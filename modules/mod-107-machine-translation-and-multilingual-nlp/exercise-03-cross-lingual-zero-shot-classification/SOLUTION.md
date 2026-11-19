# exercise-03 — Cross-Lingual Zero-Shot Classification

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-107-machine-translation-and-multilingual-nlp/exercises/exercise-03-cross-lingual-zero-shot-classification.md`.
> The target learning outcomes implied by that filename: fine-tune a **cross-lingual encoder** (XLM-R or mDeBERTa) on **English labelled data only**, then **evaluate zero-shot on other languages** on the XNLI (or MASSIVE / AmazonReviews-ML) benchmark, and report **per-language accuracy and the transfer gap** honestly — including the baseline where the English backbone is replaced with a monolingual encoder to show the cross-lingual effect.

## 1. Solution overview

Zero-shot cross-lingual classification is the pattern that lets a team support 10+ languages without 10+ labelled datasets. The three failure modes that decide whether the numbers are trustworthy:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **The "zero-shot" line is a translation line** | Evaluating on machine-translated test sets measures translation quality plus classifier quality; reporting that as "zero-shot cross-lingual" is widespread and misleading | Evaluate on **human-translated** test sets (XNLI, MASSIVE, AmericasNLI) and report translate-test as a *separate* baseline |
| **Using a monolingual English encoder** | BERT / RoBERTa have no vocabulary coverage for non-Latin scripts; "zero-shot" numbers are near-random on CJK / Arabic / Devanagari and the user never knows | Use a **genuinely multilingual encoder**: XLM-R (100 languages) or mDeBERTa-v3-base (100 languages). Document language coverage from the model card |
| **Train-dev-test language leakage** | Fine-tuning on English-only is fine; adding "a little French" to the train set to lift French numbers makes the experiment not-zero-shot. Common in tutorials | Freeze a single invariant: **training data is English only**. Every non-English evaluation is on a held-out, translated-by-humans split. Log the language inventory of every row that touches training |

The deliverable is a classification harness with four components:

1. **Backbone wrapper** — `AutoModelForSequenceClassification` on XLM-R-base (`xlm-roberta-base`) or mDeBERTa-v3-base. The exercise defaults to XLM-R because its multilingual coverage and licence are the cleanest.
2. **English-only trainer** — standard HF `Trainer` on the English XNLI split (or MNLI mapped to XNLI's 3-class schema).
3. **Multilingual evaluator** — iterate each non-English XNLI language and score with the same model; report per-language accuracy and the transfer gap `acc(en) - acc(lang)`.
4. **Translate-test baseline** — translate each non-English test sentence to English (via NLLB or Marian from exercise-01) and feed through the same classifier. Compare against zero-shot to show which languages benefit most from translate-test.

Reference datasets:
- **XNLI** (Conneau et al. 2018) — 15 languages, 7 500 test sentences per language, human-translated. The canonical cross-lingual NLI benchmark. Hub: `facebook/xnli`.
- **MASSIVE** (FitzGerald et al. 2022) — 51 languages, intent + slot labels. Harder because many languages are low-resource; stresses the backbone.
- **AmericasNLI** (Ebrahimi et al. 2022) — XNLI translated into 10 indigenous languages of the Americas. The right benchmark when you want to show how zero-shot breaks for truly underrepresented languages.

Reference backbones:
- **XLM-R-base** (Conneau et al. 2020) — 270M, 100 languages, CC100 pre-training; the standard baseline.
- **mDeBERTa-v3-base** (He et al. 2021) — 278M, 100 languages, disentangled attention; usually +1–2 points over XLM-R on XNLI.

## 2. Worked implementation

### 2.1 Project layout

```
xlingual_zs/
  data.py           # XNLI loader; language iterators
  model.py          # AutoModelForSequenceClassification wrapper
  train.py          # English-only fine-tune
  eval.py           # zero-shot across XNLI languages
  translate_test.py # baseline: translate to English, then classify
  metrics.py        # per-language accuracy, transfer gap, bootstrap CIs
  run.py            # CLI
  tests/
    test_language_coverage.py
    test_tokenizer_scripts.py
    test_transfer_gap.py
  requirements.txt
```

### 2.2 Data: the English-only training invariant

```python
# data.py
from datasets import load_dataset, concatenate_datasets

XNLI_LANGUAGES = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi",
    "ru", "sw", "th", "tr", "ur", "vi", "zh",
]


def load_xnli_english_train():
    """Train on English only. This is the zero-shot invariant."""
    ds = load_dataset("facebook/xnli", "en")
    return ds["train"], ds["validation"]  # 393k / 2.5k


def load_xnli_eval_all_languages():
    """Returns a dict {lang: test_split}."""
    return {
        lang: load_dataset("facebook/xnli", lang, split="test")
        for lang in XNLI_LANGUAGES
    }
```

Points worth defending in review:

- **No data augmentation with other languages.** The only way "zero-shot" has meaning is if the training data is exactly one language (English, by convention). If a reviewer asks "but can we add German to help German accuracy?" — that is a different experiment (multilingual supervised), not a zero-shot one.
- **XNLI test is human-translated.** XNLI's dev/test were translated by professional translators from the English MNLI matched/mismatched sets. *That* is the "zero-shot" line. Machine-translating MNLI into other languages would collapse the experiment into "measuring how well our MT system preserves NLI labels".
- **Record the language inventory of every batch.** Log the `language` column of each row that touches training; assert it is the singleton `{"en"}` before starting the trainer. This is a one-line safeguard against the most common zero-shot regression.

### 2.3 Tokenisation and the "my accuracy is random on Arabic" check

```python
# model.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer


XNLI_NUM_LABELS = 3   # entailment, neutral, contradiction


def build_model_and_tokenizer(model_id: str = "xlm-roberta-base"):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=XNLI_NUM_LABELS,
    )
    return model, tok


def tokenize_xnli(examples, tokenizer, max_length: int = 128):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],      # NLI is a two-sentence task; pair-classification
        truncation=True,
        max_length=max_length,
        padding=False,                # let the collator pad per-batch
    )
```

- **Pair-classification input, `(premise, hypothesis)`.** XLM-R's special tokens for pair input are `<s> premise </s></s> hypothesis </s>`. Concatenating the two with `" "` and passing as a single string works but drops the segment-separator and loses 1–2 accuracy points.
- **Fast tokenizer on XLM-R.** `AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)` uses the SentencePiece-backed fast tokenizer that handles non-Latin scripts natively. The slow one is not wrong, just ~10× slower.
- **`max_length=128` covers the XNLI distribution.** P99 of `len(premise)+len(hypothesis)` on XNLI is under 100 subwords for most languages; 128 is safe and keeps training cheap.

### 2.4 English-only fine-tune

```python
# train.py
from functools import partial
from transformers import (
    DataCollatorWithPadding, Trainer, TrainingArguments, set_seed,
)
from .data import load_xnli_english_train
from .model import build_model_and_tokenizer, tokenize_xnli


def train(model_id="xlm-roberta-base", output_dir="runs/xnli_en", seed=20260101):
    set_seed(seed)
    model, tok = build_model_and_tokenizer(model_id)
    train_ds, val_ds = load_xnli_english_train()

    # ASSERTION: this is the zero-shot invariant
    if "language" in train_ds.column_names:
        langs = set(train_ds.unique("language"))
        assert langs == {"en"}, f"Zero-shot invariant violated: train languages {langs}"

    train_ds = train_ds.map(partial(tokenize_xnli, tokenizer=tok), batched=True)
    val_ds = val_ds.map(partial(tokenize_xnli, tokenizer=tok), batched=True)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,                 # standard XNLI recipe: 2-3 epochs
        learning_rate=2e-5,                 # XLM-R recipe; 5e-5 destabilises
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=seed,
        fp16=True,
        report_to="none",
    )

    import numpy as np
    def metric(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        return {"accuracy": (preds == eval_pred.label_ids).mean()}

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metric,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return trainer
```

The `assert` line is the invariant. Without it, the first team member who tries `concatenate_datasets([en_train, de_train])` to help German accuracy turns zero-shot into multilingual supervised and nobody notices until the paper is written.

### 2.5 Zero-shot evaluation across languages

```python
# eval.py
import numpy as np, torch
from functools import partial
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from .model import tokenize_xnli
from .data import load_xnli_eval_all_languages


def evaluate_language(model, tok, test_split, batch_size: int = 64) -> float:
    test = test_split.map(partial(tokenize_xnli, tokenizer=tok), batched=True)
    cols = ["input_ids", "attention_mask"]
    if "token_type_ids" in test.column_names:
        cols.append("token_type_ids")
    test.set_format("torch", columns=cols + ["label"])
    loader = DataLoader(test, batch_size=batch_size,
                        collate_fn=DataCollatorWithPadding(tok))
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("label")
            out = model(**{k: v.to(model.device) for k, v in batch.items()})
            preds = out.logits.argmax(dim=-1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total


def evaluate_all(model, tok) -> dict:
    model.eval()
    eval_sets = load_xnli_eval_all_languages()
    accs = {lang: evaluate_language(model, tok, split) for lang, split in eval_sets.items()}
    accs["transfer_gap_mean"] = accs["en"] - np.mean(
        [v for k, v in accs.items() if k not in {"en", "transfer_gap_mean"}]
    )
    return accs
```

- **Transfer gap = `acc(en) − mean(acc(other))`.** The honest summary of "how zero-shot is this?" The smaller the gap, the better the cross-lingual transfer. XLM-R-base on XNLI typically lands at `acc(en) ≈ 84.6` and `transfer_gap ≈ 9–11` points; mDeBERTa-v3-base narrows that gap to ~6.
- **Per-language numbers too.** The mean hides the Swahili / Thai / Urdu tails, where the gap is often 15–20 points. Reporting only the mean is a product failure waiting to happen.

### 2.6 Translate-test baseline

```python
# translate_test.py
import torch
from transformers import pipeline

NLLB_LANG_MAP = {
    "ar": "arb_Arab", "bg": "bul_Cyrl", "de": "deu_Latn", "el": "ell_Grek",
    "es": "spa_Latn", "fr": "fra_Latn", "hi": "hin_Deva", "ru": "rus_Cyrl",
    "sw": "swh_Latn", "th": "tha_Thai", "tr": "tur_Latn", "ur": "urd_Arab",
    "vi": "vie_Latn", "zh": "zho_Hans",
}


def translate_to_english(sentences: list[str], lang: str) -> list[str]:
    src_code = NLLB_LANG_MAP[lang]
    translator = pipeline(
        "translation",
        model="facebook/nllb-200-distilled-600M",
        src_lang=src_code, tgt_lang="eng_Latn",
        device=0 if torch.cuda.is_available() else -1,
    )
    outs = translator(sentences, batch_size=16, max_length=256)
    return [o["translation_text"] for o in outs]
```

The translate-test pipeline passes `translate_to_english(premises, lang)` and `translate_to_english(hypotheses, lang)` through the English classifier. Report translate-test alongside zero-shot: for high-resource targets (de, fr, es) the two numbers are usually within 1–2 points; for low-resource targets (sw, ur) translate-test wins by 3–5 points because XLM-R's pre-training coverage for those languages is thin.

## 3. Validation steps

1. **Zero-shot invariant assertion.** Run the training script once with a corrupted dataset that mixes a few French rows into the English train set. The assertion in §2.4 must raise. If it doesn't, you have no guarantee future runs stay zero-shot.
2. **Tokenizer script coverage.** For each XNLI language, assert `tokenizer(test_split[0]["premise"])["input_ids"]` has length > 3 (i.e. non-`<s><unk></s>`). If a language returns mostly `<unk>` tokens, the backbone can't read the script and the zero-shot number will be noise. XLM-R should produce plausible token counts for all 15 XNLI languages; fail loudly otherwise.
3. **Per-language sanity floors.** After training, accuracies on `en` should exceed ~84 for XLM-R-base; `fr`/`de`/`es` should be ~79–82; `sw`/`ur`/`th` should be ~69–75. A language >3 points below these ranges flags a tokenizer bug (often `use_fast=False` for `th`) or an evaluation split loaded with the wrong language code.
4. **Bootstrap CI per language.** 7 500 test examples per language give a 95 % CI of ±~1 point; report it. A reviewer who sees `fr=80.1`, `de=80.4` without CIs may wrongly conclude `de > fr` when the difference is noise.
5. **Translate-test contrast.** For each language, compute `acc_zs(lang)` and `acc_tt(lang)`. For high-resource targets the delta is typically within noise; for low-resource it is positive for translate-test. If translate-test *loses* on a high-resource target by >2 points, your MT model (NLLB variant / beam size) is the bottleneck, not the classifier.
6. **Transfer-gap regression.** Record the mean transfer gap in the output JSON. A new backbone or recipe is only "better" if (a) English accuracy does not regress by more than 0.3 and (b) the transfer gap shrinks. Single-number reports lose this.
7. **Label-set sanity.** XNLI uses the same three-way label order as MNLI: `0=entailment, 1=neutral, 2=contradiction`. HF's `datasets` respects this, but hand-written loaders have flipped labels in the wild (XNLI's Chinese split historically used reversed labels in one early mirror). Pin a fixed `(premise, hypothesis, expected_label)` triple per language and assert at eval time.
8. **Seed stability.** Train with 3 seeds. XLM-R-base on English XNLI varies <0.3 points in English accuracy and <0.5 points in transfer gap across seeds. A run >1 point off suggests a learning-rate or tokenizer regression.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Zero-shot invariant** | Explicit assertion that train languages == {en}; invariant documented | English-only by convention, no assertion | Any non-English rows in train |
| **Multilingual backbone** | XLM-R or mDeBERTa; language coverage documented from model card | Multilingual model, no documentation | Monolingual English encoder; near-random on CJK/Arabic |
| **Pair-classification inputs** | `tokenizer(premise, hypothesis)` using special-token pair; unit test | Concatenated with `" "`; works but not canonical | Single-sentence input; silently dropped hypothesis |
| **Per-language reporting** | Accuracy per language + mean transfer gap + bootstrap CIs | Per-language, no CIs | Mean only |
| **Translate-test baseline** | Baseline reported alongside zero-shot | Mentioned but not run | Not reported |
| **Reproducibility** | `set_seed`; seed, dataset revision, backbone id in output JSON | Seed set, no echo | No seed |
| **Dataset hygiene** | XNLI licence recorded (CC-BY-NC-4.0); language inventory logged | One of those | Neither |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks noted (adversarial multilingual inputs, label-flip poisoning via unsupervised pre-training data, bias in low-resource language outputs) with mitigations | Mentioned without mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure: supported-language list with per-language accuracy + CI; known gaps for low-resource languages; recommended use (not for high-stakes decisions on languages with acc < threshold) | Partial | None |

## 5. Common mistakes

- **Using BERT / RoBERTa as the backbone.** No vocabulary coverage for non-Latin scripts; "zero-shot" on Chinese / Arabic / Thai will be near-random and the failure is silent. Use XLM-R or mDeBERTa.
- **Training on translated data and calling it zero-shot.** If you back-translate MNLI into German and add it to the train set, that is translate-train, not zero-shot. Different experiment, different claim.
- **Evaluating on machine-translated test.** XNLI's splits are human-translated; using Google-Translate'd MNLI inflates numbers and makes comparisons dishonest. Load the human-translated splits from `facebook/xnli`.
- **Reporting only the mean accuracy.** Mean hides the Swahili / Urdu / Thai tails. Always report per-language and the transfer gap.
- **Dropping segment separators.** XLM-R expects `</s></s>` between premise and hypothesis; concatenating with a space loses 1–2 accuracy points. Use the pair-classification tokeniser form.
- **Wrong tokenizer across scripts.** `use_fast=False` on XLM-R sometimes falls back to a slow SentencePiece path that mis-segments Thai. Always `use_fast=True` and assert non-trivial token counts on every XNLI language before trusting the numbers.
- **Confusing XNLI labels and MNLI labels.** XNLI follows MNLI's `0/1/2 = entailment/neutral/contradiction`. If you re-map through a classifier that was trained on a different order, your "zero-shot" accuracy is random. Pin a round-trip test.
- **Reporting translate-test as the zero-shot baseline.** Translate-test is a different, legitimate baseline. Report *both* — translate-test wins on low-resource targets, zero-shot wins on latency.
- **Fine-tuning too long.** XLM-R-base on English XNLI overfits past 3 epochs; validation accuracy climbs, test on non-English degrades. Use early stopping or hard-cap at 3 epochs.
- **Ignoring language inventory in logs.** Without a per-batch language counter, a future PR that accidentally mixes languages into train will not be caught until someone notices the transfer gap shrinking "mysteriously". The assert in §2.4 is the one-line fix.

## 6. References

Primary sources used in this solution:

- Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-R). ACL. <https://aclanthology.org/2020.acl-main.747/> — the XLM-R backbone; CC100 pre-training; the zero-shot transfer story mirrored here.
- Conneau, A. et al. (2018). *XNLI: Evaluating Cross-lingual Sentence Representations*. EMNLP. <https://aclanthology.org/D18-1269/> — the XNLI benchmark, the human-translation protocol, the 15-language test splits used in §2.2.
- He, P. et al. (2021). *DeBERTa-V3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*. <https://arxiv.org/abs/2111.09543> — the mDeBERTa variant that narrows the transfer gap vs XLM-R.
- Hu, J. et al. (2020). *XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalisation*. ICML. <https://proceedings.mlr.press/v119/hu20b.html> — the broader multi-task benchmark; defines translate-train, translate-test, and zero-shot as distinct evaluation protocols.
- FitzGerald, J. et al. (2022). *MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages*. ACL. <https://aclanthology.org/2023.acl-long.235/> — the intent+slot alternative benchmark used as a stress-test for low-resource scripts.
- Ebrahimi, A. et al. (2022). *AmericasNLI: Evaluating Zero-shot Natural Language Understanding of Pretrained Multilingual Models in Truly Low-Resource Languages*. ACL. <https://aclanthology.org/2022.acl-long.435/> — the benchmark that makes the low-resource ceiling explicit.
- Hugging Face `transformers` reference script `examples/pytorch/text-classification/run_xnli.py` — <https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_xnli.py> — canonical XNLI fine-tuning script mirrored in §2.4.
- NLLB Team (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation*. <https://arxiv.org/abs/2207.04672> — the NLLB-200 model used as the translate-test engine in §2.6.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation*: cross-lingual classifiers are routinely attacked with transliteration and homoglyph tricks (`H e l l o` with zero-width spaces, Cyrillic lookalikes); mitigate with NFKC normalisation and script-consistency checks before classification. *ML02 Data Poisoning*: XLM-R pre-training on CommonCrawl includes language-specific adversarial slices; prefer models with documented decontamination (XLM-R Large and mDeBERTa publish this).
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue; relevant whenever cross-lingual classification is used for moderation or safety filtering (bypass by code-switching, by Latin-script transliteration of non-Latin content).
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: publish the supported-language inventory with per-language accuracy and CI; mark languages below a stakeholder-set threshold as "not supported" rather than "experimental". *Measure*: transfer gap and worst-language accuracy are the metrics; the overall mean is not sufficient.
