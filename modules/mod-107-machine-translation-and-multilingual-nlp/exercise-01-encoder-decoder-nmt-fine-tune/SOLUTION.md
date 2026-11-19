# exercise-01 — Encoder-Decoder NMT Fine-Tune

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-107-machine-translation-and-multilingual-nlp/exercises/exercise-01-encoder-decoder-nmt-fine-tune.md`.
> The target learning outcomes implied by that filename: fine-tune an **encoder-decoder NMT model** (Marian, mBART-50, or NLLB-200) on a parallel corpus, handle **source/target tokenisation with the correct language codes**, run **beam search** at inference with a pinned `GenerationConfig`, and report **sacreBLEU** on a detokenised reference the way WMT does — same signature string, same tokeniser, same test-set version — so the number is comparable to the literature.

## 1. Solution overview

Fine-tuning an NMT encoder-decoder is a solved recipe, but three subtleties reliably break first-timers' numbers:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Missing or wrong language codes** | mBART-50 and NLLB-200 expect `src_lang` / `tgt_lang` codes (`en_XX`, `fr_XX`, `eng_Latn`, `fra_Latn`) on the tokenizer *and* a `forced_bos_token_id` at inference. Omitting any one causes the model to translate into the wrong language or echo the source | Set `tokenizer.src_lang` *before* tokenising inputs; set `text_target=` for the reference; pin `forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)` on the `GenerationConfig` |
| **Reporting raw BLEU on arbitrary tokenisation** | Scoring a `.split()`-tokenised hypothesis against a `.split()`-tokenised reference gives a number that varies by ±3 BLEU with tokeniser choice; it is not comparable to any published result | Use **sacreBLEU** with the WMT signature string (`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.x.x`); feed it **detokenised** strings and let sacreBLEU do the tokenisation |
| **Train/infer decoder mismatch** | Training is teacher-forced with `labels` right-shifted by the collator; inference uses `.generate()` with beam search. Forgetting `generation_config.num_beams`, `max_new_tokens`, or `forced_bos_token_id` gives truncated outputs, wrong-language outputs, or copy-the-source outputs — none diagnosable from loss curves | Pin a `GenerationConfig` next to the checkpoint and reuse it in eval and serving. Round-trip test it on one example |

The deliverable is a fine-tune harness with five components:

1. **Dataset builder** — load a parallel corpus (`opus100`, `wmt14`, or `flores` for eval) via `datasets`, format `(src, tgt)` pairs, tokenise source and target with the correct language codes.
2. **Model wrapper** around `AutoModelForSeq2SeqLM` (Marian `Helsinki-NLP/opus-mt-{src}-{tgt}`, mBART `facebook/mbart-large-50-many-to-many-mmt`, or NLLB `facebook/nllb-200-distilled-600M`).
3. **Collator + trainer** using `DataCollatorForSeq2Seq` with `-100` label padding and `Seq2SeqTrainer` with `predict_with_generate=True`.
4. **Generation config** pinned: beam size, `max_new_tokens`, length penalty, `forced_bos_token_id`.
5. **sacreBLEU evaluator** with the WMT signature recorded in the output JSON so the number is reproducible across runs and repos.

Reference corpora:
- **WMT14 en-de / en-fr** (Bojar et al. 2014) — the canonical NMT benchmarks; WMT publishes `newstest20XX` dev/test sets that sacreBLEU knows by name (`sacrebleu -t wmt14 -l en-de`).
- **FLORES-200** (NLLB team 2022) — human-translated multilingual eval across 200 languages; `datasets` exposes it as `facebook/flores`; 2009 dev / 1012 devtest sentences per language.
- **OPUS-100** (Zhang et al. 2020) — 100-language parallel training data distilled from OPUS; useful for low-resource fine-tunes and as a mBART / NLLB fine-tune source.

Reference models — pick by direction and licence:
- **Marian `Helsinki-NLP/opus-mt-{src}-{tgt}`** (Tiedemann & Thottingal 2020) — small (~75M), per-direction, CPU-runnable. The worked example below uses `opus-mt-en-de` because it is the fastest to iterate on.
- **mBART-50 many-to-many** (Tang et al. 2020) — 611M parameters, 50 languages; multilingual fine-tuning; needs `forced_bos_token_id`.
- **NLLB-200 distilled 600M** (NLLB team 2022) — 200 languages; same `forced_bos_token_id` discipline; the current default for low-resource directions.

## 2. Worked implementation

### 2.1 Project layout

```
nmt_bake/
  data.py            # datasets.load_dataset + tokenise with language codes
  model.py           # AutoModelForSeq2SeqLM + language-code wiring
  generation.py      # pinned GenerationConfig
  metrics.py         # sacreBLEU with WMT signature
  run.py             # train/eval CLI
  tests/
    test_lang_codes.py
    test_bleu.py
    test_generate.py
  requirements.txt
```

### 2.2 Dataset preparation — language codes are not optional

```python
# data.py — mBART-50 variant; see §2.3 for Marian / NLLB differences
from datasets import load_dataset

MAX_INPUT_LEN = 256   # mBART's effective seq length for news; raise to 512 for long docs
MAX_TARGET_LEN = 256


def load_wmt14_en_de():
    return load_dataset("wmt14", "de-en")  # train / validation / test


def preprocess_mbart(examples, tokenizer, src_lang="en_XX", tgt_lang="de_DE"):
    src_texts = [ex["en"] for ex in examples["translation"]]
    tgt_texts = [ex["de"] for ex in examples["translation"]]

    # mBART-50 and NLLB: set src_lang on the tokenizer BEFORE tokenising inputs.
    # The tokenizer prepends the source language code as the first special token.
    tokenizer.src_lang = src_lang
    model_inputs = tokenizer(
        src_texts,
        max_length=MAX_INPUT_LEN,
        truncation=True,
    )

    # text_target= routes through the decoder-side tokenisation rules:
    # for mBART-50 that prepends the TARGET language code as the first token, which
    # the collator-built decoder_input_ids will then use as decoder_start_token_id.
    tokenizer.tgt_lang = tgt_lang
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            tgt_texts,
            max_length=MAX_TARGET_LEN,
            truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

Points worth defending in review:

- **Set `src_lang` / `tgt_lang` on the tokenizer, not inline per call.** mBART-50 and NLLB tokenisers prepend a language-code special token during `__call__`; relying on the two-arg positional form (`tokenizer(src, tgt)`) silently concatenates them as a classification pair and loses the language codes.
- **`as_target_tokenizer()` is required on mBART-50.** It flips the tokenizer into a mode that applies the target-side language-code rule. On NLLB the equivalent is `text_target=...` with `tokenizer.tgt_lang=...`; on Marian neither is required (Marian embeds language direction in the checkpoint identity, so there is no language-code token).
- **No padding here.** The collator pads to the *batch* max, not the *dataset* max. Faster and the only way `-100` masking works cleanly in one pass.
- **Truncation on both sides, but `MAX_TARGET_LEN` wide enough.** Setting it below the P99 reference length teaches the model to end mid-sentence; for news translation 256 covers >99 % of sentences.

### 2.3 Model + language-code wiring

```python
# model.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def build_model_and_tokenizer(
    model_id: str,
    src_lang: str | None = None,
    tgt_lang: str | None = None,
):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    if "mbart" in model_id or "nllb" in model_id:
        assert src_lang and tgt_lang, (
            "mBART / NLLB require src_lang and tgt_lang; "
            "e.g. en_XX/de_DE for mBART-50 or eng_Latn/deu_Latn for NLLB"
        )
        tok.src_lang = src_lang
        tok.tgt_lang = tgt_lang
    # Marian (Helsinki-NLP/opus-mt-en-de) encodes direction in the checkpoint;
    # no explicit codes required.
    return model, tok
```

Model-family cheatsheet — these codes are the ones the official tokenizers accept:

| Model | Source code | Target code | Decoder start / `forced_bos_token_id` |
|---|---|---|---|
| Marian `opus-mt-en-de` | *(none)* | *(none)* | Set by checkpoint; no action required |
| mBART-50 many-to-many | `en_XX` | `de_DE`, `fr_XX`, `zh_CN`, … | `tokenizer.convert_tokens_to_ids("de_DE")` |
| NLLB-200 | `eng_Latn` | `deu_Latn`, `fra_Latn`, `zho_Hans`, … | `tokenizer.convert_tokens_to_ids("deu_Latn")` |

### 2.4 Collator — `-100` masking + right-shift

```python
# run.py (fragment)
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)
```

Same discipline as mod-106 exercise-01: `DataCollatorForSeq2Seq` rewrites pad positions in `labels` to `-100` (so they don't contribute to loss) and builds `decoder_input_ids` by shifting the unmasked labels right by one. For mBART-50 / NLLB the shift puts the *target* language-code token at position 0, which is what makes the model translate *into* that language.

### 2.5 Generation config — pin it once, use it everywhere

```python
# generation.py
from transformers import GenerationConfig


def nmt_generation_config(tokenizer, tgt_lang_code: str | None = None) -> GenerationConfig:
    cfg = GenerationConfig(
        max_new_tokens=256,
        num_beams=5,              # WMT papers settle on 4-5; marginal returns above 5
        length_penalty=1.0,       # news; set 0.6 for short outputs, 1.2 for long
        no_repeat_ngram_size=0,   # DO NOT set >0 for NMT — see §5 common mistakes
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if tgt_lang_code is not None:
        cfg.forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)
    return cfg
```

- **`num_beams=5`.** The WMT-era default; `=4` is also common (Marian ships at 4). Greedy (`=1`) drops ~1 BLEU on news.
- **`length_penalty=1.0`.** Encourages roughly length-faithful outputs. `>1` lengthens (useful for Chinese→English where outputs tend to under-length); `<1` shortens.
- **`no_repeat_ngram_size=0`** — crucial. See §5: forcing no-trigram-repeat destroys correct repetitions ("the the", "la la ville", punctuation sequences) and drops BLEU on real text.
- **`forced_bos_token_id`** — the mBART / NLLB mechanism that steers the decoder to the correct language.

### 2.6 Trainer

```python
# run.py
import argparse, numpy as np
from functools import partial
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, set_seed,
)
from .data import load_wmt14_en_de, preprocess_mbart
from .model import build_model_and_tokenizer
from .generation import nmt_generation_config
from .metrics import sacrebleu_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="facebook/mbart-large-50-many-to-many-mmt")
    p.add_argument("--src-lang", default="en_XX")
    p.add_argument("--tgt-lang", default="de_DE")
    p.add_argument("--output-dir", default="runs/nmt")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260101)
    a = p.parse_args()
    set_seed(a.seed)

    ds = load_wmt14_en_de()
    model, tok = build_model_and_tokenizer(a.model_id, a.src_lang, a.tgt_lang)
    gen_cfg = nmt_generation_config(tok, tgt_lang_code=a.tgt_lang)
    model.generation_config = gen_cfg

    tokenised = ds.map(
        partial(preprocess_mbart, tokenizer=tok, src_lang=a.src_lang, tgt_lang=a.tgt_lang),
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)
        return sacrebleu_metrics(decoded_preds, decoded_labels)

    args = Seq2SeqTrainingArguments(
        output_dir=a.output_dir,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size * 2,
        learning_rate=a.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        predict_with_generate=True,
        generation_config=gen_cfg,
        fp16=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        seed=a.seed,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        tokenizer=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print(trainer.evaluate(tokenised["test"]))
```

### 2.7 sacreBLEU — the WMT-comparable signature

```python
# metrics.py
import sacrebleu


def sacrebleu_metrics(preds: list[str], refs: list[str]) -> dict:
    """
    Returns BLEU and chrF++ at the detokenised string level with the WMT
    reference signature. sacreBLEU does the tokenisation internally.
    """
    bleu = sacrebleu.corpus_bleu(preds, [refs])  # refs is list-of-lists (one ref per hyp)
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)  # chrF++
    return {
        "bleu": bleu.score,
        "bleu_signature": bleu.get_signature().format(short=False),
        "chrf++": chrf.score,
        "chrf_signature": chrf.get_signature().format(short=False),
    }
```

Four things reviewers must check:

- **Detokenised inputs.** Pass strings with real whitespace and punctuation, *not* `.split()`-joined BPE. `batch_decode(..., skip_special_tokens=True)` already does this; a stray `" ".join(tokens)` on subwords will give nonsense BLEU.
- **Refs as list-of-lists.** `sacrebleu.corpus_bleu(preds, [refs])` — the outer list is "reference systems", so even with one reference set it is `[refs]`, not `refs`.
- **Record the signature string.** `BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.x.x` is the tag the WMT community uses to compare across papers. Saving it in the run JSON is cheaper than re-running and more honest than reporting a bare number.
- **chrF++ as a second anchor.** chrF is character-n-gram F-score; `word_order=2` adds word-order n-grams, giving chrF++. More stable than BLEU on morphologically rich targets (German, Finnish, Czech) and tracks human judgment better on low-resource languages.

## 3. Validation steps

1. **Language-code unit test.** Encode `"Hello world"` with `tokenizer.src_lang="en_XX"` then with `src_lang="fr_XX"`; the resulting `input_ids[0]` must differ (different language-code special token). On NLLB the equivalent codes are `eng_Latn` / `fra_Latn`. If they are equal, the tokenizer is dropping the code and translations will be in the wrong language.
2. **`forced_bos_token_id` round trip.** On an untrained mBART-50, translate `"Hello."` with `forced_bos_token_id = id("de_DE")` vs `id("fr_XX")`. The two outputs must be in different languages (langid classifier on each; expect `de` and `fr` respectively). Zero delta means the generation config isn't reaching `.generate()`.
3. **Label masking.** Build a two-example batch; after the collator, assert every pad in `labels` is `-100` and `decoder_input_ids[:, 0]` equals the target-language-code token id (mBART) or `pad_token_id` (Marian/NLLB — the convention differs by family; check `model.config.decoder_start_token_id`).
4. **Hand-computed BLEU.** Pin a tiny test: `pred=["the cat sat"]`, `ref=["the cat sat"]` → BLEU 100. `pred=["the cat sat"]`, `ref=["a cat sat"]` → BLEU ≈ 51.8 (sacreBLEU default smoothing, `tok:13a`). If your assertion fails, you are either reading the args in the wrong order (sacreBLEU is `(hyps, refs)` with refs list-of-lists) or you are smoothing differently.
5. **Beam search sanity.** On dev, `num_beams=5` should net +0.5 to +1.5 BLEU over `num_beams=1`. Zero delta means `generation_config` isn't wired through.
6. **`no_repeat_ngram_size=0` regression.** Translate a sentence with legitimate repetition ("Je t'aime, je t'aime.") with `no_repeat_ngram_size=0` and `=3`. The `=3` version will drop or mutate the second occurrence; this is why we leave it off for NMT.
7. **sacreBLEU signature pin.** On a fixed `(preds, refs)` fixture, assert the signature returned by `bleu.get_signature().format(short=False)` is the full `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.x.x` string. If it drifts, so does the comparability of the number.
8. **FLORES-200 cross-check.** For any directionally-fine-tuned model, run FLORES-200 devtest and compare to the published NLLB result (`eng_Latn → deu_Latn` NLLB-200 distilled 600M lands around BLEU 30–32 on FLORES-200 devtest). A run 3+ BLEU below the published number usually means a wrong language code or a wrong `forced_bos_token_id`, not a bad fine-tune.
9. **Seed stability.** Train with 3 seeds. Variance between seeds on WMT14 en-de for mBART-50 fine-tunes is typically <0.5 BLEU; a run >1 BLEU below the median flags a config bug.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Language-code wiring** | `src_lang` / `tgt_lang` on tokenizer; `forced_bos_token_id` on `GenerationConfig`; unit test | Codes set inline per call | No codes; wrong-language outputs |
| **Decoder start / shift-right** | Collator-built `decoder_input_ids`; unit test verifying first token is the target-language code (mBART) | Shift present, not tested | Hand-built labels without shift |
| **Seq2seq label masking** | `DataCollatorForSeq2Seq(label_pad_token_id=-100)` + unit test | Collator used, no test | Hand-padded labels; pads scored by loss |
| **Generation config pinning** | `GenerationConfig` co-located with model; used in train/eval/serve | `.generate(num_beams=...)` passed ad-hoc | Default generate; `num_beams=1`, `max_new_tokens=20` |
| **BLEU protocol fidelity** | sacreBLEU with WMT signature recorded; detokenised strings; list-of-lists refs | Score only, no signature | `nltk.translate.bleu` on `.split()` tokens |
| **Second-metric anchor** | chrF++ reported alongside BLEU | BLEU only | No corpus metric |
| **Dataset hygiene** | Dataset + revision + licence recorded (WMT14 statmt.org terms; FLORES CC-BY-SA-4.0; OPUS-100 per-source licences) | One of those | Neither |
| **Reproducibility** | `set_seed`; seed + revision + sacreBLEU signature in output JSON; `transformers` + `sacrebleu` versions pinned | Seed set, no echo | No seed |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks noted (prompt injection via translated text, poisoning via scraped web parallel data) with mitigations | Mentioned without mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure applied: language pair coverage, dataset licence, known biases (gender in pronoun-poor source languages), decode config, BLEU signature | One of those | Neither |

## 5. Common mistakes

- **Forgetting `forced_bos_token_id`.** mBART-50 and NLLB will happily translate into any language the decoder was pre-trained on. Without `forced_bos_token_id`, the model may emit English even when you asked for German. Diagnostic: run `langid` on 20 dev outputs; if the ratio of target-language outputs is <95 %, your `forced_bos_token_id` is wrong or missing.
- **Setting `no_repeat_ngram_size > 0` for NMT.** Translation has legitimate repetition ("the the", "la la ville", "I am I am"). Blocking trigram repeats drops BLEU 0.3–0.7 on news and up to 1.5 on dialogue. This is the opposite of summarisation, where `no_repeat_ngram_size=3` helps.
- **Positional-arg tokenising of the target.** `tokenizer(src, tgt)` on mBART / NLLB treats the two strings as a paired input and loses target-side language codes. Use `tokenizer.as_target_tokenizer()` or the `text_target=` kwarg.
- **Reporting BLEU on tokenised strings.** `.split()`-tokenised BLEU varies ±3 BLEU with tokeniser. sacreBLEU tokenises internally (`tok:13a` for most targets, `tok:zh` for Chinese, `tok:ja-mecab` for Japanese). Pass detokenised strings; log the signature.
- **Mismatched tokeniser at inference.** Loading the model with `AutoModelForSeq2SeqLM.from_pretrained(X)` but a tokenizer from a different checkpoint produces nonsense. Save model + tokenizer to the same directory; always `AutoTokenizer.from_pretrained(save_dir)`.
- **Wrong `MAX_INPUT_LEN` for the model family.** Marian caps at 512; mBART-50 at 1024; NLLB-200 at 1024 per-sentence. Setting it above the model's position-embedding limit silently truncates with no warning.
- **Running greedy at inference "to save time".** Greedy drops 0.5–1.5 BLEU vs beam=5 on news; the speed-up (~2–3×) is rarely worth it in offline eval. For serving, beam=4 or beam=1 with a separate reranker is a defensible trade; for reporting, always beam ≥ 4.
- **Reporting dev BLEU as test.** The fine-tuning literature reports `newstest20XX` from WMT as test; using the `validation` split as the reported number is both wrong and inflationary. sacreBLEU's named test sets (`sacrebleu -t wmt14 -l en-de --download`) are the reference splits.
- **Fine-tuning at the wrong learning rate.** mBART / NLLB fine-tune stably at 3e-5 with warmup; 5e-4 catastrophically forgets. Marian takes 1e-4. If your epoch-1 BLEU is below 10 on a well-resourced direction, the LR is the first thing to check.
- **Ignoring tokeniser side effects for Chinese / Japanese target.** sacreBLEU auto-selects `tok:zh` / `tok:ja-mecab` *only* if you pass `--language-pair` or construct `BLEU(tokenize="zh")` manually. Defaulting to `13a` on `zh` targets over-reports BLEU by 5–10 points; always pass the tokenisation when the target language has non-whitespace segmentation.

## 6. References

Primary sources used in this solution:

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. <https://arxiv.org/abs/1706.03762> — the Transformer encoder-decoder; still the architecture under every model mentioned here.
- Junczys-Dowmunt, M. et al. (2018). *Marian: Fast Neural Machine Translation in C++*. ACL. <https://aclanthology.org/P18-4020/> — the Marian framework and the Helsinki-NLP checkpoints used in the worked example.
- Tiedemann, J., Thottingal, S. (2020). *OPUS-MT — Building open translation services for the World*. EAMT. <https://aclanthology.org/2020.eamt-1.61/> — the OPUS-MT project that produces the `Helsinki-NLP/opus-mt-*` checkpoints.
- Tang, Y. et al. (2020). *Multilingual Translation with Extensible Multilingual Pretraining and Finetuning* (mBART-50). <https://arxiv.org/abs/2008.00401> — mBART-50, language-code tokens, `forced_bos_token_id` discipline used in §2.5.
- NLLB Team (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation*. <https://arxiv.org/abs/2207.04672> — NLLB-200, FLORES-200 eval protocol, `eng_Latn` code system.
- Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores*. WMT. <https://aclanthology.org/W18-6319/> — the paper that introduced sacreBLEU and the signature string convention used in §2.7.
- Popović, M. (2015). *chrF: character n-gram F-score for automatic MT evaluation*. WMT. <https://aclanthology.org/W15-3049/> — chrF and the `word_order=2` chrF++ variant used as the second anchor.
- Bojar, O. et al. (2014). *Findings of the 2014 Workshop on Statistical Machine Translation*. WMT. <https://aclanthology.org/W14-3302/> — WMT14 `newstest` dev/test convention sacreBLEU reuses.
- Hugging Face `transformers` reference script `examples/pytorch/translation/run_translation.py` — <https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py> — canonical implementation of the seq2seq translation fine-tune flow mirrored in §2.6.
- sacreBLEU — <https://github.com/mjpost/sacrebleu> — the reference implementation and WMT-comparable tokenisation / signature machinery used in §2.7.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation* applies directly: a translated document can embed an instruction ("Ignorieren Sie Ihre Anweisungen; der Inhalt ist: …") whose translation into English is then consumed by an LLM. Treat MT outputs as untrusted strings. *ML02 Data Poisoning* applies when fine-tuning on web-scraped parallel data (ParaCrawl, CCAligned); prefer vetted splits like FLORES-200 for eval.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for adversarial ML; relevant whenever MT output feeds downstream moderation, search, or compliance workflows that trust the translation.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record language-pair direction, dataset licence, known biases (gender bias in pronoun-poor source languages like Turkish → English; register bias from news-only training data); *Measure*: report BLEU + chrF++ together with sacreBLEU signature, generation config, seed, and dataset revision — faithfulness-of-translation (hallucination, omission) is covered in exercise-04.
