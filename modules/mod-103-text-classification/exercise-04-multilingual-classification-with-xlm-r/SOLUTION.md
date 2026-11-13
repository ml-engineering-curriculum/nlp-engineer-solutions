# exercise-04 — Multilingual Classification with XLM-R

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-103-text-classification/exercises/exercise-04-multilingual-classification-with-xlm-r.md`.
> This solution targets the learning goals that filename implies: train a classifier that works across many languages using **XLM-RoBERTa** (XLM-R) and evaluate it on the **cross-lingual zero-shot** setting — fine-tune on one language, test on others — plus the **joint multilingual** setting — fine-tune on the concatenation, test per language.

## 1. Solution overview

The core claim behind XLM-R (Conneau et al. 2020): a single transformer pre-trained with masked language modelling on CommonCrawl in 100 languages learns a shared subword representation strong enough that a classifier head fine-tuned in one language transfers zero-shot to the others. This is what makes XLM-R the production default for "one classifier, many locales" problems.

Two deployment settings that dominate real work:

| Setting | Train on | Test on | Typical use |
|---|---|---|---|
| **Cross-lingual zero-shot (XLT)** | English labels only (usually largest) | Every target language, unseen | Bootstrap a new locale with no human labels in that locale yet |
| **Joint multilingual** | Concatenation of all labelled languages, balanced | Each language separately | Mature product with uneven label budgets per locale |

**Benchmark.** XNLI (Conneau et al. 2018) is the standard cross-lingual benchmark for classification; it is premium NLI, not sentiment, but the protocol is the same and it is the reference used in the XLM-R paper. For a sentiment-style exercise, MARC (Keung et al. 2020) and MultiEURLEX (Chalkidis et al. 2021) both ship aligned multilingual training/test splits.

**Metric contract.** Per-language macro-F1 plus a **cross-lingual transfer gap** (English macro-F1 minus mean target-language macro-F1). Report the gap, not just the headline: a 90-point model on English that drops to 50 on Swahili is often worse in production than an 80-point model that holds 75 everywhere.

The deliverable: a fine-tune recipe for `xlm-roberta-base`, a training-data assembler that supports both XLT and joint, and an evaluator that emits per-language metrics with confidence intervals and a transfer-gap table.

## 2. Worked implementation

### 2.1 Project layout

```
multilingual/
  data.py              # per-language splits + joint assembly + language-balanced sampling
  model.py             # XLM-R classification head
  train.py             # Trainer wiring, XLT and joint modes
  evaluate.py          # per-language metrics + transfer gap
  run.py               # CLI
  tests/
    test_data.py
    test_train_smoke.py
    test_eval.py
```

### 2.2 Data: per-language splits and joint assembly

```python
# data.py
from dataclasses import dataclass
from datasets import Dataset, concatenate_datasets, interleave_datasets
import pandas as pd


@dataclass(frozen=True)
class MultilingualData:
    train_by_lang: dict[str, Dataset]
    val_by_lang: dict[str, Dataset]
    test_by_lang: dict[str, Dataset]
    labels: list[str]


def load(path_tmpl: str, languages: list[str], text_col: str, label_col: str,
         lang_col: str = "lang") -> MultilingualData:
    """path_tmpl: e.g. 'data/marc/{lang}/{split}.csv' with {lang}, {split} slots."""
    train, val, test = {}, {}, {}
    for lg in languages:
        splits = {
            s: pd.read_csv(path_tmpl.format(lang=lg, split=s)) for s in ("train", "val", "test")
        }
        if not train:
            labels = sorted(splits["train"][label_col].unique().tolist())
            l2i = {l: i for i, l in enumerate(labels)}
        for s, df in splits.items():
            df = df.rename(columns={text_col: "text", label_col: "label"})
            df["label"] = df["label"].map(l2i)
            df[lang_col] = lg
            d = Dataset.from_pandas(df[["text", "label", lang_col]].reset_index(drop=True),
                                    preserve_index=False)
            {"train": train, "val": val, "test": test}[s][lg] = d
    return MultilingualData(train, val, test, labels)


def joint_train(data: MultilingualData, balance: str = "upsample",
                seed: int = 20260101) -> Dataset:
    """Assemble a joint training set from per-language splits.

    balance='concat': simple concatenation; majority language dominates.
    balance='upsample': each language sampled to the size of the largest.
    balance='downsample': each sampled to the size of the smallest.
    balance='temperature': pT sampling at T=0.5 — the XLM-R paper default.
    """
    parts = list(data.train_by_lang.values())
    if balance == "concat":
        return concatenate_datasets(parts)
    sizes = [len(p) for p in parts]
    if balance == "upsample":
        target = max(sizes)
        return concatenate_datasets([
            p.shuffle(seed=seed).select(range(target)) if len(p) >= target
            else _upsample_to(p, target, seed) for p in parts
        ])
    if balance == "downsample":
        target = min(sizes)
        return concatenate_datasets([p.shuffle(seed=seed).select(range(target)) for p in parts])
    if balance == "temperature":
        # pT = (n_l / Σ n) ^ (1/T), normalised; realised via interleave_datasets probabilities.
        T = 0.5
        probs = [(s ** (1 / T)) for s in sizes]
        probs = [x / sum(probs) for x in probs]
        return interleave_datasets(parts, probabilities=probs, seed=seed,
                                   stopping_strategy="all_exhausted")
    raise ValueError(balance)


def _upsample_to(ds: Dataset, n: int, seed: int) -> Dataset:
    reps = (n // len(ds)) + 1
    extended = concatenate_datasets([ds.shuffle(seed=seed + i) for i in range(reps)])
    return extended.select(range(n))
```

**Why temperature sampling is the paper default.** Downsampling throws away signal from high-resource languages; upsampling wastes training steps on duplicates in low-resource ones. **Temperature sampling** at `T = 0.5` (Arivazhagan et al. 2019; Conneau & Lample 2019) interpolates between "proportional to data" (`T → 1`) and "uniform across languages" (`T → ∞`). XLM-R uses `T = 0.3`; `T = 0.5` is a reasonable classifier default when you have less extreme size skew.

### 2.3 Model

```python
# model.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "xlm-roberta-base"


def build(num_labels: int, labels: list[str]):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=num_labels,
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)},
    )
    return model, tok
```

Two capacity choices worth stating:

- **`xlm-roberta-base`** (278 M parameters) is the default; **`xlm-roberta-large`** (560 M) gains ~3-5 points on XNLI but doubles fine-tune time and pushes a `batch_size=8` into gradient-accumulation territory on a single 24 GB GPU.
- **SentencePiece Unigram tokeniser** with 250 K vocab (shared across all 100 languages). Do NOT pre-tokenise with language-specific tools; XLM-R's tokeniser is the whole point.

### 2.4 Training (both XLT and joint)

```python
# train.py
import torch
from transformers import (
    DataCollatorWithPadding, Trainer, TrainingArguments, set_seed,
)
from .model import build
from .data import MultilingualData, joint_train


def tokenise_fn(tokenizer, max_length: int = 128):
    def _fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)
    return _fn


def train_xlt(data: MultilingualData, source_lang: str, output_dir: str,
              epochs: int = 3, lr: float = 2e-5, batch_size: int = 32,
              seed: int = 20260101):
    """Fine-tune on source_lang train; val is source_lang val; test is every lang."""
    set_seed(seed)
    model, tok = build(len(data.labels), data.labels)
    tf = tokenise_fn(tok)

    tr = data.train_by_lang[source_lang].map(tf, batched=True)
    vl = data.val_by_lang[source_lang].map(tf, batched=True)

    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr, weight_decay=0.01, warmup_ratio=0.1,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, seed=seed,
        fp16=torch.cuda.is_available(), report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=vl,
                      tokenizer=tok, data_collator=DataCollatorWithPadding(tok))
    trainer.train()
    return trainer, tok


def train_joint(data: MultilingualData, output_dir: str,
                balance: str = "temperature", epochs: int = 3,
                lr: float = 2e-5, batch_size: int = 32, seed: int = 20260101):
    set_seed(seed)
    model, tok = build(len(data.labels), data.labels)
    tf = tokenise_fn(tok)

    tr = joint_train(data, balance=balance, seed=seed).map(tf, batched=True)
    # Balanced joint val: concatenation of per-language val splits.
    vl_parts = [v.map(tf, batched=True) for v in data.val_by_lang.values()]
    from datasets import concatenate_datasets
    vl = concatenate_datasets(vl_parts)

    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr, weight_decay=0.01, warmup_ratio=0.1,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, seed=seed,
        fp16=torch.cuda.is_available(), report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=vl,
                      tokenizer=tok, data_collator=DataCollatorWithPadding(tok))
    trainer.train()
    return trainer, tok
```

Hyperparameters pulled from the XLM-R paper's fine-tune appendix: learning rate `1e-5` to `3e-5`, 3-5 epochs, batch size `16-32`, max length `128` for sentence-pair tasks or `256` for single-sentence long inputs. These are robust across languages; do not tune separately per locale (that would be a second hidden form of leakage).

### 2.5 Per-language evaluation

```python
# evaluate.py
import numpy as np
from sklearn.metrics import f1_score
from .data import MultilingualData


def macro_f1(y, yp) -> float:
    return f1_score(y, yp, average="macro", zero_division=0.0)


def bootstrap_ci(y, yp, n: int = 10_000, seed: int = 20260101):
    rng = np.random.default_rng(seed)
    y, yp = np.asarray(y), np.asarray(yp)
    scores = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        scores[b] = macro_f1(y[idx], yp[idx])
    lo, hi = np.quantile(scores, [0.025, 0.975])
    return float(macro_f1(y, yp)), float(lo), float(hi)


def evaluate_all(trainer, tokenizer, data: MultilingualData) -> dict:
    tf = lambda b: tokenizer(b["text"], truncation=True, max_length=128)
    rows = []
    for lg, ds in data.test_by_lang.items():
        ds_t = ds.map(tf, batched=True)
        preds = trainer.predict(ds_t).predictions.argmax(-1)
        y = ds["label"]
        p, lo, hi = bootstrap_ci(y, preds)
        rows.append({"lang": lg, "macro_f1": p, "ci95": [lo, hi], "n": len(y)})
    report = {"per_language": rows}
    f1s = {r["lang"]: r["macro_f1"] for r in rows}
    if "en" in f1s:
        others = [v for k, v in f1s.items() if k != "en"]
        report["transfer_gap_en_to_others"] = float(f1s["en"] - np.mean(others))
    report["mean_target_macro_f1"] = float(np.mean(list(f1s.values())))
    return report
```

The transfer gap is the headline cross-lingual number. Report both it and per-language macro-F1 — the mean alone can hide a catastrophic failure on one locale.

### 2.6 End-to-end runs

```python
# run.py
import argparse, json
from .data import load
from .train import train_xlt, train_joint
from .evaluate import evaluate_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["xlt", "joint"], required=True)
    p.add_argument("--path-tmpl", required=True)
    p.add_argument("--languages", nargs="+", required=True)
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--source-lang", default="en")
    p.add_argument("--balance", default="temperature")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--report", default="report.json")
    a = p.parse_args()

    data = load(a.path_tmpl, a.languages, a.text_col, a.label_col)
    if a.mode == "xlt":
        trainer, tok = train_xlt(data, a.source_lang, a.out_dir)
    else:
        trainer, tok = train_joint(data, a.out_dir, balance=a.balance)

    report = evaluate_all(trainer, tok, data)
    report["mode"] = a.mode
    report["source_lang"] = a.source_lang if a.mode == "xlt" else None
    report["balance"] = a.balance if a.mode == "joint" else None
    with open(a.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
```

## 3. Validation steps

1. **Smoke run.** 500 rows per language, 1 epoch, `max_length=64`, `--mode xlt`. The end-to-end pipeline should emit a `report.json` with all languages present and macro-F1s in a non-trivial range (not 0.0 or 1.0 everywhere). Catches loader / column drift.
2. **Tokeniser integrity.** For each language, pick 10 held-out strings, assert `tokenizer.decode(tokenizer.encode(x))` reconstructs the input up to the documented sentencepiece normalisation (trailing / leading whitespace handling differs across HF versions — `tokenizer.decode(..., skip_special_tokens=True)` is the fair comparison).
3. **Script coverage.** Assert each language in the test set has characters from its expected script (Devanagari for Hindi, Cyrillic for Russian, CJK for Chinese) — a corrupted/transliterated split silently wrecks XLT. Use Unicode block ranges, not language-detection libraries (those are circular).
4. **Baseline floor.** The majority-class baseline per language must be beaten by ≥10 points by the XLM-R model on languages with ≥1 K test examples. If XLT closely tracks majority-class on some language, that is the transfer failure to investigate (not a bug in the code).
5. **Transfer gap monotonicity.** Train with `source_lang=en` and `source_lang=fr`; the French-source model should transfer worse to English than the English-source model transfers to French, roughly symmetric, with the English-source numbers usually higher because of pre-training imbalance. If the gaps are wildly asymmetric (>10 points) your source-language train set is leaked into the target test set or the source has extra data.
6. **Joint balance ablation.** Report joint performance under `balance ∈ {concat, upsample, downsample, temperature}` on the same model. Temperature (`T=0.5`) should beat `concat` on low-resource languages while sacrificing ≤1 point on high-resource ones. If `concat` wins everywhere, either the data is nearly balanced to begin with or the temperature implementation is wrong.
7. **Seed sensitivity.** Three seeds for XLT; publish min/max/mean. Small-fine-tune variance on XLM-R is known to be 1-3 points per language; do not report single-seed deltas smaller than the spread.
8. **Repro envelope.** Pin `transformers`, `torch`, `datasets`, `sentencepiece` versions and the tokeniser SHA (`xlm-roberta-base` has had tokeniser spec updates). A missing pin can swap vocab IDs under the floor.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Model + tokeniser** | `xlm-roberta-base` or larger; its own SentencePiece tokeniser used end-to-end | Base model only; no justification | Language-specific pre-tokenisation applied |
| **Protocol** | Both XLT and joint evaluated on the same test splits; transfer gap reported | One of the two | Only English, called "multilingual" |
| **Joint sampling** | Temperature (or justified alternative) with the knob explicit | `concat` only | Majority language dominates, unreported |
| **Metrics** | Per-language macro-F1 + bootstrap CI + transfer gap | Aggregate macro-F1 only | Accuracy on language-imbalanced test |
| **Hyperparameters** | Within the XLM-R paper's fine-tune envelope; same per language | Per-language tuning, undocumented | No stated hyperparameters |
| **Determinism** | `set_seed`; CI tolerance named; seeds re-run | Partial seeding | No seed; results not reproducible |
| **Low-resource languages** | Reported separately; failure cases flagged (e.g. Burmese) | Averaged into the mean | Not covered; mean hides failures |
| **Data licence + fairness** | Dataset licences declared per language; per-locale fairness report per NIST AI RMF (Measure); input-side risks per OWASP ML Top 10 / MITRE ATLAS when model is exposed cross-locale | Mentions one | Ignored |

## 5. Common mistakes

- **Language-specific pre-tokenisation.** Running jieba/MeCab/ICU over Chinese/Japanese inputs *before* the XLM-R tokeniser destroys the shared subword space the model was pre-trained with. Let the HF tokeniser do it all.
- **Concatenating splits with imbalanced sizes.** If English has 200 K train rows and Swahili has 2 K, `concat` turns the joint model into "English with a Swahili rounding error". Use temperature sampling and report the knob.
- **Evaluating only on an aggregated set.** A single macro-F1 computed on the concatenation of all languages hides catastrophic per-locale failures. Always publish the per-language table.
- **XLT source leakage via translation.** If "zero-shot" means "trained on English + machine-translated to target language", that is **not** zero-shot and the paper's numbers do not apply. Document the source split exactly.
- **Over-reading the transfer gap.** A smaller gap is not strictly better if the English number dropped to meet the target languages. Report the two numbers, not just the gap.
- **Large model on a tiny dataset.** `xlm-roberta-large` overfits on <10 K per-language splits; the base model is often more honest. Pick by data size, not vanity.
- **Max-length mismatch across languages.** Agglutinative languages (Finnish, Turkish) produce longer subword sequences than English for the same semantic content. A `max_length=128` fine on English may silently truncate 30 % of Finnish inputs. Log truncation rate per language.
- **Reporting XNLI accuracy as macro-F1 or vice versa.** The XLM-R paper headline is XNLI accuracy; MARC/MultiEURLEX are reported with macro-F1. Mixing metrics across sources produces nonsense comparisons.
- **Forgetting the XLM-R paper's temperature.** XLM-R itself was pre-trained with `T=0.3` sampling over languages; some fine-tune recipes recommend `T=0.5`. State which and why — they are not interchangeable.
- **Public models trained on user text.** XLM-R is public, but downstream fine-tunes on multilingual user data carry the data-provenance / membership-inference risks the OWASP ML Top 10 names under "ML04 Membership Inference" and "ML06 AI Supply Chain". Document the training split per locale.

## 6. References

Primary sources used in this solution:

- Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale*. ACL. <https://aclanthology.org/2020.acl-main.747/> — the XLM-R paper; architecture, pre-training recipe, zero-shot transfer results, temperature sampling.
- Conneau, A., Lample, G. (2019). *Cross-lingual Language Model Pretraining*. NeurIPS. <https://arxiv.org/abs/1901.07291> — XLM, the direct predecessor to XLM-R; introduces the temperature-sampling schedule for multilingual MLM.
- Conneau, A. et al. (2018). *XNLI: Evaluating Cross-lingual Sentence Representations*. EMNLP. <https://aclanthology.org/D18-1269/> — the standard cross-lingual NLI benchmark and the source of the XLT evaluation protocol.
- Arivazhagan, N. et al. (2019). *Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges*. <https://arxiv.org/abs/1907.05019> — the industrial reference for temperature-based multilingual sampling.
- Hu, J. et al. (2020). *XTREME: A Massively Multilingual Multi-task Benchmark*. ICML. <https://arxiv.org/abs/2003.11080> — multi-task multilingual evaluation suite; where "transfer gap" is formally defined as a reporting standard.
- Keung, P. et al. (2020). *The Multilingual Amazon Reviews Corpus (MARC)*. EMNLP. <https://aclanthology.org/2020.emnlp-main.369/> — a drop-in aligned multilingual classification dataset.
- Chalkidis, I. et al. (2021). *MultiEURLEX — A multi-lingual and multi-label legal document classification dataset*. EMNLP. <https://aclanthology.org/2021.emnlp-main.559/> — multi-label multilingual classification benchmark.
- Hugging Face `transformers` — XLM-RoBERTa model card and tokeniser docs. <https://huggingface.co/docs/transformers/model_doc/xlm-roberta>.
- Hugging Face `datasets` — `interleave_datasets` with `probabilities` for temperature sampling. <https://huggingface.co/docs/datasets/process#interleave>.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — "ML02 Data Poisoning" especially applies when assembling multilingual datasets from crawled sources; different per-locale integrity guarantees.
- MITRE ATLAS — <https://atlas.mitre.org/> — adversarial-input techniques against classifiers, including script-swap and transliteration evasion that specifically target multilingual models.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — Measure function: report per-locale metrics; Govern function: document which locales are supported with human-labelled vs. zero-shot data and commit to a review path for the weaker ones.
