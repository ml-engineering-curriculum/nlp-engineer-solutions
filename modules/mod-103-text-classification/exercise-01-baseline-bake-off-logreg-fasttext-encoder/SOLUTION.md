# exercise-01 — Baseline Bake-off: Logreg, fastText, and Encoder

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-103-text-classification/exercises/exercise-01-baseline-bake-off-logreg-fasttext-encoder.md`.
> This solution targets the learning goals that filename implies: train three principled classification baselines — a TF-IDF + linear model, a `fastText` subword-averaging model, and a fine-tuned transformer encoder — on the **same split**, with the **same metrics** and **the same seed policy**, so the "encoder wins" or "logreg is enough" claim is defensible rather than folklore.

## 1. Solution overview

A classification "bake-off" fails when the models are compared on different splits, different label normalisations, or different evaluation seeds — the delta then measures the protocol, not the model. The deliverable is a small harness that pins the protocol and runs three baselines against it.

| Baseline | Representation | Classifier | Why keep it in the bake-off |
|---|---|---|---|
| **Logreg** | TF-IDF unigrams + bigrams | L2-regularised logistic regression | Strong baseline on topical, formal text; linear + interpretable (per-feature coefficients); CPU-only; trains in seconds |
| **fastText** | Averaged subword n-gram embeddings | Shallow softmax head (Joulin et al. 2017) | Robust to OOV, handles morphology, trains in minutes on a laptop; supports multi-label and hierarchical softmax natively |
| **Encoder** | `distilbert-base-uncased` or domain-matched checkpoint | Pooled `[CLS]` → linear head, fine-tuned | Captures semantic/compositional cues the other two miss; pays for it in GPU time and OOD sensitivity |

The shared scaffolding — stratified split, label normalisation, metric definitions, seed discipline — is more valuable than any single model. It is what turns the bake-off from a demo into a defensible decision.

**Metric contract.** Report macro-F1 as the headline (treats every class equally — important for imbalanced class mixes), plus accuracy and per-class F1. Add a 95 % bootstrap confidence interval on the test macro-F1 (10 000 resamples with replacement) so "encoder beats logreg" is not a 0.3-point coin flip. Follow NIST AI RMF (Measure) by reporting the metric with its protocol, not just the number.

## 2. Worked implementation

### 2.1 Project layout

```
bakeoff/
  data.py              # load + stratified split + label encoder
  metrics.py           # macro-F1, accuracy, bootstrap CI
  baselines/
    logreg.py          # TF-IDF + sklearn LogisticRegression
    fasttext_model.py  # fasttext.train_supervised wrapper
    encoder.py         # HF Trainer fine-tune of distilbert
  run.py               # CLI: --baseline {logreg,fasttext,encoder}
  tests/
    test_data.py
    test_metrics.py
    test_runs.py       # smoke: 100-row fixture, each baseline, <60 s
  requirements.txt
```

### 2.2 Shared data split and label encoding

```python
# data.py
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd

SEED = 20260101  # fixed; any change to the split must change this value.


@dataclass(frozen=True)
class Split:
    X_train: list[str]
    y_train: list[int]
    X_val:   list[str]
    y_val:   list[int]
    X_test:  list[str]
    y_test:  list[int]
    labels:  list[str]  # index -> label name


def load_and_split(path: str, text_col: str, label_col: str) -> Split:
    df = pd.read_csv(path)
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)

    labels = sorted(df[label_col].unique().tolist())
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    y = df[label_col].map(label_to_idx).to_numpy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        df[text_col].tolist(), y, test_size=0.10, stratify=y, random_state=SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1111, stratify=y_trainval, random_state=SEED,
    )
    return Split(X_train, list(y_train), X_val, list(y_val), X_test, list(y_test), labels)
```

Two decisions encoded here, both common review targets:

- **Stratified** split so rare classes are in every partition. Random splits leave entire classes out of the test set on imbalanced data.
- **90/10 test, then 90/10 of the remainder for val** — a single ~0.80/0.10/0.10 split, but computed as two calls so the test set is built first and never touched by model selection.

### 2.3 Metrics

```python
# metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0.0)


def bootstrap_ci(y_true, y_pred, metric=macro_f1, n: int = 10_000,
                 seed: int = 20260101, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (point_estimate, lower, upper) for a 1-alpha confidence interval."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    N = len(y_true)
    scores = np.empty(n, dtype=np.float64)
    for b in range(n):
        idx = rng.integers(0, N, size=N)
        scores[b] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(metric(y_true, y_pred)), float(lo), float(hi)
```

Bootstrap resampling with replacement treats the test set as a draw from an unknown population; the percentile interval is good enough for a bake-off and is the standard presentation in the NLP evaluation literature (e.g. Dror et al. 2018).

### 2.4 Baseline A — TF-IDF + logistic regression

```python
# baselines/logreg.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_logreg() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            dtype="float32",
        )),
        ("clf", LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear",
            max_iter=1000, class_weight="balanced", random_state=20260101,
        )),
    ])
```

Features worth defending in review:

- **Unigrams + bigrams** capture short multi-word cues ("not good", "customer service") that unigrams alone drop.
- **`sublinear_tf=True`** dampens term-frequency saturation (`1 + log(tf)`), which helps on long documents.
- **`class_weight="balanced"`** scales loss contributions by inverse class frequency, which is the cheap counterpart of resampling.
- **`solver="liblinear"`** is coordinate descent, best for small/medium sparse matrices; switch to `saga` if you cross ~10⁶ features or need elastic-net.

### 2.5 Baseline B — fastText

```python
# baselines/fasttext_model.py
import fasttext
import tempfile
from pathlib import Path


def _to_fasttext_file(texts, labels, label_names, out: Path) -> None:
    with out.open("w", encoding="utf-8") as f:
        for t, y in zip(texts, labels):
            t = t.replace("\n", " ").replace("\r", " ").strip()
            f.write(f"__label__{label_names[y]} {t}\n")


def train_fasttext(split, epoch: int = 25, lr: float = 0.5, wordNgrams: int = 2,
                   dim: int = 100, minn: int = 3, maxn: int = 6) -> "fasttext.FastText._FastText":
    with tempfile.TemporaryDirectory() as tmp:
        train_path = Path(tmp) / "train.txt"
        _to_fasttext_file(split.X_train, split.y_train, split.labels, train_path)
        model = fasttext.train_supervised(
            input=str(train_path),
            epoch=epoch, lr=lr, wordNgrams=wordNgrams,
            dim=dim, minn=minn, maxn=maxn, loss="softmax", thread=4,
        )
    return model


def predict_fasttext(model, texts, label_names) -> list[int]:
    cleaned = [t.replace("\n", " ") for t in texts]
    preds, _ = model.predict(cleaned, k=1)
    idx = {f"__label__{l}": i for i, l in enumerate(label_names)}
    return [idx[p[0]] for p in preds]
```

Why fastText earns a slot: Joulin et al. 2017 show the classifier matches deep models on topic classification tasks at a fraction of the compute. Its secret is averaging subword *n*-gram vectors (`minn=3`, `maxn=6`), which gives you a reasonable representation for OOV and morphologically rich tokens without a pre-trained model.

### 2.6 Baseline C — fine-tuned transformer encoder

```python
# baselines/encoder.py
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorWithPadding, set_seed,
)
from datasets import Dataset


MODEL_ID = "distilbert-base-uncased"


def train_encoder(split, output_dir: str, epochs: int = 3, lr: float = 2e-5,
                  batch_size: int = 16, max_length: int = 256, seed: int = 20260101):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    def tokenise(batch):
        return tok(batch["text"], truncation=True, max_length=max_length)

    ds_train = Dataset.from_dict({"text": split.X_train, "label": split.y_train}).map(tokenise, batched=True)
    ds_val   = Dataset.from_dict({"text": split.X_val,   "label": split.y_val}).map(tokenise, batched=True)
    ds_test  = Dataset.from_dict({"text": split.X_test,  "label": split.y_test}).map(tokenise, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(split.labels),
        id2label={i: l for i, l in enumerate(split.labels)},
        label2id={l: i for i, l in enumerate(split.labels)},
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
    )
    trainer.train()

    logits = trainer.predict(ds_test).predictions
    return model, tok, np.argmax(logits, axis=-1).tolist()
```

Encoder-side decisions worth defending:

- **`distilbert-base-uncased`** as the default: ~40 % smaller and ~60 % faster than `bert-base` with ~97 % of its GLUE performance (Sanh et al. 2019). Swap for a domain-matched checkpoint (e.g. `biobert`, `legal-bert`, `xlm-r-base` for multilingual) when the domain demands it.
- **`load_best_model_at_end=True`** combined with per-epoch eval picks the checkpoint that wins on `eval_loss` *before* the model starts overfitting. Picking by `eval_loss` rather than `eval_f1` is deliberate: loss is smoother and less noisy on small val sets.
- **`fp16`** is used only on GPU; CPU training stays in fp32.
- **`set_seed`** seeds Python, NumPy, and PyTorch at once. Without it the "encoder is +1.2 points" headline is one rerun away from vanishing.

### 2.7 Pulling it together

```python
# run.py
import argparse, json
from .data import load_and_split
from .metrics import macro_f1, bootstrap_ci
from .baselines import logreg, fasttext_model, encoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", choices=["logreg", "fasttext", "encoder"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--out", default="results.json")
    a = p.parse_args()

    split = load_and_split(a.data, a.text_col, a.label_col)

    if a.baseline == "logreg":
        pipe = logreg.build_logreg().fit(split.X_train, split.y_train)
        y_pred = pipe.predict(split.X_test).tolist()
    elif a.baseline == "fasttext":
        model = fasttext_model.train_fasttext(split)
        y_pred = fasttext_model.predict_fasttext(model, split.X_test, split.labels)
    else:
        _, _, y_pred = encoder.train_encoder(split, output_dir=f"runs/{a.baseline}")

    point, lo, hi = bootstrap_ci(split.y_test, y_pred, metric=macro_f1)
    out = {
        "baseline": a.baseline,
        "macro_f1": point,
        "macro_f1_ci95": [lo, hi],
        "n_test": len(split.y_test),
        "n_classes": len(split.labels),
        "seed": 20260101,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
```

## 3. Validation steps

1. **Smoke test on a 100-row fixture.** Each baseline must train and emit a `results.json` end-to-end in under a minute on CPU (encoder gets a tiny `max_length` and 1 epoch in the test fixture). This catches split-leaks, column-name drift, and import-time surprises.
2. **No train/test leakage.** Compute `set(split.X_train) & set(split.X_test)` and assert it is empty. This is the cheap version of near-duplicate detection; a stricter version uses MinHash (see mod-110 exercise-01) before splitting.
3. **Metric sanity.** For the logreg pipe, train and score twice with the same seed — macro-F1 must match to ≥6 decimal places. For the encoder, assert deterministic *tokeniser* behaviour (same inputs → identical input IDs), then run twice with `set_seed` and check that macro-F1 lands within ±0.3 points. Pure byte-level determinism on GPUs is not achievable with standard HF/Trainer settings — document the tolerance you accept.
4. **Bootstrap CI covers the point estimate.** Trivial sanity check: the returned `(lo, hi)` should bracket the point estimate every time.
5. **Confusion matrix review.** For each baseline, render `sklearn.metrics.confusion_matrix(normalize='true')` on the test set. If a class has >20 % off-diagonal into a single competing class, that pair is the most informative artefact of the run, not the headline macro-F1.
6. **Baseline ordering.** On topical classification datasets (AG News, 20-newsgroups) the expected ordering is `logreg ≲ fasttext ≲ encoder`. On noisy/short-text datasets (tweet sentiment) logreg sometimes beats fastText; that is informative, not a bug. Report what you observe; do not shoehorn the numbers toward "encoder wins".
7. **Seed sensitivity.** Run the encoder with ≥3 seeds and report the min/max/mean of test macro-F1. If the spread is larger than the gap to the next-best baseline, the headline is a coin flip. Flag it.
8. **Dependency pins.** Record versions of `scikit-learn`, `fasttext`, `transformers`, `torch`, and the dataset snapshot hash in the JSON output.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Shared protocol** | One split + one label encoder + one metric function reused by all baselines | Separate splits; headline numbers nonetheless compared | Each baseline rolls its own split; comparison is meaningless |
| **Seed discipline** | Global `SEED` constant; `set_seed` for the encoder; seed echoed in output JSON | Seed set for some baselines | No seed; results not reproducible |
| **Metric completeness** | Macro-F1 + accuracy + per-class F1 + bootstrap CI | Macro-F1 only | Accuracy only on an imbalanced set |
| **Class imbalance** | `class_weight="balanced"` or inverse-frequency sampling; metric choice defended | Mentions imbalance but does not act on it | Reports accuracy on 95 %-majority data |
| **Encoder hygiene** | `load_best_model_at_end`, warmup, weight decay, seed; checkpoint picked on val not test | Some of those; test used as val | No early stopping; test seen during training |
| **Confusion-matrix read-out** | Shown per baseline; cross-class errors analysed | Shown without analysis | Not shown |
| **Reproducibility** | Pinned deps + dataset hash + JSON output of every knob | Pinned deps only | Nothing pinned |
| **Responsible deployment** | Dataset licence + PII disclaimer per NIST AI RMF (Map/Govern); input-manipulation risks listed per OWASP ML Top 10 and MITRE ATLAS | Mentions one | Ignored |

## 5. Common mistakes

- **Comparing across different splits.** The most common bake-off failure: the paper's logreg number came from an 80/20 split and the student's encoder came from a 70/30 split. The 1.5-point gap is the split, not the model. Share the split object across every baseline.
- **Picking hyperparameters on the test set.** Any tuning on test (encoder learning rate, logreg `C`, fastText epochs) inflates the reported number. Tune on val, freeze, then score test exactly once.
- **Macro vs weighted vs micro confusion.** Accuracy and micro-F1 are the *same* number for single-label classification; weighted-F1 weights by class frequency and masks rare-class failure; macro-F1 weights every class equally. Use macro-F1 as the headline unless the deployment explicitly weights classes differently.
- **`fasttext` label-leakage bug.** `fasttext.train_supervised` expects `__label__<name> <text>` with no newlines in the text. A stray `\n` inside a field makes subsequent lines look like new examples; the trained model will not complain but val/test metrics will be subtly wrong. The `_to_fasttext_file` helper above normalises whitespace defensively.
- **Encoder OOM in production and silence at review time.** `max_length=256` is fine for sentences; it quietly truncates long documents, which changes the label for any example where the signal lives after token 256. Log the truncation rate and either widen `max_length`, chunk, or use a long-context encoder (see mod-108).
- **`class_weight="balanced"` with calibration goals.** Reweighting changes decision thresholds in ways that break Platt/isotonic calibration downstream (see exercise-03). Either reweight *or* calibrate — do not combine without correcting the inverse frequency out before fitting the calibrator.
- **Reporting one seed.** Encoder fine-tuning on small datasets has multi-point variance across seeds (Dodge et al. 2020). A single-seed "encoder beats logreg by 1.2 points" is often noise.
- **Dropping the dataset licence.** Training on scraped or user-generated text without a data-use notice makes the whole pipeline non-shippable and violates most compliance regimes. Record the licence next to the dataset hash in the output JSON.
- **No near-duplicate removal.** Many public datasets have near-duplicates across train/test (e.g. retweets, boilerplate). Without MinHash dedup the encoder "learns" the test set and the logreg does not; the gap is a leak metric, not a quality metric.

## 6. References

Primary sources used in this solution:

- Jurafsky, D., Martin, J. (2024). *Speech and Language Processing*, 3rd ed. draft, ch. 4 (*Naive Bayes, Text Classification, and Sentiment*) and ch. 5 (*Logistic Regression*). <https://web.stanford.edu/~jurafsky/slp3/> — standard treatment of TF-IDF + linear baselines and evaluation.
- Joulin, A., Grave, E., Bojanowski, P., Mikolov, T. (2017). *Bag of Tricks for Efficient Text Classification*. EACL. <https://aclanthology.org/E17-2068/> — the `fastText` classifier paper; motivates subword n-grams and softmax head.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL. <https://aclanthology.org/N19-1423/> — reference for `[CLS]`-pooled classification heads and the fine-tune recipe.
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT*. <https://arxiv.org/abs/1910.01108> — size/speed/quality tradeoff that justifies using DistilBERT as the default encoder.
- Dror, R., Baumer, G., Shlomov, S., Reichart, R. (2018). *The Hitchhiker's Guide to Testing Statistical Significance in NLP*. ACL. <https://aclanthology.org/P18-1128/> — practice for bootstrap CIs and when to use them.
- Dodge, J. et al. (2020). *Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping*. <https://arxiv.org/abs/2002.06305> — the seed-variance result cited in §5.
- `scikit-learn` user guide — <https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting> and `LogisticRegression` reference.
- Hugging Face `transformers` docs — <https://huggingface.co/docs/transformers/tasks/sequence_classification> — authoritative source for the `Trainer`-based fine-tune recipe.
- `fastText` library — <https://fasttext.cc/docs/en/supervised-tutorial.html> — official reference for `train_supervised`.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — "ML02 Data Poisoning" and "ML06 AI Supply Chain" apply when training data or pre-trained checkpoints are sourced externally.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for adversarial inputs and supply-chain compromise against classification services.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — Measure function: report metric + protocol + uncertainty, not a single number; Map function: document dataset provenance and licence.
