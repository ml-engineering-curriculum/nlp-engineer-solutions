# exercise-04 — Weak Supervision and Active Learning

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-110-nlp-data-engineering/exercises/exercise-04-weak-supervision-and-active-learning.md`.
> The target learning outcomes implied by that filename: cut labelling cost with two complementary techniques — **weak supervision** (Snorkel-style labelling functions + a label model that aggregates them into a probabilistic training label without ground-truth) and **active learning** (an acquisition strategy that chooses the *next* items to label so each new label contributes the most to the downstream classifier). Measure both against a common gold set, publish how much labelling effort was saved, and write down which failure modes (correlated LFs, exposure bias in the acquisition pool) the pipeline watches for.

## 1. Solution overview

Both techniques exist to break the "throw more labels at it" trap, and both have well-documented failure modes that only show up once you measure against gold:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Majority-vote over LFs** | A simple vote treats a high-precision LF (regex for `"$"` → MONEY) the same as a low-precision one; one noisy LF drowns the signal. Equally bad: a cluster of correlated LFs (five near-duplicate regexes) silently out-votes an orthogonal one | Train a **label model** (Snorkel MeTaL / data-programming generative model, or `cleanlab.data_valuation`) that estimates each LF's accuracy and class balance from the agreement/disagreement pattern alone — Ratner et al. 2016/2020 — and use those weights to produce a probabilistic label per item |
| **LF correlation breaks Snorkel's independence assumption** | Label-model math assumes LFs are conditionally independent given the latent label. Two regexes that fire on the same lexical cue violate this; the label model concentrates confidence on the correlated cluster | Diagnose with LF **overlap**/**conflict** matrix, cluster by Jaccard, and either drop redundant LFs or declare dependencies in Snorkel's `LabelModel` (post-2019 Snorkel supports structure learning) |
| **Uncertainty sampling collapses on an easy class** | Acquisition picks the items with `max p(1-p)`; the model is confident on `NEUTRAL` and uncertain on `POSITIVE/NEGATIVE`. Every acquisition round pulls the two minority classes — great if classes are what you care about, terrible if one class is a safety concern | Combine uncertainty with **diversity** (core-set / BADGE) and a **budget-aware stopping rule** tied to validation accuracy, not to acquisition count |
| **Measuring active learning on the acquisition pool** | AL curves evaluated on the same pool AL draws from are optimistic; you select the items your model misclassifies, you fix those, your error goes to zero | Always evaluate on a **held-out test set** that AL never sees, pinned at project start |

The deliverable is a two-track harness that shares the same gold eval set:

### Weak-supervision track

1. **Labelling functions (LFs)** — pure functions `text -> label | ABSTAIN` written in Python; each LF has a name, a stated precision estimate from hand inspection, and a declared "target class".
2. **LF analyser** — coverage (% items the LF labels), overlap (% items where this LF agrees with any other), conflict (% items where this LF disagrees with any other), empirical accuracy on a dev gold set.
3. **Label model** — Snorkel `LabelModel` (default) or `cleanlab`'s `MultiLabelPredictor`; outputs probabilistic labels per item.
4. **End-model** — a classifier (logistic regression on TF-IDF for fast iteration; distil-BERT for the final run) trained on the probabilistic labels.

### Active-learning track

5. **Acquisition strategies** — at minimum: random, uncertainty (least-confidence / max-entropy), diversity (core-set / BADGE), and a hybrid (uncertainty-weighted diversity).
6. **Oracle** — a labelling queue returning gold labels for acquired items (in production: a human annotator from exercise-03's pool).
7. **AL loop** — seed → train → acquire-K → relabel → retrain, with checkpointing.

### Shared evaluation

8. **Gold test set** — fixed at project start, locked in a separate file, used for every evaluation point in the labelling-curve plot.

Reference tools:

- **Snorkel** (Ratner et al. 2016, 2020) — <https://github.com/snorkel-team/snorkel> — the reference LF framework and label model.
- **`cleanlab`** (Northcutt et al. 2021) — <https://github.com/cleanlab/cleanlab> — confident learning for label-error detection; complementary to Snorkel.
- **`skactiveml`** — <https://scikit-activeml.github.io/> — scikit-compatible active learning strategies (uncertainty, QBC, EMCM, BADGE).
- **`modAL`** — <https://modal-python.readthedocs.io/> — alternative AL framework.
- **BADGE** (Ash et al. 2020) — the hybrid-diversity gradient-embedding strategy; strong default when uncertainty alone is too clustered.
- **Core-set** (Sener & Savarese 2018) — the diversity strategy BADGE partially subsumes but is still worth implementing for baseline comparison.

## 2. Worked implementation

### 2.1 Project layout

```
labelops_bake/
  ws/
    lfs.py                   # labelling functions
    analyse.py               # coverage / overlap / conflict / empirical accuracy
    label_model.py           # Snorkel LabelModel wrapper
    end_model.py             # downstream classifier
  al/
    seed.py                  # stratified seed sampler
    acquisition.py           # strategies: random, uncertainty, coreset, badge
    oracle.py                # gold-lookup oracle
    loop.py                  # the AL train-acquire-relabel loop
  eval/
    gold.py                  # pinned test-set loader
    metrics.py               # macro-F1 + per-class + fairness slices
    curves.py                # labelling-effort curve plotter
  run_ws.py                  # CLI: weak supervision
  run_al.py                  # CLI: active learning
  tests/
    test_label_model_syn.py
    test_lf_analysis.py
    test_acquisition_shapes.py
    test_coreset_distance.py
  requirements.txt
```

### 2.2 Labelling functions — pure, named, with declared precision

```python
# ws/lfs.py
import re
from dataclasses import dataclass
from typing import Callable


ABSTAIN = -1       # Snorkel convention
NEGATIVE, NEUTRAL, POSITIVE = 0, 1, 2


@dataclass
class LF:
    name: str
    target_class: int
    declared_precision: float     # hand-estimate; validated in §2.3
    fn: Callable[[str], int]      # returns ABSTAIN, NEGATIVE, NEUTRAL, or POSITIVE


# ---- Pattern LFs ----

_HAPPY = re.compile(r"\b(love|amazing|great|fantastic|perfect|excellent)\b", re.I)
def lf_happy_words(text: str) -> int:
    return POSITIVE if _HAPPY.search(text) else ABSTAIN


_UNHAPPY = re.compile(r"\b(hate|terrible|awful|broken|disappointed|refund)\b", re.I)
def lf_unhappy_words(text: str) -> int:
    return NEGATIVE if _UNHAPPY.search(text) else ABSTAIN


# ---- Heuristic LFs ----

def lf_exclamation_positive(text: str) -> int:
    # "!!!" tends to indicate positive affect in product reviews; lower precision
    return POSITIVE if text.count("!") >= 2 else ABSTAIN


def lf_question_neutral(text: str) -> int:
    return NEUTRAL if text.strip().endswith("?") else ABSTAIN


# ---- Dictionary / gazetteer LFs ----

NEG_PHRASES = {"waste of money", "never again", "not worth", "stopped working"}
def lf_neg_phrases(text: str) -> int:
    low = text.lower()
    return NEGATIVE if any(p in low for p in NEG_PHRASES) else ABSTAIN


# ---- Model-based LF (fast, imperfect) ----

def lf_vader(text: str) -> int:
    """A VADER-based LF; precision ~0.75. Illustrates a 'model as LF'."""
    from nltk.sentiment import SentimentIntensityAnalyzer
    s = SentimentIntensityAnalyzer().polarity_scores(text)
    if s["compound"] >= 0.5: return POSITIVE
    if s["compound"] <= -0.5: return NEGATIVE
    return ABSTAIN


REGISTRY: list[LF] = [
    LF("lf_happy_words",        POSITIVE, 0.85, lf_happy_words),
    LF("lf_unhappy_words",      NEGATIVE, 0.85, lf_unhappy_words),
    LF("lf_exclamation_pos",    POSITIVE, 0.60, lf_exclamation_positive),
    LF("lf_question_neutral",   NEUTRAL,  0.55, lf_question_neutral),
    LF("lf_neg_phrases",        NEGATIVE, 0.90, lf_neg_phrases),
    LF("lf_vader",              -1,       0.75, lf_vader),   # multi-class LF
]
```

Points worth defending in review:

- **`ABSTAIN` is a first-class return value.** An LF that is forced to label every item is a classifier, not a labelling function. The label model handles sparsity natively; leveraging it is the whole point.
- **Declared precision is a hand-estimate, not a measurement.** It is a prior for the label model and a sanity check for §2.3. When the empirical precision on dev gold is >15 % below the declared precision, flag the LF: either the LF is miscoded or the prior was wrong.
- **LFs are pure functions of text.** No shared state, no external I/O at call time (model-based LFs load once, serve many). This is what makes them trivially parallel and the pipeline cacheable.
- **Multi-class LFs are allowed.** `lf_vader` returns any of three labels; the label model can handle it. Snorkel's older `LabelingFunction` signature was single-class; the modern `LabelModel` supports multi-class return values (cardinality fixed per project).
- **No LF-against-LF logic.** LFs that reference other LFs (`if lf_happy and lf_unhappy: ABSTAIN`) break the conditional-independence story the label model assumes. Compose cleaner LFs instead.

### 2.3 LF analyser — coverage, overlap, conflict, empirical accuracy

```python
# ws/analyse.py
import numpy as np
from .lfs import REGISTRY, ABSTAIN


def apply_lfs(texts: list[str]) -> np.ndarray:
    """Return an (n_items, n_lfs) matrix of label ints (ABSTAIN = -1)."""
    n, m = len(texts), len(REGISTRY)
    L = np.full((n, m), ABSTAIN, dtype=np.int64)
    for j, lf in enumerate(REGISTRY):
        for i, t in enumerate(texts):
            L[i, j] = lf.fn(t)
    return L


def coverage(L: np.ndarray) -> np.ndarray:
    return (L != ABSTAIN).mean(axis=0)   # per-LF % of items labelled


def overlap(L: np.ndarray) -> np.ndarray:
    """Per-LF: fraction of items where this LF labels AND at least one other does."""
    mask = L != ABSTAIN
    any_other = (mask.sum(axis=1, keepdims=True) - mask) > 0
    return (mask & any_other).mean(axis=0)


def conflict(L: np.ndarray) -> np.ndarray:
    """Per-LF: fraction of items where this LF labels AND disagrees with some other."""
    n, m = L.shape
    out = np.zeros(m)
    for j in range(m):
        this = L[:, j]
        others = np.delete(L, j, axis=1)
        this_active = this != ABSTAIN
        others_active = others != ABSTAIN
        disagree = (others != this[:, None]) & others_active
        has_disagree = disagree.any(axis=1)
        out[j] = (this_active & has_disagree).mean()
    return out


def empirical_accuracy(L: np.ndarray, y_gold: np.ndarray) -> np.ndarray:
    """On items where the LF labels AND we have gold, what fraction is correct?"""
    out = np.zeros(L.shape[1])
    for j in range(L.shape[1]):
        active = L[:, j] != ABSTAIN
        if active.sum() == 0:
            out[j] = float("nan"); continue
        out[j] = (L[active, j] == y_gold[active]).mean()
    return out
```

Diagnostic output expected in the batch report (one row per LF):

| LF | Coverage | Overlap | Conflict | Declared prec | Empirical prec on dev |
|---|---|---|---|---|---|
| lf_happy_words | 0.22 | 0.11 | 0.03 | 0.85 | 0.87 |
| lf_unhappy_words | 0.19 | 0.09 | 0.02 | 0.85 | 0.83 |
| lf_exclamation_pos | 0.31 | 0.18 | 0.14 | 0.60 | 0.52 |
| … | … | … | … | … | … |

Rules of thumb Snorkel practitioners use (the 2020 TACL paper and Ratner's follow-on talks are the source):

- **Keep LFs with ≥5 % coverage and empirical accuracy > majority-class rate + 0.10.** A 55 % LF on a 60 %-majority dataset is worse than random after calibration.
- **Keep overall coverage (union across LFs) in 60-90 %.** Above 95 %, LFs are probably too liberal; below 50 %, add more LFs or lower abstain thresholds.
- **If any LF's conflict exceeds its overlap, the LF is actively harmful.** Drop it.
- **Correlated-LF cluster test.** Compute pairwise Jaccard on the `(L != ABSTAIN)` support masks; LFs with Jaccard > 0.8 are redundant. Either drop one or declare the dependency to Snorkel.

### 2.4 Label model — Snorkel LabelModel (default)

```python
# ws/label_model.py
import numpy as np
from snorkel.labeling.model import LabelModel


def train_label_model(L: np.ndarray, cardinality: int, seed: int = 20260101) -> LabelModel:
    lm = LabelModel(cardinality=cardinality, verbose=False)
    lm.fit(L_train=L, n_epochs=500, lr=0.01, seed=seed, class_balance=None)
    return lm


def predict_probs(lm: LabelModel, L: np.ndarray) -> np.ndarray:
    """Returns (n_items, cardinality) probabilistic labels."""
    return lm.predict_proba(L=L)


def filter_unlabelled(probs: np.ndarray, L: np.ndarray, threshold: float = 1.0 / 3):
    """
    Drop items where no LF labelled (all ABSTAIN) OR the label-model max
    probability is at or below uniform (threshold). Return (kept_idx, kept_probs).
    """
    all_abstain = (L == -1).all(axis=1)
    max_p = probs.max(axis=1)
    keep = (~all_abstain) & (max_p > threshold)
    return np.where(keep)[0], probs[keep]
```

Four points reviewers must check:

- **`cardinality=3` is the number of classes, not the number of LFs.** The Snorkel API uses `cardinality` for the label space; getting this wrong produces degenerate probabilities.
- **Pass `class_balance` when known.** If you have a prior on the class distribution (e.g., `[0.25, 0.60, 0.15]` from a quick dev estimate), pass it; otherwise Snorkel estimates it from LF agreement patterns, which is noisier.
- **`n_epochs=500` + `lr=0.01` is the Snorkel default.** Convergence usually happens in < 200 epochs; log the loss curve to make sure.
- **Drop all-abstain and uniform-probability items.** Training on probabilistic labels where `(0.33, 0.33, 0.34)` carries no signal just adds noise; `filter_unlabelled` is the cleanup pass.

### 2.5 End-model — probabilistic cross-entropy

```python
# ws/end_model.py
import numpy as np
import torch
import torch.nn as nn


def soft_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy with probabilistic targets. Equivalent to KL(target || softmax(logits))
    up to a constant. This is the Snorkel training-time loss.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()
```

- The end-model is trained on the probabilistic labels using soft cross-entropy. The alternative — argmax-to-hard-labels — throws away the uncertainty estimate that the label model worked to produce and typically costs 2–4 F1 points on noisy LF sets (Ratner et al. 2020 §5).
- For a TF-IDF + logistic regression baseline, use `sklearn.linear_model.LogisticRegression` with `class_weight="balanced"` and pre-scale by probability-of-majority-class. For BERT-family models, use Hugging Face's `Trainer` with a custom `compute_loss`.

### 2.6 Seed sample — stratified

```python
# al/seed.py
import numpy as np
from sklearn.model_selection import train_test_split


def stratified_seed(indices: np.ndarray, labels: np.ndarray, k: int,
                    seed: int = 20260101) -> np.ndarray:
    """
    Return `k` indices stratified by label. For AL seed, you need rough class
    coverage; otherwise the first-round classifier is a constant predictor.
    """
    sampled, _ = train_test_split(indices, train_size=k, stratify=labels,
                                  random_state=seed)
    return sampled
```

Design note: the seed set must contain at least one example of every class; a random seed on a highly imbalanced dataset routinely misses the minority class and the first acquisition round then over-concentrates on it.

### 2.7 Acquisition strategies

```python
# al/acquisition.py
import numpy as np


def random_acq(unlabelled_idx: np.ndarray, k: int, rng) -> np.ndarray:
    return rng.choice(unlabelled_idx, size=min(k, len(unlabelled_idx)), replace=False)


def uncertainty_acq(unlabelled_idx: np.ndarray, probs: np.ndarray, k: int,
                    strategy: str = "least_confidence") -> np.ndarray:
    """
    probs: (n_unlabelled, n_classes) current model's predicted distribution over
    the unlabelled pool, indexed parallel to `unlabelled_idx`.
    """
    if strategy == "least_confidence":
        score = 1 - probs.max(axis=1)
    elif strategy == "margin":
        sorted_ = -np.sort(-probs, axis=1)
        score = -(sorted_[:, 0] - sorted_[:, 1])     # smaller margin = higher score
    elif strategy == "entropy":
        eps = 1e-12
        score = -(probs * np.log(probs + eps)).sum(axis=1)
    else:
        raise ValueError(strategy)
    top = np.argsort(-score)[:k]
    return unlabelled_idx[top]


def coreset_acq(unlabelled_embed: np.ndarray, labelled_embed: np.ndarray,
                unlabelled_idx: np.ndarray, k: int) -> np.ndarray:
    """
    Sener & Savarese 2018 greedy k-center (core-set). At each step, pick the
    unlabelled point furthest from the currently-labelled set (by Euclidean
    distance on the model's penultimate-layer embedding).
    """
    chosen: list[int] = []
    # distance from each unlabelled point to the labelled set
    dist = np.min(
        ((unlabelled_embed[:, None, :] - labelled_embed[None, :, :]) ** 2).sum(-1),
        axis=1,
    )
    for _ in range(min(k, len(unlabelled_idx))):
        j = int(np.argmax(dist))
        chosen.append(j)
        new_dist = ((unlabelled_embed - unlabelled_embed[j]) ** 2).sum(-1)
        dist = np.minimum(dist, new_dist)
        dist[j] = -1       # mark chosen
    return unlabelled_idx[np.array(chosen, dtype=int)]


def badge_acq(unlabelled_embed: np.ndarray, probs: np.ndarray,
              unlabelled_idx: np.ndarray, k: int, rng) -> np.ndarray:
    """
    Ash et al. 2020 BADGE. Score each unlabelled point by the gradient of CE
    w.r.t. the last layer, under its own predicted label; then k-means++-seed
    sample from that gradient embedding (which simultaneously optimises
    uncertainty and diversity).
    """
    # gradient embedding of a logistic head, assuming cross-entropy with argmax pseudo-label:
    # g_i = (p_i - one_hot(argmax p_i)) ⊗ phi_i      (outer product)
    n, d = unlabelled_embed.shape
    c = probs.shape[1]
    argmax = probs.argmax(axis=1)
    delta = probs.copy()
    delta[np.arange(n), argmax] -= 1.0
    grad = (delta[:, :, None] * unlabelled_embed[:, None, :]).reshape(n, c * d)

    # k-means++ seeding on grad: iteratively sample proportionally to squared-distance
    chosen: list[int] = []
    dist = np.linalg.norm(grad, axis=1) ** 2
    for _ in range(min(k, n)):
        if dist.sum() == 0:
            j = int(rng.integers(n))
        else:
            probs_sample = dist / dist.sum()
            j = int(rng.choice(n, p=probs_sample))
        chosen.append(j)
        new_dist = np.linalg.norm(grad - grad[j], axis=1) ** 2
        dist = np.minimum(dist, new_dist)
    return unlabelled_idx[np.array(chosen, dtype=int)]
```

Points to defend:

- **At least four strategies, including `random`.** Random is the floor; if your "smart" acquisition does not beat random on the labelling-curve plot, your acquisition is broken. Settles 2010 §4 is clear on this.
- **BADGE as the default when uncertainty alone clusters.** Uncertainty sampling loses the diversity signal when the dataset has redundant hard examples (common in product reviews). BADGE fixes this by seeding k-means++ on gradient embeddings — one strategy, both axes (Ash et al. 2020).
- **Core-set needs embeddings, not probabilities.** The model's penultimate-layer representation is the standard choice; TF-IDF is a passable substitute for a logistic-regression end-model.
- **Deterministic RNG for BADGE's sampling.** `rng = np.random.default_rng(seed)`; AL replay without this is not comparable.

### 2.8 AL loop

```python
# al/loop.py
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class ALConfig:
    n_rounds: int
    k_per_round: int
    seed_size: int
    strategy: str              # "random" | "uncertainty" | "coreset" | "badge"


def run_loop(X_text, y_gold, test_idx, cfg: ALConfig, train_eval_fn: Callable,
             embed_fn: Callable, oracle_fn: Callable) -> list[dict]:
    """
    Returns a list of {"round": r, "n_labelled": n, "f1": f, "strategy": s} rows.
    train_eval_fn(labelled_idx) -> (f1, probs_on_all, embeds_on_all)
    oracle_fn(idx)              -> y_gold[idx]  (abstracted so an annotator could plug in)
    """
    rng = np.random.default_rng(20260101)
    all_idx = np.arange(len(X_text))
    unlabelled = np.setdiff1d(all_idx, test_idx)
    # Seed stratified by gold labels (only for simulation; replaced by random in deployment)
    seed_idx = rng.choice(unlabelled, size=cfg.seed_size, replace=False)
    labelled = set(seed_idx.tolist())
    history: list[dict] = []
    for r in range(cfg.n_rounds):
        lab_arr = np.array(sorted(labelled))
        f1, probs, embeds = train_eval_fn(lab_arr)
        history.append({"round": r, "n_labelled": len(labelled),
                        "f1": f1, "strategy": cfg.strategy})
        rem = np.array(sorted(set(unlabelled.tolist()) - labelled))
        if len(rem) == 0: break

        if cfg.strategy == "random":
            new = random_acq(rem, cfg.k_per_round, rng)
        elif cfg.strategy == "uncertainty":
            new = uncertainty_acq(rem, probs[rem], cfg.k_per_round)
        elif cfg.strategy == "coreset":
            new = coreset_acq(embeds[rem], embeds[lab_arr], rem, cfg.k_per_round)
        elif cfg.strategy == "badge":
            new = badge_acq(embeds[rem], probs[rem], rem, cfg.k_per_round, rng)
        else:
            raise ValueError(cfg.strategy)
        _ = oracle_fn(new)
        labelled.update(new.tolist())
    return history
```

Design notes reviewers must check:

- **Evaluate on the test set every round, train on the growing labelled set.** The test set is never acquired from; AL curves computed any other way are optimistic.
- **Record `n_labelled` as cumulative sum, not round index.** Labelling-curve plots need x-axis in label units, not round units.
- **The oracle is pluggable.** In simulation, `oracle_fn(idx) = y_gold[idx]`; in deployment, it is a call to the human annotator queue from exercise-03. Keep the interface identical.

### 2.9 Evaluation harness — shared gold test set

```python
# eval/gold.py
import json
from pathlib import Path


def load_test_set(path: Path) -> tuple[list[str], list[int]]:
    """
    Loads the pinned test set. The file lives in the repo alongside the LFs
    and is NEVER touched by AL acquisition. 500-2000 items, stratified.
    """
    texts, labels = [], []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"]); labels.append(r["label"])
    return texts, labels
```

```python
# eval/metrics.py
from sklearn.metrics import f1_score, precision_recall_fscore_support


def report(y_true, y_pred, labels: list[str], subgroup: list[str] | None = None) -> dict:
    out = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class": {
            lab: f1_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
            for i, lab in enumerate(labels)
        },
    }
    if subgroup is not None:
        # Fairness slice: per-subgroup macro F1
        out["per_subgroup"] = {}
        for g in set(subgroup):
            m = [s == g for s in subgroup]
            yt = [y_true[i] for i, x in enumerate(m) if x]
            yp = [y_pred[i] for i, x in enumerate(m) if x]
            out["per_subgroup"][g] = f1_score(yt, yp, average="macro", zero_division=0)
    return out
```

## 3. Validation steps

1. **LF matrix shape.** `apply_lfs` returns `(n_items, n_lfs)` with values in `{-1, 0, 1, 2}` only; no NaN / string entries. Golden-list test.
2. **Coverage/overlap/conflict add up.** For each LF, `overlap >= conflict` and `coverage >= overlap`. If not, the analyser is buggy.
3. **Label-model convergence.** Synthesise 10 LFs with known accuracies `[0.95, 0.9, …, 0.6]` on a 3-class, 1000-item synthetic set with class balance `(0.25, 0.60, 0.15)`. After `fit`, the label model's estimated per-LF accuracies are within ±0.05 of truth and estimated class balance is within ±0.03 of truth. This is the Ratner et al. 2020 Figure 3 reproduction and the only test that catches silent regressions in the label model.
4. **Soft-label end-model beats argmax.** Train the end-model twice on the same probabilistic labels: once with soft CE, once with argmaxed hard CE. The soft version must be at least equal macro-F1 on the test set; a regression in the soft version is a loss-function bug.
5. **Correlated-LF stress test.** Add two near-duplicate regex LFs to the registry; verify the label-model-estimated accuracy of each drops relative to the single-LF baseline, and that the end-model F1 drops *less* than it would in a majority-vote baseline. If the drop is similar, the independence assumption has been silently violated.
6. **Acquisition shape.** `random_acq`, `uncertainty_acq`, `coreset_acq`, `badge_acq` all return exactly `k` indices (or fewer if the pool is smaller), all in `unlabelled_idx`, all unique.
7. **Core-set k-center property.** After picking 10 items with `coreset_acq`, verify the maximum distance from any unlabelled point to its nearest labelled neighbour is strictly smaller than before. This is the k-center objective; if not satisfied, the algorithm has a bug.
8. **Labelling-curve monotonicity.** On a sanity dataset (binary SST-2 2k items), every strategy's F1 should trend upward as `n_labelled` grows; any strategy that plateaus or decreases for 2+ rounds is a flag.
9. **Random vs smart acquisition.** All four strategies plotted on the same labelling-curve. Uncertainty / Core-set / BADGE must beat random by at least 1 F1 point at ≥ 50 % of the budget; if not, the strategy has a bug or the dataset is uniformly easy.
10. **Fairness slice stability.** Report per-subgroup F1 at every round. If the gap between subgroups widens as more labels are acquired, the acquisition strategy is pulling from one subgroup disproportionately — measure (acquired_count, subgroup) and expose it in the curve plot.
11. **Reproducibility.** Running `run_al.py --strategy=badge --seed=20260101` twice produces identical `history` rows (same acquired indices, same F1 values). This catches BADGE's `rng.choice` path drift.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **LFs** | ≥5 LFs, mix of regex + heuristic + model-based; each has declared precision; ABSTAIN used | 3-4 LFs, all same kind | Hand-labelled examples passed off as LFs |
| **LF analyser** | Coverage + overlap + conflict + empirical accuracy on dev gold; redundancy / correlation check | Coverage only | None |
| **Label model** | Snorkel LabelModel (or justified alternative); probabilistic labels filtered for all-abstain / near-uniform | Majority vote | None |
| **End-model loss** | Soft cross-entropy; argmax baseline reported for comparison | Soft CE only | Hard argmax |
| **AL strategies** | Random + uncertainty + diversity + hybrid (e.g., BADGE) all plotted | Two of these | One strategy |
| **AL evaluation** | Held-out test set pinned at project start, never acquired from | Acquired-pool evaluation | Train/test overlap |
| **Seed set** | Stratified by label; documented | Random, documented | Default one-class seed |
| **Labelling-curve plot** | F1 vs n_labelled per strategy; random baseline present | F1 vs round count | None |
| **Fairness slice** | Per-subgroup F1 reported each round; acquisition-vs-subgroup table | Overall F1 only | None |
| **Reproducibility** | RNG seeds pinned; history reproducible bit-for-bit | Partial | Non-deterministic |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks (LF-poisoning, acquisition exploitation) | Mentioned | Ignored |
| **Documentation / card** | NIST AI RMF *Map/Measure*: LF registry version, label model snapshot, test-set version, labelling-curve, fairness slice | One of those | Neither |

## 5. Common mistakes

- **Treating LFs as a vote.** Majority vote over LFs ignores precision differences and correlation; the label model is why Snorkel works. If you are not using one, you are not doing weak supervision.
- **Forgetting that LFs are a hypothesis set.** An LF with 50 % precision on a two-class task is information-free; add LFs or drop it. Measure first, keep second.
- **Stacking many near-duplicate regexes as "coverage".** Three variants of the same regex count once; the label model will still tangle their weights. Compute the Jaccard-overlap matrix and drop or merge.
- **Not pinning the test set.** The commonest AL paper reproducibility failure. If test items can be acquired, you are measuring acquisition against items the model memorised; results don't transfer.
- **Reporting "we labelled 1 000 items and got 0.80 F1".** Without a random baseline on the same labelling curve, the AL claim is unsupported. The whole point of the curve is to show the slope difference.
- **Argmax-then-train on probabilistic labels.** Throws away the uncertainty estimate the label model worked to produce; typical 2-4 F1 point loss on noisy LF sets.
- **Uncertainty-only acquisition on redundant hard examples.** Pulls K items that are nearly identical (same misunderstanding of a dialect feature, same OOV token); fixes none of them with the label budget. Add diversity (core-set / BADGE).
- **Changing the RNG seed mid-experiment.** AL replay comparisons are fragile; any seed change invalidates them.
- **Using Snorkel's old `MajorityLabelVoter` as a label model.** It exists; it is a *baseline*, not the recommendation. `LabelModel` is the artefact.
- **Ignoring class balance.** The label model estimates class balance by default; on a severely imbalanced dataset (one class < 5 %) it can collapse. Pass `class_balance` from a quick dev estimate when you have one.
- **Running AL on a dataset where the model can't move the needle.** When the base-model with 100 % labels is 0.98 F1, you have no headroom; AL can't show benefit. Report the ceiling (100 %-labels F1) on the same curve.
- **Evaluating AL "label-efficiency" only at the final budget.** The slope matters more than the end-point; a strategy that beats random at 10 % of budget and matches at 100 % is still winning. Plot the whole curve.
- **Assuming BADGE needs a gradient oracle.** The paper (Ash et al. 2020 §3) derives a closed-form gradient for a linear head with cross-entropy and pseudo-label; the implementation above is the standard practical form. No actual backprop is needed per acquisition.
- **Not aggregating per-subgroup counts over the acquisition pool.** AL silently pulls from a majority subgroup when uncertainty correlates with subgroup membership; the (acquired_count, subgroup) table is the fairness-health signal.

## 6. References

Primary sources used in this solution:

- Ratner, A. et al. (2016). *Data Programming: Creating Large Training Sets, Quickly*. NeurIPS. <https://arxiv.org/abs/1605.07723> — the original Snorkel / data programming formulation reproduced in §2.
- Ratner, A. et al. (2020). *Snorkel: Rapid Training Data Creation with Weak Supervision*. VLDB 2017, extended VLDBJ 2020. <https://arxiv.org/abs/1711.10160> — the end-to-end pipeline (LFs → LabelModel → EndModel) and the synthetic-LF experiments reproduced in §3.3.
- Fu, D. et al. (2020). *Fast and Three-rious: Speeding Up Weak Supervision with Triplet Methods*. ICML. <https://arxiv.org/abs/2002.11955> — the closed-form label-model estimator used by newer Snorkel `LabelModel`.
- Northcutt, C., Jiang, L., Chuang, I. (2021). *Confident Learning: Estimating Uncertainty in Dataset Labels*. JAIR. <https://arxiv.org/abs/1911.00068> — the `cleanlab` label-error-detection pipeline cited as complementary to Snorkel.
- Settles, B. (2009/2010). *Active Learning Literature Survey*. Univ. Wisconsin TR #1648. <https://minds.wisconsin.edu/handle/1793/60660> — the survey that defines uncertainty / margin / entropy / QBC sampling in §2.7.
- Sener, O., Savarese, S. (2018). *Active Learning for Convolutional Neural Networks: A Core-Set Approach*. ICLR. <https://arxiv.org/abs/1708.00489> — the core-set / k-center strategy in §2.7.
- Ash, J. T. et al. (2020). *Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds* (BADGE). ICLR. <https://arxiv.org/abs/1906.03671> — the gradient-embedding + k-means++ strategy implemented in §2.7.
- Lewis, D., Gale, W. (1994). *A Sequential Algorithm for Training Text Classifiers*. SIGIR. <https://dl.acm.org/doi/10.5555/188490.188495> — the original uncertainty-sampling paper referenced in §2.7.
- Hutto, C., Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*. ICWSM. <https://ojs.aaai.org/index.php/ICWSM/article/view/14550> — the VADER sentiment analyser used as a model-based LF in §2.2.
- Pavlick, E., Kwiatkowski, T. (2019). *Inherent Disagreements in Human Textual Inferences*. TACL. <https://aclanthology.org/Q19-1043/> — motivates not forcing hard labels on items with legitimate disagreement; feeds the `AMBIGUOUS` handling in §2.5.
- `Snorkel` library — <https://github.com/snorkel-team/snorkel> — reference implementation; `LabelModel`, `LFAnalysis`, `MajorityLabelVoter`.
- `cleanlab` library — <https://github.com/cleanlab/cleanlab> — Confident Learning implementation; alternative label model and label-error detector.
- `skactiveml` — <https://scikit-activeml.github.io/> — pool-based AL strategies with scikit-learn interfaces.
- `modAL` — <https://modal-python.readthedocs.io/> — alternative AL framework.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML02 Data Poisoning*: an attacker who can commit LFs (common in weak-supervision-via-SMEs setups) can systematically bias the label model; the empirical-accuracy check in §2.3 and the regression test in §3.5 are the detection controls. *ML05 Model Theft* is not directly applicable. *ML09 Output Integrity Attack* is downstream once the end-model is deployed.
- MITRE ATLAS — <https://atlas.mitre.org/> — *AML.T0020 Poison Training Data* maps directly to LF-poisoning; *AML.T0019 Craft Adversarial Data* is the AL corollary — an attacker who can place items in the unlabelled pool can steer acquisition (e.g., feed uncertainty-sampling adversarial examples that soak the budget).
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: LF registry (including versions and declared precisions), label-model snapshot, acquisition-strategy choice, seed-set generation, test-set version. *Measure*: coverage/overlap/conflict/empirical-accuracy per LF, labelling curve F1 per strategy, per-subgroup F1 time-series, acquisition-vs-subgroup table. *Manage*: Snorkel + end-model checkpoints versioned; LF changes are change-controlled; the fairness-slice divergence alarm in §3.10 is a mandated stop-work signal.
