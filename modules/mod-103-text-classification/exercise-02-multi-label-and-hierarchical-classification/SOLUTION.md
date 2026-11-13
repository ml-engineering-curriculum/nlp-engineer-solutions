# exercise-02 — Multi-Label and Hierarchical Classification

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-103-text-classification/exercises/exercise-02-multi-label-and-hierarchical-classification.md`.
> This solution targets the learning goals that filename implies: move from one-of-K single-label classification to the two settings that production catalogues actually look like — **multi-label** (an example can carry zero, one, or many labels) and **hierarchical** (labels are organised in a tree or DAG; parents must be consistent with children).

## 1. Solution overview

The jump from single-label to multi-label is more than swapping softmax for sigmoid: it reshapes the loss, the metrics, the thresholding step, and the way you report numbers.

| Setting | Output head | Loss | Decision rule | Dominant metric |
|---|---|---|---|---|
| **Single-label** | Softmax(K) | Cross-entropy | `argmax` | Accuracy / macro-F1 |
| **Multi-label** | Sigmoid(K) | Binary cross-entropy summed per class | Per-class threshold | Micro/macro-F1, mAP, label-ranking average precision |
| **Hierarchical** | Sigmoid(K) over the full taxonomy | BCE + parent-consistency constraint | Per-class threshold + tree-consistent post-processing | Hierarchical F1 (Kiritchenko et al. 2006) |

**Multi-label.** Each class is its own binary decision — the trick is that the decisions are not independent (correlated labels improve each other) and the class imbalance per label is usually extreme (most labels fire on <1 % of examples). Binary cross-entropy with sigmoid activations is the standard head; thresholding each class independently on val-F1 is the standard deploy recipe.

**Hierarchical.** A taxonomy — ICD-10 diagnoses, product categories, arXiv subject classes — carries a parent/child structure. Two consistency decisions:

1. *Mandatory ancestor inclusion.* If `(Electronics → Phones → iPhone)` is a path and the model predicts `iPhone`, it must also predict `Electronics` and `Phones`. A "flat" multi-label head trained without this constraint routinely violates it (Silla & Freitas 2011).
2. *Metric choice.* Hierarchical F1 (hF1) credits partial matches up the tree: predicting the parent of the true class is better than predicting a sibling. Flat micro-F1 treats both as zero and lies about the model's usefulness.

The deliverable is one classification head trained with sigmoid + BCE plus two post-processors: a per-class threshold tuner for the multi-label case and a tree-consistency repairer for the hierarchical case.

## 2. Worked implementation

### 2.1 Project layout

```
multilabel/
  taxonomy.py          # tree ingestion + ancestor/descendant helpers
  data.py              # multi-hot encoding
  model.py             # transformer encoder with sigmoid head (BCE loss)
  thresholds.py        # per-class threshold tuning on val
  post.py              # tree-consistent prediction repair
  metrics.py           # micro/macro-F1, mAP, hierarchical F1
  run.py
  tests/
    test_taxonomy.py
    test_thresholds.py
    test_metrics.py
    test_end_to_end.py
```

### 2.2 Encoding labels and the taxonomy

```python
# taxonomy.py
from dataclasses import dataclass, field


@dataclass
class Taxonomy:
    """Tree or DAG over label strings; root is implicit."""
    parents: dict[str, list[str]] = field(default_factory=dict)  # child -> [parents]
    labels: list[str] = field(default_factory=list)

    def add_edge(self, parent: str, child: str) -> None:
        self.parents.setdefault(child, []).append(parent)
        for n in (parent, child):
            if n not in self.labels:
                self.labels.append(n)

    def ancestors(self, label: str) -> set[str]:
        out: set[str] = set()
        stack = list(self.parents.get(label, []))
        while stack:
            p = stack.pop()
            if p not in out:
                out.add(p)
                stack.extend(self.parents.get(p, []))
        return out

    def label_index(self) -> dict[str, int]:
        return {l: i for i, l in enumerate(self.labels)}
```

```python
# data.py
import numpy as np
from .taxonomy import Taxonomy


def multihot(label_sets: list[list[str]], tax: Taxonomy) -> np.ndarray:
    """One row per example; columns match tax.labels order.

    Each positive label pulls in all of its ancestors. This is the
    'mandatory ancestor' rule from Silla & Freitas 2011 — apply it at
    encode time so train / val / test labels are all consistent.
    """
    idx = tax.label_index()
    Y = np.zeros((len(label_sets), len(tax.labels)), dtype=np.float32)
    for i, labels in enumerate(label_sets):
        for lbl in labels:
            if lbl not in idx:
                raise KeyError(f"label {lbl!r} not in taxonomy")
            Y[i, idx[lbl]] = 1.0
            for anc in tax.ancestors(lbl):
                Y[i, idx[anc]] = 1.0
    return Y
```

### 2.3 Model head and loss

```python
# model.py
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


def build_model(model_id: str, num_labels: int,
                pos_weight: torch.Tensor | None = None) -> nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )
    if pos_weight is not None:
        model.config.pos_weight = pos_weight.tolist()
    return model
```

HF's `problem_type="multi_label_classification"` swaps the Trainer's default loss to `BCEWithLogitsLoss` over the full logit vector; the model still outputs raw logits and you apply `sigmoid` at inference time. See the HF task docs for the full recipe.

**Positive-class reweighting.** For imbalanced labels, pass `pos_weight` to `BCEWithLogitsLoss`:

```python
# Weight per class: N_neg / N_pos, clipped.
import numpy as np
counts = Y_train.sum(axis=0)
pos_weight = torch.tensor(
    np.clip((len(Y_train) - counts) / np.maximum(counts, 1), 1.0, 50.0),
    dtype=torch.float32,
)
```

This is BCE's analogue of `class_weight="balanced"` and is the standard way to keep rare labels from being predicted "always off". It changes the calibration of the logits — see exercise-03 for the implications.

### 2.4 Per-class threshold tuning

```python
# thresholds.py
import numpy as np
from sklearn.metrics import f1_score


def tune_thresholds(y_true: np.ndarray, y_score: np.ndarray,
                    grid: np.ndarray | None = None) -> np.ndarray:
    """Return per-class thresholds that maximise F1 on val.

    y_true  : (N, K) multi-hot ground truth
    y_score : (N, K) sigmoid probabilities
    grid    : candidate thresholds; default is 0.05..0.95 step 0.01
    """
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.01)
    K = y_true.shape[1]
    best = np.full(K, 0.5, dtype=np.float32)
    for k in range(K):
        # Skip classes with no positives in val — keep the 0.5 default.
        if y_true[:, k].sum() == 0:
            continue
        scores = [f1_score(y_true[:, k], (y_score[:, k] >= t).astype(int),
                           zero_division=0.0) for t in grid]
        best[k] = float(grid[int(np.argmax(scores))])
    return best


def apply_thresholds(y_score: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (y_score >= thresholds[None, :]).astype(int)
```

Three decisions encoded here, each a frequent review target:

- **Per-class thresholds**, not a single global threshold. Rare classes need low thresholds (to fire at all); dominant classes need high thresholds (to suppress false positives).
- **F1 as the tuning target**, not accuracy. Accuracy at 1-of-K is >99 % trivially on sparse labels.
- **Zero-positive classes keep the 0.5 default.** With no val positives there is no F1 to optimise; collapsing to 0.0 would produce an "always-fire" classifier on test if a positive ever appears.

### 2.5 Tree-consistency repair

Even with mandatory ancestors in the training labels, the model can still predict children without their parents at inference time. Repair the prediction post-hoc:

```python
# post.py
import numpy as np
from .taxonomy import Taxonomy


def repair_upward(Y_pred: np.ndarray, tax: Taxonomy) -> np.ndarray:
    """If a label is predicted, force all of its ancestors on."""
    idx = tax.label_index()
    Y = Y_pred.copy()
    for lbl, i in idx.items():
        anc = [idx[a] for a in tax.ancestors(lbl)]
        if not anc:
            continue
        mask = Y[:, i] == 1
        Y[np.ix_(mask, anc)] = 1
    return Y


def repair_downward_prune(Y_pred: np.ndarray, tax: Taxonomy) -> np.ndarray:
    """If an ancestor is NOT predicted, suppress its descendants.

    Strictly symmetric to repair_upward. Use one or the other depending on
    which failure you'd rather make: upward = precision loss on parents,
    downward = recall loss on children.
    """
    idx = tax.label_index()
    Y = Y_pred.copy()
    changed = True
    while changed:
        changed = False
        for lbl, i in idx.items():
            anc = [idx[a] for a in tax.ancestors(lbl)]
            if not anc:
                continue
            off = (Y[:, anc] == 0).any(axis=1)
            new_zeros = off & (Y[:, i] == 1)
            if new_zeros.any():
                Y[new_zeros, i] = 0
                changed = True
    return Y
```

The two directions encode different operational preferences. Upward repair ("predict all my ancestors") favours recall on the general-purpose "what topic is this" question; downward pruning ("my ancestor disagrees — drop me") favours precision on the "fire an action" question. State which one you chose and why.

### 2.6 Metrics

```python
# metrics.py
import numpy as np
from sklearn.metrics import f1_score, average_precision_score
from .taxonomy import Taxonomy


def micro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="micro", zero_division=0.0)


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0.0)


def mAP(y_true, y_score) -> float:
    """Mean average precision across labels. Threshold-free."""
    return float(average_precision_score(y_true, y_score, average="macro"))


def hierarchical_f1(y_true: np.ndarray, y_pred: np.ndarray, tax: Taxonomy) -> float:
    """Kiritchenko et al. 2006 hF1 with ancestor expansion.

    Expand each predicted and true label set with its ancestors, then
    compute micro-precision and -recall across the expanded sets.
    """
    idx = tax.label_index()
    anc = {l: {idx[a] for a in tax.ancestors(l)} | {i} for l, i in idx.items()}
    tp = fp = fn = 0
    for yt, yp in zip(y_true, y_pred):
        t = set()
        p = set()
        for k in np.where(yt == 1)[0]:
            t |= anc[tax.labels[k]]
        for k in np.where(yp == 1)[0]:
            p |= anc[tax.labels[k]]
        tp += len(t & p)
        fp += len(p - t)
        fn += len(t - p)
    if tp == 0:
        return 0.0
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    return 2 * P * R / (P + R)
```

The three metrics answer three different questions. **Micro-F1** is what the training loss is minimising; **macro-F1** says whether rare labels work; **mAP** is threshold-free, so it compares model quality across threshold choices. **Hierarchical F1** is the one to publish for a taxonomy deployment — it is the only number that distinguishes "guessed the sibling" from "guessed the parent".

## 3. Validation steps

1. **Ancestor-closure invariant.** On every multi-hot row of `Y_train`, assert: for every column `k` that is 1, every ancestor column is also 1. If the invariant breaks, either the taxonomy is a DAG with a cycle or `multihot()` was bypassed.
2. **Loss selection.** Instantiate the model with `problem_type="multi_label_classification"`, train for one step on a 10-example batch, and assert the loss is `BCEWithLogitsLoss` (via `model(**batch).loss` existing and being finite). Default single-label CE on multi-hot targets will train without erroring and then catastrophically underfit.
3. **Threshold tuning is monotone in val F1.** Compare pre-tuning (all 0.5) and post-tuning micro-F1 on the val split; post-tuning must be ≥ pre-tuning by construction. If it is not, there is a bug in the grid search or a leak between train-time pos_weight and eval.
4. **Tree-consistency after repair.** After `repair_upward`, assert the ancestor-closure invariant also holds on `Y_pred`. After `repair_downward_prune`, assert that no row has a child predicted without all its ancestors.
5. **Hierarchical F1 ≥ flat micro F1 on the same predictions.** hF1 counts the ancestor overlap; whenever your predictions include at least some parent agreement, hF1 ≥ flat. If hF1 < flat, the `ancestors` function is wrong or your taxonomy is a cycle.
6. **Rare-label coverage.** For every class with ≥5 positives in val, assert the per-class threshold produced at least one positive prediction on val. Classes with no positive predictions are "dead" — the threshold is above every observed score.
7. **Bootstrap CI on hF1.** 10 000 resamples with replacement on the test set; publish the interval alongside the point estimate.
8. **Ablation: upward repair vs downward prune vs no repair.** Report hF1 for all three on the same predictions; one of them has to be the deployment choice and the comparison is the evidence.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Head + loss** | Sigmoid head, BCE-with-logits, HF `problem_type="multi_label_classification"`; raw logits exposed | Correct head, wrong loss name used in logs | Softmax + CE on multi-hot targets |
| **Imbalance** | `pos_weight` or focal loss, bounded by class count; justified in the writeup | Mentioned, not applied | Rare classes dead on arrival |
| **Thresholds** | Per-class, tuned on val, zero-positive policy explicit | Single global threshold | Default 0.5 everywhere |
| **Tree consistency** | Mandatory ancestors at encode time *and* a repair step (upward or downward) at predict time | One of the two | Flat multi-label; children without parents on 10 %+ of predictions |
| **Metrics** | Micro + macro + mAP + hF1 with CIs | Micro + macro only | Accuracy on multi-label |
| **Threshold leakage** | Thresholds tuned on val only; test seen once | Tuned on train + val | Tuned on test |
| **Zero-positive labels** | Policy named (default 0.5, "always off", or drop) and tested | Unclear behaviour | Threshold collapses to 0 → always-fire on test |
| **Responsible deployment** | Dataset licence + per-label fairness audit per NIST AI RMF (Measure); ATLAS/OWASP risks named when taxonomies are internet-facing | Mentions one | Ignored |

## 5. Common mistakes

- **Softmax + CE on multi-hot targets.** Trains without error, produces a model that activates at most one class, misses every "multi" example. The loss on a two-positive row looks the same as a one-positive row where the softmax point is between the two classes. Use sigmoid + BCE.
- **Single global threshold.** Caps rare-class recall and dominant-class precision simultaneously. Tune per class on val. For latency-constrained serving, cache the vector once at deploy time; it is not a runtime cost.
- **Tuning thresholds on test.** The same classic leak as exercise-01, amplified: with thousands of per-class knobs, test-leakage can gain many F1 points that evaporate in production.
- **Reporting flat micro-F1 on a hierarchical problem.** Micro-F1 counts siblings and parents as equally wrong; hierarchical F1 credits ancestor agreement. Publishing flat micro on a tree understates the model's usefulness and misleads product decisions.
- **Forgetting ancestor closure.** If `multihot()` does not add ancestors, the loss "teaches" the model that children do not imply parents. The model then outputs child-without-parent predictions whose downstream catalog pipeline rejects as inconsistent. Fix at encode time, not just at post-process.
- **Pos-weight runaway.** `pos_weight = N_neg / N_pos` on a label with 3 positives out of 100 000 becomes ≈33 000; the model learns to predict that label on everything to minimise loss. Clip it (`[1, 50]` is a common range), or switch to focal loss.
- **Confusing mAP with mean per-example AP.** `sklearn.metrics.average_precision_score(..., average="macro")` computes per-label AP and averages; `"samples"` computes per-example AP. On skewed label distributions these diverge by a lot; name which one you reported.
- **Train/val label-set divergence.** If val has labels unseen in train (common on long-tail taxonomies), your multihot encoder will raise `KeyError`. Either drop those val rows, map to `<other>`, or expand the taxonomy at encode time; silent dropping distorts macro metrics.
- **Repair applied only to one split.** Upward repair applied only on test but not val means the thresholds were tuned against unrepaired predictions and will mis-rank at inference. Apply the same repair pipeline on val during threshold tuning and on test at scoring.

## 6. References

Primary sources used in this solution:

- Tsoumakas, G., Katakis, I. (2007). *Multi-Label Classification: An Overview*. *International Journal of Data Warehousing and Mining* 3(3). <https://www.di.uoa.gr/~grigoris/MLDM2010/MultiLabelOverview.pdf> — canonical taxonomy of binary-relevance vs. classifier-chains vs. label-powerset approaches.
- Silla, C.N., Freitas, A.A. (2011). *A Survey of Hierarchical Classification Across Different Application Domains*. *Data Mining and Knowledge Discovery* 22. <https://link.springer.com/article/10.1007/s10618-010-0175-9> — the standard reference for the mandatory-ancestor rule and local vs global hierarchical classifier topologies.
- Kiritchenko, S., Matwin, S., Famili, F. (2006). *Functional Annotation of Genes Using Hierarchical Text Categorization*. *BioLINK SIG at ISMB*. <https://aclanthology.org/W05-1301/> — the hierarchical-F1 (ancestor-expanded precision/recall) metric used in §2.6.
- Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017). *Focal Loss for Dense Object Detection*. ICCV. <https://arxiv.org/abs/1708.02002> — focal loss, cited as the production alternative to `pos_weight` BCE for extreme imbalance.
- Hugging Face `transformers` — multi-label sequence classification task docs. <https://huggingface.co/docs/transformers/tasks/sequence_classification#multi-label> — reference for `problem_type="multi_label_classification"` and the BCE-with-logits loss selection.
- `scikit-learn` user guide — multi-label metrics. <https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics> — definitions of micro/macro-F1 and label-ranking metrics.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — "ML01 Input Manipulation" and "ML04 Membership Inference" apply to hierarchical catalog models exposed to user queries.
- MITRE ATLAS — <https://atlas.mitre.org/> — adversarial-input techniques against multi-label classifiers; relevant when the predicted label set drives downstream routing or moderation.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — Measure function: report per-label metrics (not aggregate only) to expose rare-class failures; Govern function: document the human-review path for low-confidence multi-label predictions.
