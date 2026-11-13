# exercise-03 — Calibration and Threshold Tuning

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-103-text-classification/exercises/exercise-03-calibration-and-threshold-tuning.md`.
> This solution targets the learning goals that filename implies: a classifier that scores well on accuracy/F1 can still be badly **miscalibrated** — its "0.9" predictions may be right 70 % of the time — and in any deployment with an action threshold (routing, moderation, alerting, review-queue sizing) **calibration + threshold choice** are the operational levers, not accuracy.

## 1. Solution overview

Two mostly-orthogonal problems:

- **Calibration.** Does `P(y=1|x) ≈ score(x)`? A calibrated score of 0.80 means "in the long run, 80 % of examples with this score are positive". Transformer softmax/sigmoid outputs are famously overconfident (Guo et al. 2017); linear/tree models can be either under- or over-confident depending on regularisation. Quantified with **Expected Calibration Error** (ECE), **Brier score**, and reliability diagrams.
- **Threshold choice.** Given calibrated scores, which operating point do we deploy? The threshold is a *product* decision (how much precision we trade for recall, what action is taken), not a metric decision. The classifier exposes the precision/recall curve; the threshold is picked at the point that satisfies the SLA.

| Calibration method | When to use | Why |
|---|---|---|
| **Platt scaling** | Scores are already roughly monotone with the true probability; small val set (≥100 positives) | Fits a single sigmoid `σ(a·s + b)`; low variance, strong parametric assumption |
| **Isotonic regression** | Scores are monotone but non-linearly shaped; larger val set (≥500 positives per class) | Non-parametric; fits an arbitrary monotone step function; prone to overfit tiny splits |
| **Temperature scaling** | Deep neural net classifier with softmax logits | Platt's single-parameter variant that preserves `argmax`; the Guo et al. 2017 default for calibrating neural-net classifiers without changing the top-1 prediction |

**ECE** partitions predictions into `M` equal-width bins and reports the sample-weighted average `|acc(bin) - conf(bin)|`. It is the headline calibration metric but has known weaknesses (equal-width bins waste resolution where predictions concentrate, and a model perfect on average can still be bin-wise biased). Report the **Brier score** (`(y - p)²` averaged) alongside — it is a proper scoring rule and decomposes into calibration + refinement components (Brier 1950, DeGroot & Fienberg 1983).

**Threshold tuning**: on the *calibrated* scores, maximise the operational objective on the val set — F1, precision at fixed recall, cost-weighted accuracy, or expected-cost-per-review for a human-in-the-loop queue. Freeze thresholds, then measure once on test.

The deliverable: a `Calibrator` protocol with three implementations, a reliability-diagram renderer, an ECE/Brier metric module, and a threshold tuner wired into the same bake-off harness as exercise-01.

## 2. Worked implementation

### 2.1 Project layout

```
calibration/
  calibrators.py       # Platt, Isotonic, TemperatureScaling
  metrics.py           # ECE, Brier, reliability bins
  thresholds.py        # F1, precision-at-recall, cost-weighted
  plot.py              # reliability diagram
  pipeline.py          # fit -> calibrate -> tune -> report
  tests/
    test_calibrators.py
    test_metrics.py
    test_thresholds.py
```

### 2.2 Calibrators

```python
# calibrators.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Platt:
    """Fit sigmoid(a*s + b) on (val_score, val_y)."""
    def __init__(self):
        self._lr = LogisticRegression(C=1e6, solver="lbfgs")

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "Platt":
        self._lr.fit(scores.reshape(-1, 1), y)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(scores.reshape(-1, 1))[:, 1]


class Isotonic:
    """Non-parametric monotone fit."""
    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "Isotonic":
        self._iso.fit(scores, y)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self._iso.transform(scores)


class TemperatureScaling:
    """Guo et al. 2017: minimise NLL over the scalar T on val logits.

    Works on *logits*, not probabilities. Preserves argmax exactly, so
    multi-class accuracy is unchanged.
    """
    def __init__(self, max_iter: int = 200):
        self.T = 1.0
        self.max_iter = max_iter

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "TemperatureScaling":
        logits_t = torch.tensor(logits, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        T = nn.Parameter(torch.ones(1))
        opt = torch.optim.LBFGS([T], lr=0.1, max_iter=self.max_iter)
        ce = nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = ce(logits_t / T.clamp(min=1e-3), y_t)
            loss.backward()
            return loss

        opt.step(closure)
        self.T = float(T.detach().clamp(min=1e-3).item())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.T
        e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
```

Design notes worth defending:

- **Platt's `C=1e6`** effectively disables L2 regularisation; Platt scaling is a *fit*, not a prior. The regularised version is called "Platt scaling with hold-out" and is overkill on val sets with <10 000 points.
- **`IsotonicRegression(out_of_bounds="clip")`** returns the learned boundary probabilities on scores outside the val range instead of extrapolating — the safer default for a deployed calibrator.
- **Temperature scaling on logits**, not on softmax outputs. Scaling after softmax is a different (non-monotone) transformation and loses the "preserves argmax" property.

### 2.3 Calibration metrics

```python
# metrics.py
import numpy as np


def reliability_bins(prob: np.ndarray, y: np.ndarray, n_bins: int = 15
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-width bins over [0, 1]. Returns (conf, acc, count) per bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf = np.zeros(n_bins)
    acc  = np.zeros(n_bins)
    count = np.zeros(n_bins, dtype=np.int64)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        count[b] = int(mask.sum())
        if count[b] == 0:
            continue
        conf[b] = prob[mask].mean()
        acc[b]  = y[mask].mean()
    return conf, acc, count


def ece(prob: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error — sample-weighted |acc - conf|."""
    conf, acc, count = reliability_bins(prob, y, n_bins)
    n = count.sum()
    if n == 0:
        return 0.0
    return float(np.sum((count / n) * np.abs(acc - conf)))


def brier(prob: np.ndarray, y: np.ndarray) -> float:
    """Brier score = mean squared error of probabilistic predictions."""
    return float(np.mean((prob - y) ** 2))
```

For multi-class, the standard choice is **top-label ECE**: evaluate ECE on `(max_prob, y == argmax_prob)`. That is what Guo et al. 2017 report and what temperature scaling directly optimises.

### 2.4 Threshold tuning

```python
# thresholds.py
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns an extra endpoint; align the slices.
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def threshold_for_recall(y_true: np.ndarray, y_score: np.ndarray,
                         target_recall: float) -> tuple[float, float]:
    """Smallest threshold with recall >= target. Returns (threshold, precision)."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    ok = np.where(rec[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return 0.0, float(prec[-1])  # can't hit target; return trivial threshold
    i = int(ok[-1])  # the largest threshold that still meets recall
    return float(thr[i]), float(prec[i])


def cost_weighted_threshold(y_true: np.ndarray, y_score: np.ndarray,
                            c_fp: float, c_fn: float) -> tuple[float, float]:
    """Pick the threshold that minimises expected cost on val."""
    thrs = np.unique(np.concatenate([[0.0], y_score, [1.0]]))
    costs = []
    for t in thrs:
        yp = (y_score >= t).astype(int)
        fp = int(((yp == 1) & (y_true == 0)).sum())
        fn = int(((yp == 0) & (y_true == 1)).sum())
        costs.append(c_fp * fp + c_fn * fn)
    i = int(np.argmin(costs))
    return float(thrs[i]), float(costs[i])
```

The three functions encode three operational objectives:

- **`best_f1_threshold`** — the "balanced" default when precision and recall are equally weighted.
- **`threshold_for_recall`** — "we must catch ≥95 % of real positives, maximise precision subject to that." Standard for moderation / safety queues.
- **`cost_weighted_threshold`** — generalises the above. `c_fp` and `c_fn` are product-supplied (e.g. "a missed fraud costs $500, a false flag costs 2 minutes of reviewer time at $1/min" → `c_fn=500`, `c_fp=2`). The chosen threshold is the Bayes-optimal operating point on val.

### 2.5 Reliability diagram

```python
# plot.py
import matplotlib.pyplot as plt
import numpy as np
from .metrics import reliability_bins


def reliability_diagram(prob: np.ndarray, y: np.ndarray,
                        n_bins: int = 15, out: str = "reliability.png") -> None:
    conf, acc, count = reliability_bins(prob, y, n_bins)
    centres = (np.linspace(0, 1, n_bins + 1)[:-1] + np.linspace(0, 1, n_bins + 1)[1:]) / 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    ax1.bar(centres, acc, width=1 / n_bins, alpha=0.6, edgecolor="black", label="accuracy")
    ax1.scatter(centres, conf, marker="x", color="red", label="confidence")
    ax1.set_ylabel("Accuracy / Confidence")
    ax1.legend(loc="upper left")

    ax2.bar(centres, count, width=1 / n_bins, alpha=0.6, edgecolor="black")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
```

The bottom panel is the most important and most often omitted part of a reliability diagram: without the count-per-bin histogram the reader cannot tell whether a "bad" bin held three examples or thirty thousand. Always include it.

### 2.6 End-to-end pipeline

```python
# pipeline.py
from dataclasses import dataclass
import numpy as np
from .calibrators import Platt, Isotonic, TemperatureScaling
from .metrics import ece, brier
from .thresholds import best_f1_threshold


@dataclass
class CalibrationReport:
    pre_ece: float
    pre_brier: float
    post_ece: float
    post_brier: float
    threshold: float
    threshold_f1_val: float


def calibrate_and_tune(val_scores: np.ndarray, val_y: np.ndarray,
                       test_scores: np.ndarray, test_y: np.ndarray,
                       calibrator="platt") -> CalibrationReport:
    cals = {"platt": Platt(), "isotonic": Isotonic()}
    cal = cals[calibrator].fit(val_scores, val_y)

    val_cal  = cal.transform(val_scores)
    test_cal = cal.transform(test_scores)

    thr, f1_val = best_f1_threshold(val_y, val_cal)
    return CalibrationReport(
        pre_ece=ece(test_scores, test_y),
        pre_brier=brier(test_scores, test_y),
        post_ece=ece(test_cal, test_y),
        post_brier=brier(test_cal, test_y),
        threshold=thr,
        threshold_f1_val=f1_val,
    )
```

## 3. Validation steps

1. **Pre- vs post-calibration ECE improves on test.** Fit the calibrator on val, transform test. Post-ECE must be lower than pre-ECE on a model that was actually miscalibrated. If post-ECE is *worse*, you likely fit the calibrator on test or used an inappropriate method (isotonic on <100 val examples overfits).
2. **Calibrator is monotone.** Platt and temperature scaling are monotone by construction. Isotonic is monotone by its fit algorithm. Sample 1 000 pairs `(s1, s2)` with `s1 < s2` and assert `cal(s1) ≤ cal(s2) + 1e-8`.
3. **Argmax is preserved by temperature scaling.** For multi-class logits, assert `np.argmax(logits, 1) == np.argmax(logits / T, 1)` for every example. A broken temperature implementation (applying it to probabilities instead of logits) can change argmax; this test catches that.
4. **Threshold is tuned on val only.** `best_f1_threshold(val_y, val_cal)` returns a threshold; test macro-F1 at that threshold is what gets reported. Tuning against test and reporting against test is the classic leak and inflates the published number by typically 1-3 F1 points.
5. **Reliability-diagram smoke.** Render pre- and post-calibration diagrams on the test split. The bars should land closer to the diagonal after calibration; the bottom histogram should look identical pre/post (counts do not change).
6. **Bootstrap CI on ECE.** 10 000 resamples with replacement; publish the interval on both pre- and post-calibration ECE. ECE confidence intervals are wide on small test sets, and a 0.01 → 0.008 "improvement" is often inside the noise.
7. **Cost sensitivity.** Tune the threshold with `cost_weighted_threshold` at `(c_fp, c_fn) = (1, 1)`, `(1, 10)`, `(10, 1)`. Confirm the chosen threshold moves monotonically with the ratio. This tests both the cost search and the business-sensibility of exposing the knob.
8. **Dependency pin and determinism.** Re-run the whole calibrate + tune pipeline twice with identical inputs; every reported number must match byte-for-byte. Random state anywhere in this pipeline is a bug.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Calibration vs discrimination** | Explicit that calibration ≠ accuracy; reports ECE/Brier alongside F1 | Mentions distinction, reports only F1 | Treats "high accuracy" as "calibrated" |
| **Method choice** | Picks Platt / isotonic / temperature based on data regime; defends choice | Picks one, no justification | "Default sigmoid" with no calibration |
| **Val/test discipline** | Calibrator fit on val, threshold tuned on val, test seen once | Fits on train, tests on val | Fits on test |
| **ECE reporting** | Equal-width bins, bin count justified, histogram shown; Brier reported alongside | ECE only | Single scalar with no diagram |
| **Reliability diagram** | Both panels (accuracy-vs-confidence AND bin counts) | Top panel only | Not produced |
| **Threshold objective** | Named (F1, precision-at-recall, cost-weighted); matches the stated deployment goal | Named but not matched to goal | "Threshold = 0.5" with no rationale |
| **Argmax preservation (multi-class)** | Temperature used on logits; argmax verified unchanged | Mentioned, not verified | Softmax rescaled; argmax quietly changes |
| **CI / uncertainty** | Bootstrap CI on ECE and on test-F1 at chosen threshold | CI on F1 only | No uncertainty quantification |
| **Responsible deployment** | NIST AI RMF Measure: calibration is part of the metric contract; reports threshold choice with reviewer-time or cost impact per MITRE ATLAS / OWASP risk | Mentions one | Ignored |

## 5. Common mistakes

- **"Accuracy is 94 % so the model is calibrated."** Calibration is a property of the score distribution, not of top-1 correctness. A model can be 94 % accurate and still have a 20-point ECE if its "0.99" predictions are only right 80 % of the time. Run the reliability diagram.
- **Temperature scaling applied to probabilities.** If you divide the softmax output by `T` and renormalise you get a different (non-monotone) transform; argmax can flip. Scale the **logits** (pre-softmax).
- **Isotonic on tiny val sets.** With <200 positives per class isotonic overfits — the resulting step function memorises val and generalises worse than Platt. Use Platt or temperature scaling as a safer default; move to isotonic once val is large.
- **Calibrator fit on train.** The model already minimises cross-entropy on train; fitting a calibrator there teaches it the identity. Fit on val only.
- **Threshold chosen after calibration but evaluated against raw scores.** The "0.3" threshold on calibrated output is not the same operating point as "0.3" on raw logits; swapping the two is a top-line F1 drop in production that is nearly impossible to debug without reading the code.
- **Reporting ECE without the histogram.** Equal-width bins can report excellent ECE because the model never predicts in 10 of the 15 bins. The bottom-panel histogram makes this obvious; without it the number looks suspiciously good.
- **Ignoring class imbalance in ECE.** On a 1-in-100 positive class, a model that always outputs 0.01 has ECE ≈ 0.0 and is useless. Report **per-class** calibration on multi-label problems and **class-stratified** reliability diagrams on imbalanced single-label ones.
- **Cost knobs with no product review.** `c_fp = 1, c_fn = 10` is an engineering guess. For high-stakes domains (medical triage, fraud, moderation) the cost ratio is the product owner's decision and should be documented in the deployment card. NIST AI RMF (Govern) expects this traceability.
- **Threshold frozen but data drifting.** A calibrator and threshold tuned on last quarter's val may be wrong this quarter. Schedule recalibration (not retraining) alongside drift monitoring (see mod-112).

## 6. References

Primary sources used in this solution:

- Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. (2017). *On Calibration of Modern Neural Networks*. ICML. <https://arxiv.org/abs/1706.04599> — the temperature-scaling paper; shows modern networks are badly miscalibrated and that a single-parameter rescale closes most of the gap.
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*. *Advances in Large Margin Classifiers*. <https://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf> — original Platt scaling.
- Zadrozny, B., Elkan, C. (2002). *Transforming Classifier Scores into Accurate Multiclass Probability Estimates*. KDD. <https://dl.acm.org/doi/10.1145/775047.775151> — isotonic regression for calibration.
- Naeini, M.P., Cooper, G.F., Hauskrecht, M. (2015). *Obtaining Well Calibrated Probabilities Using Bayesian Binning*. AAAI. <https://ojs.aaai.org/index.php/AAAI/article/view/9602> — defines ECE and the binning methodology used in §2.3.
- Brier, G.W. (1950). *Verification of Forecasts Expressed in Terms of Probability*. *Monthly Weather Review* 78. <https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml> — the Brier score.
- DeGroot, M.H., Fienberg, S.E. (1983). *The Comparison and Evaluation of Forecasters*. *The Statistician* 32. <https://www.jstor.org/stable/2987588> — calibration/refinement decomposition.
- `scikit-learn` user guide — probability calibration. <https://scikit-learn.org/stable/modules/calibration.html> — reference for `CalibratedClassifierCV`, `IsotonicRegression`, and the practical recipes.
- PyTorch documentation — `nn.CrossEntropyLoss`, `LBFGS` optimiser, used by `TemperatureScaling.fit`.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — "ML01 Input Manipulation" and threshold-evasion attacks: adversaries aware of the operating point can craft inputs that score just below it. Keeping thresholds confidential is a soft defence; monitoring near-threshold rates is the harder one.
- MITRE ATLAS — <https://atlas.mitre.org/> — adversarial-evasion techniques against classifiers; relevant when a published model card discloses the operating point.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — Measure function treats calibration as part of the metric contract (not optional); Govern function requires that threshold / cost decisions be documented alongside the model card.
