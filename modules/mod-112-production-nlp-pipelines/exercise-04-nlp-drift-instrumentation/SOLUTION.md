# exercise-04 — NLP Drift Instrumentation

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-112-production-nlp-pipelines/exercises/exercise-04-nlp-drift-instrumentation.md`.
> The target learning outcomes implied by that filename: instrument a production NLP service so
> that **three kinds of drift** (input distribution, prediction distribution, and label/quality
> drift once feedback arrives) are detected with **statistically defensible tests**, aggregated
> to **actionable alerts**, and tied back to the model-card thresholds from mod-111. The output
> is a runnable instrumentation harness, a dashboard spec, and an on-call runbook.

## 1. Solution overview

NLP models degrade silently. The dominant failure modes are not catastrophic bugs; they are
slow regressions that only show up in retrospectives:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Input distribution shift** | The vocabulary, length, and language mix of production traffic diverges from training data; the model is applied out of distribution with no warning | Log per-request input statistics (length, OOV rate, language-id, embedding) and compare to a **training reference window** on a rolling basis using a two-sample test. PSI, KL, Jensen-Shannon, or MMD per feature family |
| **Prediction distribution shift** | Class prior changes (positive-class rate trends from 12 % to 4 %); confidence distribution becomes bimodal; "unknown / none-of-the-above" rate climbs | Log the prediction distribution and confidence histogram; alert on PSI > 0.2 or a divergence CI that excludes zero |
| **Quality / performance drift** | Accuracy drops but you only see it when a customer complaints; the model card's release-gate threshold is no longer met | Collect delayed labels (customer feedback, human review, downstream outcomes); compute the primary metric on a **sliding window**; alert when it crosses the release-gate threshold |
| **Alert fatigue** | Everything alerts; nothing is actionable; the runbook is a wiki page from two releases ago | Tie every alert to a runbook step: "page on-call", "open a ticket", "roll back to checkpoint X". Alerts without a runbook step are deleted |
| **Drift detected on the whole corpus; the real problem is one slice** | Mean statistics stay flat because a regression in one slice is cancelled by an improvement in another | Compute drift tests per slice (domain, language, length bucket, demographic subgroup) in parallel with the corpus-level test |
| **Confusing drift with seasonality** | A weekly cycle in input traffic triggers a drift alarm every Monday | Use a reference window that spans a full seasonal cycle, or detrend before the test. A drift test over a 24-hour window will have a 7-day periodic false-positive |
| **No baseline captured at release time** | You cannot compare "now" to "release"; you only have "now" vs. "last week" | At model release, freeze a **reference window** of inputs + predictions; version it with the model-card artefact. Drift is measured against this reference, not against a rolling baseline |

The deliverable is a four-part instrumentation:

1. **Logging schema** — the per-request JSON row that drift analysis consumes.
2. **Drift tests** — PSI and Jensen-Shannon for categorical / binned distributions; two-sample
   Kolmogorov–Smirnov for continuous; MMD or embedding-PSI for semantic drift; paired
   bootstrap CIs for the accuracy slide.
3. **Alert rules + runbook** — mapping from drift test output to an on-call action.
4. **Dashboard spec** — what the operator sees: reference vs. current, per-slice panels, the
   release-gate threshold line.

Authoritative sources this solution draws on:

- **Gama et al. 2014** — *A survey on concept drift adaptation*. ACM Computing Surveys.
  <https://dl.acm.org/doi/10.1145/2523813>. The canonical taxonomy (virtual drift, real drift,
  concept drift) and detector classes (DDM, EDDM, ADWIN, Page–Hinkley).
- **Rabanser, Günnemann, Lipton 2019** — *Failing Loudly: An Empirical Study of Methods for
  Detecting Dataset Shift*. NeurIPS. <https://arxiv.org/abs/1810.11953>. The authoritative
  comparison of shift-detection methods on neural-network inputs; justifies using *learned
  representations* for the two-sample test rather than raw features.
- **Lipton, Wang, Smola 2018** — *Detecting and Correcting for Label Shift with Black Box
  Predictors*. ICML. <https://arxiv.org/abs/1802.03916>. Label-shift detection via the
  confusion matrix.
- **Alibi Detect** — <https://docs.seldon.io/projects/alibi-detect/en/latest/>. A reference
  open-source implementation of MMD, LSDD, FET, and KS drift detectors with API examples.
- **Evidently AI** — <https://docs.evidentlyai.com/>. Reference implementation of PSI, KL,
  JSD, WD per-feature drift computation + reports.
- **NannyML** — <https://nannyml.readthedocs.io/en/stable/>. Performance estimation without
  ground truth (Confidence-Based Performance Estimation, Direct Loss Estimation) for the
  "delayed labels" case.
- **Breck et al. 2019** — *Data Validation for Machine Learning*. MLSys.
  <https://mlsys.org/Conferences/2019/doc/2019/167.pdf>. The TensorFlow Data Validation
  (TFDV) paper; schema inference and anomaly detection on tabular features.
- **Polyzotis et al. 2017** — *Data Management Challenges in Production Machine Learning*.
  SIGMOD. The statement of the problem; an older but still-cited reference.
- **Model Cards** (Mitchell et al. 2019) — <https://arxiv.org/abs/1810.03993>. The release-
  gate threshold the drift monitor is compared against.

## 2. Worked implementation

### 2.1 Project layout

```
drift_instrumentation/
  src/drift_instrumentation/
    __init__.py
    schema.py                  # per-request JSON schema
    logger.py                  # the serving-side hook that emits the row
    reference.py               # freeze and version a reference window
    features.py                # feature extractors (length, OOV, embedding)
    tests/
      psi.py                   # Population Stability Index
      jsd.py                   # Jensen–Shannon divergence
      ks.py                    # two-sample Kolmogorov–Smirnov
      mmd.py                   # Maximum Mean Discrepancy (embedding drift)
      bootstrap_ci.py          # paired bootstrap for metric drift
    aggregator.py              # per-slice rollup
    alerts.py                  # rules + runbook mapping
  configs/
    reference_window.yaml      # frozen at release
    alerts.yaml                # thresholds + runbook refs
  dashboards/
    grafana.json               # dashboard definition
  runbooks/
    input_drift.md
    prediction_drift.md
    accuracy_drop.md
  tests/
    test_psi.py
    test_slice_aggregation.py
    test_alerts_fire.py
  pyproject.toml
  README.md
```

### 2.2 Per-request logging schema

```python
# src/drift_instrumentation/schema.py
from __future__ import annotations
from pydantic import BaseModel, Field


class RequestRow(BaseModel):
    request_id: str
    received_at: str                       # ISO-8601, UTC
    model_version: str                     # resolved model SHA, pins which release this row belongs to
    tenant: str | None = None              # multi-tenant slice axis
    # --- Inputs (features used for drift) ---
    input_char_len: int
    input_token_len: int
    language_id: str                       # "en", "es", ...
    oov_rate: float = Field(ge=0.0, le=1.0)
    input_embedding_hash: str              # SHA-256 of a hashed-embedding bucket, for cardinality-safe drift
    input_embedding_bucket: int            # 0..1023; coarse LSH bucket for JSD on embedding
    domain_tag: str | None = None          # slice axis if the domain is known
    # --- Predictions (predicted distribution + confidence) ---
    predicted_label: str
    predicted_score: float = Field(ge=0.0, le=1.0)
    top_k_labels: list[str]
    top_k_scores: list[float]
    predicted_tokens_out: int | None = None     # for generative; None for classifier
    # --- Feedback (delayed; may be absent at log time) ---
    ground_truth_label: str | None = None
    feedback_source: str | None = None          # "human_review" | "downstream_signal" | None
    feedback_arrived_at: str | None = None
```

Points worth defending in review:

- **`model_version` on every row.** Drift is measured within a model; mixing two model
  versions' rows into one drift test produces non-interpretable results.
- **Embedding drift via a **bucketed hash**, not raw embedding.** Storing the full embedding
  per request is prohibitively expensive at scale; a LSH or random-hashing bucket into
  ~1 024 bins lets us run JSD / PSI on the bucket distribution. Rabanser et al. 2019 showed
  that even coarse learned-representation distributions detect shift better than raw features.
- **`top_k_labels` + `top_k_scores`.** Confidence drift often shows up as the gap between
  top-1 and top-2 compressing, not as top-1 confidence falling. Logging the top-k is cheap.
- **Feedback optional.** Rows arrive now; feedback arrives later; the join is time-based and
  lives in the aggregator.

### 2.3 PSI — the workhorse drift test for binned distributions

```python
# src/drift_instrumentation/tests/psi.py
from __future__ import annotations
import math
from collections import Counter


def _distribution(values: list, bins: list) -> list[float]:
    counter = Counter(values)
    total = sum(counter.values()) or 1
    return [counter.get(b, 0) / total for b in bins]


def psi(reference: list, current: list, bins: list, epsilon: float = 1e-4) -> float:
    """
    Population Stability Index. Standard interpretation (Breck et al. 2019, industry rule of thumb):
        PSI < 0.1  -> no significant change
        0.1 <= PSI < 0.2  -> moderate shift, investigate
        PSI >= 0.2  -> significant shift, alert
    `bins` must be provided so reference and current share the same support.
    """
    ref_dist = _distribution(reference, bins)
    cur_dist = _distribution(current, bins)
    score = 0.0
    for r, c in zip(ref_dist, cur_dist):
        r = max(r, epsilon)
        c = max(c, epsilon)
        score += (c - r) * math.log(c / r)
    return score
```

PSI is the industry workhorse because it is interpretable and robust: `0.1 / 0.2` are
well-documented thresholds (originating in credit-scoring, re-used by TFDV / Evidently). It
needs a fixed binning scheme shared between reference and current; store it with the
reference window.

### 2.4 Jensen–Shannon — for predicted-class and embedding-bucket distributions

```python
# src/drift_instrumentation/tests/jsd.py
from __future__ import annotations
import math


def _kl(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q) if pi > 0)


def jsd(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence; bounded in [0, ln 2]."""
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
```

JSD is symmetric and bounded, which makes it easier to threshold than raw KL. For predicted-
class drift, run JSD on the probability simplex of the class prior; for embedding-bucket
drift, on the 1 024-bin bucket distribution.

### 2.5 Two-sample KS — for continuous features (length, OOV rate, score)

```python
# src/drift_instrumentation/tests/ks.py
from __future__ import annotations
import bisect


def two_sample_ks(reference: list[float], current: list[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov. Returns (D, approximate_p).

    Avoids a SciPy dependency; the approximate p is the Kolmogorov distribution tail
    sufficient for threshold-based alerting. For a formal CI, use scipy.stats.ks_2samp.
    """
    ref = sorted(reference)
    cur = sorted(current)
    n1, n2 = len(ref), len(cur)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")

    combined = sorted(set(ref) | set(cur))
    d = 0.0
    for v in combined:
        cdf_ref = bisect.bisect_right(ref, v) / n1
        cdf_cur = bisect.bisect_right(cur, v) / n2
        d = max(d, abs(cdf_ref - cdf_cur))

    # Kolmogorov approximate p-value
    en = (n1 * n2 / (n1 + n2)) ** 0.5
    stat = (en + 0.12 + 0.11 / en) * d
    # Tail approximation — two-term series is sufficient for thresholding.
    import math
    p = 2 * (math.exp(-2 * stat**2) - math.exp(-8 * stat**2))
    return d, max(0.0, min(1.0, p))
```

KS is the standard non-parametric two-sample test for continuous features. It is sensitive
to location shift and scale shift; it is less powerful against higher-moment differences,
which is the main caveat vs. MMD on embeddings.

### 2.6 MMD on embeddings — for semantic drift

```python
# src/drift_instrumentation/tests/mmd.py
from __future__ import annotations
import numpy as np


def rbf_mmd(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """
    Maximum Mean Discrepancy with an RBF kernel. Unbiased estimator (Gretton et al. 2012).
    x: (n, d) reference embeddings; y: (m, d) current embeddings.
    """
    if gamma is None:
        # Median heuristic over the stacked sample.
        combined = np.vstack([x, y])
        n = combined.shape[0]
        idx = np.random.default_rng(0).choice(n, size=min(500, n), replace=False)
        sub = combined[idx]
        dists = np.sqrt(np.maximum(0, ((sub[:, None] - sub[None, :]) ** 2).sum(-1)))
        med = np.median(dists[dists > 0])
        gamma = 1.0 / (2 * med ** 2 + 1e-12)

    def kxx(a, b):
        sq = ((a[:, None] - b[None, :]) ** 2).sum(-1)
        return np.exp(-gamma * sq)

    k_xx = kxx(x, x)
    k_yy = kxx(y, y)
    k_xy = kxx(x, y)

    n, m = x.shape[0], y.shape[0]
    # Remove diagonal for unbiased estimator
    m_xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    m_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    m_xy = k_xy.mean()

    return float(m_xx + m_yy - 2 * m_xy)
```

MMD is the shift-detection workhorse of Rabanser et al. 2019; the RBF kernel with the median
heuristic is a reasonable default. A production deployment uses Alibi Detect's MMD with
bootstrap p-values rather than hand-rolling it; the implementation above is included to make
the math explicit for the exercise.

### 2.7 Bootstrap CI for the accuracy slide (delayed labels)

```python
# src/drift_instrumentation/tests/bootstrap_ci.py
from __future__ import annotations
import random


def accuracy_bootstrap(
    labels_true: list[str],
    labels_pred: list[str],
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: random.Random | None = None,
) -> tuple[float, float, float]:
    assert len(labels_true) == len(labels_pred), "paired samples required"
    assert labels_true, "non-empty sample required"
    rng = rng or random.Random(0)
    n = len(labels_true)
    draws = []
    for _ in range(n_boot):
        correct = 0
        for _ in range(n):
            idx = rng.randrange(n)
            correct += int(labels_true[idx] == labels_pred[idx])
        draws.append(correct / n)
    draws.sort()
    point = sum(t == p for t, p in zip(labels_true, labels_pred)) / n
    lo = draws[int(alpha / 2 * n_boot)]
    hi = draws[int((1 - alpha / 2) * n_boot) - 1]
    return point, lo, hi
```

The "accuracy dropped by 3 pp" claim must come with a CI; a paired bootstrap tied back to the
mod-111 exercise-02 methodology is the standard. The alert fires on *CI overlapping below the
release-gate threshold*, not on a raw point estimate — single-point noise would page every
night.

### 2.8 Aggregator — per-slice drift rollup

```python
# src/drift_instrumentation/aggregator.py
from __future__ import annotations
from dataclasses import dataclass
from .tests.psi import psi
from .tests.jsd import jsd
from .tests.ks import two_sample_ks


@dataclass
class DriftReport:
    model_version: str
    slice_name: str                 # "overall" or "domain=legal" etc.
    n_reference: int
    n_current: int
    psi_input_length_bucket: float
    jsd_language_id: float
    jsd_predicted_label: float
    jsd_embedding_bucket: float
    ks_predicted_score_D: float
    ks_predicted_score_p: float
    metric_point: float | None
    metric_lo: float | None
    metric_hi: float | None


def aggregate(reference_rows, current_rows, slice_fn, bins) -> list[DriftReport]:
    buckets_ref: dict[str, list] = {}
    buckets_cur: dict[str, list] = {}
    for row in reference_rows:
        buckets_ref.setdefault(slice_fn(row), []).append(row)
    for row in current_rows:
        buckets_cur.setdefault(slice_fn(row), []).append(row)

    reports: list[DriftReport] = []
    for slice_name, ref in buckets_ref.items():
        cur = buckets_cur.get(slice_name, [])
        if not cur:
            continue
        reports.append(DriftReport(
            model_version=ref[0].model_version,
            slice_name=slice_name,
            n_reference=len(ref),
            n_current=len(cur),
            psi_input_length_bucket=psi(
                [_length_bucket(r.input_token_len) for r in ref],
                [_length_bucket(r.input_token_len) for r in cur],
                bins=bins["length_buckets"],
            ),
            jsd_language_id=_jsd_over(bins["languages"],
                                       [r.language_id for r in ref],
                                       [r.language_id for r in cur]),
            jsd_predicted_label=_jsd_over(bins["labels"],
                                           [r.predicted_label for r in ref],
                                           [r.predicted_label for r in cur]),
            jsd_embedding_bucket=_jsd_over(bins["embedding_buckets"],
                                            [r.input_embedding_bucket for r in ref],
                                            [r.input_embedding_bucket for r in cur]),
            ks_predicted_score_D=_ks_stat([r.predicted_score for r in ref],
                                           [r.predicted_score for r in cur])[0],
            ks_predicted_score_p=_ks_stat([r.predicted_score for r in ref],
                                           [r.predicted_score for r in cur])[1],
            metric_point=None, metric_lo=None, metric_hi=None,
        ))
    return reports


def _length_bucket(n: int) -> int:
    for b in (32, 64, 128, 256, 512, 1024):
        if n <= b:
            return b
    return 2048


def _jsd_over(support, ref_vals, cur_vals):
    from collections import Counter
    r = Counter(ref_vals); c = Counter(cur_vals)
    rn = sum(r.values()) or 1; cn = sum(c.values()) or 1
    p = [r.get(k, 0) / rn for k in support]
    q = [c.get(k, 0) / cn for k in support]
    return jsd(p, q)


def _ks_stat(ref, cur):
    return two_sample_ks(ref, cur)
```

Per-slice is where the real signal lives; corpus-level drift often cancels across slices.

### 2.9 Alerts — thresholds tied to a runbook

```yaml
# configs/alerts.yaml
alerts:
  - name: input_length_drift_major
    condition: psi_input_length_bucket >= 0.2
    severity: warning
    runbook: runbooks/input_drift.md#input-length
    owner: on-call-nlp
  - name: embedding_drift_major
    condition: jsd_embedding_bucket >= 0.1   # JSD bounded in [0, ln 2] ~ 0.693; 0.1 is meaningful
    severity: warning
    runbook: runbooks/input_drift.md#embedding
    owner: on-call-nlp
  - name: predicted_class_prior_drift
    condition: jsd_predicted_label >= 0.1
    severity: warning
    runbook: runbooks/prediction_drift.md#class-prior
    owner: on-call-nlp
  - name: accuracy_below_release_gate
    condition: metric_hi < release_gate_threshold
    severity: page
    runbook: runbooks/accuracy_drop.md
    owner: on-call-nlp
```

Point to defend: an alert without a runbook step is deleted. The `runbook:` field is enforced
by a CI check (`test_alerts_have_runbook.py`).

### 2.10 Reference-window freezing

```python
# src/drift_instrumentation/reference.py
from __future__ import annotations
import pathlib, json, hashlib
from .schema import RequestRow


def freeze_reference_window(rows: list[RequestRow], out_path: pathlib.Path) -> dict:
    """At model release, freeze a reference window. Version it with the model card."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": rows[0].model_version,
        "n_rows": len(rows),
        "rows": [r.model_dump() for r in rows],
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    out_path.write_bytes(data)
    return {
        "path": str(out_path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "n_rows": len(rows),
        "model_version": rows[0].model_version,
    }
```

The reference window is the ground truth for "how it looked at release". It is versioned with
the model artefact and the `sha256` is recorded in the model card. Rolling-window baselines
are allowed for short-horizon trend alarms, but the release-gate comparison always uses the
frozen reference.

### 2.11 Dashboard spec (Grafana / equivalent)

The dashboard has three rows:

1. **Input drift** — one panel per slice (overall, per domain, per language). Lines for
   `psi_input_length_bucket`, `jsd_language_id`, `jsd_embedding_bucket`. Threshold lines at
   0.1 and 0.2. Reference window highlighted on the time axis.
2. **Prediction drift** — one panel per slice. Lines for `jsd_predicted_label`,
   `ks_predicted_score_D`. Histograms of `predicted_score` with reference overlaid.
3. **Accuracy / performance** — the metric (mod-111 primary) on the sliding window with
   bootstrap CI shaded; the release-gate threshold line explicit. One row per slice.

Every panel links to its runbook.

## 3. Validation steps

1. **Synthetic drift injection.** Replay reference rows unchanged — all PSI / JSD / KS reports
   must stay below 0.05 and no alerts fire. Shift every input length by +50 tokens — the
   length PSI must exceed 0.2 and the alert must fire. A monitor that does not fire on an
   obvious shift is a monitor that will miss a real one.
2. **No-ops on seasonality.** Replay a week of real traffic against a one-week reference
   window; alerts must *not* fire. If a monotonic weekly cycle fires every Monday, the
   reference window is too short or the test needs detrending.
3. **Per-slice drift surfaces the one bad domain.** Inject drift into one of five domain
   slices; the per-slice report must flag that slice while the overall report may or may not
   fire. Catches the "flat corpus mean hides a slice regression" failure.
4. **Delayed-label pipeline joins correctly.** A row logged at `t`, feedback arriving at
   `t + Δ`, must appear in the window ending at `t + Δ + eval_lag`. A missing join is a
   silent "no feedback ever comes back" bug, often an ID-normalisation mismatch between
   prediction logs and feedback logs.
5. **Accuracy CI respects the release-gate threshold.** On a sample where the point estimate
   is just below the threshold but the CI straddles it, the monitor must *not* page (not
   enough evidence yet); on a sample where the CI upper bound is below the threshold, it must.
6. **Alert-to-runbook mapping.** Every alert config has a `runbook:` field pointing to a file
   that exists; the file has a heading matching the anchor; the runbook lists concrete steps.
   `test_alerts_have_runbook.py` enforces this.
7. **Model-version partitioning.** Rows from two model versions must never share a drift test.
   Load a mixed sample; the aggregator must raise or return per-version reports; it must not
   silently blend.
8. **Reference window frozen at release.** The `sha256` recorded on the model card matches the
   file contents; any re-freeze without a model-version bump is a release-process violation.
9. **MMD / embedding drift on learned representations.** On two batches of synthetically
   distinct text (e.g., financial news vs. Shakespeare), MMD and JSD(embedding_bucket) must
   both detect drift; raw length PSI may or may not. Confirms Rabanser et al.'s result that
   learned-representation drift is more sensitive than hand-crafted-feature drift.
10. **Alert fatigue budget.** Over a 30-day replay of production traffic (no injected drift),
    total pages must be ≤ a documented budget (typically ≤ 2 pages / month). Exceed it and
    thresholds need tuning or the test needs detrending.

## 4. Grading rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Logging schema** | Per-request row with inputs, predictions, confidence, model version, slice tags, and delayed-label fields | Row present but missing confidence or model version | No structured log |
| **Reference window** | Frozen at release, sha-logged on model card, versioned with the model artefact | Reference exists but is a rolling baseline | No reference |
| **Input drift tests** | PSI on length bucket, JSD on language, MMD or JSD on embedding bucket; all with reference | PSI only | Mean/median comparison only |
| **Prediction drift tests** | JSD on class prior, KS on confidence | JSD only | None |
| **Delayed-label / accuracy drift** | Primary metric on sliding window with bootstrap CI; compared to release-gate threshold | Point estimate only | None |
| **Per-slice rollup** | Overall + per-domain + per-language + per-length panels | Overall only | None |
| **Alerts tied to runbooks** | Every alert has a `runbook:` pointer that resolves; CI enforced | Runbook exists but is blank | No runbook |
| **False-positive budget** | Documented FP budget; passes replay test | Budget mentioned, not tested | No concept |
| **Model-version partitioning** | Drift tests scoped to a single `model_version`; mixed-version rows rejected | Mixed silently averaged | No awareness |
| **Documentation** | README has run command, dashboard links, and on-call ownership | README partial | Missing |

## 5. Common mistakes

- **Drift test on raw features with no learned-representation component.** Rabanser et al.
  2019 showed learned representations (even frozen embedders) detect shift better than
  hand-crafted features; a monitor that only tracks length and OOV misses semantic drift.
- **Rolling-baseline reference.** Comparing "this week" vs. "last week" catches short-horizon
  anomalies but is blind to slow drift over months; the release-reference pattern catches
  slow drift, at the cost of a once-per-release freeze step.
- **Thresholding on a p-value alone.** With a million rows, every test is significant; p
  stops being informative. Threshold on *effect size* (PSI 0.1 / 0.2, JSD 0.05 / 0.1) not p;
  use p as a secondary sanity check.
- **Alerting on corpus means, ignoring slices.** A regression in one domain is cancelled by
  an improvement in another; the mean looks flat. The per-slice panel is mandatory.
- **Accuracy drift measured without a CI.** Noise on the metric point triggers nightly pages;
  bootstrap CIs with a release-gate comparison fix it.
- **Logging prediction confidence but not the predicted class distribution.** Both drift.
  Confidence drift usually precedes accuracy drift (model losing calibration before it loses
  accuracy); class-prior drift usually reflects traffic drift (which may or may not need an
  action).
- **Reference and current with different bin edges.** PSI needs shared bins; if the reference
  has 8 bins from quantiles computed at release and the current window is quantile-binned
  afresh, the two distributions are not comparable. Fix bins at reference time; store them
  with the reference artefact.
- **No runbook step.** An alert that pages someone with no next action generates a Slack
  thread, not a fix. Delete alerts without runbooks or make the runbook mandatory.
- **MMD on 10 samples.** MMD is a two-sample test and is noisy below ~500 per side; small
  windows yield unreliable scores. Alibi Detect docs state the trade-off.
- **Mixing model versions in one test.** Drift between model versions is not the signal the
  monitor is looking for; it is "two different models". Partition by `model_version`.
- **Treating seasonality as drift.** A weekly cycle in traffic will fire every Monday. Pick
  a reference window ≥ 1 full cycle or detrend.
- **Logging every request with the full input text and embedding.** Storage blows up; PII
  risks grow. Hash the input, bucket the embedding; retention policy explicit.
- **Not tying drift to the release-gate threshold.** The point of the monitor is to tell you
  "we are below the gate"; a monitor that reports PSI but not "below gate yes/no" is one
  layer away from action.
- **Alert fatigue.** Thresholds that never fire are useless; thresholds that fire every night
  are worse. Measure the FP rate during a 30-day replay and tune.

## 6. References

Primary sources used in this solution:

- Gama, J. et al. (2014). *A survey on concept drift adaptation*. ACM Computing Surveys.
  <https://dl.acm.org/doi/10.1145/2523813> — drift taxonomy (virtual / real / concept) and
  detector families.
- Rabanser, S., Günnemann, S., Lipton, Z. C. (2019). *Failing Loudly: An Empirical Study of
  Methods for Detecting Dataset Shift*. NeurIPS. <https://arxiv.org/abs/1810.11953> —
  empirical case for learned-representation drift detection on neural-network inputs.
- Lipton, Z. C., Wang, Y.-X., Smola, A. (2018). *Detecting and Correcting for Label Shift
  with Black Box Predictors*. ICML. <https://arxiv.org/abs/1802.03916>.
- Gretton, A. et al. (2012). *A Kernel Two-Sample Test*. JMLR. — the MMD definition with the
  RBF-kernel unbiased estimator used in §2.6.
- Breck, E. et al. (2019). *Data Validation for Machine Learning*. MLSys.
  <https://mlsys.org/Conferences/2019/doc/2019/167.pdf> — TFDV schema + anomaly-detection
  foundations; the PSI 0.1 / 0.2 operational thresholds are consistent with the field
  conventions this paper surveys.
- Alibi Detect documentation. <https://docs.seldon.io/projects/alibi-detect/en/latest/> —
  reference implementation of MMD, LSDD, FET, KS drift detectors with API examples.
- Evidently AI documentation. <https://docs.evidentlyai.com/> — reference implementation of
  per-feature PSI, KL, JSD, WD; dashboard patterns.
- NannyML documentation. <https://nannyml.readthedocs.io/en/stable/> — performance
  estimation without ground truth for the delayed-label window (Confidence-Based Performance
  Estimation, Direct Loss Estimation).
- Dror, R. et al. (2018). *The Hitchhiker's Guide to Testing Statistical Significance in
  NLP*. ACL. <https://aclanthology.org/P18-1128/> — paired-bootstrap methodology used for the
  accuracy-slide CI.
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAccT.
  <https://arxiv.org/abs/1810.03993> — the release-gate threshold the accuracy drift alert
  compares against.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/>
  — *ML01 Input Manipulation Attack*: drift instrumentation is a detection surface for this;
  a sudden embedding-drift spike with no traffic-volume change is consistent with a crafted-
  input campaign. *ML02 Data Poisoning*: delayed-label accuracy drift is the detector for a
  poisoned retraining feedback loop. *ML03 Model Inversion* and *ML04 Membership Inference*
  are tangential — an instrumentation surface that logs full inputs with no retention policy
  is itself the vector; hashing and bucketing (§2.2) is the mitigation. *ML09 Output
  Integrity Attack*: prediction-distribution drift is the first signal that an upstream
  tampering has altered the model's output.
- MITRE ATLAS — <https://atlas.mitre.org/> — *AML.T0043 Craft Adversarial Data* applies to
  the input side; a sustained input-embedding drift coincident with stable output accuracy
  is a signature of adversarial robustness being eroded. *AML.T0031 Erode Integrity* is the
  pattern the delayed-label accuracy monitor is designed to detect. *AML.T0024.002 Infer
  Training Data Membership* is partially mitigated by not logging raw inputs long-term.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework>
  — *Map*: the model-version → reference-window → metric-gate linkage. *Measure*: the per-
  slice drift tests and bootstrap-CI accuracy on the sliding window. *Manage*: the alert →
  runbook → on-call ownership chain is the primary Manage artefact. *Govern*: the FP-budget
  discipline, the "every alert points to a runbook" CI gate, and the frozen reference
  window + sha on the model card are the governance controls.
