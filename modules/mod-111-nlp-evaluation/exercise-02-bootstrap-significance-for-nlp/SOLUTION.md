# exercise-02 — Bootstrap Significance for NLP

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-111-nlp-evaluation/exercises/exercise-02-bootstrap-significance-for-nlp.md`.
> The target learning outcomes implied by that filename: implement **paired bootstrap resampling**
> for NLP metric deltas (BLEU, chrF, ROUGE, accuracy, F1); decide whether a System B's advantage
> over System A on a shared test set is signal or noise; produce both a **p-value** (one-sided
> Koehn 2004 bootstrap test) and a **95 % CI on the delta**; apply the **Bonferroni / Holm /
> Benjamini-Hochberg** correction when reporting across many slices; and know when the paired
> bootstrap is the wrong tool (per-instance exchangeability violated, metric is not a mean).

## 1. Solution overview

Two models' corpus BLEU scores differ by 0.4 points. Is that a win? The answer depends on the test
set size, the per-sentence variance, and the choice of significance test — which is why the question
is answered with a resampling procedure, not a t-test:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Unpaired comparison on paired data** | Treating System A and System B as independent samples throws away the strongest variance-reduction lever: both systems saw the same inputs | **Paired bootstrap**: resample sentence *indices*, compute the delta on the paired outputs. Variance drops by the ratio of between-sentence to within-sentence variance (often 5–20×) |
| **Metric is not a mean, test is a mean-test** | BLEU is a corpus-level ratio, not a mean of sentence-level BLEUs; a paired t-test on sentence-BLEU is testing a different statistic from the one you report | Resample at the sentence index level and *recompute the corpus-level metric on the resample*. This is the Koehn 2004 "bootstrap resampling for machine translation" recipe; it works for any corpus-level statistic |
| **Hidden multiple comparisons** | Reporting 12 slice p-values, taking the lowest, and claiming significance. With α = 0.05 and 12 independent tests, the family-wise false-positive rate is 46 % | Apply a correction: Bonferroni for strict family-wise error; Holm for less conservative FWER; Benjamini-Hochberg for FDR when you expect many real effects |
| **Treating a p-value as an effect size** | "p < 0.001, therefore the model is much better" — p-values encode evidence against the null, not magnitude of improvement | Always publish **(p-value, point estimate of delta, 95 % CI on delta)**. The CI tells the stakeholder the effect size; the p-value tells them whether the sign is confident |
| **Resampling non-exchangeable units** | If sentences in the test set are grouped (same document, same user, same source) and the groups matter for the metric, independent sentence resampling breaks the i.i.d. assumption and under-estimates variance | **Hierarchical / cluster bootstrap**: resample groups first, then sentences within groups. Required for document-level MT, dialogue sessions, multi-sentence QA |

The deliverable is a paired-bootstrap test harness with five components:

1. **Paired bootstrap** — resample indices with replacement, recompute the metric on the resample, repeat `B ≥ 1000` times; produce the empirical distribution of the delta.
2. **p-value** — one-sided Koehn 2004 formulation: proportion of bootstrap samples where the delta is `≤ 0` (null is "no improvement"). Or the stricter two-sided version for exploratory analysis.
3. **95 % CI on the delta** — percentile method (BCa when the metric is skewed; see §2.4); report both bounds.
4. **Multiple-comparisons correction** — Bonferroni / Holm / Benjamini-Hochberg over slice p-values.
5. **Cluster bootstrap variant** — for document-level or session-level test sets.

Reference literature:

- **Koehn 2004** — *Statistical Significance Tests for Machine Translation Evaluation*. The canonical NLP paired-bootstrap reference; recipe reproduced below.
- **Dror et al. 2018** — *The Hitchhiker's Guide to Testing Statistical Significance in NLP*. The modern survey; recommends paired bootstrap for most NLP scenarios, documents when it fails.
- **Riezler & Maxwell 2005** — *On Some Pitfalls in Automatic Evaluation and Significance Testing for MT*. The authoritative "bootstrap vs. approximate randomization" comparison in MT; both are valid, randomization is slightly more powerful at the cost of more compute.
- **Efron & Tibshirani 1994** — *An Introduction to the Bootstrap*. The statistics-literature foundation; defines percentile, BCa, and studentised bootstrap intervals.
- **Berg-Kirkpatrick et al. 2012** — *An Empirical Investigation of Statistical Significance in NLP*. Shows how many NLP papers over-claim significance due to multiple-comparisons and under-powered tests.
- **Dror et al. 2017** — *Replicability Analysis for Natural Language Processing*. The multiple-dataset extension of significance testing.
- **Benjamini & Hochberg 1995** — *Controlling the False Discovery Rate*. The BH procedure used in §2.5.
- **Holm 1979** — *A Simple Sequentially Rejective Multiple Test Procedure*. The Holm step-down FWER procedure.
- **Søgaard et al. 2014** — *What's in a p-value in NLP?*. Argues that NLP significance tests are often mis-specified; reproduces the Koehn bootstrap.

## 2. Worked implementation

### 2.1 Project layout

```
sigtest/
  bootstrap.py           # paired / cluster bootstrap
  corrections.py         # Bonferroni / Holm / BH
  metrics/
    __init__.py
    bleu.py              # corpus-level BLEU for resampling
    chrf.py              # character-F
    rouge.py             # ROUGE-{1,2,L}
    classification.py    # accuracy / macro-F1 as corpus metrics
  cli.py                 # one-shot report
  tests/
    test_koehn_fixture.py       # reproduce Koehn 2004 numbers
    test_bca_vs_percentile.py
    test_cluster_variance.py
    test_corrections.py
  requirements.txt
```

### 2.2 Paired bootstrap — the core routine

```python
# bootstrap.py
import random
from typing import Callable, Sequence


MetricFn = Callable[[Sequence, Sequence], float]
# A metric takes (predictions, references) and returns a corpus-level score.
# For classification, references may be label lists; for MT, lists of tokenised sentences.


def paired_bootstrap_delta(
    preds_a: Sequence,
    preds_b: Sequence,
    refs: Sequence,
    metric: MetricFn,
    n_boot: int = 1000,
    seed: int = 20260101,
) -> dict:
    """
    Koehn 2004 paired bootstrap for the delta metric(B) - metric(A).

    Returns:
      {
        "delta_point": float,               # metric(B, refs) - metric(A, refs)
        "delta_samples": list[float],       # B bootstrap deltas, sorted ascending
        "p_value_one_sided": float,         # P(delta <= 0 | bootstrap); H1 is B > A
        "p_value_two_sided": float,
        "ci_95": tuple[float, float],       # percentile method; see bca_ci() for BCa
      }
    """
    n = len(refs)
    assert len(preds_a) == n == len(preds_b), "paired test requires aligned predictions"

    rng = random.Random(seed)
    m_a = metric(preds_a, refs)
    m_b = metric(preds_b, refs)
    delta_point = m_b - m_a

    deltas: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sub_refs = [refs[i] for i in idx]
        sub_a = [preds_a[i] for i in idx]
        sub_b = [preds_b[i] for i in idx]
        deltas.append(metric(sub_b, sub_refs) - metric(sub_a, sub_refs))

    deltas.sort()
    # One-sided p: proportion of bootstrap samples where B does NOT beat A.
    # The Koehn 2004 convention centres the bootstrap distribution at zero by subtracting
    # delta_point; see below.
    centred = [d - delta_point for d in deltas]
    p_one = sum(1 for d in centred if d >= delta_point) / n_boot
    p_two = sum(1 for d in centred if abs(d) >= abs(delta_point)) / n_boot

    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot) - 1]

    return {
        "delta_point": delta_point,
        "delta_samples": deltas,
        "p_value_one_sided": p_one,
        "p_value_two_sided": p_two,
        "ci_95": (lo, hi),
    }
```

Points worth defending in review:

- **Resample indices, not (pred, ref) pairs separately.** The entire variance-reduction argument of
  the paired bootstrap depends on both systems seeing the same resampled indices in each iteration.
  Independent resampling of A and B (an easy bug) reproduces the *unpaired* bootstrap and loses the
  variance reduction.
- **Recompute the corpus metric on each resample.** This is the Koehn 2004 insight. Averaging
  sentence-level metrics across the resample is a different statistic (and for BLEU / ROUGE in
  particular, the sentence-level average is known to disagree with corpus-level — Chen & Cherry 2014).
- **Centre the bootstrap distribution at zero for the p-value.** The null hypothesis is "no
  improvement" (delta = 0). To compute `P(observed delta or more extreme | null)`, subtract the
  observed delta from each resample delta and compare the shifted samples to the observed delta.
  This is Efron & Tibshirani 1994 §16.4 and the Koehn 2004 formulation; a common bug is to compute
  `P(resample delta ≤ 0)` directly, which is the CI-method p-value — related but not the same under
  skew.
- **`n_boot ≥ 1000` is the convention; `n_boot = 10 000` is prudent for publication.** Per Efron &
  Tibshirani 1994 §13.3, the Monte Carlo error on a 95 % CI with B = 1000 is ±1–2 % absolute on the
  CI bounds. For release-gate decisions near the margin, go to 10 000.
- **Seed every run.** The report must reproduce. A fixed seed + sorted sample makes the CI bounds
  bit-identical across reruns on the same data.

### 2.3 The metric interface — corpus-level, not sentence-level

```python
# metrics/bleu.py
import math
from collections import Counter
from typing import Sequence


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    predictions: Sequence[Sequence[str]],
    references: Sequence[Sequence[Sequence[str]]],  # list of lists of ref token-seqs
    max_n: int = 4,
) -> float:
    """
    Corpus-level BLEU (Papineni et al. 2002) with uniform weights and brevity penalty.
    `references` is a list of reference *lists* per prediction (BLEU supports multi-reference).
    This is the statistic bootstrapped in §2.2; it is NOT sentence-BLEU averaged.
    """
    numerators = [0] * max_n
    denominators = [0] * max_n
    hyp_len = 0
    ref_len_sum = 0
    for hyp, refs in zip(predictions, references):
        hyp_len += len(hyp)
        # Brevity-penalty reference length: closest to |hyp| (Papineni convention)
        ref_len_sum += min((abs(len(r) - len(hyp)), len(r)) for r in refs)[1]
        for n in range(1, max_n + 1):
            hyp_ngrams = _ngrams(hyp, n)
            # Clipped counts: min over references of per-reference max
            max_ref = Counter()
            for r in refs:
                for g, c in _ngrams(r, n).items():
                    if c > max_ref[g]:
                        max_ref[g] = c
            overlap = sum(min(c, max_ref[g]) for g, c in hyp_ngrams.items())
            numerators[n - 1] += overlap
            denominators[n - 1] += max(sum(hyp_ngrams.values()), 1)
    precisions = [(numerators[i] / denominators[i]) if denominators[i] else 0.0 for i in range(max_n)]
    if min(precisions) == 0.0:
        return 0.0
    log_mean = sum(math.log(p) for p in precisions) / max_n
    bp = 1.0 if hyp_len >= ref_len_sum else math.exp(1 - ref_len_sum / max(hyp_len, 1))
    return bp * math.exp(log_mean) * 100.0
```

```python
# metrics/classification.py
from typing import Sequence


def accuracy(preds: Sequence[str], refs: Sequence[str]) -> float:
    return sum(1 for p, r in zip(preds, refs) if p == r) / max(len(refs), 1)


def macro_f1(preds: Sequence[str], refs: Sequence[str]) -> float:
    labels = set(refs) | set(preds)
    f1s = []
    for lab in labels:
        tp = sum(1 for p, r in zip(preds, refs) if p == lab and r == lab)
        fp = sum(1 for p, r in zip(preds, refs) if p == lab and r != lab)
        fn = sum(1 for p, r in zip(preds, refs) if p != lab and r == lab)
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0); continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0
```

Design notes:

- **All metrics are corpus-level functions `(preds, refs) → float`.** The bootstrap routine does not
  care what the metric *is*; it only needs the ability to recompute it on a resample. BLEU, chrF,
  ROUGE, accuracy, F1, perplexity-on-held-out — all plug in.
- **BLEU is written out explicitly (not called as `sacrebleu`).** The reason is pedagogical: a
  reviewer reading this must see that the bootstrap is resampling the *inputs* to the n-gram
  counters, not resampling the per-sentence BLEUs. In production, swap to `sacrebleu.corpus_bleu`
  with the published signature for reproducibility; the bootstrap wrapper is unchanged.
- **The references list is `list[list[tokens]]` not `list[tokens]`.** BLEU supports multiple
  references. The bootstrap resample must preserve the reference-list structure per item; a common
  bug is flattening references and resampling the flat list.

### 2.4 Confidence intervals — percentile vs. BCa

The simple percentile CI in §2.2 is unbiased only if the bootstrap distribution is symmetric about
the point estimate. For metrics that are skewed near the boundary (BLEU near 0, macro-F1 with one
near-zero class), use the **BCa (bias-corrected and accelerated)** CI from Efron 1987.

```python
# bootstrap.py (continued)
from math import sqrt
from statistics import NormalDist


def bca_ci(
    samples: list[float],
    point: float,
    jackknife: list[float],
    ci: float = 0.95,
) -> tuple[float, float]:
    """
    Efron 1987 BCa interval.
      samples : B sorted bootstrap deltas
      point   : the observed delta (statistic computed on the full data)
      jackknife: leave-one-out deltas (length n)
    """
    nd = NormalDist()
    below = sum(1 for s in samples if s < point) / len(samples)
    z0 = nd.inv_cdf(below) if 0 < below < 1 else 0.0
    mean_j = sum(jackknife) / len(jackknife)
    num = sum((mean_j - j) ** 3 for j in jackknife)
    den = 6.0 * (sum((mean_j - j) ** 2 for j in jackknife) ** 1.5)
    a = num / den if den != 0 else 0.0
    alpha = (1 - ci) / 2
    z_lo = nd.inv_cdf(alpha)
    z_hi = nd.inv_cdf(1 - alpha)
    def adj(z):
        return nd.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo_p = adj(z_lo); hi_p = adj(z_hi)
    lo = samples[max(0, int(lo_p * len(samples)))]
    hi = samples[min(len(samples) - 1, int(hi_p * len(samples)))]
    return lo, hi
```

Points worth defending:

- **BCa adds the leave-one-out jackknife.** `n` extra metric evaluations, each on `n-1` items.
  For large test sets this is `O(n²)` in the metric; cache incremental n-gram counts when it matters.
  Falling back to the percentile CI is defensible if the bootstrap distribution is visually
  symmetric and `n_boot ≥ 10 000` (Efron & Tibshirani 1994 §14.3).
- **Report the CI *method*, not just the interval.** A model card that says "95 % CI [+0.3, +1.1]"
  must also record "percentile method, B = 1000" or "BCa, B = 10 000"; the two are not interchangeable
  across reviews.

### 2.5 Multiple-comparisons correction — the three standard procedures

```python
# corrections.py
from typing import Sequence


def bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """
    Reject H0 for test i if p_i <= alpha / m. Strongest FWER control, least power.
    """
    m = len(p_values)
    thresh = alpha / m
    return [p <= thresh for p in p_values]


def holm(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """
    Holm 1979 step-down FWER procedure. Uniformly more powerful than Bonferroni.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        if p_values[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break   # step-down halts on the first non-rejection
    return reject


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """
    BH 1995 FDR procedure. Rejects test i if p_i <= (rank_i / m) * alpha.
    Weaker than FWER but typically much more powerful; use when you expect many real effects.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    max_rank_rejected = -1
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= rank * alpha / m:
            max_rank_rejected = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= max_rank_rejected:
            reject[idx] = True
    return reject
```

Design notes:

- **Bonferroni is the right default when claims are strict.** If any single slice-level false
  positive is a release-blocker (e.g., fairness slice regression claims), use Bonferroni / Holm.
- **BH is the right default for exploratory slice reporting.** When you are scanning 50 slices
  looking for interesting patterns, BH controls the expected fraction of false discoveries
  rather than any-false-discovery probability.
- **Report the procedure on the card.** "12 slice comparisons, Holm-corrected at α = 0.05, 4
  rejected" is the standard phrasing; a reviewer must be able to reproduce the correction choice.
- **Dror et al. 2017** extends this to the multi-dataset setting (same model evaluated on several
  benchmarks); the logic is identical but the correction applies across datasets, not slices.

### 2.6 Cluster bootstrap — when sentences are not exchangeable

```python
# bootstrap.py (continued)
def paired_cluster_bootstrap_delta(
    preds_a: Sequence,
    preds_b: Sequence,
    refs: Sequence,
    cluster_ids: Sequence[str],   # one id per item; items with the same id move together
    metric: MetricFn,
    n_boot: int = 1000,
    seed: int = 20260101,
) -> dict:
    """
    Two-level resample: clusters with replacement, then all items in each sampled cluster.
    Required when items within a cluster are correlated (same document, same user, same source);
    independent sentence resampling under-estimates variance by the intraclass correlation (Field
    & Welsh 2007; Davison & Hinkley 1997).
    """
    rng = random.Random(seed)
    clusters: dict[str, list[int]] = {}
    for i, c in enumerate(cluster_ids):
        clusters.setdefault(c, []).append(i)
    cluster_names = list(clusters.keys())
    n_c = len(cluster_names)
    m_a = metric(preds_a, refs); m_b = metric(preds_b, refs)
    delta_point = m_b - m_a

    deltas: list[float] = []
    for _ in range(n_boot):
        chosen = [cluster_names[rng.randrange(n_c)] for _ in range(n_c)]
        idx: list[int] = []
        for c in chosen:
            idx.extend(clusters[c])
        sub_refs = [refs[i] for i in idx]
        sub_a = [preds_a[i] for i in idx]
        sub_b = [preds_b[i] for i in idx]
        deltas.append(metric(sub_b, sub_refs) - metric(sub_a, sub_refs))

    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot) - 1]
    centred = [d - delta_point for d in deltas]
    p_one = sum(1 for d in centred if d >= delta_point) / n_boot
    return {
        "delta_point": delta_point, "delta_samples": deltas,
        "p_value_one_sided": p_one, "ci_95": (lo, hi),
    }
```

Design notes:

- **Sample clusters with replacement, items within clusters by inclusion.** Resampling items *within*
  the sampled cluster with replacement is the "two-stage with replacement" variant and is slightly
  more variable; for NLP test sets the Davison & Hinkley 1997 §3.8 recommendation of all-in is
  standard.
- **Document-level MT, dialogue sessions, multi-sentence QA need this.** A test set of 1 000
  sentences drawn from 10 documents has an effective sample size closer to 10 than 1 000 when
  document-level context drives the metric. Ignoring the clustering reports CIs that are 3–10×
  too narrow.

### 2.7 CLI — one-shot report

```python
# cli.py
import argparse, json
from pathlib import Path
from .bootstrap import paired_bootstrap_delta, paired_cluster_bootstrap_delta
from .corrections import holm
from .metrics.bleu import corpus_bleu
from .metrics.classification import accuracy, macro_f1


METRICS = {"bleu": corpus_bleu, "accuracy": accuracy, "macro_f1": macro_f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-a", required=True, type=Path)
    ap.add_argument("--preds-b", required=True, type=Path)
    ap.add_argument("--refs", required=True, type=Path)
    ap.add_argument("--metric", choices=list(METRICS.keys()), required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--cluster-ids", type=Path, default=None)
    ap.add_argument("--slices", type=Path, default=None,
                    help="JSONL with {slice_name, indices} per line; prints per-slice p-values and Holm correction")
    ap.add_argument("--seed", type=int, default=20260101)
    args = ap.parse_args()

    preds_a = json.loads(args.preds_a.read_text())
    preds_b = json.loads(args.preds_b.read_text())
    refs = json.loads(args.refs.read_text())
    metric = METRICS[args.metric]

    if args.cluster_ids:
        clusters = json.loads(args.cluster_ids.read_text())
        overall = paired_cluster_bootstrap_delta(preds_a, preds_b, refs, clusters, metric,
                                                 n_boot=args.n_boot, seed=args.seed)
    else:
        overall = paired_bootstrap_delta(preds_a, preds_b, refs, metric,
                                         n_boot=args.n_boot, seed=args.seed)

    report = {
        "overall": {k: overall[k] for k in ("delta_point", "p_value_one_sided", "ci_95")},
    }
    if args.slices:
        slice_pvs: list[float] = []
        slice_names: list[str] = []
        for line in args.slices.open():
            s = json.loads(line)
            idx = s["indices"]
            sub_a = [preds_a[i] for i in idx]; sub_b = [preds_b[i] for i in idx]
            sub_r = [refs[i] for i in idx]
            res = paired_bootstrap_delta(sub_a, sub_b, sub_r, metric,
                                         n_boot=args.n_boot, seed=args.seed)
            slice_pvs.append(res["p_value_one_sided"])
            slice_names.append(s["slice_name"])
        reject = holm(slice_pvs, alpha=0.05)
        report["slices"] = [
            {"slice": n, "p": p, "holm_rejects": r}
            for n, p, r in zip(slice_names, slice_pvs, reject)
        ]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

## 3. Validation steps

1. **Reproduce the Koehn 2004 published example.** Koehn's paper reports a bootstrap on a 500-sentence
   MT test set with a BLEU delta of +0.8 and a p-value < 0.05. Build the fixture with the same
   synthetic structure (two systems, 500 items, controlled delta) and verify the harness returns
   `p < 0.05`. Catches off-by-one in the p-value formula.
2. **Null distribution has mean ≈ 0 and correct variance.** With System A = System B (identical
   predictions), `delta_point = 0` and the bootstrap distribution should be a point mass at 0;
   `p_value_one_sided ≈ 0.5`. Run on random data where the two systems are independent samples
   from the same distribution — `p_value_one_sided` should be uniform on [0, 1] across 1 000
   simulated runs.
3. **Power on a known-separated case.** Generate two systems where B deterministically beats A by
   5 BLEU points on every sentence; the paired bootstrap must reject at `p < 0.001` even with
   n = 50. If it does not, the pairing is broken.
4. **Paired variance beats unpaired variance.** On a synthetic test set where sentence-level BLEU
   has high between-sentence variance but low A-vs-B per-sentence variance, the paired CI must be
   ~5× narrower than the unpaired CI. This is the empirical demonstration that pairing matters;
   catches an implementation that accidentally resamples A and B independently.
5. **BCa vs. percentile agreement on symmetric data.** For a large (`n ≥ 5 000`) test set with
   near-symmetric per-sentence delta distribution, BCa and percentile CIs agree within the Monte
   Carlo error of `n_boot`. Where they disagree significantly, the distribution is skewed and BCa
   is preferred.
6. **Cluster bootstrap variance scaling.** Build a synthetic test set of 1 000 sentences in 10
   clusters of 100, where sentences within a cluster have delta = +5 and across clusters have delta
   = -5 half the time. Independent-sentence bootstrap will report a very narrow CI centred on the
   sentence-level mean; cluster bootstrap will report a wider CI with CIs including 0. The ratio
   of the two CI widths equals the design-effect √(1 + ρ(m-1)) where ρ is intraclass correlation
   and m is cluster size (Kish 1965).
7. **Multiple-comparisons sanity.** Generate 20 slices where only slice 1 has a real effect and
   slices 2..20 are null. Bonferroni, Holm, and BH should all reject slice 1; none should reject
   any of slices 2..20 at `α = 0.05` in >5 % of simulations.
8. **Determinism.** Two runs with the same `--seed` must produce bit-identical output. Catches
   dict iteration, floating-point associativity, or thread-pool nondeterminism.
9. **Cross-validate against `scipy.stats.bootstrap`.** For a simple statistic (difference in means),
   our percentile CI must agree with scipy's to the Monte Carlo tolerance of `n_boot`.
10. **Bootstrap with `n_boot = 10 000` vs. `n_boot = 1 000`.** The CI bounds should move by ≤ 0.1 of
    the metric unit; if they move more, the signal is too close to noise for `B = 1 000` to resolve
    and the published `B` should be `10 000`.

## 4. Grading rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Paired test** | Indices resampled jointly; corpus metric recomputed on each sample | Resampled but sentence-metrics averaged | Independent resampling of A and B |
| **Metric computed on the resample** | Corpus-level statistic recomputed from resampled items | Sentence-level average used as proxy | Pre-computed sentence scores averaged |
| **p-value formulation** | Centred bootstrap (Koehn 2004 / Efron & Tibshirani §16.4); two-sided available | One-sided only, uncentred | p-value = fraction of resamples with delta > 0 (not the same statistic) |
| **CI reporting** | Both bounds reported with method (percentile or BCa) and `n_boot` | Point estimate with stddev | None |
| **n_boot** | ≥ 1 000; ≥ 10 000 for release-gate decisions near the margin | Documented but low | Unspecified |
| **Cluster bootstrap** | Used when items are grouped (document, session, user); clusters resampled | Mentioned, not used | Independent resampling on clustered data |
| **Multiple comparisons** | Bonferroni / Holm / BH applied with procedure stated | One correction applied, procedure not named | Unadjusted slice p-values |
| **Reproducibility** | Seed fixed; sample sorted; two-run bit-identical output | Seed fixed, output varies on re-run | No seed |
| **Metric suite portability** | Works on any `(preds, refs) → float` metric (BLEU, chrF, ROUGE, F1, accuracy); no metric-specific assumptions in the bootstrap loop | Hard-coded to one metric | Metric computation inside the resample loop (recomputation each iter is fine; metric-specific bootstrap is not) |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks flagged (p-hacking via slice selection, test-set leakage affecting both systems) | Mentioned | Ignored |
| **Documentation / card** | NIST AI RMF *Map* (what tests were run), *Measure* (p-values + CIs + correction), *Manage* (release-gate rule using CI bounds) | Partial | Neither |

## 5. Common mistakes

- **Running the bootstrap on sentence-level metrics instead of the corpus-level metric.** Sentence-BLEU
  averaged ≠ corpus-BLEU (Chen & Cherry 2014); the published number for release is corpus-BLEU, so
  the significance test must bootstrap that statistic. The common bug is to precompute per-sentence
  scores and bootstrap the mean.
- **Independent resampling of System A and System B.** A one-character typo that turns `idx` into
  two separate `idx_a` and `idx_b` loops. The test then becomes the unpaired bootstrap; variance
  balloons; significant results disappear. Golden test (§3.4) catches it.
- **Using sentence-level bootstrap on document-level test sets.** Document-level MT (WMT document-
  level tasks, long-form summarisation) has strong within-document correlation; the sentence
  bootstrap under-estimates CI by the design-effect. Use `paired_cluster_bootstrap_delta`.
- **Reporting the lowest of 20 slice p-values as *the* p-value.** With `α = 0.05` and 20 independent
  tests, the FWER is 1 - 0.95^20 ≈ 64 %. Apply Bonferroni / Holm / BH.
- **Confusing p-value with effect size.** `p < 0.001` with a delta of 0.02 BLEU means the sign is
  confident but the effect is tiny; the release decision needs the CI, not the p-value. Always
  publish both.
- **Reporting only overall significance and not per-slice.** A release gated on "corpus BLEU
  improves" can ship a model that regresses on a fairness slice. Per-slice bootstrap + Holm
  correction is the standard.
- **Using `n_boot = 100` for a release decision.** Monte Carlo error at B = 100 is ±5 % absolute on
  the CI bounds; you cannot distinguish `p = 0.03` from `p = 0.07`. `B = 1 000` is the floor; `B =
  10 000` for publication.
- **Bootstrap on a non-exchangeable statistic.** Perplexity on a language model is sentence-length-
  weighted; the bootstrap must preserve that weighting. The generic `(preds, refs) → float`
  interface is fine as long as the metric is a function of the resampled items alone; implicit
  dependencies (e.g., a vocabulary shared with training) are the hidden gotcha.
- **Approximate randomisation vs. bootstrap confusion.** Approximate randomisation (Riezler &
  Maxwell 2005) is a different test: swap predictions between A and B with probability 0.5 per item
  and recompute. It is slightly more powerful at the cost of more compute. Both are valid; name the
  one you used.
- **Test-set leakage making both systems equally strong on the leaked subset.** If both A and B were
  trained on contaminated data (exercise-03), the paired bootstrap will not detect it — the delta
  is correctly computed, but the interpretation is wrong. Pair the significance test with a
  contamination check.
- **Bootstrap on tiny test sets (< 50 items).** The bootstrap distribution is dominated by sampling
  of a few items; CIs are unreliable. Use exact tests (permutation) or a larger test set.
- **Running unpaired bootstrap because the two systems used different test sets.** If A was
  evaluated on test v1 and B on test v2, you do not have a paired test at all. Fix the test set
  first; do not patch over it with a test that assumes pairing.
- **Reporting `p = 0.000` on the model card.** No bootstrap with `B = 10 000` can report `p < 1/B`.
  The correct reporting is `p < 1e-4`; writing `p = 0.000` is a precision overstatement.

## 6. References

Primary sources used in this solution:

- Koehn, P. (2004). *Statistical Significance Tests for Machine Translation Evaluation*. EMNLP. <https://aclanthology.org/W04-3250/> — the paired bootstrap recipe for MT; the §2.2 formulation is a direct reproduction.
- Efron, B., Tibshirani, R. (1994). *An Introduction to the Bootstrap*. Chapman & Hall. — §13.3 (`n_boot` requirements), §14.3 (percentile vs. BCa), §16.4 (hypothesis tests with the bootstrap).
- Efron, B. (1987). *Better Bootstrap Confidence Intervals*. JASA 82(397). — the BCa interval reproduced in §2.4.
- Davison, A. C., Hinkley, D. V. (1997). *Bootstrap Methods and their Application*. Cambridge. §3.8 — the cluster bootstrap formulation used in §2.6.
- Dror, R. et al. (2018). *The Hitchhiker's Guide to Testing Statistical Significance in NLP*. ACL. <https://aclanthology.org/P18-1128/> — the modern NLP survey; recommends paired bootstrap + approximate randomization; documents when each is appropriate.
- Riezler, S., Maxwell, J. T. (2005). *On Some Pitfalls in Automatic Evaluation and Significance Testing for MT*. ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for MT and/or Summarization. <https://aclanthology.org/W05-0908/> — bootstrap vs. approximate randomization; the §5 "which test to use" guidance.
- Berg-Kirkpatrick, T., Burkett, D., Klein, D. (2012). *An Empirical Investigation of Statistical Significance in NLP*. EMNLP. <https://aclanthology.org/D12-1091/> — empirical case that NLP papers over-claim significance; motivates §4's rubric rigour.
- Dror, R. et al. (2017). *Replicability Analysis for Natural Language Processing: Testing Significance with Multiple Datasets*. TACL. <https://aclanthology.org/Q17-1033/> — multi-dataset extension of the significance framework; the Holm / BH analogues across benchmarks.
- Søgaard, A., Johannsen, A., Plank, B., Hovy, D., Alonso, H. M. (2014). *What's in a p-value in NLP?*. CoNLL. <https://aclanthology.org/W14-1601/> — mis-specification case studies; reproduces the Koehn bootstrap.
- Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics 6. — the Holm step-down procedure in §2.5.
- Benjamini, Y., Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*. Journal of the Royal Statistical Society 57(1). — the BH procedure in §2.5.
- Papineni, K. et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. ACL. <https://aclanthology.org/P02-1040/> — the corpus-BLEU statistic the bootstrap is applied to.
- Chen, B., Cherry, C. (2014). *A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU*. WMT. <https://aclanthology.org/W14-3346/> — sentence-BLEU ≠ corpus-BLEU; justifies the §2.3 insistence on corpus-level recomputation.
- Field, C. A., Welsh, A. H. (2007). *Bootstrapping Clustered Data*. JRSSB 69(3). — the cluster-bootstrap statistical foundation.
- Kish, L. (1965). *Survey Sampling*. Wiley. — the design-effect formulation used in the §3.6 validation.
- Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores* (sacrebleu). WMT. <https://aclanthology.org/W18-6319/> — the reproducibility-of-BLEU argument; informs the §2.3 "swap to `sacrebleu` in production" note.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML06 AI Supply Chain Attacks*: if the test set itself is tampered (either systems trained on it, or predictions post-processed by a compromised pipeline), both System A and System B shift equally and the paired bootstrap reports no delta — but the *absolute* numbers are wrong. Pair the test with the exercise-03 contamination check. *ML05 Model Theft*: public bootstrap configurations make it easier for an attacker to design adversarial perturbations that evade the test; disclosing `n_boot` and the seed is a research convention, but the slice composition should be controlled.
- MITRE ATLAS — <https://atlas.mitre.org/> — *AML.T0043 Craft Adversarial Data* applies when benchmark optimisation drives subtle overfitting; the per-slice significance test is the detection control. *AML.T0024 Exfiltrate via Cyber Means* is tangentially relevant: public leaderboards with per-slice numbers leak information about the slice distribution.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record the test design (slices, cluster structure, correction procedure, `n_boot`, seed) as part of model-card metadata. *Measure*: the (p-value, delta, CI, correction) tuple per slice is the Measure artefact; `ReleaseGate` style acceptance rules key on CI lower bound, not the p-value. *Manage*: the correction procedure is a change-controlled decision — moving from BH to Bonferroni mid-project is a documented policy change. *Govern*: replicability (Dror et al. 2017) and reporting conventions (Post 2018 for BLEU) feed into the Govern function via the model-card template.
