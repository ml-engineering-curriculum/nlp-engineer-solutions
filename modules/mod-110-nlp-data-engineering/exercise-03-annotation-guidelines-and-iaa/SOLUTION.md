# exercise-03 — Annotation Guidelines and Inter-Annotator Agreement

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-110-nlp-data-engineering/exercises/exercise-03-annotation-guidelines-and-iaa.md`.
> The target learning outcomes implied by that filename: author the two artefacts that make a labelled dataset defensible — a **versioned annotation guideline** that covers scope, label definitions, decision rules, hard-case examples, and an escalation path; and the **inter-annotator-agreement (IAA)** calculation that proves a human can follow the guideline (Cohen's κ for two annotators, Fleiss's κ / Krippendorff's α for ≥3), including an **adjudication process** that turns disagreements back into guideline updates and an audit log that lets a downstream reviewer trace every gold label back to the annotators and the version of the guideline they used.

## 1. Solution overview

Annotation disputes are rarely about individual labels; they are almost always symptoms of an under-specified guideline. The job is to design the guideline *and* the measurement that forces the gaps to the surface:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Guideline is a label list, not a decision procedure** | Annotators are shown `{POSITIVE, NEGATIVE, NEUTRAL}` and told "label the sentiment". Each annotator settles on a private theory of sarcasm, mixed sentiment, and neutrality; IAA is low and stays low | Guideline is a flowchart: definitions → decision rules → tie-breakers → worked positive and negative examples → explicit *out-of-scope* handling (e.g., non-English → `SKIP`). Published with a version number; annotators record the version on every batch |
| **Chance-uncorrected agreement (raw %)** | "92 % agreement" on a 90-% majority class is indistinguishable from flipping coins. You cannot compare it across projects, across label sets, or across time | Use chance-corrected statistics: **Cohen's κ** for exactly 2 annotators and a nominal label set; **Fleiss's κ** for ≥3 annotators; **Krippendorff's α** for anything beyond nominal (ordinal/interval/ratio, or variable numbers of annotators per item) |
| **Single-stage labelling with no adjudication** | Annotators disagree, labels go to training, the disagreement signal is thrown away. The dataset learns noise and the guideline learns nothing | Every disagreed item routes to **adjudication** by a senior annotator or committee; the resolution is recorded with a *reason* that feeds the next guideline revision. The disagreement rate per label is the strongest guideline-health metric |
| **IAA measured once, at the start** | Agreement drifts over time (annotator fatigue, guideline revisions, new edge cases); without re-measurement the "0.80 κ" cited on page 1 of the dataset card may have been true six weeks ago | Pin a **calibration set** (50–200 items) re-scored on every batch; publish the κ time-series in the dataset card |

The deliverable is a four-component harness:

1. **Guideline template** — markdown with pinned sections (Scope, Label Definitions, Decision Rules, Hard Cases, Out-of-Scope, Escalation, Changelog) and a semantic version (`guideline@1.3.2`).
2. **IAA calculator** — Cohen's κ, Fleiss's κ, Krippendorff's α; per-label (one-vs-rest) breakdown; confidence intervals via non-parametric bootstrap.
3. **Adjudication workflow** — disagreed items routed to a senior reviewer; adjudicator records a resolution *and* a reason tag from a controlled vocabulary (`guideline-gap`, `annotator-error`, `legitimately-ambiguous-mark-as-such`).
4. **Audit log** — one row per `(item_id, annotator_id, label, guideline_version, timestamp)`; the gold label is derivable from this log, not stored separately.

Published standards and reference libraries:

- **Cohen 1960** — the original two-annotator κ.
- **Fleiss 1971** — κ for ≥3 annotators (confusingly named; it is a different statistic from Cohen's).
- **Krippendorff 1980 / 2018** — α, the most general; handles missing data, variable annotator counts, and ordinal/interval/ratio levels.
- **MAMA cycle** (Pustejovsky & Stubbs 2012, *Natural Language Annotation for Machine Learning*) — Model-Annotate-Model-Annotate; the canonical reference for how guideline + IAA + adjudication iterate together.
- **Datasheets for Datasets** (Gebru et al. 2018/2021) and **Data Statements for NLP** (Bender & Friedman 2018) — the templates the per-batch IAA numbers feed into on release.
- Reference IAA libraries: **`nltk.metrics.AnnotationTask`** (pure-Python, implements κ and α), **`statsmodels.stats.inter_rater`** (Fleiss κ + Cohen κ), **`irr` in R** (used for cross-checking), and **`simpledorff`** (Python α implementation).

## 2. Worked implementation

### 2.1 Project layout

```
annotate_bake/
  guideline/
    guideline.md               # the guideline itself (versioned)
    changelog.md               # per-version rationale
    calibration_set.jsonl      # 50-200 pinned items
  iaa/
    cohen.py                   # two-annotator kappa
    fleiss.py                  # multi-annotator kappa
    krippendorff.py            # alpha, with ordinal/nominal distance fns
    bootstrap.py               # CIs
    per_label.py               # one-vs-rest breakdown
  adjudication/
    workflow.py                # disagreement -> queue -> reviewer
    reasons.py                 # controlled vocabulary
  audit/
    log.py                     # append-only JSONL
    gold.py                    # materialise gold label from log
  run.py                       # CLI: batch report
  tests/
    test_cohen_fleiss_alpha.py
    test_bootstrap_ci.py
    test_adjudication_reasons.py
    test_gold_derivation.py
```

### 2.2 Guideline template — scope, rules, cases, escalation

```markdown
# Sentiment classification — annotation guideline v1.3.2

## 0. Changelog
- v1.3.2 (2026-04-07): Added *reclaimed slur* rule (§3.4) after batch #17 disagreement spike.
- v1.3.1 (2026-03-18): Clarified sarcasm (§3.2) — tie-break goes to surface sentiment.
- v1.3.0 (2026-03-01): Added `SKIP_NONENGLISH` out-of-scope label (§4).

## 1. Scope
Short English social-media posts (max 280 characters). Labels assigned at the
*post* level, not the sentence level.

## 2. Label definitions
| Label | Definition |
|---|---|
| `POSITIVE` | The author expresses approval, satisfaction, or endorsement of the main entity in the post. |
| `NEGATIVE` | The author expresses disapproval, dissatisfaction, or criticism of the main entity. |
| `NEUTRAL` | The post is factual, descriptive, or question-like without evident author stance. |

The *main entity* is the subject of the main clause unless an @mention clearly
dominates the post, in which case the @mention is the main entity.

## 3. Decision rules
1. **Mixed sentiment.** If a post contains both positive and negative claims
   about the main entity, label by the sentiment of the *final* independent
   clause (the author's last word tends to be the operative stance).
2. **Sarcasm.** If sarcasm is syntactically unmarked (no `/s`, no emoji cue),
   label by the *surface* sentiment. Rationale: a downstream classifier will
   not reliably detect sarcasm; forcing annotators to detect it transfers the
   ambiguity to κ.
3. **Reclaimed language and in-group terms** are not automatically negative;
   label by evident author stance and context.
4. **Questions** are `NEUTRAL` unless they are rhetorical and the stance is
   obvious (`"Can you believe how good this is?"` is `POSITIVE`).

## 4. Out-of-scope → `SKIP_*` codes
- `SKIP_NONENGLISH` — not predominantly English prose.
- `SKIP_PII` — contains unredacted PII; route to the scrubber (exercise-02).
- `SKIP_TOXIC` — triggers the hard-toxic threshold; route to content-safety.

## 5. Hard cases (10 worked examples)
*(Each with the post, the label, and a one-sentence rule-trace.)*

## 6. Escalation
If after re-reading §3 the annotator cannot pick a label with ≥60 % confidence:
- Use `UNSURE` (not `NEUTRAL`).
- These are routed to the adjudicator, who may: (a) resolve under the current
  guideline; (b) flag the item for the next guideline revision; (c) mark
  `legitimately-ambiguous` — the gold label becomes `AMBIGUOUS` and the item
  is excluded from accuracy reporting (included in robustness slices).
```

Points worth defending in review:

- **Scope is the first section.** The commonest cause of low IAA is annotators disagreeing on whether an item should have been labelled at all. Scope + out-of-scope labels (`SKIP_*`) make the frame explicit.
- **Decision rules, not label definitions, carry the load.** A `NEGATIVE` definition is useless at the margin (sarcasm, mixed sentiment, questions); the rule for each of those is the artefact annotators actually use.
- **Explicit tie-breakers.** Rule 3.1 ("label by the final independent clause") is arbitrary but *consistent*; any rule beats no rule. Record the rationale in the changelog.
- **`UNSURE` is a first-class label, not a cop-out.** Forcing annotators to pick a label they are not confident in corrupts the training distribution. `UNSURE` is routed to adjudication; it is one of the strongest guideline-health signals.
- **Guideline is versioned semver.** Minor version bumps on clarifications, major on new labels. Every annotation row records the guideline version it was labelled under; without this the audit log is unusable after any revision.

### 2.3 Cohen's κ — two annotators, nominal labels

```python
# iaa/cohen.py
from collections import Counter


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """
    Cohen 1960. Returns the chance-corrected agreement coefficient.
    a, b must be equal-length label sequences from two annotators on the same items.
    """
    assert len(a) == len(b) and len(a) > 0, "need paired labels"
    n = len(a)
    labels = set(a) | set(b)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = Counter(a); cb = Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0   # edge: everyone agreed on one label
    return (po - pe) / (1 - pe)
```

Points to defend:

- **κ depends on the marginal distribution.** Two annotators who agree on 95 % of items may have low κ if the agreed-upon label is 95 % of the dataset. This is a *feature*: it is why κ is used instead of raw agreement.
- **Negative κ means worse than chance.** Rare, but it happens when annotators systematically disagree. Flag it; it indicates a guideline misreading.
- **Bootstrap the CI.** κ on a 100-item sample has a ±0.1 CI, enough that a "0.75 κ" vs "0.78 κ" release-over-release comparison is noise. Always publish κ with a non-parametric CI (§2.6).

Rule-of-thumb interpretations (Landis & Koch 1977, still the standard citation; note these are suggestive, not normative):

| κ | Landis & Koch label |
|---|---|
| < 0.00 | poor |
| 0.00–0.20 | slight |
| 0.21–0.40 | fair |
| 0.41–0.60 | moderate |
| 0.61–0.80 | substantial |
| 0.81–1.00 | almost perfect |

Caveat: for subjective tasks (sentiment, abstract summarisation quality) κ=0.6 is the operational ceiling; for objective tasks (factual NER, well-defined relation extraction) κ=0.85 is the operational floor.

### 2.4 Fleiss's κ — three or more annotators

```python
# iaa/fleiss.py
from collections import Counter
from math import isclose


def fleiss_kappa(matrix: list[list[int]]) -> float:
    """
    Fleiss 1971. Expects a matrix of shape (n_items, n_categories),
    where matrix[i][k] is the number of annotators that assigned category k
    to item i. The number of annotators per row must be constant.
    """
    n = len(matrix)
    n_annotators = sum(matrix[0])
    assert all(sum(row) == n_annotators for row in matrix), "constant N required"

    # Per-item agreement
    def row_p(row: list[int]) -> float:
        return (sum(x * (x - 1) for x in row)) / (n_annotators * (n_annotators - 1))

    p_bar = sum(row_p(r) for r in matrix) / n

    # Chance-expected agreement
    totals = [sum(matrix[i][k] for i in range(n)) for k in range(len(matrix[0]))]
    pj = [t / (n * n_annotators) for t in totals]
    pe = sum(x * x for x in pj)
    if isclose(pe, 1.0):
        return 1.0 if isclose(p_bar, 1.0) else 0.0
    return (p_bar - pe) / (1 - pe)
```

Design notes:

- **Fleiss's κ requires a *constant* number of annotators per item.** When annotators drop in and out across items (the common case in crowdsourcing), either (a) subsample to a constant N or (b) use Krippendorff's α, which handles variable-N natively.
- **Fleiss's κ reduces to a different statistic than Cohen's κ even at N=2.** Do not switch between them within a single project.
- **Per-label (one-vs-rest) κ is more informative than overall κ.** It isolates which labels are the source of disagreement; an overall κ of 0.72 that decomposes as (POSITIVE: 0.88, NEGATIVE: 0.85, NEUTRAL: 0.41) tells you where to concentrate guideline work.

### 2.5 Krippendorff's α — the general case

```python
# iaa/krippendorff.py
import numpy as np
from itertools import combinations


def nominal_distance(a, b) -> float:
    return 0.0 if a == b else 1.0


def ordinal_distance(levels: list) -> callable:
    rank = {v: i for i, v in enumerate(levels)}
    def d(a, b):
        return (rank[a] - rank[b]) ** 2
    return d


def krippendorff_alpha(annotations: dict[str, dict[str, object]],
                       distance=nominal_distance) -> float:
    """
    annotations[item_id][annotator_id] = label  (missing values allowed; just
    omit the annotator from that item's dict).

    Nominal distance for categorical; ordinal_distance(levels) for ordinal.
    Returns α (Krippendorff 2018 Chapter 11 formulation).
    """
    item_ids = [i for i, a in annotations.items() if len(a) >= 2]
    if not item_ids:
        return float("nan")

    # Observed disagreement
    do_num, do_den = 0.0, 0.0
    all_labels: list = []
    for i in item_ids:
        labs = list(annotations[i].values())
        m = len(labs)
        all_labels.extend(labs)
        for a, b in combinations(labs, 2):
            do_num += distance(a, b)
        do_den += m - 1
    if do_den == 0:
        return float("nan")
    do = do_num / do_den

    # Expected disagreement (pairs sampled from the pool of all labels)
    N = len(all_labels)
    de_num = 0.0
    for a, b in combinations(all_labels, 2):
        de_num += distance(a, b)
    de = (2.0 * de_num) / (N * (N - 1)) if N > 1 else 0.0
    if de == 0:
        return 1.0 if do == 0 else 0.0
    return 1 - do / de
```

Why α is the "default" in the literature (Artstein & Poesio 2008):

- Handles **missing annotations** natively — crucial for crowdsourced data.
- Handles **ordinal / interval / ratio** label spaces via the distance function — essential for rating scales (`1`-`5` Likert, `age-group` bins).
- Reduces to the correct chance-corrected statistic for the two-nominal-annotator case; comparable to Cohen's κ in that limit.

### 2.6 Bootstrap CIs — always publish the interval

```python
# iaa/bootstrap.py
import random
from typing import Callable


def bootstrap_ci(items: list, statistic: Callable, n_boot: int = 1000,
                 ci: float = 0.95, seed: int = 20260101) -> tuple[float, float]:
    """
    Non-parametric bootstrap CI for an IAA statistic. items is the list of
    (annotator_a_label, annotator_b_label) tuples (for Cohen) or rows for
    Fleiss/α. statistic(items) returns a float.
    """
    rng = random.Random(seed)
    n = len(items)
    samples: list[float] = []
    for _ in range(n_boot):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        samples.append(statistic(sample))
    samples.sort()
    lo = int((1 - ci) / 2 * n_boot)
    hi = int((1 + ci) / 2 * n_boot) - 1
    return samples[lo], samples[hi]
```

Why the CI matters: Krippendorff 2018 §11.6 and Artstein & Poesio 2008 §7 both argue that unless the CI excludes the "unacceptable" threshold (typically α < 0.667 for exploratory tasks, α < 0.800 for release), the point estimate alone is not decision-ready.

### 2.7 Adjudication workflow — controlled vocabulary for the *why*

```python
# adjudication/reasons.py
from enum import Enum


class Reason(str, Enum):
    GUIDELINE_GAP          = "guideline-gap"
    GUIDELINE_AMBIGUITY    = "guideline-ambiguity"
    ANNOTATOR_ERROR        = "annotator-error"
    LEGITIMATELY_AMBIGUOUS = "legitimately-ambiguous"
    OUT_OF_SCOPE           = "out-of-scope"
```

```python
# adjudication/workflow.py
from dataclasses import dataclass
from .reasons import Reason


@dataclass
class Resolution:
    item_id: str
    final_label: str              # the gold label, or "AMBIGUOUS"
    adjudicator_id: str
    reason: Reason
    note: str                      # free-text rationale, feeds changelog
    guideline_version: str


def route_disagreements(batch, k: int = 2) -> list[str]:
    """
    Return item_ids that must go to adjudication. Rule:
      - any item where fewer than k annotators agreed
      - any item where any annotator used UNSURE
    """
    queue: list[str] = []
    for item_id, labels in batch.items():
        counts = {}
        for a, l in labels.items():
            counts[l] = counts.get(l, 0) + 1
        if "UNSURE" in counts:
            queue.append(item_id); continue
        if max(counts.values()) < k:
            queue.append(item_id)
    return queue
```

Points worth defending:

- **Controlled vocabulary for `reason`.** Free-text rationales are ungrepable; a controlled vocabulary of five values lets you compute "disagreement reasons by batch" and prioritise guideline edits.
- **`LEGITIMATELY_AMBIGUOUS` is a valid gold label.** Items where reasonable annotators legitimately disagree are not trainable with a single label; marking them as such (and excluding them from train/accuracy evaluation but including them in robustness slices) is better than forcing a majority vote.
- **Adjudicator records `guideline_version`.** When v1.3.2 ships a new rule for sarcasm, it is because v1.3.1 adjudications tagged `guideline-ambiguity` in sarcastic posts. The version link is how you prove it.

### 2.8 Audit log — gold is derivable, not stored

```python
# audit/log.py
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class Event:
    item_id: str
    annotator_id: str
    label: str
    guideline_version: str
    timestamp: str
    kind: str                        # "annotation" | "adjudication"


def append(path: Path, event: Event) -> None:
    with path.open("a") as f:
        f.write(json.dumps(asdict(event), separators=(",", ":")) + "\n")
```

```python
# audit/gold.py
import json
from pathlib import Path


def materialise_gold(log_path: Path) -> dict[str, str]:
    """
    Derive the gold label per item. Rule:
      - If any adjudication exists for item_id, last adjudication wins.
      - Else majority vote among annotation events; tie -> AMBIGUOUS.
    """
    events_by_item: dict[str, list[dict]] = {}
    with log_path.open() as f:
        for line in f:
            e = json.loads(line)
            events_by_item.setdefault(e["item_id"], []).append(e)

    gold: dict[str, str] = {}
    for item_id, evs in events_by_item.items():
        adjs = [e for e in evs if e["kind"] == "adjudication"]
        if adjs:
            gold[item_id] = adjs[-1]["label"]
            continue
        votes: dict[str, int] = {}
        for e in evs:
            votes[e["label"]] = votes.get(e["label"], 0) + 1
        top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
        gold[item_id] = top[0][0] if (len(top) == 1 or top[0][1] > top[1][1]) else "AMBIGUOUS"
    return gold
```

Why "gold is derivable, not stored":

- **Reproducibility.** A later change to the aggregation rule (majority vs median vs MACE) must not require re-collecting the data. The raw annotations are the source of truth; `gold` is a view.
- **Auditability.** "Who said what, under which guideline version, at what time" is one file. Every incident response begins there.
- **Re-weighting.** When a high-performing annotator leaves and a batch is suddenly biased by their absence, you can re-derive gold with equal annotator weights (MACE / Dawid-Skene) without re-collecting.

### 2.9 CLI — per-batch report

```python
# run.py
import argparse, json
from pathlib import Path
from .audit.log import append, Event
from .audit.gold import materialise_gold
from .iaa.cohen import cohen_kappa
from .iaa.fleiss import fleiss_kappa
from .iaa.krippendorff import krippendorff_alpha, nominal_distance
from .iaa.bootstrap import bootstrap_ci
from .adjudication.workflow import route_disagreements


def report(log_path: Path, out_path: Path) -> None:
    # Collect (item_id -> annotator_id -> label)
    by_item: dict[str, dict[str, str]] = {}
    versions: set[str] = set()
    with log_path.open() as f:
        for line in f:
            e = json.loads(line)
            if e["kind"] != "annotation":
                continue
            by_item.setdefault(e["item_id"], {})[e["annotator_id"]] = e["label"]
            versions.add(e["guideline_version"])

    # Per-label breakdown via one-vs-rest Cohen κ averaged across annotator pairs
    # (shown here abbreviated; see iaa/per_label.py)

    alpha = krippendorff_alpha(by_item, distance=nominal_distance)
    adjud_queue = route_disagreements(by_item, k=2)
    out = {
        "alpha": alpha,
        "guideline_versions_in_batch": sorted(versions),
        "n_items": len(by_item),
        "n_disagreed": len(adjud_queue),
        "disagreed_fraction": len(adjud_queue) / max(1, len(by_item)),
    }
    out_path.write_text(json.dumps(out, indent=2))
```

## 3. Validation steps

1. **Cohen κ reference values.** Compute κ on Cohen's original 1960 Table 2 (two annotators, 3 categories, 200 items); expect κ ≈ 0.49. Reproducing published numbers is the only validation that catches sign/denominator bugs.
2. **Fleiss κ reference values.** Compute on Fleiss 1971's Table 2 (6 annotators, 5 categories, 30 items); expect κ ≈ 0.43. Same rationale.
3. **Krippendorff α reference values.** Compute on Krippendorff 2018 §11.3 Table 11.3; expect α ≈ 0.743. The same reference is implemented by R `irr` and Python `simpledorff`; cross-check against those.
4. **Edge cases.** All-agree → κ = α = 1.0; all-disagree-same-marginals → κ = 0; empty input → `nan` (not a crash).
5. **Bootstrap determinism.** `bootstrap_ci(..., seed=20260101)` returns bit-identical bounds across runs.
6. **Per-label κ breakdown.** Simulate a dataset where annotators agree 95 % on `POSITIVE`/`NEGATIVE` and 40 % on `NEUTRAL`; the per-label breakdown must identify `NEUTRAL` as the low-agreement class. The overall κ must sit between them.
7. **Adjudication routing.** Any item with `UNSURE` is routed; any item below the agreement threshold is routed; items above are not. Golden-list test.
8. **Gold derivation.** Append (anno, anno, anno) with two of three agreeing → gold = majority label. Append an adjudication event after → gold = adjudicated label. Catches the "last adjudication wins" rule.
9. **Guideline-version traceability.** Grep the audit log for one `item_id`; the sequence of events must show every `guideline_version` the item was labelled under, in time order. If any event is missing the field, fail the batch.
10. **IAA time-series.** Re-score the calibration set on every batch; publish a plot of α over time. A drop of > 0.05 between batches is a required stop-work signal.
11. **Annotator-level reliability.** For each annotator, compute their per-label κ against the gold set (not against other annotators). An annotator with κ > 0.15 below the batch median is flagged for retraining. (Separate from gold materialisation; do not re-weight gold on this signal without an explicit change-control event.)

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Guideline structure** | Scope + Labels + Decision Rules + Hard Cases + Out-of-Scope + Escalation + Changelog; semver | Missing one or two sections | Label list only |
| **Decision rules** | Explicit tie-breakers for mixed / sarcasm / questions / reclaimed language | Some rules, inconsistent | No tie-breakers |
| **Out-of-scope labels** | `SKIP_*` codes with routing rules | Partial | Everything forced into the label set |
| **IAA statistic choice** | κ for 2 annotators, Fleiss κ for ≥3 constant-N, α for variable-N / ordinal; publicly justified | Raw agreement + one chance-corrected statistic | Raw agreement only |
| **IAA CI** | Bootstrap CI published alongside point estimate | Point estimate only | Not reported |
| **Per-label breakdown** | One-vs-rest κ per label; low-agreement labels called out | Overall only | Nothing |
| **Adjudication** | Disagreed items routed to adjudicator; reason-coded from controlled vocabulary | Resolved but reasons uncoded | Majority vote with no record |
| **Audit log** | Append-only; gold derived from log; every row carries guideline_version | Log present, no version field | None |
| **Calibration set** | 50-200 items re-scored each batch; α time-series published | Calibration done once | None |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks flagged (collusive crowd workers, label-flipping attacks, poisoned guideline) | Mentioned | Ignored |
| **Documentation / card** | NIST AI RMF *Map*: labels, guideline version, annotator pool; *Measure*: κ/α per batch + per-subgroup; Datasheets-for-Datasets fields filled | One of those | Neither |

## 5. Common mistakes

- **Shipping a label list as the guideline.** "Positive, Negative, Neutral — go" produces κ in the 0.3–0.5 range on sentiment; adding one page of decision rules and ten worked examples typically moves it to 0.65–0.75. If κ is below 0.6 and the guideline is a list, write the guideline.
- **Reporting raw agreement.** On any task with one dominant class (`NEUTRAL` is typically 60 %+ in product review sentiment), raw agreement will be near 80 % when annotators are flipping coins on the minority classes. Chance-corrected stats are not optional.
- **Using Cohen's κ with 3+ annotators by averaging pairwise.** Pairwise-averaged Cohen κ is *not* the same statistic as Fleiss's κ. Use Fleiss or α; cite which.
- **Reporting Fleiss κ on variable-N data.** Fleiss requires constant N per item; crowd data usually isn't. Use α. If you have no missing data and pin N = 3, Fleiss is fine and more interpretable.
- **Treating Landis & Koch thresholds as universal.** κ = 0.6 is "moderate" by the paper's label but may be the operational ceiling for a genuinely ambiguous task (humour, offensiveness, subtle sentiment). The threshold is domain-specific; set and publish it.
- **Collecting 3 annotations per item and taking a majority vote with no adjudication.** You get a gold label, but the disagreement signal — the single most useful guideline-health signal — is thrown away.
- **Not pinning a calibration set.** IAA drifts; without a fixed set across batches you cannot separate "the guideline got worse" from "the data got harder". Pin 50–200 items at project start; they never leave the calibration pool.
- **Storing `gold` as a column instead of deriving it.** On the next adjudication revision, you must re-run a migration to update the gold column. Keep the raw annotations; derive gold from the audit log on every pull.
- **Treating `UNSURE` as a bug.** It is a first-class signal; the `UNSURE` rate per label is the forward-looking guideline-health metric.
- **Not carrying `guideline_version` on every annotation.** When the guideline changes, you have no way to tell "this batch was labelled under v1.2 and this one under v1.3"; the audit log becomes unusable.
- **Treating annotator disagreement as noise to be majority-voted away.** Pavlick & Kwiatkowski 2019 showed that for NLI a large fraction of disagreement is *legitimate* — multiple valid interpretations — and treating it as noise discards signal. For subjective tasks, report item-level entropy as a first-class artefact and consider soft labels.
- **Running IAA once at the kickoff and never again.** Guideline versions, annotator fatigue, and dataset drift all move κ. Re-score the calibration set every batch.
- **Reporting IAA on a labelled subset but training on the full labelled set with no agreement filter.** The gold set used in training should be either (a) adjudicated or (b) items where ≥2/3 agreement held. Skipping this leaves adversarial/ambiguous items in the training distribution.
- **Not computing fairness slices of IAA.** Annotator-pool skew (geography, demographic) affects κ on subjective tasks (Sap et al. 2019 on hate-speech annotation by race of annotator). Report per-annotator-group κ on relevant subgroups.

## 6. References

Primary sources used in this solution:

- Cohen, J. (1960). *A Coefficient of Agreement for Nominal Scales*. Educational and Psychological Measurement 20(1). <https://doi.org/10.1177/001316446002000104> — the original two-annotator κ; §2.3 reproduces the formula.
- Fleiss, J. L. (1971). *Measuring Nominal Scale Agreement Among Many Raters*. Psychological Bulletin 76(5). <https://doi.org/10.1037/h0031619> — multi-annotator κ; §2.4 reproduces the formula.
- Krippendorff, K. (2018). *Content Analysis: An Introduction to Its Methodology*, 4th ed., SAGE, Chapter 11. — the general α treatment, including missing-data and non-nominal variants; §2.5 follows this.
- Artstein, R., Poesio, M. (2008). *Inter-Coder Agreement for Computational Linguistics*. Computational Linguistics 34(4). <https://doi.org/10.1162/coli.07-034-R2> — the canonical NLP-facing survey; interpretation thresholds, CI practice.
- Landis, J. R., Koch, G. G. (1977). *The Measurement of Observer Agreement for Categorical Data*. Biometrics 33(1). <https://doi.org/10.2307/2529310> — the κ-interpretation table in §2.3 (with the caveats noted).
- Pustejovsky, J., Stubbs, A. (2012). *Natural Language Annotation for Machine Learning*, O'Reilly. — the MAMA cycle and guideline-versioning discipline this exercise reproduces.
- Gebru, T. et al. (2018, extended 2021). *Datasheets for Datasets*. Communications of the ACM. <https://arxiv.org/abs/1803.09010> — the release-document template the IAA numbers feed into.
- Bender, E., Friedman, B. (2018). *Data Statements for Natural Language Processing*. TACL. <https://aclanthology.org/Q18-1041/> — annotator-pool documentation template.
- Pavlick, E., Kwiatkowski, T. (2019). *Inherent Disagreements in Human Textual Inferences*. TACL. <https://aclanthology.org/Q19-1043/> — empirical demonstration that disagreement is often legitimate; motivates `LEGITIMATELY_AMBIGUOUS`.
- Sap, M. et al. (2019). *The Risk of Racial Bias in Hate Speech Detection*. ACL. <https://aclanthology.org/P19-1163/> — fairness implications of annotator-pool skew; motivates per-subgroup κ.
- Hovy, D. et al. (2013). *Learning Whom to Trust with MACE*. NAACL. <https://aclanthology.org/N13-1132/> — the weighting / trust estimation alternative to majority voting referenced in §2.8.
- Dawid, A. P., Skene, A. M. (1979). *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm*. Applied Statistics 28(1). <https://doi.org/10.2307/2346806> — the original Dawid-Skene aggregator cited alongside MACE in §2.8.
- `nltk.metrics.AnnotationTask` — <https://www.nltk.org/_modules/nltk/metrics/agreement.html> — implements Cohen's κ, Fleiss's κ, Krippendorff's α; used in §3.1–§3.3 as the reference cross-check.
- `statsmodels.stats.inter_rater` — <https://www.statsmodels.org/stable/stats.html#inter-rater-reliability> — cross-check for Fleiss κ.
- `simpledorff` — <https://github.com/LightTag/simpledorff> — public Python α implementation; cross-check for §2.5.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML02 Data Poisoning* applies when annotators are hired via a crowdsourcing platform and can collude to label-flip; the controlled-vocabulary reason tags plus per-annotator-against-gold reliability check in §3.11 are the defensive controls. *ML08 Model Skewing* applies when a biased annotator pool systematically mis-labels one class; the per-subgroup κ in §4 catches it. *ML09 Output Integrity Attack* is downstream of the guideline — if the guideline is adversarially authored, every downstream label is compromised; the changelog + semantic-version-pinning in §2.2 is the mitigation.
- MITRE ATLAS — <https://atlas.mitre.org/> — *AML.T0020 Poison Training Data* maps to collusive crowd workers; the IAA time-series in §3.10 and the per-annotator reliability in §3.11 are the detection controls.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: the guideline itself, the annotator pool demographics, and the calibration set are the primary artefacts. *Measure*: κ/α with CI per batch, per-label breakdown, per-annotator-subgroup breakdown. *Manage*: adjudication reason codes drive guideline revisions; semantic-version bumps are change-controlled; the audit log is retained alongside the dataset for incident response. *Govern*: annotator-pool demographic reporting (Datasheets §3.2, Data Statements §2) is the policy-level artefact that this harness produces.
