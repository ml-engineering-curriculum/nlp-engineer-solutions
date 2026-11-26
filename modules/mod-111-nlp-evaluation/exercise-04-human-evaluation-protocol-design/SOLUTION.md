# exercise-04 — Human Evaluation Protocol Design

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-111-nlp-evaluation/exercises/exercise-04-human-evaluation-protocol-design.md`.
> The target learning outcomes implied by that filename: design a **human evaluation protocol**
> for a generative NLP system — pick the **elicitation form** (Direct Assessment, SxS pairwise,
> MQM, Likert), **blind the comparison**, size the sample to the smallest meaningful effect,
> compute **rater agreement + calibration**, aggregate to a release-gate statistic, and publish
> the annotator-pool composition and a reproducibility pack. Where exercise-03 (mod-110) covered
> dataset labelling, this exercise covers **evaluating model outputs**, which has different
> failure modes (within-rater drift, position bias, exposure to multiple systems).

## 1. Solution overview

Human evaluation is the ground truth for generative NLP; automatic metrics are proxies for it. The
protocol choices are where releases fail:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Mis-chosen elicitation form** | A 1–5 Likert on "overall quality" tells you two systems both score 4.1 but not whether they fail in different ways; a free-text "write feedback" cell gives rich signal but is uncoded and ungrepable | Pick the form from the task shape: **Direct Assessment** (0–100 slider) for MT and summarisation at scale; **Side-by-side pairwise** for A/B comparisons where effect sizes are small; **MQM** (error-span annotation with taxonomy) when you need actionable error categorisation; **Likert** only for aggregate acceptability on small evals |
| **Un-blinded comparison** | Raters see system names (`system_a` vs. `baseline`); novelty bias and employer bias inflate the "new system" | Randomise display order per rater, strip system identifiers, present outputs as `A` and `B` only. Record the original mapping separately for reporting |
| **Under-powered sample** | 50 items per system per rater; the SxS win-rate CI is ±14 %; a 3-point move is indistinguishable from noise | Size the sample for the minimum effect size the stakeholder cares about. A 5 pp win-rate detection at 80 % power needs ≥ 400 pairs per comparison (§2.4) |
| **Single rater per item** | No way to measure disagreement; can't separate "the task is hard" from "the rater was wrong" | ≥ 3 raters per item; publish **Krippendorff's α** (handles missing data, ordinal scales); mod-110 exercise-03 is the IAA reference |
| **No calibration or training pass** | Raters interpret the scale differently; one rater uses 20–80 on the Direct Assessment slider, another uses 60–100; aggregates are not comparable | **Calibration phase**: 20–50 pinned items with reference labels; retain raters with calibration-κ ≥ threshold; weight by rater reliability if needed (Hovy et al. 2013 MACE, Dawid-Skene) |
| **No position / order bias correction** | In SxS, the left-hand output wins more often than chance; in Likert, the first system shown anchors the scale | Counterbalance A/B order across pairs; present items in random order per rater; measure and report position bias as a diagnostic |
| **Annotator-pool composition untracked** | Fairness implications of hiring crowd-workers from a narrow geography/demographic are hidden; replication requires knowing who you hired | Publish **Data Statements** (Bender & Friedman 2018) and **Datasheet for Datasets** (Gebru et al. 2021) annotator-pool fields; record geography, L1, task experience, pay rate, platform |

The deliverable is a four-component protocol:

1. **Protocol template** — a markdown document with a fixed structure (scope, items, elicitation form, calibration set, rater pool, blinding, order, aggregation, release-gate, reproducibility pack).
2. **Sizing calculator** — effect-size → required pairs/items via a paired-bootstrap power analysis (ties back to exercise-02).
3. **Rating UI contract** — the exact prompt text, scale anchors, error-type taxonomy if MQM; a reference HTML/JSON schema so downstream tooling implements it faithfully.
4. **Aggregation + reporting** — per-system Direct Assessment mean or win-rate with bootstrap CI; per-rater reliability vs. the calibration set; position-bias and order-effect diagnostics.

Reference literature and standards:

- **Direct Assessment** — Graham et al. 2013, 2017 (WMT). The 0–100 slider methodology; the current WMT human-evaluation standard.
- **Side-by-side pairwise / preference** — Zheng et al. 2023 (MT-Bench / Chatbot Arena); Bai et al. 2022 (Anthropic HH-RLHF).
- **MQM** — Lommel et al. 2014 (*Multidimensional Quality Metrics*); Freitag et al. 2021 (*Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation*) — the authoritative MQM-for-MT reference.
- **BLEU-vs-human correlation and the "neural metrics are better" argument** — Freitag et al. 2022 WMT metrics task; drives why automatic numbers can't substitute for human evaluation.
- **Likert and its limits** — Clark et al. 2021 (*All That's Human Is Not Gold: Evaluating Human Evaluation of Generated Text*); Karpinska et al. 2021 (*The Perils of Using Mechanical Turk to Evaluate Open-Ended Text Generation*).
- **Rater calibration and aggregation** — Hovy et al. 2013 (MACE); Dawid & Skene 1979 (EM aggregation); Passonneau 2004 on reliability for annotation.
- **Fairness and annotator demographics** — Sap et al. 2019 (hate-speech annotation by annotator race); Bender & Friedman 2018 (Data Statements); Gebru et al. 2021 (Datasheets).
- **BIG-bench and HELM human-eval sections** — Srivastava et al. 2023, Liang et al. 2023 — community reference for human-eval protocol fields in large-benchmark reporting.

## 2. Worked implementation: protocol template

### 2.1 Fixed protocol structure

```markdown
# Human evaluation protocol: {task} v{semver}

## 1. Scope and research question
What system version(s) are being compared, on what items, against what baseline, to answer what
stakeholder decision? Terminate in a release-gate statement: "accept if System B's win-rate over
A is ≥ 55 % with 95 % CI excluding 50 %".

## 2. Items
- Source: {held-out eval set URI, snapshot date, item count}
- Sampling: {random / stratified / adversarial}; stratification axes and per-stratum N.
- Inclusion / exclusion rules.

## 3. Elicitation form
One of: Direct Assessment (0-100), SxS pairwise, MQM (error-span taxonomy), Likert 1-5.
Justified against §2.2 of this solution.

## 4. Prompt text and anchors
The exact words shown to the rater. Scale anchors ("0 = incomprehensible, 100 = perfect").

## 5. Calibration set
20-50 pinned items with reference labels. Raters must achieve α >= 0.7 against reference
to be retained. Calibration set is re-run every K batches.

## 6. Rater pool
- Platform: {MTurk / Prolific / in-house / specialist vendor}
- Count: {total raters}, per-item: 3 (odd for ties).
- Qualifications: {L1 = en-US, task experience >= 100 HITs, approval rate >= 98 %, etc.}
- Pay: {pay rate, time estimate, hourly}. Must meet platform-regional minimum wage.
- Demographics declaration: {Data Statements §2 fields — geography, age, education, L1}

## 7. Blinding and order
- System identities blinded: outputs labelled A/B only.
- Order randomised per rater; A/B assignment counterbalanced.
- Position bias measured as diagnostic (§2.6 of this solution).

## 8. Aggregation
Per-system DA mean (or win-rate) with 95 % paired-bootstrap CI (see mod-111 exercise-02).
Per-rater reliability (κ or α vs. calibration set); raters with reliability < threshold
excluded or down-weighted per MACE / Dawid-Skene.

## 9. Release gate
Explicit decision rule: "Accept if ...; block if ...; request more data if ..."

## 10. Reproducibility pack
- Item list with hashes
- Prompt text (exact)
- Rater instructions (exact PDF/MD)
- Rater anonymised IDs (consistent across batches)
- Raw rating file (JSONL)
- Aggregation script (code)
- Random seed for ordering and bootstrap
```

Design notes:

- **Scope ends in a release-gate statement.** A protocol that measures "quality" without a decision
  rule is a research artefact, not a release tool. Every section feeds the gate.
- **Items declare the sampling strategy.** A random sample from a narrow distribution looks cheap
  but misses rare failures; stratified sampling by difficulty / length / domain is required for a
  release decision on a multi-domain system.
- **Prompt text is part of the artefact.** "Rate fluency" and "Rate how natural this sentence sounds
  to a native speaker" elicit different distributions. The exact wording ships with the model card.
- **Reproducibility pack is the audit artefact.** Without it a human-eval claim is unverifiable;
  this is the §10 that most protocols omit and is the easiest reviewer rejection.

### 2.2 Choosing the elicitation form

| Shape | Form | Justification |
|---|---|---|
| **MT / summarisation / paraphrase, large scale** | **Direct Assessment** 0–100 slider | Graham et al. 2013, 2017 WMT: fine-grained, correlates with ranking, scales to thousands of judgements. Avoid 1–5 Likert — the WMT human evaluation task phased it out in 2015 for Direct Assessment precisely because 1–5 compresses signal. |
| **A/B release decision** | **Side-by-side pairwise** | Pairwise preference has strictly higher power than independent scoring at the same sample size (Thurstone 1927 and subsequent IR literature). The win-rate is interpretable to a stakeholder ("B is preferred 56 % of the time"). The disadvantage is that absolute quality is not measured; pair with a DA sample to also report absolute. |
| **Diagnostic error analysis** | **MQM** (error-span annotation with taxonomy) | Freitag et al. 2021 define the MQM-core taxonomy for MT (accuracy — mistranslation / omission / addition; fluency — grammar / spelling / register; style / terminology). Produces actionable per-error-type counts; the "where is my model failing" artefact the DA mean cannot give. |
| **Dialogue / open-ended generation** | **Pairwise** + **anchor Likert** | Zheng et al. 2023 (MT-Bench / Chatbot Arena) show pairwise preferences aggregate into reliable Elo rankings for LLM chat comparison; anchor Likert items calibrate the scale against a known baseline. |
| **Small, qualitative evaluation (N ≤ 100)** | **Likert 1–5 or 1–7 with explicit anchors** | At small N the extra signal from DA / MQM is lost in noise; a well-anchored Likert is the most interpretable. Publish the anchors verbatim. |

Points worth defending:

- **Direct Assessment is the default.** The WMT community moved to DA in 2017 after Graham et al.
  2013 demonstrated that 1–5 Likert suffered from end-effects and low inter-rater agreement on
  modern (high-quality) systems. 0–100 with 5-point gridlines on the slider is the Graham 2017
  recipe.
- **SxS wins on sample efficiency.** For detecting a small effect (e.g., 5 pp win-rate above
  parity), SxS is roughly 4× more sample-efficient than independent DA (Zheng et al. 2023 §4).
  Use it when the release-gate effect size is small.
- **MQM is expensive but actionable.** Trained MQM raters are typically 3–5× slower than DA raters
  per item, and the training pass is 2–4 hours. Use it for a quarterly deep audit or for a
  release that is driven by a specific error class (hallucination, terminology).
- **Likert at scale is a reviewer flag.** If the protocol is Likert 1–5 with 1 000+ items,
  propose DA instead; cite Clark et al. 2021 and Karpinska et al. 2021 on crowd-workers' use of
  compressed Likert ranges.

### 2.3 Prompt text and scale anchors — write them exactly

Example DA prompt for MT (Graham et al. 2017 phrasing, English):

```markdown
**Instructions**

Read the English sentence below and the German translation produced by a system.
Rate how well the translation conveys the meaning of the English sentence, on a
scale from 0 to 100.

- **0** — The translation is incomprehensible, has no connection to the English meaning, or is empty.
- **25** — Most of the meaning is lost; the translation is mostly wrong or unclear.
- **50** — About half the meaning is conveyed; some content is missing or incorrect.
- **75** — Most of the meaning is conveyed; a minor piece is missing, incorrect, or awkward.
- **100** — The translation perfectly conveys the meaning in natural, fluent German.

Use the slider to choose any value from 0 to 100. The slider has 5-point gridlines but you may pick any integer.
```

Example SxS prompt for summarisation:

```markdown
**Instructions**

You will be shown an article and two summaries, labelled **Summary A** and **Summary B**.
Pick the summary you think is better. If you cannot decide, choose "Tie".

For your decision, consider:
1. Does it include the most important information from the article?
2. Is it factually correct (no claims that aren't in the article)?
3. Is it fluent and readable?

Which summary is better?
( ) Summary A is clearly better
( ) Summary A is slightly better
( ) Tie — about the same
( ) Summary B is slightly better
( ) Summary B is clearly better
```

Example MQM error-span prompt (Freitag et al. 2021 taxonomy):

```markdown
**Instructions**

Mark each error in the translation by highlighting the span and assigning a category and severity.

**Categories**
- Accuracy: mistranslation, addition, omission, untranslated, do-not-translate
- Fluency: grammar, spelling, punctuation, register, inconsistency
- Style: awkward, unidiomatic, repetition
- Terminology: inappropriate-for-context, inconsistent-use-of-terminology
- Other

**Severity**
- Minor: noticeable but does not meaningfully change the meaning
- Major: meaningfully changes the meaning or significantly impedes comprehension
- Critical: changes the meaning in a way that could harm (safety, legal, medical content)
```

Points worth defending:

- **Anchor every scale point.** "50 = half the meaning conveyed" is interpretable; "50 = okay" is
  not. Graham 2017 and Freitag 2021 both include anchor text; reproduce it or your version.
- **Include the 'Tie' option in SxS.** Forcing a choice when the rater sees no difference adds
  noise. Present the Likert-pairwise continuum (clearly A / slightly A / tie / slightly B / clearly
  B) — the 5-way scale, from the Freitag 2021 comparison, gives 5 ordinal bins for free.
- **MQM severity gradation matters for aggregation.** A `major` is worth 5 `minor`s and a `critical`
  is worth 10 `minor`s in the Freitag 2021 MQM-for-MT scoring; the aggregation function must match.
- **Keep the prompts in version control.** A diff of the prompt wording between releases is a
  controlled change; silently editing it invalidates cross-release comparability.

### 2.4 Sample sizing — a paired-bootstrap power analysis

```python
# sizing.py
import random
from statistics import NormalDist
from typing import Callable


def required_sample_size_for_winrate(
    effect: float,          # minimum detectable difference from 0.5 (e.g. 0.05 for 5 pp)
    power: float = 0.80,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> int:
    """
    For a two-proportion / binomial test on win-rate, the required N is approximated as
      N = (z_alpha + z_beta)^2 * p(1-p) / effect^2
    Shortcut for the paired-bootstrap case with p = 0.5 (the null); tune to the
    expected variance of the empirical win-rate if you have pilot data.
    """
    z_a = NormalDist().inv_cdf(1 - alpha / (2 if two_sided else 1))
    z_b = NormalDist().inv_cdf(power)
    p = 0.5
    return int(((z_a + z_b) ** 2) * p * (1 - p) / (effect ** 2)) + 1


def simulated_power_paired_da(
    n: int,
    delta: float,          # expected mean DA difference, points on 0-100
    sigma_within_item: float = 15.0,  # typical sentence-level DA stddev for high-quality MT
    rho_between_systems: float = 0.5, # paired correlation of A vs B scores per item
    n_sim: int = 1000,
    seed: int = 20260101,
) -> float:
    """
    Monte Carlo power estimate for DA: proportion of simulations in which a paired
    bootstrap 95 % CI excludes zero.
    """
    rng = random.Random(seed)
    nd = NormalDist(0, 1)
    hits = 0
    for _ in range(n_sim):
        # Generate paired DA scores: A ~ N(mu_a, sigma), B ~ N(mu_a + delta, sigma),
        # correlated by rho_between_systems within item.
        a = [sigma_within_item * nd.inv_cdf(rng.random()) for _ in range(n)]
        b = [rho_between_systems * ai + ((1 - rho_between_systems ** 2) ** 0.5) * sigma_within_item * nd.inv_cdf(rng.random()) + delta for ai in a]
        # Paired bootstrap 95 % CI
        n_boot = 500
        deltas = []
        for _ in range(n_boot):
            idx = [rng.randrange(n) for _ in range(n)]
            deltas.append(sum(b[i] - a[i] for i in idx) / n)
        deltas.sort()
        lo = deltas[int(0.025 * n_boot)]
        hi = deltas[int(0.975 * n_boot) - 1]
        if lo > 0 or hi < 0:
            hits += 1
    return hits / n_sim
```

Points worth defending:

- **Sample size is driven by the stakeholder effect size, not by cost.** The stakeholder names the
  smallest difference that matters ("1 DA point", "5 pp win-rate"); the sample size is whatever
  achieves 80 % power to detect it. Picking N first and reporting underpowered results is the
  most common error in NLP human eval papers.
- **Rough rules of thumb**: 400 pairs for a 5 pp win-rate detection; 1 500 items per system for a
  1-point DA detection at 80 % power with sentence-level stddev ~15. These numbers come from the
  Graham 2017 WMT recommendations and from the Monte Carlo in `simulated_power_paired_da` above.
- **Report pilot variance.** The simulation defaults are rough; a 50-item pilot gives the real
  within-item stddev. Re-run the power calc with the measured stddev; often the required N halves.
- **Minimum 3 raters per item.** Separate from the item-count N; this is the rater-count per item.
  3 is the floor for α computation and for majority-vote aggregation; 5+ for high-stakes.

### 2.5 Blinding, order, and counterbalancing

```python
# ordering.py
import hashlib, random
from typing import Sequence


def randomised_display_order(
    items: Sequence[dict],      # each item has {"item_id", "system_a_out", "system_b_out"}
    rater_id: str,
    seed_prefix: str = "mod-111-ex-04",
) -> list[dict]:
    """
    Per-rater deterministic shuffle + A/B counterbalancing. Returns a list of
    display dicts with {"display_label_A_system", "display_label_B_system"} so the
    system identity is only in the rendered-record log, never in the rater-facing UI.
    """
    rng = random.Random(f"{seed_prefix}:{rater_id}")
    shuffled = list(items)
    rng.shuffle(shuffled)
    out: list[dict] = []
    for item in shuffled:
        flip = rng.random() < 0.5
        A_out = item["system_a_out"] if not flip else item["system_b_out"]
        B_out = item["system_b_out"] if not flip else item["system_a_out"]
        A_sys = "system_a" if not flip else "system_b"
        B_sys = "system_b" if not flip else "system_a"
        out.append({
            "item_id": item["item_id"],
            "A_display": A_out,
            "B_display": B_out,
            "A_label": "system_a" if A_sys == "system_a" else "system_b",   # for logging only
            "B_label": "system_b" if B_sys == "system_b" else "system_a",
        })
    return out


def position_bias_rate(ratings: list[dict]) -> float:
    """
    For a 2-way pairwise rating log, fraction of judgements that preferred the left-side
    option (independent of system identity). Expected 0.50 under no position bias.
    """
    left_wins = sum(1 for r in ratings if r["choice"] == "A_better" or r["choice"] == "A_slightly_better")
    total = sum(1 for r in ratings if r["choice"] != "tie")
    return left_wins / max(total, 1)
```

Points worth defending:

- **Position bias is measurable and real.** Zheng et al. 2023 §4.4 found a 5–10 pp left-bias in
  LLM-pairwise evaluation; human-rater left-bias is typically 2–5 pp. Counterbalancing (flip A/B
  per item per rater) eliminates it; reporting the raw position-bias rate as a diagnostic verifies
  the counterbalancing ran.
- **Deterministic per-rater shuffle via a seeded hash.** Lets you rebuild the exact UI order from
  the log; required for incident response ("rater X complained about item Y — what did they see?").
- **System identity is never in the rater-facing UI.** The common bug is a header `Baseline` /
  `New System`; it defeats blinding. Labels A/B only; mapping is server-side.

### 2.6 Rater reliability and aggregation

```python
# aggregation.py
from collections import Counter
from statistics import median


def majority_vote(labels: list[str]) -> str:
    """Majority vote with lexicographic tie-break; returns 'AMBIGUOUS' on an even tie."""
    c = Counter(labels)
    top = c.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    return "AMBIGUOUS"


def median_da(scores: list[float]) -> float:
    """Median DA across raters per item; robust to a single outlier rater."""
    return median(scores)


def rater_reliability_against_calibration(
    rater_calib_scores: dict[str, dict[str, float]],   # rater_id -> item_id -> score
    reference: dict[str, float],                         # item_id -> reference score
) -> dict[str, float]:
    """
    For each rater, Pearson correlation of their calibration scores vs. the reference.
    Raters with r < 0.5 are flagged for retraining or exclusion; r >= 0.7 is typical for
    well-calibrated DA raters (Graham et al. 2017).
    """
    out: dict[str, float] = {}
    for rid, scores in rater_calib_scores.items():
        shared = sorted(set(scores) & set(reference))
        if len(shared) < 5:
            out[rid] = float("nan"); continue
        xs = [reference[i] for i in shared]
        ys = [scores[i] for i in shared]
        mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
        dy = (sum((y - my) ** 2 for y in ys)) ** 0.5
        out[rid] = num / (dx * dy) if dx and dy else float("nan")
    return out
```

Points worth defending:

- **Median, not mean, for DA aggregation across raters.** One rater who uses 0 or 100 consistently
  (a known crowd-work pattern) pulls the mean; the median is robust. Graham 2017 reports both and
  uses the median for the WMT human eval ranking.
- **Reliability is measured against a pinned calibration set.** Rater-vs-rater reliability (IAA)
  measures internal consistency; rater-vs-reference measures calibration accuracy. Both matter;
  the latter catches raters who are internally consistent but wrong.
- **Excluded raters are documented.** "3 of 24 raters excluded for calibration κ < 0.5" is a
  release-gate artefact; it is not evidence of bias to remove unreliable raters, but it must be
  declared.
- **MACE / Dawid-Skene re-weighting** (mod-110 exercise-03 §2.8) is the alternative for noisy
  crowds; use it when you cannot filter raters, e.g., because the platform doesn't allow
  qualification controls.

### 2.7 Release-gate aggregation + reporting

```python
# release_gate.py
from dataclasses import dataclass


@dataclass
class HumanEvalReleaseGate:
    """
    Example: ship System B if win-rate over A is >= 55 % AND lower 95 % CI bound > 50 %.
    """
    primary_statistic: str          # "winrate" | "da_mean"
    target_winrate: float = 0.55
    target_ci_lower: float = 0.50
    required_raters_per_item: int = 3
    required_position_bias_bound: float = 0.55   # fail if left-win rate > 55 % or < 45 %

    def accepts(self, result: dict) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if self.primary_statistic == "winrate":
            if result["winrate"] < self.target_winrate:
                reasons.append(f"winrate {result['winrate']:.3f} < target {self.target_winrate}")
            if result["winrate_ci_95"][0] <= self.target_ci_lower:
                reasons.append(f"winrate CI lower {result['winrate_ci_95'][0]:.3f} <= {self.target_ci_lower}")
        if result["raters_per_item_min"] < self.required_raters_per_item:
            reasons.append(f"some items have < {self.required_raters_per_item} raters")
        pb = result.get("position_bias_rate", 0.5)
        if pb > self.required_position_bias_bound or pb < 1 - self.required_position_bias_bound:
            reasons.append(f"position bias rate {pb:.3f} out of band")
        return len(reasons) == 0, reasons
```

Points worth defending:

- **Gate on CI lower bound, not the point estimate.** A point estimate above the target with a CI
  that crosses parity is not decision-ready; reviewers must see the CI, not just the mean.
- **Minimum raters-per-item is enforced.** A gate that accepts "we tried to get 3 raters but only
  got 1 on 30 items" is not reproducible.
- **Position-bias bound is a diagnostic gate.** If counterbalancing did not run, the gate catches
  it automatically rather than relying on reviewer memory.

## 3. Validation steps

1. **Protocol template completeness.** Every required section (§2.1) must be filled before the
   release gate is run. A missing `calibration_set` field is a block.
2. **Prompt-text equivalence.** The exact prompt text is version-controlled; a diff between
   releases is flagged. Golden-list test over the markdown snapshot.
3. **Blinding round-trip.** Run the ordering harness with a synthetic rater; verify no system
   identifier appears in the rater-facing display dict; verify the mapping is recoverable from
   the log. Catches a bug where the system name leaks into a URL or metadata field.
4. **Counterbalance uniformity.** For 1 000 items × 3 raters, the fraction of `system_a_out` on
   the left side is 0.50 ± 0.03. If it is 0.40 or 0.60, the counterbalance RNG is biased.
5. **Sample-size validity.** Call `required_sample_size_for_winrate(effect=0.05)` and compare to
   the published WMT / MT-Bench 400 ± 100 range; off-by-10× is a formula bug.
6. **Power simulation.** `simulated_power_paired_da(n=1500, delta=1.0, sigma=15.0)` must return
   power ≥ 0.80. If it does not, the sentence-level stddev is higher than the default and N must
   be raised.
7. **Calibration-set drift detection.** Re-score the calibration set every K batches; a drop of
   > 0.05 in mean rater-vs-reference correlation across batches is a stop-work signal.
8. **Rater-pool fairness slice.** For each demographic bucket in the pool (declared in §2.1 §6),
   compute per-bucket median DA; report deltas. Sap et al. 2019 showed annotator demographics
   drive subjective-task labels; the pool report is the control.
9. **MQM taxonomy coverage.** For a 100-item MQM batch, publish the frequency of each error
   category; categories with 0 hits in a large batch suggest the taxonomy is misaligned to the
   data or that raters are not using a category.
10. **Reproducibility pack completeness.** On every release, the pack must include: items + hashes,
    prompt text, rater pseudonymised IDs, raw rating JSONL, aggregation script, seeds. Missing
    any field is a block.
11. **Position bias report.** `position_bias_rate(ratings)` within 0.45–0.55; if outside,
    counterbalancing failed or the UI has a confound.
12. **Release-gate round-trip.** Given a rating log and a declared gate, `accepts()` must return
    the same decision across re-runs; feeds into the audit trail.

## 4. Grading rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Scope and research question** | Explicit; terminates in a release-gate statement | Scope only | None |
| **Elicitation form** | Justified per §2.2; prompt text + anchors recorded verbatim | Form chosen, text paraphrased | 1–5 Likert at scale with no anchors |
| **Sample size** | Power analysis published; target effect declared; N sized to ≥ 80 % power | N chosen, no power calc | N chosen by cost alone |
| **Raters per item** | ≥ 3; ≥ 5 for high-stakes decisions | 2 | 1 |
| **Blinding** | System identity stripped from the rater-facing UI; A/B labels only; mapping server-side | Blinded but order fixed | Not blinded |
| **Counterbalancing** | A/B order flipped per item per rater; position-bias diagnostic reported | Random order, no flip | Fixed order |
| **Calibration set** | 20–50 pinned items; rater-vs-reference correlation per rater reported; raters below threshold excluded or weighted (MACE / Dawid-Skene) | Calibration once at start | None |
| **Aggregation** | Median DA (robust) or Elo (SxS) with 95 % paired-bootstrap CI; per-rater reliability | Mean with stddev | Point estimate only |
| **Rater-pool declaration** | Data Statements §2 fields filled (geography, L1, task experience, pay, platform) | Partial declaration | None |
| **Release gate** | Decision rule (e.g., CI lower bound > target); `accepts()` runs in CI | Target stated, not enforced | None |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS mapped (collusive raters, gaming, LM-as-judge spoofing) | Mentioned | Ignored |
| **Documentation / card** | NIST AI RMF *Map*, *Measure*, *Manage*; Data Statements; Model Card human-eval section | Partial | Neither |
| **Reproducibility pack** | Items + hashes, prompt text, pseudo-IDs, raw JSONL, aggregation script, seeds | Some fields | None |

## 5. Common mistakes

- **Running a 1–5 Likert at scale.** Clark et al. 2021 and Karpinska et al. 2021 both show
  Mechanical-Turk Likert judgements compress to the middle and produce low IAA on modern,
  high-quality generations; the WMT community moved to Direct Assessment for this reason. Use DA
  or MQM at scale; Likert for small qualitative evals only.
- **No calibration set.** Rater drift and rater disagreement are indistinguishable without a pinned
  reference. The calibration set is cheap (20–50 items) and catches both.
- **Asking raters to rate on "overall quality".** The bundled construct hides which sub-dimension
  (fluency / adequacy / faithfulness) a rater is weighting. Either ask for each sub-dimension
  separately (DA per axis) or use MQM to decompose into error-type counts.
- **Un-blinded SxS.** A `system_a` label vs. `system_b` is better than `baseline` vs. `new`; the
  latter leaks author intent. A shuffled A/B label is better still.
- **No counterbalancing.** Position bias in pairwise evals is 2–10 pp (humans) and higher for
  LM-as-judge (Zheng et al. 2023). Flip A/B per item per rater; measure the residual rate.
- **1 rater per item.** You get a number but no way to measure uncertainty or to separate rater-
  error from task-difficulty. 3 is the floor.
- **Post-hoc rater exclusion that moves the headline number.** Excluding "poorly-calibrated" raters
  based on their agreement with the result you want is p-hacking. Define the exclusion rule *in
  the protocol* (calibration-set correlation < X) *before* the main annotation.
- **Not publishing the annotator-pool composition.** Fairness and replicability both depend on it.
  Data Statements (Bender & Friedman 2018) and Datasheets §3.2 are the format; fill them.
- **Under-paying raters.** Beyond ethics, underpaid raters produce worse data (Berinsky et al.
  2012, Hara et al. 2018). Platform-regional minimum wage is the floor; task-time piloting is how
  you calibrate the per-HIT rate.
- **Reporting one number instead of a CI.** Graham et al. 2017 and Freitag 2021 both report
  CIs; a release gate on a point estimate is not decision-ready.
- **LM-as-judge without human calibration.** Zheng et al. 2023 show LM-as-judge correlates with
  human preferences only after calibration; using it as the sole metric for a release decision is
  unvalidated. LM-as-judge is a valid *scaling* tool after calibration, not a replacement.
- **Asking the same rater to score every item every release.** Within-rater drift, familiarity
  effects, and boredom all degrade judgement. Rotate the rater pool; keep a small anchor set
  consistent across batches for drift measurement.
- **No position-bias diagnostic.** Even with counterbalancing, measuring the residual bias is the
  check that counterbalancing ran. The §3.11 test takes two lines of code.
- **Reporting only aggregate numbers, not per-slice.** Per-domain, per-length, per-demographic DA
  is the actionable artefact. A corpus mean hides that System B regresses on a critical slice.
- **Treating the human eval as the ground truth without reporting its own reliability.** Human-eval
  IAA (α) on subjective tasks is often 0.4–0.6; the release gate must acknowledge that the measurement
  itself has a 10-point noise floor.

## 6. References

Primary sources used in this solution:

- Graham, Y., Baldwin, T., Moffat, A., Zobel, J. (2013). *Continuous Measurement Scales in Human Evaluation of Machine Translation*. ACL Workshop. <https://aclanthology.org/W13-2305/> — the original Direct Assessment slider for MT.
- Graham, Y., Baldwin, T., Moffat, A., Zobel, J. (2017). *Can machine translation systems be evaluated by the crowd alone?*. Natural Language Engineering 23(1). <https://doi.org/10.1017/S1351324915000339> — the WMT-adopted DA methodology; the §2.2 and §2.3 recipes.
- Freitag, M. et al. (2021). *Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation*. TACL. <https://aclanthology.org/2021.tacl-1.87/> — the authoritative MQM-for-MT reference; the §2.3 severity-and-category scheme.
- Lommel, A., Burchardt, A., Uszkoreit, H. (2014). *Multidimensional Quality Metrics (MQM): A Framework for Declaring and Describing Translation Quality Metrics*. Tradumàtica. — the MQM framework the §2.3 MQM anchors reproduce.
- Freitag, M. et al. (2022). *Results of the WMT22 Metrics Shared Task: Stop Using BLEU – Neural Metrics Are Better and More Robust*. WMT. <https://aclanthology.org/2022.wmt-1.2/> — the empirical case that automatic metrics now trail human eval as the primary; motivates exercise-04 as a first-class release artefact.
- Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS. <https://arxiv.org/abs/2306.05685> — pairwise preference methodology; Elo aggregation for chat; position-bias characterisation.
- Bai, Y. et al. (2022). *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*. arXiv. <https://arxiv.org/abs/2204.05862> — human-preference data collection at scale; the pay/platform conventions the §2.1 §6 fields are modelled on.
- Clark, E. et al. (2021). *All That's Human Is Not Gold: Evaluating Human Evaluation of Generated Text*. ACL. <https://aclanthology.org/2021.acl-long.565/> — empirical case that untrained raters cannot distinguish human from machine-generated text in modern settings; motivates the calibration + training pass.
- Karpinska, M., Akoury, N., Iyyer, M. (2021). *The Perils of Using Mechanical Turk to Evaluate Open-Ended Text Generation*. EMNLP. <https://aclanthology.org/2021.emnlp-main.97/> — low MTurk IAA on free-generation Likert; argues for trained experts on open-ended tasks.
- Hovy, D. et al. (2013). *Learning Whom to Trust with MACE*. NAACL. <https://aclanthology.org/N13-1132/> — weighted aggregation for noisy crowds; §2.6 alternative to filtering.
- Dawid, A. P., Skene, A. M. (1979). *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm*. Applied Statistics 28(1). <https://doi.org/10.2307/2346806> — the EM aggregator.
- Passonneau, R. J. (2004). *Computing Reliability for Coreference Annotation*. LREC. <https://aclanthology.org/L04-1240/> — α for subjective annotation.
- Thurstone, L. L. (1927). *A Law of Comparative Judgment*. Psychological Review 34. — foundational paper for pairwise preferences and Elo-like ranking.
- Krippendorff, K. (2018). *Content Analysis: An Introduction to Its Methodology*, 4th ed., SAGE, Chapter 11 — α for the §2.6 IAA; detailed in mod-110 exercise-03.
- Bender, E., Friedman, B. (2018). *Data Statements for Natural Language Processing*. TACL. <https://aclanthology.org/Q18-1041/> — the rater-pool declaration template.
- Gebru, T. et al. (2021). *Datasheets for Datasets*. Communications of the ACM. <https://arxiv.org/abs/1803.09010> — the §3.2 "collection-process" section informs the §2.1 §6 fields.
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAccT. <https://arxiv.org/abs/1810.03993> — the model-card "evaluation" section the human-eval protocol fills.
- Sap, M. et al. (2019). *The Risk of Racial Bias in Hate Speech Detection*. ACL. <https://aclanthology.org/P19-1163/> — annotator-pool demographic effects on subjective labels; motivates §3.8.
- Pavlick, E., Kwiatkowski, T. (2019). *Inherent Disagreements in Human Textual Inferences*. TACL. <https://aclanthology.org/Q19-1043/> — legitimate disagreement is signal; motivates `AMBIGUOUS` aggregation.
- Berinsky, A. J., Huber, G. A., Lenz, G. S. (2012). *Evaluating Online Labor Markets for Experimental Research: Amazon.com's Mechanical Turk*. Political Analysis 20(3). — crowd-worker labour-market norms; motivates the §4 pay-rate review item.
- Hara, K. et al. (2018). *A Data-Driven Analysis of Workers' Earnings on Amazon Mechanical Turk*. CHI. <https://doi.org/10.1145/3173574.3174023> — empirical earnings analysis motivating the minimum-wage floor.
- Liang, P. et al. (2023). *Holistic Evaluation of Language Models* (HELM). TMLR. <https://arxiv.org/abs/2211.09110> — the human-eval protocol fields HELM formalises.
- Srivastava, A. et al. (2023). *Beyond the Imitation Game* (BIG-bench). TMLR. <https://arxiv.org/abs/2206.04615> — the human-eval protocol patterns for a many-task benchmark.
- Koehn, P. (2004). *Statistical Significance Tests for Machine Translation Evaluation*. EMNLP. <https://aclanthology.org/W04-3250/> — the paired bootstrap for CI reporting on human-eval win-rates; ties into exercise-02.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML02 Data Poisoning* applies to raters hired via a crowd platform who can collude to systematically prefer one system; the rater-vs-calibration-set reliability in §2.6 and per-rater reliability tracking are the controls. *ML05 Model Theft*: public rating logs leak information about the model distribution; strip system identifiers from public releases. *ML09 Output Integrity Attack*: an inflated human-eval win-rate misleads a deployment decision; the §4 release gate is the mitigation.
- MITRE ATLAS — <https://atlas.mitre.org/> — *AML.T0020 Poison Training Data* has a human-eval analogue: collusive raters can corrupt the preference data used to train preference models (RLHF); controlled with per-rater reliability and anomaly detection on rating distributions. *AML.T0043 Craft Adversarial Data*: adversarial items designed to look preferable to a specific system are the attack; mitigated by random sampling plus declared adversarial slices.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: the task shape, elicitation form, rater pool composition, prompt text, and calibration set are the primary Map artefacts; every one is a required field in §2.1. *Measure*: the (per-system DA median or win-rate, 95 % CI, per-slice numbers, rater-reliability distribution, position-bias rate) tuple is the Measure artefact; `HumanEvalReleaseGate.accepts()` is the Measure-driven gate. *Manage*: protocol version bumps are change-controlled; rater pool changes between releases are declared; the reproducibility pack is retained alongside the model card for incident response. *Govern*: the Data Statements / Datasheets rater-pool declaration is the policy-level artefact; pay-rate floors and the calibration-set retention policy are governance controls.
