# exercise-01 — Task-Aware Metric Selection Rubric

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-111-nlp-evaluation/exercises/exercise-01-task-aware-metric-selection-rubric.md`.
> The target learning outcomes implied by that filename: author a **decision rubric** that, given
> an NLP task's shape (reference-supervised vs. reference-free, deterministic vs. open-ended, fluency-
> vs. faithfulness-sensitive), maps to a defensible **primary metric + secondary metric + diagnostic
> slice set**; justify every pick against the published strengths and failure modes of the metric;
> and produce a grading rubric a reviewer can use to accept or reject a model-card metric section.

## 1. Solution overview

The dominant failure mode in NLP evaluation is not "the model is bad" — it is "the metric answers a
different question than the stakeholder asked". A rubric that blocks that failure has four moving parts:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Wrong metric family for the task shape** | BLEU reported for a summarisation task with abstractive references; ROUGE reported for machine translation; exact-match reported for free-form QA | Classify the task first (reference-supervised / reference-free; extractive / abstractive; constrained / open-ended), then pick the metric family whose literature supports that shape |
| **Single-number tyranny** | A corpus-level BLEU moves 0.3 points release-over-release and the on-call engineer cannot say whether the model improved on short sentences, regressed on long ones, or did both | Pair the primary metric with a **secondary** that captures a different failure mode (e.g., BLEU + chrF or BLEU + COMET) and a **slice** report (by length, by domain, by language pair) |
| **Metric gameability** | ROUGE favours copy-paste; models that extract instead of summarise look better than models that paraphrase correctly. BLEU favours short outputs; brevity-penalty tuning hides that | Pair surface-overlap with model-based or semantic metrics (BERTScore, COMET, BLEURT) and inspect the gap; publish the ablation when the two disagree |
| **No connection to stakeholder utility** | The release blocker is "hallucination rate < 2 %"; the reported metric is corpus BLEU. These do not measure the same thing | Make the rubric terminate in a *stakeholder-utility* statement: "primary metric X with target ≥ T on slice S is sufficient evidence to ship against decision D". Metrics without a stakeholder owner do not go on the model card |

The deliverable is a four-part rubric:

1. **Task classifier** — a short questionnaire that maps the task to a shape (10 canonical shapes: classification, sequence-labelling, extractive QA, generative QA, summarisation, MT, paraphrase, dialogue, code generation, open-ended generation).
2. **Per-shape metric sheet** — for each shape, the (primary, secondary, diagnostic) metric set with the published strengths, failure modes, and a one-paragraph justification.
3. **Slicing plan** — mandatory slice axes for each shape (length bucket, domain, language, demographic subgroup) and the minimum sample size per slice.
4. **Grading rubric** — a 10-dimension checklist a reviewer uses to accept or reject a model-card metric section; failing any "pass" column is grounds to block release until addressed.

Authoritative sources this solution draws on:

- **BLEU** (Papineni et al. 2002) — the original n-gram precision + brevity penalty; established the family but documented limits.
- **chrF / chrF++** (Popović 2015, 2017) — character-F; closes several BLEU failure modes on morphology-rich target languages.
- **METEOR** (Banerjee & Lavie 2005) — alignment + stem/synonym matching.
- **ROUGE** (Lin 2004) — the recall-oriented summarisation family (ROUGE-1/2/L/LSum).
- **BERTScore** (Zhang et al. 2020) — contextual-embedding similarity; the first widely adopted model-based metric.
- **MoverScore** (Zhao et al. 2019), **BLEURT** (Sellam et al. 2020), **COMET / COMET-22** (Rei et al. 2020, 2022) — learned-metric family.
- **GLEU for grammatical error correction** (Napoles et al. 2015); **ERRANT** (Bryant et al. 2017).
- **pass@k** for code generation (Chen et al. 2021, Codex).
- **SQuAD exact-match + F1** (Rajpurkar et al. 2016, 2018) — extractive QA conventions.
- **WMT metrics shared tasks** (Freitag et al., annually) — the yearly empirical ranking of MT metrics against human judgement; the authoritative "which metric correlates best in year N" source.
- **Statistical significance for NLP** (Dror et al. 2018, *The Hitchhiker's Guide to Testing Statistical Significance in NLP*) — pairs with exercise-02 on when a reported delta is signal.
- **BIG-bench** (Srivastava et al. 2023), **HELM** (Liang et al. 2023), **lm-eval-harness** (EleutherAI) — multi-metric evaluation frameworks that the rubric's slice-plan is designed to feed.

## 2. Worked rubric

### 2.1 Task classifier — 6 questions, 10 canonical shapes

```markdown
# Task classification questionnaire v1.0

Answer in order. The first "yes" that matches picks the shape.

Q1. Is there a **finite, fixed label set** and the output is **one label per input**?
    → Shape: **CLASSIFICATION**.
Q2. Is there a **finite, fixed label set** and the output is **one label per token**?
    → Shape: **SEQUENCE_LABELLING** (NER, POS, chunking).
Q3. Is the output a **span** from the input (verbatim substring)?
    → Shape: **EXTRACTIVE_QA**.
Q4. Is the output a **short free-form answer** to a question (possibly not in the input)?
    → Shape: **GENERATIVE_QA**.
Q5. Is the output a **shorter rewrite** of a longer input, meant to preserve content?
    → Shape: **SUMMARISATION**.
Q6. Is the output a **translation** into a different language?
    → Shape: **MACHINE_TRANSLATION**.
Q7. Is the output a **meaning-preserving rewrite** in the same language?
    → Shape: **PARAPHRASE**.
Q8. Is the output a **turn in a multi-turn dialogue** with no single correct reference?
    → Shape: **DIALOGUE**.
Q9. Is the output **source code** meant to satisfy a specification?
    → Shape: **CODE_GENERATION**.
Q10. None of the above (open-ended creative or exploratory generation)?
    → Shape: **OPEN_ENDED_GENERATION**.
```

Design notes:

- **Order matters.** The questionnaire is deliberately cascading — a span-extraction task answers "yes" to Q3 before anyone evaluates whether it is also "a question answer". A reviewer who answers Q4 first picks the wrong metric family.
- **Shape is independent of model architecture.** The same encoder can serve classification and sequence labelling; the metric choice is driven by the *output* shape, not the model.
- **"None of the above" is not a residual category** — it is the hardest case. `OPEN_ENDED_GENERATION` has no reference, no verifiable constraint, and no agreed metric; the rubric forces that honesty forward instead of defaulting to BLEU.

### 2.2 Per-shape metric sheet

| Shape | Primary | Secondary | Diagnostic | Justification |
|---|---|---|---|---|
| **CLASSIFICATION** | Macro-F1 | Accuracy + per-class F1 | Confusion matrix, per-slice F1 | Macro-F1 is unaffected by class imbalance; per-class F1 exposes which class the model is failing on (the actual actionable signal). Accuracy alone is misleading on skewed label sets (Sokolova & Lapalme 2009). |
| **SEQUENCE_LABELLING** | Entity-level F1 (span-exact match) | Token-level F1 | Per-entity-type F1, boundary vs. type error split | SQuAD-style token-level F1 over-rewards near-misses on span boundaries; the CoNLL 2003 shared-task convention is span-exact entity-level F1 (Tjong Kim Sang & De Meulder 2003). |
| **EXTRACTIVE_QA** | F1 (token-overlap) | Exact-Match | HasAns-F1, NoAns-F1 (SQuAD 2.0) | SQuAD 2.0 (Rajpurkar et al. 2018) splits the metric into has-answer and no-answer slices; collapsing them hides the unanswerable-detection failure mode. |
| **GENERATIVE_QA** | Token-F1 against gold | BERTScore-F1 | Exact-Match on normalized answer | Natural Questions short-answer evaluation (Kwiatkowski et al. 2019) uses token-F1; BERTScore is the secondary when paraphrase-tolerance matters. Exact-Match as a diagnostic flags when a correct paraphrase was penalized. |
| **SUMMARISATION** | ROUGE-L + ROUGE-2 | BERTScore-F1 | Faithfulness (QAGS, FactCC) + extractivity (density, coverage) | ROUGE's weakness on abstractive summaries is well-documented (Falke et al. 2019); pairing with a faithfulness metric catches hallucination, and extractivity (Grusky et al. 2018) catches extractive gaming. |
| **MACHINE_TRANSLATION** | COMET-22 (or the current WMT metric winner) | BLEU + chrF | Per-language-pair, per-domain, per-length-bucket | WMT metrics shared tasks (Freitag et al.) show neural metrics (COMET, BLEURT) correlate best with human judgement; BLEU and chrF remain as secondary for comparability across papers. Freitag et al. 2022 recommend reporting a learned metric *and* a surface metric. |
| **PARAPHRASE** | BERTScore-F1 + iBLEU | Self-BLEU (for diversity) | Semantic equivalence + lexical divergence scatter | Paraphrase quality is simultaneously "same meaning" *and* "different surface"; a single metric cannot capture both. iBLEU (Sun & Zhou 2012) explicitly penalises surface identity to the input. |
| **DIALOGUE** | Human evaluation (Likert or SxS); automatic as diagnostic only | USR / DialoGLUE automatic metrics | Toxicity rate, repetition rate, response length distribution | Liu et al. 2016 ("How NOT to Evaluate Your Dialogue System") showed automatic metrics correlate near-zero with human judgement on open-domain dialogue; exercise-04 covers the human protocol. |
| **CODE_GENERATION** | pass@k (functional correctness) | Compile / lint pass-rate | Per-problem-difficulty pass@k, per-language | BLEU-like surface metrics do not correlate with functional correctness (Chen et al. 2021); pass@k with at least k ∈ {1, 10, 100} is the HumanEval / MBPP / APPS convention. |
| **OPEN_ENDED_GENERATION** | Human evaluation (see exercise-04) | None reliable; perplexity on a held-out reference distribution as a weak diagnostic | LM-as-judge pairwise preferences (with human calibration) | No automatic metric is validated for open-ended creativity; honestly saying so in the model card is required. LM-as-judge (Zheng et al. 2023 on MT-Bench) is documented but requires calibration against human raters. |

Points worth defending in review:

- **No surface-overlap metric is the *only* primary for any generative task.** BLEU alone for MT was standard practice until the 2019 WMT shared task (Mathur et al. 2020 showed BLEU's correlation with human judgement has fallen as systems improved). Current WMT recommendations pair a learned metric with a surface metric; reproduce that pattern.
- **ROUGE-L and ROUGE-2 together.** ROUGE-1 is too easy to game by matching content words out of order; ROUGE-2 captures bigram order; ROUGE-L captures longest-common-subsequence. Reporting all three is the standard; collapsing to one hides the trade-off.
- **COMET / BLEURT / BERTScore have model-version dependencies.** Report the exact model checkpoint (`wmt22-comet-da`, `bert-base-uncased`, etc.) in the model card. Zhang et al. 2020 explicitly warn that BERTScore values depend on the underlying BERT checkpoint and are not comparable across versions.
- **`pass@k` requires executable test suites.** Without them, you fall back to surface metrics that do not correlate with correctness. If tests are not available for your domain, say so and move to human evaluation — do not substitute BLEU.
- **Dialogue and open-ended generation terminate at human evaluation.** This is not a cop-out; it is the empirically-justified recommendation. Exercise-04 operationalises it.

### 2.3 Mandatory slicing plan

Every model-card metric section must publish the primary + secondary on these slices, at minimum:

| Slice axis | Why it is mandatory | Minimum sample size per slice |
|---|---|---|
| **Length bucket** (short / medium / long, task-dependent quartiles) | BLEU brevity-penalty; attention-span failure on long inputs; summarisation length-matching artefacts | ≥ 100 items per bucket |
| **Domain** (news, biomedical, legal, social, code, etc., when training data is multi-domain) | Out-of-domain regression is the dominant release-blocker on multi-domain models | ≥ 100 items per domain |
| **Language / language pair** (for MT and multilingual models) | Low-resource pair quality is typically 10–30 % below high-resource; a corpus mean hides it | ≥ 200 items per direction |
| **Demographic subgroup** (when relevant to task; names, dialects, countries) | Fairness regressions are usually invisible in corpus means (Dixon et al. 2018; Blodgett et al. 2020) | ≥ 100 items per subgroup |
| **Difficulty / confidence bin** | Calibration: a 90 %-confidence prediction should be right ~90 % of the time; miscalibration shows up as a bin-level gap | ≥ 50 items per bin |
| **Adversarial / robustness slice** (contrast sets, Checklist, HANS) | Corpus performance decouples from robust performance; a model-card without a robustness slice is not release-ready (Ribeiro et al. 2020 Checklist; Gardner et al. 2020 contrast sets) | ≥ 100 adversarial items |

The minimum sample sizes are chosen so that a ±2 % absolute difference is distinguishable from noise at
95 % CI with a paired bootstrap (see exercise-02). Smaller slices still get reported, but with a CI
explicitly wide enough that the reviewer sees it is non-decisive.

### 2.4 Rubric as a decision procedure (worked example)

Worked example: a summarisation release on biomedical abstracts, three language variants (en, es, zh).

1. Task shape: **SUMMARISATION** (Q5 "yes": output is a shorter rewrite preserving content).
2. Primary: **ROUGE-L + ROUGE-2** on the test set; language-separated.
3. Secondary: **BERTScore-F1** with the language-matched BERT checkpoint; report model version (`bert-base-multilingual-cased` for es/zh, `bert-base-uncased` for en).
4. Diagnostic: **FactCC** or **QAGS** faithfulness score; **extractivity density + coverage** (Grusky et al. 2018) to detect copy-paste gaming.
5. Slices: by input length (short / medium / long abstracts), by biomedical sub-domain (clinical / preclinical / review), by language. Each slice reports (ROUGE-L mean, 95 % CI via paired bootstrap).
6. Stakeholder utility statement: "Release gate: ROUGE-L ≥ 36.5 on the long-abstract slice in all three languages, AND faithfulness score ≥ 0.85, AND no demographic-slice regression > 1 absolute point vs. previous release. Evaluation sample size: 500 per slice, 10 000 total."

A reviewer runs this through §4 (grading rubric). A rubric that terminates in "release gate" language is
the artefact the stakeholder actually consumes; "we report corpus ROUGE-L" is not.

### 2.5 Reference implementation of the classifier + metric sheet

```python
# rubric.py
from dataclasses import dataclass, field


SHAPES = (
    "CLASSIFICATION", "SEQUENCE_LABELLING", "EXTRACTIVE_QA", "GENERATIVE_QA",
    "SUMMARISATION", "MACHINE_TRANSLATION", "PARAPHRASE", "DIALOGUE",
    "CODE_GENERATION", "OPEN_ENDED_GENERATION",
)


@dataclass(frozen=True)
class MetricSet:
    primary: tuple[str, ...]
    secondary: tuple[str, ...]
    diagnostic: tuple[str, ...]
    note: str


METRIC_SHEET: dict[str, MetricSet] = {
    "CLASSIFICATION": MetricSet(
        primary=("macro_f1",),
        secondary=("accuracy", "per_class_f1"),
        diagnostic=("confusion_matrix", "per_slice_f1"),
        note="Macro-F1 is unaffected by class imbalance; accuracy alone is misleading on skewed label sets.",
    ),
    "SEQUENCE_LABELLING": MetricSet(
        primary=("entity_level_f1",),
        secondary=("token_level_f1",),
        diagnostic=("per_entity_type_f1", "boundary_vs_type_error_split"),
        note="CoNLL 2003 convention: span-exact entity-level F1 is the primary; token-F1 over-rewards near-misses.",
    ),
    "EXTRACTIVE_QA": MetricSet(
        primary=("squad_f1",),
        secondary=("exact_match",),
        diagnostic=("has_ans_f1", "no_ans_f1"),
        note="SQuAD 2.0 splits HasAns and NoAns; collapsing them hides unanswerable-detection failures.",
    ),
    "GENERATIVE_QA": MetricSet(
        primary=("token_f1",),
        secondary=("bertscore_f1",),
        diagnostic=("exact_match_normalized",),
        note="Natural Questions short-answer uses token-F1; BERTScore for paraphrase tolerance.",
    ),
    "SUMMARISATION": MetricSet(
        primary=("rouge_l", "rouge_2"),
        secondary=("bertscore_f1",),
        diagnostic=("faithfulness_qags_or_factcc", "extractivity_density", "extractivity_coverage"),
        note="Pair ROUGE with a faithfulness metric to catch hallucination; extractivity catches copy-paste gaming.",
    ),
    "MACHINE_TRANSLATION": MetricSet(
        primary=("comet",),
        secondary=("bleu", "chrf"),
        diagnostic=("per_language_pair", "per_domain", "per_length_bucket"),
        note="WMT metrics task: learned metrics correlate best with human judgement; report BLEU/chrF for comparability.",
    ),
    "PARAPHRASE": MetricSet(
        primary=("bertscore_f1", "ibleu"),
        secondary=("self_bleu_for_diversity",),
        diagnostic=("semantic_vs_lexical_scatter",),
        note="Paraphrase is simultaneously meaning-preservation and surface-divergence; one metric cannot capture both.",
    ),
    "DIALOGUE": MetricSet(
        primary=("human_evaluation",),
        secondary=("usr_or_dialoglue",),
        diagnostic=("toxicity_rate", "repetition_rate", "length_distribution"),
        note="Liu et al. 2016: automatic metrics correlate near-zero with human judgement on open-domain dialogue.",
    ),
    "CODE_GENERATION": MetricSet(
        primary=("pass_at_k",),
        secondary=("compile_or_lint_pass_rate",),
        diagnostic=("per_problem_difficulty_pass_at_k", "per_language"),
        note="Surface metrics do not correlate with functional correctness; pass@1,10,100 is the HumanEval convention.",
    ),
    "OPEN_ENDED_GENERATION": MetricSet(
        primary=("human_evaluation",),
        secondary=(),
        diagnostic=("perplexity_held_out", "lm_as_judge_pairwise"),
        note="No automatic metric is validated; saying so in the model card is required.",
    ),
}


@dataclass(frozen=True)
class Question:
    text: str
    yes_shape: str


QUESTIONNAIRE: tuple[Question, ...] = (
    Question("Is there a finite, fixed label set and the output is one label per input?", "CLASSIFICATION"),
    Question("Is there a finite, fixed label set and the output is one label per token?", "SEQUENCE_LABELLING"),
    Question("Is the output a span from the input (verbatim substring)?", "EXTRACTIVE_QA"),
    Question("Is the output a short free-form answer to a question?", "GENERATIVE_QA"),
    Question("Is the output a shorter rewrite of a longer input, meant to preserve content?", "SUMMARISATION"),
    Question("Is the output a translation into a different language?", "MACHINE_TRANSLATION"),
    Question("Is the output a meaning-preserving rewrite in the same language?", "PARAPHRASE"),
    Question("Is the output a turn in a multi-turn dialogue with no single correct reference?", "DIALOGUE"),
    Question("Is the output source code meant to satisfy a specification?", "CODE_GENERATION"),
)


def classify(answers: list[bool]) -> str:
    """Return the first shape whose question got a Yes; otherwise OPEN_ENDED_GENERATION."""
    assert len(answers) == len(QUESTIONNAIRE), "need one answer per question"
    for ans, q in zip(answers, QUESTIONNAIRE):
        if ans:
            return q.yes_shape
    return "OPEN_ENDED_GENERATION"


def recommend(shape: str) -> MetricSet:
    assert shape in SHAPES, f"unknown shape {shape!r}"
    return METRIC_SHEET[shape]


@dataclass
class ReleaseGate:
    shape: str
    primary_target: dict[str, float]
    slice_min_n: int = 100
    regression_tolerance_pts: float = 1.0

    def accepts(self, measured: dict[str, float], per_slice_n: dict[str, int]) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        for metric, target in self.primary_target.items():
            if measured.get(metric, float("-inf")) < target:
                reasons.append(f"{metric} = {measured.get(metric):.3f} < target {target:.3f}")
        for slice_name, n in per_slice_n.items():
            if n < self.slice_min_n:
                reasons.append(f"slice {slice_name!r} has n={n} < {self.slice_min_n}")
        return len(reasons) == 0, reasons
```

The classifier and metric sheet are code so the rubric itself is testable; a change to the metric
sheet is a code review, not a wiki edit.

## 3. Validation steps

1. **Classifier terminates on every task.** Feed the questionnaire 30 published-task descriptions
   (SQuAD, CoNLL 2003, WMT en-de, XSum, CNN/DailyMail, HumanEval, MBPP, MultiWOZ, DSTC, QNLI,
   MNLI, PAWS, SST-2, etc.); every one must land on the published shape. Discrepancies flag an
   ambiguous question wording.
2. **Metric choice matches the published benchmark convention.** For each of the 30 tasks, the
   `recommend(shape)` primary must match the metric the benchmark's leaderboard uses as its primary
   (SQuAD → F1; CoNLL → entity-F1; WMT → COMET + BLEU; HumanEval → pass@k; etc.). A mismatch means
   either the shape questionnaire miscategorised the task or the metric sheet needs a split.
3. **Secondary metric exposes known failure modes.** On a toy summarisation set, construct three
   systems: (a) extractive copy-paste (will score high ROUGE, low faithfulness, high extractivity),
   (b) fluent hallucination (low ROUGE, low faithfulness, low extractivity), (c) abstract-and-faithful
   (medium ROUGE, high faithfulness, medium extractivity). The secondary + diagnostic metrics must
   rank them in that order for faithfulness; ROUGE alone will not.
4. **Slice min-sample-size enforcement.** `ReleaseGate.accepts()` must reject a submission with any
   slice below the minimum. Golden-list test.
5. **Model-version pinning.** Report a BERTScore value with (a) `bert-base-uncased` and (b) `roberta-large`
   on the same system; publish both numbers and show the delta. Downstream re-runs that change the
   checkpoint without updating the card must fail review.
6. **Cross-rubric consistency.** For each of the 10 shapes, every entry in `MetricSet.primary` has at
   least one published paper citing it as the primary (recorded in §6 references). Entries without a
   citation are not shippable; this is enforced as part of code review.
7. **Correlation with human judgement, where available.** For MT, re-produce the WMT 2022 metric
   correlation table (Freitag et al. 2022) on a sample; COMET should out-correlate BLEU. If the delta
   is reversed, the implementation is mis-wired.
8. **Fairness-slice presence.** For every classification or extractive task, the slicing plan must
   include at least one demographic subgroup when the data warrants (Blodgett et al. 2020; Mitchell
   et al. 2019 Model Cards). A model card with no fairness slice on an NLI dataset is incomplete.
9. **Release-gate round-trip.** Take a prior-release model card, extract its claimed metrics, feed
   through `ReleaseGate.accepts()`; the result must match the published decision. Catches gates that
   were authored but never actually enforced.

## 4. Grading rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Task shape** | Shape explicitly identified from the 10 canonical shapes, with the question answers recorded | Shape named, questionnaire not shown | No shape declared |
| **Primary metric** | Matches the per-shape sheet and is cited to a primary source | Matches but uncited | Mismatched metric (e.g., BLEU for summarisation) |
| **Secondary metric** | Different failure-mode family from primary; cited | One metric only | Only redundant metrics (e.g., ROUGE-1 + ROUGE-2 with no semantic metric) |
| **Diagnostic slice** | Length + domain (+ language if multilingual) + one demographic subgroup when relevant; min sample size enforced | Length only | None |
| **Model-version pinning** | BERTScore / COMET / BLEURT checkpoint name recorded on model card | Metric named, no version | Neither |
| **Significance** | Point estimate + 95 % CI via paired bootstrap (see exercise-02) | Point estimate + stddev | Point estimate only |
| **Robustness slice** | Contrast set / Checklist / HANS or equivalent reported | Mentioned, not reported | None |
| **Stakeholder utility** | Metric section terminates in a release-gate statement with target thresholds per slice | Thresholds listed, no gate statement | No connection to stakeholder decision |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS metric-specific risks (metric-gaming, benchmark contamination) flagged | Mentioned | Ignored |
| **Documentation / card** | NIST AI RMF *Map* (what is measured), *Measure* (how), *Manage* (release gate); Model Cards for Model Reporting (Mitchell et al. 2019) fields filled | Partial | Neither |

## 5. Common mistakes

- **Reporting BLEU for summarisation, ROUGE for MT.** These are shape mismatches: BLEU is a precision-
  oriented n-gram metric designed for multi-reference MT; ROUGE is recall-oriented for summarisation.
  Swapping them is an immediate review-block.
- **Treating ROUGE-L alone as sufficient for summarisation.** ROUGE-L is longest-common-subsequence
  and is gameable by extractive methods that pick long spans verbatim from the input. Always pair
  with ROUGE-2 (bigram, order-sensitive) and a semantic metric.
- **Reporting corpus-BLEU as a single number.** Freitag et al. 2022 and the WMT metrics shared task
  recommendations are explicit: publish per-system corpus BLEU alongside at least one learned metric
  (COMET) and per-system chrF. A single BLEU number is non-decision-ready.
- **Using BERTScore without recording the underlying checkpoint.** Zhang et al. 2020 note that
  BERTScore values vary with the base model; publishing "BERTScore = 0.87" without the checkpoint is
  equivalent to publishing "accuracy = 87 %" without the test set.
- **Running significance tests at the system level and reporting a single p-value for a release.**
  Per-slice significance is the actionable signal (see exercise-02); corpus significance can be
  driven by one noisy domain. Report per-slice CIs.
- **Reporting `pass@1` only for code generation.** HumanEval / MBPP conventions report pass@{1, 10,
  100}; pass@1 is uniquely sensitive to decoding temperature and is the easiest to game by over-tuning
  sampling parameters.
- **Choosing a single metric to optimise against during training.** Every learned metric has known
  failure modes; optimising BLEU inflates length-mismatch gaming, optimising ROUGE inflates
  extractivity, optimising COMET inflates known COMET biases (Freitag et al. 2022 §5). Keep a
  separate held-out metric suite that was *not* used in training-signal design.
- **Picking a metric because an older paper used it.** BLEU was the standard in MT papers 2002–2018;
  by 2023 the WMT metrics task recommends against reporting BLEU as the primary. Use the current
  year's WMT metrics task recommendations.
- **Reporting fairness by averaging across demographic groups.** A group-mean hides within-group
  regressions. Publish per-group numbers with CIs; do not average them.
- **Running LM-as-judge without human calibration.** Zheng et al. 2023 (MT-Bench) show LM-as-judge
  correlates with human preferences only after calibration; using it as the sole metric for release
  acceptance is unvalidated.
- **No release gate.** A model card with eight metrics and no decision rule is not operational.
  Stakeholders will invent a rule; make it explicit instead.
- **Benchmark contamination treated as a separate concern from metric selection.** A high BLEU on a
  test set that is in the training corpus is not a valid metric; the metric rubric must include a
  contamination check (see exercise-03). Numbers reported on contaminated test sets must be flagged.

## 6. References

Primary sources used in this solution:

- Papineni, K. et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. ACL. <https://aclanthology.org/P02-1040/> — the foundational n-gram-precision + brevity-penalty metric; §2.2 references its known limits.
- Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*. ACL Workshop. <https://aclanthology.org/W04-1013/> — ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-SU defined; §2.2.
- Banerjee, S., Lavie, A. (2005). *METEOR: An Automatic Metric for MT Evaluation With Improved Correlation with Human Judgments*. ACL Workshop. <https://aclanthology.org/W05-0909/> — alignment + stem/synonym matching.
- Popović, M. (2015). *chrF: character n-gram F-score for automatic MT evaluation*. WMT. <https://aclanthology.org/W15-3049/> — character-F; closes several BLEU morphology limitations.
- Popović, M. (2017). *chrF++: words helping character n-grams*. WMT. <https://aclanthology.org/W17-4770/> — chrF with word-order extensions.
- Zhang, T. et al. (2020). *BERTScore: Evaluating Text Generation with BERT*. ICLR. <https://arxiv.org/abs/1904.09675> — contextual-embedding similarity; §2.2 and model-version caveat.
- Zhao, W. et al. (2019). *MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance*. EMNLP. <https://aclanthology.org/D19-1053/> — learned-metric family.
- Sellam, T. et al. (2020). *BLEURT: Learning Robust Metrics for Text Generation*. ACL. <https://aclanthology.org/2020.acl-main.704/> — BERT-based regression metric.
- Rei, R. et al. (2020). *COMET: A Neural Framework for MT Evaluation*. EMNLP. <https://aclanthology.org/2020.emnlp-main.213/> — the COMET family.
- Rei, R. et al. (2022). *COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task*. WMT. <https://aclanthology.org/2022.wmt-1.52/> — the current (as of the 2022 WMT shared task) recommended COMET checkpoint; the §2.2 MT primary.
- Freitag, M. et al. (2022). *Results of the WMT22 Metrics Shared Task: Stop Using BLEU – Neural Metrics Are Better and More Robust*. WMT. <https://aclanthology.org/2022.wmt-1.2/> — the authoritative "which MT metric correlates best" ranking for 2022; drives the §2.2 MT row.
- Mathur, N. et al. (2020). *Tangled up in BLEU: Reevaluating the Evaluation of Automatic Machine Translation Evaluation Metrics*. ACL. <https://aclanthology.org/2020.acl-main.448/> — the empirical case that BLEU under-discriminates modern MT systems; §2.2 note.
- Chen, M. et al. (2021). *Evaluating Large Language Models Trained on Code* (Codex / HumanEval). arXiv. <https://arxiv.org/abs/2107.03374> — defines pass@k; §2.2 code row.
- Rajpurkar, P. et al. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text*. EMNLP. <https://aclanthology.org/D16-1264/> — SQuAD 1.1 exact-match + F1 conventions.
- Rajpurkar, P. et al. (2018). *Know What You Don't Know: Unanswerable Questions for SQuAD*. ACL. <https://aclanthology.org/P18-2124/> — SQuAD 2.0 HasAns / NoAns split.
- Kwiatkowski, T. et al. (2019). *Natural Questions: A Benchmark for Question Answering Research*. TACL. <https://aclanthology.org/Q19-1026/> — NQ token-F1 convention for short answers.
- Tjong Kim Sang, E. F., De Meulder, F. (2003). *Introduction to the CoNLL-2003 Shared Task*. CoNLL. <https://aclanthology.org/W03-0419/> — entity-level span-exact F1 for NER.
- Liu, C. W. et al. (2016). *How NOT to Evaluate Your Dialogue System*. EMNLP. <https://aclanthology.org/D16-1230/> — automatic metrics correlate near-zero with human judgement on open-domain dialogue; §2.2 dialogue row.
- Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS. <https://arxiv.org/abs/2306.05685> — LM-as-judge calibration; §2.2 and §5.
- Falke, T. et al. (2019). *Ranking Generated Summaries by Correctness: An Interesting but Challenging Application for Natural Language Inference*. ACL. <https://aclanthology.org/P19-1213/> — ROUGE's weakness on abstractive hallucination.
- Grusky, M. et al. (2018). *Newsroom: A Dataset of 1.3 Million Summaries with Diverse Extractive Strategies*. NAACL. <https://aclanthology.org/N18-1065/> — defines extractivity density/coverage used as a §2.2 diagnostic.
- Dror, R. et al. (2018). *The Hitchhiker's Guide to Testing Statistical Significance in NLP*. ACL. <https://aclanthology.org/P18-1128/> — paired-bootstrap methodology used in §3 and §4; pairs with exercise-02.
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAccT. <https://arxiv.org/abs/1810.03993> — the §4 "release gate" and documentation row.
- Blodgett, S. et al. (2020). *Language (Technology) is Power: A Critical Survey of "Bias" in NLP*. ACL. <https://aclanthology.org/2020.acl-main.485/> — fairness-slice motivation for §2.3.
- Ribeiro, M. T. et al. (2020). *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*. ACL. <https://aclanthology.org/2020.acl-main.442/> — robustness-slice motivation.
- Gardner, M. et al. (2020). *Evaluating Models' Local Decision Boundaries via Contrast Sets*. Findings of EMNLP. <https://aclanthology.org/2020.findings-emnlp.117/> — contrast-set robustness evaluation.
- Sokolova, M., Lapalme, G. (2009). *A systematic analysis of performance measures for classification tasks*. Information Processing & Management 45(4). — macro-F1 vs. accuracy on imbalanced data.
- Napoles, C. et al. (2015). *Ground Truth for Grammatical Error Correction Metrics*. ACL. <https://aclanthology.org/P15-2097/> — GLEU for GEC.
- Bryant, C. et al. (2017). *Automatic Annotation and Evaluation of Error Types for Grammatical Error Correction*. ACL. <https://aclanthology.org/P17-1074/> — ERRANT.
- Liang, P. et al. (2023). *Holistic Evaluation of Language Models* (HELM). TMLR. <https://arxiv.org/abs/2211.09110> — the multi-metric, multi-slice evaluation framework the §2.3 slicing plan is designed to feed.
- Srivastava, A. et al. (2023). *Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models* (BIG-bench). TMLR. <https://arxiv.org/abs/2206.04615>.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML06 AI Supply Chain Attacks* applies to learned metrics: if BERTScore / COMET / BLEURT model artefacts are tampered in the supply chain, every downstream metric value is compromised. Pin checkpoint hashes in the model card. *ML07 Transfer Learning Attack* applies when the base checkpoint for a learned metric was trained on data that overlaps the evaluation set; publishing the metric checkpoint provenance (training corpus) is the mitigation.
- MITRE ATLAS — <https://atlas.mitre.org/> — *AML.T0043 Craft Adversarial Data* applies to benchmark gaming where models over-fit to the surface patterns a metric rewards; the §2.2 "secondary + diagnostic" pattern (always pair surface and semantic metrics) is the primary mitigation. *AML.T0005 Discover Machine Learning Model Family* is tangentially relevant: public metric suites make it easy to profile systems.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: the task shape, the chosen metric set, the slicing plan, and the stakeholder decision the metric supports are the primary Map artefacts. *Measure*: the point estimate + CI per slice is the Measure artefact; `ReleaseGate.accepts()` is the Measure-driven gate. *Manage*: metric suite changes are code-review controlled; benchmark contamination (exercise-03) is a Manage risk; metric-checkpoint supply chain (ML06 above) is a Manage risk. *Govern*: the model-card template (Mitchell et al. 2019) is the policy artefact the rubric produces.
