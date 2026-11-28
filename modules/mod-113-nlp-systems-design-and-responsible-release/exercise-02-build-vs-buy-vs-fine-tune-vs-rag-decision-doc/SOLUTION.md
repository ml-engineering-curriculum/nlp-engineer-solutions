# exercise-02 — Build vs. Buy vs. Fine-Tune vs. RAG Decision Doc

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-113-nlp-systems-design-and-responsible-release/exercises/exercise-02-build-vs-buy-vs-fine-tune-vs-rag-decision-doc.md`.
> The target learning outcomes implied by that filename: given a concrete NLP use case
> and a set of constraints, write a **decision document** that evaluates four build
> options — (a) build-from-scratch, (b) buy a managed API, (c) fine-tune an open-weights
> model, (d) assemble a RAG system — against the use case's actual requirements, and
> picks one with an auditable trail.

## 1. Solution overview

Teams default to whichever option is culturally in fashion at the time: "just call GPT",
"fine-tune Llama", "do a RAG". The decision document exists to surface the question
*behind* the choice: which dimensions of the use case actually force the hand, and which
are preferences dressed up as requirements.

The four options compared in this exercise, with their canonical strengths and
weaknesses:

| Option | Strong when | Weak when | Watch out for |
|---|---|---|---|
| **a) Build from scratch** (train a task-specific model, usually a smaller encoder) | Dedicated, stable task; sufficient labelled data; strict latency/cost SLO; need full IP ownership | Data is sparse or shifting; task is open-ended; team lacks training infra | High fixed cost; underperforms on long-tail inputs; risks reinventing solved problems |
| **b) Buy a managed API** (hosted LLM / vendor NLP endpoint) | Short time-to-first-value; general-purpose text tasks; low volume or variable load; contract allows the data flow | Data is regulated and can't leave the perimeter; volume is high enough that per-call pricing beats self-hosting; SLO or availability must be stricter than the vendor's | Vendor lock-in; silent model updates breaking your eval; data-retention terms |
| **c) Fine-tune open-weights** | You have task-specific labelled data (often < 10 k rows is enough for LoRA/PEFT); you need to deploy inside your perimeter; you need deterministic, pin-able behaviour | Zero labelled data; constantly-changing world knowledge; team lacks eval infra | Supply-chain risk on the base model; license obligations (e.g. commercial-use clauses); drift of base checkpoint |
| **d) Retrieval-Augmented Generation (RAG)** | The *answer* is inside documents you control; freshness matters; traceability to a source is a requirement | The task is not grounded in a retrievable corpus (e.g. summarisation of the input itself); retrieval failures dominate error modes | Retrieval quality dominates generator quality; evaluation is multi-stage (retriever + generator); prompt-injection via retrieved content |

Dimensions the decision must score each option against (these are non-negotiable
columns, not a wishlist):

| Dimension | Why it matters | How to measure |
|---|---|---|
| **Task fit** | An option that structurally cannot produce the required output is disqualified immediately | Map to section 3 of the task brief (exercise-01); does the option produce the output shape? |
| **Data constraints** | Regulated data cannot cross organisational perimeters; an API that takes the data out is disqualified | Data classification register; vendor DPA / data-residency terms |
| **Quality (primary metric + slices)** | The release gates in the task brief must be met in the worst slice, not just on average | Offline eval on the labelled test set; per-slice decomposition |
| **Latency and throughput SLO** | A chosen option that cannot meet p95 is a release blocker regardless of accuracy | Measured on the target SKU under realistic load |
| **Cost at projected volume** | Per-call economics flip at some volume; you have to compute, not guess, the crossover | $/1 k requests; include GPU hours for self-hosted; include labelling cost for fine-tune |
| **Time-to-first-value** | For product validation, 2 weeks vs. 6 months is often the deciding factor | Realistic engineering estimate with a p50 and p90 |
| **Operational fit** | The team has to run this. An option that requires skills the team doesn't have is a hidden tax | Honest skills audit; what you'll need to hire, train, or outsource |
| **Risk surface (OWASP ML / ATLAS)** | Each option has different risks; the comparison must be apples-to-apples | Map each option to OWASP ML Top 10 and MITRE ATLAS techniques |
| **Governance fit** | NIST AI RMF `Govern` imposes documentation and oversight obligations that are cheaper/harder depending on the option | Named owner, model-card obligations, change-management process |
| **Exit cost** | The cost of being wrong — moving off the option later | Does switching require re-labelling? Re-training? Renegotiating a contract? |

The deliverable is a **one-page decision summary** (final choice + top three reasons)
plus a **three-to-five-page comparison matrix** with scored rationale per dimension.

Authoritative sources this solution draws on:

- **NIST AI Risk Management Framework (AI RMF 1.0)** —
  <https://www.nist.gov/itl/ai-risk-management-framework>. The decision document is the
  AI RMF *Map 3 (categorise the AI system)* + *Manage 1 (prioritise risks)* artefact.
- **OWASP Machine Learning Security Top 10** —
  <https://owasp.org/www-project-machine-learning-security-top-10/>. Each option scored
  on the OWASP ML entries it is exposed to.
- **MITRE ATLAS** — <https://atlas.mitre.org/>. Adversary-centric techniques scored per
  option.
- **Model Cards** (Mitchell et al. 2019). <https://arxiv.org/abs/1810.03993>. The
  governance obligation is to produce one per option that goes to production.
- **Datasheets for Datasets** (Gebru et al. 2021). <https://arxiv.org/abs/1803.09010>.
  The data-provenance obligation for options (a) and (c).

## 2. Worked implementation — decision doc for the email-triage case from exercise-01

Reuse the use case from exercise-01 — **support-email intent classification + entity
summary** — so the decision has concrete numeric constraints to compare against:

- **Primary metric gate:** macro-F1 ≥ 0.82; per-language slice ≥ 0.78; enterprise-tier
  slice ≥ 0.80.
- **Latency SLO:** p95 ≤ 300 ms on target CPU SKU.
- **Volume:** 200 emails/agent/shift × 400 agents × 3 shifts = 240 000 emails/day.
- **Data classification:** PII, must stay within the organisational perimeter (GDPR).
- **Time-to-first-value:** 6 weeks to a shippable v1.
- **Rolling re-label budget:** 100 emails/week.

### 2.1 Comparison matrix

Scoring is 0 (disqualifier), 1 (poor fit, major cost), 2 (acceptable), 3 (good fit).
Any 0 is a disqualifier regardless of totals. Weighted total omitted intentionally — if
two options tie on hard constraints, the team picks on soft dimensions in a meeting,
not by arithmetic.

| Dimension | (a) Build from scratch | (b) Buy managed LLM API | (c) Fine-tune open-weights encoder | (d) RAG over emails |
|---|---|---|---|---|
| **Task fit** | 3 — classifier + NER is a solved shape | 2 — generator must be constrained to the JSON schema | 3 — fine-tuned classifier is the canonical fit | 1 — the input *is* the document; nothing to retrieve |
| **Data constraints (GDPR)** | 3 — inside perimeter | **0** — PII leaves perimeter under default terms; conditional on vendor DPA with EU residency + no-training clause | 3 — inside perimeter | 2 — RAG index is inside perimeter; generator is the same issue as (b) |
| **Quality: primary metric** | 2 — majority-class baseline is macro-F1 ≈ 0.42; reaching 0.82 requires substantially more labels than the 2 000 available | 3 — zero-/few-shot LLM baselines commonly exceed 0.82 on intent classification *with careful prompt engineering* — verify on the real test set, do not assume | 3 — `xlm-roberta-base` fine-tuned on 2 000 rows reaches ≥ 0.82 macro-F1 on comparable intent tasks (verify on the real test set) | 2 — the generator is as good as (b) once constrained; retrieval adds no signal |
| **Quality: per-language slice** | 2 — multilingual model from scratch is harder | 2 — vendor LLMs cover en/es/fr strongly but slice performance varies and is not disclosed | 3 — XLM-RoBERTa is multilingual by design | 2 — same as (b) |
| **Latency p95 ≤ 300 ms** | 3 — small classifier on CPU | 1 — vendor LLM over network, typical p95 400–1 500 ms; may need streaming + caching | 3 — `xlm-roberta-base` on CPU meets 300 ms; verified in mod-112 | 1 — retriever + generator; each adds latency |
| **Cost at 240 k/day** | 2 — one GPU-hour amortised; mostly engineering | 1 — at $0.001/call, $240/day = $87 k/yr; at higher token counts, much more | 3 — self-hosted CPU inference is cheapest; one GPU for training only | 2 — index host + generator calls |
| **Time-to-first-value** | 1 — building a labelled dataset and training from scratch is > 6 weeks | 3 — days to a working v1 | 2 — 4–6 weeks with existing labelling plan | 2 — retriever build adds setup |
| **Operational fit** | 1 — team can label and fine-tune; building a classifier from scratch adds infra not needed | 2 — team has to build eval and rate-limit handling; no training infra needed | 3 — matches team skills; aligns with mod-111/112 outputs | 2 — new retrieval infra to run |
| **Risk surface (OWASP ML / ATLAS)** | 2 — ML02 (poisoning via labels), ML06 (dep chain) | 2 — ML06 (vendor model changes silently), ATLAS `AML.T0051 LLM Prompt Injection`; ML09 (output integrity) if JSON parse is tolerant | 2 — ML06 (base checkpoint supply chain), mitigated by revision pin | 2 — ML06 + ATLAS prompt injection via retrieved content |
| **Governance (NIST AI RMF)** | 3 — full `Govern` possible; we own the model card | 1 — vendor-controlled model; our `Govern` artefact must note the vendor dependency | 3 — full `Govern`; model card for the fine-tune | 2 — mixed; generator is as (b) |
| **Exit cost** | 3 — artefacts are ours | 1 — swap requires re-prompting, re-eval; data in the vendor's logs | 2 — labels transfer; weights don't but pipeline does | 2 — index transfers; generator swap required |

### 2.2 Shortlist and scoring

- **(b) Managed LLM API** carries a `0` on data constraints under default terms. Not a
  disqualifier *forever* — a vendor DPA with EU residency, a no-training clause, and a
  short retention period can lift the score to `2`. But the exercise's constraint is
  "treated as personal data"; absent the DPA, this is out.
- **(a) Build from scratch** is dominated by **(c)** on every dimension where they
  differ. "Build from scratch" means not using a pre-trained base; with 2 000 labels and
  a 6-week deadline, this is a slow path to a worse outcome. Scoring reflects that.
- **(d) RAG** scores `1` on task fit. The input is the document; there is no retrieval
  corpus that would help classify the email's intent. This is the use-case-wrong-tool
  pattern — a disqualifier.
- **(c) Fine-tune open-weights encoder** scores `3` on data, quality, latency, operations,
  and governance. Cost at volume is favourable. Exit cost is moderate (labels are
  transferable).

### 2.3 Chosen option — with rationale

**Chosen: (c) Fine-tune `xlm-roberta-base` + spaCy NER.** Three reasons, in order of
weight:

1. **Data constraint is a hard one.** The data is PII; sending it to a managed API
   requires a vendor DPA that is not in hand. Option (c) keeps the data inside the
   perimeter and satisfies GDPR Article 28 processor obligations directly.
2. **Latency SLO is a hard one.** The 300 ms p95 budget rules out any option that
   involves an external network call on the hot path. Managed APIs and RAG stacks pay
   that cost; the encoder on CPU does not.
3. **Quality gate is reachable with the labels we have.** XLM-RoBERTa fine-tuned on
   a few thousand intent-labelled examples is a well-trodden path; the risk is
   execution, not feasibility. (Verify on the real test set before release; do not
   assume the published benchmark numbers transfer.)

The combination of (1) and (2) is what collapses the decision: the two hard constraints
remove two options, and option (a) is dominated. Option (c) is the only one left.

**Explicit assumptions documented to make the decision falsifiable:**

- The 2 000 labelled emails give an achievable macro-F1 ≥ 0.82 at release. If the
  calibration experiment at week 3 shows this is not holding, re-open the decision:
  the fallback path is (b) *with a vendor DPA and a strict JSON-schema wrapper*.
- The target CPU SKU can run `xlm-roberta-base` at p95 ≤ 300 ms (per mod-112 exercise-02
  measurements). If the latency harness shows otherwise, the fallback is quantised ONNX
  per mod-112 exercise-03 or a smaller multilingual encoder.
- GDPR counsel reviewed the data classification and did not identify an additional
  constraint that disqualifies option (c). This review is captured in the sign-off
  trail.

**Explicit tripwires for revisiting the decision:**

- If the rolling re-label stream shows intent prior drift > 15 pp over one quarter
  (concept drift), re-evaluate; a constantly-shifting task may favour a prompt-based
  option with fewer retraining overheads.
- If annual run-cost of (c) exceeds a managed API's projected price at current
  volume (including DPA + no-training terms), re-open at next planning cycle.
- If a new regulation or a new PII class appears in the input, re-open.

## 3. Validation steps

1. **Disqualifier check.** For each option, verify the scored `0`s are actual
   disqualifiers and not pessimistic scoring. "Data residency" is a disqualifier only
   if the vendor cannot provide it; confirm with the vendor DPA before marking.
2. **Baseline floors.** The quality row cannot be filled with "the model is good enough"
   from the vendor sales deck. Run each candidate against the labelled test set, or at
   minimum against a small pilot set, and record the numbers in the doc.
3. **Latency numbers measured, not estimated.** The latency row must cite a measurement
   on the target SKU, not a vendor-published SLA. SLAs are p50; your constraint is p95.
4. **Cost model is honest.** Include labelling cost in option (c)'s TCO; include
   engineering time in option (a); include vendor DPA legal cost in option (b); include
   index build and generator cost in option (d). A cost column with only one line-item
   per option is under-specified.
5. **Risk entries cite identifiers.** For each option, the OWASP ML and MITRE ATLAS
   entries are cited by identifier (ML06, AML.T0051, etc.), not by paraphrase. A reviewer
   should be able to open the standard and verify.
6. **Governance obligations named.** For each option, name the model-card (or vendor
   substitute), datasheet, release-gate owner, and change-management path. "We'll figure
   that out" means the governance row is unfilled.
7. **Sign-off trail recorded.** PM + ML lead + security + (for options (b)/(d)) legal
   counsel are the minimum signatures. Name each; a document signed only by the author
   does not meet AI RMF `Govern` requirements.
8. **Tripwires on calendar.** The "explicit tripwires" are added to the quarterly
   planning calendar. A decision doc with tripwires that are never revisited is theatre.

## 4. Grading rubric / review checklist

| Dimension | Pass (2) | Partial (1) | Fail (0) |
|---|---|---|---|
| **Options compared** | All four options (build, buy, fine-tune, RAG) scored | Three options scored | Fewer than three options; or options are not meaningfully compared |
| **Comparison dimensions** | At least the 10 dimensions listed in section 1 scored per option | Some dimensions skipped | Decision made without a comparison matrix |
| **Disqualifier handling** | Hard constraints identified explicitly; any `0` is justified from a named source (regulation, SLO, contract) | Disqualifiers listed informally | No distinction between disqualifier and preference |
| **Quality numbers** | Primary metric compared per option against the release-gate; per-slice decomposition | Overall quality compared only | Vendor claims cited uncritically |
| **Latency numbers** | p95 measured on the target SKU | p50 only or vendor SLA cited | No latency numbers |
| **Cost model** | TCO with labelling, infra, vendor, engineering, exit | Operational cost only | None |
| **Risk entries** | OWASP ML + MITRE ATLAS cited by identifier per option | Generic risk list | None |
| **Governance** | Named sign-offs; model-card / datasheet obligations; change-management path per option | Sign-offs only | None |
| **Explicit assumptions** | Assumptions listed and tied to a verification activity | Assumptions implicit | None |
| **Tripwires for revisit** | Named triggers with owners and cadence | Generic "we'll revisit" | None |

A document missing hard-constraint analysis (disqualifiers), measured latency/quality
numbers, or named sign-offs is not a decision document; it is a preference.

## 5. Common mistakes

- **Letting fashion drive the comparison.** "We used LLMs for the last project" is not
  a technical requirement. The decision matrix exists to make the fashion visible so it
  can be debated instead of assumed.
- **Scoring on total-of-scores.** Hard constraints are disqualifiers; they do not
  average with soft preferences. An option that scores 3 on nine rows and 0 on one row
  is out, not a near-winner.
- **Using vendor-published benchmarks as the quality number.** Vendor leaderboards are
  not your task. The quality column must show numbers from your held-out test set, or
  at minimum a pilot run on representative data.
- **Measuring latency at p50 for a p95 SLO.** p50 latency hides the tail. For a hot-path
  system with a 300 ms p95 budget, you need p95 and p99 — not p50 — on the target SKU
  with the target batch size.
- **Omitting labelling cost from fine-tune TCO.** Labelling is often the dominant line
  item for option (c); omitting it makes it look artificially cheap.
- **Treating option (b) as free of operational cost.** Managed APIs have ops cost: rate
  limits, retries, timeouts, vendor-side drift detection, DPA upkeep. "We just call the
  API" is not a complete operational model.
- **Treating option (d) RAG as task-independent.** RAG is the right choice when the
  answer is in a retrievable corpus. Applied to a task where the input *is* the
  document (classification, summarisation of the single input), it adds latency without
  adding signal.
- **Ignoring exit cost.** Being locked into a vendor's prompt format or a bespoke base
  model has a cost that shows up on the day you try to swap. Score it.
- **Risk analysis that lists OWASP generically.** "OWASP ML Top 10 applies" is not
  analysis. Each applicable entry is cited by identifier and tied to a specific exposure
  under this option, with a mitigation or an accepted-risk entry.
- **No sign-offs or no tripwires.** An unsigned, unmonitored decision doc is a wiki
  page. NIST AI RMF `Govern` requires accountability and oversight; put the names on the
  doc and put the tripwires on the calendar.

## 6. References

Primary standards and official projects used in this solution:

- NIST AI Risk Management Framework (AI RMF 1.0). <https://www.nist.gov/itl/ai-risk-management-framework>
  — the `Map 3` (categorise the AI system) + `Manage 1` (prioritise risks) framing of
  the decision document.
- OWASP Machine Learning Security Top 10. <https://owasp.org/www-project-machine-learning-security-top-10/>
  — risk-column entries scored per option.
- MITRE ATLAS. <https://atlas.mitre.org/> — adversary-technique entries scored per option.

Supporting literature:

- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAccT.
  <https://arxiv.org/abs/1810.03993> — the governance artefact each option must produce.
- Gebru, T. et al. (2021). *Datasheets for Datasets*. CACM.
  <https://arxiv.org/abs/1803.09010> — the data-provenance obligation for options (a)
  and (c).

Cross-module references:

- Measured latency numbers (section 2.1) draw on `mod-112-production-nlp-pipelines/exercise-02`.
- Per-slice evaluation (quality row) is implemented in `mod-111-nlp-evaluation`.
- Supply-chain pinning for option (c) base-checkpoint risk is implemented in
  `mod-112-production-nlp-pipelines/exercise-01`.
