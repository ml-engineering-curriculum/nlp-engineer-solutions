# exercise-04 — Domain NLP Patterns Survey

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-113-nlp-systems-design-and-responsible-release/exercises/exercise-04-domain-nlp-patterns-survey.md`.
> The target learning outcomes implied by that filename: survey the recurring **NLP
> system patterns** across a set of high-regulated or high-stakes domains (healthcare,
> legal, finance, customer support, developer tooling, scientific literature), identify
> which **architecture, evaluation, and governance** choices are domain-specific and
> which are generic, and produce a survey a product/engineering reader can use to pick
> a reference pattern for a new project.

## 1. Solution overview

Every domain has its own "default stack" that evolves because its inputs and
regulatory environment force certain choices. A survey is useful for two reasons:

- A team starting in a new domain can orient faster by pattern-matching.
- A team crossing domains (e.g. support → healthcare triage) can see which assumptions
  have to be re-examined.

What the survey is **not**: a vendor shootout, a benchmark leaderboard, or a claim about
specific accuracy numbers on specific datasets. We do not quote numbers we cannot
reproduce; where a specific metric would be useful but has not been verified in this
exercise, we state the gap qualitatively and leave the numerical claim to a cited
follow-up, rather than publish invented statistics.

Dimensions used to compare patterns across domains:

| Dimension | What it captures |
|---|---|
| **Typical input modality** | Short conversational turn, long document, semi-structured form, transcript of speech, code, chart/table embedded in text |
| **Typical output shape** | Class label, extracted span, structured record, free text, retrieval citation, code suggestion |
| **Dominant task family** | Classification, extraction, retrieval+generation, summarisation, translation, code completion |
| **Ground-truth source** | Expert annotators (clinician/lawyer/analyst), proxy labels from downstream outcomes, user feedback, rule-based gold set |
| **Typical evaluation axes** | Per-slice metrics, faithfulness/hallucination rate, factuality citations, calibration, abstain rate, safety policy adherence |
| **Governance regime** | HIPAA / FDA SaMD (healthcare), SEC / MiFID / SR 11-7 (finance), legal-privilege (law), GDPR + corporate AI policy (support), SOC2 (developer tooling) — each informs data handling, audit, and change-management |
| **Dominant failure modes** | Hallucination with authority, privacy leakage, slice regression, jurisdiction mismatch, stale knowledge, prompt injection via document |
| **Rollout constraint** | Human-in-the-loop required, shadow-only in regulated domains, red-team approval before user-facing, continuous supervised eval |

The deliverable is a one-page-per-domain survey with a cross-domain synthesis at the
end. Below is the full survey for six domains.

Authoritative sources this solution draws on:

- **NIST AI Risk Management Framework (AI RMF 1.0)** —
  <https://www.nist.gov/itl/ai-risk-management-framework>. The `Govern`-`Map`-`Measure`-
  `Manage` loop is cross-domain; the specific controls vary.
- **OWASP Machine Learning Security Top 10** —
  <https://owasp.org/www-project-machine-learning-security-top-10/>. Which entries are
  dominant varies by domain.
- **MITRE ATLAS** — <https://atlas.mitre.org/>. Adversary techniques vary by the
  attacker's access — retrieval corpora in RAG expose `AML.T0051 Prompt Injection`,
  training data pipelines expose `AML.T0020 Poison Training Data`, etc.
- **Mitchell et al. 2019** — *Model Cards for Model Reporting*. <https://arxiv.org/abs/1810.03993>.
- **Gebru et al. 2021** — *Datasheets for Datasets*. <https://arxiv.org/abs/1803.09010>.

Domain-specific pointers are cited in each section only when the source is official
(regulator, SDO, or domain-standard body) or when it is clearly labelled as a
practitioner implementation example.

## 2. Worked implementation — survey by domain

### 2.1 Healthcare NLP (clinical text, triage, coding)

- **Inputs.** Clinical notes (physician dictations, nursing notes), discharge summaries,
  patient messages, radiology / pathology reports. Typically long, English-heavy
  historically, but multilingual production deployments are increasingly common.
  Contains PHI; data-residency and de-identification constraints are first-order.
- **Outputs.** ICD / SNOMED / RxNorm codes, triage severity labels, extracted problem
  lists, summaries of a case for handoff, question answering grounded in a chart.
- **Dominant task family.** Named entity recognition + linking to ontologies;
  classification for triage; summarisation for handoff; retrieval over patient history
  for QA.
- **Ground-truth source.** Expert clinicians; inter-annotator agreement is measurable
  but expensive. Proxy labels (billing-code assignments, downstream outcomes) are used
  to scale but introduce label noise.
- **Evaluation axes.** Per-specialty slice (cardiology, oncology, etc.); per-language;
  faithfulness (did the summary introduce a diagnosis that is not in the source?);
  abstain / "unclear" rate; coding completeness.
- **Governance regime.** HIPAA (US) / GDPR Article 9 (EU "special categories"). For
  device-class software the FDA's *Software as a Medical Device (SaMD)* framework
  applies when the output drives a clinical decision — see
  <https://www.fda.gov/medical-devices/software-medical-device-samd>. The FDA's 2021
  *Good Machine Learning Practice* guiding principles co-authored with Health Canada
  and the UK MHRA are the public reference — see
  <https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles>.
- **Dominant failure modes.** Hallucinated diagnoses in summaries; privacy leakage via
  memorisation; poor performance on under-represented patient populations (slice
  regression); misalignment between proxy labels and clinical truth.
- **Rollout constraint.** Human-in-the-loop by default; shadow-only deployments are
  common for months before any user-visible change; red-team / clinical-safety review
  precedes release; the clinical evaluation is auditable and versioned.
- **Representative architecture pattern.**
  - Pre-trained biomedical encoder (e.g. one of the BioBERT / ClinicalBERT / PubMedBERT
    family — each is on the Hub with a public model card) fine-tuned on a task-specific
    labelled set, with ontology-linker post-processing against SNOMED / UMLS.
  - Summarisation: either an extractive model or a constrained-generation setup (long-
    context model, source-attribution required in the output). Pure open-ended
    generation is usually not shippable without human review.
  - Evaluation harness per specialty slice; abstention rate explicit; forensics
    capture for any output used in a clinical decision.
- **Relevant OWASP ML / ATLAS.** `ML02 Data Poisoning` (labels in a large hospital
  system are a plausible poisoning vector), `ML03 Model Inversion` (memorisation of
  PHI), `ML06 Supply Chain`, `AML.T0020 Poison Training Data`.

### 2.2 Legal NLP (contract review, discovery, legal search)

- **Inputs.** Long documents (contracts, pleadings, case law, deposition transcripts),
  often with heavy citations and footnotes; jurisdiction-tagged.
- **Outputs.** Extracted clauses, risk-flagged clause types, contract summaries,
  question answering with citation to source paragraphs.
- **Dominant task family.** Span extraction + classification (clause type / risk);
  RAG over case law and precedent; summarisation with mandatory citations.
- **Ground-truth source.** Attorneys (expensive, limited availability); historically
  often a senior-partner-led gold set; agreement is measurable but the standard is low
  (legal interpretation is itself disputed).
- **Evaluation axes.** Per-jurisdiction slice; citation faithfulness (did the generator
  cite the paragraph it quoted?); refusal / "no finding" rate; recall on risk clauses
  (recall > precision in review workflows); summary coverage.
- **Governance regime.** Legal privilege is domain-specific and jurisdiction-specific;
  data leaving a privileged corpus is often itself a privilege waiver. No single
  universal SDO analogous to FDA/SEC exists — professional-conduct rules vary. A
  hosted model that logs prompts is typically disqualified from privileged workflows
  without an airtight DPA.
- **Dominant failure modes.** Hallucinated citations (a well-documented hallucination
  pattern in generative models applied to law — the system fabricates a case name or
  cites to a non-existent paragraph); jurisdiction mismatch (US-trained model cites
  US precedent on UK contracts); stale knowledge (case law changes).
- **Rollout constraint.** Attorneys review every user-facing output in regulated
  workflows. "Co-pilot" rather than "pilot" is the default framing. RAG with mandatory
  source citation is standard.
- **Representative architecture pattern.**
  - RAG over the firm's / provider's own case corpus + statutes. Retriever on a legal-
    domain embedder; generator constrained to answer only from retrieved passages with
    explicit paragraph-level citations.
  - Extraction tasks (clause review) use a fine-tuned encoder with a strict output
    schema; the reviewer sees the extracted span highlighted in the source.
  - Faithfulness evaluation is the first-class metric; accuracy is secondary.
- **Relevant OWASP ML / ATLAS.** `ML09 Output Integrity` (hallucinated citation),
  `AML.T0051 LLM Prompt Injection` (adversarial text in a document under review),
  `ML06 Supply Chain`.
- **Outstanding citation to verify.** The numerical rate of citation hallucination in
  named legal LLMs is discussed in multiple published pieces; a verifiable headline
  number should be cited by study (with URL and date) rather than restated from memory.

### 2.3 Finance NLP (news/filings, compliance monitoring, trade-surveillance)

- **Inputs.** Earnings transcripts, 10-K / 10-Q filings, analyst notes, news feeds,
  internal communications (email, chat) for surveillance. Mix of short and long
  documents; dense numerical content embedded in text.
- **Outputs.** Sentiment / event-type classification, extracted entities (ticker,
  figure, date), structured summaries, surveillance flags.
- **Dominant task family.** Classification; event extraction; retrieval over filings;
  anomaly detection on communications.
- **Ground-truth source.** Compliance analysts for surveillance (expensive);
  programmatically-derived labels for market-moving-news tasks; downstream trading
  outcomes as proxy labels (noisy and subject to leakage).
- **Evaluation axes.** Precision-oriented for surveillance (false positives are
  expensive for compliance workflows); recall-oriented for research signals; per-
  language slice for global firms; temporal robustness (re-evaluate monthly as
  language of filings evolves).
- **Governance regime.** US: *SR 11-7 Supervisory Guidance on Model Risk Management* —
  Federal Reserve Board. See
  <https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm>. It mandates
  independent validation, documentation, ongoing monitoring, and governance; it is
  older than modern NLP but applies unchanged. SEC and FINRA rules govern market-
  conduct monitoring. EU: MiFID II / GDPR. Additionally, the EU AI Act classifies
  certain financial uses (credit scoring, creditworthiness) as high-risk.
- **Dominant failure modes.** Overfitting to a market regime (model trained in a bull
  market mis-reads a downturn); data leakage from forward-looking information into
  training labels; slice regression on non-English filings; false-positive flood in
  surveillance.
- **Rollout constraint.** Independent model validation per SR 11-7; documentation and
  challenger-model requirement; retraining has to be scheduled and approved.
- **Representative architecture pattern.**
  - Fine-tuned encoder for classification; a classical baseline (e.g. lexicon /
    logistic regression) serves as the required challenger model.
  - For surveillance: rule+model ensemble; the rules are auditable, the model boosts
    recall; review workflow with human disposition.
  - Documentation is extensive; change-management is a gated process.
- **Relevant OWASP ML / ATLAS.** `ML02 Data Poisoning` (adversarial posters in chat
  channels), `ML06 Supply Chain`, `ML09 Output Integrity`.

### 2.4 Customer support / conversational NLP

- **Inputs.** Email, chat messages, call transcripts (ASR output from mod-109). Short
  turns, multilingual, noisy (typos, abbreviations, emoji, mixed language).
- **Outputs.** Intent label, entity extraction, routing destination, canned-reply
  suggestions, draft replies.
- **Dominant task family.** Classification, extraction, response generation (increasingly
  via LLM + retrieval over knowledge base), summarisation of long chats.
- **Ground-truth source.** Support team leads for intent; agent overrides / resolution
  outcomes as feedback; user-satisfaction ratings as proxy labels.
- **Evaluation axes.** Per-language, per-tier, per-channel; agent-override rate as a
  live-traffic metric; deflection rate for self-service; CSAT as a product metric.
- **Governance regime.** GDPR + corporate AI policy. Lower-regulation domain overall
  but subject to per-company responsible-AI policy; many companies publish or enforce
  a hallucination-rate policy on customer-facing generative replies.
- **Dominant failure modes.** Hallucinated policy (generator invents a refund policy
  that does not exist); routing errors; prompt-injection from adversarial customer
  messages; misunderstanding of code-mixed languages.
- **Rollout constraint.** Progressive rollout and A/B test; agent-in-the-loop for
  generative replies; "assist, don't send" mode by default.
- **Representative architecture pattern.** (This matches the use case in exercises
  01 / 02 / 03.)
  - Multilingual encoder classifier for intent; NER + dictionary linker for products.
  - For generative drafts: RAG over the knowledge base + prompt-templated LLM + agent
    review step. Faithfulness evaluated against the knowledge base.
- **Relevant OWASP ML / ATLAS.** `ML09 Output Integrity`, `AML.T0051 Prompt Injection`
  (adversarial user messages), `ML01 Input Manipulation`.

### 2.5 Developer-tooling NLP (code completion, review, search)

- **Inputs.** Source code, issue-tracker text, PR descriptions, commit messages,
  natural-language instructions for code generation.
- **Outputs.** Code completions, bug/vulnerability flags, PR summaries, test
  suggestions, natural-language search over code.
- **Dominant task family.** Code generation (constrained to a language); classification
  (vulnerability detection); retrieval over the codebase.
- **Ground-truth source.** Accepted vs. rejected completions (accept rate is the
  default online metric); commit history; code-review comments; curated vulnerability
  datasets for security tasks.
- **Evaluation axes.** Accept rate; pass-rate on execution tests; per-language slice
  (programming language, not natural language); security-vulnerability recall.
- **Governance regime.** SOC 2 / ISO 27001 for the serving platform. For open-weights
  training, the base-model's licence governs what downstream outputs can be used for
  (e.g. training-data origin, output attribution). Internal policy on telemetry
  (can prompts be sent to the model host?) is a first-order concern.
- **Dominant failure modes.** Verbatim leakage of training-data code (licence /
  attribution exposure); hallucinated APIs that do not exist; prompt injection via
  README or doc-comments in third-party code; silent degradation on languages with
  less training data.
- **Rollout constraint.** Opt-in per repository; on-prem / self-hosted for sensitive
  codebases; usage telemetry decisions reviewed per customer.
- **Representative architecture pattern.** Open-weights code model (self-hosted or
  provider-hosted per deployment). Retrieval over the active workspace for context.
  Lightweight feedback loop on accept/reject. Strong offline eval on execution-based
  benchmarks per language.
- **Relevant OWASP ML / ATLAS.** `ML03 Model Inversion` (training-data leakage of
  proprietary code), `ML06 Supply Chain`, `AML.T0051 Prompt Injection` via code
  comments.

### 2.6 Scientific-literature NLP (search, synthesis, systematic review)

- **Inputs.** PubMed / arXiv / institutional corpus documents; abstracts and full text;
  figure captions and tables embedded in PDF.
- **Outputs.** Literature search results, extractive answers with citations,
  structured evidence tables for systematic reviews, hypothesis / claim extraction.
- **Dominant task family.** Retrieval + generation with mandatory citation;
  extraction of claims and evidence; NER for scientific entities (genes, compounds,
  methods).
- **Ground-truth source.** Expert reviewers; small gold sets curated for specific
  tasks; crowd-sourced annotations with expert review for scale.
- **Evaluation axes.** Citation faithfulness; recall on known-relevant documents for
  a query; freshness (literature changes daily); handling of retractions and
  corrections.
- **Governance regime.** Licensing of the corpus is the main lever (publisher
  contracts). Reproducibility and auditability are strong norms; outputs cited in
  published work are expected to be traceable to source.
- **Dominant failure modes.** Fabricated citations; stale answers (missing a retraction
  that happened after training); PDF parsing errors silently dropping information;
  misinterpreting a table or figure caption.
- **Rollout constraint.** Human expert review for any output used downstream; fresh
  retrieval index essential; version-pinned citation style.
- **Representative architecture pattern.** RAG with strict source attribution, built on
  an updated literature index (daily or faster). PDF ingestion is often the hardest
  engineering problem; parsing errors propagate everywhere.
- **Relevant OWASP ML / ATLAS.** `ML09 Output Integrity` (citation accuracy),
  `AML.T0051 Prompt Injection` via retrieved content, `ML06 Supply Chain`.

### 2.7 Cross-domain synthesis — what is generic, what is domain-specific

**Generic (same across domains, often implemented in `mod-110`–`mod-112`):**

- Supply-chain hygiene (pinned revisions, signed artefacts, model cards) — `ML06`
  applies everywhere.
- Pipeline composition and schema-boundary testing — generic engineering practice.
- Per-slice evaluation harness — the *dimensions* vary by domain but the harness is the
  same.
- Drift monitors for input, prediction, and performance distributions.
- Release gates + rollback + post-release monitoring loop.

**Domain-specific (do not generalise without re-examination):**

| Pattern | Healthcare | Legal | Finance | Support | Dev tooling | Science |
|---|---|---|---|---|---|---|
| Dominant architecture | Encoder + ontology linker | RAG + citation | Rules + encoder challenger | Encoder + optional LLM assist | Code-specialist model + workspace retrieval | RAG + strict citation |
| Faithfulness metric priority | High (clinical safety) | Highest (citation accuracy legally binding) | High (compliance) | Medium | High for security tasks | Highest (academic integrity) |
| Human-in-the-loop by default | Yes | Yes | Yes for surveillance | Agent-in-the-loop for generative | Developer review for all suggestions | Expert review for downstream use |
| Regulated data by default | Yes (PHI) | Yes (privilege) | Yes (MNPI, PII) | PII | Usually not; IP of codebase is | Mostly not (public literature) |
| External-API deployment allowed? | Rarely | Rarely | Sometimes, with DPA | Common with DPA | Varies per customer policy | Common |
| Dominant OWASP ML entry | ML02 / ML03 | ML09 / prompt injection | ML02 / ML06 | ML09 / prompt injection | ML03 / ML06 | ML09 / prompt injection |
| Rollout cadence | Months of shadow | Shadow + co-pilot | SR 11-7 validation cycle | Progressive A/B | Per-repo opt-in | Expert review gates |

Patterns that recur across all domains:

- **Retrieval-grounded generation becomes mandatory the moment the output must be
  auditable** (legal, science, clinical handoff). RAG is not a general-purpose tool; it
  is the specific pattern for auditable generation.
- **Human-in-the-loop defaults are the norm in high-stakes domains** and are only
  relaxed after sustained live-traffic evidence.
- **Supply-chain and data-governance controls (OWASP ML06, datasheets, model cards)
  travel unchanged across domains**; the specific threats and regulations vary.
- **"Which metric matters" is the biggest cross-domain difference**: accuracy in
  support, faithfulness in legal/science, recall in surveillance, slice fairness in
  healthcare. Picking the right metric is the single highest-leverage design choice.

## 3. Validation steps

1. **Every domain covered answers all survey dimensions.** A section that skips
   "governance regime" or "evaluation axes" is incomplete. Check with a per-dimension
   fill-rate table.
2. **Governance citations are official sources.** Per the source policy, healthcare
   cites FDA; finance cites SR 11-7. Where a specific citation is not verified, the
   section flags the gap as an outstanding citation — not an unsupported claim.
3. **No invented metrics.** Any specific numerical claim ("hallucination rate X %",
   "accept rate Y %") must cite a reproducible source. If the number is important
   enough to state, cite it; otherwise, state qualitatively ("rates are non-trivial and
   vary widely across published studies") and move on.
4. **Cross-domain synthesis actually synthesises.** The final section must identify at
   least three generic-vs-domain-specific distinctions and justify each. A survey that
   lists six domains without a synthesis is a notebook, not a survey.
5. **OWASP ML / ATLAS entries cited by identifier.** Each domain's risk entry names the
   OWASP ML entry and the relevant ATLAS technique. Per the house style, generic
   paraphrase is not acceptable.
6. **Architecture pattern is representative, not vendor-specific.** The survey
   describes patterns (encoder + ontology linker, RAG with citation, challenger model)
   rather than particular vendors.
7. **Freshness of regulatory references.** Governance regimes change; the survey cites
   the version in force at the time of writing and is dated. A reader a year later can
   see when the survey was last synced with regulation.
8. **Model-card / datasheet obligation named per domain.** Each domain's governance
   section lists the specific documentation artefacts the deployed system must carry.

## 4. Grading rubric / review checklist

| Dimension | Pass (2) | Partial (1) | Fail (0) |
|---|---|---|---|
| **Domain coverage** | At least 5 domains with distinct patterns | 3–4 domains | ≤ 2 |
| **Dimensional completeness per domain** | Inputs / outputs / tasks / labels / eval / governance / failure modes / rollout / pattern / risks | Some dimensions skipped | Narrative only |
| **Governance citations** | Official regulator or SDO per domain | Mixed quality | Blogs only |
| **OWASP ML / ATLAS tie-in** | Cited by identifier per domain | Generic mention | None |
| **Cross-domain synthesis** | ≥ 3 generic vs. domain-specific distinctions with justification | 1–2 distinctions | None |
| **Source discipline** | No invented metrics; outstanding citations flagged for follow-up where appropriate | Some unsupported numbers | Many unsupported numbers |
| **Architecture patterns** | Representative families, not vendor shootout | Mix of pattern and vendor | Vendor-only |
| **Dating / freshness** | Survey dated; regulatory references versioned | Undated | No dates |

A survey that reads as a leaderboard, quotes unverified metrics, or omits governance
citations does not meet the bar.

## 5. Common mistakes

- **Writing a vendor shootout instead of a pattern survey.** The survey is about
  architectural *families* and the governance context that forces them; the specific
  model or vendor is replaceable.
- **Quoting numerical headline metrics without citation.** "Legal LLMs hallucinate in
  X % of citations" — unless the specific study is cited, the number is invented. Use
  qualitative statements and flag outstanding citations for follow-up.
- **Over-generalising from one domain.** "All domains need RAG" is wrong; RAG is a
  pattern for tasks where the answer is in a retrievable corpus. Classification of the
  input itself does not need retrieval.
- **Ignoring governance.** The survey only differentiates meaningfully if governance is
  taken seriously. A pattern that is architecturally identical across domains is often
  operationally impossible because of the governance delta.
- **Treating evaluation as domain-agnostic.** Macro-F1 is not the universal primary.
  Faithfulness is the primary in legal and science; recall is the primary in finance
  surveillance; slice fairness is the primary in healthcare. Picking the wrong metric
  reads to a reviewer as not knowing the domain.
- **Letting the synthesis be "they are all different".** The point is the pattern of
  differences, not the differences themselves. If the survey ends with "it depends",
  it has not done the work.
- **Missing OWASP ML / ATLAS identifiers.** A risk entry that paraphrases without
  citing the identifier is not auditable; a reviewer cannot look up the control.
- **Undated regulatory references.** Regulatory text changes; an undated citation ages
  into a wrong citation. Date each section and plan a re-sync cadence.
- **Writing the survey as a blog post.** The survey is a decision aid; it should be
  structured for scanning (tables, dimensional headers), not prose.
- **Confusing "pattern" with "best model on a benchmark".** Benchmarks move monthly;
  patterns move over years. The survey is about the stable layer.

## 6. References

Primary standards and official projects used in this solution:

- NIST AI Risk Management Framework (AI RMF 1.0). <https://www.nist.gov/itl/ai-risk-management-framework>
  — cross-domain framework the survey hangs off.
- OWASP Machine Learning Security Top 10. <https://owasp.org/www-project-machine-learning-security-top-10/>
  — risk entries cited per domain.
- MITRE ATLAS. <https://atlas.mitre.org/> — adversary techniques cited per domain.

Domain-specific regulatory references cited in the survey:

- US FDA. *Software as a Medical Device (SaMD)* and *Good Machine Learning Practice for
  Medical Device Development: Guiding Principles* (co-authored with Health Canada and
  the UK MHRA). <https://www.fda.gov/medical-devices/software-medical-device-samd>
  and <https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles>.
- US Federal Reserve. *Supervisory Guidance on Model Risk Management* (SR 11-7).
  <https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm>.
- EU AI Act — <https://eur-lex.europa.eu/eli/reg/2024/1689/oj> — risk-classification
  framework; several financial and healthcare uses are "high-risk".
- EU General Data Protection Regulation (GDPR) — <https://eur-lex.europa.eu/eli/reg/2016/679/oj>
  — Article 9 "special categories" applies to healthcare data.

Supporting literature:

- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAccT.
  <https://arxiv.org/abs/1810.03993>.
- Gebru, T. et al. (2021). *Datasheets for Datasets*. CACM.
  <https://arxiv.org/abs/1803.09010>.

Cross-module references:

- Evaluation harness patterns (section 2.7 "generic" row) are implemented in
  `mod-111-nlp-evaluation`.
- Supply-chain and pipeline-composition patterns are implemented in
  `mod-112-production-nlp-pipelines`.
- Drift monitoring patterns are implemented in
  `mod-112-production-nlp-pipelines/exercise-04`.

<!-- Open items left for follow-up:
     - Outstanding citation: legal-LLM citation hallucination rates from named
       published studies, with study URLs and dates, to replace the qualitative
       discussion in section 2.2.
     - Outstanding citation: representative accept-rate and pass-rate ranges for
       named developer-tooling systems, cited to published write-ups rather than
       vendor marketing.
     - Outstanding citation: specific SaMD-classified NLP deployments as case studies
       in section 2.1 to make the governance discussion more concrete.
-->
