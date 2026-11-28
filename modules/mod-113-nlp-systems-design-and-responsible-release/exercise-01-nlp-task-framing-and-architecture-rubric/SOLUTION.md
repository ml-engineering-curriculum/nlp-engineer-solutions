# exercise-01 — NLP Task Framing and Architecture Rubric

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-113-nlp-systems-design-and-responsible-release/exercises/exercise-01-nlp-task-framing-and-architecture-rubric.md`.
> The target learning outcomes implied by that filename: translate a vague product ask
> ("make our docs searchable", "summarise these calls") into a **formal NLP task**
> (inputs, outputs, labels, metric, slices, release gates), pick a defensible **architecture
> family** (retrieval, extractive, generative, hybrid), and ship a **rubric** a reviewer can
> use to grade the framing before any model is trained or any vendor contract is signed.

## 1. Solution overview

Most NLP projects fail at framing, not at modelling. By the time a team is debating LLM
provider vs. open-weights fine-tune, the real decision — "what exactly is the system
supposed to output, and how will we know it is right" — has already been skipped. The
worked artefact for this exercise is a two-page **task brief** paired with the **rubric**
a reviewer applies to the brief.

The brief has to answer, in order:

| Section | Must contain | Common failure |
|---|---|---|
| **1. Problem statement** | The user, the trigger event, and the decision the user makes with the output | "We want a chatbot" — no user, no decision |
| **2. Inputs** | Modality, language mix, length distribution, source, PII / regulated-data status | Text-only assumption; later discovery that 30 % of inputs are PDFs |
| **3. Outputs** | Type (class label, span, score, free text, structured JSON); downstream consumer; refusal / abstain behaviour | Free-form text output with no schema; downstream team parses with regex |
| **4. Labels and ground truth** | Who labels, inter-annotator agreement target, how disagreements resolve, cost per label | "We'll figure labels out later"; 10 k rows arrive before the guideline is written |
| **5. Primary metric and slices** | One primary metric, two to four secondary metrics, slice definition (language, domain, length, demographic) | Average accuracy on a flat test set; subgroup regressions invisible |
| **6. Architecture choice** | Retrieval / extractive / generative / hybrid, and *why* the inputs and outputs make the others wrong | "LLM because LLM"; no comparison to the rejected options |
| **7. Risk surface** | OWASP ML Top 10 / MITRE ATLAS techniques in scope; NIST AI RMF Map category | Risk section copy-pasted from a template; no concrete threats |
| **8. Release gates** | The pass/fail numbers on primary + slices required to ship | Gates are "LGTM in review"; no numeric commitment |
| **9. Rollback plan** | What "roll back" means for this system (previous model, fall back to keyword search, disable the feature) | No rollback plan; rollback is "deploy old container, hope" |

A rubric grader does not have to be a modelling expert; the rubric drives on the
*presence and specificity* of each section, not on whether the modelling choice is the
one the grader would have made. Two defensible architectures with full briefs both pass;
one "LLM everywhere" brief that cannot answer section 2 or 5 fails.

Authoritative sources this solution draws on:

- **NIST AI Risk Management Framework (AI RMF 1.0)** — <https://www.nist.gov/itl/ai-risk-management-framework>.
  The *Map*, *Measure*, *Manage*, *Govern* functions anchor the brief sections: the brief
  is the *Map* deliverable, the metric + slices are *Measure*, the release gates and
  rollback plan are *Manage*, and the sign-off step is *Govern*.
- **OWASP Machine Learning Security Top 10** — <https://owasp.org/www-project-machine-learning-security-top-10/>.
  Risk surface section cites concrete entries (ML01 input manipulation, ML02 data
  poisoning, ML03 model inversion, ML06 supply chain, ML09 output integrity) so the risk
  review is not generic.
- **MITRE ATLAS** — <https://atlas.mitre.org/>. The attacker-centric framing: `AML.T0051
  LLM Prompt Injection`, `AML.T0010 ML Supply Chain Compromise`, `AML.T0043 Craft
  Adversarial Data` give the adversary's playbook, which the brief's "risk surface"
  section must cover for the specific input modality.
- **Mitchell et al. 2019** — *Model Cards for Model Reporting*, FAccT.
  <https://arxiv.org/abs/1810.03993>. The brief's "intended use", "out-of-scope use", and
  "evaluation" sections are the model-card fields pulled forward to design time.
- **Gebru et al. 2021** — *Datasheets for Datasets*, CACM. <https://arxiv.org/abs/1803.09010>.
  The "inputs" and "labels" sections are datasheet fields pulled forward to design time.

## 2. Worked implementation — a filled task brief + rubric

The worked example frames a real-sounding product ask: *"Reduce the time our support
agents spend reading inbound customer emails before replying."* The brief below is what a
passing submission looks like.

### 2.1 Filled task brief — **Email triage classifier + entity summary**

**1. Problem statement.** A customer support agent receives ~200 emails per shift.
Before replying, the agent reads the email, identifies the *customer*, the *product
mentioned*, and the *intent* (refund / technical issue / billing / sales / other), then
routes to a specialist queue. The system produces that identification automatically so
the agent spends shift time replying, not routing. The decision the agent makes with the
output is: *accept the routing suggestion, or override*.

**2. Inputs.**

- Modality: plain-text email bodies (HTML stripped upstream); subject line separate.
- Language mix: 78 % en, 14 % es, 6 % fr, 2 % other (from Q3 traffic sample).
- Length: p50 = 112 words, p95 = 740 words, max observed 6 k words.
- Source: inbound support mailbox; already PII-filtered to *remove card numbers* but
  retains names, addresses, email, phone, order id.
- Regulated-data status: contains PII (GDPR); treated as personal data in the data
  classification register. No PHI.

**3. Outputs.** One JSON object per email with:

```json
{
  "message_id": "…",
  "intent": "refund | technical | billing | sales | other",
  "intent_confidence": 0.0,
  "customer": {"name": "…", "account_id": "… or null"},
  "products_mentioned": [{"span": "…", "normalised": "sku-1234"}],
  "abstain_reason": null
}
```

Abstain behaviour: `abstain_reason` is set, intent falls back to `"other"`, and a human
reviewer gets the email first, when:
- `intent_confidence < 0.60`, or
- language is not in {en, es, fr}, or
- length > 2 000 words (pipeline input cap).

The *downstream consumer* is the support-queue router; the contract is this JSON schema,
version-pinned, with additive-only changes.

**4. Labels and ground truth.**

- Label source: two support team leads label a 2 000-email sample over four weeks;
  disagreements resolved by a third lead. Cohen's κ target ≥ 0.75 on intent.
- Guideline: short label guide covers "ambiguous intent → pick the most *actionable*
  one"; explicitly carves refund-with-technical-issue to `refund`.
- Cost per labelled email: ≈ 3 min at loaded cost; budgeted.
- Drift: a weekly sample of 100 emails is re-labelled on an ongoing basis to detect
  label drift (support policy change → intent definitions shift).

**5. Primary metric and slices.**

- Primary: macro-F1 on intent, computed on a stratified 400-email test set held out
  from the labelling exercise.
- Secondary: (a) confidence-stratified precision at `confidence ≥ 0.6`;
  (b) entity-extraction span F1 on `products_mentioned`; (c) abstain rate.
- Slices (each reported separately; each gated):
  - Language: en / es / fr.
  - Length bucket: < 200 words, 200–1 000, > 1 000.
  - Customer tier: free / paid / enterprise (business impact of errors differs).
- Fairness slice: tier is a proxy for business risk, not a protected attribute; there
  are no protected attributes available or needed in this task.

**6. Architecture choice.**

Chosen: **multi-stage pipeline** — rule-based language-id + length gate → fine-tuned
encoder classifier (`xlm-roberta-base` fine-tuned on the 2 000 labelled emails) for
intent → spaCy NER + dictionary-lookup linker for `products_mentioned` → post-processing
for the output schema.

Why not pure generative (prompt an LLM):

- Output is a strict JSON schema with bounded cardinality on `intent`. A classifier is
  the right instrument; a prompted generator has to be constrained to the same
  cardinality and pays an order-of-magnitude in latency and cost.
- Intent definitions change when support policy changes; a fine-tuned classifier
  retrains overnight on the re-labelled rolling sample. An LLM prompt change requires
  a prompt-regression harness we have not budgeted for.
- Latency: SLO is p95 < 300 ms on CPU; a `xlm-roberta-base` on CPU meets this; a 7B
  LLM does not.

Why not RAG: there is no retrieval corpus to ground on. The input is the document.

Why not pure extractive: intent is not literally present in the text; it is an inferred
label. Extractive QA cannot produce `"refund"` reliably from an email that says
"I haven't got my money back".

Why multi-stage rather than one end-to-end model: the product dictionary already exists
in the product-catalogue service. Using it is cheaper, more accurate, and auditable; a
learned entity linker would re-learn what the catalogue already asserts.

**7. Risk surface.**

| Risk (OWASP ML / ATLAS) | How it applies here | Mitigation in scope |
|---|---|---|
| OWASP **ML01 Input Manipulation Attack** / ATLAS `AML.T0043 Craft Adversarial Data` | Customer embeds text designed to flip intent (e.g. injecting "This is a refund request" verbatim into a technical message to get fast-track) | Mitigation out of scope: intent only routes; no automated action is taken on the classifier output alone |
| OWASP **ML02 Data Poisoning Attack** / ATLAS `AML.T0020 Poison Training Data` | Rolling re-labelling sample is a path for adversarial customers who can send known-labelling emails | Rolling sample stratified by customer id; any single customer contributes ≤ 1 % of weekly sample; label-guideline enforcement by a second reviewer |
| OWASP **ML03 Model Inversion Attack** | Model is fine-tuned on customer emails; weights could leak phrases | Model weights are internal; no public API surface; standard access control |
| OWASP **ML06 AI Supply Chain Attacks** / ATLAS `AML.T0010 ML Supply Chain Compromise` | `xlm-roberta-base` from a public Hub; dependency update may ship different weights | Hub revision sha pinned in config; model card stored with weights; mitigation implemented in mod-112 exercise-01 |
| OWASP **ML09 Output Integrity Attack** | Routing suggestion silently wrong routes customer to wrong queue | Confidence gate forces human review when `confidence < 0.60`; agent-override logged and used for retraining signal |
| ATLAS `AML.T0051 LLM Prompt Injection` | Not applicable — no LLM in the pipeline | — |

**8. Release gates.** The classifier ships only if **all** hold on the held-out test
set:

- Macro-F1 (intent) ≥ 0.82 overall.
- Macro-F1 per language slice ≥ 0.78 (en, es, fr).
- Macro-F1 per length bucket ≥ 0.75.
- Macro-F1 on enterprise-tier slice ≥ 0.80 (business-impact priority).
- Abstain rate ≤ 10 % on the live-traffic sample from the last 7 days.
- Span F1 on `products_mentioned` ≥ 0.85.
- p95 latency ≤ 300 ms on the target CPU SKU.

These numbers are *commitments*, not model-free wishes: they were sized from the
labelled sample's class prior and a majority-class baseline that already achieves
macro-F1 ≈ 0.42. Any slice that falls below its gate at release time blocks the
release; it is not overridden by a strong overall number.

**9. Rollback plan.** Three rollback steps, each with a trigger and an owner:

- **Trigger A** — macro-F1 on last 1 000 agent-overrides drops > 5 pp from release.
  **Action** — redeploy the prior model-card revision via the CD pipeline. **Owner** —
  on-call ML engineer. **Expected restore time** — ≤ 10 minutes.
- **Trigger B** — latency SLO breached for > 15 minutes. **Action** — flip feature flag
  `email_triage_classifier_enabled = false`; UI falls back to agent-only routing (the
  pre-release workflow). **Owner** — on-call SRE. **Expected restore time** — ≤ 5 minutes.
- **Trigger C** — security incident (model-supply or prompt-injection-style). **Action**
  — same as Trigger B plus freeze the retraining pipeline until root cause is known.
  **Owner** — security on-call + ML lead.

**Sign-off**: product manager, ML lead, security reviewer. Revisit the brief on any
change to intent cardinality, input language distribution shift, or SLO.

### 2.2 Decision rationale — what a reviewer should understand

The brief is defensible because each section ties to an authoritative framework:

- Sections 1–3 are the NIST AI RMF **Map 1 / Map 3** functions: *context, categorisation
  of the AI system*. The explicit input modality and output schema are the context-
  bounding artefacts.
- Sections 4–5 are **Measure 2**: *metrics and performance evaluation*. Per-slice gates
  are the Measure 2.11 anti-spec-gaming control.
- Section 7 is **Map 5** (*impacts* / *risk*). Citing OWASP ML + ATLAS by identifier,
  not by paraphrase, is the auditable artefact.
- Sections 8–9 are **Manage 1 / Manage 4**: *risk prioritisation + response*. The
  rollback plan is the explicit *response* that distinguishes a managed system from an
  unmanaged one.

A "fail" grade is reserved for briefs that cannot answer sections 2, 5, or 8. Those
three sections are the minimum to run a pre-launch review at all.

### 2.3 Rubric — graded on *specificity*, not on *choice*

The rubric is applied per section (0–2 points each; 0 = missing, 1 = present but
generic, 2 = specific and testable). Total out of 18. ≥ 14 passes; 9–13 revise and
resubmit; ≤ 8 restart the brief.

| Section | 0 (missing) | 1 (generic) | 2 (specific and testable) |
|---|---|---|---|
| 1. Problem statement | No user, no decision | "Help users with X" | Named user, named trigger, named decision taken on the output |
| 2. Inputs | Not described | "Text emails" | Modality, language mix, length distribution, source, regulated-data classification named |
| 3. Outputs | Not described | "Returns the intent" | Explicit schema with field types, abstain behaviour and triggers, downstream consumer named |
| 4. Labels and ground truth | Not addressed | "Humans will label" | Labeller identity, κ target, disagreement resolution, cost/label, drift sampling plan |
| 5. Primary metric and slices | Not stated | "Accuracy" | One primary, named secondaries, slice axes enumerated, gate per slice |
| 6. Architecture choice | Not compared | "Chose LLM" | Chosen family + explicit rejection of the other three with cited constraint |
| 7. Risk surface | Not addressed | "OWASP applies" | OWASP ML / ATLAS entries cited by identifier, application to this system named, mitigation in scope |
| 8. Release gates | Not stated | "Must be good" | Numeric gates per primary and per slice; abstain / latency gates named |
| 9. Rollback plan | Not addressed | "We can roll back" | Named triggers, actions, owners, expected restore time |

A reviewer who scores section 7 as 2/2 should be able to open OWASP ML Top 10 and
MITRE ATLAS, find the cited identifier, and verify the application is accurate. Section
8 being 2/2 means another engineer can run the test suite and tell you pass/fail without
any further judgement call.

## 3. Validation steps

1. **Independent-reviewer dry run.** Hand the brief to an engineer who has not seen the
   product. They should be able to answer, unaided: *who uses this*, *what goes in*,
   *what comes out*, *what the primary metric is*, *what blocks a release*. If any of
   the five questions requires asking the author, that section fails the rubric.
2. **Metric-to-gate consistency.** For every metric mentioned in section 5, there must
   be a numeric gate in section 8. For every gate in section 8, there must be a metric
   in section 5. A unilateral entry on either side is a drafting error.
3. **Risk-to-mitigation consistency.** For each OWASP ML / ATLAS entry cited in section
   7 that is marked "in scope", section 8 or 9 must reference the mitigation. "ML06
   pinning" without the pinning being in a release gate or a rollback trigger is
   decorative.
4. **Output-schema freeze test.** Write one mock downstream consumer that parses the
   section 3 schema. If the schema cannot be parsed (missing field type, free-text
   fields where structured fields are required), section 3 is a 1 not a 2.
5. **Reject-the-alternatives test.** Each alternative architecture listed in section 6
   must be rejected on a *constraint specific to this problem* (latency, labels,
   retrieval corpus, output schema). "Because LLMs are expensive" is generic and scores
   1; "Because the p95 latency SLO is 300 ms on CPU and a 7B LLM does not meet it"
   scores 2.
6. **Drift / re-label plan is executable.** Section 4's rolling re-label plan should
   have a named budget (e.g. "100 emails / week") and a named owner. "We will monitor
   for drift" is a 1.
7. **Rollback drill.** Section 9 should pass a tabletop drill: an SRE reads the brief,
   simulates each trigger, and confirms the named owner and action are reachable in
   the quoted restore time.
8. **Sign-off trail.** The brief has to name the human sign-offs (PM, ML lead, security
   reviewer at minimum). A brief signed only by the author is not a NIST AI RMF
   *Govern* artefact; it is a design doc.

## 4. Grading rubric / review checklist

| Dimension | Pass (2) | Partial (1) | Fail (0) |
|---|---|---|---|
| **Problem statement** | Named user, trigger, decision | User and task present, decision implicit | Generic product goal |
| **Inputs** | Modality, language mix, length, source, PII status | Modality + one of the rest | "Text" |
| **Outputs** | Schema + abstain + consumer | Schema only | Free-form |
| **Labels** | Labeller, κ, resolution, cost, drift sampling | Two of five | Not addressed |
| **Metric & slices** | Primary, secondaries, slice axes, per-slice gates | Primary + slices without gates | "Accuracy" |
| **Architecture** | Chosen family + specific rejection of each alternative | Choice made, no comparison | Not addressed |
| **Risk surface** | OWASP ML / ATLAS / NIST entries cited by identifier, applied, mitigation in scope | Risks listed generically | None |
| **Release gates** | Numeric gates per metric and per slice; abstain / latency gates | Numeric gates on overall only | "LGTM" |
| **Rollback** | Triggers, actions, owners, restore time | Triggers only | None |

A brief scoring < 14/18 is not allowed to proceed to modelling. The rubric is the gate.

## 5. Common mistakes

- **Choosing architecture before writing section 3.** Picking "LLM" before the output
  schema is defined guarantees the pipeline will be over-general and the evaluation
  harness will be under-specified. The schema constrains the architecture, not the other
  way around.
- **One flat accuracy number as the primary metric.** Macro-F1 (or ROC-AUC, or
  calibrated precision) with explicit slice decomposition is the minimum. A primary
  metric that cannot be broken down by slice cannot drive a per-slice release gate.
- **Omitting the abstain / refusal behaviour.** Every production NLP system needs an
  explicit answer to "what does the system do when it is not confident". A section 3
  without that is incomplete even if the schema is detailed.
- **Risk section as framework copy-paste.** Writing "OWASP ML Top 10 applies" is not
  risk analysis. Each applicable entry must be named by identifier and tied to a
  concrete mitigation or an in-scope accepted risk.
- **Treating slices as fairness-only.** Length, language, and customer tier are
  slices that matter for *reliability*, not just fairness. A brief that only reports
  fairness slices and ignores reliability slices misses the dominant failure mode.
- **Release gates set to "whatever the model achieves".** The gates must be set from
  product requirements and compared to a majority-class or retrieval baseline, *before*
  the final model is trained. Setting gates after the fact invalidates them.
- **Rollback plan as "deploy the old container".** The three concrete triggers (quality,
  SLO, security) each need an action with an expected restore time. "Roll back the
  deploy" is not a plan for quality drift that was introduced two weeks ago.
- **Skipping sign-off.** NIST AI RMF `Govern` requires named human accountability. An
  author-only sign-off is not the governance artefact.
- **Reusing another product's brief verbatim.** Sections 2, 4, and 7 are inputs-specific;
  reuse here means the risk analysis is wrong.
- **Framing the brief in terms of the model, not the user.** "Train a classifier on
  emails" is not a problem statement; "support agents spend 40 % of shift time routing"
  is. Model framings skip over the fallback-to-no-ML option.

## 6. References

Primary standards and official projects used in this solution:

- NIST AI Risk Management Framework (AI RMF 1.0). <https://www.nist.gov/itl/ai-risk-management-framework>
  — the `Map`, `Measure`, `Manage`, `Govern` framework the brief sections map to.
- OWASP Machine Learning Security Top 10. <https://owasp.org/www-project-machine-learning-security-top-10/>
  — cited by identifier in section 7 (`ML01`, `ML02`, `ML03`, `ML06`, `ML09`).
- MITRE ATLAS. <https://atlas.mitre.org/> — adversary techniques (`AML.T0010`,
  `AML.T0020`, `AML.T0043`, `AML.T0051`) cited in section 7.

Supporting literature:

- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAccT.
  <https://arxiv.org/abs/1810.03993> — the intended-use, out-of-scope-use, and
  evaluation fields pulled forward to design time.
- Gebru, T. et al. (2021). *Datasheets for Datasets*. CACM.
  <https://arxiv.org/abs/1803.09010> — the input- and label-provenance fields pulled
  forward to design time.

Cross-module references:

- Supply-chain pinning (section 7 `ML06`) is implemented in `mod-112-production-nlp-pipelines/exercise-01`.
- Per-slice evaluation (section 5) is implemented in `mod-111-nlp-evaluation`.
- Drift monitoring (section 9 Trigger A) is implemented in `mod-112-production-nlp-pipelines/exercise-04`.
