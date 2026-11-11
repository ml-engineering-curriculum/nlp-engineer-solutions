# exercise-05 — Transformer Family Decision Rubric

## 1. Solution overview

This is a design exercise: given a task, pick the transformer family
(encoder-only, decoder-only, encoder–decoder) and justify the choice
against task properties — not vibes. The reference solution delivers a
structured rubric plus two worked examples that exercise the rubric
end to end.

Deliverables:

1. A **decision rubric** mapping task attributes to recommended family,
   grounded in the attention patterns described by Vaswani et al.
   (2017) and the task framing from BERT (Devlin et al., 2019), T5
   (Raffel et al., 2020), and GPT-family papers (Radford et al.,
   2018/2019; Brown et al., 2020).
2. Two **worked examples** applying the rubric: (i) biomedical entity
   extraction over fixed-length abstracts; (ii) support-ticket
   summarisation with constrained style.
3. A **grading rubric** instructors can apply to student submissions.

## 2. Worked answer or implementation

### 2.1 Family primer (one paragraph each, from the primary sources)

- **Encoder-only (e.g. BERT, RoBERTa).** Bidirectional self-attention
  over every token, trained with a masked-token objective. The entire
  input is visible to every position, which is the ideal structure for
  tasks where you *consume* text and emit a label, a span, or an
  embedding (Devlin et al., 2019).
- **Decoder-only (e.g. GPT-family, LLaMA-family).** Causal self-
  attention: each position sees only earlier positions. Trained with
  next-token prediction. Natural fit for *generation*, and — via
  in-context learning and instruction tuning — also competitive on
  consumption tasks given enough scale (Radford et al., 2018/2019;
  Brown et al., 2020).
- **Encoder–decoder (e.g. T5, BART).** Bidirectional encoder feeds a
  causal decoder via cross-attention. Natural fit for *sequence-to-
  sequence* tasks where the output is derived from the input but is
  not an extraction of it (Vaswani et al., 2017; Raffel et al., 2020;
  Lewis et al., 2020).

### 2.2 Decision rubric

Use these questions in order. Each row lists what the answer implies.
When two rows disagree, follow the one closer to the top.

| # | Question | Answer → family |
|---|---|---|
| 1 | Is the output a **span or label** over the input (classification, NER, QA extractive, embedding)? | Encoder-only |
| 2 | Is the output a **free-form sequence** that is a transformation of the input (translation, summarisation, structured generation)? | Encoder–decoder (or decoder-only if scale/latency budget allows) |
| 3 | Is the output **unconstrained open-ended text** or dialogue, with the input as a prompt rather than a source document? | Decoder-only |
| 4 | Is the input **short and bidirectional context matters** (coreference, entity relations, arithmetic over the whole sentence)? | Prefer encoder or encoder–decoder |
| 5 | Is the input **long, with the output derived from both halves**? | Encoder–decoder, with a long-context encoder |
| 6 | Do you need **in-context learning / few-shot** at inference time? | Decoder-only (with scale) |
| 7 | Are **latency and footprint** tight (edge, CPU-only, sub-20ms)? | Encoder-only, small size |
| 8 | Are **training data labels** plentiful and task-specific? | Encoder or encoder–decoder fine-tune beats decoder-only prompting at similar sizes |
| 9 | Do you have **no task-specific labels** but can write good instructions? | Decoder-only with instruction tuning or prompting |
| 10 | Must the system be **auditable and reproducible** for a regulated use case? | Fine-tuned encoder or encoder–decoder is easier to pin and evaluate than a prompted decoder-only model |

**Hard constraints override the rubric.** If a procurement rule
requires a specific model family, if licencing excludes a candidate,
or if the context window needed exceeds any available model's budget,
the rubric only ranks the remaining options.

### 2.3 Worked example A — Biomedical NER on PubMed abstracts

| Factor | Observation |
|---|---|
| Output shape | BIO-tagged spans over tokens (row 1 → encoder) |
| Input length | ~250 tokens on average (short, row 4 → encoder/enc–dec) |
| Label availability | NCBI-disease, BC5CDR provide tens of thousands of labelled abstracts (row 8 → fine-tune) |
| Latency | Batch processing of nightly dumps, 100ms/doc acceptable |
| Regulated? | Downstream clinical review — reproducibility matters (row 10) |

**Decision.** Encoder-only (BERT-family), fine-tuned on the domain
with the tokenizer extended from exercise-04. Expected win: encoder
attention directly supports the span structure, and fine-tuning on
labelled abstracts beats few-shot prompting at the scale a clinical
pipeline can afford to run.

**What the solution rejects and why.**

- Decoder-only with prompting: would require careful span-extraction
  prompt engineering and post-processing, and still underperforms a
  fine-tuned encoder on the classic biomedical NER benchmarks
  according to their published leaderboards.
- Encoder–decoder: works but adds parameters and latency to emit a
  sequence that is already recoverable from token-level labels.

### 2.4 Worked example B — Support ticket summarisation with style constraints

| Factor | Observation |
|---|---|
| Output shape | Free-form summary, controlled style ("≤3 bullets, no customer names") (row 2 → enc–dec or decoder) |
| Input length | 200–3,000 tokens per ticket thread |
| Label availability | Limited — a few hundred human-written summaries |
| Latency | Interactive, 1–3s acceptable |
| Regulated? | PII must be excluded; output goes to internal users |

**Decision.** First choice: fine-tune an encoder–decoder (BART/T5
scale) on the labelled summaries with explicit style instructions in
the prompt. If the labelled set is too small to converge, fall back to
a decoder-only model with few-shot prompting and a post-hoc style
checker.

**What the solution rejects and why.**

- Encoder-only: no native way to produce free-form text.
- Decoder-only without fine-tuning at the start: feasible but harder
  to evaluate against a style spec; prefer fine-tuning while the
  labelled set is the bottleneck, not the decoder choice.

### 2.5 Template students can fill in

```markdown
# Decision record: <task name>

## Task attributes
- Output shape:
- Input length (p50, p99):
- Label availability (labelled examples, unlabelled corpus):
- Latency and footprint budget:
- Regulatory / audit constraints:
- Hard constraints (licence, procurement, hardware):

## Rubric walk-through
| Rubric row | Answer | Implication |
|---|---|---|
| 1. | | |
| 2. | | |
| ... | | |

## Decision
<family, base model candidate, fine-tuning strategy>

## Alternatives rejected
<family> — why not

## Review triggers
<what signals would make us revisit this choice>
```

## 3. Validation steps

The exercise is a design task; validation is review-based rather than
test-based. The reference solution checks:

1. **Rubric coverage.** Every row in the rubric is answered for the
   worked examples, including "N/A" where applicable. Skipped rows
   invalidate the walk-through.
2. **Primary sources cited.** Each family claim ties back to a
   canonical paper or an official project page. Vendor marketing is
   not a primary source.
3. **Rejections are present.** A decision without rejected
   alternatives is a preference, not a design. The review checks that
   at least one alternative family is explicitly considered and
   declined.
4. **Review triggers listed.** The decision record names the signals
   that would cause a re-evaluation (new labelled data, context
   window changes, latency regressions, deprecation of the base
   model).
5. **Risk and governance notes.** The record references NIST AI RMF
   functions (Govern, Map, Measure, Manage) and names at least one
   ATLAS technique or OWASP ML risk that the chosen family could be
   exposed to.

## 4. Rubric or review checklist

| Criterion | Weight | Evidence |
|---|---|---|
| Rubric is applied in order and every row answered | 20% | Walk-through table completed |
| Decision names a family, a base model class, and a training/inference strategy | 15% | Decision section |
| At least one rejected alternative with a reason tied to task attributes | 15% | Rejections section |
| Primary sources cited for each family claim | 10% | References section |
| Hard constraints enumerated and respected | 10% | Constraints section |
| Review triggers listed | 10% | Review triggers section |
| Governance tie-in: NIST AI RMF function mapping and OWASP / ATLAS risk pointer | 10% | Risk section |
| Worked example shows numerical or categorical task inputs, not generic prose | 10% | Task attributes section |

## 5. Common mistakes

- **Choosing family based on familiarity.** "We use BERT for
  everything" or "GPT can do anything" are not decisions. The rubric
  is there to force at least one alternative to be weighed.
- **Confusing scale with family.** A larger decoder-only model is
  often competitive with a smaller fine-tuned encoder on consumption
  tasks, but only at scales most teams cannot afford to serve. Decide
  at the size and budget that will actually ship.
- **Ignoring context window.** An encoder that can only see 512 tokens
  is a bad fit for a task whose documents are 10k tokens, no matter
  how good its accuracy is on short inputs. The rubric row 5 catches
  this.
- **Picking encoder–decoder because the task "sounds like
  translation".** Many tasks that read like seq2seq (classification
  into a canonical label, structured extraction) are better served by
  an encoder that emits a structured prediction.
- **Over-indexing on leaderboards.** Leaderboard winners are selected
  under constraints (no latency budget, unlimited compute) that rarely
  match deployment. The rubric puts latency/footprint explicitly in
  the decision.
- **No review triggers.** A decision without a "when to revisit" line
  becomes load-bearing permanently and resists improvement.

## 6. References

- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
  arXiv:1706.03762. Introduces the transformer encoder, decoder, and
  encoder–decoder.
- Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. (2019). *BERT:
  Pre-training of Deep Bidirectional Transformers for Language
  Understanding.* NAACL 2019. arXiv:1810.04805. Encoder-only family.
- Radford, A. et al. (2018). *Improving Language Understanding by
  Generative Pre-Training.* OpenAI technical report. GPT-1.
- Radford, A. et al. (2019). *Language Models are Unsupervised
  Multitask Learners.* OpenAI technical report. GPT-2.
- Brown, T. B. et al. (2020). *Language Models are Few-Shot Learners.*
  NeurIPS 2020. arXiv:2005.14165. GPT-3 / decoder-only few-shot.
- Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning
  with a Unified Text-to-Text Transformer.* JMLR 21. arXiv:1910.10683.
  T5 / encoder–decoder.
- Lewis, M. et al. (2020). *BART: Denoising Sequence-to-Sequence
  Pre-training for Natural Language Generation, Translation, and
  Comprehension.* ACL 2020. arXiv:1910.13461.
- NIST AI Risk Management Framework (AI RMF 1.0)
  (`https://www.nist.gov/itl/ai-risk-management-framework`). Govern /
  Map / Measure / Manage functions as the frame for the decision
  record's risk section.
- OWASP Machine Learning Security Top 10
  (`https://owasp.org/www-project-machine-learning-security-top-10/`).
  Model-selection choices shift the risk surface (e.g. prompt
  injection on decoder-only is a different risk from adversarial
  input on an encoder classifier).
- MITRE ATLAS (`https://atlas.mitre.org/`). Use for mapping the
  chosen family's exposure to adversarial ML tactics.
