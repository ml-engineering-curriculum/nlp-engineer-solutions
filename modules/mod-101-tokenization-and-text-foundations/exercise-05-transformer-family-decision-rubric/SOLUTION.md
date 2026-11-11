# exercise-05 — Transformer Family Decision Rubric

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-101-tokenization-and-text-foundations/exercises/exercise-05-transformer-family-decision-rubric.md`.
> This solution targets the learning goals that filename implies: produce a reusable rubric for choosing among the three canonical Transformer architectures — **encoder-only**, **decoder-only**, and **encoder-decoder** — for a given NLP task, and demonstrate the rubric on worked examples.

## 1. Solution overview

Choosing an architecture is choosing an inductive bias about the *shape of the task*, not an opinion about which family is "best":

| Family | Attention pattern | Typical pre-training | Natural tasks | Representative models |
|---|---|---|---|---|
| **Encoder-only** | Bidirectional; every position attends to every other | Masked LM | Classification, span extraction, representation / retrieval | BERT, RoBERTa, DeBERTa |
| **Decoder-only** | Causal (left-to-right only) | Next-token prediction | Open-ended generation, few-shot in-context learning, instruction following | GPT family, Llama, Mistral |
| **Encoder-decoder** | Encoder bidirectional, decoder causal with cross-attention to encoder | Denoising / seq2seq | Conditional generation where the conditioning text is substantial (translation, summarisation) | T5, BART, mT5, Flan-T5 |

The exercise is to convert this table into a decision rubric that handles messy real-world asks — "my team wants to do both classification and generation", "latency budget is 50 ms per request", "we have 10 k labelled examples, 0 unlabelled" — and to show it working on a handful of concrete scenarios.

## 2. Worked answer / implementation — the rubric

### 2.1 Decision flow

Walk a candidate task through these questions in order. The first definitive answer wins; otherwise fall through to the tiebreaks in §2.3.

1. **Does the task produce free-form text as output?**
   - **No** (label, span, score, embedding) → **encoder-only** is the default.
   - **Yes** → go to (2).

2. **Does the output primarily depend on a long, *substantive* input text (translation, summarisation, structured QA with long context, data-to-text where the data is the conditioning signal)?**
   - **Yes** → **encoder-decoder** is the natural fit. The bidirectional encoder gives a strong representation of the source; cross-attention gives the decoder targeted access to it.
   - **No** (open-ended generation, chat, code, instruction following, in-context learning) → go to (3).

3. **Is the generation open-ended, conversational, or driven by prompts rather than a specific source document?**
   - **Yes** → **decoder-only** is the default; it is what instruction-tuned LLMs are optimised for and what the serving ecosystem assumes.

### 2.2 Scoring dimensions

For a quantitative rubric, score each candidate family on these axes (1 = poor fit, 5 = excellent fit). Pick the highest total; tie-break on the constraints in §2.3.

| Dimension | What to assess | How to score it |
|---|---|---|
| **Task shape fit** | Does the family's inductive bias match the task? | From the flow in §2.1 |
| **Pre-trained availability** | Is there a strong open-weights model in this family for the required language(s) and licence? | 5 if multiple SOTA options; 1 if nothing comparable |
| **Data regime** | Supervised label volume vs. need to leverage pre-training | Encoders fine-tune well with 1k-10k labels; decoders can zero-/few-shot but fine-tuning data needs can be larger |
| **Latency & cost** | Autoregressive generation is O(output_len) decoder calls; encoder classification is one forward pass | Favour encoder for tight-latency classification; favour small decoder with KV cache for short generation |
| **Interpretability** | Do you need to return input-aligned spans, attributions, offsets? | Encoders make this easy; decoders make it awkward |
| **Multilinguality / length** | Does the task need long-context or many languages? | Check existing checkpoints; context-window varies widely |
| **Serving ecosystem** | Does your serving stack (vLLM, TGI, Triton, ONNX) support this family well? | Decoder-only has the most momentum today; encoder serving is cheap but less exciting |

### 2.3 Tie-breaks

- **"We need both classification and generation."** Build two heads on one decoder-only model (prompt-the-label) if sample volume is low and latency tolerable, OR deploy an encoder for the classification path and a decoder for the generation path — pay the ops cost. Do *not* ship an encoder-decoder just because the Venn diagram is pleasing; it usually means two training regimes you cannot staff.
- **"Latency ≤ 50 ms, high QPS, classification."** Encoder-only (DistilBERT / MiniLM / DeBERTa-v3-small). Decoder-only models are overkill and their generation path is wasted.
- **"We need faithful conditional generation on long documents."** Encoder-decoder. T5 / BART / Flan-T5 carry a stronger inductive bias for faithfulness than a decoder-only model prompted the same way, and the cross-attention pattern makes abstraction vs. extraction tuning cleaner.
- **"We want in-context learning / instruction following."** Decoder-only. The entire instruction-tuning literature (and the serving ecosystem) is built on this family.
- **"The customer will self-host and wants smallest possible footprint."** Consider encoder-only distilled models for classification; consider small decoder-only models (3B-7B class) for generation.

### 2.4 Worked examples

Each example walks the rubric and names a representative model class, not a specific checkpoint (which rots).

**Example A — Toxicity classifier for user-generated comments, 50 ms SLA.**
- Flow: free-form output? No → encoder-only.
- Scoring: encoder-only wins on every axis; distilled encoder clinches cost and latency.
- Decision: **encoder-only** (DeBERTa-v3-small or DistilBERT class). Fine-tune on labels; return logits.

**Example B — Summarise 5-page legal filings into 200-word executive summaries.**
- Flow: free-form output? Yes → does it depend on a substantive input doc? Yes → encoder-decoder is the natural fit.
- Scoring: encoder-decoder wins on task-shape fit and faithfulness. A decoder-only ≥7B instruction-tuned model is a credible runner-up if you need very broad generality, but at higher cost.
- Decision: **encoder-decoder** (Flan-T5 / BART class). Alternative: decoder-only ≥7B, if the serving stack and cost allow and the summary style is best controlled by prompting.

**Example C — Internal chatbot with retrieval-augmented answers.**
- Flow: free-form output? Yes → depends on a single source doc? No, it's prompt + retrieved passages → decoder-only.
- Scoring: decoder-only wins on instruction-following and ecosystem. Encoder-only is still useful *as the retriever* (dense bi-encoder), which is a reminder the rubric applies per-component, not per-product.
- Decision: **decoder-only** generator + **encoder-only** retriever.

**Example D — Named-entity recognition with returned character spans.**
- Flow: free-form output? No (labels + spans) → encoder-only.
- Scoring: encoder-only wins on interpretability (span heads sit naturally on encoder outputs). Decoder-only "generate the annotated string" works but fights the architecture.
- Decision: **encoder-only** (BERT / DeBERTa class).

**Example E — Few-shot structured extraction where labels are scarce (≤100 examples) and schemas evolve weekly.**
- Flow: free-form output? Yes (structured JSON counts as generation when schemas vary) → depends on substantive source doc? Sometimes → ambiguous.
- Tiebreak: evolving schemas and tiny label counts favour the family that excels in-context — **decoder-only** with careful prompting and output validation. Encoder-decoder is the alternative if the input is always a single long document and the schema is stable within a weekly cycle.
- Decision: **decoder-only** with constrained decoding or JSON-schema validation.

**Example F — Machine translation, EN↔ZH, published checkpoints available in both families.**
- Flow: free-form output? Yes → depends on substantive source? Yes → encoder-decoder is the classical fit.
- Nuance: modern decoder-only LLMs now rival or exceed classical MT systems in many pairs. Decide by *evaluation on your own test set*, not by architectural purity. Keep encoder-decoder as the default *starting point* given the stronger inductive bias for faithful conditional generation.
- Decision: **encoder-decoder** as default; benchmark a decoder-only LLM alongside.

## 3. Validation steps

A good rubric is not just internally consistent — it must survive contact with real cases. Validate yours by:

1. **Replay historical decisions.** Take 3-5 architecture decisions your team or the published literature has already made (e.g., "BERT for GLUE", "T5 for SuperGLUE / summarisation", "GPT-4 for instruction following") and confirm the rubric points the same way. If it disagrees, interrogate which side is wrong.
2. **Adversarial scenarios.** Write 2-3 edge cases that intentionally stress the rubric: "classification with 2 labels but an SLA of 500 ms", "generation that must return spans", "encoder-decoder vs decoder-only for 10-page summarisation". Document how the rubric resolves them (or where it punts to measurement).
3. **Rubric stability across reviewers.** Hand the rubric and 3 scenarios to two peers independently; disagreement on the final family answer flags under-specified axes.
4. **Constraint-change sensitivity.** For one scenario, flip exactly one constraint (latency, labelled data, licence) and check that the rubric's answer changes in a direction that makes sense.
5. **Document the escape hatch.** For every "default", state the measurement that would change it. (E.g., *"Default encoder-decoder for MT; switch to decoder-only if BLEU/COMET on our held-out set differ by less than one point and inference cost is lower."*)

## 4. Rubric / review checklist for the student's rubric

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Task-shape primary axis** | Explicit yes/no question about free-form output and source-document grounding | One of the two | Not represented |
| **Secondary axes** | At least 4 of: pre-trained availability, data regime, latency, interpretability, multilinguality, serving ecosystem | 2-3 | 0-1 |
| **Tiebreaks** | Named for at least 3 common conflicts (classification+generation, tight latency, long-context faithfulness) | 1-2 | None |
| **Worked examples** | ≥4 scenarios walked through, with at least one that resolves to each family | 2-3 scenarios | 0-1 |
| **Measurement escape hatch** | For each "default", a metric that would overturn it | Mentioned only | Dogmatic |
| **Responsible deployment context** | Mentions licence, provenance, and harm considerations for whichever family is chosen (NIST AI RMF Govern/Map; OWASP ML Top 10) | Mentioned once | Ignored |
| **Readability** | Decision flow diagrammed or enumerated; family / model separation clear | Flow present but confusing | Prose only |

## 5. Common mistakes

- **"Decoder-only can do everything, so always pick it."** It can — at a cost. For classification at scale, a 110M-parameter encoder is 10-100× cheaper per request and more interpretable. The rubric must weigh cost and SLA, not just capability.
- **Confusing *family* with *specific model*.** The rubric picks a family. Model selection (`bert-base-uncased` vs DeBERTa-v3-base vs RoBERTa-large) is a *next-step* decision driven by language coverage, licence, and benchmarks. Mixing the two produces unreadable rubrics.
- **Ignoring the retriever.** Many "generation" products are retrieval-augmented, which means a product uses *multiple* families (encoder bi-encoder for retrieval, decoder for generation). The rubric should apply per-component.
- **Over-weighting trendiness.** Decoder-only LLMs dominate headlines; encoder-only models dominate everything that ships to a browser extension or a mobile device. Default to the family that fits the *task*, not the family on the latest leaderboard.
- **No measurement escape hatch.** A rubric that never admits "run the experiment" is a philosophy, not a rubric. Every default must name the metric that would overturn it.
- **Omitting responsible-deployment context.** Architecture choice interacts with safety and licence constraints (e.g., instruction-tuned decoder-only models inherit known jailbreak surfaces per MITRE ATLAS; data provenance for both decoder and encoder pre-training surfaces for NIST AI RMF). A rubric that does not touch these is incomplete.

## 6. References

Primary sources used in this solution:

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. <https://arxiv.org/abs/1706.03762> — original Transformer (encoder-decoder).
- Devlin, J. et al. (2019). *BERT*. NAACL. <https://aclanthology.org/N19-1423/> — encoder-only, masked LM pre-training.
- Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2). <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf> — decoder-only, next-token pre-training.
- Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5). JMLR. <https://arxiv.org/abs/1910.10683> — encoder-decoder, text-to-text framing.
- Lewis, M. et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training*. ACL. <https://aclanthology.org/2020.acl-main.703/> — encoder-decoder for conditional generation.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — risks that vary by architecture (prompt injection on decoder-only; supply-chain and model-poisoning on any pre-trained).
- MITRE ATLAS — <https://atlas.mitre.org/> — adversarial techniques catalogued across model families; jailbreak and prompt-injection techniques map predominantly onto decoder-only instruction-tuned deployments.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — Govern / Map / Measure / Manage functions; architecture choice is a Map-function decision whose risks must be Measured and Managed downstream.
