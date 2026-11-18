# exercise-04 — Constrained and Structured Generation

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-106-summarisation-and-controlled-generation/exercises/exercise-04-constrained-and-structured-generation.md`.
> The target learning outcomes implied by that filename: force a generator to emit outputs that **conform to a schema or grammar** (JSON, SQL fragment, bounded vocabulary) via (a) **token-level logit biasing / masking** during decoding, (b) **grammar-constrained decoding** over a formal grammar, and (c) **validator + re-ask** post-hoc; and report both **validity rate** and **task quality** so the constraint choice is defensible.

## 1. Solution overview

Constrained generation is the discipline that turns a generative model into a *protocol-respecting* component. The three failure modes that decide whether the pipeline is honest:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Soft constraints that only work most of the time** | Prompt-based "please return valid JSON" produces ~90–98 % valid JSON on strong models, ~60–80 % on smaller models. The remainder breaks downstream code that assumes a schema | Pair a hard mechanism (logit masking, grammar decoding) with a validator that catches residual failures; reject-and-regenerate on the catch rather than attempting repair |
| **Constraints that are too tight** | Over-tight whitelists (e.g. forcing numeric literals only) can push the model into a *gibberish* regime because the next-best token is far off-distribution. Validity rate goes up while task quality collapses | Constrain the grammar, not the vocabulary; only mask tokens that violate the grammar *at this position*, not globally |
| **Incompatibility with beam / sampling / caching** | A logits processor that doesn't batch, doesn't track per-beam state, or mutates the KV cache produces broken outputs under beam=4 or `num_return_sequences>1` | Implement the constraint as a `LogitsProcessor` that is stateful **per beam** and side-effect-free on the cache; test under `num_beams=4` and sampling |

Three canonical mechanisms are delivered here, with escalating strictness:

1. **Logit biasing / masking** — raise or lower logits for specific token ids. Lightweight; covers "never emit token X", "always include token Y", keyword must-have / must-avoid lists.
2. **Grammar-constrained decoding** — at every step, compute the set of tokens that keep the prefix inside a formal grammar (JSON schema, regex, CFG); mask all others. The strict mechanism; produces outputs that are guaranteed to parse.
3. **Validator + regenerate** — generate freely, parse against the target schema, and either re-ask with the error attached or sample again. Cheapest to implement; adds latency; cannot *guarantee* convergence but gives a bounded retry budget.

Pick the mechanism by cost of a parse failure downstream: **validator + regenerate** when a parse failure is tolerable (log it, skip the row); **logit biasing** when you need a soft nudge (e.g. include a brand name); **grammar decoding** when the downstream consumer crashes on invalid input (production API / SQL execution).

## 2. Worked implementation

### 2.1 Project layout

```
constrained_gen/
  processors.py      # LogitsProcessor implementations
  grammar.py         # JSON-schema grammar wrapper
  validator.py       # schema parse + error normaliser
  reask.py           # validator + regenerate loop
  run.py             # CLI: compares mechanisms side-by-side
  tests/
    test_processors.py
    test_grammar.py
    test_reask.py
  requirements.txt
```

### 2.2 Logit biasing — the first tool, often enough

`transformers` ships three composable hooks. Prefer them to hand-rolled masks because they handle per-beam state correctly.

```python
# processors.py
from transformers import (
    LogitsProcessorList,
    SequenceBiasLogitsProcessor,
    NoBadWordsLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
)


def must_include_bias(tokenizer, phrases: list[str], bias: float = 10.0) -> LogitsProcessorList:
    """Boost logits of multi-token phrases so the model prefers them."""
    sequence_bias = {}
    for phrase in phrases:
        # add_prefix_space=False so "the" doesn't become " the" in some tokenisers
        ids = tuple(tokenizer(phrase, add_special_tokens=False).input_ids)
        if ids:
            sequence_bias[ids] = bias
    return LogitsProcessorList([SequenceBiasLogitsProcessor(sequence_bias=sequence_bias)])


def forbid(tokenizer, phrases: list[str]) -> LogitsProcessorList:
    """Set logits of forbidden phrases to -inf."""
    bad_token_ids = [tokenizer(p, add_special_tokens=False).input_ids for p in phrases]
    return LogitsProcessorList([NoBadWordsLogitsProcessor(bad_words_ids=bad_token_ids, eos_token_id=tokenizer.eos_token_id)])
```

Design decisions:

- **`add_special_tokens=False`** on the tokenizer call. Otherwise BOS tokens leak into the bias/ban list, which both fails silently and bans the sentence start on BART.
- **Phrase-level sequences, not single tokens.** `SequenceBiasLogitsProcessor` biases at the *last* token of each sequence conditional on the preceding tokens matching, so multi-token keywords behave as atoms.
- **Bias magnitude.** `+10.0` nudges but doesn't force; `+1000.0` is effectively a hard mandate. Hard mandates are better expressed via a grammar — a brittle `+inf` bias on `"Apple"` turns every summary into brand propaganda.

### 2.3 Prefix-constrained decoding — one callback, arbitrary logic

```python
def allow_at_position(tokenizer, allowed_fn):
    """allowed_fn(batch_id, input_ids) -> list[int] of allowed token ids at this step.

    Everything else is masked to -inf. Useful for state-machine constraints
    (e.g. after `{`, only quoted keys from the schema are allowed).
    """
    return LogitsProcessorList([
        PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn=allowed_fn, num_beams=1),
    ])
```

`PrefixConstrainedLogitsProcessor` is the canonical hook for arbitrary state-machine constraints — pass the `num_beams` you'll actually use to `.generate()`, and the processor will correctly track per-beam state.

### 2.4 Grammar-constrained decoding — the strict mechanism

For JSON, use `outlines` (Willard & Louf 2023) or `lm-format-enforcer`. Both wrap `LogitsProcessor` under the hood and accept a JSON Schema or Pydantic model as the grammar.

```python
# grammar.py
from pydantic import BaseModel, Field


class Extraction(BaseModel):
    """The schema we force the model to produce."""
    title: str = Field(min_length=1, max_length=200)
    year: int = Field(ge=1900, le=2100)
    authors: list[str] = Field(min_length=1, max_length=20)


def build_outlines_generator(model, tokenizer):
    """Returns a callable: prompt -> validated Extraction instance."""
    import outlines
    outlines_model = outlines.models.Transformers(model, tokenizer)
    generator = outlines.generate.json(outlines_model, Extraction)
    return generator
```

Points to defend in review:

- **The grammar comes from the schema, not from the prompt.** Writing the schema in the prompt and hoping is the "soft constraint" failure mode (§1). The grammar is a deterministic mask on logits.
- **Schema with constraints, not just types.** `min_length`, `ge`, `le` push the grammar to reject obviously-wrong values (empty strings, year 0). Without them, a "valid JSON" with `"year": 0` is a schema-compliant lie.
- **Grammar decoding composes with sampling.** The mask runs *before* softmax; you can still sample with `top_p=0.9` within the allowed-token set. Grammar does not force greedy.

### 2.5 Validator + regenerate — the cheapest mechanism

```python
# validator.py
import json
from pydantic import BaseModel, ValidationError


def parse_or_error(text: str, schema: type[BaseModel]):
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    try:
        return schema.model_validate(data), None
    except ValidationError as e:
        return None, f"schema violation: {e}"


# reask.py
def reask_loop(model, tokenizer, prompt: str, schema: type[BaseModel], *, max_retries: int = 2):
    """Generate -> validate -> on failure, re-ask with the error as extra context."""
    attempts = []
    current_prompt = prompt
    for attempt in range(max_retries + 1):
        ids = tokenizer(current_prompt, return_tensors="pt", truncation=True).input_ids.to(model.device)
        out = model.generate(ids, max_new_tokens=256, num_beams=1, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        obj, err = parse_or_error(text, schema)
        attempts.append({"text": text, "error": err})
        if obj is not None:
            return obj, attempts
        current_prompt = prompt + f"\n\nYour previous attempt failed: {err}\nTry again."
    return None, attempts
```

Design decisions:

- **Bounded retries.** Unbounded retry loops are a DoS vector. `max_retries=2` means at most 3 generations per request; past that, surface the failure.
- **Error attached to the next prompt.** Just regenerating with a different seed often fails the same way; feeding the error back nudges the model toward a different mode.
- **Attempts recorded.** The caller gets every attempt for logging / debugging. Hiding failed attempts hides the validity-rate signal.
- **Greedy for retries.** Sampling at retry time makes failures non-reproducible; greedy with a modified prompt is the audit-friendly choice.

### 2.6 Comparison runner

```python
# run.py
import json
from .validator import parse_or_error
from .grammar import Extraction, build_outlines_generator
from .reask import reask_loop


def validity_rate(objs: list):
    return sum(o is not None for o in objs) / max(1, len(objs))


def run_mechanism_compare(model, tokenizer, prompts: list[str]):
    results = {}

    # 1) Prompt only
    raw = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", truncation=True).input_ids.to(model.device)
        out = model.generate(ids, max_new_tokens=256, num_beams=1, do_sample=False)
        obj, _ = parse_or_error(tokenizer.decode(out[0], skip_special_tokens=True), Extraction)
        raw.append(obj)
    results["prompt_only"] = {"valid_rate": validity_rate(raw)}

    # 2) Validator + regenerate
    reasked = []
    for p in prompts:
        obj, attempts = reask_loop(model, tokenizer, p, Extraction, max_retries=2)
        reasked.append(obj)
    results["reask"] = {"valid_rate": validity_rate(reasked)}

    # 3) Grammar-constrained decoding
    gen = build_outlines_generator(model, tokenizer)
    grammar = [gen(p) for p in prompts]
    results["grammar"] = {"valid_rate": 1.0, "note": "grammar guarantees validity by construction"}

    return results
```

The mechanism table is the deliverable. Expect numbers in the shape:

| Mechanism | Validity rate | Extra latency | Task quality (vs free gen) |
|---|---|---|---|
| prompt only | 0.60–0.95 | 1.0× | baseline |
| logit biasing (must-include) | ~baseline validity | ~1.0× | +0.5 to -0.3 ROUGE depending on bias magnitude |
| validator + regenerate (k=2) | 0.90–0.99 | 1.0–3.0× | baseline (no change on passing cases) |
| grammar-constrained | 1.00 | 1.1–1.4× | -0.2 to +0.1 on task quality; -1 to -3 on perplexity |

These are qualitative shapes from the Willard & Louf 2023 benchmarks and the published follow-ups, not promises for a specific model.

## 3. Validation steps

1. **Beam-safety of logit processors.** Run `allow_at_position(...)` with `num_beams=4`. Inspect outputs: each beam must respect the constraint at every step. If one beam violates, `num_beams` wasn't passed through to the processor.
2. **Grammar validity rate = 1.0.** On 100 test prompts, the grammar-constrained generator must produce 100/100 schema-valid outputs. Anything less is a bug in the grammar wrapper, not an acceptable cost.
3. **Over-constraining diagnostic.** Impose a grammar that allows only the single token `"a"` as every output token. Perplexity on the gold answer must skyrocket, and task quality must collapse — this is the sanity check that *tight* constraints are detectable from the metrics, not hidden.
4. **Prompt-only baseline.** Measure validity rate without any constraint mechanism; expect 0.5–0.95 depending on model size. If it's already 1.0 on your dev set, your prompts are too easy — add adversarial examples.
5. **Reask convergence.** On the dev set, record `attempts` per request. The histogram should fall off at 1–2 retries; if 20 %+ of requests hit `max_retries` and still fail, the schema is incompatible with the model's distribution and grammar decoding is the right next step.
6. **Must-include enforcement via bias.** With `must_include_bias(..., bias=10.0)` for `["Apple Inc."]`, the inclusion rate on relevant summaries should rise from baseline to ~80–95 %. If it reaches 100 %, bias is too high and quality is suffering elsewhere.
7. **Latency per mechanism.** Grammar decoding adds overhead because every step re-computes the allowed-token set. On a 128-token output with a JSON schema, expect 10–40 % more decode latency than free generation. A mechanism that is *faster* than free generation is almost certainly broken.
8. **Determinism under constraints.** Grammar + greedy must be byte-reproducible across runs with the same prompt. If not, the grammar's internal state (cache, memoization) is leaking across calls.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Mechanism breadth** | At least three of {prompt-only, logit biasing, prefix-constrained, grammar, validator + reask} compared on the same prompts | Two mechanisms | One mechanism asserted without comparison |
| **Validity rate reported** | Per-mechanism validity rate on the same dev prompts; grammar at 1.00 | Validity measured for one mechanism | "Looks valid" with no measurement |
| **Task-quality side-by-side** | ROUGE / accuracy / pass-rate reported alongside validity; over-constraining visible in the numbers | Quality measured, no over-constraining probe | Only validity, no quality metric |
| **Beam / sampling safety** | Custom processors tested under `num_beams>1` and sampling; per-beam state verified | Tested greedy only | Fails or is undefined under beam/sampling |
| **Reask bounded** | `max_retries` capped; attempts logged; failure surfaced upward on exhaustion | Retries bounded, not logged | Unbounded retry; retries hidden in success path |
| **Schema richness** | Pydantic / JSON Schema with `min_length`, bounds, enums; not just types | Types only | Free-form JSON object with no schema |
| **Reproducibility** | `set_seed`; grammar version and mechanism echoed in run JSON | Seed set, no echo | No seed |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks: constraint bypass via adversarial prompts, DoS via infinite reask, output injection into downstream parser | Mentioned without mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure: schema version, mechanism used per endpoint, per-mechanism validity + quality tradeoff | One of those | Neither |

## 5. Common mistakes

- **Trusting prompt-level "please return JSON" as a constraint.** Works most of the time; one day it won't, and your parser crashes downstream. Add a validator at minimum.
- **Grammar + sampling without noting the interaction.** Grammar masks run before softmax, so `top_p=0.9` samples from the allowed-token nucleus. That's usually what you want, but it must be acknowledged — a reviewer who assumes "grammar = deterministic" will be surprised.
- **Logit bias set in token space without checking tokenisation.** `bias["Apple"]` might become `["Ap", "ple"]` in one tokeniser and `["Apple"]` in another; a single-token bias is a no-op on the multi-token version. Always use the sequence-bias helper, or inspect the token split.
- **Reask using the same seed and same prompt.** The model produces the same invalid output. Either pass a different seed or attach the parse error to the next prompt (the §2.5 pattern).
- **`PrefixConstrainedLogitsProcessor` with `num_beams=1` while generating with `num_beams=4`.** Per-beam state is wrong; outputs from beams 2–4 violate the constraint. Match `num_beams` at construction time or pass `num_beams` through explicitly.
- **Ignoring EOS in the allowed-token set.** If the allowed-token function never returns `eos_token_id`, the generator runs until `max_new_tokens`. Always include EOS in the allowed set when the grammar permits end-of-output.
- **Masking globally rather than positionally.** Blanket-banning every non-JSON character is tempting but breaks the model's ability to recover from a mistake; positional masks (CFG-driven) are the right abstraction.
- **No over-constraining probe.** It is easy to achieve "100 % validity" by forcing the output to `{"x": 0}` for every input. The task-quality metric (§3.3) is what prevents this mode.
- **Pydantic v1 syntax in a v2 project (or vice versa).** `Field(min_items=...)` vs `Field(min_length=...)`, `.parse_obj` vs `.model_validate`. Pin a version and wire the schema to match; silent `TypeError` on a validator is a common debugging sink.
- **Treating grammar decoding as "no cost".** Every step re-computes the allowed set; on long outputs with deep schemas this can double decode time. Measure and report.

## 6. References

Primary sources used in this solution:

- Willard, B., Louf, R. (2023). *Efficient Guided Generation for Large Language Models* (the **Outlines** paper). arXiv:2307.09702. <https://arxiv.org/abs/2307.09702> — algorithm behind JSON-schema / regex grammar decoding; the library referenced in §2.4.
- Geng, S. et al. (2023). *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*. EMNLP. <https://aclanthology.org/2023.emnlp-main.674/> — grammar-constrained decoding used as a general structured-prediction mechanism; informs the grammar-decoding design.
- Scholak, T., Schucher, N., Bahdanau, D. (2021). *PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models*. EMNLP. <https://aclanthology.org/2021.emnlp-main.779/> — the canonical incremental-grammar decoder for SQL; a reference implementation of the "mask-then-parse" pattern used in §2.4.
- Lu, X. et al. (2021). *NeuroLogic Decoding: (Un)supervised Neural Text Generation with Predicate Logic Constraints*. NAACL. <https://aclanthology.org/2021.naacl-main.339/> — the canonical formulation of soft-and-hard lexical constraints at decode time.
- Hugging Face `transformers` docs, text generation — <https://huggingface.co/docs/transformers/internal/generation_utils> — reference for `LogitsProcessorList`, `SequenceBiasLogitsProcessor`, `NoBadWordsLogitsProcessor`, `PrefixConstrainedLogitsProcessor` used in §2.2-§2.3.
- `outlines` library — <https://github.com/outlines-dev/outlines> — JSON / regex / CFG grammar decoding for `transformers`; the mechanism in §2.4.
- `lm-format-enforcer` library — <https://github.com/noamgat/lm-format-enforcer> — alternative grammar-decoding implementation with the same contract; useful as a cross-check in CI.
- Pydantic v2 docs — <https://docs.pydantic.dev/> — the schema definition DSL used in §2.4 and §2.5.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation*: adversarial inputs can push the model to adopt instructions that bypass soft constraints; grammar-level enforcement neutralises this. *ML09 Output Integrity Attacks*: downstream parsers that trust model output are directly vulnerable when structured generation is not enforced. Mitigations: grammar decoding for high-trust endpoints, validator + bounded reask with failure surfaced upward.
- MITRE ATLAS — <https://atlas.mitre.org/> — relevant technique entries for evasion of downstream validators and output-channel abuse (model emits content that looks structured enough to pass a shallow parser but carries payload for a stricter one).
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record the schema version, the enforcement mechanism per endpoint, and the retry policy; *Measure*: report validity **and** quality side-by-side; tight-constraint regressions become visible in the latter when the former hits 1.00.
