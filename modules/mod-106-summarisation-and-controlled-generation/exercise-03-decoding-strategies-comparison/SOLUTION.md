# exercise-03 — Decoding Strategies Comparison

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-106-summarisation-and-controlled-generation/exercises/exercise-03-decoding-strategies-comparison.md`.
> The target learning outcomes implied by that filename: compare the canonical decoding strategies — **greedy**, **beam search**, **temperature / top-k / top-p (nucleus) sampling**, and the published **repetition controls** (`no_repeat_ngram_size`, `repetition_penalty`) — on the *same* model and prompts, and report both **quality** (ROUGE / BLEU / pass-rate) and **diversity** (self-BLEU, distinct-n) so the strategy choice is defensible.

## 1. Solution overview

Decoding is where "the model is fine" turns into "the output is bad" or vice-versa. The three failure modes that decide whether the comparison is honest:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **One-metric comparison** | Reporting only ROUGE picks beam search; reporting only diversity picks pure sampling. Both decisions are wrong for the opposite use case | Report a **quality** metric (ROUGE / pass-rate) and a **diversity** metric (distinct-n, self-BLEU) side-by-side, and pick the knee of the tradeoff |
| **Non-identical conditioning** | Changing seed, prompt, or `GenerationConfig` defaults between strategies hides the effect of the strategy itself; e.g. beam with `length_penalty=2.0` vs sampling with `length_penalty=1.0` conflates the length prior with the decoder family | Freeze everything except the decoding kwargs and the sampling seed; log the effective `GenerationConfig` next to every result |
| **Single-sample sampling** | Sampling-based metrics computed on `num_return_sequences=1` have 3–8× the variance of beam at the same compute cost; a "sampling is worse than beam" conclusion from n=1 is unsafe | Take `num_return_sequences>=5` for sampling and report the sample distribution, not a point estimate |

The deliverable is a small, reproducible benchmark harness with four components:

1. **Fixed-prompt dataset** — a slice of CNN/DM dev (for summarisation) *and* a slice of WritingPrompts or an open-ended seed (for open-ended generation), so both regimes are exercised.
2. **Strategy catalogue** — a dict of named `GenerationConfig` objects covering greedy / beam-4 / beam-4 + no-repeat / top-k=50 / top-p=0.9 / top-p=0.9 + temperature=0.7 / contrastive decoding.
3. **Generation runner** — one function that takes `(prompt, strategy, seed, n_samples)` and returns the raw decoded strings.
4. **Evaluation** — pairing a quality metric (ROUGE-L for summarisation, pass-rate for open-ended) with diversity metrics (**distinct-1/2/3**, **self-BLEU-4**) and a latency metric (tokens/sec, median samples/sec).

The exercise is not "find the best decoder"; it is "characterise the tradeoff and document the choice". Reviewers should push back on any claim of "best" without the two-axis plot.

## 2. Worked implementation

### 2.1 Project layout

```
decode_compare/
  configs.py         # GenerationConfig catalogue
  generate.py        # strategy runner
  diversity.py       # distinct-n + self-BLEU
  quality.py         # ROUGE + (optional) pass-rate
  run.py             # CLI: writes a long-form table + plots
  tests/
    test_configs.py
    test_diversity.py
    test_determinism.py
  requirements.txt
```

### 2.2 Strategy catalogue — the canonical set, pinned

```python
# configs.py
from transformers import GenerationConfig


def strategy_configs(tokenizer, max_new_tokens: int = 128) -> dict[str, GenerationConfig]:
    base = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return {
        "greedy":          GenerationConfig(**base, do_sample=False, num_beams=1),
        "beam4":           GenerationConfig(**base, do_sample=False, num_beams=4,
                                            length_penalty=1.0, early_stopping=True),
        "beam4_noreptri":  GenerationConfig(**base, do_sample=False, num_beams=4,
                                            length_penalty=1.0, early_stopping=True,
                                            no_repeat_ngram_size=3),
        "topk50":          GenerationConfig(**base, do_sample=True, top_k=50, top_p=1.0,
                                            temperature=1.0),
        "topp090":         GenerationConfig(**base, do_sample=True, top_k=0, top_p=0.90,
                                            temperature=1.0),
        "topp090_t07":     GenerationConfig(**base, do_sample=True, top_k=0, top_p=0.90,
                                            temperature=0.70),
        "contrastive":     GenerationConfig(**base, penalty_alpha=0.6, top_k=4,
                                            do_sample=False),
    }
```

Design decisions worth defending:

- **`length_penalty=1.0` across all strategies.** Beam search with `length_penalty=2.0` (BART-CNN/DM default) is a *length prior*, not a decoder family. Holding it constant is the only way to attribute differences to the strategy.
- **`top_k=0` when `top_p` is set.** Leaving `top_k=50` on while varying `top_p` runs top-k ∩ top-p, which is not nucleus sampling and conflates the axes. Explicitly set `top_k=0` to disable it.
- **Contrastive search (Su et al. 2022) included.** `penalty_alpha=0.6, top_k=4` is the paper's recommended setting; it often matches sampling diversity at near-greedy quality on open-ended tasks.
- **Temperature + nucleus combo included.** `top_p=0.9, temperature=0.7` is the common "chat" default and catches whether temperature alone changes the quality/diversity tradeoff vs raw nucleus.

### 2.3 Generation runner — freeze everything else

```python
# generate.py
import torch
from transformers import set_seed


def generate_with(model, tokenizer, prompt: str, cfg, *, seed: int, n_samples: int = 1):
    set_seed(seed)                             # covers torch, numpy, random
    ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(model.device)
    with torch.inference_mode():
        out = model.generate(
            ids,
            generation_config=cfg,
            num_return_sequences=n_samples if cfg.do_sample else 1,
        )
    texts = [tokenizer.decode(o, skip_special_tokens=True) for o in out]
    # Strip the prompt echo for decoder-only models; seq2seq models don't echo.
    if model.config.is_encoder_decoder:
        return texts
    return [t[len(prompt):] if t.startswith(prompt) else t for t in texts]
```

Three things reviewers should look for:

- **`set_seed(seed)` before every call.** Sampling without this is pseudo-random from torch's global state and the same strategy returns different samples run-to-run, which destroys reproducibility.
- **`num_return_sequences=1` for deterministic strategies.** Beam/greedy return the same string no matter how many samples you ask for; only sampling methods benefit from multiple returns.
- **`torch.inference_mode()`.** Shaves 5–15 % off decode latency and prevents gradient storage; important for the latency metric.

### 2.4 Diversity metrics — distinct-n and self-BLEU

```python
# diversity.py
import nltk
from collections import Counter


def distinct_n(texts: list[str], n: int) -> float:
    ngrams, total = [], 0
    for t in texts:
        tokens = t.split()
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i : i + n]))
            total += 1
    if total == 0:
        return 0.0
    return len(set(ngrams)) / total


def self_bleu(samples: list[str], n: int = 4) -> float:
    if len(samples) < 2:
        return 0.0
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    weights = tuple(1.0 / n for _ in range(n))
    scores = []
    for i, cand in enumerate(samples):
        refs = [samples[j].split() for j in range(len(samples)) if j != i]
        scores.append(sentence_bleu(refs, cand.split(), weights=weights, smoothing_function=smooth))
    return sum(scores) / len(scores)
```

Two definitions, both from the published literature:

- **Distinct-n** (Li et al. 2016) — fraction of unique n-grams across generated texts. Higher is more diverse. Distinct-1 ∈ [~0.05 model-looping, ~0.5 sampling]; distinct-3 is more sensitive and cleaner to plot.
- **Self-BLEU** (Zhu et al. 2018) — treat each sample as candidate and the rest as references; low self-BLEU = high diversity. Beam often lands around 0.6–0.8; nucleus around 0.2–0.4.

### 2.5 Quality metrics — reuse exercise-01's ROUGE

For summarisation, the quality metric is the same `rouge_metrics` from exercise-01 (stemming on, `rougeLsum` with sentence-split, bootstrap F1). For open-ended generation there is no reference text; replace ROUGE with one of:

- **Perplexity under a held-out LM** (a different model from the generator).
- **MAUVE** (Pillutla et al. 2021) — distribution divergence against human text; the published open-ended quality proxy.
- **Pass-rate** — binary human / LLM-judge verdict on whether the output satisfies the prompt.

```python
# quality.py
from rouge_score import rouge_scorer


def rouge_l_f(pred: str, ref: str) -> float:
    s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(ref, pred)
    return 100.0 * s["rougeL"].fmeasure
```

### 2.6 Runner — the comparison table

```python
# run.py
import json, time
from statistics import mean
from .configs import strategy_configs
from .generate import generate_with
from .diversity import distinct_n, self_bleu
from .quality import rouge_l_f


def run_comparison(model, tokenizer, prompts_with_refs: list[tuple[str, str]], *, seed: int = 20260101, n_samples: int = 5):
    cfgs = strategy_configs(tokenizer)
    rows = []
    for name, cfg in cfgs.items():
        rouges, latencies, all_outputs = [], [], []
        for prompt, ref in prompts_with_refs:
            n = n_samples if cfg.do_sample else 1
            t0 = time.perf_counter()
            outputs = generate_with(model, tokenizer, prompt, cfg, seed=seed, n_samples=n)
            latencies.append((time.perf_counter() - t0) / n)
            all_outputs.extend(outputs)
            # Quality is averaged over samples for sampling strategies.
            rouges.append(mean(rouge_l_f(o, ref) for o in outputs))
        rows.append({
            "strategy": name,
            "rouge_l_mean": mean(rouges),
            "distinct_1": distinct_n(all_outputs, 1),
            "distinct_3": distinct_n(all_outputs, 3),
            "self_bleu_4": self_bleu(all_outputs[:50]),  # cap for O(n^2) metric
            "sec_per_sample": mean(latencies),
            "n_samples_per_prompt": n_samples if cfg.do_sample else 1,
            "config_echo": cfg.to_dict(),
        })
    return rows


if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from datasets import load_dataset

    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="facebook/bart-base")
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260101)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model_id).eval()

    ds = load_dataset("cnn_dailymail", "3.0.0", split="validation").shuffle(seed=0).select(range(a.n_prompts))
    prompts_with_refs = [(ex["article"], ex["highlights"]) for ex in ds]

    rows = run_comparison(model, tok, prompts_with_refs, seed=a.seed)
    print(json.dumps(rows, indent=2))
```

The output is the whole point of the exercise: a table with one row per strategy and a column each for `rouge_l_mean`, `distinct_3`, `self_bleu_4`, and `sec_per_sample`. A reviewer must be able to read the table and defend the deployed choice. Plot `rouge_l` vs `distinct_3`: beam typically sits top-left (high quality, low diversity), nucleus bottom-right, contrastive and `topp090_t07` on the Pareto frontier.

### 2.7 Expected qualitative pattern

On a BART-base CNN/DM run (n=50 prompts, 5 samples per sampling strategy):

| Strategy | ROUGE-L (↑) | Distinct-3 (↑) | Self-BLEU-4 (↓) | sec/sample |
|---|---|---|---|---|
| greedy | ~39.0 | ~0.60 | ~0.95 | 0.4 |
| beam4 | ~40.5 | ~0.55 | ~0.98 | 1.3 |
| beam4_noreptri | ~40.9 | ~0.70 | ~0.90 | 1.3 |
| topk50 | ~36.0 | ~0.85 | ~0.30 | 0.4 |
| topp090 | ~36.5 | ~0.88 | ~0.25 | 0.4 |
| topp090_t07 | ~38.5 | ~0.80 | ~0.45 | 0.4 |
| contrastive | ~39.5 | ~0.78 | ~0.55 | 0.6 |

These are **qualitative** expectations drawn from the Welleck et al. 2020 and Holtzman et al. 2020 literature on BART-size models, not promises for a specific run. The *shape* is what the exercise should produce: beam wins ROUGE, sampling wins diversity, contrastive sits on the knee, `no_repeat_ngram_size=3` improves beam on both axes.

## 3. Validation steps

1. **Determinism of beam and greedy.** Run `beam4` and `greedy` twice with the same seed and the same prompt. Outputs must be byte-identical. If not, something is leaking non-determinism — commonly `torch.use_deterministic_algorithms(False)` or CUDA cuBLAS workspaces.
2. **Non-determinism of sampling.** Run `topp090` twice with different seeds. Outputs must differ. If identical, `do_sample=True` isn't reaching `.generate()` — most often because an outer `with torch.no_grad():` isn't the problem, but a caller forgot to pass `generation_config` and the default `do_sample=False` wins.
3. **Length-penalty invariance.** Hold `length_penalty=1.0` across strategies; the median output length should not vary by more than ~15 % across strategies on the same prompt set. A 2× length difference means a strategy has a different length prior and the comparison is confounded.
4. **Diversity-metric sanity.** Build a trivial case: 5 identical outputs → `distinct_3 ≈ 0`, `self_bleu_4 ≈ 1.0`. Five completely disjoint outputs → `distinct_3 ≈ 1`, `self_bleu_4 ≈ 0`.
5. **Config echo in output.** Every row must contain `config_echo`; reviewers must be able to reproduce the exact decoding pass from the row alone.
6. **Pareto-frontier sanity.** Plot `rouge_l_mean` vs `distinct_3`. Beam / greedy should land on the top-left; pure sampling on the bottom-right. Contrastive and temperature-nucleus should land between them. If beam has higher `distinct_3` than nucleus, `no_repeat_ngram_size` isn't the cause and something is wrong.
7. **Repetition-penalty ablation.** Drop `no_repeat_ngram_size=3` on `beam4_noreptri` → expect `distinct_3` to fall by 0.10–0.15 and ROUGE-L to drop by 0.3–0.5 points. If dropping it has no effect, the model isn't repeating on this dataset and the knob is irrelevant (fine, but note it in the report).
8. **Latency floor.** `sec_per_sample` for beam-4 should be ~2–4× greedy; for contrastive ~1.3–1.8× greedy; sampling ~1× greedy. A beam that is 10× greedy latency is a batch-size or caching bug, not a strategy property.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Strategy catalogue** | At least {greedy, beam, beam+no-repeat, top-k, top-p, top-p+temperature, contrastive} with pinned `GenerationConfig` per strategy | 3–4 strategies | 2 or fewer; or hand-tuned per-prompt configs |
| **Fair comparison** | Same model, same prompts, same seed for deterministic strategies, same `length_penalty`; config echoed | One axis varies silently (length penalty, prompt order) | Different models or prompt sets per strategy |
| **Sampling variance handled** | `num_return_sequences >= 5` for sampling; metric aggregated across samples | Single sample per sampling run | Reports one sample as representative of the strategy |
| **Quality metric** | ROUGE-L with the exercise-01 protocol (stemming, bootstrap F1) or published open-ended proxy (MAUVE, pass-rate) | Raw ROUGE / perplexity only | Single-number with no protocol stated |
| **Diversity metric** | Distinct-1 and -3 **and** self-BLEU-4 reported | One of the two | Neither; diversity asserted without measurement |
| **Latency reported** | `sec_per_sample` per strategy; batching and `use_cache` noted | Mentioned, not measured | Omitted |
| **Reproducibility** | `set_seed`; rows contain `config_echo`; dataset revision + model id pinned | Seed set, no echo | No seed |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks: sampling increases hallucination variance, prompt injection amplified under high-temperature; mitigations stated | Mentioned without mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure: decoding policy, variance across seeds and samples, latency vs quality tradeoff | One of those | Neither |

## 5. Common mistakes

- **`length_penalty` drifts between strategies.** Beam at 2.0 vs sampling at 1.0 makes beam look 1.5–2 points better than the strategy actually is. Pin to 1.0 everywhere for a fair comparison.
- **Top-k ∩ top-p by accident.** `do_sample=True, top_k=50, top_p=0.9` is a different distribution from either knob alone; the recent nucleus-sampling paper is at `top_k=0, top_p=0.9`. State the mode explicitly.
- **Measuring on n=1 sampling.** A single nucleus sample is a noisy estimator of "how the strategy behaves". n≥5 is the minimum defensible count; 20 is better if latency permits.
- **Reporting distinct-1 only.** Distinct-1 is dominated by stop-word distribution and barely moves across strategies. Distinct-3 is where the differences show.
- **Omitting `set_seed` for sampling.** Different seeds for the same strategy make re-runs irreproducible; the published practice is to seed per-call and record the seed with the output.
- **Beam with `num_beams=1`.** `do_sample=False, num_beams=1` is greedy, not beam. Easy to type as "beam" in a config dict and never notice; the determinism test (§3.1) catches it.
- **Running BART with GPT-2 defaults.** `temperature=1.0, top_k=50, top_p=1.0` is a GPT-2 OpenAI-blog default. BART at the same settings is slightly less coherent because its training distribution is tighter; `top_p=0.9, temperature=0.7` is more defensible for encoder-decoders.
- **Reporting the best sample across samples.** `max(rouge_l_f(o, ref) for o in outputs)` is oracle-selection; it's a valid upper bound but not a deployable metric. Report `mean`; put `max` next to it as an upper-bound only.
- **Latency on the first call.** CUDA kernels compile and caches warm on the first call. Always run a 3-prompt warm-up before timing.
- **Treating "beam search is better" as universal.** Beam wins ROUGE on summarisation (short, reference-constrained outputs); on open-ended generation it produces repetitive or short outputs (Welleck et al. 2020). The decoding choice depends on the task regime; the exercise's output is a tradeoff plot, not a winner.

## 6. References

Primary sources used in this solution:

- Holtzman, A. et al. (2020). *The Curious Case of Neural Text Degeneration*. ICLR. <https://openreview.net/forum?id=rygGQyrFvH> — introduces **nucleus (top-p)** sampling, documents beam's repetition pathology, and defines **distinct-n** reporting; informs §2.4.
- Fan, A., Lewis, M., Dauphin, Y. (2018). *Hierarchical Neural Story Generation*. ACL. <https://aclanthology.org/P18-1082/> — introduces **top-k** sampling and compares greedy / beam / sampling on open-ended generation; informs §2.2.
- Welleck, S. et al. (2020). *Neural Text Generation with Unlikelihood Training*. ICLR. <https://openreview.net/forum?id=SJeYe0NtvH> — documents beam search's degenerate-repetition failure and motivates `no_repeat_ngram_size`.
- Su, Y. et al. (2022). *A Contrastive Framework for Neural Text Generation* (Contrastive Search). NeurIPS. <https://arxiv.org/abs/2202.06417> — defines `penalty_alpha`, the decoding mechanism used in the `contrastive` row.
- Li, J. et al. (2016). *A Diversity-Promoting Objective Function for Neural Conversation Models*. NAACL. <https://aclanthology.org/N16-1014/> — definition of **distinct-n**.
- Zhu, Y. et al. (2018). *Texygen: A Benchmarking Platform for Text Generation Models*. SIGIR. <https://dl.acm.org/doi/10.1145/3209978.3210080> — definition of **self-BLEU**.
- Pillutla, K. et al. (2021). *MAUVE: Measuring the Gap Between Neural Text and Human Text*. NeurIPS. <https://proceedings.neurips.cc/paper/2021/hash/260c2432a0eecc28ce03c10dadc078a4-Abstract.html> — distribution-divergence quality metric for open-ended generation.
- Hugging Face `transformers` text-generation guide — <https://huggingface.co/docs/transformers/generation_strategies> — authoritative reference for every `GenerationConfig` flag used in §2.2.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation*: sampling expands the output distribution and therefore the surface area for prompt-injection payloads to materialise in the output. Mitigations: cap temperature in high-trust contexts; use beam or low-temperature nucleus; validate outputs against an allow-list before forwarding.
- MITRE ATLAS — <https://atlas.mitre.org/> — relevant technique categories for generation-stage attacks (evasion through decoder temperature, adversarial prompts that exploit nucleus tail mass).
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record the decoding policy, including strategy, seed policy, and all `GenerationConfig` fields; *Measure*: report quality **and** diversity, and record inter-sample variance for stochastic strategies.
