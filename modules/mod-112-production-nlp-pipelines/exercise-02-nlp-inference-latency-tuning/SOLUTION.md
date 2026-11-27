# exercise-02 — NLP Inference Latency Tuning

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-112-production-nlp-pipelines/exercises/exercise-02-nlp-inference-latency-tuning.md`.
> The target learning outcomes implied by that filename: take a baseline NLP inference service
> (transformer encoder or encoder-decoder, served via `transformers` or `onnxruntime`), **measure
> its latency distribution** under realistic load, apply a defensible sequence of tuning techniques
> (**dynamic batching, sequence-length bucketing, tokenizer-side truncation, mixed precision,
> KV-cache for decoders, continuous batching for generation**), and show the **tail-latency and
> throughput delta** for each, with a per-change artefact a reviewer can audit.

## 1. Solution overview

Latency tuning without measurement is folklore. The dominant failure modes are not slow
kernels; they are measurement failures that let a "faster" variant ship before anyone notices
the tail got worse:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Reporting mean latency** | Mean collapses a bimodal "fast tokens / slow long sequences" distribution to one number; the user experience is p95 / p99, not mean | Report p50 / p95 / p99 / max; compute CIs via bootstrap. Dror et al. 2018 argue the same for NLP metrics; the argument carries to latency |
| **Measuring warm-run throughput only** | The first 50 requests include lazy-init cost; the published number excludes them and the production cold-start is worse | Report **cold** (first N requests) **and** steady-state. Pre-warm before publishing steady-state only when the deployment is similarly pre-warmed |
| **Static batch size** | Batching latency = time for the *slowest* sequence in the batch + queueing. A fixed-size batch wastes padding on short sequences and times out on long ones | Dynamic batching with a timeout (NVIDIA Triton; vLLM; TorchServe) + **length bucketing** so same-length sequences are batched together |
| **No KV-cache for autoregressive decoding** | Each generation step re-encodes the entire prefix. For a 512-token prompt and 128-token continuation, that is ~40× redundant compute vs. caching | Enable KV-cache (`model.generate(..., use_cache=True)`); for high-concurrency generation use **continuous batching** (PagedAttention / vLLM / TGI) which also solves the padded-batch problem |
| **Max-length truncation that silently drops content** | `truncation=True, max_length=512` but inputs routinely run 2 000 tokens; the service returns technically valid outputs on silently truncated inputs | Measure input-length distribution first; pick `max_length` from the 99th percentile; log a truncation counter so drift is visible |
| **Precision changed without accuracy check** | fp16 / bf16 / int8 all save latency; one of them silently drops accuracy on your data | Every precision change is paired with an accuracy re-measurement on the validation slice (see mod-111 exercise-01 for metric choice) and the delta is published alongside the latency delta |

The deliverable is a four-part tuning harness:

1. **Load generator** — closed-loop or open-loop load, configurable concurrency, realistic
   input-length distribution sampled from production traces.
2. **Measurement harness** — per-request timings (queue, tokenise, encode, decode, de-tokenise);
   percentile summary with bootstrap CIs; cold-vs-warm split.
3. **Variants** — a numbered sequence of changes (`v0 baseline`, `v1 mixed_precision`,
   `v2 length_bucketing`, `v3 dynamic_batching`, `v4 kv_cache`, `v5 onnxruntime`, etc.), each
   isolated so the delta is attributable.
4. **Report** — a markdown table + a CSV with p50/p95/p99/throughput for each variant, and an
   accuracy delta per variant where the change could affect it.

Authoritative sources this solution draws on:

- **Hugging Face `generate` reference** — <https://huggingface.co/docs/transformers/main_classes/text_generation>.
  `use_cache`, `num_beams`, `do_sample`, `pad_token_id`, `max_new_tokens`.
- **PyTorch `torch.cuda.amp` autocast** — <https://pytorch.org/docs/stable/amp.html>. The
  official mixed-precision reference; `autocast(dtype=torch.bfloat16)` on Ampere+ GPUs.
- **ONNX Runtime** — <https://onnxruntime.ai/docs/>. Execution providers (`CUDAExecutionProvider`,
  `TensorrtExecutionProvider`), IO binding, graph optimisation levels.
- **NVIDIA Triton Inference Server dynamic batching** —
  <https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md>.
  `dynamic_batching { max_queue_delay_microseconds: ... preferred_batch_size: [...] }`.
- **vLLM** — <https://docs.vllm.ai/> and Kwon et al. 2023 *Efficient Memory Management for Large
  Language Model Serving with PagedAttention* (SOSP). <https://arxiv.org/abs/2309.06180>. The
  authoritative continuous-batching + paged-KV reference.
- **Hugging Face Text Generation Inference (TGI)** — <https://github.com/huggingface/text-generation-inference>.
  The continuous-batching implementation used by many serving stacks.
- **Measuring: `locust`** — <https://docs.locust.io/>; **`k6`** — <https://k6.io/docs/>;
  **`wrk2`** — <https://github.com/giltene/wrk2>. Open-loop load with coordinated-omission
  correction is what you want; closed-loop under-reports tail latency.
- **Gil Tene's "How NOT to Measure Latency"** talk (coordinated omission) —
  <https://www.youtube.com/watch?v=lJ8ydIuPFeU>. The reason open-loop load matters.
- **FlashAttention 2** (Dao 2023) — <https://arxiv.org/abs/2307.08691>. The memory-IO-bound
  attention kernel that is the default in modern inference stacks; relevant when the baseline
  doesn't use it.

## 2. Worked implementation

### 2.1 Project layout

```
latency_tuning/
  src/latency_tuning/
    __init__.py
    config.py                 # CLI args + variant registry
    model_loader.py           # loads HF or ONNX model given a variant
    server.py                 # FastAPI inference service
    load_gen.py               # open-loop load generator with coord-omission correction
    report.py                 # summarise per-variant results into table + csv
    variants/
      v0_baseline.py
      v1_bf16.py
      v2_length_bucketing.py
      v3_dynamic_batching.py
      v4_kv_cache.py
      v5_onnxruntime.py
  configs/
    traffic_profile.json      # input-length distribution sampled from traces
  tests/
    test_percentiles.py
    test_coord_omission.py
    test_accuracy_unchanged.py
  reports/
    2026-11-26_baseline.csv
    2026-11-26_tuning.md
  pyproject.toml
  README.md
```

### 2.2 Measurement harness — percentiles, bootstrap CIs, cold-vs-warm

```python
# src/latency_tuning/load_gen.py
from __future__ import annotations
import asyncio, random, statistics, time, json, pathlib
from dataclasses import dataclass, field
import httpx


@dataclass
class Sample:
    request_sent_at_s: float
    first_byte_at_s: float
    done_at_s: float
    input_tokens: int
    output_tokens: int | None
    status: int


@dataclass
class LatencyReport:
    label: str
    cold_samples: list[Sample] = field(default_factory=list)
    warm_samples: list[Sample] = field(default_factory=list)

    def _latencies_ms(self, samples: list[Sample]) -> list[float]:
        return [(s.done_at_s - s.request_sent_at_s) * 1000.0 for s in samples if s.status == 200]

    def percentiles(self, samples: list[Sample]) -> dict[str, float]:
        lats = self._latencies_ms(samples)
        if not lats:
            return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
        lats_sorted = sorted(lats)
        def pct(p: float) -> float:
            idx = min(len(lats_sorted) - 1, int(round(p * (len(lats_sorted) - 1))))
            return lats_sorted[idx]
        return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99), "max": max(lats_sorted),
                "mean": statistics.fmean(lats_sorted), "n": len(lats_sorted)}


async def open_loop(
    url: str,
    traffic: list[dict],
    qps: float,
    duration_s: float,
    cold_count: int = 50,
) -> LatencyReport:
    """
    Open-loop (Poisson-arrival) load. Each request is dispatched on its own task so the generator
    rate does not back-pressure on slow responses — this is what captures tail latency honestly
    (Gil Tene "How NOT to Measure Latency"). Closed-loop k-workers-in-a-ring is the wrong tool.
    """
    report = LatencyReport(label=url)
    end_at = time.perf_counter() + duration_s

    async with httpx.AsyncClient(timeout=30.0) as client:
        async def fire(payload: dict) -> None:
            sent_at = time.perf_counter()
            resp = await client.post(url, json=payload)
            done_at = time.perf_counter()
            bucket = report.cold_samples if len(report.cold_samples) < cold_count else report.warm_samples
            bucket.append(Sample(
                request_sent_at_s=sent_at,
                first_byte_at_s=done_at,          # first-byte==done for non-streaming
                done_at_s=done_at,
                input_tokens=payload.get("input_tokens", -1),
                output_tokens=resp.json().get("output_tokens") if resp.status_code == 200 else None,
                status=resp.status_code,
            ))

        tasks: list[asyncio.Task] = []
        next_fire = time.perf_counter()
        while time.perf_counter() < end_at:
            payload = random.choice(traffic)
            tasks.append(asyncio.create_task(fire(payload)))
            next_fire += random.expovariate(qps)
            sleep_s = max(0.0, next_fire - time.perf_counter())
            await asyncio.sleep(sleep_s)
        await asyncio.gather(*tasks, return_exceptions=True)
    return report


def bootstrap_ci(values: list[float], pct: float, n_boot: int = 1000, alpha: float = 0.05,
                 rng: random.Random | None = None) -> tuple[float, float]:
    """Paired bootstrap for a percentile; Dror et al. 2018."""
    assert values, "cannot bootstrap empty sample"
    rng = rng or random.Random(0)
    n = len(values)
    draws: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        sample.sort()
        idx = min(n - 1, int(round(pct * (n - 1))))
        draws.append(sample[idx])
    draws.sort()
    lo = draws[int(alpha / 2 * n_boot)]
    hi = draws[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi
```

Points worth defending in review:

- **Open-loop load with Poisson arrivals** is the only honest measurement. Closed-loop load
  back-pressures slow responses and hides tail latency (coordinated omission); Gil Tene's talk
  and `wrk2` document this at length.
- **`cold_count=50` first requests separated from warm.** A service with lazy weight-loading,
  JIT compilation, or first-batch ONNX graph optimisation can have a 10× cold/warm ratio;
  reporting only the warm number lies about restart recovery.
- **Bootstrap CIs on percentiles.** A single p95 draw is noisy; paired-bootstrap CIs from
  Dror et al. 2018 translate directly.

### 2.3 Variant 0 — baseline (fp32, batch size 1)

```python
# src/latency_tuning/variants/v0_baseline.py
from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load(model_name: str, revision: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
    mdl = mdl.to(device).eval()
    return tok, mdl


@torch.inference_mode()
def infer(tok, mdl, texts: list[str], device: str) -> list[int]:
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    logits = mdl(**enc).logits
    return logits.argmax(dim=-1).tolist()
```

### 2.4 Variant 1 — bf16 autocast (A100/H100; use fp16 on T4/V100)

```python
# src/latency_tuning/variants/v1_bf16.py
import torch
from .v0_baseline import load as base_load


def load(model_name: str, revision: str, device: str):
    tok, mdl = base_load(model_name, revision, device)
    # Weights stay fp32; autocast casts activations. This is the safe default.
    return tok, mdl


@torch.inference_mode()
def infer(tok, mdl, texts: list[str], device: str) -> list[int]:
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                        dtype=torch.bfloat16):
        logits = mdl(**enc).logits
    return logits.argmax(dim=-1).tolist()
```

A learner that converts weights to bf16 permanently (`mdl.bfloat16()`) is not wrong, but should
re-measure accuracy; autocast is the lower-risk variant.

### 2.5 Variant 2 — length bucketing

```python
# src/latency_tuning/variants/v2_length_bucketing.py
from __future__ import annotations
from collections import defaultdict
import torch


BUCKETS = (32, 64, 128, 256, 512)   # token-length ceilings


def bucket_for(length: int) -> int:
    for b in BUCKETS:
        if length <= b:
            return b
    return BUCKETS[-1]


@torch.inference_mode()
def infer_batched(tok, mdl, texts: list[str], device: str) -> list[int]:
    # Tokenise once to know lengths, then bucket.
    lengths = [len(tok.encode(t, add_special_tokens=True, truncation=True, max_length=512))
               for t in texts]
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, L in enumerate(lengths):
        buckets[bucket_for(L)].append(i)

    out = [-1] * len(texts)
    for max_len, indices in buckets.items():
        batch = [texts[i] for i in indices]
        enc = tok(batch, padding="max_length", truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        logits = mdl(**enc).logits
        preds = logits.argmax(dim=-1).tolist()
        for idx, pred in zip(indices, preds):
            out[idx] = pred
    return out
```

Padding to `max_length=512` on every sequence wastes `(512 - L)` positions of compute per item.
Bucketing shrinks the padded length to the bucket ceiling; the win is proportional to the
right-skew of your length distribution and is typically 2–5× for mixed-length NLP traffic.

### 2.6 Variant 3 — dynamic batching with a Triton-style configuration

A Python-side equivalent for environments without Triton; the `max_queue_delay_ms` plus
`preferred_batch_size` is the pattern Triton exposes:

```python
# src/latency_tuning/variants/v3_dynamic_batching.py
from __future__ import annotations
import asyncio, time
from collections import deque


class DynamicBatcher:
    """
    Accumulates single-request calls into batches of up to `max_batch` within `max_delay_s`.
    Mirrors Triton's dynamic batching: trade a small wait for a fat batch when the queue is hot.
    """

    def __init__(self, infer_fn, max_batch: int = 16, max_delay_s: float = 0.008):
        self._infer = infer_fn
        self._max_batch = max_batch
        self._max_delay = max_delay_s
        self._queue: deque[tuple[str, asyncio.Future]] = deque()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    async def submit(self, text: str) -> int:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        async with self._lock:
            self._queue.append((text, fut))
            if self._task is None:
                self._task = asyncio.create_task(self._drain_loop())
        return await fut

    async def _drain_loop(self) -> None:
        try:
            while True:
                deadline = time.perf_counter() + self._max_delay
                while True:
                    async with self._lock:
                        qsize = len(self._queue)
                    if qsize >= self._max_batch or time.perf_counter() >= deadline:
                        break
                    await asyncio.sleep(0.0005)
                async with self._lock:
                    if not self._queue:
                        self._task = None
                        return
                    take = []
                    while self._queue and len(take) < self._max_batch:
                        take.append(self._queue.popleft())
                texts = [t for t, _ in take]
                preds = self._infer(texts)
                for (_, fut), pred in zip(take, preds):
                    fut.set_result(pred)
        except Exception as exc:
            # Fail all pending requests to prevent hang.
            async with self._lock:
                for _, fut in self._queue:
                    if not fut.done():
                        fut.set_exception(exc)
                self._queue.clear()
                self._task = None
            raise
```

The win is throughput-via-batching with bounded added latency; the knobs are `max_batch` and
`max_delay_s`. Triton exposes the same two knobs (`preferred_batch_size` and
`max_queue_delay_microseconds`) in a model's `config.pbtxt`.

### 2.7 Variant 4 — KV-cache for autoregressive generation

```python
# src/latency_tuning/variants/v4_kv_cache.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load(model_name: str, revision: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    mdl = AutoModelForCausalLM.from_pretrained(model_name, revision=revision).to(device).eval()
    return tok, mdl


@torch.inference_mode()
def generate(tok, mdl, prompts: list[str], device: str, max_new_tokens: int = 128) -> list[str]:
    enc = tok(prompts, padding=True, return_tensors="pt").to(device)
    out = mdl.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        use_cache=True,                 # KV-cache ON — THE change under test
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)
```

The same function with `use_cache=False` is the baseline. The latency delta is typically
8–40× for prompts in the 512-token range with 128-token continuation; it is the single
largest single-flag win in decoder serving.

For a serving variant with **continuous batching**, hand the work to vLLM / TGI / Triton +
TensorRT-LLM; the Python-side re-implementation is not production-grade and the published
implementations handle paged KV + eviction correctly.

### 2.8 Variant 5 — ONNX Runtime export + graph optimisation

```python
# src/latency_tuning/variants/v5_onnxruntime.py
from __future__ import annotations
import pathlib, numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def load(model_dir: pathlib.Path, provider: str = "CUDAExecutionProvider"):
    tok = AutoTokenizer.from_pretrained(model_dir)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_dir / "model.onnx"), so, providers=[provider])
    return tok, sess


def infer(tok, sess: "ort.InferenceSession", texts: list[str]) -> list[int]:
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    out = sess.run(None, {k: v.astype(np.int64) for k, v in enc.items()})
    return out[0].argmax(axis=-1).tolist()
```

Export is covered in exercise-03. In the latency harness, variant 5 is just "use the ONNX
session instead of the torch model." The latency delta is typically 1.3–2× over torch-eager
on CPU (graph fusion + MLAS) and modest on GPU (where torch.compile + CUDA graphs can close
the gap).

### 2.9 Report format

A reviewable report is a single markdown table. Example shape (numbers are illustrative; a real
submission reports its own measured numbers from the harness):

```markdown
| Variant | Change | p50 ms | p95 ms | p99 ms | QPS | Accuracy Δ |
|---|---|---|---|---|---|---|
| v0 | baseline fp32, batch=1                 |  <n> | <n> | <n> | <n> | 0 |
| v1 | bf16 autocast (A100)                   |  <n> | <n> | <n> | <n> | <Δ ≤ 0.2 pp> |
| v2 | length bucketing                       |  <n> | <n> | <n> | <n> | 0 |
| v3 | dynamic batching (bs=16, delay=8ms)    |  <n> | <n> | <n> | <n> | 0 |
| v4 | KV-cache ON (decoder)                  |  <n> | <n> | <n> | <n> | 0 |
| v5 | onnxruntime + graph optim              |  <n> | <n> | <n> | <n> | <Δ ≤ 0.2 pp> |
```

Each row is a separate load-test run; percentiles come with bootstrap CIs; the "accuracy Δ" column
is non-zero only for variants that touch numerics (bf16, fp16, int8, ORT quantisation).

## 3. Validation steps

1. **Measurement harness correctness.** Hand `load_gen.open_loop` a service that sleeps a known
   distribution (`uniform(10, 20)` ms, say); the reported percentiles must recover the
   theoretical percentiles within bootstrap CI. Catches harness bugs before model bugs.
2. **Coordinated-omission test.** Run the harness against a service that stalls for 2 s on 5 %
   of requests; open-loop must show p95 ≈ 2 000 ms, closed-loop will under-report. A harness
   that passes closed-loop and fails this test is producing fiction.
3. **Cold vs warm separation.** The first 50-request cold bucket must show measurably higher
   p99 than the warm bucket on a fresh process; if not, either the model is not actually
   lazy-initialising or the cold count is too small.
4. **Variant isolation.** Each variant is a diff against `v0`; running the harness against v0
   twice should yield percentiles within overlapping CIs. Non-overlapping CIs on repeated runs
   means the harness is not deterministic enough to attribute wins.
5. **Accuracy re-check on numerics-touching variants.** v1 (bf16), v5 (ORT), any int8 variant
   must publish accuracy on the mod-111 validation slice. The accepted tolerance is
   task-dependent; for classification, ≤ 0.2 pp macro-F1 drop is a soft threshold.
6. **Truncation counter.** The service logs when `max_length` truncates an input. Load a
   long-tail input corpus; the counter must be non-zero and the p95 must not improve by
   "truncating everything to 32 tokens." Silent truncation is a correctness bug, not a win.
7. **GPU utilisation during batching variants.** `nvidia-smi dmon -s u` during a v3 run must
   show GPU-util > 60 %; if it hovers near 10 %, the batcher is not actually batching (queue
   is empty) and the test is degenerate — raise QPS and re-run.
8. **KV-cache sanity.** v4 `use_cache=True` vs `use_cache=False` on the same prompt must
   produce bit-identical output tokens for greedy decoding. If they differ, KV-cache
   configuration is wrong (positional embeddings off, pad tokens leaking in).
9. **ONNX output parity.** v5 ORT output logits must match torch output logits within
   `atol=1e-3, rtol=1e-3` on a 100-item sample; a bigger gap is an export bug (see exercise-03).
10. **Load-test reproducibility.** Run v0 on the same hardware twice at different times of day;
    percentiles within overlapping CIs is pass. If not, note the confounder (shared GPU,
    noisy neighbour) in the report; benchmarks on shared hardware have a caveat.

## 4. Grading rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Baseline measurement** | p50/p95/p99/max with bootstrap CIs; cold + warm split; open-loop load | Percentiles no CIs | Mean only |
| **Load generator** | Open-loop Poisson arrivals; coord-omission-safe; input-length distribution from a real traffic profile | Open-loop, fixed-length inputs | Closed-loop |
| **Variant isolation** | Each variant is one change against v0; variants are repeatable within overlapping CIs | Variants named, not isolated | Multiple changes bundled |
| **KV-cache (decoder)** | `use_cache=True` pathway + equivalence check vs `use_cache=False` | KV-cache on, no equivalence check | Not tested |
| **Batching variant** | Dynamic batching or length bucketing with a configured `max_delay`; GPU-util verified | Static batch | No batching |
| **Precision variant** | bf16/fp16 autocast or int8; accuracy delta re-measured and reported | Precision changed, no accuracy check | Mixed precision toggled without any validation |
| **ONNX / compiled variant** | ORT export with parity test; graph optimisation on; execution provider named | ORT without parity test | None, or ORT with silent fp mismatch |
| **Report** | Markdown table + CSV; one row per variant; accuracy Δ column present for numerics variants | Table, no CSV, or no accuracy column | Narrative only |
| **Truncation / drop logging** | `max_length` truncation counter published alongside percentiles | Mentioned in README, not in report | Silent truncation |
| **Reproducibility** | Harness is scripted (`make bench` or CLI); hardware, kernel, torch/cuda versions recorded | Scripted, no hardware record | Manual, not reproducible |

## 5. Common mistakes

- **Reporting mean latency.** Mean is the number stakeholders ask for and the number you should
  refuse to publish alone. p50 / p95 / p99 / max with CIs is the deliverable.
- **Closed-loop load generators ("hey I run it in a tight loop and divide").** Closed-loop
  back-pressures slow responses and hides the tail. Use `wrk2`, `k6`, or an open-loop asyncio
  generator with Poisson arrivals.
- **Pre-warming before every measurement.** Pre-warming the service before publishing steady-
  state is fine *if* production has the same pre-warm. Pre-warming to inflate your numbers
  without the matching production pre-warm is dishonest measurement.
- **`torch.compile()` without re-warming.** The first call triggers compilation; include it in
  the cold bucket. Omitting the compilation cost from the cold number lies about restart cost.
- **Dynamic batching with no queue-delay cap.** A batcher that waits for a full batch starves
  under low QPS. Always cap `max_delay_s`; typical values: 2–10 ms.
- **Converting weights to fp16/bf16 and never re-measuring accuracy.** A quiet 2 pp regression
  in production is the modal story. Autocast + accuracy re-measure, or weight conversion +
  accuracy re-measure; never weight conversion alone.
- **`torch.inference_mode()` missed.** Running without `inference_mode()` or `no_grad()` keeps
  autograd state and uses ~2× the GPU memory; latency and throughput both suffer.
- **Measuring on the dev laptop and extrapolating.** The whole exercise is useless unless the
  hardware matches target. Record the exact GPU / CPU / driver / torch / cuda versions in the
  report header.
- **Ignoring tokenizer latency.** For fast models on short inputs, tokenisation is a measurable
  fraction of p50. Use `use_fast=True` on HF tokenizers; prefer `AutoTokenizer.from_pretrained(
  ..., use_fast=True)`.
- **Changing `max_length` without logging truncation.** A learner who ships "p95 dropped 40 %"
  without noticing that 30 % of inputs are now truncated has broken correctness, not improved
  latency.
- **Replacing the service with ONNX and skipping the parity test.** Operator differences and
  fp32-vs-fp16-vs-fp32 cast ordering at export time can silently change outputs. Always
  compare logits on a sample before trusting the latency delta.
- **Continuous-batching variant claimed, not implemented.** Hand-rolled "continuous batching"
  in Python without paged-KV is not continuous batching; it's dynamic batching plus hope.
  Reference vLLM / TGI / Triton+TensorRT-LLM.
- **Un-attributable variant stack.** "v6 = v1+v3+v5" bundled; impossible to tell which change
  matters. Isolate.

## 6. References

Primary sources used in this solution:

- Hugging Face `text_generation` docs. <https://huggingface.co/docs/transformers/main_classes/text_generation>
  — `use_cache`, `max_new_tokens`, `num_beams`, `pad_token_id`.
- PyTorch Automatic Mixed Precision reference. <https://pytorch.org/docs/stable/amp.html> —
  `torch.autocast`, `bfloat16`, `float16`.
- ONNX Runtime documentation. <https://onnxruntime.ai/docs/> — execution providers, session
  options, graph optimisation levels.
- NVIDIA Triton Inference Server model configuration docs.
  <https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md>
  — `dynamic_batching { preferred_batch_size, max_queue_delay_microseconds }`.
- Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with
  PagedAttention*. SOSP. <https://arxiv.org/abs/2309.06180> — the continuous-batching + paged-KV
  reference; vLLM's design document.
- vLLM documentation. <https://docs.vllm.ai/>.
- Hugging Face Text Generation Inference. <https://github.com/huggingface/text-generation-inference>
  — the HF continuous-batching server used by many stacks.
- Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work
  Partitioning*. <https://arxiv.org/abs/2307.08691>.
- Dror, R. et al. (2018). *The Hitchhiker's Guide to Testing Statistical Significance in NLP*.
  ACL. <https://aclanthology.org/P18-1128/> — the paired-bootstrap methodology.
- Gil Tene, *How NOT to Measure Latency*. <https://www.youtube.com/watch?v=lJ8ydIuPFeU> — the
  coordinated-omission reference that drives the open-loop load argument.
- `wrk2`. <https://github.com/giltene/wrk2>; `k6`. <https://k6.io/docs/>; `locust`.
  <https://docs.locust.io/>.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/>
  — *ML04 Membership Inference Attack* and *ML02 Data Poisoning* are not directly latency
  concerns; however, *latency side-channels* can leak membership information (differences in
  cache-hit vs cache-miss time per input). Constant-time inference is impractical for large
  models, but deliberately stabilising p50/p99 ratios and monitoring input-conditional latency
  is a mitigation to be aware of.
- MITRE ATLAS — <https://atlas.mitre.org/> — *AML.T0029 Denial of ML Service* applies directly:
  an attacker crafts long-prefix inputs that dominate the dynamic-batching queue, degrading
  p99 for everyone else. `max_length` bucketing + per-tenant rate-limits + a max-tokens cap
  are the mitigations; the harness is the detector (a synthetic DoS scenario should be part of
  the benchmark suite).
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework>
  — *Map*: the variant list and traffic profile. *Measure*: percentile report with CIs.
  *Manage*: the release gate ties a p99 SLO + an accuracy floor; any variant that improves p99
  but regresses accuracy past the floor is blocked. *Govern*: hardware + versions recorded in
  every report so prior numbers remain comparable.
