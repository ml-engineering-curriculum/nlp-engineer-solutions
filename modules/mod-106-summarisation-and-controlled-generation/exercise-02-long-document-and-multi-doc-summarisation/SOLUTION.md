# exercise-02 — Long-Document and Multi-Document Summarisation

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-106-summarisation-and-controlled-generation/exercises/exercise-02-long-document-and-multi-doc-summarisation.md`.
> The target learning outcomes implied by that filename: summarise inputs that do not fit in a 1024-token encoder — both a **single long document** (e.g. an arXiv paper or `govreport` legal brief) and a **cluster of related documents** (e.g. `multi_news`) — using one of two canonical strategies (**long-context** model with sparse/linear attention, or **chunk-and-combine** map-reduce over a short-context model), and evaluate with the length-scaled ROUGE protocol the long-doc papers use.

## 1. Solution overview

Long-input summarisation is a *routing problem* dressed up as a *modelling problem*. The three failure modes that decide whether the pipeline is honest:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Truncation hides everything downstream** | A 16 k-token arXiv paper fed through BART-base (1024-token encoder) loses 94 % of the body; the summary looks fluent and ROUGE looks mediocre, with no error signal that the model *never saw* the key sections | Make the ingest length check explicit: log distribution of input lengths, fail loud when >N % of the batch is truncated, and pick a routing strategy per document length |
| **Chunking without redundancy / dedup** | Naïve map-reduce (split → summarise each → concat) emits the same point from every chunk, loses cross-chunk entities, and scores worse than vanilla truncation | Overlap chunks (stride), carry a running **brief** of what has been summarised so far as additional conditioning, and run an explicit dedup / reduce stage |
| **Multi-doc input order leaks into the output** | On `multi_news` the concatenation order determines which article the model copies from; same documents in different order → different summary, which is a reproducibility and fairness issue | Sort documents by a deterministic key (publication date, source id, or length), record the order in the run metadata, and sanity-check with order-invariance tests |

Two canonical strategies, both delivered here:

1. **Long-context model.** `allenai/led-base-16384` (Longformer Encoder-Decoder, Beltagy et al. 2020) — a BART variant with sliding-window + global attention that accepts up to 16 384 input tokens. Drop-in replacement for the exercise-01 pipeline; the only new mechanic is **global attention masks**, which must be set on BOS, section headers, or other spans the model should be able to attend to from everywhere.
2. **Map-reduce / chunk-and-combine.** Split the document into overlapping chunks that fit in a short-context model (BART-base at 1024 tokens), summarise each, then run a second-stage *reduce* model that compresses the concatenated chunk summaries into a final answer. Works with any short-context summariser; the trade-off is two forward passes per document and a dedup stage.

Pick the strategy per corpus length: **long-context** when P99 input length is 2–16 k tokens; **map-reduce** when P99 > 16 k, or when latency per document is critical and you can parallelise chunks.

Reference datasets:
- **arXiv** and **PubMed** (Cohan et al. 2018) — ~215 k scientific papers paired with their abstracts; P95 length ~8–16 k tokens.
- **GovReport** (Huang et al. 2021) — 19 k congressional / GAO reports paired with multi-paragraph executive summaries; P95 ~9 k tokens, longest ~150 k.
- **Multi-News** (Fabbri et al. 2019) — 56 k news events with 2–10 source articles each and a human-written summary; the canonical multi-document benchmark.

Reference models:
- **LED-base-16384** (Beltagy et al. 2020) via `AutoModelForSeq2SeqLM`.
- **BigBird-Pegasus** (Zaheer et al. 2020) as an alternative long-context model.
- Any exercise-01 BART/T5 checkpoint, used as the map / reduce model in the chunking strategy.

## 2. Worked implementation

### 2.1 Project layout

```
longsum/
  data.py            # dataset load + length audit
  routing.py         # per-document strategy selection
  led.py             # long-context fine-tune (LED)
  chunk.py           # map-reduce over a short-context model
  dedup.py           # cross-chunk sentence deduplication
  metrics.py         # ROUGE with the long-doc protocol
  run.py             # CLI
  tests/
    test_routing.py
    test_chunk.py
    test_dedup.py
    test_global_attention.py
  requirements.txt
```

### 2.2 Length audit — the first thing you ship

```python
# data.py
from datasets import load_dataset
import numpy as np


def load_arxiv():
    return load_dataset("scientific_papers", "arxiv")


def length_histogram(ds, tokenizer, split="train", key="article", sample=2000):
    xs = ds[split].shuffle(seed=0).select(range(sample))
    lens = [len(tokenizer(x[key], truncation=False)["input_ids"]) for x in xs]
    return {
        "p50": int(np.percentile(lens, 50)),
        "p95": int(np.percentile(lens, 95)),
        "p99": int(np.percentile(lens, 99)),
        "max": int(max(lens)),
        "pct_over_1024": float((np.array(lens) > 1024).mean()),
        "pct_over_16384": float((np.array(lens) > 16384).mean()),
    }
```

Reviewers should expect this report to be **printed to the training log**, not just available as a utility. Three numbers carry the design:

- If `pct_over_1024 > 0.3`, exercise-01's BART pipeline silently truncates a third of inputs; routing is required.
- If `pct_over_16384 < 0.05`, LED-16384 is sufficient.
- If `pct_over_16384 > 0.2`, map-reduce is required (or LED-16384 with hierarchical chunking at ingest).

### 2.3 Routing

```python
# routing.py
from typing import Literal

Strategy = Literal["direct", "led", "map_reduce"]


def pick_strategy(n_tokens: int) -> Strategy:
    if n_tokens <= 1024:
        return "direct"       # exercise-01 pipeline
    if n_tokens <= 16384:
        return "led"
    return "map_reduce"
```

A single integer check with a documented table is the honest version of "we chose LED because it's long-context". Reviewers should push back on any strategy choice that is not length-driven.

### 2.4 Long-context path: LED with global attention

```python
# led.py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LED_ID = "allenai/led-base-16384"
MAX_INPUT_LEN = 16384
MAX_TARGET_LEN = 512


def build_led():
    tok = AutoTokenizer.from_pretrained(LED_ID, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(LED_ID)
    return model, tok


def preprocess_led(examples, tokenizer):
    model_inputs = tokenizer(
        examples["article"],
        max_length=MAX_INPUT_LEN,
        truncation=True,
    )
    labels = tokenizer(
        text_target=examples["abstract"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
    )
    # Global attention: every token can attend to a *global*-marked token
    # from any position. The LED paper (Beltagy et al. 2020 §3.2) sets
    # global attention on <s> (BOS). For section-aware summarisation,
    # also set global on recognised section-header tokens.
    global_attention_mask = []
    for ids in model_inputs["input_ids"]:
        mask = [0] * len(ids)
        mask[0] = 1               # BOS token is always global
        global_attention_mask.append(mask)
    model_inputs["global_attention_mask"] = global_attention_mask
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

The single most common "LED didn't help" mistake is omitting `global_attention_mask`. Without any global tokens, LED collapses to local sliding-window attention over the whole document — the model physically cannot aggregate across chunk boundaries. The paper sets global on BOS; for domain-specific inputs, also mark section anchors (e.g. `"Introduction"`, `"Methods"`) if the tokeniser emits them as single tokens.

### 2.5 Map-reduce path: chunk → summarise → dedup → summarise

```python
# chunk.py
from typing import Iterator, List


def chunk_with_overlap(
    tokens: List[int], *, chunk_size: int = 896, stride: int = 128
) -> Iterator[List[int]]:
    i = 0
    while i < len(tokens):
        yield tokens[i : i + chunk_size]
        if i + chunk_size >= len(tokens):
            return
        i += chunk_size - stride


def map_summarise(text: str, model, tokenizer, max_new_tokens: int = 128) -> str:
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).input_ids
    out = model.generate(ids, max_new_tokens=max_new_tokens, num_beams=4,
                         length_penalty=2.0, no_repeat_ngram_size=3)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def reduce_summarise(partials: List[str], model, tokenizer, *, max_new_tokens: int = 256) -> str:
    joined = " ".join(partials)
    # Re-apply the same short-context summariser on the concat of partials.
    return map_summarise(joined, model, tokenizer, max_new_tokens=max_new_tokens)


def map_reduce_summarise(doc: str, model, tokenizer, *, chunk_size: int = 896, stride: int = 128):
    ids = tokenizer(doc, add_special_tokens=False)["input_ids"]
    chunks = [tokenizer.decode(c, skip_special_tokens=True) for c in chunk_with_overlap(ids, chunk_size=chunk_size, stride=stride)]
    partials = [map_summarise(c, model, tokenizer) for c in chunks]
    partials = dedup_sentences(partials)
    return reduce_summarise(partials, model, tokenizer)
```

Design decisions worth defending:

- **`chunk_size=896, stride=128`.** 896 + ~128 tokens of prompt / separator gives ≤1024 after re-encoding; 128-token stride ensures a sentence spanning chunks is seen in at least one chunk in full.
- **Decode → re-encode for chunks.** Splitting tokens and decoding on the way out is safer than feeding raw token IDs into `.generate()`, because the tokeniser may add special tokens or merge subwords differently.
- **Same model for map and reduce.** Keeps the pipeline reproducible. Using a stronger reduce model is fine and often helps, but adds a dependency.

```python
# dedup.py
import nltk


def dedup_sentences(partials: list[str]) -> list[str]:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    seen = set()
    out = []
    for partial in partials:
        kept = []
        for sent in nltk.sent_tokenize(partial):
            key = " ".join(sent.lower().split())
            if key in seen:
                continue
            seen.add(key)
            kept.append(sent)
        if kept:
            out.append(" ".join(kept))
    return out
```

Exact-dedup at the sentence level removes the "every chunk mentions Section 1" failure mode. A stronger version uses MinHash or sentence-embedding cosine with a 0.85 cutoff; the exact match is enough for a reference solution and avoids a new dependency.

### 2.6 Multi-document inputs

```python
# data.py (continued)
def flatten_multi_news(example):
    # multi_news stores articles separated by `|||||`. The flatten step must
    # choose a deterministic order so eval is reproducible — we sort by length
    # (longest first) to keep the most informative article closer to the model.
    articles = [a.strip() for a in example["document"].split("|||||") if a.strip()]
    articles.sort(key=len, reverse=True)
    return {"article": "\n\n".join(articles), "summary": example["summary"]}
```

The ordering choice should be documented in the run metadata. Alternatives — chronological, alphabetical source id, random with a fixed seed — are all fine; silent order is not.

### 2.7 Evaluation — long-doc ROUGE protocol

The ROUGE pipeline from exercise-01 applies unchanged for the lexical metrics. Long-document papers add two reporting conventions:

- **Report ROUGE on the truncated reference you actually fed to the model.** If `MAX_TARGET_LEN` is 512 and the gold abstract is 800 tokens, scoring against the full abstract is fine but should be noted — short outputs will cap recall.
- **Report chunk-coverage stats.** On map-reduce: fraction of chunks whose partial summary contributed at least one sentence to the final reduce output. A coverage below 30 % means the reduce stage is throwing away most of the work; a coverage of 100 % means dedup isn't firing.

```python
# metrics.py (continued)
def chunk_coverage(final_summary: str, partials: list[str]) -> float:
    final_sents = {" ".join(s.lower().split()) for s in nltk.sent_tokenize(final_summary)}
    kept = sum(1 for p in partials if any(
        " ".join(s.lower().split()) in final_sents for s in nltk.sent_tokenize(p)))
    return kept / max(1, len(partials))
```

### 2.8 CLI wiring

```python
# run.py
from .data import load_arxiv, length_histogram
from .routing import pick_strategy
from .led import build_led, preprocess_led
from .chunk import map_reduce_summarise
# ... assemble per the exercise-01 Trainer pattern, branching on pick_strategy.
```

## 3. Validation steps

1. **Length histogram printed.** On arXiv (train, 2000-sample): expect roughly `p50≈6k, p95≈15k, p99≈20k, pct_over_1024≈0.98`. On `multi_news`: `p95≈3k`. If your numbers are wildly off, your tokeniser isn't the one you think it is.
2. **Routing boundary test.** `pick_strategy(1024) == "direct"`; `pick_strategy(1025) == "led"`; `pick_strategy(16385) == "map_reduce"`. One-line table-driven test.
3. **Global attention sanity.** Build a 2000-token input, run LED with `global_attention_mask` all zeros vs `global_attention_mask[0]=1`. ROUGE should jump by 1–3 points with global on BOS; equal numbers mean the mask isn't reaching the model (common cause: `data_collator` not forwarding the key).
4. **Chunk-overlap guarantee.** For a 2500-token document with `chunk_size=896, stride=128`: assert 4 chunks, each shares ≥128 tokens with its neighbour. Covers the single-off-by-one-step-size family of bugs.
5. **Dedup firing.** Feed three partials where the first sentence of each is identical (e.g. the paper's opening claim). `dedup_sentences` must return one copy, not three.
6. **Multi-doc order invariance.** Reverse the article list on `multi_news` and re-run. ROUGE should change by ≤0.5 points (the sort stabilises it); a 3-point swing means the sort isn't firing or the model is order-sensitive beyond the sort's dampening.
7. **Chunk coverage report.** On dev, expect coverage 50–70 % on arXiv with these defaults. A coverage of 90 %+ indicates dedup isn't deduping; <30 % means `reduce_summarise` is collapsing to a single-chunk paraphrase and raising `max_new_tokens` on the reduce stage is the fix.
8. **Seed stability (LED).** Train LED-base on arXiv for 3 epochs, 3 seeds. Expect ROUGE-L ≈ 20.9–21.5 (close to the LED paper's 20.8 for LED-base-16384 fine-tuned); variance >0.6 points usually indicates a missing global-attention mask on at least some batches.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Length audit** | `length_histogram` reported at ingest; routing driven by its percentiles | Histogram computed, not acted on | No length check; silent truncation |
| **Strategy routing** | Explicit `pick_strategy` with documented thresholds; test covers boundaries | One strategy used for all doc lengths with justification | Hand-chosen strategy with no length link |
| **Long-context (LED) correctness** | `global_attention_mask` set on BOS (and optionally headers), forwarded through collator; ablation test confirming non-trivial effect | Mask present without ablation | No global attention; LED reduces to local attention |
| **Map-reduce correctness** | Overlapping chunks with documented stride; explicit dedup stage; reduce stage with higher `max_new_tokens` than map | Chunking present; no dedup | Non-overlapping chunks, no dedup, or reduce=map concatenation |
| **Multi-document ordering** | Deterministic sort key recorded; order-invariance test within tolerance | One ordering, undocumented | Random / input-dependent order |
| **ROUGE protocol** | Same long-doc convention as exercise-01 (stemming, `rougeLsum` sentence-split, F1, bootstrap CI); chunk-coverage reported for map-reduce | ROUGE only | `pyrouge` defaults or inflated protocol |
| **Dataset hygiene** | Dataset licence recorded (`scientific_papers`: `cc-by-sa-nc-4.0`; `govreport`: public domain US government; `multi_news`: Apache 2.0 via `datasets`) | One of those | Neither |
| **Reproducibility** | `set_seed`; routing table, generation config, chunk size, stride, ordering key echoed in run JSON | Seed set, no echo | No seed |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks for multi-source input (injection via one adversarial source article; mixed-tenant leakage) with mitigations | Mentioned without mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure: dataset licences, language scope, routing policy, chunking params, known failure mode (section ordering) | One of those | Neither |

## 5. Common mistakes

- **Truncating to 1024 and calling it long-doc.** If you feed arXiv into a BART-base without routing, you are evaluating a short-doc model — the "long-doc" label is a lie. The length audit in §2.2 is the fix.
- **LED without `global_attention_mask`.** Local sliding-window attention over 16 k tokens cannot aggregate; your ROUGE barely moves above BART-base. Always set global on BOS at minimum.
- **Not forwarding `global_attention_mask` through the collator.** `DataCollatorForSeq2Seq` does not know about LED's global mask by default and drops it unless the input key is preserved. Verify with a batch dump before training.
- **Non-overlapping chunks.** A cross-chunk sentence lands half in each chunk and the model's output is correct for neither. Stride of at least one sentence length is the fix.
- **Reduce stage with the same `max_new_tokens` as map.** The reduce input is N partials, each ~128 tokens; compressing N×128 tokens into 128 tokens throws away >80 % of the map stage's work. Raise `max_new_tokens` on reduce (256–512 for arXiv; 64 for multi-news).
- **Reduce is "concat partials" with no model pass.** This is the documented failure mode of naïve map-reduce: duplicate points, no cross-chunk coherence, worse than vanilla truncation. The reduce model's job is to compress AND dedupe.
- **Scoring against the un-truncated gold while `MAX_TARGET_LEN` is clipping outputs.** Recall caps at ~`MAX_TARGET_LEN / len(gold)`. Either raise `MAX_TARGET_LEN` to the dataset P99 or report "ROUGE on truncated gold" explicitly.
- **Ignoring multi-doc order sensitivity.** Random ordering across runs produces irreproducible metrics *and* the first article in the list biases the summary. Sort deterministically.
- **Running `.generate()` with FP16 on LED without `use_cache=True` guarded.** LED's sparse-attention kernels interact badly with the default caching path in some `transformers` versions; if generation is 10× slower than training throughput would predict, set `use_cache=False` on LED or upgrade `transformers`.
- **Scoring a 400-token gold summary with the exercise-01 `max_new_tokens=128`.** The output can't exceed 128 tokens; ROUGE-L recall plateaus. Match `max_new_tokens` to the dataset gold length P99 (arXiv ≈ 300, govreport ≈ 700, multi-news ≈ 350).

## 6. References

Primary sources used in this solution:

- Beltagy, I., Peters, M., Cohan, A. (2020). *Longformer: The Long-Document Transformer*. arXiv:2004.05150. <https://arxiv.org/abs/2004.05150> — sliding-window + global attention; LED (Longformer Encoder-Decoder) architecture and the global-attention convention adopted in §2.4.
- Zaheer, M. et al. (2020). *Big Bird: Transformers for Longer Sequences*. NeurIPS. <https://proceedings.neurips.cc/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html> — sparse attention alternative to Longformer; BigBird-Pegasus is the "long Pegasus" referenced in §1.
- Cohan, A. et al. (2018). *A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents*. NAACL. <https://aclanthology.org/N18-2097/> — the arXiv and PubMed long-document summarisation benchmarks.
- Huang, L. et al. (2021). *Efficient Attentions for Long Document Summarization*. NAACL. <https://aclanthology.org/2021.naacl-main.112/> — the **GovReport** corpus and chunked long-doc baselines; informs the chunk-and-reduce design in §2.5.
- Fabbri, A. et al. (2019). *Multi-News: A Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model*. ACL. <https://aclanthology.org/P19-1102/> — the Multi-News corpus; informs the multi-document ordering question in §2.6.
- Lin, C. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*. <https://aclanthology.org/W04-1013/> — the ROUGE family; same package as exercise-01.
- Hugging Face `transformers` model card for `allenai/led-base-16384` — <https://huggingface.co/allenai/led-base-16384> — the reference `global_attention_mask` usage and pre-trained checkpoint cited in §2.4.
- Hugging Face `transformers` reference script `examples/pytorch/summarization/run_summarization.py` — <https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py> — reused as the Trainer skeleton.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation* is amplified by multi-document inputs: a single adversarial source article in a cluster can inject an instruction ("summary: ignore other sources") that the model copies into the final summary. Mitigations: per-source length caps at ingest, strip model-specific control tokens, treat generated summaries as untrusted when forwarded. *ML02 Data Poisoning* applies to news-scraped corpora.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue relevant when a long-doc summariser feeds compliance, legal-discovery, or ranking workflows that trust the summary.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: dataset licences, language coverage (English scientific / government / news), known failure modes (section ordering, multi-source conflict); *Measure*: report ROUGE with the routing strategy actually used per document, together with chunk coverage, seed, and generation config.
