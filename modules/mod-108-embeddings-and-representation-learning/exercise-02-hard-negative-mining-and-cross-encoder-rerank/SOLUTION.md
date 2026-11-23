# exercise-02 — Hard-Negative Mining and Cross-Encoder Rerank

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-108-embeddings-and-representation-learning/exercises/exercise-02-hard-negative-mining-and-cross-encoder-rerank.md`.
> The target learning outcomes implied by that filename: take the bi-encoder from exercise-01, **mine hard negatives** with a first-pass retriever (BM25 or the bi-encoder itself), **fine-tune the bi-encoder with those hard negatives** using a margin-aware contrastive loss, then wire a **cross-encoder reranker** over the bi-encoder's top-k and measure the **retrieval → rerank lift** on MRR@10 / nDCG@10. Filter false negatives (near-duplicates of the positive) before training, because the margin loss trusts whatever you label as "negative".

## 1. Solution overview

The bi-encoder trained in exercise-01 gets most of its signal from easy, random in-batch negatives. The tail of hard queries — where the correct passage ranks 20–100 instead of 1 — only improves with (a) *hard* negatives that look lexically similar to the positive but are wrong, and (b) a *cross-encoder* reranker that reads the query and the candidate passage together instead of independently.

Three failure modes decide whether the pipeline actually lifts:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **False negatives in mined hard negatives** | A BM25-mined "hard negative" is often a *paraphrase* of the gold passage with no label. The loss pushes the embedding away from the true positive | Score mined negatives against the gold positive with a stronger cross-encoder (ms-marco-MiniLM or a GPT judge) and **drop negatives whose relevance score exceeds a threshold** (ANCE §4.3; the "false negative filter" step) |
| **Rerank-over-too-few-candidates** | The bi-encoder returns top-10, you rerank 10; if the gold passage was at rank 23 it is gone | Rerank over top-100 (or top-200) from the bi-encoder. Latency grows with `k`, but nDCG@10 grows faster. Published two-stage systems rerank 100–1000 candidates |
| **Rerank score reported as a probability without a calibration check** | The cross-encoder returns a raw logit; downstream treats it as `P(relevant)` and sets a threshold on it. Calibration drifts across domains | Return `sigmoid(logit)` only after calibrating on a dev slice (reliability diagram or expected-calibration-error); otherwise return the raw score and sort by it, do not threshold |

The deliverable is a two-stage retrieval harness with five components:

1. **BM25 first-pass index** (`pyserini` / `rank_bm25`) and a **bi-encoder first-pass index** (FAISS or an in-process `numpy` matrix). Both expose a `search(query, k)` interface returning `(doc_id, score)` tuples.
2. **Hard-negative miner** — for each training query: retrieve top-N (N=100) with the first-pass system, drop known positives, filter false negatives with a strong relevance model, sample k=7 negatives per positive.
3. **Bi-encoder fine-tuner** — Multiple-Negatives-Ranking loss *with explicit hard negatives attached* (SBERT `InputExample(texts=[query, positive, neg1, neg2, …])` form), or `MarginMSELoss` using cross-encoder scores as distillation targets (Hofstätter et al. 2020).
4. **Cross-encoder reranker** — `cross-encoder/ms-marco-MiniLM-L-6-v2` as the strong published baseline; `cross-encoder/ms-marco-TinyBERT-L-2-v2` for latency-bound deployments.
5. **Two-stage evaluator** — Recall@100 for stage-1, MRR@10 and nDCG@10 after stage-2; log the lift and the latency for both stages.

Reference numbers to pin the exercise against (published model cards, MS MARCO passage-ranking dev-small):

- `BM25` baseline: MRR@10 ≈ 0.19 (Thakur et al. 2021, BEIR Table 2).
- `sentence-transformers/msmarco-MiniLM-L-6-v3` bi-encoder: MRR@10 ≈ 0.32–0.34 (SBERT model card).
- `cross-encoder/ms-marco-MiniLM-L-6-v2` over BM25 top-100: MRR@10 ≈ 0.39 (SBERT cross-encoder model card).
- `cross-encoder/ms-marco-MiniLM-L-6-v2` over a dense bi-encoder top-100: typically reports MRR@10 in the 0.38–0.41 range depending on the first stage. Exact two-stage numbers vary across papers and model-card revisions (bi-encoder choice, top-k, evaluator settings), so cite the specific card or paper you reproduce rather than hard-pinning a single figure.

## 2. Worked implementation

### 2.1 Project layout

```
rerank_bake/
  bm25_index.py           # pyserini wrapper, Lucene BM25 index
  dense_index.py          # FAISS IndexFlatIP over bi-encoder embeddings
  mine_hard_negatives.py  # top-N retrieval + false-negative filter
  margin_loss.py          # MNR-with-hard-negatives and MarginMSE variants
  train_biencoder.py      # second-stage bi-encoder fine-tune
  rerank.py               # cross-encoder rerank over top-k
  evaluate.py             # two-stage IR metrics; latency breakdown
  tests/
    test_false_negative_filter.py
    test_rerank_latency.py
    test_lift.py
  requirements.txt
```

### 2.2 First-pass retrievers — BM25 and dense, both

```python
# bm25_index.py
from pyserini.search.lucene import LuceneSearcher


def build_bm25(index_path: str) -> LuceneSearcher:
    """
    pyserini wraps the Anserini Lucene BM25 implementation (k1=0.9, b=0.4 defaults,
    same as the MS MARCO baselines). For a non-Java-free environment, rank_bm25 is a
    pure-python fallback with slightly weaker numbers.
    """
    s = LuceneSearcher(index_path)
    s.set_bm25(k1=0.9, b=0.4)
    return s


def search_bm25(searcher, query: str, k: int = 100):
    hits = searcher.search(query, k=k)
    return [(h.docid, h.score) for h in hits]
```

```python
# dense_index.py
import numpy as np, faiss


def build_dense_index(embeddings: np.ndarray) -> faiss.Index:
    """
    IndexFlatIP over L2-normalised embeddings -> exact cosine nearest neighbours.
    For corpora >1M, use IndexHNSWFlat with M=32, efConstruction=200, efSearch=64
    to trade 0.3–0.5 nDCG@10 for 10-50x latency.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    return index


def search_dense(index: faiss.Index, query_vec: np.ndarray, k: int = 100):
    D, I = index.search(query_vec[None, :].astype(np.float32), k)
    return list(zip(I[0].tolist(), D[0].tolist()))
```

Points worth defending in review:

- **Keep both first-pass retrievers in the harness.** BM25-mined negatives and dense-mined negatives have different failure distributions; DPR and ANCE both train on a mix, and the combination out-performs either alone (Karpukhin et al. 2020 §4.4; Xiong et al. 2020).
- **`IndexFlatIP` is honest.** For dev-scale corpora (<1M) it is exact, fast, and the ranking is not perturbed by ANN approximation. Switch to HNSW or IVF-PQ only at scale, and log the Recall@k gap vs flat at swap time.
- **Unit-normalise embeddings before indexing.** `IndexFlatIP` on unnormalised vectors is dot product, which is not what the bi-encoder was trained on (exercise-01 §2.3). The `models.Normalize` module from exercise-01 guarantees this; reassert at index build.

### 2.3 Hard-negative mining — with a false-negative filter

```python
# mine_hard_negatives.py
import json, random
from pathlib import Path
from sentence_transformers import CrossEncoder


RELEVANCE_JUDGE = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # strong but affordable
FILTER_THRESHOLD = 0.7  # if judge(query, mined) > 0.7, treat as a false negative


def mine(
    queries: dict[str, str],
    qrels: dict[str, set[str]],
    first_pass_search,          # callable: (qid, k) -> list[(doc_id, score)]
    corpus: dict[str, str],
    k: int = 100,
    n_negatives: int = 7,
    seed: int = 20260101,
):
    rng = random.Random(seed)
    judge = CrossEncoder(RELEVANCE_JUDGE)
    triples = []

    for qid, query in queries.items():
        positives = qrels.get(qid, set())
        if not positives:
            continue
        candidates = [doc_id for doc_id, _ in first_pass_search(qid, k) if doc_id not in positives]

        # False-negative filter: ask the stronger judge to score every candidate
        # against the query; drop everything above the threshold.
        pairs = [(query, corpus[doc_id]) for doc_id in candidates]
        scores = judge.predict(pairs, batch_size=64, show_progress_bar=False)
        kept = [doc_id for doc_id, s in zip(candidates, scores) if s < FILTER_THRESHOLD]
        if len(kept) < n_negatives:
            continue   # not enough clean negatives; skip this query

        sampled = rng.sample(kept, n_negatives)
        for pos_id in positives:
            triples.append({
                "query": query,
                "positive": corpus[pos_id],
                "negatives": [corpus[nid] for nid in sampled],
            })
    return triples
```

Why this is the critical step:

- **`FILTER_THRESHOLD = 0.7`.** The threshold is domain-specific: on MS MARCO passage-ranking, scores above ~0.7 from `ms-marco-MiniLM-L-6-v2` are overwhelmingly relevant passages, so they are almost certainly *un-annotated* positives that would poison the loss. Tune on a 500-query held-out slice before running the full mine; log the kept-vs-dropped ratio.
- **Skip queries with too few clean negatives.** Easier than inventing synthetic negatives; the training loss handles variable `n_negatives` per query only if you pad, and padding breaks the symmetric MNR shape.
- **Mine with *multiple* retrievers for diversity.** ANCE (Xiong et al. 2020 §3.3) re-mines with the current bi-encoder every N epochs — the "approximate nearest-neighbour negative contrastive estimation" loop. A cheaper alternative is to pre-mine once with BM25 and once with the exercise-01 checkpoint, union the candidate sets, filter, and sample.
- **The judge model must be stronger than the bi-encoder.** Using the same bi-encoder to filter its own hard negatives is circular; it will keep what it already ranks low (which is often the correct ranking, not the false-negative noise). Cross-encoders win here because they read query and passage jointly.

### 2.4 Second-stage bi-encoder fine-tune — MNR with hard negatives attached

```python
# train_biencoder.py (loss wiring)
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader


def to_mnr_examples(triples):
    """
    sentence-transformers' MultipleNegativesRankingLoss accepts
    InputExample(texts=[query, positive, hard_neg_1, hard_neg_2, ...]).
    Every entry past index 1 is treated as an additional in-batch-level negative
    for that specific query.
    """
    return [
        InputExample(texts=[t["query"], t["positive"], *t["negatives"]])
        for t in triples
    ]


def build_loss(model, scale: float = 20.0):
    # Same loss class as exercise-01; the number of negative columns is inferred
    # from the first example in the batch (all examples must have the same count).
    return losses.MultipleNegativesRankingLoss(model, scale=scale)
```

Alternative — margin distillation from the cross-encoder (MarginMSE, Hofstätter et al. 2020):

```python
# margin_loss.py
import torch, torch.nn as nn


class MarginMSELoss(nn.Module):
    """
    Distils cross-encoder margins into the bi-encoder.
    For each (query, pos, neg), teacher_margin = CE(q, pos) - CE(q, neg) and
    student_margin = <q_emb, pos_emb> - <q_emb, neg_emb>. Loss = MSE(student, teacher).
    Strictly better than MNR when the cross-encoder teacher is available (MarginMSE
    paper §4.2); strictly worse when it is not.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mse = nn.MSELoss()

    def forward(self, sentence_features, labels):
        # sentence_features: list of 3 tokenised batches: [query, pos, neg]
        q = self.model(sentence_features[0])["sentence_embedding"]
        p = self.model(sentence_features[1])["sentence_embedding"]
        n = self.model(sentence_features[2])["sentence_embedding"]
        student = (q * p).sum(-1) - (q * n).sum(-1)
        # labels are the pre-computed teacher margins CE(q,pos) - CE(q,neg)
        return self.mse(student, labels)
```

Two defensible claims the reviewer should test:

- **MNR-with-hard-negatives is the strong baseline; MarginMSE is a step up.** MNR cannot express "pos should beat neg1 by more than it beats neg2"; MarginMSE can. On MS MARCO dev-small, MarginMSE distillation lifts the bi-encoder by ~2 MRR@10 over MNR-with-hard-negatives (Hofstätter et al. 2020 Table 2). This exercise lists both so the student can pick based on whether the cross-encoder scoring budget is available.
- **Do not combine MarginMSE and in-batch negatives in the same loss.** MarginMSE already includes a hard-negative term; adding in-batch negatives on top (as `MultipleNegativesRankingLoss(..., ..., include_negatives=True)` would do) introduces a second loss surface and destabilises training. Pick one.

### 2.5 Cross-encoder reranker

```python
# rerank.py
from sentence_transformers import CrossEncoder


def build_reranker(model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    return CrossEncoder(model_id, max_length=512)


def rerank(reranker, query: str, candidates: list[tuple[str, str]], top_k: int = 10):
    """
    candidates: list of (doc_id, passage_text) from the first-pass retriever.
    Returns top_k reranked (doc_id, score) tuples.
    """
    pairs = [(query, text) for _, text in candidates]
    scores = reranker.predict(pairs, batch_size=64, show_progress_bar=False)
    scored = sorted(
        zip([doc_id for doc_id, _ in candidates], scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    return scored[:top_k]
```

Reranker cheatsheet — pick by latency budget and corpus:

| Cross-encoder | Params | Latency (100 pairs, A100, fp16) | MRR@10 over BM25 top-100 | Use when |
|---|---|---|---|---|
| `ms-marco-TinyBERT-L-2-v2` | 4.4M | ~5 ms | ~0.35 | Edge / strict p99 latency |
| `ms-marco-MiniLM-L-4-v2` | 22M | ~20 ms | ~0.37 | Balanced default |
| `ms-marco-MiniLM-L-6-v2` | 23M | ~30 ms | ~0.39 | Server-side ranking; the paper-matching default |
| `ms-marco-MiniLM-L-12-v2` | 33M | ~60 ms | ~0.40 | Best open MiniLM reranker; when marginal quality matters |

Latency figures are approximate — reproduce them against your hardware before quoting. The MRR numbers come from the `sentence-transformers` cross-encoder model cards (MS MARCO dev-small, BM25 top-100). The exact numbers per card are maintained by the sentence-transformers project and are re-tuned across card revisions, so pin against the specific card revision you deploy rather than this table.

### 2.6 Two-stage evaluator — report the lift, not just the final score

```python
# evaluate.py
import json, time
from pytrec_eval import RelevanceEvaluator


def run_two_stage(queries, qrels, first_pass, reranker, corpus, k=100, top_k=10):
    stage1_runs, stage2_runs, timings = {}, {}, {"stage1_ms": [], "stage2_ms": []}
    for qid, query in queries.items():
        t0 = time.perf_counter()
        hits = first_pass(qid, k)
        timings["stage1_ms"].append((time.perf_counter() - t0) * 1000)
        stage1_runs[qid] = {doc_id: float(score) for doc_id, score in hits}

        t1 = time.perf_counter()
        candidates = [(doc_id, corpus[doc_id]) for doc_id, _ in hits]
        reranked = reranker.rerank(query, candidates, top_k=k)  # rerank ALL k, then cut
        timings["stage2_ms"].append((time.perf_counter() - t1) * 1000)
        stage2_runs[qid] = {doc_id: float(score) for doc_id, score in reranked}

    ev = RelevanceEvaluator(qrels, {"map", "mrr_cut_10", "ndcg_cut_10", "recall_100"})
    return {
        "stage1": _agg(ev.evaluate(stage1_runs)),
        "stage2": _agg(ev.evaluate(stage2_runs)),
        "latency_ms": {
            "stage1_p50": _percentile(timings["stage1_ms"], 50),
            "stage1_p95": _percentile(timings["stage1_ms"], 95),
            "stage2_p50": _percentile(timings["stage2_ms"], 50),
            "stage2_p95": _percentile(timings["stage2_ms"], 95),
        },
    }
```

What the output must contain:

- **Stage-1 Recall@100.** The ceiling on stage-2. If stage-1 Recall@100 is 0.85, stage-2 MRR@10 cannot exceed 0.85.
- **Stage-2 MRR@10 and nDCG@10.** The reported numbers; always both, because MRR cares only about the first correct hit and nDCG cares about the full top-10 ordering.
- **Stage-1 latency and stage-2 latency separately.** Hiding the cross-encoder cost inside the total is dishonest; it is the reason cross-encoders are not used as first-pass retrievers.
- **`qrels` format for `pytrec_eval`.** `dict[qid, dict[doc_id, grade]]` where grade is `0`/`1` for binary. Not a `set`; different library from exercise-01's SBERT evaluator. Both are built on `trec_eval` semantics.

## 3. Validation steps

1. **False-negative filter unit test.** Put a known paraphrase of the positive into the candidate list; assert the judge scores it above `FILTER_THRESHOLD` and the miner drops it.
2. **Hard-negative count parity.** After mining, assert every triple has exactly `n_negatives` negatives and that the `(query, positive, negative_i)` triples are all distinct. The MNR loss requires a constant number of negative columns per batch.
3. **Round-trip latency sanity.** Rerank 100 pairs with the chosen cross-encoder; wall-clock on your hardware. If stage-2 latency exceeds your p95 SLA, back off to a smaller cross-encoder or to `top_k=20` reranking.
4. **Lift test.** Stage-1 MRR@10 vs stage-2 MRR@10 on dev: the cross-encoder must lift by at least +3 MRR@10 over a dense bi-encoder first stage, or +10 over a BM25 first stage. A smaller lift means the cross-encoder is being asked to rerank already-ordered candidates and is not adding signal.
5. **Recall@100 ceiling check.** Stage-2 MRR@10 ≤ stage-1 Recall@100. If stage-2 MRR@10 appears to beat the ceiling, the evaluator is leaking positives into the candidate set (usually: `qrels` document IDs appear in the corpus but with a different casing).
6. **Rerank-over-k sweep.** Rerank at `k ∈ {10, 20, 50, 100, 200}`; nDCG@10 is monotone non-decreasing in `k`, with diminishing returns past 100. If it is non-monotone, the first-stage score is *negatively* correlated with the cross-encoder score, which means the first stage is broken.
7. **Calibration check.** Fit `sigmoid(logit) → P(relevant)` on a dev slice; plot a reliability diagram across 10 bins; expected calibration error (ECE) should be <0.05 for the cross-encoder to be trustworthy as a probability. If ECE is 0.15+, use scores for ranking only, not thresholding.
8. **Seed stability on the mined set.** Three seeds of the second-stage fine-tune over the *same* mined negatives; MRR@10 variance should be <0.3. Larger variance usually means too few negatives per query.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **First-pass stage** | BM25 or dense first-pass with Recall@100 reported; dense index is `IndexFlatIP` on normalised vectors at dev scale | First-pass present, no Recall@100 | First-pass bypassed; cross-encoder on the full corpus |
| **Hard-negative mining** | Top-N candidates from first-pass; false-negative filter with a *stronger* judge; per-query negative count fixed | Mining present, no filter | Random negatives passed off as "hard" |
| **Second-stage loss** | MNR-with-hard-negatives *or* MarginMSE distillation from a cross-encoder teacher | MNR without hard negatives | Triplet loss with random negatives, or no second-stage fine-tune |
| **Reranker choice** | Documented cross-encoder family + size + max_length + latency | Model id only | Reranker is the same bi-encoder |
| **Two-stage evaluator** | Recall@100 (stage-1) + MRR@10 + nDCG@10 (stage-2) + separate latency | Metric set complete, latency absent | Only final score |
| **Reproducibility** | Seed; mined-negatives dataset dumped to disk; mining script version + judge model revision recorded | Seed only | None |
| **Calibration / threshold hygiene** | Raw scores for ranking; calibration reported before thresholding | Threshold used without calibration | Score treated as probability without evidence |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks noted (adversarial passages that spoof BM25 to displace positives; cross-encoder prompt injection via passage content) | Mentioned, no mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure: judge model, filter threshold, mined-set statistics (kept/dropped), two-stage latency breakdown | One of those | Neither |

## 5. Common mistakes

- **No false-negative filter.** Mining BM25 top-100 negatives for MS MARCO queries pulls ~5–15 % false negatives (passages that are actually relevant but un-annotated). Training with them is *worse* than training with random negatives because the loss confidently pushes the embedding of a relevant passage away from the query (ANCE §4.3 "FN filtering"; Xiong et al. 2020). Always filter.
- **Reranking top-10 instead of top-100.** The cross-encoder's job is to re-order a *recall-ed* set. If the recall-ed set is tiny, there is nothing to rerank. Read Recall@k of the first pass, pick `k` so Recall@k ≥ your target MRR@10, then rerank that.
- **Using the exercise-01 bi-encoder as both first-pass retriever and reranker.** They are the same model, so the rerank is a monotone transformation of the first-pass score — zero lift. The reranker must be a cross-encoder (query and passage concatenated into one forward pass) or a stronger bi-encoder.
- **Tokenising the cross-encoder input past `max_length`.** `max_length=512` on BERT-family cross-encoders is a hard cap. If passages average 300 tokens and the query 15, you fit; above that, the tail of the passage is dropped — often the part that contained the relevance signal. Chunk long passages and max-pool scores, or raise `max_length` for a long-context base.
- **Returning `sigmoid(logit)` as a probability without calibration.** The cross-encoder logit is a ranking score; its sigmoid is not `P(relevant)` out of the box (ECE on MS MARCO is routinely >0.1). Use scores for sorting; calibrate (Platt or temperature scaling) before thresholding.
- **Reranking with the wrong tokenizer.** `CrossEncoder(model_id)` loads both weights and tokenizer from `model_id`. Using a different tokenizer because "it's the same base" produces garbage scores. Always load together.
- **Re-mining negatives after every epoch on a laptop.** ANCE's periodic re-mining is expensive; it was designed for the DPR scale budget. For a course exercise, mining once with BM25 + the exercise-01 checkpoint is enough; the marginal gain from re-mining is 1–2 MRR@10 at 10× the training cost.
- **Shipping the reranker without logging latency per k.** `rerank(..., top_k=10)` where the cross-encoder was asked to score only 10 pairs does not reflect the production latency of reranking `k=100`. Measure the actual reranked-set size, not the returned-set size.
- **Comparing to a BM25 baseline that was never tuned.** Default BM25 (`k1=0.9, b=0.4` on MS MARCO) is a well-calibrated baseline; `rank_bm25`'s defaults are different (`k1=1.5, b=0.75`) and produce lower numbers. Use `pyserini` or set the parameters explicitly before claiming a "beats-BM25" delta.
- **Reporting the wrong Recall@k.** `Recall@k = (relevant retrieved in top-k) / (total relevant)`. The MS MARCO qrels have on average ~1.1 relevant passages per query; that is *not* a reason for Recall@1 to be low — single-digit changes in Recall@10 are real.

## 6. References

Primary sources used in this solution:

- Nogueira, R., Cho, K. (2019). *Passage Re-ranking with BERT*. <https://arxiv.org/abs/1901.04085> — the monoBERT cross-encoder reranking recipe; established the two-stage retrieval pattern this exercise reproduces.
- Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP. <https://arxiv.org/abs/2004.04906> — DPR; BM25-mined hard-negative mixing and the in-batch-negatives baseline over which hard negatives lift.
- Xiong, L. et al. (2020). *Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval* (ANCE). ICLR 2021. <https://arxiv.org/abs/2007.00808> — iterative hard-negative mining with the current model; false-negative filtering.
- Hofstätter, S. et al. (2020). *Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation* (MarginMSE). SIGIR 2021. <https://arxiv.org/abs/2010.02666> — cross-encoder-to-bi-encoder margin distillation used in §2.4.
- Thakur, N. et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. NeurIPS Datasets & Benchmarks. <https://arxiv.org/abs/2104.08663> — BM25 baseline numbers and the Recall-vs-MRR relationship used throughout §1.
- Reimers, N., Gurevych, I. (2019). *Sentence-BERT*. EMNLP. <https://arxiv.org/abs/1908.10084> — the library and `CrossEncoder` class used in §2.5.
- Nguyen, T. et al. (2016). *MS MARCO*. <https://arxiv.org/abs/1611.09268> — the dataset the published numbers are scored on.
- Robertson, S., Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. FnT IR. <https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf> — BM25 as implemented by Anserini / pyserini.
- `sentence-transformers` CrossEncoder documentation — <https://www.sbert.net/docs/cross_encoder/usage/usage.html> — reference API for the reranker used in §2.5.
- `cross-encoder/ms-marco-MiniLM-L-6-v2` model card — <https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2> — the canonical reranker and its reported MS MARCO numbers.
- `pyserini` documentation — <https://github.com/castorini/pyserini> — Anserini Lucene BM25 wrapper used in §2.2.
- `pytrec_eval` — <https://github.com/cvangysel/pytrec_eval> — Python binding to `trec_eval`; the reference implementation for Recall@k / nDCG@10 / MRR@10 used in §2.6.
- Faiss — <https://github.com/facebookresearch/faiss> — the vector index library used in §2.2 (`IndexFlatIP`, `IndexHNSWFlat`).

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation*: adversarial passages crafted to be ranked highly for sensitive queries ("search poisoning"); mitigate by scoring candidate passages with the cross-encoder and logging out-of-distribution scores. *ML08 Model Skewing*: skewed training data from biased hard-negative mining (e.g., mining only from one source domain).
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for adversarial ML; the two-stage retrieval pipeline is a frequent target for ranking manipulation when the downstream system acts on top-k results without a human in the loop.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record first-pass retriever (BM25 / dense), judge model used for FN filtering, filter threshold, mined-set size, kept/dropped ratio, reranker family + size + max_length. *Measure*: Recall@100 (stage-1), MRR@10 + nDCG@10 (stage-2), stage-1 and stage-2 latency separately, calibration (ECE) if scores are thresholded.
