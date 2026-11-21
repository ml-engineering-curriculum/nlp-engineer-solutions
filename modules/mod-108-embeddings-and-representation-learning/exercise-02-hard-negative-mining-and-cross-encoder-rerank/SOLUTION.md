# exercise-02 — Hard-Negative Mining and Cross-Encoder Rerank

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-108-embeddings-and-representation-learning/exercises/exercise-02-hard-negative-mining-and-cross-encoder-rerank.md`.
> Target learning outcomes implied by that filename: **mine hard negatives** from a first-stage retriever
> (BM25 + ANN), **train a cross-encoder reranker** on the mined pairs, and compose the two into a **retrieve-then-rerank
> pipeline** with measurable nDCG@10 / MRR@10 lift and defensible **latency and cost** numbers.

## 1. Solution overview

Retrieve-then-rerank is the standard open-domain retrieval shape:

1. A bi-encoder (or BM25) returns the top-K candidates per query in milliseconds, across millions of documents.
2. A cross-encoder scores each `(query, candidate)` pair *jointly* — the two strings concatenated into one Transformer input — and re-orders the top-K.

The cross-encoder is strictly stronger than the bi-encoder at the same parameter count (it sees query-document cross-attention) and strictly slower at serving (quadratic over K per query, no precomputable passage index). The two-stage pipeline keeps both properties.

Three subtleties reliably break first-timers' numbers:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Easy negatives only** | Training a cross-encoder on random negatives gives a model that ranks "obviously unrelated" below "slightly related" but cannot separate "almost right" from "right" — exactly the decisions the reranker exists to make | Mine negatives from the first-stage retriever (BM25 top-50 excluding golds, or a bi-encoder's top-100), ideally from *two* retrievers to avoid one-retriever's blind spot |
| **False-negative leakage** | MS MARCO / NQ labels are sparse — a query has one or two labelled golds, but many other passages in the corpus are also relevant. Treating every non-labelled passage as a hard negative trains the model to downrank real relevance | De-noise with a **teacher cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-12-v2` is the public reference), drop candidates the teacher scores > 0.5 above the known-negative floor, or restrict negatives to retrieval rank ≥ 20 |
| **Score-fusion mismatch at serving** | Training the reranker with a cross-entropy or MarginMSE head produces calibrated scores in one space; concatenating them with bi-encoder cosines at query time via a naive sum gives arbitrary rankings dominated by whichever scale is larger | Rerank by the cross-encoder *alone* on top-K, or score-fuse with Reciprocal Rank Fusion (Cormack et al. 2009) which ignores raw magnitudes |

The deliverable is a two-stage pipeline with six components:

1. **First-stage retriever** — either BM25 (via `rank_bm25` or `pyserini` for a production-grade impl) or a bi-encoder index (exercise-01 artifact) in FAISS / ScaNN / HNSW.
2. **Hard-negative miner** — per query, retrieve top-K from the first-stage, drop known-positives, apply false-negative de-noising, keep N hard negatives per query.
3. **Cross-encoder trainer** — `AutoModelForSequenceClassification(num_labels=1)` on `(query, passage)` pairs with a regression or margin loss.
4. **Reranker** — at inference, call the cross-encoder on `(query, candidate_i)` pairs; return sorted by score.
5. **Evaluator** — nDCG@10 / MRR@10 before and after reranking on a held-out dev set, with latency numbers per stage.
6. **Latency / cost card** — p50, p95, p99 end-to-end, plus per-stage breakdown and the knobs that trade quality for latency (K, batch size, model size).

Reference models and baselines:

- **BM25** via `rank_bm25` (quick) or **Pyserini** (Lucene-backed, production-grade; Lin et al. 2021). Pyserini is the de facto baseline for MS MARCO / BEIR comparisons.
- **Bi-encoder first stage** — `sentence-transformers/msmarco-distilbert-base-tas-b` (Hofstätter et al. 2021), `intfloat/e5-base-v2`, or your exercise-01 artifact.
- **Cross-encoder reranker** — `cross-encoder/ms-marco-MiniLM-L-6-v2` (public baseline, 22M params, ~1 ms/pair on A10) or the stronger `cross-encoder/ms-marco-MiniLM-L-12-v2`. Fine-tune from these for domain data.
- **Teacher for distillation / de-noising** — `cross-encoder/ms-marco-MiniLM-L-12-v2` or `monoT5-3b` (Nogueira et al. 2020).

## 2. Worked implementation

### 2.1 Project layout

```
rerank_bake/
  mine.py              # hard-negative mining: BM25 + bi-encoder
  data.py              # pair-and-label dataset for cross-encoder training
  train_ce.py          # cross-encoder trainer
  pipeline.py          # retrieve -> rerank orchestration
  evaluate.py          # staged nDCG/MRR + latency
  tests/
    test_dedup_golds.py
    test_score_shape.py
    test_pipeline_order.py
  requirements.txt
```

### 2.2 Hard-negative mining — BM25 top-K minus golds

```python
# mine.py
import json
import random
from collections import defaultdict
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


def _tokenise(s: str) -> list[str]:
    return s.lower().split()


def mine_bm25_negatives(
    queries: dict[str, str],          # qid -> query text
    corpus: dict[str, str],            # pid -> passage text
    qrels: dict[str, set[str]],        # qid -> {relevant_pid, ...}
    top_k: int = 100,
    n_negatives_per_query: int = 20,
    rank_floor: int = 20,              # skip top-N even if non-gold; those often ARE relevant
    denoise_with_teacher: bool = True,
    teacher_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    teacher_margin: float = 3.0,
    seed: int = 20260101,
) -> list[dict]:
    rng = random.Random(seed)
    pids = list(corpus.keys())
    bm25 = BM25Okapi([_tokenise(corpus[p]) for p in pids])

    teacher = CrossEncoder(teacher_model) if denoise_with_teacher else None

    mined: list[dict] = []
    for qid, qtext in queries.items():
        gold_pids = qrels.get(qid, set())
        if not gold_pids:
            continue
        scores = bm25.get_scores(_tokenise(qtext))
        ranked = sorted(range(len(pids)), key=lambda i: scores[i], reverse=True)[:top_k]
        # Drop golds and the top ranks (likely-relevant-but-unlabelled).
        candidates = [pids[i] for rank, i in enumerate(ranked)
                      if pids[i] not in gold_pids and rank >= rank_floor]
        if not candidates:
            continue

        if denoise_with_teacher:
            gold_pid = next(iter(gold_pids))
            gold_score = teacher.predict([(qtext, corpus[gold_pid])])[0]
            cand_scores = teacher.predict([(qtext, corpus[p]) for p in candidates])
            # Drop any candidate the teacher thinks is close to the gold —
            # likely a false negative.
            candidates = [p for p, s in zip(candidates, cand_scores)
                          if gold_score - s >= teacher_margin]

        if len(candidates) > n_negatives_per_query:
            candidates = rng.sample(candidates, n_negatives_per_query)

        for pid in gold_pids:
            for neg_pid in candidates:
                mined.append({
                    "qid": qid,
                    "query": qtext,
                    "positive": corpus[pid],
                    "negative": corpus[neg_pid],
                })
    return mined
```

Points worth defending in review:

- **`rank_floor=20`.** MS MARCO labels are sparse (median 1 gold / query). The top 1–20 BM25 hits often contain *other* relevant passages that are not labelled; using them as negatives trains a reranker to downrank real relevance. Skipping ranks 1–19 drops a small amount of difficulty signal to buy a substantially cleaner label.
- **`denoise_with_teacher`.** A stronger cross-encoder scoring the candidate against the query gives a probability the candidate is actually relevant. Dropping candidates whose teacher score is within `teacher_margin` (= 3.0 logits ≈ "10× less relevant than the known positive") removes the worst false-negative offenders. On MS MARCO this adds 1–2 nDCG@10 over raw BM25 mining in follow-up fine-tunes (Qu et al. 2021, RocketQA; Formal et al. 2021, SPLADE).
- **`n_negatives_per_query=20`.** Typical range 5–50. More negatives = more compute per epoch but better training signal. 20 is the RocketQA / BEIR convention.
- **Two retrievers for diversity.** In practice, union the top-K from BM25 *and* a bi-encoder, then dedupe. BM25 and dense retrievers surface different kinds of distractor: BM25 surfaces lexical traps ("heart attack" matched to "attack on a heart"), dense surfaces semantic traps. Exercise-04's E5 or your exercise-01 bi-encoder is a good second miner.

### 2.3 Cross-encoder dataset — regression target, not classification

```python
# data.py
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder


def mined_to_pairs(mined: list[dict]) -> list[InputExample]:
    """
    One row per positive and one row per mined negative, labelled 1.0 / 0.0.
    Cross-encoder regression is the sentence-transformers default and is more
    stable than binary CE on sparse-label data — see the loss section.
    """
    rows = []
    for ex in mined:
        rows.append(InputExample(texts=[ex["query"], ex["positive"]], label=1.0))
        rows.append(InputExample(texts=[ex["query"], ex["negative"]], label=0.0))
    return rows


def mined_to_margin_mse(
    mined: list[dict],
    teacher: CrossEncoder,
) -> list[InputExample]:
    """
    MarginMSE (Hofstätter et al. 2020): the student regresses the *margin* between
    teacher(q, p+) and teacher(q, p-) rather than fitting hard 0/1 labels. Much
    more sample-efficient than CE when a teacher is available.
    """
    rows = []
    batch_pos = [(e["query"], e["positive"]) for e in mined]
    batch_neg = [(e["query"], e["negative"]) for e in mined]
    s_pos = teacher.predict(batch_pos)
    s_neg = teacher.predict(batch_neg)
    for ex, sp, sn in zip(mined, s_pos, s_neg):
        rows.append(
            InputExample(
                texts=[ex["query"], ex["positive"], ex["negative"]],
                label=float(sp - sn),
            )
        )
    return rows
```

Loss options:

- **`MSELoss` on `(0.0, 1.0)` labels** (the simplest; sentence-transformers' `CrossEncoder.fit(loss_fct=nn.MSELoss())`). Reliable baseline.
- **`BCEWithLogitsLoss`** on `(0, 1)` labels. Numerically equivalent to logistic regression on the cross-encoder logit. Preferred if you want calibrated probabilities out.
- **MarginMSE** (Hofstätter et al. 2020) — regress the teacher-cross-encoder's margin. Gains 1–3 nDCG@10 on MS MARCO over MSE/BCE when a strong teacher is available. The method is the de facto reference for training distilled rerankers.

### 2.4 Cross-encoder trainer

```python
# train_ce.py
import argparse
import torch
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader
from .mine import mine_bm25_negatives
from .data import mined_to_pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="microsoft/MiniLM-L12-H384-uncased")
    p.add_argument("--output-dir", default="runs/cross-encoder")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260101)
    a = p.parse_args()

    ce = CrossEncoder(a.base_model, num_labels=1, max_length=a.max_length)

    # Assume mine_bm25_negatives has been run and cached as JSONL.
    import json
    with open("mined.jsonl") as f:
        mined = [json.loads(line) for line in f]

    train_examples = mined_to_pairs(mined)
    train_loader = DataLoader(train_examples, batch_size=a.batch_size, shuffle=True)

    ce.fit(
        train_dataloader=train_loader,
        epochs=a.epochs,
        warmup_steps=min(1000, len(train_loader) // 10),
        optimizer_params={"lr": a.lr},
        output_path=a.output_dir,
        use_amp=True,
        loss_fct=torch.nn.MSELoss(),
    )
```

Key points:

- **`num_labels=1`.** The cross-encoder is a regression head over `[CLS]`; the output is one scalar per `(query, passage)` pair. For calibrated probabilities use `num_labels=1` + `BCEWithLogitsLoss` + `torch.sigmoid` at inference.
- **`max_length=256` for MS MARCO passages.** For legal / clinical paragraphs raise to 384 or 512. The cross-encoder sees `[CLS] query [SEP] passage [SEP]`, so the budget is shared.
- **Batch size ~32 at 256 tokens on an A10 (24 GB).** Smaller than bi-encoder because the sequence length is roughly doubled.
- **`use_amp=True`.** Mixed precision is safe for cross-encoder regression; no softmax over large logit matrix means no overflow risk.
- **2 epochs is the sentence-transformers / BEIR convention.** More overfits quickly on hard-negative-mined data.

### 2.5 Pipeline — retrieve K, rerank, serve top-N

```python
# pipeline.py
import time
from sentence_transformers import CrossEncoder, SentenceTransformer
import faiss
import numpy as np


class RetrieveRerank:
    def __init__(
        self,
        bi_encoder: SentenceTransformer,
        cross_encoder: CrossEncoder,
        corpus_texts: list[str],
        corpus_ids: list[str],
        retrieve_k: int = 100,
        rerank_batch_size: int = 64,
    ):
        self.bi = bi_encoder
        self.ce = cross_encoder
        self.corpus_texts = corpus_texts
        self.corpus_ids = corpus_ids
        self.retrieve_k = retrieve_k
        self.rerank_batch_size = rerank_batch_size

        # Build index at init.
        emb = self.bi.encode(corpus_texts, convert_to_numpy=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb.astype(np.float32))

    def search(self, query: str, top_n: int = 10) -> tuple[list[tuple[str, float]], dict[str, float]]:
        t0 = time.perf_counter()
        q_emb = self.bi.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q_emb.astype(np.float32), self.retrieve_k)
        t1 = time.perf_counter()

        candidates = [(self.corpus_ids[i], self.corpus_texts[i]) for i in I[0]]
        ce_scores = self.ce.predict(
            [(query, text) for _, text in candidates],
            batch_size=self.rerank_batch_size,
        )
        t2 = time.perf_counter()

        ranked = sorted(zip(candidates, ce_scores), key=lambda t: t[1], reverse=True)
        top = [(pid, float(score)) for (pid, _), score in ranked[:top_n]]
        return top, {"retrieve_ms": (t1 - t0) * 1000, "rerank_ms": (t2 - t1) * 1000}
```

Worth defending:

- **Normalise at index time** (`IndexFlatIP` with unit vectors ≡ cosine).
- **`retrieve_k=100` default.** nDCG@10 lift from reranking plateaus around K=100 on MS MARCO; K=1000 adds latency without quality gain. For legal / clinical domains with denser relevance, K=50 is often fine.
- **Report `retrieve_ms` and `rerank_ms` separately.** The first scales with corpus size (sub-ms on 100k, ~5 ms on 10M), the second with K × cross-encoder cost (linear in K). Confusing the two during a perf regression investigation wastes hours.
- **Rerank by cross-encoder alone**, not by `α · cos + β · ce_score`. The scales don't compose. If you want fusion for robustness, use RRF (`1 / (k + rank)`) instead.

### 2.6 Evaluation — before/after rerank on the same K

```python
# evaluate.py
from collections import defaultdict
import numpy as np


def ndcg_at_k(relevances: list[int], k: int) -> float:
    r = np.asarray(relevances)[:k]
    if r.sum() == 0:
        return 0.0
    gains = (2 ** r - 1) / np.log2(np.arange(len(r)) + 2)
    ideal = sorted(relevances, reverse=True)[:k]
    i_r = np.asarray(ideal)
    i_gains = (2 ** i_r - 1) / np.log2(np.arange(len(i_r)) + 2)
    return gains.sum() / i_gains.sum() if i_gains.sum() > 0 else 0.0


def mrr_at_k(relevances: list[int], k: int) -> float:
    for rank, rel in enumerate(relevances[:k], 1):
        if rel > 0:
            return 1.0 / rank
    return 0.0


def evaluate(pipeline, queries: dict, qrels: dict, k_rerank: int = 100, k_metric: int = 10):
    before, after = defaultdict(list), defaultdict(list)
    for qid, qtext in queries.items():
        gold = qrels.get(qid, set())
        if not gold:
            continue
        # Retrieve K and record ranking at retrieval stage *before* reranking.
        q_emb = pipeline.bi.encode([qtext], convert_to_numpy=True, normalize_embeddings=True)
        _, I = pipeline.index.search(q_emb, k_rerank)
        retrieved_pids = [pipeline.corpus_ids[i] for i in I[0]]
        before_rel = [1 if pid in gold else 0 for pid in retrieved_pids]
        before["ndcg@10"].append(ndcg_at_k(before_rel, k_metric))
        before["mrr@10"].append(mrr_at_k(before_rel, k_metric))

        # Rerank and compute same metrics on the reordered list.
        texts = [pipeline.corpus_texts[i] for i in I[0]]
        scores = pipeline.ce.predict([(qtext, t) for t in texts])
        order = np.argsort(-scores)
        reranked_pids = [retrieved_pids[i] for i in order]
        after_rel = [1 if pid in gold else 0 for pid in reranked_pids]
        after["ndcg@10"].append(ndcg_at_k(after_rel, k_metric))
        after["mrr@10"].append(mrr_at_k(after_rel, k_metric))

    def mean(d):
        return {k: float(np.mean(v)) for k, v in d.items()}
    return {"before_rerank": mean(before), "after_rerank": mean(after)}
```

Report both numbers — the lift is the value of the reranker, and the floor is what the retriever delivers if reranking is disabled.

## 3. Validation steps

1. **Gold-leakage check in mined negatives.** After mining, assert no mined negative text appears in any gold-positive pair for the same query. Collision count must be 0; if > 0, your qrels map is stale or your mining missed a gold entry.
2. **Teacher de-noise unit test.** Pick 50 queries. For each, log the teacher's score on `(query, mined_negative)`. Assert the max is < `teacher(gold) - teacher_margin`. If any sneak above, the de-noise filter is broken or `teacher_margin` is too tight.
3. **Cross-encoder shape test.** Feed `[(q, pos), (q, neg)]`; the pos score must be higher *after* training. On the first batch of training it may not be — check again at step 500.
4. **Pipeline-order determinism.** With `retrieve_k=100` and a fixed seed, run the same query twice; the top-10 reranked list must be identical. Non-determinism usually means a shuffled `DataLoader` on the eval set or a non-deterministic FAISS backend.
5. **Lift test.** On 1k dev queries, nDCG@10 *after* rerank must be strictly greater than *before*. For MS MARCO dev with BM25 first-stage and `cross-encoder/ms-marco-MiniLM-L-6-v2`, the baseline lift is ≈ +15 nDCG@10 (BM25 ≈ 0.19 → + rerank ≈ 0.35). A lift of < +5 means either the first-stage is already optimal (unlikely for BM25) or the cross-encoder is miscalibrated.
6. **Latency budget.** Measure end-to-end p95 at K=100 on your target hardware. For `MiniLM-L-6-v2` on a single A10, expect ~150 ms rerank + ~5 ms retrieve ≈ 155 ms at K=100. If p95 > 500 ms, reduce K or distil to a smaller cross-encoder.
7. **Score-fusion regression.** If you *do* fuse `cos` + `ce_score`, A/B against pure CE rerank on 1k dev queries. If fusion is not strictly better, drop it.
8. **BEIR cross-dataset test.** Train on MS MARCO; evaluate on at least one out-of-domain BEIR dataset (SciFact, FiQA, or NFCorpus). If nDCG@10 on the OOD dataset *drops below* BM25-only there, the cross-encoder is overfitting MS MARCO's style (web passages, short news-y queries) — consider a domain adaptation pass (exercise-04).

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Negative-mining source** | BM25 **and** bi-encoder; union then dedupe | One retriever | Random negatives only |
| **False-negative de-noise** | Teacher-based filtering with margin + `rank_floor`; dropped count logged | Rank-floor only | None; every top-K non-gold treated as negative |
| **Cross-encoder loss** | MarginMSE with teacher, OR MSE/BCE with justification; `num_labels=1` | Loss used, not justified | Softmax classification over K candidates (that's a bi-encoder pattern, not CE) |
| **Score fusion discipline** | Rerank-only OR RRF, A/B'd against pure rerank | Linear sum of cos + ce_score | Ad-hoc weighted sum, no A/B |
| **Pipeline K** | `retrieve_k` chosen with lift-vs-K curve on dev; documented | K hard-coded | K chosen "because the paper used 1000" with no justification |
| **Latency reporting** | p50 / p95 / p99 per stage; corpus size in the card | Overall latency only | No latency numbers |
| **Lift reported** | nDCG@10 and MRR@10, before and after rerank, same K, same dev set | After-only | Classifier accuracy only |
| **Reproducibility** | Seed; miner config; teacher model id; CE base model id; dataset revisions | Seed only | None |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks called out (prompt-injection via retrieved passages feeding an LLM; adversarial SEO; poisoning of training data) | Mentioned without mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure: retriever, reranker, K, dev set, per-stage latency, corpus licence | One of those | None |

## 5. Common mistakes

- **Mining negatives from the training set only, then evaluating on the same qrels.** If the same (query, passage) pair appears in training *as a negative* but in the dev qrels as a positive, your dev scores crater. Split by `qid` before mining; never let a dev `qid` leak into the miner.
- **Treating `rank_floor=0`.** The top 1–5 BM25 hits often contain other relevant passages; marking them as negatives teaches the model to downrank genuine relevance. The bigger the `rank_floor`, the cleaner the negatives, the smaller the training signal — tune on dev.
- **Linear-weighting bi-encoder cosine and CE score at fusion time.** The two scales are different (unit cosine [-1, 1] vs raw CE logit [-10, 10]). A naive sum is dominated by the CE score, which is *fine*, but then the bi-encoder term is noise. If you must fuse, use **Reciprocal Rank Fusion** (Cormack et al. 2009): `rrf(d) = Σ_r 1 / (k + rank_r(d))` ignores raw scores and is robust.
- **Over-reranking K.** Increasing `retrieve_k` from 100 to 1000 rarely helps nDCG@10 and multiplies latency 10×. Plot lift vs K on dev; stop at the first K where the curve flattens.
- **Confusing cross-encoder logits with calibrated probabilities.** Without sigmoid + BCE training, the raw cross-encoder logit is not a probability; thresholds from one model do not transfer to another. For "is this passage relevant?" boolean calls, train with BCE or Platt-scale post-hoc.
- **Reranking with the *same* model that mined.** If you mine negatives with model `A` and train model `A` on those negatives, you fit `A`'s blind spots — the "hard" negatives are easy for *other* models but hard for *this one*. Mine with a strong teacher (or an ensemble) distinct from the student.
- **Running rerank at K=100 on CPU in prod.** `MiniLM-L-6-v2` at K=100 on CPU is ~1 s/query — not a serving budget. Either GPU-serve the reranker, distil to L-4-v2 / L-2-v2, or drop to K=20.
- **Using `rank_bm25` on large corpora.** `rank_bm25` is pure-Python; above 100k documents the ranking becomes the bottleneck. Switch to Pyserini / Lucene (Elasticsearch, OpenSearch, Vespa) for production.
- **Evaluating with too few queries.** nDCG@10 variance on < 500 queries is ~0.02–0.04, which is the same order as the reranker lift on out-of-domain data. Use ≥ 1000 dev queries; bootstrap CIs for close comparisons.
- **Not pinning the teacher.** The teacher cross-encoder's checkpoint id (and its HF revision hash) must be recorded; swapping teachers between mine runs and distillation runs silently changes the margin labels. `CrossEncoder(..., revision=...)` locks the revision.

## 6. References

Primary sources used in this solution:

- Reimers, N., Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP. <https://arxiv.org/abs/1908.10084> — the bi-encoder / cross-encoder distinction at the base of §1.
- Nogueira, R., Cho, K. (2019). *Passage Re-ranking with BERT*. <https://arxiv.org/abs/1901.04085> — the original BERT cross-encoder reranker on MS MARCO.
- Qu, Y. et al. (2021). *RocketQA: An Optimized Training Approach to Dense Passage Retrieval* (hard-neg mining + cross-batch neg + denoising). ACL. <https://arxiv.org/abs/2010.08191> — the de-noising-with-teacher pattern used in §2.2.
- Hofstätter, S. et al. (2020). *Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation* (MarginMSE). <https://arxiv.org/abs/2010.02666> — MarginMSE loss used in §2.3.
- Hofstätter, S. et al. (2021). *Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling* (TAS-B). SIGIR. <https://arxiv.org/abs/2104.06967> — the `msmarco-distilbert-base-tas-b` bi-encoder suggested as first-stage.
- Formal, T., Piwowarski, B., Clinchant, S. (2021). *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*. SIGIR. <https://arxiv.org/abs/2107.05720> — context on sparse-lexical first-stage retrievers that pair with CE reranking.
- Cormack, G., Clarke, C., Buettcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*. SIGIR. — the RRF formula used in §2.5 / §5.
- Thakur, N. et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. NeurIPS. <https://arxiv.org/abs/2104.08663> — the OOD-eval protocol used in validation step 8.
- Lin, J. et al. (2021). *Pyserini: A Python Toolkit for Reproducible Information Retrieval Research with Sparse and Dense Representations*. SIGIR. <https://arxiv.org/abs/2102.10073> — the BM25 implementation referenced in §1.
- Robertson, S., Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in IR. — the BM25 formulation.
- sentence-transformers cross-encoder docs — <https://www.sbert.net/examples/applications/cross-encoder/README.html> — canonical API for `CrossEncoder.fit()` used in §2.4.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation* and *ML05 Model Theft* both apply: an adversarial passage retrieved and then reranked at the top is a classic prompt-injection vehicle when the stack feeds results into an LLM, and the reranker's scoring can be surfaced as a cheap oracle for extraction attacks. *ML02 Data Poisoning* applies to the labelled training pairs — a public MS MARCO-derived training set can be injected with adversarial documents that make the reranker surface attacker-controlled text.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for adversarial ML; relevant whenever rerank outputs feed LLMs, moderation, or search ranking that downstream systems trust.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record retriever, reranker, K, teacher; *Measure*: nDCG@10 / MRR@10 before and after rerank, per-stage latency, corpus licence; *Govern*: reranker outputs that feed LLM inputs are part of the LLM prompt supply chain and inherit its risk posture.
