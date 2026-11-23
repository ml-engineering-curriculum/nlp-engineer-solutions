# exercise-01 — Bi-Encoder Training With Contrastive Loss

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-108-embeddings-and-representation-learning/exercises/exercise-01-bi-encoder-training-with-contrastive-loss.md`.
> The target learning outcomes implied by that filename: train a **bi-encoder** sentence-embedding model with a **contrastive objective** (InfoNCE / Multiple-Negatives-Ranking) using **in-batch negatives**, pick a pooling strategy that is consistent between training and inference, normalise embeddings before scoring, and evaluate with the metrics the retrieval literature uses (Recall@k, MRR@10, nDCG@10) on a held-out query set. The bi-encoder trained here is the first stage of the two-stage pipeline completed in exercise-02.

## 1. Solution overview

Bi-encoders are the workhorse of semantic search: one encoder embeds queries, the same (or a tied) encoder embeds documents, and relevance is the cosine similarity of the two vectors. The architecture is straightforward; what reliably breaks first-timers' numbers:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Pooling mismatch between train and inference** | Training uses `mean` pooling, serving uses `[CLS]`, or the model card says `cls` and the production index was built with `mean`. The two spaces are not aligned; cosine similarities collapse to near-constant values | Pin the pooling strategy in a config next to the checkpoint; expose it on the `SentenceTransformer` `modules.json` and round-trip test an encoded query vs the saved index |
| **Unnormalised vectors + cosine scoring** | The loss uses `cosine_similarity`, which expects unit vectors; inference computes raw dot products on unnormalised vectors (or vice versa). The similarity distribution shifts; thresholds and top-k rankings silently drift | Apply L2 normalisation inside the forward pass (`F.normalize(..., dim=-1)`) or an explicit `Normalize` module. Score with inner product on the index to match |
| **Too-small batch size for in-batch negatives** | The Multiple-Negatives-Ranking loss uses every *other* example in the batch as a negative; at `batch_size=8` you get 7 negatives per query, barely a classifier. The loss plateaus early and the model never learns hard distinctions | Target batch size ≥ 64 (DPR used 128; E5 used 32 768 with gradient accumulation / GradCache). Use `CachedMultipleNegativesRankingLoss` or GradCache if VRAM-bound |

The deliverable is a bi-encoder training harness with six components:

1. **Dataset builder** — load a retrieval or NLI corpus (MS MARCO passage-ranking, NLI-pair, or an in-domain `(query, positive)` TSV); emit `InputExample(texts=[q, pos])` for MNR-style training, or `(anchor, positive, negative)` triples for triplet training.
2. **Model wrapper** around `sentence-transformers` with an explicit `modules.json`: `Transformer` → `Pooling` → `Normalize`. Pooling is one of `mean`, `cls`, `max`; the choice is a hyperparameter, not a default.
3. **Loss** — `MultipleNegativesRankingLoss` (MNR, equivalent to InfoNCE with symmetric in-batch negatives) is the strong default; `TripletLoss` with mined negatives is the alternative when explicit hard negatives are available.
4. **Trainer** — `SentenceTransformer.fit` or (newer) `SentenceTransformerTrainer` with `IRCorpusEvaluator` on a dev slice.
5. **Information-retrieval evaluator** — Recall@{1, 5, 10, 100}, MRR@10, nDCG@10 on held-out `(query, judged_passages)` pairs; BEIR’s evaluation protocol and TREC `trec_eval` are the references.
6. **Reproducibility envelope** — seed, dataset revision, base-model revision, pooling, batch size, learning rate, and the final evaluator scores dumped to a single JSON next to the checkpoint.

Reference datasets (pick by domain and licence):

- **MS MARCO passage ranking** (Nguyen et al. 2016; v1 passage collection): 8.8M passages, ~500K training queries with sparse positives. Microsoft Research non-commercial licence. The canonical dense-retrieval training set.
- **NLI pair data** — SNLI (Bowman et al. 2015, CC-BY-SA-4.0) + MultiNLI (Williams et al. 2018, MultiNLI OANC licence) reformulated as `(premise, entailment)` positives; the Reimers-Gurevych SBERT recipe.
- **BEIR benchmark** (Thakur et al. 2021) — 18 zero-shot retrieval datasets; the standard "does your embedder generalise beyond the training distribution" test. Per-dataset licences; use the ones that match your deployment.

Reference base encoders:

- `sentence-transformers/all-MiniLM-L6-v2` (6-layer, 384-dim) — the default lightweight baseline, Apache-2.0; ~22M params, CPU-runnable.
- `sentence-transformers/all-mpnet-base-v2` (12-layer, 768-dim) — the stronger general-purpose English baseline, Apache-2.0.
- `intfloat/e5-base-v2` / `intfloat/e5-large-v2` (Wang et al. 2022) — contrastively pre-trained on web pairs; needs the `"query: "` / `"passage: "` instruction prefixes at inference *and* at training for the result to match the model card.
- `BAAI/bge-base-en-v1.5` (Xiao et al. 2023) — another strong general-purpose encoder; uses a `"Represent this sentence for retrieval: "` prefix for queries only.

## 2. Worked implementation

### 2.1 Project layout

```
biencoder_bake/
  data.py            # dataset loaders; MNR pairs and triplets
  model.py           # SentenceTransformer modules: Transformer + Pooling + Normalize
  loss.py            # MNR / InfoNCE wiring; optional GradCache
  evaluate.py        # IR evaluator: Recall@k, MRR@10, nDCG@10
  run.py             # CLI
  tests/
    test_pooling_parity.py
    test_normalize.py
    test_mnr_loss_shape.py
  requirements.txt
```

### 2.2 Dataset preparation — positives, no hand-crafted negatives (yet)

```python
# data.py
from datasets import load_dataset
from sentence_transformers import InputExample


def load_msmarco_pairs(split: str = "train"):
    """
    Returns an iterable of InputExample(texts=[query, positive_passage]).
    MNR loss uses in-batch negatives, so no explicit negative column is required.
    """
    ds = load_dataset("sentence-transformers/msmarco-hard-negatives", "triplet", split=split)
    for row in ds:
        yield InputExample(texts=[row["query"], row["positive"]])


def load_nli_pairs(split: str = "train"):
    """
    SNLI + MNLI reformulated as (premise, entailment_hypothesis) positives.
    """
    ds = load_dataset("sentence-transformers/all-nli", "pair", split=split)
    for row in ds:
        yield InputExample(texts=[row["anchor"], row["positive"]])
```

Points worth defending in review:

- **No explicit negatives yet.** MNR treats the other `B-1` positives in the batch as negatives for each query. This is the "in-batch negatives" trick from Henderson et al. 2017; it is the whole reason bi-encoder training is cheap. Exercise-02 adds *hard* negatives mined with a first-pass retriever.
- **The `sentence-transformers/msmarco-hard-negatives` dataset is pre-sampled.** It is the canonical MS MARCO training split with mined hard negatives already attached (we only use the positives here — the hard-negatives column is the input to exercise-02).
- **If using E5 or BGE, add the instruction prefixes at training time.** `"query: " + q` and `"passage: " + p` for E5; `"Represent this sentence for retrieval: " + q` only for BGE queries. Omitting them at train time and adding at inference puts the model in a distribution it has not been trained on and silently costs 2–5 nDCG@10.

### 2.3 Model + pooling — pin it; default to mean

```python
# model.py
from sentence_transformers import SentenceTransformer, models


def build_biencoder(
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    pooling: str = "mean",
    max_seq_length: int = 256,
):
    word_embedding = models.Transformer(base_model, max_seq_length=max_seq_length)
    pool = models.Pooling(
        word_embedding.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=(pooling == "mean"),
        pooling_mode_cls_token=(pooling == "cls"),
        pooling_mode_max_tokens=(pooling == "max"),
    )
    normalize = models.Normalize()  # L2-normalise so cosine == dot product
    return SentenceTransformer(modules=[word_embedding, pool, normalize])
```

Pooling cheatsheet — the three choices that matter and how to pick:

| Pooling | When it wins | Risk |
|---|---|---|
| `mean` | Default. Works for every encoder family (BERT, RoBERTa, MPNet, MiniLM, E5). Averages the token hidden states excluding padding | None notable; the SBERT paper used it and later work has not displaced it |
| `cls` | Encoders explicitly trained with a `[CLS]` objective (SimCSE, some BGE variants). Check the *base* model card | If the base model's `[CLS]` is not aligned with the sentence-embedding task (e.g., vanilla BERT), `cls` pooling performs 2–4 points worse than `mean` on STS-B |
| `max` | Legacy; max-over-tokens. Useful only for short sequences where one token carries all the signal (rare) | Loses most of the sequence information; only use for ablation |

Add `models.Normalize()` as the last module. That makes `encoder.encode(..., normalize_embeddings=True)` a no-op (idempotent) and guarantees the forward pass and the index use the same vector length.

### 2.4 Loss — MultipleNegativesRankingLoss (InfoNCE, symmetric in-batch)

```python
# loss.py
from sentence_transformers import losses


def build_loss(model, scale: float = 20.0):
    """
    MultipleNegativesRankingLoss is InfoNCE with:
      - every other positive in the batch used as a negative for each anchor,
      - a symmetric log-softmax (both directions scored and averaged),
      - a cosine-similarity-with-temperature scoring function (scale == 1/T).
    scale=20.0 (i.e. T=0.05) is the SBERT default and the Reimers-Gurevych-tuned value.
    """
    return losses.MultipleNegativesRankingLoss(model, scale=scale)
```

Four points reviewers must check:

- **`scale` is the inverse temperature, not a loss weight.** `scale=20.0` corresponds to temperature `T=0.05`. The SimCSE paper (Gao et al. 2021) uses `T=0.05` for supervised SimCSE on NLI; E5 uses `T=0.01` for high-batch pretraining. Do not treat it as a dial without justifying the move; a swing from 20 to 50 can shift the loss landscape enough to require relearning the LR.
- **Symmetric loss.** `MultipleNegativesRankingLoss` averages the query-to-doc and doc-to-query log-softmax. This is the same formulation DPR uses. Asymmetric variants exist but are rarely worth the configuration surface.
- **Requires batch size ≥ 32 to be competitive, ≥ 64 for strong results.** At `B=64` each query sees 63 negatives; at `B=256` it sees 255. For large batches on a single GPU, use `losses.CachedMultipleNegativesRankingLoss` (GradCache; Gao et al. 2021b) to accumulate without running out of VRAM.
- **No manual negatives in the batch.** MNR assumes every `texts[1]` in the batch is a *valid* positive. If a batch duplicates a passage across two queries, that passage becomes a false negative for one of them. Deduplicate per-batch (`sentence-transformers` has `NoDuplicatesDataLoader` for this).

### 2.5 Training loop — SentenceTransformer.fit

```python
# run.py
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
from sentence_transformers import evaluation

from .data import load_msmarco_pairs
from .model import build_biencoder
from .loss import build_loss
from .evaluate import build_ir_evaluator


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--pooling", default="mean", choices=["mean", "cls", "max"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--scale", type=float, default=20.0)
    p.add_argument("--output-dir", default="runs/biencoder")
    p.add_argument("--seed", type=int, default=20260101)
    a = p.parse_args()
    set_seed(a.seed)

    model = build_biencoder(a.base_model, a.pooling, a.max_seq_length)
    loss_fn = build_loss(model, scale=a.scale)
    train_examples = list(load_msmarco_pairs("train"))

    loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=a.batch_size,
        collate_fn=model.smart_batching_collate,
        drop_last=True,       # keep in-batch-negative count constant
    )
    total_steps = len(loader) * a.epochs
    warmup_steps = int(a.warmup_ratio * total_steps)

    dev_evaluator = build_ir_evaluator(split="dev")

    model.fit(
        train_objectives=[(loader, loss_fn)],
        evaluator=dev_evaluator,
        epochs=a.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": a.lr},
        weight_decay=0.01,
        output_path=a.output_dir,
        save_best_model=True,
        use_amp=True,
    )

    # Final report, pinned next to the checkpoint
    (Path(a.output_dir) / "run.json").write_text(json.dumps({
        "base_model": a.base_model, "pooling": a.pooling,
        "batch_size": a.batch_size, "epochs": a.epochs, "lr": a.lr,
        "scale": a.scale, "seed": a.seed,
    }, indent=2))
```

Defensible choices:

- **`lr=2e-5` + `warmup_ratio=0.1` + `weight_decay=0.01`.** The SBERT paper's recipe; it has survived every subsequent replication. 3e-5 is also stable; anything above 5e-5 catastrophically forgets the pre-training signal for 110M-parameter base models.
- **`drop_last=True`.** The MNR loss shape depends on batch size; a short trailing batch produces a different number of negatives and biases the optimizer slightly. Drop it.
- **`save_best_model=True` with an evaluator.** The evaluator's score (default: cosine similarity on an IR task, weighted average of MRR@10 and nDCG@10 depending on the subclass) is what picks the saved checkpoint. Without an evaluator, `fit` saves the last epoch, which is usually worse than the peak.
- **`use_amp=True`.** Mixed precision is a 1.6–1.8× throughput win on A100 / H100 with no measurable quality loss for retrieval. Confirm on your hardware; some older GPUs regress.

### 2.6 Evaluation — IR metrics on a held-out dev split

```python
# evaluate.py
from datasets import load_dataset
from sentence_transformers.evaluation import InformationRetrievalEvaluator


def build_ir_evaluator(split: str = "dev"):
    """
    Builds an InformationRetrievalEvaluator that reports Recall@{1,5,10,100},
    MRR@10, nDCG@10 on the MS MARCO passage-ranking small-dev slice.
    """
    qrels = load_dataset("sentence-transformers/msmarco-corpus", "qrels", split=split)
    queries_ds = load_dataset("sentence-transformers/msmarco-corpus", "queries", split=split)
    corpus_ds = load_dataset("sentence-transformers/msmarco-corpus", "corpus", split="corpus")

    queries = {str(r["qid"]): r["query"] for r in queries_ds}
    corpus = {str(r["pid"]): r["passage"] for r in corpus_ds}
    relevant_docs = {}
    for row in qrels:
        relevant_docs.setdefault(str(row["qid"]), set()).add(str(row["pid"]))

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[1, 5, 10, 100],
        show_progress_bar=True,
        batch_size=128,
    )
```

Three things reviewers must check:

- **`relevant_docs` is a `dict[str, set[str]]`.** The evaluator expects *sets* of relevant document IDs per query, not lists; duplicates inflate Recall silently.
- **Encode the corpus with the same model and the same pooling that produced the queries.** The evaluator does this automatically when you pass the model in; a mismatch is only possible if you cache corpus embeddings from a previous run. Don't.
- **The MS MARCO `dev-small` split has sparse labels (~1.1 relevant passages per query, no explicit negatives).** Recall@10 of ~0.85 is a well-trained baseline for MiniLM-L6; `all-mpnet-base-v2` lands around 0.87–0.90; E5-base-v2 lands around 0.90–0.93. Numbers much higher than that on *this* dev slice usually mean query-to-corpus leakage in the training split. Treat these ranges as reproduction targets rather than fixed constants: the exact MS MARCO `dev-small` MRR@10 figures reported by model cards drift with evaluator settings (whether in-batch negatives are mined, which split revision is used, whether the `MTEB`/SBERT evaluator is run), so re-run the eval yourself and cite the specific card revision rather than hard-pinning a single significant digit.

## 3. Validation steps

1. **Pooling-parity test.** Build the model with `pooling="mean"`; encode `["hello world"]`; detach the embedding. Reload the model from `output_dir` and encode the same string; assert `torch.allclose(a, b, atol=1e-6)`. If it fails, either the pooling is not being serialised into `modules.json` or the `Normalize` module is missing on the reload path.
2. **Normalisation test.** After encoding, assert `embeddings.norm(dim=-1)` is within `1e-6` of `1.0` for every row. If not, the `Normalize` module is missing and cosine scoring on the index will silently drift against training.
3. **MNR loss shape test.** With a 4-example batch and a 384-dim model, assert the similarity matrix the loss computes is `(4, 4)` and the loss is `≈ log(4)` at initialisation (CE of a uniform softmax over 4 classes). Anything else means the loss is scoring the wrong tensor.
4. **In-batch-negatives count test.** Assert that with batch size `B`, each query has `B-1` negatives (not `B`, not `2B`; there is a common bug where symmetric loss implementations double-count). `MultipleNegativesRankingLoss` symmetrises by averaging two separate losses, not by doubling the matrix.
5. **Smoke train.** 1 epoch on 10 000 MS MARCO pairs; dev MRR@10 should be above `0.25` for a MiniLM-L6 base (the untrained-from-random baseline is ~0.15; pre-trained all-MiniLM-L6-v2 starts around 0.35). If it is under 0.20, the pooling or normalisation is wrong.
6. **Index/query parity on BEIR.** On a BEIR dataset (SciFact, FiQA; both are small and CC-licensed), embed the corpus and queries with the trained model, score with inner product, and compare to the `sentence-transformers` published model cards. A gap >5 nDCG@10 flags a train/eval configuration drift (prefix, pooling, or normalisation).
7. **Instruction-prefix regression.** If using E5 or BGE: re-run the dev eval *without* the prefix. The drop is typically 3–7 MRR@10; if it is zero, you never applied the prefix at training.
8. **Seed stability.** Three seeds; MRR@10 variance on MS MARCO dev-small for a MiniLM-L6 fine-tune is typically <0.4 points; a run >1 point below the median flags a config bug.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Pooling + normalise pinned** | `modules.json` carries `Transformer → Pooling(mode) → Normalize`; parity test present | Pooling consistent, no `Normalize` module | Different pooling used at train vs. serve |
| **Contrastive loss** | `MultipleNegativesRankingLoss` (or explicit InfoNCE); `scale` documented; unit test on matrix shape | Loss used, no test | Triplet-only without mined negatives, or classification-style MSE |
| **Batch size for in-batch negatives** | ≥ 64; GradCache / `Cached…Loss` for larger | ≥ 32 | ≤ 16 with no justification |
| **Instruction prefixes (if E5/BGE)** | Applied at both train and inference; regression test | Applied at inference only | Not applied, or model card prefix ignored |
| **IR evaluator** | Recall@{1,5,10,100} + MRR@10 + nDCG@10 on held-out dev; `relevant_docs` is `dict[str, set[str]]` | Only Recall or only MRR | Loss curve reported in place of IR metrics |
| **Reproducibility** | `set_seed`; seed + dataset + base-model revision + sacre-signature-style config echoed to `run.json` | Seed only | None |
| **Dataset hygiene** | Dataset + revision + licence (MS MARCO MSRLA, SNLI CC-BY-SA-4.0, BEIR per-dataset) recorded | One of those | Neither |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks noted (query-injection to steer ranking; poisoning via public pair datasets) with mitigations | Mentioned, no mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure: pooling, normalisation, loss, scale, prefix convention, licence, known failure modes (asymmetric domains, long docs) | One of those | Neither |

## 5. Common mistakes

- **Pooling drift between train and serve.** `SentenceTransformer` writes `modules.json` on save; always load the saved directory rather than reconstructing the pipeline by hand. If a reviewer sees `AutoModel.from_pretrained(path).mean_pooling(...)` in the serving code, flag it.
- **Forgetting `Normalize()` at the end of the module list.** If the loss computes `cosine_similarity(a, b)` and inference indexes raw embeddings for inner-product scoring, the loss and the serve path are not in the same space. Add `models.Normalize()` and index with inner product; the two paths then match exactly.
- **Not using in-batch negatives.** The TripletLoss / ContrastiveLoss variants with explicit `(anchor, pos, neg)` triples are *strictly weaker* than MNR + in-batch negatives at the same compute budget (Reimers & Gurevych 2019 §4.2). If the exercise is "train a bi-encoder from scratch", MNR is the default; TripletLoss is the right answer only when you already have mined hard negatives (that's exercise-02).
- **Mixing E5's `"query: " / "passage: "` prefixes with BGE's `"Represent this sentence for retrieval: "` prefix.** They are different conventions baked into the pre-training. Using the wrong prefix on the wrong model silently degrades by 3–7 nDCG@10. Each model card documents its own prefix — treat it like a tokenizer setting, not a prompt.
- **Setting `max_seq_length` too low.** MS MARCO passages have P95 length ~180 BPE tokens; `max_seq_length=128` truncates 7–9 % of passages. For RAG corpora with long passages, 256 or 512 is standard; above 512 use a long-context base encoder like `jina-embeddings-v2-base-en`.
- **Comparing MRR to Recall across papers without aligning the `k`.** MS MARCO official leaderboard uses MRR@10. BEIR uses nDCG@10. The dense-retrieval literature sometimes reports Recall@1000 (the RAG recall ceiling). A number without an `@k` is unverifiable; always report `@k` and the evaluator.
- **Using cosine similarity on the index but dot product in the loss.** They are equivalent *only* if the vectors are unit-normalised. If one side is unnormalised, the loss optimises something the index does not score. Normalise both sides.
- **Shipping with `batch_size=8` because of VRAM.** That gives MNR seven negatives per query — a classification problem with 8 classes. Use `CachedMultipleNegativesRankingLoss` (gradient caching; one forward per sub-batch, one global backward) to hit the effective batch size you actually need without the VRAM.
- **Over-reporting NLI-fine-tuned numbers on BEIR.** The classic SBERT recipe (NLI + STS) produces an encoder that is excellent on STS and competitive on some BEIR tasks but weak on MS MARCO-style passage retrieval (Thakur et al. 2021 Table 2). Train on retrieval data if the deployment is retrieval; don't cite NLI-STS numbers to justify a retrieval system.
- **Using the full MS MARCO `train` split and the `dev` split from the same version together without checking leakage.** The official v1 `train` and `dev-small` splits are query-disjoint; the pre-processed `sentence-transformers/msmarco-hard-negatives` set respects that. Rolling your own split from the passage collection risks seeing dev queries in training.

## 6. References

Primary sources used in this solution:

- Reimers, N., Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP. <https://arxiv.org/abs/1908.10084> — the SBERT paper; mean pooling + Siamese bi-encoder + NLI training, and the reference `sentence-transformers` library.
- Henderson, M. et al. (2017). *Efficient Natural Language Response Suggestion for Smart Reply*. <https://arxiv.org/abs/1705.00652> — the original in-batch-negatives / Multiple-Negatives-Ranking formulation that MNR loss descends from.
- van den Oord, A. et al. (2018). *Representation Learning with Contrastive Predictive Coding*. <https://arxiv.org/abs/1807.03748> — the InfoNCE objective; MNR loss is the symmetric batched instantiation of this.
- Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP. <https://arxiv.org/abs/2004.04906> — DPR; the dense-retrieval recipe that established in-batch negatives + BM25-mined hard negatives + bi-encoder architecture.
- Gao, T. et al. (2021a). *SimCSE: Simple Contrastive Learning of Sentence Embeddings*. EMNLP. <https://arxiv.org/abs/2104.08821> — temperature-tuned supervised and unsupervised contrastive sentence embeddings; justification for `T≈0.05`.
- Gao, L. et al. (2021b). *Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup*. RepL4NLP. <https://arxiv.org/abs/2101.06983> — GradCache; the memory-efficient technique behind `CachedMultipleNegativesRankingLoss`.
- Wang, L. et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5). <https://arxiv.org/abs/2212.03533> — E5 family and the `"query: " / "passage: "` prefix convention.
- Xiao, S. et al. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding* (BGE). <https://arxiv.org/abs/2309.07597> — BGE family and the `"Represent this sentence for retrieval: "` query prefix.
- Thakur, N. et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. NeurIPS Datasets & Benchmarks. <https://arxiv.org/abs/2104.08663> — the BEIR zero-shot IR benchmark used for generalisation sanity checks.
- Nguyen, T. et al. (2016). *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset*. <https://arxiv.org/abs/1611.09268> — MS MARCO v1 passage-ranking collection; the canonical dense-retrieval training set.
- Bowman, S. et al. (2015). *A Large Annotated Corpus for Learning Natural Language Inference* (SNLI). EMNLP. <https://nlp.stanford.edu/pubs/snli_paper.pdf> — SNLI; licence CC-BY-SA-4.0.
- Williams, A. et al. (2018). *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference* (MultiNLI). NAACL. <https://aclanthology.org/N18-1101/> — MultiNLI, the second half of the standard SBERT NLI training mix.
- `sentence-transformers` library — <https://www.sbert.net/> — the canonical reference implementation of SBERT and the `MultipleNegativesRankingLoss` / `CachedMultipleNegativesRankingLoss` used in §2.4.
- Hugging Face `datasets` `sentence-transformers/msmarco-hard-negatives` — <https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives> — the pre-processed MS MARCO training set with mined hard negatives used in §2.2.
- TREC `trec_eval` — <https://github.com/usnistgov/trec_eval> — the reference implementation for IR metrics (MAP, nDCG, R-Precision) that `sentence-transformers`’s `InformationRetrievalEvaluator` mirrors.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation Attack* applies when adversarial queries are crafted to over-match a target passage (search ranking manipulation). *ML02 Data Poisoning* applies when fine-tuning on public pair data (MS MARCO community mines, scraped web pairs); verify the dataset origin before training.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for adversarial ML; relevant whenever the embedder feeds a retrieval or recommendation system that acts on the result.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record base model, pooling, normalisation, prefix convention, training data licence, max-seq-length, and known failure modes (long documents, cross-domain). *Measure*: Recall@k + MRR@10 + nDCG@10 on a held-out IR task, with evaluator settings pinned. Fairness and robustness are deferred to exercise-04 (domain / multilingual).
