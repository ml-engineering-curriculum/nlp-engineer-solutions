# exercise-01 — Bi-Encoder Training With Contrastive Loss

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-108-embeddings-and-representation-learning/exercises/exercise-01-bi-encoder-training-with-contrastive-loss.md`.
> Target learning outcomes implied by that filename: train a **bi-encoder** (Siamese Transformer) to produce
> sentence or passage embeddings, pick a sensible **contrastive loss** (triplet, Multiple Negatives Ranking /
> InfoNCE), use **in-batch negatives** safely, pool correctly, and report **retrieval quality** on a held-out
> set with a protocol you can compare across runs.

## 1. Solution overview

A bi-encoder is two tied Transformer towers that produce a single fixed-size vector per input, scored by
cosine similarity (or dot product). It trains in minutes-to-hours on a single GPU and serves in milliseconds
per query. The architecture is originally Reimers & Gurevych's SBERT (2019), and sentence-transformers is
the de-facto reference implementation.

Three subtleties reliably break first-timers' numbers:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Wrong pooling** | Using `pooler_output` (BERT's `[CLS]` + tanh) on a model that was not trained for sentence embedding gives a near-random vector. **Mean-pooling with attention-mask** is what SBERT trained with and the only safe default for a from-scratch bi-encoder | Pool as `(sum(h * mask) / sum(mask))` across the last hidden state with the attention mask applied; never trust `pooler_output` on a vanilla encoder |
| **Dirty in-batch negatives** | MultipleNegativesRankingLoss / InfoNCE treats every other item in the batch as a negative. If your data has duplicates, paraphrases, or near-duplicate passages within one batch you train the model to push *correct* answers apart | De-duplicate by a hash of the document text (not the id); shuffle so `(q, d+)` pairs are independent batch-to-batch; log the collision rate as a health metric |
| **Scoring mismatch at eval** | Training with cosine similarity (which normalises) but computing retrieval with raw dot product (which does not) gives a different ranking function and a different number. The opposite mistake — training with dot product but reporting cosine — is equally common | Pick one (cosine for from-scratch bi-encoders is standard) and use it in loss, index, and eval. If you want dot product serving, train with dot product and *skip* the `nn.functional.normalize` step |

The deliverable is a training harness with six components:

1. **Pair dataset** — `(anchor, positive)` or `(anchor, positive, hard_negative)` triples. Public datasets: MS MARCO passage, Natural Questions, Stack Exchange duplicates, GooAQ.
2. **Bi-encoder module** wrapping an `AutoModel` with a documented pooling head (mean pooling + L2 normalise for cosine).
3. **Loss** — one of `MultipleNegativesRankingLoss` (InfoNCE with in-batch negatives), `TripletLoss`, or `CosineSimilarityLoss` depending on the data shape.
4. **Training loop** — `sentence_transformers.SentenceTransformer.fit(...)` or a plain `Trainer` with the right scheduler and warmup.
5. **Evaluator** — `InformationRetrievalEvaluator` (nDCG@10, MRR@10) or `BinaryClassificationEvaluator` (accuracy, F1) depending on task.
6. **Reproducibility card** — seed, dataset revision, base model, pooling, loss, per-device batch size (because in-batch negatives mean the *effective* number of negatives is `batch_size - 1`).

Reference checkpoints — pick by scale and licence:

- **`sentence-transformers/all-MiniLM-L6-v2`** (Reimers et al. 2022) — 22M parameters, English, trained on a 1B-pair mixture; the canonical starter checkpoint for retrieval exercises. CPU-runnable.
- **`sentence-transformers/all-mpnet-base-v2`** (Song et al. 2020 base + Reimers 2022 fine-tune) — 110M parameters; the strongest "free, English, general" bi-encoder in sentence-transformers before the E5 / BGE wave.
- **`intfloat/e5-base-v2`** (Wang et al. 2022) — 110M parameters; adds a `"query: "` / `"passage: "` prefix convention that the model was trained with. Skipping the prefix at inference drops nDCG@10 by ~5 points.
- **`BAAI/bge-base-en-v1.5`** (Xiao et al. 2023) — 110M parameters; another strong, open-weights English bi-encoder. The **v1.5 instruction prefix** is `"Represent this sentence for searching relevant passages: "`, also required at query time.

## 2. Worked implementation

### 2.1 Project layout

```
biencoder_bake/
  data.py              # pairs / triples loader with dedup
  model.py             # Bi-encoder module with mean-pool head
  loss.py              # MultipleNegativesRankingLoss wrapper (or InfoNCE in torch)
  train.py             # CLI
  evaluate.py          # IR evaluator + metric dump
  tests/
    test_pooling.py
    test_dedup.py
    test_scoring.py
  requirements.txt
```

### 2.2 Dataset — pairs with hash-based de-dup

```python
# data.py
import hashlib
from datasets import load_dataset
from sentence_transformers import InputExample


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _h(s: str) -> str:
    return hashlib.sha1(_norm(s).encode("utf-8")).hexdigest()[:16]


def ms_marco_pairs(split="train", min_pos_score: int = 1):
    """
    (query, positive_passage) pairs from MS MARCO v1.1. Drops pairs whose positive
    passage hash collides with a prior query's positive passage within the *same*
    batch-feed order, since those duplicates become false negatives in-batch.
    """
    ds = load_dataset("ms_marco", "v1.1", split=split)
    seen_doc_hashes: set[str] = set()
    pairs = []
    for ex in ds:
        for passage, is_selected in zip(
            ex["passages"]["passage_text"], ex["passages"]["is_selected"]
        ):
            if is_selected < min_pos_score:
                continue
            h = _h(passage)
            if h in seen_doc_hashes:
                # Same passage already paired with another query — dropping avoids
                # creating a false negative when both appear in one batch.
                continue
            seen_doc_hashes.add(h)
            pairs.append(InputExample(texts=[ex["query"], passage]))
    return pairs
```

Points worth defending in review:

- **Hash on the normalised string, not on the integer id.** MS MARCO re-uses passages across queries; the id can differ while the text is identical. The id-based dedup misses these.
- **Track `seen_doc_hashes` across the whole stream, not per-batch.** In-batch negatives mean you want independence over the *epoch*, not just one batch. Collisions within a *shuffled* epoch distribute uniformly, so per-stream dedup is sufficient and cheaper than per-batch.
- **`min_pos_score` is adjustable.** MS MARCO has graded labels; clinical / legal / closed-domain exercises often ship binary. Make the threshold a flag.
- **Mining hard negatives is a separate step** — see exercise-02 for BM25 + ANN hard-negative mining. For this exercise, in-batch negatives are sufficient and give a strong floor.

### 2.3 Bi-encoder module — mean pool with attention mask

```python
# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BiEncoder(nn.Module):
    """
    Siamese Transformer with mean-pool + L2 normalise.

    Mean-pool rule: sum(last_hidden * mask) / sum(mask). This is what SBERT trained
    with (Reimers & Gurevych 2019). `pooler_output` on a vanilla encoder is a
    tanh-on-CLS head that was tuned for NSP, not for sentence embedding, and will
    give near-random vectors on a from-scratch bi-encoder.
    """

    def __init__(self, model_id: str, normalize: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.normalize = normalize

    def mean_pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: list[str], device: str = "cuda") -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        out = self.encoder(**enc).last_hidden_state
        pooled = self.mean_pool(out, enc["attention_mask"])
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=1)
        return pooled

    def forward(self, anchors: list[str], positives: list[str]):
        return self.encode(anchors), self.encode(positives)
```

Worth defending:

- **L2-normalise before scoring** when you intend to use cosine similarity. `cos(a, b) = <a, b> / (||a|| ||b||)`, which for unit vectors reduces to the dot product. Downstream FAISS / ScaNN with `IndexFlatIP` then computes cosine for free.
- **Mean pooling over `last_hidden_state`**, not `pooler_output`. The 2019 SBERT paper benchmarks three pools (mean, CLS, max) and recommends mean. "Max" is better on a subset of STS tasks but worse on retrieval; "CLS" is worst unless the model was pre-trained with a sentence-level objective (SimCSE, MPNet-base sentence head).
- **`max_length=256`** is a sensible default for MS MARCO / NQ passages. For legal / clinical paragraphs raise to 512; above that, chunk (exercise-04 for the multilingual case).

### 2.4 Loss — Multiple Negatives Ranking (InfoNCE with in-batch negatives)

```python
# loss.py
import torch
import torch.nn.functional as F


class MultipleNegativesRankingLoss(torch.nn.Module):
    """
    InfoNCE with in-batch negatives (Henderson et al. 2017; Reimers 2020).

    For a batch of (anchor_i, positive_i) pairs and encoded vectors a, p of shape
    (B, D), the matrix S = scale * a @ p.T has S[i, i] as the positive logit and
    S[i, j != i] as negative logits. Labels are the diagonal:

        L = CrossEntropy(S, torch.arange(B))

    With unit-normalised vectors and `scale=20.0` (a learned temperature stand-in
    that sentence-transformers hard-codes), the softmax has useful dynamic range.
    """

    def __init__(self, scale: float = 20.0):
        super().__init__()
        self.scale = scale

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        logits = self.scale * (anchors @ positives.T)
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)
```

Key points:

- **One pair per anchor, in-batch negatives, symmetric targets.** The effective number of negatives is `batch_size - 1`. Doubling the batch roughly doubles the useful negative count — this is why contrastive training rewards larger batches more than supervised classification does.
- **Scale ≈ 20.** sentence-transformers' default. Equivalent to a temperature of `1/20 = 0.05` on unit vectors. Higher scale sharpens the softmax; lower scale softens it. Papers that learn the temperature (SimCSE, CLIP) end up near 0.05–0.07.
- **If the data has `(anchor, positive, hard_negative)` triples**, you can either append the hard negatives to the `positives` matrix (so each row sees its own hard negative plus all other anchors' positives/hard-negatives) — this is `MultipleNegativesRankingLoss` with `hard_negatives=True` — or use `TripletLoss`:

  ```python
  def triplet_loss(a, p, n, margin=0.2):
      pos_dist = 1 - (a * p).sum(dim=-1)   # cosine distance
      neg_dist = 1 - (a * n).sum(dim=-1)
      return F.relu(pos_dist - neg_dist + margin).mean()
  ```

  TripletLoss trains more slowly per step but is less sensitive to batch dirtiness; MNRL converges faster but requires clean pairs. See common mistakes.

### 2.5 Training loop — sentence-transformers path

```python
# train.py
import argparse
from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from .data import ms_marco_pairs


def build_sbert(model_id: str) -> SentenceTransformer:
    # Explicitly wire mean-pool + L2 normalise so there is no ambiguity about
    # what the final SentenceTransformer returns.
    word_emb = models.Transformer(model_id, max_seq_length=256)
    pooling = models.Pooling(
        word_emb.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    normalize = models.Normalize()
    return SentenceTransformer(modules=[word_emb, pooling, normalize])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="microsoft/mpnet-base")
    p.add_argument("--output-dir", default="runs/biencoder")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260101)
    args = p.parse_args()

    model = build_sbert(args.base_model)
    train_examples = ms_marco_pairs(split="train")
    train_loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model=model, scale=20.0)

    # Hold out ~1k queries with a labelled qrels dict for the evaluator.
    # Keys are strings; values on `corpus` are passages, on `queries` are question strings,
    # and `relevant_docs` maps qid -> set of relevant pids.
    evaluator = InformationRetrievalEvaluator(
        queries=..., corpus=..., relevant_docs=..., name="msmarco-dev-1k",
        mrr_at_k=[10], ndcg_at_k=[10], accuracy_at_k=[1, 5, 10],
    )

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=args.warmup,
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        save_best_model=True,
        use_amp=True,
    )
```

- **`warmup_steps=1000` on 10–100k steps** — the sentence-transformers default. Linear warmup to `lr=2e-5` then linear decay. Lower `lr` for larger base models; 2e-5 is stable on `bert-base` / `mpnet-base`.
- **`use_amp=True`** — mixed precision; roughly 2× throughput on A100/4090. Verify that losses stay finite for the first 200 steps; if they go NaN, scale is usually the issue (too-large `scale` + fp16 = overflow).
- **Save best by evaluator score**, not final-epoch. Contrastive retrieval can be rough-and-ready at epoch 1 and worse at epoch 3 if the pair set is small — nDCG@10 on the dev set is the only trustworthy early-stopping signal.

### 2.6 Evaluation — `InformationRetrievalEvaluator` or BEIR

```python
# evaluate.py
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator


def evaluate(model_dir: str, queries: dict, corpus: dict, relevant_docs: dict) -> dict:
    model = SentenceTransformer(model_dir)
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="eval",
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        show_progress_bar=True,
    )
    # evaluator returns the primary metric but also dumps a CSV with all metrics
    score = evaluator(model, output_path=".")
    return {"ndcg@10": score}
```

For a heavier, portable harness, run **BEIR** (Thakur et al. 2021) — a 19-dataset retrieval benchmark bundle. BEIR runs on top of sentence-transformers and reports nDCG@10 the way the retrieval community does. Exercise-03 covers MTEB, which supersets BEIR.

## 3. Validation steps

1. **Pooling unit test.** Encode `["hello world", "hello"]` with the bi-encoder. Assert the embedding of `"hello world"` is *not* equal to the embedding of `"hello"` (sanity-check mask is not degenerate), and assert both are unit vectors (`vec.norm().item() ≈ 1.0`). If they are equal, your mean-pool is pooling before applying the attention mask and averaging over pad tokens.
2. **Scoring round trip.** On a 10-pair held-out set, compute `cos(a, p)` manually with `(a * p).sum()` on unit vectors, and compare to `model.similarity(a, p)`. Delta must be < 1e-5.
3. **In-batch negative diagonal.** On a random batch, compute `S = scale * A @ P.T`; `S.argmax(dim=1)` should equal `arange(B)` on a converged model and *does not* on a freshly initialised one. If the fresh model also predicts the diagonal, your batches contain only self-paired duplicates — go check your dedup.
4. **Duplicate audit.** Before training, hash every `(q, p)` pair; the number of hash collisions in the final shuffled training stream divided by the number of pairs is the "false-negative rate" on in-batch negatives. Target < 0.5 %; above 2 %, dedup harder.
5. **Prefix convention check.** If you fine-tune from an E5 or BGE checkpoint, encode the dev query *with* and *without* the model's prescribed prefix and compare nDCG@10 on 200 queries. The prefix-less number is typically 3–5 points lower on E5. If your "without" number is *higher*, your dataloader is pushing prefix into both sides and the scores are gibberish.
6. **Base-model floor.** Evaluate the base `all-MiniLM-L6-v2` (no fine-tune) on your dev set; your fine-tune should at least match it. If it is worse, the loss is not learning (check `scale`, the sign of the loss, and the dedup rate).
7. **Latency.** Encode 10k passages at batch 64 on your target hardware; report ms/passage and tokens/sec. Bi-encoders are the fast tower — a 110M-param model on a single A10 should push > 2k passages/sec.
8. **Seed stability.** Train with 3 seeds. Variance on MS MARCO dev nDCG@10 for an `mpnet-base` MNRL fine-tune is typically < 0.5; a run > 1 point below the median flags a config bug.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Pooling discipline** | Mean-pool with attention mask, L2 normalise; pooling unit-tested | Mean pool, no mask or no test | `pooler_output` on a vanilla encoder |
| **Loss + scoring consistency** | Train and eval both use cosine (or both dot) end-to-end | Documented mismatch | Trained cosine, indexed dot — scores don't match |
| **In-batch negative hygiene** | Hash-based dedup across the training stream; collision rate logged < 0.5 % | Dedup by id only | No dedup; MNRL on raw pairs with known duplicates |
| **Loss choice justified** | MNRL for clean pairs OR Triplet for triples, with written rationale | Loss picked, not explained | Classification `BCEWithLogits` on cosine |
| **Prefix / instruction convention** | If base model requires a prefix (E5, BGE, `instructor`), it is applied at both train and inference | Applied at inference only | Prefix dropped silently |
| **Evaluator fidelity** | `InformationRetrievalEvaluator` or BEIR/MTEB task with qrels; nDCG@10 + MRR@10 reported | Accuracy@1 only | Loss curve as the only "quality" number |
| **Reproducibility** | `set_seed`; dataset revision, base model, pooling, scale, batch size dumped as JSON next to the checkpoint | Seed only | None of those |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks called out (embedding inversion, membership inference, poisoning of training pairs) with mitigations | Mentioned without mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure applied: training corpus licence, language coverage, pooling, scoring, prefix convention, dev set + number + signature | One of those | None |

## 5. Common mistakes

- **Using `pooler_output` on a vanilla encoder.** Near-random vectors; nDCG@10 typically < 0.2. Diagnostic: cosine of `"hello world"` vs `"random noise about elephants"` with your encoder should be *lower* than cosine of `"hello world"` vs `"greetings earth"`. If the two near-random cosines are identical to three decimal places, you are reading from the untrained CLS head.
- **Normalising after indexing.** If your FAISS index was built on unnormalised vectors and you normalise queries at search time, you silently compute `cos(q, d) / ||d||` — the ranking is still roughly correct but calibration and thresholds are off. Normalise both sides at index build time, then use `IndexFlatIP` for cosine-equivalent scoring.
- **Treating MNRL batch size as "just a throughput knob".** It is a *quality* knob — `batch_size - 1` is your negative count. Going from 16 to 64 can move nDCG@10 by 2–4 points on MS MARCO. If you must reduce batch size for memory, add gradient accumulation only for the forward side cannot recover the missing negatives — instead, use `GradCache` (Gao et al. 2021) to maintain effective-batch size with reduced memory.
- **Mixing different pair sources without balancing.** Training on `(NQ, MS MARCO, StackExchange, Quora)` concatenated unweighted lets the largest dataset (MS MARCO) dominate. Interleave with per-dataset probabilities or use `datasets.interleave_datasets` with `probabilities=[...]`.
- **Dropping the E5 / BGE prefix at train time but keeping it at eval.** You fine-tuned without the prefix, then the evaluator added it back; the model never learned that prefix pattern. Decide on prefix policy *before* training and never deviate.
- **Reporting Accuracy@1 as "quality".** For retrieval with graded or sparse labels, accuracy@1 collapses because most queries have one gold passage; it tells you nothing about ranking quality below rank 1. Always add nDCG@10 and MRR@10.
- **Evaluating on the training set after a tight augmentation pass.** If you augmented pairs with paraphrases and did not remove them from the dev split, your nDCG@10 is inflated. Split by `query_id` (or by `doc_id` if queries recur) *before* augmentation.
- **Scale too high with fp16.** `scale=50` + fp16 can overflow after a few steps and silently NaN the loss. If you need a higher scale, run in bf16 or keep the softmax in fp32 (`logits.float()` before cross-entropy).
- **Confusing cosine distance and cosine similarity in TripletLoss.** The margin is on the distance (`1 - cos(a, p) + margin < 1 - cos(a, n)`), and some implementations pass similarities directly. If your TripletLoss is negative or stuck at zero, check whether the implementation takes similarity or distance.
- **Not reporting the sentence-transformers version.** The library changes defaults across minor versions (scale, pooling, warmup); pin it (`sentence-transformers==x.y.z`) and record it in the run card.

## 6. References

Primary sources used in this solution:

- Reimers, N., Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP. <https://arxiv.org/abs/1908.10084> — the SBERT architecture (shared-weight bi-encoder, mean pooling, cosine scoring) at the core of §2.3.
- Henderson, M. et al. (2017). *Efficient Natural Language Response Suggestion for Smart Reply*. <https://arxiv.org/abs/1705.00652> — the in-batch negative / sampled-softmax formulation that `MultipleNegativesRankingLoss` implements in §2.4.
- van den Oord, A., Li, Y., Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding*. <https://arxiv.org/abs/1807.03748> — the InfoNCE loss that MNRL is a practical instantiation of.
- Gao, T., Yao, X., Chen, D. (2021). *SimCSE: Simple Contrastive Learning of Sentence Embeddings*. EMNLP. <https://arxiv.org/abs/2104.08821> — dropout-noise contrastive pairs and the temperature analysis behind the `scale ≈ 20` default.
- Wang, L. et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5). <https://arxiv.org/abs/2212.03533> — the `query:` / `passage:` prefix convention at the base of §2.3 / §2.5 and in the common-mistakes list.
- Xiao, S. et al. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding* (BGE). <https://arxiv.org/abs/2309.07597> — the BGE recipe (instruction prefix, two-stage contrastive + fine-tune) referenced in §2.1 / §5.
- Thakur, N. et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. NeurIPS. <https://arxiv.org/abs/2104.08663> — the retrieval benchmark used in §2.6 for cross-run comparability.
- Gao, L. et al. (2021). *Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup* (GradCache). <https://arxiv.org/abs/2101.06983> — the trick cited in §5 for recovering effective batch size without the memory.
- Nguyen, T. et al. (2016). *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset*. NeurIPS Workshop. <https://arxiv.org/abs/1611.09268> — the dataset used in the worked example.
- Reimers, N. (2022). sentence-transformers release notes and documentation — <https://www.sbert.net/docs/training/overview.html> — canonical API for `SentenceTransformer`, `InformationRetrievalEvaluator`, and `MultipleNegativesRankingLoss` used in §2.5 / §2.6.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML02 Data Poisoning* applies: a bi-encoder trained on scraped `(question, answer)` pairs (StackExchange, Reddit, Quora dumps) is one adversarial pair-injection away from biased retrieval. *ML06 Model Inversion* applies to the embedding tower itself: sentence embeddings are partially invertible (Song & Raghunathan 2020), which means publishing embeddings of private documents is effectively partial text disclosure. Treat the embedding index as PII-equivalent when the source was PII.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for adversarial ML; relevant whenever embeddings drive search, deduplication, moderation, or RAG grounding that a downstream system trusts.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record base model, training-pair licence, language coverage, pooling, prefix convention; *Measure*: nDCG@10 and MRR@10 on a named dev split with seed and sentence-transformers version; *Govern*: embedding tables inherit the data-handling classification of their source documents.
