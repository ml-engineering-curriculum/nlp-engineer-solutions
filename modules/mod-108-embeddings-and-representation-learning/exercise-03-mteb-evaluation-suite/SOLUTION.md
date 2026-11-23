# exercise-03 — MTEB Evaluation Suite

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-108-embeddings-and-representation-learning/exercises/exercise-03-mteb-evaluation-suite.md`.
> The target learning outcomes implied by that filename: evaluate a sentence-embedding model on the **Massive Text Embedding Benchmark (MTEB)** by Muennighoff et al. (2023), run a **meaningful subset of task types** (Retrieval, STS, Classification, Clustering, Reranking, Pair Classification, Summarisation), report the **per-task primary metric alongside the task-group average**, and publish an **MTEB-schema JSON** that the public leaderboard scorer can ingest. Do not report "MTEB score" without listing the tasks that produced it — the benchmark is intentionally plural.

## 1. Solution overview

MTEB is the standard cross-task embedding benchmark: 8 task types across 58 English datasets at v1 (Muennighoff et al. 2023), extended to 100+ datasets and 8 multilingual expansions in subsequent releases. The benchmark is valuable because *no single-task metric* captures embedding quality — a model can win on STS and lose on retrieval; a model can win on retrieval and be useless for clustering. The point of the suite is to force you to look at the full task-type distribution.

Three failure modes decide whether the report is actually comparable:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Cherry-picked task subset** | Running MTEB-Retrieval only and reporting "MTEB score X" is misleading; the published leaderboard averages across *all* eight task types and that is the headline number | List the tasks evaluated; report per-task-type averages (not a single "MTEB score") unless every task was run. If a subset was run, say which and why |
| **Prefix / pooling mismatch between evaluation and training** | E5 was trained with `"query: "` / `"passage: "`; BGE with `"Represent this sentence for retrieval: "` on queries only. Running MTEB with `encode(sentences)` and no prefix drops 2–7 points per task type and silently under-reports the model | Register a prefix convention with `MTEB`'s `encoder` interface via the `instruction` / `query_instruction` kwargs. The benchmark harness supports it; the model card documents the exact string |
| **Reporting `MTEB score` without the benchmark revision** | MTEB is versioned; tasks are added, removed, or re-split between revisions. A 2023 score is not comparable to a 2024 score even on the same model | Record the `mteb` library version, the task list, and the per-task primary metric. The MTEB leaderboard expects results in a specific per-task JSON schema |

The deliverable is an MTEB evaluation harness with four components:

1. **Encoder adapter** — a thin wrapper that exposes `encode(sentences, **kwargs) -> np.ndarray` and handles the model's required prefixes (E5 / BGE / jina).
2. **Task runner** — the `mteb` library; configure the task list explicitly, do not take the default "all tasks" in a time-bound exercise.
3. **Result aggregator** — per-task primary metric, per-task-type average, overall average; scores emitted in the MTEB result schema (one JSON per task under `results/{model}/{revision}/{task}.json`).
4. **Comparison report** — the trained model's scores side-by-side with the published numbers of 2–3 reference encoders (your base model + a stronger published model), so a reviewer can see whether the fine-tune lifted or regressed.

MTEB task-type cheat sheet (all eight types; primary metrics are what the leaderboard averages):

| Task type | Datasets at v1 | Primary metric | What it tests |
|---|---|---|---|
| **Retrieval** | 15 (MS MARCO, NQ, HotpotQA, FiQA, SciFact, …) | nDCG@10 | Dense retrieval; the hardest task type |
| **Reranking** | 4 (MindSmall, StackOverflowDupQ, …) | MAP | The embedder as a lightweight reranker |
| **Clustering** | 11 (ArxivClusteringP2P, RedditClustering, …) | V-measure | How well embeddings cluster by label |
| **Pair Classification** | 3 (SprintDuplicateQuestions, TwitterSemEval, TwitterURLCorpus) | AP (average precision) | Duplicate detection |
| **Classification** | 12 (AmazonPolarity, Banking77, EmotionClassification, …) | Accuracy | Linear probe on frozen embeddings |
| **STS** | 10 (STS12-16, STS-B, SICK-R, STS22, BIOSSES, …) | Spearman (cosine) | Sentence-pair similarity |
| **Summarisation** | 1 (SummEval) | Spearman | Correlation of embedding similarity with human summary scores |
| **Bitext Mining** | 0 in English v1, ≥9 in multilingual | F1 | Cross-lingual sentence alignment (in scope for exercise-04) |

Dataset counts and primary metrics are those stated in Muennighoff et al. (2023, Table 1) for MTEB v1 English. The `mteb` library hard-codes them; you should not override them per task.

## 2. Worked implementation

### 2.1 Project layout

```
mteb_bake/
  adapter.py             # encoder wrapper with prefix handling
  tasks.py               # task subset selection; whitelist helpers
  run.py                 # CLI: run MTEB tasks, emit per-task JSON
  aggregate.py           # task-type averages; comparison report
  tests/
    test_adapter_prefixes.py
    test_result_schema.py
  requirements.txt
```

### 2.2 Encoder adapter — prefixes are not optional

```python
# adapter.py
import numpy as np
from sentence_transformers import SentenceTransformer


class MTEBEncoder:
    """
    Thin adapter around sentence-transformers.SentenceTransformer that implements
    the mteb Encoder protocol.

    The `mteb` library calls `encode(sentences, prompt_name=None, prompt=None, ...)`;
    newer `mteb` releases use `encode_queries` / `encode_corpus` for retrieval tasks,
    which lets you apply different prefixes to each side. We implement both.
    """

    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        normalize: bool = True,
    ):
        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.normalize = normalize
        self.model_name = model_name

    def encode(self, sentences, **kwargs):
        # Classification / STS / Clustering don't distinguish query vs passage.
        # Use the passage prefix by convention (E5: "passage: "; BGE: no prefix).
        return self._encode(sentences, self.passage_prefix)

    def encode_queries(self, queries, **kwargs):
        return self._encode(queries, self.query_prefix)

    def encode_corpus(self, corpus, **kwargs):
        texts = [doc.get("text", doc) if isinstance(doc, dict) else doc for doc in corpus]
        return self._encode(texts, self.passage_prefix)

    def _encode(self, texts, prefix):
        prefixed = [f"{prefix}{t}" for t in texts] if prefix else texts
        emb = self.model.encode(
            prefixed,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return emb.astype(np.float32)
```

Prefix convention cheatsheet — applies at *evaluation* time, must match training:

| Model family | Query prefix | Passage prefix |
|---|---|---|
| `intfloat/e5-*-v2` | `"query: "` | `"passage: "` |
| `intfloat/multilingual-e5-*` | `"query: "` | `"passage: "` |
| `BAAI/bge-*-en-v1.5` | `"Represent this sentence for retrieval: "` | *(none)* |
| `BAAI/bge-m3` | *(none, except where the model card specifies a task prompt)* | *(none)* |
| `sentence-transformers/all-MiniLM-L6-v2` | *(none)* | *(none)* |
| `sentence-transformers/all-mpnet-base-v2` | *(none)* | *(none)* |
| `jinaai/jina-embeddings-v2-base-en` | *(none)* | *(none)* |

Values are what the respective model cards document. Deviating from the prefix the model was trained with silently under-reports on MTEB; applying a prefix to a model that was not trained with one adds noise and silently under-reports too. If unsure, run a prefix-ablation smoke test (§3 validation step 2).

### 2.3 Task runner

```python
# tasks.py
RETRIEVAL_SUBSET = [
    # Picked for coverage + download size manageability. Full retrieval list
    # in mteb is 15 datasets; these 5 cover a range of domains and lengths.
    "MSMARCO",            # web passages, sparse labels
    "NQ",                 # Natural Questions open-domain QA
    "FiQA2018",           # finance QA
    "SciFact",            # scientific claim verification
    "TRECCOVID",          # pandemic retrieval; short, single-topic
]
STS_SUBSET = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICK-R", "BIOSSES"]
CLASSIFICATION_SUBSET = [
    "Banking77Classification", "EmotionClassification",
    "AmazonPolarityClassification", "ImdbClassification",
    "MTOPDomainClassification",
]
CLUSTERING_SUBSET = [
    "ArxivClusteringP2P", "ArxivClusteringS2S",
    "RedditClustering", "StackExchangeClustering",
    "TwentyNewsgroupsClustering",
]
RERANKING_SUBSET = ["AskUbuntuDupQuestions", "StackOverflowDupQuestions", "SciDocsRR", "MindSmallReranking"]
PAIR_CLASSIFICATION_SUBSET = ["SprintDuplicateQuestions", "TwitterSemEval2015", "TwitterURLCorpus"]
SUMMARIZATION_SUBSET = ["SummEval"]
```

```python
# run.py
import argparse, json
from pathlib import Path
from mteb import MTEB

from .adapter import MTEBEncoder
from .tasks import (
    RETRIEVAL_SUBSET, STS_SUBSET, CLASSIFICATION_SUBSET,
    CLUSTERING_SUBSET, RERANKING_SUBSET, PAIR_CLASSIFICATION_SUBSET,
    SUMMARIZATION_SUBSET,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF repo id or local path")
    p.add_argument("--query-prefix", default="")
    p.add_argument("--passage-prefix", default="")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--task-groups", nargs="+", default=["retrieval", "sts", "classification"])
    a = p.parse_args()

    encoder = MTEBEncoder(
        model_name=a.model,
        query_prefix=a.query_prefix,
        passage_prefix=a.passage_prefix,
    )

    groups = {
        "retrieval": RETRIEVAL_SUBSET,
        "sts": STS_SUBSET,
        "classification": CLASSIFICATION_SUBSET,
        "clustering": CLUSTERING_SUBSET,
        "reranking": RERANKING_SUBSET,
        "pair_classification": PAIR_CLASSIFICATION_SUBSET,
        "summarization": SUMMARIZATION_SUBSET,
    }

    all_tasks = [t for g in a.task_groups for t in groups[g]]
    evaluator = MTEB(tasks=all_tasks)
    evaluator.run(encoder, output_folder=a.output_dir)
    # mteb writes per-task JSON under <output_dir>/<model_name>/<revision>/<task>.json
```

Three defensible choices:

- **Explicit task list, not `MTEB()`-defaults.** `MTEB()` with no arguments runs *every* task the library knows about; on first install that is 100+ tasks and >50 GB of downloaded data. For a graded exercise, pick the subset that covers the task types you want to assess and record it.
- **Task-group averages, not the full MTEB average.** The MTEB leaderboard averages across 56 English tasks (v1); averaging across only 5 retrieval + 8 STS + 5 classification tasks is a different number. Report per-task-type averages; if you also compute a single "overall" for comparison, make it the mean of the per-type averages, not the mean of the per-task scores (so each type is equally weighted).
- **`mteb` writes results into a stable directory layout.** `<output>/<model>/<revision>/<task>.json`. That layout is what the leaderboard scorer ingests; preserve it in your report artefact.

### 2.4 Result aggregator

```python
# aggregate.py
import json
from pathlib import Path


PRIMARY_METRIC_BY_TYPE = {
    "Retrieval": "ndcg_at_10",
    "Reranking": "map",
    "Clustering": "v_measure",
    "PairClassification": "cos_sim_ap",
    "Classification": "accuracy",
    "STS": "cos_sim_spearman",
    "Summarization": "cos_sim_spearman",
    "BitextMining": "f1",
}


def load_task_results(run_dir: Path):
    out = {}
    for f in run_dir.rglob("*.json"):
        data = json.loads(f.read_text())
        # The mteb v1 JSON exposes `test` (or dev; depends on task) → split → metrics.
        task_name = data.get("mteb_dataset_name") or f.stem
        task_type = data.get("mteb_task_type")   # e.g., "Retrieval"
        primary = PRIMARY_METRIC_BY_TYPE.get(task_type)
        if primary is None:
            continue
        split_results = data.get("test") or data.get("dev") or {}
        if isinstance(split_results, dict):
            score = _find_metric(split_results, primary)
            if score is not None:
                out[task_name] = {"task_type": task_type, "metric": primary, "score": score}
    return out


def _find_metric(d: dict, key: str):
    """mteb nests metrics under language codes or splits; descend to find the leaf."""
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict):
            r = _find_metric(v, key)
            if r is not None:
                return r
    return None


def task_type_averages(results: dict):
    by_type = {}
    for r in results.values():
        by_type.setdefault(r["task_type"], []).append(r["score"])
    return {t: sum(scores) / len(scores) for t, scores in by_type.items()}
```

Four protocol points reviewers must verify:

- **Primary metric per task type is fixed by the benchmark.** Do not swap nDCG@10 for Recall@100 on retrieval and claim comparability; the leaderboard averages nDCG@10.
- **`cos_sim_spearman` vs `cos_sim_pearson` on STS.** MTEB's primary STS metric is cosine-similarity Spearman. Pearson is reported but not the headline.
- **`cos_sim_ap` vs `dot_ap` on pair classification.** `cos_sim_ap` is the primary; it assumes normalised embeddings. If your encoder ships with `normalize_embeddings=True`, `cos_sim` and `dot` collapse to the same ranking, but the metric field on the leaderboard is still `cos_sim_ap`.
- **Nested dict traversal is necessary.** Retrieval tasks nest metrics under the eval split; multilingual tasks nest under a language code. `_find_metric` descends until it finds the key; a flat-key lookup will silently miss scores.

### 2.5 Comparison report — against *published* numbers

```python
# aggregate.py (continued)
def comparison_table(my_results: dict, reference_results: dict):
    """
    reference_results: dict of {model_name: {task_name: score}} loaded from the
    published model cards. The MTEB leaderboard CSV is the canonical source.
    """
    rows = []
    for task, r in my_results.items():
        row = {"task": task, "task_type": r["task_type"], "metric": r["metric"], "mine": r["score"]}
        for ref_model, ref_scores in reference_results.items():
            row[ref_model] = ref_scores.get(task)
        rows.append(row)
    return rows
```

The comparison is only meaningful against models that reported the *same* task list on the *same* benchmark version. Pull reference numbers from:

- The model card on Hugging Face (every MTEB-scored model reports its per-task numbers).
- The MTEB leaderboard CSV at <https://huggingface.co/spaces/mteb/leaderboard> — downloadable as a snapshot.
- The original MTEB paper's Appendix tables for the baselines in v1.

Record *which* leaderboard snapshot you pulled; the ranking shifts weekly as new models are added.

## 3. Validation steps

1. **Adapter prefix round-trip.** Build the `MTEBEncoder` with `query_prefix="query: "`; encode `["hello"]` via `encode_queries` and again via `encode`; assert they produce different vectors. If equal, the prefix branch is dead.
2. **Prefix ablation smoke test.** Run MTEB-STS on `intfloat/e5-base-v2` with and without the `"query: "` / `"passage: "` prefixes; the prefixed run should be strictly above. If not, the prefix is being dropped in the adapter.
3. **Normalisation check.** Encode a batch; assert `np.linalg.norm(emb, axis=-1)` is within `1e-6` of `1.0`. If not, `normalize_embeddings=True` is being ignored by the downstream model and `cos_sim_*` metrics drift.
4. **Reproducibility against the model card.** For a published model (`all-MiniLM-L6-v2`, say), rerun one task from the card's reported list (STSBenchmark, Banking77); your Spearman / accuracy should land within ±0.3 of the card's number. A bigger gap flags an adapter bug.
5. **MTEB result-schema shape.** Open the written JSON; assert the top-level keys include `mteb_dataset_name`, `mteb_task_type`, and a split (`test` or `dev`) with the expected primary metric key under it. The leaderboard scorer errors on missing fields.
6. **Per-task-type average sanity.** Compute the macro average of the primary metric over each task type separately; assert each task type has at least two tasks in your subset (one-task averages are noise). Exception: Summarisation has one dataset (SummEval) in v1 — note it.
7. **Fairness-of-comparison check.** When reporting "my model beats E5-base-v2 by X", the comparison must be on the same task list; MTEB's full-v1 average is 56 tasks, your subset is <20, and the two numbers are not interchangeable. Report the exact overlap.
8. **Determinism.** Rerun the whole suite twice; task scores should be identical (MTEB's encoders are deterministic given the same input) except for Clustering, which uses KMeans with a seed. `mteb` seeds KMeans; if your run differs, check that you are not overriding the random state in the adapter.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Task coverage** | ≥ 5 task types with ≥ 3 tasks each (except Summarisation = 1 in v1) | 3 task types | 1 task type ("MTEB score" reported) |
| **Prefix handling** | Adapter applies the correct prefix per model family; `encode_queries` vs `encode_corpus` branch present; smoke test | Prefix applied, no smoke test | Prefix ignored or wrong family |
| **Normalisation** | `normalize_embeddings=True`; assertion test | Normalisation applied, no test | Raw vectors with `cos_sim_*` metrics |
| **Primary metric fidelity** | Per task type: nDCG@10 (Retrieval), MAP (Reranking), V-measure (Clustering), AP (Pair Classification), Accuracy (Classification), Spearman (STS, Summarisation) | Mostly correct, one off | Metric swapped on one or more task types |
| **Result schema** | MTEB JSON schema preserved; one JSON per task under `<out>/<model>/<rev>/<task>.json` | Non-standard JSON layout | Free-form logs; not leaderboard-compatible |
| **Reference comparison** | At least 2 published reference encoders; comparison on exact same task list | One reference, loose task overlap | No reference, or MTEB-v1 average vs subset |
| **Version pinning** | `mteb` library version + model revision + benchmark revision recorded | One of those | None |
| **Reproducibility** | Seed; KMeans seed respected; per-run JSON regenerates to byte parity (excluding timestamps) | Seed only | Non-deterministic |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks noted (benchmark overfitting; evaluation dataset leakage into training data of reported baselines) | Mentioned, no mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure: task list, task-type averages, per-task primary metric, prefix convention, `mteb` version, known coverage gaps | One of those | Neither |

## 5. Common mistakes

- **Dropping the prefix on E5 or BGE at evaluation.** The single largest unforced error in MTEB reporting. E5-base-v2 loses ~5 nDCG@10 on average retrieval and ~3 Spearman on STS without prefixes. The deltas are so systematic they can be used to detect the bug.
- **Reporting a single "MTEB score" over a subset.** MTEB-v1's "average" is defined over 56 English tasks; "MTEB-score" over your five retrieval tasks is not that number. Call it "retrieval average over {5 tasks}" or "per-task-type average over {retrieval, sts, classification}".
- **Using `dot_ap` because `cos_sim_ap` is marginally lower.** They are different primary metrics; the leaderboard cell is `cos_sim_ap`. Swapping without noting it is a form of cherry-picking.
- **Clustering-task variance ignored.** Clustering V-measure with KMeans is sensitive to initialisation. `mteb` defaults to multiple restarts with a fixed seed; if you override `n_init=1`, the variance across restarts can be >5 V-measure. Leave the defaults.
- **Running classification with the model's embeddings as inputs to a *trainable* classifier.** MTEB's Classification tasks evaluate a *linear probe* on frozen embeddings; that is the point. A fully-fine-tuned classifier head is a different task and a different number.
- **Running retrieval with `top_k < 10`.** nDCG@10 requires ranking at least 10 candidates per query; if your retriever returns top-5, the metric is undefined past rank 5. `mteb` requests top-100 by default; don't short-circuit.
- **Confusing MTEB v1 and MTEB-MIEB / MMTEB.** MTEB-v1 is English, 58 tasks (Muennighoff et al. 2023). The multilingual extension (MMTEB / "Massive Multilingual Text Embedding Benchmark", Enevoldsen et al. 2024) covers 1000+ tasks across languages. They are not interchangeable; the leaderboard is partitioned. Record which.
- **Pulling reference numbers from one snapshot and own numbers from a newer `mteb`.** The library changes task splits between releases; the published card number for a model may be from `mteb==1.1.0` and yours from `mteb==1.11.0`. Pin the library version in `requirements.txt` and re-run the reference if you upgrade.
- **Treating Summarisation (SummEval) as the summarisation-quality headline.** MTEB-Summarisation tests how well *the embedding similarity between a summary and the source* correlates with human quality judgments on SummEval. It is a sentence-similarity test using a summarisation corpus; it does not measure a model's summarisation ability. Module mod-106 covers true summarisation evaluation (ROUGE, BERTScore, SummaC).
- **Averaging Spearman rank correlations with an arithmetic mean.** Fisher z-transform is the honest way to average correlations. The MTEB paper does report arithmetic means (it is the community convention); match whichever your comparison target used, but flag the choice.
- **Not re-encoding for each task.** `mteb` caches corpus embeddings per task; if you share a global cache across models, you can accidentally score model B's queries against model A's passage index. Clear the cache between models.

## 6. References

Primary sources used in this solution:

- Muennighoff, N. et al. (2023). *MTEB: Massive Text Embedding Benchmark*. EACL 2023. <https://arxiv.org/abs/2210.07316> — the benchmark, its eight task types, and the primary metrics for each type (Table 1).
- `mteb` library — <https://github.com/embeddings-benchmark/mteb> — the reference implementation; the `MTEB` class, the task list, and the result-schema JSON used in §2.3–§2.4.
- MTEB Leaderboard — <https://huggingface.co/spaces/mteb/leaderboard> — the canonical public ranking; source of comparison numbers for §2.5.
- Enevoldsen, K. et al. (2024). *MMTEB: Massive Multilingual Text Embedding Benchmark*. <https://arxiv.org/abs/2502.13595> — the multilingual extension; cited in common-mistakes for version disambiguation. The MMTEB paper's exact title and task count have shifted across preprint revisions, so verify against the current arXiv version at citation time rather than quoting this reference verbatim.
- Reimers, N., Gurevych, I. (2019). *Sentence-BERT*. EMNLP. <https://arxiv.org/abs/1908.10084> — the `SentenceTransformer` library the adapter wraps.
- Wang, L. et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5). <https://arxiv.org/abs/2212.03533> — E5 and the `"query: " / "passage: "` prefix convention used in §2.2.
- Xiao, S. et al. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding* (BGE). <https://arxiv.org/abs/2309.07597> — BGE and its asymmetric query-only prefix.
- Fabbri, A. R. et al. (2021). *SummEval: Re-evaluating Summarization Evaluation*. TACL. <https://arxiv.org/abs/2007.12626> — the MTEB-Summarisation task source dataset and what it actually measures.
- Thakur, N. et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. NeurIPS Datasets & Benchmarks. <https://arxiv.org/abs/2104.08663> — the BEIR datasets reused in MTEB-Retrieval.
- Agirre, E. et al. (2012-2017). *SemEval Semantic Textual Similarity shared tasks*. — STS12-16 and STSBenchmark references used in MTEB-STS. Overview: <https://aclanthology.org/S12-1051/>.
- Rosenberg, A., Hirschberg, J. (2007). *V-measure: A Conditional Entropy-Based External Cluster Evaluation Measure*. EMNLP. <https://aclanthology.org/D07-1043/> — the MTEB-Clustering primary metric.
- Fisher, R. A. (1915). *Frequency Distribution of the Values of the Correlation Coefficient in Samples from an Indefinitely Large Population*. — the Fisher z-transform; relevant when averaging correlations.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML06 AI Supply Chain Attacks* and *ML08 Model Skewing*: published embedding models may have been trained on datasets that overlap with MTEB evaluation splits (benchmark contamination); document the model's training data origin when citing MTEB numbers.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for adversarial ML; benchmark gaming (selective reporting, prompt-tuning on eval splits) is a relevant pattern when comparing public leaderboard entries.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record benchmark revision, task list, prefix convention, `mteb` library version, and known coverage gaps (e.g., English-only, legal/medical underrepresented in v1). *Measure*: per-task-type primary metric and per-task-type macro-average; disclose when a single "MTEB score" is actually a subset.
