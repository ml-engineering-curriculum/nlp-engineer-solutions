# exercise-03 — MTEB Evaluation Suite

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-108-embeddings-and-representation-learning/exercises/exercise-03-mteb-evaluation-suite.md`.
> Target learning outcomes implied by that filename: run the **Massive Text Embedding Benchmark (MTEB)**
> against your own embedding model, read its **task-category metrics correctly**, and produce a reproducible
> report that is comparable to the public leaderboard — without inflating scores by overfitting the benchmark.

## 1. Solution overview

MTEB (Muennighoff et al. 2022) is a benchmark for general-purpose text embeddings that aggregates
tasks from BEIR (retrieval), STS datasets (semantic similarity), clustering benchmarks, reranking
benchmarks, and short-text classification suites into **one standard evaluation harness**. MTEB is
run via the open `mteb` Python package, which wraps datasets, metric calculation, and result
serialisation. The companion leaderboard on Hugging Face Hub is the de-facto reference for comparing
general-purpose embedding models.

Three subtleties reliably break first-timers' numbers:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Prefix / instruction mismatch** | E5, BGE, GTE, Instructor, and Nomic each have a *different* required prefix or instruction template. Running them bare drops task scores by 3–15 points, especially on retrieval and STS. The MTEB leaderboard numbers were produced with the prefixes in place | Encode with the model's documented prefix convention. `mteb.MTEB(..., model_prompts=...)` supports per-task prompts; the cleanest path is a thin wrapper that adds the prefix in `encode()` |
| **Misreading the aggregate "average"** | MTEB's headline "average" score is the arithmetic mean across task categories, each normalised to its own primary metric (nDCG@10 for retrieval, Spearman for STS, accuracy for classification, V-measure for clustering). A model that is best at STS but terrible at retrieval can top one category average and still trail at aggregate | Report the **category-level breakdown** alongside the aggregate; never cite a single number without the per-category table |
| **Benchmark contamination** | Training on MS MARCO + NFCorpus + SciFact + QuoraRetrieval and then reporting MTEB retrieval shows the model's memorisation of those same splits, not generalisation. Several top leaderboard entries have been retracted or asterisked for this | Hold out the MTEB splits from training; if the training data includes any MTEB-listed dataset, disclose it on the model card. The leaderboard's **training-data column** exists for this reason |

The deliverable is an evaluation harness with five components:

1. **Model wrapper** — your embedding model exposed via the `mteb` model API (either `SentenceTransformer`, which `mteb` recognises, or a thin class with an `encode` method).
2. **Prefix / instruction policy** — explicit code path that ensures the model's required prefixes are applied.
3. **Task runner** — either the full 56-task English MTEB, a named subset (`mteb.MTEB(task_types=["Retrieval"])`), or a language-specific subset (e.g. MTEB-French, MTEB-Chinese).
4. **Results card** — JSON / Markdown dump of per-task scores, per-category averages, aggregate average, model revision, mteb version, and runtime per task.
5. **Leaderboard-submission checklist** — training-data disclosure, prefix documented, model card filled.

Reference models for comparison:

- **`sentence-transformers/all-MiniLM-L6-v2`** — 22M params; the "cheap but reasonable" baseline on English MTEB.
- **`sentence-transformers/all-mpnet-base-v2`** — 110M params; pre-E5 era general-purpose.
- **`intfloat/e5-base-v2`** / **`e5-large-v2`** — require `"query: "` / `"passage: "` prefixes.
- **`BAAI/bge-base-en-v1.5`** / **`bge-large-en-v1.5`** — require `"Represent this sentence for searching relevant passages: "` on queries for retrieval tasks.
- **`BAAI/bge-m3`** — multilingual dense + sparse + ColBERT-style; evaluated via `MTEB` task-by-task because each head has a different scoring function.

MTEB English task mix (as of mteb 1.x) — totals the headline 56 tasks:

| Category | # tasks | Primary metric | Example tasks |
|---|---|---|---|
| **Classification** | 12 | Accuracy (or AP for binary) | `AmazonCounterfactualClassification`, `Banking77Classification`, `EmotionClassification` |
| **Clustering** | 11 | V-measure | `ArxivClusteringS2S`, `BiorxivClusteringP2P`, `RedditClustering` |
| **Pair Classification** | 3 | Average precision | `SprintDuplicateQuestions`, `TwitterSemEval2015`, `TwitterURLCorpus` |
| **Reranking** | 4 | mean MRR / MAP | `AskUbuntuDupQuestions`, `MindSmallReranking`, `SciDocsRR`, `StackOverflowDupQuestions` |
| **Retrieval** | 15 (BEIR subset + MSMARCO) | nDCG@10 | `MSMARCO`, `NQ`, `HotpotQA`, `FiQA2018`, `SCIDOCS`, `NFCorpus`, `ArguAna`, `ClimateFEVER`, … |
| **STS** | 10 | Spearman cosine | `STS12`, `STS13`, `STS14`, `STS15`, `STS16`, `STSBenchmark`, `STS17`, `STS22`, `SICK-R`, `BIOSSES` |
| **Summarization** | 1 | Spearman | `SummEval` |

The headline **aggregate** on the leaderboard is the mean of the category-wise means, not the mean of all 56 task scores. These differ when categories have different task counts — which they always do.

## 2. Worked implementation

### 2.1 Project layout

```
mteb_bake/
  model.py             # SentenceTransformer or custom encoder wrapper with prefix policy
  run.py               # mteb.MTEB(...) runner
  report.py            # aggregate per-category + emit results.json / results.md
  tests/
    test_prefix_applied.py
    test_small_subset.py
  requirements.txt
```

### 2.2 Model wrapper — prefix policy is non-negotiable

```python
# model.py
from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer


class PrefixedBiEncoder:
    """
    Thin wrapper that enforces the model's query / passage prefix convention.

    MTEB task runners call `encode_queries` and `encode_corpus` for retrieval /
    reranking, and `encode` for classification / clustering / STS / pair tasks.
    Implementing both pairs of methods is the only way to apply retrieval-specific
    prefixes while leaving non-retrieval tasks unprefixed.
    """

    # Prefix templates per model family. These strings are exactly what the model
    # card recommends; using a paraphrase silently drops a few points.
    PREFIX = {
        "intfloat/e5": {"query": "query: ", "passage": "passage: "},
        "BAAI/bge": {"query": "Represent this sentence for searching relevant passages: ", "passage": ""},
        "sentence-transformers": {"query": "", "passage": ""},
    }

    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.model = SentenceTransformer(model_id, device=device)
        self.family = self._detect_family(model_id)
        self.query_prefix = self.PREFIX[self.family]["query"]
        self.passage_prefix = self.PREFIX[self.family]["passage"]

    @staticmethod
    def _detect_family(model_id: str) -> str:
        if model_id.startswith("intfloat/e5"):
            return "intfloat/e5"
        if model_id.startswith("BAAI/bge"):
            return "BAAI/bge"
        return "sentence-transformers"

    # --- MTEB retrieval / reranking hooks ---
    def encode_queries(self, queries, batch_size: int = 64, **kwargs) -> np.ndarray:
        texts = [self.query_prefix + q for q in queries]
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, **kwargs)

    def encode_corpus(self, corpus, batch_size: int = 64, **kwargs) -> np.ndarray:
        # mteb passes corpus as list[dict] with "title" and "text" keys.
        texts = []
        for doc in corpus:
            if isinstance(doc, dict):
                t = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            else:
                t = doc
            texts.append(self.passage_prefix + t)
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, **kwargs)

    # --- MTEB non-retrieval hook ---
    def encode(self, sentences, batch_size: int = 64, **kwargs) -> np.ndarray:
        # For non-retrieval tasks, the sentence-transformers default (no prefix)
        # is what the leaderboard numbers used for E5/BGE. Do NOT add a prefix here.
        return self.model.encode(sentences, batch_size=batch_size, normalize_embeddings=True, **kwargs)
```

Points worth defending in review:

- **`encode_queries` vs `encode_corpus` vs `encode`.** MTEB inspects the model for these method names; if `encode_queries` is absent it falls back to `encode` for everything, which is how prefixes get dropped silently.
- **Prefixes only on retrieval / reranking.** E5's `"query: "` prefix on an STS task hurts STS. The published leaderboard numbers for E5 used the prefix only on retrieval.
- **`normalize_embeddings=True`.** MTEB's retrieval scorer expects unit vectors (or dot-product-equivalent). Leaving it off drops retrieval nDCG by 2–5 points on cosine-trained models.
- **Explicit family detection.** Grepping the model id is better than a config flag because you cannot forget it. If you add a new family (Nomic, Jina, Cohere, Mixedbread), add its prefix here.

### 2.3 Running MTEB — full English, subset, or multilingual

```python
# run.py
import argparse
from pathlib import Path
import mteb
from .model import PrefixedBiEncoder


FAST_SUBSET = [
    # Smoke-test subset — ~10 min on a single A10. Keep retrieval, STS, clustering,
    # classification represented. Full English MTEB takes hours.
    "Banking77Classification",
    "STSBenchmark",
    "SciDocsRR",
    "NFCorpus",
    "TwentyNewsgroupsClustering",
    "SprintDuplicateQuestions",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="intfloat/e5-base-v2")
    p.add_argument("--output-dir", default="mteb_runs")
    p.add_argument("--subset", choices=["fast", "english", "retrieval", "sts", "custom"], default="fast")
    p.add_argument("--tasks", nargs="*", default=None)  # with --subset custom
    p.add_argument("--revision", default=None, help="Pin the HF revision for reproducibility")
    a = p.parse_args()

    encoder = PrefixedBiEncoder(a.model_id)

    if a.subset == "fast":
        tasks = mteb.get_tasks(tasks=FAST_SUBSET)
    elif a.subset == "english":
        tasks = mteb.get_benchmark("MTEB(eng, v1)")
    elif a.subset == "retrieval":
        tasks = mteb.get_tasks(task_types=["Retrieval"], languages=["eng"])
    elif a.subset == "sts":
        tasks = mteb.get_tasks(task_types=["STS"], languages=["eng"])
    else:
        tasks = mteb.get_tasks(tasks=a.tasks)

    evaluation = mteb.MTEB(tasks=tasks)
    out = Path(a.output_dir) / a.model_id.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    evaluation.run(encoder, output_folder=str(out), eval_splits=["test"])
```

Key choices:

- **`get_benchmark("MTEB(eng, v1)")` returns the *canonical* 56-task English set.** The exact list has changed across `mteb` versions; always pin the version (`mteb==1.x.y`) and state which benchmark snapshot you used. From `mteb` ≥ 1.11, benchmark versioning is first-class.
- **`eval_splits=["test"]`.** Dev splits are for your own tuning; the leaderboard is computed on test. Never cite dev numbers as MTEB numbers.
- **`--subset fast` for iteration.** Running the full English set on a single A10 takes 2–6 hours (dominated by retrieval tasks); the fast subset is ~10 minutes and catches 80 % of regressions.
- **Pin `--revision`.** `SentenceTransformer(model_id, revision="abc123")` locks the HF snapshot. The same `model_id` can point to a different checkpoint six months later and silently change your numbers.

### 2.4 Aggregating — category means, then category-mean-of-means

```python
# report.py
import json
import statistics
from pathlib import Path


# MTEB primary metric per task type (mteb 1.x convention).
PRIMARY = {
    "Classification": "accuracy",
    "Clustering": "v_measure",
    "PairClassification": "ap",
    "Reranking": "map",
    "Retrieval": "ndcg_at_10",
    "STS": "cos_sim.spearman",
    "Summarization": "cos_sim.spearman",
}

# Mapping task name -> category; populate from mteb.get_tasks() in real code.
# Kept here as a sketch.
TASK_CATEGORY = {"NFCorpus": "Retrieval", "STSBenchmark": "STS", ...}


def collect_scores(results_dir: Path) -> dict:
    out = {}
    for f in results_dir.glob("**/*.json"):
        data = json.loads(f.read_text())
        task = data["mteb_dataset_name"]
        cat = TASK_CATEGORY.get(task)
        if not cat:
            continue
        metric = PRIMARY[cat]
        # mteb results have nested structure: data["scores"]["test"][lang][metric].
        score = _extract_nested(data["scores"]["test"]["eng"], metric)
        out.setdefault(cat, {})[task] = score
    return out


def category_means(scores: dict) -> dict:
    return {cat: statistics.mean(v.values()) for cat, v in scores.items()}


def mteb_aggregate(scores: dict) -> float:
    """The headline MTEB number: the mean of category means (NOT the mean of task scores)."""
    cat_means = category_means(scores)
    return statistics.mean(cat_means.values())


def _extract_nested(d: dict, dotted: str):
    cur = d
    for k in dotted.split("."):
        cur = cur[k]
    return cur
```

The leaderboard's "average" column is `mean_over_categories(mean_over_tasks_in_category(primary_metric))`. Computing `mean_over_all_task_scores` instead over-weights retrieval (15 tasks) relative to pair classification (3 tasks) and will not match the leaderboard.

### 2.5 Report format — what to publish

```
runs/e5-base-v2/
  results.json         # raw per-task
  results.md           # human-readable table
  METADATA.json        # model_id, revision, mteb version, prefix policy, hardware, runtime
```

`results.md` should include:

- A one-line "aggregate" with the category-mean-of-means.
- A per-category table: category name, # tasks, mean score.
- A per-task appendix with scores and runtime.
- The **prefix policy section**: which prefix was applied to which task type, and the source for each (model card URL).
- The **training-data disclosure**: a short list of training pairs with sources, plus an explicit "no MTEB task test splits were in the training corpus" assertion (or the honest disclosure if they were).

## 3. Validation steps

1. **Prefix-applied unit test.** On your `PrefixedBiEncoder`, assert that `encode_queries(["hello"])[0]` is *not* equal to `encode(["hello"])[0]` for an E5 / BGE family model, and that the two *are* equal for a `sentence-transformers` family model. If the first assertion fails, your retrieval tasks are running without the prefix.
2. **Fast-subset parity.** Run `FAST_SUBSET` with the published checkpoint of `intfloat/e5-base-v2` and compare each per-task score to the published number. All six should match within ±0.3 (the Spearman / nDCG rounding noise). If any is off by more than 1 point, the prefix, pooling, or normalization is wrong.
3. **Retrieval sanity.** For `NFCorpus` specifically, nDCG@10 on `all-MiniLM-L6-v2` should land around 0.316; `e5-base-v2` around 0.372; `bge-base-en-v1.5` around 0.373 (approximate, within the model-card range). A number outside ±2 points of these flags a config bug.
4. **Category breakdown read.** After running the English suite, write the aggregate and per-category table by hand; confirm the aggregate from §2.4 matches your `report.py` output. If the aggregate from `report.py` is *higher* than the leaderboard's for the same model, you're probably averaging over tasks instead of over categories.
5. **Training-data contamination audit.** Write a script that iterates every training `(anchor, positive)` pair and checks the (query, corpus_doc) pairs for each MTEB retrieval task; count any exact-string overlaps. The report must either say "0 overlaps" or disclose the count.
6. **Reproducibility replay.** With `mteb` version, model revision, prefix policy, and hardware pinned, rerun the fast subset after a week; scores must match to within rounding. If they drift, something (HF Hub snapshot? torch version? `faiss` nn-backend?) is not pinned.
7. **Runtime vs model size check.** Expect roughly linear scaling of retrieval runtime with parameter count. If `bge-large` takes *less* time than `bge-base`, batching is misconfigured.
8. **Spearman-vs-Pearson check.** MTEB STS uses **Spearman** (rank correlation), not Pearson. If your STS number is suspiciously high (>0.9 on STSBenchmark for a base-size model), you may be reading Pearson from the results blob — look for `cos_sim.spearman`, not `cos_sim.pearson`.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Prefix policy** | Explicit `encode_queries`/`encode_corpus`/`encode` with family-specific prefixes; unit-tested | Prefix on all encode calls | No prefixes; E5 / BGE score 5+ points below card |
| **Benchmark version pinned** | `mteb` version + benchmark snapshot name recorded | `mteb` version only | None |
| **Model revision pinned** | HF model id + revision hash in METADATA | Model id only | None |
| **Correct aggregation** | Category-mean-of-means; matches leaderboard for a known model | Mean of all task scores | A single-task score called "the MTEB number" |
| **Per-category breakdown** | Reports category means and per-task scores | Aggregate only | None |
| **Test split only** | `eval_splits=["test"]`; dev-split numbers clearly labelled "dev" when shown at all | Mix of dev and test, labelled | Dev numbers reported as MTEB numbers |
| **Training-data disclosure** | Explicit audit; either 0-overlap claim or per-dataset disclosure | Mentioned informally | None |
| **Runtime / cost card** | Wall time per task, GPU type, total GPU-hours | Total runtime only | None |
| **Reproducibility** | Seed, prefix, revision, mteb version, task list all in METADATA | Some of those | None |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS noted (benchmark gaming / prompt-injection via training contamination) | Mentioned without mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure: tasks, categories, aggregate, prefixes, training-data provenance | Partial | Missing |

## 5. Common mistakes

- **Running E5 without the `"query: "` / `"passage: "` prefix.** Costs 3–8 points on retrieval and 2–4 points on STS. Diagnostic: run `MSMARCO` on `e5-base-v2` with and without; if the delta is under 1 point, the prefix is being dropped at both ends (you are comparing bare-vs-bare).
- **Running BGE without the retrieval instruction.** BGE v1.5's `"Represent this sentence for searching relevant passages: "` is applied **only to the query side** for retrieval. Applying it to passages (or to STS) slightly hurts.
- **Reporting dev MTEB numbers.** The leaderboard is test-split-only. Dev numbers are typically higher (dev is used by model authors for tuning) and not comparable.
- **Aggregating across tasks instead of categories.** The leaderboard reports category-mean-of-means. A model with 15 retrieval wins and 3 pair-classification losses will beat on task-mean but tie on category-mean, and only the latter is the leaderboard number.
- **Overfitting MTEB by training on its evaluated splits.** The leaderboard added a "Training Data" column after several models were disqualified for this. If any of MS MARCO, NQ, HotpotQA, FiQA, SciFact, NFCorpus, Quora, TREC-COVID, Touché-2020, DBPedia, CQADupStack, ArguAna, ClimateFEVER, SCIDOCS, or FEVER is in your training corpus, you must disclose it.
- **Comparing against stale leaderboard snapshots.** The MTEB leaderboard rankings move; a number 6 months old may be mid-tier now. State the leaderboard date you compared to.
- **Running MTEB on a laptop CPU and giving up on retrieval tasks.** The retrieval category dominates runtime. A reasonable iteration loop is CPU → `FAST_SUBSET`; GPU → full English suite. Full English on CPU is a day+.
- **Letting `mteb` auto-download without caching.** Each task downloads a dataset; without `HF_DATASETS_CACHE` pinned, CI re-downloads every run. Set a persistent cache.
- **Leaving `normalize_embeddings=False` on a cosine-trained model.** Retrieval scores flip from cosine to raw dot product; `nDCG@10` can move 2–5 points. Match the model card's scoring.
- **Not pinning `mteb` version.** Benchmark composition, dataset revisions, and primary metrics have shifted between `mteb` 1.0 and 1.x. Pin the version and quote it in the report.
- **Reporting individual task deltas as "MTEB performance".** A model may top `SciFact` by 2 points and lose `MSMARCO` by 6; a single-task win is not an MTEB win. Report the category breakdown.
- **Interpreting MTEB as a universal-quality measure.** MTEB skews English, web-and-academic, short-sentence; it does not measure long-document retrieval well, does not measure code embedding (`CoIR` is a separate benchmark), and does not measure multilingual retrieval exhaustively (see MTEB-French / MTEB-Chinese / multilingual MTEB for those). Pair MTEB with task-specific evals for your actual use case.

## 6. References

Primary sources used in this solution:

- Muennighoff, N., Tazi, N., Magne, L., Reimers, N. (2022). *MTEB: Massive Text Embedding Benchmark*. EACL 2023. <https://arxiv.org/abs/2210.07316> — the benchmark construction, task categories, and aggregation rules used throughout §1–§2.
- Thakur, N. et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. NeurIPS. <https://arxiv.org/abs/2104.08663> — the retrieval subset of MTEB inherits BEIR's `nDCG@10` convention and dataset list.
- Wang, L. et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5). <https://arxiv.org/abs/2212.03533> — E5 prefix convention and leaderboard numbers used in §2.2 / §3.
- Xiao, S. et al. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding* (BGE). <https://arxiv.org/abs/2309.07597> — BGE's retrieval instruction prefix.
- Chen, J. et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. <https://arxiv.org/abs/2402.03216> — the multilingual MTEB variant and model.
- Rosenberg, A., Hirschberg, J. (2007). *V-measure: A conditional entropy-based external cluster evaluation measure*. EMNLP. <https://aclanthology.org/D07-1043/> — V-measure, the clustering primary metric.
- Agirre, E. et al. (2012–2017). STS SemEval shared-task papers — the STS12–STS17 datasets evaluated in the STS category.
- Nguyen, T. et al. (2016). *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset*. <https://arxiv.org/abs/1611.09268> — a retrieval task in MTEB.
- MTEB leaderboard on Hugging Face — <https://huggingface.co/spaces/mteb/leaderboard> — the canonical dashboard whose aggregation convention §2.4 reproduces.
- `mteb` Python package (Muennighoff et al.) — <https://github.com/embeddings-benchmark/mteb> — API and task registry used in §2.3.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML02 Data Poisoning* applies to contamination: a model trained on a benchmark's test split is effectively poisoned toward that benchmark. *ML09 Output Integrity* applies when published leaderboard numbers are used to pick a model that is then deployed — if the leaderboard number was inflated by contamination, downstream users consume the inflated signal as truth.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for adversarial ML; `AML.T0042 Craft Adversarial Data` is relevant whenever benchmark leakage lets a model memorise test queries.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: benchmark + version + task list + model revision; *Measure*: category breakdown with primary metrics and category-mean-of-means aggregate; *Govern*: the "Training Data" disclosure column on the MTEB leaderboard is the community's answer to the AI RMF's transparency imperative — reproduce it on your own model card.
