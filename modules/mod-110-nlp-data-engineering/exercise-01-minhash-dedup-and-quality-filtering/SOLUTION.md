# exercise-01 — MinHash Dedup and Quality Filtering

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-110-nlp-data-engineering/exercises/exercise-01-minhash-dedup-and-quality-filtering.md`.
> The target learning outcomes implied by that filename: build the two first-pass filters that turn a raw web/text crawl into a training-grade corpus — **MinHash+LSH near-duplicate removal** at document and paragraph granularity, and **quality heuristics** in the Gopher / C4 / RefinedWeb tradition — report what fraction of the corpus each filter drops, and publish a reproducible manifest so a downstream reviewer can tell whether a given document was kept, deduped-out, or quality-dropped and why.

## 1. Solution overview

Large-text corpora reach the embedding/LM trainer with two dominant failure modes that cheap filters can catch before expensive ones (classifier-based filters, perplexity filters, language-ID) run:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Near-duplicates inflate effective-epoch counts** | Boilerplate, re-hosted articles, template pages, and licence footers are identical or one-edit-different across millions of documents. Training on them over-weights those sequences; Lee et al. 2022 showed dedup alone recovers ~1 % perplexity and halves memorisation | Shingle each document, MinHash-sketch each shingle set, LSH-bucket by Jaccard threshold; keep one representative per connected-component. Run at both **document** and **paragraph** granularity because document-dedup leaves paragraph-level boilerplate in place |
| **Low-quality documents dominate the long tail** | Common Crawl is ~40 % non-prose boilerplate (menus, linkfarms, SEO-generated pages). A trained model learns their n-gram statistics as readily as the prose | Apply the Gopher / C4 cheap-heuristic battery (length, symbol ratio, stopword ratio, repeated-line ratio, bullet-point ratio, ellipsis ratio) and record a per-filter drop-rate dashboard so regressions after a crawler change are visible |
| **Silent corpus drift** | You rebuild the corpus, "the numbers move", and no one knows whether the model changed or the corpus did | Emit a manifest `{doc_id, kept/dup/quality, filter_hits, cluster_id}` next to the final JSONL; archive it with the checkpoint. The manifest + the raw snapshot is enough to re-derive the training corpus bit-for-bit |

The deliverable is a two-stage cleaner with five components:

1. **Shingler** — token- or char-n-gram shingles of each document; `n=5` for word-tokens is the RefinedWeb / The-Pile standard.
2. **MinHash + LSH index** — `datasketch` `MinHash`/`MinHashLSH` with a **Jaccard threshold of 0.8**, **128 permutations** (RefinedWeb / SlimPajama) or **256** if false-positive budget is tighter. Band/row counts fall out of the threshold.
3. **Connected-component clustering** — union-find over LSH candidate pairs; keep one representative per cluster (longest, or lowest `doc_id`, pinned to the manifest).
4. **Quality filter battery** — the Gopher cheap-heuristic set plus C4's noisy-word filter; each filter is a pure function `doc -> (kept: bool, reason: str)` so hits are counted and logged.
5. **Manifest** — one JSONL row per *input* document recording the keep/drop decision and the filter(s) that fired; this is the audit artefact.

Reference pipelines (public, documented) that implement this pattern:

- **C4** (Raffel et al. 2020, "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer") — the "Colossal Clean Crawled Corpus" filters and the famous naughty-word list.
- **The Pile** (Gao et al. 2020) — SHA-256 exact-dup then MinHash near-dup across constituent sub-corpora; well documented in appendix.
- **Gopher / MassiveText** (Rae et al. 2021, DeepMind) — the canonical 5-rule cheap-heuristic quality filter (mean-word-length, symbol-to-word-ratio, hash-to-word-ratio, ellipsis ratio, alphabetic ratio, stopword presence).
- **RefinedWeb** (Penedo et al. 2023, Falcon) — document + line-level MinHash dedup; 5-gram shingles; threshold 0.8; 9 128-perm MinHash bands. The paper is the clearest end-to-end recipe.
- **SlimPajama** (Soboleva et al. 2023, Cerebras) — 627 GB redpajama MinHash dedup at threshold 0.8, `datasketch`-based reference implementation.
- **FineWeb / FineWeb-Edu** (Penedo et al. 2024, HuggingFace) — same lineage, with the exact filter thresholds and ablation tables published alongside the dataset.

## 2. Worked implementation

### 2.1 Project layout

```
corpus_bake/
  shingling.py          # word/char n-gram shingles
  minhash_lsh.py        # MinHash sketches + LSH bucketing
  cluster.py            # union-find connected components
  quality/
    __init__.py
    gopher.py           # the Gopher 2021 heuristics
    c4.py               # C4 noisy-word + line-level filters
    runner.py           # compose filters, count hits
  manifest.py           # JSONL audit artefact
  run.py                # CLI
  tests/
    test_minhash_jaccard.py
    test_shingling.py
    test_gopher_thresholds.py
    test_manifest_idempotence.py
  requirements.txt
```

### 2.2 Shingling — word 5-grams, lower-cased, whitespace-split

```python
# shingling.py
from typing import Iterator


def word_shingles(text: str, n: int = 5) -> Iterator[str]:
    """
    RefinedWeb/SlimPajama default: n=5 word shingles, lower-cased, whitespace tokenised.
    Yields shingles as joined strings so MinHash can hash them directly.
    """
    toks = text.lower().split()
    if len(toks) < n:
        # Short docs: yield the whole thing as one shingle so MinHash still produces a sketch
        if toks:
            yield " ".join(toks)
        return
    for i in range(len(toks) - n + 1):
        yield " ".join(toks[i : i + n])
```

Points worth defending in review:

- **Word 5-grams, not char n-grams, for English prose.** Char n-grams (used in BigCode StarCoder data pipeline) are the right choice for code, where whitespace is less reliable; for prose, word-grams give higher Jaccard separation between near-duplicates and legitimately similar texts (both RefinedWeb §3.1 and SlimPajama §3.2 justify the choice).
- **Lower-case + whitespace tokenisation, no stemming.** Stemming or BPE tokenisation couples the dedup pass to a specific tokenizer; a tokenizer rebuild would then invalidate the dedup cluster IDs. Word-lower is portable.
- **Short-document fallback.** A 3-token headline has no 5-gram shingles; emitting one whole-doc shingle keeps it representable in LSH (where it still won't collide with anything substantive, but will collide cheaply with other 3-token repeats).

### 2.3 MinHash + LSH — `datasketch`, threshold 0.8, 128 permutations

```python
# minhash_lsh.py
from datasketch import MinHash, MinHashLSH
from typing import Iterable

NUM_PERM = 128           # RefinedWeb default; 256 if FP budget is tighter
JACCARD_THRESHOLD = 0.80  # RefinedWeb / SlimPajama / FineWeb default


def sketch(shingles: Iterable[str], num_perm: int = NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def build_lsh(threshold: float = JACCARD_THRESHOLD, num_perm: int = NUM_PERM) -> MinHashLSH:
    """
    datasketch picks band/row counts to minimise the integral of the probability
    of (false positive + false negative) at the target Jaccard threshold.
    For threshold=0.8, num_perm=128, this is typically bands=25, rows=5.
    """
    return MinHashLSH(threshold=threshold, num_perm=num_perm)


def candidate_pairs(lsh: MinHashLSH, sketches: dict[str, MinHash]) -> set[tuple[str, str]]:
    """
    Insert all sketches, then query each to get its bucket-mates. Candidate
    *pairs*, not true Jaccard — the caller must verify or accept LSH FPs.
    """
    for doc_id, m in sketches.items():
        lsh.insert(doc_id, m)
    pairs: set[tuple[str, str]] = set()
    for doc_id, m in sketches.items():
        for other in lsh.query(m):
            if other == doc_id:
                continue
            pairs.add(tuple(sorted((doc_id, other))))  # canonicalise
    return pairs
```

Four points reviewers must check:

- **`num_perm` is a cost/accuracy knob, not a free parameter.** 128 is the published RefinedWeb and SlimPajama choice and the datasketch default for mid-scale corpora. 256 is used when the false-positive rate at threshold 0.8 is the dominant concern (the FP rate at the LSH stage halves roughly as `num_perm` doubles). Below 64 the MinHash estimator's variance destroys threshold discrimination.
- **The LSH threshold is the Jaccard threshold on *shingled sets*, not on document length or edit distance.** At `(n=5, threshold=0.8)` two documents of a few hundred words need to share roughly 80 % of their 5-gram shingles; that is empirically the boundary between "boilerplate + 2 % of unique material" and "two distinct documents on the same topic".
- **`MinHashLSH` is in-memory by default.** For >10 M documents use `MinHashLSHForest` or back `MinHashLSH` with a RedisStorage; the chunked / out-of-core path is the SlimPajama reference. Document this in the README; a reviewer who sees a 10 M corpus loaded into a single process has reason to ask.
- **Reverse-query after insert.** The naïve pattern (insert and then skip because you've seen yourself) is what the bucket-mates filter above does. If you implement it with a single `insert → query` pass you miss early-inserted documents that later ones would have matched; always insert first, query second.

### 2.4 Connected-component clustering — one representative per cluster

```python
# cluster.py
from typing import Iterable


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def cluster(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """
    Returns doc_id -> cluster_representative (the find() root). Callers pick
    one representative per cluster downstream (longest, or lowest id).
    """
    uf = UnionFind()
    for a, b in pairs:
        uf.union(a, b)
    return {x: uf.find(x) for x in uf.parent}
```

Three design notes worth their one line each in the README:

- **Transitive dedup is intentional.** If A ≈ B and B ≈ C but A ≈ C is below threshold, you still want to drop two of {A, B, C}; otherwise you preserve a chain. All production pipelines (RefinedWeb, FineWeb, SlimPajama) run union-find; some LLM-pretraining papers (e.g., the first TinyLlama release) did *not* transitively cluster and over-represented hub pages as a result.
- **Pick the representative deterministically.** Longest `doc_id`'s content, lowest `doc_id` string, or lexicographically first `url` — pin one in the manifest so re-runs on the same snapshot produce bit-identical output.
- **Record the cluster size in the manifest.** A tiny fraction of clusters account for most of the drops; their sizes are the diagnostic artefact when the dedup pass looks suspicious.

### 2.5 Quality filters — Gopher 2021 heuristics + C4 line-level filter

```python
# quality/gopher.py
# Reference: Rae et al. 2021, "Scaling Language Models: Methods, Analysis & Insights from
# Training Gopher", Appendix A2 "Text Quality Filtering". Thresholds are the Gopher-published
# values; teams typically tune 2-3 of them on a 1000-doc sample from their own corpus.
STOPWORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}


def gopher_filters(text: str) -> dict[str, bool]:
    """
    Returns a dict of filter_name -> passes? (True == document kept by that filter).
    Compose with AND: a doc is kept only if every filter passes.
    """
    words = text.split()
    n_words = len(words)
    n_lines = max(text.count("\n") + 1, 1)

    # 1. Word count
    len_ok = 50 <= n_words <= 100_000

    # 2. Mean word length
    mean_word_len = (sum(len(w) for w in words) / n_words) if n_words else 0.0
    word_len_ok = 3.0 <= mean_word_len <= 10.0

    # 3. Symbol-to-word ratio ("#", "...") combined
    n_hash = text.count("#")
    n_ellipsis = text.count("...") + text.count("…")
    symbol_ratio = (n_hash + n_ellipsis) / max(n_words, 1)
    symbol_ok = symbol_ratio < 0.10

    # 4. Bullet-point / ellipsis line ratio
    bullet_lines = sum(1 for ln in text.split("\n") if ln.lstrip().startswith(("•", "-", "*")))
    ellipsis_lines = sum(1 for ln in text.split("\n") if ln.rstrip().endswith(("...", "…")))
    bullet_ok = bullet_lines / n_lines < 0.90
    ellipsis_ok = ellipsis_lines / n_lines < 0.30

    # 5. Alphabetic-token ratio
    alpha_tokens = sum(1 for w in words if any(c.isalpha() for c in w))
    alpha_ok = (alpha_tokens / n_words if n_words else 0.0) >= 0.80

    # 6. Stopword presence (any of the top-8)
    tokens_lower = {w.lower().strip(".,!?;:") for w in words}
    stop_ok = bool(tokens_lower & STOPWORDS)

    return {
        "gopher_len": len_ok,
        "gopher_word_len": word_len_ok,
        "gopher_symbol": symbol_ok,
        "gopher_bullet": bullet_ok,
        "gopher_ellipsis": ellipsis_ok,
        "gopher_alpha": alpha_ok,
        "gopher_stopwords": stop_ok,
    }
```

```python
# quality/c4.py
import re

NAUGHTY_WORDS = {"...load the published C4 list at build time..."}
_JS_RE = re.compile(r"\bjavascript\b", re.IGNORECASE)
_LOREM_RE = re.compile(r"\blorem ipsum\b", re.IGNORECASE)


def c4_line_filter(text: str) -> tuple[str, dict[str, int]]:
    """
    Reference: Raffel et al. 2020, §2.2. Keep only lines that end with a terminal
    punctuation mark, contain at least 5 words, and do not mention 'javascript'
    or 'lorem ipsum'. Returns (kept_text, hits).
    """
    hits = {"short": 0, "no_terminal": 0, "javascript": 0, "lorem": 0}
    kept: list[str] = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if len(s.split()) < 5:
            hits["short"] += 1; continue
        if s[-1] not in ".!?\"'":
            hits["no_terminal"] += 1; continue
        if _JS_RE.search(s):
            hits["javascript"] += 1; continue
        if _LOREM_RE.search(s):
            hits["lorem"] += 1; continue
        kept.append(s)
    return "\n".join(kept), hits


def c4_doc_filter(text: str) -> dict[str, bool]:
    """Document-level filters from C4 §2.2."""
    low = text.lower()
    return {
        "c4_min_3_sents": sum(low.count(p) for p in ".!?") >= 3,
        "c4_min_200_chars": len(text) >= 200,
        "c4_no_naughty": not any(w in low for w in NAUGHTY_WORDS),
    }
```

Points worth defending:

- **Thresholds are reproductions of published values, not inventions.** The Gopher 50-word minimum, 3–10 mean-word-length band, and ≥80 % alphabetic ratio are from Rae et al. 2021 Appendix A2; the C4 ≥5-word terminal-punctuation line filter is from Raffel et al. 2020 §2.2. Teams typically tune the symbol/bullet/ellipsis thresholds on a labelled 1 000-doc sample from their own corpus (e.g., scientific text legitimately uses more ellipsis and bullets) and publish the deltas.
- **Compose with AND, not OR, inside the Gopher battery.** Each rule is a cheap targeted filter; false-negatives add up fast. Measure the drop-rate per rule on a 10k-doc sample and publish the breakdown; rules that drop >5 % on your domain are suspect.
- **C4's naughty-word list is controversial.** Dodge et al. 2021 ("Documenting Large Webtext Corpora") showed the list disproportionately filters LGBTQ and minority-dialect content; a serious deployment either (a) drops the naughty-word filter in favour of a separate toxicity model (exercise-02) or (b) publishes the fairness-impact analysis Dodge et al. introduced. Do not silently use the list.
- **Line-level filters go *before* MinHash dedup when the goal is paragraph dedup.** Boilerplate lines (cookie banners, nav menus) dominate MinHash sketch collisions; remove them first and your paragraph-dedup cluster sizes become interpretable.

### 2.6 Manifest — one JSONL row per input doc, every drop traceable

```python
# manifest.py
import json
from pathlib import Path
from typing import Any


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    """
    Writes the audit trail. One row per input doc:
      {
        "doc_id": "...",
        "status": "kept" | "dup" | "quality",
        "cluster_id": "...",         # the MinHash representative (== doc_id for singletons)
        "cluster_size": 17,          # number of docs in that cluster before pruning
        "filter_hits": ["gopher_symbol", "c4_min_3_sents"],  # empty if kept
        "source_url": "https://...",
        "sha256": "deadbeef..."      # exact-dup key for the raw text
      }
    """
    with path.open("w") as f:
        for r in sorted(rows, key=lambda r: r["doc_id"]):   # deterministic byte order
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
```

Why the manifest is load-bearing:

- **Auditability.** A reviewer asking "why was *this* doc dropped" gets a single, grep-able answer. The manifest is checked into the dataset release next to the final JSONL.
- **Reproducibility.** The manifest plus the raw snapshot URIs is sufficient to re-derive the training corpus bit-identical. Without it, you have the artefact but not the *process*; a re-run after changing a threshold cannot be compared.
- **Fairness analysis.** The domain (URL)–filter-hits cross-tab is the one-shot answer to "is our quality filter accidentally removing a demographic". Dodge et al. 2021 ran exactly this cross-tab on C4.

### 2.7 CLI — one pass, deterministic output

```python
# run.py
import argparse, hashlib, json
from pathlib import Path
from .shingling import word_shingles
from .minhash_lsh import sketch, build_lsh, candidate_pairs
from .cluster import cluster
from .quality.gopher import gopher_filters
from .quality.c4 import c4_line_filter, c4_doc_filter
from .manifest import write_manifest


def process(input_path: Path, output_path: Path, manifest_path: Path, num_perm: int = 128, threshold: float = 0.80):
    docs: dict[str, dict] = {}
    with input_path.open() as f:
        for line in f:
            r = json.loads(line)
            docs[r["doc_id"]] = r

    # Quality pass first (removes boilerplate lines, shrinks dedup workload)
    quality_rows: dict[str, list[str]] = {}
    for doc_id, r in docs.items():
        cleaned, line_hits = c4_line_filter(r["text"])
        r["text"] = cleaned
        doc_hits = {**c4_doc_filter(cleaned), **gopher_filters(cleaned)}
        failed = [k for k, passed in doc_hits.items() if not passed]
        quality_rows[doc_id] = failed

    # MinHash dedup on the survivors
    survivors = [d for d, hits in quality_rows.items() if not hits]
    sketches = {d: sketch(word_shingles(docs[d]["text"])) for d in survivors}
    lsh = build_lsh(threshold=threshold, num_perm=num_perm)
    pairs = candidate_pairs(lsh, sketches)
    rep_of = cluster(pairs)
    kept: set[str] = set()
    seen_reps: set[str] = set()
    for d in sorted(survivors):           # deterministic
        rep = rep_of.get(d, d)
        if rep not in seen_reps:
            kept.add(d); seen_reps.add(rep)

    # Emit final corpus + manifest
    manifest: list[dict] = []
    with output_path.open("w") as out:
        for doc_id, r in docs.items():
            if doc_id in kept:
                out.write(json.dumps({"doc_id": doc_id, "text": r["text"]}) + "\n")
                status = "kept"
            elif quality_rows[doc_id]:
                status = "quality"
            else:
                status = "dup"
            manifest.append({
                "doc_id": doc_id,
                "status": status,
                "cluster_id": rep_of.get(doc_id, doc_id),
                "filter_hits": quality_rows[doc_id],
                "source_url": r.get("url"),
                "sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
            })
    write_manifest(manifest, manifest_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=0.80)
    a = ap.parse_args()
    process(a.input, a.output, a.manifest, a.num_perm, a.threshold)
```

Defensible choices:

- **Quality filter runs before dedup.** Three reasons: (1) cheaper — dedup is the slow pass, so shrinking the input first saves wall-clock; (2) more interpretable — paragraph dedup after boilerplate removal gives cluster sizes that mean "near-duplicate article", not "all share a cookie banner"; (3) consistent with FineWeb and RefinedWeb published pipelines.
- **Sorted iteration where output depends on insertion order.** Python dict iteration is insertion-ordered in CPython 3.7+ but the input ordering depends on file-system order. Sorted survivors + sorted manifest = bit-identical reruns on the same input.
- **SHA-256 of the raw text in the manifest.** This is the cheap exact-dup key; near-dup pipelines still want it so pure exact-dup can be run before MinHash as a 100× cheaper pre-pass.

## 3. Validation steps

1. **Shingle parity.** `list(word_shingles("the quick brown fox jumps over the lazy dog", n=5))` returns 5 shingles; the first is `"the quick brown fox jumps"`. Golden test; catches whitespace/case regressions.
2. **MinHash Jaccard estimator variance.** Build two random 1 000-shingle sets with known true Jaccard ∈ {0.5, 0.8, 0.95}; verify `m1.jaccard(m2)` is within `±0.05` of true at `num_perm=128` (the datasketch-documented bound is `1/sqrt(num_perm)` ≈ 0.088 at 128 perms). If estimator variance is higher, `num_perm` is being dropped somewhere.
3. **LSH recall at threshold.** Synthesize 1 000 pairs with Jaccard uniformly in [0.5, 1.0]; verify that `MinHashLSH(threshold=0.8)` returns ≥ 99 % of the true ≥0.8 pairs (LSH's false-negative probability at Jaccard 0.8 with bands=25, rows=5 is ~0.7 %). Separately measure the false-positive rate at Jaccard 0.6 (expected: < 5 %).
4. **Transitive clustering.** Insert the triplet `{A≈B@0.82, B≈C@0.82, A≈C@0.65}`; `cluster()` must return one representative for all three. If `A` is kept separately, union-find is not being called on the LSH pairs.
5. **Gopher filter thresholds.** Unit-test every published threshold against an example doc that lies just inside and just outside. E.g., a 49-word doc fails `gopher_len`; a 50-word doc passes. Thresholds that are ±1 off the Gopher paper are almost always typos; catch them here.
6. **Manifest idempotence.** Run the pipeline twice on the same input JSONL; assert `sha256(manifest_v1) == sha256(manifest_v2)`. This catches hidden non-determinism (dict iteration, random tie-breaks, dense_rank differences).
7. **Drop-rate dashboard regression.** On a 10k-doc sample from your domain, compute the per-filter drop-rate; publish it as a table. A new crawler release that changes any rate by > 2 absolute points is a flag.
8. **Fairness cross-tab.** Compute `(domain, filter_hit)` co-occurrence on the manifest. If a single domain/topic accounts for > 50 % of a filter's hits, the filter is domain-specific — decide if that is intended.
9. **"Known pair" integration test.** Hand-curate 20 known-duplicate pairs (same article on two mirrors) and 20 known-not-duplicate pairs (two articles on the same topic from the same mirror). Assert recall ≥ 95 % on the first, specificity ≥ 95 % on the second. This is the only test that catches tokenisation/regional-casing regressions that shingle-level tests miss.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Shingling** | Word n-gram at `n=5`, lowercased, documented | Char n-gram or different `n`, justified | No shingling / exact-dup only |
| **MinHash + LSH** | `datasketch` or equivalent, `num_perm ≥ 128`, Jaccard threshold 0.8, LSH used (not full pairwise) | Correct Jaccard but no LSH (full pairwise; works at small scale) | Hash collisions used as Jaccard proxy (wrong metric) |
| **Transitive clustering** | Union-find across LSH candidate pairs; deterministic representative | Pairwise dedup only (misses A-B-C chains) | No clustering |
| **Quality filters** | Gopher battery + C4 line-level, every filter reports hits individually | Gopher only or C4 only | Hand-written regex grab-bag without named thresholds |
| **Line-level dedup** | Paragraph dedup after boilerplate removal, or boilerplate line-filter | Document dedup only, acknowledged | Neither |
| **Manifest** | One row per *input* doc; status + filter_hits + cluster_id; deterministic byte order | Only kept docs logged | No manifest |
| **Drop-rate dashboard** | Per-filter drop-rate table reproduced on a sample; published next to corpus | Logs present, no table | Nothing logged |
| **Threat model** | OWASP ML Top 10 & MITRE ATLAS risks flagged (data poisoning via dedup-evasion content, naughty-word list fairness harm) | Mentioned, no mitigation | Ignored |
| **Documentation / card** | NIST AI RMF *Map* entries (snapshot URI, threshold, num_perm, filters, drop-rate); *Measure* entries (sample fairness cross-tab) | One of those | Neither |
| **Reproducibility** | Single-pass, deterministic order, sha256 of each input in manifest | Deterministic but no hashes | Manifest regenerates differently each run |

## 5. Common mistakes

- **Using LSH candidate pairs as final pairs without a Jaccard verification pass when the FP budget is strict.** LSH is a filter, not a decision. At `(threshold=0.8, num_perm=128)` the FP probability at true Jaccard 0.6 is ~3–5 %. For dataset release (where a false-drop is less costly than a false-keep of boilerplate) this is usually acceptable; for near-duplicate detection in a legal/plagiarism setting it is not — compute real Jaccard on candidate pairs and filter.
- **Running MinHash in `num_perm=64` "because the sketches are smaller".** The estimator standard deviation at 64 perms is ~0.125 at Jaccard 0.8 — enough to misclassify a quarter of near-dup pairs. Memory per sketch is `num_perm × 8 bytes`; 128 perms is 1 KB per doc, trivial for < 100 M docs.
- **Word shingles on CJK / no-space languages.** `text.split()` is a no-op on Chinese/Japanese text; every doc becomes one shingle. For CJK use char n-grams (`n=5` char-grams is the standard RefinedWeb-multilingual choice) and gate the shingler on language-ID.
- **Mixing exact-dup (SHA-256) and MinHash near-dup clusters in the manifest.** They are different relationships; preserve both. A doc that is an exact dup of three others and a near-dup of twelve has a `sha_cluster_size=4` and `minhash_cluster_size=16`; these answer different questions (is this verbatim re-hosted? vs. is this a template page?).
- **Using the C4 naughty-word list unmodified.** Dodge et al. 2021 documented systematic over-filtering of LGBTQ and minority dialect content. Either substitute a classifier-based toxicity filter (the exercise-02 approach) or publish the fairness-impact table. Silent inclusion is now a flagged reviewer concern.
- **Not rerunning the pipeline on the test fixtures after a dependency bump.** `datasketch` has changed its default hashing (`xxhash` vs `farmhash`) between major versions; MinHash sketches are not comparable across versions. Pin the version in `requirements.txt` and include at least one golden-sketch test.
- **Reporting corpus size in GB after dedup without the before-number.** The dedup-rate on CommonCrawl-derived corpora is typically 20–40 %; publishing only "50 GB clean corpus" hides whether you dropped 50 GB or 500 GB. Publish (raw-GB, after-quality-GB, after-dedup-GB) as a strict chain.
- **Serial dedup across the *concatenation* of corpora instead of per-sub-corpus then global.** The Pile approach is: dedup within each component, then dedup across. Jamming everything into one LSH index up front works, but attributing the drop-rate becomes impossible — you can no longer answer "how much of Common Crawl survived once The Pile's deduplicates were removed?".
- **Dropping the manifest to save disk.** The manifest is 100–200 bytes per input doc (~100 MB for 1 M docs). It is the audit artefact; dataset releases without one are not reproducible. If size is a concern, Parquet-compress it (10×) and ship it alongside the corpus.
- **Setting the Jaccard threshold below 0.7 "to be safe".** At 0.5, `the cat sat on the mat` and `the cat sat on the hat` collapse; so does any two-articles-on-same-template pair. You lose signal faster than you gain dedup. Stick with 0.8 (RefinedWeb/SlimPajama/FineWeb) unless you have corpus-specific evidence.

## 6. References

Primary sources used in this solution:

- Broder, A. (1997). *On the Resemblance and Containment of Documents*. SEQUENCES. <https://www.cs.princeton.edu/courses/archive/spring13/cos598C/broder97resemblance.pdf> — the original MinHash paper; defines Jaccard-estimator properties used in §2.3 and §3.2.
- Indyk, P., Motwani, R. (1998). *Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality*. STOC. <https://dl.acm.org/doi/10.1145/276698.276876> — the LSH family used for the bucketing scheme in `MinHashLSH`.
- Leskovec, J., Rajaraman, A., Ullman, J. (2020). *Mining of Massive Datasets*, 3rd ed., Chapter 3. <http://www.mmds.org/> — the canonical pedagogical reference for MinHash + LSH band/row tuning used in §2.3.
- Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5 / C4). JMLR. <https://arxiv.org/abs/1910.10683> — the C4 corpus construction, line-level filters, and the naughty-word list referenced in §2.5.
- Rae, J. et al. (2021). *Scaling Language Models: Methods, Analysis & Insights from Training Gopher*. DeepMind. <https://arxiv.org/abs/2112.11446> — Appendix A2 "Text Quality Filtering" is the source of the Gopher heuristic thresholds reproduced in §2.5.
- Lee, K. et al. (2022). *Deduplicating Training Data Makes Language Models Better*. ACL. <https://arxiv.org/abs/2107.06499> — the empirical case for MinHash dedup; shows perplexity and memorisation effects used to motivate §1.
- Gao, L. et al. (2020). *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*. <https://arxiv.org/abs/2101.00027> — Appendix documents the SHA-256 + MinHash pipeline used by the sub-corpora.
- Penedo, G. et al. (2023). *The RefinedWeb Dataset for Falcon LLM* (Penedo et al.). <https://arxiv.org/abs/2306.01116> — the five-gram, `threshold=0.8`, `num_perm=128` recipe reproduced in §2.3; §3.1 is the clearest filter-order reference.
- Soboleva, D. et al. (2023). *SlimPajama: A 627B token cleaned and deduplicated version of RedPajama*. Cerebras. <https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama> — public reference implementation using `datasketch`; the one to mirror for out-of-core scaling.
- Penedo, G. et al. (2024). *FineWeb: decanting the web for the finest text data at scale*. HuggingFace. <https://huggingface.co/datasets/HuggingFaceFW/fineweb> — publishes per-filter drop-rates and ablation tables that justify the §3.7 dashboard.
- Dodge, J. et al. (2021). *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus*. EMNLP. <https://arxiv.org/abs/2104.08758> — the fairness analysis of C4's naughty-word list that motivates the manifest cross-tab in §3.8 and the common-mistake flag.
- `datasketch` — <https://ekzhu.com/datasketch/> — reference implementation for `MinHash`, `MinHashLSH`, and `MinHashLSHForest` used in §2.3.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML02 Data Poisoning* is the primary concern: an adversary can publish deliberately-near-duplicate SEO content that evades `threshold=0.8` MinHash and still floods the corpus; the mitigation is to track `cluster_size` per source domain and alert when one domain is responsible for an outsized number of near-duplicates. *ML05 Model Theft* is not directly applicable here; *ML07 Transfer Learning Attack* applies downstream once the corpus trains a model.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique *AML.T0020 Poison Training Data* maps to the dedup-evasion scenario; the manifest's `(domain, cluster_size)` table is the observable control.
- NIST AI RMF — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record the raw-snapshot URIs, filter thresholds (`num_perm`, `threshold`, Gopher values), and the manifest's per-filter drop rates alongside the final JSONL. *Measure*: publish the domain × filter_hit cross-tab (the Dodge et al. fairness check) as the standard release artefact. *Manage*: version the pipeline; a dependency bump that changes `datasketch` hashing invalidates MinHash cluster IDs and is a controlled change.
