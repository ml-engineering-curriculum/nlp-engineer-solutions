# exercise-03 — Contamination Detection and Decontamination

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-111-nlp-evaluation/exercises/exercise-03-contamination-detection-and-decontamination.md`.
> The target learning outcomes implied by that filename: build the pipeline that answers "did my
> training data contain the test set?" before every release — **n-gram overlap detection** at the
> input and input+output level, **MinHash-based fuzzy match** for paraphrased leakage, **membership-
> inference probes** as a complementary signal, a **decontamination pass** that removes the
> contaminated training examples (not the test items), and a **contamination report** that is
> produced for every model card and blocks release when the overlap exceeds a declared threshold.

## 1. Solution overview

A metric number on a contaminated test set measures memorisation, not generalisation. The job is to
detect three progressively weaker signals of contamination and to decontaminate from the training
side so the evaluation remains interpretable:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Verbatim test items in the training corpus** | GPT-3 ablations (Brown et al. 2020) and PaLM analysis (Chowdhery et al. 2023) both found non-trivial overlap between public test sets and the training corpora. Metric values on contaminated items are unreliable | **Exact n-gram overlap**: for each test item, flag if any 13-gram (or 50-character) span appears verbatim in the training corpus. This is the GPT-3 recipe. Use a 13-gram hash index of the training data |
| **Paraphrased or lightly-edited test items** | Benchmarks get paraphrased, re-hosted, or translated-and-back before they enter the training corpus. 13-gram exact match misses these | **MinHash fuzzy overlap**: shingle both corpora, build a MinHash LSH on the training set, query every test item. Jaccard threshold 0.7–0.8 catches moderate paraphrase |
| **Latent / semantic leakage** | An item may not match verbatim or by shingle, but the model has been trained on the answer. The strongest tell is an abnormally-low loss on the test item relative to a held-out distribution | **Membership-inference probe** (Carlini et al. 2022; Shi et al. 2023 Min-K% Prob): rank items by their LM likelihood; the test items with unusually low loss are flagged. This is a *weak* signal but catches cases the first two miss |
| **Decontaminating the test set** | A standard-but-wrong fix: drop test items that overlap the training corpus. This destroys cross-paper comparability | **Decontaminate the training corpus, not the test set.** Remove the overlapping training documents; re-train or at minimum re-run eval with the leaked data masked. Document the removed count in the model card |
| **One-shot contamination check** | Running the check once at the end of training catches contamination but not "how much did it affect the reported metric". The fair comparison is dirty-vs-clean on the same model | Report metrics **both** on the full test set and on the **decontaminated subset** (test items not matched to any training item). The difference is the contamination effect size |

The deliverable is a four-component pipeline:

1. **Exact n-gram index** — a hash set of every 13-gram (or 50-char span) in the training data; test-item lookup is O(items × ngrams-per-item).
2. **MinHash + LSH fuzzy index** — exercise-01-style shingling + MinHash, queried with every test item as the probe; Jaccard threshold 0.7–0.8.
3. **Membership-inference probe** — compute per-item LM loss (or the Min-K% Prob of Shi et al. 2023); rank against a baseline held-out distribution; flag the lowest-loss tail.
4. **Decontamination report + CLI** — one JSONL row per test item recording (verbatim-hit, fuzzy-hit, MI-flag, matched-training-doc-ids); a release-gate check that blocks when any item hits the "contaminated" threshold *and* the reported metric differs meaningfully between contaminated and decontaminated subsets.

Published precedent and standards:

- **GPT-3 contamination analysis** (Brown et al. 2020, Appendix C) — the 13-gram recipe, with per-benchmark dirty-vs-clean metric tables.
- **PaLM contamination analysis** (Chowdhery et al. 2023, §B.1) — extended the recipe to 15+ public benchmarks; introduced the "high / medium / low overlap" tri-categorisation per item.
- **C4 / The Pile audits** — Dodge et al. 2021 and the SlimPajama audit in Soboleva et al. 2023 both run test-set-in-corpus checks; the Dodge et al. appendix is the canonical "how to run it on CommonCrawl" reference.
- **Magar & Schwartz 2022** (*Data Contamination: From Memorization to Exploitation*) — shows the effect-size argument for the dirty-vs-clean comparison; contaminated test items can raise metrics by 5–20 points on some benchmarks.
- **Carlini et al. 2022** (*Quantifying Memorization Across Neural Language Models*) — the extraction / membership-inference methodology.
- **Shi et al. 2023** (*Detecting Pretraining Data from Large Language Models*) — the Min-K% Prob membership-inference probe used in §2.4.
- **Oren et al. 2024** (*Proving Test Set Contamination in Black-Box Language Models*) — the exchangeability-based probe that catches contamination without white-box loss access.
- **Sainz et al. 2023** (*NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each Benchmark*) — community survey and best-practice recommendations.

## 2. Worked implementation

### 2.1 Project layout

```
decontam/
  ngram_index.py          # 13-gram hash index over training corpus
  fuzzy_index.py          # MinHash + LSH fuzzy index
  membership.py           # LM loss + Min-K% Prob
  report.py               # one JSONL row per test item
  decontaminate.py        # remove overlapping training docs
  cli.py                  # end-to-end CLI
  tests/
    test_ngram_shingles.py
    test_lsh_recall.py
    test_min_k_prob.py
    test_report_schema.py
  requirements.txt
```

### 2.2 Exact n-gram overlap — the GPT-3 13-gram recipe

```python
# ngram_index.py
from typing import Iterator


def whitespace_tokens(text: str) -> list[str]:
    return text.lower().split()


def word_ngrams(tokens: list[str], n: int) -> Iterator[tuple[str, ...]]:
    if len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


class NgramIndex:
    """
    Hash set of training-corpus n-grams. Lookup: for each test item, split into
    n-grams and count hits. Follows Brown et al. 2020 Appendix C: n=13, whitespace
    tokens, lowercased. Memory: ~40 bytes per n-gram; for 100M-token corpus of
    English prose, ~2-3 GB in RAM.
    """

    def __init__(self, n: int = 13):
        self.n = n
        self.grams: set[tuple[str, ...]] = set()
        self.ngram_to_docs: dict[tuple[str, ...], set[str]] = {}

    def add_document(self, doc_id: str, text: str) -> None:
        for g in word_ngrams(whitespace_tokens(text), self.n):
            self.grams.add(g)
            self.ngram_to_docs.setdefault(g, set()).add(doc_id)

    def check(self, text: str) -> dict:
        """Returns {n_overlap_grams, total_grams, overlap_fraction, matched_docs}."""
        total = 0
        hits = 0
        matched: set[str] = set()
        for g in word_ngrams(whitespace_tokens(text), self.n):
            total += 1
            if g in self.grams:
                hits += 1
                matched.update(self.ngram_to_docs.get(g, ()))
        return {
            "n_overlap_grams": hits,
            "total_grams": total,
            "overlap_fraction": (hits / total) if total else 0.0,
            "matched_training_docs": sorted(matched),
        }
```

Points worth defending:

- **n = 13 is the Brown et al. 2020 choice.** A single 13-gram collision is extremely unlikely by
  chance for English prose (entropy ~1 nat/word ⇒ ~2^18 unique 13-grams per typical document); shorter
  n-grams produce too many false positives on common phrases. Shorter n (8–10) is acceptable for
  code or structured text where the vocabulary is smaller.
- **Whitespace tokens, lowercased, no stemming.** Same portability argument as in mod-110
  exercise-01: a tokeniser-specific pipeline breaks when the tokeniser changes. The GPT-3 appendix
  uses this exact tokeniser.
- **Memory is real.** On a 1-T-token corpus the n-gram set is ~10^11 entries (hundreds of GB). The
  production pattern is a Bloom filter (Brown et al. 2020 used one) or a sharded Rocks-backed set.
  For courseware scale (~100 M tokens), an in-memory set is fine; document the scaling limit.
- **Record `matched_training_docs` per hit.** Without it the report answers "there is overlap" but
  not "where". The decontamination pass needs the doc IDs; the fairness audit needs the training
  sources that cause the leak.

### 2.3 Fuzzy overlap — MinHash + LSH probe

```python
# fuzzy_index.py
from datasketch import MinHash, MinHashLSH


NUM_PERM = 128
JACCARD_THRESHOLD = 0.70   # looser than mod-110 dedup (0.8) because we are catching paraphrase


def word_shingles(text: str, n: int = 5) -> list[str]:
    toks = text.lower().split()
    if len(toks) < n:
        return [" ".join(toks)] if toks else []
    return [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]


def sketch(text: str, num_perm: int = NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in word_shingles(text):
        m.update(s.encode("utf-8"))
    return m


class FuzzyIndex:
    def __init__(self, threshold: float = JACCARD_THRESHOLD, num_perm: int = NUM_PERM):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm

    def add_document(self, doc_id: str, text: str) -> None:
        self.lsh.insert(doc_id, sketch(text, self.num_perm))

    def query(self, text: str) -> list[str]:
        return self.lsh.query(sketch(text, self.num_perm))
```

Points worth defending:

- **Threshold 0.70, not 0.80.** The mod-110 dedup threshold is for *identifying near-duplicates
  within a corpus* and tolerates a small false-positive rate. Contamination detection has the
  opposite trade-off: a false-negative (missed contamination) is worse than a false-positive
  (flagged a legitimate similarity). 0.70 catches paraphrase; the §3 validation set covers how to
  pick the exact value per benchmark.
- **Word 5-grams, lowercased.** The RefinedWeb / SlimPajama convention; empirical justification in
  mod-110 exercise-01 §2.2.
- **MinHash probe is a filter, not a decision.** Candidate pairs from LSH must be verified with
  real Jaccard; keep this a separate step in the pipeline. For contamination reporting the LSH
  candidate list is usually sufficient (false positives at threshold 0.7 are < 5 %), but strict
  settings should verify.
- **Both the training corpus and the test set index.** For small test sets you may prefer to build
  the LSH on the test set and probe with training documents (fewer queries, fewer index inserts);
  the result is symmetric.

### 2.4 Membership-inference probe — Min-K% Prob

```python
# membership.py
import math
from typing import Callable, Sequence

LogProbFn = Callable[[str], list[float]]
# Returns per-token log-probabilities under the model for the given string.


def min_k_prob(token_logprobs: Sequence[float], k_fraction: float = 0.2) -> float:
    """
    Shi et al. 2023, "Detecting Pretraining Data from Large Language Models".
    Returns the mean of the k% smallest token log-probabilities. Items seen in
    pretraining tend to have higher (less negative) mean-log-prob on their rarest
    tokens than held-out items.
    """
    k = max(1, int(len(token_logprobs) * k_fraction))
    sorted_lp = sorted(token_logprobs)
    bottom_k = sorted_lp[:k]
    return sum(bottom_k) / k


def membership_scores(
    items: Sequence[str],
    logprob_fn: LogProbFn,
    k_fraction: float = 0.2,
) -> list[float]:
    return [min_k_prob(logprob_fn(x), k_fraction=k_fraction) for x in items]


def flag_low_loss_tail(
    scores: Sequence[float],
    held_out_scores: Sequence[float],
    q: float = 0.95,
) -> list[bool]:
    """
    Flag test items whose Min-K% Prob is higher than the q-quantile of a held-out
    distribution. "Higher" (less negative) means "less surprising" => potentially seen.
    """
    cutoff = sorted(held_out_scores)[int(q * len(held_out_scores))]
    return [s >= cutoff for s in scores]
```

Points worth defending:

- **Min-K% Prob outperforms simple perplexity.** Shi et al. 2023 show that the lowest-k% tokens
  carry most of the memorisation signal; a trained model memorises rare phrasings but is fluent on
  common ones. Using whole-sequence loss washes out the signal.
- **Calibrate against a held-out baseline.** The absolute score is uninterpretable; what matters is
  "lower than the q-quantile of items known not to be in training". Build the held-out set from
  post-training data (e.g., news articles published after the training cutoff) or from a dataset
  you control.
- **Membership inference is a *weak* signal.** Carlini et al. 2022 show the false-positive rate is
  non-trivial for items that are highly fluent (news headlines, idioms). Use it as a tie-breaker
  and audit trigger, not as the primary flag.
- **Black-box setting is harder.** Oren et al. 2024 describe an exchangeability test that works
  without white-box loss access (submit the test set in two orderings and compare). If the model is
  API-only, implement that; for open-weight models, Min-K% Prob is directly available.

### 2.5 Contamination report — one row per test item

```python
# report.py
import json
from pathlib import Path
from typing import Any


def write_report(rows: list[dict[str, Any]], path: Path) -> None:
    """
    One row per test item:
      {
        "test_id": "...",
        "verbatim": {"n_overlap_grams": 2, "total_grams": 18, "overlap_fraction": 0.11,
                     "matched_training_docs": ["doc_12", "doc_93"]},
        "fuzzy":    {"matched_training_docs": ["doc_17"]},
        "membership": {"min_k_prob": -2.41, "flagged": True},
        "category": "HIGH" | "MEDIUM" | "LOW" | "CLEAN"
      }
    """
    with path.open("w") as f:
        for r in sorted(rows, key=lambda r: r["test_id"]):
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def categorise(row: dict[str, Any]) -> str:
    """
    Chowdhery 2023 PaLM-style tri-categorisation with an added CLEAN bucket.
      HIGH    - any verbatim 13-gram match OR fuzzy Jaccard >= 0.9
      MEDIUM  - fuzzy Jaccard in [0.7, 0.9) OR membership flagged with any secondary signal
      LOW     - only membership flagged, no verbatim and no fuzzy
      CLEAN   - no signal
    """
    v = row["verbatim"]["n_overlap_grams"] > 0
    f = bool(row["fuzzy"]["matched_training_docs"])
    m = row["membership"]["flagged"]
    if v or f:
        return "HIGH"
    if m:
        return "LOW"
    return "CLEAN"
```

Points worth defending:

- **`HIGH` / `MEDIUM` / `LOW` / `CLEAN` is the PaLM convention.** Chowdhery et al. 2023 Appendix B.1
  reports dirty-vs-clean metric deltas per category; reviewers expect the same categorisation. Keep
  the thresholds configurable in a pinned config.
- **Verbatim overrides fuzzy.** A verbatim 13-gram match is `HIGH` even if the fuzzy score is low.
  (A fuzzy score can be low because a single sentence match in a long document dilutes Jaccard; the
  verbatim hit is still a leak.)
- **Report the matched training docs.** The audit value of the report is "which training document
  contaminated which test item"; without it a reviewer cannot trace the leak back to the ingest step.
- **Sort rows by `test_id`.** Deterministic byte order so reruns on the same data produce
  bit-identical reports; this matches the mod-110 manifest convention.

### 2.6 Decontamination — remove the training docs, not the test items

```python
# decontaminate.py
import json
from pathlib import Path


def collect_contaminating_training_docs(report_path: Path) -> set[str]:
    docs: set[str] = set()
    with report_path.open() as f:
        for line in f:
            r = json.loads(line)
            docs.update(r["verbatim"]["matched_training_docs"])
            docs.update(r["fuzzy"]["matched_training_docs"])
    return docs


def filter_training_corpus(
    input_corpus: Path, output_corpus: Path, to_remove: set[str]
) -> int:
    removed = 0
    with input_corpus.open() as inp, output_corpus.open("w") as out:
        for line in inp:
            doc = json.loads(line)
            if doc["doc_id"] in to_remove:
                removed += 1
                continue
            out.write(line)
    return removed
```

Points worth defending:

- **Decontaminate the training corpus, not the test set.** Removing test items destroys cross-paper
  comparability and is a one-way street. Removing training items preserves the test as a reference
  and quantifies the leak as "N training documents were removed".
- **Re-train (or at least re-evaluate) after decontamination.** The decontaminated training corpus
  is the one the released model should have been trained on; if full retraining is not feasible,
  the fallback is to evaluate only on the decontaminated-subset-of-test (test items with no
  `HIGH`/`MEDIUM` category) and publish the dirty-vs-clean delta.
- **Publish the `removed` count.** A model card that says "decontaminated" without a number is
  unverifiable; the reviewer needs `(training docs before, after, removed)` to assess the pipeline
  ran.

### 2.7 CLI — end-to-end

```python
# cli.py
import argparse, json
from pathlib import Path
from .ngram_index import NgramIndex
from .fuzzy_index import FuzzyIndex
from .membership import membership_scores, flag_low_loss_tail
from .report import write_report, categorise
from .decontaminate import collect_contaminating_training_docs, filter_training_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-corpus", type=Path, required=True, help="JSONL {doc_id, text}")
    ap.add_argument("--test-set", type=Path, required=True, help="JSONL {test_id, text}")
    ap.add_argument("--report-out", type=Path, required=True)
    ap.add_argument("--decontam-corpus-out", type=Path, default=None,
                    help="If set, write the decontaminated training corpus here")
    ap.add_argument("--held-out-scores", type=Path, default=None,
                    help="JSON list of membership scores for held-out (non-training) items")
    ap.add_argument("--logprob-cache", type=Path, default=None,
                    help="JSON dict test_id -> list of token logprobs under the model")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--fuzzy-threshold", type=float, default=0.70)
    args = ap.parse_args()

    # Build indices
    ngram = NgramIndex(n=args.n)
    fuzzy = FuzzyIndex(threshold=args.fuzzy_threshold)
    with args.training_corpus.open() as f:
        for line in f:
            d = json.loads(line)
            ngram.add_document(d["doc_id"], d["text"])
            fuzzy.add_document(d["doc_id"], d["text"])

    # Score every test item
    rows: list[dict] = []
    tests = [json.loads(l) for l in args.test_set.open()]
    held_out = json.loads(args.held_out_scores.read_text()) if args.held_out_scores else []
    cache = json.loads(args.logprob_cache.read_text()) if args.logprob_cache else {}
    for t in tests:
        tid = t["test_id"]
        v = ngram.check(t["text"])
        f_matches = fuzzy.query(t["text"])
        # Membership probe: only if a logprob cache is supplied
        m = {"min_k_prob": None, "flagged": False}
        if cache and tid in cache:
            from .membership import min_k_prob
            mk = min_k_prob(cache[tid], k_fraction=0.2)
            flagged = any(flag_low_loss_tail([mk], held_out)) if held_out else False
            m = {"min_k_prob": mk, "flagged": flagged}
        row = {
            "test_id": tid,
            "verbatim": v,
            "fuzzy": {"matched_training_docs": f_matches},
            "membership": m,
        }
        row["category"] = categorise(row)
        rows.append(row)
    write_report(rows, args.report_out)

    # Optional decontamination pass
    if args.decontam_corpus_out:
        to_remove = collect_contaminating_training_docs(args.report_out)
        removed = filter_training_corpus(args.training_corpus, args.decontam_corpus_out, to_remove)
        print(f"removed {removed} training documents implicated in test contamination")


if __name__ == "__main__":
    main()
```

Design notes:

- **Membership probe is opt-in.** It requires access to per-token log-probs under the trained model;
  that may not be available (closed API, pre-training black-box). The CLI accepts a `--logprob-cache`
  path so the probe can be run offline after model inference.
- **Report is written before decontamination.** Two independent artefacts: the report answers "is
  there contamination" and ships with the model card; the decontaminated corpus is the input to the
  next training round.
- **Held-out score file is a first-class input.** Don't bake the baseline into the code; it is
  benchmark- and model-dependent. The release gate records the held-out URI and sample size.

## 3. Validation steps

1. **Round-trip a known-leaked pair.** Insert a test item verbatim into the training corpus; the
   report must show `verbatim.n_overlap_grams == total_grams` and `category == "HIGH"` for that
   item, and list the matching training doc. Catches whitespace-normalisation or encoding regressions.
2. **Paraphrase detection on controlled edit distance.** Build 100 pairs with Jaccard 0.6, 0.7, 0.8,
   0.9 (via word deletions / substitutions); verify the fuzzy index flags at ≥ 95 % recall at
   Jaccard ≥ 0.75 and ≤ 5 % FP at Jaccard ≤ 0.5. Picks the operational threshold.
3. **Negative control.** Run the pipeline with training corpus = `[]` or with a training corpus
   disjoint from the test set; `category` must be `CLEAN` for every test item. Catches bugs where
   hash collisions produce spurious matches.
4. **GPT-3 benchmark reproduction.** Build a synthetic 1 M-token training corpus plus a 1 000-item
   test set with known 5 % overlap; run the pipeline; the report's `HIGH` count must match 50 ± 2
   (deterministic bootstrap resample — see exercise-02 for CI methodology).
5. **Dirty-vs-clean metric comparison.** Score a model on the full test set and on the
   decontaminated subset; the published metric must be the decontaminated-subset number *and* the
   delta between them. A delta > 2 absolute points is a flag; > 5 is a release block.
6. **Min-K% Prob calibration.** On a model with known training data, Min-K% Prob on in-training
   items must be *higher* (less negative) than on held-out items on average; a histogram overlay
   is the canonical visualisation. If the signal reverses, the implementation is wrong.
7. **Decontamination round-trip.** Run the pipeline; collect the removed training docs; re-run the
   pipeline against the decontaminated corpus; every previously-`HIGH` item must now be `CLEAN` or
   `LOW` (verbatim and fuzzy clear; membership may still trip).
8. **Deterministic report.** Two runs with the same inputs produce bit-identical JSONL. Catches
   `datasketch` version drift (sketches are not comparable across versions), dict iteration order,
   set-element-order non-determinism.
9. **Scaling smoke test.** Run on 10 M-token training corpus and 10 k-item test set; wall-clock and
   peak memory within declared budgets. Verifies the pipeline is usable, not just correct.
10. **Published-benchmark check.** Run the pipeline on the training corpus against a published
    public benchmark (SQuAD dev, MMLU, HumanEval); publish the `HIGH` count and matched-doc
    examples in the model card. Reviewers expect at least a non-trivial count on CommonCrawl-
    derived corpora.

## 4. Grading rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Exact overlap** | 13-gram index (Brown 2020 recipe); per-item overlap fraction + matched training docs | n-gram check with different `n`, justified | None or substring regex |
| **Fuzzy overlap** | MinHash + LSH at Jaccard 0.7; per-item matched training docs | Exact overlap only; paraphrase caveat noted | None |
| **Membership probe** | Min-K% Prob (Shi 2023) or Oren 2024 exchangeability; calibrated against held-out | Perplexity-only probe | None |
| **Tri-categorisation** | HIGH / MEDIUM / LOW / CLEAN (PaLM convention) | Binary flag only | None |
| **Decontamination action** | Training docs removed; count published; re-evaluation on decontaminated subset | Flag only, no action | Test set modified instead |
| **Dirty-vs-clean metric delta** | Metric reported on (full, decontaminated-subset); delta published | One number only | None |
| **Report schema** | Per-test-item row with verbatim + fuzzy + membership + category + matched doc IDs; deterministic order | Aggregate only | None |
| **Release gate** | Threshold (e.g., delta ≤ 1 absolute point OR HIGH rate ≤ 0.5 %) blocks release when exceeded | Threshold mentioned, not enforced | None |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS mapped (test-set poisoning, benchmark gaming) | Mentioned | Ignored |
| **Documentation / card** | NIST AI RMF *Map* (benchmarks checked, thresholds), *Measure* (HIGH/MEDIUM/LOW counts, dirty-vs-clean), *Manage* (removed training docs + re-eval record) | One of those | Neither |

## 5. Common mistakes

- **Removing contaminated items from the *test* set instead of the training corpus.** Shrinks the
  test, destroys cross-paper comparability, and leaves the model trained on the leak. The correct
  fix is to remove the training documents and (ideally) retrain, or at minimum publish metrics on
  the decontaminated subset.
- **Running the contamination check only at release time.** By then the model is trained; the
  decontamination choices are "remove from eval" (bad) or "retrain" (expensive). Run the check
  during corpus construction (mod-110 pipeline), not at release.
- **Using `n = 8` or lower for n-gram overlap.** 8-grams of English prose collide on "the quick
  brown fox" style idioms; the false-positive rate becomes punitive. Stick with `n = 13`
  (Brown 2020); go lower only for code or structured text with a smaller vocabulary.
- **Treating the Bloom filter false-positive rate as negligible.** At 10^11 n-grams, a 1 % FP rate
  flags 10^9 spurious hits. Record the FP rate in the model card alongside the raw counts.
- **No held-out baseline for Min-K% Prob.** The absolute score is meaningless; you need a
  distribution of Min-K% Prob on items the model has *not* seen. Build it from post-training
  content (dated after the training cutoff) or from a dataset you fully control.
- **Single-threshold fuzzy probe.** 0.70 is a sensible default; benchmarks with heavy paraphrase
  risk (back-translation, question rephrasing) justify 0.60; strict code-benchmark settings may
  want 0.80. Sweep the threshold on a controlled set (§3.2); publish the operating point.
- **Not recording the matched training documents.** A report that says "50 items contaminated"
  without identifying the training source is not actionable. The audit value depends on
  `test_id → list[training_doc_id]`.
- **Treating the Min-K% Prob flag as high-confidence.** False positives on idiomatic text are
  non-trivial; use it as a tie-breaker and audit trigger, not a standalone release block.
- **Only checking publicly-known benchmarks.** Private evaluation sets must also be checked;
  internal corporate benchmarks have a non-zero leak rate when ingested from shared repos.
- **Re-building the LSH index per release without pinning `datasketch`.** MinHash sketches depend
  on the library's hashing scheme (farmhash vs. xxhash). Pin the dependency version in
  `requirements.txt` and golden-test a sketch.
- **Ignoring the "latent" contamination case (Oren 2024).** A model may have been trained on a
  paraphrased or translated version of a test set; only an exchangeability-based probe catches
  this. For closed-API models this is the only available tool.
- **Not publishing the pre- and post-decontamination training-corpus sizes.** Reviewers need
  `(before, after, removed)` to verify the decontamination actually ran; "decontaminated" without
  a number is unverifiable.
- **Running the pipeline but not setting a release gate on the output.** If the report is
  cosmetic — produced but never blocks — the pipeline does nothing. The gate is the artefact;
  the report is its input.
- **Reporting contamination at the corpus level only.** "0.3 % of the test set is contaminated"
  hides which benchmark is affected. Report per-benchmark; a 0.3 % corpus rate can be 100 % on
  a small benchmark that happens to be fully in the corpus.

## 6. References

Primary sources used in this solution:

- Brown, T. et al. (2020). *Language Models are Few-Shot Learners* (GPT-3). NeurIPS. <https://arxiv.org/abs/2005.14165> — Appendix C is the 13-gram overlap methodology reproduced in §2.2; the dirty-vs-clean metric comparison in §3.5 follows the same template.
- Chowdhery, A. et al. (2023). *PaLM: Scaling Language Modeling with Pathways*. JMLR. <https://arxiv.org/abs/2204.02311> — Appendix B.1 introduces the HIGH / MEDIUM / LOW categorisation used in §2.5.
- Magar, I., Schwartz, R. (2022). *Data Contamination: From Memorization to Exploitation*. ACL. <https://aclanthology.org/2022.acl-short.18/> — the empirical effect size of contamination on reported metrics; motivates §3.5.
- Carlini, N. et al. (2022). *Quantifying Memorization Across Neural Language Models*. ICLR. <https://arxiv.org/abs/2202.07646> — the membership-inference / memorisation framework that §2.4 operationalises.
- Shi, W. et al. (2023). *Detecting Pretraining Data from Large Language Models*. ICLR. <https://arxiv.org/abs/2310.16789> — Min-K% Prob reproduced in §2.4.
- Oren, Y. et al. (2024). *Proving Test Set Contamination in Black-Box Language Models*. ICLR. <https://arxiv.org/abs/2310.17623> — the exchangeability-based black-box probe referenced in §5.
- Sainz, O. et al. (2023). *NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each Benchmark*. Findings of EMNLP. <https://aclanthology.org/2023.findings-emnlp.722/> — the community-recommendations survey motivating §4's rubric structure.
- Dodge, J. et al. (2021). *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus* (C4 audit). EMNLP. <https://arxiv.org/abs/2104.08758> — audit methodology; reports benchmark overlap in CommonCrawl-derived corpora.
- Soboleva, D. et al. (2023). *SlimPajama: A 627B token cleaned and deduplicated version of RedPajama*. Cerebras. <https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama> — the SlimPajama audit section; practical reference for scaling the fuzzy index.
- Gao, L. et al. (2020). *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*. <https://arxiv.org/abs/2101.00027> — includes a benchmark-overlap audit in the appendix.
- Dong, Y. et al. (2024). *Generalization or Memorization? Data Contamination and Trustworthy Evaluation for Large Language Models*. arXiv. <https://arxiv.org/abs/2402.15938> — recent survey of contamination-detection methods; cited for the Min-K / Min-K++ comparison and emerging black-box probes.
- Broder, A. (1997). *On the Resemblance and Containment of Documents*. SEQUENCES. — MinHash foundation; §2.3.
- Indyk, P., Motwani, R. (1998). *Approximate Nearest Neighbors*. STOC. — LSH foundation; §2.3.
- Zhang, Y., Yang, D., Lapata, M. et al. (2024) and follow-ups on contamination of code benchmarks — see Riddell et al. (2024) *Quantifying Contamination in Evaluating Code Generation Capabilities of Language Models*, ACL. <https://aclanthology.org/2024.acl-long.761/> — code-specific contamination case; relevant to §2.2's note about tighter n for code.
- `datasketch` — <https://ekzhu.com/datasketch/> — MinHash + MinHashLSH reference implementation used in §2.3.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML02 Data Poisoning* has a direct corollary: an adversary can publish paraphrased copies of a target benchmark to inflate the trained model's benchmark score. Detection controls: the fuzzy probe in §2.3 and the dirty-vs-clean delta in §3.5. *ML06 AI Supply Chain Attacks*: if a third-party dataset (e.g., a public crawl) is tampered to include benchmark leakage, the exercise-03 pipeline is the detection control on ingest. *ML09 Output Integrity Attack*: inflated benchmark numbers mislead downstream deployment decisions; the release gate in §4 is the mitigation.
- MITRE ATLAS — <https://atlas.mitre.org/> — *AML.T0020 Poison Training Data* maps to deliberate benchmark leakage via corpus injection; the n-gram + fuzzy indices are the detection controls. *AML.T0018 Backdoor ML Model* is tangentially relevant: a backdoor trigger phrase, if present in a test set, is captured by the same verbatim probe. *AML.T0043 Craft Adversarial Data* applies to benchmark gaming via paraphrase; the §2.3 fuzzy probe and §2.4 membership probe are the layered defences.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record every benchmark checked, the thresholds (`n`, `fuzzy_threshold`, `min_k_fraction`), and the held-out URI. *Measure*: publish HIGH/MEDIUM/LOW/CLEAN counts per benchmark, per-category metric deltas, and the pre- / post-decontamination training-corpus sizes. *Manage*: the release gate (§4 and §2.7's `--decontam-corpus-out`) is the change-controlled artefact; retraining vs. decontaminated-subset evaluation is a documented policy choice. *Govern*: contamination checks are part of the pre-release quality gate alongside fairness and robustness; a model card without a contamination report is non-release-ready.
