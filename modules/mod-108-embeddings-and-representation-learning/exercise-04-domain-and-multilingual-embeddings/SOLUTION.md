# exercise-04 — Domain and Multilingual Embeddings

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-108-embeddings-and-representation-learning/exercises/exercise-04-domain-and-multilingual-embeddings.md`.
> Target learning outcomes implied by that filename: adapt a general-purpose embedding model to a **specific
> domain** (clinical, legal, financial, code) **or** a **specific language mix**, pick an appropriate
> multilingual base, evaluate **cross-lingually and cross-domain** without fooling yourself, and document
> the resulting coverage honestly.

## 1. Solution overview

A general-purpose embedding model trained on web-scale `(query, passage)` pairs (E5, BGE, all-MiniLM) is
strong on English Wikipedia / news / Stack Exchange style text and much weaker on everything else —
clinical notes, legal filings, SEC 10-Ks, scientific abstracts, codebases, non-English text. Domain /
multilingual adaptation is the discipline of picking the right base, training on the right pairs, and
measuring on the right set.

Three failure modes recur:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **English-only base for a multilingual target** | Fine-tuning `all-MiniLM-L6-v2` on Spanish pairs produces a model that handles Spanish queries slightly better than before but fails any cross-lingual task — because the base vocabulary and encoder were never exposed to non-English | Start from a **multilingual base**: LaBSE (Feng et al. 2022) for strong sentence-level, `paraphrase-multilingual-mpnet-base-v2` for general multilingual STS, `intfloat/multilingual-e5-base` for retrieval, or `BAAI/bge-m3` for retrieval + sparse + multi-vector |
| **Evaluating only in the training language** | A multilingual model trained on English→Spanish pairs may be perfect at English→Spanish and terrible at Spanish→Spanish, Spanish→English, or any third language. Headline numbers that use only the training-language direction are misleading | Evaluate on **cross-lingual retrieval** sets (Mr. TyDi, MIRACL, XOR-TyDi, MLDR) and on **same-language retrieval in each target language** separately; report both diagonals and off-diagonals |
| **Overfitting the domain to the point of forgetting** | Clinical fine-tuning on a small MIMIC dataset for too many epochs collapses the model's general-domain performance; on the next business case the team reaches back for `all-MiniLM-L6-v2` and loses the clinical lift | Use **continual-pretraining-then-contrastive** (MLM on domain corpus, *then* contrastive on pairs), low LR, few epochs, and **evaluate on BOTH a domain benchmark and a general benchmark** so the regression is visible |

The deliverable is an adapted-embedding harness with six components:

1. **Base selection rationale** — written justification for the multilingual / domain base, with the alternative bases considered and why they were rejected.
2. **Domain / parallel corpus** — labelled or weakly-labelled `(anchor, positive)` pairs. For multilingual: parallel sentences from CCMatrix / NLLB-mined-bitext / FLORES / Tatoeba / OPUS. For clinical: MIMIC-III / IV discharge summary pairs, PubMedQA title-abstract. For legal: EU statutes with paragraph-paragraph reformulations, Edgar 10-K similar-chunk pairs. For code: CodeSearchNet docstring-function pairs.
3. **Training recipe** — usually MNRL / InfoNCE as in exercise-01, with careful de-duplication and language balancing. For code embeddings, a two-stage recipe (MLM on code + contrastive on docstring-function) is standard.
4. **Multilingual-specific tweaks** — language-weighted sampling to avoid dominance by high-resource languages; optional language-code prefixes (model-dependent).
5. **Cross-lingual evaluator** — diagonal (same-language retrieval) + off-diagonal (cross-language retrieval) matrix on a held-out multilingual benchmark.
6. **Documentation card** — NIST AI RMF Map: languages and domains covered; dataset licences; known biases (resource imbalance, dialect coverage); scoring convention.

Reference bases by target:

| Target | Recommended base | Why |
|---|---|---|
| General multilingual retrieval | `intfloat/multilingual-e5-base` or `BAAI/bge-m3` | Both trained on broad multilingual pair sets; both ship published MIRACL / Mr. TyDi scores. BGE-M3 adds sparse + ColBERT heads. |
| Cross-lingual STS / sentence similarity | `sentence-transformers/LaBSE` (Feng et al. 2022) or `paraphrase-multilingual-mpnet-base-v2` | LaBSE trained on 109 languages, dedicated cross-lingual objective (translation-ranking). |
| Low-resource language retrieval | `intfloat/multilingual-e5-large` or `BAAI/bge-m3` | Both include the FLORES-200 language set. For languages not in either, start from XLM-R base and train. |
| Clinical English | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` (Gu et al. 2020) via SBERT fine-tune; or `S-BioBert` / `BioLORD` | PubMedBERT's vocabulary covers medical terms the general BERT WordPiece splits into fragments. |
| Legal English | `nlpaueb/legal-bert-base-uncased` (Chalkidis et al. 2020) | Pretrained on EU + US law; general BERT tokenisers fragment Latin legal terms. |
| Scientific | `allenai/scibert_scivocab_uncased` (Beltagy et al. 2019) or `allenai/specter2` (Singh et al. 2022) | SPECTER2 is a paper-embedding model trained on citation pairs; drop-in for scholarly retrieval. |
| Code / NL↔code | `microsoft/codebert-base` (Feng et al. 2020) or `Salesforce/codet5p-110m-embedding` | Both pretrained on code+docstring pairs; trained on CodeSearchNet. |
| Clinical multilingual | `intfloat/multilingual-e5-large-instruct` + continual pretraining | No off-the-shelf exists; the common path is a general multilingual base + a small MIMIC/NLLB-mined clinical pair set. |

## 2. Worked implementation

### 2.1 Project layout

```
multilingual_bake/
  base.py              # base-selection justification + loader
  data.py              # parallel pair loader with language balancing
  train.py             # MNRL fine-tune with per-language sampling
  evaluate.py          # diagonal + off-diagonal cross-lingual eval
  tests/
    test_balancing.py
    test_cross_lingual_sign.py
  requirements.txt
```

### 2.2 Language-balanced parallel pair loader

```python
# data.py
import random
from collections import defaultdict
from datasets import load_dataset
from sentence_transformers import InputExample


# FLORES-200 ISO codes (NLLB convention). Expand to the project's language set.
LANGS = ["eng_Latn", "spa_Latn", "fra_Latn", "deu_Latn", "zho_Hans", "arb_Arab", "hin_Deva", "swh_Latn"]


def load_flores_pairs(target_langs: list[str], split: str = "dev") -> dict[str, list[tuple[str, str]]]:
    """
    FLORES-200 devtest covers 2009/1012 sentences per language on the same source
    (Wikipedia articles), so sentence index i in any language is a translation
    of sentence index i in any other. We pivot around English to produce
    (eng, L) pairs for each L in target_langs.
    """
    ds = {lang: load_dataset("facebook/flores", lang, split=split)
          for lang in set(target_langs + ["eng_Latn"])}
    pairs = defaultdict(list)
    eng = ds["eng_Latn"]
    for lang in target_langs:
        if lang == "eng_Latn":
            continue
        other = ds[lang]
        for e, o in zip(eng, other):
            pairs[lang].append((e["sentence"], o["sentence"]))
    return pairs


def balance_pairs(
    pairs_by_lang: dict[str, list[tuple[str, str]]],
    policy: str = "sqrt",
    alpha: float = 0.5,
    seed: int = 20260101,
    epoch_size: int | None = None,
) -> list[InputExample]:
    """
    policy='proportional': sample each language with weight proportional to its
                           size — lets high-resource languages dominate.
    policy='sqrt':         weight by `count^alpha` with alpha=0.5 — the NLLB
                           convention; downweights high-resource languages
                           without zeroing them.
    policy='uniform':      every language gets the same number of pairs —
                           simplest, strongest low-resource signal, wastes
                           high-resource data.

    Weights are converted into **sampling probabilities**, not absolute counts.
    `epoch_size` bounds the total number of `InputExample`s produced per epoch
    (default: `sum(sizes.values())`). This prevents the pathological case
    where a language with ~1k FLORES pairs is expanded to millions of examples
    simply because `count^alpha * 100_000` happens to be a large constant.
    """
    rng = random.Random(seed)
    sizes = {lang: len(v) for lang, v in pairs_by_lang.items()}
    if epoch_size is None:
        epoch_size = sum(sizes.values())

    if policy == "uniform":
        target = min(sizes.values())
        return [InputExample(texts=[a, b]) for lang in sizes
                for (a, b) in rng.sample(pairs_by_lang[lang], target)]
    if policy == "proportional":
        raw = {lang: float(size) for lang, size in sizes.items()}
    else:  # sqrt
        raw = {lang: float(size) ** alpha for lang, size in sizes.items()}

    total_w = sum(raw.values())
    counts = {lang: max(1, int(round(w / total_w * epoch_size))) for lang, w in raw.items()}

    out = []
    for lang, w in counts.items():
        subset = pairs_by_lang[lang]
        if len(subset) >= w:
            picks = rng.sample(subset, w)
        else:
            # Repeat only when the requested probability exceeds available pairs.
            # This is explicit oversampling, not an accidental 1000x blow-up.
            picks = (subset * ((w // len(subset)) + 1))[:w]
        out.extend(InputExample(texts=[a, b]) for a, b in picks)
    rng.shuffle(out)
    return out
```

Points worth defending in review:

- **`sqrt` / `α=0.5` sampling is the NLLB and XLM-R default.** Pure proportional sampling trains a model that is essentially monolingual English because the parallel data is dominated by English. Uniform sampling wastes most of the available data on the saturated high-resource side.
- **Shuffle after assembly, not before.** If you shuffle each language then interleave, in-batch negatives of exercise-01's MNRL loss happen *across* languages, which is what you want — a Spanish query's positive is then contrasted against French, German, Chinese, and Arabic negatives, teaching cross-lingual separation.
- **`NoDuplicatesDataLoader`, not a plain shuffled `DataLoader`.** FLORES pivots each English source across all target languages, so the same English anchor appears in multiple `(eng, L)` pairs. A plain `DataLoader` would allow two such pairs into one MNRL batch, where each translation becomes a negative for an identical anchor — a sign-flipped signal. `NoDuplicatesDataLoader` guarantees each unique text appears at most once per batch.
- **FLORES-200 is for *eval*, not for training.** Training on FLORES then reporting FLORES is contamination. For training, use CCMatrix, NLLB-mined-bitext, OPUS-100, or Tatoeba; reserve FLORES devtest for test numbers.

### 2.3 Fine-tuning from a multilingual base

```python
# train.py
import argparse
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.datasets import NoDuplicatesDataLoader
from .data import load_flores_pairs, balance_pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="intfloat/multilingual-e5-base")
    p.add_argument("--target-langs", nargs="+", default=["eng_Latn", "spa_Latn", "fra_Latn", "zho_Hans"])
    p.add_argument("--output-dir", default="runs/mling-adapted")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--policy", choices=["proportional", "sqrt", "uniform"], default="sqrt")
    p.add_argument("--seed", type=int, default=20260101)
    a = p.parse_args()

    # NB: for multilingual E5, keep its `query: ` / `passage: ` convention.
    # For retrieval pairs: use the prefix. For generic bitext (FLORES), either
    # both prefixed or both bare — never mixed.
    model = SentenceTransformer(a.base_model)

    # Weak-supervision multilingual training uses CCMatrix / NLLB-mined-bitext
    # in practice; FLORES devtest is here for clarity but should be held out for eval.
    pairs_by_lang = load_flores_pairs(a.target_langs, split="dev")
    examples = balance_pairs(pairs_by_lang, policy=a.policy, seed=a.seed)
    # `NoDuplicatesDataLoader` dedupes texts within a batch: because FLORES pivots
    # around English, the same English anchor is reused as `(eng, spa)`,
    # `(eng, fra)`, `(eng, zho)` positives. A plain shuffled `DataLoader` would
    # let two such pairs land in the same batch, where MNRL would treat each
    # translation as a negative for the other identical anchor — a sign-flipped
    # training signal. `NoDuplicatesDataLoader` ensures each unique text appears
    # at most once per batch.
    loader = NoDuplicatesDataLoader(examples, batch_size=a.batch_size)
    loss = losses.MultipleNegativesRankingLoss(model=model, scale=20.0)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=a.epochs,
        warmup_steps=min(1000, len(loader) // 10),
        optimizer_params={"lr": a.lr},
        output_path=a.output_dir,
        use_amp=True,
    )
```

Key choices:

- **Multilingual base is non-negotiable.** Starting from `all-MiniLM-L6-v2` or `all-mpnet-base-v2` and feeding it Chinese or Arabic pairs produces a model whose Chinese / Arabic vocabulary is mostly byte-fallback; no amount of fine-tuning recovers the information that is lost at the tokeniser layer.
- **LR 2e-5.** Higher LRs (5e-4) catastrophically forget the pretraining; lower (5e-6) barely moves the model. 2e-5 with warmup is the sentence-transformers default and a stable choice.
- **Batch size 128 if hardware allows.** Multilingual MNRL benefits even more than monolingual from large in-batch negative counts because the cross-lingual contrast only works when the batch contains multiple languages.
- **One epoch, maybe two.** Multilingual-E5 is already contrastively trained; a long fine-tune overwrites that signal.

### 2.4 Domain variant — same loss, different base and corpus

For clinical / legal / scientific / code domains the training scaffold is the same but:

- Base: domain-specific (PubMedBERT, LegalBERT, SciBERT / SPECTER2, CodeBERT / CodeT5+).
- Corpus: domain-specific pair source — PubMedQA (title-abstract), MedQA, Clinical-STS; EU Court case-paragraph pairs; CodeSearchNet `(docstring, function)`.
- Max length: longer (384 / 512) — clinical and legal paragraphs run long.
- Pre-contrastive MLM stage: if the domain gap is large, run MLM on the domain corpus for 1–3 epochs before the contrastive fine-tune; this is **continual pretraining** (Gururangan et al. 2020, *Don't Stop Pretraining*).

### 2.5 Cross-lingual evaluation — diagonal + off-diagonal

```python
# evaluate.py
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


def retrieval_scores(model, queries, corpus, qrels, top_k: int = 10) -> dict:
    """
    Return nDCG@10 and MRR@10 on one (query_lang, corpus_lang) pair.
    Each of `queries`, `corpus` is a dict id->text; `qrels` maps qid -> set[pid].
    """
    import heapq
    q_ids, q_texts = zip(*queries.items())
    c_ids, c_texts = zip(*corpus.items())
    q_emb = model.encode(list(q_texts), convert_to_numpy=True, normalize_embeddings=True)
    c_emb = model.encode(list(c_texts), convert_to_numpy=True, normalize_embeddings=True)
    sims = q_emb @ c_emb.T

    ndcgs, mrrs = [], []
    for i, qid in enumerate(q_ids):
        gold = qrels.get(qid, set())
        if not gold:
            continue
        topk_idx = np.argsort(-sims[i])[:top_k]
        rels = [1 if c_ids[j] in gold else 0 for j in topk_idx]
        # nDCG / MRR.
        dcg = sum(r / np.log2(k + 2) for k, r in enumerate(rels))
        idcg = sum(1 / np.log2(k + 2) for k in range(min(len(gold), top_k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        mrrs.append(next((1 / (k + 1) for k, r in enumerate(rels) if r), 0.0))
    return {"ndcg@10": float(np.mean(ndcgs)), "mrr@10": float(np.mean(mrrs)), "n": len(ndcgs)}


def cross_lingual_matrix(model_dir: str, langs: list[str], tasks_by_lang: dict) -> dict:
    """
    tasks_by_lang[(q_lang, c_lang)] -> (queries, corpus, qrels)
    Produces the full |langs| x |langs| retrieval matrix. Diagonal = same-lang.
    """
    model = SentenceTransformer(model_dir)
    matrix = {}
    for q_lang in langs:
        for c_lang in langs:
            key = (q_lang, c_lang)
            if key not in tasks_by_lang:
                continue
            queries, corpus, qrels = tasks_by_lang[key]
            matrix[f"{q_lang}->{c_lang}"] = retrieval_scores(model, queries, corpus, qrels)
    return matrix
```

Recommended cross-lingual benchmarks:

- **MIRACL** (Zhang et al. 2023) — 18-language same-language retrieval; standard benchmark for multilingual dense retrieval. Hugging Face: `miracl/miracl`.
- **Mr. TyDi** (Zhang et al. 2021) — 11-language same-language retrieval, Wikipedia-based; simpler scope than MIRACL.
- **MKQA** (Longpre et al. 2021) — 26-language parallel questions aligned to English Wikipedia answers; strong cross-lingual test.
- **XOR-TyDi** (Asai et al. 2021) — cross-lingual open-retrieval where the query is in language L and the answer passage is in English.
- **MLDR** (`BAAI/MLDR`) — long-document multilingual retrieval; used by BGE-M3 for benchmarking.
- **Multilingual MTEB** — `mteb.get_benchmark("MTEB(Multilingual, v1)")` — the superset.

For a domain variant, swap in: **BEIR SciFact / NFCorpus / TREC-COVID** (scientific), **MedQA / PubMedQA** (clinical), **CaseHOLD / LegalBench** (legal), **CodeSearchNet** (code).

### 2.6 Report shape

Minimum publishable card for a domain / multilingual adaptation:

- Languages or domain covered; data provenance and licences.
- Base model, revision, pooling, prefix convention.
- Training corpus sizes per language / sub-domain, sampling policy (`sqrt(α=0.5)`).
- Diagonal retrieval scores (same-lang / in-domain).
- Off-diagonal retrieval scores (cross-lang / cross-domain).
- Comparison against base model on both diagonal and off-diagonal — must show the adaptation improved what it targeted *without regressing the untargeted axes* (or disclose the regression).
- A "known blind spots" paragraph: dialects not represented, scripts partially covered, low-resource languages below some threshold.

## 3. Validation steps

1. **Base-selection justification.** The run card contains one paragraph explaining why the chosen base was picked over the next-best alternative (e.g., "chose multilingual-E5 over LaBSE because the primary use case is retrieval, where E5 holds a ~5-point MIRACL lead; LaBSE remains stronger on STS/BUCC-style bitext mining but that is not our use case").
2. **Tokeniser coverage check.** For every target language, encode 1000 sampled sentences and measure the mean subword fragments per word. A value > 3× English's suggests the tokeniser is byte-falling-back and the model will struggle; swap bases if so.
3. **Language-balance smoke test.** On a batch sampled from your `balance_pairs()` output, compute the per-language frequency; it must match the policy's prediction (±2 %). If English is 60 % with `policy=sqrt`, the sqrt is on the wrong axis.
4. **Cross-lingual retrieval sign test.** On a 100-query MIRACL subset, run the model you trained and the base. The diagonal (same-lang) numbers must come up; the off-diagonal (cross-lang) numbers must not come *down* by more than noise. If cross-lingual dropped, you overfit the diagonal.
5. **Prefix-forgetting test.** If you fine-tuned from multilingual-E5, re-run the FAST_SUBSET of MTEB with the `query:` / `passage:` prefixes. Scores must track the base within ±1 point on general-domain tasks; a drop > 2 points means the fine-tune has partially forgotten the prefix convention.
6. **Catastrophic-forgetting check.** For a domain adaptation, run a general-domain retrieval set (BEIR NFCorpus at minimum) before and after fine-tuning. A regression > 5 points on general-domain is a sign of too many epochs or too high an LR; add continual pretraining with a smaller LR or reduce epochs.
7. **Dataset-split hygiene for parallel data.** If training pairs come from CCMatrix / OPUS-100, ensure that the FLORES-200 sentences you intend to use for eval are not present (literal-string overlap check). CCMatrix contains Wikipedia text; FLORES devtest is Wikipedia-derived; the collision rate must be checked, not assumed.
8. **Low-resource floor.** For each target language, report the number of training pairs and the resulting diagonal nDCG@10. Below ~50k pairs and/or diagonal nDCG@10 < 0.2, state explicitly in the card that the language is **experimental coverage**, not production-ready.
9. **Reproducibility replay.** With `sentence-transformers` version, torch version, and HF cache snapshot pinned, re-run training on a tiny fraction and verify loss curves and final-checkpoint numbers to within rounding noise.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Base model match to target** | Multilingual base for multilingual target / domain base for domain target; justification written | Base chosen without justification | English-only base for multilingual target; general base for heavily specialised domain |
| **Language balancing** | `sqrt` / `α` policy with the balance verified on a sampled batch | Uniform or proportional with justification | High-resource language > 50 % of sampled batch with no acknowledgment |
| **Prefix / instruction policy** | Base's convention preserved end-to-end; prefix-forgetting unit-tested | Prefix applied inconsistently | Prefix dropped or added arbitrarily per batch |
| **Cross-lingual eval** | Diagonal **and** off-diagonal reported on MIRACL / Mr. TyDi / MKQA | Diagonal only | Training-language retrieval only |
| **Regression disclosure** | Pre- and post-fine-tune numbers on at least one general-domain benchmark | Post-only | No general-domain eval |
| **Low-resource caveats** | Per-language pair count + diagonal number + "experimental" tag for thin languages | Flat language list | No per-language reporting |
| **Dataset contamination** | FLORES / MIRACL eval splits audited against training corpus | Assumed clean | Trained on FLORES devtest |
| **Reproducibility** | Seed, revision, policy, batch size, library versions all recorded | Some of those | None |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks called out (poisoning of parallel data, membership inference on private domain corpora, embedding inversion on PHI / legal content) with mitigations | Mentioned without mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure: language / domain coverage, licences, sampling policy, diagonal + off-diagonal, known blind spots | Partial | Missing |

## 5. Common mistakes

- **Fine-tuning an English base on Chinese pairs.** The tokeniser cannot represent Chinese cleanly, so fine-tuning distorts vectors but cannot rebuild lost information. Start from a multilingual base (XLM-R family, mBERT, LaBSE, multilingual-E5, BGE-M3).
- **Treating "multilingual" as a single axis.** Mandarin, Hindi, Swahili, and Spanish have radically different scripts, morphologies, and resource levels. "Our model works in 50 languages" is almost always false — some of those 50 land at nDCG@10 < 0.1. Report per-language.
- **Reporting only the training-direction retrieval.** English→Spanish on a model trained on English-Spanish pairs is cheap; Spanish→English and Spanish→Spanish are the honest tests.
- **FLORES-200 contamination.** FLORES devtest is Wikipedia-derived, and CCMatrix / NLLB-mined-bitext include Wikipedia; naive concatenation of training data leaks FLORES into training. Filter by hash.
- **Catastrophic forgetting on domain fine-tunes.** Clinical adaptation on 50k pairs for 10 epochs can tank general-domain nDCG@10 by 10+ points. Early stopping on a *mixed* eval set (domain + general) catches this.
- **Mixing prefix and bare encodings.** Fine-tuning multilingual-E5 with prefix on queries and bare on passages (or vice versa) is a subtle bug that degrades both retrieval and STS. Document the prefix policy and unit-test it.
- **Tokenizer truncation on long legal / clinical paragraphs.** Default `max_length=128` silently truncates most clinical paragraphs mid-sentence. Set `max_length=512` and verify your GPU memory with a batch size reduced to match.
- **Embedding inversion on sensitive domains.** Publishing embeddings of clinical notes or legal filings is partial text disclosure; Song & Raghunathan (2020) and later work show substantial recovery is possible. For protected data, keep embeddings inside the trust boundary of the raw text.
- **Assuming BGE-M3's three outputs are interchangeable.** BGE-M3 ships dense, sparse, and ColBERT-style multi-vector heads; each needs its own scoring function and index. Mixing them at query time requires a documented fusion (the model card ships recommended weights).
- **Using `pooler_output` on a multilingual base.** Same bug as exercise-01, but costlier here: `bert-base-multilingual`'s pooler was NSP-tuned and useless for sentence embedding. Always mean-pool.
- **Comparing against the wrong leaderboard.** MIRACL and Mr. TyDi are *same-language* benchmarks; using them to validate a "cross-lingual" claim is category-confused. Use XOR-TyDi / MKQA for cross-lingual.
- **Training domain pairs that are near-duplicates.** Clinical discharge summaries have strong template structure ("CHIEF COMPLAINT:", "HISTORY OF PRESENT ILLNESS:"); without near-duplicate filtering, MNRL contrasts template strings against themselves. Use MinHash / SimHash near-duplicate filtering on the pair text before training.

## 6. References

Primary sources used in this solution:

- Reimers, N., Gurevych, I. (2020). *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*. EMNLP. <https://arxiv.org/abs/2004.09813> — the distillation recipe behind `paraphrase-multilingual-mpnet-base-v2` referenced in §1.
- Feng, F. et al. (2022). *Language-agnostic BERT Sentence Embedding* (LaBSE). ACL. <https://arxiv.org/abs/2007.01852> — the 109-language bi-encoder cited as a cross-lingual STS/bitext default.
- Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-R). ACL. <https://arxiv.org/abs/1911.02116> — the multilingual encoder underpinning the multilingual-E5 and BGE-M3 families.
- NLLB Team (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation*. <https://arxiv.org/abs/2207.04672> — FLORES-200 eval corpus, NLLB-mined-bitext training corpus, and the `sqrt(α=0.5)` sampling policy used in §2.2.
- Schwenk, H. et al. (2021). *CCMatrix: Mining Billions of High-Quality Parallel Sentences on the Web*. ACL. <https://arxiv.org/abs/1911.04944> — large-scale mined bitext source for multilingual contrastive training.
- Wang, L. et al. (2024). *Multilingual E5 Text Embeddings: A Technical Report*. <https://arxiv.org/abs/2402.05672> — multilingual-E5 architecture, prefix policy, and reported MIRACL / Mr. TyDi numbers.
- Chen, J. et al. (2024). *BGE M3-Embedding*. <https://arxiv.org/abs/2402.03216> — dense + sparse + multi-vector multilingual retrieval model and the MLDR benchmark.
- Zhang, X. et al. (2023). *MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages*. TACL. <https://arxiv.org/abs/2210.09984> — the cross-lingual retrieval benchmark used in §2.5 and §3.
- Zhang, X. et al. (2021). *Mr. TyDi: A Multi-lingual Benchmark for Dense Retrieval*. MRL Workshop. <https://arxiv.org/abs/2108.08787> — same-language 11-language retrieval.
- Asai, A. et al. (2021). *XOR QA: Cross-lingual Open-Retrieval Question Answering*. NAACL. <https://arxiv.org/abs/2010.11856> — cross-lingual open-retrieval eval.
- Longpre, S. et al. (2021). *MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering*. TACL. <https://arxiv.org/abs/2007.15207> — 26-language parallel-question benchmark.
- Gu, Y. et al. (2020). *Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing* (PubMedBERT). ACM Transactions on Computing for Healthcare. <https://arxiv.org/abs/2007.15779> — the biomedical base cited in the recommended-bases table.
- Chalkidis, I. et al. (2020). *LEGAL-BERT: The Muppets straight out of Law School*. EMNLP Findings. <https://arxiv.org/abs/2010.02559> — the legal domain base.
- Beltagy, I., Lo, K., Cohan, A. (2019). *SciBERT: A Pretrained Language Model for Scientific Text*. EMNLP. <https://arxiv.org/abs/1903.10676> — scientific-text base.
- Singh, A. et al. (2022). *SciRepEval & SPECTER2: Scientific Document Embedding at Scale*. <https://arxiv.org/abs/2211.13308> — citation-trained document embedding model.
- Feng, Z. et al. (2020). *CodeBERT: A Pre-Trained Model for Programming and Natural Languages*. EMNLP Findings. <https://arxiv.org/abs/2002.08155> — code/NL embedding base.
- Husain, H. et al. (2019). *CodeSearchNet Challenge: Evaluating the State of Semantic Code Search*. <https://arxiv.org/abs/1909.09436> — the docstring-function pair corpus used for code embeddings.
- Gururangan, S. et al. (2020). *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks*. ACL. <https://arxiv.org/abs/2004.10964> — the continual pretraining pattern cited in §2.4.
- Song, C., Raghunathan, A. (2020). *Information Leakage in Embedding Models*. CCS. <https://arxiv.org/abs/2004.00053> — embedding-inversion risk cited in §5 and the threat model.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML02 Data Poisoning*: mined bitext (CCMatrix, NLLB) is web-sourced and susceptible to adversarial pair injection. *ML06 Model Inversion*: embedding inversion on clinical / legal / financial content is partial disclosure of PHI / privileged material; treat embedding indices as classified at the highest level of the underlying documents. *ML08 Transfer Learning Attack*: adapting from a poisoned base propagates the base's backdoor into the fine-tune.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for adversarial ML; relevant whenever multilingual embeddings drive translation-mediated search, moderation, or compliance.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: language / domain coverage, licences, sampling policy, prefix convention; *Measure*: diagonal + off-diagonal cross-lingual retrieval, pre/post-fine-tune general-domain regression, per-language low-resource floor; *Govern*: domain / multilingual coverage is a stated-limitation axis on every model card produced by this pipeline, not a marketing axis.
