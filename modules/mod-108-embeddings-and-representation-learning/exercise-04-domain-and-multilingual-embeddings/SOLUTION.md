# exercise-04 — Domain and Multilingual Embeddings

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-108-embeddings-and-representation-learning/exercises/exercise-04-domain-and-multilingual-embeddings.md`.
> The target learning outcomes implied by that filename: adapt a sentence-embedding model to **(a) a specialist domain** (legal, biomedical, code, finance) where general encoders under-perform, and **(b) a multilingual setting** where queries and documents may be in different languages. Use **domain-adaptive pre-training (DAPT)** or **unsupervised sentence adaptation (TSDAE / GPL)** when labelled pairs are scarce; pick a **multilingual encoder aligned by parallel data or knowledge distillation**; evaluate on **in-domain retrieval** (BEIR task that matches the domain, or a held-out proprietary slice) and **cross-lingual retrieval** (BUCC bitext mining, Tatoeba, or the multilingual MTEB subset).

## 1. Solution overview

A general-purpose English embedder (`all-mpnet-base-v2`, `e5-base-v2`) is strong on web and news; it is weaker on medical ("myocardial infarction" ≠ "heart attack" in the embedding space unless the model was trained to know), weaker on code ("`O(n log n)`" and "n log n"), and useless when queries are in Spanish and documents in Portuguese. The two problems share one failure mode and one fix each:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **General encoder on specialist domain** | The retrieval model "works" (returns results) but relevance is low because the embedding space does not separate domain concepts (symptom codes, drug names, legal citations, programming identifiers) | Domain-adaptive pre-training on a domain MLM corpus (Gururangan et al. 2020), or sentence-level adaptation with TSDAE (Wang et al. 2021) when only unlabelled domain text is available, or Generative Pseudo-Labelling (GPL, Wang et al. 2022) when a T5 query generator is available |
| **Monolingual encoder on multilingual retrieval** | Query in Spanish, document in English → near-zero similarity; the embedding spaces are disjoint per language | Use a *multilingual* encoder trained with cross-lingual alignment: LaBSE (Feng et al. 2020), multilingual E5 (Wang et al. 2024), BGE-m3 (Chen et al. 2024), or the knowledge-distillation approach of Reimers & Gurevych (2020) — "Make Monolingual Sentence Embeddings Multilingual" |
| **Reporting in-domain gain on the training distribution** | DAPT or TSDAE on legal text + evaluation on the same legal subset the adaptation sampled from ≈ training-on-test. The reported lift evaporates when the held-out split is held out properly | Split the domain corpus before any adaptation; keep the eval slice sealed. Report both the pre-adapt and post-adapt score on the *same* held-out slice |

The deliverable is two coupled harnesses:

**Part A — Domain adaptation.** Pick one of three paths, graded on which signal is available:

1. **No labelled pairs, only domain text** — TSDAE (denoising autoencoder loss on unlabelled domain sentences).
2. **No labelled pairs, generative budget available** — GPL (generate synthetic queries for domain passages with a `doc2query` T5; mine hard negatives with the base encoder; distil cross-encoder margins).
3. **Labelled domain pairs available** — fine-tune the exercise-01/02 bi-encoder on them directly, with MNR loss and in-batch negatives (small learning rate, weight-decay anchored to base).

**Part B — Multilingual.** Pick a multilingual base and evaluate cross-lingual retrieval:

1. **LaBSE** (Feng et al. 2020) — 109 languages, CC-BY-SA-4.0, strong at bitext mining; 470M params.
2. **multilingual-E5-base** (Wang et al. 2024) — 94 languages, MIT licence, strong at multilingual retrieval; 278M params.
3. **BGE-m3** (Chen et al. 2024) — 100+ languages, MIT licence; supports dense + sparse + multi-vector in one model; the current open SOTA for multilingual retrieval. BGE-m3's per-language retrieval numbers vary across model-card revisions, so verify the specific card revision you cite rather than quoting the headline MIRACL/MMTEB numbers from the paper.

Reference evaluations (what the paper comparisons use):

- **In-domain retrieval:** `BEIR-SciFact` (scientific), `BEIR-TRECCOVID` (medical/pandemic), `BEIR-FiQA` (finance), `CodeSearchNet` (code) — all have held-out query/corpus splits that were not used in general encoder pre-training.
- **Cross-lingual retrieval:** `MIRACL` (Zhang et al. 2023) is the standard multilingual retrieval benchmark (18 languages, Wikipedia-based); `Tatoeba` (bitext mining across 112 languages); `BUCC 2017/2018` (parallel-sentence mining in zh/de/fr/ru against en); the multilingual-MTEB (MMTEB) "Retrieval" and "BitextMining" task groups.

## 2. Worked implementation

### 2.1 Project layout

```
domain_multilingual_bake/
  domain/
    tsdae_train.py        # Transformers-based denoising AE fine-tune
    gpl_pipeline.py       # doc2query + hard-negative mining + MarginMSE
    dapt_mlm.py           # Gururangan-style MLM on raw domain text
    eval_in_domain.py     # BEIR / in-house in-domain retrieval eval
  multilingual/
    adapter.py            # multilingual encoder adapter; language codes
    distill_from_monolingual.py  # Reimers-Gurevych 2020 distillation
    eval_crosslingual.py  # Tatoeba + MIRACL + MMTEB-BitextMining
  tests/
    test_domain_split_leakage.py
    test_crosslingual_alignment.py
  requirements.txt
```

### 2.2 Domain path 1 — TSDAE (unsupervised, no pairs)

TSDAE (Transformer-based Sequential Denoising Auto-Encoder; Wang et al. 2021) learns sentence embeddings by (a) corrupting a sentence with random token deletion, (b) encoding the corrupted version with a `[CLS]`-pooled encoder, (c) decoding the original with a Transformer decoder initialised from the same base, (d) training end-to-end with cross-entropy. The encoder retains the sentence-embedding capability afterwards. It is the simplest unsupervised domain-adaptation recipe when all you have is a pile of in-domain text.

```python
# domain/tsdae_train.py
from pathlib import Path
from sentence_transformers import SentenceTransformer, LoggingHandler, models, datasets, losses
from torch.utils.data import DataLoader


def train_tsdae(
    base_model: str = "distilbert-base-uncased",
    corpus_path: str = "data/domain_sentences.txt",
    output_dir: str = "runs/tsdae-domain",
    batch_size: int = 8,
    epochs: int = 1,
    lr: float = 3e-5,
):
    # TSDAE in sentence-transformers requires CLS pooling on the encoder.
    word_embedding = models.Transformer(base_model)
    pool = models.Pooling(word_embedding.get_word_embedding_dimension(), pooling_mode="cls")
    model = SentenceTransformer(modules=[word_embedding, pool])

    with open(corpus_path) as f:
        train_sentences = [line.strip() for line in f if line.strip()]

    # DenoisingAutoEncoderDataset applies random word-deletion (noise_fn default).
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_loss = losses.DenoisingAutoEncoderLoss(
        model,
        decoder_name_or_path=base_model,
        tie_encoder_decoder=True,  # halves memory; TSDAE paper default
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        weight_decay=0,                # TSDAE paper uses 0 weight decay
        scheduler="constantlr",        # TSDAE paper uses a constant LR
        optimizer_params={"lr": lr},
        show_progress_bar=True,
        output_path=output_dir,
    )
```

Three points worth defending in review:

- **`pooling_mode="cls"` is non-default.** TSDAE learns a `[CLS]` sentence representation because the decoder reads only the CLS token's hidden state. Mean-pooling here would decouple the encoder's sentence representation from what the decoder sees. The paper is explicit about this.
- **`tie_encoder_decoder=True`.** Initialising the decoder from the encoder weights and tying them halves memory and — per TSDAE §4.3 — produces better sentence embeddings than an untied decoder of the same size. Only untie if you have evidence that it helps on your corpus.
- **Constant LR, no weight decay.** The paper's defaults. Using the exercise-01 cosine-schedule + weight-decay recipe is a plausible override but should be ablated; many TSDAE replications see it hurt.

### 2.3 Domain path 2 — GPL (unsupervised, generative)

GPL (Generative Pseudo-Labelling; Wang et al. 2022) is the stronger unsupervised recipe when you have GPU budget for query generation. The pipeline:

1. **Query generation.** Run a `doc2query` T5 (`BeIR/query-gen-msmarco-t5-base-v1`) over each domain passage; sample 3 queries per passage with top-k sampling.
2. **Hard-negative mining.** For each synthetic query, retrieve top-50 passages with a general-purpose bi-encoder (the exercise-01 checkpoint); the positive is the originating passage; the negatives are the rest, filtered for false negatives (exercise-02 §2.3 pattern).
3. **Margin scoring.** Score `(query, positive)` and `(query, negative)` pairs with a strong cross-encoder teacher (`cross-encoder/ms-marco-MiniLM-L-6-v2`); the teacher margin is `CE(q, pos) - CE(q, neg)`.
4. **Student fine-tune.** Train the bi-encoder with MarginMSELoss (exercise-02 §2.4) on the synthetic (query, pos, neg, margin) tuples.

GPL out-performs TSDAE on BEIR by ~3–5 nDCG@10 on the domains in the paper; the compute cost is dominated by query generation (~1 s per passage at batch=32 on an A100), so it is mostly feasible for corpora below 500K passages.

```python
# domain/gpl_pipeline.py (skeleton — full implementation is the sum of exercise-02's
# mining + margin-MSE loss + a doc2query generation step up front)
from transformers import T5Tokenizer, T5ForConditionalGeneration


QUERY_GEN_MODEL = "BeIR/query-gen-msmarco-t5-base-v1"


def generate_queries(passages: list[str], n_per_passage: int = 3) -> list[tuple[str, str]]:
    tok = T5Tokenizer.from_pretrained(QUERY_GEN_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(QUERY_GEN_MODEL)
    model.eval().cuda()

    pairs = []
    for passage in passages:
        inputs = tok(passage, return_tensors="pt", truncation=True, max_length=256).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            top_k=10,
            num_return_sequences=n_per_passage,
        )
        for out in outputs:
            q = tok.decode(out, skip_special_tokens=True).strip()
            if q:
                pairs.append((q, passage))
    return pairs
```

After this step, the training loop is exactly the MarginMSE pipeline from exercise-02; the only difference is the source of the (query, passage) positives (synthetic, not human-labelled).

### 2.4 Domain path 3 — labelled pairs

When domain-labelled pairs exist (HR ticket pairs, legal contract clauses with human similarity scores, medical question-retrieval pairs with physician review), the exercise-01 MNR recipe is the right answer, with two adjustments:

- **Small learning rate (1e-5) and strong weight decay (0.01) anchored on the base encoder.** You want to shift the embedding space, not rewrite it.
- **Mix with ~10 % of the base training data** (MS MARCO, NLI) to prevent catastrophic forgetting of general semantics. The TSDAE / GPL paths do this implicitly via the generic base model.

### 2.5 Multilingual encoder adapter

```python
# multilingual/adapter.py
import numpy as np
from sentence_transformers import SentenceTransformer


MODELS = {
    "labse": "sentence-transformers/LaBSE",
    "me5-base": "intfloat/multilingual-e5-base",
    "me5-large": "intfloat/multilingual-e5-large",
    "bge-m3": "BAAI/bge-m3",
}


class MultilingualEncoder:
    def __init__(self, model_key: str):
        self.model = SentenceTransformer(MODELS[model_key])
        self.model_key = model_key

    def encode(self, texts, is_query=False):
        # multilingual-E5 uses the same English E5 prefix convention.
        if self.model_key.startswith("me5"):
            prefix = "query: " if is_query else "passage: "
            texts = [f"{prefix}{t}" for t in texts]
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
```

Multilingual encoder selection cheatsheet:

| Encoder | Languages | Licence | Strongest at | Weakest at |
|---|---|---|---|---|
| **LaBSE** | 109 | CC-BY-SA-4.0 | Bitext mining (Tatoeba, BUCC) — purpose-built for it | Retrieval quality within English (worse than mE5-base on MTEB-Retrieval) |
| **multilingual-E5-base** | 94 | MIT | General-purpose multilingual retrieval; strong on MIRACL and MMTEB | Low-resource languages outside the 94; some African and Indic underperform |
| **multilingual-E5-large** | 94 | MIT | Top-tier retrieval quality; slower | Same coverage gap as mE5-base; 560M params |
| **BGE-m3** | 100+ | MIT | Combined dense + sparse + multi-vector retrieval; long-context (8192 tokens) | Harder to serve (three retrieval paths); the dense-only mode trails mE5-large on some MIRACL languages |

Pick by deployment constraint. For a cross-lingual search engine: BGE-m3 if long contexts matter, mE5-large otherwise. For bitext mining (finding parallel sentences in a document pair): LaBSE.

### 2.6 Multilingual distillation (alternative to pre-trained multilingual)

If you already have a strong *English* encoder (your exercise-02 checkpoint) and you want it to work in `N` other languages without replacing it, the Reimers-Gurevych (2020) knowledge-distillation recipe is the canonical path:

1. **Pair a parallel corpus** `(en_i, xx_i)` for each target language `xx` (OPUS, Tatoeba, FLORES-200).
2. **Teacher** = the English SentenceTransformer; `teacher_emb = encode(en_i)`.
3. **Student** = a *multilingual* base (XLM-R, or `distiluse-base-multilingual-cased-v2`); initialise with the student's own pretraining.
4. **Objective** = MSE between `student.encode(xx_i)` and `teacher.encode(en_i)`. Also align `student.encode(en_i)` to `teacher.encode(en_i)` (so English stays).

```python
# multilingual/distill_from_monolingual.py (skeleton)
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.datasets import ParallelSentencesDataset
from torch.utils.data import DataLoader


def distill(teacher_ckpt: str, student_base: str, parallel_files: list[str], out_dir: str):
    teacher = SentenceTransformer(teacher_ckpt)
    student = SentenceTransformer(student_base)  # must produce same dim as teacher

    train = ParallelSentencesDataset(student_model=student, teacher_model=teacher, batch_size=32)
    for f in parallel_files:
        train.load_data(f)  # TSV: "english TAB target-language"

    loader = DataLoader(train, shuffle=True, batch_size=32)
    loss_fn = losses.MSELoss(model=student)
    student.fit(train_objectives=[(loader, loss_fn)], epochs=5, output_path=out_dir)
```

The resulting student encoder places `(en_i, xx_i)` pairs close together in the embedding space (because both are trained to match the teacher's English vector). This is the technique behind `distiluse-base-multilingual-cased-v2` and `paraphrase-multilingual-MiniLM-L12-v2`.

### 2.7 Evaluation — in-domain and cross-lingual

```python
# domain/eval_in_domain.py
from mteb import MTEB


def eval_scifact(encoder):
    # BEIR-SciFact is in MTEB's Retrieval task list; exact split match with BEIR.
    MTEB(tasks=["SciFact"]).run(encoder, output_folder="results/scifact")
```

```python
# multilingual/eval_crosslingual.py
from mteb import MTEB


def eval_multilingual(encoder):
    # Tatoeba (bitext mining, 112 languages) and MIRACL (18-language retrieval)
    # are MMTEB tasks; mteb exposes them under the same interface.
    MTEB(tasks=["Tatoeba", "MIRACLRetrieval"]).run(encoder, output_folder="results/mling")
```

Three things to report in the results JSON:

- **Pre-adapt vs post-adapt score on the same held-out split.** For domain: the base encoder's SciFact nDCG@10 vs the TSDAE/GPL-adapted encoder's SciFact nDCG@10. For multilingual-by-distillation: the base student's Tatoeba F1 vs the distilled student's.
- **Per-language breakdown on multilingual retrieval.** The average over 18 MIRACL languages hides that some languages (Thai, Swahili, Yoruba) are 10–20 points below the average; report the full table.
- **Dataset licences.** MIRACL: Apache-2.0. Tatoeba: CC-BY-SA-2.0 FR. BEIR per-dataset. LaBSE's training data is proprietary (no fine-tune redistribution risk, but cite the paper). Record all of them.

## 3. Validation steps

1. **Domain-split leakage test.** Hash every sentence in the adaptation corpus and every sentence in the eval split; assert the intersection is empty. A single leaked sentence inflates eval numbers in ways that are hard to attribute.
2. **TSDAE sanity check.** Encode `["patient presents with chest pain"]` before and after TSDAE adaptation on medical text. The vectors should differ by cosine distance ≥ 0.3. If they are near-identical, TSDAE did not run (`.fit` returned without a single update — common when the dataloader is empty due to a path bug).
3. **GPL query quality spot check.** Sample 30 synthetic queries from `doc2query` on your domain; a human reviewer should rate ≥ 60 % as plausible search queries. If the rate is under 40 %, the T5 is off-domain; retrain `doc2query` on 10K in-domain `(query, passage)` seeds before running the full GPL pipeline.
4. **Multilingual round-trip test.** Encode `[("Hello world", "en"), ("Hola mundo", "es"), ("Bonjour le monde", "fr")]`; assert pairwise cosine similarities between the translations are ≥ 0.80 (LaBSE / mE5 / BGE-m3 land around 0.90–0.95 on this trivial case). Below 0.6 flags that the model is not aligned across languages — i.e., you loaded a monolingual encoder by mistake.
5. **Prefix regression on mE5.** multilingual-E5 uses the same `"query: " / "passage: "` prefixes as English E5; without them, MIRACL nDCG@10 drops ~4 points on average. Run with and without; if there is no difference, the prefix is not being applied.
6. **Cross-lingual bitext mining baseline.** On a 10K parallel sentence slice, compute pairwise cosine similarity between the English and target-language sides; the "argmax matches the aligned pair" rate (P@1) must be ≥ 0.90 for LaBSE and mE5-base on major European languages. Below 0.80 flags a configuration bug.
7. **Per-language table completeness.** For any MIRACL run, the output JSON must contain scores for all 18 languages. A missing language typically means a download error that silently skipped the task.
8. **Pre- vs post-adapt comparison on the identical corpus.** On BEIR-SciFact: `(base encoder, post-adapt encoder)` scored on the *same* query set; post-adapt should strictly improve, or the adaptation did not help on this domain and should not be shipped.
9. **Licence check.** For every dataset used in adaptation, record the licence; for every pre-trained multilingual encoder, record the licence. LaBSE is CC-BY-SA-4.0 (share-alike propagates); mE5 and BGE-m3 are MIT. If a derivative model is published, the licence terms transitively apply.
10. **Seed stability.** Three seeds; MIRACL nDCG@10 variance should be <0.5 per language; TSDAE adaptation variance on BEIR-SciFact typically <1.0 nDCG@10.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Domain path selected with justification** | TSDAE / GPL / supervised chosen based on signal available; justification on paper | Path picked, no justification | Path mismatched to available signal (supervised recipe on unlabelled corpus) |
| **Domain adaptation hygiene** | Held-out eval slice sealed before adaptation; leakage test; pre- vs post-adapt on same split | Split present, no leakage test | Adaptation + eval on the same slice |
| **Multilingual encoder selection** | Documented language coverage, licence, strength (bitext vs retrieval); matched to task | Encoder named without justification | Monolingual encoder used for multilingual task |
| **Cross-lingual eval** | Tatoeba / MIRACL / MMTEB-BitextMining run; per-language table | Average only | No cross-lingual eval |
| **Prefix / normalisation** | mE5 prefixes applied; normalisation asserted; parity test | Prefix applied, no test | Dropped prefixes; silent under-reporting |
| **Reproducibility** | Seed; adaptation data revision; `sentence-transformers` + `mteb` versions pinned | Seed only | None |
| **Licence tracking** | Per-dataset and per-model licences recorded; LaBSE share-alike noted if used | Licences for some artefacts | None |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks noted (poisoning via adaptation corpus; adversarial multilingual inputs that sidestep safety classifiers trained on English) with mitigations | Mentioned, no mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure: domain, languages, adaptation data, pre-/post-adapt deltas, per-language retrieval table, known failure modes (code-mixed text, low-resource scripts) | One of those | Neither |

## 5. Common mistakes

- **Evaluating domain adaptation on the adaptation corpus.** If your adaptation sampled from the full SciFact corpus and you evaluate on SciFact without a held-out split, the measured lift is a memorisation artefact. Split before adapting, and seal the held-out slice.
- **TSDAE with mean pooling instead of CLS.** The TSDAE paper's whole mechanism depends on the decoder reading the pooled sentence representation. Mean pooling and `[CLS]` pooling produce different sentence vectors; the decoder was trained against the latter. Switching pooling at inference destroys the alignment.
- **Running GPL without the false-negative filter.** The hard negatives for GPL are mined from *synthetic* queries, which are often *paraphrases* of the positive passage. Without exercise-02's FN filter, 20-40 % of GPL negatives are false. The paper applies the filter; replication without it under-performs the paper by 2-4 nDCG@10.
- **Over-fitting the student in multilingual distillation.** The teacher (English encoder) has a fixed embedding geometry; the student's MSE loss asymptotically copies it. Training for too many epochs degrades generalisation on languages with few parallel sentences. The Reimers-Gurevych paper reports best results at 5 epochs on ~5M parallel sentences.
- **Using LaBSE for English-only retrieval.** LaBSE is *optimised for bitext mining*. On MTEB-Retrieval (English), it under-performs mE5-base by 5-7 nDCG@10 and all-mpnet-base-v2 by 3-5. Pick the model for the task.
- **Comparing multilingual-E5 to BGE-m3 on English only.** Both are multilingual models; comparing them on an English benchmark is fair for English quality but hides the multilingual trade-off (BGE-m3 trades some English quality for longer context and multi-vector support).
- **Ignoring tokenizer character coverage for low-resource languages.** XLM-R and its descendants tokenise most languages into reasonable subwords; but for Burmese, Tigrinya, or less-represented scripts, the tokenizer may fall back to byte-level with 4-8× the subword length, which saturates `max_seq_length`. Check the subword count per language before concluding "the model is bad at Burmese".
- **Treating a `distiluse-base-multilingual-cased-v2` result as representative of the state of the art.** DistilUSE was strong in 2020; mE5-base and BGE-m3 surpass it on every multilingual retrieval benchmark by 2024 (per their published model cards). It is a reasonable baseline, not a reference target.
- **Adapting on paraphrases of the eval corpus.** A common source of leakage: the "unlabelled domain corpus" for TSDAE is scraped from the web; the eval corpus (BEIR-FiQA) is from StackExchange. If the scrape hit StackExchange, the adaptation has seen the eval set. Always check the source URLs of the adaptation data against the eval dataset's source.
- **Shipping a multilingual retrieval system without a per-language latency SLO.** Non-Latin-script tokenisation inflates sequence length and latency per query; the p99 for Thai or Chinese can be 1.5–2× the English p50. Report latency per language, not a single number.

## 6. References

Primary sources used in this solution:

- Reimers, N., Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP. <https://arxiv.org/abs/1908.10084> — the SBERT base library and the sentence-embedding architecture adapted here.
- Reimers, N., Gurevych, I. (2020). *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*. EMNLP. <https://arxiv.org/abs/2004.09813> — the teacher-student distillation recipe in §2.6.
- Feng, F. et al. (2020). *Language-agnostic BERT Sentence Embedding* (LaBSE). <https://arxiv.org/abs/2007.01852> — the LaBSE multilingual encoder, 109 languages.
- Wang, K. et al. (2021). *TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning*. Findings of EMNLP. <https://arxiv.org/abs/2104.06979> — TSDAE pre-training / adaptation recipe in §2.2.
- Wang, K. et al. (2022). *GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval*. NAACL. <https://arxiv.org/abs/2112.07577> — the Generative Pseudo-Labelling pipeline in §2.3.
- Gururangan, S. et al. (2020). *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks* (DAPT). ACL. <https://arxiv.org/abs/2004.10964> — foundational DAPT recipe when only domain text is available.
- Wang, L. et al. (2024). *Multilingual E5 Text Embeddings: A Technical Report*. <https://arxiv.org/abs/2402.05672> — multilingual E5 (94 languages), prefix convention.
- Chen, J. et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation* (BGE-m3). <https://arxiv.org/abs/2402.03216> — BGE-m3 model and its dense/sparse/multi-vector retrieval paths.
- Thakur, N. et al. (2021). *BEIR*. <https://arxiv.org/abs/2104.08663> — in-domain retrieval evaluation datasets (SciFact, TRECCOVID, FiQA, ...).
- Zhang, X. et al. (2023). *MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages*. TACL. <https://arxiv.org/abs/2210.09984> — the MIRACL cross-lingual retrieval benchmark used in §2.7.
- Artetxe, M., Schwenk, H. (2019). *Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond* (LASER). TACL. <https://arxiv.org/abs/1812.10464> — Tatoeba bitext-mining benchmark; LASER baseline for multilingual sentence embeddings.
- Zweigenbaum, P. et al. (2017). *Overview of the Second BUCC Shared Task: Spotting Parallel Sentences in Comparable Corpora*. WMT. <https://aclanthology.org/W17-2512/> — BUCC bitext-mining benchmark.
- Muennighoff, N. et al. (2023). *MTEB: Massive Text Embedding Benchmark*. EACL. <https://arxiv.org/abs/2210.07316> — the MTEB library used in §2.7; the in-domain SciFact run plugs into it.
- `sentence-transformers` TSDAE example — <https://www.sbert.net/examples/unsupervised_learning/TSDAE/README.html> — canonical implementation mirrored in §2.2.
- `sentence-transformers` multilingual distillation example — <https://www.sbert.net/examples/training/multilingual/README.html> — canonical implementation mirrored in §2.6.
- `BeIR/query-gen-msmarco-t5-base-v1` model card — <https://huggingface.co/BeIR/query-gen-msmarco-t5-base-v1> — the `doc2query` T5 used in §2.3.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML02 Data Poisoning*: scraped domain corpora (web-scraped medical / legal / finance text) are a known poisoning surface; vet the source. *ML01 Input Manipulation*: multilingual retrieval systems are routinely attacked by phrasing a query in a less-safety-filtered language to reach a target document; align safety classifiers across languages.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for adversarial ML; adversarial examples crossing language boundaries are a documented pattern for sidestepping English-trained guardrails.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record domain (legal / medical / code / finance), languages supported, adaptation corpus source and licence, multilingual-encoder choice and its coverage gaps (which languages are *not* supported), and the pre- vs post-adaptation delta. *Measure*: in-domain retrieval metric (nDCG@10 on sealed held-out) and cross-lingual retrieval per language (MIRACL table; Tatoeba P@1); latency per language for production systems.
