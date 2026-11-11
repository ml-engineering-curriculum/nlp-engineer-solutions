# exercise-04 — Domain-Adapted Tokenizer Training

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-101-tokenization-and-text-foundations/exercises/exercise-04-domain-adapted-tokenizer-training.md`.
> This solution targets the learning goals that filename implies: train (or extend) a subword tokeniser so it compresses domain text more efficiently than a general-purpose off-the-shelf tokeniser, and quantify the gain.

## 1. Solution overview

A tokeniser trained on general web text (GPT-2 byte BPE, BERT WordPiece, etc.) spends many tokens per word on domain-specific vocabulary — medical terms, legal boilerplate, SQL keywords, protein sequences. Domain adaptation gets those terms into the vocabulary as single pieces, which translates directly into shorter contexts, cheaper inference, and often better downstream model performance when the embedding matrix is retrained.

There are two legitimate strategies, and the exercise is to pick one deliberately and justify it:

| Strategy | What you do | Pros | Cons |
|---|---|---|---|
| **Train from scratch on domain corpus** | New vocabulary, new merges/pieces | Maximally efficient for the domain; clean algorithmic choice | Cannot reuse any pre-trained model weights; need to train the LM from scratch too |
| **Extend an existing tokeniser's vocabulary** | Keep original pieces + add N domain pieces | Preserves backwards compatibility; works with existing pre-trained model if embedding matrix is resized and new rows are trained | Hybrid vocab can be suboptimal; need to handle the new embedding rows during fine-tuning |

The deliverable is: the trained tokeniser artifact, a comparison against a baseline tokeniser on a held-out domain corpus, and a short memo documenting the choice.

## 2. Worked answer / implementation

### 2.1 Corpus preparation

Quality matters more than quantity for domain adaptation. For a 32k-vocab tokeniser, 100-500 MB of in-domain text is usually enough; more helps a little, less works surprisingly well.

```python
# corpus_prep.py
import unicodedata
from pathlib import Path

def normalise_line(line: str) -> str:
    # Pin NFC before anyone else gets a chance to vary it (see exercise-03).
    line = unicodedata.normalize("NFC", line.strip())
    return line

def write_corpus(src_glob: str, out_path: str) -> int:
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for p in Path().glob(src_glob):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = normalise_line(line)
                    if line:
                        out.write(line + "\n")
                        n += 1
    return n
```

Checklist:

- Fix normalisation form (NFC) explicitly.
- Deduplicate near-duplicate lines (minhash or just line-level `sort -u` for a small corpus).
- Keep a disjoint ~5% held-out slice for fertility evaluation, never shown to the trainer.
- Document the corpus provenance and licence — the trained tokeniser inherits those terms.

### 2.2 Option A — Train from scratch (SentencePiece Unigram)

Default choice for a new project where you also plan to train / fine-tune a model from scratch.

```python
# train_domain_scratch.py
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="domain_unigram_32k",
    model_type="unigram",
    vocab_size=32000,
    character_coverage=0.9995,
    normalization_rule_name="identity",   # we normalised upstream
    user_defined_symbols=["[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    byte_fallback=True,                    # robust to unseen bytes
    unk_surface=" ⁇ ",
    seed_sentencepiece_size=1000000,
)
```

`byte_fallback=True` matters: with it set, any character not in the final vocabulary falls back to byte pieces, so the tokeniser never emits `[UNK]` in practice. This is the approach used by many recent models (e.g., Llama family); it is documented in the SentencePiece repo.

### 2.3 Option B — Extend an existing tokeniser

Default choice when you plan to continue pre-training on top of an existing model (e.g., start from `bert-base-uncased` and keep its first-stage knowledge).

```python
# extend_vocab.py
from collections import Counter
from transformers import AutoTokenizer

BASE = "bert-base-uncased"
N_NEW = 2000

def count_fertile_tokens(tokeniser, lines, tokens_per_word_threshold=3):
    """Find words the base tokeniser splits into many pieces — candidates to add whole."""
    fert: Counter = Counter()
    for line in lines:
        for word in line.split():
            pieces = tokeniser.tokenize(word)
            if len(pieces) >= tokens_per_word_threshold:
                fert[word] += 1
    return fert

base = AutoTokenizer.from_pretrained(BASE)
with open("corpus.txt", encoding="utf-8") as f:
    corpus_lines = [line.strip() for line in f if line.strip()]

candidates = count_fertile_tokens(base, corpus_lines).most_common(N_NEW * 3)

# Filter: keep only words that are mostly letters and appear in multiple contexts.
keep = [w for w, c in candidates if w.isalpha() and c >= 5][:N_NEW]

added = base.add_tokens(keep)   # resizes vocab only; model.resize_token_embeddings needed downstream
base.save_pretrained("./bert-domain-extended")
print(f"added {added} new tokens")
```

Memo should state:

- Which model's weights are being carried forward.
- Expected `model.resize_token_embeddings(len(tokenizer))` call during fine-tuning.
- Plan for initialising the new embedding rows (random init → the model has to learn them; mean-of-constituent-pieces initialisation is the common trick).

### 2.4 Evaluation protocol

Compute these on the held-out domain corpus against a baseline tokeniser (the general-purpose one you would have otherwise used):

| Metric | Direction | Why |
|---|---|---|
| Mean tokens per word | lower is better for domain-adapted | Headline fertility |
| Mean tokens per *domain-term* | lower is better | Fertility focused on target vocabulary |
| Token count per 1 000 chars | lower is better | Compression; translates to context-window savings |
| UNK / byte-fallback rate | should be ~0 | Sanity |
| Coverage of a curated domain-term list | higher is better | Ensures the pieces you *care about* are single tokens |
| Overlap with baseline vocab | descriptive | High overlap = conservative extension; low overlap = aggressive new vocab |

Minimal evaluator:

```python
# eval_domain.py
from transformers import AutoTokenizer
import sentencepiece as spm

def fertility(enc, lines):
    toks = sum(len(enc(line)) for line in lines)
    words = sum(len(line.split()) for line in lines)
    return toks / max(words, 1)

baseline = AutoTokenizer.from_pretrained("bert-base-uncased")
domain = spm.SentencePieceProcessor(model_file="domain_unigram_32k.model")

held = open("heldout.txt", encoding="utf-8").read().splitlines()
print("baseline fertility:", fertility(lambda s: baseline.tokenize(s), held))
print("domain fertility  :", fertility(lambda s: domain.encode(s, out_type=str), held))
```

### 2.5 Expected result and how to report it

For a typical specialised domain (e.g. biomedical abstracts, SEC filings, SQL logs), a from-scratch domain tokeniser at matched vocab size will reduce mean tokens per word by **20-50%** relative to GPT-2 byte BPE or `bert-base-uncased` WordPiece, concentrated on the terms the baseline splits into 3+ pieces. The extension strategy typically captures **60-80% of that gain** while preserving compatibility with the base model. These ranges are what writers should aim to measure empirically; concrete numbers depend on corpus and vocab size and must be reported from the actual experiment, not assumed.

The writeup should include:

1. Strategy chosen (A or B) with one-paragraph justification tied to the downstream plan (train from scratch vs continue pre-training).
2. Corpus stats: raw size, deduped size, language mix, held-out size.
3. Vocab-size parity note (matched to baseline where possible).
4. Full metric table.
5. Five to ten qualitative segmentation examples of domain terms, side by side.
6. A limitations paragraph (overfitting to corpus style, data provenance, licence inheritance, how the added embeddings will be initialised downstream).

## 3. Validation steps

1. **Round-trip test.** `decode(encode(x)) == x` modulo documented whitespace normalisation on 1 000 held-out lines.
2. **Byte-fallback safety.** For SP Unigram with `byte_fallback`, the UNK rate on held-out data must be effectively zero. If not, investigate — a non-zero UNK means something in the encoder path is bypassing fallback.
3. **Vocab-size parity.** `len(baseline.vocab) == len(domain.vocab)` within specials. Otherwise fertility differences include a size effect.
4. **Fertility diff significance.** Bootstrap the mean fertility over held-out lines; report a 95% CI. Domain tokeniser's CI should not overlap baseline's.
5. **Downstream compatibility check (Option B only).** Instantiate the base model, call `model.resize_token_embeddings(len(tokeniser))`, run one forward pass on a domain sample — no shape errors, finite logits.
6. **Determinism.** Fixed seed and fixed input ordering yield byte-identical model files (`sha256sum`).

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Strategy choice** | Option A or B picked, justified against downstream plan | Picked but not justified | Both tried, neither evaluated |
| **Corpus hygiene** | NFC fixed, dedup, held-out slice disjoint, provenance documented | 2 of 4 | Trained on raw mixed data |
| **Fair comparison** | Matched vocab size; baseline named and pinned | Baseline named only | Comparing to a different-size baseline |
| **Metric coverage** | Fertility (word & char), UNK/fallback rate, domain-term coverage | 2 of 3 | One metric only |
| **Qualitative evidence** | 5-10 domain-term side-by-side segmentations | 2-4 | None |
| **Downstream compatibility (B only)** | Resize + forward pass demonstrated | Discussed only | Ignored |
| **Limitations / risks** | Data-licence inheritance, overfitting to corpus style, embedding-init plan addressed | 1-2 of 3 | None |
| **Reproducibility** | Fixed seed, pinned deps, one-command repro | Partial | None |

## 5. Common mistakes

- **Comparing against a different-size baseline.** A 50k-vocab domain tokeniser vs a 30k-vocab baseline will always "win" on fertility — but for the wrong reason. Match vocab size.
- **Training on leaked held-out data.** If the held-out set is a random sample from the training file, fertility understates real-world token count. Hold out *lines* before training, not after.
- **Forgetting embedding resize (Option B).** Adding tokens with `tokenizer.add_tokens()` changes the vocab size; the model's embedding matrix still has the old shape. Downstream code must call `model.resize_token_embeddings(len(tokeniser))` or inference will crash or silently point to garbage indices.
- **New-row initialisation with random noise and then freezing.** New embedding rows initialised randomly have no useful signal; fine-tuning must actually update them. Alternatively, initialise each new row as the mean of the pieces it would have decomposed into under the old tokeniser — a well-known trick that reduces cold-start.
- **Shipping NFKC-trained tokeniser for display-oriented tasks.** NFKC collapses ligatures, superscripts, full-width forms. If the downstream task preserves display formatting, this is lossy.
- **Ignoring licence inheritance.** A tokeniser trained on licensed text embeds vocabulary derived from that text. Document provenance; some licences restrict derivative artefacts.
- **Overlooking byte fallback.** Without byte fallback (or an equivalent OOV plan) the tokeniser will emit `[UNK]` on unseen characters in production — unrecoverable for the model.

## 6. References

Primary sources used in this solution:

- Kudo, T. (2018). *Subword Regularization* — Unigram LM algorithm. <https://aclanthology.org/P18-1007/>
- Kudo, T. & Richardson, J. (2018). *SentencePiece* — tokenizer + CLI; `byte_fallback`, `character_coverage`, normalisation rules. <https://aclanthology.org/D18-2012/>
- Google SentencePiece project — training options, including `byte_fallback`. <https://github.com/google/sentencepiece>
- Hugging Face Transformers — `tokenizer.add_tokens`, `model.resize_token_embeddings` documentation. <https://huggingface.co/docs/transformers/main_classes/tokenizer>
- Hugging Face `tokenizers` library — <https://huggingface.co/docs/tokenizers> — WordPiece / BPE / Unigram trainers used for extension strategies.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — data-provenance risks in training artefacts.
- MITRE ATLAS — <https://atlas.mitre.org/> — adversarial manipulation of training corpora.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — Govern/Map functions require documenting training-data provenance and derivative artefacts; a domain tokeniser is such an artefact.
