# exercise-02 — SentencePiece Unigram vs. WordPiece Experiment

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-101-tokenization-and-text-foundations/exercises/exercise-02-sentencepiece-unigram-vs-wordpiece-experiment.md`.
> This solution targets the learning goals that filename implies: train two subword tokenisers (Unigram LM via SentencePiece, and WordPiece via Hugging Face `tokenizers`) on the same corpus, measure them against the same held-out set, and explain the differences in algorithm, segmentation behaviour, and downstream suitability.

## 1. Solution overview

Unigram (Kudo 2018) and WordPiece (Schuster & Nakajima 2012, popularised by BERT) are both *probabilistic* subword methods, but they get there by opposite directions:

- **Unigram LM** starts from a large seed vocabulary (often derived from suffix-array substrings) and *prunes* pieces that contribute least to a unigram language-model likelihood of the training corpus. At inference it keeps the full vocabulary with per-piece log-probabilities and can produce multiple candidate segmentations (Viterbi for the best, sampling for regularisation).
- **WordPiece** starts from a character vocabulary and *grows* it by greedily merging pieces whose merged form maximises training-corpus likelihood under a unigram model. Inference is longest-match-first from the left (greedy), not Viterbi.

The exercise is to see these differences in a controlled experiment: one corpus, matched vocab size, same held-out text, side-by-side metrics. The deliverable is a numeric comparison plus a short written interpretation.

## 2. Worked answer / implementation

### 2.1 Training both tokenisers

The canonical implementations are:

- **Unigram** — Google's `sentencepiece` library. See <https://github.com/google/sentencepiece> for the training CLI and the `SentencePieceTrainer` Python API.
- **WordPiece** — Hugging Face `tokenizers` ships `WordPiece` + `WordPieceTrainer`. See <https://huggingface.co/docs/tokenizers>.

```python
# train_tokenizers.py
import sentencepiece as spm
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer


def train_unigram(corpus_path: str, vocab_size: int, model_prefix: str) -> None:
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        model_type="unigram",
        vocab_size=vocab_size,
        character_coverage=0.9995,  # per SP guidance for Latin scripts
        normalization_rule_name="nmt_nfkc",
    )


def train_wordpiece(corpus_path: str, vocab_size: int, out_path: str) -> None:
    tok = Tokenizer(WordPiece(unk_token="[UNK]"))
    # Match SentencePiece's `nmt_nfkc`: NFKC is the core compatibility normalisation
    # (ligatures, full-width → half-width, superscripts); the "nmt_" prefix also strips
    # control characters and trims whitespace, which `BertPreTokenizer` already handles
    # downstream. `BertPreTokenizer` is pre-tokenisation only — it does NOT normalise —
    # so without an explicit `tok.normalizer` the WordPiece pipeline would skip
    # compatibility normalisation entirely and the Unigram/WordPiece delta would
    # conflate algorithm differences with normalisation differences.
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = BertPreTokenizer()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        continuing_subword_prefix="##",
    )
    tok.train([corpus_path], trainer)
    tok.save(out_path)
```

Two pragmatic choices worth flagging:

- **Matched vocab size.** For a fair comparison, both trainers must *target* the same total vocabulary size (e.g. 16 000 or 32 000). SentencePiece's count includes special tokens and the initial character set; so does `WordPieceTrainer`. Specify the same number on both sides and verify after training (`len(sp.get_piece_size())` vs `tokenizer.get_vocab_size()`).
- **Matched normalisation.** SentencePiece applies `nmt_nfkc` by default (NFKC plus whitespace collapsing); `BertPreTokenizer` is pre-tokenisation only and performs no normalisation, so WordPiece without an explicit `tok.normalizer` sees raw compatibility characters (`ﬁ`, `Ⅳ`, full-width letters) that SentencePiece has already folded. The snippet above attaches a matching `NFKC` normaliser to WordPiece so the two trainers see the same input; if you prefer the classic BERT recipe (`BertNormalizer(lowercase=True, strip_accents=True)`), apply the *same* recipe upstream of SentencePiece — either way, keep normalisation identical so the delta you measure is the *algorithm* delta.

### 2.2 Evaluation protocol

Compute the following on a held-out slice disjoint from the training corpus. (A 90/10 line split is enough for a small assignment; prefer 95/5 if the corpus is large.)

| Metric | Why it matters | How to compute |
|---|---|---|
| **Mean tokens per word** | Fertility — lower = more efficient packing | `sum(len(tokens_i)) / sum(words_in_line_i)` over the held-out set |
| **Mean tokens per character** | Script-independent fertility | `sum(len(tokens_i)) / sum(chars_in_line_i)` |
| **OOV / `[UNK]` rate** | How often each tokeniser falls back | fraction of tokens equal to the unk piece |
| **Vocabulary coverage of held-out words** | How many held-out word types get at least one in-vocab piece | set-intersection of held-out word-forms and vocabulary |
| **Segmentation entropy** (Unigram only) | Non-determinism cost at inference | shannon entropy over top-k Viterbi outputs at a fixed temperature |
| **Round-trip fidelity** | Does `decode(encode(x)) == x` | assert on every held-out line |
| **Longest-match divergence** | Qualitative | print 10 lines where segmentations differ most |

Minimal evaluator:

```python
# eval.py
import sentencepiece as spm
from tokenizers import Tokenizer

def encode_sp(sp: spm.SentencePieceProcessor, text: str):
    return sp.encode(text, out_type=str)

def encode_wp(tok: Tokenizer, text: str):
    return tok.encode(text).tokens

def fertility(encoder, lines):
    n_words = sum(len(line.split()) for line in lines)
    n_toks = sum(len(encoder(line)) for line in lines)
    return n_toks / max(n_words, 1)
```

### 2.3 What to expect and what to report

Published and widely-reproduced observations about these algorithms (see §6 refs):

- **Unigram typically yields slightly *fewer* tokens per word** on natural-language text at matched vocab size, because Viterbi at inference can choose longer in-vocab pieces that greedy left-to-right WordPiece misses.
- **WordPiece's greedy inference is faster** and simpler (one pass per word) and has no ambiguity, which is why BERT-family models retained it.
- **Unigram supports subword regularisation** (`sample_encode_as_pieces`) — sampling different segmentations at training time improves downstream model robustness; WordPiece has no analogue.
- **Rare and compound words diverge most.** WordPiece's `##`-continuation and greedy inference give characteristic left-anchored splits (`unbelievable → un + ##bel + ##iev + ##able`), while Unigram often picks a more balanced decomposition driven by piece probabilities.

The writeup should include (a) the full table of metrics above, (b) a histogram or short list of fertility per-line, (c) 5-10 concrete side-by-side segmentations with a sentence of interpretation each.

## 3. Validation steps

1. **Sanity-check vocab sizes.** After training, assert `abs(len(sp_vocab) - wp_vocab_size) < len(special_tokens) + 10`. If the gap is large, your special-token / character-coverage config disagrees.
2. **Round-trip test both tokenisers** on 100 random held-out lines. `sp.decode(sp.encode(x)) == x` modulo whitespace normalisation; `tok.decode(tok.encode(x).ids) == x` modulo the WordPiece `##`-stripping decoder.
3. **Entropy-sample Unigram** with `sample_encode_as_pieces(..., nbest_size=-1, alpha=0.1)` and confirm segmentations vary — this verifies Unigram is actually probabilistic in your pipeline.
4. **Deterministic replay.** `SentencePieceTrainer.train(..., seed_sentencepiece_size=...)` and a fixed input ordering should reproduce byte-identical model files across runs; verify with `sha256sum`.
5. **Held-out leakage check.** `assert not (train_lines & heldout_lines)` at the set-of-lines level before computing fertility.
6. **Document all config.** Normalisation, character coverage, special-token list, vocab size, corpus size, seed — the experiment is not comparable without them.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Fair comparison** | Matched vocab size, matched normalisation, disjoint held-out | Matched vocab only | Different corpora or vocab sizes |
| **Metric coverage** | Fertility (word & char), OOV rate, coverage, round-trip | 2-3 metrics | Only one metric or qualitative only |
| **Algorithmic explanation** | Articulates top-down-prune vs bottom-up-grow and Viterbi vs greedy | Explains one axis only | No algorithmic framing |
| **Qualitative evidence** | ≥5 side-by-side segmentations with interpretation | Only aggregate numbers | None |
| **Regularisation** | Demonstrates Unigram subword regularisation and discusses when to use it | Mentions but does not demonstrate | Not mentioned |
| **Reproducibility** | Fixed seed, pinned deps, config captured | Partial | Not reproducible |

## 5. Common mistakes

- **Mixing normalisation into the result.** If SentencePiece does NFKC but WordPiece does not (or vice versa), half the "algorithm" differences you measure are actually normalisation differences. Pre-normalise once, disable further normalisation on both sides.
- **Comparing against different vocab sizes.** Fertility scales with vocab size; a 32k Unigram vs 16k WordPiece comparison is uninformative. Always matched.
- **Ignoring the character-coverage knob.** `character_coverage=1.0` (SP default for Western text is 0.9995) decides whether rare glyphs become their own pieces or fall back to byte-level. This dominates OOV-rate comparisons on multilingual text.
- **Treating `[UNK]` as always-fatal.** WordPiece emits `[UNK]` when no prefix matches; Unigram practically never emits UNK thanks to its fallback characters. Cross-tabulating UNK rates without noting this leads to the conclusion "Unigram is always better" when the real cause is fallback policy.
- **Measuring only on training data.** Fertility on the training corpus flatters whichever tokeniser overfit. Always hold out.
- **Using Unigram's greedy `encode` and calling it an algorithm comparison.** `sp.encode(text)` returns the Viterbi-best segmentation; if you want a determinism comparison, say so explicitly and keep the Viterbi comparison as the primary number.

## 6. References

Primary sources used in this solution:

- Kudo, T. (2018). *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*. ACL. <https://aclanthology.org/P18-1007/> — the Unigram LM algorithm and subword sampling.
- Kudo, T. & Richardson, J. (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*. EMNLP demo. <https://aclanthology.org/D18-2012/> — the library and its normalisation/coverage controls.
- Schuster, M. & Nakajima, K. (2012). *Japanese and Korean Voice Search*. ICASSP. <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf> — the original WordPiece algorithm.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL. <https://aclanthology.org/N19-1423/> — WordPiece as used in BERT, with `##` continuation marker.
- Google SentencePiece project — <https://github.com/google/sentencepiece> — training and inference API, model types, normalisation rules.
- Hugging Face `tokenizers` documentation — <https://huggingface.co/docs/tokenizers> — `WordPiece`, `WordPieceTrainer`, `BertPreTokenizer`.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — <https://atlas.mitre.org/>
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework>
