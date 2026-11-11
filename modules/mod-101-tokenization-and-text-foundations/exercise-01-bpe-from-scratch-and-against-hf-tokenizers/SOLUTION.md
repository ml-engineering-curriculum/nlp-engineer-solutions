# exercise-01 — BPE From Scratch and Against Hugging Face `tokenizers`

## 1. Solution overview

The exercise asks learners to (a) implement byte-pair encoding (BPE) from
scratch on a small corpus, (b) train a comparable Hugging Face `tokenizers`
BPE model on the same corpus, and (c) analyse differences in the learned
merge table and the resulting token sequences.

The reference solution has three artifacts:

1. `bpe_from_scratch.py` — a self-contained trainer and encoder that follows
   Sennrich, Haddow and Birch (2016) "Neural Machine Translation of Rare
   Words with Subword Units" (ACL 2016 / arXiv:1508.07909). It operates on
   pre-tokenised word counts and learns a merge list by greedy frequency.
2. `train_hf_bpe.py` — a Hugging Face `tokenizers` recipe that trains
   `BpeTrainer` on the same corpus with explicit pre-tokeniser,
   normaliser, and decoder settings so that the comparison is apples to
   apples.
3. `compare.py` — loads both tokenizers, encodes a shared probe set
   (English prose, a non-ASCII paragraph, a URL, and an uncommon
   identifier), and prints merge-order diffs and token-sequence diffs.

Scope deliberately avoids byte-level BPE (covered in exercise-02's
unigram/WordPiece pairing and in GPT-family notes in exercise-05). The
solution uses character-level BPE so students can read the merges by eye.

## 2. Worked answer or implementation

### 2.1 Minimal from-scratch BPE

```python
# bpe_from_scratch.py
"""Minimal character-level BPE, following Sennrich, Haddow & Birch (2016).

The trainer operates over a word-count dictionary. Each word is stored as a
tuple of characters with an end-of-word marker ``</w>`` so that merges at
word boundaries stay bounded.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


END_OF_WORD = "</w>"


def word_counts(corpus: Iterable[str]) -> Counter[str]:
    """Lowercase, split on whitespace, count words. Keep punctuation attached
    so that learners can see how punctuation-adjacent merges emerge."""
    counts: Counter[str] = Counter()
    for line in corpus:
        for word in line.strip().split():
            if word:
                counts[word] += 1
    return counts


def initial_vocab(counts: Counter[str]) -> dict[tuple[str, ...], int]:
    """Represent each word as a tuple of symbols with an end-of-word marker."""
    return {tuple(word) + (END_OF_WORD,): freq for word, freq in counts.items()}


def pair_stats(vocab: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
    stats: Counter[tuple[str, str]] = Counter()
    for symbols, freq in vocab.items():
        for a, b in zip(symbols, symbols[1:]):
            stats[(a, b)] += freq
    return stats


def merge_vocab(
    pair: tuple[str, str], vocab: dict[tuple[str, ...], int]
) -> dict[tuple[str, ...], int]:
    merged: dict[tuple[str, ...], int] = {}
    joined = pair[0] + pair[1]
    for symbols, freq in vocab.items():
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if (
                i < len(symbols) - 1
                and symbols[i] == pair[0]
                and symbols[i + 1] == pair[1]
            ):
                new_symbols.append(joined)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        merged[tuple(new_symbols)] = merged.get(tuple(new_symbols), 0) + freq
    return merged


@dataclass
class BPEModel:
    merges: list[tuple[str, str]]
    vocab: set[str]

    def encode_word(self, word: str) -> list[str]:
        symbols = list(word) + [END_OF_WORD]
        rank = {pair: i for i, pair in enumerate(self.merges)}
        while True:
            pairs = list(zip(symbols, symbols[1:]))
            if not pairs:
                break
            candidate = min(pairs, key=lambda p: rank.get(p, float("inf")))
            if candidate not in rank:
                break
            # apply the earliest-merged pair once (left-to-right).
            new_symbols: list[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and (symbols[i], symbols[i + 1]) == candidate
                ):
                    new_symbols.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols


def train_bpe(corpus: Iterable[str], num_merges: int) -> BPEModel:
    vocab = initial_vocab(word_counts(corpus))
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        stats = pair_stats(vocab)
        if not stats:
            break
        best = max(stats, key=stats.get)
        if stats[best] < 2:  # no further compressive merges available
            break
        vocab = merge_vocab(best, vocab)
        merges.append(best)
    tokens: set[str] = set()
    for symbols in vocab:
        tokens.update(symbols)
    return BPEModel(merges=merges, vocab=tokens)


if __name__ == "__main__":
    import sys

    text = sys.stdin.read().splitlines()
    model = train_bpe(text, num_merges=200)
    for i, (a, b) in enumerate(model.merges):
        print(f"{i:03d}\t{a}\t{b}")
```

### 2.2 Equivalent Hugging Face training recipe

```python
# train_hf_bpe.py
"""Train a HF `tokenizers` BPE model with settings that mirror the
from-scratch trainer: whitespace pre-tokenisation, no byte-level fallback,
no lowercasing beyond what the user passes in."""
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.BPEDecoder()
    return tok


def train(corpus_path: Path, vocab_size: int = 500) -> Tokenizer:
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        show_progress=False,
        min_frequency=2,
    )
    tok.train(files=[str(corpus_path)], trainer=trainer)
    return tok


if __name__ == "__main__":
    import sys

    tok = train(Path(sys.argv[1]))
    tok.save("hf_bpe.json")
```

### 2.3 Comparison probes

```python
# compare.py
from pathlib import Path

from tokenizers import Tokenizer

import bpe_from_scratch as scratch


PROBES = [
    "the quick brown fox jumped over the lazy dog",
    "naïve café résumé",            # NFC + diacritics
    "https://example.com/path?x=1",  # URL-like
    "tokeniser tokenizer tokenization",  # morphological family
]


def main(corpus: Path) -> None:
    scratch_model = scratch.train_bpe(corpus.read_text().splitlines(), 200)
    hf_model = Tokenizer.from_file("hf_bpe.json")
    for probe in PROBES:
        scratch_tokens = [
            tok for word in probe.split() for tok in scratch_model.encode_word(word)
        ]
        hf_tokens = hf_model.encode(probe).tokens
        print(probe)
        print("  scratch:", scratch_tokens)
        print("  hf     :", hf_tokens)


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]))
```

### 2.4 What the comparison should reveal

- **Merge order differs for ties.** The 2016 BPE paper does not specify
  tie-breaking; the reference implementation uses `Counter.max`, which
  returns insertion order. HF `tokenizers` defines its own deterministic
  rule. Ties on small corpora can therefore permute merges 1–20 even when
  the frequency counts are identical.
- **End-of-word handling.** The from-scratch model uses an explicit
  `</w>` marker; HF's `BPE` model stores whether a token is word-initial
  via its serialised form instead. Learners should note the `Ġ`-style
  prefix is only introduced by `ByteLevel` pre-tokenisation, not by the
  `Whitespace` pre-tokeniser used here.
- **Non-ASCII input.** On `naïve`, the from-scratch trainer learns merges
  over Unicode scalar values; HF does the same under `Whitespace`. Byte-
  level BPE (e.g. GPT-2) would instead operate on UTF-8 bytes — covered as
  a stretch bullet rather than in the main solution.

## 3. Validation steps

1. **Deterministic on a fixed corpus.** Running `bpe_from_scratch.py` twice
   on the same input must produce byte-identical merge lists. The trainer
   has no randomness, so a diff should be empty.
2. **Round-trip encode → decode.** For HF, `tok.decode(tok.encode(s).ids)`
   on each probe must equal `s` after whitespace normalisation. The HF
   `BPEDecoder` reconstructs words by concatenating sub-tokens; verify the
   output matches the pre-tokeniser's view, not the raw string with
   collapsed spaces.
3. **Compression check.** On the training corpus, `len(tokens) /
   len(words)` should decrease monotonically as `num_merges` grows, up to
   the point where the merge table saturates the vocabulary. Plot this to
   confirm the trainer is making progress.
4. **Behavioural parity test.** Pick a word that appears many times in the
   corpus. After enough merges it should encode to a single token under
   both tokenizers. If it does not, inspect the pre-tokeniser — the usual
   cause is Unicode normalisation differences.
5. **Vocabulary inclusion.** Assert that every character present in the
   training corpus appears in `model.vocab`. Characters never covered by a
   merge still need to live in the vocabulary as singletons, otherwise
   encoding unseen words produces `<unk>`.

## 4. Rubric or review checklist

| Criterion | Weight | Evidence |
|---|---|---|
| From-scratch trainer follows Sennrich et al. (pair counting, greedy merge) | 25% | `train_bpe` loop; merge list matches a hand-traced toy example |
| HF trainer configured with explicit pre-tokeniser, decoder, special tokens | 15% | `build_tokenizer` block; no hidden defaults |
| Comparison surfaces at least 2 real differences (merge order, pre-tokenisation, end-of-word marker, normaliser behaviour) | 20% | `compare.py` output discussed in write-up |
| Round-trip and determinism tests present and passing | 15% | `pytest` or notebook assertions |
| Discussion links algorithmic choice to downstream impact (OOV rate, sequence length, cross-lingual behaviour) | 15% | Written analysis cites corpus-specific counts, not generic claims |
| Code readable: types, docstrings, one responsibility per function | 10% | Lint/format clean |

A passing submission scores ≥70% with all "must" rows (trainer correctness,
round-trip test) above zero. Anything that ships a non-deterministic
trainer or silently drops characters fails regardless of total score.

## 5. Common mistakes

- **Treating `max(stats, key=stats.get)` as stable.** On CPython it is
  effectively insertion-ordered, but relying on that for cross-implementation
  parity bites learners when they try to reproduce HF's merge order. The
  fix is to tie-break on `(count, pair)` explicitly if parity is desired.
- **Forgetting the end-of-word marker.** Without it, merges learned at
  word boundaries bleed across words, producing a tokenizer that cannot
  tell `er` at the end of `walker` from `er` at the start of `erase`.
- **Running HF with `ByteLevel` pre-tokenisation and comparing to a
  Unicode-char trainer.** The two will never match token-for-token; use
  `Whitespace` for the apples-to-apples case and note the byte-level case
  separately.
- **Treating the merge list as a vocabulary.** The vocabulary is the set
  of symbols reachable after all merges plus the base alphabet, not the
  merge list itself.
- **Not freezing the random seed on HF.** The HF `BpeTrainer` is
  deterministic given a fixed corpus and settings, but shuffled input
  files introduce apparent nondeterminism. Sort files before passing them
  in when reproducibility matters.

## 6. References

- Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine
  Translation of Rare Words with Subword Units.* ACL 2016. arXiv:1508.07909.
  Original BPE algorithm for NMT.
- Hugging Face Tokenizers documentation — `models.BPE`, `BpeTrainer`,
  `pre_tokenizers.Whitespace`, `decoders.BPEDecoder`
  (`https://huggingface.co/docs/tokenizers`).
- Unicode Standard Annex #29, *Unicode Text Segmentation* — background on
  word-boundary rules that the HF `Whitespace` pre-tokeniser approximates
  (`https://unicode.org/reports/tr29/`).
- NIST AI Risk Management Framework (AI RMF 1.0), section on *Measure* —
  tokenization choices contribute to the "representativeness" measurement
  obligations when the downstream model is used in high-impact contexts
  (`https://www.nist.gov/itl/ai-risk-management-framework`).
