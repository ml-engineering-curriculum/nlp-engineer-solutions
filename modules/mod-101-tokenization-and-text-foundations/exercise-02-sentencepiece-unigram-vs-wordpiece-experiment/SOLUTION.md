# exercise-02 — SentencePiece Unigram vs. WordPiece Experiment

## 1. Solution overview

Learners train two tokenizers on the same corpus and vocabulary budget:

- **SentencePiece Unigram** (Kudo, 2018, "Subword Regularization") via
  the `sentencepiece` reference implementation, and
- **WordPiece** (Schuster & Nakajima, 2012; Wu et al., 2016 for the NMT
  variant) via Hugging Face `tokenizers` (`models.WordPiece` +
  `trainers.WordPieceTrainer`).

The reference solution structures the experiment as a small script
suite with one shared evaluation harness so that the comparison is
quantitative, not anecdotal. Four measurements are collected:

1. **Token/word ratio** on a held-out evaluation set.
2. **OOV rate** — fraction of evaluation characters that fall back to
   `<unk>`.
3. **Round-trip exactness** — percentage of evaluation sentences that
   decode to the input after NFC normalisation.
4. **Vocabulary overlap** between the two models.

Results are reported as a small table rather than narrative claims —
exact numbers depend on the corpus and are not pre-populated.

## 2. Worked answer or implementation

### 2.1 Project layout

```
exercise-02-sentencepiece-unigram-vs-wordpiece-experiment/
├── SOLUTION.md
├── train_unigram.py
├── train_wordpiece.py
├── evaluate.py
└── data/            # supplied by the learner
```

### 2.2 Unigram training

```python
# train_unigram.py
"""Train a SentencePiece Unigram model.

SentencePiece treats the input as a raw Unicode stream; a space is
represented by the ``▁`` meta-symbol. This is the key architectural
difference from WordPiece, which assumes a word-level pre-tokenisation.
"""
from pathlib import Path

import sentencepiece as spm


def train(corpus: Path, out_prefix: str, vocab_size: int = 8000) -> None:
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=out_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,       # Kudo et al. recommend < 1.0 for CJK
        normalization_rule_name="nmt_nfkc_cf",
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        shuffle_input_sentence=True,
    )


if __name__ == "__main__":
    import sys

    train(Path(sys.argv[1]), sys.argv[2])
```

### 2.3 WordPiece training

```python
# train_wordpiece.py
"""Train a WordPiece model via the HF `tokenizers` library.

WordPiece requires an explicit pre-tokeniser (we use BERT-style
whitespace+punctuation). Continuation sub-tokens are prefixed with
``##``, matching the BERT convention (Devlin et al., 2019).
"""
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


def build() -> Tokenizer:
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
    ])
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece(prefix="##")
    return tok


def train(corpus: Path, out_file: str, vocab_size: int = 8000) -> None:
    tok = build()
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        continuing_subword_prefix="##",
        min_frequency=2,
        show_progress=False,
    )
    tok.train([str(corpus)], trainer=trainer)
    tok.save(out_file)


if __name__ == "__main__":
    import sys

    train(Path(sys.argv[1]), sys.argv[2])
```

### 2.4 Shared evaluation harness

```python
# evaluate.py
"""Compare a SentencePiece model and a WordPiece model on an eval file."""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import sentencepiece as spm
from tokenizers import Tokenizer


@dataclass
class Metrics:
    name: str
    sentences: int
    words: int
    tokens: int
    unk_chars: int
    chars: int
    roundtrip_exact: int

    def token_word_ratio(self) -> float:
        return self.tokens / max(1, self.words)

    def oov_rate(self) -> float:
        return self.unk_chars / max(1, self.chars)

    def roundtrip_rate(self) -> float:
        return self.roundtrip_exact / max(1, self.sentences)


def _lines(path: Path) -> Iterable[str]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                yield line


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def eval_sp(model_path: Path, eval_path: Path) -> Metrics:
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    unk_id = sp.unk_id()
    m = Metrics(name=model_path.name, sentences=0, words=0, tokens=0,
                unk_chars=0, chars=0, roundtrip_exact=0)
    for line in _lines(eval_path):
        ids = sp.encode(line, out_type=int)
        pieces = sp.encode(line, out_type=str)
        m.sentences += 1
        m.words += len(line.split())
        m.tokens += len(ids)
        m.chars += len(line)
        m.unk_chars += sum(len(p) for p, i in zip(pieces, ids) if i == unk_id)
        if _nfc(sp.decode(ids)) == _nfc(line):
            m.roundtrip_exact += 1
    return m


def eval_hf(model_path: Path, eval_path: Path) -> Metrics:
    tok = Tokenizer.from_file(str(model_path))
    unk = "[UNK]"
    m = Metrics(name=model_path.name, sentences=0, words=0, tokens=0,
                unk_chars=0, chars=0, roundtrip_exact=0)
    for line in _lines(eval_path):
        enc = tok.encode(line)
        m.sentences += 1
        m.words += len(line.split())
        m.tokens += len(enc.tokens)
        m.chars += len(line)
        m.unk_chars += sum(len(t) for t in enc.tokens if t == unk)
        if _nfc(tok.decode(enc.ids)) == _nfc(line):
            m.roundtrip_exact += 1
    return m


def report(ms: list[Metrics]) -> None:
    rows = [
        {
            "model": m.name,
            "tokens_per_word": round(m.token_word_ratio(), 3),
            "oov_rate": round(m.oov_rate(), 5),
            "roundtrip_rate": round(m.roundtrip_rate(), 3),
        }
        for m in ms
    ]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    import sys

    sp_metrics = eval_sp(Path(sys.argv[1]), Path(sys.argv[3]))
    wp_metrics = eval_hf(Path(sys.argv[2]), Path(sys.argv[3]))
    report([sp_metrics, wp_metrics])
```

### 2.5 Expected qualitative findings

The exact numbers are corpus dependent and should not be pre-filled.
Learners should still see the following structural effects:

- **Unigram tends to produce fewer tokens per word** on morphologically
  rich input because it is trained by maximum likelihood over a lattice
  of segmentations rather than a greedy merge list (Kudo, 2018).
- **WordPiece's `##` continuation prefix** makes tokenization sensitive
  to the pre-tokeniser's word boundaries, so punctuation-heavy inputs
  (URLs, code identifiers) are more fragmented than under SentencePiece,
  which operates on the raw stream.
- **Round-trip exactness favours SentencePiece** when the normaliser is
  held constant: WordPiece's BERT-style pipeline applies
  lowercase/strip-accents, which destroys information before merging and
  therefore cannot reconstruct the original casing/diacritics.
- **Vocabulary overlap** is non-trivial in theory (both models can
  represent any character), but set overlap at a fixed vocabulary size is
  usually moderate — expect substantial disagreement on which multi-
  character pieces are kept.

<!-- research-note: cite a published head-to-head benchmark if one is
required by the exercise; Kudo (2018) reports BLEU deltas vs. BPE on
WMT en-de and en-ja but does not evaluate against WordPiece directly. -->

## 3. Validation steps

1. **Vocabulary size matches request.** Load each model and assert the
   vocabulary is within 1% of the requested size. Both trainers can fall
   short if the corpus is too small.
2. **Normaliser parity for eval.** Before computing round-trip rates,
   apply the same Unicode normalisation (NFC) to both model outputs.
   Otherwise a decode that differs only in composition form is
   mis-scored.
3. **No silent `<unk>` explosion.** Plot `oov_rate` vs. vocabulary size
   for both tokenizers at {1k, 4k, 8k, 16k}. The curves should be
   monotonically decreasing; a flat curve usually means the pre-tokeniser
   is throwing away input.
4. **Deterministic re-runs.** Fix the shuffle seed
   (`SentencePieceTrainer.train(input_sentence_size=..., seed=...)` and
   HF `BpeTrainer(show_progress=False)` on sorted inputs) and confirm the
   same vocabulary is produced twice in a row.
5. **Held-out token coverage.** Run evaluation on a domain-shifted file
   (e.g. a wiki paragraph, a snippet of code). Expect the token/word
   ratio to rise sharply for both tokenizers; this is the diagnostic that
   motivates exercise-04's domain-adapted trainer.

## 4. Rubric or review checklist

| Criterion | Weight | Evidence |
|---|---|---|
| Both tokenizers trained with matched vocabulary budget and matched normaliser (or normalisers are reported) | 20% | Training scripts committed; CLI invocations captured in README |
| Evaluation harness computes all four metrics on a held-out set | 20% | `evaluate.py` output table included |
| Analysis identifies at least two *mechanism-level* differences (training objective, pre-tokeniser, continuation marker, normaliser) | 20% | Written analysis, not just numbers |
| Results reproducible from a fixed seed | 10% | README documents seed / sort order |
| Round-trip correctness called out separately from token/word ratio | 10% | A WordPiece model with lowercase normaliser is correctly flagged as lossy |
| Discussion of downstream implications (sequence length budget, OOV risk, cross-lingual behaviour) | 20% | Links to a concrete downstream task, not generic advice |

## 5. Common mistakes

- **Comparing with different normalisers and attributing the delta to
  the algorithm.** If WordPiece lowercases and Unigram does not, most of
  the OOV and round-trip difference is the normaliser, not the training
  objective. Hold normalisation constant or report it explicitly.
- **Using `character_coverage=1.0` on a corpus with rare CJK or emoji.**
  SentencePiece will reserve a slot for every unique character and
  starve the subword vocabulary. The documented default is 0.9995 for
  mixed corpora and 1.0 only for Latin-script corpora.
- **Comparing WordPiece `##foo` directly to SentencePiece `▁foo`.** They
  encode the same information (word-initial vs. continuation) with
  opposite conventions. The vocabulary overlap metric must strip these
  markers before comparing.
- **Measuring OOV by token count instead of character count.** A single
  `<unk>` can absorb an arbitrarily long span; counting tokens
  undercounts information loss. Use character coverage.
- **Treating "fewer tokens per word" as "better".** On tasks sensitive
  to morphology (NER, POS tagging), over-aggressive merging hurts
  downstream performance. The exercise's rubric credits analysis, not
  minimisation.

## 6. References

- Kudo, T. (2018). *Subword Regularization: Improving Neural Network
  Translation Models with Multiple Subword Candidates.* ACL 2018.
  arXiv:1804.10959. Introduces the Unigram LM tokenizer.
- Kudo, T. & Richardson, J. (2018). *SentencePiece: A simple and
  language independent subword tokenizer and detokenizer.* EMNLP demo.
  arXiv:1808.06226. Reference implementation + the `▁` convention.
- Schuster, M. & Nakajima, K. (2012). *Japanese and Korean Voice
  Search.* ICASSP 2012. Original WordPiece algorithm description.
- Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. (2019). *BERT:
  Pre-training of Deep Bidirectional Transformers for Language
  Understanding.* NAACL 2019. arXiv:1810.04805. WordPiece as used in
  practice, with the `##` convention.
- Hugging Face Tokenizers documentation — `models.WordPiece`,
  `WordPieceTrainer`, `pre_tokenizers.BertPreTokenizer`
  (`https://huggingface.co/docs/tokenizers`).
- SentencePiece documentation — options for normalisation, character
  coverage, and vocabulary budgeting
  (`https://github.com/google/sentencepiece`).
- NIST AI Risk Management Framework (AI RMF 1.0) — the Measure function
  requires documenting pre-processing (including tokenization) that can
  change evaluation outcomes
  (`https://www.nist.gov/itl/ai-risk-management-framework`).
