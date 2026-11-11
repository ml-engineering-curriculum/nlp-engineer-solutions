# exercise-04 — Domain-Adapted Tokenizer Training

## 1. Solution overview

The exercise asks learners to train a domain-adapted tokenizer and
decide whether to **extend** an existing tokenizer's vocabulary or
**retrain** from scratch. The reference solution works through the
decision explicitly, then executes both paths and evaluates them on
domain text so that the trade-off is visible in numbers.

The worked domain is **biomedical abstracts** (a canonical case where
general-purpose tokenizers over-fragment terminology). The methodology
is domain-agnostic; learners should re-run the recipe on their own
corpus.

Three deliverables:

1. `train_from_scratch.py` — fresh SentencePiece Unigram on the domain
   corpus.
2. `extend_vocab.py` — take an existing BPE tokenizer and append
   high-value domain tokens using the corpus's own merge statistics.
3. `evaluate_domain.py` — measure compression and preservation of
   multi-word technical terms against a held-out domain evaluation set.

## 2. Worked answer or implementation

### 2.1 Decision rubric: extend vs. retrain

Use this table before training. It encodes the trade-offs that come
from the design of subword tokenizers and the realities of downstream
fine-tuning.

| Factor | Extend an existing tokenizer | Retrain from scratch |
|---|---|---|
| Will the base model be fine-tuned, not retrained? | Preferred — keeps most embeddings aligned | Not an option |
| Is domain vocabulary a small overlay (≤5k new tokens)? | Preferred | Overkill |
| Does the domain share script and language with the base? | Required for extension to work | Still fine |
| Is the domain mostly non-English or non-Latin script? | Risky — base merges fight the new vocabulary | Preferred |
| Do you need deterministic token IDs for evaluation parity with the base model? | Only extension preserves the existing IDs | — |
| Are you training a new model anyway? | — | Preferred — you have no legacy to protect |

When in doubt, extension is the lower-risk default because downstream
embeddings are the expensive artefact to replace.

### 2.2 Retrain from scratch

```python
# train_from_scratch.py
"""Train a domain-only SentencePiece Unigram tokenizer.

The script deliberately mirrors the settings from exercise-02 so that
differences between the general and domain models are attributable to
training data, not hyperparameters."""
from pathlib import Path

import sentencepiece as spm


def train(corpus: Path, out_prefix: str, vocab_size: int = 16000) -> None:
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=out_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        normalization_rule_name="nmt_nfkc_cf",
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    )


if __name__ == "__main__":
    import sys

    train(Path(sys.argv[1]), sys.argv[2])
```

### 2.3 Extend an existing tokenizer's vocabulary

```python
# extend_vocab.py
"""Extend a Hugging Face BPE tokenizer with domain tokens selected from
the corpus's own n-gram statistics.

The selection heuristic is: a candidate token is valuable if it is
*frequent* in the domain corpus and *under-represented* by the base
tokenizer (i.e. the base splits it into many sub-tokens). This matches
the practitioner approach used by biomedical BERT variants, which
report adding on the order of a few thousand domain tokens on top of a
general base.
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

from tokenizers import Tokenizer

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")


def word_counts(path: Path) -> collections.Counter[str]:
    counter: collections.Counter[str] = collections.Counter()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            counter.update(WORD_RE.findall(line))
    return counter


def fragmentation(tok: Tokenizer, word: str) -> int:
    """Number of sub-tokens the base tokenizer uses for ``word``."""
    return len(tok.encode(word, add_special_tokens=False).tokens)


def select_additions(
    counts: collections.Counter[str],
    tok: Tokenizer,
    *,
    min_count: int = 50,
    min_fragmentation: int = 3,
    budget: int = 3000,
) -> list[str]:
    scored: list[tuple[int, str]] = []
    for word, count in counts.items():
        if count < min_count:
            continue
        frag = fragmentation(tok, word)
        if frag < min_fragmentation:
            continue
        # value ≈ tokens saved per document occurrence × frequency
        scored.append(((frag - 1) * count, word))
    scored.sort(reverse=True)
    return [word for _, word in scored[:budget]]


def extend(base_tok_file: Path, corpus: Path, out_file: Path) -> list[str]:
    tok = Tokenizer.from_file(str(base_tok_file))
    counts = word_counts(corpus)
    additions = select_additions(counts, tok)
    tok.add_tokens(additions)      # appended to the end of the vocab
    tok.save(str(out_file))
    return additions


if __name__ == "__main__":
    import sys

    added = extend(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    for word in added[:50]:
        print(word)
    print(f"... {len(added)} tokens added")
```

A few important points about extension:

- `add_tokens` appends to the vocabulary without rewriting existing
  merges or IDs, so a model that was trained against the base
  tokenizer still produces consistent embeddings for existing IDs.
  **New IDs need new embedding rows**; the downstream fine-tuning
  script must resize the embedding matrix and initialise the new rows
  (common choice: average of the sub-tokens the base used to split the
  word into).
- Added tokens are *whole-word* tokens; they do not participate in BPE
  merges, so they are chosen by exact match. This means
  capitalisation, hyphenation, and casing must match the forms seen at
  inference.

### 2.4 Domain evaluation

```python
# evaluate_domain.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm
from tokenizers import Tokenizer


@dataclass
class DomainMetrics:
    name: str
    tokens: int
    words: int
    kept_terms: int
    probed_terms: int

    def ratio(self) -> float:
        return self.tokens / max(1, self.words)

    def term_preservation(self) -> float:
        return self.kept_terms / max(1, self.probed_terms)


def _tokens(
    tok_fn, word: str
) -> list[str]:  # tok_fn returns tokens for a single word
    return tok_fn(word)


def eval_tokenizer(
    name: str,
    tok_fn,
    eval_path: Path,
    terms_path: Path,
) -> DomainMetrics:
    words_total = 0
    tokens_total = 0
    with eval_path.open(encoding="utf-8") as fh:
        for line in fh:
            words = line.split()
            words_total += len(words)
            for w in words:
                tokens_total += len(_tokens(tok_fn, w))

    kept, probed = 0, 0
    with terms_path.open(encoding="utf-8") as fh:
        for term in fh:
            term = term.strip()
            if not term:
                continue
            probed += 1
            if len(_tokens(tok_fn, term)) == 1:
                kept += 1
    return DomainMetrics(
        name=name,
        tokens=tokens_total,
        words=words_total,
        kept_terms=kept,
        probed_terms=probed,
    )


def main(sp_path: Path, hf_path: Path, eval_path: Path, terms_path: Path) -> None:
    sp = spm.SentencePieceProcessor(model_file=str(sp_path))
    hf = Tokenizer.from_file(str(hf_path))
    metrics = [
        eval_tokenizer(
            sp_path.name, lambda w: sp.encode(w, out_type=str), eval_path, terms_path
        ),
        eval_tokenizer(
            hf_path.name,
            lambda w: hf.encode(w, add_special_tokens=False).tokens,
            eval_path,
            terms_path,
        ),
    ]
    print(json.dumps([m.__dict__ for m in metrics], indent=2))


if __name__ == "__main__":
    import sys

    main(*[Path(p) for p in sys.argv[1:5]])
```

### 2.5 What the comparison should show

- **Compression (tokens/word).** Domain-trained tokenizers should
  produce fewer tokens per word on domain text. Extension partially
  closes the gap; retraining usually closes more of it.
- **Term preservation.** On a hand-curated list of domain terms (e.g.
  gene names, chemical formulae, legal citations, log line signatures),
  the domain tokenizers should encode most as single tokens. The base
  tokenizer typically fragments them into 3–7 pieces.
- **Carry cost for retraining.** A fresh tokenizer invalidates every
  embedding row of the base model. If fine-tuning is the plan, the
  cost of retraining goes up by a full epoch of embedding re-learning.

<!-- research-note: specific tokens-saved metrics for named biomedical
tokenizers (e.g. BioBERT, PubMedBERT) are published in their respective
papers but should be re-verified before quoting in a solution. -->

## 3. Validation steps

1. **Corpus hygiene.** Deduplicate the training corpus (even near-
   duplicates matter — SentencePiece's Unigram objective is sensitive
   to repetition). Record corpus size in characters and words in the
   README.
2. **Train/eval split.** The evaluation corpus must come from a source
   or time period not seen in training. Otherwise the compression
   measurement is overly optimistic.
3. **Term list independence.** The curated term list used for the
   `term_preservation` metric must be sourced *before* training (e.g.
   from an external ontology) so the metric is not a proxy for the
   training data itself.
4. **Embedding init sanity (for extension).** After adding tokens,
   train a tiny model head for a few hundred steps and confirm the new
   token embeddings move. If they stay at the init value, the
   embedding resize wiring is broken.
5. **Backwards compatibility (for extension).** Encode a probe set of
   general-purpose sentences with the extended tokenizer and the base
   tokenizer; the token ID sequences for tokens that already existed
   must match.
6. **Reproducibility.** Record the tokenizer trainer version, the
   corpus checksum, and the seed. Unigram is deterministic given
   those; BPE extensions are deterministic given sort order.

## 4. Rubric or review checklist

| Criterion | Weight | Evidence |
|---|---|---|
| Learner states the extend-vs-retrain decision with rationale before training | 15% | Written decision section |
| Both options implemented with matched vocabulary budget (or budgets reported) | 20% | Scripts committed, parameters documented |
| Domain evaluation uses an externally sourced term list | 15% | README cites the term source |
| Results reported for compression and term preservation, not just one | 15% | Evaluation output |
| Discussion of downstream cost (embedding init, retraining, inference cache invalidation) | 15% | Written analysis |
| Reproducibility artefacts: corpus checksum, seed, trainer version | 10% | README |
| Acknowledges limitations (term list bias, corpus leakage, locale assumptions) | 10% | Written limitations section |

## 5. Common mistakes

- **Training on raw scraped text without dedup.** SentencePiece
  Unigram treats repeated sentences as independent evidence; a corpus
  with heavy near-duplicates learns a vocabulary dominated by whatever
  paragraphs happened to appear most often.
- **Treating `add_tokens` as sufficient.** It adds the string to the
  vocabulary but does not resize the model's embedding matrix. The
  downstream training script must call
  `model.resize_token_embeddings(len(tok))` (or equivalent) or new IDs
  index into uninitialised memory.
- **Measuring compression on in-domain training data.** Produces
  misleadingly good numbers. Always measure on held-out domain text.
- **Using a term list that was derived from the training corpus.**
  Guarantees high preservation by construction. Use an external
  ontology (UMLS, SNOMED, a legal citation style guide, Common Crawl
  n-grams depending on domain).
- **Forgetting that extension does not help cross-lingual coverage.**
  If the base tokenizer never saw Arabic, no amount of adding full-
  word tokens will give you reasonable sub-word fallback for unseen
  Arabic words. In that case, retrain or use a multilingual base.
- **Ignoring tokenization drift at inference.** Added tokens are
  exact-match. If the inference-time normaliser differs from the
  training-time normaliser, added tokens silently stop matching.

## 6. References

- Kudo, T. (2018). *Subword Regularization: Improving Neural Network
  Translation Models with Multiple Subword Candidates.* ACL 2018.
  arXiv:1804.10959.
- Kudo, T. & Richardson, J. (2018). *SentencePiece: A simple and
  language independent subword tokenizer and detokenizer.* EMNLP demo.
  arXiv:1808.06226.
- Hugging Face Tokenizers documentation — `Tokenizer.add_tokens`,
  `Tokenizer.from_file`
  (`https://huggingface.co/docs/tokenizers`).
- Hugging Face Transformers documentation —
  `PreTrainedModel.resize_token_embeddings`, which is the required
  companion to vocabulary extension
  (`https://huggingface.co/docs/transformers`).
- NIST AI Risk Management Framework (AI RMF 1.0) — the Govern function
  calls out data provenance and version tracking, which apply to
  training corpora for tokenizers as much as for models
  (`https://www.nist.gov/itl/ai-risk-management-framework`).
- OWASP Machine Learning Security Top 10
  (`https://owasp.org/www-project-machine-learning-security-top-10/`).
  Tokenizer training corpora are part of the "Data Poisoning Attack"
  surface; domain adaptation that pulls from untrusted sources
  inherits this risk.
