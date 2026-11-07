# exercise-01 — BPE From Scratch and Against Hugging Face `tokenizers`

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-101-tokenization-and-text-foundations/exercises/exercise-01-bpe-from-scratch-and-against-hf-tokenizers.md`.
> This solution targets the learning goals that filename implies: implement byte-pair encoding end-to-end, train it on a small corpus, and compare the learned vocabulary and segmentations against a Hugging Face `tokenizers`-trained BPE on the same corpus.

## 1. Solution overview

The exercise exercises three ideas at once:

1. **The BPE training loop** (Sennrich, Haddow, Birch 2016) — initialise a symbol vocabulary as characters, repeatedly merge the most frequent adjacent symbol pair, and record the merge list. Stop at a target vocabulary size.
2. **Deterministic, reproducible encoding** — given the merge list, segment a new string by applying merges in the order they were learned. This separation between *training* (statistics over a corpus) and *encoding* (rule application over one string) is the mental model to anchor on.
3. **Compatibility with a production library** — `tokenizers` (Hugging Face) ships a `BPE` model plus a `BpeTrainer`. A faithful from-scratch implementation should produce the same merges on the same input, modulo pre-tokenisation and byte-vs-char alphabet choices (GPT-2-style byte BPE uses a 256-entry byte alphabet rather than raw Unicode characters).

The deliverable is: a short Python module with a training function and an encoding function, a comparison harness that trains both tokenizers on the same corpus, and a written analysis of any differences (vocab overlap, token count on a held-out sample, longest divergent sequence).

## 2. Worked answer / implementation

### 2.1 Minimal BPE in pure Python

The reference implementation below is adapted from the pseudocode in the original BPE paper (Sennrich et al. 2016, §3.2). It treats a *word* as a space-delimited tuple of symbols terminated by `</w>`, matching the paper's convention for marking word boundaries.

```python
# bpe_from_scratch.py
from collections import Counter, defaultdict
from typing import Iterable

END = "</w>"


def _word_tuples(corpus: Iterable[str]) -> Counter:
    """Count whitespace-split words, each represented as a tuple of chars + end marker."""
    counts: Counter = Counter()
    for line in corpus:
        for word in line.split():
            counts[tuple(word) + (END,)] += 1
    return counts


def _pair_stats(vocab: Counter) -> Counter:
    pairs: Counter = Counter()
    for word, freq in vocab.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += freq
    return pairs


def _merge(vocab: Counter, pair: tuple[str, str]) -> Counter:
    merged = Counter()
    a, b = pair
    ab = a + b
    for word, freq in vocab.items():
        new_word, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(ab)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        merged[tuple(new_word)] = freq
    return merged


def train_bpe(corpus: Iterable[str], num_merges: int):
    """Return (merges, final_vocab). `merges` is an ordered list of pairs."""
    vocab = _word_tuples(corpus)
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        stats = _pair_stats(vocab)
        if not stats:
            break
        best = max(stats, key=lambda p: (stats[p], p))  # deterministic tiebreak
        merges.append(best)
        vocab = _merge(vocab, best)
    return merges, vocab


def encode(text: str, merges: list[tuple[str, str]]) -> list[str]:
    """Apply learned merges to a new string, in order."""
    rank = {pair: i for i, pair in enumerate(merges)}
    out: list[str] = []
    for word in text.split():
        symbols = list(word) + [END]
        while True:
            best_i, best_rank = -1, len(rank)
            for i in range(len(symbols) - 1):
                r = rank.get((symbols[i], symbols[i + 1]), len(rank))
                if r < best_rank:
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            symbols = (
                symbols[:best_i]
                + [symbols[best_i] + symbols[best_i + 1]]
                + symbols[best_i + 2 :]
            )
        out.extend(symbols)
    return out
```

Two deliberate choices worth calling out to a learner:

- **Deterministic tiebreak.** When two pairs are tied in frequency, sorting on `(freq, pair)` gives reproducible merges across runs. Without it, Python dict iteration order can leak non-determinism into the merge list.
- **Encoding picks the earliest learned merge each step.** That is how production BPE ranks pairs: merges earlier in the list win. This matches the BPE inference loop that libraries like `tokenizers` and `tiktoken` implement.

### 2.2 Training the Hugging Face baseline

The companion baseline uses `tokenizers` (Hugging Face's Rust-backed library). The Python API documented at <https://huggingface.co/docs/tokenizers> provides the `BPE` model and `BpeTrainer`.

```python
# hf_baseline.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_hf_bpe(corpus_path: str, vocab_size: int) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token="[UNK]", end_of_word_suffix="</w>"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]"],
        end_of_word_suffix="</w>",
        show_progress=False,
    )
    tok.train([corpus_path], trainer)
    return tok
```

`Whitespace` pre-tokenisation is the right choice when matching the from-scratch trainer above, which also splits on whitespace. The `end_of_word_suffix="</w>"` on both the `BPE` model and the `BpeTrainer` is what makes the comparison honest: the scratch implementation appends `</w>` to every training word and allows merges involving the marker, so the HF baseline must learn the same suffix or its symbol inventory and merge statistics will diverge by construction. Pick `ByteLevel` if you want GPT-2-style byte BPE instead — then drop the `end_of_word_suffix` on both sides, and switch the from-scratch implementation's alphabet to the 256-entry byte→unicode map that `tokenizers` uses (see `tokenizers.pre_tokenizers.ByteLevel.alphabet()`).

### 2.3 Comparison harness

```python
# compare.py
from bpe_from_scratch import train_bpe, encode
from hf_baseline import train_hf_bpe


def load_corpus(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def compare(corpus_path: str, num_merges: int, sample_lines: list[str]):
    corpus = load_corpus(corpus_path)

    # From-scratch
    merges, _ = train_bpe(corpus, num_merges)
    scratch_tokens = [encode(line, merges) for line in sample_lines]

    # HF baseline — vocab_size includes initial alphabet and specials,
    # so pass num_merges + alphabet_size + len(specials) for parity.
    initial_alphabet = sorted({c for line in corpus for c in line})
    hf_vocab_size = num_merges + len(initial_alphabet) + 1  # +1 for [UNK]
    hf = train_hf_bpe(corpus_path, hf_vocab_size)
    hf_tokens = [hf.encode(line).tokens for line in sample_lines]

    for line, s, h in zip(sample_lines, scratch_tokens, hf_tokens):
        print(line)
        print("  scratch:", s)
        print("  hf     :", h)
        print("  len diff:", len(s) - len(h))
```

### 2.4 Expected result

On a small clean corpus (a few thousand lines of ASCII English, whitespace pre-tokenisation on both sides, matching alphabet), the two tokenisers typically land within a few percent on token count per sentence, and the top merges (shortest, highest-frequency) agree verbatim. Divergence grows at the tail of the merge list where ties are broken differently, and anywhere the pre-tokeniser disagrees (punctuation handling, leading-space markers, byte vs. char alphabet).

A worked expectation for the written-up answer: report (a) merge-list Jaccard, (b) mean tokens-per-sentence on a held-out slice, (c) one or two concrete divergent sequences with a plausible explanation. See §5 for common traps.

## 3. Validation steps

1. **Unit-test the merge loop on a toy corpus** — the paper's `low lower newest widest` example should produce the merges `(e, s), (es, t), (l, o)…` in that order; a correct implementation reproduces this.
2. **Round-trip test** — `"".join(encode(x, merges)).replace(END, " ").rstrip()` should equal `" ".join(x.split())` for any whitespace-tokenised input. Join the pieces *without* spaces (they are subword pieces, not words), then let the end-of-word marker reconstitute the single inter-word space; `rstrip` drops the trailing space from the final word's marker. Joining with `" "` instead inserts whitespace *inside* words and will fail on even trivial inputs — e.g. after two merges on `"hello world"`, the pieces might be `["h", "e", "l", "l", "o</w>", "wor", "l", "d</w>"]`, so the space-join expression produces `"h e l l o   wor l d  "` rather than `"hello world"`.
3. **Determinism test** — train twice with the same seed and corpus, assert identical merge lists.
4. **Vocab-size parity** — confirm that `len(set(merges))` plus the initial alphabet plus specials equals the `vocab_size` you asked `BpeTrainer` for. If not, your parity math (§2.3) is off.
5. **Overlap report** — compute `len(set(my_merges) & set(hf_merges)) / len(set(my_merges) | set(hf_merges))` on the first `k` merges for `k ∈ {100, 500, num_merges}`. High-k overlap is what demonstrates parity.
6. **Divergence hunt** — run both tokenisers on a held-out paragraph; store every line where segmentations disagree; inspect the longest.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **BPE training loop** | Deterministic, correct merges on paper's toy example | Correct on toy but non-deterministic ties | Produces wrong merges or never terminates |
| **Encoding** | Picks earliest learned merge each step; round-trips | Greedy left-to-right (common bug, wrong for BPE) | No encoder, or hard-codes training vocab |
| **HF baseline parity** | Matches pre-tokeniser and alphabet; vocab-size math explained | Trained but parameters not explained | HF baseline ignored or copy-pasted from a tutorial |
| **Comparison** | ≥2 quantitative metrics + ≥1 qualitative example with explanation | Quantitative only, no explanation | Just "they look the same" with no numbers |
| **Reproducibility** | Fixed seed, pinned deps, one-command repro | Partially reproducible | Can't rerun from the writeup |
| **Writeup** | Explains divergences in terms of pre-tokeniser / byte alphabet / tiebreak | Describes differences but not causes | No analysis |

## 5. Common mistakes

- **Greedy-left-to-right encoding.** Scanning the string left-to-right and applying the first applicable merge is not BPE — it's a different algorithm and will diverge from `tokenizers` almost immediately. BPE encoding must repeatedly pick the *lowest-rank* (earliest-learned) applicable pair across the whole word.
- **Mismatched alphabet.** Comparing character BPE against GPT-2-style byte BPE will make the two disagree on every non-ASCII input. Pick one representation and keep both tokenisers on it.
- **Forgotten end-of-word marker.** Without `</w>` (or the equivalent `Ġ`/space marker in byte BPE) the tokeniser cannot distinguish `er` at the end of `lower` from `er` inside `erupt`, and merges leak across word boundaries.
- **`vocab_size` confusion.** `BpeTrainer`'s `vocab_size` counts the initial alphabet and special tokens, not just merges. If you ask for `vocab_size=1000` and your alphabet already has 96 characters, you get at most ~903 merges. Learners frequently report "my merge list is shorter than HF's" and the cause is this.
- **Non-deterministic tiebreak.** Letting `max(stats, key=stats.get)` pick the winner on ties silently varies across Python versions. Always provide a secondary key.
- **Over-fitting the comparison corpus.** A tiny corpus makes divergence look much larger than it will be at scale. Report on a held-out sample as well as training data.

## 6. References

Primary sources used in this solution:

- Sennrich, R., Haddow, B., Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. ACL. <https://aclanthology.org/P16-1162/> — introduces BPE for NMT; §3.2 contains the pseudocode this implementation follows.
- Hugging Face `tokenizers` documentation — <https://huggingface.co/docs/tokenizers> — authoritative reference for `BPE`, `BpeTrainer`, `Whitespace`/`ByteLevel` pre-tokenisers, and the byte→unicode alphabet used by GPT-2-style tokenisers.
- Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2 tech report) — the byte-level BPE variant. <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>

Context references listed on the work packet (not directly load-bearing for this exercise but linked for the module context):

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — <https://atlas.mitre.org/>
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework>
