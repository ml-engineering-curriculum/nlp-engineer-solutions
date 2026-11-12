# exercise-02 — N-gram Language Model and Perplexity

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-102-classical-nlp/exercises/exercise-02-ngram-language-model-and-perplexity.md`.
> This solution targets the learning goals that filename implies: train an *n*-gram language model with at least one principled smoothing method, score a held-out corpus, and report perplexity — the standard intrinsic metric for language models.

## 1. Solution overview

An *n*-gram language model assigns probability to a sentence as a product of conditional probabilities over fixed-length contexts:

```
P(w_1 ... w_T) = Π_{t=1..T} P(w_t | w_{t-n+1} ... w_{t-1})
```

The two decisions that make or break the model are **context length** (`n`) and **smoothing** (how probability mass is redistributed to unseen *n*-grams).

- Maximum-likelihood estimates set `P(w|context) = count(context, w) / count(context)`. On any realistic corpus this assigns probability zero to roughly 20-40 % of held-out *n*-grams, which makes perplexity infinite. Smoothing fixes that.
- **Laplace (add-one)** is the simplest. It is almost always worse than **Kneser-Ney** on natural text (Chen & Goodman 1999 is the canonical empirical comparison), but it is simpler to implement and a reasonable teaching baseline.
- **Modified Kneser-Ney** (Chen & Goodman 1999, §4.3) remains the state of the art among count-based smoothers; it uses *continuation* counts (how many unique contexts a word completes) rather than raw counts, which specifically addresses the "San Francisco" problem: `Francisco` is frequent as a word but rare as a *continuation* of contexts, so KN gives it low backoff mass where additive smoothing would overestimate.

**Perplexity** on a held-out corpus of `N` tokens (including end-of-sentence markers) is:

```
PP(W) = exp(-(1/N) · Σ_t log P(w_t | context_t))
```

It is the *geometric mean inverse probability per token*: a model with perplexity 100 is as surprised per token on average as if it were choosing uniformly from 100 words. Lower is better. Perplexity is only comparable across models that tokenise the same way and count tokens the same way; a model that drops sentence boundaries will report an artificially low number.

The deliverable is a Python module that trains a smoothed *n*-gram LM, scores held-out text, and returns perplexity, plus a short report comparing at least two smoothing choices (e.g. Laplace vs. Kneser-Ney) and two context lengths (e.g. bigram vs. trigram).

## 2. Worked implementation

### 2.1 Project layout

```
ngram_lm/
  tokenise.py        # sentence splitting + <s>/</s> handling
  counts.py          # Counter-based n-gram tabulation
  smoothing.py       # Laplace, Kneser-Ney
  model.py           # NgramLM class with .train() / .logprob() / .perplexity()
  cli.py
  tests/
    test_counts.py
    test_smoothing.py
    test_perplexity.py
```

### 2.2 Tokenising with sentence markers

Every sentence gets `n-1` begin-of-sentence markers and one end-of-sentence marker. Without this, the bigram `P(w_1 | <s>)` is undefined and perplexity is not comparable across models (SLP3 §3.1).

```python
# tokenise.py
from typing import Iterable

BOS = "<s>"
EOS = "</s>"


def sentences(path: str) -> Iterable[list[str]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if toks:
                yield toks


def pad(tokens: list[str], n: int) -> list[str]:
    """n-1 BOS, then tokens, then one EOS."""
    return [BOS] * (n - 1) + tokens + [EOS]
```

### 2.3 Counting *n*-grams

```python
# counts.py
from collections import Counter
from typing import Iterable
from .tokenise import pad


def ngram_counts(sentences: Iterable[list[str]], n: int) -> tuple[Counter, Counter]:
    """Return (ngram_counts, context_counts) over padded sentences."""
    ngrams: Counter = Counter()
    contexts: Counter = Counter()
    for sent in sentences:
        padded = pad(sent, n)
        for i in range(len(padded) - n + 1):
            ng = tuple(padded[i : i + n])
            ngrams[ng] += 1
            contexts[ng[:-1]] += 1
    return ngrams, contexts
```

### 2.4 Smoothing

```python
# smoothing.py
import math
from collections import Counter, defaultdict
from typing import Protocol


class Smoother(Protocol):
    def logprob(self, context: tuple[str, ...], word: str) -> float: ...


class Laplace:
    """Add-one smoothing. Simple but weak; a didactic baseline."""
    def __init__(self, ngrams: Counter, contexts: Counter, vocab_size: int):
        self.ngrams = ngrams
        self.contexts = contexts
        self.V = vocab_size

    def logprob(self, context, word):
        c_ctx = self.contexts.get(context, 0)
        c_ng  = self.ngrams.get(context + (word,), 0)
        return math.log((c_ng + 1) / (c_ctx + self.V))


class KneserNey:
    """Interpolated Kneser-Ney with a single fixed discount.

    See Chen & Goodman 1999 §4.3. For teaching purposes we use a single discount
    `D` rather than the three-discount `modified Kneser-Ney` variant; the extra
    discounts improve perplexity but add bookkeeping.
    """
    def __init__(self, ngrams: Counter, contexts: Counter,
                 lower_ngrams: Counter | None, vocab: set[str], D: float = 0.75):
        self.ngrams = ngrams
        self.contexts = contexts
        self.D = D
        self.vocab = vocab
        # Continuation counts: N1+( • w ) = # of distinct contexts that precede w.
        cont: dict[str, int] = defaultdict(int)
        total_unique_bigrams = 0
        for ng in ngrams:
            cont[ng[-1]] += 1
            total_unique_bigrams += 1
        self.cont = cont
        self.total_unique = max(total_unique_bigrams, 1)
        # Number of distinct words seen following each context, for backoff weight.
        seen_after: dict[tuple, int] = defaultdict(int)
        for ng in ngrams:
            seen_after[ng[:-1]] += 1
        self.seen_after = seen_after

    def _pcont(self, word: str) -> float:
        """Continuation probability for the lowest-order model."""
        return self.cont.get(word, 0) / self.total_unique

    def logprob(self, context, word):
        c_ctx = self.contexts.get(context, 0)
        if c_ctx == 0:
            return math.log(max(self._pcont(word), 1e-12))
        c_ng = self.ngrams.get(context + (word,), 0)
        discounted = max(c_ng - self.D, 0.0) / c_ctx
        lam = (self.D / c_ctx) * self.seen_after.get(context, 0)
        p = discounted + lam * self._pcont(word)
        return math.log(max(p, 1e-12))
```

A pragmatic note: the `1e-12` floor exists because a word unseen in training and unseen as any *continuation* will still receive zero continuation probability. Production KN implementations fall back to a uniform-over-vocabulary term for that case; the floor above is the teaching shortcut.

### 2.5 Model

```python
# model.py
import math
from collections import Counter
from .counts import ngram_counts
from .tokenise import pad, BOS, EOS
from .smoothing import Laplace, KneserNey


class NgramLM:
    def __init__(self, n: int, smoother: str = "kn", D: float = 0.75):
        if n < 2:
            raise ValueError("n must be >= 2; unigrams are a degenerate case")
        self.n = n
        self.smoother_name = smoother
        self.D = D

    def train(self, sentences):
        sents = list(sentences)
        self.vocab = {t for s in sents for t in s} | {BOS, EOS}
        self.ngrams, self.contexts = ngram_counts(sents, self.n)
        if self.smoother_name == "laplace":
            self.sm = Laplace(self.ngrams, self.contexts, len(self.vocab))
        elif self.smoother_name == "kn":
            self.sm = KneserNey(self.ngrams, self.contexts, None, self.vocab, self.D)
        else:
            raise ValueError(self.smoother_name)
        return self

    def logprob_sentence(self, sent: list[str]) -> tuple[float, int]:
        """Return (sum log P, number of scored tokens)."""
        padded = pad(sent, self.n)
        total = 0.0
        scored = 0
        for i in range(self.n - 1, len(padded)):
            ctx = tuple(padded[i - self.n + 1 : i])
            total += self.sm.logprob(ctx, padded[i])
            scored += 1  # every token after the initial BOS padding is scored
        return total, scored

    def perplexity(self, sentences) -> float:
        total_logp, total_tok = 0.0, 0
        for s in sentences:
            lp, n = self.logprob_sentence(s)
            total_logp += lp
            total_tok += n
        if total_tok == 0:
            return float("inf")
        return math.exp(-total_logp / total_tok)
```

The scored-token count includes `</s>` but not the leading `<s>` padding — this matches the SLP3 convention and makes perplexity comparable across models with different `n`.

### 2.6 End-to-end example

```python
from ngram_lm.model import NgramLM
from ngram_lm.tokenise import sentences

train = list(sentences("data/wikitext-103/train.tokens"))
dev   = list(sentences("data/wikitext-103/valid.tokens"))

for n in (2, 3):
    for sm in ("laplace", "kn"):
        lm = NgramLM(n=n, smoother=sm).train(train)
        print(f"n={n} smoother={sm:7s} ppl={lm.perplexity(dev):.2f}")
```

On WikiText-103 you should see Kneser-Ney beat Laplace by a wide margin (typically 2-5×) at every `n`, and trigram beat bigram by a modest margin at each smoother. The exact numbers depend on tokenisation; do not report numbers you did not measure.

## 3. Validation steps

1. **Probabilities sum to one per context.** For a sample of 100 observed contexts, compute `sum(exp(logprob(ctx, w)) for w in vocab)` and assert it is within `1e-6` of 1.0 (Laplace) or within `1e-3` of 1.0 (Kneser-Ney with the floor). KN without the uniform fallback will undershoot slightly for contexts with no continuation mass; log the residual and fail loud if it exceeds a threshold.
2. **Monotone in training data.** Train on the first 10 %, 50 %, 100 % of the corpus and confirm dev perplexity decreases monotonically. A non-monotone curve points at a bookkeeping bug in the smoother.
3. **Baseline comparison.** Report Laplace vs Kneser-Ney on the same dev split, same tokeniser, same `n`. KN should beat Laplace by a substantial margin on any natural-text corpus (Chen & Goodman 1999 Table 4, replicated across many corpora).
4. **Sanity-check against a toolkit.** Train a KN model on the same data with KenLM (<https://github.com/kpu/kenlm>). The perplexities will not match exactly — tokenisation and the modified-KN variant differ — but they should land within a factor of 2. A factor-10 gap is a bug on your side.
5. **Held-out vs test split.** Tune `D` on dev, report final number on a *separate* held-out test. Reporting perplexity on the split you used to pick `D` leaks and inflates the apparent quality.
6. **Vocabulary closure.** Every token seen in the dev split should either be in the training vocab or be mapped to `<unk>` *before* perplexity is computed. Many beginner implementations silently drop OOV tokens, which cuts the denominator and lowers reported perplexity dishonestly.
7. **Reproducibility.** Pin tokenisation (tool + version), the random seed if any subsampling occurs, and the exact training/dev/test files. Perplexity numbers reported without this metadata are meaningless.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Padding / sentence boundaries** | `<s>` × (n-1) and `</s>` on every sentence; scored-token count defined and matches SLP3 convention | Padding present but convention undocumented | No padding; cross-sentence *n*-grams; comparable-across-*n* broken |
| **Smoothing** | At least one of Laplace or KN implemented correctly + at least one reference check against a toolkit or published number | Laplace only | MLE with no smoothing; perplexity infinite on first OOV |
| **Perplexity formula** | Correct denominator (total scored tokens); log-sum-exp done in log-space to avoid underflow | Correct formula, numerical issues at long sentences | Uses product-of-probabilities in float; underflows on >50-token sentences |
| **Comparison report** | ≥2 smoothers × ≥2 `n` values on shared dev split with pinned tokeniser | Reports one grid point | Just one number, no baselines |
| **Hyperparameter hygiene** | Tunes `D` (or `α` for Laplace) on dev, reports on held-out test | Tunes and tests on same split | No tuning; defaults unexplained |
| **OOV handling** | Explicit `<unk>` policy, applied before scoring | Mentions OOV | Silently drops OOV tokens, denominators mis-counted |
| **Numerical safety** | All computation in log-space; floors or fallbacks for zero-probability events | Mostly log-space | Float products; `inf` perplexities glossed |
| **Reproducibility** | Pinned tokeniser, data splits, seeds; one-command rerun | Partial pinning | Not rerunnable |
| **Responsible deployment context** | Notes licence of training corpus; data-provenance per NIST AI RMF (Map); ATLAS/OWASP considerations when LM output is user-visible | Mentions one | Ignored |

## 5. Common mistakes

- **Float underflow.** Multiplying 300 probabilities in float64 underflows to 0 well before the end of a paragraph, so perplexity jumps to infinity. Sum log-probabilities instead; never compute `math.prod(probs)`.
- **Scoring padding tokens.** Including every `<s>` in the token count inflates the denominator and underreports perplexity. SLP3's convention is to score tokens 1..T and `</s>`, not the leading BOS padding; stick to it so your numbers compare to published ones.
- **Reporting per-sentence perplexity.** Perplexity is corpus-level (geometric mean per *token*), not an average of per-sentence perplexities. The latter over-weights short sentences and is not comparable across models.
- **Cross-sentence *n*-grams.** Sliding an *n*-window across sentence boundaries without `</s>` / `<s>` bookends leaks context and conflates "end of sentence" with "beginning of next sentence". It also makes perplexity across different `n` incomparable.
- **Tuning `D` on the test split.** The classic leak. Keep train/dev/test separate; report only on test.
- **OOV silently dropped.** If `dev.tokens` contains words that never appear in `train`, and your scoring loop skips them, your perplexity denominator is smaller than you think and the number you publish is not what you computed on the full split. Map OOV to `<unk>` (with a known training-time `<unk>` probability) before the loop runs.
- **Kneser-Ney on raw counts instead of continuation counts.** A KN implementation that forgets to compute `N1+(• w)` is just interpolated absolute discounting — still better than Laplace, but not KN. The whole point is the *continuation* count.
- **Reporting perplexity without tokenisation metadata.** A perplexity of 70 with word-level tokenisation, a perplexity of 70 with BPE tokenisation, and a perplexity of 70 after you dropped punctuation are three different numbers. State the tokeniser, the vocab size, and the splits every time.
- **Using `<unk>` in training but not in dev.** If the training-time `<unk>` probability is not applied to dev OOVs, dev scoring is undefined for those positions. Apply the same `<unk>` substitution to both.

## 6. References

Primary sources used in this solution:

- Jurafsky, D., Martin, J. (2024). *Speech and Language Processing*, 3rd ed. draft, ch. 3 (*N-gram Language Models*). <https://web.stanford.edu/~jurafsky/slp3/3.pdf> — the standard textbook treatment of counts, smoothing, and perplexity.
- Chen, S.F., Goodman, J. (1999). *An Empirical Study of Smoothing Techniques for Language Modeling*. *Computer Speech & Language* 13(4). <https://www.cs.cmu.edu/~roni/11761/PreviousYearsHandouts/chen-goodman-99.pdf> — the standard empirical reference for Kneser-Ney and modified Kneser-Ney, including the discount schedule in §4.3.
- Heafield, K. (2011). *KenLM: Faster and Smaller Language Model Queries*. WMT. <https://aclanthology.org/W11-2123/> — reference implementation of modified Kneser-Ney for toolkit-level validation of perplexity.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — "ML03 Model Inversion" and "ML06 AI Supply Chain" apply when n-gram LMs are trained on sensitive or licensed corpora.
- MITRE ATLAS — <https://atlas.mitre.org/> — training-data provenance and membership-inference techniques map onto any statistical LM whose training set is not fully public.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — Map / Measure functions require perplexity to be reported with tokeniser, split, and corpus metadata, not as a bare number.
