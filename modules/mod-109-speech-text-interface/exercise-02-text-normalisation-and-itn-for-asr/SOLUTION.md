# exercise-02 — Text Normalisation and ITN for ASR

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-109-speech-text-interface/exercises/exercise-02-text-normalisation-and-itn-for-asr.md`.
> The target learning outcomes implied by that filename: build the two halves of the ASR text-normalisation stack — **TN (text → spoken form)** used for TTS front-ends and training-label cleanup, and **ITN (spoken form → written form)** used to turn an ASR transcript like `"twenty five dollars"` into the display string `"$25"` — pick between a deterministic **WFST** approach (NeMo text-processing, Sparrowhawk) and a seq2seq approach, and **score both WER-with-normaliser and a separate WER-no-norm** so the effect of the ITN stage is isolated.

## 1. Solution overview

Text normalisation sits on both sides of an ASR system and is almost always the quiet reason "the WER is fine but the output looks terrible" (or the louder reason "we agreed on the same reference and still can't agree on WER"):

- **TN (verbaliser / text→spoken).** Needed to build TTS training data and to normalise references before WER. "He paid $25 on 3/4/2024" → "he paid twenty five dollars on march fourth twenty twenty four". Deterministic, idempotent, language-specific.
- **ITN (inverse / spoken→written).** Needed to make ASR output display-usable. "he paid twenty five dollars on march fourth" → "He paid $25 on March 4th". Orders of magnitude more ambiguous than TN because the ASR text has lost the formatting signal.
- **The Whisper normaliser.** `EnglishTextNormalizer` is a *scoring* normaliser, not a display normaliser: it is TN-adjacent (it fold numbers, strips casing, strips punctuation) and exists to make WER comparable across systems. It is not what you ship to a product UI.

Three subtleties define the design space:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **One-pass regexes on free text** | Hand-written regex normalisers compose badly: a rule that turns "25" → "twenty five" fires inside tokens ("H2O" → "H twenty five O"), inside URLs, inside already-normalised output | Use a tagger→classify→verbaliser pipeline (Sparrowhawk / NeMo structure): first tag spans by semiotic class (`DATE`, `MONEY`, `CARDINAL`, `MEASURE`, `PLAIN`, `VERBATIM`), *then* apply a class-specific grammar inside each span |
| **Non-idempotent normalisation** | Running TN twice changes the result; WER depends on how many times the pipeline was run. In log analysis this is a nightmare | Enforce idempotence by test: `TN(TN(x)) == TN(x)` on a corpus; add a `PLAIN` class that is preserved verbatim |
| **Mixing scoring and display ITN** | Using the Whisper scoring normaliser to generate the user-facing transcript strips the casing and punctuation users expect ("He paid $25." → "he paid 25 dollars") and routinely ends up in production | Keep two pipelines: **display** (full ITN, cased, punctuated) and **scoring** (Whisper `EnglishTextNormalizer` / `BasicTextNormalizer`); never substitute one for the other |

The deliverable is a bidirectional normalisation harness with four components:

1. **Classifier / tagger** — a semiotic-class tagger that labels spans as `DATE`, `TIME`, `MONEY`, `MEASURE`, `CARDINAL`, `ORDINAL`, `TELEPHONE`, `ADDRESS`, `URL`, `EMAIL`, `PUNCT`, `VERBATIM`, `PLAIN`.
2. **TN verbaliser** — per-class rules that turn each tagged span into spoken form.
3. **ITN verbaliser** — per-class rules that turn each tagged span (on the ASR side) into display form.
4. **Scoring normaliser** — Whisper's `EnglishTextNormalizer` (English) or `BasicTextNormalizer` (other languages), pinned to a specific `openai-whisper` version, with the pre/post-normalisation WERs both reported.

Reference implementations:
- **NVIDIA NeMo text processing (`nemo_text_processing`)** — production-grade WFST-based TN/ITN for 20+ languages; used in Riva TTS/ASR. The reference open-source implementation; built on Pynini.
- **Sparrowhawk** (Google 2017) — the WFST tagger/classifier/verbaliser *schema* NeMo mirrors; original paper defines the semiotic-class taxonomy used above.
- **Whisper `normalizers/`** — <https://github.com/openai/whisper/tree/main/whisper/normalizers> — the `EnglishTextNormalizer` and `BasicTextNormalizer` used for WER reporting.
- **Hugging Face `transformers` seq2seq ITN** — fine-tune a small T5/ByT5 to go from spoken→written, used as the "model" track in the worked example.

## 2. Worked implementation

### 2.1 Project layout

```
norm_bake/
  classes.py          # semiotic-class enum
  tagger.py           # rule + optional ML tagger → class spans
  tn.py               # text → spoken verbalisers per class
  itn.py              # spoken → text verbalisers per class
  whisper_norm.py     # thin wrapper on Whisper's EnglishTextNormalizer
  run.py              # CLI: --direction=tn|itn|score
  tests/
    test_idempotence.py
    test_classes.py
    test_wer_delta.py
  requirements.txt
```

### 2.2 Semiotic classes — Sparrowhawk's taxonomy

```python
# classes.py
from enum import Enum

class SemClass(str, Enum):
    DATE      = "DATE"       # "3/4/2024", "March 4th", "4 March 2024"
    TIME      = "TIME"       # "3:00 pm", "15:00", "quarter past three"
    MONEY     = "MONEY"      # "$25", "25 dollars", "twenty-five euros"
    MEASURE   = "MEASURE"    # "10 km", "25 kg", "100 mph"
    CARDINAL  = "CARDINAL"   # "42", "forty-two"
    ORDINAL   = "ORDINAL"    # "1st", "first"
    TELEPHONE = "TELEPHONE"  # "+1 (555) 123-4567"
    ADDRESS   = "ADDRESS"    # "221B Baker Street"
    URL       = "URL"
    EMAIL     = "EMAIL"
    PUNCT     = "PUNCT"      # standalone punctuation
    VERBATIM  = "VERBATIM"   # "H2O", "iPhone", keep literally
    PLAIN     = "PLAIN"      # everything else — DO NOTHING
```

This is Ebden & Sproat's 2015 Sparrowhawk / RNN Text Normalisation taxonomy, which NeMo and most production pipelines extend. The `PLAIN` class is the safety valve: anything not matched by the tagger passes through untouched.

### 2.3 Tagger — rules first, model second

```python
# tagger.py
import re
from dataclasses import dataclass
from .classes import SemClass

@dataclass
class Span:
    start: int
    end: int
    cls: SemClass
    text: str

# Order matters: longer / more specific patterns first.
_PATTERNS: list[tuple[SemClass, re.Pattern]] = [
    (SemClass.URL,       re.compile(r"\bhttps?://\S+\b", re.IGNORECASE)),
    (SemClass.EMAIL,     re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    (SemClass.TELEPHONE, re.compile(r"\+?\d[\d\s().-]{7,}\d")),
    (SemClass.MONEY,     re.compile(r"[$£€¥]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:dollars|euros|pounds|yen)\b", re.IGNORECASE)),
    (SemClass.DATE,      re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s\d{4}\b", re.IGNORECASE)),
    (SemClass.TIME,      re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:am|pm)?\b", re.IGNORECASE)),
    (SemClass.MEASURE,   re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(km|kg|lb|mi|mph|cm|mm|ml|l|g|°C|°F|%)\b")),
    (SemClass.ORDINAL,   re.compile(r"\b\d+(?:st|nd|rd|th)\b")),
    (SemClass.CARDINAL,  re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")),
]

def tag(text: str) -> list[Span]:
    spans: list[Span] = []
    taken = [False] * len(text)
    for cls, pat in _PATTERNS:
        for m in pat.finditer(text):
            if any(taken[m.start():m.end()]):
                continue   # already claimed by a more specific class
            spans.append(Span(m.start(), m.end(), cls, m.group(0)))
            for i in range(m.start(), m.end()):
                taken[i] = True
    spans.sort(key=lambda s: s.start)
    return spans
```

- **Order the patterns by specificity.** URL before EMAIL before TELEPHONE before DATE before MONEY before ORDINAL before CARDINAL. Otherwise CARDINAL eats the digit halves of dates.
- **`taken` mask prevents overlap.** A single MONEY span `$25.99` must not also be tagged as three CARDINALs.
- **`PLAIN` is implicit.** Anything not tagged stays as-is. This is what makes the whole system idempotent.

A seq2seq tagger (fine-tuned ByT5-small on the Google Text Normalization Challenge data, Sproat & Jaitly 2016) will outperform regex on messy inputs ("1k", "₹1.5 cr"), but the regex gets you to ~95 % of production coverage and is debuggable. Ship rules first; add the model as a fallback for `UNK` spans.

### 2.4 TN verbalisers — text → spoken

```python
# tn.py
import inflect
from datetime import datetime
from .classes import SemClass

_p = inflect.engine()
_CURRENCY = {"$": "dollars", "£": "pounds", "€": "euros", "¥": "yen"}
_MONTHS = ["january","february","march","april","may","june",
           "july","august","september","october","november","december"]


def verbalise_cardinal(text: str) -> str:
    n = int(text.replace(",", ""))
    return _p.number_to_words(n).replace("-", " ")


def verbalise_ordinal(text: str) -> str:
    n = int(text.rstrip("stndrh"))
    return _p.ordinal(_p.number_to_words(n)).replace("-", " ")


def verbalise_money(text: str) -> str:
    text = text.strip()
    if text[0] in _CURRENCY:
        amount, unit = text[1:].strip(), _CURRENCY[text[0]]
    else:
        amount, unit = text.rsplit(" ", 1)
        unit = unit.lower()
    if "." in amount:
        whole, cents = amount.split(".")
        return (
            f"{verbalise_cardinal(whole)} {unit} and "
            f"{verbalise_cardinal(cents)} cents"
        )
    return f"{verbalise_cardinal(amount)} {unit}"


def verbalise_date(text: str) -> str:
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%d %B %Y", "%d %b %Y"):
        try:
            d = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue
    else:
        return text
    month = _MONTHS[d.month - 1]
    day = verbalise_ordinal(f"{d.day}th")
    year = verbalise_cardinal(str(d.year))
    return f"{month} {day} {year}"


VERBALISERS: dict[SemClass, callable] = {
    SemClass.CARDINAL: verbalise_cardinal,
    SemClass.ORDINAL:  verbalise_ordinal,
    SemClass.MONEY:    verbalise_money,
    SemClass.DATE:     verbalise_date,
    # TIME, MEASURE, TELEPHONE, ADDRESS: left as exercises; same pattern.
}


def tn(text: str, spans) -> str:
    out, cursor = [], 0
    for sp in spans:
        out.append(text[cursor:sp.start])
        vb = VERBALISERS.get(sp.cls)
        out.append(vb(sp.text) if vb else sp.text)
        cursor = sp.end
    out.append(text[cursor:])
    return " ".join("".join(out).split())
```

- **`inflect` handles irregulars.** Hand-rolling "twenty-first" vs "twenty first" vs "21st" is a well-known tar pit; use a library.
- **Months as strings, ordinals for day.** "March fourth twenty twenty four" is how English news TTS reads dates. "Four March twenty twenty four" is the UK-ish alternative; pick one and document it.
- **Punctuation becomes whitespace in TN output.** TTS training data is tokenised on whitespace; keeping punctuation in TN output breaks downstream.

### 2.5 ITN verbalisers — spoken → display (the harder direction)

```python
# itn.py
# ITN turns "twenty five dollars" into "$25". The hard part is tagging the
# spans, because the ASR output has no formatting signal.
import re
from word2number import w2n   # pip install word2number
from .classes import SemClass

_WORD_NUMBERS = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|and|-| )+"
_PATTERNS = [
    (SemClass.MONEY,   re.compile(rf"\b({_WORD_NUMBERS})\s+(dollars|euros|pounds|yen)\b", re.IGNORECASE)),
    (SemClass.CARDINAL, re.compile(rf"\b{_WORD_NUMBERS}\b", re.IGNORECASE)),
    # dates, time, measures follow the same recipe
]


def itn(text: str) -> str:
    def _money(m: re.Match) -> str:
        n_words, unit = m.group(1).strip(), m.group(2).lower()
        try:
            n = w2n.word_to_num(n_words)
        except ValueError:
            return m.group(0)
        sym = {"dollars": "$", "euros": "€", "pounds": "£", "yen": "¥"}[unit]
        return f"{sym}{n}"

    def _cardinal(m: re.Match) -> str:
        try:
            return str(w2n.word_to_num(m.group(0)))
        except ValueError:
            return m.group(0)

    text = _PATTERNS[0][1].sub(_money, text)
    text = _PATTERNS[1][1].sub(_cardinal, text)
    return text
```

Three design decisions this bakes in:

- **Punctuation restoration is a separate stage.** ITN here does not add commas, periods, or capitalisation; exercise-03 covers that. Keep the stages decoupled.
- **Word-number parsing via `word2number` (not regex arithmetic).** The library handles "twenty five", "twenty-five", "a hundred and five", and "two thousand and six". Hand-rolled versions miss conjunctions and hyphenation and silently corrupt numbers.
- **Fail soft.** If `word_to_num` raises on an ambiguous span ("one two three" — three cardinals or a phone number?), return the original text. Never corrupt input you can't confidently rewrite.

### 2.6 Scoring normaliser — the Whisper protocol

```python
# whisper_norm.py
from whisper.normalizers import EnglishTextNormalizer, BasicTextNormalizer


def scoring_normaliser(language: str = "en"):
    """
    The normaliser used ONLY to compute WER. Never ship its output to a user:
    it strips casing, punctuation, and contractions.

    English: Whisper's EnglishTextNormalizer.
    Other:   Whisper's BasicTextNormalizer (unicode-based, case-fold + symbol strip).
    """
    return EnglishTextNormalizer() if language == "en" else BasicTextNormalizer()
```

- **Two pipelines, not one.** `display = casing + punctuation + ITN`; `scoring = Whisper normaliser`. Confusing them is the single most common source of "our ASR is worse than Whisper paper" and "our transcripts look bad".
- **Report both pre- and post-normalisation WER.** `WER_raw = jiwer.wer(ref, hyp)`; `WER_norm = jiwer.wer(norm(ref), norm(hyp))`. The delta is how much of your error is actually a formatting disagreement.
- **Pin the `openai-whisper` version.** `EnglishTextNormalizer` has changed between `openai-whisper==20231117` and later releases; a pipeline that didn't record the version can't defend its WER to a reviewer.

## 3. Validation steps

1. **Class-tagger coverage.** Run `tag()` on a fixed 200-utterance regression corpus. Assert every known span is tagged with the right class and no overlap. Add new patterns only after writing a failing test.
2. **TN idempotence.** `assert tn(tag(tn(tag(x)))) == tn(tag(x))` across the regression corpus. Double-TN output must be stable. If this fails the TN rules are emitting substrings that re-match patterns ("twenty five" → "twenty five" that then accidentally retags as `CARDINAL`).
3. **ITN identity on already-formatted text.** `itn("He paid $25")` must return `"He paid $25"` unchanged. ITN must only fire on spoken-form spans; if it mangles already-written numbers, the pattern has no word-boundary anchor.
4. **Round-trip sanity.** `tn("$25")` → `"twenty five dollars"`. `itn("twenty five dollars")` → `"$25"`. Identical in → out. This should hold for every class in the corpus.
5. **Non-idempotent ITN.** ITN *is not* idempotent in general ("twenty five" is a valid cardinal; the ITN output "25" is not a cardinal). Document this and run ITN exactly once.
6. **WER before vs after Whisper normaliser on a known fixture.** `ref="He said 25 dollars"`, `hyp="he said twenty five dollars"` → raw WER > 0 (case + number form), normalised WER ≈ 0. Pin this.
7. **Normaliser version drift.** Record `openai-whisper.__version__` in the output JSON; CI fails if it changes without a re-baseline.
8. **ITN latency budget.** On a 1 k-line batch, ITN should complete in <200 ms (pure-Python regex). If it takes seconds, the patterns are backtracking (nested quantifiers); profile with `re.DEBUG`.
9. **Language escape.** On a non-English language, `scoring_normaliser("fr")` must return `BasicTextNormalizer`. If an English normaliser leaks into French ASR scoring, the French number spell-out does not cancel and WER is systematically lower than reality.
10. **Known-edge fixtures.** Include at least one each of: ambiguous cardinal in a phone context ("five five five one two three four"), inline unit ("H2O"), currency without symbol ("25 USD"), mixed case brand ("iPhone 15 Pro"). Hand-grade each; use as regression gates.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Class taxonomy** | Named semiotic classes (Sparrowhawk / NeMo style); `PLAIN` fallback | Ad-hoc classes | Monolithic "normalise()" regex pile |
| **Tagger / verbaliser separation** | Tagger emits spans + class; verbalisers per class | Separation partial | Rules rewrite in one pass |
| **TN uses a number library** | `inflect` or NeMo for cardinal/ordinal | Hand-rolled with gaps | Regex arithmetic |
| **ITN uses `word2number` or equivalent** | ASR words → int done by a library; soft-fail on ambiguity | Hand-rolled | Numbers corrupted silently |
| **Idempotence of TN** | `tn(tn(x)) == tn(x)` passes on the corpus | Fails on a few classes | Not tested |
| **Scoring vs display separation** | Two distinct pipelines with named entry points | One pipeline with a flag | Whisper normaliser used as display |
| **WER reporting** | Both raw and normalised WER reported; normaliser class name + `openai-whisper` version recorded | One of the two | Only post-normalisation |
| **Known-edge fixtures** | Regression suite with tagged spans + expected output | Some edges covered | None |
| **Language coverage** | English TN/ITN + at least a stub for one other language; `BasicTextNormalizer` for non-English scoring | English only | English only + English normaliser used for all languages |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks noted (ITN as prompt-injection surface, PII exposure in normalised logs) with mitigations | Mentioned only | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure applied: locale coverage, idempotence proof, normaliser version, known classes supported, known failure modes | Some of those | None |

## 5. Common mistakes

- **One monolithic `normalise()` regex pile.** Rules fire in unintended order, you can't tell which rule caused a bug, and you can't add a new class without breaking three existing ones. The tagger→verbaliser split is the tested-in-production structure NeMo and Sparrowhawk converge on for exactly this reason.
- **Using the Whisper `EnglishTextNormalizer` as a display normaliser.** It case-folds, strips contractions ("don't" → "do not"), and is designed to be aggressive — a scoring normaliser's whole point is to collapse legitimate variation so WER is stable. Ship its output to a user and the user will see "he said twenty five dollars he said he d go to new york".
- **Wrong tagger order.** CARDINAL before DATE eats the digits inside dates. MONEY after CARDINAL means "25 dollars" gets tagged as a `CARDINAL` span followed by a word. Order is specificity-first, generic-last.
- **No idempotence test.** TN that runs twice produces a different result is a time-bomb for logs, metrics, and reproducibility. `tn(tn(x)) == tn(x)` on a corpus is a one-line test; skipping it is a false economy.
- **ITN that rewrites already-formatted numbers.** Without word-boundary anchors, `itn("page 25")` tries to interpret "page" as a number-adjacent token and corrupts the input. Always anchor on `\b` and only on spans that match word-number patterns.
- **Blind confidence on ambiguous spans.** "five five five one two three four" could be seven cardinals or a phone number. ITN must soft-fail (return input unchanged) when the pattern is ambiguous; never silently produce a wrong-but-plausible rewrite.
- **Not recording the Whisper normaliser version.** `EnglishTextNormalizer`'s behaviour on edge cases has changed across releases. A WER number without `openai-whisper.__version__` is not reproducible.
- **Treating ITN as a replacement for punctuation restoration.** ITN ≠ cased + punctuated output. Casing and punctuation are the next stage (exercise-03). Trying to do both in ITN produces a fragile mega-regex.
- **English-only `BasicTextNormalizer`.** `BasicTextNormalizer` is the Whisper paper's non-English scoring default; it is not language-specific and should be used *for all non-English* scoring. If your pipeline silently applies `EnglishTextNormalizer` to Spanish ASR, you're measuring something that nobody else measures.
- **Regex backtracking on long text.** Patterns with nested `.*` on paragraph-long inputs can go exponential. Keep the quantifiers greedy-bounded or use `regex` with `fuzzy_match` off; benchmark with `timeit` on a realistic input length.

## 6. References

Primary sources used in this solution:

- Sproat, R., Jaitly, N. (2016). *RNN Approaches to Text Normalization: A Challenge*. <https://arxiv.org/abs/1611.00068> — the Google Text Normalization Challenge; defines the semiotic-class taxonomy (`DATE`, `MONEY`, `MEASURE`, etc.) used in §2.2.
- Ebden, P., Sproat, R. (2015). *The Kestrel TTS text normalization system*. Natural Language Engineering. <https://doi.org/10.1017/S1351324914000175> — the Kestrel/Sparrowhawk tagger/classifier/verbaliser architecture mirrored in NeMo.
- Zhang, Y. et al. (2021). *NeMo Inverse Text Normalization: From Development to Production*. INTERSPEECH. <https://arxiv.org/abs/2104.05055> — the NeMo ITN paper; WFST grammars + classifier + verbaliser schema for 20+ languages.
- NVIDIA NeMo text processing — <https://github.com/NVIDIA/NeMo-text-processing> — the reference open-source TN / ITN implementation; Pynini-based WFSTs.
- Sparrowhawk — <https://github.com/google/sparrowhawk> — Google's open-source WFST runtime that NeMo text processing is built on; the architectural blueprint for §2.3.
- Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). <https://arxiv.org/abs/2212.04356> — §4.4 defines the `EnglishTextNormalizer` and `BasicTextNormalizer` used for WER reporting.
- OpenAI Whisper `normalizers/` — <https://github.com/openai/whisper/tree/main/whisper/normalizers> — the reference implementation of the scoring normalisers in §2.6.
- `inflect` — <https://github.com/jaraco/inflect> — Python library for cardinal/ordinal/plural English word forms used in §2.4.
- `word2number` — <https://github.com/akshaynagpal/w2n> — library for parsing English number words to int used in §2.5.
- `jiwer` — <https://github.com/jitsi/jiwer> — WER / CER reference implementation used in validation.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation* applies: ITN may be invoked in a chain `ASR → ITN → LLM`; an attacker who controls the audio controls the ITN-rewritten string. Treat ITN output as untrusted; do not use it to compose commands or SQL without a downstream validator. *ML09 Output Integrity Attack* applies when ITN silently corrupts numbers (phone numbers, invoice amounts) with no error signal.
- MITRE ATLAS — <https://atlas.mitre.org/> — adversarial-example catalogue; the audio→ASR→ITN chain inherits the audio adversarial surface documented there.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record locale coverage, supported semiotic classes, normaliser version, and known failure modes (ambiguous cardinals, brand-case verbatim); *Measure*: report both raw and normalised WER; log idempotence-test outcomes; keep a regression suite of hand-graded edge fixtures as part of the release gate.
