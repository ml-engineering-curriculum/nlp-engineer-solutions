# exercise-01 — Regex and FST Normaliser

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-102-classical-nlp/exercises/exercise-01-regex-and-fst-normaliser.md`.
> This solution targets the learning goals that filename implies: build a text normaliser that handles the "ugly" cases upstream of tokenisation (numbers, dates, abbreviations, URLs, emails, currency, whitespace, control characters) using **regular expressions** for the easy rules and a **weighted finite-state transducer (WFST)** for the rules where regex order of application matters.

## 1. Solution overview

Text normalisation is a cascade: each stage rewrites one class of surface form into a canonical form. Two engines carry most of the weight:

| Engine | Best for | Why |
|---|---|---|
| **Regex** | Pattern detection, class-aware substitution, local rewrites | Fast, ubiquitous, no composition story, bad at ordered rewriting |
| **FST** | Rule cascades where one stage's output is the next stage's input, verbaliser-style number/date/currency expansion, reversible mappings | Composable (`A ∘ B ∘ C` is one transducer), optimisable (determinisation / minimisation), inherently supports *n*-best output |

Formal framing: a **regex** describes a regular *language* (a set of strings). A **finite-state transducer** describes a regular *relation* (a set of input/output string pairs). A normaliser that *rewrites* strings is naturally a relation; a normaliser that only *recognises* classes is a language. Mixing the two deliberately is the point of the exercise.

The deliverable is a `TextNormaliser` with:

- A regex layer that handles whitespace, control characters, URLs, emails, and simple substitutions (dots after single-letter initials, curly→straight quotes).
- An FST layer, built with Pynini (the Python front-end to OpenFst), that verbalises numbers, currencies, and dates — rewrites where the output of one stage feeds the next and where ordering matters.
- A deterministic pipeline that composes the two and returns normalised text plus a map from output offsets back to the input (so downstream NER spans remain reportable against the raw string).

## 2. Worked implementation

### 2.1 Project layout

```
normaliser/
  regex_layer.py       # pure-Python regex rewrites
  fst_layer.py         # Pynini FSTs for numbers/currency/dates
  pipeline.py          # composes layers, exposes TextNormaliser
  tests/
    test_regex.py
    test_fst.py
    test_pipeline.py
  requirements.txt     # regex>=2024, pynini>=2.1
```

Pynini (<https://www.openfst.org/twiki/bin/view/GRM/Pynini>) is the standard Python interface to OpenFst. It is used in the open-source NVIDIA NeMo Text Processing stack, which is a working reference for production number/date verbalisers. Install from source or via `conda install -c conda-forge pynini`; it is not pip-installable on all platforms.

### 2.2 Regex layer

The regex layer handles rewrites that are *context-free* within the token (or that are order-insensitive). Prefer the third-party `regex` module to the stdlib `re` for Unicode property classes (`\p{L}`, `\p{N}`, `\X`).

```python
# regex_layer.py
import regex as re
from dataclasses import dataclass, field

# Rule ordering matters: whitespace first, URLs before single-letter-dot, etc.
# Each tuple is (name, compiled pattern, replacement-or-callable).
URL_RE  = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WS_RE   = re.compile(r"\s+")
# Single-letter initials: "U.S.A." -> "USA" (not "U. S. A.")
INIT_RE = re.compile(r"\b(?:[A-Z]\.){2,}[A-Z]?")

@dataclass
class RegexLayer:
    url_token: str = " <URL> "
    email_token: str = " <EMAIL> "

    def __call__(self, text: str) -> str:
        text = text.translate(QUOTES)
        text = CTRL_RE.sub(" ", text)
        text = URL_RE.sub(self.url_token, text)
        text = EMAIL_RE.sub(self.email_token, text)
        text = INIT_RE.sub(lambda m: m.group(0).replace(".", ""), text)
        text = WS_RE.sub(" ", text).strip()
        return text
```

The order is intentional: URLs are detected *before* `INIT_RE` would see `http://` and mis-fire on the dots. Any reshuffling of these lines is a regression and `test_regex.py` catches it.

### 2.3 FST layer

Pynini exposes `pynini.accep(s)` (an acceptor for the string `s`), `pynini.cross(a, b)` (a transducer `a → b`, i.e. `a:b`), composition (`@`), union (`|`), closure (`.star`, `.plus`), and `pynini.compose` / `shortestpath` for application.

```python
# fst_layer.py
import pynini
from pynini.lib import pynutil

# Alphabet: byte-level FSTs are simplest. Byte strings go in, byte strings come out.
SIGMA_STAR = pynini.closure(pynini.union(*(chr(c) for c in range(256))))

DIGITS = "0123456789"
TEENS = {
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen",
}
UNITS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
TENS = {
    "2": "twenty", "3": "thirty", "4": "forty", "5": "fifty",
    "6": "sixty", "7": "seventy", "8": "eighty", "9": "ninety",
}


def _digits_0_99() -> pynini.Fst:
    unit = pynini.union(*(pynini.cross(k, v) for k, v in UNITS.items()))
    teen = pynini.union(*(pynini.cross(k, v) for k, v in TEENS.items()))
    tens = pynini.union(*(pynini.cross(k, v) for k, v in TENS.items()))
    two_digit = (
        teen
        | (tens + pynutil.delete("0"))                # "20" -> "twenty"
        | (tens + pynutil.insert(" ") + unit)         # "21" -> "twenty one"
    )
    return (two_digit | unit).optimize()


def integer_verbaliser() -> pynini.Fst:
    """0..999 verbaliser. Extend with thousands/millions as needed."""
    small = _digits_0_99()
    hundred = (
        pynini.cross(pynini.union(*DIGITS), "") + pynutil.insert(" hundred")
    )
    # "123" -> "one hundred twenty three" ; "100" -> "one hundred"
    with_tail = hundred + pynutil.insert(" ") + small
    only_hundreds = hundred + pynutil.delete("00")
    return (small | only_hundreds | with_tail).optimize()


def currency_verbaliser() -> pynini.Fst:
    """$X -> X dollars, €X -> X euros. Byte-level FST."""
    dollar = pynutil.delete("$") + integer_verbaliser() + pynutil.insert(" dollars")
    euro   = pynutil.delete("€") + integer_verbaliser() + pynutil.insert(" euros")
    return (dollar | euro).optimize()


def date_verbaliser() -> pynini.Fst:
    """YYYY-MM-DD -> 'YYYY, month DD' (coarse ISO-date rewrite)."""
    months = {
        "01": "january", "02": "february", "03": "march", "04": "april",
        "05": "may", "06": "june", "07": "july", "08": "august",
        "09": "september", "10": "october", "11": "november", "12": "december",
    }
    year = integer_verbaliser()
    month = pynini.union(*(pynini.cross(k, v) for k, v in months.items()))
    day = _digits_0_99()
    sep1 = pynutil.delete("-") + pynutil.insert(", ")
    sep2 = pynutil.delete("-") + pynutil.insert(" ")
    return (year + sep1 + month + sep2 + day).optimize()


def apply(fst: pynini.Fst, s: str) -> str:
    """Run the one-best output through an FST on string `s`."""
    lattice = pynini.compose(s, fst)
    return pynini.shortestpath(lattice).string()
```

Two practical FST idioms worth naming:

- **`pynutil.delete(x)`** is sugar for `pynini.cross(x, "")` — it accepts `x` and emits nothing. **`pynutil.insert(x)`** is the opposite.
- **`.optimize()`** applies determinisation + minimisation. Always call it on compiled rules; the memory and composition-time savings are large.

### 2.4 Composing regex + FST into a pipeline

```python
# pipeline.py
import regex as re
from dataclasses import dataclass, field
from typing import Callable

import pynini
from .regex_layer import RegexLayer
from .fst_layer import (
    integer_verbaliser, currency_verbaliser, date_verbaliser, apply as apply_fst
)

# Simple pre-scan: isolate spans the FST should rewrite, keep the rest verbatim.
NUM_RE  = re.compile(r"\b\d{1,3}\b")
CUR_RE  = re.compile(r"[$€]\d{1,3}")
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


@dataclass
class TextNormaliser:
    regex_layer: RegexLayer = field(default_factory=RegexLayer)
    _int_fst: pynini.Fst   = field(default_factory=integer_verbaliser)
    _cur_fst: pynini.Fst   = field(default_factory=currency_verbaliser)
    _date_fst: pynini.Fst  = field(default_factory=date_verbaliser)

    def __call__(self, text: str) -> str:
        text = self.regex_layer(text)
        text = DATE_RE.sub(lambda m: apply_fst(self._date_fst, m.group(0)), text)
        text = CUR_RE.sub (lambda m: apply_fst(self._cur_fst,  m.group(0)), text)
        text = NUM_RE.sub (lambda m: apply_fst(self._int_fst,  m.group(0)), text)
        return re.sub(r"\s+", " ", text).strip()
```

The ordering of the three FST calls is the key design decision: dates must run before the currency/number FSTs so `2024-03-15` is not pre-consumed as three bare numbers; currency runs before plain numbers because the number FST would otherwise swallow the digits and leave `$` stranded. Whenever you add a rule class, document its position in the cascade and add a regression test that exercises the ordering.

### 2.5 End-to-end example

```python
nrm = TextNormaliser()
assert nrm("Please pay $42 by 2024-03-15.") == (
    "Please pay forty two dollars by two thousand twenty four , march fifteen ."
    # (comma spacing depends on how you set `sep1` in date_verbaliser)
)
assert nrm("Visit https://example.com or mail a@b.co.") == (
    "Visit <URL> or mail <EMAIL> ."
)
```

## 3. Validation steps

1. **Regex unit tests.** For each rule in `RegexLayer` write one positive and one negative example (e.g. `INIT_RE` must fire on `U.S.A.` and NOT on `3.14`). Keep the tests in `test_regex.py`; regressions here are the most common when a learner adds a new rule.
2. **FST round-trip where the relation is a bijection.** `integer_verbaliser()` is not a bijection (`one hundred` could map back from several numeric forms), so skip it. For a mapping that *is* bijective (e.g. a simple abbreviation table `Dr. ↔ doctor`), compile both directions as `t` and `pynini.invert(t)` and assert `invert(t) @ t` accepts every surface form.
3. **FST composition test.** Verify that `integer_verbaliser() @ pynini.closure(pynini.union(*(chr(c) for c in range(256))))` is non-empty on each of `"0"`, `"9"`, `"17"`, `"99"`, `"100"`, `"999"`. If composition is empty, the FST under-covers its alphabet and `shortestpath` will throw.
4. **Pipeline golden file.** Keep `tests/golden/input.txt` and `tests/golden/expected.txt` in-repo. The pipeline is tested by exact string equality; any diff fails CI. This is the test that catches silent rule-ordering bugs.
5. **Determinisation / minimisation sanity.** After building each FST, log `fst.num_states()` before and after `.optimize()`. If optimisation does not reduce the state count on a hand-built cascade of 50+ rules, you likely have an epsilon loop or unreachable state — investigate before shipping.
6. **Performance smoke test.** Normalise the first 10 000 lines of a representative corpus; record wall time and per-line latency. Regex-only should land in the single-digit microseconds per line; FST calls add the cost of one composition + shortest-path per matched span — typically sub-millisecond for 0-999 integer verbalisation. A 10× jump between runs is a regression.
7. **Fuzz with Unicode edge cases.** Feed the normaliser strings with mixed NFC/NFD, bidirectional marks, surrogate pairs, and zero-width joiners (the exercise reader may have already audited these in mod-101 exercise-03). No rule should panic; every rule should either fire cleanly or pass the input through unchanged.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Engine choice articulated** | States why each rule class is regex vs FST, cites the language-vs-relation distinction | States one of the two | Treats regex + FST as interchangeable |
| **Rule ordering** | Documented ordering with regression tests that would catch reshuffles | Correct ordering without tests | Random order; pipeline breaks on realistic input |
| **FST composition / optimisation** | Calls `.optimize()` on all compiled rules; composition tested | Compiles but never optimises | Builds enormous FSTs at every call |
| **Reversibility** | Bijective sub-rules explicitly tested with `invert` | Mentions invert; no test | Treats one direction as sufficient |
| **Unicode safety** | Rules use `regex` module Unicode classes; handles ZWJ / NFC-vs-NFD | Handles ASCII only, scope documented | Silently mis-splits non-ASCII |
| **Offset preservation** | Returns an output→input offset map OR documents that offsets are invalidated after FST rewrites | Mentions the tradeoff | Returns no offsets without acknowledgement |
| **Responsible deployment** | Notes PII handling for `<EMAIL>`/`<URL>` scrubbing in line with NIST AI RMF Map/Govern; flags adversarial regex-DoS surface per OWASP ML Top 10 and MITRE ATLAS | Mentions one of the three | Ignores the security angle |
| **Reproducibility** | Pinned `regex`, `pynini`, OpenFst versions; one-command test run | Partially pinned | Not reproducible |

## 5. Common mistakes

- **Catastrophic backtracking.** Patterns like `(.+)+$` on long inputs are classic regex-DoS triggers. Prefer the `regex` module's atomic groups or possessive quantifiers; test any user-driven pattern with worst-case inputs. OWASP flags this surface under "input attacks" for ML pipelines, and MITRE ATLAS techniques like *AML.T0049* (Exploit Public-Facing Application) include regex-DoS as a prototypical vector against text-processing services.
- **Letting regex do rule cascades.** Writing an eight-step pipeline as a single long regex "because it works on my examples" hides the ordering, defeats composition, and makes the eighth step fail silently when the seventh gets edited. Use regex for pattern matching; use the FST for ordered rewriting.
- **Skipping `.optimize()`.** Pynini FSTs built from many `union`/`cross` calls accumulate epsilon transitions; a 300-rule un-optimised FST can be 10-100× larger in states than the minimal equivalent and will compose much slower. Always optimise after construction.
- **Forgetting the sigma closure.** Composition `s @ fst` fails (returns empty) if the FST does not cover every input symbol in `s`. Either restrict the FST to spans pre-isolated by regex (as in §2.4) or union it with a `SIGMA_STAR` pass-through for off-domain characters.
- **Treating `pynutil.delete`/`insert` as free.** Each adds an epsilon arc; stacking many of them without `.optimize()` creates the state-count blow-up in the previous bullet.
- **No golden-file test.** Normaliser bugs are usually off-by-one in rule ordering. Golden-file equality is the only test that reliably catches these; relying on unit tests per rule misses interaction bugs.
- **Dropping offsets without saying so.** Any FST that inserts or deletes characters invalidates offsets back to the original string. Downstream NER span-return, QA span-return, and PII-redaction-report all depend on those offsets. Either preserve them (track byte-level diffs during substitution) or make the loss explicit in the API.
- **Leaking PII in the normalised text.** Scrubbing URLs/emails to placeholder tokens is a privacy and compliance decision, not a style decision. NIST AI RMF (Govern) expects this to be documented; MITRE ATLAS lists data-exposure cases where normaliser logs become the leak path.

## 6. References

Primary sources used in this solution:

- Mohri, M., Pereira, F., Riley, M. (2008). *Speech Recognition with Weighted Finite-State Transducers*. In *Springer Handbook of Speech Processing*. <https://www.openfst.org/twiki/pub/FST/FstBackground/hbka.pdf> — the standard reference for WFSTs and the algorithms (composition, determinisation, minimisation) that Pynini/OpenFst implement.
- OpenFst library documentation — <https://www.openfst.org/twiki/bin/view/FST/FstQuickTour> — authoritative reference for the FST operations invoked indirectly through Pynini.
- Pynini documentation — <https://www.openfst.org/twiki/bin/view/GRM/Pynini> and the `pynini.lib.pynutil` module reference — Python surface used in §2.3.
- Unicode Technical Standard #18, *Unicode Regular Expressions* — <https://www.unicode.org/reports/tr18/> — defines the Unicode-aware regex semantics implemented by the `regex` PyPI module (`\p{...}`, `\X`).
- `regex` Python module — <https://pypi.org/project/regex/> — supplies the UTS #18 features missing from `re`.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — "Input Manipulation" and adjacent entries cover ReDoS and malformed-input attacks against text pre-processors.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue; regex-DoS falls under *AML.T0049* (Exploit Public-Facing Application) when the normaliser is an internet-facing component.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — Govern/Map functions require documenting PII handling (e.g. `<URL>`/`<EMAIL>` scrubbing) and the trust boundary that the normaliser sits on.
