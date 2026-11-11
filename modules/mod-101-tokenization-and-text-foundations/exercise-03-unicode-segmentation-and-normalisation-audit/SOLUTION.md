# exercise-03 — Unicode Segmentation and Normalisation Audit

## 1. Solution overview

The exercise asks learners to audit a text pipeline for Unicode
correctness: pick an input tokenizer and preprocessing stack, probe it
with inputs that stress segmentation and normalisation, document where
it diverges from the Unicode Standard, and recommend fixes.

The reference audit uses the Hugging Face BERT WordPiece tokenizer as
the system under test because its defaults (NFD → strip accents →
lowercase → whitespace+punct split) exercise the full chain. The
methodology generalises to any tokenizer.

Audit deliverables:

1. A **probe corpus** that covers the Unicode features the audit must
   check (composition forms, grapheme clusters, bidi-sensitive input,
   confusables, invisibles).
2. An **audit script** that runs each probe through the pipeline and
   records the outputs alongside expected Unicode-standard behaviour.
3. A **findings table** with severity ratings and references to the
   relevant Unicode Standard Annex (UAX) or Technical Report (UTR).
4. A **remediation note** for each finding mapped to concrete pipeline
   changes.

## 2. Worked answer or implementation

### 2.1 Probe corpus

| ID | Input (shown as a Python literal) | What it tests | Expected per Unicode standard |
|---|---|---|---|
| P1 | `"café"` NFC vs `"café"` NFD | Canonical composition (UAX #15 / UTR #15) | Both forms are canonically equivalent and must tokenise identically after normalisation |
| P2 | `"ﬀ"` (ligature `ﬀ`) | Compatibility decomposition (NFKC) | NFKC maps to `"ff"`; NFC preserves the ligature |
| P3 | `"👨‍👩‍👧"` (ZWJ family emoji) | Extended grapheme clusters (UAX #29) | Counts as 1 user-perceived character; splitting it is a visible bug |
| P4 | `"Café"` with turkish dotted/undotted i: `"İstanbul"` / `"istanbul"` | Locale-sensitive casing (UAX #44 and CLDR) | ASCII lowercase on `"İ"` gives `"i̇"`; Turkish casing differs |
| P5 | `"pаypal"` with Cyrillic `а` (U+0430) | Confusables (UTS #39) | Must either be flagged or normalised; silent acceptance is a security finding |
| P6 | Any string containing `U+200B` (ZWSP) or `U+2028` (line separator) | Invisibles and format controls (UAX #44 General_Category = Cf) | Should be stripped or normalised, not treated as word-internal characters |
| P7 | `"‮evil.exe"` (right-to-left override) | Bidi control characters (UAX #9) | Must be stripped from untrusted input to prevent spoofing (CVE-2021-42574 family) |
| P8 | `"straße".upper()` → `"STRASSE"` | Case folding (UAX #44 / UCD `SpecialCasing.txt`) | Default casefold doubles `ß`; round-trip lowercase does not restore it |

### 2.2 Audit script

```python
# audit.py
"""Unicode audit harness. Produces a per-probe report comparing pipeline
output to the behaviour mandated by the Unicode Standard."""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, asdict
from typing import Callable

try:
    import regex  # the `regex` package implements \X (grapheme clusters)
except ImportError:  # pragma: no cover
    regex = None


def count_graphemes(s: str) -> int:
    """Count extended grapheme clusters per UAX #29. Falls back to a
    code-point count if the `regex` library is not installed, which the
    audit must then flag explicitly."""
    if regex is None:
        return len(s)
    return len(regex.findall(r"\X", s))


@dataclass
class ProbeResult:
    probe_id: str
    input_repr: str
    tokens: list[str]
    token_ids: list[int]
    grapheme_count: int
    code_point_count: int
    nfc_equal: bool
    nfkc_equal: bool
    finding: str


def run_probe(probe_id: str, text: str, tokenize: Callable[[str], list[str]],
              ids: Callable[[str], list[int]]) -> ProbeResult:
    nfc = unicodedata.normalize("NFC", text)
    nfkc = unicodedata.normalize("NFKC", text)
    return ProbeResult(
        probe_id=probe_id,
        input_repr=repr(text),
        tokens=tokenize(text),
        token_ids=ids(text),
        grapheme_count=count_graphemes(text),
        code_point_count=len(text),
        nfc_equal=(tokenize(text) == tokenize(nfc)),
        nfkc_equal=(tokenize(text) == tokenize(nfkc)),
        finding="",   # filled in by the auditor
    )
```

### 2.3 Running the audit against a WordPiece tokenizer

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_pretrained("bert-base-uncased")

def encode(s: str) -> list[str]:
    return tok.encode(s).tokens

def ids(s: str) -> list[int]:
    return tok.encode(s).ids

PROBES = {
    "P1-nfc": "café",
    "P1-nfd": "café",
    "P2":     "ﬀtest",
    "P3":     "family: \U0001F468‍\U0001F469‍\U0001F467",
    "P4":     "İstanbul istanbul",
    "P5":     "pаypal paypal",
    "P6":     "foo​bar",
    "P7":     "‮evil.exe",
    "P8":     "straße".upper(),
}

results = [run_probe(pid, text, encode, ids) for pid, text in PROBES.items()]
print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
```

### 2.4 Findings template

| ID | Severity | Observation | Unicode source | Recommended fix |
|---|---|---|---|---|
| P1 | Low (expected behaviour, but worth asserting) | NFC and NFD forms tokenise identically under `bert-base-uncased` because the tokenizer applies NFD + strip-accents. | UAX #15 | Add an explicit `unicodedata.normalize("NFC", s)` at the ingress boundary so downstream models receive a single canonical form. |
| P2 | Medium | NFC preserves `ﬀ` as one code point; `bert-base-uncased`'s NFD + strip-accents does not decompose it, so the ligature is treated as `<unk>`. | UTR #15, section "Compatibility Equivalence" | Apply NFKC before tokenisation if downstream users are expected to paste typographically rich text. |
| P3 | High | A ZWJ emoji family is split into component code points, which can break length-sensitive logic (SMS char limits, UI truncation). | UAX #29 | Measure length in grapheme clusters, not code points, in any user-facing surface. Do not change tokenizer input unless the downstream model was trained on grapheme-cluster-aware input. |
| P4 | High | ASCII lowercase of `İ` yields `i` + combining dot above, splitting it across two tokens. Turkish users see different model behaviour than English users on otherwise identical inputs. | UAX #44, CLDR `SpecialCasing.txt` | Decide explicitly whether the locale-sensitive `az`/`tr` casing rules are needed. If not, document that the pipeline is locale-insensitive. |
| P5 | High (security) | Confusable Cyrillic `а` passes through unchanged and produces a different `<unk>` or sub-token path than ASCII `a`. | UTS #39, section "Confusable Detection" | Add a confusable filter at the trust boundary for user-controlled input in fields where impersonation matters (usernames, URLs, brand text). |
| P6 | Medium | ZWSP is preserved and becomes part of a sub-token. This leaks invisible content into logs and vector stores. | UAX #44 (`Cf` = Format) | Strip `Cf` characters at normalisation time unless the pipeline has a documented reason to keep them. |
| P7 | Critical (security) | Bidi override is preserved. If user input is rendered in a log viewer or a web UI, this enables spoofing attacks of the same class as CVE-2021-42574 ("Trojan Source"). | UAX #9 | Reject or escape bidi controls on ingress; filtering alone is not enough for displayed text. |
| P8 | Medium | `"straße".upper()` is `"STRASSE"`, which cannot be lowercased back to `"straße"`. Casing round-trips are lossy in general. | UAX #44, `SpecialCasing.txt` | Never assume casing is reversible. Store canonical form separately from display form. |

Findings should be reported with the exact Unicode version in effect
(e.g. 15.1.0) because character properties can shift between Unicode
releases; the audit must be re-run when the Unicode database is
upgraded.

## 3. Validation steps

1. **Unicode version pinned.** Record `unicodedata.unidata_version` in
   the audit output. A finding that depends on character properties is
   only valid against that version.
2. **Grapheme counting verified on known inputs.** `count_graphemes`
   should return `1` for every single-user-perceived character in the
   probe set, including the ZWJ family. If the `regex` library is not
   installed, the audit must explicitly flag that grapheme-level
   findings are unverified.
3. **Normalisation idempotence.** For every probe, verify
   `nfc(nfc(x)) == nfc(x)` and `nfkc(nfkc(x)) == nfkc(x)`. This catches
   corrupted combining-character sequences.
4. **Confusable detection.** Run the P5 probe through a confusable
   detector (`uts46` from the `idna` package, or the ICU `UnicodeSet`
   confusable mapping). Audit passes only if the detector flags at
   least one of the probe inputs.
5. **Security probes re-run on every pipeline change.** P5 and P7 are
   regression tests; they should live in the pipeline's unit-test suite
   after the audit completes.

## 4. Rubric or review checklist

| Criterion | Weight | Evidence |
|---|---|---|
| Probe set covers all 8 categories (composition, compatibility, grapheme cluster, locale casing, confusables, invisibles, bidi, lossy casing) | 25% | Probe table in deliverable |
| Each finding cites the specific UAX / UTR / UTS section | 15% | Findings table has source column populated |
| Severity ratings follow a documented rubric (impact × likelihood, or equivalent) | 10% | Rubric included in audit report |
| Security-relevant findings (P5, P7) flagged and mapped to known CVE patterns | 15% | Written mapping (e.g. CVE-2021-42574 for Trojan Source) |
| Remediations are concrete code-level changes, not "sanitise input" | 15% | Each finding links to a diff or a function to add |
| Audit is reproducible: Unicode version pinned, deterministic output | 10% | README documents versions |
| Uses `regex.\X` or ICU for grapheme clustering; does not treat `len()` as character count | 10% | Code review |

## 5. Common mistakes

- **Using `str.lower()` and `str.upper()` without caring about locale.**
  These apply the default (locale-independent) casing rules; they get
  Turkish `İ` wrong, silently. If the pipeline must be locale-aware,
  pass the locale explicitly through ICU or document the limitation.
- **Measuring string length with `len()`.** `len("👨‍👩‍👧")` is 5 or 6
  depending on representation, not 1. UIs that use `len()` for
  truncation can split the grapheme cluster mid-sequence and produce
  corrupted glyphs.
- **Treating NFC as "the one true normalisation."** NFC is right for
  storage and matching; NFKC is right for search and tokenization of
  typographically rich input. The audit must state which form is used
  at each pipeline stage.
- **Stripping all non-ASCII.** This was common as "sanitation" and
  still appears in legacy pipelines. It silently destroys legitimate
  non-English input and masks the real problems the audit is looking
  for.
- **Ignoring `Cf` (format) characters.** ZWSP, ZWNJ, LRM, RLM, and the
  bidi overrides are not whitespace and do not render, but they do
  carry through to logs and embeddings.
- **Assuming confusables only matter for domain names.** They matter
  anywhere a decision is made on string equality: usernames, dataset
  labels, moderation lists.

## 6. References

- Unicode Standard Annex #15, *Unicode Normalisation Forms*
  (`https://unicode.org/reports/tr15/`). Definitions of NFC, NFD, NFKC,
  NFKD.
- Unicode Standard Annex #29, *Unicode Text Segmentation*
  (`https://unicode.org/reports/tr29/`). Grapheme cluster, word, and
  sentence boundary rules.
- Unicode Standard Annex #9, *Unicode Bidirectional Algorithm*
  (`https://unicode.org/reports/tr9/`). Bidi controls and their security
  implications.
- Unicode Technical Standard #39, *Unicode Security Mechanisms*
  (`https://unicode.org/reports/tr39/`). Confusable detection, mixed-
  script detection, restricted identifier profiles.
- Unicode Character Database, *SpecialCasing.txt*
  (`https://unicode.org/Public/UCD/latest/ucd/SpecialCasing.txt`).
  Context-sensitive and locale-sensitive casing.
- CVE-2021-42574 — *Trojan Source: Invisible Vulnerabilities*. Motivates
  P7's severity rating.
- OWASP Machine Learning Security Top 10
  (`https://owasp.org/www-project-machine-learning-security-top-10/`).
  Confusables and homoglyph attacks map onto the "Input Manipulation"
  risk category when the ML system makes decisions on text.
- MITRE ATLAS (`https://atlas.mitre.org/`). Confusable inputs and bidi
  spoofing fall under the techniques covered by "Craft adversarial
  data" tactics.
- NIST AI Risk Management Framework (AI RMF 1.0)
  (`https://www.nist.gov/itl/ai-risk-management-framework`). The Measure
  function calls out input-processing assumptions as a required
  documentation artefact.
