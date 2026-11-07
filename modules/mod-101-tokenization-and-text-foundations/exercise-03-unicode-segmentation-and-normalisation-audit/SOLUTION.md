# exercise-03 — Unicode Segmentation and Normalisation Audit

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-101-tokenization-and-text-foundations/exercises/exercise-03-unicode-segmentation-and-normalisation-audit.md`.
> This solution targets the learning goals that filename implies: audit a text-processing pipeline for correct Unicode behaviour — the normalisation form applied (NFC/NFD/NFKC/NFKD), the segmentation rules used for grapheme clusters, words, and sentences (UAX #29), and the consequences when these are wrong.

## 1. Solution overview

A tokeniser that does not pin its normalisation form and its segmentation algorithm is a tokeniser whose output depends on how the input was keyed in. Two visually identical strings (`"café"` as NFC vs NFD) will hash differently, split differently, and embed differently; a string that happens to contain a ZWJ or a combining mark can silently split a grapheme and poison downstream tasks (NER spans, offset mapping for QA, retrieval hit-highlighting).

The audit has three parts:

1. **Decide and pin a normalisation form.** NFC is the Unicode Consortium's default recommendation for stored text (UAX #15). NFKC additionally collapses "compatibility" variants (e.g. half-width → full-width, superscript `²` → `2`) and is appropriate for *search and matching* but destroys meaning-bearing distinctions for *display* and *downstream NLP that cares about casing/typography*. Choose one and document the trade-off.
2. **Segment using UAX #29.** Python's stdlib `str` methods split on Python's internal notion of whitespace/alpha, not on Unicode grapheme/word boundaries. Use `regex` (with `\X` for graphemes and `\p{...}` for Unicode categories) or ICU (`PyICU.BreakIterator`) for correct behaviour. Grapheme-cluster boundaries matter whenever offsets are returned to a caller (e.g. QA span starts/ends, PII redaction).
3. **Produce an audit report.** The deliverable lists each pipeline stage, the normalisation it applies, the segmentation it uses, and the risks it inherits.

## 2. Worked answer / implementation

### 2.1 Audit scope template

For each stage of the pipeline, record the following fields. The example below is populated for a typical tokeniser-plus-BERT inference path.

| Stage | Input form | Normalisation applied | Segmentation | Risks |
|---|---|---|---|---|
| **Ingest (HTTP body)** | arbitrary UTF-8 | none | n/a | Mixed NFC/NFD from clients; invalid UTF-8; BOM |
| **Pre-clean** | bytes → `str` | NFC (explicit) | n/a | Must choose NFC vs NFKC before this stage |
| **Pre-tokeniser** | `str` | none (NFC already) | Unicode word boundaries (UAX #29) | Native `str.split()` misses ZWJ/ZWNJ, emoji joiners, Indic clusters |
| **Subword tokeniser** | `str` per word | depends on model (BERT: NFD + strip accents + lowercase; GPT-2: byte-level, no normalisation) | WordPiece or byte-level BPE | Normalisation collisions (`micro µ` vs `micro m`), accent-loss for languages that need accents |
| **Offset return** | token ids + offsets | n/a | should be *grapheme-aware* | Returning offsets on a different normalisation form than the client sent will misalign NER spans and highlights |

### 2.2 Reference audit script

```python
# unicode_audit.py
import unicodedata
import regex  # third-party; `re` does not implement UAX #29

NORMAL_FORMS = ("NFC", "NFD", "NFKC", "NFKD")


def normalise_report(s: str) -> dict:
    """Show how a string differs across the four normal forms."""
    return {form: unicodedata.normalize(form, s) for form in NORMAL_FORMS}


def grapheme_clusters(s: str) -> list[str]:
    """UAX #29 extended grapheme clusters via the `regex` module's \X."""
    return regex.findall(r"\X", s)


def word_tokens_uax29(s: str) -> list[str]:
    """Approximation of UAX #29 word boundaries using Unicode properties."""
    # \w in `regex` module respects Unicode categories (L*, N*, Mn, Pc) by default.
    return regex.findall(r"\p{L}[\p{L}\p{M}\p{N}_‍‌]*|\p{N}+", s)


def mixed_script_runs(s: str) -> list[tuple[str, str]]:
    """Return (script_name, substring) runs; useful for confusable detection."""
    out, current, current_script = [], [], None
    for ch in s:
        try:
            script = unicodedata.name(ch).split(" ")[0]
        except ValueError:
            script = "UNKNOWN"
        if script != current_script and current:
            out.append((current_script, "".join(current)))
            current = []
        current.append(ch)
        current_script = script
    if current:
        out.append((current_script, "".join(current)))
    return out
```

### 2.3 Canonical worked example

Run the audit on a short corpus that stresses every category:

```python
samples = [
    "café",                       # 1: NFC vs NFD; 4 vs 5 code points
    "ﬁ",                          # 2: ligature; NFKC → "fi"
    "Ⅳ",                          # 3: Roman numeral; NFKC → "IV"
    "👩‍🔬",                        # 4: ZWJ sequence; 1 grapheme, 3 code points
    "नमस्ते",                      # 5: Devanagari conjunct; graphemes ≠ chars
    "paypaΙ.com",                 # 6: Latin 'l' replaced with Greek capital iota (confusable)
    "    Hello World",       # 7: NBSP vs regular space
]
```

Expected observations, which the writeup should enumerate:

- **(1) `"café"`.** `len(NFC) == 4`, `len(NFD) == 5` because `é` → `e` + U+0301. A tokeniser that normalises to NFC and then diffs against user input in NFD will report a false mismatch on the trailing combining mark.
- **(2) `"ﬁ"` ligature.** NFC leaves it; NFKC turns it into `"fi"`. Matters for search ("find all 'fi'") and for any string equality check. UAX #15 is explicit that NFKC is lossy and should not be reversed.
- **(3) `"Ⅳ"` roman numeral.** NFKC → `"IV"`. For numeric extraction this is a *gain*; for preserving the original glyph (e.g. legal document faithful rendering) it is a *loss*.
- **(4) Emoji ZWJ sequence.** `len(str) == 3` code points (woman + U+200D ZWJ + microscope); `grapheme_clusters(s) == ["👩‍🔬"]` — one perceived character. A naive tokeniser that returns character offsets will split the woman-scientist emoji across three tokens, and offset-mapping back to the source will land mid-grapheme.
- **(5) Devanagari `नमस्ते`.** 6 code points, 4 grapheme clusters. Any UI highlight that uses code-point offsets will highlight the wrong substring.
- **(6) Confusable-script host.** `"paypaΙ.com"` contains a Greek capital iota where ASCII expects `l`. NFKC does *not* fix this (iota and L are different letters). The audit must flag mixed-script runs explicitly — this is the UTS #39 confusable-detection problem.
- **(7) NBSP vs space.** Python's `str.split()` will handle U+00A0 (NBSP) as whitespace, but naive `line.split(" ")` will not. Models trained with one and served with the other show silent fertility inflation.

### 2.4 Recommended pipeline defaults

For a typical NLP service, the audited and documented defaults are:

| Concern | Default | Rationale |
|---|---|---|
| Stored / canonical form | **NFC** | UAX #15 default; reversible; smallest code-point count |
| Equality / search / dedup | **NFKC on a *copy*** | Collapses compatibility variants; keep the original around |
| Lowercasing | `str.casefold()`, *not* `str.lower()` | Casefold handles German `ß → ss`, Turkish dotted-i, etc.; `lower()` is locale-naive |
| Grapheme boundaries | `regex.findall(r"\X", s)` or ICU | UAX #29 compliant; code-point indexing is not safe |
| Word boundaries | UAX #29 via `regex` or ICU; otherwise explicit pre-tokeniser | `str.split()` loses ZWJ-joined words, Indic conjuncts, CJK |
| Mixed-script check | Flag runs where `Script_Extensions` disagrees | UTS #39; helps catch homograph attacks |

## 3. Validation steps

1. **Round-trip the canonical form.** `assert unicodedata.normalize("NFC", unicodedata.normalize("NFC", x)) == unicodedata.normalize("NFC", x)` — normalisation is idempotent; a second pass should be a no-op.
2. **Idempotence of the full pipeline.** `pipeline(pipeline(x)) == pipeline(x)` for representative samples. If false, you have a stage that re-normalises differently on second pass (often `lower()` interacting with Turkish locale).
3. **Grapheme-vs-codepoint test.** `len(regex.findall(r"\X", "👩‍🔬")) == 1`, but `len("👩‍🔬") == 3` because the string is three code points (woman + ZWJ + microscope). The full first grapheme therefore has *grapheme-cluster* offsets `(0, 1)` and *code-point* offsets `(0, 3)`. If your offset-mapping returns `(0, 3)` for the first user-visible character, you are returning code-point offsets; if it returns `(0, 1)`, you are returning grapheme-cluster offsets. Document which, and ensure the choice matches what your clients expect — reversing the diagnosis sends learners off to "fix" correct offset handling.
4. **ICU parity spot-check.** On 100 held-out multilingual lines, compare your UAX #29 word tokens against `PyICU.BreakIterator.createWordInstance(Locale("root"))`. Non-trivial divergence means your regex isn't truly UAX #29.
5. **Mixed-script scan.** Run `mixed_script_runs` on a corpus; flag any non-trivial run that mixes Latin with Greek, Cyrillic, or Han — classic homograph pattern.
6. **Normalisation-collision scan.** Count distinct NFC strings whose NFKC form collides; investigate the top collisions. Expected offenders: compatibility ligatures, superscript digits, full-width punctuation.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Normalisation decision** | Explicit choice of NFC (or NFKC on a copy) with rationale | Normalises but doesn't say which form or why | Not normalised |
| **Segmentation** | Uses UAX #29 (regex `\X` / ICU) for graphemes and words | Uses Unicode-aware regex but not `\X` | `str.split()` only |
| **Casefolding** | `str.casefold()` where lowercase needed; locale noted | `str.lower()` with caveat | `str.lower()` silently |
| **Offset returns** | Documented as code-point *or* grapheme offsets; chosen to match clients | Documented but inconsistent | Off-by-grapheme on emoji / Indic |
| **Audit report** | Stage-by-stage table + representative samples + fix list | Table only | Prose without evidence |
| **Confusables** | Mixed-script detection present | Mentioned, not implemented | Not considered |
| **Reproducibility** | `regex`/ICU/`unicodedata` versions pinned; Python Unicode DB version noted | Partial | None |

## 5. Common mistakes

- **Assuming code points = characters.** `len("👩‍🔬") == 4`. Any offset API that counts code points and claims to return "character offsets" will silently split emoji, flags, and Indic conjuncts.
- **Using `str.lower()` for case-insensitive matching.** `"STRAẞE".lower() == "straße"`, but you cannot reverse it, and `"straße".lower() == "straße"` while `"STRASSE".lower() == "strasse"` — so "STRAẞE" and "STRASSE" compare unequal after lower but equal after `casefold()`. Casefold is the correct primitive.
- **NFKC-everywhere.** Applying NFKC in the ingest path destroys information that later stages may need: ligatures, superscripts, full-width punctuation — all legitimately distinct in display. Keep the original and NFKC a *copy* for search/dedup.
- **Ignoring BOM.** Files produced on Windows often start with a BOM (U+FEFF); left in, it becomes a visible leading token in most tokenisers. Strip explicitly in ingest.
- **Regex `\w` under `re`, not `regex`.** Python stdlib `re`'s `\w` is Unicode-aware for letters but does not match UAX #29 word boundaries. Use the third-party `regex` module for UAX #29 semantics.
- **Normalising inside the model's tokeniser silently.** Many pre-trained tokenisers apply their own normalisation (e.g. BERT does NFD + strip-accents). If you also normalise upstream, document that the model sees *your* output after *its* normalisation; otherwise accented inputs get stripped twice and offsets drift.
- **Treating "visually identical" as "equal".** Confusable-script attacks (`рaypal.com` with Cyrillic `р`) survive naïve normalisation. Needs UTS #39 handling.

## 6. References

Primary sources used in this solution:

- Unicode Standard Annex #15 — *Unicode Normalization Forms*. <https://unicode.org/reports/tr15/> — defines NFC, NFD, NFKC, NFKD; recommends NFC as the canonical interchange form; states NFKC lossy.
- Unicode Standard Annex #29 — *Unicode Text Segmentation*. <https://unicode.org/reports/tr29/> — grapheme-cluster, word, and sentence boundary rules.
- Unicode Technical Standard #39 — *Unicode Security Mechanisms*. <https://unicode.org/reports/tr39/> — confusables and mixed-script detection.
- Python Unicode HOWTO — <https://docs.python.org/3/howto/unicode.html> — `unicodedata`, `str.casefold()`, and caveats.
- `regex` module documentation — <https://pypi.org/project/regex/> — `\X` for grapheme clusters, `\p{...}` for Unicode categories, Script_Extensions properties.
- ICU User Guide — *BreakIterator* — <https://unicode-org.github.io/icu/userguide/boundaryanalysis/> — reference implementation of UAX #29.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — relevant lens for homograph/input-manipulation risks at the tokeniser layer.
- MITRE ATLAS — <https://atlas.mitre.org/> — adversarial ML techniques; confusables can appear in prompt-injection vectors.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — Map/Measure functions call for documented data-handling decisions, of which normalisation is one.
