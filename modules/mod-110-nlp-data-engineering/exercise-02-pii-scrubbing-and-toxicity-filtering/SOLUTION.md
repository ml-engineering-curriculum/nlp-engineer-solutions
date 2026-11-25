# exercise-02 — PII Scrubbing and Toxicity Filtering

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-110-nlp-data-engineering/exercises/exercise-02-pii-scrubbing-and-toxicity-filtering.md`.
> The target learning outcomes implied by that filename: assemble the two *content safety* filters that must run on a training corpus before it ships — **PII detection + redaction** (Microsoft Presidio-style: pattern + model + context + confidence) and **toxicity filtering** (classifier-based, not word-list-based) — report both *coverage* (what fraction of true PII / toxic content is caught) and *collateral damage* (what legitimate content the filter removes), and keep a reproducible audit trail connecting each redaction to the rule or model that produced it.

## 1. Solution overview

PII scrubbing and toxicity filtering are the two filters most likely to be done wrong in a hurry and silently cause real-world harm:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Regex-only PII** | `[0-9]{3}-[0-9]{2}-[0-9]{4}` catches US SSNs that look like SSNs, misses ones in prose (`"my social is 123 45 6789"`), misses non-US identifiers entirely, and flags every three-dash number-string in the corpus (dates, phone fragments) as PII | Use a layered detector: regex for well-formed tokens, NER for names/orgs/locations, context-window scoring for low-entropy numerics (`"account number: 4111 1111 ..."`), and checksum validation for Luhn/IBAN. Microsoft **Presidio** is the reference open-source implementation of exactly this layered pattern |
| **Word-list toxicity** | A static bad-word list (the C4 naughty-word list; Dodge et al. 2021) over-filters LGBTQ and minority-dialect content while missing euphemisms, misspellings, and newer slurs | Use a **classifier** (`unitary/toxic-bert`, `Detoxify`, Jigsaw's Perspective API) at paragraph or document granularity; calibrate the threshold on a held-out *labelled* sample with fairness slices (Jigsaw's identity subgroups) rather than a global PR curve |
| **Redaction that destroys downstream utility** | Overwriting PII with `[REDACTED]` kills co-reference signals (`John called John`), destroys exact-match evaluation on entity-linking tasks, and makes the scrubbed corpus non-invertible | Use **surrogate replacement** (Faker-generated placeholder, same semiotic type) by default; use `[PERSON_1]`-style placeholders when the downstream task is information extraction; keep a *sealed* mapping table (encrypted-at-rest) for authorised re-identification. Reference: Presidio Anonymizer's surrogate strategy |
| **No ground-truth sample** | You cannot measure coverage or collateral damage without a labelled evaluation set | Hand-label 500–1 000 sentences stratified by (domain, PII type, toxicity subgroup); re-label on every filter rev. Publish precision/recall/F1 per PII entity and per toxicity subgroup in the dataset card |

The deliverable is a two-pass safety filter with six components:

1. **PII detectors** — a `RecognizerRegistry`-style composition of per-entity detectors (regex, NER, checksum, context). Presidio-equivalent entities: `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `US_SSN`, `US_DRIVER_LICENSE`, `CREDIT_CARD`, `IP_ADDRESS`, `IBAN_CODE`, `URL`, `DATE_TIME`, `LOCATION`, `ORGANIZATION`, `MEDICAL_LICENSE`. Each emits `(start, end, type, confidence, source_rule)`.
2. **PII arbitration** — resolve overlapping detections by highest confidence; publish the tie-break rule.
3. **Anonymiser** — strategy per entity: `REPLACE("[TYPE_N]")`, `SURROGATE(Faker)`, `HASH(sha256-salt)`, `MASK("***")`. Pick by downstream need.
4. **Toxicity classifier** — a fine-tuned transformer (`unitary/toxic-bert` / `Detoxify`) returning per-subgroup scores; threshold calibrated with fairness slices.
5. **Decision log** — one row per detection or classification, linked back to `doc_id` and `(span_start, span_end)` or the paragraph index.
6. **Evaluation harness** — precision/recall by PII type and toxicity subgroup on a held-out *labelled* set; the dataset card quotes these numbers.

Reference implementations and tools:

- **Microsoft Presidio** (presidio-analyzer + presidio-anonymizer) — <https://microsoft.github.io/presidio/> — open-source (MIT); the layered regex/NER/context pattern this exercise mirrors. Supports English + multilingual via spaCy / stanza / transformers backends.
- **spaCy** / **flair** / **transformers NER** — the model backbone the `PERSON`/`LOCATION`/`ORGANIZATION` detectors use (`en_core_web_lg`, `dslim/bert-base-NER`, etc.).
- **`unitary/toxic-bert` and `Detoxify`** (Hanu & Unitary 2020) — <https://github.com/unitaryai/detoxify> — fine-tuned RoBERTa/BERT on Jigsaw Unintended Bias in Toxicity Classification; Apache-2.0 model weights.
- **Jigsaw Unintended Bias in Toxicity Classification** (Borkan et al. 2019) — the dataset the Detoxify family is trained on and the standard source of fairness slices (identity subgroups).
- **Faker** — <https://faker.readthedocs.io/> — synthetic surrogate generator used for the anonymiser's realistic-placeholder strategy.

## 2. Worked implementation

### 2.1 Project layout

```
safety_bake/
  pii/
    __init__.py
    entities.py            # PII entity enum
    regex_detectors.py     # email, credit-card, SSN, IP, URL, phone
    model_detectors.py     # PERSON/LOC/ORG via spaCy or HF NER
    context.py             # proximity boosting ("ssn:", "dob:")
    checksum.py            # Luhn, IBAN mod-97
    registry.py            # aggregate + conflict resolution
    anonymizer.py          # REPLACE, SURROGATE, HASH, MASK strategies
  toxicity/
    classifier.py          # Detoxify / toxic-bert wrapper
    threshold.py           # calibration + fairness slice evaluation
  evaluate/
    gold_labels.py         # labelled 500-sentence evaluation set loader
    metrics.py             # precision/recall/F1 by entity & subgroup
  decision_log.py          # JSONL audit trail
  run.py                   # CLI
  tests/
    test_regex_detectors.py
    test_luhn.py
    test_overlap_arbitration.py
    test_surrogate_preserves_cooc.py
  requirements.txt
```

### 2.2 PII entities and regex detectors — strict first, loose last

```python
# pii/entities.py
from enum import Enum

class Entity(str, Enum):
    PERSON          = "PERSON"
    EMAIL_ADDRESS   = "EMAIL_ADDRESS"
    PHONE_NUMBER    = "PHONE_NUMBER"
    US_SSN          = "US_SSN"
    US_DRIVER_LIC   = "US_DRIVER_LICENSE"
    CREDIT_CARD     = "CREDIT_CARD"
    IP_ADDRESS      = "IP_ADDRESS"
    IBAN_CODE       = "IBAN_CODE"
    URL             = "URL"
    DATE_TIME       = "DATE_TIME"
    LOCATION        = "LOCATION"
    ORGANIZATION    = "ORGANIZATION"
    MEDICAL_LICENSE = "MEDICAL_LICENSE"
```

```python
# pii/regex_detectors.py
import re
from dataclasses import dataclass
from .entities import Entity

@dataclass(frozen=True)
class Detection:
    start: int
    end: int
    entity: Entity
    confidence: float   # 0.0 - 1.0
    source: str         # "regex:email", "ner:spacy", "context+luhn"

# Email — RFC 5322-pragmatic (not the full BNF; Presidio ships a similar form)
_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
# IPv4
_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
)
# US SSN in canonical form (loose form handled by context+checksum below)
_US_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
# Credit-card-like 13-19 digit groups (Luhn verified separately in checksum.py)
_CC_CANDIDATE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def regex_detect(text: str) -> list[Detection]:
    out: list[Detection] = []
    for m in _EMAIL.finditer(text):
        out.append(Detection(m.start(), m.end(), Entity.EMAIL_ADDRESS, 0.95, "regex:email"))
    for m in _IPV4.finditer(text):
        out.append(Detection(m.start(), m.end(), Entity.IP_ADDRESS, 0.85, "regex:ipv4"))
    for m in _US_SSN.finditer(text):
        # Area/group/serial exclusions baked into the regex; still weak-confidence
        out.append(Detection(m.start(), m.end(), Entity.US_SSN, 0.70, "regex:ssn-strict"))
    for m in _CC_CANDIDATE.finditer(text):
        out.append(Detection(m.start(), m.end(), Entity.CREDIT_CARD, 0.30, "regex:cc-candidate"))
    return out
```

Points worth defending in review:

- **Low baseline confidence for CC candidates.** `(\d[ -]*?){13,19}` matches every medium-length number including phone numbers, ISBNs, and timestamps. The confidence comes from the Luhn checksum (§2.3) and the context-window boost (§2.4); the regex is a *candidate generator*, not a decision.
- **SSN area/group/serial exclusions.** The SSA explicitly reserves `000`, `666`, `9xx` area numbers; `00` group number; and `0000` serial. Including them in the regex as negative lookaheads makes the pattern specific without a checksum, and matches what Presidio's recognizer does.
- **Email regex is pragmatic, not RFC-perfect.** RFC 5322's BNF matches strings almost no one writes. The pragmatic regex is what Presidio and virtually every production detector use.
- **No PERSON/LOCATION/ORG at the regex layer.** Regex over capitalisation (`[A-Z][a-z]+ [A-Z][a-z]+`) has ~20 % precision on prose and triggers constantly in headlines and code. The NER layer below is the only acceptable answer for name-like entities.

### 2.3 Checksums — Luhn for credit cards, mod-97 for IBAN

```python
# pii/checksum.py
def luhn_ok(digits: str) -> bool:
    """True if the digits (stripped of spaces/dashes) pass the Luhn check."""
    d = [int(c) for c in digits if c.isdigit()]
    if len(d) < 13 or len(d) > 19:
        return False
    s = 0
    for i, x in enumerate(reversed(d)):
        if i % 2 == 1:
            x *= 2
            if x > 9:
                x -= 9
        s += x
    return s % 10 == 0


def iban_ok(iban: str) -> bool:
    """IBAN mod-97 check (ISO 13616); returns True iff the check digits validate."""
    s = iban.replace(" ", "").upper()
    if len(s) < 15 or len(s) > 34:
        return False
    s = s[4:] + s[:4]
    digits = "".join(str(ord(c) - 55) if c.isalpha() else c for c in s)
    return int(digits) % 97 == 1
```

Then the arbitration rule in `registry.py` boosts a regex-CC-candidate to `confidence=0.99` *only* if `luhn_ok` passes. This is how Presidio's `CreditCardRecognizer` works; it is the difference between 10× false positives on dates and millisecond-stable detection.

### 2.4 Context — proximity boost, Presidio-style

```python
# pii/context.py
import re
from .entities import Entity

CONTEXT_CUES = {
    Entity.US_SSN:         [r"\bssn\b", r"social security", r"ss#"],
    Entity.CREDIT_CARD:    [r"\bcard\b", r"\bcc\b", r"visa", r"mastercard", r"amex"],
    Entity.PHONE_NUMBER:   [r"\bphone\b", r"\btel\b", r"\bmobile\b", r"\bcell\b"],
    Entity.EMAIL_ADDRESS:  [r"\bemail\b", r"\be-?mail\b"],
    Entity.IBAN_CODE:      [r"\biban\b", r"\baccount\b"],
    Entity.DATE_TIME:      [r"\bdob\b", r"date of birth", r"birthday"],
}
WINDOW = 50  # characters either side


def context_multiplier(text: str, det_start: int, entity: Entity) -> float:
    """
    Returns a multiplier to apply to detection confidence based on surrounding
    cue words (Presidio-style). 1.0 == no boost; 1.4 == strong cue present.
    """
    lo = max(0, det_start - WINDOW)
    window = text[lo:det_start].lower()
    for pat in CONTEXT_CUES.get(entity, []):
        if re.search(pat, window):
            return 1.4
    return 1.0
```

Why this matters:

- **Numeric PII is the hardest class.** Phone numbers, SSNs, IDs are all strings of digits with weak lexical signals. The `"ssn:"` / `"dob:"` / `"account number:"` prefix on an administrative form is typically the strongest evidence; the Presidio recognizer depends on exactly these cues to hit production precision.
- **Confidence multiplication, not replacement.** A detection without a context cue is still a detection (`confidence=0.70`); with the cue it climbs toward 1.0. The threshold for redaction is separate from confidence; publish both.

### 2.5 NER backbone — spaCy or HF, batched

```python
# pii/model_detectors.py
import spacy
from .entities import Entity
from .regex_detectors import Detection

_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_lg")
    return _NLP


_MAP = {
    "PERSON":  Entity.PERSON,
    "GPE":     Entity.LOCATION,
    "LOC":     Entity.LOCATION,
    "ORG":     Entity.ORGANIZATION,
    "DATE":    Entity.DATE_TIME,
    "TIME":    Entity.DATE_TIME,
}


def ner_detect(text: str) -> list[Detection]:
    doc = _nlp()(text)
    out: list[Detection] = []
    for ent in doc.ents:
        if ent.label_ not in _MAP:
            continue
        out.append(Detection(
            start=ent.start_char, end=ent.end_char,
            entity=_MAP[ent.label_],
            confidence=0.80,          # spaCy exposes no per-span score by default
            source=f"ner:spacy:{ent.label_}",
        ))
    return out
```

Reviewers must check:

- **Two NER backends available, exchangeable.** spaCy's `en_core_web_lg` is CPU-fast but lags transformer NER on recall of lowercased text and foreign names; `dslim/bert-base-NER` (HF, Apache-2.0) is 3–5 F1 better on CoNLL-03. The exercise's `requirements.txt` should allow either and the choice should be recorded in the manifest.
- **spaCy does not expose a reliable per-span score.** Use `0.80` as a pinned default and compensate via the context layer. For HF NER use the softmax-of-aggregation score; it is calibrated enough for threshold tuning.
- **NER is the weakest layer for non-English text.** For a multilingual corpus either (a) switch to a multilingual NER checkpoint (`xlm-roberta-large-finetuned-conll03-english` + per-language fallbacks) or (b) gate NER on language-ID and treat as unresolved below the LID confidence threshold.

### 2.6 Registry + overlap arbitration

```python
# pii/registry.py
from .regex_detectors import Detection, regex_detect
from .model_detectors import ner_detect
from .context import context_multiplier
from .checksum import luhn_ok, iban_ok
from .entities import Entity


def detect_all(text: str) -> list[Detection]:
    dets = regex_detect(text) + ner_detect(text)

    # Checksum boost
    out: list[Detection] = []
    for d in dets:
        c = d.confidence * context_multiplier(text, d.start, d.entity)
        if d.entity is Entity.CREDIT_CARD and luhn_ok(text[d.start:d.end]):
            c = max(c, 0.99)
        if d.entity is Entity.IBAN_CODE and iban_ok(text[d.start:d.end]):
            c = max(c, 0.99)
        out.append(Detection(d.start, d.end, d.entity, min(c, 1.0), d.source))

    # Overlap resolution: when two detections overlap, keep the higher confidence.
    # Tie-break: longer span, then (deterministic) earlier source alphabetically.
    out.sort(key=lambda d: (-d.confidence, -(d.end - d.start), d.source))
    kept: list[Detection] = []
    for d in out:
        if any(not (d.end <= k.start or k.end <= d.start) for k in kept):
            continue
        kept.append(d)
    kept.sort(key=lambda d: d.start)
    return kept
```

Design notes:

- **Overlap resolution must be deterministic.** The naïve "keep both" leaves two redactions on the same span, producing malformed output. The naïve "keep first" is order-dependent and silently non-reproducible.
- **Confidence floor for redaction, not for logging.** Log every detection with its confidence; redact only those above the configured floor (0.5 for high-recall mode, 0.8 for high-precision mode). The threshold is pinned in the manifest.
- **"Known allowlist" strings pass through.** Common unredactable strings (public email addresses in headers, publicly-named entities in Wikipedia training data) are an operational concern; the registry should accept an allowlist and exclude it from redaction while *still logging*.

### 2.7 Anonymiser — surrogate by default, [TYPE_N] for IE tasks

```python
# pii/anonymizer.py
import hashlib
from faker import Faker
from .entities import Entity
from .regex_detectors import Detection

_FAKE = Faker()
_FAKE.seed_instance(20260101)  # deterministic by default

STRATEGIES = {
    "replace_tagged": "replace_tagged",    # [PERSON_0], [PERSON_1], ...
    "surrogate":      "surrogate",         # realistic fake value
    "hash":           "hash",              # sha256(salt||value)[:8]
    "mask":           "mask",              # fixed asterisks
}


def _surrogate(ent: Entity) -> str:
    match ent:
        case Entity.PERSON:         return _FAKE.name()
        case Entity.EMAIL_ADDRESS:  return _FAKE.email()
        case Entity.PHONE_NUMBER:   return _FAKE.phone_number()
        case Entity.CREDIT_CARD:    return _FAKE.credit_card_number()
        case Entity.LOCATION:       return _FAKE.city()
        case Entity.ORGANIZATION:   return _FAKE.company()
        case _:                     return "[" + ent.name + "]"


def anonymize(text: str, dets: list[Detection], strategy: str = "replace_tagged",
              salt: str = "") -> tuple[str, dict[int, str]]:
    """
    Returns (anonymised_text, mapping) where mapping[detection_index] is the
    *surrogate* value. Caller is responsible for sealing the mapping if the
    original values are retained anywhere.
    """
    out: list[str] = []
    cursor = 0
    tagged: dict[Entity, int] = {}
    mapping: dict[int, str] = {}
    for i, d in enumerate(dets):
        out.append(text[cursor:d.start])
        value = text[d.start:d.end]
        match strategy:
            case "replace_tagged":
                n = tagged.get(d.entity, 0); tagged[d.entity] = n + 1
                repl = f"[{d.entity.name}_{n}]"
            case "surrogate":
                repl = _surrogate(d.entity)
            case "hash":
                repl = hashlib.sha256((salt + value).encode()).hexdigest()[:8]
            case "mask":
                repl = "*" * (d.end - d.start)
            case _:
                raise ValueError(f"unknown strategy {strategy}")
        out.append(repl)
        mapping[i] = repl
        cursor = d.end
    out.append(text[cursor:])
    return "".join(out), mapping
```

Points worth defending:

- **`replace_tagged` preserves co-reference.** `"John called John"` becomes `"[PERSON_0] called [PERSON_0]"`, not `"[PERSON] called [PERSON]"`. This is critical when the scrubbed corpus feeds a co-reference or entity-linking model; the alternative loses signal silently.
- **`surrogate` preserves distribution but destroys traceability.** A BERT-pretraining corpus wants `surrogate` so the model sees realistic frequencies of names and emails; a downstream IE training corpus wants `replace_tagged` so it can learn `[PERSON_0]` as a type. State which in the dataset card.
- **The mapping dict is sensitive.** If retained, it is the key to re-identify every scrubbed value. Treat it as PII-grade data (encrypted at rest, access-logged) or discard it on commit. The anonymizer returns the mapping so the *caller* makes the retention decision explicitly.
- **Deterministic Faker seed.** Non-deterministic surrogates make dataset diffs a nightmare. Seed per-run and record the seed in the manifest.

### 2.8 Toxicity classifier — Detoxify + fairness-aware threshold

```python
# toxicity/classifier.py
from detoxify import Detoxify

# Detoxify's "unbiased" model is trained on Jigsaw Unintended Bias in Toxicity
# (Borkan et al. 2019) and exposes seven heads: toxicity, severe_toxicity,
# obscene, identity_attack, insult, threat, sexual_explicit.
_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = Detoxify("unbiased")
    return _MODEL


def score(texts: list[str]) -> list[dict]:
    """Returns per-text dict of head -> probability in [0, 1]."""
    return model().predict(texts)
```

```python
# toxicity/threshold.py
from typing import Iterable

HEADS = ["toxicity", "severe_toxicity", "obscene",
         "identity_attack", "insult", "threat", "sexual_explicit"]


def decision(score: dict, thresholds: dict[str, float]) -> tuple[bool, list[str]]:
    """
    Returns (block, triggered_heads). A paragraph is blocked if any head
    exceeds its threshold. Thresholds are separate per head because operating
    points differ: 'identity_attack' is typically set lower than 'obscene'.
    """
    triggered = [h for h in HEADS if score.get(h, 0.0) >= thresholds.get(h, 1.0)]
    return (bool(triggered), triggered)


def calibrate_thresholds(scored: Iterable[dict], labels: Iterable[int],
                         target_fpr: float = 0.05) -> dict[str, float]:
    """
    Pick per-head thresholds that hit a target FPR on the labelled dev set.
    Report per-head precision/recall after calibration. Fairness subgroup
    FPR must be measured separately (Borkan et al. 2019 subgroup metrics).
    """
    # ... implementation omitted; sweep thresholds, pick per head, report ...
```

Design notes reviewers must check:

- **Use the `unbiased` Detoxify checkpoint, not the original `toxic-bert`.** The original was trained on Jigsaw's "Toxic Comment Classification" set and inherited well-documented biases (Borkan et al. 2019 Table 2); the `unbiased` checkpoint was trained on the follow-on dataset specifically to reduce identity-subgroup FPR delta.
- **Per-head thresholds, not one global threshold.** The heads have very different operating distributions; `identity_attack` has a lower prevalence and needs a lower threshold to hit the same recall. Record all seven in the manifest.
- **Measure fairness slices, not just a PR curve.** Borkan et al. 2019 define three subgroup metrics (Subgroup AUC, BPSN, BNSP); the Jigsaw leaderboard uses the generalised mean of them. Report these on identity subgroups (religion, race, sexual orientation, gender identity, disability). A model with 0.95 overall AUC and 0.7 on a subgroup is actively harmful.
- **Threshold is pinned, not auto-tuned on the train corpus.** Auto-tuning on the corpus you are filtering guarantees the threshold moves with every corpus rev; this is non-reproducible. Tune on a *fixed* held-out labelled set.

### 2.9 Decision log — JSONL audit trail

```python
# decision_log.py
import json
from pathlib import Path
from typing import Any


def emit(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    One row per decision:
      {"doc_id": "...", "paragraph": 7, "stage": "pii" | "toxicity",
       "detections": [{"span": [12, 24], "entity": "EMAIL_ADDRESS",
                        "confidence": 0.96, "source": "regex:email",
                        "replaced_with": "[EMAIL_ADDRESS_0]"}],
       "toxicity_scores": {"toxicity": 0.08, ...},
       "blocked": false, "triggered_heads": []}
    """
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
```

Why this is the load-bearing artefact:

- **Legal.** A privacy incident ("this PII was never scrubbed") is answerable by grepping the decision log for `doc_id` and `entity`. Without it the answer is "we don't know".
- **Debugging.** When precision drops on a release (previously-benign content now flagged toxic), the log is the only way to see which head and which threshold did the flagging.
- **Dataset card.** The published corpus card aggregates the log: counts per entity, per toxicity head, per subgroup. Users of the dataset use this card to decide whether to trust it.

### 2.10 Evaluation — labelled set, per-entity / per-subgroup metrics

```python
# evaluate/metrics.py
from collections import defaultdict


def pii_prf(gold: list[tuple[int, int, str]],  # [(start, end, entity), ...]
            pred: list[tuple[int, int, str]]) -> dict[str, dict[str, float]]:
    """
    Span-level exact-match precision/recall/F1 per entity type.
    Partial overlaps score 0; this is the stricter CoNLL-03 convention and
    appropriate for PII where partial redaction is a privacy incident.
    """
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    gset = set(gold)
    pset = set(pred)
    for d in pset:
        (tp if d in gset else fp)[d[2]] += 1
    for d in gset:
        if d not in pset:
            fn[d[2]] += 1
    out: dict[str, dict[str, float]] = {}
    for ent in set(list(tp) + list(fp) + list(fn)):
        p = tp[ent] / (tp[ent] + fp[ent]) if (tp[ent] + fp[ent]) else 0.0
        r = tp[ent] / (tp[ent] + fn[ent]) if (tp[ent] + fn[ent]) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        out[ent] = {"precision": p, "recall": r, "f1": f,
                    "tp": tp[ent], "fp": fp[ent], "fn": fn[ent]}
    return out
```

Points reviewers must check:

- **Span-exact scoring, not token-level or character-overlap.** For PII, a half-redacted SSN is a *failed* redaction; it must count as FN. Token-level or F1-over-overlap scores hide this.
- **Per-entity breakdown.** Macro-F1 hides uneven behaviour. Credit-card detection at 0.98 F1 and SSN detection at 0.60 F1 is a very different operational story from both at 0.82. Report both the breakdown and the macro.
- **Per-subgroup fairness for toxicity.** Overall AUC + per-identity-subgroup AUC (Borkan et al. 2019 metrics). A 10-point AUC gap between subgroups is the published fail criterion.
- **Re-label on every filter revision.** Labels drift (what counts as "toxic" today vs five years ago); the gold set has to be versioned alongside the filter.

## 3. Validation steps

1. **Regex unit tests.** 20 positives and 20 known negatives per regex detector; regex must produce span-exact matches on positives and no matches on negatives.
2. **Luhn checksum.** All 60 Visa/Mastercard/Amex/Discover BIN prefixes from `test_credit_cards.yml` must pass; 100 random 16-digit non-cards must fail.
3. **Overlap arbitration.** Construct a text where an SSN regex match overlaps a DATE NER match (`"born 123-45-6789 in 1990"`); assert exactly one detection survives and it is the SSN (higher confidence).
4. **Surrogate co-reference preservation.** `"John called John"` → `"[PERSON_0] called [PERSON_0]"` under `replace_tagged`; two different surnames under `surrogate`. Catch regressions where the `tagged` counter resets per detection.
5. **Context boost.** `"ssn: 123-45-6789"` must score higher than `"123-45-6789"` alone; verify the confidence multiplier fires.
6. **Toxicity threshold calibration.** On the held-out evaluation set, verify FPR per subgroup differs by at most 5 percentage points after calibration. Publish the table.
7. **Decision log round-trip.** Re-run the scrubber on one document; assert the decision log row is byte-identical to a previous run (use `json.dumps(sort_keys=True)` for stability). Catches hidden non-determinism (dict iteration order).
8. **End-to-end eval.** Score the scrubbed corpus on a 500-sentence hand-labelled gold set; publish precision/recall/F1 per entity in the dataset card. Flag any entity below 0.80 F1 for review.
9. **Collateral-damage sample.** Hand-review 50 docs *dropped* by the toxicity filter; count how many were actually toxic and how many were false-positive (minority dialect, reclaimed slur, clinical discussion). Report in the dataset card.
10. **Smoke test on non-English.** On a Spanish or German sample, verify (a) regex detectors still fire on EMAIL / IPV4 / CC, (b) NER drops in quality (quantify, do not hide), and (c) multilingual backbone reinstates quality. Publish per-language numbers.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Layered PII detection** | Regex + NER + context + checksum, documented confidence scores | Regex + NER only | Regex only |
| **Overlap arbitration** | Deterministic; confidence + tie-break documented | Deterministic but undocumented | First-match wins / non-deterministic |
| **Checksums** | Luhn (CC) + mod-97 (IBAN); unit-tested | Luhn only | None |
| **Anonymisation strategies** | ≥2 strategies supported; co-reference preserved; strategy recorded per doc | Single strategy | Overwrites with `[REDACTED]` / loses co-reference |
| **Toxicity model** | Classifier (Detoxify / toxic-bert); per-head scoring; per-subgroup fairness evaluation | Classifier, no subgroup slicing | Word-list only |
| **Threshold calibration** | Thresholds set on held-out labelled set; per-head; published in card | Global threshold | Default threshold used without justification |
| **Decision log** | JSONL audit row per decision; linkable to doc_id & span | Only drops logged | No log |
| **Labelled eval set** | ≥500 sentences; versioned; per-entity precision/recall/F1 published | <500, published | None |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks + privacy frameworks (NIST Privacy Framework, GDPR Art. 4(5) pseudonymisation vs Art. 4(5) anonymisation) | Mentioned | Ignored |
| **Documentation / card** | NIST AI RMF *Map/Measure*: entities covered, model version, thresholds, fairness subgroups, labelled-set version | Partial | None |

## 5. Common mistakes

- **Regex-only PII shipped to production.** Misses `PERSON`, `LOCATION`, `ORGANIZATION` entirely; misses "my SSN is one two three four five six seven eight nine"; silently fails on non-US ID formats. The audit incident when it is found is more expensive than the model work needed to add NER.
- **Using a static bad-word list for toxicity.** Over-filters LGBTQ and minority-dialect content while missing euphemisms and misspellings. Dodge et al. 2021 is the public case against this; a dataset released with a word-list-based filter will be reviewed under that light.
- **Overwriting PII with `[REDACTED]`.** Destroys co-reference, destroys IE training signal, is irreversible. Use `[PERSON_0]` or Faker surrogates; preserve co-reference by re-using the same counter per entity type within the document.
- **Keeping the surrogate-mapping dict unencrypted in the output artefact.** The point of pseudonymisation is that the mapping is sealed; shipping it alongside the corpus turns the pseudonymisation into decorative obfuscation (GDPR Art. 4(5) explicitly: the mapping must be "kept separately and subject to technical and organisational measures").
- **Not measuring collateral damage.** Precision on a labelled set is not the same as collateral damage on the full corpus; the labelled set rarely covers all contexts. Hand-review 50 drops per release.
- **Calibrating thresholds on the train corpus.** The corpus distribution moves between releases; the threshold moves with it; "the same filter" silently becomes a different filter. Calibrate on a fixed held-out labelled set whose version is pinned.
- **Measuring toxicity with aggregate AUC only.** A model with 0.95 overall AUC and 0.70 AUC on an identity subgroup is the published definition of "unintended bias" (Borkan et al. 2019). Report per-subgroup AUC or do not report AUC.
- **Mixing `replace_tagged` and `surrogate` within the same corpus.** Some docs end up as `[PERSON_0]` and others as realistic names; downstream IE training confuses the two as two separate labels. Pick one strategy and record it in the manifest.
- **Using `en_core_web_sm` for `PERSON` detection "because it's smaller".** `en_core_web_sm` has 2–3 F1 lower PERSON recall than `en_core_web_lg`; false negatives are PII leaks. The model size difference is 50 MB vs 500 MB; pay the disk.
- **Logging the raw PII values in the decision log.** The audit log is itself PII if it contains the detected spans verbatim. Log `(doc_id, span_start, span_end, entity, confidence, source)` — the *span* locator and the *type*, not the *value*. If the original doc is retained, the locator is enough to re-derive the value for authorised review.
- **Running toxicity classifier at document granularity on 100k-token docs.** Detoxify's backbone is a 512-token BERT; longer docs get truncated and the classifier scores only the head. Run at paragraph granularity, report per-paragraph scores, and roll up to document only for a kill decision.
- **Skipping regex *and* NER on URLs because "URLs are not PII".** A URL containing `?user=alice&token=abc123` very much is. Add URL detection + parameter-stripping as a dedicated detector; it typically catches 1–3 % of corpus leaks.

## 6. References

Primary sources used in this solution:

- Microsoft Presidio. <https://microsoft.github.io/presidio/> — reference open-source PII analyser + anonymiser; the layered recognizer architecture mirrored in §2.
- Microsoft Presidio paper: Mendels, O., Peled, C., Vaisman Levy, N. (2021). *Presidio: Context aware, pluggable and customizable data protection and anonymization service for text and images*. <https://arxiv.org/abs/2107.01775> — the recognizer-registry pattern and context-boost mechanism used in §2.4 and §2.6.
- Borkan, D. et al. (2019). *Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification*. WWW. <https://arxiv.org/abs/1903.04561> — the Jigsaw Unintended Bias dataset and the Subgroup-AUC / BPSN / BNSP fairness metrics used in §2.8 and §4.
- Hanu, L., Unitary team (2020). *Detoxify*. <https://github.com/unitaryai/detoxify> — the Detoxify family of classifiers (Apache-2.0) wrapped in §2.8.
- Dodge, J. et al. (2021). *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus*. EMNLP. <https://arxiv.org/abs/2104.08758> — the fairness analysis of C4's naughty-word list; the principal public argument against static bad-word filters in §1.
- Honnibal, M., Montani, I. (2017+). *spaCy*. <https://spacy.io/> — the NER backbone used in §2.5; `en_core_web_lg` is the reference English model.
- Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL. <https://arxiv.org/abs/1810.04805> — the backbone family for `unitary/toxic-bert` and `dslim/bert-base-NER`.
- Tjong Kim Sang, E., De Meulder, F. (2003). *Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition*. CoNLL. <https://aclanthology.org/W03-0419/> — the NER span-exact evaluation convention used in §2.10.
- `Faker` — <https://faker.readthedocs.io/> — the synthetic-surrogate library wrapped in §2.7.
- ISO 13616 (IBAN check-digit mod-97). <https://www.iso.org/standard/41031.html> — the checksum used in §2.3.
- Luhn, H. (1954). US Patent 2,950,048. — the credit-card check digit used in §2.3.
- Perspective API (Jigsaw / Google). <https://perspectiveapi.com/> — the alternative hosted toxicity classifier; shares the Jigsaw Unintended Bias dataset lineage with Detoxify.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML02 Data Poisoning* applies when adversaries publish content designed to (a) evade PII detectors (e.g., SSNs split with zero-width joiners) or (b) manipulate toxicity scoring via adversarial paraphrasing; mitigation is the decision-log cross-tab in §2.9 and the regular labelled-set re-eval in §2.10. *ML06 AI Supply Chain Attack* applies to the Detoxify/toxic-bert checkpoints; pin by SHA.
- MITRE ATLAS — <https://atlas.mitre.org/> — *AML.T0020 Poison Training Data* maps to PII-evasion; *AML.T0057 LLM Data Leakage* is downstream (if the LM trained on this corpus memorises PII) and is the motivation for strict redaction.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: entities covered, detectors used, NER backbone version, toxicity model checkpoint, per-head thresholds, anonymisation strategy, labelled-set version. *Measure*: per-entity P/R/F1, per-toxicity-subgroup AUC, fraction of docs with any redaction, fraction blocked by toxicity filter. *Manage*: decision log retained; labelled set re-versioned on every detector change; dataset card updated.
- Additional cross-reference: NIST Privacy Framework (<https://www.nist.gov/privacy-framework>) and GDPR Art. 4(5) (pseudonymisation vs anonymisation) are the policy framings under which the "seal the mapping or discard it" rule in §2.7 is non-optional.
