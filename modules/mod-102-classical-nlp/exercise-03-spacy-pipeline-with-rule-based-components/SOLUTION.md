# exercise-03 — spaCy Pipeline with Rule-Based Components

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-102-classical-nlp/exercises/exercise-03-spacy-pipeline-with-rule-based-components.md`.
> This solution targets the learning goals that filename implies: assemble a working spaCy pipeline that mixes the statistical components shipped in `en_core_web_sm` with **rule-based** components — `Matcher`, `PhraseMatcher`, `EntityRuler`, and a custom pipeline component — and show where rule-based layers belong relative to the statistical ones.

## 1. Solution overview

Rule-based NLP earns its keep in exactly three situations:

| Situation | Why rules over a statistical model |
|---|---|
| **Known, enumerable surface forms** | Product names, SKUs, drug names, legal citation formats — a `PhraseMatcher` is fast and perfectly precise |
| **Label gaps in the pre-trained model** | `en_core_web_sm`'s NER misses tickers like `$MSFT`, custom entity types like `DOSE`, or evolving slang — rules add new label types without retraining |
| **Guardrail / override** | A deployed NER model that occasionally mis-labels `Washington` as `GPE` when the context says `PERSON` — rules placed *after* NER correct known errors |

spaCy's architecture makes the ordering explicit: the pipeline is a list of components (`nlp.pipe_names`) and each component reads/writes `Doc` attributes. Placement matters. Two canonical placements:

- **`EntityRuler` before NER** (`nlp.add_pipe("entity_ruler", before="ner")`) — the ruler seeds entities *before* the statistical NER runs; the NER model then uses those spans as hard constraints (it will not predict over them).
- **`EntityRuler` after NER** (`...after="ner"`) — the ruler augments or overrides the NER output. New entity types are added; duplicate spans use the first-written label.

The deliverable is a `build_pipeline()` function that returns an `nlp` with: (1) a default statistical base (`en_core_web_sm`), (2) a `PhraseMatcher`-backed custom component for a vocabulary of product names, (3) an `EntityRuler` for a new label type with regex + token patterns, and (4) a custom component that adds an extension attribute (`doc._.has_sensitive_term`). Two worked examples showing which placement choice changes behaviour.

## 2. Worked implementation

### 2.1 Setup

```python
# pyproject.toml (excerpt)
# spacy >= 3.7, < 4
# No model download at install time — the component loader fetches it lazily.
```

```python
import spacy
assert spacy.__version__.startswith("3."), "This solution is tested on spaCy 3.x"
```

spaCy 3's pipeline API (`add_pipe`, `pipe_analysis`, `@Language.component`) is stable within `3.x`. A pipeline written for 2.x will not run unchanged — the registration decorator and the component-ordering API both changed. See <https://spacy.io/usage/v3> for the migration notes if you inherit older code.

### 2.2 The base pipeline

```python
# pipeline.py
import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span

PRODUCT_TERMS = ["Acme Widget Pro", "Acme Widget Lite", "Acme Toolkit"]
SENSITIVE_TERMS = ["termination", "layoff", "redundancy"]


def build_pipeline(model: str = "en_core_web_sm") -> Language:
    nlp = spacy.load(model)
    _add_product_matcher(nlp)
    _add_entity_ruler(nlp)
    _add_sensitive_flag(nlp)
    return nlp
```

### 2.3 `PhraseMatcher` as a pipeline component

The `PhraseMatcher` is built for large exact-phrase vocabularies. It is O(len(doc)) per match thanks to the Aho-Corasick-like trie backing it, and far faster than compiling the same vocabulary as regex alternation.

```python
# Register a custom extension once at import time; spaCy will raise on
# double-registration otherwise.
if not Doc.has_extension("products"):
    Doc.set_extension("products", default=[])


@Language.factory("product_matcher")
def make_product_matcher(nlp, name):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(t) for t in PRODUCT_TERMS]
    matcher.add("PRODUCT", patterns)
    return _ProductMatcherComponent(matcher)


class _ProductMatcherComponent:
    def __init__(self, matcher: PhraseMatcher):
        self.matcher = matcher

    def __call__(self, doc: Doc) -> Doc:
        spans = [Span(doc, start, end, label="PRODUCT")
                 for _, start, end in self.matcher(doc)]
        # Filter overlapping spans; longest wins.
        spans = spacy.util.filter_spans(spans)
        doc._.products = spans
        return doc


def _add_product_matcher(nlp: Language) -> None:
    nlp.add_pipe("product_matcher", before="ner")
```

Two deliberate choices:

- **`attr="LOWER"`** matches on the lowercased surface form. This is what you want for product-name recognition in user chat, where `acme widget pro` and `Acme Widget Pro` should both match. Match on `attr="ORTH"` instead if case is load-bearing (e.g., `JavaScript` vs `javascript`).
- **`filter_spans`** is non-negotiable for overlapping patterns. Without it, two overlapping products (e.g. `"Acme Widget"` and `"Acme Widget Pro"`) both annotate the same doc and later components see an inconsistent `doc.ents`.

### 2.4 `EntityRuler` for a new entity label

The `EntityRuler` supports token-level patterns (full access to spaCy's attribute matcher — lemma, POS, shape, regex per token) and phrase-level patterns.

```python
EQUIPMENT_PATTERNS = [
    # Token pattern: a sequence of three digits followed by "mL" or "mg", case-insensitive.
    {"label": "DOSE", "pattern": [
        {"IS_DIGIT": True},
        {"LOWER": {"IN": ["ml", "mg", "g"]}},
    ]},
    # Phrase pattern: well-known equipment names.
    {"label": "EQUIPMENT", "pattern": "centrifuge"},
    {"label": "EQUIPMENT", "pattern": "spectrometer"},
    # Regex-bearing pattern: internal ticket IDs like "TICK-12345".
    {"label": "TICKET", "pattern": [{"TEXT": {"REGEX": r"^TICK-\d{4,6}$"}}]},
]


def _add_entity_ruler(nlp: Language) -> None:
    ruler = nlp.add_pipe("entity_ruler", before="ner",
                         config={"overwrite_ents": False,
                                 "validate": True})
    ruler.add_patterns(EQUIPMENT_PATTERNS)
```

Placement knob: **before NER** means these labels are pinned and the statistical NER will not predict over them. **After NER** (`after="ner"`) with `overwrite_ents=True` means the ruler can replace NER-assigned labels. Pick based on who you trust more on this label — the pre-trained NER or the rule author. For label types the pre-trained model does not know about (e.g. `DOSE`, `TICKET`), either placement works and "before NER" is slightly safer because it prevents NER from overwriting with a nearby `CARDINAL`.

### 2.5 Custom guardrail component

A one-function component that flags any document containing a sensitivity-trigger term. Pure demonstration of the component API.

```python
if not Doc.has_extension("has_sensitive_term"):
    Doc.set_extension("has_sensitive_term", default=False)


@Language.component("sensitive_flag")
def sensitive_flag(doc: Doc) -> Doc:
    lowered = {t.lower_ for t in doc}
    doc._.has_sensitive_term = any(w in lowered for w in SENSITIVE_TERMS)
    return doc


def _add_sensitive_flag(nlp: Language) -> None:
    nlp.add_pipe("sensitive_flag", last=True)
```

### 2.6 End-to-end usage

```python
nlp = build_pipeline()
print(nlp.pipe_names)
# ['tok2vec', 'tagger', 'parser', 'entity_ruler', 'product_matcher',
#  'ner', 'attribute_ruler', 'lemmatizer', 'sensitive_flag']

doc = nlp("We shipped 250 mg of the Acme Widget Pro to spectrometer unit TICK-10234. "
          "HR warned about a possible layoff.")
for ent in doc.ents:
    print(ent.text, ent.label_)
# 250 mg       DOSE
# Acme Widget Pro PRODUCT
# spectrometer EQUIPMENT
# TICK-10234   TICKET
print([s.text for s in doc._.products])  # ['Acme Widget Pro']
print(doc._.has_sensitive_term)          # True
```

### 2.7 Placement-matters worked example

Swap `before="ner"` to `after="ner"` on the `product_matcher` and the same sentence may now show `Acme Widget Pro` as `ORG` instead of `PRODUCT` whenever the statistical NER happens to tag it as an organisation first. The rule still runs — but `doc.ents` is already populated and spaCy's default conflict resolution keeps the NER span. Decide the placement in writing; do not let defaults decide it for you.

## 3. Validation steps

1. **Component order assertion.** After `build_pipeline()`, assert `nlp.pipe_names` matches an expected list. The test will fail loudly if a future contributor moves `entity_ruler` after the NER or forgets to register `sensitive_flag`.
2. **Factory registration round-trip.** Serialise the pipeline with `nlp.to_disk()` and reload. The custom components must round-trip; `@Language.factory` registrations must be importable from the same module path at load time.
3. **Placement test.** Build two copies of the pipeline — one with `product_matcher` before NER, one after. Score the same sentence that has a product-name/ORG ambiguity (`"Acme Widget Pro filed earnings"`) and assert each produces the expected label. This test is the placement rationale made executable.
4. **Pattern validation.** Call `ruler.add_patterns(..., validate=True)` or configure `validate=True`; spaCy will raise on malformed patterns at load time rather than silently ignoring them at match time.
5. **No overlapping entities.** Assert `len(doc.ents) == len(set((e.start, e.end) for e in doc.ents))` for a corpus of 100 docs; overlapping entities are a legal state in `Span`-land but illegal in `doc.ents`, so the invariant tells you the merge logic is intact.
6. **Performance smoke.** Run the full pipeline on 1 000 lines of representative text; record wall time. The `PhraseMatcher` and `EntityRuler` layers should add single-digit milliseconds per thousand tokens on CPU. Blowups usually mean a token pattern with an unbounded quantifier.
7. **Reproducibility.** Pin `spacy` and the exact model version (e.g. `en_core_web_sm-3.7.1`). Reproducing a model's NER outputs requires the exact minor-version model wheel, not just `en_core_web_sm` latest.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Component registration** | Uses `@Language.component` or `@Language.factory` correctly; serialisable | Works in a notebook, fails on `to_disk` | Monkey-patched `nlp.pipeline` directly |
| **Placement rationale** | States why each custom component is before/after NER; placement test present | States placement, no test | Added everywhere by default |
| **Vocabulary vs patterns** | Uses `PhraseMatcher` for exact phrases, `EntityRuler` token patterns for structural matches | Uses one tool for everything | Uses regex strings concatenated into patterns |
| **Attribute hygiene** | Extensions registered once, guarded against double-registration | Registered at module top-level, crashes on reimport | No `has_extension` check |
| **Conflict resolution** | `filter_spans` or `overwrite_ents` chosen deliberately; documented | Mentions one | Overlapping ents leak into `doc.ents` |
| **Pattern validation** | `validate=True` on the ruler; patterns tested on positive + negative examples | Positive tests only | No validation; silent no-ops |
| **Performance** | Smoke test on 1 k docs; latency recorded | Mentioned | Unknown |
| **Reproducibility** | Pins `spacy` and the exact model wheel version | Pins library only | Not pinned |
| **Responsible deployment** | Notes PII in the sensitive-flag component (NIST AI RMF Measure); guards against adversarial inputs (ATLAS *AML.T0049*); patterns reviewed for over-matching | Mentions one | Ignored |

## 5. Common mistakes

- **Adding components without a placement.** `nlp.add_pipe(...)` with no `before=`/`after=`/`first=`/`last=` arg appends to the end of the pipeline, which is almost always wrong for rulers that should run before NER. Always name the placement.
- **Overlapping spans in `doc.ents`.** spaCy raises on overlapping entities assigned to `doc.ents`, but overlapping spans from a `Matcher`/`PhraseMatcher` are easy to produce. Always run `spacy.util.filter_spans` before setting `doc.ents` from your own matches.
- **Setting `doc.ents` directly without preserving existing entities.** Assigning `doc.ents = my_spans` *replaces* the entity list. If you want to add, do `doc.ents = list(doc.ents) + filtered_new_spans` after de-dup.
- **Registering extensions at request time.** `Doc.set_extension` throws on re-registration. Guard with `Doc.has_extension(...)` at module import, not inside the per-doc function.
- **Using regex strings where token patterns belong.** `EntityRuler` token patterns are the right tool when you need structural constraints (POS, shape, lemma). Regex-only patterns bypass spaCy's tokeniser and will mis-match on hyphenated or apostrophised tokens.
- **Forgetting that the `PhraseMatcher` uses doc tokenisation.** If you build patterns from `nlp.make_doc("Acme Widget Pro")`, tokenisation differences between training and inference (different `prefix_search` / `suffix_search` settings) silently break matching. Keep tokeniser config identical between pattern creation and match time.
- **Mis-ordering `EntityRuler` and `NER`.** Before NER: ruler-pinned entities block NER predictions. After NER: ruler augments/overrides. Both are valid; the mistake is treating this as a default — document it per component.
- **Loading a different model wheel version in prod than in tests.** `en_core_web_sm` ships multiple minor-version wheels; a wheel refresh changes NER outputs subtly. Pin to a specific wheel URL in requirements, not to the alias.
- **Stuffing business logic into the pipeline.** The sensitive-flag component in §2.5 is intentionally stateless. A component that calls an external API, writes to disk, or mutates global state turns `nlp.pipe()` into something that no longer behaves like a pure function and breaks spaCy's batching assumptions. Put business logic downstream of `nlp()`.

## 6. References

Primary sources used in this solution:

- spaCy *Usage: Rule-Based Matching* — <https://spacy.io/usage/rule-based-matching> — `Matcher`, `PhraseMatcher`, `EntityRuler` APIs, `filter_spans`, `overwrite_ents`.
- spaCy *Usage: Processing Pipelines* — <https://spacy.io/usage/processing-pipelines> — `@Language.component` / `@Language.factory`, `add_pipe` placement, `pipe_analysis`, serialisation.
- spaCy *API: Language* and *API: PhraseMatcher* — <https://spacy.io/api/language> and <https://spacy.io/api/phrasematcher> — authoritative reference for the methods used in §2.
- spaCy *Migrating from v2.x* — <https://spacy.io/usage/v3> — notes on the factory / component-ordering API changes used here.
- spaCy *Models: en_core_web_sm* — <https://spacy.io/models/en> — identifies the statistical base pipeline and its NER label inventory.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — adversarial input handling applies to any production text pipeline; rules must be reviewed for over-matching that could be exploited.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue; *AML.T0049* (Exploit Public-Facing Application) and prompt/pattern injection surfaces apply to production spaCy deployments.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — Measure function covers the sensitivity-flag component (documented false-positive and false-negative rates); Govern covers the decision to ship rules vs retrain.
