# exercise-04 — Language Identification and Routing

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-102-classical-nlp/exercises/exercise-04-language-identification-and-routing.md`.
> This solution targets the learning goals that filename implies: detect the language of an incoming piece of text, and **route** it to the appropriate downstream pipeline — including an explicit "unknown / mixed / too short" fallback path — rather than silently coercing it into a single-language model.

## 1. Solution overview

Language identification (LID) is a short-document classification problem with three properties that the design must take seriously:

1. **The input is often too short to be sure.** Chat messages of 1-3 tokens cannot be reliably classified; confusables ("ok", numbers, emoji-only) are language-agnostic. A production LID must emit a **confidence** and a caller must have an *abstention* route.
2. **The label set includes "mixed".** Code-switched text (English + Hindi in the same message, or English + Chinese) will confuse a single-label model. Either train a multilabel head, segment the input and classify spans, or explicitly route "below threshold" to a mixed-handling path.
3. **The default "just use the top prediction" is the wrong default.** Routing without a threshold means a 51 %-confidence Portuguese message gets sent to the English summariser — the downstream model will happily produce fluent nonsense.

Two production-grade open-weights LID choices:

| Model | Scope | License / weights | Notes |
|---|---|---|---|
| **fastText `lid.176`** | 176 languages | CC BY-SA 3.0; pre-trained weights <https://fasttext.cc/docs/en/language-identification.html> | Original fastText paper (Joulin et al. 2017); fast, ships with probabilities per class, well-tested |
| **CLD3** | 107 languages | Apache 2.0; <https://github.com/google/cld3> | Compact neural model, Chrome's built-in LID |

This solution uses fastText `lid.176` for the model and layers a routing wrapper around it. The wrapper (a) normalises input, (b) emits top-k labels + confidences, (c) applies a confidence threshold, (d) returns a routing decision (`Route.handle_with(lang)`, `Route.mixed`, `Route.abstain`), and (e) logs everything the downstream caller needs to debug a mis-route.

## 2. Worked implementation

### 2.1 Install and load the model

```python
# pyproject.toml excerpt
# fasttext == 0.9.2
# Model weight: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
#   (~126 MB). Pin the exact SHA when downloaded.
```

```python
# lid/model.py
import fasttext
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load(model_path: str = "./lid.176.bin") -> fasttext.FastText._FastText:
    assert Path(model_path).exists(), (
        f"Download lid.176.bin to {model_path}. See fasttext.cc docs."
    )
    return fasttext.load_model(model_path)
```

Lazy loading via `lru_cache` is deliberate — the model is ~126 MB and you do not want to pay that at import time or re-pay it per request.

### 2.2 Prediction wrapper

```python
# lid/predict.py
import re
from dataclasses import dataclass
from .model import load

# fastText reports labels as `__label__en`, `__label__fr`, etc.
_LABEL_PREFIX = "__label__"
_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Strip control chars and collapse whitespace. fastText expects single-line input."""
    text = text.replace("\n", " ").replace("\r", " ")
    return _WS_RE.sub(" ", text).strip()


@dataclass(frozen=True)
class LangPrediction:
    lang: str        # ISO 639-1 if two-letter available
    confidence: float


def predict_topk(text: str, k: int = 3, model_path: str = "./lid.176.bin"
                 ) -> list[LangPrediction]:
    model = load(model_path)
    cleaned = _clean(text)
    if not cleaned:
        return []
    labels, probs = model.predict(cleaned, k=k)
    return [LangPrediction(lbl.removeprefix(_LABEL_PREFIX), float(p))
            for lbl, p in zip(labels, probs)]
```

Two things the wrapper fixes for you:

- **Newlines.** `model.predict` raises on newlines. The clean step turns them into spaces.
- **Label prefix.** fastText returns `__label__en`; the caller wants `en`. Stripping the prefix at the boundary is tidier than letting it leak downstream.

### 2.3 Routing

```python
# lid/router.py
from dataclasses import dataclass
from enum import Enum
from .predict import predict_topk, LangPrediction


class RouteKind(Enum):
    HANDLE = "handle"       # confident single-language
    MIXED = "mixed"         # multiple high-probability languages
    ABSTAIN = "abstain"     # low confidence or empty input


@dataclass(frozen=True)
class Route:
    kind: RouteKind
    lang: str | None
    confidences: list[LangPrediction]  # top-k, for logging / debugging


# Default routing config. Tune on your traffic; see §3.
HANDLE_THRESHOLD = 0.70   # top-1 confidence to accept a single language
MIXED_GAP = 0.20          # if p1 - p2 < MIXED_GAP AND p2 >= 0.25 → mixed
MIN_CHARS = 10            # shorter than this: always abstain unless p1 >= 0.95


def route(text: str, handle_threshold: float = HANDLE_THRESHOLD,
          mixed_gap: float = MIXED_GAP, min_chars: int = MIN_CHARS) -> Route:
    preds = predict_topk(text, k=3)
    if not preds:
        return Route(RouteKind.ABSTAIN, None, [])

    p1, *rest = preds
    cleaned_len = len(text.strip())

    # Short input: require a very high top-1 to accept.
    if cleaned_len < min_chars and p1.confidence < 0.95:
        return Route(RouteKind.ABSTAIN, None, preds)

    # Mixed detection: two high-ish probabilities that are close together.
    if rest and (p1.confidence - rest[0].confidence) < mixed_gap and rest[0].confidence >= 0.25:
        return Route(RouteKind.MIXED, None, preds)

    if p1.confidence >= handle_threshold:
        return Route(RouteKind.HANDLE, p1.lang, preds)

    return Route(RouteKind.ABSTAIN, None, preds)
```

Thresholds have defaults, not magic numbers: the three constants are *tuning targets*. On your traffic, measure (a) how much of the stream falls below each threshold, (b) what the downstream pipeline does with routed-but-wrong-language input, and (c) what humans say about routed-but-mixed content. Adjust accordingly.

### 2.4 Downstream dispatch

```python
# lid/dispatch.py
from .router import route, RouteKind

HANDLERS: dict[str, callable] = {
    "en": lambda t: _en_pipeline(t),
    "es": lambda t: _es_pipeline(t),
    "fr": lambda t: _fr_pipeline(t),
    # ... populate per supported language
}

FALLBACK = lambda t, reason: _multilingual_pipeline(t, reason=reason)


def process(text: str) -> dict:
    r = route(text)
    if r.kind == RouteKind.HANDLE and r.lang in HANDLERS:
        return {"route": r.kind.value, "lang": r.lang,
                "confidences": [(p.lang, p.confidence) for p in r.confidences],
                "result": HANDLERS[r.lang](text)}
    # Mixed, abstain, or language we don't support → multilingual fallback.
    reason = r.kind.value if r.kind != RouteKind.HANDLE else "unsupported_lang"
    return {"route": reason,
            "lang": r.lang,
            "confidences": [(p.lang, p.confidence) for p in r.confidences],
            "result": FALLBACK(text, reason=reason)}
```

Two production details:

- **Log the top-k with probabilities on every request.** The single most valuable signal for debugging a mis-route is the exact probability vector the router saw. Store it alongside the result for at least sampling-rate retention.
- **Have a multilingual fallback.** XLM-R-based or mBART-based multilingual pipelines handle "I don't know" gracefully; monolingual pipelines do not. If you have no multilingual option, the fallback must at minimum return "could not process" and escalate, not silently guess.

### 2.5 End-to-end example

```python
for text in [
    "How do I reset my password?",
    "¿Cómo cambio mi contraseña?",
    "ok",                               # too short
    "Reset password por favor",         # code-switched
    "Comment réinitialiser mon mot de passe?",
    "🙃🙃🙃",                             # content-free
]:
    r = route(text)
    print(f"{r.kind.value:7s} lang={r.lang!s:4s}  topk={r.confidences[:2]}")
```

On a well-tuned router, lines 1-5 produce: `handle en`, `handle es`, `abstain`, `mixed`, `handle fr`. The emoji line produces `abstain`.

## 3. Validation steps

1. **Per-language test set.** Collect 100-500 short messages per supported language from a public source (e.g., Tatoeba sentences <https://tatoeba.org>, Universal Dependencies). Measure accuracy, abstain rate, and mis-route rate per language. Report the confusion matrix.
2. **Code-switched test set.** Build a small (50-200) code-switched set — e.g., English/Hindi, English/Spanish — and confirm the router produces `MIXED` rather than silently picking one. The LinCE benchmark (<https://ritual.uh.edu/lince/>) is the standard source if you want a published set.
3. **Short-input robustness.** Measure accuracy on messages of 1, 2, 3, 5, 10 characters. Confidence should be flat near 50/50 on 1-2 chars; the router should abstain. If it does not, raise `min_chars` or `handle_threshold`.
4. **Threshold-sweep table.** For `handle_threshold ∈ {0.5, 0.6, 0.7, 0.8, 0.9}` report (abstain rate, mis-route rate, correct-handle rate). This table is the tuning deliverable; pick the point where mis-route rate hits your SLO.
5. **Drift check.** Re-run the above on a sample of production traffic once a week. Script it so a 2 %+ swing in per-language accuracy pages someone.
6. **Fallback coverage.** Confirm every `RouteKind.MIXED` and `RouteKind.ABSTAIN` path reaches a defined handler (multilingual pipeline, human-review queue, "we don't support this" polite error). Any un-covered branch is an invisible failure mode.
7. **Dependency pinning.** `fasttext` v0.9.2 is current; the model file `lid.176.bin` must be pinned by SHA-256, not by URL alone (fastText's model URLs are stable but the policy should be "verify the hash").

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Model choice & justification** | Picks fastText `lid.176` or CLD3 with a reason tied to language coverage and licence; alternative considered | Picks a model, no comparison | Hand-rolls a trigram classifier for the exercise |
| **Confidence surfaced** | API exposes top-k + probabilities, not just top-1 | Top-1 only | No probabilities at all |
| **Abstention path** | Explicit `ABSTAIN` route with a defined downstream handler | Returns `None` quietly | Picks top-1 regardless |
| **Mixed-language handling** | Explicit `MIXED` route with defined behaviour; tested on code-switched data | Handles mixed as top-1 | Code-switched content silently mis-routed |
| **Threshold hygiene** | Thresholds named, documented, swept; chosen point tied to SLO | Thresholds hardcoded | Thresholds absent |
| **Short-input robustness** | Different behaviour below `min_chars` with justification | Same thresholds everywhere | Any-length inputs trusted equally |
| **Observability** | Top-k + chosen route logged per request (sampled OK) | Only top-1 logged | No logging |
| **Fallback coverage** | Every non-HANDLE route reaches a defined handler | Some routes undefined | Fallback crashes or returns nothing |
| **Reproducibility** | `fasttext` version + `lid.176.bin` SHA pinned | Library pinned only | Not pinned |
| **Responsible deployment** | Mis-routing framed as a documented risk per NIST AI RMF (Measure/Manage); adversarial inputs (e.g. homoglyph attacks that spoof language) considered per OWASP ML Top 10 and MITRE ATLAS | Mentions one | No risk analysis |

## 5. Common mistakes

- **Trusting top-1 unconditionally.** A 51 %-confidence prediction between two languages is not the same as a 99 %-confidence prediction, and the router must act on the difference. The fix is a threshold and an abstain path, not a better model.
- **No log of the probability vector.** When a user complains about a mis-route, the only diagnostic that matters is "what did the model actually see and score?". Log the top-k on every request or you cannot debug production.
- **Handling mixed as top-1.** Code-switched text systematically beats a single-language pipeline. A quiet top-1 route to English for an English/Spanish message produces an English-language response to a Spanish-speaking user. Treat mixed as its own route.
- **Short-input overconfidence.** fastText will emit a confident label for `"ok"` or `"hi"` or `"yes"` — but these tokens are *multilingual*. The router has to know this; the model does not. Set a `min_chars` guard and require higher confidence below it.
- **Cleaning the text with the wrong normaliser.** Stripping emoji before LID is usually fine. Stripping diacritics (`cafés` → `cafes`) is not — it makes French look like English. Apply NFC normalisation only; do not case-fold or strip accents before LID.
- **Using `langdetect` for latency-sensitive production.** `langdetect` is widely cited but is a pure-Python port of a 2010 Google tool with no official maintainer and non-deterministic behaviour unless you set a seed. fastText `lid.176` and CLD3 are the production options; `langdetect` belongs in notebooks.
- **No per-language accuracy report.** Reporting one accuracy number hides the long tail. A model with 98 % English accuracy and 40 % Norwegian accuracy is a disaster for Norwegian users and no one sees it in the headline metric. Always report per-language.
- **Not SHA-pinning the model file.** The fastText repo hosts weights by URL; if the hosting org replaces the file (e.g., for a bug fix), your pipeline's behaviour changes silently. Verify a SHA-256 on download and keep it in `requirements` or CI.
- **Routing to a monolingual pipeline with no fallback.** When LID abstains, the downstream monolingual pipeline will often return gibberish. The fallback path must exist; a `KeyError` on an unsupported `lang` key is a shipping bug, not a nice-to-have.
- **Treating LID as a one-time choice.** Traffic drifts; new languages appear; code-switch rates shift. LID deserves periodic evaluation against a labelled sample of live traffic, not just the test set you built once.

## 6. References

Primary sources used in this solution:

- fastText *Language Identification* — <https://fasttext.cc/docs/en/language-identification.html> — the `lid.176` model page, including class list, licence, and the direct link to the weights binary.
- Joulin, A., Grave, E., Bojanowski, P., Mikolov, T. (2017). *Bag of Tricks for Efficient Text Classification*. EACL. <https://aclanthology.org/E17-2068/> — the fastText classifier architecture the LID model is trained on.
- Google CLD3 — <https://github.com/google/cld3> — alternative production LID model.
- Universal Dependencies treebanks — <https://universaldependencies.org/> — standard source for per-language evaluation data.
- LinCE benchmark — <https://ritual.uh.edu/lince/> — curated code-switching corpora for mixed-language evaluation.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — "Input Manipulation" covers homoglyph / visually-confusable attacks that can spoof language in text.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue; evasion-via-mixed-script-or-language is documented as an input-side adversarial technique against ML services.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — Measure / Manage functions require per-language accuracy metrics and documented fallback behaviour, both of which are spelled out in §3 and §4 above.
