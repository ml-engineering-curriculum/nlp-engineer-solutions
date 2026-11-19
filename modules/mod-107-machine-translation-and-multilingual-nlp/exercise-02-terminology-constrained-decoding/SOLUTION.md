# exercise-02 — Terminology-Constrained Decoding

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-107-machine-translation-and-multilingual-nlp/exercises/exercise-02-terminology-constrained-decoding.md`.
> The target learning outcomes implied by that filename: force an NMT decoder to **emit a specific target-language phrase** when a corresponding source phrase is present — the translation-industry "terminology injection" problem — using (a) **`force_words_ids` / `DisjunctiveConstraint`** in Hugging Face's `constrained_beam_search`, (b) **prefix-constrained decoding** when the constraint depends on the already-generated prefix, and (c) **validator + re-ask** as a fallback — and report both **constraint satisfaction rate** and **BLEU / chrF** so the quality tradeoff is visible.

## 1. Solution overview

Terminology constraints are the first concrete demand from any professional MT buyer: "when the source says `heparin`, the German output must say `Heparin`, not `Blutverdünner`." The three failure modes that decide whether your solution is honest:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Soft constraints that only work most of the time** | Prompt-engineered "please use the term X" works ~70–90 % with large models and ~40–60 % with small NMT models. The remainder is exactly the rows a glossary customer wanted to control | Pair a hard mechanism (`force_words_ids` with disjunctive inflections, lexically-constrained beam search) with a post-hoc validator; log misses rather than silently passing them |
| **Hard constraints that destroy fluency** | A naive "copy the English term" injection produces outputs that violate German case, number, or word order. BLEU drops 2–4 points; human raters call the translations ungrammatical | Allow a **disjunction over inflections** (`Heparin` / `Heparins` / `Heparine`) and let beam search choose the one that fits. Never force a single surface form |
| **Constraint × beam-size interactions** | Constrained beam search enlarges the effective beam (`num_beams × constraint_states`). A `num_beams=4` run that worked unconstrained can run out of memory under constraints, or starve non-constrained beams | Set `num_beams` ≥ `num_beam_groups ≥ 2 × max_simultaneous_constraints`; benchmark memory under the worst-case constraint set before shipping |

The deliverable is a constrained-decoding harness with four components:

1. **Terminology bilingual glossary** — `{src_term: [tgt_term, tgt_inflections, …]}`. The inflection list is the mechanism that keeps BLEU from collapsing.
2. **Constraint builder** — turns a `(source_sentence, glossary)` pair into a list of `transformers.PhrasalConstraint` / `DisjunctiveConstraint` objects.
3. **Constrained generator** — `model.generate(..., constraints=..., num_beams=5)`; Hugging Face's `constrained_beam_search` enforces that every constraint is satisfied in the returned sequence.
4. **Validator + reporter** — parses the output, confirms every constraint fired, logs the rate; falls back to unconstrained + re-ask-with-prompt when a constraint fails under memory pressure.

Reference baseline (unconstrained + prompt) — the honest comparison. You cannot defend hard constraints without showing how often the soft version gets it wrong.

## 2. Worked implementation

### 2.1 Project layout

```
term_constrain/
  glossary.py        # bilingual term dictionary + inflection expansion
  constraints.py     # build transformers.PhrasalConstraint / DisjunctiveConstraint
  generate.py        # constrained + unconstrained decoders
  validator.py       # parse outputs, check satisfaction
  metrics.py         # satisfaction rate, BLEU/chrF, prompt-baseline comparison
  run.py             # CLI
  tests/
    test_constraints.py
    test_validator.py
    test_bleu_delta.py
  requirements.txt
```

### 2.2 The glossary — inflection awareness is the whole trick

```python
# glossary.py
from dataclasses import dataclass, field


@dataclass
class TermEntry:
    """A source term and its allowed target surface forms."""
    src: str
    tgt: list[str] = field(default_factory=list)  # each list entry is one allowed form

    def all_forms(self) -> list[str]:
        return self.tgt


# Minimal en→de medical glossary used in the worked example
MEDICAL_EN_DE = [
    TermEntry("heparin", ["Heparin", "Heparins", "Heparine", "Heparinen"]),
    TermEntry("aspirin", ["Aspirin", "Aspirins"]),
    TermEntry("patient",  ["Patient", "Patienten", "Patienten.", "Patientin", "Patientinnen"]),
]
```

Points worth defending in review:

- **Inflection list, not a single form.** `Heparin` (nominative) is wrong after `des` (genitive → `Heparins`); `Patient` is wrong for a female patient (`Patientin`). Forcing a single form costs BLEU 2–4 and alienates professional translators who read the output as mechanical.
- **Punctuation in the allowed forms.** `"Patienten."` captures sentence-final use; the decoder can then emit the period inside the constraint without another beam-split. Optional but measurably helps.
- **No uppercased / lowercased duplicates.** German nouns are always capitalised; lower-casing would introduce false constraints that satisfy "contains substring" but produce ungrammatical output. If your validator is case-insensitive, your rate is a lie.

### 2.3 Building `transformers` constraints

```python
# constraints.py
from transformers import PhrasalConstraint, DisjunctiveConstraint
from .glossary import TermEntry


def build_constraints(
    source_sentence: str,
    glossary: list[TermEntry],
    tokenizer,
) -> list:
    """
    For each glossary term that appears (case-insensitive) in the source, produce a
    DisjunctiveConstraint over its target inflections. The decoder must emit at
    least one of the inflections somewhere in the output.
    """
    src_lower = source_sentence.lower()
    constraints = []
    for entry in glossary:
        if entry.src.lower() not in src_lower:
            continue

        # Each allowed form is tokenised into a token-id sequence. DisjunctiveConstraint
        # takes a list of such sequences and is satisfied when ANY of them appears.
        token_id_seqs = [
            tokenizer(form, add_special_tokens=False).input_ids
            for form in entry.all_forms()
        ]
        # Drop empty encodings (tokeniser could not encode a form at all)
        token_id_seqs = [s for s in token_id_seqs if s]
        if not token_id_seqs:
            continue

        if len(token_id_seqs) == 1:
            # Single surface form → PhrasalConstraint is simpler and uses less memory
            constraints.append(PhrasalConstraint(token_id_seqs[0]))
        else:
            constraints.append(DisjunctiveConstraint(token_id_seqs))
    return constraints
```

- **`add_special_tokens=False`.** Otherwise BOS tokens leak into the constraint, and the decoder can never satisfy it because BOS only appears at position 0.
- **Case-insensitive source match, case-correct target forms.** The glossary lookup does not care about case on the source side; the target-side forms are rigid. This is the only combination that is both ergonomic and grammatical.
- **Fallback to `PhrasalConstraint` when disjunction size is 1.** Disjunctions over a single sequence inflate memory; the flat `PhrasalConstraint` path is cheaper.

### 2.4 Constrained generation

```python
# generate.py
import torch
from transformers import GenerationConfig


def translate_constrained(
    model,
    tokenizer,
    source: str,
    constraints: list,
    tgt_lang_code: str | None = None,
    num_beams: int = 5,
    max_new_tokens: int = 256,
) -> str:
    """One-sentence constrained translation. See generation.py in exercise-01 for the
    unconstrained path. Note: constrained_beam_search does NOT support sampling."""
    inputs = tokenizer(source, return_tensors="pt", truncation=True, max_length=512)
    gen_cfg = GenerationConfig(
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        length_penalty=1.0,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if tgt_lang_code is not None:
        gen_cfg.forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            generation_config=gen_cfg,
            constraints=constraints if constraints else None,
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]
```

- **`constraints=` is the Hugging Face API for lexically-constrained beam search** (Post & Vilar 2018; Hu et al. 2019). It enlarges the beam internally into "banks" — one bank per number of satisfied constraints — and picks the best completion that reaches the all-satisfied bank by EOS. See the `transformers` constrained-generation docs and the WMT paper linked in §6.
- **No sampling with constraints.** `do_sample=True` + `constraints=...` raises in Hugging Face; the algorithm is a beam-search extension and cannot mix with stochastic sampling. If you need sampling + terms, fall back to validator + re-ask.
- **Empty-constraints branch.** Passing `constraints=[]` currently raises in some `transformers` versions; always coerce empty → `None` and dispatch to vanilla beam search.

### 2.5 Validator — never trust the constraint algorithm alone

```python
# validator.py
import re
from .glossary import TermEntry


def constraint_satisfied(output: str, entry: TermEntry) -> bool:
    """Match exact-word occurrence of any inflection; case-sensitive on the target
    side because German noun capitalisation matters."""
    for form in entry.all_forms():
        pattern = r"(?<!\w)" + re.escape(form) + r"(?!\w)"
        if re.search(pattern, output):
            return True
    return False


def report_satisfaction(output: str, source: str, glossary: list[TermEntry]) -> dict:
    applicable = [e for e in glossary if e.src.lower() in source.lower()]
    satisfied = [e for e in applicable if constraint_satisfied(output, e)]
    return {
        "applicable": len(applicable),
        "satisfied": len(satisfied),
        "missed_terms": [e.src for e in applicable if e not in satisfied],
    }
```

- **Word-boundary regex, not substring match.** `Heparin` is a substring of `Heparinase`; substring matching inflates the satisfaction rate.
- **Case-sensitive target match.** `heparin` lowercase is wrong in German; counting it as a hit lies about quality.
- **Report `missed_terms`.** The absolute count is never what the customer asks for; they ask "which terms are we still missing?" The misses are the backlog.

### 2.6 Validator + re-ask fallback

For models or settings that cannot use `constrained_beam_search` (sampling-only pipelines, hosted APIs without a constraints argument), the fallback is:

```python
# generate.py (continued)
def translate_with_reask(
    model, tokenizer, source: str, glossary, tgt_lang_code=None,
    max_attempts: int = 3,
) -> tuple[str, int]:
    from .constraints import build_constraints
    from .validator import report_satisfaction

    for attempt in range(max_attempts):
        if attempt == 0:
            out = translate_constrained(
                model, tokenizer, source,
                constraints=build_constraints(source, glossary, tokenizer),
                tgt_lang_code=tgt_lang_code,
            )
        else:
            # Soft re-ask: inject the required terms into the source with a marker
            # the model learned from its pre-training to copy literally.
            missed = report_satisfaction(out, source, glossary)["missed_terms"]
            augmented_source = source + " [TERMS: " + ", ".join(missed) + "]"
            out = translate_constrained(
                model, tokenizer, augmented_source,
                constraints=build_constraints(source, glossary, tokenizer),
                tgt_lang_code=tgt_lang_code,
            )
        if not report_satisfaction(out, source, glossary)["missed_terms"]:
            return out, attempt + 1
    return out, max_attempts
```

The re-ask is only useful for terms where `constrained_beam_search` failed under memory pressure (constraint dropped to stay within beam budget). On a well-sized beam it should fire <1 % of the time; if it fires more than that, raise `num_beams`.

### 2.7 CLI + metrics

```python
# run.py
import argparse, json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .glossary import MEDICAL_EN_DE
from .constraints import build_constraints
from .generate import translate_constrained
from .validator import report_satisfaction
from .metrics import corpus_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Helsinki-NLP/opus-mt-en-de")
    p.add_argument("--src-file", required=True)
    p.add_argument("--ref-file", required=True)
    p.add_argument("--out-file", required=True)
    p.add_argument("--num-beams", type=int, default=5)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model_id)

    srcs = open(a.src_file).read().splitlines()
    refs = open(a.ref_file).read().splitlines()
    outs, sats = [], []
    for s in srcs:
        c = build_constraints(s, MEDICAL_EN_DE, tok)
        o = translate_constrained(model, tok, s, c, num_beams=a.num_beams)
        outs.append(o)
        sats.append(report_satisfaction(o, s, MEDICAL_EN_DE))

    with open(a.out_file, "w") as f:
        json.dump({
            "bleu_chrf": corpus_metrics(outs, refs),
            "constraint_rate": sum(s["satisfied"] for s in sats) / max(1, sum(s["applicable"] for s in sats)),
            "misses": [m for s in sats for m in s["missed_terms"]],
        }, f, indent=2)
```

`corpus_metrics` is the sacreBLEU wrapper from exercise-01 §2.7; reporting BLEU + chrF alongside the constraint rate is the only way to see when constraints have cost quality.

## 3. Validation steps

1. **Glossary round-trip test.** For each `TermEntry`, encode every form with `add_special_tokens=False` and assert the resulting id sequence is non-empty. Dropped encodings silently become un-enforced constraints.
2. **Case-boundary regex test.** Pin `constraint_satisfied("Der Patient nimmt Heparinase.", heparin_entry)` → `False` (substring `Heparin` but not word). Pin `constraint_satisfied("Der Patient nimmt Heparin ein.", heparin_entry)` → `True`. Catches the single most common homegrown-validator bug.
3. **Soft-baseline comparison.** Run the same corpus unconstrained with a prompt-engineered "use the term X" header; record the satisfaction rate. The constrained path should outperform by ≥10 percentage points on small NMT models and ≥5 pp on large multilingual models. Zero or negative delta means the constraints are not reaching `.generate()`.
4. **BLEU delta regression.** Record BLEU + chrF on unconstrained-no-glossary, unconstrained-with-glossary-in-prompt, and constrained-with-disjunctive-inflections. The constrained-disjunctive path should be within ≤0.5 BLEU of unconstrained-no-glossary. If the drop is >1.5 BLEU, your constraint lists are missing inflections.
5. **Beam-size scaling.** Translate the same 100 sentences at `num_beams=1` (greedy, no constraints), `num_beams=5` (unconstrained), `num_beams=5` (constrained), `num_beams=10` (constrained). The constrained-5 path should satisfy 95–99 % of applicable constraints; constrained-10 should hit 99 %+. If constrained-5 is below 90 %, your beam is starved; raise it.
6. **Memory regression.** Report peak CUDA memory for `num_beams=5` unconstrained vs constrained on a batch of 32. Constrained memory grows roughly as `num_beams × num_simultaneous_constraints`; a 3×+ blow-up means a constraint is encoded as many short sequences instead of a disjunction.
7. **Re-ask-rate ceiling.** Over a 500-sentence dev set, re-ask should fire < 1 % of the time at `num_beams=5`. Higher rate means raise beams first, then enlarge the inflection list, then relax the constraint to a `PrefixConstrainedLogitsProcessor` with a looser check.
8. **End-to-end example.** Source: `"The patient was given heparin and aspirin."`; glossary includes both. At `num_beams=5`, the mBART-50 en→de output should include `Heparin` and `Aspirin` (or their inflections) with satisfaction = 2/2. Pin this as a system test.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Inflection-aware disjunctive constraints** | `DisjunctiveConstraint` over inflection list; glossary curated per direction | `PhrasalConstraint` on single form | No disjunction; single-form constraints that destroy grammar |
| **Validator rigor** | Word-boundary regex, case-sensitive target match, misses reported | Substring match | `in output` string check |
| **Prompt-baseline comparison** | Soft-prompt rate reported alongside constrained rate | Constrained rate only | Only a single number reported |
| **Quality tradeoff visible** | BLEU + chrF reported for constrained vs unconstrained | BLEU only | No quality metric |
| **Beam budget sanity** | `num_beams` ≥ 2 × typical simultaneous constraint count; memory logged | Beam fixed at 5 without justification | Default beams; OOM under constraints |
| **Empty-constraints branch** | `constraints=[] → None` coercion with unit test | Handled ad-hoc | Crashes on sentences with no term matches |
| **Terminology governance** | Glossary stored as versioned artifact; updates recorded with reviewer and date | Glossary in repo but unversioned | Hard-coded in the generator |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks noted (adversarial glossary injection, PII leakage via forced terms) with mitigations | Mentioned without mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure: direction, glossary source + licence, constraint satisfaction, BLEU delta, re-ask rate | Partial | None |

## 5. Common mistakes

- **Single surface form in the constraint.** `PhrasalConstraint(tokenize("Heparin"))` fails grammatically whenever the sentence context requires a case marking other than nominative. Always use `DisjunctiveConstraint` over inflections.
- **Substring validator.** `"Heparin" in output` returns `True` for `Heparinase`. Use a word-boundary regex and keep it case-sensitive on the target side.
- **Setting `do_sample=True` with constraints.** Raises in `transformers`. Constrained beam search is deterministic.
- **Beam too small for simultaneous constraints.** `num_beams=4` with 3 simultaneous constraints will drop constraints under memory pressure and silently emit outputs that fail them. Hard rule: `num_beams ≥ 2 × max_simultaneous_constraints`.
- **Forgetting `add_special_tokens=False` on the constraint tokenisation.** Leaked BOS/EOS tokens produce constraints that can never fire mid-sentence; the generator either emits them at the wrong position or starves.
- **Using greedy for the baseline.** Your unconstrained baseline must be `num_beams=5` too; otherwise the BLEU delta from "constraints" is partly a beam-size effect.
- **Case-insensitive validator on German / French targets.** `heparin` is grammatically wrong in German; counting it as a hit is a bug. Case-sensitive on target.
- **Treating the re-ask path as production.** Re-ask is a fallback for the <1 % of cases where constrained beam search genuinely cannot satisfy. If you depend on it for everyday rates, your beam / glossary / model is wrong.
- **Not logging misses.** A satisfaction rate without the list of missed terms is unactionable. Log every miss; this is the terminologist backlog.
- **Ignoring tokenisation of hyphenated / compound terms.** German compounds (`Blutverdünnungsmittel`) can tokenise into many subwords; constraints over very long sequences are expensive and failure-prone. Prefer compact lemma forms and let the decoder inflect, rather than forcing compounds verbatim.
- **Constraints on common words.** Adding `the → der/die/das` to the glossary saturates the constraint budget and destroys BLEU. Terminology constraints are for *terms* (medical, legal, product names), not closed-class words.

## 6. References

Primary sources used in this solution:

- Post, M., Vilar, D. (2018). *Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation*. NAACL. <https://aclanthology.org/N18-1119/> — the "banked beam" algorithm `transformers` `constrained_beam_search` implements; the paper that made terminology constraints practical.
- Hu, J. et al. (2019). *Improved Lexically Constrained Decoding for Translation and Monolingual Rewriting*. NAACL. <https://aclanthology.org/N19-1090/> — the vectorised banked-beam variant used by Marian and `transformers`; the one you actually run.
- Hokamp, C., Liu, Q. (2017). *Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search*. ACL. <https://aclanthology.org/P17-1141/> — the original grid-beam-search formulation; important background even if the Post–Vilar algorithm is what ships.
- Dinu, G. et al. (2019). *Training Neural Machine Translation to Apply Terminology Constraints*. ACL. <https://aclanthology.org/P19-1294/> — the training-time alternative (annotation-tag injection in the source); useful context for when hard constraints are too expensive at inference.
- Hugging Face `transformers` constrained generation docs — <https://huggingface.co/docs/transformers/main/en/internal/generation_utils> — `PhrasalConstraint`, `DisjunctiveConstraint`, the `constraints=` argument to `.generate()`.
- Hugging Face `transformers` reference blog *Guiding Text Generation with Constrained Beam Search* — <https://huggingface.co/blog/constrained-beam-search> — worked example mirrored in §2.4; the canonical introduction.
- WMT 2021 Terminology Shared Task overview — <https://aclanthology.org/2021.wmt-1.69/> — the shared task that turned the use case into a benchmark; useful for sourcing en→fr medical terminology eval.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation*: a user-supplied glossary is attacker-controlled input to `.generate()`; untrusted entries can force outputs that contain specific strings (marketing, propaganda, PII). Mitigate by reviewing and signing glossaries, bounding entry count, and rate-limiting edits. *ML07 Transfer Learning Attack*: fine-tunes that embed terminology rules in the weights can be poisoned to prefer attacker-chosen inflections; keep terminology outside the model (as runtime constraints) when the glossary is untrusted.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for adversarial ML; relevant when terminology is enforced server-side for regulated outputs (pharmaceutical dosages, legal citations) and tampering carries liability.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record glossary source, reviewer, and effective date; record language direction and quality baselines; *Measure*: report constraint satisfaction *and* BLEU/chrF so a reviewer can see the quality price of each glossary update.
