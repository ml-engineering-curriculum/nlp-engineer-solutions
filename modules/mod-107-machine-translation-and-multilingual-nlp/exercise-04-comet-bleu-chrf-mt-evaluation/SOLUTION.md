# exercise-04 — COMET / BLEU / chrF MT Evaluation

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-107-machine-translation-and-multilingual-nlp/exercises/exercise-04-comet-bleu-chrf-mt-evaluation.md`.
> The target learning outcomes implied by that filename: evaluate an NMT system with the **three-metric panel** the WMT Metrics Shared Task recommends — **sacreBLEU** with its signature string for cross-paper comparability, **chrF / chrF++** for morphologically rich languages, and **COMET** as a learned metric trained on human direct-assessment scores — and report **sentence-level deltas with bootstrap confidence intervals**, not just a corpus number.

## 1. Solution overview

Evaluating MT is where every team loses time. The three failure modes that decide whether your number means anything:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Single-metric reporting** | BLEU correlates with human judgment at the ρ ≈ 0.2–0.5 range on sentence-level; teams shipping on BLEU alone routinely miss regressions that human raters would flag (Freitag et al. 2021 WMT Metrics) | Report BLEU + chrF (or chrF++) + a learned metric (COMET / BLEURT) together; disagreements between them are the diagnostic |
| **Tokenisation drift** | `.split()`-BLEU, `nltk.translate.bleu`, and hand-rolled n-gram BLEU all vary ±3 points vs sacreBLEU on the same data; comparisons across papers that don't pin the signature are not comparable | Always use sacreBLEU with the signature string recorded; for Chinese/Japanese/Korean targets use the language-specific tokenisation (`tok:zh`, `tok:ja-mecab`, `tok:ko-mecab`) |
| **No confidence intervals** | A corpus BLEU delta of 0.3 is reported as "the new model is better"; it is inside the bootstrap noise band and the claim is unsupported | Report **paired bootstrap resampling** (1 000 samples) to produce a CI and a p-value for the hypothesis that system A > system B |

The deliverable is an evaluation harness with four components:

1. **sacreBLEU runner** — `BLEU` + `CHRF` with the signature string recorded; language-pair-aware tokenisation.
2. **COMET runner** — `Unbabel/wmt22-comet-da` (reference-based) and `Unbabel/wmt22-cometkiwi-da` (reference-free); documents the backbone and training data, which is what makes the number comparable.
3. **Paired bootstrap tester** — given two systems and a reference, produce `(delta_mean, CI_95, p_value)` for each metric.
4. **Report builder** — a single JSON that captures every system's score, every signature string, and every bootstrap CI, so the artefact a reviewer sees is reproducible without re-running.

Reference metrics (WMT Metrics Shared Task family):
- **sacreBLEU** (Post 2018) — reference BLEU implementation with pinned tokenisation.
- **chrF / chrF++** (Popović 2015, 2017) — character-n-gram F-score; chrF++ adds word-level n-grams. More robust to morphology; higher correlation with humans than BLEU on inflected languages.
- **COMET** (Rei et al. 2020; `wmt22-comet-da`) — regression head on XLM-R that predicts human direct-assessment scores; the state-of-the-art reference-based metric in WMT 2021–2023.
- **COMET-Kiwi** (Rei et al. 2022) — same family, reference-free; the quality-estimation variant used when you have no reference.
- **BLEURT** (Sellam et al. 2020) — alternative learned metric; useful cross-check against COMET.

Reference dataset: **FLORES-200 devtest** (NLLB Team 2022) for cross-lingual coverage; **WMT newstest20XX** for the published-literature baselines.

## 2. Worked implementation

### 2.1 Project layout

```
mt_eval/
  sacre.py          # sacreBLEU + chrF wrappers; signature recording
  comet_score.py    # COMET (ref-based) + COMET-Kiwi (ref-free)
  bootstrap.py      # paired bootstrap + CI + p-value
  report.py         # JSON aggregator
  run.py            # CLI
  tests/
    test_signature.py
    test_bootstrap.py
    test_comet_shape.py
  requirements.txt
```

### 2.2 sacreBLEU + chrF — pin the signature, print it loudly

```python
# sacre.py
import sacrebleu


DEFAULT_TOKENIZE = {
    # Only override for CJK / targets where whitespace tokenisation lies
    "zh": "zh",
    "ja": "ja-mecab",
    "ko": "ko-mecab",
}


def corpus_sacre_metrics(
    hyps: list[str],
    refs: list[str],
    tgt_lang: str | None = None,
) -> dict:
    tokenize = DEFAULT_TOKENIZE.get(tgt_lang, "13a")
    bleu = sacrebleu.BLEU(tokenize=tokenize)
    chrf = sacrebleu.CHRF(word_order=2)   # chrF++
    b = bleu.corpus_score(hyps, [refs])
    c = chrf.corpus_score(hyps, [refs])
    return {
        "bleu": b.score,
        "bleu_signature": b.get_signature().format(short=False),
        "chrf++": c.score,
        "chrf_signature": c.get_signature().format(short=False),
    }
```

Points worth defending in review:

- **Tokenisation per target language.** `tok:13a` (the default) is Moses-like and correct for Latin-script and most European targets. On Chinese `tok:13a` under-tokenises and inflates BLEU by 5–10 points; on Japanese `tok:ja-mecab` is the WMT convention. Overriding is a two-line table; forgetting is a public mistake.
- **chrF with `word_order=2` (= chrF++).** chrF++ correlates better with human judgment than either plain BLEU or plain chrF on WMT Metrics sets, and is the default WMT "second score" since 2017.
- **Signature recorded in the output.** The signature string (`BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.x.x`) is the handshake that lets another team reproduce your score. Papers that omit it are not reproducible on this metric.
- **Refs as list-of-lists.** `corpus_score(hyps, [refs])` — the outer list is "reference sets" (for the rare multi-reference case); even with one reference, it is `[refs]`.

### 2.3 COMET and COMET-Kiwi

```python
# comet_score.py
from comet import download_model, load_from_checkpoint

REF_BASED_MODEL = "Unbabel/wmt22-comet-da"
REF_FREE_MODEL = "Unbabel/wmt22-cometkiwi-da"


def load_comet(ref_based: bool = True):
    name = REF_BASED_MODEL if ref_based else REF_FREE_MODEL
    model_path = download_model(name)
    return load_from_checkpoint(model_path), name


def comet_scores(
    srcs: list[str], hyps: list[str], refs: list[str] | None,
    gpus: int = 1, batch_size: int = 16,
) -> dict:
    ref_based = refs is not None
    model, name = load_comet(ref_based)
    data = [
        {"src": s, "mt": h, "ref": r} if ref_based else {"src": s, "mt": h}
        for s, h, r in zip(srcs, hyps, refs if refs else hyps)
    ]
    result = model.predict(data, batch_size=batch_size, gpus=gpus)
    return {
        "comet_mean": float(result.system_score),
        "comet_sentence": result.scores,
        "comet_model": name,
    }
```

- **Model identity recorded.** `wmt22-comet-da` (reference-based) and `wmt22-cometkiwi-da` (reference-free) are the WMT 2022 state-of-the-art regression heads; they publish their training data and the Direct Assessment protocol they were trained on. Reporting a bare "COMET score" without the model id is like reporting BLEU without the signature.
- **Per-sentence scores retained.** The system-level score is what you report; the per-sentence scores are what you bootstrap and what lets you audit individual regressions.
- **Reference-free mode** (`refs=None`) uses CometKiwi; use it for quality-estimation-only pipelines (no references available), not as a replacement for the reference-based score when references exist.

### 2.4 Paired bootstrap resampling — the only honest delta

```python
# bootstrap.py
import numpy as np


def paired_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    num_samples: int = 1000,
    seed: int = 20260101,
) -> dict:
    """
    Compute a paired-bootstrap 95% CI and p-value for mean(A - B).
    - scores_a, scores_b: per-sentence metric values aligned by index
    - Returns: delta, ci_low, ci_high, p_value for H0 that A <= B
    """
    assert scores_a.shape == scores_b.shape
    rng = np.random.default_rng(seed)
    n = scores_a.shape[0]
    deltas = []
    for _ in range(num_samples):
        idx = rng.integers(0, n, size=n)
        deltas.append(scores_a[idx].mean() - scores_b[idx].mean())
    deltas = np.array(deltas)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    # One-sided p-value against H0: A does not improve on B
    p = (deltas <= 0).mean()
    return {
        "delta_mean": float(deltas.mean()),
        "ci_95": [float(ci_low), float(ci_high)],
        "p_value_a_gt_b": float(p),
        "n_bootstrap": num_samples,
    }
```

- **Paired, not unpaired.** The two systems translate the same sentences; the bootstrap must preserve the per-sentence pairing. Un-pairing inflates variance ~2× and hides significant deltas.
- **Per-sentence metric values.** For BLEU, use the per-sentence `sacrebleu.sentence_bleu(...).score`; for chrF, `sacrebleu.sentence_chrf(...).score`; for COMET, the `result.scores` list from §2.3. Per-sentence BLEU is noisy (especially on short sentences) but the bootstrap averages it correctly; the alternative — per-system corpus BLEU over resampled corpora — is documented in Koehn 2004 (see §6).
- **One-sided p-value.** `delta > 0` means A is better than B by this metric. A 0.3 BLEU delta with `p = 0.3` is not a win, regardless of how satisfying the number looks.

### 2.5 Report builder

```python
# report.py
import json, numpy as np
from .sacre import corpus_sacre_metrics
from .comet_score import comet_scores
from .bootstrap import paired_bootstrap


def evaluate_system(name, srcs, hyps, refs, tgt_lang=None):
    sacre = corpus_sacre_metrics(hyps, refs, tgt_lang=tgt_lang)
    import sacrebleu
    sent_bleu = np.array([sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hyps, refs)])
    sent_chrf = np.array([sacrebleu.sentence_chrf(h, [r], word_order=2).score for h, r in zip(hyps, refs)])
    comet = comet_scores(srcs, hyps, refs)
    return {
        "system": name,
        **sacre,
        "comet_mean": comet["comet_mean"],
        "comet_model": comet["comet_model"],
        "_per_sentence": {
            "bleu": sent_bleu, "chrf": sent_chrf, "comet": np.array(comet["comet_sentence"]),
        },
    }


def compare_two(sys_a, sys_b):
    out = {"a": sys_a["system"], "b": sys_b["system"], "metrics": {}}
    for m in ("bleu", "chrf", "comet"):
        out["metrics"][m] = paired_bootstrap(
            sys_a["_per_sentence"][m], sys_b["_per_sentence"][m],
        )
    return out


def write_json(path, systems, pairs):
    doc = {
        "systems": [{k: v for k, v in s.items() if k != "_per_sentence"} for s in systems],
        "comparisons": pairs,
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
```

### 2.6 CLI

```python
# run.py
import argparse
from .report import evaluate_system, compare_two, write_json


def read_lines(p):
    return open(p, encoding="utf-8").read().splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--sys", action="append", required=True,
                    help="name:path, may be repeated", nargs="+")
    ap.add_argument("--tgt-lang", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    srcs = read_lines(a.src)
    refs = read_lines(a.ref)

    systems = []
    for s in [item for sub in a.sys for item in sub]:
        name, path = s.split(":", 1)
        hyps = read_lines(path)
        assert len(hyps) == len(refs), f"{name}: hyps/refs length mismatch"
        systems.append(evaluate_system(name, srcs, hyps, refs, tgt_lang=a.tgt_lang))

    pairs = []
    for i in range(len(systems)):
        for j in range(i + 1, len(systems)):
            pairs.append(compare_two(systems[i], systems[j]))

    write_json(a.out, systems, pairs)
```

## 3. Validation steps

1. **Signature pin.** On a fixed `(hyps, refs)` fixture, assert the signature returned by sacreBLEU is the full `BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.x.x`. If it drifts, so does the comparability of the number.
2. **Tokenisation table coverage.** `test_signature.py` runs for `(hyps_zh, refs_zh)`, asserts the signature includes `tok:zh`. Same for Japanese (`tok:ja-mecab`) and Korean (`tok:ko-mecab`). Catches the "my Chinese BLEU is 60 and I believe it" failure.
3. **Identity translation = BLEU 100.** `corpus_sacre_metrics(refs, refs)["bleu"] == 100.0`. One-line sanity check that catches swapped-arg bugs; the sacreBLEU API is `corpus_score(hyps, [refs])` and getting the order wrong flips precision and recall.
4. **Hand-computed chrF.** Pin a tiny test: `hyps=["the cat"]`, `refs=["the cat"]` → chrF 100. `hyps=["the cat"]`, `refs=["the hat"]` → chrF ≈ 71 (default order). These are pinned so a library upgrade that changes defaults is caught immediately.
5. **Bootstrap under the null.** Pass `paired_bootstrap(x, x)` (same scores both sides); assert `ci_95` straddles 0 and `p_value_a_gt_b` ≈ 0.5. Catches a direction-flip bug in the percentile code.
6. **Bootstrap seed stability.** Two runs with the same seed must produce identical `(delta_mean, ci_95, p_value)`. If they drift, the RNG is not being seeded; the paper's CI becomes irreproducible.
7. **COMET score ordering.** For a toy pair `src="Hello"`, `ref="Bonjour"`, compare `hyps=["Bonjour"]` and `hyps=["xyzzy"]`. The first must produce a higher COMET score. Direction swaps in a bespoke wrapper are the single most common COMET bug.
8. **Metric-disagreement regression.** On a 500-sentence dev set, compute Spearman between corpus BLEU and COMET across 10 system variants. Expect ρ > 0.6 on well-behaved systems; ρ < 0.4 is a signal that one of the metrics is seeing something the other isn't — a *good* signal. Report both.
9. **Language-pair sanity on FLORES-200 devtest.** NLLB-200 distilled 600M on `eng_Latn → deu_Latn` FLORES-200 devtest should land near BLEU 30–32 / chrF++ ≈ 58 / COMET ≈ 0.85. Numbers 3+ BLEU below that range flag a config problem (wrong `forced_bos_token_id`, wrong tokenisation).

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Three-metric panel** | BLEU + chrF++ + COMET all reported | Two of the three | BLEU only |
| **Signature discipline** | sacreBLEU signature and COMET model id recorded in output JSON | Signature shown in log but not persisted | Bare numbers |
| **Target-language tokenisation** | Tokeniser override for zh/ja/ko with unit test | Override present, no test | 13a everywhere |
| **Paired bootstrap** | 1000-sample paired bootstrap; CI + p-value reported | Non-paired bootstrap or no CI | Point estimate only |
| **Reference-free variant** | COMET-Kiwi reported when refs absent | Noted but not implemented | Only ref-based |
| **Sentence-level retained** | Per-sentence scores persisted alongside corpus | One of them | Corpus only |
| **FLORES or WMT anchor** | At least one standard eval set result reported for sanity | Internal set only | None |
| **Reproducibility** | Seed, metric versions, model ids in output JSON; `sacrebleu`, `unbabel-comet` versions pinned in requirements | Partial | None |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks noted (gameable metrics, reference contamination of COMET training data) with mitigations | Mentioned without mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure: language pair, test set + revision, metric panel + CI, known gaming modes | Partial | None |

## 5. Common mistakes

- **`.split()`-BLEU or `nltk.translate.bleu_score`.** Both drift ±3 points from sacreBLEU on the same data. Use sacreBLEU.
- **Forgetting target-language tokenisation.** `tok:13a` on Chinese inflates BLEU by 5–10 points. Pass `tokenize="zh"` (or let sacreBLEU auto-select via `-l en-zh` on the CLI).
- **Dropping the signature.** A paper that reports "BLEU = 35.1" is not comparable to any other paper; always print the signature next to the number.
- **Non-paired bootstrap.** Translating the same corpus with two systems and un-pairing before resampling inflates variance ~2× and hides real deltas. The paired bootstrap *must* keep the per-sentence indices aligned.
- **Reporting corpus BLEU delta without a CI.** A 0.3 BLEU delta on 1 000 sentences is inside the noise band. The CI from the bootstrap is the only honest answer to "is it better?".
- **Using COMET without recording the model id.** `wmt21-comet-da`, `wmt22-comet-da`, and `xcomet-xl` are different models and produce different scores on the same data. Record the exact model id.
- **Confusing COMET with COMET-Kiwi.** The reference-based model takes `{src, mt, ref}`; the reference-free model takes `{src, mt}`. Swapping them silently scores against a wrong input and under-reports.
- **Treating COMET as "the right answer".** COMET correlates with human judgment better than BLEU but is not ground truth; it can be gamed by producing fluent-but-wrong translations (because the backbone rewards fluency). Report both learned and surface-form metrics.
- **Reference contamination.** COMET's training data (WMT DA scores) includes public test sets; evaluating on WMT newstest2017–2020 with COMET risks test-set leakage into the metric itself. Cross-check with chrF on those sets, or move to FLORES-200.
- **Averaging per-sentence BLEU for the corpus number.** Corpus BLEU is a global n-gram statistic, *not* the mean of per-sentence BLEU. `np.mean(sentence_bleus)` is a different metric and is typically lower; use `corpus_bleu` for the headline number and keep `sentence_bleu` only for bootstrap resampling.

## 6. References

Primary sources used in this solution:

- Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores*. WMT. <https://aclanthology.org/W18-6319/> — the paper that introduced sacreBLEU and the signature-string convention used in §2.2.
- Papineni, K. et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. ACL. <https://aclanthology.org/P02-1040/> — the original BLEU; important background even if sacreBLEU is what runs.
- Popović, M. (2015). *chrF: character n-gram F-score for automatic MT evaluation*. WMT. <https://aclanthology.org/W15-3049/> — chrF.
- Popović, M. (2017). *chrF++: words helping character n-grams*. WMT. <https://aclanthology.org/W17-4770/> — the chrF++ variant used in §2.2.
- Rei, R. et al. (2020). *COMET: A Neural Framework for MT Evaluation*. EMNLP. <https://aclanthology.org/2020.emnlp-main.213/> — the COMET backbone and training protocol used in §2.3.
- Rei, R. et al. (2022). *COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task*. WMT. <https://aclanthology.org/2022.wmt-1.52/> — the `wmt22-comet-da` and `wmt22-cometkiwi-da` releases used in §2.3.
- Sellam, T., Das, D., Parikh, A. (2020). *BLEURT: Learning Robust Metrics for Text Generation*. ACL. <https://aclanthology.org/2020.acl-main.704/> — the sibling learned metric; useful cross-check against COMET.
- Koehn, P. (2004). *Statistical Significance Tests for Machine Translation Evaluation*. EMNLP. <https://aclanthology.org/W04-3250/> — the paired-bootstrap protocol used in §2.4.
- Freitag, M. et al. (2021). *Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation*. TACL. <https://aclanthology.org/2021.tacl-1.87/> — the MQM study showing that BLEU-only reporting misses errors human raters catch; the empirical backbone for the three-metric-panel recommendation.
- Freitag, M. et al. (2022). *Results of WMT22 Metrics Shared Task: Stop Using BLEU – Neural Metrics Are Better and More Robust*. WMT. <https://aclanthology.org/2022.wmt-1.2/> — the official WMT finding that drives §1's "report a learned metric" recommendation.
- NLLB Team (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation*. <https://arxiv.org/abs/2207.04672> — the FLORES-200 eval protocol used as a sanity anchor in §3.
- sacreBLEU — <https://github.com/mjpost/sacrebleu> — reference implementation used in §2.2.
- Unbabel COMET — <https://github.com/Unbabel/COMET> — reference implementation used in §2.3.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML06 Model Serialization Attack*: the COMET checkpoint is a downloaded PyTorch file; verify `download_model` destinations and pin the model revision. *ML02 Data Poisoning*: reference-based metrics can be gamed by overfitting to published test sets; check for reference contamination when comparing against prior literature.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for adversarial ML; relevant when metrics gate release decisions (a system that games BLEU without gaining real quality is an adversarial-example-at-scale against a learned release process).
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Measure*: a three-metric panel with CIs is the minimum Measure-quality report for an MT system; *Govern*: pin metric versions and test-set revisions in the system card so the "better" claim can be audited later.
