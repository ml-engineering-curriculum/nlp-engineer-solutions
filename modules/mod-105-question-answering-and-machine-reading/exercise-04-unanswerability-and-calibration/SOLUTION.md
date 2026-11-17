# exercise-04 — Unanswerability and Calibration

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-105-question-answering-and-machine-reading/exercises/exercise-04-unanswerability-and-calibration.md`.
> The target learning outcomes implied by that filename: train an extractive QA model that can **abstain** (SQuAD v2.0-style unanswerable detection), **calibrate** its confidence so that the reported probability matches the empirical correctness rate, and pick an **operating threshold** that trades off coverage for risk — with reliability diagrams, ECE, and a selective-prediction curve to defend the choice.

## 1. Solution overview

Three failure modes decide whether an unanswerable-aware QA pipeline is honest or theatre:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Confusing "unanswered" with "low confidence"** | A model that always predicts `""` on hard questions has 100 % precision on its answered subset — and is useless. A model that always answers has high recall and miscalibrated confidence | Separate **selection** (should I answer?) from **prediction** (what is the answer?). Train with explicit no-answer supervision; at inference, compute a no-answer score and compare to a tuned threshold |
| **Confidence = raw softmax** | `max(softmax(start_logits)) * max(softmax(end_logits))` is a product-of-marginals, not a joint probability of the span. On SQuAD v2.0 it is systematically over-confident on long spans and under-confident on short ones. Reliability diagrams curve strongly off the diagonal | Fit a **post-hoc calibrator** (temperature scaling on the start/end logits, or Platt / isotonic on the answerable score) using the held-out dev split. ECE should drop 3–10× |
| **Threshold picked on test** | Tuning the no-answer threshold on the test set is a leak. Thresholds tuned on test routinely score 2–4 points higher than thresholds tuned on dev | Split the data: train → dev (threshold + calibrator) → test (reported numbers). Never touch the test set until the threshold is frozen |

The deliverable is a small, auditable extractive-QA harness with four components:

1. **No-answer supervision**: point `(start, end) = (cls_idx, cls_idx)` for unanswerable examples, so the `[CLS]` token's logits become the no-answer score.
2. **Null-score decoder**: compute `null_score = start_logits[cls] + end_logits[cls]` and `best_non_null_score = max over valid spans`. Predict `""` iff `null_score - best_non_null_score > threshold`.
3. **Calibrator**: one temperature `T` fit on dev to minimise NLL / ECE of the `(answerable | question)` probability.
4. **Threshold-and-coverage selection**: sweep `threshold` on dev, report **HasAns-F1, NoAns-F1, Overall-F1, ECE, coverage-F1 curve, risk-coverage curve**.

Reference datasets: **SQuAD v2.0** (Rajpurkar, Jia, Liang 2018) — 150 k questions, ~50 k unanswerable; v2.0 is the canonical selective-prediction benchmark for extractive QA.

Reference model: **DistilBERT-base-uncased** via `AutoModelForQuestionAnswering`; the techniques port unchanged to any BERT-family encoder.

## 2. Worked implementation

### 2.1 Project layout

```
qa_abstain/
  data.py            # SQuAD v2.0 load + striding + CLS-pointing for unanswerable
  model.py           # AutoModelForQuestionAnswering wrapper
  postprocess.py     # null-score decoder with threshold
  calibrate.py       # temperature scaling + Platt
  metrics.py         # HasAns / NoAns / Overall F1 + ECE + risk-coverage
  run.py             # train/eval CLI
  tests/
    test_null_score.py
    test_calibration.py
    test_threshold.py
  requirements.txt
```

### 2.2 Training features — the CLS-pointing contract

For v2.0, unanswerable examples arrive with `answers = {"text": [], "answer_start": []}`. The training pipeline (shared with exercise-01) already treats "answer not in this feature window" as pointing at `[CLS]`. We extend the invariant to apply to *genuinely* unanswerable examples too:

```python
# data.py (v2.0 delta vs exercise-01)
def prepare_train_features(examples, tokenizer):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    starts, ends = [], []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_idx = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)
        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]

        # v2.0 unanswerable: empty answer list → both positions point at CLS.
        if len(answers["answer_start"]) == 0:
            starts.append(cls_idx); ends.append(cls_idx); continue

        # Otherwise identical to exercise-01's char→token alignment.
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        ctx_start = next(k for k, s in enumerate(sequence_ids) if s == 1)
        ctx_end = len(sequence_ids) - 1
        while sequence_ids[ctx_end] != 1: ctx_end -= 1

        if not (offsets[ctx_start][0] <= start_char and offsets[ctx_end][1] >= end_char):
            starts.append(cls_idx); ends.append(cls_idx); continue

        tok_start = ctx_start
        while tok_start <= ctx_end and offsets[tok_start][0] <= start_char: tok_start += 1
        tok_end = ctx_end
        while tok_end >= ctx_start and offsets[tok_end][1] >= end_char: tok_end -= 1
        starts.append(tok_start - 1); ends.append(tok_end + 1)

    tokenized["start_positions"] = starts
    tokenized["end_positions"] = ends
    return tokenized
```

Design decision to defend in review:

- **No separate "answerable" head.** Pointing both positions at `[CLS]` lets the model learn a single joint signal: if the best span under the model's distribution is `([CLS], [CLS])`, the question is unanswerable. This is the original SQuAD v2.0 baseline recipe (Rajpurkar et al. 2018 §4.3) and works surprisingly well. Adding a separate binary head is a reasonable upgrade but is strictly additive — keep the baseline honest first.

### 2.3 Null-score decoder

```python
# postprocess.py
import collections, numpy as np

N_BEST = 20
MAX_ANSWER_LENGTH = 30


def postprocess_v2(examples, features, raw_predictions, null_threshold=0.0,
                   n_best=N_BEST, max_answer_length=MAX_ANSWER_LENGTH):
    start_all, end_all = raw_predictions
    feat_per = collections.defaultdict(list)
    for i, f in enumerate(features):
        feat_per[f["example_id"]].append(i)

    predictions = collections.OrderedDict()
    scores = collections.OrderedDict()

    for example in examples:
        min_null_score = None
        valid_candidates = []
        for fi in feat_per[example["id"]]:
            s_logits, e_logits = start_all[fi], end_all[fi]
            offset_mapping = features[fi]["offset_mapping"]
            cls_idx = features[fi]["input_ids"].index(0)  # [CLS]=0 for BERT family
            feat_null = float(s_logits[cls_idx] + e_logits[cls_idx])
            if min_null_score is None or feat_null < min_null_score:
                min_null_score = feat_null

            s_idx = np.argsort(s_logits)[-1 : -n_best - 1 : -1].tolist()
            e_idx = np.argsort(e_logits)[-1 : -n_best - 1 : -1].tolist()
            for si in s_idx:
                for ei in e_idx:
                    if (si >= len(offset_mapping) or ei >= len(offset_mapping)
                        or offset_mapping[si] is None or offset_mapping[ei] is None):
                        continue
                    if ei < si or ei - si + 1 > max_answer_length:
                        continue
                    if si == cls_idx or ei == cls_idx:
                        continue
                    valid_candidates.append({
                        "score": float(s_logits[si] + e_logits[ei]),
                        "text": example["context"][offset_mapping[si][0] : offset_mapping[ei][1]],
                    })

        if valid_candidates:
            best = max(valid_candidates, key=lambda c: c["score"])
            score_diff = min_null_score - best["score"]
            scores[example["id"]] = float(score_diff)
            predictions[example["id"]] = "" if score_diff > null_threshold else best["text"]
        else:
            scores[example["id"]] = float(min_null_score if min_null_score is not None else 0.0)
            predictions[example["id"]] = ""

    return predictions, scores
```

Three design decisions worth pointing to in review:

- **`null_score - best_non_null_score > threshold` → abstain.** Taking the *difference* removes most of the per-example scale. The threshold is on a bounded signed quantity and generalises across examples.
- **Minimum null score across features.** A long document can produce 5 feature windows; the minimum null score is the strongest "no evidence in any window" signal and is what the decoder uses. Using the mean dilutes the signal.
- **Return the `score_diff` alongside the prediction.** The calibrator and the risk-coverage analyses in §2.5 need that scalar; computing it twice (once for decoding, once for metrics) is a frequent source of off-by-one between what the model picked and what the metric measures.

### 2.4 Calibration — temperature scaling on the answerable score

```python
# calibrate.py
import torch
import numpy as np


def _answerable_prob(score_diff: np.ndarray, T: float) -> np.ndarray:
    """P(answerable) = sigmoid( -score_diff / T ).

    score_diff > 0 → model's best answer is weaker than the null → abstain → low p.
    T > 1 softens the probabilities; T < 1 sharpens them.
    """
    return 1.0 / (1.0 + np.exp(score_diff / T))


def fit_temperature(score_diff: np.ndarray, is_correct_when_answered: np.ndarray,
                    is_answerable: np.ndarray) -> float:
    """Fit T by minimising NLL of the binary 'answerable & correct if answered' label.

    Target = 1 iff the question is answerable AND we would have gotten it right if we
    chose to answer.  This is the operationally correct target for selective prediction:
    we want high confidence to track the chance that answering is the right move.
    """
    y = (is_answerable & is_correct_when_answered).astype(np.float32)
    s = torch.tensor(score_diff, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    T = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        logits = -s / T
        loss = bce(logits, y_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=1e-3))


def platt(score_diff, is_correct, is_answerable):
    """Logistic regression fallback: p = sigmoid(a * score_diff + b). Fitted on dev.

    Platt is a two-parameter calibrator; it fits better when the raw signal is non-monotone
    in the right direction, at the cost of overfitting on small dev sets. Temperature
    scaling (one parameter, no intercept) is almost always strictly better when the raw
    signal is already monotone — which score_diff is.
    """
    from sklearn.linear_model import LogisticRegression
    X = score_diff.reshape(-1, 1)
    y = (is_answerable & is_correct).astype(int)
    lr = LogisticRegression(penalty=None).fit(-X, y)  # note the sign
    return lr
```

Three decisions worth defending in review:

- **Target = `is_answerable & is_correct_when_answered`.** A calibrator that only models `is_answerable` tells you the base rate of answerability, which is useful but not the risk the decoder actually takes. The joint target "the system should have answered *and* been right" makes calibrated probability the right quantity to threshold on in deployment.
- **Temperature over Platt as the default.** Temperature is strictly less prone to overfit on small dev sets (1 parameter vs 2), and on SQuAD v2.0 the raw score-diff is already monotone in correctness. Platt is the fallback when temperature's reliability diagram still curves systematically.
- **Fit on dev, evaluate on test, never both.** Temperature scaling is a one-parameter model, but fitting on test still leaks. The dev / test split is load-bearing for the ECE numbers you publish.

### 2.5 Metrics — F1 breakdown, ECE, risk-coverage

```python
# metrics.py
import re, string, collections
import numpy as np


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _em(p, g):
    return int(_normalize(p) == _normalize(g))


def _f1(p, g):
    pt, gt = _normalize(p).split(), _normalize(g).split()
    if not pt or not gt:
        return float(pt == gt)
    common = collections.Counter(pt) & collections.Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    pr = n / len(pt); rc = n / len(gt)
    return 2 * pr * rc / (pr + rc)


def squad_v2(predictions, references):
    has_em = has_f1 = no_em = no_f1 = all_em = all_f1 = 0.0
    has_n = no_n = 0
    for ref in references:
        pred = predictions[ref["id"]]
        golds = ref["answers"]["text"]
        if golds:  # HasAns
            e = max(_em(pred, g) for g in golds)
            f = max(_f1(pred, g) for g in golds)
            has_em += e; has_f1 += f
            all_em += e; all_f1 += f
            has_n += 1
        else:      # NoAns: correct iff we predicted empty string
            e = int(pred == "")
            no_em += e; no_f1 += e
            all_em += e; all_f1 += e
            no_n += 1
    n = len(references)
    return {
        "HasAns_EM": 100 * has_em / max(has_n, 1),
        "HasAns_F1": 100 * has_f1 / max(has_n, 1),
        "NoAns_EM":  100 * no_em / max(no_n, 1),
        "NoAns_F1":  100 * no_f1 / max(no_n, 1),
        "EM":        100 * all_em / n,
        "F1":        100 * all_f1 / n,
    }


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error, equal-mass bins (Nguyen & O'Connor 2015 style).

    For each bin, |confidence_mean - accuracy_mean| weighted by bin mass; summed.
    """
    order = np.argsort(probs)
    probs_sorted = probs[order]
    labels_sorted = labels[order]
    N = len(probs)
    bin_edges = np.linspace(0, N, n_bins + 1, dtype=int)
    total = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi == lo: continue
        conf = probs_sorted[lo:hi].mean()
        acc  = labels_sorted[lo:hi].mean()
        total += (hi - lo) / N * abs(conf - acc)
    return float(total)


def risk_coverage(score_diff: np.ndarray, is_correct_if_answered: np.ndarray):
    """Return (coverage, risk) pairs sweeping the answer threshold.

    coverage = fraction of examples we answered
    risk     = error rate among answered examples
    Lower-is-better at every coverage.
    """
    order = np.argsort(score_diff)  # most-confident-answerable first
    sc = score_diff[order]
    correct = is_correct_if_answered[order]
    coverages, risks = [], []
    cum_correct = 0
    for k in range(1, len(sc) + 1):
        cum_correct += correct[k - 1]
        coverages.append(k / len(sc))
        risks.append(1 - cum_correct / k)
    return np.array(coverages), np.array(risks)
```

### 2.6 Threshold selection

```python
# run.py (sketch)
# 1. Train on SQuAD v2.0 train with CLS-pointing for unanswerables.
# 2. Run model on dev: produce (predictions_at_threshold_0, score_diff) for every example.
# 3. Sweep threshold in np.linspace(score_diff.min(), score_diff.max(), 1000).
# 4. For each threshold, compute Overall F1 on dev (applying `"" if score_diff > thr`).
# 5. Pick thr* = argmax Overall F1.
# 6. Fit temperature T on dev with fit_temperature.
# 7. On TEST, apply (thr*, T), report HasAns/NoAns/Overall + ECE (using T-scaled probs)
#    + risk-coverage curve.
```

The rule-of-thumb reviewers should verify: the dev-F1 curve as a function of threshold should be unimodal-ish, peaking in the interior of the range. If the maximum is at the extreme (threshold = min or max), the CLS head has not been trained — either `starts/ends` were not pointed at `[CLS]` for unanswerable examples, or every example is being pointed at `[CLS]`.

## 3. Validation steps

1. **CLS-pointing audit.** On a batch of 32 training features containing a mix of answerable and unanswerable examples, assert that for every unanswerable example `start_positions == end_positions == cls_idx`; for every answerable, the decoded span equals the gold text.
2. **Null-score monotonicity.** On dev, compute `score_diff` for 100 known-unanswerable and 100 known-answerable examples. The distribution of `score_diff` for unanswerables should sit above the distribution for answerables (mass shifted right). If they overlap completely, the CLS head did not learn.
3. **Threshold is interior.** Plot `Overall F1(threshold)`; the optimum must be interior. Interior-optimum confirms the model learned a usable no-answer signal; boundary-optimum means something is broken.
4. **ECE before/after temperature.** Pre-calibration ECE on SQuAD v2.0 dev with DistilBERT is typically 0.08–0.15; post-temperature ECE lands at 0.01–0.03. A post-calibration ECE above 0.05 suggests the score signal is non-monotone — try Platt as a fallback, or inspect the reliability diagram for a specific bin.
5. **Reliability diagram.** Plot binned confidence vs accuracy; the ideal is the diagonal. Pre-calibration you typically see over-confidence in the top quintile (confident but wrong); post-calibration the top quintile should snap to the diagonal.
6. **Risk-coverage curve.** At coverage 50 % on SQuAD v2.0 dev with a well-trained DistilBERT model, risk should be well below the overall error rate — the whole point of selective prediction. If risk is flat as coverage decreases, the model has no confidence signal; something upstream is broken.
7. **Train/dev/test hygiene.** Hash the dev predictions and store the threshold alongside them. Any change to the dev set must invalidate the threshold; any report that uses the threshold on test must cite the hash.
8. **Leak probe.** Grep for the gold answer string inside the dev question for the answerable subset. Rates >5 % suggest construction shortcuts that inflate HasAns metrics.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **No-answer supervision** | `(start, end) = (cls_idx, cls_idx)` for unanswerable training examples; unit-tested | CLS-pointing for overflow-only, not for genuinely unanswerable | Trained as v1.1; empty answers dropped or silently labelled 0 |
| **Null-score decoding** | `null_score - best_non_null_score > threshold`; threshold a real scalar | Hard-coded threshold = 0 with no selection | Argmax span always returned; no abstention mechanism |
| **Threshold hygiene** | Threshold chosen on dev, frozen before test; dev→test split documented | Threshold on dev but numbers reported on dev | Threshold fit on test |
| **Metrics breakdown** | `HasAns_F1`, `NoAns_F1`, `Overall_F1` separately | Overall only | v1.1-style single F1 |
| **Calibration** | Temperature scaling (or Platt) fit on dev; pre- and post-ECE reported | Calibrator present but one of ECE values missing | No calibration, confidence = raw softmax |
| **Reliability + risk-coverage** | Reliability diagram and risk-coverage curve in writeup | One of the two | Neither |
| **Threat model** | Discussion of how adversarial questions (prompt injection, out-of-distribution) interact with the no-answer threshold per OWASP ML Top 10 and ATLAS; abstention as a defence | Mentioned without concrete scenario | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure: dataset licence, intended use, selective-prediction target (operating point, tolerable risk) | One of those | Neither |

## 5. Common mistakes

- **Not pointing unanswerable examples at `[CLS]`.** Without supervision the CLS logits are random and `null_score` is noise; no choice of threshold recovers useful selective behaviour.
- **Threshold fit on test.** Even small test sets give 1.5–3 F1 points of overfit to the threshold. Dev→test split must be strict; threshold frozen before any test number is viewed.
- **Reporting a single "F1" on v2.0.** The v2.0 convention is HasAns-F1, NoAns-F1, and overall. A model with overall F1 70 might be 90/50 (strong extractor, no abstain) or 40/95 (abstains on everything); the breakdown is the only informative number.
- **`max_softmax` as confidence.** Softmax is over the sequence (which token is the start) not over "is there an answer". On long contexts the softmax max can be high simply because the distribution is sharp, regardless of whether the answer is right. Use `null_score - best_span_score` (post-calibration) as the confidence signal.
- **Per-bin accuracy instead of per-bin confidence vs accuracy.** ECE is the gap between the two, weighted by bin mass. Many home-grown implementations plot accuracy only and call it a reliability diagram; it is informative but not ECE.
- **Fitting temperature by minimising ECE directly.** ECE is non-differentiable and binning-dependent; minimising it directly with grid search often finds local optima that overfit the chosen bin count. Minimise NLL (as in `fit_temperature`) and report ECE as an evaluation metric; Guo et al. 2017 show these give the same result when the signal is monotone and NLL does not depend on bin count.
- **Equal-width bins on an uneven score distribution.** Most of the mass sits in the middle of the score range; equal-width bins put 90 % of examples in a few bins and give a misleading ECE. Use **equal-mass** bins (as in `ece()` above) to spread resolution across the distribution.
- **Treating `""` as a valid span for F1 scoring.** In v2.0 on an answerable question, predicting `""` scores 0 for both EM and F1; on an unanswerable question, predicting `""` scores 1 for both. The metric's asymmetry is doing load-bearing work — don't overwrite it with a "custom" variant.
- **Confusing abstention with answerability detection.** Abstention is a *decision*: given a confidence, do we answer? Answerability detection is a *signal*: how likely is the question answerable? The two can be combined (answer iff p(answerable) > τ) but the terms are not interchangeable in docs or dashboards.
- **No downstream calibration monitoring.** Model inputs drift in production; calibration goes with them. A post-launch reliability diagram should be a weekly job, not a launch-day artefact.

## 6. References

Primary sources used in this solution:

- Rajpurkar, P., Jia, R., Liang, P. (2018). *Know What You Don't Know: Unanswerable Questions for SQuAD*. ACL. <https://aclanthology.org/P18-2124/> — SQuAD v2.0 dataset; §4.3 defines the CLS-pointing training recipe and the `null_score - best_span_score` decoder used in §2.3.
- Rajpurkar, P. et al. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text*. EMNLP. <https://aclanthology.org/D16-1264/> — base dataset and metric normalisation reused from exercise-01.
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML. <https://proceedings.mlr.press/v70/guo17a.html> — temperature scaling as the default post-hoc calibrator; justifies fitting on dev by NLL and evaluating by ECE.
- Platt, J. C. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*. In *Advances in Large Margin Classifiers*. — Platt scaling as the two-parameter alternative used in `calibrate.py`.
- Nguyen, K., O'Connor, B. (2015). *Posterior Calibration and Exploratory Analysis for Natural Language Processing Models*. EMNLP. <https://aclanthology.org/D15-1182/> — reliability-diagram methodology for NLP classifiers; motivates equal-mass binning.
- El-Yaniv, R., Wiener, Y. (2010). *On the Foundations of Noise-free Selective Classification*. JMLR. <https://www.jmlr.org/papers/v11/el-yaniv10a.html> — formal framework for the risk-coverage curve used in §2.5.
- Kamath, A., Jia, R., Liang, P. (2020). *Selective Question Answering under Domain Shift*. ACL. <https://aclanthology.org/2020.acl-main.503/> — reference treatment of selective QA including SQuAD v2.0 operating-point analysis; directly maps to this exercise.
- Jurafsky, D., Martin, J. (2024). *Speech and Language Processing*, 3rd ed. draft, ch. 15 (*Question Answering and Information Retrieval*) and ch. 18 (*Calibration*). <https://web.stanford.edu/~jurafsky/slp3/> — reference treatment of QA evaluation protocols and confidence calibration.
- Hugging Face `transformers` reference script `examples/pytorch/question-answering/run_qa.py` with `--version_2_with_negative` — <https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py> — canonical implementation of the null-score decoder and threshold selection; this solution's `postprocess_v2` is a close variant.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation* is the direct threat vector relevant to abstention: adversarial inputs often sit at or near the decision boundary and defeat the model. A tuned threshold on `null_score - best_span_score`, together with a calibrator, is a defence surface: if confidence drops below the operating point, the system abstains rather than answering wrongly. *ML08 Model Skewing* applies when the deployed calibrator drifts away from the data distribution it was fit on; re-calibrate periodically.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for crafted-input attacks on QA readers; selective abstention narrows the attack surface by refusing low-confidence inputs instead of answering them confidently.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map* function: document the intended operating point (coverage at tolerable risk) and the data split used to select it; *Measure* function: report HasAns-F1, NoAns-F1, Overall-F1, ECE, and the risk-coverage curve — not a single scalar; *Manage* function: define a release gate on ECE regression and a monitoring job that re-plots the reliability diagram on production data weekly.
