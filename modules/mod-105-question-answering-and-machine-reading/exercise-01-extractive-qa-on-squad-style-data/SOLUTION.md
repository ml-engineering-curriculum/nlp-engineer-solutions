# exercise-01 — Extractive QA on SQuAD-style Data

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-105-question-answering-and-machine-reading/exercises/exercise-01-extractive-qa-on-squad-style-data.md`.
> The target learning outcomes implied by that filename: fine-tune a transformer encoder for **extractive span prediction** on SQuAD-style `(question, context, answer_start, answer_text)` tuples, handle **long contexts** with doc-stride windowing, decode predictions back to **character spans** via `offset_mapping`, and report **SQuAD-protocol EM/F1** — the numbers everyone else reports, computed the same way.

## 1. Solution overview

Extractive QA on SQuAD is deceptively simple on paper — predict a `(start, end)` pair of token positions over the context — and failure-rich in practice. The three failure modes that decide whether the pipeline is honest:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Character→token alignment** | The gold `answer_start` is a character offset into the raw context; models operate on tokens. Naïvely truncating/encoding loses the mapping and produces off-by-one or wildly-wrong spans | Tokenise with `return_offsets_mapping=True`; walk the offsets to find the token indices whose character spans cover `[answer_start, answer_start + len(answer_text))` |
| **Context overflow with no stride** | SQuAD passages are short but the model sees `question + context`. Legal, medical, Wikipedia-article variants routinely exceed 384/512 tokens, and plain truncation silently drops the answer | Use `return_overflowing_tokens=True` + `doc_stride`; produce multiple features per example; mark features whose context does not contain the answer as `(cls_idx, cls_idx)` |
| **Metric drift** | Reporting token-level accuracy, substring match, or sklearn F1 inflates numbers by 5–20 points vs the SQuAD evaluation protocol | Use the official SQuAD evaluator: lowercasing, article/punctuation removal, whitespace tokenisation, then EM and per-token F1 against *every* gold answer |

The deliverable is a small, auditable fine-tune harness with five components:

1. **Feature builder** that encodes `(question, context)` pairs with striding and offset mappings, and produces `(start_position, end_position)` token labels.
2. **Model wrapper** around `AutoModelForQuestionAnswering` (two linear heads on the encoder output for start and end logits).
3. **Post-processing** that walks each example's feature group, picks the best valid `(start, end)` span under constraints (`end >= start`, `end - start < max_answer_length`, neither index is special-token), and converts back to a character span → answer string via `offset_mapping`.
4. **Metric** using the SQuAD-v1.1 evaluation script (normalised EM / token-F1, max over all reference answers).
5. **CLI** that trains, evaluates on dev, and writes `predictions.json` in the official SQuAD submission format.

Reference datasets: **SQuAD v1.1** (Rajpurkar et al. 2016) — 100 k answerable questions over Wikipedia paragraphs. SQuAD v2.0 unanswerable handling is deferred to exercise-04.

## 2. Worked implementation

### 2.1 Project layout

```
qa_bake/
  data.py            # dataset load + tokenise with stride
  align.py           # answer_start/answer_text → token start/end
  model.py           # AutoModelForQuestionAnswering wrapper
  postprocess.py     # logits → character span → answer string
  metrics.py         # SQuAD normalise + EM/F1
  run.py             # train/eval CLI
  tests/
    test_align.py    # hand-built offset case
    test_postprocess.py
    test_metrics.py  # EM/F1 on worked examples from the paper
  requirements.txt
```

### 2.2 Feature building — the step that decides whether the model ever sees the answer

Two tokenisation flags carry the exercise:

- `truncation="only_second"` so only the **context** is truncated, never the question.
- `return_overflowing_tokens=True` plus `stride=128` so long contexts produce multiple features with overlapping windows; the overflow map tells you which features belong to which original example.

```python
# data.py
from datasets import load_dataset

MAX_LENGTH = 384
DOC_STRIDE = 128


def load_squad():
    return load_dataset("squad")  # train / validation splits


def prepare_train_features(examples, tokenizer):
    # Pad on the right; questions on the left, contexts on the right.
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
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

        if len(answers["answer_start"]) == 0:
            starts.append(cls_idx)
            ends.append(cls_idx)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Find the span of tokens that belong to the context.
        ctx_start = 0
        while sequence_ids[ctx_start] != 1:
            ctx_start += 1
        ctx_end = len(input_ids) - 1
        while sequence_ids[ctx_end] != 1:
            ctx_end -= 1

        # If the answer is outside this window, point to CLS.
        if not (offsets[ctx_start][0] <= start_char and offsets[ctx_end][1] >= end_char):
            starts.append(cls_idx)
            ends.append(cls_idx)
            continue

        # Shrink the context window to the tokens whose char span covers the answer.
        tok_start = ctx_start
        while tok_start <= ctx_end and offsets[tok_start][0] <= start_char:
            tok_start += 1
        starts.append(tok_start - 1)

        tok_end = ctx_end
        while tok_end >= ctx_start and offsets[tok_end][1] >= end_char:
            tok_end -= 1
        ends.append(tok_end + 1)

    tokenized["start_positions"] = starts
    tokenized["end_positions"] = ends
    return tokenized
```

Decisions worth defending in review:

- **Why point to `CLS` when the answer isn't in this window.** `CrossEntropyLoss` is computed against the `(start, end)` indices; pointing both at the `[CLS]` position produces a legitimate gradient toward "no answer in this feature" without requiring a separate no-answer head. For SQuAD v1.1 — which guarantees every question is answerable — this only fires on overflow features that *don't* contain the gold span; the sibling feature that *does* contain it carries the real label.
- **Why left-aligned `tok_start - 1` and right-aligned `tok_end + 1`.** The `while` loops walk *past* the covering token; backing off by one lands on the first/last token whose offsets strictly overlap the character span. Off-by-one here is the single most common SQuAD bug; the `test_align.py` case below pins it down.

### 2.3 Eval-time feature building (keep offset mapping, drop labels)

At evaluation we need the offsets to go from predicted token span → character span → string — but we don't need token-level labels.

```python
def prepare_validation_features(examples, tokenizer):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    # Store a pointer back to the original example, and null out offsets outside the context.
    tokenized["example_id"] = [examples["id"][m] for m in sample_mapping]
    for i, seq_ids in enumerate(map(tokenized.sequence_ids, range(len(tokenized["input_ids"])))):
        tokenized["offset_mapping"][i] = [
            (o if seq_ids[k] == 1 else None)
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]
    return tokenized
```

The `None`-masking on offsets outside the context is what prevents the decoder from accidentally emitting the question text as the answer.

### 2.4 Model

```python
# model.py
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODEL_ID = "distilbert-base-uncased"


def build_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID)
    return model, tok
```

`AutoModelForQuestionAnswering` adds two linear heads (start and end) on top of the last hidden state; the loss is `(CE(start_logits, start_positions) + CE(end_logits, end_positions)) / 2`. Nothing exotic — the work is in the data pipeline and the decoder. `use_fast=True` is non-negotiable for extractive QA: the slow tokenisers do not return `offset_mapping`, which breaks the entire post-processing path.

### 2.5 Post-processing — turn logits into strings

```python
# postprocess.py
import collections
import numpy as np

N_BEST = 20
MAX_ANSWER_LENGTH = 30


def postprocess_qa_predictions(examples, features, raw_predictions, n_best=N_BEST, max_answer_length=MAX_ANSWER_LENGTH):
    all_start_logits, all_end_logits = raw_predictions

    # Group features by their original example.
    feat_per_example = collections.defaultdict(list)
    for i, feat in enumerate(features):
        feat_per_example[feat["example_id"]].append(i)

    predictions = collections.OrderedDict()

    for example in examples:
        context = example["context"]
        candidates = []
        for fi in feat_per_example[example["id"]]:
            start_logits = all_start_logits[fi]
            end_logits = all_end_logits[fi]
            offset_mapping = features[fi]["offset_mapping"]

            start_idx = np.argsort(start_logits)[-1 : -n_best - 1 : -1].tolist()
            end_idx   = np.argsort(end_logits)[-1 : -n_best - 1 : -1].tolist()

            for s in start_idx:
                for e in end_idx:
                    if (
                        s >= len(offset_mapping) or e >= len(offset_mapping)
                        or offset_mapping[s] is None or offset_mapping[e] is None
                    ):
                        continue  # not in context
                    if e < s or e - s + 1 > max_answer_length:
                        continue
                    candidates.append({
                        "score": float(start_logits[s] + end_logits[e]),
                        "text": context[offset_mapping[s][0] : offset_mapping[e][1]],
                    })

        if candidates:
            best = max(candidates, key=lambda c: c["score"])
            predictions[example["id"]] = best["text"]
        else:
            predictions[example["id"]] = ""

    return predictions
```

Design decisions worth pointing to in review:

- **Top-k on each side, cross-product under constraints.** Taking `argmax(start_logits) + argmax(end_logits)` is cheap but frequently violates `end >= start`, especially on short contexts where the tails of the two distributions overlap `[CLS]`. The top-k × top-k loop over valid pairs is the canonical SQuAD decoder (Devlin et al. 2019 §4.2; HF `run_qa.py`).
- **Multiple features → one answer.** A long document that spans three windows produces three sets of logits for the same `example_id`. The decoder pools all candidates across windows before picking the best, so overflow features contribute to recall rather than being discarded.
- **`None`-offset tokens dropped.** This is what the eval-time `None`-masking in §2.3 is for. Without it, a `[CLS]` or question token can be the top start or end and the decoder returns gibberish or empty strings with high confidence.

### 2.6 Metrics — the SQuAD normaliser is the contract

```python
# metrics.py
import re, string, collections

def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gold: str) -> int:
    return int(_normalize(pred) == _normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    pt = _normalize(pred).split()
    gt = _normalize(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = collections.Counter(pt) & collections.Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pt)
    r = num_same / len(gt)
    return 2 * p * r / (p + r)


def squad_metrics(predictions, references):
    em = f1 = 0
    for ref in references:
        pred = predictions[ref["id"]]
        golds = ref["answers"]["text"] or [""]
        em += max(exact_match(pred, g) for g in golds)
        f1 += max(token_f1(pred, g) for g in golds)
    n = len(references)
    return {"exact_match": 100.0 * em / n, "f1": 100.0 * f1 / n}
```

Four normalisation rules, in this order: lowercase, strip punctuation, drop English articles (`a`, `an`, `the`), collapse whitespace. Then EM is string equality on normalised strings; F1 is per-token F1 over whitespace splits. The `max(... for g in golds)` is the thing most home-grown metrics drop — SQuAD dev has up to three reference answers per question, and the official protocol takes the max over them.

### 2.7 Training loop

```python
# run.py
import argparse
from functools import partial
from transformers import (
    Trainer, TrainingArguments, default_data_collator, set_seed,
)
from .data import load_squad, prepare_train_features, prepare_validation_features
from .model import build_model_and_tokenizer
from .postprocess import postprocess_qa_predictions
from .metrics import squad_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="runs/qa")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260101)
    a = p.parse_args()
    set_seed(a.seed)

    ds = load_squad()
    model, tok = build_model_and_tokenizer()

    train_ds = ds["train"].map(
        partial(prepare_train_features, tokenizer=tok),
        batched=True,
        remove_columns=ds["train"].column_names,
    )
    val_feats = ds["validation"].map(
        partial(prepare_validation_features, tokenizer=tok),
        batched=True,
        remove_columns=ds["validation"].column_names,
    )

    args = TrainingArguments(
        output_dir=a.output_dir,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size * 2,
        learning_rate=a.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        save_strategy="epoch",
        seed=a.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        tokenizer=tok,
        data_collator=default_data_collator,
    )
    trainer.train()

    raw_preds = trainer.predict(val_feats.remove_columns(["example_id", "offset_mapping"]))
    predictions = postprocess_qa_predictions(
        examples=ds["validation"],
        features=val_feats,
        raw_predictions=raw_preds.predictions,
    )
    print(squad_metrics(predictions, ds["validation"]))
```

`default_data_collator` works for extractive QA because padding is already fixed at `max_length` inside `prepare_*_features` — label alignment (`start_positions`, `end_positions`) is to real token indices, so there's no dynamic padding to break.

## 3. Validation steps

1. **Hand-built alignment case.** `question="Who?"`, `context="The quick brown fox."`, `answer_text="brown fox"`, `answer_start=10`. Call `prepare_train_features`. Assert decoded `(start, end)` tokens round-trip to `"brown fox"` via `tokenizer.decode(input_ids[s:e+1])` after stripping specials.
2. **Overflow coverage.** Build a synthetic 1 500-character context with the gold answer near the end. With `max_length=384, stride=128`, confirm that at least one feature contains the answer and that feature carries real `(start, end)` positions; all other features carry `(cls_idx, cls_idx)`.
3. **SQuAD-normaliser unit tests.** From the paper's own worked examples: `pred="Denver Broncos"`, `gold="the Denver Broncos"` → EM=1 (article stripped), F1=1. `pred="Carolina"`, `gold="the Carolina Panthers"` → EM=0, F1≈0.67.
4. **Top-k decoder sanity.** Feed synthetic logits where `argmax(start)=5, argmax(end)=3` (`end < start`); the decoder must pick a valid lower-ranked pair rather than empty-string. If it returns `""`, your cross-product constraint-check is on the wrong side of the if.
5. **`None`-offset handling.** Patch a validation feature so the top start-logit lands on a question token. Decoder must skip it and fall through to the next valid candidate.
6. **Overflow fan-in.** Pick a dev example whose context is >384 tokens. Confirm `len(feat_per_example[example_id]) > 1` and that the final chosen span could have come from any of the overlapping windows — not just the first.
7. **Prediction file format.** `predictions.json` must be `{question_id: answer_text}` — the exact shape the SQuAD leaderboard evaluator consumes. Validate by loading and running the official evaluator against it; the number should match `squad_metrics` within floating-point.
8. **Seed stability.** Train with 3 seeds, 2 epochs each. DistilBERT on SQuAD v1.1 lands around EM 77 / F1 85 across seeds; a run >1.5 F1 below the mean usually means a broken alignment, not bad luck.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Char→token alignment** | Offset-mapping walk with hand-built unit test; `tok_start - 1 / tok_end + 1` boundary documented | Alignment present, no test | Uses `answer_start // 4` or whitespace-split estimation |
| **Long-context handling** | `return_overflowing_tokens=True` + `stride=128`; features not containing the answer point to `[CLS]`; decoder fans in over all features | Striding present but decoder only reads one feature | No striding; long contexts truncated |
| **Decoder constraints** | Top-k × top-k cross-product; `end >= start`; `max_answer_length`; `None`-offset guard | `argmax` on each side; manual `max(s, e)` fix | No span validity check; returns `[CLS]`-containing strings |
| **SQuAD metric fidelity** | Normaliser matches the official v1.1 evaluator (lowercase, drop punct + articles, collapse spaces); `max` over multiple gold answers | Normaliser present but single-gold | sklearn F1 or substring match |
| **Prediction format** | `predictions.json` as `{id: str}`; passes the official evaluator | File exists but wrong keys | No submission file |
| **Reproducibility** | `set_seed`; seed and dataset revision echoed in output JSON | Seed set, no echo | No seed |
| **Threat model** | Prompt-injection / context-injection risks noted per OWASP ML Top 10 and MITRE ATLAS; mitigation (strip control tokens, cap context length) stated | Mentioned without mitigation | Ignored |
| **Documentation / card** | Model card lists dataset licence (CC-BY-SA-4.0 for SQuAD) and caveats per NIST AI RMF (Map, Measure) | One of those | Neither |

## 5. Common mistakes

- **Using the slow tokeniser.** No `offset_mapping` → no character-level decode. The entire pipeline silently falls back to whitespace guessing, and F1 drops 15–20 points with no error message.
- **Truncating without striding.** On SQuAD v1.1 you lose maybe 2 % of answers; on domain data with long contexts you lose 30–60 %. The single biggest free F1 gain on real-world extractive-QA deployments is adding `doc_stride`.
- **Off-by-one on `tok_end`.** The `while offsets[tok_end][1] >= end_char` loop must back off *past* the covering token and then `+1`. If you stop at the covering token and don't `+1`, the decoded span is missing its final word on roughly 8 % of examples. The alignment unit test catches this.
- **Decoding without the context-only mask.** If `offset_mapping` isn't `None`-masked outside the context, the decoder happily returns substrings of the question. Always run the eval-time preparer (§2.3), not the training-time preparer.
- **`argmax` top-1 on each side.** Violates `end >= start` or exceeds `max_answer_length` on 3–6 % of predictions. Top-k × top-k with constraints is the one-line fix that nets ~1 F1.
- **Single-gold F1.** Computing F1 against only `answers["text"][0]` misses the maximum-over-references rule and under-reports F1 by 1–3 points. Always `max(f1(pred, g) for g in golds)`.
- **Padding context on the left.** `padding_side="left"` interacts badly with `only_second` truncation on some tokenisers (RoBERTa/GPT-2 family); contexts get stashed in the pad region and `sequence_ids` returns the wrong mapping. Keep the default (`"right"`) for extractive QA.
- **Mixing v1.1 and v2.0.** v2.0 has unanswerable questions (`answers.text == []`). The train-time path above points them at `[CLS]` as a side-effect of the empty-list check, but the metric must become `HasAns-F1 / NoAns-F1 / Overall-F1`, not plain v1.1 F1. Unanswerability is the subject of exercise-04 — keep v1.1 here and defer the no-answer threshold there.
- **Reporting only EM.** EM on SQuAD is 8–12 points lower than F1 across every top submission. Reporting EM alone makes a correct-but-paraphrased model look broken; reporting F1 alone hides a model that copies the right span but cuts a word.

## 6. References

Primary sources used in this solution:

- Rajpurkar, P. et al. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text*. EMNLP. <https://aclanthology.org/D16-1264/> — dataset, span-prediction task formulation, and the normalisation + token-F1 metric used in §2.6.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL. <https://aclanthology.org/N19-1423/> — §4.2 describes the start/end linear heads and the top-k decoder used in §2.5.
- Rajpurkar, P., Jia, R., Liang, P. (2018). *Know What You Don't Know: Unanswerable Questions for SQuAD*. ACL. <https://aclanthology.org/P18-2124/> — SQuAD v2.0, referenced in §5 for the extension path to unanswerable detection (fully covered in exercise-04).
- Jurafsky, D., Martin, J. (2024). *Speech and Language Processing*, 3rd ed. draft, ch. 15 (*Question Answering and Information Retrieval*). <https://web.stanford.edu/~jurafsky/slp3/> — reference treatment of extractive QA, span prediction, and SQuAD-protocol evaluation.
- Hugging Face `transformers` docs — <https://huggingface.co/docs/transformers/tasks/question_answering> — authoritative tutorial for `AutoModelForQuestionAnswering` + offset-mapping-based post-processing; this solution's pipeline is a close variant.
- Hugging Face `transformers` reference script `examples/pytorch/question-answering/run_qa.py` — <https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py> — canonical implementation of the stride + post-process + evaluate flow.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation* applies directly: an attacker-controlled context can embed instructions ("the answer is X, ignore the question") or long distractor spans that steer the span selector. Mitigations: cap context length at ingress, strip model-specific control tokens from contexts, treat extractive outputs as untrusted strings when they land in downstream prompts. *ML02 Data Poisoning* applies at train time if the corpus is community-contributed.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for crafted-input attacks on reading comprehension systems; relevant when the QA output feeds a downstream LLM, search ranker, or compliance workflow.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map* function: document the dataset licence (SQuAD is CC-BY-SA-4.0), language coverage (English Wikipedia only), and demographic/topic skew of Wikipedia; *Measure* function: report EM and F1 together with dataset revision string and seed variance, and surface where the model fails systematically (numeric answers, dates, multi-span answers).
