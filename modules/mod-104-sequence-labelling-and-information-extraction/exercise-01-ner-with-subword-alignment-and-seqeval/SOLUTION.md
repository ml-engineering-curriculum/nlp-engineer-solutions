# exercise-01 — NER with Subword Alignment and seqeval

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-104-sequence-labelling-and-information-extraction/exercises/exercise-01-ner-with-subword-alignment-and-seqeval.md`.
> This solution targets the learning goals that filename implies: fine-tune a subword-tokenised transformer (BERT/DistilBERT/RoBERTa family) for token classification, align **word-level IOB/IOB2 labels** to the model's **subword tokens** without leaking or dropping tags, and evaluate with **`seqeval`** so the number you report is span-F1 — not token-level accuracy that inflates by 10+ points for free.

## 1. Solution overview

Three failure modes decide whether a token-classification fine-tune is honest or theatre:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Subword misalignment** | Each word becomes 1..N wordpieces; attaching the IOB tag to only some of them produces wrong gradients and wrong decoded spans | Align labels to the *first* subword of each word, mark remaining subwords with `-100` so cross-entropy ignores them, keep offsets for decoding |
| **Token-level "accuracy"** | ~85 % of tokens are `O`; a model that predicts `O` everywhere scores >90 % token accuracy with zero useful entities | Score with entity-level span-F1 via `seqeval`, which parses the IOB stream and compares spans, not tokens |
| **Tag-scheme drift** | Mixing IOB1 (only break-ties `B-`), IOB2 (strict begin-inside), and BILOU in training vs eval silently mislabels spans | Normalise to a single scheme (IOB2 by default — matches `seqeval` defaults and HF `AutoModelForTokenClassification`) at load time and assert it |

The deliverable is a small, auditable fine-tune harness with four components:

1. **Alignment function** that converts word-level labels + `BatchEncoding.word_ids()` into subword-level label sequences with `-100` masking.
2. **Collator** — standard `DataCollatorForTokenClassification`, which pads labels with `-100` so dynamic padding does not break masking.
3. **`compute_metrics`** that un-pads predictions, drops `-100`, maps back to tag strings, and hands to `seqeval.metrics.classification_report`.
4. **CLI** that trains, evaluates on a held-out test fold, and prints an entity-level report plus a confusion matrix over entity *types* (not tokens).

Reference dataset: **CoNLL-2003** English NER (Tjong Kim Sang & De Meulder 2003). 4 types — `PER`, `ORG`, `LOC`, `MISC` — IOB1 on disk, we normalise to IOB2 at load time.

## 2. Worked implementation

### 2.1 Project layout

```
ner_bake/
  data.py            # dataset load + IOB1→IOB2 normalisation
  align.py           # word_ids → subword label alignment
  model.py           # AutoModelForTokenClassification wrapper
  metrics.py         # seqeval-backed compute_metrics
  run.py             # train/eval CLI
  tests/
    test_align.py
    test_metrics.py
    test_scheme.py   # IOB2 round-trip sanity
  requirements.txt
```

### 2.2 Dataset loading and tag-scheme normalisation

CoNLL-2003 on disk uses IOB1: `B-X` only appears when two same-type entities abut. HF `datasets` already exposes it as IOB2 in the `conll2003` script (which converted it), but **do not rely on that**; assert it.

```python
# data.py
from datasets import load_dataset, ClassLabel
from typing import Iterable


def load_conll2003():
    ds = load_dataset("conll2003")  # fields: tokens, ner_tags (ClassLabel)
    label_list = ds["train"].features["ner_tags"].feature.names
    assert _is_iob2(label_list), f"Expected IOB2 label set, got {label_list}"
    return ds, label_list


def _is_iob2(label_list: Iterable[str]) -> bool:
    # Every non-O tag is prefixed B- or I- and the base types match.
    types = set()
    for t in label_list:
        if t == "O":
            continue
        if not (t.startswith("B-") or t.startswith("I-")):
            return False
        types.add(t[2:])
    return len(types) > 0
```

If you ingest raw CoNLL files instead of the HF cache, convert IOB1→IOB2 before training: walk the tag sequence left-to-right and promote `I-X` to `B-X` whenever the previous tag is `O` or a different type. The confusion between IOB1 and IOB2 is responsible for a disproportionate share of "my F1 dropped 3 points after I switched tokenisers" debugging sessions.

### 2.3 Subword alignment — the exercise's core function

```python
# align.py
def align_labels_to_subwords(
    examples,
    tokenizer,
    label_all_subwords: bool = False,
    max_length: int = 256,
):
    """Return a BatchEncoding with 'labels' aligned to subwords.

    - First subword of each word carries that word's label.
    - If label_all_subwords: remaining subwords carry the same label (useful
      for BILOU or for span-pointer heads).
      Else: remaining subwords are set to -100, so CE loss ignores them
      and `seqeval` sees one tag per word after reconstruction.
    - Special tokens (CLS, SEP, PAD) are -100.
    """
    tokenised = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
    )
    all_labels = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenised.word_ids(batch_index=i)
        aligned, prev_word = [], None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)
            elif wid != prev_word:
                aligned.append(int(word_labels[wid]))
            else:
                aligned.append(int(word_labels[wid]) if label_all_subwords else -100)
            prev_word = wid
        all_labels.append(aligned)
    tokenised["labels"] = all_labels
    return tokenised
```

Two design decisions worth defending in review:

- **Why `-100` on the *subsequent* subwords.** PyTorch cross-entropy treats `ignore_index=-100` as a no-op — no gradient, no contribution to the mean. The model is therefore trained to predict the right tag on the *first* subword of each word, and the decoder reads the first-subword prediction at eval time. This mirrors Devlin et al. 2019 §5.3 and is the default Hugging Face recipe.
- **Why not duplicate the label on every subword?** If you duplicate a `B-PER`, every subword of a two-piece surname is "begin", which breaks the `B-/I-` boundary semantics `seqeval` relies on to count spans. Duplicate-tag training is possible but needs a BILOU scheme and a custom decoder; the default here stays in IOB2 and aligns only to the first subword.

### 2.4 Model and tokeniser

```python
# model.py
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_ID = "distilbert-base-uncased"


def build_model_and_tokenizer(label_list):
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}
    tok = AutoTokenizer.from_pretrained(MODEL_ID, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )
    return model, tok
```

- `add_prefix_space=True` is required for BPE tokenisers (RoBERTa, GPT-2) when the input is already a list of words; WordPiece tokenisers (BERT, DistilBERT) ignore it harmlessly, so it is safe to set unconditionally.
- Register `id2label`/`label2id` on the config so downstream users get strings back instead of integer ids when they call `pipeline("token-classification", model=...)`.

### 2.5 Metrics with `seqeval`

```python
# metrics.py
import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def make_compute_metrics(label_list):
    def _compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        # Drop -100 (special tokens + non-first subwords).
        true_tags, pred_tags = [], []
        for pred_row, label_row in zip(preds, labels):
            true_row, pred_row_clean = [], []
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                true_row.append(label_list[l])
                pred_row_clean.append(label_list[p])
            true_tags.append(true_row)
            pred_tags.append(pred_row_clean)

        return {
            "precision": precision_score(true_tags, pred_tags),
            "recall":    recall_score(true_tags, pred_tags),
            "f1":        f1_score(true_tags, pred_tags),
            "report":    classification_report(true_tags, pred_tags, digits=4),
        }
    return _compute
```

Why `seqeval` and not sklearn: sklearn scores token-level, which rewards the model for predicting `O` on every token that is already `O`. `seqeval` reconstructs spans from the IOB stream (`[B-PER, I-PER, O]` → one `PER` span covering tokens 0..1) and scores precision/recall/F1 at the *span* level. This is the CoNLL-2003 convention; reporting token-level accuracy on CoNLL is a common red flag in reviews.

### 2.6 Training loop

```python
# run.py
import argparse
from functools import partial
from transformers import (
    DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed,
)
from .data import load_conll2003
from .align import align_labels_to_subwords
from .model import build_model_and_tokenizer
from .metrics import make_compute_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="runs/ner")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260101)
    a = p.parse_args()
    set_seed(a.seed)

    ds, label_list = load_conll2003()
    model, tok = build_model_and_tokenizer(label_list)

    tokenised = ds.map(
        partial(align_labels_to_subwords, tokenizer=tok),
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    args = TrainingArguments(
        output_dir=a.output_dir,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size * 2,
        learning_rate=a.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=a.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        tokenizer=tok,
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=make_compute_metrics(label_list),
    )
    trainer.train()
    print(trainer.evaluate(tokenised["test"]))
```

`DataCollatorForTokenClassification` is the one line the recipe cannot cheat on: it pads `labels` to the batch max length with `-100` (not `0`), preserving the mask. A plain `default_data_collator` pads with `0`, which collides with whatever label-0 means (often `O`, but on datasets where label-0 is `B-PER` the result is silently wrong).

## 3. Validation steps

1. **Unit test the aligner against a known case.** Hand-build `tokens=["EU", "rejects", "German", "call"]`, `ner_tags=["B-ORG","O","B-MISC","O"]`. Call `align_labels_to_subwords`. Assert exactly one non-`-100` label per *word*, in the right positions.
2. **IOB2 assertion.** The scheme-check (`_is_iob2`) must pass at load time; a regression that leaves IOB1 tags in the stream produces inflated F1 (spans bleed across type boundaries).
3. **`seqeval` sanity on a trivial case.** For `true=[['B-PER','I-PER','O']]`, `pred=[['B-PER','I-PER','O']]` F1 must be 1.0; for `pred=[['B-PER','I-ORG','O']]` F1 must be 0.0 (the type mismatch breaks the span). If your pipeline reports 0.5 or 0.67 here, you are scoring token-level.
4. **Loss excludes special tokens.** Forward one batch, print the mean of `outputs.loss`. Then swap `-100` for `0` in the labels tensor and recompute — the loss should jump substantially because `O` targets on padding positions pull the model toward "predict O".
5. **Reconstruction round-trip.** Run the trained model on one validation batch, decode predictions to word-level tags via `word_ids`, and assert the reconstructed length matches the original `len(tokens)` for every sentence. Off-by-one here is the #1 source of "F1 is high but my downstream consumer breaks".
6. **Seed stability.** Train with 3 seeds. Report min/max/mean test F1. CoNLL-2003 DistilBERT typically lands at ≈0.90–0.91 F1 across seeds; a run >1 point below the mean suggests a broken collator or alignment.
7. **Entity-type confusion.** Compute an entity-level confusion matrix (one row/col per type, plus "missed" and "spurious"). `MISC` vs `ORG` is the dominant confusion on CoNLL and is informative about the dataset, not a bug.
8. **Deps + hashes recorded.** `transformers`, `datasets`, `seqeval`, `torch` versions and the dataset revision string belong in the JSON output alongside the metric numbers.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Alignment correctness** | One label per word on the first subword; `-100` elsewhere; unit-tested against a hand case | Alignment present but no unit test | Labels broadcast to every subword, or labels zip against subwords positionally |
| **Loss masking** | `DataCollatorForTokenClassification`; `-100` padded by collator | Manual padding with `-100` in `map` but default collator | Default collator pads labels with 0 |
| **Scheme discipline** | IOB2 asserted at load; test case for the IOB1→IOB2 promotion rule | Normalised but not asserted | Mixed schemes in training vs eval |
| **Metric honesty** | `seqeval` entity-F1 as headline; per-type breakdown + confusion included | `seqeval` F1 only | Token-level accuracy reported as "NER accuracy" |
| **Reproducibility** | `set_seed` wraps HF/torch/numpy; seed echoed in output | Seed set once but no echo | No seed |
| **Decoding path** | Predictions reconstructed back to word-level tags for downstream consumers | Only subword predictions saved | No decode path documented |
| **Model card** | `id2label`/`label2id` on config; dataset licence noted per NIST AI RMF (Map) | One of those | Neither |
| **Threat model** | Input-manipulation / prompt-injection via NER inputs noted per OWASP ML Top 10 and MITRE ATLAS | Mentioned without concrete mitigation | Ignored |

## 5. Common mistakes

- **Broadcasting the label to all subwords.** Trains, evaluates, and *looks fine* — but decoded spans drift because every `I-X` that should have been a continuation becomes an independent `B-X` after `argmax`. Measured F1 can be off by 1–3 points for a bug that has no runtime error.
- **Padding labels with `0`.** Classic silent corruption: wherever label-0 is `O`, the model gets a tiny but consistent gradient to predict `O` on the padding region, which also leaks into real tokens near the end of each batch item. Always `DataCollatorForTokenClassification`.
- **Scoring with sklearn.** `sklearn.metrics.f1_score(..., average="macro")` on flattened per-token labels treats `O` as a class, which dominates the macro average. The resulting number is always higher and always wrong for span-level NER; CoNLL leaderboards use `seqeval`.
- **Truncating long sentences without striding.** CoNLL sentences are short, but legal/biomedical NER corpora routinely exceed 512 tokens. Truncation silently drops tail entities. Use `return_overflowing_tokens=True` with `stride`, or move to a long-context encoder (see exercise-02).
- **Mixing IOB1 and IOB2.** Training on IOB1 and evaluating on IOB2 (or vice versa) produces `seqeval` numbers that look plausible but are 2–4 points off. Normalise at load; assert the scheme.
- **Forgetting `add_prefix_space=True` on BPE.** RoBERTa/DeBERTa with `is_split_into_words=True` silently produce the wrong tokenisation for the first word of each sentence without it. WordPiece tokenisers do not need it.
- **Overlapping / nested entities crammed into flat IOB.** Flat IOB cannot represent nested spans (a `PER` inside an `ORG`). If the dataset has nested entities, treat it as a span-classification problem (exercise-02), not as sequence labelling.
- **Reporting a single seed.** Fine-tuning variance is 0.3–0.8 F1 points on CoNLL-2003. A one-shot "I got 91.2" is not reproducible headline material; report mean ± std over ≥3 seeds.
- **Leaking dev into training.** `load_best_model_at_end=True` picks on validation; if the "validation" set was created by random-sampling train, the picked checkpoint overfits. Use the dataset's canonical split.

## 6. References

Primary sources used in this solution:

- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL. <https://aclanthology.org/N19-1423/> — §5.3 defines the first-subword alignment recipe used by `AutoModelForTokenClassification`.
- Tjong Kim Sang, E. F., De Meulder, F. (2003). *Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition*. CoNLL. <https://aclanthology.org/W03-0419/> — the reference dataset, tag scheme, and span-F1 evaluation protocol.
- Ramshaw, L. A., Marcus, M. P. (1995). *Text Chunking Using Transformation-Based Learning*. VLC. <https://aclanthology.org/W95-0107/> — origin of the IOB scheme that IOB2 is derived from.
- Nakayama, H. (2018). *seqeval: A Python framework for sequence labeling evaluation*. <https://github.com/chakki-works/seqeval> — the metric library used in §2.5; implementation of CoNLL-2003-compatible span-F1.
- Jurafsky, D., Martin, J. (2024). *Speech and Language Processing*, 3rd ed. draft, ch. 8 (*Sequence Labeling for Parts of Speech and Named Entities*). <https://web.stanford.edu/~jurafsky/slp3/> — reference treatment of IOB1/IOB2/BILOU and span-level evaluation.
- Hugging Face `transformers` docs — <https://huggingface.co/docs/transformers/tasks/token_classification> — the authoritative tutorial for `AutoModelForTokenClassification` + `DataCollatorForTokenClassification`.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation Attack* and *ML02 Data Poisoning* both apply to NER services that accept user-controlled text (attacker can craft inputs that force mislabelled entities for downstream decisions).
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for crafted-input attacks on NLP classifiers; relevant when the NER output gates access, routing, or redaction.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Measure* function: report entity-level span-F1 with its protocol (seed variance, scheme, data revision), not just a single number; *Map* function: document dataset licence and PII risk for NER training corpora.
