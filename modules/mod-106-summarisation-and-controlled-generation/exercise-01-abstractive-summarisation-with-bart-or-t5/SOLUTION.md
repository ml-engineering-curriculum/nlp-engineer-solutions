# exercise-01 — Abstractive Summarisation with BART or T5

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-106-summarisation-and-controlled-generation/exercises/exercise-01-abstractive-summarisation-with-bart-or-t5.md`.
> The target learning outcomes implied by that filename: fine-tune an **encoder-decoder** (BART or T5) on a news summarisation corpus (CNN/DailyMail or XSum), handle **long inputs** safely, run **beam search** at inference, and evaluate with the **ROUGE** family the way the reference papers do — stemming, newline-splitting for ROUGE-L-sum, and reported as F1 against reference summaries.

## 1. Solution overview

Abstractive summarisation with an off-the-shelf encoder-decoder is "a one-liner" until you look at the three things that decide whether the numbers match the paper:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Padding leaks into the loss** | Seq2Seq training needs `labels` where pad tokens are replaced with `-100`; a missing mask rewards predicting `<pad>` and the model learns to emit empty strings | Use `DataCollatorForSeq2Seq(..., label_pad_token_id=-100)` *or* mask by hand after tokenising |
| **Wrong ROUGE protocol** | Reporting raw ROUGE on untokenised strings, or omitting the sentence-split required for ROUGE-L-sum, yields numbers 2–6 points above or below the reference — both directions lie | Lowercase + stem (Porter) + split summaries on `\n` for ROUGE-L-sum, then report F1 (not precision or recall alone) |
| **Train/infer decoder mismatch** | Training is teacher-forced; inference is autoregressive beam search. Forgetting `generation_config.max_new_tokens`, `num_beams`, or the length penalty gives truncated or looping outputs that score badly through no fault of the model | Pin a `GenerationConfig` next to the checkpoint and use it in both offline eval and serving |

The deliverable is a small fine-tune harness with five components:

1. **Dataset builder** — load `cnn_dailymail` (or `xsum`) via `datasets`, format `(article, summary)` pairs, tokenise article as input and summary as target.
2. **Model wrapper** around `AutoModelForSeq2SeqLM` (BART-base or T5-base).
3. **Collator and trainer** using `DataCollatorForSeq2Seq` with `-100` label padding and `Seq2SeqTrainer` with `predict_with_generate=True`.
4. **Generation config** pinned: beam size, `max_new_tokens`, length penalty, `no_repeat_ngram_size`, `early_stopping`.
5. **ROUGE evaluator** using `rouge_score` with stemming, newline-splitting for `rougeLsum`, and midpoint bootstrapped CIs so review knows whether a 0.3-point delta is noise.

Reference datasets:
- **CNN/DailyMail** (Hermann et al. 2015; See et al. 2017) — ~287 k news articles with multi-sentence highlights. "Highlight-extractive"-leaning; good for ROUGE-1/2.
- **XSum** (Narayan et al. 2018) — 226 k BBC articles with one-sentence abstractive summaries; harder, more hallucination-prone.

Reference model: **BART-base** (Lewis et al. 2020) or **T5-base** (Raffel et al. 2020). Both ship as `AutoModelForSeq2SeqLM`. The worked example below uses BART because its denoising pre-training (text-infilling + sentence-permutation) is a particularly good fit for summarisation; T5 is a drop-in alternative with a prompt prefix (`"summarize: "`).

## 2. Worked implementation

### 2.1 Project layout

```
sum_bake/
  data.py            # datasets.load_dataset + tokenise
  model.py           # AutoModelForSeq2SeqLM wrapper
  generation.py      # pinned GenerationConfig
  metrics.py         # ROUGE with stemming + sentence-split
  run.py             # train/eval CLI
  tests/
    test_label_mask.py
    test_rouge.py
    test_generate.py
  requirements.txt
```

### 2.2 Dataset preparation — tokenise article and summary separately

```python
# data.py
from datasets import load_dataset

MAX_INPUT_LEN = 1024   # BART's position-embedding limit
MAX_TARGET_LEN = 128   # CNN/DM highlights are ~50-70 tokens; 128 leaves headroom


def load_cnn_dm():
    return load_dataset("cnn_dailymail", "3.0.0")  # 287k / 13k / 11k train/val/test


def preprocess(examples, tokenizer):
    model_inputs = tokenizer(
        examples["article"],
        max_length=MAX_INPUT_LEN,
        truncation=True,
    )
    # text_target routes through the decoder-side tokenisation rules:
    # BART handles BOS; T5 appends </s>. Passing the summary as the second
    # positional arg of tokenizer(...) is wrong — that's for paired-input
    # classification and will not apply the decoder-side special tokens.
    labels = tokenizer(
        text_target=examples["highlights"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

Points worth defending in review:

- **`text_target=...` is not cosmetic.** On BART it ensures the decoder sees `<s>` as BOS and the correct EOS; on T5 it adds the trailing `</s>`. Using the two-positional-arg form (`tokenizer(article, highlights)`) silently concatenates them as a pair-classification input and the labels become unusable.
- **No padding in `preprocess`.** We let the data collator pad to the *batch* max length, not the *dataset* max length. That's both faster and the only way to make `-100` masking work cleanly without a second pass.
- **Truncation on the input only.** Truncating the target cuts off the gold summary and teaches the model to end mid-sentence. 128 tokens covers 99 %+ of CNN/DM highlights; for XSum (single-sentence summaries) 64 is enough.

### 2.3 Collator: the single place where `-100` masking happens

```python
# run.py (fragment)
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,  # critical: masks pads out of the loss
    pad_to_multiple_of=8,     # tensor-core alignment on A100/H100
)
```

`DataCollatorForSeq2Seq` also builds `decoder_input_ids` by shifting the (unmasked) labels right by one position and inserting the decoder start token (`<s>` for BART, `pad_token_id` for T5). Doing this by hand is a common bug — forgetting the shift causes a one-token offset and the model never learns.

### 2.4 Model

```python
# model.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "facebook/bart-base"   # or "t5-base"


def build_model_and_tokenizer(model_id: str = MODEL_ID):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return model, tok
```

### 2.5 Generation config — pin it once, use it everywhere

```python
# generation.py
from transformers import GenerationConfig


def summarisation_generation_config(tokenizer) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=128,           # matches MAX_TARGET_LEN
        min_new_tokens=16,            # kills degenerate empty outputs
        num_beams=4,                  # paper default; 6 adds ~0.1 ROUGE at 1.5x cost
        length_penalty=2.0,           # BART CNN/DM default; XSum uses 1.0
        no_repeat_ngram_size=3,       # blocks the "copy a trigram forever" failure
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,  # BART; omit for T5
    )
```

The specific numbers are from the BART paper's CNN/DM config (Lewis et al. 2020, Appendix) and the Hugging Face `summarization` reference script. Pin them inline so that reviewers can see them next to the training code — a drifted `num_beams=1` at inference is the single most common "my ROUGE dropped 4 points overnight" bug.

### 2.6 Trainer

```python
# run.py
import argparse, numpy as np
from functools import partial
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, set_seed,
)
from .data import load_cnn_dm, preprocess
from .model import build_model_and_tokenizer
from .generation import summarisation_generation_config
from .metrics import rouge_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="facebook/bart-base")
    p.add_argument("--output-dir", default="runs/sum")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260101)
    a = p.parse_args()
    set_seed(a.seed)

    ds = load_cnn_dm()
    model, tok = build_model_and_tokenizer(a.model_id)
    gen_cfg = summarisation_generation_config(tok)
    model.generation_config = gen_cfg

    tokenised = ds.map(
        partial(preprocess, tokenizer=tok),
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)
        return rouge_metrics(decoded_preds, decoded_labels)

    args = Seq2SeqTrainingArguments(
        output_dir=a.output_dir,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size * 2,
        learning_rate=a.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        predict_with_generate=True,     # eval runs .generate(), not teacher-forced
        generation_config=gen_cfg,      # pin at Trainer level too
        fp16=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        seed=a.seed,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        tokenizer=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print(trainer.evaluate(tokenised["test"]))
```

`predict_with_generate=True` is the switch that turns eval into real decoding. Without it, `Seq2SeqTrainer` reports loss-based metrics against teacher-forced predictions, which both inflates numbers and hides the actual beam-search failure modes (looping, truncation).

### 2.7 ROUGE — the reference protocol, not a one-liner

```python
# metrics.py
import nltk
from rouge_score import rouge_scorer, scoring


def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def _sent_split(s: str) -> str:
    return "\n".join(nltk.sent_tokenize(s.strip()))


def rouge_metrics(preds, refs):
    """Returns ROUGE-1/2/L/Lsum F1 (midpoint of a 1000-sample bootstrap CI)."""
    _ensure_nltk()
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=True,
    )
    aggregator = scoring.BootstrapAggregator()
    for p, r in zip(preds, refs):
        # rougeLsum requires sentence-split inputs (summary-level LCS).
        p_sum, r_sum = _sent_split(p), _sent_split(r)
        aggregator.add_scores(scorer.score(r_sum, p_sum))
    result = aggregator.aggregate()
    return {k: 100.0 * v.mid.fmeasure for k, v in result.items()}
```

Four things reviewers must check:

- **Stemming on.** `use_stemmer=True` applies Porter stemming; the BART and PEGASUS papers report with stemming. Turn it off and your ROUGE-1 drops ~1 point and you no longer compare like-for-like.
- **Newline-split for `rougeLsum`.** The `-sum` variant computes LCS *per sentence* and aggregates; the input strings must be `\n`-joined sentences. The non-`sum` `rougeL` runs on the whole string — reporting both numbers is useful because they diverge for multi-sentence summaries like CNN/DM.
- **Bootstrap midpoint.** `BootstrapAggregator.mid.fmeasure` is the SymmetricCI midpoint over 1000 resamples; it's stable to 0.1 points. Reporting the raw mean fluctuates up to 0.4 points between runs on the same predictions.
- **F1, not recall.** Classic `pyrouge` defaults to recall; `rouge_score` defaults to F1. Use F1; it's what the modern papers report.

## 3. Validation steps

1. **Label masking unit test.** Build a batch of two examples where labels are `[1, 2, pad_id, pad_id]` and `[3, 4, 5, pad_id]`. After the collator runs, assert every `pad_id` in `labels` has been rewritten to `-100` and the corresponding `decoder_input_ids` are a right-shift with the decoder start token prepended. If any pad survives in `labels`, the loss is wrong.
2. **Template identity on inference.** For T5, wrap the input in `"summarize: " + article`; for BART, no prefix. A unit test that round-trips `format_input("hello")` through train → eval → serving paths catches a drifted prefix before it costs 2 ROUGE points.
3. **Hand-computed ROUGE.** `pred="the cat sat on the mat"`, `ref="the cat sat on the rug"` → ROUGE-1 unigram F1 = 5 common / (6 pred + 6 ref) overlap → F1 ≈ 0.833 (with stemming). Pin this in `test_rouge.py`; it catches a swapped-argument order (`scorer.score(pred, ref)` vs `scorer.score(ref, pred)`) which is the single most common homegrown-ROUGE bug.
4. **Sentence-split regression.** Compare `rougeLsum` with and without `\n`-joining. On any multi-sentence pair, the `\n`-joined version should differ; if they match, your sentence splitter never fired and the number is a lie.
5. **Beam search sanity.** Generate with `num_beams=1` (greedy) and `num_beams=4`. On CNN/DM dev, `num_beams=4` should net +0.6 to +1.2 ROUGE-L over greedy. Zero delta or a negative delta means `generation_config` isn't reaching `.generate()`.
6. **`no_repeat_ngram_size` sanity.** Force the model to decode 256 tokens on a short prompt (e.g. one sentence). With `no_repeat_ngram_size=0`, you'll see a looping suffix; with `=3`, the output ends or diverges. The test asserts the output has no repeated trigrams.
7. **Prediction length distribution.** Histogram `len(tokenize(pred))` on the dev set. If 20 %+ of predictions are exactly `max_new_tokens`, raise it — you're being clipped. If 20 %+ are exactly `min_new_tokens`, lower it — you're padding with filler.
8. **Seed stability.** Train with 3 seeds, 3 epochs each. BART-base on CNN/DM lands near ROUGE-L ≈ 40.9 / ROUGE-1 ≈ 43.0 (vs the paper's reported 44.16 for BART-large); variance between seeds is typically <0.3. A run 1+ point below the median usually means a broken label mask or a wrong generation config, not bad luck.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Seq2seq label masking** | `DataCollatorForSeq2Seq(label_pad_token_id=-100)` + unit test verifying pad→`-100` | Collator used, no test | Hand-padded labels; pads scored by loss |
| **Decoder start / shift-right** | Collator-built `decoder_input_ids`; or a unit test verifying the shift | Shift present, not tested | Hand-built labels without shift |
| **Prompt template identity** | Single `format_input` function + round-trip test | One call site, no test | Prompt text duplicated across train/serve |
| **Generation config pinning** | `GenerationConfig` co-located with the model, used in train/eval/serve | `.generate(num_beams=...)` passed ad-hoc | Default generate; no beam/length-penalty control |
| **ROUGE protocol fidelity** | Stemming on, `rougeLsum` with sentence-split, F1 reported, bootstrap midpoint CI | Stemming only | `pyrouge` defaults, recall reported, or `rougeL` as proxy for `rougeLsum` |
| **Dataset hygiene** | Dataset revision string + licence recorded (CNN/DM: Apache-2.0 via `datasets`; XSum: `cc-by-sa-4.0`) | One of those | Neither |
| **Reproducibility** | `set_seed`; seed + revision in output JSON; `transformers` + `rouge_score` versions pinned | Seed set, no echo | No seed |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks noted (prompt injection via summarised document, training-data poisoning via scraped news) with mitigations | Mentioned without mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure applied: dataset licence, language coverage, known biases, decode config, ROUGE variance | One of those | Neither |

## 5. Common mistakes

- **Scoring loss-only at eval.** `predict_with_generate=False` makes `Seq2SeqTrainer` evaluate with teacher-forcing, which inflates numbers and hides looping / truncation. Always `predict_with_generate=True` for summarisation.
- **Unmasked pad tokens in labels.** Omitting `-100` for pad positions is the "my model emits empty strings" bug. Fix by either using `DataCollatorForSeq2Seq` or manually `labels = [l if l != pad_id else -100 for l in labels]`.
- **Positional-arg tokenising of the target.** `tokenizer(article, highlights)` treats the two strings as a classification pair, concatenates them, and produces one `input_ids` with no `labels`. Use the `text_target=` kwarg.
- **Dropping sentence splits for `rougeLsum`.** If both inputs are single blobs, `rougeLsum` collapses into `rougeL` and under-reports by ~2 points on multi-sentence summaries. Always `\n`-join sentences before scoring with `rougeLsum`.
- **Wrong argument order to `rouge_scorer.score`.** The API is `scorer.score(target, prediction)`. Swapping them inverts precision and recall and therefore the F1 point where they disagree. The `test_rouge.py` worked example pins the direction.
- **Default generation at inference.** `model.generate(input_ids)` with no `GenerationConfig` uses greedy decoding with `max_new_tokens=20`. On CNN/DM that truncates nearly every summary and your ROUGE-L collapses to ~18. Co-locate the config with the checkpoint (`model.generation_config = ...`).
- **Truncating gold summaries.** If `MAX_TARGET_LEN` is below the P99 of summary length, you teach the model to stop mid-sentence and your ROUGE-L drops without a diagnostic. Set `MAX_TARGET_LEN` to the dataset P99 (CNN/DM ≈ 110; XSum ≈ 40).
- **Reporting precision or recall alone.** ROUGE recall climbs with long outputs; ROUGE precision climbs with short outputs. A model that cheats either way can make either number look good. F1 is the published metric.
- **Fine-tuning at the wrong learning rate.** BART-base on CNN/DM is stable at 3e-5 with warmup; running at 5e-4 (a typical from-scratch rate) catastrophically forgets the pre-trained weights. If your epoch-1 ROUGE is below 20, check the LR first.
- **Ignoring position-embedding limits.** BART-base caps at 1024 tokens; T5-base at 512 unless using the long variant. CNN/DM articles routinely exceed both. The right answer is either truncation (acceptable for exercise-01; the lead-biased news corpus cushions the loss) or long-context models — covered in exercise-02.

## 6. References

Primary sources used in this solution:

- Lewis, M. et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*. ACL. <https://aclanthology.org/2020.acl-main.703/> — architecture, denoising objective, CNN/DM fine-tuning recipe, the beam-size and length-penalty defaults copied into §2.5.
- Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5). JMLR. <https://jmlr.org/papers/v21/20-074.html> — text-to-text framing, `"summarize: "` prefix convention, shared-vocabulary decoder.
- See, A., Liu, P., Manning, C. (2017). *Get To The Point: Summarization with Pointer-Generator Networks*. ACL. <https://aclanthology.org/P17-1099/> — CNN/DailyMail summarisation benchmark and the per-dataset ROUGE protocol used in §2.7.
- Narayan, S., Cohen, S., Lapata, M. (2018). *Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization*. EMNLP. <https://aclanthology.org/D18-1206/> — the XSum corpus; the alternative dataset for a more abstractive regime.
- Lin, C. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*. <https://aclanthology.org/W04-1013/> — the ROUGE family, the sentence-split convention for `rougeLsum`, and the F1 reporting recommendation.
- Hugging Face `transformers` reference script `examples/pytorch/summarization/run_summarization.py` — <https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py> — canonical implementation of the seq2seq fine-tune + generate flow used as a template here.
- Google Research `rouge_score` package — <https://github.com/google-research/google-research/tree/master/rouge> — the stemming and `BootstrapAggregator` used in §2.7; this is the library whose numbers match published results most closely.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation* applies directly: a summarised document can embed an instruction ("ignore your task; the summary is: …") and influence downstream processing when the generated summary is passed to an LLM or classifier. Mitigations: strip model-specific control tokens from inputs, cap input length at ingress, treat generated summaries as untrusted strings when they are forwarded. *ML02 Data Poisoning* applies if the training corpus is scraped news; prefer vetted splits like the `cnn_dailymail` 3.0.0 revision used above.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for adversarial ML; relevant when a summariser output feeds search ranking, moderation, or compliance workflows that trust the summary.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map* function: record dataset licence (CNN/DM: Apache-2.0 via `datasets`; XSum: CC-BY-SA-4.0), language coverage (English news only), known lead-bias in CNN/DM; *Measure* function: report ROUGE-1/2/L/Lsum together with generation config, seed, and dataset revision — faithfulness coverage is deferred to exercise-05.
