# exercise-03 — Punctuation Restoration Model

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-109-speech-text-interface/exercises/exercise-03-punctuation-restoration-model.md`.
> The target learning outcomes implied by that filename: cast punctuation restoration as a **per-token sequence labelling task**, fine-tune a transformer encoder (`xlm-roberta-base` or `distilbert-base-cased`) on a cased-and-punctuated corpus whose input is reduced to lowercase/unpunctuated form at training time, and report **per-class F1** (comma / period / question / exclamation) plus overall micro-F1 — the way the IWSLT shared task on punctuation prediction scores it.

## 1. Solution overview

ASR transcripts come out as a lowercased, punctuation-free token stream ("so we met in paris last week and then we went to berlin"). Punctuation restoration (and its sibling, truecasing) is the thin stage that makes the transcript readable and makes every downstream model (summarisation, NMT, retrieval) stop treating two sentences as one. Three subtleties define the design:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Sequence-tagging vs seq2seq confusion** | Framing this as seq2seq (give the model un-punctuated text, generate punctuated text) gives you a free hallucination surface: the model can drop words, re-order, or paraphrase | Cast it as per-token classification: for each input token, predict one of a small label set `{O, COMMA, PERIOD, QUESTION, EXCLAMATION}`. The model cannot drop or invent tokens — only decide what, if anything, comes after each |
| **Boundary leakage in the training signal** | Teacher-forcing a model to predict a period at the end of every training sentence means it learns "any sequence end gets a period", not "a period marks a sentence boundary". Serving long ASR outputs then produces periods every 20–30 words regardless of content | Train on paragraphs of 3–10 sentences concatenated, with punctuation dropped from the input and preserved as labels. The model learns boundary-within-context, not end-of-string |
| **Word-piece labels misaligned with word labels** | A token-classification head over a subword tokeniser needs one label per subword; naively copying the word's label to every subword double-counts in the loss and inflates F1 | Label the *first subword* of each word with the word's label and label the rest `-100`. This is the HF-canonical trick; mirror it exactly for comparability with published numbers |

The deliverable is a punctuation-restoration model + eval harness with five components:

1. **Dataset builder** — load a cased-and-punctuated corpus (IWSLT 2011 / 2012 punctuation task TED Talks, or an open equivalent like OpenSubtitles), strip casing and punctuation from the input, convert punctuation to per-token labels.
2. **Label scheme** — 5 classes: `O`, `COMMA`, `PERIOD`, `QUESTION`, `EXCLAMATION` (optional: `COLON`, `SEMICOLON`). Keep it small; sparse classes drown.
3. **Tokeniser + label alignment** — `xlm-roberta-base` for multilingual, `distilbert-base-cased` for English-only; first-subword labelling, `-100` for the rest.
4. **Model + trainer** — `AutoModelForTokenClassification`, `Trainer` or `accelerate`, `compute_metrics` returning per-class and micro F1 via `seqeval`.
5. **Service wrapper** — a function that takes an un-punctuated string, runs the model, re-inserts punctuation (and optionally capitalises sentence-initial words based on predicted periods / questions / exclamations).

Reference corpora:
- **IWSLT 2011 / 2012 punctuation prediction task** — TED Talks, the canonical English punctuation-restoration benchmark; defined the per-class F1 reporting convention used below.
- **TED Talks transcripts** via `datasets`'s `ted_talks_iwslt` or OpenSubtitles — large, cased, multi-lingual, free.
- **Europarl** (Koehn 2005) — large multilingual parliamentary transcripts; well-punctuated; useful for multilingual training.

Reference checkpoints:
- **`distilbert-base-cased`** — 66 M; English-only; the worked example. Case-sensitive (we need the model to see the case signal in the training data; the *input* is still lowercased).
- **`xlm-roberta-base`** — 278 M; 100 languages; the default for multilingual punctuation restoration.
- **`oliverguhr/fullstop-punctuation-multilang-large`** — a popular off-the-shelf 4-language (EN/DE/FR/IT) XLM-R-based model; good baseline / off-the-shelf comparator.
- **`felflare/bert-restore-punctuation`** — English BERT-base punctuation restoration; also a common comparator.

## 2. Worked implementation

### 2.1 Project layout

```
punct_bake/
  data.py            # text → (tokens, labels)
  labels.py          # id2label / label2id
  model.py           # AutoModelForTokenClassification
  align.py           # word labels → subword labels (first-subword trick)
  metrics.py         # per-class F1 + micro F1 via seqeval
  service.py         # apply predictions to produce final cased + punctuated text
  run.py             # train / eval CLI
  tests/
    test_alignment.py
    test_labels.py
    test_apply.py
  requirements.txt
```

### 2.2 Label scheme — small is better

```python
# labels.py
LABELS = ["O", "COMMA", "PERIOD", "QUESTION", "EXCLAMATION"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

PUNCT_OF = {"COMMA": ",", "PERIOD": ".", "QUESTION": "?", "EXCLAMATION": "!"}
LABEL_OF_PUNCT = {v: k for k, v in PUNCT_OF.items()}
```

- **Five classes, not fifty.** Adding colons, semicolons, em-dashes, and parentheses looks tempting; in practice F1 on the rare classes is <20 % and the common classes degrade. Ship 4–5 classes and let the long tail be `O`.
- **No `BEGIN_SENT` / `END_SENT` meta-labels.** The downstream capitaliser will derive sentence boundaries from PERIOD / QUESTION / EXCLAMATION predictions.
- **Case is handled by the model.** The *input* is lowercased un-punctuated text; the *training signal* is punctuation; the *post-processing* re-capitalises based on the predicted boundaries. Treating casing as a separate 2-class token-classification head over `{keep_lower, capitalise}` is a reasonable extension but doubles the per-class cost.

### 2.3 Data preparation — from punctuated text to (tokens, labels)

```python
# data.py
import re
from .labels import LABEL2ID, LABEL_OF_PUNCT

PUNCT = ".,?!"
_RE = re.compile(rf"([{re.escape(PUNCT)}])")


def text_to_tokens_labels(text: str) -> tuple[list[str], list[int]]:
    """
    Take a cased, punctuated sentence/paragraph and return:
      tokens: list of lowercased words (no punctuation)
      labels: label id for each word (the punctuation that FOLLOWS it, or O)
    """
    # 1. tokenise on whitespace, keeping punctuation attached to the preceding word
    pieces = _RE.sub(r" \1", text).split()
    tokens, labels = [], []
    for piece in pieces:
        if piece in PUNCT:
            if tokens:
                label_name = LABEL_OF_PUNCT.get(piece, "O")
                # overwrite O with the actual punctuation; keep earliest real label on sequence
                labels[-1] = LABEL2ID[label_name]
            continue
        tokens.append(piece.lower())
        labels.append(LABEL2ID["O"])
    return tokens, labels
```

- **Punctuation is a *following* label.** The word before the punctuation carries the label; this is the Tilk & Alumäe 2015 convention and is how published IWSLT numbers are reported.
- **Lowercase the token, keep the label.** The model sees lowercase; the gold label system is still case-independent at this stage. Casing is restored downstream.
- **Multiple punctuation marks on one word.** "Really?!" → label the word with `EXCLAMATION` (or train a two-head extension); do not try to represent both as one label.

### 2.4 Subword alignment — the HF-canonical "label first subword" trick

```python
# align.py
def align_labels(
    tokens: list[str],
    labels: list[int],
    tokenizer,
    max_length: int = 256,
) -> dict:
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    word_ids = enc.word_ids()
    aligned: list[int] = []
    prev = None
    for wid in word_ids:
        if wid is None:
            aligned.append(-100)                 # special tokens
        elif wid != prev:
            aligned.append(labels[wid])          # first subword of a word → real label
        else:
            aligned.append(-100)                 # subsequent subwords → ignored by loss
        prev = wid
    enc["labels"] = aligned
    return enc
```

Points worth defending in review:

- **`is_split_into_words=True`.** Pass pre-tokenised words, not a raw string. The tokeniser then exposes `word_ids()` which maps each resulting subword back to its source word.
- **`-100` on special tokens and non-first subwords.** `nn.CrossEntropyLoss(ignore_index=-100)` skips those positions in the loss. If you put the same label on every subword of a word, the loss over-weights multi-subword words and F1 inflates by a few points.
- **Truncation at 256.** Punctuation restoration works on paragraphs up to a few sentences; 256 covers >99 % of utterances in IWSLT. Longer windows (512) cost compute with marginal F1 gain.

### 2.5 Model

```python
# model.py
from transformers import AutoModelForTokenClassification, AutoTokenizer
from .labels import LABELS, LABEL2ID, ID2LABEL


def build(model_id: str = "distilbert-base-cased"):
    tok = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model, tok
```

- **`add_prefix_space=True`.** RoBERTa / XLM-R family tokenisers require this when `is_split_into_words=True`; without it, the first word's label alignment is off by one.
- **`id2label` / `label2id` on the config.** When you `.save_pretrained()` and reload, the mapping is preserved; `pipeline("token-classification", model=dir)` then returns human-readable labels instead of `LABEL_0`.
- **`distilbert-base-cased`, not `-uncased`.** We lowercase the *input*; the model is case-sensitive so its pre-training signal (BPE with case info) is preserved for the fine-tune. Using the uncased variant throws away capacity for no benefit.

### 2.6 Trainer + metrics

```python
# metrics.py
from seqeval.metrics import classification_report, f1_score
from .labels import ID2LABEL


def seqeval_from_ids(preds, labels):
    """Convert id tensors (preds, labels) into the (list[list[str]]) shape seqeval wants,
    dropping -100 positions."""
    true_seqs, pred_seqs = [], []
    for p, l in zip(preds, labels):
        t_seq, p_seq = [], []
        for pi, li in zip(p, l):
            if li == -100:
                continue
            t_seq.append(ID2LABEL[int(li)])
            p_seq.append(ID2LABEL[int(pi)])
        true_seqs.append(t_seq)
        pred_seqs.append(p_seq)
    return true_seqs, pred_seqs


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    true_seqs, pred_seqs = seqeval_from_ids(preds, labels)
    report = classification_report(true_seqs, pred_seqs, output_dict=True)
    return {
        "f1_micro": f1_score(true_seqs, pred_seqs, average="micro"),
        "f1_macro": f1_score(true_seqs, pred_seqs, average="macro"),
        "f1_comma": report.get("COMMA", {}).get("f1-score", 0.0),
        "f1_period": report.get("PERIOD", {}).get("f1-score", 0.0),
        "f1_question": report.get("QUESTION", {}).get("f1-score", 0.0),
        "f1_exclamation": report.get("EXCLAMATION", {}).get("f1-score", 0.0),
    }
```

- **`seqeval` over token-level `sklearn`.** `seqeval` treats the labels as sequence tags (same convention as NER); per-class F1 reported this way matches the IWSLT shared-task convention.
- **Drop `-100` before scoring.** Subword positions with label `-100` contribute neither to the numerator nor the denominator of any class's precision/recall. If you forget, F1 is diluted by the implicit "correctly predicted `-100` for every subword".
- **Report `f1_comma`, `f1_period`, `f1_question`, `f1_exclamation` separately.** Commas are the common class (~8 % of tokens); periods the next; questions and exclamations are rare (<1 %). Micro-F1 is dominated by comma + period; only per-class F1 shows whether the rare classes are learned at all.

```python
# run.py
import argparse, numpy as np
from functools import partial
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed
from datasets import load_dataset
from .model import build
from .align import align_labels
from .data import text_to_tokens_labels
from .metrics import compute_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="distilbert-base-cased")
    p.add_argument("--output-dir", default="runs/punct")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260101)
    a = p.parse_args()
    set_seed(a.seed)

    model, tok = build(a.model_id)

    ds = load_dataset("ted_talks_iwslt", language_pair=("en", "en"))  # or OpenSubtitles

    def prepare(ex):
        tokens, labels = text_to_tokens_labels(ex["translation"]["en"])
        return align_labels(tokens, labels, tok)

    ds_prepared = ds.map(prepare, remove_columns=ds["train"].column_names)

    args = TrainingArguments(
        output_dir=a.output_dir,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size * 2,
        learning_rate=a.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        seed=a.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds_prepared["train"],
        eval_dataset=ds_prepared["validation"],
        tokenizer=tok,
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()
```

### 2.7 Service wrapper — predictions back to display text

```python
# service.py
import torch
from .labels import ID2LABEL, PUNCT_OF


def restore(text: str, model, tokenizer, device: str = "cuda") -> str:
    tokens = text.lower().split()
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    preds = logits.argmax(dim=-1)[0].tolist()
    word_ids = enc.word_ids(batch_index=0)

    # first-subword prediction per word
    word_preds: list[str] = [None] * len(tokens)
    for sub_i, wid in enumerate(word_ids):
        if wid is None or word_preds[wid] is not None:
            continue
        word_preds[wid] = ID2LABEL[preds[sub_i]]

    # stitch
    out: list[str] = []
    capitalise = True
    for w, lbl in zip(tokens, word_preds):
        out.append(w.capitalize() if capitalise else w)
        capitalise = False
        if lbl in PUNCT_OF:
            out[-1] = out[-1] + PUNCT_OF[lbl]
            if lbl in {"PERIOD", "QUESTION", "EXCLAMATION"}:
                capitalise = True
    return " ".join(out)
```

Three design decisions this bakes in:

- **Capitalise the first token and every post-boundary token.** Driven by predicted PERIOD / QUESTION / EXCLAMATION labels. If the model is uncertain, you under-capitalise mid-document; this is a graceful failure mode.
- **Attach punctuation to the preceding word.** `" ".join(out)` joins with spaces; the punctuation lives inside the token so there is no `"word ,"` artefact.
- **No proper-noun casing.** "paris" and "berlin" stay lowercase unless they begin a sentence. Proper-noun truecasing is a separate NER-adjacent problem (see mod-104 exercise on sequence labelling); call it out explicitly.

## 3. Validation steps

1. **Label alignment unit test.** Tokens `["so", "we", "met"]`, labels `[O, O, PERIOD]`. Encode with `xlm-roberta-base`. Assert the first subword of `"met"` has label id `PERIOD`, every other non-special-token subword has `-100`. If this fails, every F1 number downstream is wrong.
2. **Round-trip through `text_to_tokens_labels` + service wrapper.** `text = "hello, world."` → `(tokens=["hello","world"], labels=[COMMA, PERIOD])`. Feed the tokens back through a model that predicts the same labels; `restore(...)` should return `"Hello, world."`. If capitalisation or punctuation drops, the stitcher is wrong.
3. **Class imbalance smoke test.** Count labels in the training set. Commas should be ~8–10 % of tokens, periods ~5 %, questions / exclamations <1 %. If periods are >30 % you are training on single-sentence examples and the model will learn "end of input → period".
4. **Per-class F1 floor.** On IWSLT 2011 TED Talks, `distilbert-base-cased` fine-tuned 3 epochs lands at micro-F1 ≈ 0.78, comma F1 ≈ 0.70, period F1 ≈ 0.80, question F1 ≈ 0.60. If you get micro-F1 ≈ 0.9+ something is leaking (likely the model is seeing original casing / punctuation in the input).
5. **Order-of-magnitude sanity.** A 1 k-paragraph eval should take <30 s on a single GPU with `distilbert`. If it takes 5 min, you forgot `torch.no_grad()` or the batch size is 1.
6. **Model-config round trip.** `model.save_pretrained(dir)`; `AutoModelForTokenClassification.from_pretrained(dir)`; `pipeline("token-classification", model=dir)` returns real class names, not `LABEL_0`.
7. **Known off-the-shelf baseline.** Run `oliverguhr/fullstop-punctuation-multilang-large` on the same eval. If your fine-tuned model is >5 F1 behind it on English, something is wrong with your data pipeline (likely the tokeniser alignment) — this is a baseline anyone can reproduce in one `pipeline()` call.
8. **Idempotence test on a punctuated input.** `restore("Hello, world.")` must return `"Hello, world."` (or something very close). If running the model on already-punctuated text destroys the punctuation, your preprocessing is wrong (it should accept pre-punctuated input too, or you should enforce lowercasing + strip-punct upstream).
9. **Fairness per utterance length.** Report micro-F1 on short (<10 words), medium (10–30), long (30+) utterances separately. Expect long-utterance comma F1 to be the weakest; if short-utterance F1 is much lower, the model is overfitting to paragraph boundaries.
10. **Seed stability.** Three seeds, same config. Variance on micro-F1 should be <0.5 points. A 2-point swing between seeds flags a config bug (LR too high, batch too small).

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Task framing** | Per-token classification with an explicit label set | Seq2seq framing with safeguards | Seq2seq with free-form output |
| **Label scheme** | 4–5 compact classes; rare classes dropped | 7–10 classes with sparse ones at <10 % F1 | Dozens of classes; overall F1 unreported |
| **Subword alignment** | First-subword label; rest `-100`; alignment unit test | First-subword label, no test | Same label on all subwords |
| **Case + punctuation strip** | Inputs lowercased and stripped during training data build; test asserts no case / punct leaks through | One of the two | Training sees cased / punctuated inputs; task is trivial, F1 near 1 |
| **Metrics protocol** | Per-class + micro-F1 via `seqeval`; IWSLT-style reporting | Overall F1 only | Accuracy on subword tokens |
| **Baseline comparator** | Numbers reported against a known off-the-shelf model (`oliverguhr/fullstop-...`) | Baseline mentioned | Only internal numbers |
| **Service wrapper** | Produces cased + punctuated display text; sentence-initial capitalisation driven by predicted boundaries | Punctuation restored, no casing | Only emits labels |
| **Reproducibility** | Seed set; model config has `id2label`/`label2id`; `transformers` + `seqeval` versions pinned | Seed set, no label map | None |
| **Dataset hygiene** | Corpus + revision + licence recorded (TED/IWSLT, Europarl, OpenSubtitles) | One of those | None |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks noted (punctuation as semantic switch, PII capitalisation leaks in logs) with mitigations | Mentioned only | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure applied: language coverage, label set, per-class F1, utterance-length slice, known failure modes | Some of those | None |

## 5. Common mistakes

- **Framing as seq2seq.** Punctuation restoration is a classic sequence-labelling task; seq2seq lets the model drop or paraphrase tokens. The IWSLT convention, every published model (`oliverguhr/fullstop-...`, `felflare/bert-restore-punctuation`), and every production system does token classification. If you reach for a seq2seq model, you are opting into hallucinations for no accuracy gain.
- **Labelling every subword with the word's label.** Over-counts in the loss, inflates F1 by 2–4 points, and makes numbers incomparable to published results. `-100` on non-first subwords is the correct pattern.
- **Training on single-sentence examples.** The model then equates "end of input" with "period". On long ASR outputs it either predicts a period at the end or cycles periods every few tokens. Concatenate 3–10 sentences in training; keep paragraphs together.
- **Letting the original input leak cased / punctuated characters.** If you forget to lowercase or forget to strip punctuation from the training input, the task becomes "copy the punctuation you can already see" and F1 is near 1. On real ASR output at serving time, F1 craters because there is no punctuation to copy. Always build a pair-level test asserting the input stream has no `[.,?!]` and no uppercase.
- **Dropping question marks from the label set.** Questions are rare (<1 % of tokens) but high-value for downstream understanding ("did you book the flight" vs "did you book the flight?"). Keep QUESTION even with low F1; log it separately so review can notice it degrading.
- **Not comparing to an off-the-shelf baseline.** `pipeline("token-classification", model="oliverguhr/fullstop-punctuation-multilang-large")` is a 10-line baseline. A fine-tune that is 5 F1 behind it on the same eval is a signal — usually an alignment bug.
- **Micro-F1 only.** Micro-F1 is dominated by commas and periods. The model can be useless on questions and your micro-F1 still looks good. Always report per-class F1.
- **Sentence-initial capitalisation from naive rules.** `string.capitalize()` lowercases the rest of the token, so "Paris" becomes "Paris" the first time and "paris" after that. Use `word[0].upper() + word[1:]` to preserve any mid-word case the tokeniser kept.
- **Treating casing as a solved sub-problem.** Proper-noun truecasing is NER-adjacent and *not* solved by this model. "paris" stays "paris" unless it's at a sentence boundary. If the product needs proper-noun casing, that's a separate NER-driven stage and should be scoped as such (mod-104).
- **No seed; no `set_seed`.** Token-classification F1 can swing 1–2 points between seeds; a change-in-F1 of 0.5 is meaningless without a seed baseline. Pin the seed and report variance.
- **`add_prefix_space=False` on RoBERTa / XLM-R.** When `is_split_into_words=True`, these tokenisers need `add_prefix_space=True` to insert a leading-space marker on the first word. Without it, the first word's label alignment is off by one. Model trains, numbers look fine, production is subtly wrong.

## 6. References

Primary sources used in this solution:

- Tilk, O., Alumäe, T. (2015). *LSTM for Punctuation Restoration in Speech Transcripts*. INTERSPEECH. <https://www.isca-speech.org/archive/interspeech_2015/papers/i15_0683.pdf> — the sequence-tagging formulation (label-the-word-before-the-punctuation) used in §2.3.
- Che, X. et al. (2016). *Punctuation Prediction for Unsegmented Transcript Based on Word Vector*. LREC. <https://aclanthology.org/L16-1103/> — per-class F1 reporting + label scheme that IWSLT adopted.
- Federico, M. et al. (2012). *Overview of the IWSLT 2012 Evaluation Campaign*. IWSLT. <https://aclanthology.org/2012.iwslt-evaluation.1/> — the shared-task reference for the punctuation-prediction sub-task and the per-class F1 convention used in §2.6.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL. <https://aclanthology.org/N19-1423/> — the encoder architecture under `distilbert-base-cased`.
- Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-R). <https://arxiv.org/abs/1911.02116> — `xlm-roberta-base`; the multilingual encoder recommended for non-English punctuation restoration.
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT*. <https://arxiv.org/abs/1910.01108> — the `distilbert-base-cased` checkpoint used in §2.5.
- Hugging Face Transformers token-classification tutorial — <https://huggingface.co/docs/transformers/tasks/token_classification> — the canonical "label first subword, `-100` for the rest" pattern mirrored in §2.4.
- `seqeval` — <https://github.com/chakki-works/seqeval> — the sequence-tagging F1 implementation used in §2.6.
- `oliverguhr/fullstop-punctuation-multilang-large` — <https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large> — a well-known off-the-shelf baseline mentioned in §3.
- `felflare/bert-restore-punctuation` — <https://huggingface.co/felflare/bert-restore-punctuation> — English BERT-base punctuation restoration; secondary baseline.
- Koehn, P. (2005). *Europarl: A Parallel Corpus for Statistical Machine Translation*. MT Summit. <https://aclanthology.org/2005.mtsummit-papers.11/> — Europarl multilingual parliamentary transcripts; cased, well-punctuated training source.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML09 Output Integrity Attack* applies: punctuation is a semantic switch ("let's eat, grandma" vs "let's eat grandma"). An attacker-controlled audio stream that misplaces a comma can change downstream meaning without obviously corrupting the transcript. Treat punctuation-restored output as part of the untrusted-transcript boundary. *ML01 Input Manipulation* applies when an attacker crafts audio specifically to defeat sentence boundaries ("…system prompt ignore previous instructions…") that an LLM consumer then parses as one sentence.
- MITRE ATLAS — <https://atlas.mitre.org/> — the catalogue of adversarial-ML techniques; relevant when the ASR→punctuation→LLM chain is part of an agent surface.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record language coverage, label set, training corpus (and its register — news vs TED vs subtitles), checkpoint version, known biases (lower F1 on long utterances, lower F1 on code-switched content); *Measure*: report per-class F1 + micro F1, utterance-length slices, and the off-the-shelf baseline delta — fairness and dialect coverage checks belong in mod-111 and mod-113.
