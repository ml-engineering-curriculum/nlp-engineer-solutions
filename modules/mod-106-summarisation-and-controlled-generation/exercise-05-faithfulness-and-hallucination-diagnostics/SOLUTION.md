# exercise-05 — Faithfulness and Hallucination Diagnostics

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-106-summarisation-and-controlled-generation/exercises/exercise-05-faithfulness-and-hallucination-diagnostics.md`.
> The target learning outcomes implied by that filename: measure how **faithful** a generated summary is to its source — distinguishing **intrinsic** (contradicts the source) from **extrinsic** (new content not in the source) hallucinations — using multiple automatic metrics (**entity overlap**, **NLI-based entailment**, **QA-based (QAGS / QuestEval)** probes) and report them alongside ROUGE so a "high-ROUGE, low-faithfulness" regression is visible.

## 1. Solution overview

Faithfulness is the metric that ROUGE cannot see. A summariser can score 42 ROUGE-L and routinely invent facts; the Maynez et al. 2020 analysis of XSum reports >70 % of neural summaries contain some hallucination. The three failure modes that decide whether a faithfulness pipeline is honest:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **A single "faithfulness score" is treated as truth** | Each proxy (entity overlap, NLI, QAGS) measures a different thing; relying on one and declaring a system faithful is routinely falsified by human eval | Report at least three metrics drawn from different families (lexical, entailment, QA-based) and correlate against a small human-labelled slice |
| **No separation of intrinsic vs extrinsic** | "Hallucination rate = 0.22" tells you nothing about *what kind*. Intrinsic (contradictions) require a different fix than extrinsic (invented entities) | Decompose: entity-level checks catch most extrinsic; NLI catches intrinsic; QA-based catches both but with its own biases |
| **Metric ↔ model collapse** | If the faithfulness metric uses the same base model or same training data as the system under test, the metric's errors correlate with the model's errors; faithfulness scores inflate | Use a **different** backbone for the metric (e.g. a RoBERTa NLI model from MNLI) than the generator; document the pairing and swap periodically |

The deliverable is a diagnostic harness with four components:

1. **Entity-overlap probe** — extract named entities from source and summary with spaCy or Stanza; report the fraction of summary entities that appear in the source.
2. **NLI entailment probe** — for each summary sentence, treat the source as the premise and run a RoBERTa-MNLI model; label `entailment / neutral / contradiction`; aggregate to a faithfulness score.
3. **QA-based probe** — generate questions from the summary, answer them against the summary and against the source, and compare; a disagreement is evidence of hallucination (QAGS: Wang et al. 2020).
4. **Human-labelled calibration slice** — ~100 hand-annotated summaries from the dev set labelled `faithful` / `intrinsic` / `extrinsic`; used to compute Pearson / Spearman correlation between each automatic metric and human judgment.

Reference datasets:
- **CNN/DailyMail** (less hallucination-prone; good baseline).
- **XSum** (highly abstractive; the reference corpus for hallucination research — Maynez et al. 2020 report ~70 % of fine-tuned-BART summaries contain hallucinations).
- **FRANK** benchmark (Pagnoni et al. 2021) — 2 250 summaries with fine-grained faithfulness labels; the canonical benchmark for automatic faithfulness metrics.

Reference models for the metrics:
- **spaCy** `en_core_web_sm` or `_trf` for entity extraction.
- **`roberta-large-mnli`** (Liu et al. 2019 + Williams et al. 2018 MNLI) for NLI entailment.
- Any seq2seq summariser from exercise-01 as the system under test.

## 2. Worked implementation

### 2.1 Project layout

```
faithfulness/
  entities.py        # spaCy-based entity overlap probe
  nli.py             # RoBERTa-MNLI entailment probe
  qags.py            # QAGS-style QA agreement probe
  calibrate.py       # human-label slice + correlation
  metrics.py         # aggregator producing the diagnostic table
  run.py             # CLI
  tests/
    test_entities.py
    test_nli.py
    test_qags.py
    test_calibration.py
  data/
    human_labels.jsonl   # 100-sample hand-annotated slice
  requirements.txt
```

### 2.2 Entity-overlap probe

```python
# entities.py
import spacy

_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


_HALLUCINATION_SAFE_LABELS = {"DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT"}


def _ents(text: str):
    doc = _nlp()(text)
    return {(e.text.lower().strip(), e.label_) for e in doc.ents}


def entity_overlap(source: str, summary: str, *, include_quant: bool = True):
    src_ents = _ents(source)
    sum_ents = _ents(summary)
    if include_quant is False:
        src_ents = {e for e in src_ents if e[1] not in _HALLUCINATION_SAFE_LABELS}
        sum_ents = {e for e in sum_ents if e[1] not in _HALLUCINATION_SAFE_LABELS}
    if not sum_ents:
        return {"precision": 1.0, "missing": [], "n_sum": 0, "n_src": len(src_ents)}
    missing = sum_ents - src_ents
    return {
        "precision": 1 - len(missing) / len(sum_ents),
        "missing": sorted(missing),
        "n_sum": len(sum_ents),
        "n_src": len(src_ents),
    }
```

Decisions worth defending in review:

- **Precision, not recall.** Faithfulness is "what the summary says is in the source", not "the summary covers all source entities" (that's informativeness, measured by ROUGE-recall). Pointing the ratio in the opposite direction makes verbose summaries look faithful by construction.
- **Return the missing list, not just the number.** Reviewing the specific invented entities is the fastest way to triage a faithfulness regression. A number alone hides which type of entity is being hallucinated (`PERSON`? `ORG`? `GPE`?).
- **`include_quant=True` by default.** Numeric hallucinations (dates, amounts) are the most dangerous category in domains like finance / healthcare; keeping them in the overlap is deliberate. The flag exists for comparisons with papers that excluded them.
- **Case-folded, whitespace-normalised match.** "the United States" vs "United States" must not count as a hallucination; a loose string match is the practical choice. For higher precision, replace with coreference-resolved entity clustering (adds a dependency).

### 2.3 NLI entailment probe

```python
# nli.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "roberta-large-mnli"
LABELS = ["contradiction", "neutral", "entailment"]


class NLIFaithfulness:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def score(self, source: str, summary_sentence: str) -> dict:
        enc = self.tok(source, summary_sentence, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        logits = self.model(**enc).logits[0]
        probs = logits.softmax(-1).tolist()
        return dict(zip(LABELS, probs))


def sentence_level_nli(source: str, summary: str, nli: NLIFaithfulness) -> dict:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    sents = nltk.sent_tokenize(summary)
    scores = [nli.score(source, s) for s in sents]
    entail = sum(s["entailment"] for s in scores) / max(1, len(scores))
    contra = sum(s["contradiction"] for s in scores) / max(1, len(scores))
    n_strong_contra = sum(1 for s in scores if s["contradiction"] > 0.5)
    return {
        "mean_entailment": entail,
        "mean_contradiction": contra,
        "n_sentences": len(scores),
        "n_strong_contradictions": n_strong_contra,
    }
```

Design decisions:

- **Sentence-level, not summary-level.** Running NLI on the whole summary against the whole source lets the model trade a contradicting sentence against a supported one. Per-sentence aggregation preserves the signal.
- **Three outputs: `mean_entailment`, `mean_contradiction`, and `n_strong_contradictions`.** A summary with one strongly contradicted sentence is less trustworthy than one with three weakly neutral sentences and the same `mean_entailment`.
- **Truncation on the premise side.** RoBERTa-MNLI tops out at 512 tokens total; for long sources, `truncation=True` drops premise tail. A better long-doc version chunks the premise and takes the per-sentence max over chunks — see §2.5 for the matching pattern in QAGS.
- **MNLI, not SNLI.** MNLI covers more domains (government, slate, telephone, fiction, travel); SNLI is Flickr captions only and generalises poorly to news summaries.

### 2.4 QAGS probe

QAGS (Wang et al. 2020) and QuestEval (Scialom et al. 2021) are the published "generate questions from the summary, answer from source AND from summary, compare" pattern. The reference implementation uses two pre-trained models: a question generator and an extractive QA model.

```python
# qags.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

QG_ID = "mrm8488/t5-base-finetuned-question-generation-ap"    # question generator
QA_ID = "distilbert-base-uncased-distilled-squad"             # extractive QA


class QAGS:
    def __init__(self):
        self.qg_tok = AutoTokenizer.from_pretrained(QG_ID)
        self.qg = AutoModelForSeq2SeqLM.from_pretrained(QG_ID).eval()
        self.qa = pipeline("question-answering", model=QA_ID)

    def generate_questions(self, summary: str, max_qs: int = 5) -> list[str]:
        prompt = f"generate questions: {summary}"
        ids = self.qg_tok(prompt, return_tensors="pt", truncation=True).input_ids
        out = self.qg.generate(
            ids,
            num_beams=4,
            num_return_sequences=max_qs,
            max_new_tokens=32,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        return list({self.qg_tok.decode(o, skip_special_tokens=True).strip() for o in out})

    def answer(self, question: str, context: str) -> str:
        if not context.strip():
            return ""
        return self.qa(question=question, context=context[:5000])["answer"]

    def score(self, source: str, summary: str) -> dict:
        questions = self.generate_questions(summary)
        if not questions:
            return {"qags_f1": 1.0, "n_questions": 0}
        f1s = []
        for q in questions:
            a_src = self.answer(q, source)
            a_sum = self.answer(q, summary)
            f1s.append(_token_f1(a_src, a_sum))
        return {"qags_f1": sum(f1s) / len(f1s), "n_questions": len(questions)}


def _token_f1(a: str, b: str) -> float:
    at, bt = a.lower().split(), b.lower().split()
    if not at or not bt:
        return float(at == bt)
    common = set(at) & set(bt)
    if not common:
        return 0.0
    p = len(common) / len(at)
    r = len(common) / len(bt)
    return 2 * p * r / (p + r)
```

Design decisions:

- **Question generation off the summary, not the source.** The QAGS invariant: "a summary should answer the same questions against the source as against itself". Generating from the source and testing against the summary measures *coverage*, not faithfulness.
- **Extractive QA, not generative.** A generative QA head can hallucinate its own answers, confounding the probe. Extractive QA either finds an answer span or returns an empty string.
- **Token-F1 agreement metric.** Exact-match on free-form questions is too strict (paraphrase penalises); F1 over normalised tokens is the QAGS-paper choice.
- **Empty source answer → zero F1.** If the QA model cannot find the answer in the source, that's prima facie evidence the summary invented it — don't fall back to "treat as neutral".

### 2.5 Calibration — the human-labelled slice

```python
# calibrate.py
import json
from scipy.stats import pearsonr, spearmanr


def load_human_labels(path: str) -> list[dict]:
    """Each record: {id, source, summary, label}; label in {faithful, intrinsic, extrinsic}."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def correlate(metrics: list[dict], human_labels: list[dict], metric_key: str) -> dict:
    label_to_score = {"faithful": 1.0, "intrinsic": 0.0, "extrinsic": 0.0}
    ys = [label_to_score[r["label"]] for r in human_labels]
    xs = [m[metric_key] for m in metrics]
    if len(xs) < 5:
        return {"pearson_r": None, "spearman_r": None, "n": len(xs), "note": "insufficient samples"}
    pr, _ = pearsonr(xs, ys)
    sr, _ = spearmanr(xs, ys)
    return {"pearson_r": pr, "spearman_r": sr, "n": len(xs)}
```

The ~100-sample slice is the discipline that keeps the metrics honest. Without it, a drop in `mean_entailment` from 0.72 to 0.68 is just a number; with it, you know whether the metric correlates with human judgment above r=0.3 (useful) or below r=0.1 (noise). Pagnoni et al. 2021's FRANK finding was precisely that single automatic metrics correlate 0.2–0.5 with humans; combining three raises the ceiling to ~0.6.

### 2.6 Aggregator — the diagnostic table

```python
# metrics.py
from .entities import entity_overlap
from .nli import NLIFaithfulness, sentence_level_nli
from .qags import QAGS


def faithfulness_report(source: str, summary: str, nli: NLIFaithfulness, qags: QAGS) -> dict:
    return {
        "entity": entity_overlap(source, summary),
        "nli": sentence_level_nli(source, summary, nli),
        "qags": qags.score(source, summary),
    }
```

The row is designed so a regression is visible in at least one metric even if the others miss it. In practice:

- **Extrinsic entity hallucination** → entity precision drops, NLI mean_entailment drops slightly, QAGS QA agreement drops.
- **Intrinsic contradiction (e.g. "X said Y" where X said not-Y)** → entity precision stays high, NLI contradiction spikes, QAGS QA F1 drops to zero for the relevant question.
- **Vague / under-informative summary** → entity precision high (nothing to hallucinate), NLI entailment medium, QAGS F1 medium (QA returns empty/short answers on both sides, which agree trivially).

### 2.7 CLI wiring

```python
# run.py
import json, argparse
from datasets import load_dataset
from .nli import NLIFaithfulness
from .qags import QAGS
from .metrics import faithfulness_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="JSONL with id, source, summary")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    nli = NLIFaithfulness()
    qags = QAGS()
    results = []
    with open(a.predictions) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            r = faithfulness_report(row["source"], row["summary"], nli, qags)
            r["id"] = row["id"]
            results.append(r)
    with open(a.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
```

## 3. Validation steps

1. **Known-faithful control.** Pick 5 dev examples where the summary is `source[:200]` (literal copy). Entity precision = 1.0, NLI mean_entailment > 0.90, QAGS F1 > 0.80. Any lower = a metric is broken.
2. **Known-hallucinating control.** Edit a summary to insert `", said President Zelensky"` on an unrelated article. Entity precision drops (`PERSON` added), NLI mean_contradiction rises (or neutral at least), QAGS generates a `Who said …?` question whose source answer is empty → F1 low. All three signals must move.
3. **Known-contradiction control.** Negate a quantitative claim: swap "increased by 5%" → "decreased by 5%". Entity precision may stay at 1.0 (numbers match); NLI contradiction must spike for the sentence; QAGS F1 drops on the quantitative question. The intrinsic/extrinsic decomposition is the point.
4. **Calibration slice correlation.** On the 100-sample human slice: `pearson_r(entity_precision, human_label) >= 0.25`, `pearson_r(mean_entailment, human_label) >= 0.30`, `pearson_r(qags_f1, human_label) >= 0.25`. Below those floors, the pairing of metric and corpus is wrong (common cause: XSum with CNN/DM-tuned QG model).
5. **Metric-model independence.** Using a BART-MNLI classifier to score BART-base summaries is the "metric ↔ model collapse" failure. Swap the NLI backbone to DeBERTa-MNLI and verify the ranking of 10 systems does not flip. If it does, the metric is scoring BART-style artefacts, not faithfulness.
6. **ROUGE-faithfulness scatter.** On the dev set, plot ROUGE-L against `mean_entailment`. The correlation should be weak (r < 0.4); otherwise you've measured a ROUGE proxy, not faithfulness.
7. **Sentence-level aggregation stability.** Replace `mean` with `min` across sentences. A summary that scores highly on mean but low on min has one-bad-sentence structure — surface this in the report with `n_strong_contradictions`.
8. **Latency floor.** QAGS is the slowest probe (two model passes per question, 5 questions per summary). Budget 1–3 seconds per summary on CPU; 100–300 ms on a single GPU. If it's <50 ms, the QG model is returning cached outputs.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Metric breadth** | At least one metric each from {lexical/entity, NLI-entailment, QA-based} | Two of the three | Single metric reported as "faithfulness" |
| **Intrinsic/extrinsic split** | Report decomposes by type (entity-level for extrinsic, NLI contradiction for intrinsic) | One axis of decomposition | Single scalar |
| **Metric-model independence** | NLI / QA backbone distinct from summariser backbone; swap test documented | Same family, documented | Same checkpoint used as metric and generator |
| **Human calibration slice** | ≥50 hand-labelled examples; Pearson / Spearman reported per metric | 20–50 examples | No human calibration |
| **ROUGE reported alongside** | ROUGE-L + at least one faithfulness metric in the same table; low correlation documented | ROUGE table and faithfulness table separate | ROUGE only (missed the exercise) |
| **Entity-overlap direction** | **Precision** (summary → source), not recall; the `missing` list surfaced per-row | Precision only | Recall reported as "faithfulness" |
| **QAGS wiring** | Questions generated from summary, answered against both; extractive QA; F1 metric | Correct direction, wrong metric | Generative QA on both sides |
| **Reproducibility** | NLI / QG / QA model ids + revisions pinned; seed set; report JSON emitted | Models pinned, no seed | Nothing pinned |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS: adversarial source text designed to pass faithfulness checks, metric-model identity exploited by attacker | Mentioned without mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure: each metric's model + revision + intended use; known failure modes (vague summaries) | One of those | Neither |

## 5. Common mistakes

- **Reporting a single "faithfulness" number.** Any such number is either defensive (collapses three signals into one) or misleading. Report the three (entity precision, NLI mean_entailment / mean_contradiction, QAGS F1) and let the reader see the shape.
- **Running NLI on the whole summary as a single hypothesis.** A long multi-sentence hypothesis is almost always `neutral` because truncation and sentence-level contradictions average out. Split summaries into sentences; score each.
- **Entity overlap as recall.** "The summary mentions all source entities" is informativeness; the faithfulness direction is "all summary entities come from the source" — i.e. precision.
- **Metric-model collapse.** Using BART-MNLI to score BART-base summaries looks fine on paper and scores higher than the metric should. Pair a different backbone family (RoBERTa or DeBERTa for the NLI head when the generator is BART).
- **QG off the source instead of the summary.** QAGS's whole invariant is "same questions, answered against both". Generate from the summary; otherwise you've measured coverage, not faithfulness.
- **Generative QA for QAGS.** A generative QA head can produce an answer even when none is present in the source; compared to the summary's answer, they can agree while both are hallucinated. Extractive QA fails loudly on absent answers; that's the point.
- **Ignoring vague-summary trivial agreement.** A summary that says "an event occurred" generates questions like "what occurred?" where both answers are "an event" → F1=1.0. Flag summaries with fewer than N entities or fewer than N content words before scoring.
- **Omitting human calibration.** Without a human-labelled slice, the correlation of your automatic metrics with "actual" faithfulness is unknown. ~100 labels is the minimum defensible count; Pagnoni et al. 2021 use 2 250 for FRANK.
- **Comparing ROUGE and faithfulness on the same axis as "quality".** They measure different things; a system that gains 2 ROUGE and loses 0.1 entailment is worse for high-trust applications and better for low-trust ones. Report both and let the use case decide.
- **Averaging NLI probabilities across sentences and reporting one number.** A single `mean_contradiction > 0.5` sentence is a production bug; averaging it down to 0.08 hides it. Surface `n_strong_contradictions` or `min_entailment` next to the mean.
- **Token-F1 without normalisation.** QAGS answers from two models may differ in punctuation / articles; use the SQuAD-style normaliser from exercise-01 §2.6 to avoid false disagreements.

## 6. References

Primary sources used in this solution:

- Maynez, J. et al. (2020). *On Faithfulness and Factuality in Abstractive Summarization*. ACL. <https://aclanthology.org/2020.acl-main.173/> — the paper that quantified hallucination rates on XSum; defines the **intrinsic / extrinsic** decomposition used in §1 and §4.
- Kryscinski, W. et al. (2020). *Evaluating the Factual Consistency of Abstractive Text Summarization* (FactCC). EMNLP. <https://aclanthology.org/2020.emnlp-main.750/> — early NLI-based faithfulness metric; informs the sentence-level NLI design in §2.3.
- Wang, A., Cho, K., Lewis, M. (2020). *Asking and Answering Questions to Evaluate the Factual Consistency of Summaries* (QAGS). ACL. <https://aclanthology.org/2020.acl-main.450/> — defines the QG + QA agreement pattern reimplemented in §2.4.
- Scialom, T. et al. (2021). *QuestEval: Summarization Asks for Fact-based Evaluation*. EMNLP. <https://aclanthology.org/2021.emnlp-main.529/> — extension of the QA-based pattern; adds answerability weighting; a drop-in alternative to §2.4.
- Pagnoni, A. et al. (2021). *Understanding Factuality in Abstractive Summarization with FRANK: A Benchmark for Factuality Metrics*. NAACL. <https://aclanthology.org/2021.naacl-main.383/> — the FRANK benchmark; informs the calibration-slice approach in §2.5.
- Falke, T. et al. (2019). *Ranking Generated Summaries by Correctness: An Interesting but Challenging Application for Natural Language Inference*. ACL. <https://aclanthology.org/P19-1213/> — motivates NLI as a faithfulness proxy and documents its known failures.
- Williams, A., Nangia, N., Bowman, S. (2018). *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference* (MNLI). NAACL. <https://aclanthology.org/N18-1101/> — the training data behind `roberta-large-mnli`; motivates MNLI-over-SNLI choice in §2.3.
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. arXiv:1907.11692. <https://arxiv.org/abs/1907.11692> — the base model behind `roberta-large-mnli`.
- Rajpurkar, P. et al. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text*. EMNLP. <https://aclanthology.org/D16-1264/> — SQuAD dataset and normaliser; the F1 convention used in §2.4.
- Hugging Face model card — `roberta-large-mnli` — <https://huggingface.co/roberta-large-mnli> — label-id ordering (contradiction=0, neutral=1, entailment=2) required to decode logits correctly in §2.3.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation*: an attacker can craft source text that induces specific hallucinations (prompt-injection via the document). *ML09 Output Integrity Attacks*: when a summariser's output drives downstream decisions (ticket routing, moderation), hallucinations are not a quality issue but a security one. Mitigations: faithfulness gates in the pipeline, flagging summaries with `n_strong_contradictions > 0` for human review, metric-model independence to resist attacker tuning against a specific metric.
- MITRE ATLAS — <https://atlas.mitre.org/> — relevant technique entries for data-evasion attacks that exploit metric-model identity (adversarial text crafted to score high under a known faithfulness metric while still misleading humans).
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: document each metric's backbone model, revision, and pairing with the system under test; record the human-labelled calibration slice as part of the release artefact. *Measure*: report faithfulness with ROUGE, not instead of it; surface decomposed metrics (entity, NLI, QAGS) so trade-offs across metrics are visible. *Manage*: faithfulness regressions gate release; a system cannot ship with a regression in `mean_contradiction` above a documented threshold even if ROUGE improves.
