# exercise-03 — Long-Context and Multi-Hop QA

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-105-question-answering-and-machine-reading/exercises/exercise-03-long-context-and-multi-hop-qa.md`.
> The target learning outcomes implied by that filename: handle contexts that exceed a dense encoder's window (**long-context QA**: Longformer / BigBird / retrieve-then-read), and answer questions whose evidence is spread across **multiple passages** (**multi-hop QA** on HotpotQA / 2WikiMultiHopQA), including reporting **supporting-fact** metrics, not just answer EM/F1.

## 1. Solution overview

Two failure modes dominate once the context no longer fits in a 512-token encoder:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **One window, hope for the best** | Truncating a 20-page document to the first 512 tokens drops the answer on the majority of questions whose evidence lives past position 1 000. Striding (exercise-01) gives you many windows but no way to combine their evidence | Pick one of two architectures: **sparse-attention encoder** (Longformer / BigBird) that extends the window to 4 096 / 8 192 tokens natively; or **retrieve-then-read** (passage retriever + reader) that narrows the window before reading. For multi-doc questions, **Fusion-in-Decoder** (FiD) encodes top-k passages independently and lets the decoder attend over all of them |
| **Single-hop reader on multi-hop questions** | On HotpotQA, 100 % of questions need ≥2 passages to answer. A reader given one gold passage at random scores ~F1 40 and looks fine on "answer F1" while being 0 % correct on "joint F1 with supporting facts" | Evaluate with HotpotQA's **joint metric**: a prediction counts only if *both* the answer *and* the set of supporting sentences are correct. Train a model that also predicts supporting-fact sentences (binary head over sentences, or sentence tokens in the decoder) |

And one failure mode that spans both:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Position-embedding mismatch** | Loading a BERT checkpoint and running it on >512 tokens triggers an index error — or, worse, silently wraps the position IDs | Either pick a model whose position embeddings cover the length you need (Longformer: 4 096; BigBird: 4 096; T5 relative-position: unbounded in principle) or extend the embeddings explicitly (and retrain) rather than hoping the mask hides it |

The deliverable is two interchangeable readers plus a shared evaluator:

1. **Long-context single-reader** — Longformer-base with 4 096-token windows, extractive head, global attention on question tokens + supporting-fact head.
2. **Retrieve-then-read** — BM25 or dense retriever picks top-k paragraphs, Fusion-in-Decoder reads them, generates the answer plus a list of cited titles.
3. **Evaluator** — HotpotQA protocol: **answer-EM / answer-F1**, **sup-EM / sup-F1**, **joint-EM / joint-F1**.

Reference datasets: **HotpotQA** (Yang et al. 2018) with the **distractor** setting (10 paragraphs: 2 gold, 8 distractors). **2WikiMultiHopQA** (Ho et al. 2020) is a harder second benchmark where reasoning chains are annotated. **Natural Questions** (Kwiatkowski et al. 2019) is an alternative open-domain long-context benchmark.

Reference models: **Longformer-base** (Beltagy, Peters, Cohan 2020) for the long-context single reader; **FiD** (Izacard & Grave 2021) layered on **T5-base** for the retrieve-then-read path.

## 2. Worked implementation

### 2.1 Project layout

```
qa_multi/
  data.py              # HotpotQA load + per-architecture feature building
  retriever.py         # BM25 + optional DPR dense retriever
  reader_long.py       # Longformer extractive + sup-fact head
  reader_fid.py        # FiD wrapper over T5
  metrics.py           # answer / sup-fact / joint EM & F1
  run_long.py          # train/eval CLI for Longformer path
  run_fid.py           # train/eval CLI for FiD path
  tests/
    test_global_mask.py
    test_hotpot_metrics.py
    test_fid_fanout.py
  requirements.txt
```

### 2.2 Dataset shape

HotpotQA-distractor items look like:

```python
{
  "question": "Which film came out first: Movie A or Movie B?",
  "answer": "Movie A",
  "context": [            # 10 paragraphs; 2 are gold
    {"title": "Movie A", "sentences": ["Movie A is a 1999 film.", "..."]},
    ...
  ],
  "supporting_facts": [   # sentences needed to answer
    {"title": "Movie A", "sent_id": 0},
    {"title": "Movie B", "sent_id": 0},
  ],
  "type": "comparison",   # or "bridge"
  "level": "medium",
}
```

Two engineered columns are useful downstream: a **flattened** `context_text` (title-prefixed, sentence-delimited) that the reader consumes, and a **sent_span** map `[(sent_global_idx, char_start, char_end)]` so sup-fact predictions can be scored against the gold `(title, sent_id)` pairs.

### 2.3 Path A — Longformer single reader

```python
# reader_long.py
import torch
from transformers import LongformerModel, LongformerTokenizerFast

MODEL_ID = "allenai/longformer-base-4096"


class LongformerQAWithSupFacts(torch.nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.backbone = LongformerModel.from_pretrained(MODEL_ID)
        hidden = self.backbone.config.hidden_size
        self.qa_outputs = torch.nn.Linear(hidden, 2)          # start / end logits
        self.sent_cls   = torch.nn.Linear(hidden, 1)          # per-sentence relevance

    def forward(self, input_ids, attention_mask, global_attention_mask,
                sent_token_positions, start_positions=None, end_positions=None,
                sent_labels=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        seq = out.last_hidden_state

        start_end = self.qa_outputs(seq)
        start_logits, end_logits = start_end.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits   = end_logits.squeeze(-1)

        # sent_token_positions: [B, N_sents] -- index of the first token of each sentence.
        sent_hidden = torch.gather(
            seq, 1, sent_token_positions.unsqueeze(-1).expand(-1, -1, seq.size(-1))
        )
        sent_logits = self.sent_cls(sent_hidden).squeeze(-1)

        total = None
        if start_positions is not None:
            ce = torch.nn.CrossEntropyLoss()
            bce = torch.nn.BCEWithLogitsLoss()
            total = ce(start_logits, start_positions) + ce(end_logits, end_positions)
            if sent_labels is not None:
                total = total + 0.5 * bce(sent_logits, sent_labels.float())
        return {"loss": total, "start_logits": start_logits, "end_logits": end_logits,
                "sent_logits": sent_logits}
```

Four design decisions worth defending in review:

- **Global attention on the question (and on each sentence-start token).** Longformer uses windowed local attention everywhere except on tokens flagged in `global_attention_mask`, which get full attention. Setting global attention on question tokens is required by the Longformer paper (§3.2); adding it to each sentence-start token is what lets the sentence head see the question even when it is far away in the sequence.
- **Separate linear head for sentence relevance.** Reusing the start-logits is a tempting shortcut that corrupts both signals. Keep them separate; use `BCEWithLogitsLoss` on the sentence head so it's calibrated as independent per-sentence probabilities (sup-fact labels are sparse multi-label).
- **Loss weighting at 0.5.** The QA loss operates over 4 096 tokens; the sentence loss operates over ~40 sentences. Without a scale, the sentence loss is dwarfed and the sup-fact head never converges. 0.5 is the reference weight from the HotpotQA baseline model; tune in {0.25, 0.5, 1.0} if sup-F1 lags.
- **`attention_mask` ≠ `global_attention_mask`.** The former gates padding (1 for real, 0 for pad); the latter gates attention *pattern* (0 for local-window, 1 for global). The two must be constructed separately — zeroing a global token because it is padding is correct; zeroing a global token because the mask was copied wholesale from `attention_mask` is the most common Longformer bug.

### 2.4 Path B — Retrieve-then-read with Fusion-in-Decoder

```python
# retriever.py (BM25 baseline; DPR optional)
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, paragraphs, tokenise):
        self.paragraphs = paragraphs
        self.tokenise = tokenise
        self.index = BM25Okapi([tokenise(p["text"]) for p in paragraphs])

    def retrieve(self, question: str, k: int = 10):
        scores = self.index.get_scores(self.tokenise(question))
        topk = scores.argsort()[::-1][:k]
        return [self.paragraphs[i] | {"score": float(scores[i])} for i in topk]
```

```python
# reader_fid.py
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_ID = "t5-base"


def encode_fid(question: str, passages, tokenizer, max_len=256):
    """Each passage is encoded independently as "question: Q context: title. text" and
    the k encoded tensors are concatenated on the sequence axis before the decoder
    cross-attends to them. This is the Fusion-in-Decoder trick: the encoder stays small
    (k passes of length 256), but the decoder sees the equivalent of a k*256-token
    context without ever instantiating that long sequence.
    """
    input_ids_list, mask_list = [], []
    for p in passages:
        text = f"question: {question} title: {p['title']} context: {p['text']}"
        enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length",
                        return_tensors="pt")
        input_ids_list.append(enc.input_ids)
        mask_list.append(enc.attention_mask)
    # Shape: [k, 1, L] → [1, k, L] → consumed by FiD-wrapped T5.
    import torch
    return (
        torch.cat(input_ids_list, dim=0).unsqueeze(0),
        torch.cat(mask_list,      dim=0).unsqueeze(0),
    )
```

FiD differs from the T5 baseline in one place: the encoder is called `k` times (once per passage) and the resulting `[k, L, H]` tensor is reshaped to `[1, k*L, H]` before the decoder's cross-attention. In practice this is done by overriding `T5Stack.forward` or using the `FiDT5` wrapper from the Izacard & Grave reference implementation. The key property is that **encoder compute scales linearly in k** (not k²), so you can condition on 10 passages without the quadratic blow-up of putting them all into one long sequence.

Train the model to generate not only the answer but also the list of cited paragraph titles, e.g. `"Movie A // Movie A, Movie B"` — the second segment lets you evaluate supporting-document F1 without a separate head. (For sentence-level supporting-fact F1 you still need a sentence head or sentence-token generation; the title-level is a reasonable approximation and matches the HotpotQA fullwiki setting.)

### 2.5 HotpotQA metrics — the joint score is the point

```python
# metrics.py
import collections, re, string


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _em(p, g): return int(_normalize(p) == _normalize(g))


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


def _set_f1(pred_set, gold_set):
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0
    tp = len(pred_set & gold_set)
    pr = tp / len(pred_set)
    rc = tp / len(gold_set)
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
    return pr, rc, f1


def hotpot_metrics(predictions):
    """predictions: list of
        {"id","answer_pred","answer_gold","sup_pred":{(title,sent_id)},
         "sup_gold":{(title,sent_id)}}
    """
    a_em = a_f1 = s_em = s_f1 = j_em = j_f1 = 0.0
    n = len(predictions)
    for ex in predictions:
        ae = _em(ex["answer_pred"], ex["answer_gold"])
        af = _f1(ex["answer_pred"], ex["answer_gold"])
        se = int(ex["sup_pred"] == ex["sup_gold"])
        _, _, sf = _set_f1(ex["sup_pred"], ex["sup_gold"])
        a_em += ae; a_f1 += af
        s_em += se; s_f1 += sf
        j_em += ae * se
        j_f1 += af * sf           # product, not min — the HotpotQA convention
    return {
        "ans_em": 100 * a_em / n, "ans_f1": 100 * a_f1 / n,
        "sup_em": 100 * s_em / n, "sup_f1": 100 * s_f1 / n,
        "joint_em": 100 * j_em / n, "joint_f1": 100 * j_f1 / n,
    }
```

The **product** aggregation for joint-F1 (`answer_F1 * sup_F1`) is the HotpotQA paper's definition (Yang et al. 2018 §5). It is what makes joint-F1 punish single-hop shortcuts: a reader that gets the answer by copying surface patterns scores high `ans_f1` and low `sup_f1`, and the product flattens.

### 2.6 Training loop sketch (Longformer path)

```python
# run_long.py (abridged)
# 1. Flatten HotpotQA context into one long passage, tracking sentence spans.
# 2. Tokenise with Longformer tokenizer (max_length=4096, truncation=True).
# 3. Build start/end positions by searching flattened text for the gold answer string.
# 4. Build sent_labels as a 0/1 vector over every sentence in the flattened passage.
# 5. Build global_attention_mask: 1 on every question token and on sentence-start tokens.
# 6. Train with HuggingFace Trainer, custom compute_loss = model output's "loss".
# 7. At eval: argmax(start_logits, end_logits) within context, decode; sigmoid(sent_logits)>0.5
#    → predicted supporting facts; compare to gold with hotpot_metrics.
```

### 2.7 Training loop sketch (FiD path)

```python
# run_fid.py (abridged)
# 1. Build a BM25 (or DPR) index over all paragraphs in the training corpus.
# 2. For each example, retrieve top-k paragraphs; if the gold paragraphs are missing
#    from the top-k, inject them (training only!) — otherwise the reader cannot learn.
# 3. Encode each of the k paragraphs with T5 tokenizer; stack to shape [k, L].
# 4. Target string: "answer // title1, title2" (so FiD learns to cite).
# 5. Train with Seq2SeqTrainer, predict_with_generate=True, num_beams=4.
# 6. At eval: retrieve top-k, generate, split answer and titles; compare to gold
#    using hotpot_metrics with sup_pred = set of titles, sup_gold = set of gold titles.
```

The injection of gold paragraphs at train time is important and easy to get wrong at eval time. **Never inject gold at eval** — that is a leak and inflates both answer-F1 and sup-F1. At train time injection is a noise-robustness trick: the reader learns to find the answer even when the retriever is imperfect.

## 3. Validation steps

1. **Global mask construction test.** Build a batch of 2 examples with questions of length 8 and 12 tokens. Assert that `global_attention_mask[:, :question_len]` is all 1, and that it's 1 at every `sent_token_position`. If the test accidentally sets the global mask equal to the attention mask, Longformer silently degrades to full attention and OOMs on 4 096-token inputs.
2. **Position-embedding bound.** Try to encode a 5 000-token example with `max_length=5000`. Must raise `IndexError`; must not silently wrap. If it does not raise, you forgot to `truncation=True` and are feeding positions past 4 095.
3. **Metric worked case.** From the HotpotQA paper's example: `answer_pred="Scott Derrickson"`, `answer_gold="Scott Derrickson"`, `sup_pred={("Scott Derrickson",0)}`, `sup_gold={("Scott Derrickson",0),("Ed Wood",0)}`. Hand-compute: `ans_em=1, ans_f1=1, sup_em=0, sup_f1 ≈ 2*(0.5)*(1.0)/(0.5+1.0)*?` → precision 1.0, recall 0.5, F1 ≈ 0.67; joint_f1 = 1 * 0.67 = 0.67. Unit-test matches.
4. **Retrieval ceiling.** Before training the FiD reader, measure what fraction of dev questions have *both* gold paragraphs in the top-k retrieved. This is the recall ceiling for the reader; if it is 70 % at k=10, your final sup-F1 cannot exceed ~70 % no matter how good the reader is. A common mistake is to tune reader hyperparameters while the retriever is the bottleneck.
5. **Single-hop baseline.** Train the same reader on *one* randomly chosen passage per example. This should score roughly the same `ans_f1` as the multi-passage version on single-hop-solvable questions (`type=comparison` or questions whose answer appears in one passage). If it scores *higher*, your multi-passage pipeline has a bug — likely a padding issue in `encode_fid`.
6. **Supporting-fact head sanity.** On a batch of 32 dev examples, compute precision/recall at threshold 0.5 on `sent_logits`. Sup-fact recall > 0.95 with precision < 0.2 means the head is predicting "everything is supporting" — a sign that `BCE` loss has not been weighted enough (increase 0.5 → 1.0) or that the sentence-start tokens were not given global attention.
7. **Leak probe.** Grep for the gold answer string inside `question`. A 1–3 % leak rate is normal (the question paraphrases a word from the answer); a double-digit leak rate indicates the dataset was built with a shortcut and that an unreasonably simple baseline will score well.
8. **Seed stability.** HotpotQA joint-F1 variance for Longformer baselines is typically 0.5–1.0 points across seeds; FiD is slightly more variable. Report mean ± std over ≥3 seeds.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Architecture fit for length** | Longformer / BigBird for >512-token single-pass; FiD or RAG for >4 096 and multi-doc; justified in writeup | One architecture chosen without justification | Default BERT truncated to 512 |
| **Global attention pattern (Longformer)** | Global on question tokens + sentence-starts; distinct from `attention_mask`; unit-tested | Global on question only | Global = attention mask (no sparse benefit) |
| **Supporting-fact head** | Dedicated sentence-level head with `BCEWithLogitsLoss`, weighted into the total loss | Head present but loss unweighted → head does not converge | No sup-fact prediction; sup-F1 reported as 0 or omitted |
| **Retrieval evaluation** | Top-k gold-recall measured independently; reader never sees gold-injected passages at eval | Gold-recall measured but reader eats gold at eval | No retrieval metric; reader fed gold passages directly |
| **Joint metric** | `ans_em / ans_f1`, `sup_em / sup_f1`, `joint_em / joint_f1`; product aggregation for joint-F1 | Answer metrics only + a separate sup-F1 line | Only answer F1 reported |
| **Train-eval retrieval parity** | Train on retrieved (+ optional gold injection); eval on retrieved only | Same retriever both sides but injects gold at eval | Trains on gold, evaluates on gold (leaked) |
| **Threat model** | Context-injection risk discussed per OWASP ML Top 10; retrieval-poisoning (ATLAS) covered when applicable | Mentioned | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure notes on dataset composition (HotpotQA shortcut-biases per Chen & Durrett 2019) and evaluation protocol (joint metric) | One of those | Neither |

## 5. Common mistakes

- **Global attention set to the whole attention mask.** Longformer then runs full attention and OOMs on 4 096 tokens; the speedup is gone. Always build `global_attention_mask` separately from `attention_mask`, starting from zeros and setting 1 only where needed.
- **Training on gold passages, serving on retrieved passages.** The reader learns to assume the gold is present and cannot handle noisy retrieval. If you must train with gold, inject it into the top-k and make the model learn to ignore the distractors.
- **Evaluating with gold passages present.** If `eval_passages = topk_retrieved + gold_paragraphs`, the reader has a 100 % recall ceiling and reported metrics are not reproducible against real retrieval. Keep the eval set clean.
- **Reporting answer-F1 only on HotpotQA.** A single-hop reader that cheats via shortcuts (Chen & Durrett 2019) can score ans-F1 65 while sup-F1 is 20 and joint-F1 is ~13. The number everyone compares against is joint-F1; reporting only ans-F1 looks like evasion.
- **Position-embedding wrap.** Loading a 512-position BERT and feeding 1 024 tokens may silently wrap the position IDs on some transformer forks, producing garbage. Always assert `len(input_ids) <= model.config.max_position_embeddings` at encode time.
- **`BCE` on sup-facts without class weighting.** On HotpotQA, sup-facts are ~5 % of sentences. Unweighted BCE lets the model predict "no" everywhere; precision is 0 and recall is 0. Add a `pos_weight = n_neg / n_pos` on the BCE, or oversample positives.
- **FiD without independent passage encoding.** If you concatenate all k passages into one long encoder input instead of calling the encoder k times, you lose FiD's linear-in-k scaling and OOM on anything >k=4. The reference FiD codebase does the loop explicitly for this reason.
- **BM25 index built after the questions arrive.** Rebuilding the index per query is 10–100× slower than building once over the corpus and querying. Cache the index on disk; invalidate when the corpus changes.
- **Not budgeting for retriever error.** If retriever-recall@10 is 70 %, your reader cannot exceed 70 % on sup-F1. Teams routinely burn reader-side effort on a 10 % reader fix when a 40 % retriever fix is available.
- **Ignoring HotpotQA's known biases.** Chen & Durrett (2019) showed many HotpotQA questions are answerable from one paragraph via shortcut patterns. Report joint-F1 split by `level` (easy/medium/hard) and `type` (bridge/comparison); hard+bridge is the honest benchmark.

## 6. References

Primary sources used in this solution:

- Beltagy, I., Peters, M. E., Cohan, A. (2020). *Longformer: The Long-Document Transformer*. arXiv. <https://arxiv.org/abs/2004.05150> — sparse-attention architecture used in Path A; §3.2 defines the local + global attention pattern.
- Zaheer, M. et al. (2020). *Big Bird: Transformers for Longer Sequences*. NeurIPS. <https://arxiv.org/abs/2007.14062> — alternative sparse-attention architecture for long contexts; interchangeable with Longformer in Path A.
- Izacard, G., Grave, E. (2021). *Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering*. EACL. <https://aclanthology.org/2021.eacl-main.74/> — Fusion-in-Decoder architecture used in Path B.
- Yang, Z. et al. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. EMNLP. <https://aclanthology.org/D18-1259/> — reference dataset; §5 defines the joint-EM / joint-F1 metric computed in §2.5.
- Ho, X., Nguyen, A.-K. D., Sugawara, S., Aizawa, A. (2020). *Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps* (2WikiMultiHopQA). COLING. <https://aclanthology.org/2020.coling-main.580/> — complementary benchmark with annotated reasoning chains.
- Kwiatkowski, T. et al. (2019). *Natural Questions: A Benchmark for Question Answering Research*. TACL. <https://aclanthology.org/Q19-1026/> — long-context open-domain benchmark referenced as an alternative evaluation target.
- Chen, J., Durrett, G. (2019). *Understanding Dataset Design Choices for Multi-hop Reasoning*. NAACL. <https://aclanthology.org/N19-1405/> — documents HotpotQA shortcut biases; justifies reporting joint-F1 and splitting by level/type (§5).
- Robertson, S., Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in IR. <https://doi.org/10.1561/1500000019> — reference definition of BM25, used in `retriever.py`.
- Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering* (DPR). EMNLP. <https://aclanthology.org/2020.emnlp-main.550/> — dense retriever alternative to BM25; cited as the optional upgrade path in §2.4.
- Hugging Face `transformers` docs — <https://huggingface.co/docs/transformers/model_doc/longformer> — authoritative reference for the Longformer tokenizer and `global_attention_mask` mechanics.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation* applies to every passage passed to the reader: an attacker who can insert a paragraph into the retrieval corpus (wiki vandalism, SEO injection, retrieval-store write access) can plant content that steers the reader. Mitigations: provenance filter at retrieval ingest; attribute output passages and surface them to the user. *ML02 Data Poisoning* is the training-time variant: a corpus scraper that ingests arbitrary web pages is a poisoning vector.
- MITRE ATLAS — <https://atlas.mitre.org/> — catalogue of adversarial-ML techniques including retrieval-corpus poisoning and prompt-injection via retrieved passages. Specifically relevant when the multi-hop reader's output drives automated decisions.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map* function: document the dataset's known shortcut biases (Chen & Durrett 2019) and the retriever's demographic / topical coverage; *Measure* function: report joint-F1 and split by difficulty level; *Manage* function: define a release gate on joint-F1 regression, not on answer-F1 alone, so a "better answer, worse citations" change is caught.
