# exercise-02 — Nested and Document-Level NER

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-104-sequence-labelling-and-information-extraction/exercises/exercise-02-nested-and-document-level-ner.md`.
> This solution targets the learning goals that filename implies: move beyond the **flat, sentence-scoped IOB** recipe from exercise-01 to the two settings production corpora routinely contain — **nested entities** (where "University of Washington" is an `ORG` that contains a `LOC` "Washington") and **document-level NER** (where the signal that fixes a mention's label lives several paragraphs away). The output is a **span-classification** model that natively represents nesting plus a **windowed/document-level** inference recipe.

## 1. Solution overview

Flat IOB tagging has two hard structural limits:

| Limit | Why it fails | Consequence |
|---|---|---|
| **One label per token** | Nested entities overlap on the same token span; IOB cannot encode both labels on token `t` | ~10–30 % of entities in biomedical (GENIA) and legal corpora are nested and simply *vanish* |
| **Sentence-scoped context** | 512-token BERT models process sentences in isolation; coreference and document-level disambiguation cues are discarded | "The Court" is mis-labelled `ORG` on page 7 because "the Supreme Court" was defined on page 1 |

Two families of fixes used in this solution:

- **Span-based classification** (Sohrab & Miwa 2018; Zhong & Chen 2021; Yu et al. 2020). Enumerate candidate spans `(i, j)` with `0 ≤ i ≤ j < n` and `j - i < L`; classify each as `NONE` or one of the entity types. Because spans are classified independently, two overlapping spans can both carry labels — nesting is a first-class citizen.
- **Document windowing with context flow** (Beltagy et al. 2020 for Longformer; the "Document-level NER" line in Luo et al. 2020; Wang & Xia 2022). Slide a fixed window across the document with stride, let each token appear in multiple windows, aggregate logits (mean or logit-averaging), and keep the span with the highest confidence when two windows disagree. Coreference-aware variants add a second pass that re-labels mentions using cluster-level evidence.

The deliverable has two parts — a **span-pair biaffine head** on top of a transformer encoder for nested NER, and a **windowed inference harness** that applies that model to multi-page documents and resolves cross-window conflicts.

Reference corpora: **GENIA** (biomedical nested NER, Kim et al. 2003) for the nested case; **CoNLL-2003 documents** (full documents, not sentences) as a document-level smoke test; **ACE 2005** for both at once. Metric: entity-level span-F1 via `seqeval` for the flat sentences, **nested span-F1** (exact `(start, end, type)` match) for the nested dev set.

## 2. Worked implementation

### 2.1 Project layout

```
nested_ner/
  spans.py              # candidate span enumeration with length cap
  model.py              # encoder + biaffine span classifier
  data.py               # (tokens, spans, labels) dataset conversion
  windowing.py          # document → overlapping windows + aggregation
  decode.py             # nested-aware greedy / ILP decoding
  metrics.py            # nested span-F1
  run.py
  tests/
    test_spans.py
    test_windowing.py
    test_decode.py
  requirements.txt
```

### 2.2 Candidate span enumeration

```python
# spans.py
from itertools import product


def enumerate_spans(seq_len: int, max_width: int) -> list[tuple[int, int]]:
    """All (start, end_inclusive) with end - start < max_width."""
    return [(i, j) for i, j in product(range(seq_len), repeat=2)
            if i <= j and j - i < max_width]
```

`max_width` matters: biomedical entities in GENIA cap out around ~8 tokens, legal entities in contracts around ~20. Enumerating spans up to 30 on a 512-token document gives ~15 000 candidates per document — tractable; raising to 60 quadruples that and spends most of the compute on spans that will always be `NONE`. Set `max_width` from the 99th percentile of gold span lengths in the training data, not folklore.

### 2.3 Span representation: biaffine head

Following Yu, Bohnet, & Poesio (2020) *Named Entity Recognition as Dependency Parsing*, we score a span `(i, j)` by applying two FFNNs — one to produce a "start" vector from the start token's encoder output, one for the end token — and combining them with a biaffine form:

```python
# model.py
import torch
import torch.nn as nn
from transformers import AutoModel


class BiaffineSpanClassifier(nn.Module):
    def __init__(self, encoder_id: str, num_labels: int, hidden: int = 150):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_id)
        H = self.encoder.config.hidden_size
        self.ffn_start = nn.Sequential(nn.Linear(H, hidden), nn.GELU(), nn.Dropout(0.2))
        self.ffn_end   = nn.Sequential(nn.Linear(H, hidden), nn.GELU(), nn.Dropout(0.2))
        # Biaffine: start^T U end + V [start; end] + b, per class.
        self.U = nn.Parameter(torch.empty(num_labels, hidden, hidden))
        self.V = nn.Parameter(torch.empty(num_labels, 2 * hidden))
        self.b = nn.Parameter(torch.zeros(num_labels))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, span_starts, span_ends):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        s = self.ffn_start(h.gather(1, span_starts.unsqueeze(-1).expand(-1, -1, h.size(-1))))
        e = self.ffn_end  (h.gather(1, span_ends  .unsqueeze(-1).expand(-1, -1, h.size(-1))))
        # scores: (B, S, num_labels)
        bilinear = torch.einsum("bsh,chg,bsg->bsc", s, self.U, e)
        linear   = torch.einsum("bsh,ch->bsc",      torch.cat([s, e], dim=-1), self.V)
        return bilinear + linear + self.b
```

Why biaffine: the bilinear term `start^T U end` lets the model learn start/end *interaction* rather than treating them as independent features; the 2020 paper reports substantial gains over pure concat-MLP heads on both flat (CoNLL-2003) and nested (ACE2004/2005, GENIA) corpora.

### 2.4 Dataset conversion

Each training example is a tuple `(input_ids, span_starts, span_ends, span_labels)`. For every enumerated span we assign the gold label if it matches an annotated entity exactly, else `NONE`. Imbalance is severe (99 %+ `NONE`); negative sub-sampling or a focal loss both help. Default here is to **downsample negatives** to 10× the positive count per batch.

```python
# data.py (sketch)
NONE_ID = 0  # reserve label 0 for NONE

def build_example(tokens, gold_entities, label_to_id, max_width):
    # gold_entities: list of (start, end_inclusive, type_name)
    gold = {(s, e): label_to_id[t] for s, e, t in gold_entities}
    cands = enumerate_spans(len(tokens), max_width)
    starts = [s for s, _ in cands]
    ends   = [e for _, e in cands]
    labels = [gold.get((s, e), NONE_ID) for s, e in cands]
    return starts, ends, labels
```

### 2.5 Document-level windowing with stride

```python
# windowing.py
from dataclasses import dataclass


@dataclass
class Window:
    start: int   # absolute token index
    end:   int   # exclusive
    span_logits: dict[tuple[int, int], list[float]] = None  # (absolute, absolute)


def window_document(doc_tokens, window_size: int = 384, stride: int = 128):
    windows = []
    i = 0
    while i < len(doc_tokens):
        j = min(i + window_size, len(doc_tokens))
        windows.append(Window(start=i, end=j))
        if j == len(doc_tokens):
            break
        i += stride
    return windows
```

Each absolute token index typically appears in 2–3 windows (`window_size / stride`). After inference, aggregate span logits per absolute `(start, end)`:

```python
def aggregate(windows, agg: str = "mean"):
    bucket = {}
    for w in windows:
        for (s, e), logits in w.span_logits.items():
            bucket.setdefault((s, e), []).append(logits)
    out = {}
    for key, lst in bucket.items():
        logits = torch.tensor(lst)
        out[key] = logits.mean(0) if agg == "mean" else logits.max(0).values
    return out
```

`mean` is the robust default; `max` amplifies a single high-confidence window and is worse when one window is corrupted by a truncated entity. Beltagy et al. 2020 report that **logit-averaging beats hard-voting by ~1 F1** on document-scale WikiNER; our harness defaults to mean-logits.

### 2.6 Decoding: nested-aware

A flat decoder picks one label per token; a nested decoder keeps every span with `argmax != NONE`. Two decoding regimes are included:

```python
# decode.py
def decode_nested(span_scores, threshold: float = 0.0, keep_nested: bool = True):
    """span_scores: dict[(start, end)] -> tensor(num_labels)."""
    kept = []
    for (s, e), scores in span_scores.items():
        label = int(scores.argmax())
        if label == 0:  # NONE
            continue
        margin = float(scores[label] - scores[0])
        if margin < threshold:
            continue
        kept.append((s, e, label, margin))
    if keep_nested:
        return kept
    # Drop lower-confidence spans that overlap higher-confidence ones.
    kept.sort(key=lambda x: -x[3])
    accepted, occupied = [], set()
    for s, e, l, m in kept:
        if any(t in occupied for t in range(s, e + 1)):
            continue
        accepted.append((s, e, l, m))
        for t in range(s, e + 1):
            occupied.add(t)
    return accepted
```

- `keep_nested=True` is the correct setting for nested-NER evaluation (GENIA, ACE).
- `keep_nested=False` drops overlaps greedily by confidence — this is the compatibility path when a downstream consumer (search index, redaction, UI highlight) cannot render nested spans.

### 2.7 Metrics

Nested evaluation is exact match on the triple `(start, end, type)`:

```python
# metrics.py
def nested_span_f1(gold, pred):
    gold_set = {(s, e, t) for s, e, t in gold}
    pred_set = {(s, e, t) for s, e, t in pred}
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}
```

`seqeval` does **not** support nested evaluation — it decodes the IOB stream into spans and assumes non-overlap. Trying to force nested entities through `seqeval` by "flattening" gives a number that is systematically wrong. For the flat sentence-level sanity run, keep `seqeval`; for the nested headline, use the function above.

## 3. Validation steps

1. **Span enumeration count.** For `seq_len=50, max_width=12`, the number of candidates must be `sum_{w=1..12}(50 - w + 1) = 522`. Unit-test this against the formula.
2. **Nested round-trip.** On one GENIA sentence with a known nested pair (e.g. `IL-2 gene` inside `human IL-2 gene expression`), run the trained model and assert both spans are returned with `keep_nested=True`, and only the outer span survives with `keep_nested=False`.
3. **Window coverage.** For any document length `L` and `(window_size, stride)`, assert every token `t ∈ [0, L)` is in at least one window — common bug in stride loops is the final partial window being skipped.
4. **Aggregation equivalence.** For a document shorter than `window_size`, the windowed path and the single-pass path must produce bitwise-identical logits (one window, no aggregation needed).
5. **Negative sampling honesty.** Train with 10× negative subsampling and evaluate with **no** subsampling (all spans scored). A pipeline that also subsamples at eval time inflates precision by ignoring hard negatives.
6. **Flat compatibility.** On CoNLL-2003 sentences the nested decoder must match (within ~0.3 F1) the flat IOB model from exercise-01. If the nested model is 2+ points lower, the span cap or negative ratio is wrong.
7. **GENIA headline.** Published numbers on GENIA nested-F1 with biaffine + BERT-base land in the 78–81 range (Yu et al. 2020). A result below 70 points to a scheme or evaluation bug.
8. **Seed variance.** Run ≥3 seeds and report mean ± std. Span-classifier variance is higher than flat IOB because negative sampling is stochastic; typical std is 0.4–0.8 F1 on GENIA.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Nested representation** | Spans classified independently; overlapping outputs preserved under `keep_nested=True` | Flat IOB with post-hoc span merging | Flat IOB only; nesting silently lost |
| **Span cap** | `max_width` chosen from the 99th-percentile of gold spans; documented | Fixed `max_width` without data check | No cap; O(n²) blows up on long documents |
| **Negative sampling** | Subsample negatives at train time only; eval sees all spans | Subsampled at both stages | No subsampling, or subsampling at eval |
| **Windowing** | Overlap + logit-mean aggregation; full coverage asserted | Overlap but takes last window on conflict | Non-overlapping chunks; tokens near boundaries silently mis-classified |
| **Nested metric** | Exact-triple F1 on `(start, end, type)`; no `seqeval` on nested data | `seqeval` on nested data (invalid but reported) | Token accuracy only |
| **Decoder flexibility** | Both `keep_nested` and greedy-overlap-free modes available and tested | One mode only | Neither documented |
| **Reproducibility** | Seed, deps, span cap, negative ratio all echoed in output | Partial | None |
| **Threat model** | Adversarial-span injection / long-document DoS noted per OWASP ML Top 10 and MITRE ATLAS | Mentioned without concrete mitigation | Ignored |

## 5. Common mistakes

- **Trying to force nesting into IOB.** BIOES/BILOU still can only encode one label per token. If your dataset has nested entities, flat sequence labelling will always ceiling out well below the span-classification result.
- **Scoring nested data with `seqeval`.** `seqeval` flattens the stream into non-overlapping spans and silently drops nesting. The reported "0.85 F1" is on the flattened view, not the true gold.
- **No negative subsampling.** Spans are dominated 99 %+ by `NONE`. Without subsampling the model learns a strong `NONE` prior and misses rare types; with uniform subsampling the loss landscape is tractable and macro-F1 on rare types improves 5–10 points.
- **Picking `max_width` by default.** `max_width=30` on GENIA is overkill (99th percentile is ~8); on legal contracts 30 cuts off many long entity names. Pick from the data.
- **Non-overlapping windows.** A chunk at `[0, 512)` and the next at `[512, 1024)` leaves any entity straddling token 511/512 unlabelable. Stride `≤ window_size / 2` and aggregate.
- **Aggregating with `argmax` per window.** Picking the per-window `argmax` first and voting loses the magnitude information — two low-confidence predictions and one high-confidence one get equal weight. Aggregate *logits* (mean), then decode once.
- **Forgetting the span-cap in inference.** Training with `max_width=12` and inferring with no cap enumerates far more spans than the model has seen in support, and the tail spans fire spuriously. Pin the cap at inference too.
- **Assuming the encoder sees the whole document.** `distilbert-base-uncased` has a 512-token context. Beyond that your windowing is load-bearing; without it, you silently truncate. Use Longformer / BigBird (Beltagy et al. 2020; Zaheer et al. 2020) or make the windowing explicit.
- **Dropping document identity in the loader.** Document-level inference requires all tokens of a document to be processed in the same pass (or at least batched contiguously). A shuffle that interleaves sentences from different documents breaks cross-window aggregation.

## 6. References

Primary sources used in this solution:

- Yu, J., Bohnet, B., Poesio, M. (2020). *Named Entity Recognition as Dependency Parsing*. ACL. <https://aclanthology.org/2020.acl-main.577/> — the biaffine-span head used in §2.3; state-of-the-art for flat and nested NER at publication.
- Sohrab, M. G., Miwa, M. (2018). *Deep Exhaustive Model for Nested Named Entity Recognition*. EMNLP. <https://aclanthology.org/D18-1309/> — exhaustive span enumeration with a span-cap, the recipe underlying §2.2.
- Kim, J.-D., Ohta, T., Tateisi, Y., Tsujii, J. (2003). *GENIA corpus — a semantically annotated corpus for bio-textmining*. Bioinformatics 19. <https://doi.org/10.1093/bioinformatics/btg1023> — the nested biomedical corpus used as reference.
- Beltagy, I., Peters, M. E., Cohan, A. (2020). *Longformer: The Long-Document Transformer*. <https://arxiv.org/abs/2004.05150> — long-context attention; alternative to windowing with aggregation.
- Zaheer, M. et al. (2020). *Big Bird: Transformers for Longer Sequences*. NeurIPS. <https://arxiv.org/abs/2007.14062> — sparse-attention long-context model.
- Luo, Y. et al. (2020). *Hierarchical Contextualized Representation for Named Entity Recognition*. AAAI. <https://arxiv.org/abs/1911.02257> — document-level context flow for NER.
- Zhong, Z., Chen, D. (2021). *A Frustratingly Easy Approach for Entity and Relation Extraction*. NAACL. <https://aclanthology.org/2021.naacl-main.5/> — span-pair classification; relevant for exercise-03.
- Jurafsky, D., Martin, J. (2024). *Speech and Language Processing*, 3rd ed. draft, ch. 8 and ch. 17 (coreference). <https://web.stanford.edu/~jurafsky/slp3/> — text treatment of nested vs flat NER and document context.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation Attack*: adversarial text can force spurious nested spans; long-document inputs are also a *ML05 Model Denial of Service* vector against span-enumeration systems that are quadratic in token count.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for crafted-input attacks and model-DoS; applicable when NER output gates redaction or compliance decisions.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Measure* function: report nested-F1 with the exact span-triple protocol, not a flattened proxy; *Map* function: document corpus licence and PII / clinical-data risk for GENIA and clinical derivatives.
