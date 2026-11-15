# exercise-03 — Relation and Event Extraction

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-104-sequence-labelling-and-information-extraction/exercises/exercise-03-relation-and-event-extraction.md`.
> This solution targets the learning goals that filename implies: go beyond NER (exercises 1–2) to the structures that depend on it — **binary relations between entity pairs** (e.g. `works_for(Alice, AcmeCorp)`) and **events with multiple arguments** (e.g. `Acquisition(buyer=AcmeCorp, target=FooInc, date=2025-03-14)`). Build both a **pipelined** extractor (NER → RE) and a **joint span-pair** extractor, then show where each one wins and where each one breaks.

## 1. Solution overview

Relation and event extraction share a core question: **once you have entity spans, how do you decide which spans combine into which structured fact?** Two dominant architectures:

| Architecture | What it is | Strength | Weakness |
|---|---|---|---|
| **Pipelined (NER → RE)** | Run NER first, then for every entity pair in the same sentence, classify the relation type | Simple, reusable NER, interpretable error attribution | NER errors cascade; same encoder forward pass is run many times |
| **Joint span-pair** (Zhong & Chen 2021, "PURE"; Luan et al. 2019, DyGIE) | Enumerate candidate spans, classify entity types; classify relation on each span pair; share encoder | Higher F1 on benchmarks; single forward pass per document | Harder to debug; coupling between heads means an NER regression also degrades RE |

Event extraction adds a third structural element — the **trigger** (the verb/noun that evokes the event) — and treats arguments as typed roles rather than unlabelled relations. The ACE 2005 formulation (Doddington et al. 2004) is the industry standard: a trigger, an event type, and a set of `(entity, role)` arguments.

The deliverable has two parts:

1. **Pipelined relation extractor** on TACRED (Zhang et al. 2017) with **entity-marker typed tokens** (Soares et al. 2019) — the "PURE" trick that encodes entity spans and types directly into the input stream so the encoder attends to them.
2. **Event extraction harness** on ACE-2005-style schema with a trigger-detection head and an argument-classification head that pairs each trigger with each entity in the sentence.

Both use the same `AutoModel` encoder; the two heads differ only in which pair of spans they score.

## 2. Worked implementation

### 2.1 Project layout

```
re_ee/
  data.py                 # TACRED + ACE schema loaders
  markers.py              # entity-marker insertion
  model_re.py             # pipelined relation classifier
  model_ee.py             # trigger + argument heads
  decode.py               # decode role assignments
  metrics.py              # macro-F1 (RE); trigger-F1 + argument-F1 (EE)
  run_re.py / run_ee.py
  tests/
    test_markers.py
    test_decode.py
  requirements.txt
```

### 2.2 Entity-marker typed tokens

For every training pair `(e1, e2, relation)` in the same sentence, we insert special markers around each entity that encode its **position and type**:

```
Original : "Alice joined AcmeCorp in 2019."
Marked   : "[E1:PER] Alice [/E1:PER] joined [E2:ORG] AcmeCorp [/E2:ORG] in 2019."
```

```python
# markers.py
E1_START = "[E1:{type}]"
E1_END   = "[/E1:{type}]"
E2_START = "[E2:{type}]"
E2_END   = "[/E2:{type}]"


def add_markers(tokens, e1_span, e1_type, e2_span, e2_type):
    """Return a new list of tokens with marker tokens around e1 and e2.

    Spans are (start, end_inclusive) in the original `tokens` list.
    Caller must resize the model's token embedding matrix to include the new
    marker vocabulary; otherwise the markers map to <unk> and the gain disappears.
    """
    s1, e1 = e1_span
    s2, e2 = e2_span
    # Normalise so that e1 comes before e2 (relations are typed, order matters).
    swap = False
    if s1 > s2:
        s1, e1, s2, e2 = s2, e2, s1, e1
        e1_type, e2_type = e2_type, e1_type
        swap = True
    out, i = [], 0
    for cursor in sorted([(s1, "s1"), (e1, "e1"), (s2, "s2"), (e2, "e2")], key=lambda x: x[0]):
        while i <= cursor[0]:
            out.append(tokens[i]); i += 1
        if cursor[1] == "s1": out.append(E1_START.format(type=e1_type))
        if cursor[1] == "e1": out.append(E1_END.format(type=e1_type))
        if cursor[1] == "s2": out.append(E2_START.format(type=e2_type))
        if cursor[1] == "e2": out.append(E2_END.format(type=e2_type))
    out.extend(tokens[i:])
    return out, swap
```

Why marker tokens: Soares et al. 2019 show that marking entities in the input and reading the marker representations at the output outperforms every other pair-representation they tested (concat, pooling, attention) on TACRED and SemEval-2010-Task-8. Zhong & Chen 2021 replicate the gain and attribute it to the model learning *position-aware* and *type-aware* pair representations without a dedicated architecture.

### 2.3 Relation classifier head

```python
# model_re.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class RelationClassifier(nn.Module):
    def __init__(self, encoder_id: str, num_relations: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_id)
        H = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(2 * H, H), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(H, num_relations),
        )

    def forward(self, input_ids, attention_mask, e1_marker_idx, e2_marker_idx):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Gather the [E1:type] start-marker and [E2:type] start-marker hidden states.
        e1 = h.gather(1, e1_marker_idx.view(-1, 1, 1).expand(-1, 1, h.size(-1))).squeeze(1)
        e2 = h.gather(1, e2_marker_idx.view(-1, 1, 1).expand(-1, 1, h.size(-1))).squeeze(1)
        return self.classifier(torch.cat([e1, e2], dim=-1))
```

Note: after adding marker tokens to the tokeniser, `model.resize_token_embeddings(len(tokenizer))` is mandatory. Skipping that step silently maps markers to `<unk>`, which gives a result roughly equal to the no-marker baseline — a quiet way to lose 2–3 F1 and not notice.

### 2.4 Negative sampling for `no_relation`

TACRED labels ~80 % of entity pairs as `no_relation`. Without explicit handling, the model learns a strong "nothing" prior and underfits the positive classes. Two tactics applied together:

- **Downsample negatives** to 3× the positive count during training (not eval).
- **Micro-F1 that excludes the `no_relation` class** as the headline metric. This is the TACRED convention (Zhang et al. 2017) — otherwise precision/recall are dominated by the majority class.

```python
# metrics.py
from sklearn.metrics import precision_recall_fscore_support

NO_REL = "no_relation"

def tacred_f1(y_true, y_pred, label_names):
    idxs = [i for i, l in enumerate(label_names) if l != NO_REL]
    # micro over positive classes only
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=idxs, average="micro", zero_division=0.0,
    )
    return {"precision": p, "recall": r, "f1": f}
```

### 2.5 Event extraction: triggers and arguments

An event example is `(sentence, trigger_span, event_type, [(entity_span, role), ...])`. Two heads on the shared encoder:

- **Trigger head** — a token-classification head over event types (plus `NONE`), run on every token (restrict to candidate heads via dependency parse if available, else all tokens).
- **Argument head** — for every detected trigger and every candidate entity span in the sentence, classify the role or `NONE`.

```python
# model_ee.py
class EventExtractor(nn.Module):
    def __init__(self, encoder_id: str, num_event_types: int, num_roles: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_id)
        H = self.encoder.config.hidden_size
        self.trigger_head = nn.Linear(H, num_event_types)
        self.argument_head = nn.Sequential(
            nn.Linear(3 * H, H), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(H, num_roles),
        )

    def forward(self, input_ids, attention_mask, pair_indices=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        trig_logits = self.trigger_head(h)  # (B, T, E)
        if pair_indices is None:
            return trig_logits, None
        # pair_indices: (B, K, 3) — trigger token, entity start, entity end
        B, K, _ = pair_indices.shape
        gather = lambda idx: h.gather(1, idx.view(B, K, 1).expand(-1, -1, h.size(-1)))
        t  = gather(pair_indices[:, :, 0])
        es = gather(pair_indices[:, :, 1])
        ee = gather(pair_indices[:, :, 2])
        arg_logits = self.argument_head(torch.cat([t, es, ee], dim=-1))
        return trig_logits, arg_logits
```

**Role constraints.** ACE event types have a fixed set of valid roles (e.g. a `Life:Die` event has `Victim`, `Agent`, `Place`, `Time` — but not `Buyer`). A clean decoder applies a per-event-type **role mask** that zeroes out invalid roles before `argmax`; this prevents a confident model from emitting structurally impossible events.

### 2.6 Metrics

Two ACE-standard numbers for event extraction:

- **Trigger-F1** — a trigger prediction is correct if the token span and event type both match gold.
- **Argument-F1** — an argument prediction is correct if the gold *trigger* was also predicted *and* the argument span and role both match.

```python
def trigger_f1(gold_trigs, pred_trigs):
    g = {(sp, et) for sp, et in gold_trigs}
    p = {(sp, et) for sp, et in pred_trigs}
    return _pr_f(g, p)

def argument_f1(gold_args, pred_args, pred_trigs, gold_trigs):
    # Only count arguments whose trigger was correctly predicted.
    correct_trigs = {(sp, et) for (sp, et) in gold_trigs} & {(sp, et) for (sp, et) in pred_trigs}
    g = {(trig, sp, role) for (trig, sp, role) in gold_args if trig in correct_trigs}
    p = {(trig, sp, role) for (trig, sp, role) in pred_args if trig in correct_trigs}
    return _pr_f(g, p)

def _pr_f(g, p):
    tp = len(g & p); fp = len(p - g); fn = len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
```

This definition of argument-F1 is *trigger-conditioned* — the dominant ACE convention. An alternative (*unconstrained argument-F1*) counts arguments against any predicted trigger; it is more lenient and will show a different number. Report which one you use.

## 3. Validation steps

1. **Marker unit test.** Hand-build a short sentence with known `(e1_span, e1_type, e2_span, e2_type)` and assert the marker positions in the output exactly match the input spans. Test the `s1 > s2` swap branch.
2. **Vocabulary resize.** After `tokenizer.add_special_tokens({'additional_special_tokens': [E1_START, ...]})`, call `model.resize_token_embeddings(len(tokenizer))` and assert `model.get_input_embeddings().weight.shape[0] == len(tokenizer)`. A regression here is silent and costly.
3. **TACRED headline range.** DistilBERT + markers + 3× negative subsampling should land at ~68–71 micro-F1 on TACRED positive classes (Zhang et al. 2017 report ~66 for a plain BERT baseline; marker gains take it into the low 70s). Below 60 points to markers mapping to `<unk>` or negatives not being subsampled.
4. **Role-mask invariance.** Apply the role mask after argument prediction and assert every predicted role `r` on a trigger of event type `e` satisfies `r ∈ valid_roles(e)` — else the decoder is broken.
5. **Pipeline failure cascade.** Deliberately corrupt NER on 10 % of test pairs (drop one of the two entities). Measure the delta in RE F1 — it should drop roughly proportionally. This is the number you quote when a stakeholder asks "do I need better NER or better RE".
6. **No-relation leakage.** Train with 3× negative subsampling; eval on **all** pairs. If you also subsample at eval, precision inflates by excluding hard negatives — common pitfall.
7. **Seed stability.** Report mean ± std over ≥3 seeds. Marker-based RE on TACRED typically has σ ≈ 0.5 F1.
8. **Deps + dataset hash.** Record TACRED revision, `transformers` version, marker-vocabulary hash in the output JSON.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Pair representation** | Typed entity markers inserted + special tokens registered; embedding matrix resized | Markers inserted without resize (silent regression) | No positional/type information in pair representation |
| **Negative handling** | Subsample `no_relation` at train only; eval on full set | Subsample at both stages | No subsampling; model collapses to `no_relation` |
| **Metric honesty** | Micro-F1 excluding `no_relation` (TACRED convention); macro also reported | Macro-F1 over all classes including `no_relation` | Accuracy only |
| **Event trigger-F1 / arg-F1** | Both reported with trigger-conditioned argument definition stated | Trigger-F1 only | Token accuracy on triggers |
| **Role mask** | Per-event-type role mask applied before argmax; invalid-role rate asserted zero | Role vocabulary flat; no mask | Invalid `(event, role)` pairs appear in output |
| **Cascade analysis** | NER failure-mode experiment reported as part of error budget | Mentioned without data | Pipeline evaluated as black box |
| **Reproducibility** | Seed, deps, dataset hash, marker vocab echoed | Seed only | None |
| **Threat model** | Adversarial pair-ordering / marker-injection noted per OWASP ML Top 10 and MITRE ATLAS | Mentioned without concrete mitigation | Ignored |

## 5. Common mistakes

- **Forgetting `resize_token_embeddings`.** The most common silent regression in marker-based RE. The model trains, metrics report, but marker tokens are indistinguishable from `<unk>` and the +3 F1 the markers promise does not appear.
- **Reporting macro-F1 across all classes including `no_relation`.** `no_relation` dominates; macro inflates toward the easy class. TACRED uses micro over positive classes — follow it.
- **Enumerating pairs across sentence boundaries.** TACRED and most RE benchmarks are sentence-scoped. Cross-sentence pairs (document-level RE, DocRED) need a different architecture; mixing scopes silently degrades both.
- **Treating RE as entirely independent of NER.** The headline F1 on RE depends on the span boundaries NER produces. Reporting RE on gold spans and deploying with predicted spans is a common bait-and-switch — always report both.
- **No role mask on event arguments.** Without it, the model can emit a `Buyer` role on a `Life:Die` event with no runtime error and a plausible-looking JSON output. Downstream consumers trust the schema and corrupt their state.
- **Mixing trigger-conditioned and unconstrained argument-F1.** The former is stricter by 3–5 points on ACE. Comparing a paper's unconstrained number to your conditioned number is a bake-off bug.
- **Dropping entity types from the marker.** Soares et al. 2019 and Zhong & Chen 2021 both find that **typed** markers (`[E1:PER]`) outperform untyped (`[E1]`) by ~1 F1 — the type gives the encoder a free prior on which relations are plausible. Many implementations drop the type for simplicity and lose the gain.
- **Training the trigger head without the "no event" class.** A trigger head that only distinguishes event types (and relies on some span-selection step for the "no-event" case) is brittle. Include `NONE` and predict on every token.
- **Negative pairs created only from entities in the same sentence.** Within-sentence pairs under-represent the real failure mode of RE at scale — spurious cross-sentence confusion. If your downstream consumer will see document-level inputs, add some cross-sentence negatives even if the benchmark does not.

## 6. References

Primary sources used in this solution:

- Zhang, Y., Zhong, V., Chen, D., Angeli, G., Manning, C. D. (2017). *Position-aware Attention and Supervised Data Improve Slot Filling*. EMNLP. <https://aclanthology.org/D17-1004/> — the TACRED dataset and the TACRED-F1 evaluation convention.
- Soares, L. B., FitzGerald, N., Ling, J., Kwiatkowski, T. (2019). *Matching the Blanks: Distributional Similarity for Relation Learning*. ACL. <https://aclanthology.org/P19-1279/> — typed entity-marker representation used in §2.2.
- Zhong, Z., Chen, D. (2021). *A Frustratingly Easy Approach for Entity and Relation Extraction*. NAACL. <https://aclanthology.org/2021.naacl-main.5/> — the "PURE" pipelined-vs-joint comparison; justifies the pipeline we build in §2.3.
- Luan, Y., Wadden, D., He, L., Shah, A., Ostendorf, M., Hajishirzi, H. (2019). *A general framework for information extraction using dynamic span graphs* (DyGIE). NAACL. <https://aclanthology.org/N19-1308/> — joint-span-pair architecture; included as the alternative worth knowing.
- Doddington, G. R. et al. (2004). *The Automatic Content Extraction (ACE) Program — Tasks, Data, and Evaluation*. LREC. <https://aclanthology.org/L04-1011/> — the ACE event schema (trigger + arguments + roles) used in §2.5.
- Walker, C., Strassel, S., Medero, J., Maeda, K. (2006). *ACE 2005 Multilingual Training Corpus*. LDC2006T06. <https://catalog.ldc.upenn.edu/LDC2006T06> — the ACE 2005 corpus that defines the trigger-F1 / argument-F1 evaluation.
- Jurafsky, D., Martin, J. (2024). *Speech and Language Processing*, 3rd ed. draft, ch. 18 (*Information Extraction*). <https://web.stanford.edu/~jurafsky/slp3/> — reference treatment of relation and event extraction.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation Attack*: crafted text can force spurious relations between entities (useful for information-operations and disinfo-style attacks on knowledge bases); *ML03 Model Inversion* becomes relevant when RE models memorise named-entity pairs from training data.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue for adversarial input and extraction attacks; relevant when RE output feeds a KB or decision engine.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Measure* function: report both gold-span and predicted-span F1 for RE; name the argument-F1 variant explicitly; *Map* function: document the training-data licence (TACRED is LDC-licensed; ACE is LDC-licensed with event annotation).
