# exercise-04 — Entity Linking and Coreference

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-104-sequence-labelling-and-information-extraction/exercises/exercise-04-entity-linking-and-coreference.md`.
> This solution targets the learning goals that filename implies: once you have detected entity spans (exercises 1–2), you still need to answer two questions — **which real-world thing is this mention?** (entity linking against a KB such as Wikipedia/Wikidata) and **which mentions in the document refer to the same thing?** (coreference resolution). Build a **two-stage EL** pipeline (candidate generation + disambiguation) and a **mention-pair / higher-order coreference** resolver, then show how they interact when deployed together.

## 1. Solution overview

Two problems, one underlying structure. Both ask "assign this mention to a cluster"; the cluster identity differs:

| Task | Cluster identity | Canonical dataset | Dominant metric |
|---|---|---|---|
| **Entity linking (EL)** | A KB entity id (e.g. Wikidata `Q76` = Barack Obama) | AIDA-CoNLL (Hoffart et al. 2011), KILT ( Petroni et al. 2021) | InKB-accuracy, micro-F1 |
| **Coreference (coref)** | A mention cluster within the document | OntoNotes 5.0 CoNLL-2012 shared task (Pradhan et al. 2012) | MUC, B³, CEAF-ϕ4; averaged as CoNLL-F1 |

### Entity linking architecture

A production EL system almost always has two stages (Mihalcea & Csomai 2007; Hoffart et al. 2011; Kolitsas et al. 2018; De Cao et al. 2021 for the GENRE generative variant; Ayoola et al. 2022 for ReFinED):

1. **Candidate generation** — for a mention string `m`, return a small set `C(m)` of KB ids that `m` plausibly refers to. Backed by (a) an **alias table** derived from Wikipedia redirects + anchor text, (b) **prior probability** `p(e | m)` from anchor-count statistics, and (c) optionally a dense retrieval index (BLINK; Wu et al. 2020).
2. **Disambiguation** — a scorer that ranks candidates using local context (the sentence around the mention) and global coherence (other mentions in the document agreeing on related entities).

### Coreference architecture

The modern baseline is the **mention-ranking / higher-order coreference** model (Lee et al. 2017; Lee et al. 2018; Joshi et al. 2020 for SpanBERT). Enumerate spans up to a width cap, prune to a candidate set, and for each mention greedily pick one antecedent (or `null`) by score. Transitive closure of antecedent arrows produces clusters.

The deliverable has two parts:

- **EL pipeline** with an alias-table candidate generator, a cross-encoder disambiguator (`AutoModelForSequenceClassification` scoring `(mention + context, candidate description)` pairs), and a NIL-threshold for mentions absent from the KB.
- **Coreference resolver** built on top of the exercise-02 span encoder, with a mention-pair scorer and standard cluster metrics from `coval` (Moosavi et al. 2019).

## 2. Worked implementation

### 2.1 Project layout

```
el_coref/
  el/
    alias.py          # alias table + prior p(e|m)
    retriever.py      # dense retrieval (optional; BLINK-style)
    disambiguator.py  # cross-encoder scorer
    nil.py            # NIL threshold calibration
  coref/
    spans.py          # span enumeration (shared with ex-02)
    model.py          # mention-pair scorer
    decode.py         # antecedent → cluster (union-find)
  metrics.py          # EL accuracy; MUC / B³ / CEAF-ϕ4
  run_el.py
  run_coref.py
  tests/
  requirements.txt
```

### 2.2 EL — candidate generation with alias table + prior

Build an alias table from Wikipedia anchor text (every anchor `[[wikititle|surface]]` is evidence that `surface` can refer to `wikititle`). Count `c(e, m)` (how many times surface `m` linked to entity `e`) and `c(m)` (total anchor count for `m`). The **prior** `p(e | m) = c(e, m) / c(m)` is the most useful single feature in classical EL — Hoffart et al. 2011 reports that prior alone gets ~71 % accuracy on AIDA-CoNLL.

```python
# el/alias.py
from collections import defaultdict


class AliasTable:
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))  # alias -> {entity: count}
        self.totals = defaultdict(int)

    def add(self, alias: str, entity: str, n: int = 1) -> None:
        a = _normalize(alias)
        self.counts[a][entity] += n
        self.totals[a] += n

    def candidates(self, alias: str, k: int = 30) -> list[tuple[str, float]]:
        a = _normalize(alias)
        if self.totals[a] == 0:
            return []
        items = sorted(self.counts[a].items(), key=lambda x: -x[1])[:k]
        return [(e, c / self.totals[a]) for e, c in items]


def _normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())
```

Two productionisation decisions worth defending:

- **Normalise before lookup.** Case-insensitive, whitespace-collapsed, stripped. Case-matters mentions ("Apple" company vs "apple" fruit) are a bug 5 % of the time in real text and a 90 % win when you normalise.
- **Top-k cap.** Keeping only top-30 candidates per mention trades recall for a tractable disambiguation step. `k=30` recovers >95 % of gold candidates on AIDA; raising to 100 adds <1 point and triples disambiguation cost.

### 2.3 EL — cross-encoder disambiguation

Given a mention in context and a candidate's description from the KB, score the pair with a cross-encoder:

```python
# el/disambiguator.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


def build_disambiguator(encoder_id: str):
    tok = AutoTokenizer.from_pretrained(encoder_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(encoder_id, num_labels=1)
    return tok, mdl


def _format_pair(mention: str, left_ctx: str, right_ctx: str, cand_title: str, cand_desc: str):
    # The entity marker [M] ... [/M] tells the encoder which span to attend to.
    a = f"{left_ctx} [M] {mention} [/M] {right_ctx}"
    b = f"{cand_title}. {cand_desc}"
    return a, b


@torch.no_grad()
def score_candidates(tok, mdl, mention, left_ctx, right_ctx, candidates, prior_weight: float = 0.3):
    """candidates: list of (entity_id, prior, title, description)."""
    pairs = [_format_pair(mention, left_ctx, right_ctx, t, d) for _, _, t, d in candidates]
    batch = tok([a for a, _ in pairs], [b for _, b in pairs],
                padding=True, truncation=True, max_length=256, return_tensors="pt")
    scores = mdl(**batch).logits.squeeze(-1)  # (K,)
    priors = torch.tensor([p for _, p, _, _ in candidates]).log()
    combined = scores + prior_weight * priors  # log-linear combination
    best = int(combined.argmax())
    return candidates[best][0], float(combined[best])
```

The log-linear combination of the cross-encoder score and the prior log-probability is the standard recipe (Ganea & Hofmann 2017; Hoffart et al. 2011): priors drop long-tail errors cheaply, cross-encoder catches context-dependent cases the prior cannot resolve.

### 2.4 EL — NIL (unlinkable) handling

Not every mention has a KB entity (e.g. "my cousin Sam"). A deployed EL system **must** have a NIL path, or it will confidently map every mention to the closest Wikipedia page:

```python
# el/nil.py
def link_or_nil(cand_best_entity, cand_best_score, cand_second_score, thresholds):
    if cand_best_score < thresholds["min_score"]:
        return "NIL"
    if cand_best_score - cand_second_score < thresholds["margin"]:
        return "NIL"
    return cand_best_entity
```

Calibrate `thresholds["min_score"]` and `thresholds["margin"]` on a validation set that **includes NILs** — AIDA-CoNLL has them. If your training corpus does not have NILs, add synthetic NILs by holding out a random 20 % of KB entities at train time; otherwise the thresholds are tuned on a distribution that never fires NIL.

### 2.5 Coreference — mention pair scorer

Reuse the span enumerator from exercise-02. Produce span representations, then score every `(mention, antecedent)` pair:

```python
# coref/model.py
import torch
import torch.nn as nn
from transformers import AutoModel


class CorefScorer(nn.Module):
    def __init__(self, encoder_id: str, hidden: int = 150, max_antecedents: int = 50):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_id)
        H = self.encoder.config.hidden_size
        self.span_repr = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Dropout(0.3))
        self.mention_score = nn.Linear(H, 1)
        self.pair_score = nn.Sequential(
            nn.Linear(3 * H, hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )
        self.max_antecedents = max_antecedents

    def spans(self, token_hidden, starts, ends):
        s = token_hidden.gather(1, starts.unsqueeze(-1).expand(-1, -1, token_hidden.size(-1)))
        e = token_hidden.gather(1, ends  .unsqueeze(-1).expand(-1, -1, token_hidden.size(-1)))
        return self.span_repr(torch.cat([s, e], dim=-1))  # (B, S, H)

    def forward(self, input_ids, attention_mask, starts, ends):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        g = self.spans(h, starts, ends)
        m = self.mention_score(g).squeeze(-1)  # (B, S)
        # Pair score for each span against its K nearest previous spans.
        # Lee et al. 2017: antecedent pruning by distance + mention score.
        # ... (full windowed pairing omitted for brevity; see Lee et al. 2017 for the exact loop)
        return g, m
```

Key ideas from the e2e-coref line:

- **Antecedent pruning.** Full O(n²) pair scoring is too expensive. Prune to the top-K antecedents (`K≈50`) per mention using the sum of mention scores plus a cheap distance feature.
- **Dummy antecedent.** Each mention has a "no antecedent" option scored at 0. This makes the loss a softmax over `K + 1` options; the model learns to pick it when nothing fits.
- **Transitive closure.** After decoding the per-mention antecedent, build the mention-cluster graph (each `(mention → antecedent)` is an edge) and extract connected components via union-find. The result is a partition of mentions into clusters.

### 2.6 Cluster metrics

Coreference is evaluated with **three** cluster metrics averaged into CoNLL-F1 — no single metric is sufficient (Pradhan et al. 2012):

- **MUC** — link-based; favours systems that create many links (biased toward recall).
- **B³ (B-cubed)** — mention-based; averages precision/recall over mentions. Standard for most reporting.
- **CEAF-ϕ4** — entity-based; penalises singleton explosion.

Use the reference implementations in `coval` (Moosavi et al. 2019) — rolling your own is almost always slightly wrong, and the deltas are decisive for publication-quality comparisons.

## 3. Validation steps

1. **Alias normalisation test.** `AliasTable.candidates("Apple")`, `AliasTable.candidates("apple ")`, `AliasTable.candidates("APPLE")` must all return the same list. A regression here is a common silent recall loss.
2. **Prior-alone baseline.** Report accuracy from "pick the top candidate by prior" with no disambiguator. On AIDA-CoNLL it should land near 71 %. If your implementation reports much lower, the alias table is missing redirects or under-normalised.
3. **NIL threshold calibration.** Sweep `margin` on validation and report precision/recall at each point. The deployed threshold must sit on the validation curve, not be picked by inspection of the test set.
4. **Coreference cluster monotonicity.** Transitive closure of antecedent arrows must produce clusters whose size is monotonic in the number of linked antecedents. Unit-test: `[(1→0), (2→1), (3→null)]` must yield `{{0,1,2}, {3}}`.
5. **Use `coval` for the headline.** Report MUC / B³ / CEAF-ϕ4 / CoNLL-F1 computed by `coval`. Rolling your own B³ is a trap; the reference implementation is 10 lines of configuration away.
6. **EL + coref interaction.** If you deploy both, run them in order **coref → EL** and report the EL-on-cluster-head accuracy. Running EL per mention and merging after loses the cluster-level evidence that fixes long-tail disambiguation.
7. **KB freshness.** Record the Wikipedia/Wikidata dump date in the output. An EL model trained in January scoring an August test set has known coverage drift for anything newer than the dump.
8. **Seed and dep echoing.** Record encoder id, seed, candidate `k`, NIL thresholds, and `coval` version.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Two-stage EL** | Alias table + prior + cross-encoder; top-k documented | Prior only, or cross-encoder only | Single-step classifier over full KB (intractable, often secretly fakes it) |
| **Prior normalisation** | Case-insensitive, whitespace-normalised, redirects applied | Normalised at store time but not query time | None |
| **NIL handling** | Threshold calibrated on val with NILs; synthetic NILs added when absent | Hardcoded threshold | No NIL path; every mention links |
| **Coreference architecture** | Mention-ranking with dummy antecedent, pruning, transitive closure | One of those | Ad hoc pronoun-resolution heuristics |
| **Cluster metrics** | CoNLL-F1 via `coval`: MUC + B³ + CEAF-ϕ4 all reported | B³ only | Mention-level F1 reported as "coref F1" |
| **EL–coref interaction** | Coref → EL on cluster heads; interaction reported | Independent, no cluster integration | Each mention linked independently; cluster evidence discarded |
| **KB provenance** | Dump date, redirect source, alias count recorded | Dump date only | None |
| **Threat model** | Adversarial alias pollution / KB poisoning noted per OWASP ML Top 10 and MITRE ATLAS | Mentioned without concrete mitigation | Ignored |

## 5. Common mistakes

- **No NIL path.** Links every mention to something; silently maps "my cousin Sam" to `Sam_Houston_(singer)`. Any system without an explicit NIL threshold is bluffing on the ~15–20 % of mentions that have no KB entity.
- **Testing NIL thresholds on test.** If you pick the margin by inspecting test accuracy, the reported number is an upper bound that production will not see. Calibrate on val with NILs; freeze.
- **Ignoring Wikipedia redirects.** "Barack Obama" and "Obama" and "President Obama" all redirect to the same page; without the redirect table your alias stats split across them and the prior under-performs by 3–5 points.
- **Rolling your own B³.** The implementation has edge cases (singletons, empty cluster) that `coval` handles and most hand-rolls do not. A homebrew B³ can differ from the standard by 1–2 points and no one spots it until publication.
- **Reporting only MUC.** MUC alone biases toward systems that link aggressively; it's useless as a solo headline. CoNLL-F1 is the agreed average because the three metrics disagree in known ways.
- **Running EL mention-by-mention after coref.** Coref cluster heads carry more context than any single mention; linking the head and propagating to the cluster is more accurate than linking each mention independently and voting. Many pipelines get this wrong by default.
- **KB dump drift.** Linking a 2024 news article against a 2019 KB silently fails on every post-2019 entity. Record the dump date and surface it in the API response so consumers know.
- **Case-sensitive alias lookup.** User-generated text ("apple", "Apple", "APPLE") splits across three alias-table entries; normalise at both build and query time.
- **Treating coreference as pronoun resolution.** Pronouns are a minority of coreferent mentions. Named-mention coreference ("Obama" and "the president") is the harder and more common case; a pronoun-only system misses 60 %+ of real links.
- **No antecedent pruning in training.** Scoring every span pair is O(n²) at best; without pruning the model trains slowly and eval times are dominated by uninformative far-apart pairs. Prune by distance + mention score.

## 6. References

Primary sources used in this solution:

- Hoffart, J. et al. (2011). *Robust Disambiguation of Named Entities in Text*. EMNLP. <https://aclanthology.org/D11-1072/> — AIDA-CoNLL dataset and the prior + context disambiguation recipe in §2.3.
- Mihalcea, R., Csomai, A. (2007). *Wikify! Linking Documents to Encyclopedic Knowledge*. CIKM. <https://doi.org/10.1145/1321440.1321475> — foundational two-stage EL (candidate gen + disambiguation).
- Ganea, O.-E., Hofmann, T. (2017). *Deep Joint Entity Disambiguation with Local Neural Attention*. EMNLP. <https://aclanthology.org/D17-1277/> — log-linear combination of local score and prior used in §2.3.
- Kolitsas, N., Ganea, O.-E., Hofmann, T. (2018). *End-to-End Neural Entity Linking*. CoNLL. <https://aclanthology.org/K18-1050/> — joint mention detection + linking.
- Wu, L., Petroni, F., Josifoski, M., Riedel, S., Zettlemoyer, L. (2020). *Scalable Zero-shot Entity Linking with Dense Entity Retrieval* (BLINK). EMNLP. <https://aclanthology.org/2020.emnlp-main.519/> — dense retrieval for candidate generation.
- De Cao, N., Izacard, G., Riedel, S., Petroni, F. (2021). *Autoregressive Entity Retrieval* (GENRE). ICLR. <https://arxiv.org/abs/2010.00904> — generative EL as an alternative to retrieval-based.
- Ayoola, T. et al. (2022). *ReFinED: An Efficient Zero-shot-capable Approach to End-to-End Entity Linking*. NAACL. <https://aclanthology.org/2022.naacl-industry.24/> — current industry baseline; relevant for deployment choices.
- Petroni, F. et al. (2021). *KILT: A Benchmark for Knowledge-Intensive Language Tasks*. NAACL. <https://aclanthology.org/2021.naacl-main.200/> — EL alongside the other KB-grounded tasks.
- Lee, K., He, L., Lewis, M., Zettlemoyer, L. (2017). *End-to-end Neural Coreference Resolution*. EMNLP. <https://aclanthology.org/D17-1018/> — mention-ranking architecture used in §2.5.
- Lee, K., He, L., Zettlemoyer, L. (2018). *Higher-order Coreference Resolution with Coarse-to-fine Inference*. NAACL. <https://aclanthology.org/N18-2108/> — higher-order refinement.
- Joshi, M., Chen, D. et al. (2020). *SpanBERT: Improving Pre-training by Representing and Predicting Spans*. TACL. <https://aclanthology.org/2020.tacl-1.5/> — the canonical encoder for coreference (+ a few F1 points for free).
- Pradhan, S. et al. (2012). *CoNLL-2012 Shared Task: Modeling Multilingual Unrestricted Coreference in OntoNotes*. EMNLP-CoNLL. <https://aclanthology.org/W12-4501/> — the MUC + B³ + CEAF-ϕ4 CoNLL-F1 metric.
- Moosavi, N. S., Rahman, S., Strube, M. (2019). *coval*: a toolkit for coreference evaluation. <https://github.com/ns-moosavi/coval> — the reference metric implementation used in §2.6.
- Jurafsky, D., Martin, J. (2024). *Speech and Language Processing*, 3rd ed. draft, ch. 23 (*Coreference Resolution*). <https://web.stanford.edu/~jurafsky/slp3/> — text treatment.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML02 Data Poisoning*: EL depends on anchor-text statistics and Wikipedia dump; adversarial edits to Wikipedia poison the alias table and shift the prior. *ML01 Input Manipulation Attack*: crafted mentions can force NIL or mislink.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue applicable to KB-poisoning and alias-table attacks against EL services.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Measure*: report CoNLL-F1 via `coval` rather than homebrew; document KB dump date; *Map*: document the KB licence (Wikipedia is CC BY-SA; Wikidata is CC0) and the training-data licence.
