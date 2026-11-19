# exercise-05 — Low-Resource MT with Back-Translation

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-107-machine-translation-and-multilingual-nlp/exercises/exercise-05-low-resource-mt-with-back-translation.md`.
> The target learning outcomes implied by that filename: lift a low-resource direction (e.g. `en→sw`, `en→ur`) by **synthesising parallel data via back-translation** (Sennrich et al. 2016), understand the difference between **beam-search** and **sampled / noised** back-translation (Edunov et al. 2018), **tag synthetic data** to prevent register drift (Caswell et al. 2019), and report the quality curve as a function of synthetic-to-authentic mixing ratio.

## 1. Solution overview

Back-translation is the single most effective trick in low-resource NMT, but naive back-translation makes the model *worse*, not better. The three failure modes that decide whether the lift is real:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Using beam-search back-translations only** | Beam-search BT produces fluent but repetitive, translation-ese source side; the forward model overfits to that distribution and generalises poorly to real inputs. Edunov et al. 2018 showed sampled or noised BT gives +0.5 to +2 BLEU over beam BT at scale | Use **sampled** (`do_sample=True`, `top_k=40` or `top_p=0.9`) or **noised** back-translation (add word drop/swap/replace noise on the back-translated source). Keep temperature ≥ 0.8 |
| **Unbalanced synthetic/authentic mixing** | 90 % synthetic + 10 % authentic overwhelms the model's exposure to real source distributions; quality on a held-out real test set collapses even as training loss falls | Cap synthetic ≤ 1–3× authentic; upsample authentic to match; report the quality curve for several mixing ratios so the sweet spot is defended, not guessed |
| **Register drift from untagged synthetic data** | Synthetic source looks slightly off (translation-ese: SVO rigidity, under-use of pronouns, missing idioms). Without a marker the model cannot tell "real" from "synthetic" at train time and blurs its outputs | **Tag synthetic examples** at the start of the source sentence (`<BT>`) following Caswell et al. 2019. The tag is dropped at inference; the model learns to produce "real-register" output when no tag is present |

The deliverable is a BT training harness with five components:

1. **Monolingual target-language corpus** — the thing you want more of. For `en→sw`, download high-quality `sw` text (News Commentary, Wikipedia, OPUS CCMatrix filtered).
2. **Reverse direction model** — fine-tuned `sw→en` NMT (either from scratch on the authentic parallel data or an NLLB-200 initialisation).
3. **Back-translator** — runs the reverse model on the monolingual `sw` text to produce synthetic `(en_synth, sw_real)` pairs.
4. **Forward direction model** — trained on (authentic `(en, sw)`) + (tagged synthetic `(<BT> en_synth, sw_real)`) at a controlled ratio.
5. **Evaluator** — reports BLEU / chrF / COMET on the real FLORES devtest for each mixing ratio; produces the quality curve that justifies the final ratio.

Reference recipes:
- **Sennrich et al. 2016** — the original back-translation paper; established the pattern for both high-resource (news) and low-resource (en-tr) directions.
- **Edunov et al. 2018** — showed that *sampled* back-translation outperforms beam-search BT at large scale, and that *noised* BT closes the gap at small scale; the paper that moves BT from "works" to "works well".
- **Caswell et al. 2019** — tagged back-translation; a `<BT>` prefix lets the model distinguish synthetic from authentic without an extra loss term.

Reference directions and data:
- **Authentic parallel data**: `OPUS-100` or `FLORES-200` direction-specific splits; always confirm licence (CC-BY-SA-4.0 for FLORES-200; per-source for OPUS-100).
- **Monolingual target data**: `wmt-news` monolingual releases, Wikipedia dumps filtered to the target language, OSCAR / CC-100 monolingual subsets.

## 2. Worked implementation

### 2.1 Project layout

```
bt_lift/
  reverse.py        # train or load the reverse-direction NMT
  back_translate.py # sampled BT over monolingual corpus
  tag.py            # <BT> tagging / stripping
  mix.py            # authentic + synthetic mixing with capped ratio
  forward.py        # train the forward direction on the mixed set
  curve.py          # ratio sweep + metric panel
  run.py            # CLI
  tests/
    test_tag.py
    test_sampling_diversity.py
    test_mix_ratio.py
  requirements.txt
```

### 2.2 Reverse-direction model

```python
# reverse.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_reverse(model_id: str = "facebook/nllb-200-distilled-600M"):
    """Load the reverse direction (target_lang -> source_lang) model.
    For truly low-resource reverse directions, prefer fine-tuning NLLB on the
    small authentic parallel data in the REVERSE direction rather than training
    from scratch. See exercise-01 for the fine-tune recipe.
    """
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return model, tok
```

The reverse model determines the *quality floor* of the synthetic source side. A weak reverse model produces nonsense English back-translations; the forward model then learns to produce nonsense from nonsense. In practice this means: if you have under 10k authentic sentence pairs, bootstrap the reverse model from NLLB-200 (which has broad coverage) and fine-tune lightly rather than training from scratch.

### 2.3 Sampled back-translation — not beam search

```python
# back_translate.py
import torch
from transformers import GenerationConfig


def sampled_back_translate(
    model, tokenizer,
    target_sentences: list[str],
    src_lang_code: str = "swh_Latn",   # NLLB code for the "target of BT" = source of reverse model
    tgt_lang_code: str = "eng_Latn",
    batch_size: int = 16,
    top_k: int = 40,
    top_p: float = 0.95,
    temperature: float = 0.9,
) -> list[str]:
    tokenizer.src_lang = src_lang_code
    gen_cfg = GenerationConfig(
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=256,
        num_beams=1,  # Edunov: sampling beats beams for BT at scale
        early_stopping=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang_code),
    )
    outs: list[str] = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(target_sentences), batch_size):
            batch = target_sentences[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            gen = model.generate(**enc, generation_config=gen_cfg)
            outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outs
```

Points worth defending in review:

- **`do_sample=True` with `top_k=40, top_p=0.95, temperature=0.9`.** The Edunov recipe. Beam-search BT gives repetitive, translation-ese source side; sampled BT gives diverse source side that the forward model generalises from.
- **`num_beams=1`.** Mixing sampling and beams is unusual; the Edunov paper's "sampled BT" is `num_beams=1` with sampling on. The alternative *noised beam* path (`num_beams=5` + token noise) is in §2.4.
- **Record RNG seed for the sampling pass.** Sampled BT varies run-to-run; without a seed your "sampled BT corpus" is not reproducible and a future reviewer cannot re-derive your result.
- **Direction codes.** `src_lang_code` is the *target* of back-translation = the *source* of the reverse model. If `sw` is the forward-direction target, the reverse model reads `sw` (`swh_Latn`) and writes `en` (`eng_Latn`). Swapping these produces a "back-translation" that is actually a copy, and the forward model's BLEU drops silently.

### 2.4 Noised back-translation — the small-data alternative

```python
# back_translate.py (continued)
import random


def word_noise(sentence: str, p_drop=0.1, p_swap=0.1, p_blank=0.1, seed=None) -> str:
    """Edunov-style word noise: drop, swap-with-neighbour, replace with <BLANK>.
    Applied to the back-translated source *after* it is produced."""
    rng = random.Random(seed)
    tokens = sentence.split()
    # Drop
    tokens = [t for t in tokens if rng.random() > p_drop]
    # Swap neighbours
    for i in range(len(tokens) - 1):
        if rng.random() < p_swap:
            tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
    # Blank
    tokens = ["<BLANK>" if rng.random() < p_blank else t for t in tokens]
    return " ".join(tokens)
```

Noised BT is the recommended path when the monolingual corpus is small (< 1 M sentences) and sampling diversity alone is not producing enough variety. Edunov et al. 2018 Table 4 shows that noised-beam BT matches sampled BT on low-resource directions at a fraction of the compute.

### 2.5 Tagging synthetic data — the Caswell trick

```python
# tag.py
BT_TAG = "<BT>"


def tag_synthetic(source: str) -> str:
    return f"{BT_TAG} {source}"


def strip_tag(source: str) -> str:
    return source[len(BT_TAG) + 1:] if source.startswith(BT_TAG + " ") else source


def add_tag_to_vocab(tokenizer):
    if BT_TAG not in tokenizer.get_vocab():
        tokenizer.add_tokens([BT_TAG], special_tokens=True)
    return tokenizer.convert_tokens_to_ids(BT_TAG)
```

- **Prepend `<BT>` to synthetic source sentences only.** Authentic parallel sentences stay untagged.
- **Add `<BT>` as a special token on the tokenizer.** Without this, the tag fragments into 3–4 subwords (`<`, `B`, `T`, `>`) and loses its identity. `resize_token_embeddings` must follow the tokenizer update.
- **Strip the tag at inference time.** Production inputs are untagged; the forward model was trained to produce "real-register" output when no tag is present. This is the whole point of the Caswell protocol.

### 2.6 Mixing — the ratio is the experiment

```python
# mix.py
import random
from datasets import Dataset, concatenate_datasets


def mix_authentic_and_synthetic(
    authentic: Dataset,          # columns: src, tgt
    synthetic: Dataset,          # columns: src (tagged), tgt
    synthetic_to_authentic: float = 1.0,
    seed: int = 20260101,
) -> Dataset:
    """Cap synthetic:authentic ≤ ratio by truncation (not upsampling of synthetic);
    optionally upsample authentic so both appear more often per epoch."""
    n_auth = len(authentic)
    max_syn = int(n_auth * synthetic_to_authentic)
    rng = random.Random(seed)
    idx = list(range(len(synthetic)))
    rng.shuffle(idx)
    synthetic = synthetic.select(idx[:max_syn])
    return concatenate_datasets([authentic, synthetic]).shuffle(seed=seed)
```

- **Cap, don't fill.** If your synthetic corpus is 10× your authentic one, truncating synthetic down to the ratio is cheaper and more honest than upsampling authentic. If you *must* use all synthetic, upsample authentic to match; never silently let `synthetic:authentic` exceed 3:1 without reporting it.
- **Shuffle after concatenation.** Otherwise all authentic sentences land in the early mini-batches and all synthetic sentences in the later ones; the model sees a huge distribution shift mid-epoch.

### 2.7 Forward training

```python
# forward.py
from .mix import mix_authentic_and_synthetic
from .tag import add_tag_to_vocab
from .reverse import load_reverse  # reuse for the forward model init too if NLLB-based


def train_forward(
    authentic_ds, synthetic_ds,
    ratio: float,
    model_id: str = "facebook/nllb-200-distilled-600M",
    src_lang: str = "eng_Latn",
    tgt_lang: str = "swh_Latn",
    output_dir: str = "runs/forward",
    seed: int = 20260101,
):
    from .reverse import AutoModelForSeq2SeqLM, AutoTokenizer  # re-import for clarity
    tok = AutoTokenizer.from_pretrained(model_id)
    add_tag_to_vocab(tok)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.resize_token_embeddings(len(tok))
    mixed = mix_authentic_and_synthetic(authentic_ds, synthetic_ds, ratio, seed=seed)
    # ... standard Seq2SeqTrainer flow from exercise-01 §2.6 ...
    return mixed  # for the demo; full trainer wiring mirrors exercise-01
```

### 2.8 The quality curve — the artefact that defends the choice

```python
# curve.py
import json
from .forward import train_forward
from .reverse import load_reverse  # for the reverse model used by BT
from .back_translate import sampled_back_translate
from .tag import tag_synthetic

RATIOS = [0.0, 0.5, 1.0, 2.0, 3.0]


def run_curve(authentic_ds, monolingual_target, output):
    reverse_model, reverse_tok = load_reverse()
    synth_src = sampled_back_translate(reverse_model, reverse_tok, monolingual_target)
    synthetic_ds = authentic_ds.from_dict({
        "src": [tag_synthetic(s) for s in synth_src],
        "tgt": monolingual_target,
    })
    rows = []
    for r in RATIOS:
        trainer = train_forward(authentic_ds, synthetic_ds, ratio=r)
        # Evaluate on real FLORES devtest with the exercise-04 metric panel
        rows.append({"ratio": r, **evaluate_real_devtest(trainer)})  # from mt_eval/
    with open(output, "w") as f:
        json.dump(rows, f, indent=2)
```

The curve is the artefact. A single-ratio run cannot answer "is BT helping?"; a curve shows that BLEU rises from `ratio=0` to a peak (often around 1–2×) and then flattens or regresses. Shipping without the curve is shipping a guess.

## 3. Validation steps

1. **Tag round-trip.** `strip_tag(tag_synthetic("hello"))` → `"hello"`. Catches off-by-one in the slicing.
2. **Tag tokenisation.** `tokenizer("<BT> hello", add_special_tokens=False).input_ids[0] == tokenizer.convert_tokens_to_ids("<BT>")`. If the tag fragments into subwords, `add_tokens` wasn't followed by `resize_token_embeddings` and your tagged corpus is actually a lightly-noised untagged corpus.
3. **Sampling diversity floor.** Back-translate the same `sw` sentence 10 times at `temperature=0.9`; distinct output count should be ≥ 7. If it is 1 or 2, sampling is not on (`do_sample=True`) and you are running beam-BT despite the config.
4. **Direction-code sanity.** Pin a round-trip: back-translate `"Habari za asubuhi"` (sw) with `src_lang=swh_Latn, tgt_lang=eng_Latn`. Output must be English, roughly `"Good morning"`. If it comes back in Swahili, your `src_lang` / `tgt_lang` are swapped.
5. **Mixing ratio invariant.** `len(mix_authentic_and_synthetic(auth, synth, ratio=2.0)) ≤ 3 * len(auth)`. If it isn't, the cap is broken and you're training on more synthetic than you reported.
6. **No-synthetic baseline = authentic-only.** `ratio=0.0` must produce a dataset whose contents are exactly `authentic`. Catches a bug where `synthetic` leaks when the cap underflows.
7. **Curve shape sanity.** On FLORES devtest for a direction with 10k authentic pairs, the BLEU curve should rise from `ratio=0` through `ratio=1` by 2–5 BLEU and then flatten. A curve that peaks at `ratio=0` means either the reverse model is too weak (output is nonsense) or you forgot to tag (register drift).
8. **Untagged-at-inference check.** Translate 100 real en sentences *without* a `<BT>` prefix; translate the same 100 with a `<BT>` prefix. Compare BLEU. The no-tag path should win (that is the Caswell result). If the tag helps at inference, the model has learnt to translate better in "synthetic mode" — a bug in the training data (e.g. all authentic also tagged).
9. **Monolingual filtering.** Audit 100 random back-translated sentences by eye. If you see repeated loops ("the the the"), the reverse-model generation config has `no_repeat_ngram_size=0` *and* sampling off; raise temperature or enable `no_repeat_ngram_size` on the reverse model only (safe because we are not scoring BT outputs by BLEU).
10. **Real-eval invariant.** Metrics *must* come from the real FLORES-200 devtest (or a human-held test set), never from the synthetic pool. Add an assertion that the eval split has no overlap with the monolingual-target corpus; a 50-line check that saves weeks of mystery wins.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Sampled or noised BT** | `do_sample=True` + top-k/top-p; or beam + Edunov noise; seed recorded | One of them | Beam-search BT only |
| **Tagging discipline** | `<BT>` added as special token; tag applied to synthetic only; strip-tag at inference tested | Tagging present, not tested | Untagged synthetic; register drift |
| **Reverse-model quality** | Reverse model BLEU on held-out reverse-direction test reported alongside forward quality | Reverse model used, no eval | Reverse model unknown; synthetic data not characterised |
| **Mixing ratio curve** | Curve over ≥ 3 ratios reported with BLEU + chrF + COMET per ratio | Single ratio reported | No ratio justification |
| **Cap invariant** | Hard cap on synthetic:authentic enforced and asserted | Cap documented, not enforced | No cap; synthetic >> authentic silently |
| **Real-eval invariant** | Eval uses human-translated held-out set (FLORES-200 devtest / WMT); no overlap with monolingual corpus | Overlap not checked | Eval set is self-translated |
| **Reproducibility** | Seeds for BT sampling, mixing, forward training; dataset revisions, model ids all in output JSON | Seeds set, no echo | No seeds |
| **Dataset hygiene** | Monolingual corpus licence recorded (OSCAR / CC-100 / Wikipedia terms); authentic parallel licence recorded | One of those | Neither |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks noted (data poisoning via adversarial monolingual corpus, bias amplification from unfiltered web data) with mitigations | Mentioned without mitigation | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure: direction, authentic/synthetic counts, ratio, reverse-model id, metric curve, known gaps for monolingual coverage | Partial | None |

## 5. Common mistakes

- **Beam-search back-translation at scale.** Fluent, repetitive, translation-ese; forward model overfits. Use sampled or noised BT (Edunov et al. 2018).
- **Un-tagged synthetic data.** The model blends synthetic and real distributions and produces translation-ese at inference. Tag with `<BT>` (Caswell et al. 2019); strip at inference.
- **Tag that isn't a special token.** `<BT>` fragments to subwords without `add_tokens` + `resize_token_embeddings`. The effect is the same as un-tagged.
- **Direction swap in BT.** `src_lang` should be the *target of the forward model* = *source of the reverse model*. Swapping produces synthetic pairs that are target-on-both-sides; the forward model learns to copy.
- **Synthetic ≫ authentic without a cap.** 10× or 100× synthetic drowns out real distributions. Cap at 1–3×.
- **Reporting a single ratio.** A one-point curve cannot show the sweet spot. Report ≥ 3 ratios.
- **Eval on the synthetic pool.** Translating back-translated test sentences inflates numbers (the forward model has already seen the target side). Eval must be on a human-held test set (FLORES-200 devtest or WMT newstest).
- **Weak reverse model.** If the reverse direction BLEU is below 15, synthetic sources are nonsense and the forward model gets *worse*. Characterise the reverse model first; if it is too weak, fine-tune it harder or initialise from a stronger pre-trained model (NLLB-200).
- **No-shuffle after mixing.** Authentic-then-synthetic batch order causes a huge distribution shift mid-epoch. Always `shuffle(seed=...)` after `concatenate_datasets`.
- **Monolingual corpus from the same source as the authentic parallel data.** If your monolingual `sw` is CCMatrix-filtered and your authentic parallel `(en, sw)` is also CCMatrix-filtered, you have target-side leakage and your BLEU is inflated. Hold out the monolingual target from any corpus that overlaps with authentic parallel data.
- **Reading dataset licences too late.** OSCAR / CC-100 / Wikipedia all have specific attribution and share-alike terms. Your ship-ready MT system must comply with the licences of the monolingual corpus too, not just the parallel one.

## 6. References

Primary sources used in this solution:

- Sennrich, R., Haddow, B., Birch, A. (2016). *Improving Neural Machine Translation Models with Monolingual Data*. ACL. <https://aclanthology.org/P16-1009/> — the original back-translation paper; the recipe that defines the authentic/synthetic mix in §2.6.
- Edunov, S. et al. (2018). *Understanding Back-Translation at Scale*. EMNLP. <https://aclanthology.org/D18-1045/> — the scaling study showing sampled BT > beam BT; the noised-BT variant used in §2.4.
- Caswell, I., Chelba, C., Grangier, D. (2019). *Tagged Back-Translation*. WMT. <https://aclanthology.org/W19-5206/> — the `<BT>` tag discipline in §2.5.
- Hoang, V. et al. (2018). *Iterative Back-Translation for Neural Machine Translation*. WMT. <https://aclanthology.org/W18-2703/> — the iterative BT extension that repeats §2.2–§2.7 with the improved forward model as the next reverse model; useful once the single-pass curve is understood.
- Fadaee, M., Monz, C. (2018). *Back-Translation Sampling by Targeting Difficult Words in Neural Machine Translation*. EMNLP. <https://aclanthology.org/D18-1040/> — the "sample hard" extension; good follow-up for truly small-corpus directions.
- NLLB Team (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation*. <https://arxiv.org/abs/2207.04672> — the NLLB-200 model used as reverse-direction initialisation; FLORES-200 eval protocol used in §2.8.
- Zhang, B. et al. (2020). *Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation*. ACL. <https://aclanthology.org/2020.acl-main.148/> — the OPUS-100 corpus used in §1 as the authentic-parallel source.
- Suárez, P. J., Sagot, B., Romary, L. (2019). *Asynchronous Pipeline for Processing Huge Corpora on Medium to Low Resource Infrastructures* (OSCAR). <https://ids-pub.bsz-bw.de/files/9021/Suarez_Sagot_Romary_Asynchronous_pipeline_for_processing_huge_corpora_2019.pdf> — the monolingual corpus used in low-resource BT pipelines.
- Freitag, M. et al. (2022). *Results of WMT22 Metrics Shared Task: Stop Using BLEU – Neural Metrics Are Better and More Robust*. WMT. <https://aclanthology.org/2022.wmt-1.2/> — the metric choice in §2.8 (COMET + chrF alongside BLEU).

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML02 Data Poisoning* applies directly to BT: a monolingual corpus is attacker-controllable (web-scraped data can be poisoned to embed target-side triggers); when the reverse model translates it and the pair is added to training, the trigger propagates. Mitigations: pin monolingual corpus revisions (OSCAR versions are hash-identified), filter for obvious adversarial content before BT, tag synthetic with `<BT>` so the trigger is at least bounded to one register. *ML07 Transfer Learning Attack* applies when the reverse model is a pre-trained public checkpoint; verify provenance.
- MITRE ATLAS — <https://atlas.mitre.org/> — technique catalogue; relevant whenever an MT system is updated via BT in a CI pipeline and the monolingual source is scraped on each run.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record authentic parallel count, monolingual count, synthetic cap, reverse-model id, forward-model recipe; *Measure*: report the ratio curve, not a single point; *Govern*: pin monolingual corpus revisions so a ship-ready BT corpus is auditable after the fact.
