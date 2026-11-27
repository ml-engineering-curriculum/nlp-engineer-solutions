# exercise-03 — ONNX or TorchScript Export and Quantisation

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-112-production-nlp-pipelines/exercises/exercise-03-onnx-or-torchscript-export-and-quantisation.md`.
> The target learning outcomes implied by that filename: take a fine-tuned transformer checkpoint,
> **export it to a portable runtime format** (ONNX or TorchScript), verify **numeric parity** with
> the PyTorch source, apply **dynamic or static INT8 quantisation**, re-verify parity and task
> accuracy, and publish a report that makes the latency / size / accuracy trade-off auditable.

## 1. Solution overview

Export + quantisation is where production NLP pipelines pick up their biggest latency and size
wins — and where silent correctness regressions most often land in production. The dominant
failure modes are:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **"It exports" treated as success** | `torch.onnx.export(...)` finishes with no error; the ONNX graph has subtle differences (dropped `attention_mask`, wrong dynamic axes, cast ordering) that silently change outputs | Export → immediately run a **parity test**: for ≥ 100 held-out inputs, max `abs(onnx_logits - torch_logits)` must be below `atol=1e-3, rtol=1e-3`. A bigger gap means the export is broken, not that ORT is inaccurate |
| **Dynamic axes mis-declared** | Fixed batch size and sequence length baked into the graph; at serve time any shape change throws or silently pads | Declare `dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, ...}`. Test with batch∈{1, 4, 16} and seq∈{32, 128, 512} |
| **Dynamic quantisation applied to models where static is required** | Dynamic quant quantises weights, activations stay fp32; works fine for linear-dominated encoders. On conv-heavy or activation-sensitive models, accuracy craters | Pick dynamic for encoder-only transformers (BERT, RoBERTa, DistilBERT). Pick static (with calibration dataset) when the dynamic variant exceeds the accuracy-loss budget |
| **No accuracy re-measurement** | Quantised model ships; accuracy regression caught by a user complaint six weeks later | Every export and every quantisation step is paired with an accuracy run on a held-out slice; delta is in the report |
| **TorchScript / ONNX opset mismatch with the serving runtime** | Export on opset 17; serve on ORT ≤ 1.14 which supports opset ≤ 16; runtime errors at load time | Pin the opset to the serving runtime's maximum supported; check `onnxruntime` release notes for the target version |
| **Export captures training-mode behaviour** | `model.train()` still active at export; `Dropout` fires at inference, outputs are non-deterministic | Call `model.eval()` before export and verify no `nn.Dropout` appears in the graph |
| **Tokenizer excluded from the export** | ONNX model exists for the transformer; tokenisation still happens in Python; the serving container needs the same tokenizer version | Ship the tokenizer alongside the ONNX graph (same directory); record the tokenizer `revision` in the manifest. Or export the tokenizer itself via `optimum` / `transformers.onnx` tokenizer wrappers when supported |

The deliverable is a four-part export-and-quantise harness:

1. **Export script** — `export_onnx.py` (or `export_torchscript.py`) that loads the fine-tuned
   checkpoint, exports with pinned opset and dynamic axes, saves the graph + tokenizer + a
   `manifest.json` with source sha, opset, exporter version.
2. **Parity test** — numeric parity between PyTorch and the exported graph on a 100-input
   sample; fails CI on `atol/rtol` miss.
3. **Quantisation script** — dynamic quant for encoders; static quant with a named calibration
   dataset for cases that need it; separate output directory for each variant.
4. **Accuracy + size + latency report** — markdown table with columns: {fp32 torch, fp32 ONNX,
   INT8 dynamic ONNX, INT8 static ONNX}, rows: accuracy (task metric), size (MB),
   p50/p95 latency (ms) on a fixed hardware / batch configuration.

Authoritative sources this solution draws on:

- **ONNX specification** — <https://github.com/onnx/onnx/blob/main/docs/IR.md>. Operator set
  (opset) versioning; shape inference; the formal IR.
- **ONNX Runtime quantisation** — <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>.
  `quantize_dynamic` and `quantize_static` APIs; QDQ format; calibration-data reader contract.
- **PyTorch ONNX export** — <https://pytorch.org/docs/stable/onnx.html>. `torch.onnx.export`,
  `dynamic_axes`, `opset_version`, the new `torch.onnx.dynamo_export` path.
- **PyTorch TorchScript** — <https://pytorch.org/docs/stable/jit.html>. `torch.jit.trace`,
  `torch.jit.script`, saving / loading, `torch.utils.mobile_optimizer`.
- **Hugging Face `optimum`** — <https://huggingface.co/docs/optimum/>. `optimum.onnxruntime`
  wrappers that handle the transformer-specific export boilerplate (`past_key_values`,
  attention-mask broadcasting, decoder-with-cache split) correctly.
- **Hugging Face `transformers.onnx`** — <https://huggingface.co/docs/transformers/serialization>.
  First-party export recipe; superseded by `optimum` for most cases but still authoritative for
  the configuration surface.
- **PyTorch quantisation reference** — <https://pytorch.org/docs/stable/quantization.html>.
  Backend quantisation (eager + FX) and the tradeoffs vs. ORT quantisation.
- **Transformers inference deployment reference** — <https://huggingface.co/docs/transformers/perf_infer_gpu_one>.

## 2. Worked implementation

### 2.1 Project layout

```
export_quantise/
  src/export_quantise/
    __init__.py
    export_onnx.py        # torch.onnx.export path
    export_torchscript.py # torch.jit.trace path
    quantise.py           # dynamic + static quant
    parity.py             # atol/rtol parity harness
    accuracy.py           # task metric on held-out slice
    manifest.py           # per-variant provenance
  configs/
    export.yaml           # source checkpoint, opset, dynamic axes, atol/rtol
    quant_static.yaml     # calibration dataset path, num-calibration-samples
  tests/
    test_parity.py
    test_dynamic_axes.py
    test_quantisation_accuracy.py
  reports/
    2026-11-26_export.md
    2026-11-26_export.csv
  pyproject.toml
  README.md
```

### 2.2 Export — ONNX with pinned opset and dynamic axes

```python
# src/export_quantise/export_onnx.py
from __future__ import annotations
import pathlib, hashlib, json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def export(
    checkpoint: str,
    revision: str,
    out_dir: pathlib.Path,
    opset: int = 17,
    max_seq_len: int = 512,
) -> dict:
    """Export an encoder classifier to ONNX. Returns a manifest dict."""
    tok = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    mdl = AutoModelForSequenceClassification.from_pretrained(checkpoint, revision=revision)
    mdl.eval()  # mandatory: dropout etc. must be off at export time

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    # A small dummy input; shapes are placeholders because dynamic_axes makes them symbolic.
    dummy = tok("export-time dummy text", padding="max_length",
                truncation=True, max_length=max_seq_len, return_tensors="pt")
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]
    token_type_ids = dummy.get("token_type_ids")

    inputs = (input_ids, attention_mask) if token_type_ids is None else \
             (input_ids, attention_mask, token_type_ids)
    input_names = ["input_ids", "attention_mask"] + (["token_type_ids"] if token_type_ids is not None else [])

    with torch.inference_mode():
        torch.onnx.export(
            mdl,
            inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                **({"token_type_ids": {0: "batch", 1: "seq"}} if token_type_ids is not None else {}),
                "logits":         {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )

    tok.save_pretrained(out_dir)
    manifest = {
        "kind": "onnx",
        "source_checkpoint": checkpoint,
        "source_revision": revision,
        "opset": opset,
        "max_seq_len": max_seq_len,
        "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "size_bytes": onnx_path.stat().st_size,
        "torch_version": torch.__version__,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
```

Points worth defending in review:

- **`model.eval()` is mandatory.** Dropout and BatchNorm statistics differ between train and
  eval; an exported graph that captured train-mode dropout is non-deterministic at inference.
- **`dynamic_axes` on *both* batch and seq.** Fixing either will reject real traffic at serve
  time (variable batch) or truncate silently (variable seq).
- **`do_constant_folding=True`.** Collapses literal-constant subgraphs. Makes the graph smaller
  and inference slightly faster; rarely changes numerics materially but always re-run parity.
- **`opset=17`** was chosen because ORT ≥ 1.15 and all major deploy targets support it as of
  late 2024. A project targeting an older runtime must lower this; a project targeting
  TensorRT may need to push it based on operator coverage. Always check the runtime's release
  notes.
- **Tokenizer saved alongside** so a downstream worker can `AutoTokenizer.from_pretrained(out_dir)`
  without reaching for the HF Hub.

### 2.3 Parity — ONNX vs. PyTorch on 100 inputs

```python
# src/export_quantise/parity.py
from __future__ import annotations
import pathlib, numpy as np, torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parity_check(
    torch_checkpoint: str,
    torch_revision: str,
    onnx_dir: pathlib.Path,
    sample_texts: list[str],
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> dict:
    tok = AutoTokenizer.from_pretrained(onnx_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(torch_checkpoint, revision=torch_revision).eval()

    # Torch reference
    enc = tok(sample_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.inference_mode():
        torch_logits = mdl(**enc).logits.cpu().numpy()

    # ONNX
    sess = ort.InferenceSession(str(onnx_dir / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    feed = {k: v.cpu().numpy().astype(np.int64) for k, v in enc.items()}
    onnx_logits = sess.run(None, feed)[0]

    abs_diff = np.abs(torch_logits - onnx_logits)
    return {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "atol": atol,
        "rtol": rtol,
        "passed": bool(np.allclose(torch_logits, onnx_logits, atol=atol, rtol=rtol)),
        "n_samples": len(sample_texts),
    }
```

A parity test that passes with `max_abs_diff ≈ 1e-6` is excellent. `1e-4` is routine. `1e-2` on
fp32 is a red flag — investigate the export; usually `do_constant_folding` plus a bad opset,
or an `attention_mask` dtype cast mismatch. For fp16 / bf16 exports, loosen `atol` to `1e-2`
and *re-run accuracy*, not *loosen the parity test and declare victory*.

### 2.4 Quantisation — dynamic (encoder default) and static (calibrated)

```python
# src/export_quantise/quantise.py
from __future__ import annotations
import pathlib, numpy as np
from typing import Iterable
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_dynamic, quantize_static, QuantType, CalibrationDataReader,
)


def dynamic_quantise(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Weight-only INT8; activations stay fp32. Safe default for encoder transformers."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        # per_channel=True gives tighter ranges on linear weights; the ORT quantisation docs
        # note it is the recommended default for transformer encoders.
        per_channel=True,
    )


class HFCalibrationDataReader(CalibrationDataReader):
    def __init__(self, texts: Iterable[str], tokenizer, max_len: int = 128):
        from transformers import AutoTokenizer  # noqa: F401
        self._iter = iter(texts)
        self._tok = tokenizer
        self._max_len = max_len

    def get_next(self):
        try:
            text = next(self._iter)
        except StopIteration:
            return None
        enc = self._tok(text, padding="max_length", truncation=True,
                        max_length=self._max_len, return_tensors="np")
        return {k: v.astype(np.int64) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}


def static_quantise(
    src: pathlib.Path,
    dst: pathlib.Path,
    calibration_texts: list[str],
    tokenizer,
) -> None:
    """Static INT8; both weights and activations quantised. Requires a calibration dataset."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    reader = HFCalibrationDataReader(calibration_texts, tokenizer)
    quantize_static(
        model_input=str(src),
        model_output=str(dst),
        calibration_data_reader=reader,
        quant_format=ort.quantization.QuantFormat.QDQ,   # QDQ is the modern recommendation
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
```

Dynamic quant is one call away; static quant needs a representative calibration set. The
calibration set should be drawn from the same distribution as serving traffic, not from the
validation set used to report accuracy — sharing data introduces a subtle form of
test-set leakage into the quantisation decision.

### 2.5 TorchScript equivalent (for projects not using ONNX)

```python
# src/export_quantise/export_torchscript.py
import pathlib, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def export_torchscript(checkpoint: str, revision: str, out_dir: pathlib.Path,
                       max_seq_len: int = 512) -> None:
    tok = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    mdl = AutoModelForSequenceClassification.from_pretrained(checkpoint, revision=revision)
    mdl.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    dummy = tok("tracing dummy", padding="max_length", truncation=True,
                max_length=max_seq_len, return_tensors="pt")
    example = (dummy["input_ids"], dummy["attention_mask"])

    with torch.inference_mode():
        # Trace captures the executed operators. For models with input-dependent control flow
        # (branching on attention_mask contents, etc.), prefer torch.jit.script.
        traced = torch.jit.trace(mdl, example, strict=False)
        traced.save(str(out_dir / "model.pt"))
    tok.save_pretrained(out_dir)
```

TorchScript traces have the same `train/eval` and dropout caveats; they additionally require
either `trace` for pure-data-flow graphs or `script` for control-flow-sensitive models.
Transformer encoders are typically pure-dataflow; decoder models with KV-cache paths usually
need `script`.

### 2.6 Dynamic-axes test — rectangular sanity check

```python
# tests/test_dynamic_axes.py
import numpy as np, onnxruntime as ort, pathlib


def _run(sess: ort.InferenceSession, batch: int, seq: int) -> np.ndarray:
    feed = {
        "input_ids":      np.zeros((batch, seq), dtype=np.int64),
        "attention_mask": np.ones((batch, seq),  dtype=np.int64),
    }
    if "token_type_ids" in {i.name for i in sess.get_inputs()}:
        feed["token_type_ids"] = np.zeros((batch, seq), dtype=np.int64)
    return sess.run(None, feed)[0]


def test_dynamic_batch_and_seq(tmp_path: pathlib.Path) -> None:
    sess = ort.InferenceSession("reports/model.onnx", providers=["CPUExecutionProvider"])
    for bs in (1, 4, 16):
        for sl in (32, 128, 512):
            out = _run(sess, bs, sl)
            assert out.shape[0] == bs, f"batch axis not dynamic at (bs={bs}, sl={sl})"
```

If either axis is not dynamic, this test fails loudly at CI time. Catching it before serving
avoids the "mysterious batch-1 timeouts" symptom in production.

### 2.7 Accuracy + size + latency report

A reviewable report is a single markdown table plus a CSV. Illustrative shape (numbers are
placeholders a submitter fills in from their measurements):

```markdown
# export_quantise/reports/2026-11-26_export.md

| Variant                | Size (MB) | Task metric | Δ vs. fp32 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| fp32 torch (baseline)  | <n> | <m> | 0 pp     | <n> | <n> |
| fp32 ONNX              | <n> | <m> | ~0 pp    | <n> | <n> |
| INT8 dynamic ONNX      | <n> | <m> | Δ_dyn pp | <n> | <n> |
| INT8 static QDQ ONNX   | <n> | <m> | Δ_stat pp| <n> | <n> |

Hardware: Intel Xeon 8375C, 32 cores, no GPU
Metric: macro-F1 on <validation slice> (n=<N>)
Parity vs torch (fp32 ONNX): max_abs_diff = <d>
```

If `Δ_stat > 1 pp` and `Δ_dyn ≤ 1 pp`, dynamic is the shipping variant; static earned the
space / latency win but failed the accuracy gate. If both fail, the exercise's answer is
"return to fp32 ONNX" — that is a legitimate finding and should be documented, not hidden.

## 3. Validation steps

1. **Export parity.** `parity_check(...)` returns `passed=True` with `max_abs_diff < 1e-3` on
   ≥ 100 held-out texts on CPU. A failure here is an export bug, not a runtime bug.
2. **Dynamic axes work.** `test_dynamic_axes.py` passes for `batch∈{1,4,16}` × `seq∈{32,128,512}`.
   Catches `dynamic_axes` misconfiguration before it reaches serving.
3. **Opset ↔ runtime compatibility.** `ort.get_available_providers()` + `onnx.load(path).opset_import`
   printed to the manifest; manually verify the target runtime supports it.
4. **Quantisation accuracy re-measured.** For each quant variant, run the task's primary metric
   (mod-111 rubric) on the validation slice; report Δ. Drop > 1 pp without a documented reason
   is a release block.
5. **Determinism.** Running ORT inference twice with the same inputs must produce bit-identical
   outputs on CPU. On GPU with enabled EP optimisations, allow a small `rtol`; if it exceeds
   `1e-3`, the EP is doing non-deterministic reductions and the caller should know.
6. **Size claim verifiable.** `size_bytes` in the manifest matches `os.path.getsize` of the
   artefact; the summary table matches the manifest. A mismatched size column is a report-
   plumbing bug.
7. **Latency on the target hardware.** The latency columns in the report were measured with the
   mod-112 exercise-02 harness on the hardware listed in the report header. A report that
   measures latency on the dev laptop and extrapolates is not shippable.
8. **Tokenizer version pinned.** The manifest records tokenizer `revision`; the tokenizer
   bundled with the export is the same revision. Catches the "trained with v1 tokenizer,
   served with v2 tokenizer, outputs are gibberish" regression.
9. **Static quant calibration data not from the validation set.** Calibration texts in
   `quant_static.yaml` point to a disjoint subset from the accuracy slice. A shared subset
   inflates the accuracy number of the static variant.
10. **Operator coverage.** ORT loads the model with the chosen EP (CPU / CUDA / TRT) with no
    "Unsupported op" warnings. A warning is a correctness risk even if inference succeeds —
    the op may have fallen back to CPU and the GPU latency number is incorrect.

## 4. Grading rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Exported artefact** | ONNX (or TorchScript) + tokenizer + manifest.json saved together | Artefact only | Script that exports but artefact not persisted |
| **Opset / runtime pin** | Opset chosen to match target runtime's max; recorded in manifest | Opset set; runtime not verified | Default opset, no documentation |
| **Dynamic axes** | Batch + seq dimensions dynamic; test matrix across {1,4,16}×{32,128,512} passes | One axis dynamic | Fixed shapes |
| **Parity test** | ≥ 100 samples, `atol=1e-3, rtol=1e-3`, logs max and mean diff | Smaller sample or looser tol with justification | Parity not tested |
| **Quant variants** | Dynamic + static (or documented reason to skip static); separate directories; separate manifests | Dynamic only, no reason | None |
| **Accuracy delta** | Task primary metric (mod-111) re-measured for every variant on a held-out slice | Measured on calibration set only | Not re-measured |
| **Size / latency report** | Markdown table + CSV with size, p50/p95, accuracy for every variant on named hardware | Table missing one column or CSV | Narrative only |
| **Determinism** | CPU inference bit-identical across runs; GPU rtol documented | CPU only | Non-determinism unexplained |
| **Supply-chain / provenance** | Source checkpoint sha, tokenizer revision, torch / ORT versions recorded | Partial | None |
| **Documentation** | README with run command, expected parity numbers, and the decision rule for choosing the shipping variant | README exists but no decision rule | None |

## 5. Common mistakes

- **Declaring success on `torch.onnx.export` finishing.** The exporter completing without an
  exception is necessary, not sufficient. Always run the parity harness.
- **Trusting fp16 export parity.** Casting to fp16 widens parity tolerance to ~1e-2 by itself;
  separate the cast change from the export change so the parity test still catches genuine
  export bugs.
- **Forgetting `model.eval()`.** Dropout fires and your logits become stochastic. A parity test
  flags it immediately if run with fixed seeds.
- **Dynamic quantising a decoder model.** `past_key_values` paths often confuse ORT's dynamic
  quant pass. Use `optimum.onnxruntime.ORTModelForCausalLM` which handles the decoder-with-cache
  export surface correctly, then quantise.
- **Static quant with 10 calibration samples.** 10 is too few; ORT quantisation guidance is
  hundreds to low thousands. Under-calibrated static quant produces activation ranges tuned to
  the quirks of the tiny sample.
- **Calibration data = validation data.** Tempting and wrong; leaks test-set statistics into
  quantisation and inflates the accuracy you report for the quantised model.
- **Shipping a graph with warnings.** ORT prints "Op 'X' is not optimised for this EP; falling
  back to CPU" — the GPU latency number you measured does not reflect the full compute path.
  Fix the op (upgrade opset, change an operator, update ORT).
- **Mixing `torch.jit.trace` with a control-flow-dependent model.** Trace captures the specific
  branch executed on the example input. A model with `if` on tensor contents (common in
  decoder generation paths) must be `torch.jit.script`-ed, not traced.
- **Exporting with `do_constant_folding=False`.** Larger graph, possibly slower; rarely needed.
  Turn it on and re-run parity.
- **Pinning an opset higher than the serving runtime supports.** The export succeeds on your
  dev machine; the serving pod fails to load with an error you did not test for. Serving-side
  smoke tests catch this; exporter-side opset-vs-runtime cross-check prevents it.
- **Quantising, reporting latency, forgetting accuracy.** The quantised variant is faster;
  that is not the ship criterion. The ship criterion is faster *and* within the accuracy
  tolerance.
- **Not pinning the exporter version.** `torch==2.X.Y` or `optimum==W.Z` must be pinned in
  `pyproject.toml`; exporter changes can produce different graphs from the same source.

## 6. References

Primary sources used in this solution:

- ONNX IR specification. <https://github.com/onnx/onnx/blob/main/docs/IR.md> — operator set
  versioning, shape inference, the formal IR.
- PyTorch ONNX export reference. <https://pytorch.org/docs/stable/onnx.html> —
  `torch.onnx.export`, `dynamic_axes`, `opset_version`, `do_constant_folding`.
- ONNX Runtime quantisation documentation.
  <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html> —
  `quantize_dynamic`, `quantize_static`, QDQ format, calibration data reader contract,
  per-channel recommendation for transformers.
- ONNX Runtime documentation. <https://onnxruntime.ai/docs/> — execution providers, session
  options, graph optimisation levels.
- PyTorch TorchScript reference. <https://pytorch.org/docs/stable/jit.html> — `trace` vs.
  `script`, save/load.
- PyTorch quantisation reference. <https://pytorch.org/docs/stable/quantization.html> —
  backend quantisation (eager + FX) and when to prefer PyTorch native over ORT quant.
- Hugging Face `optimum` ONNX documentation. <https://huggingface.co/docs/optimum/> —
  transformer-aware wrappers around ONNX export + ORT.
- Hugging Face `transformers.onnx` reference. <https://huggingface.co/docs/transformers/serialization>
  — first-party transformer export configuration.
- Hugging Face Hub revision pinning guide. <https://huggingface.co/docs/huggingface_hub/guides/download>
  — `revision=<sha>` for source-checkpoint provenance.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/>
  — *ML06 AI Supply Chain Attacks* applies to the exported artefact itself: a model file in
  `s3://.../model.onnx` with no integrity check is a drop-in replacement target. The
  `sha256` in the manifest and a verification step at load time are the mitigation. *ML07
  Transfer Learning Attack* applies to the source checkpoint; revision-pinning the source is
  the provenance control. *ML05 Model Theft* is tangentially relevant: a model served as an
  ONNX file in a container is easier to exfiltrate than a weights-in-RAM-only pattern; a
  threat-model note (not a mitigation) in the deploy documentation.
- MITRE ATLAS — <https://atlas.mitre.org/> — *AML.T0010 ML Supply Chain Compromise* applies
  to both the pre-trained source and the exported artefact; dual-provenance recording (source
  checkpoint sha + exported artefact sha256 + exporter version) is the minimum control.
  *AML.T0020 Poison Training Data* is upstream (data engineering), but the quantisation step
  can amplify a poisoned-model's robustness gap, so the accuracy re-measurement surface is a
  partial detector.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework>
  — *Map*: the artefact inventory (torch checkpoint → fp32 ONNX → INT8 dynamic / static). *Measure*:
  parity max_abs_diff, accuracy delta, size, latency. *Manage*: the ship decision rule (lowest-
  latency variant within the accuracy tolerance) encoded in the report. *Govern*: the
  manifest.json surface — shas, versions, dates — is the governance artefact; without it the
  pipeline has no audit trail.
