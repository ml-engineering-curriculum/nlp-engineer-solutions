# exercise-01 — spaCy Projects or HF Pipeline Composition

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-112-production-nlp-pipelines/exercises/exercise-01-spacy-projects-or-hf-pipeline-composition.md`.
> The target learning outcomes implied by that filename: compose a **multi-stage NLP pipeline**
> (tokenise → NER → linking → classification → post-processing) as a reproducible, versioned
> project using either **spaCy Projects** or the **Hugging Face `transformers.pipeline` + `datasets`**
> stack; declare every input/output artefact, pin every model/dataset version, make the pipeline
> runnable with a single command, and ship it with a README, a config, and tests that catch
> upstream breakage.

## 1. Solution overview

Production NLP rarely consists of a single model call. A realistic pipeline — say, extracting
regulated entities from inbound email, linking them to a canonical register, classifying the
email intent, and writing the structured output to a downstream queue — has four to eight
stages, each with its own model artefact, config, and failure mode. The dominant failure modes
are not model quality; they are composition failures:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Unpinned model versions** | `spacy.load("en_core_web_trf")` or `pipeline("ner", model="dslim/bert-base-NER")` resolves to whatever is cached locally; dev machine and prod machine silently disagree | Pin every artefact by *name + revision hash*. `transformers` supports `revision=<sha>`; spaCy supports `--pin` in Projects and immutable asset URLs |
| **Hidden I/O shapes** | Stage N emits `List[Dict]` with a key that stage N+1 did not expect; discovered at runtime, often in prod | Declare each stage's input and output schema (Pydantic / dataclass / spaCy `Doc.ext`); validate at stage boundaries so the failure is at stage N, not stage N+3 |
| **Non-reproducible runs** | "The numbers moved" between re-runs; source of move is unlogged | spaCy Projects assets have SHA checksums; HF pipelines log `model_sha`, `tokenizer_sha`, `datasets_version`. Capture both in the run manifest |
| **Monolithic pipeline** | One Python file wires 7 stages; a change to stage 3 forces re-running stages 1–2 that are expensive | Split into independently-cacheable commands (spaCy Projects' DAG; `datasets.map(..., cache_file_name=...)`); downstream re-runs reuse upstream outputs |
| **No CI on the pipeline graph** | A refactor changes the schema at stage 4; downstream stages keep running but drop fields silently | Every stage has a contract test with a golden fixture; the DAG runs end-to-end on a 10-document fixture in CI |
| **Model-artefact bloat in git** | 500 MB transformers checkpoints committed into the repo; clone time goes from seconds to minutes | Keep code in git; assets in a project asset store (DVC, Hub, S3) referenced by URL + SHA |

The deliverable is a four-part pipeline project:

1. **Project config** — `project.yml` (spaCy) or `config.yaml` + `pipeline.py` (HF). Declares assets,
   stages, inputs, outputs, commands, and the model/dataset revisions.
2. **Stage modules** — one file per stage with a typed `run(input) -> output` function and a
   schema at each boundary.
3. **Run manifest** — per-run JSON logging the resolved model shas, dataset shas, timings, and
   per-stage counts; dropped next to the final output so a downstream reviewer can reproduce.
4. **Tests + CI** — a 10-document golden fixture flowed end-to-end; a schema test per stage; a
   smoke test that `python -m <project> run end-to-end` exits 0 on a fresh clone.

Authoritative sources this solution draws on:

- **spaCy Projects** — official documentation (<https://spacy.io/usage/projects>). The commands /
  assets / workflows model is first-party. spaCy's per-asset checksum and DVC integration are
  the official reproducibility mechanism.
- **spaCy v3 `Language` pipeline API** — <https://spacy.io/api/language>. The component contract
  (each component is a `Doc -> Doc` callable; disables and factory names are public API).
- **Hugging Face `transformers.pipeline`** — <https://huggingface.co/docs/transformers/main_classes/pipelines>.
  The task-string → model class resolution, the `device`, `batch_size`, and `revision` kwargs are
  documented and stable.
- **Hugging Face `datasets`** — <https://huggingface.co/docs/datasets/>. `Dataset.map` with
  `cache_file_name` is the canonical caching primitive; `fingerprint` propagation is how
  reproducibility is enforced.
- **Hugging Face Hub revisions** — <https://huggingface.co/docs/huggingface_hub/guides/download>.
  `revision=<sha>` pins the commit; the name alone pins only to `main`.
- **Model Cards** (Mitchell et al. 2019) and **Datasheets** (Gebru et al. 2021) — the documentation
  surface the pipeline must carry forward from each component.
- **OWASP ML Security Top 10** — <https://owasp.org/www-project-machine-learning-security-top-10/>.
  *ML06 AI Supply Chain Attacks* is why revision-pinning is a security control, not a nicety.
- **MITRE ATLAS** — <https://atlas.mitre.org/>. *AML.T0010 ML Supply Chain Compromise*
  documents the attacker objective the pinning defends against.
- **NIST AI RMF** — <https://www.nist.gov/itl/ai-risk-management-framework>. The pipeline project
  is the *Map* + *Measure* artefact — what the system does, and how each stage is measured.

## 2. Worked implementation

The worked example uses the **Hugging Face stack** (HF `pipeline` + `datasets` + `huggingface_hub`)
because it covers the widest range of modern task families and uses revision-pinning explicitly;
section 2.7 shows the equivalent **spaCy Projects** config so a reviewer can see the parallel
structure. Pick one for an assignment; a learner does not need to implement both.

### 2.1 Project layout

```
email_triage/
  config.yaml              # pipeline config; model revisions pinned
  pyproject.toml           # pinned deps including transformers, datasets, huggingface_hub
  src/email_triage/
    __init__.py
    schemas.py             # Pydantic models for every stage boundary
    stages/
      __init__.py
      s01_ingest.py        # raw emails -> normalised records
      s02_tokenise.py      # text -> token offsets + language id
      s03_ner.py           # NER via transformers pipeline(task="ner")
      s04_linker.py        # entity -> canonical id via fuzzy match
      s05_intent.py        # intent classification via transformers pipeline(task="text-classification")
      s06_emit.py          # structured output + manifest
    run.py                 # CLI: python -m email_triage run --input <dir> --output <dir>
    manifest.py            # per-run run manifest (sha, timings, counts)
  tests/
    fixtures/
      emails_golden.jsonl  # 10 input records
      emails_golden.expected.jsonl  # expected pipeline output
    test_schemas.py        # each stage boundary parses/round-trips
    test_stages.py         # per-stage unit test on a 1-item sample
    test_end_to_end.py     # full DAG on fixtures; compared to golden
  .github/workflows/ci.yml
  README.md
```

### 2.2 `config.yaml` — every model artefact pinned by revision

```yaml
# config.yaml
project:
  name: email_triage
  version: 0.1.0

inputs:
  emails_dir: ./data/inbox/

outputs:
  structured_dir: ./data/structured/
  manifest_path:  ./data/structured/run_manifest.json

models:
  ner:
    task: "ner"
    repo: "dslim/bert-base-NER"
    revision: "6e8ace3c2a9c3d3f6a4c8e8e3a2b9c3d4e5f6a7b"   # placeholder: real sha from the model card
    aggregation_strategy: "simple"
  intent:
    task: "text-classification"
    repo: "SamLowe/roberta-base-go_emotions"
    revision: "1e8d46ea4fe6d3f5e8a7b2c3d4e5f6a7b8c9d0e1"   # placeholder
    top_k: 3
  tokenizer_fast: true

runtime:
  device: "cpu"              # "cuda:0" when the runner has a GPU; validated at boot
  batch_size: 16
  max_workers: 4
  cache_dir: "./.hf_cache"
```

Points worth defending in review:

- **`revision` is mandatory.** Running `pipeline("ner", model="dslim/bert-base-NER")` without
  `revision=` resolves to `main` and silently updates when the upstream repo moves. The HF
  Hub docs are explicit that a name alone is not a pin.
- **`aggregation_strategy="simple"`** on NER collapses `B-ORG`, `I-ORG` into an `ORG` span with
  character offsets; without it the downstream linker must re-assemble spans and the stage
  boundary leaks tokenizer internals.
- **`device` is validated at boot, not at request time.** A pipeline that discovers at the first
  request that `cuda:0` is unavailable will fall back to CPU and your p95 latency explodes
  without alerting.

### 2.3 Stage boundaries — typed with Pydantic

```python
# src/email_triage/schemas.py
from pydantic import BaseModel, Field


class Email(BaseModel):
    message_id: str
    sent_at: str              # ISO-8601
    subject: str
    body: str
    language: str | None = None


class TokenisedEmail(BaseModel):
    message_id: str
    text: str                 # concatenated subject + body for NER
    language: str


class EntitySpan(BaseModel):
    start: int                # char offset in `text`
    end: int
    label: str                # ORG | PER | LOC | MISC (model-dependent)
    text: str
    score: float = Field(ge=0.0, le=1.0)


class LinkedEntity(EntitySpan):
    canonical_id: str | None  # None when linker found no match above threshold
    link_score: float = Field(ge=0.0, le=1.0)


class IntentPrediction(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)


class TriagedEmail(BaseModel):
    message_id: str
    language: str
    entities: list[LinkedEntity]
    intent_top_k: list[IntentPrediction]
    model_revisions: dict[str, str]    # resolved shas; audit field
```

Every stage consumes the previous stage's model and emits the next. A schema mismatch raises at
the boundary, not two stages later with a `KeyError`.

### 2.4 Stages — each one is a pure function + an HF pipeline handle

```python
# src/email_triage/stages/s03_ner.py
from __future__ import annotations
from functools import lru_cache
from transformers import pipeline

from ..schemas import TokenisedEmail, EntitySpan


@lru_cache(maxsize=1)
def _ner_pipeline(repo: str, revision: str, device: str):
    # revision= pins the exact commit; cache the resolved pipeline across calls.
    return pipeline(
        task="ner",
        model=repo,
        revision=revision,
        aggregation_strategy="simple",
        device=device,
    )


def run(tokenised: TokenisedEmail, *, repo: str, revision: str, device: str) -> list[EntitySpan]:
    ner = _ner_pipeline(repo, revision, device)
    raw = ner(tokenised.text)
    return [
        EntitySpan(
            start=int(ent["start"]),
            end=int(ent["end"]),
            label=ent["entity_group"],
            text=ent["word"],
            score=float(ent["score"]),
        )
        for ent in raw
    ]
```

Repeat the pattern for `s05_intent.py`. The `@lru_cache` keeps the pipeline warm across the
batch so the model is loaded once per worker.

### 2.5 Driver — DAG runner with manifest

```python
# src/email_triage/run.py
from __future__ import annotations
import argparse, json, pathlib, time, hashlib, yaml
from huggingface_hub import model_info

from .schemas import Email, TokenisedEmail, TriagedEmail, LinkedEntity, IntentPrediction
from .stages import s01_ingest, s02_tokenise, s03_ner, s04_linker, s05_intent, s06_emit
from .manifest import RunManifest


def _resolve_sha(repo: str, revision: str) -> str:
    # model_info(revision=<sha>) returns that sha; the call is a verification + presence check.
    info = model_info(repo, revision=revision)
    return info.sha


def run_pipeline(config_path: pathlib.Path) -> RunManifest:
    cfg = yaml.safe_load(config_path.read_text())
    ner_cfg = cfg["models"]["ner"]
    intent_cfg = cfg["models"]["intent"]
    device = cfg["runtime"]["device"]

    manifest = RunManifest(
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config_sha=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        model_revisions={
            "ner": _resolve_sha(ner_cfg["repo"], ner_cfg["revision"]),
            "intent": _resolve_sha(intent_cfg["repo"], intent_cfg["revision"]),
        },
    )

    inputs_dir = pathlib.Path(cfg["inputs"]["emails_dir"])
    outputs_dir = pathlib.Path(cfg["outputs"]["structured_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for email in s01_ingest.iter_emails(inputs_dir):
        t0 = time.perf_counter()
        tokenised: TokenisedEmail = s02_tokenise.run(email)
        entities = s03_ner.run(
            tokenised,
            repo=ner_cfg["repo"],
            revision=ner_cfg["revision"],
            device=device,
        )
        linked: list[LinkedEntity] = [s04_linker.run(e) for e in entities]
        intent: list[IntentPrediction] = s05_intent.run(
            tokenised,
            repo=intent_cfg["repo"],
            revision=intent_cfg["revision"],
            device=device,
            top_k=intent_cfg["top_k"],
        )
        triaged = TriagedEmail(
            message_id=email.message_id,
            language=tokenised.language,
            entities=linked,
            intent_top_k=intent,
            model_revisions=manifest.model_revisions,
        )
        s06_emit.write(triaged, outputs_dir)
        manifest.record_item(message_id=email.message_id, elapsed_s=time.perf_counter() - t0)

    manifest.finalise(pathlib.Path(cfg["outputs"]["manifest_path"]))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(prog="email_triage")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"))
    args = parser.parse_args()
    run_pipeline(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 2.6 Manifest — the audit artefact

```python
# src/email_triage/manifest.py
from __future__ import annotations
import json, pathlib, statistics
from dataclasses import dataclass, field


@dataclass
class RunManifest:
    started_at: str
    config_sha: str
    model_revisions: dict[str, str]
    item_timings_s: list[float] = field(default_factory=list)
    item_count: int = 0

    def record_item(self, message_id: str, elapsed_s: float) -> None:
        self.item_timings_s.append(elapsed_s)
        self.item_count += 1

    def finalise(self, out_path: pathlib.Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        timings = self.item_timings_s or [0.0]
        payload = {
            "started_at": self.started_at,
            "config_sha": self.config_sha,
            "model_revisions": self.model_revisions,
            "item_count": self.item_count,
            "per_item_latency_s": {
                "p50": statistics.median(timings),
                "p95": statistics.quantiles(timings, n=20)[-1] if len(timings) >= 20 else max(timings),
                "max": max(timings),
            },
        }
        out_path.write_text(json.dumps(payload, indent=2))
```

The manifest is the artefact a downstream reviewer reads. It answers two questions
authoritatively: *which model weights produced these outputs* and *was the run healthy*.

### 2.7 spaCy Projects equivalent

```yaml
# project.yml
title: "email_triage"
description: "Pipeline composition exercise; see mod-112 exercise-01 SOLUTION.md."
vars:
  ner_model: "en_core_web_trf"
  spacy_version: "3.7.4"
assets:
  - dest: "assets/emails.jsonl"
    url:  "s3://<org>-nlp-assets/email_triage/emails-v1.jsonl"
    checksum: "<sha256>"
workflows:
  end-to-end:
    - ingest
    - ner
    - link
    - intent
    - emit
commands:
  - name: "ingest"
    help: "Normalise raw emails into the pipeline-input schema."
    script:
      - "python -m email_triage.stages.s01_ingest --input assets/emails.jsonl --output corpus/normalised.jsonl"
    deps: ["assets/emails.jsonl"]
    outputs: ["corpus/normalised.jsonl"]
  - name: "ner"
    help: "Run spaCy NER."
    script:
      - "python -m email_triage.stages.s03_ner_spacy --input corpus/normalised.jsonl --output corpus/ner.jsonl --model ${vars.ner_model}"
    deps: ["corpus/normalised.jsonl"]
    outputs: ["corpus/ner.jsonl"]
  # link, intent, emit analogous
```

spaCy Projects enforces the DAG dependency structure natively: a `commands` entry declares `deps`
and `outputs`; `spacy project run end-to-end` only re-runs commands whose `deps` changed. Asset
`checksum:` fields are the equivalent of HF `revision=`. The equivalent of the HF manifest is
spaCy's per-command metadata emitted by `spacy project document`.

### 2.8 Tests — one golden fixture + schema checks

```python
# tests/test_end_to_end.py
import json, pathlib
from email_triage.run import run_pipeline


def test_end_to_end_matches_golden(tmp_path: pathlib.Path) -> None:
    fixture_input = pathlib.Path("tests/fixtures/emails_golden.jsonl")
    fixture_expected = pathlib.Path("tests/fixtures/emails_golden.expected.jsonl")

    # Clone the config, point IO at tmp_path so the run is hermetic.
    cfg = (pathlib.Path("config.yaml").read_text()
           .replace("./data/inbox/", str(fixture_input.parent) + "/inbox/")
           .replace("./data/structured/", str(tmp_path) + "/out/"))
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg)

    run_pipeline(cfg_path)

    got = sorted(json.loads(line) for line in (tmp_path / "out" / "triaged.jsonl").read_text().splitlines())
    want = sorted(json.loads(line) for line in fixture_expected.read_text().splitlines())
    assert got == want, "pipeline output drifted from golden fixture; investigate stage emitting the diff"
```

The golden-fixture test is the single most valuable test a pipeline project has: a change to any
stage that affects output is caught in CI, and a git diff on `emails_golden.expected.jsonl` is
the *review artefact* for the model-version bump.

## 3. Validation steps

1. **Fresh-clone smoke test.** `git clone`, `python -m pip install -e .`, `python -m email_triage run`
   exits 0 on the fixture emails. If this fails, the project has an undocumented dependency
   (environment variable, cached asset, HOME-directory state) and must be fixed before merge.
2. **Revision pinning verified.** Remove internet access, warm the HF cache with the pinned
   revisions, run the pipeline. Pass. Then change one `revision:` in `config.yaml`, re-run offline
   with no new cache — the run must fail with a clear "revision not cached" error, *not* silently
   pull `main`.
3. **Schema validation at boundaries.** Add a stage that returns a dict with an extra key; the
   downstream stage's Pydantic parse must raise `ValidationError`. Add a stage that returns a
   dict missing a required key; same requirement.
4. **Golden fixture regression.** Change one word in `emails_golden.jsonl`; the test should fail
   with a diff pointing to the *first* stage whose output changed. This is the test that
   amortises review cost over the life of the pipeline.
5. **Deterministic re-run.** Running the pipeline twice on the same inputs with the same config
   must produce byte-identical outputs (or, for transformer models, the same outputs up to
   floating-point reproducibility — pin `torch.use_deterministic_algorithms(True)` and
   `PYTHONHASHSEED`). If outputs vary, the pipeline is not shippable as a reference artefact.
6. **Manifest completeness.** Every run writes `run_manifest.json` with resolved model shas,
   config sha, per-item timing distribution, and item count. A manifest missing any of those
   fails review.
7. **Stage isolation.** Running `python -m email_triage.stages.s03_ner --input corpus/normalised.jsonl
   --output corpus/ner.jsonl` succeeds standalone. If a stage only runs as part of `run.py`, it
   is not independently cacheable and the DAG is a shell.
8. **`spacy project document`** (spaCy variant) or `datasets --list-cache-files` (HF variant)
   lists exactly the artefacts the config declares. Extra outputs or missing outputs indicate
   drift between the config and the code.
9. **CI passes end-to-end on a cold cache.** Matrix the CI across `python==3.11,3.12` and
   `transformers==<pinned>` only. Failures on unpinned transitive deps are a dependency-pin
   gap, fixed in `pyproject.toml`.
10. **Model-artefact provenance.** `run_manifest.json` contains the Hub revision sha for every
    model; `huggingface_hub.model_info(repo, revision=<sha>)` must resolve it (verifies the
    revision still exists on the Hub; a deleted revision is a release-blocker).

## 4. Grading rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Revision pinning** | Every model in `config.yaml` carries `revision=<sha>`; `_resolve_sha()` called at boot and recorded in manifest | Revision pinned; sha not logged | No `revision=`; `pipeline(...)` resolves to `main` |
| **Stage schemas** | Pydantic (or equivalent) model at every stage boundary; schema test per boundary | Dataclasses, no validation | Raw dicts |
| **DAG / independent stages** | Each stage is runnable standalone with its own CLI; spaCy Projects `deps`/`outputs` or HF `datasets.map(cache_file_name=…)` so re-runs are incremental | Monolithic driver that calls stages in sequence but stages are importable | One file, all logic |
| **Config** | `config.yaml` or `project.yml` lists assets, commands, revisions, and runtime knobs; nothing hard-coded in code | Config present; some constants in code | No config |
| **Manifest** | Per-run JSON with resolved shas, config sha, per-item latency distribution, item count | JSON present, missing one field | No manifest |
| **Tests** | Golden-fixture end-to-end + per-stage schema test + smoke `--help` test | End-to-end only | None |
| **CI** | `.github/workflows/ci.yml` runs the end-to-end on a 10-doc fixture; matrix over pinned Python and `transformers` | CI exists but skips the end-to-end | No CI |
| **Device handling** | Device validated at boot; CPU fallback explicit and logged | Device read from config but not validated | `cuda:0` assumed |
| **Supply-chain controls** | OWASP ML06 mitigation: revision sha pinned, model card linked in README, `huggingface_hub` offline-mode supported | Revision pinned, no model-card link | No control |
| **Documentation** | README has run command, expected output, and a stage diagram; each stage has a docstring | README present but incomplete | Neither |

## 5. Common mistakes

- **Loading the pipeline inside the per-item loop.** `pipeline("ner", ...)` inside `for email in …`
  reloads the model from disk every item. Instance-level `@lru_cache` (or module-level
  initialisation) is the fix; the symptom is 50× too-slow throughput.
- **Mixing `pipeline` kwargs with model-side kwargs.** `pipeline("ner", model=..., max_length=512)`
  — `max_length` is a tokenizer kwarg and silently has no effect unless the tokenizer honours it
  on the pipeline's internal call. Set tokenizer/model kwargs on the explicit objects, pass them
  into `pipeline`, and test with a long input.
- **Not setting `aggregation_strategy`.** Default NER emits BIO-tagged subword predictions; a
  linker that expects entity spans will mis-align offsets. Pick `"simple"` or `"first"` and lock
  it in the config.
- **Pinning only `transformers` version, not the model revision.** Pinning the library catches
  library breakage; pinning the model revision catches *model* breakage. Both matter; pin both.
- **Running `datasets.map` without `cache_file_name`.** HF `datasets` caches on a fingerprint
  that hashes the function source; a trivial edit invalidates the cache and the pipeline
  re-encodes 10 GB of text. Supply `cache_file_name` for every expensive map.
- **Calling spaCy `nlp.add_pipe("ner", source=other_nlp)` without `before=` / `after=`.** Order
  determines which component sees which annotations; silent order change is a common regression.
  Pin the component order in the config and test it.
- **Using `device_map="auto"` for a one-model pipeline.** Accelerate's `device_map` is designed
  for large models that need sharding; on single-GPU, single-model pipelines it adds overhead
  and complicates reproducibility. Use `device=0` or `device="cuda:0"` explicitly.
- **Not pinning `torch` CUDA/CPU wheels.** A deploy that installs `torch==2.X.Y` can land either
  CPU or GPU wheels depending on the environment; p95 latency silently changes by 10×. Pin the
  wheel index (`--extra-index-url https://download.pytorch.org/whl/cu121`) in the lockfile.
- **Streaming large datasets with `datasets.load_dataset(..., streaming=True)` but caching
  downstream.** Streaming skips the arrow cache; subsequent runs re-download. Pick streaming
  *or* caching depending on whether the input is larger than disk.
- **Shipping with the HF cache baked into the container image.** Image size blows up; cache
  invalidation becomes a container rebuild. Mount the cache volume instead; or build a
  minimal image that pulls the pinned revisions at boot (with a connectivity test).
- **Treating the manifest as optional.** The manifest is the only artefact a downstream auditor
  reads. "The numbers moved" with no manifest means the pipeline is not shippable.

## 6. References

Primary sources used in this solution:

- spaCy Projects documentation. <https://spacy.io/usage/projects> — official reference for
  `project.yml`, asset checksums, commands, workflows, and DVC integration.
- spaCy `Language` pipeline API. <https://spacy.io/api/language> — component contract,
  `add_pipe`, `disable`, factory names.
- Hugging Face `transformers.pipeline` documentation. <https://huggingface.co/docs/transformers/main_classes/pipelines>
  — task strings, `revision=`, `device=`, `batch_size=`, `aggregation_strategy=` kwargs.
- Hugging Face `datasets` documentation. <https://huggingface.co/docs/datasets/> — `Dataset.map`,
  `cache_file_name`, `fingerprint` propagation.
- Hugging Face Hub download/revision reference. <https://huggingface.co/docs/huggingface_hub/guides/download>
  — pinning by `revision=<commit_sha>`.
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAccT. <https://arxiv.org/abs/1810.03993>
  — the documentation surface each model component must carry.
- Gebru, T. et al. (2021). *Datasheets for Datasets*. CACM. <https://arxiv.org/abs/1803.09010>
  — the equivalent for datasets the pipeline consumes.
- Pydantic v2 documentation. <https://docs.pydantic.dev/latest/> — the schema-validation primitive
  used for stage boundaries.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/>
  — *ML06 AI Supply Chain Attacks* applies directly: unpinned `transformers` revisions are a
  supply-chain exposure (a compromised upstream model repo can ship new weights on `main` with
  no version bump). Pinning `revision=<sha>` and recording the resolved sha in the run manifest
  is the mitigation. *ML07 Transfer Learning Attack* applies when a pre-trained checkpoint was
  adversarially fine-tuned to leak training data or ship a backdoor; the model-card provenance
  record in the manifest is the minimum documentation for that risk.
- MITRE ATLAS — <https://atlas.mitre.org/> — *AML.T0010 ML Supply Chain Compromise* and
  *AML.T0019 Publish Poisoned Datasets* document the attacker objectives the pin + manifest
  defend against. *AML.T0017 Develop Capabilities: Craft Adversarial Data* applies to the input
  side; validation at stage boundaries is a partial mitigation (blatantly malformed inputs do
  not reach the model).
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework>
  — *Map*: the pipeline DAG + per-stage schema is the map of what the system does. *Measure*:
  the manifest's per-item latency distribution and item count is the measure artefact per run.
  *Manage*: `revision=<sha>` pins and the golden-fixture CI are the manage controls against
  silent drift. *Govern*: `config.yaml` as the single policy surface (every reviewable knob
  lives there) is the governance artefact.
