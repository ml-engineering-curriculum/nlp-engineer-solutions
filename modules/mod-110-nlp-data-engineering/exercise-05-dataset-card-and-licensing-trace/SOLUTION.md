# exercise-05 — Dataset Card and Licensing Trace

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-110-nlp-data-engineering/exercises/exercise-05-dataset-card-and-licensing-trace.md`.
> The target learning outcomes implied by that filename: author the two artefacts a dataset needs *before* it ships — a **dataset card** (Datasheets-for-Datasets / HuggingFace Dataset Card template, with the fields NIST AI RMF names explicitly) and a **licensing trace** that tracks every upstream source's licence through every transform to a *derivative compatibility matrix* that answers "may this derived dataset be used for X" without a lawyer having to re-derive the chain each time. Both artefacts are versioned, reproducible from the dataset's own manifest, and referenced by name from the model card of anything trained on the dataset.

## 1. Solution overview

A dataset is only as useful as the questions it can answer *before* someone downloads it. The two failure modes that catch teams late:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **No dataset card** | A future user cannot tell: what the dataset covers, how it was collected, what the known risks are, what the licence allows. They either reinvent your decisions or ship assumptions. Gebru et al. 2018 is the seminal case against the "data as exhaust" mindset | Ship a **Datasheets-for-Datasets**-style card in-repo at `DATASHEET.md` with *every* field filled or explicitly marked `N/A — reason`. The HuggingFace Dataset Card template is the pragmatic default because downstream tooling expects it |
| **No licence trace** | Public corpora are built from dozens of upstream sources with heterogeneous licences (CC-BY, CC-BY-SA, MIT, Apache, proprietary-with-exceptions, no-licence-stated). A downstream user asking "can I commercialise a model fine-tuned on this" gets no answer; the risk is quietly offloaded to them | Every upstream source has a row in a **licensing manifest** (`LICENSES.yml`): identifier, URL, licence SPDX ID, attribution requirement, share-alike flag, commercial-use flag. Derive a **compatibility matrix** for intended downstream uses; mechanise so a lint check blocks adding incompatible sources |
| **Non-reproducible card numbers** | The card says "92 % English, 500M tokens, 0.73 κ"; the figures were measured six corpus-revs ago and no one knows which | Card metrics are **derived from the pipeline's manifest** (exercise-01's manifest, exercise-02's decision log, exercise-03's audit log). Rebuilds of the card are a single script invocation |
| **Silent copy of a non-redistributable sub-corpus** | A permissive licence on your corpus does not grant permissive redistribution of a non-commercial sub-corpus you included; downstream licensees believe they received more rights than they did | Publish per-sub-corpus licences **separately**; downstream tools read `LICENSES.yml` and surface the most restrictive upstream term; redistributions include the attribution file and the chain |

The deliverable is a four-component harness:

1. **`DATASHEET.md`** — the Datasheets-for-Datasets template (Gebru et al. 2018) with the seven standard sections filled from manifests.
2. **`LICENSES.yml`** — per-upstream-source licensing manifest (SPDX IDs, attribution requirements, share-alike / NC / ND flags).
3. **Compatibility matrix** — `COMPATIBILITY.md` (generated) listing intended downstream uses × allow/deny with rationale.
4. **Licence lint** — a CI check that reads `LICENSES.yml` and the manifest, and fails on known-incompatible combinations (e.g., CC-BY-NC in a corpus advertised as commercial-use).

Reference templates and tools:

- **Datasheets for Datasets** (Gebru et al. 2018, extended 2021) — <https://arxiv.org/abs/1803.09010> — the canonical 7-section questionnaire; the fields this solution fills.
- **Data Statements for NLP** (Bender & Friedman 2018) — <https://aclanthology.org/Q18-1041/> — NLP-specific template focused on annotator demographics, language variety, and curation rationale. Complementary to Datasheets.
- **HuggingFace Dataset Card template** — <https://github.com/huggingface/datasets/blob/main/templates/README_guide.md> — the pragmatic default; the Hub auto-renders required YAML header fields (`task_categories`, `language`, `license`, `size_categories`).
- **Croissant metadata** (MLCommons 2024) — <https://docs.mlcommons.org/croissant/> — the machine-readable schema that HuggingFace, Kaggle, and OpenML now consume; a card without it is harder to catalog.
- **SPDX** — <https://spdx.org/licenses/> — the canonical identifier list; use the identifiers verbatim (`CC-BY-SA-4.0`, not "CC Share-Alike 4.0").
- **ML Commons CDLA** — the Linux Foundation Community Data License Agreement (`CDLA-Permissive-2.0`, `CDLA-Sharing-1.0`); the licence most new permissive ML datasets use because it's more precise about model-training.
- **The Pile** (Gao et al. 2020) — <https://arxiv.org/abs/2101.00027> — the reference example of per-sub-corpus licence trace; §A.1 has the per-component table.

## 2. Worked implementation

### 2.1 Project layout

```
dataset_bake/
  DATASHEET.md                   # the Datasheets-for-Datasets card (generated + hand-edited)
  LICENSES.yml                   # per-upstream manifest (authoritative)
  COMPATIBILITY.md               # derived from LICENSES.yml
  NOTICES.txt                    # attribution file (CC-BY requires)
  tools/
    render_datasheet.py          # fill derived fields from manifest
    compat_matrix.py             # LICENSES.yml → COMPATIBILITY.md
    lint_licenses.py             # CI check
  tests/
    test_license_parse.py
    test_compat_matrix.py
    test_notices_coverage.py
  requirements.txt
```

### 2.2 `LICENSES.yml` — the authoritative licensing manifest

```yaml
# LICENSES.yml — one entry per upstream sub-corpus / source URL cluster
# All SPDX identifiers from https://spdx.org/licenses/. "licence_text_sha256"
# pins the exact wording we received to defend against upstream edits.

version: 1.2.0
dataset_name: internal-rag-corpus
compiled_at: "2026-07-14"
sources:
  - id: wikipedia-en-20260401
    url: "https://dumps.wikimedia.org/enwiki/20260401/"
    description: "English Wikipedia article text dump 2026-04-01"
    licence_spdx: "CC-BY-SA-4.0"
    attribution_required: true
    share_alike: true
    noncommercial: false
    no_derivatives: false
    licence_text_sha256: "0d8f...e3"
    attribution_template: "Wikipedia contributors, '{title}', CC-BY-SA 4.0, retrieved 2026-04-01"
    inclusion_rationale: "coverage of general-knowledge entities"
    notes: "Share-alike applies to the dataset as a whole if we include derivatives."

  - id: pubmed-abstracts-2026q2
    url: "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
    description: "PubMed abstracts (2026 baseline)"
    licence_spdx: "LicenseRef-NCBI-Public-Domain"
    attribution_required: false
    share_alike: false
    noncommercial: false
    no_derivatives: false
    licence_text_sha256: "7a21...fc"
    inclusion_rationale: "biomedical domain coverage"
    notes: "NCBI terms say 'public domain in the US'; retained for international audit."

  - id: stackoverflow-q-cc-by-sa-2024
    url: "https://archive.org/details/stackexchange-20240401"
    description: "Stack Overflow Q/A dump 2024-04-01"
    licence_spdx: "CC-BY-SA-4.0"
    attribution_required: true
    share_alike: true
    noncommercial: false
    no_derivatives: false
    licence_text_sha256: "4b13...9c"
    attribution_template: "Stack Overflow users, CC-BY-SA 4.0, archive 2024-04-01, question id: {qid}"
    inclusion_rationale: "programming Q/A domain coverage"
    notes: "License changed to CC-BY-SA-4.0 on 2018-05-02; pre-2018 content is CC-BY-SA-3.0 (compatible for inclusion, but attribution templates differ)."

  - id: reuters-trc2-2009
    url: "proprietary"
    description: "Reuters TRC2"
    licence_spdx: "LicenseRef-Reuters-TRC2"
    attribution_required: true
    share_alike: false
    noncommercial: true
    no_derivatives: false
    licence_text_sha256: "c0ff...1a"
    inclusion_rationale: "domain coverage - financial news"
    notes: "NON-COMMERCIAL. Taints the entire derivative for commercial use; see COMPATIBILITY.md."

  - id: common-crawl-2026-07-sample
    url: "https://data.commoncrawl.org/CC-MAIN-2026-27/"
    description: "Sample from Common Crawl 2026-27 dump"
    licence_spdx: "LicenseRef-CommonCrawl-ToU"
    attribution_required: false
    share_alike: false
    noncommercial: false
    no_derivatives: false
    licence_text_sha256: "9a5e...04"
    inclusion_rationale: "open-web coverage"
    notes: "Common Crawl ToU grants redistribution but does not clear third-party rights in the crawled content; downstream training is at user's own risk. See notes block below."
```

Points worth defending in review:

- **SPDX identifiers, verbatim.** `CC-BY-SA-4.0` is machine-comparable; "Creative Commons Attribution-ShareAlike 4.0 International" is not. The SPDX list is the canonical source; `LicenseRef-...` is the SPDX pattern for licences not on the list (NCBI, proprietary, platform ToUs).
- **`licence_text_sha256`.** The upstream page you viewed may be edited later; the hash pins what *you* accepted. Essential for audit when an upstream silently changes terms.
- **`inclusion_rationale`.** One sentence per source, explicit. If the auditor asks "why is this here", the answer is in the file; if the answer is weak ("volume"), that's a review signal.
- **Common Crawl's own terms do not grant third-party rights.** The ToU permits redistribution of the *crawl*; the crawled content retains whatever licence it had. Serious deployments either (a) restrict to CC-flagged sub-domains or (b) accept the risk in writing; either way the note must be in the manifest.
- **NC flags "taint" the derivative.** If any source is non-commercial, the whole derived dataset is non-commercial *unless* that source is excludable (i.e., keep it as a separate sub-corpus). This is the reason for the sub-corpus granularity.

### 2.3 Compatibility matrix — derived, not hand-authored

```python
# tools/compat_matrix.py
import yaml
from pathlib import Path


INTENDED_USES = [
    ("research", "Non-commercial academic research"),
    ("internal_eval", "Internal evaluation inside one organisation"),
    ("commercial_model_training", "Training models for commercial product use"),
    ("redistribution_derivative", "Redistribute a derivative dataset (same or different licence)"),
    ("llm_output_redistribution", "Redistribute LLM-generated outputs of models trained on this"),
]


def classify_use(sources: list[dict], use: str) -> tuple[str, list[str]]:
    """
    Returns (verdict, reasons) where verdict ∈ {allow, allow_with_attribution,
    allow_with_share_alike, deny, requires_legal_review}.
    """
    reasons: list[str] = []
    any_nc = any(s["noncommercial"] for s in sources)
    any_nd = any(s["no_derivatives"] for s in sources)
    any_sa = any(s["share_alike"] for s in sources)
    any_attr = any(s["attribution_required"] for s in sources)
    uses_cc_toU = any(s["licence_spdx"].startswith("LicenseRef-CommonCrawl") for s in sources)
    uses_proprietary = any(s["licence_spdx"].startswith("LicenseRef-Reuters") for s in sources)

    if use == "commercial_model_training" and any_nc:
        nc = [s["id"] for s in sources if s["noncommercial"]]
        reasons.append(f"non-commercial source(s): {', '.join(nc)}")
        return ("deny", reasons)

    if use == "redistribution_derivative":
        if any_nd:
            nd = [s["id"] for s in sources if s["no_derivatives"]]
            reasons.append(f"no-derivatives source(s): {', '.join(nd)}")
            return ("deny", reasons)
        if any_sa:
            reasons.append("share-alike propagates; derivative must be CC-BY-SA-4.0 or compatible")
            return ("allow_with_share_alike", reasons)
        if any_attr:
            return ("allow_with_attribution", reasons)
        return ("allow", reasons)

    if use == "llm_output_redistribution":
        # Not purely a licensing question; see notes below.
        reasons.append("legally unsettled area (model outputs vs training set rights)")
        return ("requires_legal_review", reasons)

    if uses_proprietary:
        reasons.append("proprietary licence present; carve-outs may apply")
        return ("requires_legal_review", reasons)

    if uses_cc_toU:
        reasons.append("Common Crawl ToU grants crawl redistribution but not third-party content rights")
        return ("requires_legal_review", reasons)

    if any_attr:
        return ("allow_with_attribution", reasons)
    return ("allow", reasons)


def render(path_in: Path, path_out: Path) -> None:
    data = yaml.safe_load(path_in.read_text())
    rows = ["| Intended use | Verdict | Reasons |", "|---|---|---|"]
    for key, desc in INTENDED_USES:
        verdict, reasons = classify_use(data["sources"], key)
        reasons_str = "; ".join(reasons) or "no restrictions found"
        rows.append(f"| {desc} | **{verdict}** | {reasons_str} |")
    path_out.write_text("# Compatibility matrix\n\n" + "\n".join(rows) + "\n")
```

Points reviewers must check:

- **The matrix is a *derivation*, not a human judgement.** This is the only tractable way to keep it in sync with sources that come and go between dataset revisions.
- **`requires_legal_review` is an honest verdict.** The common-crawl-third-party-rights question and the LLM-output-redistribution question are legally unsettled (as of late 2026); claiming "allow" on either is substantive misrepresentation. Surface the question, do not auto-answer it.
- **Deny for `commercial_model_training` when any NC source is present.** This is the strongest hard rule; the only escape is to redistribute two sub-corpora (one commercial, one not) with separate licences.
- **Share-alike propagation.** CC-BY-SA requires the *derivative* be licensed under compatible terms. The matrix flags this; the datasheet reports it explicitly in §4.

### 2.4 `DATASHEET.md` — Datasheets-for-Datasets, every field filled

```markdown
# DATASHEET: internal-rag-corpus

> Generated by `tools/render_datasheet.py` on 2026-07-14 from `LICENSES.yml@1.2.0`
> and `manifest.jsonl.gz@c0ff…ab`. Edit the `# Motivation`, `# Composition`,
> and `# Uses` sections by hand; all numeric fields are re-derived on each build.

## Motivation

- **For what purpose was the dataset created?**
  To train a domain-specific retrieval model for internal RAG, covering
  general-knowledge, biomedical, financial news, and programming domains.
- **Who created the dataset and on behalf of whom?**
  [team name] on behalf of [org].
- **Who funded the creation of the dataset?**
  Internal R&D funding; no external sponsors.

## Composition

- **What do the instances represent?**
  Each instance is a cleaned document (paragraph-dedup'd, boilerplate-stripped)
  with a canonical URL, source ID, and language code.
- **How many instances are there?**
  *Derived from manifest*: 12 471 338 documents, 4.8 B whitespace tokens.
- **What is the language variety / geographic coverage / time range?**
  English (en-US, en-GB predominant; en-IN, en-NG represented but <5 % each);
  sources from 2009–2026-04.
- **Is the dataset a sample from a larger set?**
  Yes; 2.3 % sample of Common Crawl's 2026-27 dump (random by domain hash),
  full inclusion of Wikipedia/PubMed/Stack Overflow/Reuters TRC2 within the
  listed snapshots.
- **Does the dataset contain data that might be considered confidential or sensitive?**
  *Derived from exercise-02 decision log*: 4 213 827 PII detections, of which
  99.7 % were replaced with `[TYPE_N]` placeholders; 12 831 (0.3 %) were held
  back for manual review and replaced on 2026-07-10.
- **Does the dataset contain data that might be offensive, insulting, threatening,
  or might otherwise cause anxiety?**
  *Derived from exercise-02 toxicity log*: 1.8 % of paragraphs scored above the
  `toxicity` head threshold of 0.75 and were dropped from the corpus. The
  identity-attack head threshold was 0.50; subgroup AUC was reported between
  0.84 (religion) and 0.91 (gender). See `TOXICITY_FAIRNESS.md`.

## Collection

- **How was the data acquired?**
  Direct download from documented URIs (listed in `LICENSES.yml`).
- **What mechanisms or procedures were used to collect the data?**
  Common Crawl WARC streaming; Wikipedia dump mirror; PubMed FTP; Stack Overflow
  archive on Internet Archive; Reuters TRC2 licensed distribution.
- **If the dataset is a sample from a larger set, what was the sampling strategy?**
  Common Crawl: random by second-level domain hash.
- **Who was involved in the data collection process?**
  No external annotators at the *collection* stage. For the *labelled subset*
  (10 000 documents with gold retrieval labels), see §Annotation.

## Preprocessing / cleaning / labeling

- **Was preprocessing done?**
  Yes: (a) exercise-01 MinHash dedup at word-5-gram, Jaccard 0.8, 128 perms,
  dropped 24.3 % of inputs; (b) C4 line-level + Gopher quality filters, dropped
  8.1 %; (c) exercise-02 PII detection (Presidio + spaCy `en_core_web_lg`) and
  surrogate replacement, dropped 0.2 % of documents, replaced spans in 31 %;
  (d) exercise-02 toxicity filter (Detoxify `unbiased`), dropped 1.8 %.
- **Is the "raw" data saved?**
  Yes; raw WARCs retained in cold storage for 90 days, `manifest.jsonl` retained
  indefinitely so any document can be re-derived from raw URIs.

## Annotation

- **Was the data labeled?**
  10 000 documents in the "query → relevance" subset. See `nlp-engineer-solutions/mod-110/exercise-03`
  for guidelines, inter-annotator-agreement (IAA), and adjudication process.
- **Who were the annotators?**
  5 in-house annotators with NLP-adjacent backgrounds; demographic survey in
  `DATA_STATEMENT.md` per Bender & Friedman 2018.
- **Were annotators compensated?**
  Yes, at internal-engineer rates; not piecework.
- **What is the inter-annotator agreement?**
  Krippendorff's α = 0.82 (CI 95 %: [0.78, 0.84]) on the pinned 150-item
  calibration set; see `IAA_CURVE.png`.

## Uses

- **For what tasks has the dataset been used?**
  Bi-encoder retrieval training (exercise-01 of mod-108) and rerank cross-encoder
  training (exercise-02 of mod-108).
- **Is there a repository that links to papers / systems that use the dataset?**
  Yes; see `USAGE.md`.
- **What (other) tasks could the dataset be used for?**
  Any English retrieval or passage-ranking task in the covered domains.
- **Is there anything that could cause the dataset to be used in a way that is
  biased or otherwise harmful?**
  Yes; see `COMPATIBILITY.md` for licence constraints and `TOXICITY_FAIRNESS.md`
  for subgroup AUC gaps. Reuters TRC2's non-commercial carve-out means the
  dataset is non-commercial without excluding the Reuters sub-corpus.

## Distribution

- **Will the dataset be distributed to third parties outside the entity creating it?**
  Internal distribution only; no external redistribution.
- **How will it be distributed?**
  Internal object store with signed access; manifests included.
- **When will it be distributed?**
  Dataset `v1.2.0` published 2026-07-14.
- **Will the dataset be distributed under a copyright or other intellectual
  property license?**
  Each upstream source retains its own licence (see `LICENSES.yml`); the derived
  dataset carries the most restrictive upstream term (Reuters TRC2 → NC). The
  overall derivative is distributed internally under `LicenseRef-Internal-NC`.
- **Have any third parties imposed IP-based or other restrictions on the data?**
  Yes: Reuters TRC2 non-commercial, CC-BY-SA share-alike from Wikipedia and
  Stack Overflow.

## Maintenance

- **Who will be supporting / hosting / maintaining the dataset?**
  [team name].
- **How can the owner / curator / manager be contacted?**
  [email / team channel].
- **Is there an erratum?**
  Yes; see `ERRATA.md`.
- **Will the dataset be updated?**
  Quarterly; update triggers a new semver release (`v1.3.0` on new sub-corpus;
  `v1.2.1` on scrub pass; `v2.0.0` on breaking schema change).
- **If the dataset relates to people, are there applicable limits on retention?**
  Yes; raw crawl snapshots held 90 days, then expired. Labelled subset retained
  with hashed annotator-IDs per Data Statement for NLP guidance.
- **Will older versions continue to be supported / hosted / maintained?**
  Yes; previous 4 quarterly releases are retained.
```

Points worth defending:

- **Numeric fields are derived, not typed.** Any sentence of the form "N % of documents were …" is populated by `render_datasheet.py` from the manifests; hand-edit only the Motivation / Composition rationale fields.
- **Datasheets' question list is reproduced verbatim.** The questionnaire is the artefact; answering only the "easy" ones silently drops the harder audit questions (sensitive content, sampling, consent, retention).
- **Links to Data Statement (Bender & Friedman 2018).** Datasheets is dataset-level; Data Statement is NLP-specific and covers language variety, annotator demographics, curation rationale. Both, not either.
- **"Internal distribution only" is still a licensing event.** Even inside one org, downstream users need the compatibility matrix to decide "can I ship a product with this".

### 2.5 `NOTICES.txt` — attribution file

```text
# Attribution notices for internal-rag-corpus v1.2.0
# Generated by tools/render_notices.py from LICENSES.yml.
# CC-BY-SA-4.0 requires that this file travel with the dataset and derivatives.

== English Wikipedia (CC-BY-SA-4.0) ==
Source: https://dumps.wikimedia.org/enwiki/20260401/
Attribution: Wikipedia contributors, CC-BY-SA 4.0, retrieved 2026-04-01.
Full license text SHA-256: 0d8f...e3
Full license text file: licenses/CC-BY-SA-4.0.txt

== Stack Overflow Q/A (CC-BY-SA-4.0) ==
Source: https://archive.org/details/stackexchange-20240401
Attribution: Stack Overflow users, CC-BY-SA 4.0, archive 2024-04-01.
Full license text SHA-256: 4b13...9c

== PubMed abstracts (NCBI public domain, US) ==
Source: https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/

== Reuters TRC2 (proprietary, non-commercial) ==
Source: Reuters distribution (contact [licensor]).
Full license text SHA-256: c0ff...1a
Carve-out: this sub-corpus is non-commercial. See COMPATIBILITY.md.
```

- **CC-BY-SA requires the attribution and the licence text to travel with the dataset.** `NOTICES.txt` + a `licenses/` directory with the full text are the standard open-source practice; the SPDX identifier alone is not sufficient.
- **Generated, not hand-authored.** Every source listed in `LICENSES.yml` with `attribution_required: true` must appear. The CI test in §3 enforces coverage.

### 2.6 Licence lint — the CI check

```python
# tools/lint_licenses.py
import yaml, sys
from pathlib import Path


DECLARED_USE = "internal_eval"   # the dataset's declared/released use
ALLOWED_VERDICTS = {"allow", "allow_with_attribution", "allow_with_share_alike"}


def main(license_path: Path, declared_use: str = DECLARED_USE) -> int:
    data = yaml.safe_load(license_path.read_text())
    from .compat_matrix import classify_use
    verdict, reasons = classify_use(data["sources"], declared_use)
    if verdict not in ALLOWED_VERDICTS:
        print(f"LICENSE LINT FAIL: declared use '{declared_use}' got verdict '{verdict}'")
        for r in reasons:
            print(f"  - {r}")
        return 1
    # Attribution coverage: every attribution_required source must appear in NOTICES.txt
    notices = Path(license_path.parent, "NOTICES.txt").read_text()
    for s in data["sources"]:
        if s["attribution_required"] and s["id"] not in notices and s["description"] not in notices:
            print(f"LICENSE LINT FAIL: missing NOTICES entry for '{s['id']}'")
            return 1
    print("LICENSE LINT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path("LICENSES.yml")))
```

Why this is load-bearing:

- A reviewer or a developer who adds `pubmed-fulltext-proprietary` to the sources without noticing the `noncommercial: true` is now shipping a corpus that silently violates its declared `commercial_model_training` intent. The CI check makes that an immediate error, not a post-release surprise.
- The attribution-coverage check is the only mechanical control against "CC-BY-SA ships without `NOTICES.txt`" — the most common compliance failure in dataset releases.

## 3. Validation steps

1. **SPDX ID validity.** Every `licence_spdx` either matches the SPDX list or starts with `LicenseRef-`. Catch via `test_license_parse.py`; a typo (`CC-BY-SA` without `-4.0`) is a silent downstream break.
2. **Hash pinning.** `licence_text_sha256` matches `sha256(open(licenses/<spdx>.txt).read())`. If an upstream reissues its licence page, the hash diverges and the test fails; the file forces a review.
3. **Compatibility matrix determinism.** `compat_matrix.render` is a pure function of `LICENSES.yml`; re-run twice, assert byte-identical output.
4. **Attribution coverage.** Every `attribution_required: true` source has an entry in `NOTICES.txt`. Enforced by `lint_licenses.py`.
5. **Non-commercial taint.** Add a synthetic source with `noncommercial: true`; assert `commercial_model_training` verdict flips to `deny`.
6. **Share-alike propagation.** Add a `share_alike: true` source; assert `redistribution_derivative` verdict is `allow_with_share_alike` and the derivative-licence requirement is mentioned in the datasheet §Distribution.
7. **Common-Crawl-only sanity.** A dataset built *only* from Common Crawl receives `requires_legal_review` for every use except `internal_eval` + `research`. If the matrix says `allow` for `commercial_model_training`, the classifier has a bug (or the user has substantive legal advice they must record in the matrix overrides, not edit the code).
8. **Datasheet renderability.** `tools/render_datasheet.py` runs clean on the manifest; every `<!-- derived:... -->` placeholder in the template is substituted with a numeric figure or `N/A — reason`.
9. **Missing-source audit.** Remove one entry from `LICENSES.yml` and keep its data in the pipeline; the datasheet's "Composition / sub-corpus coverage" numbers disagree with the manifest and the drift is visible. Document the test.
10. **Croissant / HuggingFace header roundtrip.** The YAML frontmatter at the top of `DATASHEET.md` (HuggingFace format) round-trips through `huggingface_hub.DatasetCard`; `python -c "from huggingface_hub import DatasetCard; DatasetCard.load('DATASHEET.md')"` succeeds.
11. **Reproducibility of numbers.** Re-run the manifest derivation on the same manifest; assert every numeric field in the datasheet is bit-identical. Catches ordering-non-determinism in aggregation.

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Datasheet template** | Datasheets-for-Datasets (Gebru et al.) or Data Statements (Bender & Friedman) sections filled; every field answered or explicit `N/A — reason` | One template, some gaps | README-style marketing text only |
| **HuggingFace YAML header** | `task_categories`, `language`, `license`, `size_categories`, `pretty_name` set; renderable via `DatasetCard.load` | Some fields set | None |
| **Numeric fields derivation** | All counts/percentages derived from manifest; re-renderable | Some derived, some hand-typed | All hand-typed |
| **`LICENSES.yml`** | Every sub-corpus has SPDX ID + URL + attribution flag + SA flag + NC flag + ND flag + hash | Partial | Missing / "see attached" |
| **Attribution notices** | `NOTICES.txt` + licence text files; coverage lint passes | Notices present, no lint | None |
| **Compatibility matrix** | Derived table of intended uses × verdict + reasons; `requires_legal_review` used honestly | Hand-authored | None |
| **Share-alike / NC propagation** | Explicitly surfaced; derivative licence named in §Distribution | Mentioned | Ignored |
| **Croissant / metadata** | Croissant JSON-LD generated; valid against schema | HuggingFace YAML only | None |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks (poisoned attribution chain, misused CC-NC) flagged | Mentioned | Ignored |
| **NIST AI RMF alignment** | Explicit Map/Measure/Manage/Govern anchors in the datasheet | One of those | Neither |
| **Versioning** | Semver; `compiled_at`; older versions retained | Semver only | Tracking by filename |

## 5. Common mistakes

- **Writing "MIT" for the dataset licence when the data is from CC-BY-SA sources.** You cannot sub-license upstream CC-BY-SA under MIT. The derivative must be CC-BY-SA (or compatible); the *tooling* in your repo can be MIT. State both.
- **Forgetting the attribution file on redistribution.** CC-BY and CC-BY-SA require the licence text and attribution travel with the work. Shipping a corpus without `NOTICES.txt` is a technical breach even when all sources are permissive.
- **Omitting Common Crawl's third-party-rights carve-out.** Common Crawl grants redistribution of the *crawl*; the crawled content retains its own rights. A dataset card that asserts "freely redistributable because we used Common Crawl" is wrong.
- **Treating "no licence stated" as "public domain".** Code and text without a stated licence is default-copyrighted in most jurisdictions. GitHub's README is explicit on this; "no licence = all rights reserved". Flag these in the manifest as `LicenseRef-NoLicenseStated` and treat as deny-by-default.
- **Not pinning the licence text hash.** Upstream pages change; the licence version you accepted may differ from the one a future auditor finds. Hash it.
- **Letting the datasheet drift from the data.** Any card where the token-counts were measured six revisions ago is misleading. Derive the numeric fields; re-render on every release.
- **Collapsing "allow_with_share_alike" into "allow".** Downstream users then redistribute under a non-SA licence and inherit a compliance exposure you created. Keep the three verdicts distinct.
- **Claiming `commercial_model_training: allow` when any source is NC.** The hard rule is: a single NC source taints the whole derivative unless the NC source is excluded. There is no "we only used the NC source for X" carve-out; downstream can't tell.
- **Not listing failure modes in the datasheet.** The Datasheets template has explicit questions about potentially-offensive content, sampling bias, and known gaps; leaving them blank is tantamount to claiming no gaps exist.
- **Using "CC-BY" without a version.** `CC-BY-4.0` and `CC-BY-3.0` differ meaningfully (anti-DRM clauses, moral-rights waivers); SPDX always carries the version.
- **Assuming the Hugging Face auto-generated README is the dataset card.** The HF template is a *header*; it does not fill the Datasheets questionnaire. A card needs both.
- **Making the compatibility matrix human-authored.** Hand-authored matrices drift with sources; a lawyer who reviewed v1.0 cannot re-approve v1.3 without re-reviewing every verdict. Mechanise.
- **Forgetting that LLM outputs and training data have independent legal statuses.** The matrix's `llm_output_redistribution` honestly says `requires_legal_review`; the question is unsettled in 2026 (see USCO guidance and open court cases). Do not auto-answer.
- **Ignoring GDPR / privacy law separate from copyright.** A CC-BY dataset can still be unlawful to redistribute if it contains personal data that was never lawfully processed. The PII scrubbing from exercise-02 is the concrete control; the datasheet should link it explicitly.
- **Not versioning the datasheet.** The datasheet's `compiled_at` and the `LICENSES.yml` `version` should bump together; a datasheet without a version is a snapshot that cannot be referenced from a model card.

## 6. References

Primary sources used in this solution:

- Gebru, T. et al. (2018, extended 2021). *Datasheets for Datasets*. CACM. <https://arxiv.org/abs/1803.09010> — the questionnaire reproduced in §2.4; the sectioning (Motivation / Composition / Collection / Preprocessing / Annotation / Uses / Distribution / Maintenance) is canonical.
- Bender, E., Friedman, B. (2018). *Data Statements for Natural Language Processing*. TACL. <https://aclanthology.org/Q18-1041/> — the NLP-specific template focused on language variety, annotator demographics, and curation rationale; §Annotation of the datasheet links to it.
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAccT. <https://arxiv.org/abs/1810.03993> — the sibling model-card template; the dataset card is referenced from any model card of a model trained on the data.
- Pushkarna, M., Zaldivar, A., Kjartansson, O. (2022). *Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI*. FAccT. <https://arxiv.org/abs/2204.01075> — Google's framing; useful cross-reference for release-review processes.
- MLCommons Croissant (2024). *Croissant: ML-ready dataset metadata*. <https://docs.mlcommons.org/croissant/> — the machine-readable schema; HuggingFace / Kaggle / OpenML integration.
- SPDX License List — <https://spdx.org/licenses/> — the canonical identifier registry; `LicenseRef-` syntax covers proprietary / platform-ToU cases.
- Open Source Initiative — <https://opensource.org/licenses/> — the OSI-approved licence catalogue cross-referenced with SPDX.
- Creative Commons v4.0 licence family — <https://creativecommons.org/licenses/by/4.0/legalcode> — CC-BY and CC-BY-SA full text; the licence files that must travel with the dataset.
- Linux Foundation CDLA — <https://cdla.dev/> — community data licence agreements (Permissive-2.0, Sharing-1.0); the licence many new permissive ML datasets use.
- HuggingFace Dataset Card Guide — <https://huggingface.co/docs/datasets/dataset_card> — the YAML-headered template the Hub auto-renders.
- GitHub "No License" guidance — <https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository> — the default "all rights reserved" posture for absent licences.
- Gao, L. et al. (2020). *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*. <https://arxiv.org/abs/2101.00027> — Appendix A.1 is the reference per-sub-corpus licence table; the clearest example of a per-source license trace in practice.
- Longpre, S. et al. (2023). *The Data Provenance Initiative: A Large Scale Audit of Dataset Licensing & Attribution in AI*. <https://arxiv.org/abs/2310.16787> — empirical audit of licence drift across ML datasets; the case for machine-readable manifests like `LICENSES.yml`.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML06 AI Supply Chain Attack*: the dataset supply chain is a mirror of the software supply chain; `LICENSES.yml` + hash pinning + CI lint is the dataset-side SBOM-equivalent. *ML09 Output Integrity Attack*: a model trained on a licence-non-compliant dataset carries that non-compliance into its outputs; model cards must link back to the dataset card.
- MITRE ATLAS — <https://atlas.mitre.org/> — *AML.T0002 Acquire Public ML Artifacts*: adversary considerations include publishing permissively-licensed-looking datasets whose actual content contains non-redistributable third-party material; the hash-pinning in §2.2 and the lint in §2.6 defend against inadvertent inclusion.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: the dataset card is the primary Map artefact; every field maps to a question the framework requires. *Measure*: the numeric derivations (coverage, PII drop-rate, toxicity drop-rate, IAA, per-subgroup AUC) feed the Measure function from upstream manifests. *Manage*: `LICENSES.yml` versioning + lint + ERRATA process is the change-controlled governance loop. *Govern*: the *existence* of a datasheet alongside a dataset is itself a governance policy; the framework's AI Actors section expects a named maintainer (the Maintenance section).
- Additional cross-reference: OECD AI Principles — <https://oecd.ai/en/ai-principles> — section on transparency and explainability underwrites the datasheet's existence as a disclosure artefact; EU AI Act (OJ 2024/1689) Annex IV and Art. 10 impose dataset-documentation obligations that the Datasheets-for-Datasets template maps to directly.
