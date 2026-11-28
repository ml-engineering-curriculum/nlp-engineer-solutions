# exercise-03 — Release Gates and Rollback Design

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-113-nlp-systems-design-and-responsible-release/exercises/exercise-03-release-gates-and-rollback-design.md`.
> The target learning outcomes implied by that filename: design the **release-gate set**
> that a NLP model must pass to ship, define the **rollout strategy** (shadow / canary /
> progressive), and the **rollback triggers + actions** that undo the release if the
> gates degrade in production. Deliverable is a short policy document plus the
> configuration files that enforce it.

## 1. Solution overview

A release gate is the numeric bar a candidate must clear before traffic is routed to it.
A rollback trigger is the numeric condition under which traffic is routed back. The pair
makes shipping an engineering decision, not a judgement call.

The dominant failure modes in this area:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Gates set by the model, not by the product** | "Ship if F1 > current prod" — every release ratchets up on the eval set but breaks on reality | Gates set from the task brief (exercise-01) against a *majority-class baseline* and a *human floor*, not from the best candidate |
| **Only aggregate gates; no slice gates** | Overall accuracy improves while the Spanish-language slice regresses by 8 pp; no alarm | One gate per slice (language, length, domain, tier). The weakest slice must meet its own gate |
| **No separate gates for safety-adjacent metrics** | Model is more accurate but also more confidently wrong when wrong (calibration regression); downstream agent trusts the wrong answers | Gate on calibration (ECE, Brier) and on refusal/abstain rate, not just accuracy |
| **Big-bang releases** | New model goes to 100 % at once; a regression appears only after the rollback window has closed | Progressive rollout (1 % → 5 % → 25 % → 100 %) with hold times; each stage requires live-gate re-confirmation |
| **Rollback is manual and slow** | Alarm fires at 03:00; the on-call does not have the credentials/runbook to flip the flag; MTTR is hours | Rollback is a *flag flip* or a *re-point to the previous artefact*, with a documented restore time ≤ the alarm window |
| **Rollback trigger is noisy; the team learns to ignore it** | Per-request alerts firing every hour; eventually muted; a real regression slips through | Gate on windowed statistics with explicit false-positive budgets; tie each trigger to a runbook step |
| **No shadow-eval before ship** | Candidate is not evaluated on live traffic distribution; offline test set hides the distribution shift | Shadow run the candidate alongside prod for N days; compare on *live inputs*, not just the frozen test set |
| **No post-release evaluation window** | "Shipped; done". Any slow regression (drift) is invisible until the next release | Post-release monitoring window (minimum 14 days) with per-slice tracking and a named owner; post-mortem if any gate drops |
| **Rollback deletes the problem evidence** | Rollback discards logs, errors, and feature vectors needed for root-cause | Rollback preserves the last N hours of inputs + predictions in the rollback-forensics bucket |
| **Model-card does not reflect gates** | The gates are in a wiki; the deployed artefact has no record of them; auditor cannot verify | Gates are stored in the model card and in the deployment config; `config.yaml` in `mod-112-production-nlp-pipelines/exercise-01` is the single source |

The deliverable is five parts:

1. **Gate set** — numeric thresholds for the primary metric, per slice, for
   calibration, refusal, latency, and safety-adjacent metrics. From the task brief
   (exercise-01).
2. **Rollout policy** — shadow / canary / progressive, with stage definitions and hold
   times.
3. **Rollback triggers** — observable conditions that revert traffic.
4. **Rollback runbook** — concrete actions, owners, and expected restore time.
5. **Post-release monitoring plan** — what is watched for how long after ship.

Authoritative sources this solution draws on:

- **NIST AI Risk Management Framework (AI RMF 1.0)** —
  <https://www.nist.gov/itl/ai-risk-management-framework>. Release gates are the
  `Measure 2` artefact; rollback triggers and runbook are the `Manage 4.1` response
  artefact.
- **OWASP Machine Learning Security Top 10** —
  <https://owasp.org/www-project-machine-learning-security-top-10/>. `ML05 Model Theft`,
  `ML06 Supply Chain`, `ML09 Output Integrity` rates are release-gate inputs; rollback
  triggers cover `ML01` input manipulation and `ML09` output-integrity signals.
- **MITRE ATLAS** — <https://atlas.mitre.org/>. `AML.T0043 Craft Adversarial Data`
  and `AML.T0051 LLM Prompt Injection` are rollback-trigger-adjacent signals; the
  detection-side techniques inform the monitor.
- **Model Cards** (Mitchell et al. 2019). <https://arxiv.org/abs/1810.03993>. The gates
  are stored in the model-card `evaluation` and `release-gate` sections.
- **Google's "What's your ML test score?"** (Breck et al. 2017).
  <https://research.google/pubs/whats-your-ml-test-score-a-rubric-for-ml-production-systems/>
  — classical reference for the pre-release test rubric that gates on data, features,
  model, and infrastructure.

## 2. Worked implementation — gates + rollout + rollback for the email-triage classifier

Reuse the use case from exercise-01/02: intent classifier + entity summary, 240 k emails/day,
PII inputs, CPU inference, release gate macro-F1 ≥ 0.82 overall and per-slice.

### 2.1 Release-gate set — stored next to the model artefact

```yaml
# release_gates.yaml  (versioned with the model card)
model_card_ref: "email_triage@v1.0.0"

offline_gates:                       # computed on the held-out test set
  primary:
    metric: "macro_f1"
    slice: "overall"
    min: 0.82
  per_slice:
    - name: "language:en"
      metric: "macro_f1"
      min: 0.80
    - name: "language:es"
      metric: "macro_f1"
      min: 0.78
    - name: "language:fr"
      metric: "macro_f1"
      min: 0.78
    - name: "length_bucket:<200"
      metric: "macro_f1"
      min: 0.80
    - name: "length_bucket:200-1000"
      metric: "macro_f1"
      min: 0.80
    - name: "length_bucket:>1000"
      metric: "macro_f1"
      min: 0.75
    - name: "customer_tier:enterprise"
      metric: "macro_f1"
      min: 0.80
  calibration:
    - metric: "expected_calibration_error"
      bins: 10
      max: 0.05
    - metric: "brier_score"
      max: 0.18
  behaviour:
    - metric: "abstain_rate"
      max: 0.10
    - metric: "span_f1_products_mentioned"
      min: 0.85
  robustness:
    - name: "punctuation_perturbation"
      metric: "macro_f1_delta"
      max_absolute: 0.03
    - name: "mixed_language_perturbation"
      metric: "macro_f1_delta"
      max_absolute: 0.04
  fairness_proxies:
    - name: "language_slice_spread"
      metric: "macro_f1_spread_across_languages"
      max: 0.08       # no language slice may lag the best by more than 8 pp

operational_gates:                   # computed on the target SKU
  latency_p50_ms:
    max: 120
  latency_p95_ms:
    max: 300
  latency_p99_ms:
    max: 600
  throughput_rps_per_core:
    min: 30
  memory_mb:
    max: 2048

security_gates:                      # required to cross prod boundary
  model_supply_chain:
    - name: "hub_revision_pinned"
      required: true                  # OWASP ML06
    - name: "model_signature_verified"
      required: true
    - name: "dependency_audit_passed"
      required: true
  data_governance:
    - name: "datasheet_present"
      required: true
    - name: "pii_classification_recorded"
      required: true
  evaluation_governance:
    - name: "test_set_signed_off"
      required: true
    - name: "model_card_signed_off"
      required: true
      signers: ["pm", "ml_lead", "security_reviewer"]
```

Points worth defending in review:

- **Gates are stored in the same artefact as the model.** The model card and the release
  gate set are versioned together; rolling the model back rolls the gates back. If they
  live in a wiki, they are not auditable.
- **Per-slice gates are not a "nice-to-have".** A release that clears overall macro-F1
  but fails a slice is a regression for the users in that slice. The weakest-slice-passes
  rule is the one that prevents "ship on average, break for Spanish".
- **Calibration gates are peers of accuracy gates.** A better-accuracy, worse-calibration
  model routes with high confidence to the wrong queue and the agent trusts it. The
  downstream workflow depends on `intent_confidence` thresholds; calibration drift
  breaks those.
- **Robustness perturbation gates matter for text inputs.** Email inputs are noisy;
  punctuation and language-mix perturbations are realistic. A `3 pp` tolerance on these
  perturbations is a cheap, high-signal gate.
- **Security gates are binary.** You cannot numerically trade off a missing model
  signature. Either it is present or the release is blocked.

### 2.2 Rollout policy — shadow + canary + progressive

```yaml
# rollout.yaml
model: "email_triage@v1.0.0"
baseline: "email_triage@v0.9.3"       # the model currently serving prod

stages:
  - name: "shadow"
    traffic_pct: 0
    duration: "7d"
    description: >
      New model receives every request in parallel with prod, predictions written to
      the shadow table. No user-visible effect.
    exit_criteria:
      - "offline_gates: all pass"
      - "shadow_traffic_gates: all pass"
      - "disagreement_rate_vs_baseline <= 0.15"
      - "prediction_distribution_js_divergence_vs_baseline <= 0.05"

  - name: "canary"
    traffic_pct: 1
    duration: "48h"
    description: >
      1% of traffic routed to the candidate; agent-override rate, latency, abstain
      rate, and error rate tracked separately. Hold for one full business cycle.
    exit_criteria:
      - "p95_latency <= 300ms"
      - "error_rate <= baseline * 1.1"
      - "agent_override_rate <= baseline * 1.1"
      - "per_language_slice: macro_f1 >= gate (from labelled sample within window)"
    rollback_on:
      - "p95_latency > 400ms for 15min"
      - "error_rate > baseline * 2.0 for 15min"

  - name: "progressive-5"
    traffic_pct: 5
    duration: "48h"
    description: "Broaden the sample; re-check slice performance."
    exit_criteria: *canary_exit_criteria
    rollback_on: *canary_rollback_on

  - name: "progressive-25"
    traffic_pct: 25
    duration: "48h"
    exit_criteria: *canary_exit_criteria
    rollback_on: *canary_rollback_on

  - name: "full"
    traffic_pct: 100
    duration: "indefinite"
    exit_criteria: ["14-day post-release window passes without a rollback trigger"]
    rollback_on: *canary_rollback_on
```

Points worth defending in review:

- **Shadow stage is non-negotiable for a replacement.** Offline evaluation on a frozen
  test set does not catch distribution shift. A shadow stage evaluates on live inputs
  before any traffic is affected. The cost is doubled inference on 100 % of traffic for
  7 days; this is small compared to the cost of a bad release.
- **Canary hold time covers a full business cycle.** Two weekdays catches patterns
  (weekend traffic, Monday spike) that a one-hour canary misses. For email triage, where
  inputs cycle weekly, a 48 h canary is the minimum.
- **Progressive stages exist so a partial failure costs 5 % of users, not 100 %.**
  The `1 → 5 → 25 → 100` ramp is conservative; aggressive teams use `5 → 25 → 100`.
  Which to pick is a function of blast radius, not team preference.
- **Disagreement-rate and JS divergence between shadow and baseline is the leading
  indicator of a bad model.** Even when offline gates pass, a candidate that disagrees
  with the baseline on 25 % of inputs is behaviourally different; treat as a release
  risk.

### 2.3 Rollback triggers — windowed, with named forensics

```yaml
# rollback.yaml
rollback_triggers:
  - id: "LATENCY_P95_BREACH"
    condition: "p95_latency_ms > 400 sustained for 15m"
    action: "flip feature flag email_triage_classifier_enabled = false"
    owner: "oncall_sre"
    expected_restore_time_minutes: 5
    forensics:
      capture: "last 2h of requests + predictions to s3://.../rollback-forensics/<ts>/"
      alert: "page oncall_ml"

  - id: "ERROR_RATE_BREACH"
    condition: "5xx_rate > baseline * 2.0 sustained for 15m"
    action: "flip feature flag email_triage_classifier_enabled = false"
    owner: "oncall_sre"
    expected_restore_time_minutes: 5
    forensics:
      capture: "last 2h of requests + exceptions"
      alert: "page oncall_ml"

  - id: "AGENT_OVERRIDE_SPIKE"
    condition: "agent_override_rate on last 1000 overrides > baseline * 1.25 sustained for 2h"
    action: "redeploy prior model_card revision (baseline)"
    owner: "oncall_ml"
    expected_restore_time_minutes: 10
    forensics:
      capture: "last 24h of overridden predictions with inputs"
      alert: "page oncall_ml, open high-priority ticket"

  - id: "PER_SLICE_QUALITY_DROP"
    condition: "any per-slice macro_f1 (from rolling 500-email labelled sample) < gate for 48h"
    action: "redeploy prior model_card revision (baseline)"
    owner: "oncall_ml"
    expected_restore_time_minutes: 30
    forensics:
      capture: "rolling labelled sample for the affected slice"
      alert: "page oncall_ml, post-mortem required"

  - id: "ABSTAIN_RATE_SPIKE"
    condition: "abstain_rate > 0.15 sustained for 24h"
    action: "freeze candidate at current traffic %; open investigation ticket"
    owner: "oncall_ml"
    expected_restore_time_minutes: 60
    forensics:
      capture: "last 7d of abstained predictions with inputs (sampled)"
      alert: "page oncall_ml"

  - id: "SECURITY_EVENT"
    condition: >
      any of:
      - upstream model_card revision check fails
      - signature verification fails
      - model-supply-chain advisory affects the base checkpoint
      - prompt-injection-adjacent signals exceed threshold
    action: "flip feature flag + freeze retraining pipeline"
    owner: "security_oncall + ml_lead"
    expected_restore_time_minutes: 10
    forensics:
      capture: "full request/prediction stream for last 24h"
      alert: "page security_oncall, escalate per incident policy"
```

Points worth defending in review:

- **Every trigger has a window, not a single-observation rule.** Single-request alerts
  are noise; sustained-over-N-minutes triggers are signal. The windows are calibrated
  against baseline false-positive rate so an alerting team does not learn to ignore.
- **Two action types: flag flip and artefact re-point.** Flag flip is for availability
  and security incidents; artefact re-point is for quality incidents. The response type
  depends on *how reversible the problem is*; a slow quality drop does not require a
  5-minute response, a 5xx spike does.
- **Forensics capture is part of the trigger, not a separate step.** A rollback that
  discards the inputs and predictions makes root-cause impossible. The forensics bucket
  retains the evidence for 90 days and is the artefact the post-mortem reads.
- **Security triggers freeze retraining.** The default for a quality-regressed model is
  "retrain and re-ship". The default for a supply-chain security event is "stop" — the
  retraining pipeline consumes the compromised upstream.

### 2.4 Rollback runbook — concrete steps per trigger

```markdown
# Rollback Runbook — email_triage

On trigger LATENCY_P95_BREACH / ERROR_RATE_BREACH:
  1. SRE on-call confirms the trigger (prometheus dashboard: email_triage_latency).
  2. `kubectl -n ml-serving rollout undo deployment/email-triage` — no, use the flag:
     `kubectl -n ml-serving set env deployment/email-triage EMAIL_TRIAGE_CLASSIFIER_ENABLED=false`
  3. Verify traffic drops on the classifier dashboard within 60 s (the ingress service
     checks the flag per request).
  4. Confirm agent UI shows the "manual routing" fallback (screenshot in #release-ops).
  5. Capture forensics: trigger the rollback-forensics lambda:
     `aws lambda invoke --function-name email-triage-forensics --payload '{"since":"-2h"}' out.json`
  6. Page oncall_ml for ownership of the investigation.
  7. Update status page if the fallback UX degrades agent SLA.

On trigger AGENT_OVERRIDE_SPIKE / PER_SLICE_QUALITY_DROP:
  1. ML on-call confirms the signal on the override-rate dashboard.
  2. Re-point the serving deployment at the previous model-card revision:
     `kubectl -n ml-serving set image deployment/email-triage \
        serving=<registry>/email-triage:v0.9.3`
  3. Verify traffic on the baseline model within the expected rollout window (5–10 min).
  4. Capture forensics: run `python -m email_triage.forensics dump-sample \
       --since=-24h --slice=<slice-id> --out=s3://.../rollback-forensics/<ts>/`
  5. Freeze the candidate: `python -m email_triage.release lock-at --version=v0.9.3`
  6. Open a post-mortem ticket; owner is ml_lead; required within 5 business days.

On trigger SECURITY_EVENT:
  1. Security on-call + ML lead both paged; escalation per incident policy.
  2. Flip feature flag (as above).
  3. Freeze retraining: `python -m email_triage.release freeze-retraining --reason="SEC-<ticket>"`
  4. Snapshot: training data as of last successful run; current model artefacts;
     current config.yaml; current supply-chain audit output.
  5. Hand to security IR; do not re-enable until clearance.
```

### 2.5 Post-release monitoring plan

Minimum 14 days after full rollout; the monitors feed the same triggers that drive
rollback. The monitored metrics are:

| Metric | Source | Cadence | Threshold / gate |
|---|---|---|---|
| p50, p95, p99 latency | serving ingress | 1-min buckets | p95 ≤ 300 ms |
| 5xx rate | serving ingress | 1-min | baseline × 1.1 |
| Abstain rate | serving log | 1-h | ≤ 0.10 |
| Agent-override rate | agent UI event stream | 1-h | baseline × 1.1 |
| Macro-F1 per slice (rolling labelled sample) | labelling queue | 7-d rolling | per-slice gate |
| Calibration (ECE on confidence ≥ 0.6 bucket) | serving log + labels | 7-d rolling | ≤ 0.05 |
| Input distribution shift (JS divergence per feature) | serving log vs. reference window | 1-h | ≤ 0.05 |

A gate drop on any of these for the 14-day window re-opens the release decision. A
rollback-and-post-mortem is the default response; "accept and keep" requires named
sign-off.

## 3. Validation steps

1. **Gates versioned with the model.** `release_gates.yaml` lives next to the model
   card and is tagged with the same version. A gate change without a model-card rev
   is a governance error.
2. **Per-slice gate enforcement in CI.** The release-gate check in CI reads the slice
   gates from the YAML; a candidate with overall macro-F1 above `primary` but a slice
   below its gate must be rejected by the check. Verify with a fixture candidate.
3. **Rollback mechanism tested in staging.** The rollback actions (flag flip; artefact
   re-point) are exercised in staging with a fault-injection test. Expected restore
   time is measured, not estimated.
4. **Forensics bucket exists and has retention policy.** `s3://.../rollback-forensics/`
   exists, has write permissions for the lambda, read permissions for the post-mortem
   role, and a 90-day retention lifecycle rule.
5. **Alert noise budget.** For each trigger, measure the false-positive rate in staging
   or against a historical traffic replay. Any trigger with > 1 false positive per week
   must be widened; otherwise the team will silently mute it.
6. **Sign-off trail is enforced.** `security_gates.evaluation_governance.model_card_signed_off`
   is blocking in CI; a release without the three named signers cannot pass. Verify by
   attempting to deploy without the sign-off and confirming rejection.
7. **Post-release owner named.** The 14-day post-release monitoring plan names a single
   owner. "The team" is not an owner.
8. **Shadow stage produces a diff report.** The shadow stage emits a
   `shadow_diff_report.json` summarising per-slice disagreement with baseline; the
   report is reviewed and signed off before entering canary.
9. **The runbook is reachable at 03:00.** The runbook location is in the alert payload,
   not only in a wiki. Confirm by acknowledging an alert from oncall and following the
   link.

## 4. Grading rubric / review checklist

| Dimension | Pass (2) | Partial (1) | Fail (0) |
|---|---|---|---|
| **Gate set** | Primary + per-slice + calibration + behaviour + robustness + operational + security | Primary + per-slice only | Only overall primary metric |
| **Gate storage** | Versioned with the model card; machine-readable; enforced in CI | Human-readable doc; manual enforcement | Wiki / Slack |
| **Rollout stages** | Shadow + canary + ≥ 2 progressive stages + full; hold times named | Canary + full | Big-bang |
| **Stage exit criteria** | Numeric; include disagreement/JSD vs. baseline in shadow | Numeric for overall only | "Looks good" |
| **Rollback triggers** | Windowed conditions per failure mode; forensics capture named per trigger | Triggers without windows | None |
| **Rollback actions** | Flag flip for availability/security; artefact re-point for quality; owners + expected restore | One action type covers everything | None |
| **Runbook** | Concrete per-trigger steps, exact commands, named escalation | Prose description | Missing |
| **Post-release monitoring** | 14-day minimum, named owner, same gate set as release | Shorter window or no owner | Not addressed |
| **Security gates** | Supply-chain + signature + data-governance + sign-off; binary | Partial | Not addressed |
| **Forensics** | Rollback preserves inputs + predictions; retention policy; access controlled | Preserved but no retention policy | Lost on rollback |

A release design that only addresses the "gate set" and not the "rollback runbook" is
incomplete; the two are peers. Likewise, a design without a shadow stage cannot pass:
offline evaluation on a frozen test set is not the live-traffic distribution.

## 5. Common mistakes

- **Setting gates from the candidate, not from the task.** If every release's gate is
  "beat the previous candidate on the eval set", gates ratchet on what is easy to
  measure and drift from what the product needs. Gates come from the task brief
  (exercise-01) and from a baseline (majority-class, retrieval, human floor).
- **Overall accuracy without per-slice decomposition.** Macro-F1 of 0.85 overall can hide
  macro-F1 of 0.68 on a language slice; without a slice gate, the release ships and the
  users in that slice see the regression.
- **Calibration regression unaddressed.** A more-accurate, worse-calibrated model routes
  with high confidence to the wrong queue and downstream workflows that use the
  confidence as a gate break silently.
- **Skipping shadow mode.** Jumping from offline eval to canary at 1 % skips the
  cheapest, highest-information step. Shadow mode catches distribution shift before any
  user is affected.
- **Rollback as "deploy the old image".** This is not a plan: it has no trigger, no
  owner, no expected restore time, and no forensics capture. Rollback is a procedure,
  not a reflex.
- **Alerts without rate limiting.** A noisy trigger is a muted trigger. Measure the
  false-positive rate before shipping the alert; widen the window if needed.
- **Rollback loses the evidence.** If the rollback discards the inputs and predictions
  that caused it, there is no root-cause. Forensics capture is part of the rollback
  action, not a bonus.
- **Retraining on autopilot after a security event.** A supply-chain compromise is
  consumed by the retraining pipeline. Security-triggered rollback must freeze the
  retraining.
- **No signed sign-off trail.** NIST AI RMF `Govern` requires named accountability.
  Author-only sign-off is not governance.
- **No post-release monitoring window.** The release is "done" only after N days of
  live-traffic stability. A model that passes offline gates but regresses in week 2 is
  still a bad release.

## 6. References

Primary standards and official projects used in this solution:

- NIST AI Risk Management Framework (AI RMF 1.0). <https://www.nist.gov/itl/ai-risk-management-framework>
  — `Measure 2` (performance measurement) and `Manage 4.1` (risk response) are the
  anchors for the gate set and the rollback plan.
- OWASP Machine Learning Security Top 10. <https://owasp.org/www-project-machine-learning-security-top-10/>
  — `ML05`, `ML06`, `ML09` appear in security gates and rollback triggers.
- MITRE ATLAS. <https://atlas.mitre.org/> — `AML.T0043`, `AML.T0051` motivate rollback
  signals on adversarial and prompt-injection-adjacent inputs.

Supporting literature:

- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAccT.
  <https://arxiv.org/abs/1810.03993> — gates stored in the model-card evaluation section.
- Breck, E. et al. (2017). *What's your ML test score? A rubric for ML production systems*.
  Google Research. <https://research.google/pubs/whats-your-ml-test-score-a-rubric-for-ml-production-systems/>
  — the canonical pre-release test rubric for data, features, model, and infrastructure.

Cross-module references:

- Supply-chain pinning gates (`security_gates.model_supply_chain`) are implemented in
  `mod-112-production-nlp-pipelines/exercise-01`.
- Per-slice metric computation driving offline gates is in `mod-111-nlp-evaluation`.
- Input / prediction / performance drift monitors feeding the post-release plan are in
  `mod-112-production-nlp-pipelines/exercise-04`.
