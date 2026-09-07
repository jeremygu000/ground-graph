# ADR-012 — Post-MVP Controlled Improvement Loop

## Status

Accepted

## Context

M12 establishes a controlled feedback-to-improvement pipeline so pilot user signals
drive prioritized, evaluated changes without destabilizing the system.

## Decisions

### 1. Feedback Capture
User feedback is captured via the operator API (`POST /operator/feedback`):
- `execution_run_id`: ties feedback to a specific answer
- `category`: one of `correct_answer`, `incorrect_answer`, `missing_citation`,
  `unclear_citation`, `other`
- `correction`: optional user-provided correct answer
- `notes`: free-text context

Feedback is stored in the `evaluation_feedback` table with tenant/principal isolation.

### 2. Feedback-Driven Review Queue
The operator dashboard exposes a review queue (`GET /operator/review/queue`) that:
- Aggregates feedback by claim/fact
- Prioritizes high-impact issues (most-frequently-reported claims)
- Provides auditors with original question, answer, and user correction
- Emits audit trail events to the execution log

### 3. Controlled Improvement Pipeline
All model/prompt/index changes must pass through:
1. **Offline evaluation**: run the evaluation suite against the proposed change
2. **Canary evaluation**: `scripts/canary_eval.py` against live traffic shadow
3. **Regression gate**: no aggregate quality regression vs. baseline (ADR-006)
4. **Human review**: sample review of changed answers before promotion

### 4. Evaluation Baseline
A golden eval dataset (`evals/datasets/golden_cases.jsonl`) is versioned in the repo.
Changes to the retrieval pipeline, prompt templates, or model routing require:
- Full evaluation run against the golden dataset
- Delta report with per-metric change
- Threshold check per ADR-006 before any promotion

### 5. SLO Monitoring
- Faithfulness score tracked per release via Prometheus metric `groundgraph_eval_faithfulness_score`
- Alert fired if score drops below 0.70 (critical threshold from ADR-006)
- Automated rollback triggered if canary evaluation fails per `scripts/canary_eval.py`

## Consequences

- Pilot feedback has a structured path from user to code change
- No direct user feedback can change production behavior without evaluation gate
- Evaluation is the guardrail for all model/prompt changes
