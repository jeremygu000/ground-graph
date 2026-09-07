# ADR-006 Evaluation Method and Release Thresholds

> **Status**: Accepted
> **Date**: 2026-09-07
> **Deciders**: Plan author
> **Related**: ADR-004, ADR-005, plan.md §M8

## Context

The system must measure quality in a reproducible, enforceable way during development and in CI. We need to answer:

- **What metrics** to compute (faithfulness, citation accuracy, recall, latency, etc.)?
- **How to aggregate** per-case scores into an overall quality signal?
- **What thresholds** gate a release or block a PR?
- **How to ensure critical gates cannot be overridden** by a good aggregate score?

## Decision

### 1. Metric layers

Evaluation is organized into three layers, each with different aggregation rules:

| Layer | Metrics | Aggregation | Failure mode |
|---|---|---|---|
| **Critical** (must-pass) | citation_accuracy, provenance_coverage, acl_filter_rate | All must pass | Blocks release regardless of aggregate |
| **Quality** | faithfulness, answer_relevancy, contextual_recall | Mean across cases | Weighted into aggregate |
| **Operational** | p50_latency_ms, error_rate | Percentiles | Warning only |

### 2. Critical gates are binary and non-overridable

Any case failing a **critical** metric produces a `critical_failure` event that:
- Cannot be offset by high scores elsewhere
- Is stored in `evaluation_results` with `severity=critical`
- Causes the CI gate to fail
- Is reported separately in the evaluation report

### 3. Aggregate score formula

```
aggregate = (
    0.35 * faithfulness
  + 0.25 * answer_relevancy
  + 0.20 * citation_accuracy
  + 0.20 * contextual_recall
)
```

- `faithfulness` and `citation_accuracy` below threshold → **FAIL**
- `aggregate` below `0.70` → **FAIL**
- `p50_latency_ms` above `2000` → **WARNING** (not blocking)
- `error_rate` above `0.05` → **FAIL**

### 4. Regression classification

When a new evaluation run regresses against the baseline:

| Severity | Condition | Action |
|---|---|---|
| `critical` | Any critical metric failure | Block release |
| `major` | Aggregate drop > 5pp OR any quality metric drop > 10pp | Require ADR before merge |
| `minor` | Aggregate drop 1-5pp | Require note in PR |
| `informational` | Any other change | No blocking action |

### 5. Baseline is versioned

The baseline is the most recent **production deployment** evaluation run, stored with its commit hash and `evaluation_run_id`. New runs are compared against the active baseline.

### 6. Evaluation runner interface

All evaluators implement the `MetricEvaluator` protocol:

```python
class MetricEvaluator(Protocol):
    name: str
    threshold: float

    async def evaluate(
        self,
        response: QueryResponse,
        evidence: list[Evidence],
    ) -> EvaluationResult: ...
```

The `GroundGraphFaithfulnessMetric` and `GroundGraphAnswerRelevancyMetric` use DeepEval as the backing implementation (see `deepeval_adapter.py`). Custom metrics (`GroundGraphCitationMetric`, `GroundGraphNoHallucinationMetric`) are implemented directly.

## Consequences

**Easier:**
- CI gates fail fast on citation/provenance regressions
- Metric adapters are swappable (DeepEval or custom)
- Regression classification is automated and consistent

**Harder:**
- Multiple thresholds require careful tuning per metric
- Critical gate non-overridability means occasional legitimate regressions block merges

**Accepted trade-offs:**
- DeepEval adds a dependency but provides battle-tested LLM-based metrics
- Binary critical gates may over-block minor regressions; the `minor` classification handles this

## References

- `src/groundgraph/application/evaluation/deepeval_adapter.py` — metric implementations
- `evals/runners/m6_hybrid_evaluation.py` — existing synthetic evaluation runner
- `src/groundgraph/application/ports.py:EvaluationRepository` — result storage
- plan.md §M8 (evaluation system and CI quality gates)
