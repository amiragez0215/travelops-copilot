# TravelOps Agent Workflow Eval Report

## Scope

- Dataset: 32 deterministic workflow cases.
- Categories: request_extraction=5, policy_safety=4, tool_data=5, rag=6, verifier=5, hitl_revision=4, commit_idempotency=3.
- Graph: production topology plus real DecisionGate interrupt/resume; external dependencies are deterministic substitutes.
- Verifier fault injection and commit assertions use the real production components.

## Core Metrics

| Metric | Result |
| --- | ---: |
| task_success_rate | 1.0000 |
| expected_path_accuracy | 1.0000 |
| expected_tool_precision | 1.0000 |
| expected_tool_recall | 1.0000 |
| verifier_detection_recall | 1.0000 |
| verifier_false_positive_rate | 0.0000 |
| grounded_citation_rate | 1.0000 |
| no_side_effect_before_approval | 1.0000 |
| revision_merge_correctness | 1.0000 |
| commit_idempotency | 1.0000 |
| mean_latency_ms | 176.9161 |
| p95_latency_ms | 321.3297 |

## Bad Cases

No workflow case failed the current acceptance criteria.
## Interpretation Boundary

This is an offline, deterministic workflow harness. It proves routing, HITL, verifier fault detection, and database side-effect boundaries without claiming live-provider or live-LLM quality. End-to-end grounded citations remain a future run-level measurement that joins a real Proposal with its exact Evidence Pool.
