from __future__ import annotations

from pathlib import Path

from eval.workflow_eval import (
    WORKFLOW_CATEGORIES,
    load_cases,
    run_workflow_eval,
    validate_case_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_workflow_dataset_has_stage_four_coverage():
    cases = load_cases(PROJECT_ROOT / "eval" / "workflow_cases.jsonl")

    assert len(cases) == 32
    assert {case.category for case in cases} == WORKFLOW_CATEGORIES
    assert sum(case.scenario == "verifier_fault" for case in cases) == 5
    assert sum(case.scenario == "revision" for case in cases) == 2
    validate_case_coverage(cases)


def test_workflow_eval_runs_the_full_deterministic_harness():
    report = run_workflow_eval(
        load_cases(PROJECT_ROOT / "eval" / "workflow_cases.jsonl")
    )

    assert report["dataset_case_count"] == 32
    assert report["bad_cases"] == []
    assert report["summary"]["expected_path_accuracy"] == 1.0
    assert report["summary"]["expected_tool_precision"] == 1.0
    assert report["summary"]["expected_tool_recall"] == 1.0
    assert report["summary"]["verifier_detection_recall"] == 1.0
    assert report["summary"]["grounded_citation_rate"] == 1.0
    assert report["summary"]["no_side_effect_before_approval"] == 1.0
    assert report["summary"]["revision_merge_correctness"] == 1.0
    assert report["summary"]["commit_idempotency"] == 1.0
