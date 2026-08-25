from __future__ import annotations

from pathlib import Path

from eval.workflow_eval import load_cases, run_workflow_eval, write_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cases = load_cases(PROJECT_ROOT / "eval" / "workflow_cases.jsonl")
    report = run_workflow_eval(cases)
    write_report(
        report,
        json_path=PROJECT_ROOT / "eval" / "results" / "workflow_eval_results.json",
        markdown_path=PROJECT_ROOT / "eval" / "workflow_report.md",
    )
    print(f"Workflow Eval completed: cases={report['dataset_case_count']}")


if __name__ == "__main__":
    main()
