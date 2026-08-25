from __future__ import annotations

import argparse
from pathlib import Path

from eval.rag_eval import DEFAULT_VARIANTS, load_cases, run_ablation, write_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCOPE_CASES = PROJECT_ROOT / "eval" / "rag_cases.jsonl"
DEFAULT_SEMANTIC_HOLDOUT_CASES = PROJECT_ROOT / "eval" / "rag_challenge_cases.jsonl"
DEFAULT_SCOPE_RESULTS = PROJECT_ROOT / "eval" / "results" / "rag_eval_results.json"
DEFAULT_SCOPE_REPORT = PROJECT_ROOT / "eval" / "rag_report.md"
DEFAULT_SEMANTIC_HOLDOUT_RESULTS = (
    PROJECT_ROOT / "eval" / "results" / "rag_challenge_results.json"
)
DEFAULT_SEMANTIC_HOLDOUT_REPORT = PROJECT_ROOT / "eval" / "rag_challenge_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TravelOps production RAG ablation experiments.",
    )
    parser.add_argument(
        "--suite",
        choices=("scope_regression", "semantic_holdout"),
        default="scope_regression",
        help=(
            "scope_regression validates labelled chunk reachability; "
            "semantic_holdout evaluates natural-language, multi-need, and decoy cases."
        ),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=None,
        help="Optional JSONL override. It takes precedence over --suite.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=DEFAULT_VARIANTS,
        default=list(DEFAULT_VARIANTS),
        help="A=BM25, B=Vector, C=Hybrid RRF, D=Hybrid plus Cross-Encoder.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional bounded smoke run. Omit for the full Gold Dataset.",
    )
    parser.add_argument(
        "--fetch-multiplier",
        type=int,
        default=None,
        help="Override the RRF candidate-pool multiplier for a controlled experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_path = (
        DEFAULT_SCOPE_CASES
        if args.suite == "scope_regression"
        else DEFAULT_SEMANTIC_HOLDOUT_CASES
    )
    cases = load_cases(args.cases or suite_path)
    default_results = (
        DEFAULT_SCOPE_RESULTS
        if args.suite == "scope_regression"
        else DEFAULT_SEMANTIC_HOLDOUT_RESULTS
    )
    default_report = (
        DEFAULT_SCOPE_REPORT
        if args.suite == "scope_regression"
        else DEFAULT_SEMANTIC_HOLDOUT_REPORT
    )
    output_json = args.output_json or default_results
    output_report = args.output_report or default_report
    if args.max_cases is not None:
        if args.max_cases <= 0:
            raise ValueError("--max-cases must be positive")
        cases = cases[: args.max_cases]

    report = run_ablation(
        cases,
        variants=args.variants,
        fetch_multiplier=args.fetch_multiplier,
    )
    write_report(
        report,
        json_path=output_json,
        markdown_path=output_report,
    )
    print(
        "RAG Eval completed: "
        f"cases={report['dataset_case_count']} variants={len(args.variants)}"
    )
    print(f"JSON: {output_json}")
    print(f"Report: {output_report}")


if __name__ == "__main__":
    main()
