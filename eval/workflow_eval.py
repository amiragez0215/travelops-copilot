from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.workflows.plan_trip_workflow import build_plan_trip_graph
from app.workflows.snapshot_utils import get_interrupt_values, snapshot_values
from app.workflows.workflow_runtime import PlanTripWorkflowRuntime
from eval.rag_metrics import grounded_citation_rate
from eval.workflow_metrics import (
    boolean_rate,
    latency_summary,
    ordered_path_contains,
    set_precision,
    set_recall,
)

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover - compatibility with older LangGraph
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver


WORKFLOW_CATEGORIES = {
    "request_extraction",
    "policy_safety",
    "tool_data",
    "rag",
    "verifier",
    "hitl_revision",
    "commit_idempotency",
}
SCENARIOS = {
    "approve",
    "cancel",
    "revision",
    "clarify",
    "safe_reject",
    "no_candidates",
    "budget_infeasible",
    "evidence_repair",
    "invalid_decision",
    "verifier_fault",
}
TOOLS_BY_NODE = {
    # ToolExecute 是统一调度节点；四个必选业务要求对应三个 MCP Tool，
    # search_flights 的一次返回同时包含去程与返程候选。
    "tool_execute": ["get_weather", "search_flights", "search_hotels"],
}


@dataclass(frozen=True)
class WorkflowEvalCase:
    case_id: str
    category: str
    scenario: str
    input: dict[str, Any]
    expected_path_contains: list[str]
    expected_tools: list[str]
    must_interrupt: bool
    must_not_commit_before_approval: bool
    fault: str | None = None
    expected_verifier_issue: str | None = None


def load_cases(path: str | Path) -> list[WorkflowEvalCase]:
    """Load the deterministic Agent Workflow Gold Dataset."""

    cases: list[WorkflowEvalCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        case_id = _required_text(raw, "case_id", line_number)
        category = _required_text(raw, "category", line_number)
        scenario = _required_text(raw, "scenario", line_number)
        if case_id in seen_ids:
            raise ValueError(f"duplicate workflow case_id: {case_id}")
        if category not in WORKFLOW_CATEGORIES:
            raise ValueError(f"{case_id}: unsupported category={category}")
        if scenario not in SCENARIOS:
            raise ValueError(f"{case_id}: unsupported scenario={scenario}")
        input_value = raw.get("input")
        if not isinstance(input_value, dict):
            raise ValueError(f"{case_id}: input must be an object")
        case = WorkflowEvalCase(
            case_id=case_id,
            category=category,
            scenario=scenario,
            input=dict(input_value),
            expected_path_contains=[str(value) for value in raw.get("expected_path_contains") or []],
            expected_tools=[str(value) for value in raw.get("expected_tools") or []],
            must_interrupt=bool(raw.get("must_interrupt", False)),
            must_not_commit_before_approval=bool(raw.get("must_not_commit_before_approval", True)),
            fault=_optional_text(raw.get("fault")),
            expected_verifier_issue=_optional_text(raw.get("expected_verifier_issue")),
        )
        if scenario == "verifier_fault" and not case.fault:
            raise ValueError(f"{case_id}: verifier_fault requires fault")
        if case.expected_verifier_issue and scenario != "verifier_fault":
            raise ValueError(f"{case_id}: expected_verifier_issue only applies to verifier_fault")
        seen_ids.add(case_id)
        cases.append(case)
    if not cases:
        raise ValueError("Workflow Eval dataset is empty")
    return cases


def validate_case_coverage(cases: Sequence[WorkflowEvalCase]) -> None:
    """Fail early when a dataset does not cover the stage-four acceptance boundary."""

    categories = {case.category for case in cases}
    missing_categories = WORKFLOW_CATEGORIES - categories
    if missing_categories:
        raise ValueError(f"workflow dataset missing categories: {sorted(missing_categories)}")
    scenarios = {case.scenario for case in cases}
    required = {"approve", "cancel", "revision", "clarify", "safe_reject", "verifier_fault"}
    missing_scenarios = required - scenarios
    if missing_scenarios:
        raise ValueError(f"workflow dataset missing scenarios: {sorted(missing_scenarios)}")


def run_workflow_eval(cases: Sequence[WorkflowEvalCase]) -> dict[str, Any]:
    """Run reproducible workflow, verifier, HITL, and commit-boundary evaluations."""

    validate_case_coverage(cases)
    observations = [_observe_case(case) for case in cases]
    return {
        "schema_version": "workflow_eval_v1",
        "dataset_case_count": len(cases),
        "category_counts": dict(Counter(case.category for case in cases)),
        "scenario_counts": dict(Counter(case.scenario for case in cases)),
        "summary": _summarize(observations),
        "observations": observations,
        "bad_cases": [row for row in observations if not row["passed"]],
        "execution_boundary": {
            "graph": "production StateGraph topology and DecisionGate interrupt/resume",
            "deterministic_substitutes": "LLM, MCP, Chroma, and external tools are replaced by deterministic test doubles",
            "verifier": "real ProposalVerifier with fault-injected valid Proposal state",
            "commit": "real CommitDraftService with isolated in-memory SQLite",
        },
    }


def write_report(report: Mapping[str, Any], *, json_path: str | Path, markdown_path: str | Path) -> None:
    output_json = Path(json_path)
    output_markdown = Path(markdown_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_markdown.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# TravelOps Agent Workflow Eval Report",
        "",
        "## Scope",
        "",
        f"- Dataset: {report['dataset_case_count']} deterministic workflow cases.",
        f"- Categories: {', '.join(f'{key}={value}' for key, value in report['category_counts'].items())}.",
        "- Graph: production topology plus real DecisionGate interrupt/resume; external dependencies are deterministic substitutes.",
        "- Verifier fault injection and commit assertions use the real production components.",
        "",
        "## Core Metrics",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
    ]
    for key in (
        "task_success_rate",
        "expected_path_accuracy",
        "expected_tool_precision",
        "expected_tool_recall",
        "verifier_detection_recall",
        "verifier_false_positive_rate",
        "grounded_citation_rate",
        "no_side_effect_before_approval",
        "revision_merge_correctness",
        "commit_idempotency",
    ):
        lines.append(f"| {key} | {_display(summary.get(key))} |")
    lines.extend([
        f"| mean_latency_ms | {_display(summary.get('mean_latency_ms'))} |",
        f"| p95_latency_ms | {_display(summary.get('p95_latency_ms'))} |",
        "",
        "## Bad Cases",
        "",
    ])
    bad_cases = report.get("bad_cases") or []
    if not bad_cases:
        lines.append("No workflow case failed the current acceptance criteria.")
    else:
        for item in bad_cases:
            lines.extend([
                f"### {item['case_id']}",
                "",
                f"- Scenario: `{item['scenario']}`",
                f"- Failed checks: {', '.join(item['failed_checks'])}",
                f"- Path: {' → '.join(item['actual_path'])}",
                "",
            ])
    lines.extend([
        "## Interpretation Boundary",
        "",
        "This is an offline, deterministic workflow harness. It proves routing, HITL, verifier fault detection, and database side-effect boundaries without claiming live-provider or live-LLM quality. End-to-end grounded citations remain a future run-level measurement that joins a real Proposal with its exact Evidence Pool.",
        "",
    ])
    return "\n".join(lines)


def _observe_case(case: WorkflowEvalCase) -> dict[str, Any]:
    started = time.perf_counter()
    if case.scenario == "verifier_fault":
        result = _execute_verifier_fault(case)
    else:
        result = _execute_graph_case(case)
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    actual_path = result["actual_path"]
    actual_tools = result["actual_tools"]
    checks = {
        "path": bool(ordered_path_contains(actual_path, case.expected_path_contains)),
        "tool_precision": set_precision(actual_tools, case.expected_tools) == 1.0,
        "tool_recall": set_recall(actual_tools, case.expected_tools) == 1.0,
        "interrupt": result["interrupted"] == case.must_interrupt,
        "no_side_effect_before_approval": result["no_side_effect_before_approval"] if case.must_not_commit_before_approval else True,
        "revision_merge": result["revision_merge_correct"],
        "commit_idempotency": result["commit_idempotent"],
        "verifier_detection": result["verifier_detected"],
    }
    applicable = {key: value for key, value in checks.items() if result["applicable"].get(key, True)}
    return {
        "case_id": case.case_id,
        "category": case.category,
        "scenario": case.scenario,
        "actual_path": actual_path,
        "actual_tools": actual_tools,
        "checks": checks,
        "applicable": result["applicable"],
        "passed": all(applicable.values()),
        "failed_checks": [key for key, value in applicable.items() if not value],
        "latency_ms": latency_ms,
        "verifier_issue_types": result.get("verifier_issue_types", []),
        "grounded_citation_rate": result.get("grounded_citation_rate"),
        "verifier_clean_pass": result.get("verifier_clean_pass"),
    }


def _execute_graph_case(case: WorkflowEvalCase) -> dict[str, Any]:
    calls: list[str] = []
    ledger = {"commit_calls": 0}
    checkpointer = InMemorySaver()
    graph = build_plan_trip_graph(
        checkpointer=checkpointer,
        node_overrides=_scenario_nodes(case.scenario, calls, ledger),
        route_overrides=_scenario_routes(case.scenario),
    )
    runtime = PlanTripWorkflowRuntime(graph=graph, checkpointer=checkpointer, recursion_limit=100)
    started = runtime.start_plan(
        user_id=str(case.input.get("user_id") or "user_001"),
        raw_message=str(case.input.get("message") or "杭州到成都三日游"),
        reference_date=str(case.input.get("reference_date") or "2026-07-01"),
        trip_session_id=f"workflow-eval-{case.case_id}",
    )
    interrupts = get_interrupt_values(graph_output=started.output, snapshot=started.snapshot)
    interrupted = bool(interrupts)
    no_side_effect_before_approval = ledger["commit_calls"] == 0
    revision_merge_correct = True
    if case.scenario == "approve" and interrupts:
        runtime.resume_decision(thread_id=started.thread_id, decision_payload=_decision_payload(interrupts[0], "approve"))
    elif case.scenario == "cancel" and interrupts:
        runtime.resume_decision(thread_id=started.thread_id, decision_payload=_decision_payload(interrupts[0], "cancel"))
    elif case.scenario == "invalid_decision" and interrupts:
        reprompted = runtime.resume_decision(
            thread_id=started.thread_id,
            decision_payload={**_decision_payload(interrupts[0], "approve"), "proposal_id": "stale_proposal"},
        )
        retry_interrupts = get_interrupt_values(graph_output=reprompted.output, snapshot=reprompted.snapshot)
        interrupted = bool(retry_interrupts)
        if retry_interrupts:
            runtime.resume_decision(thread_id=started.thread_id, decision_payload=_decision_payload(retry_interrupts[0], "cancel"))
    elif case.scenario == "revision" and interrupts:
        revised = runtime.resume_decision(
            thread_id=started.thread_id,
            decision_payload={**_decision_payload(interrupts[0], "request_changes"), "change_request": "预算增加 1000 元，并降低步行强度。"},
        )
        retry_interrupts = get_interrupt_values(graph_output=revised.output, snapshot=revised.snapshot)
        revision_merge_correct = bool(retry_interrupts) and retry_interrupts[0].get("proposal_id") == "proposal_v2"
        if retry_interrupts:
            runtime.resume_decision(thread_id=started.thread_id, decision_payload=_decision_payload(retry_interrupts[0], "approve"))
    elif case.scenario == "evidence_repair" and interrupts:
        runtime.resume_decision(thread_id=started.thread_id, decision_payload=_decision_payload(interrupts[0], "approve"))
    commit_idempotent = True
    if case.scenario in {"approve", "revision", "evidence_repair"}:
        commit_idempotent = _verify_commit_boundary(case.case_id)
    citation_rate = _grounded_fixture_rate() if "proposal" in calls else None
    return {
        "actual_path": calls,
        "actual_tools": [
            tool_name
            for node_name in calls
            for tool_name in TOOLS_BY_NODE.get(node_name, [])
        ],
        "interrupted": interrupted,
        "no_side_effect_before_approval": no_side_effect_before_approval,
        "revision_merge_correct": revision_merge_correct,
        "commit_idempotent": commit_idempotent,
        "verifier_detected": True,
        "grounded_citation_rate": citation_rate,
        "verifier_clean_pass": True,
        "applicable": {
            "revision_merge": case.scenario == "revision",
            "commit_idempotency": case.scenario in {"approve", "revision", "evidence_repair"},
            "verifier_detection": False,
        },
    }


def _execute_verifier_fault(case: WorkflowEvalCase) -> dict[str, Any]:
    # The existing deterministic fixtures build a valid state through the production
    # ProposalGenerator. The actual ProposalVerifier is then fault-injected here.
    from tests.test_verifiers.test_proposal_verifier import _generated_state, _verify

    state = _generated_state()
    clean_output = _verify(_generated_state())
    clean_pass = bool(clean_output["verifier_result"].get("passed"))
    _inject_fault(state, str(case.fault))
    output = _verify(state)
    issue_types = [str(item["issue_type"]) for item in output["verifier_result"]["issues"]]
    detected = str(case.expected_verifier_issue) in issue_types
    return {
        "actual_path": ["verifier"],
        "actual_tools": [],
        "interrupted": False,
        "no_side_effect_before_approval": True,
        "revision_merge_correct": True,
        "commit_idempotent": True,
        "verifier_detected": detected,
        "grounded_citation_rate": None,
        "verifier_clean_pass": clean_pass,
        "verifier_issue_types": issue_types,
        "applicable": {
            "interrupt": False,
            "revision_merge": False,
            "commit_idempotency": False,
            "verifier_detection": True,
        },
    }


def _scenario_nodes(scenario: str, calls: list[str], ledger: dict[str, int]):
    from tests.fakes.plan_trip_workflow import _build_fake_nodes

    nodes = _build_fake_nodes(calls)

    # The deterministic test graph deliberately keeps the production DecisionGate
    # implementation. Wrap it only to expose its path event to the offline harness.
    real_decision_gate = nodes["decision_gate"]
    def decision_gate_node(state):
        calls.append("decision_gate")
        return real_decision_gate(state)
    nodes["decision_gate"] = decision_gate_node

    def record(name: str, update: dict[str, Any]):
        def node(state):
            del state
            calls.append(name)
            return update
        return node

    if scenario == "clarify":
        nodes["missing_info_check"] = record("missing_info_check", {"can_continue": False, "missing_fields": ["start_date"]})
    if scenario == "safe_reject":
        nodes["safety_check"] = record("safety_check", {"safety_result": {"blocked": True}})
    if scenario == "no_candidates":
        nodes["candidate_rank"] = record("candidate_rank", {"candidate_rank_result": {"status": "no_candidates"}})
    if scenario == "budget_infeasible":
        nodes["budget_optimize"] = record("budget_optimize", {"selection_result": {"status": "no_feasible_combination"}})
    if scenario == "evidence_repair":
        count = {"value": 0}
        def evidence_node(state):
            del state
            calls.append("evidence_grade")
            count["value"] += 1
            return {"evidence_result": {"status": "passed", "passed": True, "next_action": "continue" if count["value"] > 1 else "repair"}}
        nodes["evidence_grade"] = evidence_node
    if scenario == "revision":
        # Existing fake implementation already creates proposal_v2 after ApplyRevision.
        pass

    def commit_node(state):
        del state
        calls.append("commit_draft")
        ledger["commit_calls"] += 1
        return {"commit_result": {"status": "committed", "idempotent_replay": False}, "itinerary_status": "approved_simulated"}
    nodes["commit_draft"] = commit_node
    return nodes


def _scenario_routes(scenario: str):
    from tests.fakes.plan_trip_workflow import _build_fake_routes
    return _build_fake_routes()


def _decision_payload(interrupt: Mapping[str, Any], decision: str) -> dict[str, str]:
    return {"decision": decision, "proposal_id": str(interrupt["proposal_id"]), "action_id": str(interrupt["action_id"])}


def _verify_commit_boundary(case_id: str) -> bool:
    from app.db.init_db import init_db
    from app.db.models import ApprovalRecord, TripDraft, TripVersion
    from app.db.seed_data import DEFAULT_SEED_DATA, seed_database
    from app.db.session import create_db_engine, create_session_factory
    from app.repositories.trip_draft_repository import TripDraftRepository
    from app.services.commit_draft_service import CommitDraftService
    from tests.test_services.test_commit_draft_service import _approved_context

    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        seed_database(session, DEFAULT_SEED_DATA, clear=True)
        before = (session.query(TripDraft).count(), session.query(TripVersion).count(), session.query(ApprovalRecord).count())
        if before != (0, 0, 0):
            return False
        context = _approved_context()
        context["decision_result"]["decision_event_id"] = f"workflow_eval_{case_id}"
        context["pending_actions"][0]["decision_event_id"] = f"workflow_eval_{case_id}"
        service = CommitDraftService(TripDraftRepository(session))
        first = service.commit(user_id="user_001", trip_session_id=f"workflow_eval_{case_id}", **context)
        second = service.commit(user_id="user_001", trip_session_id=f"workflow_eval_{case_id}", **context)
        after = (session.query(TripDraft).count(), session.query(TripVersion).count(), session.query(ApprovalRecord).count())
        return first["commit_result"]["status"] == "committed" and second["commit_result"]["status"] == "already_committed" and after == (1, 1, 1)


def _inject_fault(state: dict[str, Any], fault: str) -> None:
    if fault == "selected_hotel_id_fake":
        state["proposal"]["selected_hotel"]["hotel_id"] = "FAKE_HOTEL"
    elif fault == "known_subtotal_minus_1000":
        state["proposal"]["budget_summary"]["known_costs"]["known_subtotal"] -= 1000
    elif fault == "fake_source":
        state["proposal"]["daily_plan"][0]["activities"][1]["evidence_refs"] = ["fake_source"]
    elif fault == "rain_day_all_outdoor":
        for activity in state["proposal"]["daily_plan"][1]["activities"]:
            activity["indoor_outdoor"] = "outdoor"
    elif fault == "real_booking_claim":
        state["proposal"]["summary"] = "已经为你预订酒店并完成付款。"
    else:
        raise ValueError(f"unsupported verifier fault: {fault}")


def _grounded_fixture_rate() -> float | None:
    """Measure proposal citations against the exact deterministic Evidence Pool."""

    from tests.test_verifiers.test_proposal_verifier import _generated_state

    state = _generated_state()
    citations = _collect_evidence_refs(state["proposal"])
    evidence_ids = [str(item.get("chunk_id") or "") for item in state["evidence_pool"]]
    return grounded_citation_rate(citations, evidence_ids)


def _collect_evidence_refs(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        refs = value.get("evidence_refs")
        output = [str(item) for item in refs] if isinstance(refs, list) else []
        for nested in value.values():
            output.extend(_collect_evidence_refs(nested))
        return output
    if isinstance(value, list):
        output: list[str] = []
        for nested in value:
            output.extend(_collect_evidence_refs(nested))
        return output
    return []


def _summarize(observations: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    def metric(name: str, *, applicable: bool = True) -> list[float]:
        return [float(row["checks"][name]) for row in observations if (not applicable or row["applicable"].get(name, True)) and row["applicable"].get(name, True)]

    verifier_rows = [row for row in observations if row["scenario"] == "verifier_fault"]
    clean_verifier_rows = [row for row in observations if row.get("verifier_clean_pass") is not None]
    latency = latency_summary(row["latency_ms"] for row in observations)
    return {
        "task_success_rate": boolean_rate(row["passed"] for row in observations),
        "expected_path_accuracy": boolean_rate(metric("path")),
        "expected_tool_precision": boolean_rate(metric("tool_precision")),
        "expected_tool_recall": boolean_rate(metric("tool_recall")),
        "verifier_detection_recall": boolean_rate(row["checks"]["verifier_detection"] for row in verifier_rows),
        "verifier_false_positive_rate": 0.0 if not clean_verifier_rows else 1.0 - (boolean_rate(row["verifier_clean_pass"] for row in clean_verifier_rows) or 0.0),
        "no_side_effect_before_approval": boolean_rate(metric("no_side_effect_before_approval")),
        "revision_merge_correctness": boolean_rate(metric("revision_merge")),
        "commit_idempotency": boolean_rate(metric("commit_idempotency")),
        "grounded_citation_rate": boolean_rate(
            row["grounded_citation_rate"]
            for row in observations
            if row.get("grounded_citation_rate") is not None
        ),
        **latency,
    }


def _required_text(raw: Mapping[str, Any], key: str, line_number: int) -> str:
    value = str(raw.get(key) or "").strip()
    if not value:
        raise ValueError(f"line {line_number}: {key} is required")
    return value


def _optional_text(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _display(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
