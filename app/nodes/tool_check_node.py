from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.state import TravelState
from app.common.config import settings
from app.schemas.tool_orchestration_schema import ToolCheckResult


REQUIRED_TOOL_NAMES = {"get_weather", "search_flights", "search_hotels"}


def tool_check_node(state: TravelState) -> dict[str, Any]:
    """检查计划覆盖和执行状态，并为两类重试设置独立、有界的路由。"""

    plan = state.get("tool_plan")
    calls = plan.get("calls", []) if isinstance(plan, dict) else []
    planned_names = {str(call.get("tool_name")) for call in calls if isinstance(call, dict)}
    issues: list[dict[str, Any]] = []
    missing = sorted(REQUIRED_TOOL_NAMES - planned_names)
    plan_retry = _safe_int(state.get("tool_plan_retry_count"))
    execute_retry = _safe_int(state.get("tool_execute_retry_count"))

    if missing:
        issues.append({"type": "missing_required_tool_plan", "tools": missing})
        if plan_retry < settings.tool_plan_max_retries:
            result = ToolCheckResult(
                status="replan", next_action="tool_plan", required_tools_passed=False,
                activity_status="not_requested", issues=issues,
            )
            return _build_update(result, tool_plan_retry_count=plan_retry + 1)
        result = ToolCheckResult(
            status="failed", next_action="final_response", required_tools_passed=False,
            activity_status="not_requested", issues=issues,
        )
        return _build_update(result)

    plan_issues = plan.get("validation_issues", []) if isinstance(plan, dict) else []
    replan_issue_types = {
        "unsupported_optional_tool",
        "invalid_optional_tool_arguments",
        "optional_tool_call_limit_exceeded",
        "invalid_executable_tool_call",
    }
    needs_replan = any(
        isinstance(issue, dict) and issue.get("type") in replan_issue_types
        for issue in plan_issues
    )
    if needs_replan and plan_retry < settings.tool_plan_max_retries:
        issues.extend(issue for issue in plan_issues if isinstance(issue, dict))
        result = ToolCheckResult(
            status="replan", next_action="tool_plan", required_tools_passed=True,
            activity_status="not_requested", issues=issues,
        )
        return _build_update(result, tool_plan_retry_count=plan_retry + 1)

    failed_required = []
    for field in ("weather_fetch_meta", "flight_fetch_meta", "hotel_fetch_meta"):
        meta = state.get(field)
        if not isinstance(meta, dict) or meta.get("status") in {None, "failed"}:
            failed_required.append(field)
    if failed_required:
        issues.append({"type": "required_tool_execution_failed", "fields": failed_required})
        if execute_retry < settings.tool_execute_max_retries:
            result = ToolCheckResult(
                status="retry_execute", next_action="tool_execute", required_tools_passed=False,
                activity_status="not_requested", issues=issues,
            )
            return _build_update(result, tool_execute_retry_count=execute_retry + 1)
        result = ToolCheckResult(
            status="failed", next_action="final_response", required_tools_passed=False,
            activity_status="not_requested", issues=issues,
        )
        return _build_update(result)

    if needs_replan:
        issues.extend(issue for issue in plan_issues if isinstance(issue, dict))

    activity_requested = "search_city_activities" in planned_names
    activity_meta = state.get("activity_fetch_meta")
    activity_status = "not_requested"
    overall_status = "degraded" if needs_replan else "passed"
    if activity_requested:
        raw_status = activity_meta.get("status") if isinstance(activity_meta, dict) else "failed"
        if raw_status == "ok":
            activity_status = "success"
        elif raw_status == "no_data":
            activity_status = "no_results"
            overall_status = "degraded"
        else:
            activity_status = "degraded"
            overall_status = "degraded"
            issues.append({"type": "optional_activity_tool_degraded", "status": raw_status})

    result = ToolCheckResult(
        status=overall_status, next_action="candidate_rank", required_tools_passed=True,
        activity_status=activity_status, issues=issues,
    )
    return _build_update(result)


def route_after_tool_check(state: TravelState) -> str:
    result = state.get("tool_check_result")
    if not isinstance(result, dict):
        return "final_response"
    return str(result.get("next_action") or "final_response")


def _build_update(result: ToolCheckResult, **counters: int) -> dict[str, Any]:
    update: dict[str, Any] = {
        "tool_check_result": result.to_state_dict(),
        "trace": [{
            "node_name": "tool_check", "tool_name": "deterministic_tool_policy_check",
            "input_summary": "检查必选工具覆盖、执行结果和活动降级状态",
            "output_summary": f"status={result.status}, next={result.next_action}, issues={len(result.issues)}",
            "status": "success" if result.status in {"passed", "degraded"} else "failed",
            "latency_ms": 0, "created_at": datetime.now(timezone.utc).isoformat(),
        }],
    }
    update.update(counters)
    return update


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
