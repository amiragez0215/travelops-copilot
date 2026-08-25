from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from app.agents.state import TravelState
from app.builders.planning_context_builder import build_planning_context


def planning_context_node(state: TravelState) -> dict[str, Any]:
    """
    PlanningContextNode：生成统一规划上下文。

    读取 State：
        - trip_request
        - user_profile

    写入 State：
        - planning_context
        - budget_plan
        - trace
        - errors，异常时写入

    为什么要写 budget_plan 到顶层？
        planning_context["budget_plan"] 是权威结果。
        同时写 state["budget_plan"] 是为了后续 BudgetOptimizeNode 或测试读取更方便。

    这个节点不调用 LLM、不读数据库、不查 RAG、不查 mock。
    它只做确定性上下文整理。
    """

    node_name = "planning_context"
    started_at = perf_counter()

    try:
        trip_request = state.get("trip_request")
        user_profile = state.get("user_profile", {})

        if not isinstance(trip_request, dict):
            raise TypeError("state['trip_request'] 必须是 dict，请先运行 InputExtractNode")

        if user_profile is not None and not isinstance(user_profile, dict):
            raise TypeError("state['user_profile'] 必须是 dict，请先运行 MemoryReadNode")

        # Builder 负责合并 trip_request 和 user_profile。
        planning_context = build_planning_context(
            trip_request=trip_request,
            user_profile=user_profile,
        )

        budget_plan = planning_context["budget_plan"]

        trace_item = _build_trace_item(
            node_name=node_name,
            status="success",
            started_at=started_at,
            input_summary=_summarize_input(trip_request, user_profile),
            output_summary=_summarize_context(planning_context),
        )

        return {
            "planning_context": planning_context,
            "budget_plan": budget_plan,
            "trace": [trace_item],
        }

    except Exception as exc:
        error_item = {
            "node": node_name,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        trace_item = _build_trace_item(
            node_name=node_name,
            status="failed",
            started_at=started_at,
            input_summary="planning context build failed",
            output_summary=str(exc),
        )

        return {
            "errors": [error_item],
            "trace": [trace_item],
        }


def _summarize_input(
    trip_request: dict[str, Any],
    user_profile: dict[str, Any] | None,
) -> str:
    """
    生成输入摘要，避免 trace 保存完整大对象。
    """

    signal_count = len(trip_request.get("preference_signals", []))
    profile_source = None

    if isinstance(user_profile, dict):
        profile_source = user_profile.get("source")

    return (
        f"origin={trip_request.get('origin')}, "
        f"destination={trip_request.get('destination')}, "
        f"days={trip_request.get('days')}, "
        f"budget={trip_request.get('budget')}, "
        f"preference_signal_count={signal_count}, "
        f"profile_source={profile_source}"
    )


def _summarize_context(
    planning_context: dict[str, Any],
) -> str:
    """
    生成 planning_context 摘要。
    """

    request = planning_context.get("request", {})
    budget_plan = planning_context.get("budget_plan", {})
    summary = planning_context.get("context_summary", {})

    return (
        f"destination={request.get('destination')}, "
        f"nights={request.get('nights')}, "
        f"budget_mode={budget_plan.get('mode')}, "
        f"main_preferences={summary.get('main_preferences')}"
    )


def _build_trace_item(
    node_name: str,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """
    生成统一 trace 记录。
    """

    latency_ms = int((perf_counter() - started_at) * 1000)

    return {
        "node_name": node_name,
        "tool_name": None,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "latency_ms": latency_ms,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }