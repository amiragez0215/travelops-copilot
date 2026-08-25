from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from app.agents.state import TravelState
from app.common.config import settings
from app.mcp.client_factory import create_travel_data_client, describe_travel_data_client
from app.mcp.travel_data_client import TravelDataClient
from app.nodes.flight_node import flight_search_node
from app.nodes.hotel_node import hotel_search_node
from app.nodes.weather_node import weather_node
from app.observability.harness import ToolExecutor


def tool_execute_node(
    state: TravelState,
    client: TravelDataClient | None = None,
) -> dict[str, Any]:
    """
    执行 ToolPlan，并保持原天气/航班/酒店 State 契约完全不变。

    已成功且查询签名一致的必选工具不会在 ToolCheck 重试或重新规划后重复
    执行；活动是只读查询，失败时记录降级状态而不阻断 Rank/Budget 主链。
    """

    started_at = perf_counter()
    plan = state.get("tool_plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("calls"), list):
        raise ValueError("state['tool_plan'] 缺失或无效")

    active_client = client or create_travel_data_client()
    client_info = describe_travel_data_client(active_client)
    update: dict[str, Any] = {"trace": [], "errors": []}
    calls = [item for item in plan["calls"] if isinstance(item, dict)]

    for call in calls:
        tool_name = str(call.get("tool_name") or "")
        arguments = dict(call.get("arguments") or {})
        if tool_name == "get_weather":
            if not _is_cached(state, "weather_fetch_meta", arguments):
                _merge_update(update, weather_node(state, client=active_client))
        elif tool_name == "search_flights":
            if not _is_cached(state, "flight_fetch_meta", arguments):
                _merge_update(update, flight_search_node(state, client=active_client))
        elif tool_name == "search_hotels":
            if not _is_cached(state, "hotel_fetch_meta", arguments):
                _merge_update(update, hotel_search_node(state, client=active_client))
        elif tool_name == "search_city_activities":
            if not _is_cached(state, "activity_fetch_meta", arguments):
                _merge_update(
                    update,
                    _execute_activity(active_client, client_info, arguments),
                )

    # LLM 没选择活动时显式写 skipped，避免空列表被误解为“工具查询失败”。
    activity_requested = any(call.get("tool_name") == "search_city_activities" for call in calls)
    if not activity_requested:
        update.update({
            "activity_result": {"status": "skipped", "items": [], "source": "policy:not_requested"},
            "activity_candidates": [],
            "activity_fetch_meta": {"tool_name": "search_city_activities", "status": "skipped"},
        })

    # ProposalGenerator 已经接收 planning_context。把规范化活动候选作为新增
    # 上下文字段写入，可以避免修改 Rank/Budget 及 ProposalGenerator 公共签名。
    planning_context = dict(state.get("planning_context") or {})
    candidates = update.get("activity_candidates", state.get("activity_candidates", []))
    planning_context["activity_candidates"] = list(candidates or [])
    update["planning_context"] = planning_context
    update["tool_execution_result"] = {
        "status": "completed",
        "executed_call_count": len(calls),
        "retry_count": _safe_int(state.get("tool_execute_retry_count")),
    }
    update["trace"].append({
        "node_name": "tool_execute",
        "tool_name": "policy_constrained_dispatcher",
        "input_summary": f"planned_calls={len(calls)}",
        "output_summary": f"activity_requested={activity_requested}",
        "status": "success",
        "latency_ms": int((perf_counter() - started_at) * 1000),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    if not update["errors"]:
        update.pop("errors")
    return update


def _execute_activity(
    client: TravelDataClient,
    client_info: dict[str, Any],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    started_at = perf_counter()
    try:
        result = ToolExecutor(timeout_seconds=settings.mcp_travel_data_timeout_seconds).execute(
            name="search_city_activities",
            operation_type=("mcp" if client_info["protocol"] == "mcp" else "tool"),
            input_value=arguments,
            model_or_tool=f"{client_info['protocol']}:search_city_activities",
            callback=lambda: client.search_city_activities(**arguments),
        )
        items = result.get("items", []) if isinstance(result, dict) else []
        return {
            "activity_result": result,
            "activity_candidates": list(items) if isinstance(items, list) else [],
            "activity_fetch_meta": {
                "tool_name": "search_city_activities", "status": result.get("status"),
                "source": result.get("source"), **client_info, "query_signature": arguments,
            },
            "trace": [{
                "node_name": "tool_execute", "tool_name": f"{client_info['protocol']}:search_city_activities",
                "input_summary": f"city={arguments.get('city')}, interests={len(arguments.get('interests', []))}",
                "output_summary": f"activities={len(items)}, status={result.get('status')}",
                "status": "success", "latency_ms": int((perf_counter() - started_at) * 1000),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }],
        }
    except Exception as exc:
        return {
            "activity_result": {"status": "failed", "items": [], "error": str(exc)},
            "activity_candidates": [],
            "activity_fetch_meta": {"tool_name": "search_city_activities", "status": "failed", **client_info, "error": str(exc), "query_signature": arguments},
            "errors": [{"node": "tool_execute", "type": exc.__class__.__name__, "message": str(exc), "created_at": datetime.now(timezone.utc).isoformat()}],
        }


def _is_cached(state: TravelState, meta_field: str, arguments: dict[str, Any]) -> bool:
    meta = state.get(meta_field)
    if not isinstance(meta, dict) or meta.get("status") in {None, "failed"}:
        return False
    signature = meta.get("query_signature")
    return isinstance(signature, dict) and all(signature.get(key) == value for key, value in arguments.items())


def _merge_update(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if key in {"trace", "errors"}:
            target.setdefault(key, []).extend(value if isinstance(value, list) else [])
        else:
            target[key] = value


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
