from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from app.agents.state import TravelState
from app.formatters.clarification_formatter import (
    build_clarification,
    build_clarification_final_response,
)
from app.validators.trip_request_validator import (
    REQUIRED_FIELD_SPECS,
    build_missing_info_message,
    build_missing_info_result,
)


def clarify_node(state: TravelState) -> dict[str, Any]:
    """
    ClarifyNode：缺少必填字段时生成追问。
    """

    node_name = "clarify"
    started_at = perf_counter()

    try:
        # 优先使用 MissingInfoCheckNode 的完整结果。
        missing_info_result = _resolve_missing_info_result(state)

        # formatter 负责把结构化缺失信息转成用户可读问题。
        clarification = build_clarification(
            missing_info_result=missing_info_result,
        )

        final_response = build_clarification_final_response(clarification)

        trace_item = _build_trace_item(
            node_name=node_name,
            status="success",
            started_at=started_at,
            input_summary=f"missing_fields={clarification.get('missing_fields', [])}",
            output_summary=clarification.get("message", ""),
        )

        return {
            "clarification": clarification,
            "final_response": final_response,
            "itinerary_status": "collecting_info",
            "trace": [trace_item],
        }

    except Exception as exc:
        fallback_clarification = {
            "type": "missing_info",
            "status": "needs_clarification",
            "message": "还需要补充旅行信息，请重新说明你的出发地、目的地、出发日期和旅行天数。",
            "missing_fields": [],
            "questions": [
                {
                    "field": "travel_request",
                    "label": "旅行需求",
                    "question": "请重新说明你的出发地、目的地、出发日期和旅行天数。",
                    "reason": "当前旅行需求结构不完整，无法继续规划。",
                    "example": "例如：2026-07-02 从杭州去成都玩 3 天。",
                }
            ],
            "trip_request_summary": {},
            "next_action": "provide_missing_info",
            "available_actions": ["provide_missing_info", "cancel"],
        }

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
            input_summary="clarification build failed",
            output_summary=str(exc),
        )

        return {
            "clarification": fallback_clarification,
            "final_response": build_clarification_final_response(fallback_clarification),
            "itinerary_status": "collecting_info",
            "errors": [error_item],
            "trace": [trace_item],
        }


def _resolve_missing_info_result(state: TravelState) -> dict[str, Any]:
    """
    从 State 中解析 missing_info_result，必要时从 trip_request 兜底生成。
    """

    result = state.get("missing_info_result")
    if isinstance(result, dict):
        return result

    trip_request_data = state.get("trip_request")
    if trip_request_data:
        return build_missing_info_result(trip_request_data)

    if "missing_fields" in state:
        missing_fields = state.get("missing_fields") or []

        if not isinstance(missing_fields, list):
            raise TypeError("state['missing_fields'] 必须是 list")

        missing_details = []
        for field in missing_fields:
            spec = REQUIRED_FIELD_SPECS.get(str(field))
            if spec:
                missing_details.append(spec.to_dict())

        can_continue = len(missing_fields) == 0

        return {
            "status": "ready" if can_continue else "missing_required_fields",
            "required_fields": list(REQUIRED_FIELD_SPECS.keys()),
            "missing_fields": [str(field) for field in missing_fields],
            "missing_details": missing_details,
            "can_continue": can_continue,
            "message": build_missing_info_message([str(field) for field in missing_fields]),
            "trip_request_summary": {},
        }

    raise ValueError("state 中缺少 missing_info_result、missing_fields 或 trip_request")


def _build_trace_item(
    node_name: str,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """
    生成统一 trace item。
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