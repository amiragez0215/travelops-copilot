from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal

from app.agents.state import TravelState
from app.validators.trip_request_validator import build_missing_info_result


def missing_info_check_node(state: TravelState) -> dict[str, Any]:
    """
    MissingInfoCheckNode：检查 trip_request 是否缺少必填字段。
    """

    node_name = "missing_info_check"
    started_at = perf_counter()

    try:
        # 当前 State 只允许从 trip_request 读取结构化需求。
        trip_request_data = state.get("trip_request")

        if not trip_request_data:
            raise ValueError("state 中缺少 trip_request，请先运行 InputExtractNode")

        # validator 负责具体字段校验。
        missing_info_result = build_missing_info_result(trip_request_data)

        missing_fields = missing_info_result["missing_fields"]
        can_continue = missing_info_result["can_continue"]
        capability_result = missing_info_result.get("capability_result", {})
        blocking_issues = missing_info_result.get("blocking_issues", [])

        trace_item = _build_trace_item(
            node_name=node_name,
            status="success",
            started_at=started_at,
            input_summary=_summarize_trip_request(trip_request_data),
            output_summary=(
                f"can_continue={can_continue}, "
                f"missing_fields={missing_fields}, "
                f"blocking_issue_count={len(blocking_issues)}, "
                f"capability_status={capability_result.get('status')}"
            ),
        )

        update: dict[str, Any] = {
            "missing_fields": missing_fields,
            "can_continue": can_continue,
            "missing_info_result": missing_info_result,
            "capability_result": capability_result,
            "trace": [trace_item],
        }

        # 缺字段时进入 collecting_info 状态。
        if not can_continue:
            update["itinerary_status"] = "collecting_info"

        return update

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
            input_summary="trip_request validation failed",
            output_summary=str(exc),
        )

        return {
            "missing_fields": ["trip_request"],
            "can_continue": False,
            "missing_info_result": {
                "status": "invalid_trip_request",
                "required_fields": ["origin", "destination", "start_date", "days"],
                "missing_fields": ["trip_request"],
                "missing_details": [],
                "validation_issues": [],
                "blocking_issues": [],
                "capability_result": {
                    "status": "not_checked",
                    "supported": False,
                    "issues": [],
                },
                "clarification_questions": [],
                "can_continue": False,
                "message": "旅行需求结构无效，请重新输入旅行需求。",
            },
            "capability_result": {
                "status": "not_checked",
                "supported": False,
                "issues": [],
            },
            "itinerary_status": "collecting_info",
            "errors": [error_item],
            "trace": [trace_item],
        }


def route_after_missing_info(
    state: TravelState,
) -> Literal["clarify", "safety_check"]:
    """
    Conditional Edge 使用：缺信息走 clarify，完整走 safety_check。
    """

    if state.get("can_continue") is True and not state.get("missing_fields"):
        return "safety_check"

    return "clarify"


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


def _summarize_trip_request(trip_request_data: Any) -> str:
    """
    为 trace 生成 trip_request 摘要。
    """

    if not isinstance(trip_request_data, dict):
        return f"type={type(trip_request_data).__name__}"

    return (
        f"origin={trip_request_data.get('origin')}, "
        f"destination={trip_request_data.get('destination')}, "
        f"start_date={trip_request_data.get('start_date')}, "
        f"days={trip_request_data.get('days')}, "
        f"budget={trip_request_data.get('budget')}"
    )