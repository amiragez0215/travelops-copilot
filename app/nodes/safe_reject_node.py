from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from app.agents.state import TravelState
from app.formatters.safe_reject_formatter import build_safe_reject_final_response


def safe_reject_node(state: TravelState) -> dict[str, Any]:
    """
    SafeRejectNode：安全检查阻断后，生成安全拒绝响应。
    """

    node_name = "safe_reject"
    started_at = perf_counter()

    try:
        safety_result = state.get("safety_result")
        safety_flags = state.get("safety_flags", [])

        if not isinstance(safety_result, dict):
            raise ValueError("state 中缺少 safety_result，无法生成安全拒绝响应")

        if safety_result.get("blocked") is not True:
            raise ValueError("safety_result.blocked 不是 True，不应该进入 SafeRejectNode")

        # formatter 负责生成用户可读响应。
        final_response = build_safe_reject_final_response(
            safety_result=safety_result,
            safety_flags=safety_flags,
        )

        trace_item = _build_trace_item(
            node_name=node_name,
            status="success",
            started_at=started_at,
            input_summary=_summarize_safety_result(safety_result),
            output_summary=final_response["message"],
        )

        return {
            "final_response": final_response,
            "itinerary_status": "rejected",
            "trace": [trace_item],
        }

    except Exception as exc:
        fallback_response = build_safe_reject_final_response(
            safety_result={
                "blocked": True,
                "reason": "安全拒绝响应生成失败，系统不会继续执行旅行规划。",
                "safe_mode": "reject_or_draft_only",
                "flags": [],
                "blocked_actions": ["system_error"],
            },
            safety_flags=[],
        )

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
            input_summary="safe reject build failed",
            output_summary=str(exc),
        )

        return {
            "final_response": fallback_response,
            "itinerary_status": "rejected",
            "errors": [error_item],
            "trace": [trace_item],
        }


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
        "output_summary": _shorten(output_summary),
        "status": status,
        "latency_ms": latency_ms,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _summarize_safety_result(safety_result: dict[str, Any]) -> str:
    """
    生成 safety_result 的 trace 摘要。
    """

    flags = safety_result.get("flags", [])
    blocked_actions = safety_result.get("blocked_actions", [])

    return (
        f"blocked={safety_result.get('blocked')}, "
        f"safe_mode={safety_result.get('safe_mode')}, "
        f"flag_count={len(flags) if isinstance(flags, list) else 0}, "
        f"blocked_actions={blocked_actions}"
    )


def _shorten(text: str, max_len: int = 120) -> str:
    """
    截断过长 trace 文本。
    """

    if len(text) <= max_len:
        return text

    return text[: max_len - 3] + "..."