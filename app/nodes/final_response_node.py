from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from app.agents.state import TravelState
from app.formatters.final_response_formatter import (
    build_final_response,
)


def final_response_node(
    state: TravelState,
) -> dict[str, Any]:
    """
    FinalResponseNode：把内部 TravelState 转换成稳定的 API / 前端响应。

    读取 State：
        根据当前结束分支读取：
            - final_response，Clarify / SafeReject 已生成时；
            - commit_result / trip_draft；
            - cancel_result；
            - revision_plan / revision_result；
            - candidate_rank_result；
            - selection_result / adjustment_plan；
            - evidence_result；
            - proposal_result；
            - verifier_result；
            - errors。

    写入 State：
        - final_response
        - trace

    这个节点不做：
        - 不调用 LLM；
        - 不查询 MCP；
        - 不执行 RAG；
        - 不写数据库；
        - 不改变 TripDraft、Proposal 或审批状态；
        - 不返回完整 TravelState。

    为什么需要单独节点？
        内部 State 包含候选池、用户画像、Trace、RAG chunk 和错误细节，
        这些不应该直接暴露给前端。FinalResponseNode 是内部状态和外部 API
        数据合同之间的最后一道边界。
    """

    node_name = "final_response"
    started_at = perf_counter()

    try:
        # 1. Formatter 负责按当前分支选择公开响应结构。
        final_response = build_final_response(
            state
        )

        trace_item = _build_trace_item(
            status="success",
            started_at=started_at,
            input_summary=_summarize_input(
                state
            ),
            output_summary=(
                f"status={final_response.get('status')}, "
                f"type={final_response.get('type')}"
            ),
        )

        return {
            "final_response": (
                final_response
            ),
            "trace": [trace_item],
        }

    except Exception as exc:
        # 2. 即使响应格式化自身失败，也必须返回一个最小安全错误响应。
        fallback = {
            "status": "failed",
            "type": "final_response_error",
            "message": (
                "系统未能生成最终响应，请稍后重试。"
            ),
            "need_user_input": False,
            "available_actions": ["retry"],
            "workflow": state.get("workflow"),
            "trip_session_id": state.get(
                "trip_session_id"
            ),
        }

        error_item = {
            "node": node_name,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        trace_item = _build_trace_item(
            status="failed",
            started_at=started_at,
            input_summary=(
                "final response formatting failed"
            ),
            output_summary=str(exc),
        )

        return {
            "final_response": fallback,
            "errors": [error_item],
            "trace": [trace_item],
        }


def _summarize_input(
    state: TravelState,
) -> str:
    """只记录结束分支摘要，不把完整 State 写进 Trace。"""

    return (
        f"itinerary_status={state.get('itinerary_status')}, "
        f"commit_status={_status(state.get('commit_result'))}, "
        f"cancel_status={_status(state.get('cancel_result'))}, "
        f"revision_status={_status(state.get('revision_result'))}, "
        f"verifier_status={_status(state.get('verifier_result'))}"
    )


def _status(
    value: Any,
) -> Any:
    """读取 dict.status。"""

    if isinstance(value, dict):
        return value.get("status")

    return None


def _build_trace_item(
    *,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """生成统一 Trace 记录。"""

    return {
        "node_name": "final_response",
        "tool_name": "final_response_formatter",
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "latency_ms": int(
            (perf_counter() - started_at)
            * 1000
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }
