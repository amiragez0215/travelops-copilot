from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal

from app.agents.state import TravelState
from app.revisions.revision_analyzer import (
    RevisionAnalyzer,
    build_default_revision_analyzer,
)
from app.schemas.decision_schema import DecisionGateResult


RevisionAnalyzeDestination = Literal[
    "apply_revision",
    "final_response",
]


def revision_analyze_node(
    state: TravelState,
    analyzer: RevisionAnalyzer | None = None,
) -> dict[str, Any]:
    """
    RevisionAnalyzeNode：把用户的自然语言修改说明转换成受控 RevisionPlan。

    读取 State：
        - trip_request
        - proposal
        - decision_result
        - change_request

    写入 State：
        - revision_plan
        - revision_result，分析阶段只保存简短状态
        - trace
        - errors，仅程序异常时

    这个节点只做“理解修改意图”：
        - 不直接修改 trip_request；
        - 不查询天气、航班或酒店；
        - 不执行 RAG；
        - 不重新生成 Proposal；
        - 不写数据库。

    为什么不能把旧输入和修改文本拼起来重新执行 InputExtractNode？
        因为“预算增加1000”“多待两天”是相对操作。
        RevisionPlan 会显式保存 increment，后续 ApplyRevisionNode 再根据旧值
        确定性计算新值。
    """

    node_name = "revision_analyze"
    started_at = perf_counter()

    try:
        trip_request = _require_mapping(
            state.get("trip_request"),
            field_name="trip_request",
        )
        proposal = _require_mapping(
            state.get("proposal"),
            field_name="proposal",
        )
        change_request = _require_mapping(
            state.get("change_request"),
            field_name="change_request",
        )
        decision_result_data = _require_mapping(
            state.get("decision_result"),
            field_name="decision_result",
        )

        # 1. RevisionAnalyze 只能由 DecisionGate 接受 request_changes 后进入。
        decision_result = DecisionGateResult(
            **decision_result_data
        )

        if (
            decision_result.status != "accepted"
            or decision_result.accepted is not True
            or decision_result.decision != "request_changes"
            or decision_result.next_action != "revision_analyze"
        ):
            raise ValueError(
                "DecisionGate 尚未接受 request_changes，"
                "不能执行 RevisionAnalyzeNode"
            )

        # 2. 测试可注入 Fake Analyzer；正式运行使用 DeepSeek + Rule fallback。
        active_analyzer = (
            analyzer
            or build_default_revision_analyzer()
        )

        revision_plan = active_analyzer.analyze(
            trip_request=trip_request,
            change_request=change_request,
            proposal=proposal,
        )

        status = str(
            revision_plan.get("status")
            or "failed"
        )

        # 3. clarification_required 是正常业务分支，不写入 state.errors。
        revision_result = {
            "status": "analyzed",
            "analysis_status": status,
            "base_proposal_id": (
                revision_plan.get(
                    "base_proposal_id"
                )
            ),
            "summary": (
                revision_plan.get("patch", {})
                .get("summary", "")
                if isinstance(
                    revision_plan.get("patch"),
                    dict,
                )
                else ""
            ),
        }

        trace_status = (
            "success"
            if status == "ready"
            else "degraded"
        )

        trace_item = _build_trace_item(
            status=trace_status,
            started_at=started_at,
            input_summary=(
                f"proposal_id={proposal.get('proposal_id')}, "
                f"change={_shorten(str(change_request.get('raw_message') or ''))}"
            ),
            output_summary=(
                f"status={status}, "
                f"fallback_used={revision_plan.get('fallback_used')}, "
                f"question_count="
                f"{len(revision_plan.get('clarification_questions', []))}"
            ),
        )

        return {
            "revision_plan": revision_plan,
            "revision_result": revision_result,
            "trace": [trace_item],
        }

    except Exception as exc:
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
                "revision analysis failed"
            ),
            output_summary=str(exc),
        )

        return {
            "revision_plan": {
                "status": "failed",
                "strategy_version": (
                    "llm_revision_patch_v1"
                ),
                "base_proposal_id": None,
                "base_proposal_version": None,
                "base_trip_request_hash": None,
                "raw_change_request": (
                    state.get("change_request", {})
                    .get("raw_message", "")
                    if isinstance(
                        state.get("change_request"),
                        dict,
                    )
                    else ""
                ),
                "patch": {},
                "clarification_questions": [],
                "model_id": "unavailable",
                "attempt_count": 0,
                "fallback_used": False,
                "validation_errors": [str(exc)],
            },
            "revision_result": {
                "status": "failed",
                "analysis_status": "failed",
                "summary": "修改请求分析失败。",
            },
            "errors": [error_item],
            "trace": [trace_item],
        }


def route_after_revision_analyze(
    state: TravelState,
) -> RevisionAnalyzeDestination:
    """
    RevisionAnalyzeNode 后的 Conditional Edge。

    ready：
        → ApplyRevisionNode

    clarification_required / failed：
        → FinalResponseNode

    当前 v1 不在这里再增加第二个人工 interrupt，避免把修改链路设计得过重。
    """

    plan = state.get("revision_plan")

    if (
        isinstance(plan, dict)
        and plan.get("status") == "ready"
    ):
        return "apply_revision"

    return "final_response"


def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    """读取必须存在的 dict State。"""

    if not isinstance(value, dict):
        raise TypeError(
            f"state['{field_name}'] 必须是 dict"
        )

    return dict(value)


def _build_trace_item(
    *,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """生成统一 Trace 记录。"""

    return {
        "node_name": "revision_analyze",
        "tool_name": "llm_revision_patch_analyzer",
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


def _shorten(
    text: str,
    max_length: int = 120,
) -> str:
    """截断 Trace 中的修改文本。"""

    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."
