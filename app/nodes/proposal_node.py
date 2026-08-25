from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal

from app.agents.state import TravelState
from app.proposals.proposal_generator import (
    PROPOSAL_GENERATOR_VERSION,
    PROPOSAL_PROMPT_VERSION,
    ProposalGenerationError,
    ProposalGenerator,
    build_default_proposal_generator,
)


def proposal_node(
    state: TravelState,
    generator: ProposalGenerator | None = None,
) -> dict[str, Any]:
    """
    ProposalNode：使用锁定事实、逐日天气和经过验证的 RAG 证据生成旅行方案。

    读取 State：
        - trip_request
        - planning_context
        - selection_result
        - weather_result
        - evidence_pool；如果不存在则回退读取 reranked_evidence
        - evidence_result
        - proposal_feedback，可选；Verifier 跨节点修复反馈

    写入 State：
        - proposal
        - current_proposal
        - proposal_result
        - proposal_feedback = {}，成功使用后清空
        - verifier_result = {}，等待重新验证
        - itinerary_status = "proposed"
        - trace
        - errors，程序或模型生成失败时写入

    这个节点不做：
        - 不重新选择航班或酒店；
        - 不重新计算预算；
        - 不重新执行 RAG；
        - 不执行真实预订、支付或通知；
        - 不允许 LLM 修改 selection_result 中的锁定事实。

    核心数据流：

        selection_result + weather_result
            → 锁定事实

        evidence_pool + planning_context
            → LLM 可生成上下文

        ProposalDraft
            → Pydantic + 业务规则校验

        锁定事实 + ProposalDraft
            → 最终 TripProposal
    """

    node_name = "proposal"
    started_at = perf_counter()

    try:
        trip_request = state.get("trip_request")
        planning_context = state.get("planning_context")
        selection_result = state.get("selection_result")
        weather_result = state.get("weather_result")
        evidence_result = state.get("evidence_result")

        # VerifierNode 回到 ProposalNode 时，会把跨节点修复问题写入这里。
        # 首次生成没有该字段，按空 dict 处理。
        proposal_feedback = state.get("proposal_feedback")
        proposal_feedback = (
            proposal_feedback
            if isinstance(proposal_feedback, dict)
            else {}
        )

        evidence_pool = state.get("evidence_pool")

        # 1. EvidenceGradeNode 会把累计证据写入 evidence_pool。
        #    为兼容早期单元测试，如果 evidence_pool 缺失，回退读取 reranked_evidence。
        if not isinstance(evidence_pool, list):
            fallback_evidence = state.get("reranked_evidence")
            evidence_pool = (
                fallback_evidence
                if isinstance(fallback_evidence, list)
                else None
            )

        # 2. 在创建真实 DeepSeek Client 之前先完成 State 类型检查。
        #    这样缺少上游字段时，不会因为 API Key 或网络配置掩盖真正错误。
        if not isinstance(trip_request, dict):
            raise TypeError(
                "state['trip_request'] 必须是 dict，请先运行 InputExtractNode"
            )

        if not isinstance(planning_context, dict):
            raise TypeError(
                "state['planning_context'] 必须是 dict，请先运行 PlanningContextNode"
            )

        if not isinstance(selection_result, dict):
            raise TypeError(
                "state['selection_result'] 必须是 dict，请先运行 BudgetOptimizeNode"
            )

        if selection_result.get("status") != "feasible":
            raise ValueError(
                "selection_result.status 必须为 feasible，当前没有可生成方案的完整组合"
            )

        if not isinstance(weather_result, dict):
            raise TypeError(
                "state['weather_result'] 必须是 dict，请先运行 WeatherNode"
            )

        if not isinstance(evidence_pool, list):
            raise TypeError(
                "state['evidence_pool'] 必须是 list，请先运行 EvidenceGradeNode"
            )

        if not isinstance(evidence_result, dict):
            raise TypeError(
                "state['evidence_result'] 必须是 dict，请先运行 EvidenceGradeNode"
            )

        if evidence_result.get("passed") is not True:
            raise ValueError(
                "evidence_result.passed 必须为 True，不能使用未通过验收的 RAG 证据"
            )

        # 3. 版本号由工作流状态确定，不交给 LLM。
        #    初次为 1；Revision 后为 base_proposal_version + 1。
        proposal_version = _resolve_proposal_version(state)

        # 4. 测试可注入 Fake Generator；正式运行使用 DeepSeek JSON Generator。
        active_generator = generator or build_default_proposal_generator()

        # 5. Generator 负责 Prompt、LLM 调用、Schema 校验和最终组装。
        output = active_generator.generate(
            trip_request=trip_request,
            planning_context=planning_context,
            selection_result=selection_result,
            weather_result=weather_result,
            evidence_pool=evidence_pool,
            evidence_result=evidence_result,
            verifier_feedback=proposal_feedback,
            proposal_version=proposal_version,
        )

        proposal = output["proposal"]
        proposal_result = output["proposal_result"]

        if not isinstance(proposal, dict):
            raise TypeError("ProposalGenerator 返回的 proposal 必须是 dict")

        if not isinstance(proposal_result, dict):
            raise TypeError("ProposalGenerator 返回的 proposal_result 必须是 dict")

        if proposal_result.get("status") != "generated":
            raise ValueError("ProposalGenerator 没有返回 generated 状态")

        trace_item = _build_trace_item(
            node_name=node_name,
            status="success",
            started_at=started_at,
            input_summary=_summarize_input(
                trip_request=trip_request,
                selection_result=selection_result,
                weather_result=weather_result,
                evidence_pool=evidence_pool,
            ),
            output_summary=_summarize_output(
                proposal=proposal,
                proposal_result=proposal_result,
            ),
        )

        return {
            "proposal": proposal,

            # 5. 在当前工作流内把新方案同时设置为 current_proposal。
            #    后续 ModifyTripWorkflow 可以直接读取，不必依赖聊天文本重建旧方案。
            "current_proposal": proposal,

            "proposal_result": proposal_result,

            # 6. 当前反馈已经用于本轮生成，成功后清空，
            #    避免下一次普通生成重复携带旧 Verifier 问题。
            "proposal_feedback": {},

            # 7. 清空上一版 Verifier 结果。
            #    新 Proposal 必须重新进入 VerifierNode 独立审查。
            "verifier_result": {},

            "pending_actions": [],
            "itinerary_status": "proposed",
            "trace": [trace_item],
        }

    except Exception as exc:
        validation_errors = (
            exc.validation_errors
            if isinstance(exc, ProposalGenerationError)
            else []
        )

        attempt_count = (
            exc.attempt_count
            if isinstance(exc, ProposalGenerationError)
            else 0
        )

        error_item = {
            "node": node_name,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "validation_errors": validation_errors,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        trace_item = _build_trace_item(
            node_name=node_name,
            status="failed",
            started_at=started_at,
            input_summary="proposal generation failed",
            output_summary=str(exc),
        )

        return {
            "proposal": {},
            "proposal_result": {
                "status": "failed",
                "generator_version": PROPOSAL_GENERATOR_VERSION,
                "prompt_version": PROPOSAL_PROMPT_VERSION,
                "model_id": getattr(generator, "model_id", "unavailable"),
                "attempt_count": attempt_count,
                "max_attempts": max(attempt_count, 1),
                "evidence_input_count": 0,
                "evidence_prompt_count": 0,
                "used_evidence_count": 0,
                "used_evidence_refs": [],
                "locked_fact_hash": None,
                "proposal_id": None,
                "daily_plan_count": 0,
                "validation_errors": validation_errors,
                "issues": [
                    {
                        "issue_type": "proposal_generation_failed",
                        "message": str(exc),
                    }
                ],
            },
            "errors": [error_item],
            "trace": [trace_item],
        }



def _resolve_proposal_version(state: TravelState) -> int:
    """
    根据当前工作流状态确定 Proposal 版本号。

    - 首次规划：1；
    - RevisionApplyResult.status == applied：base_proposal_version + 1；
    - Verifier 打回同一版重新生成：版本保持不变，不再次递增。
    """
    revision_result = state.get("revision_result")
    if isinstance(revision_result, dict) and revision_result.get("status") == "applied":
        base_version = _safe_int(revision_result.get("base_proposal_version"), default=0)
        if base_version >= 1:
            return base_version + 1
    revision_plan = state.get("revision_plan")
    if isinstance(revision_plan, dict) and revision_plan.get("status") == "ready":
        base_version = _safe_int(revision_plan.get("base_proposal_version"), default=0)
        if base_version >= 1:
            return base_version + 1
    return 1


def _safe_int(value: Any, *, default: int) -> int:
    """安全转换 int，避免脏 State 破坏版本计算。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def route_after_proposal(
    state: TravelState,
) -> Literal[
    "verifier",
    "final_response",
]:
    """
    ProposalNode 后的 Conditional Edge。

    路由规则：
        proposal_result.status == "generated"
            → VerifierNode

        其他状态或字段缺失
            → FinalResponseNode

    采用安全默认值：
        未经 ProposalGenerator 校验的输出不会继续进入 Verifier。
    """

    result = state.get("proposal_result")

    if (
        isinstance(result, dict)
        and result.get("status") == "generated"
    ):
        return "verifier"

    return "final_response"


def _summarize_input(
    *,
    trip_request: dict[str, Any],
    selection_result: dict[str, Any],
    weather_result: dict[str, Any],
    evidence_pool: list[dict[str, Any]],
) -> str:
    """生成不包含大段 Evidence 正文的输入摘要。"""

    selected_hotel_id = selection_result.get("selected_hotel_id")

    combination = selection_result.get("selected_combination")
    combination = combination if isinstance(combination, dict) else {}

    if not selected_hotel_id:
        selected_hotel_id = combination.get("hotel_id")

    daily_weather = weather_result.get("daily")
    daily_weather_count = (
        len(daily_weather)
        if isinstance(daily_weather, list)
        else 0
    )

    return (
        f"destination={trip_request.get('destination')}, "
        f"days={trip_request.get('days')}, "
        f"combination_id={combination.get('combination_id')}, "
        f"selected_hotel_id={selected_hotel_id}, "
        f"weather_day_count={daily_weather_count}, "
        f"evidence_count={len(evidence_pool)}"
    )


def _summarize_output(
    *,
    proposal: dict[str, Any],
    proposal_result: dict[str, Any],
) -> str:
    """生成 ProposalNode 输出摘要。"""

    daily_plan = proposal.get("daily_plan")
    daily_plan_count = (
        len(daily_plan)
        if isinstance(daily_plan, list)
        else 0
    )

    return (
        f"status={proposal_result.get('status')}, "
        f"proposal_id={proposal.get('proposal_id')}, "
        f"daily_plan_count={daily_plan_count}, "
        f"used_evidence_count={proposal_result.get('used_evidence_count')}, "
        f"attempt_count={proposal_result.get('attempt_count')}"
    )


def _build_trace_item(
    node_name: str,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """生成统一 Trace 记录。"""

    return {
        "node_name": node_name,
        "tool_name": "deepseek_structured_proposal",
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "latency_ms": int(
            (perf_counter() - started_at) * 1000
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }
