from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal

from app.agents.state import TravelState
from app.common.config import settings
from app.verifiers.proposal_verifier import (
    ProposalVerifier,
    ProposalVerifierConfig,
)


def verifier_node(
    state: TravelState,
    verifier: ProposalVerifier | None = None,
) -> dict[str, Any]:
    """
    VerifierNode：在人工审批前独立审查最终 TripProposal。

    读取 State：
        - proposal
        - proposal_result
        - trip_request
        - planning_context
        - selection_result
        - weather_result
        - evidence_pool
        - evidence_result
        - flight_candidates
        - hotel_candidates
        - retrieval_plan，可选，用于 RAG repair 次数保护
        - proposal_retry_count，可选
        - budget_retry_count，可选
        - rag_retry_count，可选

    写入 State：
        - verifier_result
        - proposal_feedback
        - pending_actions
        - proposal_retry_count，回 ProposalNode 时递增
        - budget_retry_count，回 BudgetOptimizeNode 时递增
        - trace
        - errors，仅 Verifier 程序异常时写入

    重要边界：
        Verifier 发现业务问题属于正常工作流结果，
        例如雨天安排不合理、Evidence 引用错误或预算不一致，
        这些不会写入 state.errors。

        只有类型错误、代码异常或 Verifier 自身执行失败才写 errors。
    """

    node_name = "verifier"
    started_at = perf_counter()

    try:
        proposal = state.get("proposal")
        proposal_result = state.get("proposal_result")
        trip_request = state.get("trip_request")
        planning_context = state.get("planning_context")
        selection_result = state.get("selection_result")
        weather_result = state.get("weather_result")
        evidence_pool = state.get("evidence_pool")
        evidence_result = state.get("evidence_result")
        flight_candidates = state.get("flight_candidates")
        hotel_candidates = state.get("hotel_candidates")

        # 1. 在创建默认 Verifier 前完成 State 类型检查。
        #    这样错误信息会明确指出缺少哪个上游节点结果。
        required_dicts = {
            "proposal": proposal,
            "proposal_result": proposal_result,
            "trip_request": trip_request,
            "planning_context": planning_context,
            "selection_result": selection_result,
            "weather_result": weather_result,
            "evidence_result": evidence_result,
        }

        for field_name, value in required_dicts.items():
            if not isinstance(value, dict):
                raise TypeError(
                    f"state['{field_name}'] 必须是 dict，"
                    "请确认 ProposalNode 及其上游节点已经完成"
                )

        required_lists = {
            "evidence_pool": evidence_pool,
            "flight_candidates": flight_candidates,
            "hotel_candidates": hotel_candidates,
        }

        for field_name, value in required_lists.items():
            if not isinstance(value, list):
                raise TypeError(
                    f"state['{field_name}'] 必须是 list，"
                    "请确认对应上游节点已经完成"
                )

        proposal_retry_count = _safe_int(
            state.get("proposal_retry_count"),
            default=0,
        )
        budget_retry_count = _safe_int(
            state.get("budget_retry_count"),
            default=0,
        )
        rag_retry_count = _safe_int(
            state.get("rag_retry_count"),
            default=0,
        )

        retrieval_plan = state.get("retrieval_plan")
        retrieval_plan = (
            retrieval_plan
            if isinstance(retrieval_plan, dict)
            else None
        )

        # 2. 默认 Verifier 使用项目 Settings；测试可以注入自定义实例。
        active_verifier = verifier or ProposalVerifier(
            ProposalVerifierConfig(
                float_tolerance=(
                    settings.verifier_float_tolerance
                ),
                max_proposal_repairs=(
                    settings.verifier_max_proposal_repairs
                ),
                max_budget_repairs=(
                    settings.verifier_max_budget_repairs
                ),
                max_issue_examples=(
                    settings.verifier_max_issue_examples
                ),
            )
        )

        # 3. ProposalVerifier 负责所有确定性检查和修复路由决策。
        output = active_verifier.verify(
            proposal=proposal,
            proposal_result=proposal_result,
            trip_request=trip_request,
            planning_context=planning_context,
            selection_result=selection_result,
            weather_result=weather_result,
            evidence_pool=evidence_pool,
            evidence_result=evidence_result,
            flight_candidates=flight_candidates,
            hotel_candidates=hotel_candidates,
            proposal_retry_count=proposal_retry_count,
            budget_retry_count=budget_retry_count,
            retrieval_plan=retrieval_plan,
            rag_retry_count=rag_retry_count,
        )

        verifier_result = output["verifier_result"]
        next_action = verifier_result["next_action"]

        update: dict[str, Any] = {
            "verifier_result": verifier_result,
            "proposal_feedback": output.get(
                "proposal_feedback",
                {},
            ),
            "pending_actions": output.get(
                "pending_actions",
                [],
            ),
        }

        # 4. 只有真正回到 ProposalNode 时才增加 proposal_retry_count。
        #    ProposalGenerator 内部的 PROPOSAL_MAX_ATTEMPTS 不使用这个字段；
        #    这里记录的是跨节点 Proposal → Verifier → Proposal 修复次数。
        if next_action == "proposal":
            update["proposal_retry_count"] = (
                proposal_retry_count + 1
            )

        # 5. 选择组合过期时回 BudgetOptimize，并增加组合修复次数。
        if next_action == "budget_optimize":
            update["budget_retry_count"] = (
                budget_retry_count + 1
            )

        trace_status = (
            "success"
            if verifier_result["status"] == "passed"
            else (
                "degraded"
                if verifier_result["status"] == "repair_required"
                else "failed"
            )
        )

        trace_item = _build_trace_item(
            node_name=node_name,
            status=trace_status,
            started_at=started_at,
            input_summary=_summarize_input(
                proposal=proposal,
                evidence_pool=evidence_pool,
                proposal_retry_count=proposal_retry_count,
                budget_retry_count=budget_retry_count,
                rag_retry_count=rag_retry_count,
            ),
            output_summary=_summarize_output(
                verifier_result
            ),
        )

        update["trace"] = [trace_item]
        return update

    except Exception as exc:
        # 6. 只有 Verifier 程序执行失败才写入 errors。
        error_item = {
            "node": node_name,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        trace_item = _build_trace_item(
            node_name=node_name,
            status="failed",
            started_at=started_at,
            input_summary="proposal verification failed",
            output_summary=str(exc),
        )

        return {
            "verifier_result": {
                "status": "failed",
                "passed": False,
                "repairable": False,
                "next_action": "final_response",
                "strategy_version": (
                    "deterministic_proposal_verifier_v1"
                ),
                "proposal_id": None,
                "expected_locked_fact_hash": None,
                "actual_locked_fact_hash": None,
                "proposal_retry_count": _safe_int(
                    state.get("proposal_retry_count"),
                    default=0,
                ),
                "max_proposal_repairs": (
                    settings.verifier_max_proposal_repairs
                ),
                "budget_retry_count": _safe_int(
                    state.get("budget_retry_count"),
                    default=0,
                ),
                "max_budget_repairs": (
                    settings.verifier_max_budget_repairs
                ),
                "approval_required": False,
                "check_count": 0,
                "passed_check_count": 0,
                "warning_count": 0,
                "error_count": 0,
                "critical_count": 1,
                "checks": {},
                "issues": [
                    {
                        "issue_type": "verifier_execution_failed",
                        "severity": "critical",
                        "repair_target": "final_response",
                        "check_name": "node_execution",
                        "message": str(exc),
                        "path": None,
                        "expected": None,
                        "actual": None,
                        "details": {},
                    }
                ],
                "repair_feedback": {},
            },
            "proposal_feedback": {},
            "pending_actions": [],
            "errors": [error_item],
            "trace": [trace_item],
        }


def route_after_verifier(
    state: TravelState,
) -> Literal[
    "decision_gate",
    "proposal",
    "retrieval_plan",
    "budget_optimize",
    "final_response",
]:
    """
    VerifierNode 后的 Conditional Edge。

    路由：
        decision_gate
            最终 Proposal 通过，等待用户 approve/request_changes/cancel。

        proposal
            模型生成内容、引用、天气适配或最终组装需要修复。

        retrieval_plan
            Evidence 状态失效，需要重新执行 RAG repair。

        budget_optimize
            选中组合或费用状态过期，需要重新组合。

        final_response
            无法自动修复或重试次数已经耗尽。

    安全默认：
        verifier_result 缺失或 next_action 非法时进入 final_response。
    """

    result = state.get("verifier_result")

    if not isinstance(result, dict):
        return "final_response"

    next_action = result.get("next_action")

    if next_action in {
        "decision_gate",
        "proposal",
        "retrieval_plan",
        "budget_optimize",
    }:
        return next_action  # type: ignore[return-value]

    return "final_response"


def _summarize_input(
    *,
    proposal: dict[str, Any],
    evidence_pool: list[dict[str, Any]],
    proposal_retry_count: int,
    budget_retry_count: int,
    rag_retry_count: int,
) -> str:
    """生成不包含完整 Proposal 和 Evidence 正文的输入摘要。"""

    daily_plan = proposal.get("daily_plan")
    daily_plan_count = (
        len(daily_plan)
        if isinstance(daily_plan, list)
        else 0
    )

    return (
        f"proposal_id={proposal.get('proposal_id')}, "
        f"daily_plan_count={daily_plan_count}, "
        f"evidence_count={len(evidence_pool)}, "
        f"proposal_retry_count={proposal_retry_count}, "
        f"budget_retry_count={budget_retry_count}, "
        f"rag_retry_count={rag_retry_count}"
    )


def _summarize_output(
    result: dict[str, Any],
) -> str:
    """生成 VerifierNode 输出摘要。"""

    return (
        f"status={result.get('status')}, "
        f"passed={result.get('passed')}, "
        f"next_action={result.get('next_action')}, "
        f"passed_checks={result.get('passed_check_count')}/"
        f"{result.get('check_count')}, "
        f"warnings={result.get('warning_count')}, "
        f"errors={result.get('error_count')}, "
        f"critical={result.get('critical_count')}"
    )


def _safe_int(
    value: Any,
    default: int,
) -> int:
    """安全转换 int。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_trace_item(
    node_name: str,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """生成统一 Trace 记录。"""

    latency_ms = int(
        (perf_counter() - started_at) * 1000
    )

    return {
        "node_name": node_name,
        "tool_name": "deterministic_proposal_verifier",
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "latency_ms": latency_ms,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }
