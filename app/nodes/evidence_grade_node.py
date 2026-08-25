from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal

from app.agents.state import TravelState
from app.rag.evidence_grader import (
    EvidenceGrader,
)


def evidence_grade_node(
    state: TravelState,
    grader: EvidenceGrader | None = None,
) -> dict[str, Any]:
    """
    EvidenceGradeNode：检查 RAG 证据是否完整、合法且可追溯。

    读取 State：
        - retrieval_plan
        - reranked_evidence
        - evidence_pool，可选
        - evidence_requirements，可选
        - weather_result，可选
        - rerank_result，可选
        - rag_retry_count，可选

    写入 State：
        - evidence_pool
        - evidence_requirements
        - reranked_evidence
        - evidence_result
        - retrieval_feedback
        - trace
        - errors，仅程序异常时写入

    注意：

        证据不足不是程序异常。

        因此：

            evidence_result.status == "repair_required"

        不会写入 state["errors"]。

        errors 只用于：
            - State 类型错误
            - Grader 内部异常
            - 代码执行失败
    """

    node_name = "evidence_grade"
    started_at = perf_counter()

    if grader is None:
        # 与 Planner 使用同一个运行模式；Agentic Grader 内部先执行 Rule Gate，
        # 只有结构和 metadata 通过后才调用 LLM Semantic Judge。
        from app.rag.agentic_runtime import get_default_evidence_grader

        active_grader = get_default_evidence_grader()
    else:
        active_grader = grader
    grader_tool_name = (
        "agentic_evidence_grader"
        if active_grader.__class__.__name__ == "AgenticEvidenceGrader"
        else "rule_evidence_grader"
    )

    try:
        retrieval_plan = state.get(
            "retrieval_plan"
        )

        reranked_evidence = state.get(
            "reranked_evidence"
        )

        if not isinstance(
            retrieval_plan,
            dict,
        ):
            raise TypeError(
                "state['retrieval_plan'] 必须是 dict，"
                "请先运行 RetrievalPlanNode"
            )

        if not isinstance(
            reranked_evidence,
            list,
        ):
            raise TypeError(
                "state['reranked_evidence'] 必须是 list，"
                "请先运行 RerankNode"
            )

        evidence_pool = state.get(
            "evidence_pool"
        )

        evidence_requirements = state.get(
            "evidence_requirements"
        )

        weather_result = state.get(
            "weather_result"
        )

        rerank_result = state.get(
            "rerank_result"
        )

        rag_retry_count = _safe_int(
            state.get(
                "rag_retry_count"
            ),
            default=0,
        )

        # 1. 调用当前 EvidenceGrader。Agentic 实现内部依然先过 Rule Gate。
        grade_output = active_grader.grade(
            retrieval_plan=retrieval_plan,
            reranked_evidence=(
                reranked_evidence
            ),
            existing_evidence_pool=(
                evidence_pool
                if isinstance(
                    evidence_pool,
                    list,
                )
                else []
            ),
            existing_requirements=(
                evidence_requirements
                if isinstance(
                    evidence_requirements,
                    dict,
                )
                else None
            ),
            weather_result=(
                weather_result
                if isinstance(
                    weather_result,
                    dict,
                )
                else {}
            ),
            rerank_result=(
                rerank_result
                if isinstance(
                    rerank_result,
                    dict,
                )
                else {}
            ),
            rag_retry_count=(
                rag_retry_count
            ),
        )

        accumulated_pool = grade_output[
            "evidence_pool"
        ]

        result = grade_output[
            "evidence_result"
        ]

        # 2. 将累计 Evidence Pool 写回 reranked_evidence。
        #
        #    这样后续 ProposalNode 和 VerifierNode
        #    不需要区分“初次证据”和“修复证据”，
        #    只读取 state["reranked_evidence"] 即可。
        update: dict[str, Any] = {
            "evidence_pool":
                accumulated_pool,

            "evidence_requirements":
                grade_output[
                    "evidence_requirements"
                ],

            "reranked_evidence":
                accumulated_pool,

            "evidence_result":
                result,

            # 3. 通过时写入空 dict，
            #    清除之前可能遗留的 repair feedback。
            "retrieval_feedback":
                grade_output.get(
                    "retrieval_feedback"
                )
                or {},
        }

        # 4. 根据 Evidence Grade 状态设置 Trace 状态。
        if result["status"] == "passed":
            trace_status = "success"

        elif result["status"] in {
            "degraded",
            "repair_required",
        }:
            trace_status = "degraded"

        else:
            trace_status = "failed"

        trace_item = _build_trace_item(
            node_name=node_name,
            tool_name=grader_tool_name,
            status=trace_status,
            started_at=started_at,
            input_summary=_summarize_input(
                retrieval_plan=(
                    retrieval_plan
                ),
                reranked_evidence=(
                    reranked_evidence
                ),
                evidence_pool=(
                    evidence_pool
                ),
                rag_retry_count=(
                    rag_retry_count
                ),
            ),
            output_summary=_summarize_output(
                result
            ),
        )

        update["trace"] = [
            trace_item
        ]

        return update

    except Exception as exc:
        error_item = {
            "node":
                node_name,

            "type":
                exc.__class__.__name__,

            "message":
                str(exc),

            "created_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        trace_item = _build_trace_item(
            node_name=node_name,
            tool_name=grader_tool_name,
            status="failed",
            started_at=started_at,
            input_summary=(
                "evidence grading failed"
            ),
            output_summary=str(exc),
        )

        return {
            "evidence_result": {
                "status": "failed",
                "passed": False,
                "repairable": False,
                "next_action": "stop",
                "strategy_version":
                    "rule_evidence_grade_v1",
                "plan_id": None,
                "mode": "initial",
                "graded_evidence_count": 0,
                "rejected_evidence_count": 0,
                "required_category_count": 0,
                "passed_required_category_count": 0,
                "fallback_used": False,
                "coverage": {},
                "issues": [],
                "retrieval_feedback": None,
            },

            "retrieval_feedback": {},

            "errors": [
                error_item
            ],

            "trace": [
                trace_item
            ],
        }


def route_after_evidence(
    state: TravelState,
) -> Literal[
    "proposal",
    "retrieval_plan",
    "final_response",
]:
    """
    EvidenceGradeNode 后的 Conditional Edge。

    路由规则：

        next_action == "continue"
            → ProposalNode

        next_action == "repair"
            → RetrievalPlanNode

        next_action == "stop"
            → FinalResponseNode

    如果 evidence_result 缺失或结构异常，
    默认走 final_response，避免工作流继续使用未验证证据。
    """

    result = state.get(
        "evidence_result"
    )

    if not isinstance(
        result,
        dict,
    ):
        return "final_response"

    next_action = result.get(
        "next_action"
    )

    if next_action == "continue":
        return "proposal"

    if next_action == "repair":
        return "retrieval_plan"

    return "final_response"


def _summarize_input(
    retrieval_plan: dict[str, Any],
    reranked_evidence: list[dict[str, Any]],
    evidence_pool: Any,
    rag_retry_count: int,
) -> str:
    """
    生成 EvidenceGradeNode 输入摘要。

    不将完整 evidence 正文写入 Trace。
    """

    old_pool_count = (
        len(evidence_pool)
        if isinstance(
            evidence_pool,
            list,
        )
        else 0
    )

    return (
        f"plan_id={retrieval_plan.get('plan_id')}, "
        f"mode={retrieval_plan.get('mode')}, "
        f"latest_evidence_count={len(reranked_evidence)}, "
        f"existing_pool_count={old_pool_count}, "
        f"rag_retry_count={rag_retry_count}"
    )


def _summarize_output(
    result: dict[str, Any],
) -> str:
    """
    生成 EvidenceGradeNode 输出摘要。
    """

    semantic = result.get("semantic_review")
    semantic_status = (
        semantic.get("status")
        if isinstance(semantic, dict)
        else "not_run"
    )
    return (
        f"status={result.get('status')}, "
        f"passed={result.get('passed')}, "
        f"repairable={result.get('repairable')}, "
        f"next_action={result.get('next_action')}, "
        f"graded_evidence={result.get('graded_evidence_count')}, "
        f"rejected_evidence={result.get('rejected_evidence_count')}, "
        f"semantic={semantic_status}"
    )


def _safe_int(
    value: Any,
    default: int,
) -> int:
    """
    安全转换 int。
    """

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _build_trace_item(
    node_name: str,
    tool_name: str,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """
    生成统一 Trace 记录。
    """

    latency_ms = int(
        (
            perf_counter()
            - started_at
        )
        * 1000
    )

    return {
        "node_name":
            node_name,

        "tool_name":
            tool_name,

        "input_summary":
            input_summary,

        "output_summary":
            output_summary,

        "status":
            status,

        "latency_ms":
            latency_ms,

        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }
