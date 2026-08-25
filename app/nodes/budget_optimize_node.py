from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal

from app.agents.state import TravelState
from app.optimization.budget_optimizer import (
    BudgetOptimizer,
)


def budget_optimize_node(
    state: TravelState,
    optimizer: BudgetOptimizer | None = None,
) -> dict[str, Any]:
    """
    BudgetOptimizeNode：从已评分候选中选择最佳完整旅行组合。

    读取 State：
        - planning_context
        - flight_candidates
        - hotel_candidates

    写入 State：
        - selection_result
        - budget_result
        - adjustment_plan
        - budget_optimize_result
        - trace
        - errors，仅程序异常时写入

    这个节点不做：
        - 不重新查询天气、航班或酒店；
        - 不重新给单个候选打分；
        - 不调用 RAG；
        - 不调用 LLM；
        - 不预测用户真实餐饮、购物和临时消费。

    它只回答：
        “哪个去程航班 + 返程航班 + 酒店组合，
         在组合级硬限制内，对当前用户的综合效用最高？”
    """

    node_name = "budget_optimize"
    started_at = perf_counter()

    # 测试时可以注入自定义 Optimizer；
    # 正常工作流使用默认确定性组合优化器。
    active_optimizer = (
        optimizer
        or BudgetOptimizer()
    )

    try:
        planning_context = state.get(
            "planning_context"
        )
        flight_candidates = state.get(
            "flight_candidates"
        )
        hotel_candidates = state.get(
            "hotel_candidates"
        )

        # 1. 必须先完成 PlanningContextNode。
        if not isinstance(
            planning_context,
            dict,
        ):
            raise TypeError(
                "state['planning_context'] 必须是 dict，"
                "请先运行 PlanningContextNode"
            )

        # 2. 必须先完成 CandidateRankNode 的航班排序。
        if not isinstance(
            flight_candidates,
            list,
        ):
            raise TypeError(
                "state['flight_candidates'] 必须是 list，"
                "请先运行 CandidateRankNode"
            )

        # 3. 必须先完成 CandidateRankNode 的酒店排序。
        if not isinstance(
            hotel_candidates,
            list,
        ):
            raise TypeError(
                "state['hotel_candidates'] 必须是 list，"
                "请先运行 CandidateRankNode"
            )

        # 4. Optimizer 只处理纯业务算法，不直接读写 TravelState。
        optimize_output = active_optimizer.optimize(
            planning_context=(
                planning_context
            ),
            flight_candidates=(
                flight_candidates
            ),
            hotel_candidates=(
                hotel_candidates
            ),
        )

        optimize_result = optimize_output[
            "budget_optimize_result"
        ]

        # 5. 无可行组合属于正常业务状态，不写 errors。
        #    只有程序异常才进入 except。
        trace_status = (
            "success"
            if optimize_result.get("status")
            == "feasible"
            else "degraded"
        )

        trace_item = _build_trace_item(
            node_name=node_name,
            status=trace_status,
            started_at=started_at,
            input_summary=_summarize_input(
                planning_context=(
                    planning_context
                ),
                flight_candidates=(
                    flight_candidates
                ),
                hotel_candidates=(
                    hotel_candidates
                ),
            ),
            output_summary=_summarize_output(
                optimize_result
            ),
        )

        return {
            "selection_result": (
                optimize_output[
                    "selection_result"
                ]
            ),
            "budget_result": (
                optimize_output[
                    "budget_result"
                ]
            ),
            "adjustment_plan": (
                optimize_output[
                    "adjustment_plan"
                ]
            ),
            "budget_optimize_result": (
                optimize_result
            ),
            "trace": [trace_item],
        }

    except Exception as exc:
        # 6. 程序异常返回统一失败结构，便于 workflow 安全停止。
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
            input_summary=(
                "budget optimization failed"
            ),
            output_summary=str(exc),
        )

        return {
            "selection_result": {
                "status": "failed",
                "strategy_version": (
                    "deterministic_budget_optimize_v1"
                ),
                "budget_mode": (
                    "budget_limited"
                ),
                "selected_combination": None,
                "selected_outbound_flight": {},
                "selected_return_flight": {},
                "selected_hotel": {},
                "selected_hotel_id": None,
                "selected_costs": None,
                "total_budget": None,
                "remaining_budget": None,
                "remaining_budget_allocation": None,
                "alternatives": [],
                "notes": [],
            },
            "budget_result": {
                "status": "failed",
                "budget_mode": (
                    "budget_limited"
                ),
                "total_budget": None,
                "selected_known_subtotal": None,
                "remaining_budget": None,
                "known_cost_note": (
                    "BudgetOptimizeNode 执行失败，"
                    "没有生成费用组合结果。"
                ),
            },
            "adjustment_plan": {
                "status": "failed",
                "reason_code": (
                    "optimizer_exception"
                ),
                "current_total_budget": None,
                "minimum_known_subtotal": None,
                "required_budget_increase": None,
                "cheapest_combination": {},
                "suggestions": [],
            },
            "budget_optimize_result": {
                "status": "failed",
                "strategy_version": (
                    "deterministic_budget_optimize_v1"
                ),
                "budget_mode": (
                    "budget_limited"
                ),
                "input_counts": {},
                "candidate_limits": {},
                "evaluated_combination_count": 0,
                "feasible_combination_count": 0,
                "rejected_by_total_budget_count": 0,
                "rejected_by_transport_limit_count": 0,
                "selected_combination_id": None,
                "selected_score": None,
                "minimum_known_subtotal": None,
                "scoring_config": {},
                "issues": [],
            },
            "errors": [error_item],
            "trace": [trace_item],
        }


def route_after_budget_optimize(
    state: TravelState,
) -> Literal[
    "retrieval_plan",
    "final_response",
]:
    """
    BudgetOptimizeNode 后的 Conditional Edge。

    路由规则：

        selection_result.status == "feasible"
            → RetrievalPlanNode

        no_candidates / no_feasible_combination / failed
            → FinalResponseNode

    为什么不自动重新调整预算？

        系统不能擅自提高用户预算，也不能偷偷放松硬约束。
        没有可行组合时，adjustment_plan 只提供建议，
        应由用户确认后再进入 Modify / Replan。
    """

    result = state.get(
        "selection_result"
    )

    if (
        isinstance(result, dict)
        and result.get("status")
        == "feasible"
    ):
        return "retrieval_plan"

    return "final_response"


def _summarize_input(
    *,
    planning_context: dict[str, Any],
    flight_candidates: list[dict[str, Any]],
    hotel_candidates: list[dict[str, Any]],
) -> str:
    """
    生成 BudgetOptimizeNode 输入摘要。

    Trace 不保存完整候选对象，避免日志过大。
    """

    budget_plan = planning_context.get(
        "budget_plan"
    )
    budget_plan = (
        budget_plan
        if isinstance(budget_plan, dict)
        else {}
    )

    outbound_count = sum(
        1
        for item in flight_candidates
        if isinstance(item, dict)
        and (
            item.get("direction")
            or item.get("query_direction")
        )
        == "outbound"
    )
    return_count = sum(
        1
        for item in flight_candidates
        if isinstance(item, dict)
        and (
            item.get("direction")
            or item.get("query_direction")
        )
        == "return"
    )

    return (
        f"budget_mode={budget_plan.get('mode')}, "
        f"total_budget={budget_plan.get('total_budget')}, "
        f"outbound_count={outbound_count}, "
        f"return_count={return_count}, "
        f"hotel_count={len(hotel_candidates)}"
    )


def _summarize_output(
    result: dict[str, Any],
) -> str:
    """生成 BudgetOptimizeNode 输出摘要。"""

    return (
        f"status={result.get('status')}, "
        f"evaluated={result.get('evaluated_combination_count')}, "
        f"feasible={result.get('feasible_combination_count')}, "
        f"over_budget={result.get('rejected_by_total_budget_count')}, "
        f"transport_limit={result.get('rejected_by_transport_limit_count')}, "
        f"selected={result.get('selected_combination_id')}"
    )


def _build_trace_item(
    node_name: str,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """生成统一 Trace 记录。"""

    latency_ms = int(
        (
            perf_counter()
            - started_at
        )
        * 1000
    )

    return {
        "node_name": node_name,
        "tool_name": (
            "deterministic_budget_optimizer"
        ),
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "latency_ms": latency_ms,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }
