from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal

from app.agents.state import TravelState
from app.ranking.candidate_ranker import CandidateRanker


def candidate_rank_node(
    state: TravelState,
    ranker: CandidateRanker | None = None,
) -> dict[str, Any]:
    """
    CandidateRankNode：对 Mock 航班和酒店执行硬过滤、评分与排序。

    读取 State：
        - planning_context
        - raw_flight_results
        - raw_hotel_results

    写入 State：
        - flight_candidates
        - hotel_candidates
        - candidate_rank_result
        - trace
        - errors，仅程序异常时写入

    这个节点不做：
        - 不查询 MCP；
        - 不读取 RAG；
        - 不组合航班和酒店；
        - 不生成旅行方案；
        - 不预测用户真实旅行消费。

    它只回答两个问题：
        1. 单个航班对当前用户有多合适？
        2. 单个酒店对当前用户有多合适？

    下一步 BudgetOptimizeNode 会从：
        出发航班候选 × 返程航班候选 × 酒店候选

    中选择总预算内综合效用最高的组合。
    """

    node_name = "candidate_rank"
    started_at = perf_counter()
    active_ranker = ranker or CandidateRanker()

    try:
        planning_context = state.get("planning_context")
        raw_flight_results = state.get("raw_flight_results")
        raw_hotel_results = state.get("raw_hotel_results")

        if not isinstance(planning_context, dict):
            raise TypeError(
                "state['planning_context'] 必须是 dict，请先运行 PlanningContextNode"
            )

        if not isinstance(raw_flight_results, dict):
            raise TypeError(
                "state['raw_flight_results'] 必须是 dict，请先运行 FlightSearchNode"
            )

        if not isinstance(raw_hotel_results, dict):
            raise TypeError(
                "state['raw_hotel_results'] 必须是 dict，请先运行 HotelSearchNode"
            )

        # 1. CandidateRanker 是纯业务算法，不直接读写 TravelState。
        rank_output = active_ranker.rank(
            planning_context=planning_context,
            raw_flight_results=raw_flight_results,
            raw_hotel_results=raw_hotel_results,
        )

        rank_result = rank_output["candidate_rank_result"]

        # 2. 无候选属于正常业务状态，不写 state.errors。
        #    只有代码异常才进入 except。
        trace_status = (
            "success"
            if rank_result.get("status") == "ok"
            else "degraded"
        )

        trace_item = _build_trace_item(
            node_name=node_name,
            status=trace_status,
            started_at=started_at,
            input_summary=_summarize_input(
                planning_context=planning_context,
                raw_flight_results=raw_flight_results,
                raw_hotel_results=raw_hotel_results,
            ),
            output_summary=_summarize_output(rank_result),
        )

        return {
            "flight_candidates": rank_output["flight_candidates"],
            "hotel_candidates": rank_output["hotel_candidates"],
            "candidate_rank_result": rank_result,
            "trace": [trace_item],
        }

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
            input_summary="candidate ranking failed",
            output_summary=str(exc),
        )

        return {
            "flight_candidates": [],
            "hotel_candidates": [],
            "candidate_rank_result": {
                "status": "failed",
                "strategy_version": "deterministic_candidate_rank_v1",
                "input_counts": {},
                "output_counts": {},
                "rejected_flights": [],
                "rejected_hotels": [],
                "ignored_preference_keys": {},
                "ranking_config": {},
                "issues": [],
            },
            "errors": [error_item],
            "trace": [trace_item],
        }


def route_after_candidate_rank(
    state: TravelState,
) -> Literal["budget_optimize", "final_response"]:
    """
    CandidateRankNode 后的 Conditional Edge。

    路由规则：

        candidate_rank_result.status == "ok"
            → BudgetOptimizeNode

        partial / no_candidates / failed
            → FinalResponseNode

    为什么 partial 不继续组合？
        BudgetOptimize 至少需要出发航班、返程航班和酒店三个候选池。
        缺少任意必要候选时，不存在完整可行组合。
    """

    result = state.get("candidate_rank_result")

    if isinstance(result, dict) and result.get("status") == "ok":
        return "budget_optimize"

    return "final_response"


def _summarize_input(
    *,
    planning_context: dict[str, Any],
    raw_flight_results: dict[str, Any],
    raw_hotel_results: dict[str, Any],
) -> str:
    """
    生成输入摘要。

    Trace 不保存完整候选内容，避免日志过大。
    """

    request = planning_context.get("request")
    request = request if isinstance(request, dict) else {}

    outbound = raw_flight_results.get("outbound")
    return_flights = raw_flight_results.get("return")
    hotels = raw_hotel_results.get("items")

    return (
        f"destination={request.get('destination')}, "
        f"people_count={request.get('people_count')}, "
        f"room_count={request.get('room_count')}, "
        f"outbound_count={len(outbound) if isinstance(outbound, list) else 0}, "
        f"return_count={len(return_flights) if isinstance(return_flights, list) else 0}, "
        f"hotel_count={len(hotels) if isinstance(hotels, list) else 0}"
    )


def _summarize_output(
    result: dict[str, Any],
) -> str:
    """生成 CandidateRankNode 输出摘要。"""

    output_counts = result.get("output_counts")
    output_counts = output_counts if isinstance(output_counts, dict) else {}

    return (
        f"status={result.get('status')}, "
        f"outbound={output_counts.get('outbound_flights', 0)}, "
        f"return={output_counts.get('return_flights', 0)}, "
        f"hotels={output_counts.get('hotels', 0)}, "
        f"rejected_flights={len(result.get('rejected_flights', []))}, "
        f"rejected_hotels={len(result.get('rejected_hotels', []))}"
    )


def _build_trace_item(
    node_name: str,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """生成统一 Trace 记录。"""

    latency_ms = int((perf_counter() - started_at) * 1000)

    return {
        "node_name": node_name,
        "tool_name": "deterministic_candidate_ranker",
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "latency_ms": latency_ms,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
