from __future__ import annotations

from app.nodes.budget_optimize_node import (
    budget_optimize_node,
    route_after_budget_optimize,
)


def _state(total_budget: float = 4000):
    """构造 BudgetOptimizeNode 所需的最小完整 State。"""

    return {
        "planning_context": {
            "request": {
                "origin": "杭州",
                "destination": "成都",
                "days": 3,
                "nights": 2,
                "people_count": 1,
                "room_count": 1,
                "total_budget": total_budget,
            },
            "budget_plan": {
                "mode": "budget_limited",
                "total_budget": total_budget,
                "hard_limits": {
                    "total_budget": total_budget,
                    "max_transport_total": None,
                },
                "adjusted_ratios": {
                    "transport": 0.35,
                    "hotel": 0.35,
                    "food_activity": 0.30,
                },
                "soft_targets": {
                    "transport_budget": 1400,
                    "hotel_budget": 1400,
                    "food_activity_budget": 1200,
                },
                "allocation_source": "default_heuristic_v1",
            },
        },
        "flight_candidates": [
            {
                "flight_id": "f_out",
                "flight_no": "CA1001",
                "direction": "outbound",
                "query_direction": "outbound",
                "rank": 1,
                "total_score": 0.92,
                "price": 700,
                "total_price": 700,
                "depart_time": "10:00",
                "arrive_time": "12:30",
                "is_direct": True,
            },
            {
                "flight_id": "f_return",
                "flight_no": "CA1002",
                "direction": "return",
                "query_direction": "return",
                "rank": 1,
                "total_score": 0.90,
                "price": 650,
                "total_price": 650,
                "depart_time": "16:00",
                "arrive_time": "18:30",
                "is_direct": True,
            },
        ],
        "hotel_candidates": [
            {
                "hotel_id": "h1",
                "name": "成都安静酒店",
                "rank": 1,
                "total_score": 0.93,
                "price_per_night": 580,
                "estimated_total_price": 1160,
                "planned_nights": 2,
                "planned_room_count": 1,
                "rating": 4.7,
            }
        ],
    }


def test_budget_optimize_node_success():
    """正常输入应写出选择结果、预算摘要、调整状态和 Trace。"""

    result = budget_optimize_node(
        _state()
    )

    assert result["selection_result"]["status"] == "feasible"
    assert result["selection_result"]["selected_hotel_id"] == "h1"
    assert (
        result["selection_result"][
            "selected_combination"
        ]["outbound_flight_id"]
        == "f_out"
    )
    assert result["budget_result"]["status"] == "within_budget"
    assert result["adjustment_plan"]["status"] == "not_needed"
    assert result["budget_optimize_result"]["status"] == "feasible"

    assert result["trace"][0]["node_name"] == "budget_optimize"
    assert (
        result["trace"][0]["tool_name"]
        == "deterministic_budget_optimizer"
    )
    assert result["trace"][0]["status"] == "success"
    assert "errors" not in result


def test_budget_optimize_node_no_feasible_combination_is_business_status():
    """
    总预算不足时不写 errors，
    但应生成 adjustment_plan 并停止后续 RAG。
    """

    result = budget_optimize_node(
        _state(total_budget=1500)
    )

    assert (
        result["selection_result"]["status"]
        == "no_feasible_combination"
    )
    assert (
        result["adjustment_plan"]["status"]
        == "need_user_change"
    )
    assert result["trace"][0]["status"] == "degraded"
    assert "errors" not in result
    assert (
        route_after_budget_optimize(result)
        == "final_response"
    )


def test_route_after_budget_optimize():
    """Conditional Edge 只允许可行组合进入 RetrievalPlanNode。"""

    assert route_after_budget_optimize(
        {
            "selection_result": {
                "status": "feasible"
            }
        }
    ) == "retrieval_plan"

    assert route_after_budget_optimize(
        {
            "selection_result": {
                "status": "no_candidates"
            }
        }
    ) == "final_response"

    assert route_after_budget_optimize(
        {}
    ) == "final_response"


def test_budget_optimize_node_missing_planning_context_returns_error():
    """缺少 PlanningContext 属于程序调用顺序错误。"""

    state = _state()
    state.pop("planning_context")

    result = budget_optimize_node(
        state
    )

    assert result["selection_result"]["status"] == "failed"
    assert result["budget_optimize_result"]["status"] == "failed"
    assert result["errors"]
    assert result["errors"][0]["node"] == "budget_optimize"
    assert result["trace"][0]["status"] == "failed"


def test_budget_optimize_node_missing_candidates_returns_error():
    """缺少 CandidateRank 输出字段属于程序错误。"""

    state = _state()
    state.pop("flight_candidates")

    result = budget_optimize_node(
        state
    )

    assert result["selection_result"]["status"] == "failed"
    assert result["errors"]
