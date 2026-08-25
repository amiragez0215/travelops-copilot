from __future__ import annotations

from app.nodes.candidate_rank_node import (
    candidate_rank_node,
    route_after_candidate_rank,
)


def _state():
    """构造 CandidateRankNode 所需的最小完整 State。"""

    return {
        "planning_context": {
            "request": {
                "origin": "杭州",
                "destination": "成都",
                "days": 3,
                "nights": 2,
                "people_count": 1,
                "room_count": 1,
            },
            "preference_weights": {
                "flight": {
                    "avoid_early_flight": 0.9,
                    "prefer_direct": 0.8,
                },
                "hotel": {
                    "quiet": 1.0,
                    "cleanliness": 0.9,
                },
            },
            "hard_constraints": {
                "trip": [],
                "flight": [],
                "hotel": [],
            },
            "named_constraints": [],
            "budget_plan": {
                "mode": "budget_limited",
                "soft_targets": {
                    "transport_budget": 1500,
                    "target_hotel_price_per_room_night": 600,
                },
            },
        },
        "raw_flight_results": {
            "status": "ok",
            "return_date": "2026-07-04",
            "outbound": [
                {
                    "flight_id": "f1",
                    "flight_no": "CA1001",
                    "departure_city": "杭州",
                    "arrival_city": "成都",
                    "depart_time": "10:00",
                    "arrive_time": "12:30",
                    "price": 700,
                    "available_seats": 3,
                    "is_direct": True,
                    "refundable": False,
                    "changeable": True,
                    "query_direction": "outbound",
                }
            ],
            "return": [
                {
                    "flight_id": "f2",
                    "flight_no": "CA1002",
                    "departure_city": "成都",
                    "arrival_city": "杭州",
                    "depart_time": "16:00",
                    "arrive_time": "18:30",
                    "price": 650,
                    "available_seats": 3,
                    "is_direct": True,
                    "refundable": False,
                    "changeable": True,
                    "query_direction": "return",
                }
            ],
        },
        "raw_hotel_results": {
            "status": "ok",
            "items": [
                {
                    "hotel_id": "h1",
                    "name": "成都安静酒店",
                    "city": "成都",
                    "price_per_night": 580,
                    "rating": 4.7,
                    "available_rooms": 2,
                    "near_subway": True,
                    "distance_to_subway_meters": 300,
                    "quiet_score": 0.92,
                    "cleanliness_score": 0.90,
                    "tags": ["安静"],
                    "amenities": ["WiFi"],
                }
            ],
        },
    }


def test_candidate_rank_node_success():
    """正常输入应写出排序候选、执行摘要和 Trace。"""

    result = candidate_rank_node(_state())

    assert len(result["flight_candidates"]) == 2
    assert len(result["hotel_candidates"]) == 1
    assert result["candidate_rank_result"]["status"] == "ok"

    assert result["hotel_candidates"][0]["hotel_id"] == "h1"
    assert result["hotel_candidates"][0]["rank"] == 1

    assert result["trace"][0]["node_name"] == "candidate_rank"
    assert (
        result["trace"][0]["tool_name"]
        == "deterministic_candidate_ranker"
    )
    assert result["trace"][0]["status"] == "success"
    assert "errors" not in result


def test_candidate_rank_node_does_not_read_rag_evidence():
    """
    CandidateRankNode 的结构化选择不能被 RAG 证据改变。

    即使 State 中存在高度正向或负向的 reranked_evidence，
    节点仍然只读取 Mock + PlanningContext。
    """

    state_without_rag = _state()
    state_with_rag = _state()
    state_with_rag["reranked_evidence"] = [
        {
            "hotel_id": "h1",
            "content": "这是一条极端负面评价。",
            "rerank_score": 0.99,
        }
    ]

    result_without_rag = candidate_rank_node(state_without_rag)
    result_with_rag = candidate_rank_node(state_with_rag)

    assert (
        result_without_rag["hotel_candidates"]
        == result_with_rag["hotel_candidates"]
    )

    assert (
        result_without_rag["flight_candidates"]
        == result_with_rag["flight_candidates"]
    )


def test_candidate_rank_node_missing_planning_context_returns_error():
    """缺少 PlanningContext 时返回程序错误。"""

    state = _state()
    state.pop("planning_context")

    result = candidate_rank_node(state)

    assert result["flight_candidates"] == []
    assert result["hotel_candidates"] == []
    assert result["candidate_rank_result"]["status"] == "failed"
    assert result["errors"]
    assert result["errors"][0]["node"] == "candidate_rank"
    assert result["trace"][0]["status"] == "failed"


def test_candidate_rank_node_empty_candidates_is_not_program_error():
    """没有候选属于正常业务状态，不写 errors。"""

    state = _state()
    state["raw_flight_results"] = {
        "status": "no_candidate",
        "outbound": [],
        "return": [],
    }
    state["raw_hotel_results"] = {
        "status": "no_candidate",
        "items": [],
    }

    result = candidate_rank_node(state)

    assert result["candidate_rank_result"]["status"] == "no_candidates"
    assert result["trace"][0]["status"] == "degraded"
    assert "errors" not in result


def test_route_after_candidate_rank():
    """只有完整候选池才进入 BudgetOptimizeNode。"""

    assert route_after_candidate_rank(
        {"candidate_rank_result": {"status": "ok"}}
    ) == "budget_optimize"

    assert route_after_candidate_rank(
        {"candidate_rank_result": {"status": "partial"}}
    ) == "final_response"

    assert route_after_candidate_rank(
        {"candidate_rank_result": {"status": "no_candidates"}}
    ) == "final_response"

    assert route_after_candidate_rank({}) == "final_response"
