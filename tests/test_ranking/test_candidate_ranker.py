from __future__ import annotations

from app.ranking.candidate_ranker import CandidateRanker


def _planning_context():
    """构造 CandidateRanker 使用的统一规划上下文。"""

    return {
        "request": {
            "origin": "杭州",
            "destination": "成都",
            "start_date": "2026-07-02",
            "end_date": "2026-07-04",
            "days": 3,
            "nights": 2,
            "people_count": 1,
            "room_count": 1,
        },
        "preference_weights": {
            "flight": {
                "avoid_early_flight": 1.0,
                "prefer_direct": 0.8,
                "price_sensitive": 0.7,
            },
            "hotel": {
                "quiet": 1.0,
                "cleanliness": 0.9,
                "near_subway": 0.6,
                "high_rating": 0.7,
                "budget_friendly": 0.5,
            },
        },
        "hard_constraints": {
            "trip": [
                {
                    "domain": "trip",
                    "field": "total_budget",
                    "operator": "<=",
                    "value": 4000,
                }
            ],
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
    }


def _flight_results():
    """构造两组往返航班。"""

    return {
        "status": "ok",
        "return_date": "2026-07-04",
        "outbound": [
            {
                "flight_id": "f_early",
                "flight_no": "CA1001",
                "airline": "示例航空",
                "departure_city": "杭州",
                "arrival_city": "成都",
                "depart_date": "2026-07-02",
                "depart_time": "06:00",
                "arrive_date": "2026-07-02",
                "arrive_time": "08:40",
                "duration_minutes": 160,
                "price": 500,
                "available_seats": 5,
                "is_direct": True,
                "refundable": False,
                "changeable": True,
                "query_direction": "outbound",
            },
            {
                "flight_id": "f_comfort",
                "flight_no": "CA1002",
                "airline": "示例航空",
                "departure_city": "杭州",
                "arrival_city": "成都",
                "depart_date": "2026-07-02",
                "depart_time": "10:00",
                "arrive_date": "2026-07-02",
                "arrive_time": "12:40",
                "duration_minutes": 160,
                "price": 680,
                "available_seats": 5,
                "is_direct": True,
                "refundable": True,
                "changeable": True,
                "query_direction": "outbound",
            },
        ],
        "return": [
            {
                "flight_id": "f_return",
                "flight_no": "CA2001",
                "airline": "示例航空",
                "departure_city": "成都",
                "arrival_city": "杭州",
                "depart_date": "2026-07-04",
                "depart_time": "16:00",
                "arrive_date": "2026-07-04",
                "arrive_time": "18:40",
                "duration_minutes": 160,
                "price": 650,
                "available_seats": 5,
                "is_direct": True,
                "refundable": False,
                "changeable": True,
                "query_direction": "return",
            }
        ],
    }


def _hotel_results():
    """构造价格和安静度存在取舍的酒店。"""

    return {
        "status": "ok",
        "nights": 2,
        "items": [
            {
                "hotel_id": "h_quiet",
                "name": "安静酒店",
                "city": "成都",
                "price_per_night": 580,
                "rating": 4.6,
                "available_rooms": 2,
                "near_subway": True,
                "distance_to_subway_meters": 300,
                "quiet_score": 0.95,
                "cleanliness_score": 0.90,
                "tags": ["安静"],
                "amenities": ["WiFi"],
            },
            {
                "hotel_id": "h_cheap_noisy",
                "name": "便宜热闹酒店",
                "city": "成都",
                "price_per_night": 420,
                "rating": 4.5,
                "available_rooms": 3,
                "near_subway": True,
                "distance_to_subway_meters": 150,
                "quiet_score": 0.40,
                "cleanliness_score": 0.86,
                "tags": ["商业中心"],
                "amenities": ["WiFi"],
            },
        ],
    }


def test_subjective_quiet_preference_ranks_but_does_not_filter():
    """
    “一定要安静”对应 quiet=1.0。

    两家酒店都应该保留，但更安静的酒店排名更高。
    """

    output = CandidateRanker().rank(
        planning_context=_planning_context(),
        raw_flight_results=_flight_results(),
        raw_hotel_results=_hotel_results(),
    )

    hotels = output["hotel_candidates"]

    assert len(hotels) == 2
    assert hotels[0]["hotel_id"] == "h_quiet"
    assert hotels[0]["rank"] == 1
    assert hotels[0]["total_score"] > hotels[1]["total_score"]

    # 较吵酒店不会被过滤，但会明确记录高权重安静偏好差距。
    noisy_gaps = hotels[1]["preference_gaps"]
    assert any(
        gap["preference_key"] == "quiet"
        for gap in noisy_gaps
    )

    assert output["candidate_rank_result"]["rejected_hotels"] == []


def test_objective_hard_constraints_filter_candidates():
    """必须直飞和酒店每晚上限属于可验证硬约束，应直接过滤。"""

    context = _planning_context()
    context["hard_constraints"]["flight"] = [
        {
            "domain": "flight",
            "field": "is_direct",
            "operator": "==",
            "value": True,
            "evidence": "必须直飞",
        }
    ]
    context["hard_constraints"]["hotel"] = [
        {
            "domain": "hotel",
            "field": "price_per_night",
            "operator": "<=",
            "value": 500,
            "evidence": "酒店每晚不能超过500",
        }
    ]

    flights = _flight_results()
    flights["outbound"].append(
        {
            "flight_id": "f_stop",
            "flight_no": "KN3001",
            "airline": "示例航空",
            "departure_city": "杭州",
            "arrival_city": "成都",
            "depart_date": "2026-07-02",
            "depart_time": "11:00",
            "arrive_date": "2026-07-02",
            "arrive_time": "16:00",
            "duration_minutes": 300,
            "price": 400,
            "available_seats": 5,
            "is_direct": False,
            "refundable": False,
            "changeable": True,
            "query_direction": "outbound",
        }
    )

    output = CandidateRanker().rank(
        planning_context=context,
        raw_flight_results=flights,
        raw_hotel_results=_hotel_results(),
    )

    flight_ids = {
        item["flight_id"]
        for item in output["flight_candidates"]
    }
    hotel_ids = {
        item["hotel_id"]
        for item in output["hotel_candidates"]
    }

    assert "f_stop" not in flight_ids
    assert hotel_ids == {"h_cheap_noisy"}

    assert any(
        item["candidate_id"] == "f_stop"
        and item["reason_code"] == "hard_constraint_violation"
        for item in output["candidate_rank_result"]["rejected_flights"]
    )

    assert any(
        item["candidate_id"] == "h_quiet"
        and item["reason_code"] == "hard_constraint_violation"
        for item in output["candidate_rank_result"]["rejected_hotels"]
    )


def test_avoid_early_flight_preference_can_outweigh_lower_price():
    """高权重避免早班机偏好可以让舒适航班排在便宜早班机之前。"""

    output = CandidateRanker().rank(
        planning_context=_planning_context(),
        raw_flight_results=_flight_results(),
        raw_hotel_results=_hotel_results(),
    )

    outbound = [
        item
        for item in output["flight_candidates"]
        if item["direction"] == "outbound"
    ]

    assert outbound[0]["flight_id"] == "f_comfort"
    assert outbound[0]["score_breakdown"]["time_fit"] > (
        outbound[1]["score_breakdown"]["time_fit"]
    )


def test_required_named_hotel_keeps_only_named_candidate():
    """required 指定酒店属于客观实体限制，只保留名称匹配的酒店。"""

    context = _planning_context()
    context["named_constraints"] = [
        {
            "entity_type": "hotel",
            "entity_name": "安静酒店",
            "constraint_mode": "required",
            "scope": "whole_trip",
        }
    ]

    output = CandidateRanker().rank(
        planning_context=context,
        raw_flight_results=_flight_results(),
        raw_hotel_results=_hotel_results(),
    )

    assert [
        item["hotel_id"]
        for item in output["hotel_candidates"]
    ] == ["h_quiet"]

    assert any(
        item["candidate_id"] == "h_cheap_noisy"
        and item["reason_code"] == "required_named_entity_mismatch"
        for item in output["candidate_rank_result"]["rejected_hotels"]
    )


def test_hotel_total_price_uses_nights_and_room_count():
    """酒店计划总价必须乘以住宿晚数和房间数。"""

    context = _planning_context()
    context["request"]["room_count"] = 2

    hotels = _hotel_results()
    hotels["items"][0]["available_rooms"] = 2
    hotels["items"][1]["available_rooms"] = 1

    output = CandidateRanker().rank(
        planning_context=context,
        raw_flight_results=_flight_results(),
        raw_hotel_results=hotels,
    )

    assert len(output["hotel_candidates"]) == 1

    hotel = output["hotel_candidates"][0]
    assert hotel["hotel_id"] == "h_quiet"
    assert hotel["estimated_total_price"] == 580 * 2 * 2

    assert any(
        item["candidate_id"] == "h_cheap_noisy"
        and item["reason_code"] == "insufficient_rooms"
        for item in output["candidate_rank_result"]["rejected_hotels"]
    )


def test_no_candidates_is_business_status_not_exception():
    """所有候选为空时返回 no_candidates，而不是抛程序异常。"""

    output = CandidateRanker().rank(
        planning_context=_planning_context(),
        raw_flight_results={
            "status": "no_candidate",
            "outbound": [],
            "return": [],
        },
        raw_hotel_results={
            "status": "no_candidate",
            "items": [],
        },
    )

    assert output["flight_candidates"] == []
    assert output["hotel_candidates"] == []
    assert output["candidate_rank_result"]["status"] == "no_candidates"

def test_round_trip_target_does_not_depend_on_candidate_availability():
    """
    往返交通软目标的拆分不能依赖实际查到了几组航班。

    当前 v1 固定支持往返旅行：

        交通总软目标 1500
        ↓
        去程参考目标 750
        返程参考目标 750

    即使 Provider 没有返回返程候选，
    也不能把整个 1500 元目标都分配给去程。
    """

    flights = _flight_results()

    # 1. 模拟返程查询没有找到任何候选。
    flights["return"] = []

    # 2. 同时删除 return_date，验证 CandidateRank 不依赖 Provider 字段
    #    判断本次旅行是否需要返程。
    flights.pop(
        "return_date",
        None,
    )

    output = CandidateRanker().rank(
        planning_context=_planning_context(),
        raw_flight_results=flights,
        raw_hotel_results=_hotel_results(),
    )

    result = output[
        "candidate_rank_result"
    ]

    ranking_config = result[
        "ranking_config"
    ]

    # 3. 当前产品边界始终是往返旅行。
    assert (
        ranking_config["trip_type"]
        == "round_trip"
    )

    assert (
        ranking_config[
            "expected_flight_leg_count"
        ]
        == 2
    )

    assert (
        ranking_config["return_required"]
        is True
    )

    # 4. 1500 元往返软目标固定平均拆成两个航段。
    assert (
        ranking_config[
            "per_leg_transport_target"
        ]
        == 750
    )

    # 5. 没有返程候选时不能进入 BudgetOptimize。
    assert result["status"] == "partial"

    assert any(
        issue["issue_type"]
        == "no_return_flight_candidates"
        for issue in result["issues"]
    )