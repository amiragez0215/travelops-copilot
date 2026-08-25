from __future__ import annotations

from app.optimization.budget_optimizer import (
    BudgetOptimizeConfig,
    BudgetOptimizer,
)


def _planning_context(
    *,
    total_budget: float | None = 4000,
    adjusted_ratios: dict[str, float] | None = None,
    transport_target: float | None = 1400,
    hotel_target: float | None = 1400,
    food_activity_target: float | None = 1200,
    max_transport_total: float | None = None,
):
    """构造 BudgetOptimizer 使用的最小 PlanningContext。"""

    budget_mode = (
        "budget_limited"
        if total_budget is not None
        else "no_budget_limit"
    )

    return {
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
            "mode": budget_mode,
            "total_budget": total_budget,
            "hard_limits": {
                "total_budget": total_budget,
                "max_transport_total": (
                    max_transport_total
                ),
            },
            "adjusted_ratios": (
                adjusted_ratios
                or {
                    "transport": 0.35,
                    "hotel": 0.35,
                    "food_activity": 0.30,
                }
            ),
            "soft_targets": {
                "transport_budget": (
                    transport_target
                ),
                "hotel_budget": hotel_target,
                "food_activity_budget": (
                    food_activity_target
                ),
            },
            "allocation_source": (
                "default_heuristic_v1"
            ),
        },
    }


def _flight(
    flight_id: str,
    *,
    direction: str,
    score: float,
    total_price: float,
    rank: int,
):
    """构造一个已通过 CandidateRankNode 的航班。"""

    return {
        "flight_id": flight_id,
        "flight_no": flight_id.upper(),
        "direction": direction,
        "query_direction": direction,
        "rank": rank,
        "total_score": score,
        "price": total_price,
        "total_price": total_price,
        "depart_time": "10:00",
        "arrive_time": "12:30",
        "is_direct": True,
    }


def _hotel(
    hotel_id: str,
    *,
    score: float,
    total_price: float,
    rank: int,
):
    """构造一个已通过 CandidateRankNode 的酒店。"""

    return {
        "hotel_id": hotel_id,
        "name": hotel_id,
        "rank": rank,
        "total_score": score,
        "price_per_night": total_price / 2,
        "estimated_total_price": total_price,
        "planned_nights": 2,
        "planned_room_count": 1,
        "rating": 4.5,
    }


def test_optimizer_selects_best_feasible_combination_not_top_individuals():
    """
    三个单项排名第一的候选组合可能超预算。

    BudgetOptimizer 应从其他组合中选择得分最高的可行解，
    而不是机械地拿三个 rank=1 候选拼在一起。
    """

    flights = [
        _flight(
            "out_high",
            direction="outbound",
            score=0.96,
            total_price=1000,
            rank=1,
        ),
        _flight(
            "out_affordable",
            direction="outbound",
            score=0.84,
            total_price=600,
            rank=2,
        ),
        _flight(
            "return_high",
            direction="return",
            score=0.95,
            total_price=900,
            rank=1,
        ),
        _flight(
            "return_affordable",
            direction="return",
            score=0.82,
            total_price=500,
            rank=2,
        ),
    ]
    hotels = [
        _hotel(
            "hotel_high",
            score=0.96,
            total_price=1600,
            rank=1,
        ),
        _hotel(
            "hotel_affordable",
            score=0.86,
            total_price=1000,
            rank=2,
        ),
    ]

    output = BudgetOptimizer().optimize(
        planning_context=_planning_context(
            total_budget=3000,
            food_activity_target=800,
        ),
        flight_candidates=flights,
        hotel_candidates=hotels,
    )

    selection = output["selection_result"]

    assert selection["status"] == "feasible"
    assert (
        selection["selected_combination"][
            "combination_score"
        ]
        <= 1.0
    )
    assert (
        selection["selected_costs"][
            "known_subtotal"
        ]
        <= 3000
    )

    # rank=1 的三个候选合计 3500，不能成为最终选择。
    selected_ids = {
        selection["selected_combination"][
            "outbound_flight_id"
        ],
        selection["selected_combination"][
            "return_flight_id"
        ],
        selection["selected_combination"][
            "hotel_id"
        ],
    }

    assert selected_ids != {
        "out_high",
        "return_high",
        "hotel_high",
    }
    assert (
        output["budget_optimize_result"][
            "rejected_by_total_budget_count"
        ]
        > 0
    )


def test_soft_category_targets_do_not_filter_feasible_combination():
    """
    航班费用超过 transport soft target 不应被直接过滤。

    只要 known_subtotal 没有超过真正 total_budget，
    组合仍然可以进入最终选择。
    """

    flights = [
        _flight(
            "out",
            direction="outbound",
            score=0.9,
            total_price=800,
            rank=1,
        ),
        _flight(
            "return",
            direction="return",
            score=0.9,
            total_price=700,
            rank=1,
        ),
    ]
    hotels = [
        _hotel(
            "hotel",
            score=0.9,
            total_price=1000,
            rank=1,
        )
    ]

    output = BudgetOptimizer().optimize(
        planning_context=_planning_context(
            total_budget=4000,
            transport_target=1200,
        ),
        flight_candidates=flights,
        hotel_candidates=hotels,
    )

    selected = output[
        "selection_result"
    ]["selected_combination"]

    assert selected is not None
    assert selected["selected_costs"]["transport_total"] == 1500
    assert selected["soft_target_deviation"]["transport"] == 300
    assert output["selection_result"]["status"] == "feasible"


def test_high_food_activity_priority_prefers_more_remaining_budget():
    """
    用户更重视餐饮活动预算时，较便宜但质量略低的组合可能更优。

    这证明 adjusted_ratios 不只是展示字段，
    它会真正影响组合效用函数。
    """

    flights = [
        _flight(
            "out_expensive",
            direction="outbound",
            score=0.96,
            total_price=900,
            rank=1,
        ),
        _flight(
            "out_saving",
            direction="outbound",
            score=0.82,
            total_price=500,
            rank=2,
        ),
        _flight(
            "return",
            direction="return",
            score=0.90,
            total_price=500,
            rank=1,
        ),
    ]
    hotels = [
        _hotel(
            "hotel",
            score=0.90,
            total_price=1000,
            rank=1,
        )
    ]

    output = BudgetOptimizer().optimize(
        planning_context=_planning_context(
            total_budget=3000,
            adjusted_ratios={
                "transport": 0.10,
                "hotel": 0.10,
                "food_activity": 0.80,
            },
            food_activity_target=1000,
        ),
        flight_candidates=flights,
        hotel_candidates=hotels,
    )

    selected = output[
        "selection_result"
    ]["selected_combination"]

    assert selected["outbound_flight_id"] == "out_saving"
    assert selected["remaining_budget"] == 1000
    assert (
        selected["score_breakdown"][
            "food_activity_reserve_fit"
        ]
        == 1.0
    )


def test_no_budget_limit_uses_quality_without_fake_remaining_budget():
    """
    用户没有总预算时，不生成具体剩余预算，也不预测消费。
    """

    flights = [
        _flight(
            "out_high",
            direction="outbound",
            score=0.95,
            total_price=1000,
            rank=1,
        ),
        _flight(
            "out_low",
            direction="outbound",
            score=0.70,
            total_price=400,
            rank=2,
        ),
        _flight(
            "return",
            direction="return",
            score=0.90,
            total_price=700,
            rank=1,
        ),
    ]
    hotels = [
        _hotel(
            "hotel_high",
            score=0.94,
            total_price=1600,
            rank=1,
        ),
        _hotel(
            "hotel_low",
            score=0.72,
            total_price=800,
            rank=2,
        ),
    ]

    output = BudgetOptimizer().optimize(
        planning_context=_planning_context(
            total_budget=None,
            transport_target=None,
            hotel_target=None,
            food_activity_target=None,
        ),
        flight_candidates=flights,
        hotel_candidates=hotels,
    )

    selection = output["selection_result"]

    assert selection["status"] == "feasible"
    assert selection["budget_mode"] == "no_budget_limit"
    assert selection["remaining_budget"] is None
    assert (
        selection["remaining_budget_allocation"][
            "allocation_basis"
        ]
        == "no_budget_limit"
    )
    assert (
        selection["selected_combination"][
            "outbound_flight_id"
        ]
        == "out_high"
    )
    assert (
        selection["selected_combination"][
            "hotel_id"
        ]
        == "hotel_high"
    )


def test_max_transport_total_is_combination_hard_limit():
    """往返交通总上限只能在组合阶段检查。"""

    flights = [
        _flight(
            "out_expensive",
            direction="outbound",
            score=0.98,
            total_price=800,
            rank=1,
        ),
        _flight(
            "out_ok",
            direction="outbound",
            score=0.80,
            total_price=500,
            rank=2,
        ),
        _flight(
            "return",
            direction="return",
            score=0.90,
            total_price=600,
            rank=1,
        ),
    ]
    hotels = [
        _hotel(
            "hotel",
            score=0.90,
            total_price=1000,
            rank=1,
        )
    ]

    output = BudgetOptimizer().optimize(
        planning_context=_planning_context(
            total_budget=4000,
            max_transport_total=1200,
        ),
        flight_candidates=flights,
        hotel_candidates=hotels,
    )

    selected = output[
        "selection_result"
    ]["selected_combination"]

    assert selected["outbound_flight_id"] == "out_ok"
    assert (
        selected["selected_costs"][
            "transport_total"
        ]
        == 1100
    )
    assert (
        output["budget_optimize_result"][
            "rejected_by_transport_limit_count"
        ]
        == 1
    )


def test_no_feasible_combination_returns_budget_gap_and_suggestions():
    """
    全部组合超过总预算时，不自动放松限制，
    而是给出最便宜已知组合和最低预算差距。
    """

    flights = [
        _flight(
            "out",
            direction="outbound",
            score=0.9,
            total_price=700,
            rank=1,
        ),
        _flight(
            "return",
            direction="return",
            score=0.9,
            total_price=600,
            rank=1,
        ),
    ]
    hotels = [
        _hotel(
            "hotel",
            score=0.9,
            total_price=900,
            rank=1,
        )
    ]

    output = BudgetOptimizer().optimize(
        planning_context=_planning_context(
            total_budget=1800,
        ),
        flight_candidates=flights,
        hotel_candidates=hotels,
    )

    assert (
        output["selection_result"]["status"]
        == "no_feasible_combination"
    )
    assert (
        output["adjustment_plan"][
            "minimum_known_subtotal"
        ]
        == 2200
    )
    assert (
        output["adjustment_plan"][
            "required_budget_increase"
        ]
        == 400
    )
    assert output["adjustment_plan"]["suggestions"]
    assert output["budget_optimize_result"][
        "rejected_by_total_budget_count"
    ] == 1


def test_missing_candidate_group_is_business_status():
    """缺少返程候选时返回 no_candidates，而不是程序异常。"""

    output = BudgetOptimizer().optimize(
        planning_context=_planning_context(),
        flight_candidates=[
            _flight(
                "out",
                direction="outbound",
                score=0.9,
                total_price=700,
                rank=1,
            )
        ],
        hotel_candidates=[
            _hotel(
                "hotel",
                score=0.9,
                total_price=1000,
                rank=1,
            )
        ],
    )

    assert output["selection_result"]["status"] == "no_candidates"
    assert output["adjustment_plan"]["status"] == "missing_candidates"
    assert output["budget_optimize_result"][
        "evaluated_combination_count"
    ] == 0


def test_known_cost_breakdown_does_not_claim_real_total_spending():
    """
    输出只应包含可确定的航班和酒店报价，
    不能出现 actual_spending 或 predicted_total_spending 等字段。
    """

    output = BudgetOptimizer().optimize(
        planning_context=_planning_context(),
        flight_candidates=[
            _flight(
                "out",
                direction="outbound",
                score=0.9,
                total_price=700,
                rank=1,
            ),
            _flight(
                "return",
                direction="return",
                score=0.9,
                total_price=600,
                rank=1,
            ),
        ],
        hotel_candidates=[
            _hotel(
                "hotel",
                score=0.9,
                total_price=1000,
                rank=1,
            )
        ],
    )

    selection = output["selection_result"]
    serialized = str(selection).lower()

    assert "actual_spending" not in serialized
    assert "predicted_total_spending" not in serialized
    assert selection["selected_costs"] == {
        "outbound_flight": 700.0,
        "return_flight": 600.0,
        "transport_total": 1300.0,
        "hotel": 1000.0,
        "known_subtotal": 2300.0,
    }


def test_candidate_limits_control_cartesian_product_size():
    """候选上限应限制实际枚举的笛卡尔积规模。"""

    flights = [
        *[
            _flight(
                f"out_{index}",
                direction="outbound",
                score=0.9 - index * 0.01,
                total_price=500 + index * 10,
                rank=index + 1,
            )
            for index in range(5)
        ],
        *[
            _flight(
                f"return_{index}",
                direction="return",
                score=0.9 - index * 0.01,
                total_price=500 + index * 10,
                rank=index + 1,
            )
            for index in range(5)
        ],
    ]
    hotels = [
        _hotel(
            f"hotel_{index}",
            score=0.9 - index * 0.01,
            total_price=900 + index * 10,
            rank=index + 1,
        )
        for index in range(5)
    ]

    optimizer = BudgetOptimizer(
        BudgetOptimizeConfig(
            max_outbound_candidates=2,
            max_return_candidates=2,
            max_hotel_candidates=2,
            max_alternatives=1,
        )
    )

    output = optimizer.optimize(
        planning_context=_planning_context(
            total_budget=5000,
        ),
        flight_candidates=flights,
        hotel_candidates=hotels,
    )

    # 2 × 2 × 2 = 8 个组合。
    assert output["budget_optimize_result"][
        "evaluated_combination_count"
    ] == 8
    assert len(output["selection_result"]["alternatives"]) <= 1
