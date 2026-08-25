from __future__ import annotations

import pytest

from app.builders.planning_context_builder import build_planning_context


def _trip_request(**overrides):
    data = {
        "origin": "杭州",
        "destination": "成都",
        "start_date": "2026-07-02",
        "end_date": "2026-07-04",
        "days": 3,
        "budget": 4000,
        "people_count": 1,
        "room_count": 1,
        "preference_signals": [],
        "spend_preferences": [],
        "hard_constraints": [],
        "named_constraints": [],
        "unmapped_requirements": [],
    }
    data.update(overrides)
    return data


def _profile():
    return {
        "source": "sql:user_profiles",
        "flight_preferences": {
            "avoid_early_flight": 0.9,
            "prefer_direct": 0.75,
        },
        "hotel_preferences": {
            "near_subway": 0.7,
            "quiet": 0.85,
        },
        "activity_preferences": {"nature_scenery": 0.55},
        "pace_preferences": {"slow": 0.85},
        "budget_preferences": {"preferred_hotel_price_max": 650},
        "diet_preferences": {"spicy_tolerance": "medium"},
    }


def test_planning_context_merges_preferences_and_uses_default_budget_heuristic():
    """没有消费倾向时使用 35/35/30，但所有分项仍是软目标。"""

    context = build_planning_context(
        trip_request=_trip_request(
            preference_signals=[
                {
                    "domain": "hotel",
                    "key": "quiet",
                    "label": "安静",
                    "strength": 1.0,
                    "polarity": "positive",
                    "evidence": "一定要安静",
                    "source": "rule",
                },
                {
                    "domain": "hotel",
                    "key": "cleanliness",
                    "label": "干净",
                    "strength": 0.9,
                    "polarity": "positive",
                    "evidence": "干净的酒店",
                    "source": "rule",
                },
            ]
        ),
        user_profile=_profile(),
    )

    assert context["request"]["nights"] == 2
    assert context["request"]["room_count"] == 1
    assert context["preference_weights"]["hotel"]["quiet"] == 1.0
    assert context["preference_weights"]["hotel"]["cleanliness"] == 0.9

    # “一定要安静”仍然只是高权重主观偏好。
    assert context["hard_constraints"]["hotel"] == []

    budget = context["budget_plan"]
    assert budget["allocation_source"] == "default_heuristic_v1"
    assert budget["base_ratios"] == {
        "transport": 0.35,
        "hotel": 0.35,
        "food_activity": 0.30,
    }
    assert budget["adjusted_ratios"] == budget["base_ratios"]
    assert budget["soft_targets"]["transport_budget"] == 1400
    assert budget["soft_targets"]["hotel_budget"] == 1400
    assert budget["soft_targets"]["food_activity_budget"] == 1200

    # 预算拆分的 700 元/晚被长期软偏好 650 元/晚下调，但不是硬限制。
    assert budget["soft_targets"]["target_hotel_price_per_room_night"] == 650
    assert budget["hard_limits"]["max_hotel_price_per_night"] is None


def test_spend_preferences_dynamically_adjust_ratios():
    """住宿和餐饮想多投入、航班想省钱时，比例应确定性调整。"""

    context = build_planning_context(
        trip_request=_trip_request(
            spend_preferences=[
                {
                    "category": "hotel",
                    "direction": "increase",
                    "strength": 0.9,
                    "evidence": "住宿好一点",
                    "source": "llm",
                },
                {
                    "category": "transport",
                    "direction": "decrease",
                    "strength": 0.85,
                    "evidence": "航班不用很贵",
                    "source": "llm",
                },
                {
                    "category": "food_activity",
                    "direction": "increase",
                    "strength": 0.85,
                    "evidence": "吃的东西好一点",
                    "source": "llm",
                },
            ]
        ),
        user_profile={},
    )

    budget = context["budget_plan"]
    ratios = budget["adjusted_ratios"]

    assert budget["allocation_source"] == "user_adjusted_heuristic_v1"
    assert ratios["transport"] < 0.35
    assert ratios["hotel"] > 0.35
    assert ratios["food_activity"] > 0.30
    assert sum(ratios.values()) == pytest.approx(1.0)
    assert budget["soft_targets"]["hotel_budget"] > 1400
    assert budget["soft_targets"]["transport_budget"] < 1400
    assert len(budget["allocation_reasons"]) == 3


def test_objective_hard_constraints_are_grouped_and_subjective_ones_are_dropped():
    """PlanningContext 只传播 Validator 白名单中的客观硬约束。"""

    context = build_planning_context(
        trip_request=_trip_request(
            hard_constraints=[
                {
                    "domain": "flight",
                    "field": "is_direct",
                    "operator": "==",
                    "value": True,
                    "evidence": "必须直飞",
                    "source": "rule",
                },
                {
                    "domain": "hotel",
                    "field": "price_per_night",
                    "operator": "<=",
                    "value": 600,
                    "evidence": "每晚不能超过600元",
                    "source": "rule",
                },
                {
                    "domain": "hotel",
                    "field": "quiet_score",
                    "operator": ">=",
                    "value": 0.9,
                    "evidence": "一定要安静",
                    "source": "llm",
                },
            ]
        ),
        user_profile={},
    )

    assert context["hard_constraints"]["flight"][0]["field"] == "is_direct"
    assert context["hard_constraints"]["hotel"][0]["field"] == "price_per_night"
    assert all(
        item["field"] != "quiet_score"
        for item in context["hard_constraints"]["hotel"]
    )
    assert context["budget_plan"]["hard_limits"]["max_hotel_price_per_night"] == 600
    assert context["budget_plan"]["hard_limits"]["total_budget"] == 4000


def test_no_user_budget_keeps_only_long_term_price_soft_target():
    """没有总预算时不生成分项金额，但可以保留长期价格偏好。"""

    context = build_planning_context(
        trip_request=_trip_request(budget=None),
        user_profile={
            "budget_preferences": {
                "preferred_hotel_price_max": 600,
            }
        },
    )

    budget = context["budget_plan"]
    assert budget["mode"] == "no_budget_limit"
    assert budget["total_budget"] is None
    assert budget["hard_limits"]["total_budget"] is None
    assert budget["soft_targets"]["transport_budget"] is None
    assert budget["soft_targets"]["hotel_budget"] is None
    assert budget["soft_targets"]["food_activity_budget"] is None
    assert budget["soft_targets"]["target_hotel_price_per_room_night"] == 600


def test_planning_context_infers_end_date():
    context = build_planning_context(
        trip_request=_trip_request(end_date=None),
        user_profile={},
    )

    assert context["request"]["end_date"] == "2026-07-04"


def test_hotel_soft_target_accounts_for_room_count():
    """住宿软目标按 room_count × nights 分摊。"""

    context = build_planning_context(
        trip_request=_trip_request(
            people_count=2,
            room_count=2,
        ),
        user_profile={},
    )

    # 酒店预算 1400，2 间 × 2 晚 = 4 个 room-nights，因此目标 350。
    assert (
        context["budget_plan"]["soft_targets"][
            "target_hotel_price_per_room_night"
        ]
        == 350
    )


def test_invalid_trip_request_type_raises():
    with pytest.raises(TypeError, match="trip_request 必须是 dict-like 对象"):
        build_planning_context("invalid", {})  # type: ignore[arg-type]
