from __future__ import annotations

from app.nodes.planning_context_node import planning_context_node


def _state(**trip_overrides):
    trip_request = {
        "origin": "杭州",
        "destination": "成都",
        "start_date": "2026-07-02",
        "end_date": "2026-07-04",
        "days": 3,
        "budget": 4000,
        "people_count": 1,
        "room_count": 1,
        "preference_signals": [
            {
                "domain": "hotel",
                "key": "quiet",
                "label": "安静",
                "strength": 1.0,
                "polarity": "positive",
                "evidence": "一定要安静",
                "source": "rule",
            }
        ],
        "spend_preferences": [],
        "hard_constraints": [],
        "named_constraints": [],
        "unmapped_requirements": [],
    }
    trip_request.update(trip_overrides)

    return {
        "trip_request": trip_request,
        "user_profile": {
            "source": "sql:user_profiles",
            "flight_preferences": {"avoid_early_flight": 0.9},
            "hotel_preferences": {"near_subway": 0.7, "quiet": 0.85},
            "activity_preferences": {"food": 0.75},
            "pace_preferences": {"slow": 0.85},
            "budget_preferences": {"preferred_hotel_price_max": 650},
        },
    }


def test_planning_context_node_success():
    result = planning_context_node(_state())

    assert "planning_context" in result
    assert "budget_plan" in result

    context = result["planning_context"]
    assert context["request"]["destination"] == "成都"
    assert context["request"]["nights"] == 2
    assert context["preference_weights"]["hotel"]["quiet"] == 1.0
    assert context["hard_constraints"]["hotel"] == []
    assert result["budget_plan"]["allocation_source"] == "default_heuristic_v1"
    assert (
        result["budget_plan"]["soft_targets"][
            "target_hotel_price_per_room_night"
        ]
        == 650
    )
    assert result["trace"][0]["node_name"] == "planning_context"
    assert result["trace"][0]["status"] == "success"


def test_planning_context_node_dynamic_budget():
    result = planning_context_node(
        _state(
            spend_preferences=[
                {
                    "category": "hotel",
                    "direction": "increase",
                    "strength": 1.0,
                    "evidence": "酒店好一点",
                    "source": "llm",
                },
                {
                    "category": "transport",
                    "direction": "decrease",
                    "strength": 1.0,
                    "evidence": "航班省一点",
                    "source": "llm",
                },
            ]
        )
    )

    ratios = result["budget_plan"]["adjusted_ratios"]
    assert ratios["hotel"] > 0.35
    assert ratios["transport"] < 0.35


def test_planning_context_node_without_user_profile():
    state = _state()
    state.pop("user_profile")

    result = planning_context_node(state)

    assert result["planning_context"]["request"]["destination"] == "成都"
    assert result["budget_plan"]["mode"] == "budget_limited"
    assert result["trace"][0]["status"] == "success"


def test_planning_context_node_missing_trip_request_returns_error():
    result = planning_context_node({})

    assert result["errors"]
    assert result["errors"][0]["node"] == "planning_context"
    assert result["trace"][0]["status"] == "failed"


def test_planning_context_node_invalid_user_profile_returns_error():
    result = planning_context_node(
        {
            "trip_request": _state()["trip_request"],
            "user_profile": "invalid",
        }
    )

    assert result["errors"]
    assert result["trace"][0]["status"] == "failed"
