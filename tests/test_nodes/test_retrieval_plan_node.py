from __future__ import annotations

from app.nodes.retrieval_plan_node import retrieval_plan_node


def _state():
    return {
        "trip_request": {
            "origin": "杭州",
            "destination": "成都",
            "start_date": "2026-07-02",
            "end_date": "2026-07-04",
            "days": 3,
            "people_count": 1,
            "room_count": 1,
            "named_constraints": [],
        },
        "planning_context": {
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
                "flight": {},
                "hotel": {"quiet": 1.0, "cleanliness": 0.9},
                "activity": {"food": 0.9},
                "pace": {"slow": 0.9},
                "risk": {},
            },
            "named_constraints": [],
        },
        "weather_result": {
            "status": "ok",
            "risks": [{"risk_type": "rain", "level": "medium"}],
            "daily": [],
        },
        "raw_hotel_results": {
            "status": "ok",
            "items": [
                {
                    "hotel_id": "hotel_001",
                    "name": "成都青羊静巷酒店",
                    "city": "成都",
                },
                {
                    "hotel_id": "hotel_002",
                    "name": "成都春熙路酒店",
                    "city": "成都",
                },
            ],
        },
        "selection_result": {
            "status": "feasible",
            "selected_combination": {
                "hotel_id": "hotel_001",
                "outbound_flight_id": "flight_out_001",
                "return_flight_id": "flight_return_001",
            },
        },
        "rag_retry_count": 0,
    }


def test_retrieval_plan_node_uses_post_selection_context():
    result = retrieval_plan_node(_state())

    plan = result["retrieval_plan"]
    assert plan["mode"] == "initial"

    hotel_task = next(
        item for item in plan["tasks"] if item["category"] == "hotel_reviews"
    )
    assert hotel_task["metadata_filter"]["hotel_ids"] == ["hotel_001"]
    assert hotel_task["required"] is False
    assert result["trace"][0]["status"] == "success"


def test_retrieval_plan_node_repair_increments_retry_count():
    state = _state()
    state["retrieval_feedback"] = {"missing_doc_types": ["guides"]}

    result = retrieval_plan_node(state)

    assert result["retrieval_plan"]["mode"] == "repair"
    assert result["retrieval_plan"]["tasks"][0]["task_id"] == "repair_guide_main"
    assert result["rag_retry_count"] == 1


def test_retrieval_plan_node_does_not_increment_when_repair_exhausted():
    state = _state()
    state["rag_retry_count"] = 1
    state["retrieval_feedback"] = {"missing_doc_types": ["guides"]}

    result = retrieval_plan_node(state)

    assert result["retrieval_plan"]["repair_exhausted"] is True
    assert result["retrieval_plan"]["tasks"] == []
    assert "rag_retry_count" not in result


def test_retrieval_plan_node_without_selection_skips_hotel_task():
    state = _state()
    state.pop("selection_result")

    result = retrieval_plan_node(state)

    categories = {item["category"] for item in result["retrieval_plan"]["tasks"]}
    assert "hotel_reviews" not in categories
    assert "guides" in categories


def test_retrieval_plan_node_missing_planning_context_returns_error():
    state = _state()
    state.pop("planning_context")

    result = retrieval_plan_node(state)

    assert result["errors"]
    assert result["errors"][0]["node"] == "retrieval_plan"
    assert result["trace"][0]["status"] == "failed"


def test_retrieval_plan_node_missing_trip_request_returns_error():
    state = _state()
    state.pop("trip_request")

    result = retrieval_plan_node(state)

    assert result["errors"]
    assert result["trace"][0]["status"] == "failed"
