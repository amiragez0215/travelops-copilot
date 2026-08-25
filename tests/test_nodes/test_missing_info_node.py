from app.nodes.missing_info_node import (
    missing_info_check_node,
    route_after_missing_info,
)


def test_missing_info_node_all_required_fields_present():
    state = {
        "trip_request": {
            "origin": "杭州",
            "destination": "北京",
            "start_date": "2026-10-01",
            "end_date": "2026-10-03",
            "days": 3,
            "budget": 4000,
        }
    }

    result = missing_info_check_node(state)

    assert result["can_continue"] is True
    assert result["missing_fields"] == []
    assert result["missing_info_result"]["status"] == "ready"
    assert result["trace"][0]["node_name"] == "missing_info_check"
    assert result["trace"][0]["status"] == "success"

    assert route_after_missing_info(result) == "safety_check"


def test_missing_info_node_missing_start_date():
    state = {
        "trip_request": {
            "origin": "杭州",
            "destination": "北京",
            "start_date": None,
            "days": 3,
            "budget": 4000,
        }
    }

    result = missing_info_check_node(state)

    assert result["can_continue"] is False
    assert result["missing_fields"] == ["start_date"]
    assert result["missing_info_result"]["status"] == "missing_required_fields"
    assert result["itinerary_status"] == "collecting_info"

    assert route_after_missing_info(result) == "clarify"


def test_missing_info_node_budget_missing_but_can_continue():
    state = {
        "trip_request": {
            "origin": "杭州",
            "destination": "北京",
            "start_date": "2026-10-01",
            "days": 3,
            "budget": None,
        }
    }

    result = missing_info_check_node(state)

    assert result["can_continue"] is True
    assert result["missing_fields"] == []
    assert route_after_missing_info(result) == "safety_check"


def test_missing_info_node_reads_trip_request_only():
    state = {
        "trip_request": {
            "origin": "杭州",
            "destination": "北京",
            "start_date": "2026-10-01",
            "days": 3,
        }
    }

    result = missing_info_check_node(state)

    assert result["can_continue"] is True
    assert result["missing_fields"] == []


def test_missing_info_node_missing_trip_request_returns_error():
    result = missing_info_check_node({})

    assert result["can_continue"] is False
    assert result["missing_fields"] == ["trip_request"]
    assert result["missing_info_result"]["status"] == "invalid_trip_request"
    assert result["errors"]
    assert result["errors"][0]["node"] == "missing_info_check"
    assert result["trace"][0]["status"] == "failed"

    assert route_after_missing_info(result) == "clarify"


def test_route_after_missing_info_defaults_to_clarify_when_uncertain():
    assert route_after_missing_info({"can_continue": False}) == "clarify"
    assert route_after_missing_info({"missing_fields": ["start_date"]}) == "clarify"
    assert route_after_missing_info({}) == "clarify"
