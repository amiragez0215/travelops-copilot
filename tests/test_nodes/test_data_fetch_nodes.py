import json

from app.mcp.travel_data_client import LocalTravelDataClient
from app.nodes.flight_node import flight_search_node
from app.nodes.hotel_node import hotel_search_node
from app.nodes.weather_node import weather_node


def _write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )


def _base_state():
    return {
        "planning_context": {
            "request": {
                "origin": "杭州",
                "destination": "成都",
                "start_date": "2026-07-02",
                "end_date": "2026-07-04",
                "days": 3,
                "nights": 2,
                "people_count": 1,
            },
            "budget_plan": {
                "mode": "budget_limited",
                "total_budget": 4000,
                "hard_limits": {
                    "total_budget": 4000,
                    "max_hotel_price_per_night": None,
                },
                "soft_targets": {
                    "target_hotel_price_per_night": 700,
                },
            },
        }
    }


def test_weather_node_success(tmp_path):
    weather_file = tmp_path / "weather_mock.json"
    _write_json(
        weather_file,
        {
            "provider": "mock_weather_provider",
            "items": [
                {
                    "weather_id": "w1",
                    "city": "成都",
                    "date": "2026-07-02",
                    "condition": "小雨",
                    "risk_tags": ["rain"],
                    "warnings": ["带伞"],
                }
            ],
        },
    )

    client = LocalTravelDataClient(weather_mock_file=weather_file)
    result = weather_node(_base_state(), client=client)

    assert result["weather_result"]["status"] == "ok"
    assert result["weather_result"]["daily"][0]["city"] == "成都"
    assert result["weather_fetch_meta"]["tool_name"] == "get_weather"
    assert result["trace"][0]["node_name"] == "weather"
    assert result["trace"][0]["status"] == "success"


def test_flight_search_node_success(tmp_path):
    flights_file = tmp_path / "flights_mock.json"
    _write_json(
        flights_file,
        {
            "provider": "mock_flight_provider",
            "items": [
                {
                    "flight_id": "f_out",
                    "departure_city": "杭州",
                    "arrival_city": "成都",
                    "depart_date": "2026-07-02",
                    "depart_time": "10:00",
                    "price": 800,
                    "available_seats": 2,
                },
                {
                    "flight_id": "f_return",
                    "departure_city": "成都",
                    "arrival_city": "杭州",
                    "depart_date": "2026-07-04",
                    "depart_time": "16:00",
                    "price": 700,
                    "available_seats": 2,
                },
            ],
        },
    )

    client = LocalTravelDataClient(flights_mock_file=flights_file)
    result = flight_search_node(_base_state(), client=client)

    assert result["raw_flight_results"]["status"] == "ok"
    assert len(result["raw_flight_results"]["outbound"]) == 1
    assert len(result["raw_flight_results"]["return"]) == 1
    assert result["flight_fetch_meta"]["tool_name"] == "search_flights"
    assert result["trace"][0]["node_name"] == "flight_search"
    assert result["trace"][0]["status"] == "success"


def test_hotel_search_node_success(tmp_path):
    hotels_file = tmp_path / "hotels_mock.json"
    _write_json(
        hotels_file,
        {
            "provider": "mock_hotel_provider",
            "items": [
                {
                    "hotel_id": "h1",
                    "name": "成都酒店",
                    "city": "成都",
                    "price_per_night": 500,
                    "rating": 4.6,
                    "available_rooms": 2,
                    "near_subway": True,
                    "quiet_score": 0.8,
                    "cleanliness_score": 0.9,
                }
            ],
        },
    )

    client = LocalTravelDataClient(hotels_mock_file=hotels_file)
    result = hotel_search_node(_base_state(), client=client)

    assert result["raw_hotel_results"]["status"] == "ok"
    assert len(result["raw_hotel_results"]["items"]) == 1
    assert result["raw_hotel_results"]["items"][0]["estimated_total_price"] == 1000
    assert result["hotel_fetch_meta"]["tool_name"] == "search_hotels"
    assert result["trace"][0]["node_name"] == "hotel_search"
    assert result["trace"][0]["status"] == "success"


def test_data_fetch_nodes_fail_without_planning_context():
    weather_result = weather_node({})
    flight_result = flight_search_node({})
    hotel_result = hotel_search_node({})

    assert weather_result["weather_result"]["status"] == "unavailable"
    assert weather_result["trace"][0]["status"] == "failed"

    assert flight_result["raw_flight_results"]["status"] == "failed"
    assert flight_result["trace"][0]["status"] == "failed"

    assert hotel_result["raw_hotel_results"]["status"] == "failed"
    assert hotel_result["trace"][0]["status"] == "failed"