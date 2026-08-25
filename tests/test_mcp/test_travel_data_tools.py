import json

from app.mcp.travel_data_tools import (
    get_weather_tool,
    search_city_activities_tool,
    search_flights_tool,
    search_hotels_tool,
)


def _write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )


def test_get_weather_tool_uses_mock_provider(tmp_path):
    mock_file = tmp_path / "weather_mock.json"
    _write_json(
        mock_file,
        {
            "provider": "mock_weather_provider",
            "items": [
                {
                    "weather_id": "w1",
                    "city": "成都",
                    "date": "2026-07-02",
                    "condition": "多云",
                    "risk_tags": [],
                    "warnings": [],
                }
            ],
        },
    )

    result = get_weather_tool(
        city="成都",
        start_date="2026-07-02",
        mock_file=mock_file,
    )

    assert result["status"] == "ok"
    assert result["source"] == "mock_weather_provider"


def test_search_flights_tool_uses_mock_provider(tmp_path):
    mock_file = tmp_path / "flights_mock.json"
    _write_json(
        mock_file,
        {
            "provider": "mock_flight_provider",
            "items": [
                {
                    "flight_id": "f1",
                    "departure_city": "杭州",
                    "arrival_city": "成都",
                    "depart_date": "2026-07-02",
                    "depart_time": "10:00",
                    "price": 800,
                    "available_seats": 3,
                }
            ],
        },
    )

    result = search_flights_tool(
        origin="杭州",
        destination="成都",
        depart_date="2026-07-02",
        people_count=1,
        mock_file=mock_file,
    )

    assert result["status"] == "ok"
    assert result["outbound"][0]["flight_id"] == "f1"


def test_search_hotels_tool_uses_mock_provider(tmp_path):
    mock_file = tmp_path / "hotels_mock.json"
    _write_json(
        mock_file,
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
                }
            ],
        },
    )

    result = search_hotels_tool(
        city="成都",
        nights=2,
        people_count=1,
        mock_file=mock_file,
    )

    assert result["status"] == "ok"
    assert result["items"][0]["hotel_id"] == "h1"


def test_search_city_activities_tool_filters_city_date_and_interest(tmp_path):
    mock_file = tmp_path / "activities_mock.json"
    _write_json(
        mock_file,
        {
            "provider": "mock_activity_provider",
            "items": [
                {
                    "activity_id": "sha-a1",
                    "name": "上海艺术展",
                    "city": "上海",
                    "category": "展览",
                    "available_dates": ["2026-10-03"],
                    "indoor_outdoor": "indoor",
                    "tags": ["展览", "艺术"],
                },
                {
                    "activity_id": "pek-a1",
                    "name": "北京展览",
                    "city": "北京",
                    "category": "展览",
                    "available_dates": ["2026-10-03"],
                    "tags": ["展览"],
                },
            ],
        },
    )

    result = search_city_activities_tool(
        city="上海",
        start_date="2026-10-03",
        end_date="2026-10-05",
        interests=["艺术"],
        mock_file=mock_file,
    )

    assert result["status"] == "ok"
    assert [item["activity_id"] for item in result["items"]] == ["sha-a1"]
