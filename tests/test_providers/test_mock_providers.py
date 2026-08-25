import json

from app.providers.flight_mock_provider import search_flights_from_mock
from app.providers.hotel_mock_provider import search_hotels_from_mock
from app.providers.weather_mock_provider import query_weather_from_mock


def _write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )


def test_weather_mock_provider_filters_city_and_date(tmp_path):
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
                    "condition": "小雨",
                    "risk_tags": ["rain"],
                    "warnings": ["带伞"],
                },
                {
                    "weather_id": "w2",
                    "city": "杭州",
                    "date": "2026-07-02",
                    "condition": "晴",
                    "risk_tags": [],
                    "warnings": [],
                },
            ],
        },
    )

    result = query_weather_from_mock(
        city="成都",
        start_date="2026-07-02",
        end_date="2026-07-02",
        mock_file=mock_file,
    )

    assert result["status"] == "ok"
    assert len(result["daily"]) == 1
    assert result["daily"][0]["city"] == "成都"
    assert result["risks"][0]["risk_type"] == "rain"


def test_flight_mock_provider_filters_data_layer_hard_errors(tmp_path):
    mock_file = tmp_path / "flights_mock.json"
    _write_json(
        mock_file,
        {
            "provider": "mock_flight_provider",
            "items": [
                {
                    "flight_id": "f_ok",
                    "flight_no": "CA100",
                    "airline": "MockAir",
                    "departure_city": "杭州",
                    "arrival_city": "成都",
                    "depart_date": "2026-07-02",
                    "depart_time": "10:00",
                    "arrive_time": "12:30",
                    "price": 800,
                    "available_seats": 2,
                    "is_direct": True,
                },
                {
                    "flight_id": "f_soldout",
                    "flight_no": "CA101",
                    "airline": "MockAir",
                    "departure_city": "杭州",
                    "arrival_city": "成都",
                    "depart_date": "2026-07-02",
                    "depart_time": "08:00",
                    "arrive_time": "10:30",
                    "price": 500,
                    "available_seats": 0,
                    "is_direct": True,
                },
                {
                    "flight_id": "f_wrong_city",
                    "flight_no": "CA102",
                    "airline": "MockAir",
                    "departure_city": "杭州",
                    "arrival_city": "长沙",
                    "depart_date": "2026-07-02",
                    "depart_time": "09:00",
                    "arrive_time": "11:00",
                    "price": 500,
                    "available_seats": 5,
                    "is_direct": True,
                },
            ],
        },
    )

    result = search_flights_from_mock(
        origin="杭州",
        destination="成都",
        depart_date="2026-07-02",
        people_count=1,
        mock_file=mock_file,
    )

    assert result["status"] == "ok"
    assert len(result["outbound"]) == 1
    assert result["outbound"][0]["flight_id"] == "f_ok"


def test_hotel_mock_provider_filters_data_layer_hard_errors(tmp_path):
    mock_file = tmp_path / "hotels_mock.json"
    _write_json(
        mock_file,
        {
            "provider": "mock_hotel_provider",
            "items": [
                {
                    "hotel_id": "h_ok",
                    "name": "成都好酒店",
                    "city": "成都",
                    "price_per_night": 500,
                    "rating": 4.6,
                    "available_rooms": 2,
                    "near_subway": True,
                    "quiet_score": 0.8,
                    "cleanliness_score": 0.9,
                },
                {
                    "hotel_id": "h_soldout",
                    "name": "成都无房酒店",
                    "city": "成都",
                    "price_per_night": 300,
                    "rating": 4.2,
                    "available_rooms": 0,
                    "near_subway": True,
                    "quiet_score": 0.7,
                },
                {
                    "hotel_id": "h_wrong_city",
                    "name": "杭州酒店",
                    "city": "杭州",
                    "price_per_night": 300,
                    "rating": 4.2,
                    "available_rooms": 5,
                    "near_subway": True,
                    "quiet_score": 0.7,
                },
            ],
        },
    )

    result = search_hotels_from_mock(
        city="成都",
        nights=2,
        people_count=1,
        mock_file=mock_file,
    )

    assert result["status"] == "ok"
    assert len(result["items"]) == 1
    assert result["items"][0]["hotel_id"] == "h_ok"
    assert result["items"][0]["estimated_total_price"] == 1000