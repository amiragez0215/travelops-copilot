from __future__ import annotations

import json

from app.validators.request_capability_validator import RequestCapabilityValidator


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _mock_dir(tmp_path):
    """创建最小能力目录。"""

    _write(
        tmp_path / "hotels_mock.json",
        {
            "provider": "test",
            "items": [
                {
                    "hotel_id": "hotel_001",
                    "name": "成都青羊静巷酒店",
                    "city": "成都",
                    "price_per_night": 500,
                    "available_rooms": 2,
                }
            ],
        },
    )
    _write(
        tmp_path / "weather_mock.json",
        {
            "provider": "test",
            "items": [
                {"city": "成都", "date": "2026-07-02"}
            ],
        },
    )
    _write(
        tmp_path / "flights_mock.json",
        {
            "provider": "test",
            "items": [
                {
                    "flight_id": "flight_001",
                    "departure_city": "杭州",
                    "arrival_city": "成都",
                    "price": 700,
                }
            ],
        },
    )
    return tmp_path


def test_supported_destination_route_and_named_hotel(tmp_path):
    validator = RequestCapabilityValidator(_mock_dir(tmp_path))

    result = validator.validate(
        {
            "origin": "杭州",
            "destination": "成都",
            "named_constraints": [
                {
                    "entity_type": "hotel",
                    "entity_name": "成都青羊静巷酒店",
                    "constraint_mode": "required",
                }
            ],
        }
    )

    assert result["supported"] is True
    assert result["supported_destinations"] == ["成都"]


def test_llm_resolved_destination_can_still_be_out_of_data_scope(tmp_path):
    validator = RequestCapabilityValidator(_mock_dir(tmp_path))

    result = validator.validate(
        {
            "origin": "杭州",
            "destination": "西安",
            "named_constraints": [],
        }
    )

    assert result["supported"] is False
    issue_types = {item["issue_type"] for item in result["issues"]}
    assert "unsupported_destination" in issue_types
    assert "unsupported_flight_route" in issue_types


def test_unknown_named_hotel_requires_clarification(tmp_path):
    validator = RequestCapabilityValidator(_mock_dir(tmp_path))

    result = validator.validate(
        {
            "origin": "杭州",
            "destination": "成都",
            "named_constraints": [
                {
                    "entity_type": "hotel",
                    "entity_name": "不存在酒店",
                    "constraint_mode": "required",
                }
            ],
        }
    )

    assert result["supported"] is False
    assert any(
        item["issue_type"] == "named_hotel_not_found"
        for item in result["issues"]
    )
