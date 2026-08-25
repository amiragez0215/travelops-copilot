from datetime import date

from app.nodes.input_extract_node import input_extract_node
from app.schemas.trip_request_schema import TripRequest


class FakeTripRequestExtractor:
    """
    测试 Node 是否能注入 extractor。
    """

    def extract(self, message, user_id=None, reference_date=None):
        return TripRequest(
            user_id=user_id,
            origin="杭州",
            destination="成都",
            start_date=date(2026, 7, 2),
            days=3,
            raw_message=message,
            extraction={
                "method": "fake_extractor",
                "confidence": 1.0,
                "missing_fields": [],
            },
        )


def test_input_extract_node_success_with_default_extractor():
    result = input_extract_node(
        {
            "user_id": "user_001",
            "raw_message": "我想从杭州去成都玩三天，预算4000，不想坐太早航班，想住安静一点",
            "reference_date": "2026-07-01",
        }
    )

    trip_request = result["trip_request"]

    assert trip_request["user_id"] == "user_001"
    assert trip_request["origin"] == "杭州"
    assert trip_request["destination"] == "成都"
    assert trip_request["days"] == 3
    assert trip_request["budget"] == 4000
    assert "avoid_early_flight" in trip_request["transport_preferences"]
    assert "quiet" in trip_request["hotel_preferences"]

    assert "preference_signals" in trip_request
    assert any(
        signal["domain"] == "flight" and signal["key"] == "avoid_early_flight"
        for signal in trip_request["preference_signals"]
    )
    assert any(
        signal["domain"] == "hotel" and signal["key"] == "quiet"
        for signal in trip_request["preference_signals"]
    )

    assert "trip_request" in result
    assert "intent" not in result
    assert "raw_input" not in result

    assert result["trace"][0]["node_name"] == "input_extract"
    assert result["trace"][0]["status"] == "success"


def test_input_extract_node_supports_extractor_injection():
    result = input_extract_node(
        {
            "user_id": "user_001",
            "raw_message": "这是一条由 fake extractor 处理的输入",
            "reference_date": "2026-07-01",
        },
        extractor=FakeTripRequestExtractor(),
    )

    trip_request = result["trip_request"]

    assert trip_request["origin"] == "杭州"
    assert trip_request["destination"] == "成都"
    assert trip_request["start_date"] == "2026-07-02"
    assert trip_request["days"] == 3
    assert trip_request["extraction"]["method"] == "fake_extractor"
    assert result["trace"][0]["status"] == "success"


def test_input_extract_node_full_date():
    result = input_extract_node(
        {
            "user_id": "user_001",
            "raw_message": "2026年7月2日从杭州去成都玩三天，预算4000",
            "reference_date": "2026-07-01",
        }
    )

    assert result["trip_request"]["origin"] == "杭州"
    assert result["trip_request"]["destination"] == "成都"
    assert result["trip_request"]["start_date"] == "2026-07-02"


def test_input_extract_node_empty_message_returns_error():
    result = input_extract_node(
        {
            "user_id": "user_001",
            "raw_message": "   ",
            "reference_date": "2026-07-01",
        }
    )

    assert result["trip_request"]["user_id"] == "user_001"
    assert result["trip_request"]["raw_message"] == ""
    assert result["errors"]
    assert result["errors"][0]["node"] == "input_extract"
    assert result["trace"][0]["status"] == "failed"

def test_input_extract_node_keeps_strong_hotel_preference_as_soft_weight():
    """
    “一定要安静”只生成 strength=1.0 的偏好，不进入 hard_constraints。
    """

    result = input_extract_node(
        {
            "user_id": "user_001",
            "raw_message": (
                "2026年7月2日从杭州去成都玩三天，"
                "一定要安静，干净的酒店"
            ),
            "reference_date": "2026-07-01",
        }
    )

    trip_request = result["trip_request"]
    quiet = next(
        signal
        for signal in trip_request["preference_signals"]
        if signal["domain"] == "hotel" and signal["key"] == "quiet"
    )

    assert quiet["strength"] == 1.0
    assert "hard_constraint" not in quiet
    assert trip_request["hard_constraints"] == []


def test_input_extract_node_separates_objective_hard_constraint():
    """必须直飞属于客观可验证硬约束。"""

    result = input_extract_node(
        {
            "raw_message": "2026年7月2日从杭州去成都玩三天，必须直飞",
            "reference_date": "2026-07-01",
        }
    )

    constraints = result["trip_request"]["hard_constraints"]

    assert any(
        item["domain"] == "flight"
        and item["field"] == "is_direct"
        and item["value"] is True
        for item in constraints
    )
