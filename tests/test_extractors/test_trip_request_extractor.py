from __future__ import annotations

from datetime import date

from app.extractors.trip_request_extractor import (
    RuleBasedTripRequestExtractor,
    extract_trip_request_from_message,
)


def test_extract_basic_request_without_date():
    """规则 fallback 应抽取基础字段和常见偏好。"""

    trip_request = RuleBasedTripRequestExtractor().extract(
        message=(
            "我想从杭州去成都玩三天，预算4000，"
            "不想坐太早航班，想住安静一点，靠近地铁"
        ),
        user_id="user_001",
        reference_date=date(2026, 7, 1),
    )
    data = trip_request.to_state_dict()

    assert data["user_id"] == "user_001"
    assert data["origin"] == "杭州"
    assert data["destination"] == "成都"
    assert data["days"] == 3
    assert data["budget"] == 4000
    assert data["people_count"] == 1
    assert data["room_count"] == 1
    assert "avoid_early_flight" in data["transport_preferences"]
    assert "quiet" in data["hotel_preferences"]
    assert "near_subway" in data["hotel_preferences"]
    assert data["start_date"] is None
    assert "start_date" in data["extraction"]["missing_fields"]


def test_extract_full_date_and_compute_end_date():
    """start_date + days 应确定性推导 end_date。"""

    data = extract_trip_request_from_message(
        message="2026年7月2日从杭州去成都玩3天，预算4000，不要早班机",
        user_id="user_001",
        reference_date=date(2026, 7, 1),
    ).to_state_dict()

    assert data["start_date"] == "2026-07-02"
    assert data["end_date"] == "2026-07-04"
    assert data["days"] == 3
    assert data["budget"] == 4000


def test_extract_date_range_people_count_and_requires_room_clarification_later():
    """多人未说明房间数时保持 room_count=None，交给 Validator 澄清。"""

    data = extract_trip_request_from_message(
        message="7月2日到7月4日杭州到成都，两个人，预算五千，想轻松一点",
        user_id="user_001",
        reference_date=date(2026, 6, 30),
    ).to_state_dict()

    assert data["start_date"] == "2026-07-02"
    assert data["end_date"] == "2026-07-04"
    assert data["days"] == 3
    assert data["people_count"] == 2
    assert data["room_count"] is None
    assert data["budget"] == 5000
    assert "slow" in data["travel_style"]


def test_extract_destination_only_and_activity_preferences():
    """单目的地表达应保留缺失出发地，同时识别活动偏好。"""

    data = extract_trip_request_from_message(
        message="想去成都三日游，喜欢美食和博物馆",
        reference_date=date(2026, 7, 1),
    ).to_state_dict()

    assert data["origin"] is None
    assert data["destination"] == "成都"
    assert data["days"] == 3
    assert "food" in data["travel_style"]
    assert "culture_history" in data["travel_style"]
    assert "origin" in data["extraction"]["missing_fields"]
    assert "start_date" in data["extraction"]["missing_fields"]


def test_extract_relative_date_and_k_budget():
    """规则版继续支持相对日期和 4k 金额。"""

    data = extract_trip_request_from_message(
        message="后天从杭州去成都玩三天，预算4k，直飞，住安静点，不要太晚到",
        reference_date=date(2026, 7, 1),
    ).to_state_dict()

    assert data["start_date"] == "2026-07-03"
    assert data["end_date"] == "2026-07-05"
    assert data["budget"] == 4000
    assert "prefer_direct" in data["transport_preferences"]
    assert "avoid_late_arrival" in data["transport_preferences"]


def test_return_date_infers_start_date_from_days():
    """“7月8号回来”必须识别为 end_date，而不是 start_date。"""

    data = extract_trip_request_from_message(
        message="从杭州去成都玩三天，7月8号回来，想轻松一点",
        reference_date=date(2026, 7, 1),
    ).to_state_dict()

    assert data["start_date"] == "2026-07-06"
    assert data["end_date"] == "2026-07-08"
    assert data["extraction"]["missing_fields"] == []


def test_strong_subjective_preference_is_not_hard_constraint():
    """“一定要安静”应成为 1.0 偏好，而不是 hard_constraints。"""

    data = extract_trip_request_from_message(
        message=(
            "2026年7月2日从杭州去成都玩三天，"
            "离地铁近，一定要安静，干净的酒店，评分较高"
        ),
        reference_date=date(2026, 7, 1),
    ).to_state_dict()

    quiet = next(
        item
        for item in data["preference_signals"]
        if item["domain"] == "hotel" and item["key"] == "quiet"
    )

    assert quiet["strength"] == 1.0
    assert "hard_constraint" not in quiet
    assert data["hard_constraints"] == []


def test_objective_constraints_are_separated_from_preferences():
    """直飞、时间和价格等客观限制应进入 hard_constraints。"""

    data = extract_trip_request_from_message(
        message=(
            "2026年7月2日从杭州去成都玩三天，"
            "必须直飞，航班不能早于8点，酒店每晚不能超过600元"
        ),
        reference_date=date(2026, 7, 1),
    ).to_state_dict()

    constraints = {
        (item["domain"], item["field"], item["operator"]): item["value"]
        for item in data["hard_constraints"]
    }

    assert constraints[("flight", "is_direct", "==")] is True
    assert constraints[("flight", "depart_time", ">=")] == "08:00"
    assert constraints[("hotel", "price_per_night", "<=")] == 600.0


def test_extract_spend_preferences_for_dynamic_budget():
    """住宿、交通、餐饮投入倾向应被单独结构化。"""

    data = extract_trip_request_from_message(
        message=(
            "2026年7月2日从杭州去成都玩三天，预算4000，"
            "住宿好一点，航班不用很贵，吃的东西好一点"
        ),
        reference_date=date(2026, 7, 1),
    ).to_state_dict()

    signals = {
        (item["category"], item["direction"]): item["strength"]
        for item in data["spend_preferences"]
    }

    assert signals[("hotel", "increase")] == 0.9
    assert signals[("transport", "decrease")] == 0.85
    assert signals[("food_activity", "increase")] == 0.85


def test_extract_named_hotel_and_food():
    """简单指定酒店和食物在规则 fallback 中也能保留。"""

    data = extract_trip_request_from_message(
        message=(
            "2026年7月2日从杭州去成都玩三天，"
            "我只想住成都青羊静巷酒店，还想吃正宗的鸭血粉丝汤"
        ),
        reference_date=date(2026, 7, 1),
    ).to_state_dict()

    named = {
        (item["entity_type"], item["entity_name"]): item["constraint_mode"]
        for item in data["named_constraints"]
    }

    assert named[("hotel", "成都青羊静巷酒店")] == "required"
    assert named[("food", "鸭血粉丝汤")] == "preferred"


def test_month_only_date_is_not_fabricated():
    """规则版只记录月份，不把“7月出发”编成某一天。"""

    data = extract_trip_request_from_message(
        message="从杭州去成都玩三天，7月出发",
        reference_date=date(2026, 6, 1),
    ).to_state_dict()

    assert data["start_date"] is None
    assert data["partial_date"]["month"] == 7
    assert data["partial_date"]["day"] is None
