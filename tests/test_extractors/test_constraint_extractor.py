from __future__ import annotations

from app.extractors.constraint_extractor import RuleBasedConstraintExtractor


def test_extracts_objective_hard_constraints():
    """只有可验证的直飞、时间和价格限制才进入 hard_constraints。"""

    result = RuleBasedConstraintExtractor().extract(
        "必须直飞，航班不能早于8点，最晚20点前到达，酒店每晚不能超过600元"
    )

    constraints = {
        (item.domain, item.field, item.operator): item.value
        for item in result.hard_constraints
    }

    assert constraints[("flight", "is_direct", "==")] is True
    assert constraints[("flight", "depart_time", ">=")] == "08:00"
    assert constraints[("flight", "arrive_time", "<=")] == "20:00"
    assert constraints[("hotel", "price_per_night", "<=")] == 600.0


def test_subjective_strong_words_do_not_become_hard_constraints():
    """安静和干净是连续主观特征，强烈措辞也不能直接硬过滤。"""

    result = RuleBasedConstraintExtractor().extract(
        "一定要安静，必须干净，评分一定要高"
    )

    assert result.hard_constraints == []


def test_extracts_spend_preferences():
    """消费倾向只表示预算方向，不直接生成最终百分比。"""

    result = RuleBasedConstraintExtractor().extract(
        "住宿好一点，航班不用很贵，吃的东西好一点"
    )

    signals = {
        (item.category, item.direction): item.strength
        for item in result.spend_preferences
    }

    assert signals[("hotel", "increase")] == 0.9
    assert signals[("transport", "decrease")] == 0.85
    assert signals[("food_activity", "increase")] == 0.85


def test_extracts_named_hotel_and_food():
    """规则 fallback 覆盖简单指定酒店和食物表达。"""

    result = RuleBasedConstraintExtractor().extract(
        "我只想住成都青羊静巷酒店，还想吃正宗的鸭血粉丝汤"
    )

    named = {
        (item.entity_type, item.entity_name): item.constraint_mode
        for item in result.named_constraints
    }

    assert named[("hotel", "成都青羊静巷酒店")] == "required"
    assert named[("food", "鸭血粉丝汤")] == "preferred"


def test_month_only_date_is_preserved_without_inventing_day():
    """用户只说月份时保存 partial_date，不擅自生成具体出发日。"""

    result = RuleBasedConstraintExtractor().extract(
        "我准备7月出发"
    )

    assert result.partial_date is not None
    assert result.partial_date.month == 7
    assert result.partial_date.day is None
    assert result.partial_date.precision == "month"
