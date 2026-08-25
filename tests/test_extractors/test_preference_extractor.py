from __future__ import annotations

from app.extractors.preference_extractor import RuleBasedPreferenceExtractor


def _find_signal(signals, domain: str, key: str):
    """从测试结果中找到指定偏好。"""

    for signal in signals:
        if signal.domain == domain and signal.key == key:
            return signal

    raise AssertionError(f"missing signal: {domain}.{key}")


def test_extract_hotel_preferences_and_strong_word_only_increases_weight():
    """
    “一定要安静”只提高 quiet 权重，不生成偏好型硬约束。
    """

    signals = RuleBasedPreferenceExtractor().extract(
        "离地铁近，一定要安静，干净的酒店，评分较高"
    )

    near_subway = _find_signal(signals, "hotel", "near_subway")
    quiet = _find_signal(signals, "hotel", "quiet")
    cleanliness = _find_signal(signals, "hotel", "cleanliness")
    high_rating = _find_signal(signals, "hotel", "high_rating")

    assert near_subway.strength >= 0.8
    assert quiet.strength == 1.0
    assert "一定要安静" in quiet.evidence
    assert not hasattr(quiet, "hard_constraint")
    assert cleanliness.strength >= 0.85
    assert high_rating.strength >= 0.75


def test_extract_activity_and_pace_preferences():
    """活动和节奏偏好应进入不同 domain。"""

    signals = RuleBasedPreferenceExtractor().extract(
        "喜欢美食和自然风景，行程不要太赶"
    )

    assert _find_signal(signals, "activity", "food").strength >= 0.85
    assert _find_signal(signals, "activity", "nature_scenery").strength >= 0.8
    assert _find_signal(signals, "pace", "slow").strength >= 0.85


def test_extract_flight_preferences():
    """常见航班偏好应被规则 fallback 识别。"""

    signals = RuleBasedPreferenceExtractor().extract(
        "不要太早出发，也不想半夜到，最好直飞"
    )

    assert _find_signal(signals, "flight", "avoid_early_flight").strength >= 0.85
    assert _find_signal(signals, "flight", "avoid_late_arrival").strength >= 0.85
    assert _find_signal(signals, "flight", "prefer_direct").strength >= 0.75


def test_deduplicate_keeps_strongest_signal():
    """同一 domain.key 多次命中时保留最高强度。"""

    signals = RuleBasedPreferenceExtractor().extract(
        "最好安静，一定要安静"
    )

    quiet_signals = [
        signal
        for signal in signals
        if signal.domain == "hotel" and signal.key == "quiet"
    ]

    assert len(quiet_signals) == 1
    assert quiet_signals[0].strength == 1.0
