from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.schemas.preference_schema import PreferenceSignal


@dataclass(frozen=True)
class PreferenceRule:
    """
    一条主观偏好抽取规则。

    这里不再保存 default_hard_constraint。
    程度词只影响 strength，不自动把主观偏好转成硬约束。
    """

    domain: str
    key: str
    label: str
    patterns: tuple[str, ...]
    default_strength: float = 0.85
    polarity: str = "positive"


# “一定、必须”在主观属性中只表示强烈偏好。
# 例如“一定要安静”会得到 strength=1.0，但不会直接过滤酒店。
VERY_STRONG_MARKERS = (
    "一定要",
    "必须",
    "务必",
    "绝对",
    "特别",
    "非常",
)

STRONG_MARKERS = (
    "很",
    "希望",
    "想要",
    "更想",
    "更喜欢",
    "优先",
)

PREFER_MARKERS = (
    "最好",
    "尽量",
    "较高",
    "比较好",
    "高一点",
)

MILD_MARKERS = (
    "有点",
    "稍微",
    "可以",
    "一点",
)

NEGATIVE_MARKERS = (
    "不要",
    "不想",
    "避免",
    "别",
)


PREFERENCE_RULES: tuple[PreferenceRule, ...] = (
    # ------------------------------------------------------------------
    # 航班偏好
    # ------------------------------------------------------------------
    PreferenceRule(
        domain="flight",
        key="avoid_early_flight",
        label="避免早班机",
        patterns=(
            "不想坐太早",
            "不要早班",
            "避开早班",
            "避免早班",
            "太早航班",
            "早班机",
            "不要太早出发",
        ),
        default_strength=0.9,
    ),
    PreferenceRule(
        domain="flight",
        key="avoid_late_arrival",
        label="避免太晚抵达",
        patterns=(
            "不要太晚到",
            "别太晚到",
            "不想半夜到",
            "不要半夜到",
            "太晚到",
            "半夜到",
        ),
        default_strength=0.9,
    ),
    PreferenceRule(
        domain="flight",
        key="prefer_direct",
        label="偏好直飞",
        patterns=("直飞", "直达", "不转机", "不要转机"),
        default_strength=0.85,
    ),
    PreferenceRule(
        domain="flight",
        key="price_sensitive",
        label="航班价格敏感",
        patterns=(
            "机票便宜",
            "航班便宜",
            "航班不用很贵",
            "交通省一点",
            "价格低",
            "预算有限",
        ),
        default_strength=0.8,
    ),

    # ------------------------------------------------------------------
    # 酒店偏好
    # ------------------------------------------------------------------
    PreferenceRule(
        domain="hotel",
        key="quiet",
        label="安静",
        patterns=("安静", "不要吵", "别太吵", "清静", "隔音"),
        default_strength=0.85,
    ),
    PreferenceRule(
        domain="hotel",
        key="cleanliness",
        label="干净",
        patterns=("干净", "清洁", "卫生", "整洁"),
        default_strength=0.9,
    ),
    PreferenceRule(
        domain="hotel",
        key="near_subway",
        label="靠近地铁",
        patterns=(
            "离地铁近",
            "近地铁",
            "靠地铁",
            "靠近地铁",
            "地铁附近",
            "交通方便",
            "交通便利",
        ),
        default_strength=0.85,
    ),
    PreferenceRule(
        domain="hotel",
        key="high_rating",
        label="高评分",
        patterns=("评分较高", "评分高", "高评分", "评价好", "口碑好"),
        default_strength=0.75,
    ),
    PreferenceRule(
        domain="hotel",
        key="budget_friendly",
        label="酒店预算友好",
        patterns=("便宜酒店", "性价比酒店", "预算友好", "酒店便宜"),
        default_strength=0.75,
    ),

    # ------------------------------------------------------------------
    # 活动偏好
    # ------------------------------------------------------------------
    PreferenceRule(
        domain="activity",
        key="food",
        label="美食",
        patterns=("美食", "小吃", "火锅", "川菜", "茶馆", "吃好一点"),
        default_strength=0.9,
    ),
    PreferenceRule(
        domain="activity",
        key="nature_scenery",
        label="自然风景",
        patterns=("自然风景", "美景", "风景", "公园", "山水", "自然"),
        default_strength=0.85,
    ),
    PreferenceRule(
        domain="activity",
        key="culture_history",
        label="文化历史",
        patterns=("文化", "历史", "博物馆", "人文", "古迹"),
        default_strength=0.8,
    ),
    PreferenceRule(
        domain="activity",
        key="entertainment",
        label="玩乐娱乐",
        patterns=("玩乐", "娱乐", "夜生活", "演出"),
        default_strength=0.75,
    ),
    PreferenceRule(
        domain="activity",
        key="fitness",
        label="锻炼运动",
        patterns=("锻炼", "健身", "跑步", "运动"),
        default_strength=0.75,
    ),
    PreferenceRule(
        domain="activity",
        key="adventure",
        label="冒险户外",
        patterns=("冒险", "刺激", "徒步", "户外"),
        default_strength=0.75,
    ),
    PreferenceRule(
        domain="activity",
        key="photography",
        label="拍照摄影",
        patterns=("拍照", "摄影", "出片"),
        default_strength=0.75,
    ),

    # ------------------------------------------------------------------
    # 节奏偏好
    # ------------------------------------------------------------------
    PreferenceRule(
        domain="pace",
        key="slow",
        label="慢节奏",
        patterns=("慢节奏", "轻松", "不要太赶", "别太赶", "不要太累", "休闲"),
        default_strength=0.9,
    ),
    PreferenceRule(
        domain="pace",
        key="low_walking_intensity",
        label="低步行强度",
        patterns=("少走路", "不想走太多", "低强度", "不累"),
        default_strength=0.8,
    ),

    # ------------------------------------------------------------------
    # 风险偏好
    # ------------------------------------------------------------------
    PreferenceRule(
        domain="risk",
        key="weather_sensitive",
        label="天气敏感",
        patterns=("怕下雨", "天气敏感", "不要淋雨", "雨天少走"),
        default_strength=0.75,
    ),
    PreferenceRule(
        domain="risk",
        key="avoid_crowds",
        label="避开人群",
        patterns=("避开人群", "不要人多", "别太挤", "人少一点"),
        default_strength=0.75,
    ),
)


class RuleBasedPreferenceExtractor:
    """
    规则版主观偏好抽取器。

    它只抽取“用于评分”的偏好。
    真正硬约束、消费倾向和指定实体由 RuleBasedConstraintExtractor 处理。
    """

    def extract(self, text: str) -> list[PreferenceSignal]:
        """
        从用户原话中抽取 PreferenceSignal。
        """

        text = text.strip()
        signals: list[PreferenceSignal] = []

        # 1. 遍历规则表，寻找每个偏好的第一个匹配表达。
        for rule in PREFERENCE_RULES:
            for pattern in rule.patterns:
                match = re.search(
                    re.escape(pattern),
                    text,
                    flags=re.IGNORECASE,
                )

                if not match:
                    continue

                # 2. 截取关键词附近的原文，用于判断程度词和保存证据。
                context = _window_text(
                    text,
                    match.start(),
                    match.end(),
                )

                strength = _infer_strength(
                    context=context,
                    default_strength=rule.default_strength,
                )

                signals.append(
                    PreferenceSignal(
                        domain=rule.domain,  # type: ignore[arg-type]
                        key=rule.key,
                        label=rule.label,
                        strength=strength,
                        polarity=rule.polarity,  # type: ignore[arg-type]
                        evidence=context,
                        source="rule",
                    )
                )
                break

        # 3. 同一 domain.key 多次命中时，只保留强度最高的一条。
        return _deduplicate_signals(signals)


def _infer_strength(
    context: str,
    default_strength: float,
) -> float:
    """
    根据程度词推断 0—1 偏好强度。

    关键规则：
        “一定、必须”只将 strength 提高到 1.0，
        不在这里创建 HardConstraint。
    """

    compact = re.sub(r"\s+", "", context)

    if _contains_any(compact, VERY_STRONG_MARKERS):
        return 1.0

    if _contains_any(compact, STRONG_MARKERS):
        return max(default_strength, 0.9)

    if _contains_any(compact, PREFER_MARKERS):
        return max(default_strength, 0.75)

    if _contains_any(compact, MILD_MARKERS):
        return min(default_strength, 0.65)

    if _contains_any(compact, NEGATIVE_MARKERS):
        return max(default_strength, 0.85)

    return default_strength


def _window_text(
    text: str,
    start: int,
    end: int,
    window: int = 8,
) -> str:
    """截取关键词前后少量原文，避免 evidence 过长。"""

    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right]


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    """判断文本是否包含任意关键词。"""

    return any(keyword in text for keyword in keywords)


def _deduplicate_signals(
    signals: list[PreferenceSignal],
) -> list[PreferenceSignal]:
    """同一 domain.key 多次命中时保留 strength 最高的一条。"""

    best: dict[tuple[str, str], PreferenceSignal] = {}

    for signal in signals:
        key = (signal.domain, signal.key)
        old = best.get(key)

        if old is None or signal.strength > old.strength:
            best[key] = signal

    return list(best.values())


def signals_to_state_dicts(
    signals: list[PreferenceSignal],
) -> list[dict]:
    """将 PreferenceSignal 列表转换成普通 dict 列表。"""

    return [signal.to_state_dict() for signal in signals]
