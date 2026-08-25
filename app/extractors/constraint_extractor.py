from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.schemas.request_constraint_schema import (
    HardConstraint,
    NamedConstraint,
    PartialDate,
    SpendPreference,
)


@dataclass(frozen=True)
class RuleConstraintExtraction:
    """
    规则版约束抽取的聚合结果。

    规则版只覆盖高确定性的常见表达。
    复杂实体、隐含语义和开放需求仍由 LLM 负责。
    """

    hard_constraints: list[HardConstraint]
    spend_preferences: list[SpendPreference]
    named_constraints: list[NamedConstraint]
    partial_date: PartialDate | None


class RuleBasedConstraintExtractor:
    """
    抽取客观硬约束、消费倾向、简单指定实体和不完整日期。

    设计边界：
        - “一定要安静”不是硬约束，只由 PreferenceExtractor 生成高权重偏好。
        - “必须直飞”“酒店每晚不能超过600”可以客观验证，进入硬约束。
    """

    def extract(self, text: str) -> RuleConstraintExtraction:
        """执行所有规则抽取，并返回结构化结果。"""

        normalized = text.strip()

        return RuleConstraintExtraction(
            hard_constraints=self._extract_hard_constraints(normalized),
            spend_preferences=self._extract_spend_preferences(normalized),
            named_constraints=self._extract_named_constraints(normalized),
            partial_date=self._extract_partial_date(normalized),
        )

    def _extract_hard_constraints(self, text: str) -> list[HardConstraint]:
        """抽取项目当前能够确定性验证的硬约束。"""

        constraints: list[HardConstraint] = []
        compact = re.sub(r"\s+", "", text)

        # 1. 只有“必须/只要/不能转机”等明确不可妥协表达，才把直飞设为硬约束。
        if re.search(r"(?:必须|一定要|只要)直飞|(?:不能|不要)转机", compact):
            constraints.append(
                HardConstraint(
                    domain="flight",
                    field="is_direct",
                    operator="==",
                    value=True,
                    evidence=_matched_text(
                        compact,
                        r"(?:必须|一定要|只要)直飞|(?:不能|不要)转机",
                    ),
                    source="rule",
                )
            )

        # 2. 起飞时间下限，例如“航班不能早于8点”。
        earliest_match = re.search(
            r"(?:起飞|航班|出发)(?:时间)?(?:不能|不要)早于(\d{1,2})(?::(\d{2}))?(?:点|时)?",
            compact,
        )

        if earliest_match:
            constraints.append(
                HardConstraint(
                    domain="flight",
                    field="depart_time",
                    operator=">=",
                    value=_format_time(
                        earliest_match.group(1),
                        earliest_match.group(2),
                    ),
                    evidence=earliest_match.group(0),
                    source="rule",
                )
            )

        # 3. 到达时间上限，例如“最晚20点前到达”。
        latest_match = re.search(
            r"(?:最晚|不能晚于|不要晚于)(\d{1,2})(?::(\d{2}))?(?:点|时)?(?:前)?(?:到达|抵达|到)?",
            compact,
        )

        if latest_match:
            constraints.append(
                HardConstraint(
                    domain="flight",
                    field="arrive_time",
                    operator="<=",
                    value=_format_time(
                        latest_match.group(1),
                        latest_match.group(2),
                    ),
                    evidence=latest_match.group(0),
                    source="rule",
                )
            )

        # 4. 酒店每晚上限，例如“酒店每晚不能超过600元”。
        hotel_price_match = re.search(
            r"(?:酒店|住宿)(?:每晚|一晚|每夜)?(?:价格)?(?:不能|不要|不)(?:超过|高于)|"
            r"(?:酒店|住宿)(?:每晚|一晚|每夜)?(?:控制在|最多|上限)",
            compact,
        )

        if hotel_price_match:
            segment = compact[hotel_price_match.start() : hotel_price_match.start() + 35]
            amount_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:元|块)?", segment)

            if amount_match:
                constraints.append(
                    HardConstraint(
                        domain="hotel",
                        field="price_per_night",
                        operator="<=",
                        value=float(amount_match.group(1)),
                        evidence=segment,
                        source="rule",
                    )
                )

        # 5. “必须靠近地铁”可以用 near_subway 布尔字段验证。
        if re.search(r"(?:必须|一定要|只住)靠近地铁|(?:必须|一定要)近地铁", compact):
            constraints.append(
                HardConstraint(
                    domain="hotel",
                    field="near_subway",
                    operator="==",
                    value=True,
                    evidence=_matched_text(
                        compact,
                        r"(?:必须|一定要|只住)靠近地铁|(?:必须|一定要)近地铁",
                    ),
                    source="rule",
                )
            )

        return _dedupe_models(
            constraints,
            key_builder=lambda item: (
                item.domain,
                item.field,
                item.operator,
                str(item.value),
            ),
        )

    def _extract_spend_preferences(self, text: str) -> list[SpendPreference]:
        """
        抽取预算投入方向。

        这里仅判断“更愿意把钱花在哪一类”，
        最终比例由 PlanningContextBuilder 使用确定性公式计算。
        """

        compact = re.sub(r"\s+", "", text)
        output: list[SpendPreference] = []

        rules: tuple[
            tuple[str, str, float, tuple[str, ...]],
            ...
        ] = (
            (
                "hotel",
                "increase",
                0.9,
                (
                    "住的地方好一点",
                    "住宿好一点",
                    "酒店好一点",
                    "住好一点",
                    "酒店可以贵一点",
                    "住宿可以贵一点",
                ),
            ),
            (
                "hotel",
                "decrease",
                0.8,
                (
                    "酒店便宜一点",
                    "住宿省一点",
                    "住的地方不用太贵",
                ),
            ),
            (
                "transport",
                "decrease",
                0.85,
                (
                    "航班不用很贵",
                    "机票便宜一点",
                    "航班省一点",
                    "交通省一点",
                ),
            ),
            (
                "transport",
                "increase",
                0.75,
                (
                    "航班舒服一点",
                    "机票可以贵一点",
                    "交通安排好一点",
                ),
            ),
            (
                "food_activity",
                "increase",
                0.85,
                (
                    "吃的东西好一点",
                    "吃好一点",
                    "餐饮好一点",
                    "美食预算多一点",
                    "玩得好一点",
                ),
            ),
            (
                "food_activity",
                "decrease",
                0.75,
                (
                    "吃饭省一点",
                    "景点省一点",
                    "餐饮不用太贵",
                ),
            ),
        )

        # 1. 每条规则命中任意表达后生成一条 SpendPreference。
        for category, direction, strength, patterns in rules:
            matched = next(
                (pattern for pattern in patterns if pattern in compact),
                None,
            )

            if not matched:
                continue

            output.append(
                SpendPreference(
                    category=category,  # type: ignore[arg-type]
                    direction=direction,  # type: ignore[arg-type]
                    strength=strength,
                    evidence=matched,
                    source="rule",
                )
            )

        return _dedupe_models(
            output,
            key_builder=lambda item: (
                item.category,
                item.direction,
            ),
        )

    def _extract_named_constraints(self, text: str) -> list[NamedConstraint]:
        """
        抽取少量高确定性的指定实体。

        规则 fallback 只覆盖简单酒店和食物表达；
        景点别名、间接描述和复杂实体关系主要由 LLM 处理。
        """

        compact = re.sub(r"\s+", "", text)
        output: list[NamedConstraint] = []

        # 1. 酒店名称通常以“酒店/宾馆/客栈/民宿”结尾。
        hotel_match = re.search(
            r"(?P<prefix>只想住|只住|必须住|想住|要住|住在)"
            r"(?P<name>[^，。；]{2,30}?(?:酒店|宾馆|客栈|民宿))",
            compact,
        )

        if hotel_match:
            prefix = hotel_match.group("prefix")
            mode = "required" if prefix in {"只想住", "只住", "必须住"} else "preferred"

            output.append(
                NamedConstraint(
                    entity_type="hotel",
                    entity_name=hotel_match.group("name"),
                    constraint_mode=mode,
                    scope="whole_trip",
                    evidence=hotel_match.group(0),
                    source="rule",
                )
            )

        # 2. 简单食物表达，例如“我想吃正宗的鸭血粉丝汤”。
        food_match = re.search(
            r"(?:想吃|要吃|一定要吃|必须吃)(?P<name>[^，。；]{2,20})",
            compact,
        )

        if food_match:
            raw_name = food_match.group("name")
            food_name = re.sub(r"^(?:正宗的|当地的|本地的)", "", raw_name)

            output.append(
                NamedConstraint(
                    entity_type="food",
                    entity_name=food_name,
                    constraint_mode=(
                        "required"
                        if food_match.group(0).startswith(("一定要吃", "必须吃"))
                        else "preferred"
                    ),
                    scope="itinerary",
                    evidence=food_match.group(0),
                    source="rule",
                )
            )

        return _dedupe_models(
            output,
            key_builder=lambda item: (
                item.entity_type,
                item.entity_name,
                item.constraint_mode,
            ),
        )

    def _extract_partial_date(self, text: str) -> PartialDate | None:
        """识别“7月出发”这类只精确到月份的日期表达。"""

        compact = re.sub(r"\s+", "", text)

        # 完整日期由 TripRequestExtractor 处理，这里只匹配没有具体日的月份。
        match = re.search(
            r"(?:(\d{4})年)?(\d{1,2})月(?:份|左右)?(?:出发|去|旅行|旅游|启程)",
            compact,
        )

        if not match:
            return None

        return PartialDate(
            year=int(match.group(1)) if match.group(1) else None,
            month=int(match.group(2)),
            day=None,
            precision="month",
            evidence=match.group(0),
        )


def _format_time(hour: str, minute: str | None) -> str:
    """把 8 点、8:30 标准化为 HH:MM。"""

    return f"{int(hour):02d}:{int(minute or 0):02d}"


def _matched_text(text: str, pattern: str) -> str:
    """返回正则命中的原文；未命中时返回空字符串。"""

    match = re.search(pattern, text)
    return match.group(0) if match else ""


def _dedupe_models(items: list[Any], key_builder) -> list[Any]:
    """按指定 key 去重，保持首次出现顺序。"""

    seen: set[Any] = set()
    output: list[Any] = []

    for item in items:
        key = key_builder(item)

        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output
