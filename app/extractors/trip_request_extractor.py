from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Iterable, Optional, Protocol

from app.extractors.constraint_extractor import RuleBasedConstraintExtractor
from app.extractors.preference_extractor import RuleBasedPreferenceExtractor
from app.schemas.preference_schema import PreferenceSignal
from app.schemas.trip_request_schema import TripRequest


# ----------------------------------------------------------------------
# 已知城市表
# ----------------------------------------------------------------------
# 规则抽取器不可能识别世界上所有城市。
# MVP 阶段先用一份常见城市列表，保证测试稳定。
#
# 后续升级方向：
# 1. 从配置文件加载城市表。
# 2. 从数据库 city_catalog 表加载。
# 3. 当规则抽取失败时，用 LLM structured output fallback。
KNOWN_CITIES = [
    "北京",
    "上海",
    "广州",
    "深圳",
    "杭州",
    "成都",
    "重庆",
    "长沙",
    "西安",
    "南京",
    "苏州",
    "厦门",
    "武汉",
    "青岛",
    "昆明",
    "大理",
    "丽江",
    "三亚",
    "贵阳",
    "天津",
    "福州",
    "郑州",
]


# ----------------------------------------------------------------------
# 中文数字映射
# ----------------------------------------------------------------------
# 规则抽取里经常遇到：
# - 三天
# - 两个人
# - 预算五千
#
# 这些不能直接用 int() 转换，所以需要简单中文数字解析。
_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

_CHINESE_UNITS = {
    "十": 10,
    "百": 100,
    "千": 1000,
    "万": 10000,
}


# ----------------------------------------------------------------------
# 常用正则片段
# ----------------------------------------------------------------------
# 这些片段集中定义，便于后续维护。
#
# _DAY_TOKEN:
#   用于匹配天数，例如 3、三、两、十。
#
# _MONEY_TOKEN:
#   用于匹配预算，例如 4000、4k、五千。
#
# _DATE_TOKEN:
#   用于匹配日期，例如：
#   - 2026年7月2日
#   - 2026-07-02
#   - 7月2日
_DAY_TOKEN = r"\d+|[零〇一二两三四五六七八九十]+"
_MONEY_TOKEN = r"\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+"
# 日期 token。
#
# 支持：
# - 2026年7月2日
# - 2026年7月2号
# - 2026-07-02
# - 2026/07/02
# - 7月2日
# - 7月2号
#
# 注意：
# 这里只负责匹配“日期长什么样”。
# 日期是 start_date 还是 end_date，需要后面的语义规则判断。
_DATE_TOKEN = r"(?:\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?|\d{1,2}月\d{1,2}[日号]?)"


class TripRequestExtractor(Protocol):
    """
    TripRequest 抽取器协议。

    为什么要定义 Protocol？

    因为 InputExtractNode 不应该绑定某一个具体抽取器。
    它只需要知道：
        你有一个 extract(...) 方法，
        输入 message/user_id/reference_date，
        输出 TripRequest。

    这样后面可以无缝替换成：

        RuleBasedTripRequestExtractor
        LLMTripRequestExtractor
        HybridTripRequestExtractor

    这就是策略模式的一种轻量写法。
    """

    def extract(
        self,
        message: str,
        user_id: str | None = None,
        reference_date: date | None = None,
    ) -> TripRequest:
        ...


class RuleBasedTripRequestExtractor:
    """
    规则版 TripRequest 抽取器。

    职责：
    - 从用户自然语言中抽取结构化旅行需求。
    - 返回 TripRequest。
    - 不读写 TravelState。
    - 不调用数据库。
    - 不调用 RAG。
    - 不调用 mock 外部数据。
    - 不调用 LLM。

    为什么不放在 Node 里？

    因为 Node 的职责应该是：
    - 读 State
    - 调用 extractor/tool
    - 写 State
    - 记录 trace/errors

    抽取规则是一个可替换策略，不应该和 State 编排代码混在一起。
    """

    def __init__(
            self,
            known_cities: Optional[list[str]] = None,
            preference_extractor: RuleBasedPreferenceExtractor | None = None,
            constraint_extractor: RuleBasedConstraintExtractor | None = None,
    ) -> None:
        self.known_cities = known_cities or KNOWN_CITIES
        self.preference_extractor = preference_extractor or RuleBasedPreferenceExtractor()
        self.constraint_extractor = constraint_extractor or RuleBasedConstraintExtractor()

        self.city_pattern = "|".join(
            re.escape(city)
            for city in sorted(self.known_cities, key=len, reverse=True)
        )

    def extract(
            self,
            message: str,
            user_id: str | None = None,
            reference_date: date | None = None,
    ) -> TripRequest:
        """
        从用户输入中抽取 TripRequest。

        这一步只做“结构化抽取”，不查天气、不查航班、不查酒店。

        日期抽取的关键规则：

        1. 如果用户说“7月2日到7月4日”，这是日期范围。
        2. 如果用户说“7月2日出发”，这是 start_date。
        3. 如果用户说“7月8号回来”，这是 end_date。
        4. 如果用户只说“7月2日”，没有角色词，才默认当作 start_date。
        """

        message = message.strip()
        reference_date = reference_date or date.today()

        # 1. 抽取出发地和目的地。
        origin, destination = self._extract_cities(message)

        # 2. 先抽取天数。
        #
        # 为什么先抽 days？
        # 因为 “7月8号回来，玩三天” 这种表达里，
        # 7月8号是 end_date，需要用 days 反推出 start_date。
        days = self._extract_days(message)

        # 3. 抽取日期。
        #
        # 这里不再简单地把第一个日期当作 start_date，
        # 而是区分出发日期、返程日期、日期范围和普通单日期。
        start_date, end_date = self._extract_dates(
            text=message,
            reference_date=reference_date,
            days=days,
        )

        # 4. 如果用户给了日期范围，但没说天数，则用日期范围推导天数。
        if start_date and end_date and end_date >= start_date and days is None:
            days = (end_date - start_date).days + 1

        # 5. 如果用户给了 start_date + days，但没给 end_date，则推导 end_date。
        if start_date and days and end_date is None:
            end_date = start_date + timedelta(days=days - 1)

        # 6. 如果用户给了 end_date + days，但没给 start_date，则反推 start_date。
        #
        # 例如：
        #   “从杭州去成都玩三天，7月8号回来”
        #
        # 结果：
        #   end_date = 2026-07-08
        #   days = 3
        #   start_date = 2026-07-06
        if end_date and days and start_date is None:
            start_date = end_date - timedelta(days=days - 1)

        # 7. 抽取预算、人数和房间数量。
        budget = self._extract_budget(message)
        people_count = self._extract_people_count(message)
        room_count = self._extract_room_count(
            text=message,
            people_count=people_count,
        )

        # 8. 抽取用于 CandidateRank 打分的主观偏好。
        preference_signals = self.preference_extractor.extract(message)

        # 9. 抽取真正硬约束、消费倾向、简单指定实体和不完整日期。
        constraint_result = self.constraint_extractor.extract(message)

        # 10. 从 PreferenceSignal 派生兼容性的轻量 list 字段。
        (
            transport_preferences,
            hotel_preferences,
            travel_style,
            raw_constraints,
        ) = self._derive_preference_lists(preference_signals)

        # 11. 将硬约束、消费倾向和指定实体的原文也加入 raw_constraints，
        #     方便 Trace 与后续调试。
        raw_constraints = _dedupe(
            [
                *raw_constraints,
                *[item.evidence for item in constraint_result.hard_constraints if item.evidence],
                *[item.evidence for item in constraint_result.spend_preferences if item.evidence],
                *[item.evidence for item in constraint_result.named_constraints if item.evidence],
            ]
        )

        trip_request = TripRequest(
            user_id=user_id,
            origin=origin,
            destination=destination,
            start_date=start_date,
            end_date=end_date,
            days=days,
            budget=budget,
            people_count=people_count,
            room_count=room_count,
            partial_date=constraint_result.partial_date,
            transport_preferences=transport_preferences,
            hotel_preferences=hotel_preferences,
            travel_style=travel_style,
            raw_constraints=raw_constraints,
            preference_signals=preference_signals,
            spend_preferences=constraint_result.spend_preferences,
            hard_constraints=constraint_result.hard_constraints,
            named_constraints=constraint_result.named_constraints,
            field_resolution=self._build_field_resolution(
                origin=origin,
                destination=destination,
                start_date=start_date,
                end_date=end_date,
                days=days,
                budget=budget,
                people_count=people_count,
                room_count=room_count,
                partial_date=constraint_result.partial_date,
            ),
            raw_message=message,
        )

        # 12. 写入抽取元信息。
        trip_request.extraction = {
            "method": "rule_based_v2",
            "confidence": self._estimate_confidence(trip_request),
            "missing_fields": trip_request.required_missing_fields(),
            "notes": [
                "规则版只覆盖确定性字段和常见偏好，复杂语义由 Hybrid Extractor 的 LLM 主路径处理。",
                "主观程度词只提高偏好权重，不自动生成硬约束。",
                "budget 不是 plan_trip MVP 必填字段。",
            ],
        }

        return trip_request

    def _extract_cities(self, text: str) -> tuple[str | None, str | None]:
        """
        抽取出发地和目的地。

        支持表达：
        - 从杭州去成都
        - 杭州到成都
        - 杭州出发去成都
        - 想去成都三日游

        返回：
            (origin, destination)
        """

        compact = _compact_text(text)

        # 常见明确路径表达。
        patterns = [
            rf"从({self.city_pattern})(?:出发)?(?:去|到|飞往|飞|前往)({self.city_pattern})",
            rf"({self.city_pattern})(?:出发|起飞)(?:.*?)(?:去|到|飞往|飞|前往)({self.city_pattern})",
            rf"({self.city_pattern})(?:去|到|飞往|飞|前往)({self.city_pattern})",
        ]

        for pattern in patterns:
            match = re.search(pattern, compact)
            if match:
                origin = match.group(1)
                destination = match.group(2)

                if origin != destination:
                    return origin, destination

        origin = None
        destination = None

        # 单独抽取出发地。
        origin_match = re.search(rf"从({self.city_pattern})(?:出发)?", compact)
        if not origin_match:
            origin_match = re.search(rf"({self.city_pattern})(?:出发|起飞)", compact)

        if origin_match:
            origin = origin_match.group(1)

        # 单独抽取目的地。
        destination_match = re.search(
            rf"(?:想去|去|到|前往|飞往)({self.city_pattern})",
            compact,
        )

        if destination_match:
            destination = destination_match.group(1)

        # 处理“成都三日游”“成都旅游”这种只有目的地的表达。
        if destination is None:
            for city in self.known_cities:
                destination_contexts = [
                    f"{city}三日游",
                    f"{city}两日游",
                    f"{city}四日游",
                    f"{city}旅游",
                    f"{city}旅行",
                    f"{city}玩",
                ]
                if any(context in compact for context in destination_contexts):
                    destination = city
                    break

        # 最后兜底：
        # 如果文本里出现两个城市，默认第一个是出发地，最后一个是目的地。
        if destination is None:
            appeared = [
                match.group(0)
                for match in re.finditer(self.city_pattern, compact)
            ]

            if len(appeared) >= 2:
                origin = origin or appeared[0]
                destination = appeared[-1]

            elif len(appeared) == 1 and origin is None:
                only_city = appeared[0]

                # 如果不是“从这个城市出发”，则更可能是目的地。
                if f"从{only_city}" not in compact and f"{only_city}出发" not in compact:
                    destination = only_city

        # 如果误抽成同一个城市，就清空 origin，避免错误路线。
        if origin == destination:
            origin = None

        return origin, destination

    def _extract_dates(
            self,
            text: str,
            reference_date: date,
            days: int | None,
    ) -> tuple[date | None, date | None]:
        """
        抽取 start_date 和 end_date。

        这是日期抽取的总入口。

        处理优先级：

        1. 明确日期范围：
           “7月2日到7月4日”

        2. 明确出发日期 / 明确返程日期：
           “7月2日出发”
           “7月8号回来”

        3. 普通单日期：
           “7月2日从杭州去成都”
           “7月2日去成都”

        4. 相对日期：
           “明天”
           “后天”
        """

        # 1. 日期范围优先级最高。
        range_start, range_end = self._extract_explicit_date_range(
            text=text,
            reference_date=reference_date,
        )
        if range_start or range_end:
            return range_start, range_end

        # 2. 明确出发日期。
        departure_date = self._extract_departure_date(
            text=text,
            reference_date=reference_date,
        )

        # 3. 明确返程日期。
        return_date = self._extract_return_date(
            text=text,
            reference_date=reference_date,
        )

        # 4. 如果同时有出发和返程日期，直接返回。
        if departure_date and return_date:
            if return_date >= departure_date:
                return departure_date, return_date

            # 如果返程日期早于出发日期，说明抽取可能有误。
            # 这里保守返回出发日期，end_date 留空，后续 Missing/Verifier 可以处理。
            return departure_date, None

        # 5. 只有出发日期。
        if departure_date:
            return departure_date, None

        # 6. 只有返程日期。
        #
        # 如果有 days，可以反推出 start_date；
        # 如果没有 days，则 start_date 仍然缺失，后续 MissingInfoCheckNode 会澄清。
        if return_date:
            if days:
                return return_date - timedelta(days=days - 1), return_date

            return None, return_date

        # 7. 普通单日期。
        single_date = self._extract_single_start_date(
            text=text,
            reference_date=reference_date,
        )
        if single_date:
            return single_date, None

        # 8. 相对日期。
        relative_date = self._extract_relative_date(
            text=_compact_text(text),
            reference_date=reference_date,
        )
        if relative_date:
            return relative_date, None

        return None, None

    def _extract_explicit_date_range(
            self,
            text: str,
            reference_date: date,
    ) -> tuple[date | None, date | None]:
        """
        抽取明确日期范围。

        支持：
        - 7月2日到7月4日
        - 7月2日至7月4日
        - 2026-07-02~2026-07-04
        """

        compact = _compact_text(text)

        range_match = re.search(
            rf"({_DATE_TOKEN})\s*(?:到|至|~|—|——|--)\s*({_DATE_TOKEN})",
            compact,
        )

        if not range_match:
            return None, None

        start_token = range_match.group(1)
        end_token = range_match.group(2)

        start = _parse_date_token(start_token, reference_date)

        # 如果结束日期没有年份，用 start 的年份作为参考更合理。
        # 例如 reference_date 是 2026-06-30，
        # 用户说 7月2日到7月4日，
        # start/end 都应该是 2026 年。
        end_reference = start or reference_date
        end = _parse_date_token(end_token, end_reference)

        if start and end and end >= start:
            return start, end

        return start, None

    def _extract_departure_date(
            self,
            text: str,
            reference_date: date,
    ) -> date | None:
        """
        抽取明确出发日期。

        支持：
        - 7月2日出发
        - 7月2日起飞
        - 7月2日从杭州去成都
        - 出发日期是7月2日

        注意：
        这个函数只处理有“出发语义”的日期。
        """

        compact = _compact_text(text)

        patterns = [
            # 日期在前，出发语义在后。
            rf"({_DATE_TOKEN})(?:从|出发|起飞|去|前往|飞往)",

            # 出发语义在前，日期在后。
            rf"(?:出发日期|出发时间|启程日期|启程|出发|起飞)(?:是|在|为)?({_DATE_TOKEN})",
        ]

        for pattern in patterns:
            match = re.search(pattern, compact)
            if match:
                return _parse_date_token(match.group(1), reference_date)

        return None

    def _extract_return_date(
            self,
            text: str,
            reference_date: date,
    ) -> date | None:
        """
        抽取明确返程日期。

        支持：
        - 7月8号回来
        - 7月8日返程
        - 7月8号返回杭州
        - 返程日期是7月8日

        这个函数是为了解决：
            “从杭州去成都玩三天，7月8号回来”

        在这个句子里，7月8号不是 start_date，而是 end_date。
        """

        compact = _compact_text(text)

        patterns = [
            # 日期在前，返程语义在后。
            rf"({_DATE_TOKEN})(?:回来|返回|返程|回程|离开|走|回)",

            # 返程语义在前，日期在后。
            rf"(?:回来|返回|返程|回程|离开|回)(?:日期|时间)?(?:是|在|为)?({_DATE_TOKEN})",
        ]

        for pattern in patterns:
            match = re.search(pattern, compact)
            if match:
                return _parse_date_token(match.group(1), reference_date)

        return None

    def _extract_single_start_date(
            self,
            text: str,
            reference_date: date,
    ) -> date | None:
        """
        抽取普通单日期。

        只有在日期没有明显“返程语义”时，才把它当作 start_date。

        例如：
        - 7月2日从杭州去成都
        - 7月2日去成都
        - 7月2日玩三天

        但下面这种不能当 start_date：
        - 7月8号回来
        """

        compact = _compact_text(text)

        for match in re.finditer(_DATE_TOKEN, compact):
            before = compact[max(0, match.start() - 8): match.start()]
            after = compact[match.end(): match.end() + 8]

            # 如果日期附近出现返程语义，则跳过。
            # 因为这种日期更可能是 end_date。
            if _contains_any(
                    before + after,
                    ["回来", "返回", "返程", "回程", "离开"],
            ):
                continue

            return _parse_date_token(match.group(0), reference_date)

        return None

    def _extract_relative_date(self, text: str, reference_date: date) -> date | None:
        """
        抽取相对日期。

        当前只支持：
        - 今天
        - 明天
        - 后天

        更复杂表达如“下周五”后续可以交给 LLM fallback。
        """

        if "后天" in text:
            return reference_date + timedelta(days=2)

        if "明天" in text:
            return reference_date + timedelta(days=1)

        if "今天" in text:
            return reference_date

        return None

    def _extract_days(self, text: str) -> int | None:
        """
        抽取旅行天数。

        支持：
        - 玩三天
        - 旅行3天
        - 三日游
        - 3天2晚
        - 住2晚 -> 推导为 3 天
        """

        compact = _compact_text(text)

        # 例如：3天2晚。
        day_night_match = re.search(
            rf"({_DAY_TOKEN})\s*(?:天|日)\s*({_DAY_TOKEN})\s*晚",
            compact,
        )
        if day_night_match:
            return _token_to_int(day_night_match.group(1))

        # 例如：玩三天、三日游。
        patterns = [
            rf"(?:玩|游玩|旅行|待|安排)\s*({_DAY_TOKEN})\s*(?:天|日)",
            rf"({_DAY_TOKEN})\s*(?:天|日)\s*(?:游|行程|旅行)",
        ]

        for pattern in patterns:
            match = re.search(pattern, compact)
            if match:
                return _token_to_int(match.group(1))

        # 例如：住2晚，通常旅行天数 = 晚数 + 1。
        night_match = re.search(rf"住\s*({_DAY_TOKEN})\s*晚", compact)
        if night_match:
            nights = _token_to_int(night_match.group(1))
            if nights is not None:
                return nights + 1

        return None

    def _extract_budget(self, text: str) -> float | None:
        """
        抽取预算。

        支持：
        - 预算4000
        - 预算五千
        - 控制在4k
        - 不超过4000
        - 4000元以内

        返回 float，单位是元。
        """

        compact = _compact_text(text)

        # 先找带预算上下文的表达。
        contexts = [
            "总预算",
            "预算",
            "控制在",
            "不超过",
            "不要超过",
            "最多",
            "上限",
            "费用",
            "花费",
        ]

        for context in contexts:
            index = compact.find(context)
            if index == -1:
                continue

            # 只看预算词后面一小段，避免误抽其他数字。
            segment = compact[index : index + 30]
            match = re.search(rf"({_MONEY_TOKEN})\s*(k|K|千|万)?", segment)
            if match:
                return _money_token_to_float(match.group(1), match.group(2))

        # 处理“4000元以内”这种没有预算关键词但有金额后缀的表达。
        suffix_match = re.search(
            r"([1-9]\d{2,5})(?:元|块|以内|以下|左右)",
            compact,
        )
        if suffix_match:
            return float(suffix_match.group(1))

        # 处理“4k以内”“5千左右”。
        unit_match = re.search(
            r"([1-9]\d?(?:\.\d+)?)(k|K|千|万)(?:元|块|以内|以下|左右)?",
            compact,
        )
        if unit_match:
            return _money_token_to_float(unit_match.group(1), unit_match.group(2))

        return None

    def _extract_people_count(self, text: str) -> int:
        """
        抽取出行人数。

        支持：
        - 一家三口
        - 两个人
        - 两人
        - 3位
        """

        compact = _compact_text(text)

        if "一家三口" in compact:
            return 3

        if "两个人" in compact or "两人" in compact or "双人" in compact:
            return 2

        match = re.search(rf"({_DAY_TOKEN})\s*(?:个人|人|位|名)", compact)
        if match:
            people = _token_to_int(match.group(1))
            if people is not None and people >= 1:
                return people

        return 1

    def _extract_room_count(
        self,
        text: str,
        people_count: int,
    ) -> int | None:
        """
        抽取房间数量。

        规则：
            - 用户明确说“2间房”时使用明确值。
            - 单人旅行默认 1 间房。
            - 多人未说明房间数时保持 None，交给 Validator 澄清。
        """

        compact = _compact_text(text)
        match = re.search(rf"({_DAY_TOKEN})\s*(?:间房|个房间|间酒店房)", compact)

        if match:
            room_count = _token_to_int(match.group(1))
            if room_count is not None and room_count >= 1:
                return room_count

        if people_count == 1:
            return 1

        return None

    def _build_field_resolution(
        self,
        *,
        origin: str | None,
        destination: str | None,
        start_date: date | None,
        end_date: date | None,
        days: int | None,
        budget: float | None,
        people_count: int,
        room_count: int | None,
        partial_date,
    ) -> dict[str, dict]:
        """
        记录规则版核心字段的解析来源。

        规则版只区分 explicit、derived、incomplete 和 default。
        LLM Hybrid 合并后会补充 stable_world_knowledge 等来源。
        """

        result: dict[str, dict] = {}

        for field_name, value in {
            "origin": origin,
            "destination": destination,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "days": days,
            "budget": budget,
        }.items():
            if value is not None:
                result[field_name] = {
                    "value": value,
                    "resolution_type": "explicit",
                    "evidence": "",
                    "needs_clarification": False,
                }

        result["people_count"] = {
            "value": people_count,
            "resolution_type": "explicit" if people_count != 1 else "default",
            "evidence": "",
            "needs_clarification": False,
        }

        result["room_count"] = {
            "value": room_count,
            "resolution_type": (
                "default"
                if people_count == 1 and room_count == 1
                else "explicit"
                if room_count is not None
                else "incomplete"
            ),
            "evidence": "",
            "needs_clarification": people_count > 1 and room_count is None,
        }

        if start_date is None and partial_date is not None:
            result["start_date"] = {
                "value": None,
                "resolution_type": "incomplete",
                "evidence": partial_date.evidence,
                "needs_clarification": True,
            }

        return result

    def _derive_preference_lists(
            self,
            preference_signals: list[PreferenceSignal],
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """
        从 PreferenceSignal 派生旧的轻量偏好字段。

        注意：
        - preference_signals 是新的权威偏好结构。
        - transport_preferences / hotel_preferences / travel_style 只是早期节点的轻量索引。
        """

        transport_preferences: list[str] = []
        hotel_preferences: list[str] = []
        travel_style: list[str] = []
        raw_constraints: list[str] = []

        for signal in preference_signals:
            if signal.domain == "flight":
                transport_preferences.append(signal.key)

            elif signal.domain == "hotel":
                hotel_preferences.append(signal.key)

            elif signal.domain in {"activity", "pace"}:
                travel_style.append(signal.key)

            if signal.evidence:
                raw_constraints.append(signal.evidence)

        return (
            _dedupe(transport_preferences),
            _dedupe(hotel_preferences),
            _dedupe(travel_style),
            _dedupe(raw_constraints),
        )

    def _estimate_confidence(self, trip_request: TripRequest) -> float:
        """
        粗略估计规则抽取置信度。

        这个分数不是机器学习模型分数。
        它只是一个工程信号，用于未来判断是否需要 LLM fallback。

        例如：
        - origin/destination/start_date/days 都齐全，置信度较高。
        - 只抽到 destination，置信度较低。
        """

        score = 0.1

        if trip_request.origin:
            score += 0.18

        if trip_request.destination:
            score += 0.18

        if trip_request.start_date:
            score += 0.18

        if trip_request.days:
            score += 0.16

        if trip_request.budget is not None:
            score += 0.1

        if (
            trip_request.transport_preferences
            or trip_request.hotel_preferences
            or trip_request.travel_style
        ):
            score += 0.1

        return round(min(score, 1.0), 2)


def extract_trip_request_from_message(
    message: str,
    user_id: str | None = None,
    reference_date: date | datetime | str | None = None,
    extractor: TripRequestExtractor | None = None,
) -> TripRequest:
    """
    单独调用抽取器的便捷函数。

    这个函数主要用于：
    - 单元测试
    - notebook 调试
    - 临时脚本

    InputExtractNode 内部也可以直接使用 extractor.extract(...)。
    """

    extractor = extractor or RuleBasedTripRequestExtractor()
    parsed_reference_date = coerce_reference_date(reference_date)

    return extractor.extract(
        message=message,
        user_id=user_id,
        reference_date=parsed_reference_date,
    )


def coerce_reference_date(value: date | datetime | str | None) -> date:
    """
    把 reference_date 统一转成 date。

    支持：
    - date
    - datetime
    - "2026-07-01"
    - None

    如果字符串解析失败，则回退到 date.today()。
    """

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return date.today()

    return date.today()


def _compact_text(text: str) -> str:
    """
    去掉文本中的连续空白。

    例如：
        "杭州 到 成都" -> "杭州到成都"

    这样正则更容易匹配。
    """

    return re.sub(r"\s+", "", text.strip())


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    """
    判断 text 是否包含任意关键词。
    """

    return any(keyword.lower() in text for keyword in keywords)


def _dedupe(items: list[str]) -> list[str]:
    """
    保持顺序去重。

    list(set(items)) 会打乱顺序。
    dict.fromkeys(items) 可以保留第一次出现的顺序。
    """

    return list(dict.fromkeys(items))


def _parse_date_token(token: str, reference_date: date) -> date | None:
    """
    解析日期 token。

    支持：
    - 2026年7月2日
    - 2026-07-02
    - 2026/07/02
    - 7月2日

    如果没有年份，则使用 reference_date.year。
    """

    token = token.strip()

    normalized = (
        token.replace("年", "-")
        .replace("月", "-")
        .replace("日", "")
        .replace("号", "")
        .replace("/", "-")
    )

    full_match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", normalized)
    if full_match:
        year, month, day = map(int, full_match.groups())
        return date(year, month, day)

    month_day_match = re.fullmatch(r"(\d{1,2})-(\d{1,2})", normalized)
    if month_day_match:
        month, day = map(int, month_day_match.groups())
        return date(reference_date.year, month, day)

    return None


def _token_to_int(token: str) -> int | None:
    """
    把数字 token 转成 int。

    支持：
    - "3"
    - "三"
    - "两"
    - "十"
    """

    value = _number_token_to_float(token)
    if value is None:
        return None

    return int(value)


def _number_token_to_float(token: str) -> float | None:
    """
    把数字 token 转成 float。

    支持阿拉伯数字和简单中文数字。
    """

    token = token.strip()

    if not token:
        return None

    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return float(token)

    value = _chinese_to_int(token)
    if value is None:
        return None

    return float(value)


def _money_token_to_float(token: str, unit: str | None = None) -> float | None:
    """
    把金额表达转成元。

    示例：
    - token="4000", unit=None -> 4000
    - token="4", unit="k" -> 4000
    - token="五", unit="千" -> 5000
    - token="1", unit="万" -> 10000
    """

    value = _number_token_to_float(token)
    if value is None:
        return None

    if unit in {"k", "K", "千"}:
        value *= 1000

    if unit == "万":
        value *= 10000

    return float(value)


def _chinese_to_int(text: str) -> int | None:
    """
    简单中文数字转整数。

    支持常见表达：
    - 三
    - 十
    - 十五
    - 二十
    - 四千
    - 五千
    - 一万
    - 一万二千

    不追求覆盖所有中文数字。
    复杂金额表达后续可以交给 LLM structured output fallback。
    """

    if not text:
        return None

    total = 0
    section = 0
    number = 0
    seen = False

    for char in text:
        if char in _CHINESE_DIGITS:
            number = _CHINESE_DIGITS[char]
            seen = True
            continue

        if char in _CHINESE_UNITS:
            unit = _CHINESE_UNITS[char]
            seen = True

            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0
                number = 0
            else:
                if number == 0:
                    number = 1
                section += number * unit
                number = 0
            continue

        return None

    if not seen:
        return None

    return total + section + number