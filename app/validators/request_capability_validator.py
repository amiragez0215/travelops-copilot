from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.providers.mock_data_loader import (
    DEFAULT_MOCK_DIR,
    load_mock_json,
)


class RequestCapabilityValidator:
    """
    检查结构化旅行需求是否位于当前 Mock 数据能力范围内。

    它不判断用户输入是否缺字段；只回答：
        “系统当前是否有能力为这个目的地、路线和指定酒店提供数据？”

    当前检查：
        - 目的地是否出现在天气或酒店 Mock 中。
        - 出发地 → 目的地是否有航班路线。
        - required/preferred 指定酒店是否存在于酒店 Mock。

    当前不检查：
        - 指定食物、景点是否有 RAG 证据。
          这部分会在 RetrievalPlan / EvidenceGrade 阶段检查。
        - 某个具体日期是否有 Mock 航班。
          日期候选不足由 FlightSearchNode 返回 no_candidate 后再澄清。
    """

    def __init__(
        self,
        mock_dir: str | Path | None = None,
    ) -> None:
        self.mock_dir = Path(mock_dir) if mock_dir else DEFAULT_MOCK_DIR

    def validate(
        self,
        trip_request: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        返回 capability_result。

        如果 Mock 文件尚未生成或测试环境没有 data/mock，
        返回 not_checked，不阻断普通 Schema 单元测试。
        """

        origin = _clean_text(trip_request.get("origin"))
        destination = _clean_text(trip_request.get("destination"))

        # 1. 核心城市还没补齐时，能力检查没有意义。
        if not origin or not destination:
            return {
                "status": "not_checked",
                "supported": True,
                "reason": "origin 或 destination 尚未完整。",
                "issues": [],
                "supported_destinations": [],
            }

        try:
            hotel_data = self._load("hotels_mock.json")
            weather_data = self._load("weather_mock.json")
            flight_data = self._load("flights_mock.json")
        except FileNotFoundError as exc:
            # 2. 测试代码快照通常不包含 data 目录。
            #    此时记录未检查，而不是把所有 Validator 测试判失败。
            return {
                "status": "not_checked",
                "supported": True,
                "reason": str(exc),
                "issues": [],
                "supported_destinations": [],
            }

        hotel_items = hotel_data.get("items", [])
        weather_items = weather_data.get("items", [])
        flight_items = flight_data.get("items", [])

        supported_hotel_cities = {
            str(item.get("city"))
            for item in hotel_items
            if isinstance(item, Mapping)
            and item.get("city")
            and item.get("hotel_id")
            and _safe_float(item.get("price_per_night"), -1) >= 0
        }

        supported_weather_cities = {
            str(item.get("city"))
            for item in weather_items
            if isinstance(item, Mapping) and item.get("city")
        }

        supported_destinations = sorted(
            supported_hotel_cities & supported_weather_cities
        )

        supported_routes = {
            (
                str(item.get("departure_city")),
                str(item.get("arrival_city")),
            )
            for item in flight_items
            if isinstance(item, Mapping)
            and item.get("departure_city")
            and item.get("arrival_city")
            and item.get("flight_id")
            and _safe_float(item.get("price"), -1) >= 0
        }

        issues: list[dict[str, Any]] = []

        # 3. 目的地至少需要酒店和天气两个结构化数据源支持。
        if destination not in supported_destinations:
            issues.append(
                {
                    "issue_type": "unsupported_destination",
                    "message": (
                        f"已识别目的地为{destination}，但当前 Mock 数据没有同时覆盖"
                        "该城市的天气和酒店。"
                    ),
                    "requires_clarification": True,
                    "suggested_question": (
                        f"当前模拟数据支持的目的地为：{_join_values(supported_destinations)}。"
                        "你是否愿意修改目的地？"
                    ),
                    "details": {
                        "destination": destination,
                        "supported_destinations": supported_destinations,
                    },
                }
            )

        # 4. 航班路线只检查城市组合，不在这里检查具体日期和余票。
        if (origin, destination) not in supported_routes:
            issues.append(
                {
                    "issue_type": "unsupported_flight_route",
                    "message": f"当前 Mock 航班没有覆盖{origin}到{destination}的路线。",
                    "requires_clarification": True,
                    "suggested_question": (
                        "请修改出发地或目的地，或者使用当前 Mock 已覆盖的路线。"
                    ),
                    "details": {
                        "origin": origin,
                        "destination": destination,
                    },
                }
            )

        # 5. 指定酒店必须存在于目标城市的酒店 Mock 中。
        destination_hotels = [
            item
            for item in hotel_items
            if isinstance(item, Mapping)
            and item.get("city") == destination
            and item.get("hotel_id")
            and _safe_float(item.get("price_per_night"), -1) >= 0
        ]

        for constraint in _iter_named_constraints(trip_request):
            if constraint.get("entity_type") != "hotel":
                continue

            hotel_name = _clean_text(constraint.get("entity_name"))

            if not hotel_name:
                continue

            matches = [
                item
                for item in destination_hotels
                if _normalize_name(item.get("name")) == _normalize_name(hotel_name)
            ]

            if matches:
                continue

            issues.append(
                {
                    "issue_type": "named_hotel_not_found",
                    "message": (
                        f"当前 Mock 酒店数据中没有找到用户指定的酒店：{hotel_name}。"
                    ),
                    "requires_clarification": True,
                    "suggested_question": (
                        "是否接受当前数据源中的其他酒店候选？"
                    ),
                    "details": {
                        "hotel_name": hotel_name,
                        "available_hotel_names": [
                            str(item.get("name"))
                            for item in destination_hotels
                            if item.get("name")
                        ],
                    },
                }
            )

        return {
            "status": "supported" if not issues else "unsupported",
            "supported": not issues,
            "reason": "能力范围检查完成。",
            "issues": issues,
            "supported_destinations": supported_destinations,
        }

    def _load(self, filename: str) -> dict[str, Any]:
        """从配置的 mock_dir 读取统一格式 JSON。"""

        return load_mock_json(
            filename,
            mock_file=self.mock_dir / filename,
        )


def _iter_named_constraints(
    trip_request: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """读取合法 named_constraints。"""

    value = trip_request.get("named_constraints")

    if not isinstance(value, list):
        return []

    return [
        dict(item)
        for item in value
        if isinstance(item, Mapping)
    ]


def _normalize_name(value: Any) -> str:
    """酒店名称比较时去空格并忽略大小写。"""

    return "".join(str(value or "").split()).casefold()


def _clean_text(value: Any) -> str | None:
    """空值转 None，其他值转去空白字符串。"""

    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _safe_float(value: Any, default: float) -> float:
    """安全转换 float。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _join_values(values: list[str]) -> str:
    """把可选城市列表转成用户可读文本。"""

    return "、".join(values) if values else "暂无可用目的地"
