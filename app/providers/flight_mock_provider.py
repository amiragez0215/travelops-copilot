from __future__ import annotations

from pathlib import Path
from typing import Any

from app.providers.mock_data_loader import load_mock_json


def search_flights_from_mock(
    origin: str,
    destination: str,
    depart_date: str,
    return_date: str | None = None,
    people_count: int = 1,
    mock_file: str | Path | None = None,
) -> dict[str, Any]:
    """
    从 flights_mock.json 中查询航班。

    Args:
        origin:
            出发城市。

        destination:
            目的地城市。

        depart_date:
            出发日期，格式 YYYY-MM-DD。

        return_date:
            返程日期，格式 YYYY-MM-DD。
            如果为 None，只返回 outbound。

        people_count:
            出行人数。MCP Provider 层会过滤 available_seats < people_count 的航班。

        mock_file:
            测试用临时 mock 文件。

    Returns:
        dict:
            raw_flight_results，可写入 state["raw_flight_results"]。

    MCP / Provider 层只处理数据层硬错误：
        - 城市不匹配
        - 日期不匹配
        - 座位不足
        - 价格非法

    不处理用户偏好：
        - 不过滤早班机
        - 不过滤太晚到达
        - 不按预算软目标过滤
        - 不做最终排序
    """

    _require_text(origin, "origin")
    _require_text(destination, "destination")
    _require_text(depart_date, "depart_date")

    people_count = max(int(people_count or 1), 1)

    data = load_mock_json("flights_mock.json", mock_file=mock_file)
    items = data["items"]

    outbound = _query_one_way(
        items=items,
        departure_city=origin,
        arrival_city=destination,
        depart_date=depart_date,
        people_count=people_count,
        direction="outbound",
    )

    return_items: list[dict[str, Any]] = []
    if return_date:
        return_items = _query_one_way(
            items=items,
            departure_city=destination,
            arrival_city=origin,
            depart_date=return_date,
            people_count=people_count,
            direction="return",
        )

    return {
        "status": "ok" if outbound and (return_items or not return_date) else "no_candidate",
        "origin": origin,
        "destination": destination,
        "depart_date": depart_date,
        "return_date": return_date,
        "people_count": people_count,
        "outbound": outbound,
        "return": return_items,
        "source": data.get("provider", "mock_flight_provider"),
        "filter_summary": {
            "outbound_count": len(outbound),
            "return_count": len(return_items),
            "basic_filters": [
                "departure_city",
                "arrival_city",
                "depart_date",
                "available_seats >= people_count",
                "price >= 0",
            ],
        },
    }


def _query_one_way(
    items: list[dict[str, Any]],
    departure_city: str,
    arrival_city: str,
    depart_date: str,
    people_count: int,
    direction: str,
) -> list[dict[str, Any]]:
    """
    查询单程航班。
    """

    matched: list[dict[str, Any]] = []

    for item in items:
        if item.get("departure_city") != departure_city:
            continue

        if item.get("arrival_city") != arrival_city:
            continue

        if item.get("depart_date") != depart_date:
            continue

        if not _is_available_flight(item=item, people_count=people_count):
            continue

        normalized = _normalize_flight_item(item)
        normalized["query_direction"] = direction
        matched.append(normalized)

    # Provider 层只做稳定排序，不做偏好排序。
    matched.sort(key=lambda flight: (flight["depart_time"], flight["price"]))

    return matched


def _is_available_flight(
    item: dict[str, Any],
    people_count: int,
) -> bool:
    """
    检查航班是否为基础可用候选。
    """

    seats = _safe_int(item.get("available_seats"), default=0)
    price = _safe_float(item.get("price"), default=-1)

    if seats < people_count:
        return False

    if price < 0:
        return False

    if not item.get("flight_id"):
        return False

    return True


def _normalize_flight_item(
    item: dict[str, Any],
) -> dict[str, Any]:
    """
    标准化航班字段。
    """

    return {
        "flight_id": item.get("flight_id"),
        "flight_no": item.get("flight_no"),
        "airline": item.get("airline"),
        "departure_city": item.get("departure_city"),
        "arrival_city": item.get("arrival_city"),
        "departure_airport": item.get("departure_airport"),
        "arrival_airport": item.get("arrival_airport"),
        "depart_date": item.get("depart_date"),
        "depart_time": item.get("depart_time"),
        "arrive_date": item.get("arrive_date"),
        "arrive_time": item.get("arrive_time"),
        "duration_minutes": _safe_int(item.get("duration_minutes"), default=0),
        "price": _safe_float(item.get("price"), default=0),
        "available_seats": _safe_int(item.get("available_seats"), default=0),
        "cabin": item.get("cabin"),
        "baggage": item.get("baggage"),
        "is_direct": bool(item.get("is_direct", True)),
        "refundable": bool(item.get("refundable", False)),
        "changeable": bool(item.get("changeable", False)),
        "updated_at": item.get("updated_at"),
        "source": "mock_flight_provider",
    }


def _require_text(
    value: str,
    field_name: str,
) -> None:
    """
    检查必填字符串。
    """

    if not value or not str(value).strip():
        raise ValueError(f"{field_name} 不能为空")


def _safe_int(
    value: Any,
    default: int,
) -> int:
    """
    安全转换 int。
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(
    value: Any,
    default: float,
) -> float:
    """
    安全转换 float。
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return default