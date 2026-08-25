from __future__ import annotations

from pathlib import Path
from typing import Any

from app.providers.mock_data_loader import load_mock_json


def search_hotels_from_mock(
    city: str,
    nights: int,
    people_count: int = 1,
    mock_file: str | Path | None = None,
) -> dict[str, Any]:
    """
    从 hotels_mock.json 中查询酒店。

    Args:
        city:
            目的地城市。

        nights:
            住宿晚数。用于计算 estimated_total_price。

        people_count:
            出行人数。当前 mock 酒店按房间数过滤，不处理复杂房型。

        mock_file:
            测试用临时 mock 文件。

    Returns:
        dict:
            raw_hotel_results，可写入 state["raw_hotel_results"]。

    Provider 层只处理数据层硬错误：
        - 城市不匹配
        - 无房
        - 价格非法
        - hotel_id 缺失

    不处理用户偏好：
        - 不按 quiet_score 过滤
        - 不按 cleanliness_score 过滤
        - 不按 soft target 价格过滤
        - 不做最终排序
    """

    _require_text(city, "city")

    nights = max(int(nights or 0), 0)
    people_count = max(int(people_count or 1), 1)

    data = load_mock_json("hotels_mock.json", mock_file=mock_file)
    items = data["items"]

    hotels = [
        _normalize_hotel_item(item=item, nights=nights)
        for item in items
        if _is_hotel_match(item=item, city=city, people_count=people_count)
    ]

    # Provider 层只做稳定排序，不做偏好排序。
    hotels.sort(key=lambda hotel: (hotel["price_per_night"], -hotel["rating"]))

    return {
        "status": "ok" if hotels else "no_candidate",
        "city": city,
        "nights": nights,
        "people_count": people_count,
        "items": hotels,
        "source": data.get("provider", "mock_hotel_provider"),
        "filter_summary": {
            "hotel_count": len(hotels),
            "basic_filters": [
                "city",
                "available_rooms >= 1",
                "price_per_night >= 0",
                "hotel_id exists",
            ],
        },
    }


def _is_hotel_match(
    item: dict[str, Any],
    city: str,
    people_count: int,
) -> bool:
    """
    判断酒店是否满足基础数据硬条件。
    """

    if item.get("city") != city:
        return False

    if not item.get("hotel_id"):
        return False

    available_rooms = _safe_int(item.get("available_rooms"), default=0)
    if available_rooms <= 0:
        return False

    price = _safe_float(item.get("price_per_night"), default=-1)
    if price < 0:
        return False

    return True


def _normalize_hotel_item(
    item: dict[str, Any],
    nights: int,
) -> dict[str, Any]:
    """
    标准化酒店字段。
    """

    price_per_night = _safe_float(item.get("price_per_night"), default=0)

    return {
        "hotel_id": item.get("hotel_id"),
        "name": item.get("name"),
        "city": item.get("city"),
        "district": item.get("district"),
        "address": item.get("address"),
        "price_per_night": price_per_night,
        "estimated_total_price": round(price_per_night * nights, 2),
        "rating": _safe_float(item.get("rating"), default=0),
        "available_rooms": _safe_int(item.get("available_rooms"), default=0),
        "near_subway": bool(item.get("near_subway", False)),
        "distance_to_subway_meters": _safe_int(
            item.get("distance_to_subway_meters"),
            default=999999,
        ),
        "quiet_score": _safe_float(item.get("quiet_score"), default=0.5),
        "cleanliness_score": _safe_float(
            item.get("cleanliness_score"),
            default=_infer_cleanliness_score(item),
        ),
        "tags": item.get("tags", []),
        "amenities": item.get("amenities", []),
        "cancel_policy": item.get("cancel_policy"),
        "updated_at": item.get("updated_at"),
        "source": "mock_hotel_provider",
    }


def _infer_cleanliness_score(
    item: dict[str, Any],
) -> float:
    """
    如果 mock 数据暂时没有 cleanliness_score，用 rating 做一个保守推断。

    后续建议在 hotels_mock.json 中显式补 cleanliness_score。
    """

    rating = _safe_float(item.get("rating"), default=4.0)

    # rating 约 4.0-5.0，转换到 0.6-0.95 左右。
    return max(0.5, min(0.95, (rating - 3.0) / 2.0))


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