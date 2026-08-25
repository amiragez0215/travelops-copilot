from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from app.providers.mock_data_loader import load_mock_json


def search_activities_from_mock(
    *,
    city: str,
    start_date: str,
    end_date: str,
    interests: list[str] | None = None,
    indoor_preference: bool | None = None,
    max_results: int = 12,
    mock_file: str | Path | None = None,
) -> dict[str, Any]:
    """
    查询日期范围内可用的结构化活动。

    这里故意只做确定性过滤和轻量匹配，不承担 CandidateRank 的职责：
    航班、酒店仍由原 Rank 节点处理；活动候选只作为 Proposal 的受控事实。
    """

    normalized_city = str(city).strip()
    if not normalized_city:
        raise ValueError("city 不能为空")

    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if end < start:
        raise ValueError("end_date 不能早于 start_date")
    if max_results < 1:
        raise ValueError("max_results 必须大于 0")

    requested = [str(item).strip() for item in (interests or []) if str(item).strip()]
    data = load_mock_json("activities_mock.json", mock_file=mock_file)
    candidates: list[tuple[int, dict[str, Any]]] = []

    for raw in data["items"]:
        if str(raw.get("city") or "") != normalized_city:
            continue
        available_dates = [str(value) for value in raw.get("available_dates", [])]
        if not any(start <= _parse_date(value, "available_dates") <= end for value in available_dates):
            continue
        if indoor_preference is True and raw.get("indoor_outdoor") == "outdoor":
            continue

        searchable = " ".join(
            [
                str(raw.get("name") or ""),
                str(raw.get("category") or ""),
                *[str(tag) for tag in raw.get("tags", [])],
            ]
        ).lower()
        match_count = sum(term.lower() in searchable for term in requested)
        normalized = _normalize_activity(raw)
        normalized["match_reasons"] = [term for term in requested if term.lower() in searchable]
        candidates.append((match_count, normalized))

    # 有明确兴趣时优先返回匹配项，但不会把所有零匹配项都删除；少量通用
    # 备选可以帮助 Proposal 在天气变化或时段冲突时仍有安全回退。
    candidates.sort(key=lambda pair: (-pair[0], str(pair[1].get("activity_id") or "")))
    items = [item for _, item in candidates[:max_results]]

    return {
        "status": "ok" if items else "no_data",
        "city": normalized_city,
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "items": items,
        "source": data.get("provider", "mock_activity_provider"),
    }


def _normalize_activity(item: dict[str, Any]) -> dict[str, Any]:
    """只暴露 Proposal 和 Verifier 真正需要的稳定字段。"""

    return {
        "activity_id": item.get("activity_id"),
        "name": item.get("name"),
        "city": item.get("city"),
        "district": item.get("district"),
        "category": item.get("category"),
        "available_dates": list(item.get("available_dates", [])),
        "opening_hours": item.get("opening_hours"),
        "estimated_duration_minutes": item.get("estimated_duration_minutes"),
        "estimated_price": item.get("estimated_price"),
        "indoor_outdoor": item.get("indoor_outdoor", "flexible"),
        "reservation_required": bool(item.get("reservation_required", False)),
        "tags": list(item.get("tags", [])),
        "address": item.get("address"),
    }


def _parse_date(value: Any, field_name: str) -> date:
    if not value:
        raise ValueError(f"{field_name} 不能为空")
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ValueError(f"{field_name} 日期格式无效: {value}") from exc
