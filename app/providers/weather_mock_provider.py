from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from app.providers.mock_data_loader import load_mock_json


def query_weather_from_mock(
    city: str,
    start_date: str,
    end_date: str | None = None,
    mock_file: str | Path | None = None,
) -> dict[str, Any]:
    """
    从 weather_mock.json 中查询天气。

    Args:
        city:
            目的地城市，例如 成都。

        start_date:
            查询开始日期，格式 YYYY-MM-DD。

        end_date:
            查询结束日期，格式 YYYY-MM-DD。
            如果为 None，则只查 start_date 当天。

        mock_file:
            测试时可传入临时 mock 文件。

    Returns:
        dict:
            weather_result，可直接写入 state["weather_result"]。

    数据层硬错误处理：
        天气没有“过滤候选”的概念。
        如果查不到天气，返回 status=no_data，不直接抛异常。
    """

    if not city or not city.strip():
        raise ValueError("city 不能为空")

    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date or start_date, "end_date")

    if end < start:
        raise ValueError("end_date 不能早于 start_date")

    data = load_mock_json("weather_mock.json", mock_file=mock_file)
    items = data["items"]

    matched = [
        _normalize_weather_item(item)
        for item in items
        if _is_weather_match(item=item, city=city, start=start, end=end)
    ]

    risks = _build_weather_risks(matched)
    warnings = _collect_warnings(matched)

    if not matched:
        return {
            "status": "no_data",
            "city": city,
            "date_range": {
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            "summary": f"未找到 {city} 在 {start.isoformat()} 至 {end.isoformat()} 的 mock 天气数据。",
            "daily": [],
            "risks": [],
            "warnings": ["天气数据缺失，后续方案需要提示用户。"],
            "source": data.get("provider", "mock_weather_provider"),
        }

    return {
        "status": "ok",
        "city": city,
        "date_range": {
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
        "summary": _build_summary(city=city, daily=matched, risks=risks),
        "daily": matched,
        "risks": risks,
        "warnings": warnings,
        "source": data.get("provider", "mock_weather_provider"),
    }


def _is_weather_match(
    item: dict[str, Any],
    city: str,
    start: date,
    end: date,
) -> bool:
    """
    判断一条天气记录是否匹配城市和日期范围。
    """

    if item.get("city") != city:
        return False

    item_date = _parse_date(item.get("date"), "item.date")

    return start <= item_date <= end


def _normalize_weather_item(item: dict[str, Any]) -> dict[str, Any]:
    """
    标准化天气记录，保证后续节点读取字段稳定。
    """

    return {
        "weather_id": item.get("weather_id"),
        "city": item.get("city"),
        "date": item.get("date"),
        "condition": item.get("condition"),
        "temperature_low": item.get("temperature_low"),
        "temperature_high": item.get("temperature_high"),
        "humidity": item.get("humidity"),
        "wind": item.get("wind"),
        "precipitation_probability": item.get("precipitation_probability"),
        "risk_tags": item.get("risk_tags", []),
        "warnings": item.get("warnings", []),
        "updated_at": item.get("updated_at"),
    }


def _build_weather_risks(
    daily: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    根据 risk_tags / condition / wind 生成结构化天气风险。
    """

    risks: list[dict[str, Any]] = []

    for item in daily:
        risk_tags = item.get("risk_tags", [])
        condition = str(item.get("condition") or "")
        wind = str(item.get("wind") or "")

        if "rain" in risk_tags or "雨" in condition:
            risks.append(
                {
                    "risk_type": "rain",
                    "level": "medium",
                    "date": item.get("date"),
                    "message": "可能有降雨，建议准备雨具并减少长时间户外排队。",
                }
            )

        if "heavy_rain" in risk_tags or "暴雨" in condition:
            risks.append(
                {
                    "risk_type": "heavy_rain",
                    "level": "high",
                    "date": item.get("date"),
                    "message": "可能有强降雨，建议优先安排室内活动并预留交通缓冲。",
                }
            )

        if "wind" in risk_tags or "大风" in wind or "强风" in wind:
            risks.append(
                {
                    "risk_type": "wind",
                    "level": "medium",
                    "date": item.get("date"),
                    "message": "可能有大风天气，建议减少长时间户外步行。",
                }
            )

        if "heat" in risk_tags or "高温" in condition:
            risks.append(
                {
                    "risk_type": "heat",
                    "level": "medium",
                    "date": item.get("date"),
                    "message": "可能有高温天气，建议减少正午户外活动。",
                }
            )

    return risks


def _collect_warnings(
    daily: list[dict[str, Any]],
) -> list[str]:
    """
    收集天气记录中的 warnings，并保持顺序去重。
    """

    warnings: list[str] = []

    for item in daily:
        for warning in item.get("warnings", []):
            if isinstance(warning, str) and warning.strip():
                warnings.append(warning.strip())

    return list(dict.fromkeys(warnings))


def _build_summary(
    city: str,
    daily: list[dict[str, Any]],
    risks: list[dict[str, Any]],
) -> str:
    """
    构造天气摘要。
    """

    dates = [item["date"] for item in daily if item.get("date")]
    conditions = [str(item.get("condition")) for item in daily if item.get("condition")]

    risk_types = list(dict.fromkeys(risk["risk_type"] for risk in risks))

    if risk_types:
        return f"{city} {dates[0]} 至 {dates[-1]} 天气包含 {', '.join(conditions)}，风险：{', '.join(risk_types)}。"

    return f"{city} {dates[0]} 至 {dates[-1]} 天气包含 {', '.join(conditions)}，暂无明显天气风险。"


def _parse_date(
    value: Any,
    field_name: str,
) -> date:
    """
    解析 YYYY-MM-DD 日期。
    """

    if not value:
        raise ValueError(f"{field_name} 不能为空")

    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ValueError(f"{field_name} 日期格式无效: {value}") from exc