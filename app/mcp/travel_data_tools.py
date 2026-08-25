from __future__ import annotations

from pathlib import Path
from typing import Any

from app.providers.flight_mock_provider import search_flights_from_mock
from app.providers.hotel_mock_provider import search_hotels_from_mock
from app.providers.activity_mock_provider import search_activities_from_mock
from app.providers.weather_mock_provider import query_weather_from_mock


def get_weather_tool(
    city: str,
    start_date: str,
    end_date: str | None = None,
    mock_file: str | Path | None = None,
) -> dict[str, Any]:
    """
    MCP 工具函数：查询天气。

    这个函数本身是普通 Python 函数，方便单元测试。
    travel_data_server.py 会把它注册成真正的 MCP tool。
    """

    return query_weather_from_mock(
        city=city,
        start_date=start_date,
        end_date=end_date,
        mock_file=mock_file,
    )


def search_flights_tool(
    origin: str,
    destination: str,
    depart_date: str,
    return_date: str | None = None,
    people_count: int = 1,
    mock_file: str | Path | None = None,
) -> dict[str, Any]:
    """
    MCP 工具函数：查询航班。

    只做数据层硬错误过滤，不做用户偏好排序。
    """

    return search_flights_from_mock(
        origin=origin,
        destination=destination,
        depart_date=depart_date,
        return_date=return_date,
        people_count=people_count,
        mock_file=mock_file,
    )


def search_hotels_tool(
    city: str,
    nights: int,
    people_count: int = 1,
    mock_file: str | Path | None = None,
) -> dict[str, Any]:
    """
    MCP 工具函数：查询酒店。

    只做数据层硬错误过滤，不做用户偏好排序。
    """

    return search_hotels_from_mock(
        city=city,
        nights=nights,
        people_count=people_count,
        mock_file=mock_file,
    )


def search_city_activities_tool(
    city: str,
    start_date: str,
    end_date: str,
    interests: list[str] | None = None,
    indoor_preference: bool | None = None,
    max_results: int = 12,
    mock_file: str | Path | None = None,
) -> dict[str, Any]:
    """
    MCP 工具函数：按城市、日期和用户兴趣查询活动。

    与航班/酒店不同，活动不进入 CandidateRank 和 BudgetOptimize；它提供
    带稳定 activity_id 的结构化事实，供 Proposal 安排并由 Verifier 核验。
    """

    return search_activities_from_mock(
        city=city,
        start_date=start_date,
        end_date=end_date,
        interests=interests,
        indoor_preference=indoor_preference,
        max_results=max_results,
        mock_file=mock_file,
    )
