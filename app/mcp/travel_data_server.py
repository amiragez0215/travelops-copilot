from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from app.mcp.travel_data_tools import (
    get_weather_tool,
    search_city_activities_tool,
    search_flights_tool,
    search_hotels_tool,
)


mcp = FastMCP("travel-data-mcp")


@mcp.tool()
def get_weather(
    city: str,
    start_date: str,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    查询 mock 天气数据。

    MCP tool 输入：
    - city
    - start_date
    - end_date

    MCP tool 输出：
    - weather_result dict
    """

    return get_weather_tool(
        city=city,
        start_date=start_date,
        end_date=end_date,
    )


@mcp.tool()
def search_flights(
    origin: str,
    destination: str,
    depart_date: str,
    return_date: str | None = None,
    people_count: int = 1,
) -> dict[str, Any]:
    """
    查询 mock 航班数据。

    MCP tool 只过滤数据层硬错误：
    - 城市
    - 日期
    - 座位
    - 价格合法性
    """

    return search_flights_tool(
        origin=origin,
        destination=destination,
        depart_date=depart_date,
        return_date=return_date,
        people_count=people_count,
    )


@mcp.tool()
def search_hotels(
    city: str,
    nights: int,
    people_count: int = 1,
) -> dict[str, Any]:
    """
    查询 mock 酒店数据。

    MCP tool 只过滤数据层硬错误：
    - 城市
    - 是否有房
    - 价格合法性
    """

    return search_hotels_tool(
        city=city,
        nights=nights,
        people_count=people_count,
    )


@mcp.tool()
def search_city_activities(
    city: str,
    start_date: str,
    end_date: str,
    interests: list[str] | None = None,
    indoor_preference: bool | None = None,
    max_results: int = 12,
) -> dict[str, Any]:
    """查询指定城市和旅行日期内可安排的结构化活动候选。"""

    return search_city_activities_tool(
        city=city,
        start_date=start_date,
        end_date=end_date,
        interests=interests,
        indoor_preference=indoor_preference,
        max_results=max_results,
    )


if __name__ == "__main__":
    # 使用 stdio transport 运行 MCP server。
    mcp.run()
