from typing import Any
from mcp.server.fastmcp import FastMCP
from app.mcp.travel_data_tools import search_city_activities_tool

mcp = FastMCP("travel-activity-mcp")

@mcp.tool()
def search_city_activities(city: str, start_date: str, end_date: str, interests: list[str] | None = None, indoor_preference: bool | None = None, max_results: int = 12) -> dict[str, Any]:
    """查询旅行日期内符合兴趣偏好的结构化城市活动候选。"""
    return search_city_activities_tool(city=city, start_date=start_date, end_date=end_date, interests=interests, indoor_preference=indoor_preference, max_results=max_results)

if __name__ == "__main__":
    mcp.run()
