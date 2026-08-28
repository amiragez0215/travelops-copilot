from typing import Any
from mcp.server.fastmcp import FastMCP
from app.mcp.travel_data_tools import get_weather_tool

mcp = FastMCP("travel-weather-mcp")

@mcp.tool()
def get_weather(city: str, start_date: str, end_date: str | None = None) -> dict[str, Any]:
    """查询指定城市和日期范围的结构化天气。"""
    return get_weather_tool(city=city, start_date=start_date, end_date=end_date)

if __name__ == "__main__":
    mcp.run()
