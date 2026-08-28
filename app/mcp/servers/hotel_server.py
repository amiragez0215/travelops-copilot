from typing import Any
from mcp.server.fastmcp import FastMCP
from app.mcp.travel_data_tools import search_hotels_tool

mcp = FastMCP("travel-hotel-mcp")

@mcp.tool()
def search_hotels(city: str, nights: int, people_count: int = 1) -> dict[str, Any]:
    """查询指定城市的结构化酒店候选。"""
    return search_hotels_tool(city=city, nights=nights, people_count=people_count)

if __name__ == "__main__":
    mcp.run()
