from typing import Any
from mcp.server.fastmcp import FastMCP
from app.mcp.travel_data_tools import search_flights_tool

mcp = FastMCP("travel-flight-mcp")

@mcp.tool()
def search_flights(origin: str, destination: str, depart_date: str, return_date: str | None = None, people_count: int = 1) -> dict[str, Any]:
    """查询指定路线和日期的结构化往返航班候选。"""
    return search_flights_tool(origin=origin, destination=destination, depart_date=depart_date, return_date=return_date, people_count=people_count)

if __name__ == "__main__":
    mcp.run()
