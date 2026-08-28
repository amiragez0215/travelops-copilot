from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.common.config import settings
from app.mcp.connection_manager import MCPConnectionManager
from app.mcp.server_registry import build_default_server_registry
from app.mcp.tool_registry import ToolDescriptor, ToolRegistry


@dataclass(frozen=True)
class MCPToolRuntime:
    connection_manager: MCPConnectionManager
    tool_registry: ToolRegistry


_runtime: MCPToolRuntime | None = None


def initialize_mcp_tool_runtime(*, discover: bool | None = None) -> MCPToolRuntime:
    global _runtime
    servers = build_default_server_registry()
    manager = MCPConnectionManager(
        servers=servers,
        timeout_seconds=settings.mcp_travel_data_timeout_seconds,
    )
    use_discovery = settings.travel_data_client_mode == "mcp" if discover is None else discover
    tools: list[ToolDescriptor] = []
    if use_discovery:
        for server_id in servers:
            tools.extend(manager.discover_server(server_id))
    else:
        tools = _local_descriptors()
    registry = ToolRegistry(tools)
    required = {"get_weather", "search_flights", "search_hotels"}
    missing = required - set(registry.names())
    if missing:
        raise RuntimeError(f"MCP Discovery 缺少必选工具：{sorted(missing)}")
    _runtime = MCPToolRuntime(manager, registry)
    return _runtime


def get_mcp_tool_runtime() -> MCPToolRuntime:
    return _runtime or initialize_mcp_tool_runtime()


def reset_mcp_tool_runtime() -> None:
    global _runtime
    _runtime = None


def _local_descriptors() -> list[ToolDescriptor]:
    schemas: dict[str, tuple[str, dict[str, Any]]] = {
        "get_weather": ("weather", _schema({"city": "string", "start_date": "string", "end_date": "string"}, ["city", "start_date"])),
        "search_flights": ("flight", _schema({"origin": "string", "destination": "string", "depart_date": "string", "return_date": "string", "people_count": "integer"}, ["origin", "destination", "depart_date"])),
        "search_hotels": ("hotel", _schema({"city": "string", "nights": "integer", "people_count": "integer"}, ["city", "nights"])),
        "search_city_activities": ("activity", {
            "type": "object",
            "properties": {
                "city": {"type": "string"}, "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "interests": {"type": "array", "items": {"type": "string"}},
                "indoor_preference": {"type": ["boolean", "null"]},
                "max_results": {"type": "integer"},
            },
            "required": ["city", "start_date", "end_date"], "additionalProperties": False,
        }),
    }
    descriptions = {
        "get_weather": "查询旅行日期内的天气事实。",
        "search_flights": "查询往返航班候选。",
        "search_hotels": "查询酒店候选。",
        "search_city_activities": "用户明确需要展览、博物馆、文化、夜间、亲子、低步行或室内活动时查询活动候选。",
    }
    return [ToolDescriptor(name, server, descriptions[name], schema) for name, (server, schema) in schemas.items()]


def _schema(properties: dict[str, str], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": {k: {"type": v} for k, v in properties.items()}, "required": required, "additionalProperties": False}
