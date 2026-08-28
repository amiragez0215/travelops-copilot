from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class MCPServerConfig:
    server_id: str
    transport: str
    command: str
    args: tuple[str, ...]


def build_default_server_registry() -> dict[str, MCPServerConfig]:
    modules = {
        "weather": "app.mcp.servers.weather_server",
        "flight": "app.mcp.servers.flight_server",
        "hotel": "app.mcp.servers.hotel_server",
        "activity": "app.mcp.servers.activity_server",
    }
    return {
        server_id: MCPServerConfig(
            server_id=server_id,
            transport="stdio",
            command=sys.executable,
            args=("-m", module),
        )
        for server_id, module in modules.items()
    }
