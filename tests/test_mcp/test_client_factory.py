from __future__ import annotations

from app.mcp.client_factory import (
    create_travel_data_client,
    describe_travel_data_client,
)
from app.mcp.travel_data_client import (
    LocalTravelDataClient,
    MCPTravelDataClient,
)


def test_factory_can_create_local_client():
    """单元测试模式可以显式创建不启动子进程的 Local Client。"""

    client = create_travel_data_client(
        mode="local"
    )

    assert isinstance(
        client,
        LocalTravelDataClient,
    )

    assert describe_travel_data_client(
        client
    ) == {
        "client_mode": "local",
        "protocol": "local",
        "transport": "in_process",
    }


def test_factory_can_create_mcp_stdio_client():
    """正式模式应创建真实 MCP stdio Client，而不是 Local Client。"""

    client = create_travel_data_client(
        mode="mcp"
    )

    assert isinstance(
        client,
        MCPTravelDataClient,
    )

    assert client.args == (
        "-m",
        "app.mcp.travel_data_server",
    )

    assert describe_travel_data_client(
        client
    ) == {
        "client_mode": "mcp",
        "protocol": "mcp",
        "transport": "stdio",
    }
