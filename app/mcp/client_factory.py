from __future__ import annotations

import sys
from typing import Any, Literal

from app.common.config import settings
from app.mcp.travel_data_client import (
    LocalTravelDataClient,
    MCPTravelDataClient,
    TravelDataClient,
)


TravelDataClientMode = Literal["local", "mcp"]


def create_travel_data_client(
    *,
    mode: TravelDataClientMode | None = None,
) -> TravelDataClient:
    """
    根据配置创建天气、航班和酒店节点使用的 TravelDataClient。

    这个工厂函数是阶段二最关键的改动之一。

    原来的三个数据节点在没有显式传入 client 时，会直接创建：

        LocalTravelDataClient()

    因此正式 Graph 虽然存在 MCP Server 和 MCP Client 代码，
    但运行主链实际上绕过了 MCP 协议，直接调用了本地 Python 工具函数。

    现在改为：

        WeatherNode / FlightSearchNode / HotelSearchNode
            ↓
        create_travel_data_client()
            ↓
        根据 TRAVEL_DATA_CLIENT_MODE 选择：
            local -> LocalTravelDataClient
            mcp   -> MCPTravelDataClient

    Args:
        mode:
            可选的显式模式。

            - None：读取 settings.travel_data_client_mode。
            - "local"：直接调用本地 Tool，适合单元测试。
            - "mcp"：通过 stdio 启动 MCP Server 并调用 Tool，适合正式演示。

    Returns:
        满足 TravelDataClient Protocol 的 Client 实例。

    Raises:
        ValueError:
            mode 不是 local 或 mcp 时抛出。
    """

    # 1. 正式运行时通常不传 mode，统一读取 .env 中的配置。
    resolved_mode = (
        mode
        or settings.travel_data_client_mode
    )

    if resolved_mode == "local":
        # 2. Local Client 仍然保留。
        #    它对单元测试很重要，因为测试不应为每个用例启动 MCP 子进程。
        return LocalTravelDataClient()

    if resolved_mode == "mcp":
        # 3. 正式主链使用 MCP Client。
        #    command 使用当前 Python 解释器，确保进入当前运行环境中的依赖。
        #    args 通过 -m 启动项目中的 MCP Server 模块。
        return MCPTravelDataClient(
            command=sys.executable,
            args=(
                "-m",
                settings.mcp_travel_data_server_module,
            ),
            timeout_seconds=(
                settings.mcp_travel_data_timeout_seconds
            ),
        )

    raise ValueError(
        "TRAVEL_DATA_CLIENT_MODE 只支持 local 或 mcp，"
        f"当前值为：{resolved_mode!r}"
    )


def describe_travel_data_client(
    client: TravelDataClient,
) -> dict[str, Any]:
    """
    返回当前 Client 的公开运行信息，供 fetch_meta 和 Trace 使用。

    为什么需要这个函数？

        以前 Node 的 Trace 固定写成：

            mcp:get_weather

        即使实际使用的是 LocalTravelDataClient，日志仍然看起来像 MCP。
        这样无法证明正式主链是否真的经过了 MCP Client -> MCP Server。

    现在每次工具调用都会记录：

        client_mode
        protocol
        transport

    示例：

        {
            "client_mode": "mcp",
            "protocol": "mcp",
            "transport": "stdio"
        }

    对自定义测试 Client，如果没有这些属性，会安全标记为 custom。
    """

    return {
        "client_mode": str(
            getattr(
                client,
                "client_mode",
                "custom",
            )
        ),
        "protocol": str(
            getattr(
                client,
                "protocol",
                "custom",
            )
        ),
        "transport": str(
            getattr(
                client,
                "transport",
                "custom",
            )
        ),
    }
