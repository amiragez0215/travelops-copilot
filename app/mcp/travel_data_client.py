from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

from app.mcp.travel_data_tools import (
    get_weather_tool,
    search_city_activities_tool,
    search_flights_tool,
    search_hotels_tool,
)


class TravelDataClient(Protocol):
    """
    天气、航班和酒店节点共同依赖的 Client 协议。

    Node 只关心下面三个业务方法，不关心底层到底是：

        - LocalTravelDataClient：进程内直接调用 Tool；
        - MCPTravelDataClient：通过 MCP stdio 调用 Server Tool。

    这种依赖倒置让正式运行可以使用 MCP，
    同时单元测试仍然可以使用快速、稳定的 Local Client。
    """

    def get_weather(
        self,
        city: str,
        start_date: str,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """查询指定城市和日期范围的天气。"""
        ...

    def search_flights(
        self,
        origin: str,
        destination: str,
        depart_date: str,
        return_date: str | None = None,
        people_count: int = 1,
    ) -> dict[str, Any]:
        """查询往返航班候选。"""
        ...

    def search_hotels(
        self,
        city: str,
        nights: int,
        people_count: int = 1,
    ) -> dict[str, Any]:
        """查询目标城市的酒店候选。"""
        ...

    def search_city_activities(
        self,
        city: str,
        start_date: str,
        end_date: str,
        interests: list[str] | None = None,
        indoor_preference: bool | None = None,
        max_results: int = 12,
    ) -> dict[str, Any]:
        """查询可选活动候选。"""
        ...


@dataclass
class LocalTravelDataClient:
    """
    进程内本地 Client，主要用于单元测试。

    调用链：

        Node
            -> LocalTravelDataClient
            -> travel_data_tools.py
            -> Mock Provider
            -> data/mock/*.json

    它不会建立 MCP Session，也不会启动 MCP Server 子进程。

    这个类没有删除，因为：

        1. 单元测试不应该频繁启动 stdio 子进程；
        2. Provider / Tool / Node 可以在不依赖 MCP 运行环境时独立测试；
        3. 当排查 MCP 问题时，可以用 local 模式快速判断业务数据层是否正常。
    """

    # 这些类属性只用于 Trace，帮助区分真实调用链。
    client_mode: ClassVar[str] = "local"
    protocol: ClassVar[str] = "local"
    transport: ClassVar[str] = "in_process"

    weather_mock_file: str | Path | None = None
    flights_mock_file: str | Path | None = None
    hotels_mock_file: str | Path | None = None
    activities_mock_file: str | Path | None = None

    def get_weather(
        self,
        city: str,
        start_date: str,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """直接调用本地 get_weather_tool。"""

        return get_weather_tool(
            city=city,
            start_date=start_date,
            end_date=end_date,
            mock_file=self.weather_mock_file,
        )

    def search_flights(
        self,
        origin: str,
        destination: str,
        depart_date: str,
        return_date: str | None = None,
        people_count: int = 1,
    ) -> dict[str, Any]:
        """直接调用本地 search_flights_tool。"""

        return search_flights_tool(
            origin=origin,
            destination=destination,
            depart_date=depart_date,
            return_date=return_date,
            people_count=people_count,
            mock_file=self.flights_mock_file,
        )

    def search_hotels(
        self,
        city: str,
        nights: int,
        people_count: int = 1,
    ) -> dict[str, Any]:
        """直接调用本地 search_hotels_tool。"""

        return search_hotels_tool(
            city=city,
            nights=nights,
            people_count=people_count,
            mock_file=self.hotels_mock_file,
        )

    def search_city_activities(
        self,
        city: str,
        start_date: str,
        end_date: str,
        interests: list[str] | None = None,
        indoor_preference: bool | None = None,
        max_results: int = 12,
    ) -> dict[str, Any]:
        """直接调用本地活动 Tool，供单测和本地诊断使用。"""

        return search_city_activities_tool(
            city=city,
            start_date=start_date,
            end_date=end_date,
            interests=interests,
            indoor_preference=indoor_preference,
            max_results=max_results,
            mock_file=self.activities_mock_file,
        )


@dataclass
class MCPTravelDataClient:
    """
    真正通过 MCP stdio 调用工具的同步 Client。

    每次业务方法调用的完整链路：

        WeatherNode / FlightSearchNode / HotelSearchNode
            -> MCPTravelDataClient
            -> stdio_client(...)
            -> 启动 python -m app.mcp.travel_data_server
            -> ClientSession.initialize()
            -> ClientSession.call_tool(...)
            -> FastMCP Server
            -> travel_data_tools.py
            -> Mock Provider
            -> data/mock/*.json

    当前 v1 采用“每次 Tool 调用建立一个短生命周期 MCP Session”的最小实现：

        - 代码简单；
        - Session 和子进程会被 async with 自动关闭；
        - 足以证明正式主链真实经过 MCP 协议；
        - 代价是天气、航班、酒店会分别启动一次 Server 子进程。

    未来只有在性能实验明确证明启动成本过高时，
    才需要升级为应用级长连接 Session。当前阶段不做这层复杂化。
    """

    client_mode: ClassVar[str] = "mcp"
    protocol: ClassVar[str] = "mcp"
    transport: ClassVar[str] = "stdio"

    # command 使用当前 Python 解释器，确保 MCP Server 使用同一依赖环境。
    command: str = sys.executable

    # args 通过 Python 模块方式启动项目内 MCP Server。
    args: tuple[str, ...] = (
        "-m",
        "app.mcp.travel_data_server",
    )

    # 一次完整 MCP 连接、初始化和 Tool 调用允许的最长时间。
    # 这是最小必要保护，避免 Server 无响应时完整 Graph 永久卡住。
    timeout_seconds: float = 30.0

    def get_weather(
        self,
        city: str,
        start_date: str,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """通过 MCP Server 调用 get_weather Tool。"""

        return self._call_tool_sync(
            tool_name="get_weather",
            arguments={
                "city": city,
                "start_date": start_date,
                "end_date": end_date,
            },
        )

    def search_flights(
        self,
        origin: str,
        destination: str,
        depart_date: str,
        return_date: str | None = None,
        people_count: int = 1,
    ) -> dict[str, Any]:
        """通过 MCP Server 调用 search_flights Tool。"""

        return self._call_tool_sync(
            tool_name="search_flights",
            arguments={
                "origin": origin,
                "destination": destination,
                "depart_date": depart_date,
                "return_date": return_date,
                "people_count": people_count,
            },
        )

    def search_hotels(
        self,
        city: str,
        nights: int,
        people_count: int = 1,
    ) -> dict[str, Any]:
        """通过 MCP Server 调用 search_hotels Tool。"""

        return self._call_tool_sync(
            tool_name="search_hotels",
            arguments={
                "city": city,
                "nights": nights,
                "people_count": people_count,
            },
        )

    def search_city_activities(
        self,
        city: str,
        start_date: str,
        end_date: str,
        interests: list[str] | None = None,
        indoor_preference: bool | None = None,
        max_results: int = 12,
    ) -> dict[str, Any]:
        """通过同一个 MCP Server 调用活动查询 Tool。"""

        return self._call_tool_sync(
            tool_name="search_city_activities",
            arguments={
                "city": city,
                "start_date": start_date,
                "end_date": end_date,
                "interests": interests or [],
                "indoor_preference": indoor_preference,
                "max_results": max_results,
            },
        )

    def _call_tool_sync(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        把 MCP SDK 的异步调用包装成当前 Graph 可用的同步方法。

        当前 PlanTrip Graph 使用同步 invoke，FastAPI 的同步 Route 也会在线程池中执行，
        因此这里通常没有正在运行的 asyncio event loop。

        核心功能只有两步：

            1. asyncio.run(...) 执行异步 MCP Session；
            2. asyncio.wait_for(...) 为完整调用设置超时。

        如果这个同步 Client 被错误地放到一个已运行的事件循环中，
        这里直接给出明确错误，而不是隐式创建复杂的跨线程事件循环。
        """

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # 没有正在运行的事件循环，这是当前同步 Graph 的正常路径。
            pass
        else:
            raise RuntimeError(
                "MCPTravelDataClient 是同步 Client，"
                "不能直接在已有 asyncio event loop 中调用。"
            )

        try:
            # 这是 MCP Client 的同步入口核心代码。
            return asyncio.run(
                asyncio.wait_for(
                    self._call_tool_async(
                        tool_name,
                        arguments,
                    ),
                    timeout=self.timeout_seconds,
                )
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"MCP Tool {tool_name!r} 调用超过 "
                f"{self.timeout_seconds} 秒。"
            ) from exc

    async def _call_tool_async(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        建立一次真实 MCP stdio Session，并调用指定 Tool。

        这里把 MCP SDK import 放在函数内部，原因是：

            - 普通单元测试只使用 LocalTravelDataClient；
            - 测试导入 Node 时不需要立即加载 MCP SDK；
            - 如果 MCP 依赖缺失，错误只会出现在真正选择 mcp 模式时。
        """

        from mcp import (
            ClientSession,
            StdioServerParameters,
        )
        from mcp.client.stdio import (
            stdio_client,
        )

        # 1. 描述如何启动 MCP Server 子进程。
        server_params = StdioServerParameters(
            command=self.command,
            args=list(self.args),
        )

        # 2. stdio_client 启动 Server，并建立读写通道。
        async with stdio_client(
            server_params
        ) as (read, write):
            # 3. ClientSession 负责 MCP 协议层的请求和响应。
            async with ClientSession(
                read,
                write,
            ) as session:
                # 4. initialize() 完成版本和能力协商。
                await session.initialize()

                # 5. call_tool() 是实际的 MCP Tool 调用核心代码。
                result = await session.call_tool(
                    tool_name,
                    arguments=arguments,
                )

        # 6. Session 退出后解析结构化返回值。
        return _extract_mcp_result(
            result
        )


def _extract_mcp_result(
    result: Any,
) -> dict[str, Any]:
    """
    将 MCP CallToolResult 解析成项目内部统一使用的 dict。

    FastMCP 返回带类型注解的 dict 时，SDK 通常会把结果放到：

        structuredContent
        或 structured_content

    为兼容当前项目已经使用过的 SDK 版本，
    如果没有结构化字段，再尝试解析第一条 TextContent 中的 JSON。
    """

    # 1. MCP Tool 明确返回错误时，不把错误文本伪装成正常业务结果。
    is_error = bool(
        getattr(
            result,
            "isError",
            getattr(
                result,
                "is_error",
                False,
            ),
        )
    )

    if is_error:
        raise RuntimeError(
            "MCP Tool 返回错误："
            + _extract_mcp_error_text(
                result
            )
        )

    # 2. 优先读取结构化 Tool Result。
    for attr in (
        "structuredContent",
        "structured_content",
    ):
        value = getattr(
            result,
            attr,
            None,
        )

        if isinstance(value, dict):
            return value

    # 3. 某些 SDK 版本会把 dict 作为 JSON 文本放进 content。
    content = getattr(
        result,
        "content",
        None,
    )

    if isinstance(content, list):
        for item in content:
            text = getattr(
                item,
                "text",
                None,
            )

            if not isinstance(text, str):
                continue

            parsed = json.loads(
                text
            )

            if isinstance(parsed, dict):
                return parsed

    raise ValueError(
        "无法把 MCP Tool 返回结果解析为 dict："
        f"{result!r}"
    )


def _extract_mcp_error_text(
    result: Any,
) -> str:
    """从 MCP 错误结果中提取简短文本，便于写入 Node errors 和 Trace。"""

    content = getattr(
        result,
        "content",
        None,
    )

    if isinstance(content, list):
        texts = [
            str(text).strip()
            for item in content
            if (
                text := getattr(
                    item,
                    "text",
                    None,
                )
            )
            if str(text).strip()
        ]

        if texts:
            return " | ".join(texts)

    return repr(result)
