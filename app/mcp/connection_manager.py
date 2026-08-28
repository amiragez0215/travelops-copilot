from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from app.mcp.server_registry import MCPServerConfig
from app.mcp.tool_registry import ToolDescriptor


@dataclass
class MCPConnectionManager:
    """应用级连接工厂；每次 discovery/call 都创建并关闭短 Session。"""

    servers: dict[str, MCPServerConfig]
    timeout_seconds: float = 30.0

    def discover_server(self, server_id: str) -> list[ToolDescriptor]:
        result = self._run(self._discover_async(server_id))
        return result

    def call_tool(
        self, *, server_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return self._run(self._call_async(server_id, tool_name, arguments))

    def _run(self, awaitable: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            if hasattr(awaitable, "close"):
                awaitable.close()
            raise RuntimeError("同步 MCPConnectionManager 不能在运行中的 event loop 内调用")
        return asyncio.run(asyncio.wait_for(awaitable, timeout=self.timeout_seconds))

    def _config(self, server_id: str) -> MCPServerConfig:
        try:
            return self.servers[server_id]
        except KeyError as exc:
            raise KeyError(f"未知 MCP Server：{server_id}") from exc

    async def _discover_async(self, server_id: str) -> list[ToolDescriptor]:
        config = self._config(server_id)
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=config.command, args=list(config.args))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
        descriptors = []
        for tool in getattr(result, "tools", []):
            schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
            descriptors.append(ToolDescriptor(
                tool_name=str(tool.name), server_id=server_id,
                description=str(getattr(tool, "description", "") or ""),
                input_schema=dict(schema), transport=config.transport,
            ))
        return descriptors

    async def _call_async(
        self, server_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        config = self._config(server_id)
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=config.command, args=list(config.args))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
        if bool(getattr(result, "isError", getattr(result, "is_error", False))):
            raise RuntimeError(f"MCP Tool {tool_name!r} 返回错误")
        for attr in ("structuredContent", "structured_content"):
            value = getattr(result, attr, None)
            if isinstance(value, dict):
                return value
        for item in getattr(result, "content", []) or []:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                value = json.loads(text)
                if isinstance(value, dict):
                    return value
        raise ValueError(f"无法解析 MCP Tool {tool_name!r} 的结构化结果")
