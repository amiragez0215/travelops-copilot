from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ToolDescriptor:
    tool_name: str
    server_id: str
    description: str
    input_schema: dict[str, Any]
    transport: str = "stdio"

    def as_llm_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class ToolRegistry:
    """应用级不可变工具目录；重复工具名会在启动期直接失败。"""

    def __init__(self, tools: Iterable[ToolDescriptor]) -> None:
        items: dict[str, ToolDescriptor] = {}
        for tool in tools:
            if tool.tool_name in items:
                raise ValueError(f"MCP Tool 名称重复：{tool.tool_name}")
            items[tool.tool_name] = tool
        self._tools: Mapping[str, ToolDescriptor] = MappingProxyType(items)

    def resolve(self, tool_name: str) -> ToolDescriptor:
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise KeyError(f"Global Tool Registry 中不存在工具：{tool_name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def validate_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """执行轻量 JSON Schema 边界检查，不依赖 Server 端二次报错。"""
        schema = self.resolve(tool_name).input_schema
        properties = schema.get("properties", {})
        missing = [name for name in schema.get("required", []) if name not in arguments]
        if missing:
            raise ValueError(f"{tool_name} 缺少必填参数：{missing}")
        if schema.get("additionalProperties") is False:
            extras = sorted(set(arguments) - set(properties))
            if extras:
                raise ValueError(f"{tool_name} 包含未声明参数：{extras}")
        python_types = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
        for name, value in arguments.items():
            declared = properties.get(name, {}).get("type") if isinstance(properties.get(name), dict) else None
            allowed = declared if isinstance(declared, list) else [declared]
            if value is None and "null" in allowed:
                continue
            expected = tuple(python_types[item] for item in allowed if item in python_types)
            if expected and (not isinstance(value, expected) or (bool in expected and isinstance(value, int) and not isinstance(value, bool))):
                raise ValueError(f"{tool_name}.{name} 参数类型不符合 Discovery Schema")

    def __len__(self) -> int:
        return len(self._tools)
