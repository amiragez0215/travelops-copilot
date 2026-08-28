from __future__ import annotations

from dataclasses import dataclass

from app.mcp.tool_registry import ToolDescriptor, ToolRegistry


REQUIRED_TOOL_NAMES = ("get_weather", "search_flights", "search_hotels")

# Discovery Tool 默认都是 LLM-visible Optional；这里只维护不能暴露给
# 模型的例外。当前四个 Tool 均为只读查询，因此默认集合为空。
DENIED_INTERNAL_TOOL_NAMES: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ToolPolicyDecision:
    required_tools: tuple[ToolDescriptor, ...]
    llm_visible_optional_tools: tuple[ToolDescriptor, ...]


def filter_tool_policy(
    registry: ToolRegistry,
    *,
    unavailable_or_disallowed: set[str] | frozenset[str] = frozenset(),
) -> ToolPolicyDecision:
    required_names = set(REQUIRED_TOOL_NAMES)
    excluded = required_names | set(DENIED_INTERNAL_TOOL_NAMES) | set(unavailable_or_disallowed)
    return ToolPolicyDecision(
        required_tools=tuple(registry.resolve(name) for name in REQUIRED_TOOL_NAMES),
        llm_visible_optional_tools=tuple(
            registry.resolve(name)
            for name in registry.names()
            if name not in excluded
        ),
    )
