from __future__ import annotations

import pytest

from app.mcp.server_registry import build_default_server_registry
from app.mcp.tool_policy import filter_tool_policy
from app.mcp.tool_registry import ToolDescriptor, ToolRegistry
from app.mcp.tool_runtime import initialize_mcp_tool_runtime, reset_mcp_tool_runtime


def test_server_registry_has_one_stdio_server_per_domain():
    registry = build_default_server_registry()
    assert set(registry) == {"weather", "flight", "hotel", "activity"}
    assert all(item.transport == "stdio" for item in registry.values())
    assert registry["activity"].args[-1].endswith("activity_server")


def test_local_runtime_builds_global_registry_and_policy_boundary():
    reset_mcp_tool_runtime()
    runtime = initialize_mcp_tool_runtime(discover=False)
    policy = filter_tool_policy(runtime.tool_registry)
    assert set(runtime.tool_registry.names()) == {"get_weather", "search_flights", "search_hotels", "search_city_activities"}
    assert {item.tool_name for item in policy.required_tools} == {"get_weather", "search_flights", "search_hotels"}
    assert [item.tool_name for item in policy.llm_visible_optional_tools] == ["search_city_activities"]


def test_registry_rejects_duplicate_tool_names():
    tool = ToolDescriptor("same", "a", "", {"type": "object"})
    with pytest.raises(ValueError, match="名称重复"):
        ToolRegistry([tool, ToolDescriptor("same", "b", "", {"type": "object"})])


def test_registry_validates_discovered_argument_schema():
    runtime = initialize_mcp_tool_runtime(discover=False)
    runtime.tool_registry.validate_arguments("search_hotels", {"city": "北京", "nights": 2, "people_count": 1})
    with pytest.raises(ValueError, match="未声明参数"):
        runtime.tool_registry.validate_arguments("search_hotels", {"city": "北京", "nights": 2, "invented": True})


def test_optional_tools_are_discovery_registry_minus_required_and_disallowed():
    tools = [
        ToolDescriptor("get_weather", "weather", "", {"type": "object"}),
        ToolDescriptor("search_flights", "flight", "", {"type": "object"}),
        ToolDescriptor("search_hotels", "hotel", "", {"type": "object"}),
        ToolDescriptor("search_city_activities", "activity", "", {"type": "object"}),
        ToolDescriptor("search_restaurants", "food", "", {"type": "object"}),
        ToolDescriptor("internal_refresh", "admin", "", {"type": "object"}),
    ]
    policy = filter_tool_policy(
        ToolRegistry(tools),
        unavailable_or_disallowed={"internal_refresh"},
    )

    assert [tool.tool_name for tool in policy.llm_visible_optional_tools] == [
        "search_city_activities",
        "search_restaurants",
    ]
