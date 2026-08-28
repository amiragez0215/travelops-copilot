from __future__ import annotations

from typing import Any

from app.nodes.tool_check_node import route_after_tool_check, tool_check_node
from app.nodes.tool_execute_node import tool_execute_node
from app.nodes.tool_plan_node import tool_plan_node
from app.nodes.tool_plan_node import _optional_tool_definitions
from app.mcp.tool_policy import filter_tool_policy
from app.mcp.tool_registry import ToolDescriptor, ToolRegistry


class FakeNativeToolClient:
    model_id = "fake-native-tool-model"

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls
        self.received_tools: list[dict[str, Any]] = []
        self.system_prompt = ""

    def generate_tool_calls(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.received_tools = kwargs["tools"]
        self.system_prompt = kwargs["system_prompt"]
        return self.calls


class FakeTravelClient:
    client_mode = "fake"
    protocol = "local"
    transport = "in_process"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call_tool(self, *, server_id: str, tool_name: str, arguments: dict[str, Any]):
        self.calls.append((tool_name, dict(arguments)))
        return getattr(self, tool_name)(**arguments)

    def get_weather(self, city: str, start_date: str, end_date: str | None = None):
        return {"status": "ok", "city": city, "daily": [], "risks": [], "warnings": [], "source": "fake"}

    def search_flights(self, origin: str, destination: str, depart_date: str, return_date: str | None = None, people_count: int = 1):
        return {
            "status": "ok",
            "outbound": [{"flight_id": "out-1"}],
            "return": [{"flight_id": "ret-1"}],
            "source": "fake",
        }

    def search_hotels(self, city: str, nights: int, people_count: int = 1):
        return {"status": "ok", "items": [{"hotel_id": "hotel-1"}], "source": "fake"}

    def search_city_activities(self, **kwargs: Any):
        return {
            "status": "ok",
            "items": [{
                "activity_id": "SHA-ACT-001", "name": "西岸美术馆当代艺术展",
                "city": kwargs["city"], "available_dates": ["2026-10-03"],
            }],
            "source": "fake",
        }


def _state() -> dict[str, Any]:
    return {
        "raw_message": "想多看展览和夜间活动。",
        "planning_context": {
            "request": {
                "origin": "杭州", "destination": "上海",
                "start_date": "2026-10-03", "end_date": "2026-10-05",
                "nights": 2, "people_count": 1,
            },
            "preference_weights": {"activities.culture": 0.9},
        },
        "tool_plan_retry_count": 0,
        "tool_execute_retry_count": 0,
    }


def test_tool_plan_policy_always_adds_required_tools_and_llm_only_adds_activity():
    fake = FakeNativeToolClient([
        {
            "call_id": "call-activity", "tool_name": "search_city_activities",
            "arguments": {"interests": ["展览", "夜间活动"]},
        }
    ])

    update = tool_plan_node(_state(), client=fake)
    plan = update["tool_plan"]
    names = [call["tool_name"] for call in plan["calls"]]

    assert names[:3] == ["get_weather", "search_flights", "search_hotels"]
    assert names[3:] == ["search_city_activities"]
    assert plan["calls"][3]["arguments"]["city"] == "上海"
    assert plan["calls"][3]["arguments"]["start_date"] == "2026-10-03"
    assert fake.received_tools[0]["function"]["name"] == "search_city_activities"
    assert "天气" not in fake.system_prompt
    assert "航班" not in fake.system_prompt
    assert "酒店" not in fake.system_prompt
    assert "search_city_activities" not in fake.system_prompt


def test_tool_plan_compiles_all_seen_calls_through_the_optional_tool_registry():
    """不再只消费第一个模型调用；每种工具仍由其注册 policy 限流。"""

    fake = FakeNativeToolClient([
        {"call_id": "a1", "tool_name": "search_city_activities", "arguments": {"interests": ["展览"]}},
        {"call_id": "a2", "tool_name": "search_city_activities", "arguments": {"interests": ["夜间活动"]}},
        {"call_id": "unknown", "tool_name": "unknown_optional", "arguments": {}},
    ])

    update = tool_plan_node(_state(), client=fake)
    plan = update["tool_plan"]

    assert [call["tool_name"] for call in plan["calls"]] == [
        "get_weather", "search_flights", "search_hotels", "search_city_activities",
    ]
    assert {issue["type"] for issue in plan["validation_issues"]} == {
        "optional_tool_call_limit_exceeded",
        "unsupported_optional_tool",
    }


def test_new_discovered_non_required_tool_is_automatically_llm_visible():
    registry = ToolRegistry([
        ToolDescriptor("get_weather", "weather", "", {"type": "object"}),
        ToolDescriptor("search_flights", "flight", "", {"type": "object"}),
        ToolDescriptor("search_hotels", "hotel", "", {"type": "object"}),
        ToolDescriptor("search_restaurants", "food", "查询餐厅", {
            "type": "object", "properties": {"cuisine": {"type": "string"}},
            "required": ["cuisine"], "additionalProperties": False,
        }),
    ])

    definitions = _optional_tool_definitions(filter_tool_policy(registry))

    assert [item["function"]["name"] for item in definitions] == ["search_restaurants"]
    assert definitions[0]["function"]["parameters"]["required"] == ["cuisine"]


def test_tool_execute_preserves_rank_budget_inputs_and_adds_activity_context():
    state = _state()
    state.update(tool_plan_node(state, client=FakeNativeToolClient([
        {"call_id": "a1", "tool_name": "search_city_activities", "arguments": {"interests": ["展览"]}}
    ])))

    client = FakeTravelClient()
    update = tool_execute_node(state, client=client)

    assert update["raw_flight_results"]["outbound"][0]["flight_id"] == "out-1"
    assert update["raw_hotel_results"]["items"][0]["hotel_id"] == "hotel-1"
    assert update["activity_candidates"][0]["activity_id"] == "SHA-ACT-001"
    assert update["planning_context"]["activity_candidates"][0]["activity_id"] == "SHA-ACT-001"
    assert client.calls[0] == ("get_weather", state["tool_plan"]["calls"][0]["arguments"])
    assert {item["server_id"] for item in update["tool_call_results"]} == {"weather", "flight", "hotel", "activity"}


def test_tool_execute_uses_compiled_arguments_instead_of_rebuilding_from_state():
    state = _state()
    state.update(tool_plan_node(state, client=FakeNativeToolClient([])))
    state["tool_plan"]["calls"][0]["arguments"]["city"] = "计划中的城市"
    client = FakeTravelClient()

    tool_execute_node(state, client=client)

    assert client.calls[0][1]["city"] == "计划中的城市"


def test_tool_check_passes_required_calls_and_allows_optional_activity_degrade():
    state = _state()
    state.update(tool_plan_node(state, client=FakeNativeToolClient([
        {"call_id": "a1", "tool_name": "search_city_activities", "arguments": {"interests": ["展览"]}}
    ])))
    state.update({
        "weather_fetch_meta": {"status": "ok"},
        "flight_fetch_meta": {"status": "ok"},
        "hotel_fetch_meta": {"status": "ok"},
        "activity_fetch_meta": {"status": "failed"},
    })

    update = tool_check_node(state)

    assert update["tool_check_result"]["status"] == "degraded"
    assert route_after_tool_check(update) == "candidate_rank"


def test_tool_check_retries_failed_required_execution_with_bound():
    state = _state()
    state.update(tool_plan_node(state, client=FakeNativeToolClient([])))
    state.update({
        "weather_fetch_meta": {"status": "failed"},
        "flight_fetch_meta": {"status": "ok"},
        "hotel_fetch_meta": {"status": "ok"},
    })

    update = tool_check_node(state)

    assert update["tool_check_result"]["next_action"] == "tool_execute"
    assert update["tool_execute_retry_count"] == 1


def test_invalid_llm_activity_call_replans_once_without_losing_required_policy():
    state = _state()
    state.update(tool_plan_node(state, client=FakeNativeToolClient([
        {"call_id": "bad", "tool_name": "unknown_tool", "arguments": {}}
    ])))
    state.update({
        "weather_fetch_meta": {"status": "ok"},
        "flight_fetch_meta": {"status": "ok"},
        "hotel_fetch_meta": {"status": "ok"},
    })

    update = tool_check_node(state)

    assert update["tool_check_result"]["next_action"] == "tool_plan"
    assert update["tool_plan_retry_count"] == 1
    assert [call["tool_name"] for call in state["tool_plan"]["calls"]] == [
        "get_weather", "search_flights", "search_hotels"
    ]
