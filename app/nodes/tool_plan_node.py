from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from app.agents.state import TravelState
from app.common.config import settings
from app.llm.deepseek_json_client import DeepSeekJSONClient, NativeToolCallingClient
from app.mcp.tool_policy import filter_tool_policy
from app.mcp.tool_runtime import get_mcp_tool_runtime
from app.schemas.tool_orchestration_schema import (
    ActivityToolArguments,
    ExecutableToolCall,
    ToolPlan,
)


ACTIVITY_TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_city_activities",
        "description": (
            "当用户明确希望安排展览、博物馆、街区体验、城市文化、夜间活动、"
            "亲子、低步行或雨天室内活动时调用。普通旅行且没有这些具体活动诉求时不要调用。"
            "城市和日期由系统注入，不要在参数中生成。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "interests": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "从用户原话提炼的活动兴趣，最多 12 项。",
                },
                "indoor_preference": {
                    "type": "boolean",
                    "description": "用户明确偏好室内时为 true；没有明确偏好时省略该字段。",
                },
            },
            "additionalProperties": False,
        },
    },
}

ACTIVITY_INTENT_TERMS = (
    "展览", "展", "博物馆", "街区", "城市文化", "夜间", "夜生活",
    "亲子", "带父母", "低步行", "少走路", "室内", "雨天", "艺术",
)

# LLM 仅能看到并选择这个注册表中的可选增强工具。天气、航班和酒店属于
# Policy 直接生成的必选调用，不会作为模型上下文的一部分暴露出去。
# 新增可选工具时注册其 Schema、参数模型与可信参数注入函数即可；Router
# Prompt 不需要写任何工具名称。
MAX_OPTIONAL_TOOL_CALLS = 4


def _inject_activity_arguments(
    semantic_args: BaseModel,
    request: dict[str, Any],
) -> dict[str, Any]:
    """为活动工具注入可信行程范围；模型只负责活动语义参数。"""

    if not isinstance(semantic_args, ActivityToolArguments):
        raise TypeError("活动工具参数必须是 ActivityToolArguments")
    return {
        "city": _require_text(request.get("destination"), "destination"),
        "start_date": _require_text(request.get("start_date"), "start_date"),
        "end_date": _require_text(request.get("end_date"), "end_date"),
        "interests": semantic_args.interests,
        "indoor_preference": semantic_args.indoor_preference,
        "max_results": settings.activity_tool_max_results,
    }


OPTIONAL_TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "search_city_activities": {
        "definition": ACTIVITY_TOOL_DEFINITION,
        "arguments_model": ActivityToolArguments,
        "inject_trusted_arguments": _inject_activity_arguments,
        "max_calls": 1,
    },
}


def tool_plan_node(
    state: TravelState,
    client: NativeToolCallingClient | None = None,
) -> dict[str, Any]:
    """
    生成受 Policy 约束的统一 ToolPlan。

    关键边界：必选工具由程序无条件加入但不暴露给 LLM；LLM 只看到当前
    注册的可选 Tool Schema，因此不能取消必选调用或重写可信约束。
    """

    node_name = "tool_plan"
    started_at = perf_counter()
    planning_context = state.get("planning_context")
    if not isinstance(planning_context, dict):
        raise ValueError("state['planning_context'] 缺失，请先运行 PlanningContextNode")
    request = planning_context.get("request")
    if not isinstance(request, dict):
        raise ValueError("planning_context.request 缺失")

    registry = get_mcp_tool_runtime().tool_registry
    policy = filter_tool_policy(registry)
    required_calls = _build_required_calls(request, registry)
    retry_count = _safe_int(state.get("tool_plan_retry_count"))
    validation_issues: list[dict[str, Any]] = []
    optional_calls: list[ExecutableToolCall] = []
    model_id = "deterministic_activity_rule_v1"
    status = "planned"

    try:
        if client is not None or settings.tool_plan_mode == "llm":
            active_client = client or _build_default_client()
            model_id = active_client.model_id
            raw_calls = active_client.generate_tool_calls(
                system_prompt=_build_system_prompt(),
                user_prompt=_build_user_prompt(state, planning_context),
                tools=_optional_tool_definitions(policy),
                max_tokens=settings.tool_plan_max_tokens,
            )
            optional_calls, validation_issues = _compile_optional_calls(
                raw_calls=raw_calls,
                request=request, registry=registry, policy=policy,
            )
        else:
            optional_calls = _rule_optional_calls(state=state, request=request, registry=registry)
    except Exception as exc:
        # LLM 失败只取消可选增强；必选调用仍然保留。这是 Policy-Constrained
        # 架构的降级边界，不能让模型服务成为航班/酒店/天气的单点故障。
        status = "degraded"
        validation_issues.append(
            {"type": "optional_tool_planner_failed", "message": str(exc)}
        )

    plan = ToolPlan(
        status=status,
        planner_mode=("llm" if client is not None or settings.tool_plan_mode == "llm" else "rule"),
        model_id=model_id,
        plan_retry_count=retry_count,
        calls=_compile_executable_calls([*required_calls, *optional_calls], registry, validation_issues),
        llm_selected_tools=[call.tool_name for call in optional_calls],
        validation_issues=validation_issues,
    )
    trace = {
        "node_name": node_name,
        "tool_name": model_id,
        "input_summary": f"destination={request.get('destination')}, retry={retry_count}",
        "output_summary": (
            f"required=3, optional={len(optional_calls)}, issues={len(validation_issues)}"
        ),
        "status": status,
        "latency_ms": int((perf_counter() - started_at) * 1000),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return {"tool_plan": plan.to_state_dict(), "trace": [trace]}


def _build_required_calls(request: dict[str, Any], registry: Any) -> list[ExecutableToolCall]:
    """从可信 State 生成三个一定执行的 MCP 调用。"""

    origin = _require_text(request.get("origin"), "origin")
    destination = _require_text(request.get("destination"), "destination")
    start_date = _require_text(request.get("start_date"), "start_date")
    end_date = _require_text(request.get("end_date"), "end_date")
    people_count = max(_safe_int(request.get("people_count"), default=1), 1)
    nights = max(_safe_int(request.get("nights"), default=0), 0)
    return [
        ExecutableToolCall(
            call_id="policy_weather", requirement_key="weather", tool_name="get_weather",
            server_id=registry.resolve("get_weather").server_id,
            source="required_policy",
            arguments={"city": destination, "start_date": start_date, "end_date": end_date},
            reason="天气是当前旅行规划的必选外部事实。",
        ),
        ExecutableToolCall(
            call_id="policy_flights", requirement_key="round_trip_flights", tool_name="search_flights",
            server_id=registry.resolve("search_flights").server_id,
            source="required_policy",
            arguments={
                "origin": origin, "destination": destination, "depart_date": start_date,
                "return_date": end_date, "people_count": people_count,
            },
            reason="去程和返程航班是当前演示范围的必选候选。",
        ),
        ExecutableToolCall(
            call_id="policy_hotels", requirement_key="hotel", tool_name="search_hotels",
            server_id=registry.resolve("search_hotels").server_id,
            source="required_policy",
            arguments={"city": destination, "nights": nights, "people_count": people_count},
            reason="酒店是当前演示范围的必选候选。",
        ),
    ]


def _compile_optional_calls(
    *, raw_calls: list[dict[str, Any]], request: dict[str, Any], registry: Any,
    policy: Any,
) -> tuple[list[ExecutableToolCall], list[dict[str, Any]]]:
    """按可选工具注册表编译 LLM 调用并注入可信参数。"""

    issues: list[dict[str, Any]] = []
    compiled: list[ExecutableToolCall] = []
    call_counts: Counter[str] = Counter()
    allowed_names = {tool.tool_name for tool in policy.llm_visible_optional_tools}
    for raw in raw_calls[:MAX_OPTIONAL_TOOL_CALLS]:
        tool_name = str(raw.get("tool_name") or "")
        if tool_name not in allowed_names:
            issues.append({"type": "unsupported_optional_tool", "tool_name": tool_name})
            continue
        spec = OPTIONAL_TOOL_REGISTRY.get(tool_name)
        max_calls = int(spec["max_calls"]) if spec is not None else 1
        if call_counts[tool_name] >= max_calls:
            issues.append({"type": "optional_tool_call_limit_exceeded", "tool_name": tool_name})
            continue
        try:
            descriptor = registry.resolve(tool_name)
            if spec is not None:
                semantic_args = spec["arguments_model"](**dict(raw.get("arguments") or {}))
                arguments = spec["inject_trusted_arguments"](semantic_args, request)
            else:
                # 普通 Discovery Optional Tool 直接使用 MCP Schema 中定义的
                # 参数；统一 Compiler 随后仍会执行 Registry Schema 校验。
                arguments = dict(raw.get("arguments") or {})
        except Exception as exc:
            issues.append({"type": "invalid_optional_tool_arguments", "tool_name": tool_name, "message": str(exc)})
            continue
        call_counts[tool_name] += 1
        compiled.append(
            ExecutableToolCall(
                call_id=str(raw.get("call_id") or f"llm_{tool_name}_{call_counts[tool_name]}"),
                requirement_key=f"optional_{tool_name}",
                tool_name=tool_name,
                server_id=descriptor.server_id,
                source="llm_selected",
                arguments=arguments,
                reason="LLM 根据用户需求选择了注册的可选工具。",
            )
        )
    return compiled, issues


def _optional_tool_definitions(policy: Any) -> list[dict[str, Any]]:
    """返回本轮可由 LLM 选择的工具 Schema，保持注册表顺序。"""

    definitions: list[dict[str, Any]] = []
    for descriptor in policy.llm_visible_optional_tools:
        spec = OPTIONAL_TOOL_REGISTRY.get(descriptor.tool_name)
        definitions.append(
            spec["definition"] if spec is not None else descriptor.as_llm_definition()
        )
    return definitions


def _rule_optional_calls(*, state: TravelState, request: dict[str, Any], registry: Any) -> list[ExecutableToolCall]:
    """无模型测试/降级路径：仅用显式关键词复现可选调用边界。"""

    raw_message = str(state.get("raw_message") or "")
    interests = [term for term in ACTIVITY_INTENT_TERMS if term in raw_message]
    if not interests:
        return []
    policy = filter_tool_policy(registry)
    return _compile_optional_calls(
        raw_calls=[{
            "call_id": "rule_activity_1", "tool_name": "search_city_activities",
            "arguments": {"interests": interests, "indoor_preference": (True if "室内" in interests else None)},
        }],
        request=request, registry=registry, policy=policy,
    )[0]


def _deduplicate_calls(
    calls: list[ExecutableToolCall], issues: list[dict[str, Any]]
) -> list[ExecutableToolCall]:
    """按 server/tool/规范化参数去重；必选调用因合并顺序而自然优先。"""

    seen: set[tuple[str, str, str]] = set()
    result: list[ExecutableToolCall] = []
    for call in calls:
        key = (call.server_id, call.tool_name, json.dumps(call.arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        if key in seen:
            issues.append({"type": "duplicate_tool_call", "tool_name": call.tool_name, "call_id": call.call_id})
            continue
        seen.add(key)
        result.append(call)
    return result


def _compile_executable_calls(
    calls: list[ExecutableToolCall], registry: Any, issues: list[dict[str, Any]]
) -> list[ExecutableToolCall]:
    """在 Merge 后执行 Registry/Server/Schema Validate，再做确定性去重。"""
    valid: list[ExecutableToolCall] = []
    for call in calls:
        try:
            descriptor = registry.resolve(call.tool_name)
            if descriptor.server_id != call.server_id:
                raise ValueError("server_id 与 Global Tool Registry 不一致")
            registry.validate_arguments(call.tool_name, call.arguments)
        except Exception as exc:
            issues.append({"type": "invalid_executable_tool_call", "tool_name": call.tool_name, "call_id": call.call_id, "message": str(exc)})
            continue
        valid.append(call)
    return _deduplicate_calls(valid, issues)


def _build_default_client() -> DeepSeekJSONClient:
    if not settings.deepseek_api_key:
        raise ValueError("TOOL_PLAN_MODE=llm 时必须配置 DEEPSEEK_API_KEY")
    return DeepSeekJSONClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.llm_base_url,
        model_id=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        thinking_enabled=settings.llm_thinking_enabled,
    )


def _build_system_prompt() -> str:
    return (
        "你是旅行规划中的可选工具路由器。请仅依据用户的明确需求、已提供的"
        "上下文摘要，以及本轮传入的工具定义，判断是否需要调用零个、一个或多个工具。"
        "只调用当前工具列表中存在的工具；没有直接必要时，不调用任何工具。"
        "工具调用只能用于补充用户明确需要的、结构化且可验证的信息，不要猜测或"
        "扩展用户未提出的偏好。不要自行构造或修改由系统管理的可信约束字段；"
        "只填写各工具 Schema 明确允许、且需要语义判断的参数。"
        "必须严格遵守工具 Schema，不得输出不存在的工具名、参数或额外字段。"
    )


def _build_user_prompt(state: TravelState, planning_context: dict[str, Any]) -> str:
    payload = {
        "raw_message": state.get("raw_message"),
        "preference_weights": planning_context.get("preference_weights", {}),
        "context_summary": planning_context.get("context_summary", {}),
    }
    return "判断是否需要可选工具查询，并仅在需要时生成原生 Tool Call：\n" + json.dumps(payload, ensure_ascii=False)


def _require_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"planning_context.request.{name} 不能为空")
    return text


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
