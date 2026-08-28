from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from app.agents.state import TravelState
from app.common.config import settings
from app.mcp.tool_runtime import get_mcp_tool_runtime
from app.observability.harness import ToolExecutor
from app.schemas.tool_orchestration_schema import ToolCallResult

META_FIELDS = {"get_weather": "weather_fetch_meta", "search_flights": "flight_fetch_meta", "search_hotels": "hotel_fetch_meta", "search_city_activities": "activity_fetch_meta"}


def tool_execute_node(state: TravelState, client: Any | None = None) -> dict[str, Any]:
    """通用串行 Dispatcher；计划 arguments 是实际调用参数的唯一来源。"""
    started_at = perf_counter()
    plan = state.get("tool_plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("calls"), list):
        raise ValueError("state['tool_plan'] 缺失或无效")
    runtime = get_mcp_tool_runtime()
    update: dict[str, Any] = {"trace": [], "errors": []}
    results: list[dict[str, Any]] = []
    calls = [item for item in plan["calls"] if isinstance(item, dict)]
    for call in calls:
        name = str(call.get("tool_name") or "")
        arguments = dict(call.get("arguments") or {})
        descriptor = runtime.tool_registry.resolve(name)
        if call.get("server_id") != descriptor.server_id:
            exc = ValueError("ToolPlan server_id 与 Global Tool Registry 不一致")
            results.append(_result(call, descriptor, arguments, "failed", exc=exc))
            _merge(update, _adapt_failure(name, descriptor, arguments, exc, client))
            continue
        meta_field = META_FIELDS.get(name)
        if meta_field and _is_cached(state, meta_field, arguments):
            results.append(_result(call, descriptor, arguments, "cached"))
            continue
        call_started = perf_counter()
        try:
            raw = ToolExecutor(timeout_seconds=settings.mcp_travel_data_timeout_seconds).execute(
                name=name, operation_type="mcp" if client is None else "tool",
                input_value=arguments, model_or_tool=f"{descriptor.server_id}:{name}",
                callback=lambda n=name, a=arguments, d=descriptor: _call(client, runtime.connection_manager, d.server_id, n, a),
            )
            results.append(_result(call, descriptor, arguments, "success", data=raw, started=call_started, client=client))
            _merge(update, _adapt_success(name, raw, descriptor, arguments, client))
        except Exception as exc:
            results.append(_result(call, descriptor, arguments, "failed", exc=exc, started=call_started, client=client))
            _merge(update, _adapt_failure(name, descriptor, arguments, exc, client))

    activity_requested = any(call.get("tool_name") == "search_city_activities" for call in calls)
    if not activity_requested:
        update.update({"activity_result": {"status": "skipped", "items": [], "source": "policy:not_requested"}, "activity_candidates": [], "activity_fetch_meta": {"tool_name": "search_city_activities", "status": "skipped"}})
    context = dict(state.get("planning_context") or {})
    context["activity_candidates"] = list(update.get("activity_candidates", state.get("activity_candidates", [])) or [])
    update["planning_context"] = context
    update["tool_call_results"] = results
    update["tool_execution_result"] = {"status": "completed", "planned_call_count": len(calls), "executed_call_count": sum(r["status"] != "cached" for r in results), "retry_count": _safe_int(state.get("tool_execute_retry_count"))}
    update["trace"].append({"node_name": "tool_execute", "tool_name": "generic_multi_mcp_dispatcher", "input_summary": f"planned_calls={len(calls)}", "output_summary": f"results={len(results)}, activity_requested={activity_requested}", "status": "success", "latency_ms": int((perf_counter() - started_at) * 1000), "created_at": datetime.now(timezone.utc).isoformat()})
    if not update["errors"]:
        update.pop("errors")
    return update


def _call(client: Any, manager: Any, server_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if client is None:
        return manager.call_tool(server_id=server_id, tool_name=name, arguments=arguments)
    if callable(getattr(client, "call_tool", None)):
        return client.call_tool(server_id=server_id, tool_name=name, arguments=arguments)
    return getattr(client, name)(**arguments)


def _info(client: Any, descriptor: Any) -> dict[str, str]:
    return {"client_mode": str(getattr(client, "client_mode", "mcp")), "protocol": str(getattr(client, "protocol", "mcp")), "transport": str(getattr(client, "transport", descriptor.transport)), "server_id": descriptor.server_id}


def _adapt_success(name: str, raw: dict[str, Any], descriptor: Any, arguments: dict[str, Any], client: Any) -> dict[str, Any]:
    meta = {"tool_name": name, "status": raw.get("status"), "source": raw.get("source"), **_info(client, descriptor), "query_signature": arguments}
    if name == "get_weather": return {"weather_result": raw, "weather_fetch_meta": meta}
    if name == "search_flights": return {"raw_flight_results": raw, "flight_fetch_meta": meta}
    if name == "search_hotels": return {"raw_hotel_results": raw, "hotel_fetch_meta": meta}
    if name == "search_city_activities": return {"activity_result": raw, "activity_candidates": list(raw.get("items", [])), "activity_fetch_meta": meta}
    return {}


def _adapt_failure(name: str, descriptor: Any, arguments: dict[str, Any], exc: Exception, client: Any) -> dict[str, Any]:
    meta = {"tool_name": name, "status": "failed", **_info(client, descriptor), "error": str(exc), "query_signature": arguments}
    error = {"node": "tool_execute", "type": exc.__class__.__name__, "message": str(exc), "created_at": datetime.now(timezone.utc).isoformat()}
    if name == "get_weather": return {"weather_result": {"status": "unavailable", "daily": [], "risks": [], "warnings": ["天气查询失败。"]}, "weather_fetch_meta": meta, "errors": [error]}
    if name == "search_flights": return {"raw_flight_results": {"status": "failed", "outbound": [], "return": [], "error": str(exc)}, "flight_fetch_meta": meta, "errors": [error]}
    if name == "search_hotels": return {"raw_hotel_results": {"status": "failed", "items": [], "error": str(exc)}, "hotel_fetch_meta": meta, "errors": [error]}
    return {"activity_result": {"status": "failed", "items": [], "error": str(exc)}, "activity_candidates": [], "activity_fetch_meta": meta, "errors": [error]}


def _result(call: dict[str, Any], descriptor: Any, arguments: dict[str, Any], status: str, *, data: Any = None, exc: Exception | None = None, started: float | None = None, client: Any = None) -> dict[str, Any]:
    return ToolCallResult(call_id=str(call.get("call_id") or "unknown"), requirement_key=str(call.get("requirement_key") or "unknown"), tool_name=str(call.get("tool_name") or ""), server_id=descriptor.server_id, source=call.get("source", "llm_selected"), status=status, arguments=arguments, data=data, latency_ms=int((perf_counter() - started) * 1000) if started else 0, transport=_info(client, descriptor)["transport"], error=({"type": exc.__class__.__name__, "message": str(exc)} if exc else None)).model_dump(mode="json")


def _is_cached(state: TravelState, field: str, arguments: dict[str, Any]) -> bool:
    meta = state.get(field)
    return isinstance(meta, dict) and meta.get("status") not in {None, "failed"} and meta.get("query_signature") == arguments


def _merge(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if key in {"trace", "errors"}: target.setdefault(key, []).extend(value if isinstance(value, list) else [])
        else: target[key] = value


def _safe_int(value: Any) -> int:
    try: return int(value)
    except (TypeError, ValueError): return 0
