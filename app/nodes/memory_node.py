from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable, Mapping

from sqlalchemy.orm import Session

from app.agents.state import TravelState
from app.db.session import SessionLocal
from app.tools.memory_tool import read_user_memory


SessionFactory = Callable[[], Session]


def memory_read_node(
    state: TravelState,
    session_factory: SessionFactory | None = None,
) -> dict[str, Any]:
    """
    MemoryReadNode：读取用户长期偏好，并合并本次请求偏好。

    读取 State：
        - user_id
        - trip_request

    写入 State：
        - user_profile
        - trace
        - errors，数据库读取异常时

    重要边界：
        这个节点只合并“偏好”。
        真正的 hard_constraints 仍然保存在 trip_request 中，
        后续由 PlanningContextNode 整理，不会写入长期记忆。
    """

    node_name = "memory_read"
    started_at = perf_counter()
    factory = session_factory or SessionLocal

    try:
        user_id = state.get("user_id")
        trip_request = state.get("trip_request")

        if trip_request is not None and not isinstance(trip_request, dict):
            raise TypeError("state['trip_request'] 必须是 dict")

        # 1. MemoryTool 只读 SQL，并返回长期偏好 + 当前偏好合并结果。
        with factory() as session:
            user_profile = read_user_memory(
                user_id=user_id,
                session=session,
                trip_request=trip_request,
                history_limit=3,
            )

        trace_item = _build_trace_item(
            node_name=node_name,
            status="success",
            started_at=started_at,
            input_summary=f"user_id={user_id}",
            output_summary=(
                f"source={user_profile.get('source')}, "
                f"profile_found={user_profile.get('profile_found')}, "
                f"history_count={len(user_profile.get('recent_trip_history', []))}"
            ),
        )

        return {
            "user_profile": user_profile,
            "trace": [trace_item],
        }

    except Exception as exc:
        # 2. 记忆系统属于增强能力。
        #    数据库临时失败时，不中断整个旅行规划，改用本次输入偏好继续。
        fallback_profile = _build_fallback_profile(state)

        error_item = {
            "node": node_name,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        trace_item = _build_trace_item(
            node_name=node_name,
            status="failed",
            started_at=started_at,
            input_summary=f"user_id={state.get('user_id')}",
            output_summary=str(exc),
        )

        return {
            "user_profile": fallback_profile,
            "errors": [error_item],
            "trace": [trace_item],
        }


def _build_fallback_profile(state: TravelState) -> dict[str, Any]:
    """
    构造数据库读取失败时的兜底用户画像。

    兜底画像只使用本次 trip_request 中已经抽取出的偏好，
    不伪造长期偏好，也不把硬约束混进用户记忆。
    """

    trip_request = state.get("trip_request")
    trip_request = trip_request if isinstance(trip_request, dict) else {}

    preference_signals = trip_request.get("preference_signals", [])
    spend_preferences = trip_request.get("spend_preferences", [])
    named_constraints = trip_request.get("named_constraints", [])

    # 1. 从 preference_signals 或旧版 list 字段构造当前偏好权重。
    current = _build_current_preferences_from_signals_or_lists(trip_request)

    legacy_travel_style = {
        **current["activity_preferences"],
        **current["pace_preferences"],
    }

    return {
        "user_id": state.get("user_id"),
        "user_found": False,
        "profile_found": False,
        "source": "fallback:memory_read_error",
        "long_term": {},
        "current_request": {
            **current,
            "constraints": trip_request.get("raw_constraints", []),
            "budget": trip_request.get("budget"),
            "preference_signals": (
                preference_signals if isinstance(preference_signals, list) else []
            ),
            "spend_preferences": (
                spend_preferences if isinstance(spend_preferences, list) else []
            ),
            "named_constraints": (
                named_constraints if isinstance(named_constraints, list) else []
            ),
        },
        "effective": {
            "flight_preferences": current["flight_preferences"],
            "hotel_preferences": current["hotel_preferences"],
            "activity_preferences": current["activity_preferences"],
            "pace_preferences": current["pace_preferences"],
            "risk_preferences": current["risk_preferences"],
            "diet_preferences": {},
            "budget_preferences": {
                "current_trip_budget": trip_request.get("budget")
            },
        },
        "recent_trip_history": [],
        "flight_preferences": current["flight_preferences"],
        "hotel_preferences": current["hotel_preferences"],
        "activity_preferences": current["activity_preferences"],
        "pace_preferences": current["pace_preferences"],
        "travel_style": legacy_travel_style,
        "risk_preferences": current["risk_preferences"],
        "diet_preferences": {},
        "budget_preferences": {
            "current_trip_budget": trip_request.get("budget")
        },
        "preference_signals": (
            preference_signals if isinstance(preference_signals, list) else []
        ),
        "spend_preferences": (
            spend_preferences if isinstance(spend_preferences, list) else []
        ),
        "named_constraints": (
            named_constraints if isinstance(named_constraints, list) else []
        ),
        "memory_summary": "记忆读取失败，已仅使用本次输入偏好继续。",
    }


def _build_current_preferences_from_signals_or_lists(
    trip_request: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """
    从 trip_request 构造兜底偏好权重。

    所有主观偏好统一是 0—1 连续权重。
    “一定要安静”只会形成 quiet=1.0，不会变成硬过滤条件。
    """

    result: dict[str, dict[str, float]] = {
        "flight_preferences": {},
        "hotel_preferences": {},
        "activity_preferences": {},
        "pace_preferences": {},
        "risk_preferences": {},
    }

    signals = trip_request.get("preference_signals")

    if isinstance(signals, list) and signals:
        # 1. 新版数据优先读取结构化 PreferenceSignal。
        for signal in signals:
            if not isinstance(signal, Mapping):
                continue

            domain = str(signal.get("domain") or "")
            key = str(signal.get("key") or "")
            strength = _clamp01(_safe_float(signal.get("strength"), 1.0))

            if not domain or not key:
                continue

            bucket = {
                "flight": "flight_preferences",
                "hotel": "hotel_preferences",
                "activity": "activity_preferences",
                "pace": "pace_preferences",
                "risk": "risk_preferences",
            }.get(domain)

            if bucket:
                result[bucket][key] = max(
                    result[bucket].get(key, 0.0),
                    strength,
                )

        return result

    # 2. 兼容旧版 trip_request 的简单 list 字段。
    result["flight_preferences"] = _list_to_score_map(
        trip_request.get("transport_preferences", [])
    )
    result["hotel_preferences"] = _list_to_score_map(
        trip_request.get("hotel_preferences", [])
    )

    for key, value in _list_to_score_map(
        trip_request.get("travel_style", [])
    ).items():
        if key in {"slow", "low_walking_intensity", "packed_schedule"}:
            result["pace_preferences"][key] = value
        else:
            result["activity_preferences"][key] = value

    return result


def _list_to_score_map(items: Any) -> dict[str, float]:
    """把旧版偏好字符串列表转成权重 1.0 的 dict。"""

    if not isinstance(items, list):
        return {}

    return {
        str(item): 1.0
        for item in items
        if isinstance(item, str) and item.strip()
    }


def _safe_float(value: Any, default: float) -> float:
    """安全转换 float。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    """把偏好强度限制在 0 到 1。"""

    return max(0.0, min(1.0, value))


def _build_trace_item(
    node_name: str,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """生成统一 Trace 记录。"""

    latency_ms = int((perf_counter() - started_at) * 1000)

    return {
        "node_name": node_name,
        "tool_name": "memory_tool",
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "latency_ms": latency_ms,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
