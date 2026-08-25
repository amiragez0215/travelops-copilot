from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal

from app.agents.state import TravelState
from app.guards.safety_guard import evaluate_safety_request


def safety_check_node(state: TravelState) -> dict[str, Any]:
    """
    SafetyCheckNode 对应的方法。

    所在层：
        app/nodes

    解决什么问题：
        检查用户请求是否要求系统执行越界动作。

    当前项目明确不做：
        - 真实支付
        - 真实预订
        - 真实邮件或短信
        - 购物车或商品购买
        - 违法或绕过安全机制的请求

    读取 State：
        - state["raw_message"]
        - state["trip_request"]

    写入 State：
        - state["safety_result"]
        - state["safety_flags"]
        - state["trace"]
        - state["errors"]，异常时写入

    不调用：
        - LLM
        - SQL
        - mock 外部数据
        - RAG
        - 外部 API

    为什么不直接在 Node 里写规则？
        Node 只负责读写 State 和记录 trace。
        具体安全规则放在 app/guards/safety_guard.py，
        这样可以独立测试，也便于后续升级为规则 + LLM classifier。
    """

    node_name = "safety_check"
    started_at = perf_counter()

    try:
        raw_message = state.get("raw_message") or ""
        raw_message = str(raw_message).strip()

        trip_request = state.get("trip_request")

        if not isinstance(trip_request, dict):
            raise ValueError("state 中缺少有效 trip_request，请先运行 InputExtractNode 和 MissingInfoCheckNode")

        # 如果 raw_message 为空，尝试从 trip_request 中取 raw_message。
        # 这不是兼容旧 State 字段，而是为了让 SafetyCheckNode 可以被单独测试。
        if not raw_message:
            raw_message = str(trip_request.get("raw_message") or "").strip()

        if not raw_message:
            raise ValueError("state 中缺少 raw_message，无法进行安全检查")

        safety_result = evaluate_safety_request(
            raw_message=raw_message,
            trip_request=trip_request,
        )

        safety_flags = safety_result["flags"]

        trace_item = _build_trace_item(
            node_name=node_name,
            status="success",
            started_at=started_at,
            input_summary=_shorten(raw_message),
            output_summary=(
                f"blocked={safety_result['blocked']}, "
                f"flag_count={len(safety_flags)}, "
                f"safe_mode={safety_result['safe_mode']}"
            ),
        )

        return {
            "safety_result": safety_result,
            "safety_flags": safety_flags,
            "trace": [trace_item],
        }

    except Exception as exc:
        # 安全检查异常时，默认不继续主规划流程。
        # 这比在安全状态不明时继续查天气、航班、酒店更安全。
        fallback_flag = {
            "rule_id": "safety_check_error",
            "category": "system_error",
            "level": "high",
            "action": "block",
            "message": "安全检查执行失败，系统已停止继续规划。",
            "suggestion": "请重新提交旅行需求，或联系开发者检查日志。",
            "matched_keyword": None,
            "evidence": "",
        }

        safety_result = {
            "blocked": True,
            "reason": "安全检查执行失败，当前请求不会继续进入旅行规划。",
            "safe_mode": "reject_or_draft_only",
            "flags": [fallback_flag],
            "allowed_actions": ["generate_safe_explanation"],
            "blocked_actions": ["system_error"],
            "project_boundary": {
                "no_real_payment": True,
                "no_real_booking": True,
                "no_real_email": True,
                "no_shopping_cart": True,
                "draft_only": True,
            },
        }

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
            input_summary="safety check failed",
            output_summary=str(exc),
        )

        return {
            "safety_result": safety_result,
            "safety_flags": [fallback_flag],
            "errors": [error_item],
            "trace": [trace_item],
        }


def route_after_safety(
    state: TravelState,
) -> Literal["safe_reject", "memory_read"]:
    """
    SafetyCheckNode 后面的 Conditional Edge 路由函数。

    对应 workflow：

        route_after_safety
        ├── blocked=True  → SafeRejectNode
        └── blocked=False → MemoryReadNode

    注意：
        这个函数不是 Node。
        它不写 State，不记录 trace。
        它只读取 state["safety_result"] 并返回路由标签。

    为什么缺 safety_result 时默认 safe_reject？
        因为如果安全状态未知，不应该继续执行后面的工具查询和规划。
    """

    safety_result = state.get("safety_result")

    if not isinstance(safety_result, dict):
        return "safe_reject"

    if safety_result.get("blocked") is True:
        return "safe_reject"

    return "memory_read"


def _build_trace_item(
    node_name: str,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """
    构造统一 trace item。

    后续我们可以把它抽成 TraceTool。
    现在先保留在节点内部，等多个节点稳定后再统一重构。
    """

    latency_ms = int((perf_counter() - started_at) * 1000)

    return {
        "node_name": node_name,
        "tool_name": None,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "latency_ms": latency_ms,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _shorten(text: str, max_len: int = 100) -> str:
    """
    截断过长文本，避免 trace 太大。
    """

    if len(text) <= max_len:
        return text

    return text[: max_len - 3] + "..."