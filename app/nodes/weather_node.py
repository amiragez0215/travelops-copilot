from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from app.agents.state import TravelState
from app.common.config import settings
from app.mcp.client_factory import (
    create_travel_data_client,
    describe_travel_data_client,
)
from app.mcp.travel_data_client import (
    TravelDataClient,
)
from app.observability.harness import ToolExecutor


def weather_node(
    state: TravelState,
    client: TravelDataClient | None = None,
) -> dict[str, Any]:
    """
    WeatherNode：查询目的地在旅行日期范围内的天气。

    读取 State：
        - planning_context.request.destination
        - planning_context.request.start_date
        - planning_context.request.end_date

    写入 State：
        - weather_result
        - weather_fetch_meta
        - trace
        - errors，查询异常时写入

    Client 选择规则：

        1. 调用方显式传入 client：
            使用该 Client。
            单元测试通常传 LocalTravelDataClient。

        2. 没有传入 client：
            调用 create_travel_data_client()。
            正式 .env 配置 TRAVEL_DATA_CLIENT_MODE=mcp 后，
            调用链会真实经过：

                WeatherNode
                    -> MCPTravelDataClient
                    -> MCP stdio Server
                    -> get_weather Tool
                    -> Weather Mock Provider

    注意：
        这个节点只负责获取结构化天气事实。
        它不生成旅行文案，也不决定具体活动。
    """

    node_name = "weather"
    started_at = perf_counter()

    # 先保留调用方注入的 Client。
    # 如果没有注入，正式运行时再由工厂根据 .env 创建。
    active_client = client

    # 默认值用于 Client 创建本身失败时的错误 Trace。
    client_info: dict[str, Any] = {
        "client_mode": "unknown",
        "protocol": "unknown",
        "transport": "unknown",
    }

    try:
        # 1. 这是阶段二的核心改动：
        #    原代码默认 LocalTravelDataClient()，正式 Graph 会绕过 MCP。
        #    现在统一通过工厂读取 TRAVEL_DATA_CLIENT_MODE。
        active_client = (
            active_client
            or create_travel_data_client()
        )

        # 2. 记录实际 Client 类型。
        #    这让 Trace 能证明本次调用究竟是 local 还是 mcp。
        client_info = (
            describe_travel_data_client(
                active_client
            )
        )

        planning_context = (
            _get_planning_context(state)
        )
        request = planning_context["request"]

        destination = _require_text(
            request.get("destination"),
            "destination",
        )
        start_date = _require_text(
            request.get("start_date"),
            "start_date",
        )
        end_date = (
            request.get("end_date")
            or start_date
        )

        # 3. 阶段五统一由 ToolExecutor 执行外部调用。
        #    它会在当前 weather Node Span 下面创建 Tool/MCP 子 Span，
        #    并统一超时和错误代码；它不替代真实 MCP Client 的协议实现。
        weather_result = ToolExecutor(
            timeout_seconds=settings.mcp_travel_data_timeout_seconds,
        ).execute(
            name="get_weather",
            operation_type=(
                "mcp"
                if client_info["protocol"] == "mcp"
                else "tool"
            ),
            input_value={
                "city": destination,
                "start_date": start_date,
                "end_date": end_date,
            },
            model_or_tool=(
                f"{client_info['protocol']}:get_weather"
            ),
            callback=lambda: active_client.get_weather(
                city=destination,
                start_date=start_date,
                end_date=end_date,
            ),
        )

        # 4. fetch_meta 保存查询签名和协议来源，
        #    便于后续 Trace、Eval 和问题排查。
        fetch_meta = {
            "tool_name": "get_weather",
            "status": weather_result.get(
                "status"
            ),
            "source": weather_result.get(
                "source"
            ),
            **client_info,
            "query_signature": {
                "city": destination,
                "start_date": start_date,
                "end_date": end_date,
            },
        }

        trace_item = _build_trace_item(
            node_name=node_name,
            tool_name=(
                f"{client_info['protocol']}:"
                "get_weather"
            ),
            status="success",
            started_at=started_at,
            input_summary=(
                f"city={destination}, "
                f"start={start_date}, "
                f"end={end_date}"
            ),
            output_summary=(
                f"client_mode="
                f"{client_info['client_mode']}, "
                f"transport="
                f"{client_info['transport']}, "
                f"status="
                f"{weather_result.get('status')}, "
                f"risks="
                f"{len(weather_result.get('risks', []))}"
            ),
        )

        return {
            "weather_result": weather_result,
            "weather_fetch_meta": fetch_meta,
            "trace": [trace_item],
        }

    except Exception as exc:
        # 5. MCP Server 启动失败、超时、Tool 错误或 State 缺失
        #    都会进入统一失败分支。
        error_item = {
            "node": node_name,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        weather_result = {
            "status": "unavailable",
            "summary": (
                "天气查询失败，后续方案需要提示用户"
                "天气数据不可用。"
            ),
            "daily": [],
            "risks": [],
            "warnings": [
                "天气查询失败。"
            ],
            "source": (
                f"{client_info['protocol']}:"
                "get_weather"
            ),
        }

        trace_item = _build_trace_item(
            node_name=node_name,
            tool_name=(
                f"{client_info['protocol']}:"
                "get_weather"
            ),
            status="failed",
            started_at=started_at,
            input_summary=(
                "weather query failed"
            ),
            output_summary=(
                f"client_mode="
                f"{client_info['client_mode']}, "
                f"transport="
                f"{client_info['transport']}, "
                f"error={exc}"
            ),
        )

        return {
            "weather_result": weather_result,
            "weather_fetch_meta": {
                "tool_name": "get_weather",
                "status": "failed",
                **client_info,
                "error": str(exc),
            },
            "errors": [error_item],
            "trace": [trace_item],
        }


def _get_planning_context(
    state: TravelState,
) -> dict[str, Any]:
    """从 State 中读取 PlanningContextNode 已生成的统一规划上下文。"""

    planning_context = state.get(
        "planning_context"
    )

    if not isinstance(
        planning_context,
        dict,
    ):
        raise ValueError(
            "state['planning_context'] 缺失或不是 dict，"
            "请先运行 PlanningContextNode"
        )

    return planning_context


def _require_text(
    value: Any,
    field_name: str,
) -> str:
    """读取必填文本字段，并统一去除首尾空白。"""

    if value is None or not str(
        value
    ).strip():
        raise ValueError(
            "planning_context.request."
            f"{field_name} 不能为空"
        )

    return str(value).strip()


def _build_trace_item(
    node_name: str,
    tool_name: str,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """生成 Node 级 Trace；tool_name 会真实反映 local 或 mcp。"""

    latency_ms = int(
        (
            perf_counter()
            - started_at
        )
        * 1000
    )

    return {
        "node_name": node_name,
        "tool_name": tool_name,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "latency_ms": latency_ms,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }
