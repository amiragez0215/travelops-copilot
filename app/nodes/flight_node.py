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


def flight_search_node(
    state: TravelState,
    client: TravelDataClient | None = None,
) -> dict[str, Any]:
    """
    FlightSearchNode：查询去程和返程航班原始候选。

    读取 State：
        - planning_context.request.origin
        - planning_context.request.destination
        - planning_context.request.start_date
        - planning_context.request.end_date
        - planning_context.request.people_count

    写入 State：
        - raw_flight_results
        - flight_fetch_meta
        - trace
        - errors，异常时写入

    Client 选择：

        - 测试显式传入 LocalTravelDataClient；
        - 正式 Graph 不传 client，统一通过工厂读取
          TRAVEL_DATA_CLIENT_MODE=mcp。

    这个节点只负责获取结构化航班候选。
    它不处理早班机偏好、价格适配和最终组合选择，
    这些分别由 CandidateRankNode 和 BudgetOptimizeNode 完成。
    """

    node_name = "flight_search"
    started_at = perf_counter()

    active_client = client
    client_info: dict[str, Any] = {
        "client_mode": "unknown",
        "protocol": "unknown",
        "transport": "unknown",
    }

    try:
        # 1. 正式运行不再固定创建 Local Client，
        #    而是通过配置工厂决定 local / mcp。
        active_client = (
            active_client
            or create_travel_data_client()
        )
        client_info = (
            describe_travel_data_client(
                active_client
            )
        )

        planning_context = (
            _get_planning_context(state)
        )
        request = planning_context["request"]

        origin = _require_text(
            request.get("origin"),
            "origin",
        )
        destination = _require_text(
            request.get("destination"),
            "destination",
        )
        start_date = _require_text(
            request.get("start_date"),
            "start_date",
        )
        end_date = request.get(
            "end_date"
        )
        people_count = int(
            request.get("people_count")
            or 1
        )

        # 2. ToolExecutor 将 Node 内的外部事实查询收敛到同一超时、
        #    错误映射和 Trace 边界。MCP 仍由 active_client 完成，
        #    所以不会把协议责任错误地搬到 Agent Harness。
        raw_flight_results = ToolExecutor(
            timeout_seconds=settings.mcp_travel_data_timeout_seconds,
        ).execute(
            name="search_flights",
            operation_type=(
                "mcp"
                if client_info["protocol"] == "mcp"
                else "tool"
            ),
            input_value={
                "origin": origin,
                "destination": destination,
                "depart_date": start_date,
                "return_date": end_date,
                "people_count": people_count,
            },
            model_or_tool=(
                f"{client_info['protocol']}:search_flights"
            ),
            callback=lambda: active_client.search_flights(
                origin=origin,
                destination=destination,
                depart_date=start_date,
                return_date=end_date,
                people_count=people_count,
            ),
        )

        fetch_meta = {
            "tool_name": "search_flights",
            "status": raw_flight_results.get(
                "status"
            ),
            "source": raw_flight_results.get(
                "source"
            ),
            **client_info,
            "query_signature": {
                "origin": origin,
                "destination": destination,
                "depart_date": start_date,
                "return_date": end_date,
                "people_count": people_count,
            },
        }

        outbound_count = len(
            raw_flight_results.get(
                "outbound",
                [],
            )
        )
        return_count = len(
            raw_flight_results.get(
                "return",
                [],
            )
        )

        trace_item = _build_trace_item(
            node_name=node_name,
            tool_name=(
                f"{client_info['protocol']}:"
                "search_flights"
            ),
            status="success",
            started_at=started_at,
            input_summary=(
                f"origin={origin}, "
                f"destination={destination}, "
                f"depart={start_date}, "
                f"return={end_date}, "
                f"people={people_count}"
            ),
            output_summary=(
                f"client_mode="
                f"{client_info['client_mode']}, "
                f"transport="
                f"{client_info['transport']}, "
                f"outbound={outbound_count}, "
                f"return={return_count}, "
                f"status="
                f"{raw_flight_results.get('status')}"
            ),
        )

        return {
            "raw_flight_results": (
                raw_flight_results
            ),
            "flight_fetch_meta": (
                fetch_meta
            ),
            "trace": [trace_item],
        }

    except Exception as exc:
        error_item = {
            "node": node_name,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        raw_flight_results = {
            "status": "failed",
            "outbound": [],
            "return": [],
            "source": (
                f"{client_info['protocol']}:"
                "search_flights"
            ),
            "error": str(exc),
        }

        trace_item = _build_trace_item(
            node_name=node_name,
            tool_name=(
                f"{client_info['protocol']}:"
                "search_flights"
            ),
            status="failed",
            started_at=started_at,
            input_summary=(
                "flight query failed"
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
            "raw_flight_results": (
                raw_flight_results
            ),
            "flight_fetch_meta": {
                "tool_name": "search_flights",
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
    """从 State 中读取 PlanningContextNode 的输出。"""

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
    """读取并标准化必填文本字段。"""

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
    """生成带真实 Client 协议信息的 Node Trace。"""

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
