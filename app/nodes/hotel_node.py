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


def hotel_search_node(
    state: TravelState,
    client: TravelDataClient | None = None,
) -> dict[str, Any]:
    """
    HotelSearchNode：查询目标城市的酒店原始候选。

    读取 State：
        - planning_context.request.destination
        - planning_context.request.nights
        - planning_context.request.people_count

    写入 State：
        - raw_hotel_results
        - hotel_fetch_meta
        - trace
        - errors，异常时写入

    正式模式下的调用链：

        HotelSearchNode
            -> create_travel_data_client()
            -> MCPTravelDataClient
            -> stdio MCP Server
            -> search_hotels Tool
            -> Hotel Mock Provider

    这里不根据安静、清洁、地铁或预算软目标排序，
    CandidateRankNode 会在拿到全部有效候选后统一处理这些偏好。
    """

    node_name = "hotel_search"
    started_at = perf_counter()

    active_client = client
    client_info: dict[str, Any] = {
        "client_mode": "unknown",
        "protocol": "unknown",
        "transport": "unknown",
    }

    try:
        # 1. 没有测试注入时，使用统一 Client 工厂。
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

        destination = _require_text(
            request.get("destination"),
            "destination",
        )
        nights = int(
            request.get("nights")
            or 0
        )
        people_count = int(
            request.get("people_count")
            or 1
        )

        # 2. ToolExecutor 让酒店查询也具有与天气、航班一致的超时、
        #    异常映射和子 Span。输入只会被哈希，不会被记录为 Trace 正文。
        raw_hotel_results = ToolExecutor(
            timeout_seconds=settings.mcp_travel_data_timeout_seconds,
        ).execute(
            name="search_hotels",
            operation_type=(
                "mcp"
                if client_info["protocol"] == "mcp"
                else "tool"
            ),
            input_value={
                "city": destination,
                "nights": nights,
                "people_count": people_count,
            },
            model_or_tool=(
                f"{client_info['protocol']}:search_hotels"
            ),
            callback=lambda: active_client.search_hotels(
                city=destination,
                nights=nights,
                people_count=people_count,
            ),
        )

        fetch_meta = {
            "tool_name": "search_hotels",
            "status": raw_hotel_results.get(
                "status"
            ),
            "source": raw_hotel_results.get(
                "source"
            ),
            **client_info,
            "query_signature": {
                "city": destination,
                "nights": nights,
                "people_count": people_count,
            },
        }

        hotel_count = len(
            raw_hotel_results.get(
                "items",
                [],
            )
        )

        trace_item = _build_trace_item(
            node_name=node_name,
            tool_name=(
                f"{client_info['protocol']}:"
                "search_hotels"
            ),
            status="success",
            started_at=started_at,
            input_summary=(
                f"city={destination}, "
                f"nights={nights}, "
                f"people={people_count}"
            ),
            output_summary=(
                f"client_mode="
                f"{client_info['client_mode']}, "
                f"transport="
                f"{client_info['transport']}, "
                f"hotels={hotel_count}, "
                f"status="
                f"{raw_hotel_results.get('status')}"
            ),
        )

        return {
            "raw_hotel_results": (
                raw_hotel_results
            ),
            "hotel_fetch_meta": fetch_meta,
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

        raw_hotel_results = {
            "status": "failed",
            "items": [],
            "source": (
                f"{client_info['protocol']}:"
                "search_hotels"
            ),
            "error": str(exc),
        }

        trace_item = _build_trace_item(
            node_name=node_name,
            tool_name=(
                f"{client_info['protocol']}:"
                "search_hotels"
            ),
            status="failed",
            started_at=started_at,
            input_summary=(
                "hotel query failed"
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
            "raw_hotel_results": (
                raw_hotel_results
            ),
            "hotel_fetch_meta": {
                "tool_name": "search_hotels",
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
