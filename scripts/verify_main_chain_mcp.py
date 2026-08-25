from __future__ import annotations

import argparse
import json
from typing import Any
from uuid import uuid4

import httpx


EXPECTED_MCP_NODES = {
    "weather": "mcp:get_weather",
    "flight_search": "mcp:search_flights",
    "hotel_search": "mcp:search_hotels",
}


def _print_json(
    title: str,
    value: Any,
) -> None:
    """以易读 JSON 打印验证结果。"""

    print(f"\n=== {title} ===")
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    """
    验证正式 PlanTrip 主链是否真实经过 MCP Client -> MCP Server。

    使用前先启动 FastAPI：

        uvicorn app.main:app --reload

    这个脚本不是单元测试替代品，而是阶段二最重要的集成验证：

        1. 通过正式 /invoke API 启动完整 Graph；
        2. 等 Graph 运行到 DecisionGate；
        3. 通过 state API 读取公开 Trace；
        4. 检查 Weather / Flight / Hotel 三个节点的 tool_name；
        5. 检查 output_summary 中是否记录 client_mode=mcp、transport=stdio。

    如果三个节点仍然使用 LocalTravelDataClient，脚本会直接失败。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
    )
    args = parser.parse_args()

    thread_id = (
        args.thread_id
        or f"mcp_verify_{uuid4().hex[:12]}"
    )

    with httpx.Client(
        base_url=args.base_url,
        timeout=300.0,
    ) as client:
        # 1. 启动正式完整 Graph。
        invoke_response = client.post(
            "/api/v1/travel/invoke",
            json={
                "user_id": "user_001",
                "message": (
                    "2026年7月2日从杭州去成都玩三天，"
                    "预算4000，想住安静一点。"
                ),
                "reference_date": "2026-07-01",
                "trip_session_id": thread_id,
            },
        )
        invoke_response.raise_for_status()
        invoke_body = invoke_response.json()
        _print_json(
            "Invoke Response",
            invoke_body,
        )

        if (
            invoke_body.get("run_status")
            != "waiting_for_decision"
        ):
            raise RuntimeError(
                "完整 Graph 没有到达 DecisionGate；"
                "请先检查 final_response 和服务端日志。"
            )

        # 2. 读取包含公开 Trace 的最新 Checkpoint 状态。
        state_response = client.get(
            f"/api/v1/travel/runs/{thread_id}",
            params={
                "include_trace": "true"
            },
        )
        state_response.raise_for_status()
        state_body = state_response.json()

        trace_items = state_body.get(
            "trace"
        ) or []

        # 3. 按 node_name 建立索引，找到三个正式数据节点。
        trace_by_node = {
            str(item.get("node_name")):
                item
            for item in trace_items
            if isinstance(item, dict)
        }

        checks: dict[str, Any] = {}

        for node_name, expected_tool in (
            EXPECTED_MCP_NODES.items()
        ):
            trace = trace_by_node.get(
                node_name
            )

            if trace is None:
                raise RuntimeError(
                    f"公开 Trace 中缺少节点 {node_name!r}。"
                )

            tool_name = trace.get(
                "tool_name"
            )
            output_summary = str(
                trace.get("output_summary")
                or ""
            )

            if tool_name != expected_tool:
                raise RuntimeError(
                    f"节点 {node_name!r} 没有使用预期 MCP Tool："
                    f"expected={expected_tool!r}, actual={tool_name!r}"
                )

            if "client_mode=mcp" not in output_summary:
                raise RuntimeError(
                    f"节点 {node_name!r} Trace 没有记录 client_mode=mcp。"
                )

            if "transport=stdio" not in output_summary:
                raise RuntimeError(
                    f"节点 {node_name!r} Trace 没有记录 transport=stdio。"
                )

            checks[node_name] = {
                "tool_name": tool_name,
                "output_summary": output_summary,
                "latency_ms": trace.get(
                    "latency_ms"
                ),
            }

        _print_json(
            "MCP Main-chain Verification",
            checks,
        )

        print(
            "\n[OK] WeatherNode、FlightSearchNode、HotelSearchNode "
            "均真实通过 MCP stdio Client -> Server 调用。"
        )


if __name__ == "__main__":
    main()
