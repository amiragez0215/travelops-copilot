from __future__ import annotations

import argparse
import json
import uuid

from langgraph.types import Command

from app.workflows.checkpoint_runtime import (
    initialize_checkpointer,
    reset_checkpointer_runtime,
)
from app.workflows.decision_gate_demo_workflow import (
    build_decision_gate_demo_graph,
    build_demo_decision_state,
)


def main() -> None:
    """
    运行 DecisionGate 的最小 Human-in-the-loop 闭环。

    示例：
        python scripts/demo_decision_gate_hitl.py --decision approve

        python scripts/demo_decision_gate_hitl.py ^
            --decision request_changes ^
            --change-request "酒店换成绿水青山酒店，预算增加1000"

        python scripts/demo_decision_gate_hitl.py --decision cancel
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--decision",
        choices=[
            "approve",
            "request_changes",
            "cancel",
        ],
        default="approve",
    )

    parser.add_argument(
        "--change-request",
        default=None,
    )

    parser.add_argument(
        "--thread-id",
        default=(
            "decision_demo_"
            + uuid.uuid4().hex[:12]
        ),
    )

    args = parser.parse_args()

    if (
        args.decision
        == "request_changes"
        and not (
            args.change_request
            and args.change_request.strip()
        )
    ):
        parser.error(
            "request_changes 必须提供 --change-request"
        )

    checkpointer = initialize_checkpointer()

    graph = build_decision_gate_demo_graph(
        checkpointer=checkpointer
    )

    config = {
        "configurable": {
            "thread_id": args.thread_id,
        }
    }

    try:
        # 1. 首次运行到 DecisionGate，Graph 在 interrupt() 处暂停。
        paused = graph.invoke(
            build_demo_decision_state(),
            config=config,
        )

        interrupt_item = paused[
            "__interrupt__"
        ][0]

        interrupt_payload = getattr(
            interrupt_item,
            "value",
            interrupt_item,
        )

        print("\n=== Graph Paused ===")
        print(
            json.dumps(
                interrupt_payload,
                ensure_ascii=False,
                indent=2,
            )
        )

        # 2. 模拟前端把用户决定通过 Command(resume=...) 提交回来。
        resume_payload = {
            "decision": args.decision,
            "proposal_id": (
                interrupt_payload[
                    "proposal_id"
                ]
            ),
            "action_id": (
                interrupt_payload[
                    "action_id"
                ]
            ),
            "change_request": (
                args.change_request
                if args.decision
                == "request_changes"
                else None
            ),
            "client_request_id": (
                "cli_" + uuid.uuid4().hex
            ),
        }

        # 3. 恢复时必须使用完全相同的 thread_id。
        final_state = graph.invoke(
            Command(
                resume=resume_payload
            ),
            config=config,
        )

        print("\n=== Graph Resumed ===")
        print(
            json.dumps(
                {
                    "thread_id": args.thread_id,
                    "decision_result": (
                        final_state.get(
                            "decision_result"
                        )
                    ),
                    "change_request": (
                        final_state.get(
                            "change_request"
                        )
                    ),
                    "final_response": (
                        final_state.get(
                            "final_response"
                        )
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    finally:
        # 4. 关闭连接，但磁盘中的 Checkpoint 会继续保留。
        reset_checkpointer_runtime()


if __name__ == "__main__":
    main()
