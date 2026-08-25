from __future__ import annotations

import sqlite3

from langgraph.types import Command

try:
    from langgraph.checkpoint.memory import (
        InMemorySaver,
    )
except ImportError:  # pragma: no cover - 兼容较早 LangGraph
    from langgraph.checkpoint.memory import (
        MemorySaver as InMemorySaver,
    )

from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)

from app.workflows.decision_gate_demo_workflow import (
    build_decision_gate_demo_graph,
    build_demo_decision_state,
)


def _interrupt_payload(result):
    """读取 graph.invoke() 返回的第一个 Interrupt Payload。"""

    item = result["__interrupt__"][0]

    return getattr(
        item,
        "value",
        item,
    )


def _compile_memory_graph():
    """创建测试用内存 Checkpointer Graph。"""

    return build_decision_gate_demo_graph(
        checkpointer=InMemorySaver()
    )


def test_approve_interrupt_resume_closed_loop():
    """approve 应暂停、恢复并路由到 commit_draft Demo。"""

    graph = _compile_memory_graph()

    config = {
        "configurable": {
            "thread_id": "approve-1"
        }
    }

    paused = graph.invoke(
        build_demo_decision_state(),
        config=config,
    )

    payload = _interrupt_payload(
        paused
    )

    assert payload["proposal_id"] == (
        "proposal_demo_001"
    )

    final_state = graph.invoke(
        Command(
            resume={
                "decision": "approve",
                "proposal_id": payload[
                    "proposal_id"
                ],
                "action_id": payload[
                    "action_id"
                ],
            }
        ),
        config=config,
    )

    assert final_state["decision"] == (
        "approve"
    )
    assert (
        final_state[
            "decision_result"
        ]["next_action"]
        == "commit_draft"
    )
    assert (
        final_state[
            "itinerary_status"
        ]
        == "approved_simulated"
    )
    assert (
        final_state[
            "final_response"
        ]["status"]
        == "approved_simulated"
    )
    assert len(
        final_state[
            "decision_history"
        ]
    ) == 1


def test_request_changes_interrupt_resume_closed_loop():
    """request_changes 应保存原始修改文本并路由 RevisionAnalyze。"""

    graph = _compile_memory_graph()
    config = {
        "configurable": {
            "thread_id": "changes-1"
        }
    }

    paused = graph.invoke(
        build_demo_decision_state(),
        config=config,
    )

    payload = _interrupt_payload(
        paused
    )

    final_state = graph.invoke(
        Command(
            resume={
                "decision": (
                    "request_changes"
                ),
                "proposal_id": payload[
                    "proposal_id"
                ],
                "action_id": payload[
                    "action_id"
                ],
                "change_request": (
                    "酒店换成绿水青山酒店，"
                    "预算增加1000。"
                ),
            }
        ),
        config=config,
    )

    assert final_state["decision"] == (
        "request_changes"
    )
    assert (
        final_state[
            "itinerary_status"
        ]
        == "revision_requested"
    )
    assert (
        "绿水青山酒店"
        in final_state[
            "change_request"
        ]["raw_message"]
    )
    assert (
        final_state[
            "final_response"
        ]["status"]
        == "revision_requested"
    )


def test_cancel_interrupt_resume_closed_loop():
    """cancel 应路由到 Cancel Demo。"""

    graph = _compile_memory_graph()
    config = {
        "configurable": {
            "thread_id": "cancel-1"
        }
    }

    paused = graph.invoke(
        build_demo_decision_state(),
        config=config,
    )

    payload = _interrupt_payload(
        paused
    )

    final_state = graph.invoke(
        Command(
            resume={
                "decision": "cancel",
                "proposal_id": payload[
                    "proposal_id"
                ],
                "action_id": payload[
                    "action_id"
                ],
                "reason": "暂时不需要",
            }
        ),
        config=config,
    )

    assert final_state["decision"] == (
        "cancel"
    )
    assert (
        final_state[
            "final_response"
        ]["status"]
        == "cancelled"
    )
    assert (
        final_state[
            "itinerary_status"
        ]
        == "rejected"
    )


def test_invalid_human_input_reprompts_once_per_node_invocation():
    """
    过期 proposal_id 不结束 Graph，而是回到 DecisionGate 再次暂停。

    该测试验证官方推荐模式：
        interrupt 一次
        → 校验
        → 非法则 State 更新并条件回环
        → 下一次节点调用再 interrupt 一次
    """

    graph = _compile_memory_graph()
    config = {
        "configurable": {
            "thread_id": "invalid-1"
        }
    }

    paused = graph.invoke(
        build_demo_decision_state(),
        config=config,
    )

    payload = _interrupt_payload(
        paused
    )

    reprompted = graph.invoke(
        Command(
            resume={
                "decision": "approve",
                "proposal_id": (
                    "stale_proposal"
                ),
                "action_id": payload[
                    "action_id"
                ],
            }
        ),
        config=config,
    )

    new_payload = _interrupt_payload(
        reprompted
    )

    assert new_payload[
        "validation_error"
    ]
    assert (
        "proposal_id"
        in new_payload[
            "validation_error"
        ]["message"]
    )

    final_state = graph.invoke(
        Command(
            resume={
                "decision": "approve",
                "proposal_id": new_payload[
                    "proposal_id"
                ],
                "action_id": new_payload[
                    "action_id"
                ],
            }
        ),
        config=config,
    )

    assert (
        final_state[
            "decision_attempt_count"
        ]
        == 2
    )
    assert (
        final_state[
            "decision_result"
        ]["status"]
        == "accepted"
    )


def test_sqlite_checkpoint_can_resume_after_graph_recreation(
    tmp_path,
):
    """
    SQLite Checkpoint 应允许关闭旧连接、重建 Graph 后继续同一 thread。

    这证明暂停状态不是只存在于 Python 内存中。
    """

    db_path = (
        tmp_path
        / "checkpoints.sqlite"
    )

    config = {
        "configurable": {
            "thread_id": (
                "sqlite-persist-1"
            )
        }
    }

    # 1. 第一个进程生命周期：运行到 interrupt 并关闭连接。
    connection_1 = sqlite3.connect(
        db_path,
        check_same_thread=False,
    )

    graph_1 = build_decision_gate_demo_graph(
        checkpointer=SqliteSaver(
            connection_1
        )
    )

    paused = graph_1.invoke(
        build_demo_decision_state(),
        config=config,
    )

    payload = _interrupt_payload(
        paused
    )

    connection_1.close()

    # 2. 第二个进程生命周期：重新打开 SQLite、重建 Graph。
    connection_2 = sqlite3.connect(
        db_path,
        check_same_thread=False,
    )

    try:
        graph_2 = build_decision_gate_demo_graph(
            checkpointer=SqliteSaver(
                connection_2
            )
        )

        final_state = graph_2.invoke(
            Command(
                resume={
                    "decision": "approve",
                    "proposal_id": payload[
                        "proposal_id"
                    ],
                    "action_id": payload[
                        "action_id"
                    ],
                }
            ),
            config=config,
        )

        assert (
            final_state[
                "final_response"
            ]["status"]
            == "approved_simulated"
        )

    finally:
        connection_2.close()
