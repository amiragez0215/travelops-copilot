from __future__ import annotations

import pytest

from app.workflows.checkpoint_runtime import (
    create_sqlite_checkpointer,
)
from app.workflows.workflow_runtime import (
    PlanTripWorkflowRuntime,
    WorkflowNotWaitingForDecisionError,
    WorkflowThreadAlreadyExistsError,
    WorkflowThreadNotFoundError,
)
from app.workflows.snapshot_utils import (
    get_interrupt_values,
    snapshot_values,
)
from tests.fakes.plan_trip_workflow import (
    build_fake_plan_trip_graph,
    build_fake_runtime,
)


def _approve_payload(
    execution,
) -> dict[str, str]:
    """根据一次暂停执行构造合法 approve Resume Payload。"""

    interrupt_payload = get_interrupt_values(
        graph_output=execution.output,
        snapshot=execution.snapshot,
    )[0]

    return {
        "decision": "approve",
        "proposal_id": (
            interrupt_payload["proposal_id"]
        ),
        "action_id": (
            interrupt_payload["action_id"]
        ),
    }


def test_runtime_start_get_history_and_unknown_thread():
    """
    Runtime 应能够启动 Thread、读取最新 Snapshot 和查询历史。
    """

    runtime = build_fake_runtime()

    started = runtime.start_plan(
        user_id="user_001",
        raw_message="杭州到成都三日游。",
        reference_date="2026-07-01",
        trip_session_id="runtime-basic-001",
    )

    interrupts = get_interrupt_values(
        graph_output=started.output,
        snapshot=started.snapshot,
    )

    assert interrupts
    assert interrupts[0]["proposal_id"] == (
        "proposal_v1"
    )

    current = runtime.get_snapshot(
        "runtime-basic-001"
    )
    assert get_interrupt_values(
        snapshot=current
    )

    history = runtime.get_history(
        "runtime-basic-001",
        limit=10,
    )
    assert history

    with pytest.raises(
        WorkflowThreadNotFoundError
    ):
        runtime.get_snapshot(
            "missing-thread"
        )


def test_runtime_rejects_duplicate_start_thread_id():
    """
    Start API 使用的 trip_session_id 必须唯一；
    重复启动不能覆盖已有 Checkpoint。
    """

    runtime = build_fake_runtime()

    runtime.start_plan(
        user_id="user_001",
        raw_message="第一次",
        trip_session_id="same-thread",
    )

    with pytest.raises(
        WorkflowThreadAlreadyExistsError
    ):
        runtime.start_plan(
            user_id="user_001",
            raw_message="第二次",
            trip_session_id="same-thread",
        )


def test_runtime_resumes_approve_and_rejects_duplicate_resume():
    """
    合法 approve 应完成 Graph；完成后的 Thread 不再接受 Resume。

    CommitDraftService 自身仍有数据库幂等保护，
    但 Runtime 层会更早拒绝“已不在 interrupt”的重复 Resume。
    """

    calls: list[str] = []
    runtime = build_fake_runtime(
        calls=calls
    )

    paused = runtime.start_plan(
        user_id="user_001",
        raw_message="杭州到成都三日游。",
        trip_session_id=(
            "runtime-approve-001"
        ),
    )

    decision = _approve_payload(paused)

    completed = runtime.resume_decision(
        thread_id="runtime-approve-001",
        decision_payload=decision,
    )

    values = snapshot_values(
        completed.snapshot
    )

    assert values["final_response"][
        "status"
    ] == "approved_simulated"
    assert calls.count("commit_draft") == 1

    with pytest.raises(
        WorkflowNotWaitingForDecisionError
    ):
        runtime.resume_decision(
            thread_id="runtime-approve-001",
            decision_payload=decision,
        )


def test_runtime_request_changes_returns_second_interrupt():
    """
    request_changes 应在同一 Resume 调用中完成 Revision Loop，
    并停在 Proposal v2 的第二次 DecisionGate interrupt。
    """

    calls: list[str] = []
    runtime = build_fake_runtime(
        calls=calls
    )

    first = runtime.start_plan(
        user_id="user_001",
        raw_message="杭州到成都三日游。",
        trip_session_id=(
            "runtime-revision-001"
        ),
    )

    first_interrupt = get_interrupt_values(
        graph_output=first.output,
        snapshot=first.snapshot,
    )[0]

    second = runtime.resume_decision(
        thread_id="runtime-revision-001",
        decision_payload={
            "decision": "request_changes",
            "proposal_id": first_interrupt[
                "proposal_id"
            ],
            "action_id": first_interrupt[
                "action_id"
            ],
            "change_request": (
                "预算增加1000元。"
            ),
        },
    )

    second_interrupt = get_interrupt_values(
        graph_output=second.output,
        snapshot=second.snapshot,
    )[0]

    assert second_interrupt[
        "proposal_id"
    ] == "proposal_v2"
    assert "revision_analyze" in calls
    assert "apply_revision" in calls
    assert calls.count("input_extract") == 1
    assert calls.count(
        "missing_info_check"
    ) == 2


def test_sqlite_checkpoint_resumes_after_graph_recreation(
    tmp_path,
):
    """
    关闭第一条 SQLite 连接并重新创建 Graph 后，
    同一 thread_id 仍能从 DecisionGate interrupt 恢复。

    这是阶段一最重要的 Durable Execution 集成测试：
        HTTP 请求结束或服务进程重建后，
        Checkpoint 仍然保存在磁盘数据库中。
    """

    checkpoint_path = (
        tmp_path
        / "langgraph_test.sqlite"
    )

    # ------------------------------------------------------------------
    # 第一个“服务生命周期”：运行到 interrupt 后关闭连接。
    # ------------------------------------------------------------------
    saver_1, connection_1 = (
        create_sqlite_checkpointer(
            checkpoint_path
        )
    )
    graph_1 = build_fake_plan_trip_graph(
        checkpointer=saver_1
    )
    runtime_1 = PlanTripWorkflowRuntime(
        graph=graph_1,
        checkpointer=saver_1,
        recursion_limit=100,
    )

    paused = runtime_1.start_plan(
        user_id="user_001",
        raw_message="杭州到成都三日游。",
        trip_session_id="sqlite-resume-001",
    )
    decision = _approve_payload(paused)

    connection_1.close()

    # ------------------------------------------------------------------
    # 第二个“服务生命周期”：重新打开同一 SQLite 文件并重建 Graph。
    # ------------------------------------------------------------------
    saver_2, connection_2 = (
        create_sqlite_checkpointer(
            checkpoint_path
        )
    )
    graph_2 = build_fake_plan_trip_graph(
        checkpointer=saver_2
    )
    runtime_2 = PlanTripWorkflowRuntime(
        graph=graph_2,
        checkpointer=saver_2,
        recursion_limit=100,
    )

    try:
        restored = runtime_2.get_snapshot(
            "sqlite-resume-001"
        )
        assert get_interrupt_values(
            snapshot=restored
        )

        completed = runtime_2.resume_decision(
            thread_id="sqlite-resume-001",
            decision_payload=decision,
        )

        values = snapshot_values(
            completed.snapshot
        )
        assert values["final_response"][
            "status"
        ] == "approved_simulated"

    finally:
        connection_2.close()
