from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.api.graph_response import (
    build_public_history_response,
    build_public_run_response,
)


@dataclass
class FakeInterrupt:
    """模拟 LangGraph Interrupt 对象。"""

    value: dict[str, Any]


@dataclass
class FakeTask:
    """模拟 StateSnapshot.tasks 中的 PregelTask。"""

    interrupts: tuple[FakeInterrupt, ...]


@dataclass
class FakeSnapshot:
    """只包含 Formatter 需要的公开 StateSnapshot 属性。"""

    values: dict[str, Any]
    next: tuple[str, ...]
    config: dict[str, Any]
    metadata: dict[str, Any]
    created_at: str
    tasks: tuple[FakeTask, ...] = ()


def _snapshot() -> FakeSnapshot:
    return FakeSnapshot(
        values={
            "workflow": "plan_trip",
            "itinerary_status": "proposed",
            "proposal": {
                "proposal_id": "proposal_001",
                "summary": "测试方案",
            },
            "verifier_result": {
                "status": "passed",
                "passed": True,
                "proposal_id": "proposal_001",
                "next_action": "decision_gate",
                "warning_count": 0,
                "error_count": 0,
                "critical_count": 0,
            },
            "user_profile": {
                "private": "不能返回"
            },
            "evidence_pool": [
                {
                    "content": "不能返回完整证据"
                }
            ],
            "trace": [
                {
                    "node_name": "proposal",
                    "tool_name": "deepseek",
                    "status": "success",
                    "latency_ms": 123,
                    "input_summary": "输入摘要",
                    "output_summary": "输出摘要",
                    "created_at": (
                        "2026-08-09T00:00:00Z"
                    ),
                    "full_prompt": (
                        "内部 Prompt 不应返回"
                    ),
                }
            ],
        },
        next=("decision_gate",),
        config={
            "configurable": {
                "thread_id": "thread_001",
                "checkpoint_id": "cp_001",
            }
        },
        metadata={
            "step": 12,
            "source": "loop",
        },
        created_at="2026-08-09T00:00:00Z",
        tasks=(
            FakeTask(
                interrupts=(
                    FakeInterrupt(
                        value={
                            "proposal_id": (
                                "proposal_001"
                            ),
                            "action_id": (
                                "approve_proposal_001"
                            ),
                        }
                    ),
                )
            ),
        ),
    )


def test_public_run_response_uses_field_whitelist():
    """
    Public Response 应展示 Proposal 和 Interrupt，
    但不能暴露用户画像、Evidence 和 Trace 中的新私有字段。
    """

    response = build_public_run_response(
        thread_id="thread_001",
        graph_output=None,
        snapshot=_snapshot(),
        include_trace=True,
        max_trace_items=10,
    )

    payload = response.model_dump(
        mode="json"
    )

    assert payload["run_status"] == (
        "waiting_for_decision"
    )
    assert payload["checkpoint_id"] == (
        "cp_001"
    )
    assert payload["proposal"][
        "proposal_id"
    ] == "proposal_001"
    assert payload["interrupt"][
        "action_id"
    ] == "approve_proposal_001"
    assert payload["verification"][
        "passed"
    ] is True
    assert "user_profile" not in payload
    assert "evidence_pool" not in payload
    assert "full_prompt" not in (
        payload["trace"][0]
    )


def test_public_history_response_only_returns_checkpoint_summary():
    """Checkpoint History 不应返回完整 values 或 tasks。"""

    response = build_public_history_response(
        thread_id="thread_001",
        snapshots=[_snapshot()],
    )

    payload = response.model_dump(
        mode="json"
    )

    assert payload["count"] == 1
    item = payload["items"][0]
    assert item["checkpoint_id"] == (
        "cp_001"
    )
    assert item["has_interrupt"] is True
    assert item["proposal_id"] == (
        "proposal_001"
    )
    assert "values" not in item
    assert "tasks" not in item
