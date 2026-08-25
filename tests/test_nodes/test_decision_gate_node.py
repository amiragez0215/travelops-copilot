from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.nodes.decision_gate_node import (
    build_decision_interrupt_payload,
)
from app.schemas.decision_schema import (
    DecisionResponse,
)
from app.workflows.decision_gate_demo_workflow import (
    build_demo_decision_state,
)


def test_build_decision_interrupt_payload():
    """Payload 应只包含审批需要的精简信息。"""

    payload = build_decision_interrupt_payload(
        build_demo_decision_state()
    )

    data = payload.to_state_dict()

    assert data["type"] == (
        "trip_proposal_decision"
    )
    assert data["proposal_id"] == (
        "proposal_demo_001"
    )
    assert data["action_id"] == (
        "approve_proposal_demo_001"
    )

    assert {
        item["decision"]
        for item in data[
            "available_actions"
        ]
    } == {
        "approve",
        "request_changes",
        "cancel",
    }

    assert (
        data["proposal_summary"][
            "destination"
        ]
        == "成都"
    )

    # Interrupt Payload 不应暴露完整 Proposal 或内部 Trace。
    assert "daily_plan" not in data
    assert "trace" not in data


def test_request_changes_requires_text():
    """修改决定必须包含具体 change_request。"""

    with pytest.raises(
        ValidationError
    ):
        DecisionResponse(
            decision="request_changes",
            proposal_id="proposal_demo_001",
            action_id=(
                "approve_proposal_demo_001"
            ),
            change_request="   ",
        )


def test_approve_rejects_change_request():
    """批准和修改不能在同一个 Resume Payload 中同时出现。"""

    with pytest.raises(
        ValidationError
    ):
        DecisionResponse(
            decision="approve",
            proposal_id="proposal_demo_001",
            action_id=(
                "approve_proposal_demo_001"
            ),
            change_request="再换一家酒店",
        )


def test_payload_requires_verified_proposal():
    """未通过 Verifier 的方案不能进入人工审批。"""

    state = build_demo_decision_state()
    state["verifier_result"][
        "passed"
    ] = False

    with pytest.raises(
        ValueError,
        match="passed",
    ):
        build_decision_interrupt_payload(
            state
        )


def test_payload_requires_single_pending_action():
    """当前 Proposal 必须且只能有一个匹配的 pending approval。"""

    state = build_demo_decision_state()
    state["pending_actions"] = []

    with pytest.raises(
        ValueError,
        match="实际数量",
    ):
        build_decision_interrupt_payload(
            state
        )
