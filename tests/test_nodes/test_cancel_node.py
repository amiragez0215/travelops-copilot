from __future__ import annotations

from copy import deepcopy

from app.nodes.cancel_node import (
    cancel_node,
    route_after_cancel,
)
from tests.test_verifiers.test_proposal_verifier import (
    _generated_state,
    _verify,
)


def _cancel_state() -> dict:
    """
    构造已经通过 Verifier、并由 DecisionGate 接受 cancel 的 State。
    """

    generated_state = _generated_state()
    verified = _verify(generated_state)

    proposal = generated_state[
        "proposal"
    ]

    verifier_result = verified[
        "verifier_result"
    ]

    action = deepcopy(
        verified["pending_actions"][0]
    )

    decision_event_id = (
        "decision_test_cancel_001"
    )

    # DecisionGateNode 在 cancel 分支会把待审批动作标记为 cancelled。
    action.update(
        {
            "status": "cancelled",
            "decision": "cancel",
            "decision_event_id": (
                decision_event_id
            ),
            "decided_at": (
                "2026-07-24T15:00:00+00:00"
            ),
            "decision_reason": (
                "用户暂时不需要该方案"
            ),
        }
    )

    decision_result = {
        "status": "accepted",
        "strategy_version": (
            "langgraph_hitl_decision_gate_v1"
        ),
        "accepted": True,
        "decision": "cancel",
        "next_action": "cancel",
        "proposal_id": (
            proposal["proposal_id"]
        ),
        "proposal_version": (
            proposal["version"]
        ),
        "action_id": action[
            "action_id"
        ],
        "decision_event_id": (
            decision_event_id
        ),
        "client_request_id": (
            "client_cancel_001"
        ),
        "attempt_count": 1,
        "decided_at": (
            "2026-07-24T15:00:00+00:00"
        ),
        "change_request_provided": False,
        "reason": (
            "用户暂时不需要该方案"
        ),
        "validation_errors": [],
    }

    return {
        "proposal": proposal,
        "verifier_result": (
            verifier_result
        ),
        "decision_result": (
            decision_result
        ),
        "pending_actions": [action],
        "itinerary_status": "proposed",
        "change_request": {
            "raw_message": "旧修改请求"
        },
        "proposal_feedback": {
            "instructions": [
                "旧 Verifier 反馈"
            ]
        },
    }


def test_cancel_node_success():
    """
    合法 cancel 决定应完成内部收尾，但不删除 Proposal。
    """

    state = _cancel_state()

    result = cancel_node(state)

    assert result["cancel_result"][
        "status"
    ] == "cancelled"

    assert result[
        "itinerary_status"
    ] == "rejected"

    # Proposal 没有被删除；Node 只返回局部更新。
    assert "proposal" not in result

    assert result[
        "pending_actions"
    ][0]["status"] == "cancelled"

    assert result[
        "pending_actions"
    ][0]["cancel_finalized"] is True

    assert result[
        "change_request"
    ] == {}

    assert result[
        "proposal_feedback"
    ] == {}

    assert result["trace"][0][
        "node_name"
    ] == "cancel"

    assert result["trace"][0][
        "status"
    ] == "success"

    assert (
        route_after_cancel(result)
        == "final_response"
    )


def test_cancel_node_is_idempotent_for_same_event():
    """
    同一个取消事件被 Checkpoint 重放时，应返回 already_cancelled。
    """

    state = _cancel_state()
    first = cancel_node(state)

    # 模拟 LangGraph 将第一次局部更新合并回 State 后再次执行。
    replay_state = {
        **state,
        **first,
    }

    second = cancel_node(
        replay_state
    )

    assert first["cancel_result"][
        "status"
    ] == "cancelled"

    assert second["cancel_result"][
        "status"
    ] == "already_cancelled"

    assert second["cancel_result"][
        "idempotent_replay"
    ] is True

    assert (
        second["cancel_result"][
            "decision_event_id"
        ]
        == first["cancel_result"][
            "decision_event_id"
        ]
    )


def test_cancel_node_rejects_non_cancel_decision():
    """
    approve 或 request_changes 不能错误进入 CancelNode。
    """

    state = _cancel_state()

    state["decision_result"] = deepcopy(
        state["decision_result"]
    )

    state["decision_result"].update(
        {
            "decision": "approve",
            "next_action": (
                "commit_draft"
            ),
        }
    )

    result = cancel_node(state)

    assert result["cancel_result"][
        "status"
    ] == "failed"

    assert result["errors"]

    assert "cancel" in result[
        "errors"
    ][0]["message"]


def test_cancel_node_rejects_stale_proposal_id():
    """
    人工取消与当前 Proposal ID 不一致时不能完成取消收尾。
    """

    state = _cancel_state()

    state["decision_result"] = deepcopy(
        state["decision_result"]
    )

    state["decision_result"][
        "proposal_id"
    ] = "stale_proposal"

    result = cancel_node(state)

    assert result["cancel_result"][
        "status"
    ] == "failed"

    assert "不一致" in result[
        "errors"
    ][0]["message"]


def test_cancel_node_requires_cancelled_pending_action():
    """
    Pending Action 尚未被 DecisionGate 标记 cancelled 时应拒绝。
    """

    state = _cancel_state()
    state["pending_actions"] = deepcopy(
        state["pending_actions"]
    )

    state["pending_actions"][0][
        "status"
    ] = "pending"

    result = cancel_node(state)

    assert result["cancel_result"][
        "status"
    ] == "failed"

    assert "cancelled" in result[
        "errors"
    ][0]["message"]


def test_cancel_node_does_not_create_final_response():
    """
    CancelNode 只负责状态收尾；用户响应仍由 FinalResponseNode 生成。
    """

    result = cancel_node(
        _cancel_state()
    )

    assert "final_response" not in result


def test_cancel_node_does_not_claim_external_cancellation():
    """
    当前项目没有真实预订，因此取消结果不能声称执行了外部取消。
    """

    result = cancel_node(
        _cancel_state()
    )

    cancel_result = result[
        "cancel_result"
    ]

    assert cancel_result[
        "proposal_deleted"
    ] is False

    assert cancel_result[
        "real_booking_cancelled"
    ] is False

    assert cancel_result[
        "real_payment_reversed"
    ] is False

    assert cancel_result[
        "real_notification_sent"
    ] is False
