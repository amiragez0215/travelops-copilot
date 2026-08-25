from __future__ import annotations

from typing import Any, Mapping

from app.nodes.revision_analyze_node import (
    revision_analyze_node,
    route_after_revision_analyze,
)


class FakeRevisionAnalyzer:
    """Node 测试不调用真实模型。"""

    def __init__(self, status: str = "ready") -> None:
        self.status = status

    def analyze(
        self,
        *,
        trip_request: Mapping[str, Any],
        change_request: Mapping[str, Any],
        proposal: Mapping[str, Any],
    ) -> dict[str, Any]:
        del trip_request
        return {
            "status": self.status,
            "strategy_version": "test",
            "base_proposal_id": proposal[
                "proposal_id"
            ],
            "base_proposal_version": proposal[
                "version"
            ],
            "base_trip_request_hash": "a" * 64,
            "raw_change_request": change_request[
                "raw_message"
            ],
            "patch": {
                "field_operations": [],
                "summary": "测试修改",
            },
            "clarification_questions": (
                ["请说明具体修改内容"]
                if self.status
                == "clarification_required"
                else []
            ),
            "model_id": "fake",
            "attempt_count": 1,
            "fallback_used": False,
            "validation_errors": [],
        }


def _state() -> dict[str, Any]:
    return {
        "trip_request": {
            "origin": "杭州",
            "destination": "成都",
            "start_date": "2026-07-02",
            "days": 3,
        },
        "proposal": {
            "proposal_id": "proposal_001",
            "version": 1,
        },
        "change_request": {
            "raw_message": "预算增加1000元",
            "base_proposal_id": "proposal_001",
            "base_proposal_version": 1,
        },
        "decision_result": {
            "status": "accepted",
            "accepted": True,
            "decision": "request_changes",
            "next_action": "revision_analyze",
            "proposal_id": "proposal_001",
            "proposal_version": 1,
            "action_id": "action_001",
            "decision_event_id": "decision_001",
            "attempt_count": 1,
            "change_request_provided": True,
            "validation_errors": [],
        },
    }


def test_revision_analyze_node_success():
    result = revision_analyze_node(
        _state(),
        analyzer=FakeRevisionAnalyzer(),
    )

    assert result["revision_plan"]["status"] == "ready"
    assert result["revision_result"]["status"] == "analyzed"
    assert result["trace"][0]["status"] == "success"
    assert (
        route_after_revision_analyze(result)
        == "apply_revision"
    )


def test_revision_analyze_node_clarification_route():
    result = revision_analyze_node(
        _state(),
        analyzer=FakeRevisionAnalyzer(
            "clarification_required"
        ),
    )

    assert (
        route_after_revision_analyze(result)
        == "final_response"
    )
    assert result["trace"][0]["status"] == "degraded"


def test_revision_analyze_node_rejects_wrong_decision():
    state = _state()
    state["decision_result"]["decision"] = "approve"
    state["decision_result"]["next_action"] = "commit_draft"

    result = revision_analyze_node(
        state,
        analyzer=FakeRevisionAnalyzer(),
    )

    assert result["revision_plan"]["status"] == "failed"
    assert result["errors"]
