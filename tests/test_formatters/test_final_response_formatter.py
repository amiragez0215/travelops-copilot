from __future__ import annotations

from app.formatters.final_response_formatter import (
    build_final_response,
)


def test_build_commit_success_response():
    response = build_final_response(
        {
            "workflow": "plan_trip",
            "trip_session_id": "session_001",
            "commit_result": {
                "status": "committed",
                "message": "保存成功。",
                "trip_id": "trip_001",
                "proposal_id": "proposal_001",
                "proposal_version": 1,
                "idempotent_replay": False,
            },
            "trip_draft": {
                "trip_id": "trip_001",
                "destination": "成都",
                "start_date": "2026-07-02",
                "end_date": "2026-07-04",
                "known_subtotal": 2560,
                "remaining_budget": 1440,
            },
            "trip_version": {
                "version_number": 1,
            },
        }
    )

    assert response["status"] == "approved_simulated"
    assert response["trip"]["trip_id"] == "trip_001"
    assert response["execution_boundary"]["draft_only"] is True
    assert response["trip_session_id"] == "session_001"


def test_build_cancel_response():
    response = build_final_response(
        {
            "cancel_result": {
                "status": "cancelled",
                "message": "当前旅行方案已取消。",
                "proposal_id": "proposal_001",
                "idempotent_replay": False,
            }
        }
    )

    assert response["status"] == "cancelled"
    assert response["proposal_deleted"] is False
    assert response["real_booking_cancelled"] is False


def test_preserves_existing_clarification_response():
    response = build_final_response(
        {
            "workflow": "plan_trip",
            "final_response": {
                "status": "needs_clarification",
                "type": "missing_info",
                "message": "请补充日期。",
            },
        }
    )

    assert response["status"] == "needs_clarification"
    assert response["message"] == "请补充日期。"
    assert response["workflow"] == "plan_trip"


def test_build_revision_clarification_response():
    response = build_final_response(
        {
            "revision_plan": {
                "status": "clarification_required",
                "base_proposal_id": "proposal_001",
                "clarification_questions": [
                    "飞机好一点具体指什么？"
                ],
                "patch": {
                    "summary": "已识别部分修改。"
                },
            }
        }
    )

    assert (
        response["status"]
        == "revision_needs_clarification"
    )
    assert response["need_user_input"] is True
    assert response["questions"] == [
        "飞机好一点具体指什么？"
    ]


def test_build_budget_failure_response():
    response = build_final_response(
        {
            "selection_result": {
                "status": "no_feasible_combination"
            },
            "adjustment_plan": {
                "current_total_budget": 1800,
                "minimum_known_subtotal": 2200,
                "required_budget_increase": 400,
                "suggestions": [
                    "增加预算400元"
                ],
            },
        }
    )

    assert response["status"] == "no_feasible_combination"
    assert response["details"]["required_budget_increase"] == 400
    assert response["need_user_input"] is True


def test_final_response_does_not_expose_internal_state():
    response = build_final_response(
        {
            "errors": [
                {
                    "node": "x",
                    "type": "ValueError",
                    "message": "bad",
                    "internal_stack": "secret",
                }
            ],
            "user_profile": {
                "secret": "private"
            },
            "evidence_pool": [
                {"content": "large internal chunk"}
            ],
        }
    )

    assert response["status"] == "failed"
    assert "user_profile" not in response
    assert "evidence_pool" not in response
    assert "internal_stack" not in str(response)


def test_semantic_evidence_failure_is_not_reported_as_empty_knowledge_base():
    response = build_final_response(
        {
            "evidence_result": {
                "status": "failed",
                "issues": [],
                "coverage": {
                    "guides": {
                        "required": True,
                        "required_chunks": 2,
                        "actual_chunks": 12,
                        "passed": True,
                    }
                },
                "semantic_review": {
                    "status": "insufficient",
                    "need_coverage": [
                        {
                            "need_id": "need_food",
                            "status": "partial",
                            "reason": "美食偏好证据不完整。",
                            "supporting_chunk_ids": ["internal_chunk"],
                        }
                    ],
                },
            }
        }
    )

    assert response["status"] == "insufficient_evidence"
    assert "语义证据检查" in response["message"]
    gaps = response["details"]["semantic_review"]["gaps"]
    assert gaps == [
        {
            "need_id": "need_food",
            "status": "partial",
            "reason": "美食偏好证据不完整。",
        }
    ]
    assert "internal_chunk" not in str(response)
