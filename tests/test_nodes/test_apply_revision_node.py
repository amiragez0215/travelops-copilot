from __future__ import annotations

from typing import Any

from app.nodes.apply_revision_node import (
    apply_revision_node,
    route_after_apply_revision,
)
from app.revisions.revision_analyzer import hash_trip_request


def _trip_request() -> dict[str, Any]:
    return {
        "user_id": "user_001",
        "origin": "杭州",
        "destination": "成都",
        "start_date": "2026-07-02",
        "end_date": "2026-07-04",
        "partial_date": None,
        "days": 3,
        "budget": 4000,
        "people_count": 1,
        "room_count": 1,
        "transport_preferences": [],
        "hotel_preferences": ["quiet"],
        "travel_style": [],
        "raw_constraints": ["喜欢安静酒店"],
        "preference_signals": [
            {
                "domain": "hotel",
                "key": "quiet",
                "label": "安静",
                "strength": 0.8,
                "polarity": "positive",
                "evidence": "喜欢安静酒店",
                "source": "llm",
            }
        ],
        "spend_preferences": [],
        "hard_constraints": [],
        "named_constraints": [
            {
                "entity_type": "hotel",
                "entity_name": "旧酒店",
                "constraint_mode": "preferred",
                "scope": "whole_trip",
                "traveler": None,
                "evidence": "优先旧酒店",
                "source": "llm",
            }
        ],
        "unmapped_requirements": [],
        "requirement_issues": [],
        "field_resolution": {},
        "raw_message": "从杭州去成都玩三天，预算4000。",
        "extraction": {
            "method": "hybrid_llm_rule_v2",
            "notes": [],
        },
    }


def _revision_plan() -> dict[str, Any]:
    request = _trip_request()
    return {
        "status": "ready",
        "strategy_version": "test",
        "base_proposal_id": "proposal_001",
        "base_proposal_version": 1,
        "base_trip_request_hash": hash_trip_request(request),
        "raw_change_request": (
            "酒店换成绿水青山酒店，多待两天，预算增加1000元。"
        ),
        "patch": {
            "field_operations": [
                {
                    "operation": "increment",
                    "field": "days",
                    "value": 2,
                    "evidence": "多待两天",
                },
                {
                    "operation": "increment",
                    "field": "budget",
                    "value": 1000,
                    "evidence": "预算增加1000元",
                },
            ],
            "preference_upserts": [
                {
                    "domain": "hotel",
                    "key": "cleanliness",
                    "label": "干净",
                    "strength": 0.95,
                    "polarity": "positive",
                    "evidence": "酒店更干净",
                    "source": "llm",
                }
            ],
            "replace_named_entity_types": [
                "hotel"
            ],
            "named_constraint_additions": [
                {
                    "entity_type": "hotel",
                    "entity_name": "绿水青山酒店",
                    "constraint_mode": "required",
                    "scope": "whole_trip",
                    "traveler": None,
                    "evidence": "酒店换成绿水青山酒店",
                    "source": "llm",
                }
            ],
            "summary": "延长两天、增加预算并替换酒店。",
            "confidence": 0.95,
        },
        "clarification_questions": [],
        "model_id": "fake",
        "attempt_count": 1,
        "fallback_used": False,
        "validation_errors": [],
    }


def _state() -> dict[str, Any]:
    return {
        "user_id": "user_001",
        "workflow": "plan_trip",
        "raw_message": "原始请求",
        "trip_request": _trip_request(),
        "proposal": {
            "proposal_id": "proposal_001",
            "version": 1,
        },
        "change_request": {
            "raw_message": (
                "酒店换成绿水青山酒店，多待两天，预算增加1000元。"
            ),
            "base_proposal_id": "proposal_001",
            "base_proposal_version": 1,
        },
        "revision_plan": _revision_plan(),
        # 这些旧结果必须被清空。
        "planning_context": {"old": True},
        "weather_result": {"old": True},
        "flight_candidates": [{"old": True}],
        "selection_result": {"status": "feasible"},
        "evidence_pool": [{"old": True}],
        "verifier_result": {"status": "passed"},
        "pending_actions": [{"old": True}],
        "decision": "request_changes",
        "decision_result": {"status": "accepted"},
        "proposal_retry_count": 1,
        "budget_retry_count": 1,
        "rag_retry_count": 1,
    }


def test_apply_revision_node_applies_patch_and_invalidates_old_results():
    result = apply_revision_node(_state())

    request = result["trip_request"]

    assert request["days"] == 5
    assert request["end_date"] == "2026-07-06"
    assert request["budget"] == 5000.0

    hotels = request["named_constraints"]
    assert len(hotels) == 1
    assert hotels[0]["entity_name"] == "绿水青山酒店"
    assert hotels[0]["constraint_mode"] == "required"

    preference_keys = {
        item["key"]
        for item in request["preference_signals"]
    }
    assert preference_keys == {
        "quiet",
        "cleanliness",
    }

    assert result["revision_result"]["status"] == "applied"
    assert result["planning_context"] == {}
    assert result["weather_result"] == {}
    assert result["flight_candidates"] == []
    assert result["selection_result"] == {}
    assert result["evidence_pool"] == []
    assert result["proposal"] == {}
    assert result["pending_actions"] == []
    assert result["decision"] is None
    assert result["change_request"] == {}
    assert result["proposal_retry_count"] == 0
    assert result["workflow"] == "modify_trip"
    assert result["raw_message"].startswith("酒店换成")

    assert (
        route_after_apply_revision(result)
        == "missing_info_check"
    )


def test_apply_revision_node_rejects_stale_trip_request_hash():
    state = _state()
    state["trip_request"]["budget"] = 4500

    result = apply_revision_node(state)

    assert result["revision_result"]["status"] == "failed"
    assert result["errors"]
    assert (
        route_after_apply_revision(result)
        == "final_response"
    )


def test_apply_revision_node_derives_days_from_new_end_date():
    state = _state()
    plan = _revision_plan()
    plan["patch"] = {
        "field_operations": [
            {
                "operation": "set",
                "field": "end_date",
                "value": "2026-07-08",
            }
        ],
        "summary": "延后返程日期。",
    }
    state["revision_plan"] = plan

    result = apply_revision_node(state)

    assert result["trip_request"]["days"] == 7
    assert result["trip_request"]["end_date"] == "2026-07-08"
