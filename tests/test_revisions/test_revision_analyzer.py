from __future__ import annotations

from typing import Any

from app.revisions.revision_analyzer import (
    LLMRevisionAnalyzer,
    RuleBasedRevisionAnalyzer,
)


class FakeJSONClient:
    """按顺序返回预设 JSON 或抛出异常。"""

    model_id = "fake-revision-model"

    def __init__(
        self,
        responses: list[Any],
    ) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.user_prompts: list[str] = []

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        del system_prompt, max_tokens
        self.user_prompts.append(user_prompt)
        value = self.responses[
            self.call_count
        ]
        self.call_count += 1

        if isinstance(value, Exception):
            raise value

        return value


def _trip_request() -> dict[str, Any]:
    return {
        "user_id": "user_001",
        "origin": "杭州",
        "destination": "成都",
        "start_date": "2026-07-02",
        "end_date": "2026-07-04",
        "days": 3,
        "budget": 4000,
        "people_count": 1,
        "room_count": 1,
        "preference_signals": [],
        "spend_preferences": [],
        "hard_constraints": [],
        "named_constraints": [],
        "unmapped_requirements": [],
        "requirement_issues": [],
        "field_resolution": {},
        "raw_message": "从杭州去成都玩三天，预算4000。",
        "extraction": {},
    }


def _change_request() -> dict[str, Any]:
    return {
        "raw_message": (
            "酒店换成绿水青山酒店，多待两天，预算增加1000元。"
        ),
        "base_proposal_id": "proposal_001",
        "base_proposal_version": 1,
    }


def _proposal() -> dict[str, Any]:
    return {
        "proposal_id": "proposal_001",
        "version": 1,
    }


def _valid_payload() -> dict[str, Any]:
    return {
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
        "replace_named_entity_types": [
            "hotel"
        ],
        "named_constraint_additions": [
            {
                "entity_type": "hotel",
                "entity_name": "绿水青山酒店",
                "constraint_mode": "required",
                "scope": "whole_trip",
                "evidence": "酒店换成绿水青山酒店",
                "source": "llm",
            }
        ],
        "summary": "延长两天、提高预算并更换指定酒店。",
        "confidence": 0.95,
    }


def test_llm_revision_analyzer_builds_relative_patch():
    client = FakeJSONClient(
        [_valid_payload()]
    )
    analyzer = LLMRevisionAnalyzer(
        client=client,
        max_attempts=2,
    )

    result = analyzer.analyze(
        trip_request=_trip_request(),
        change_request=_change_request(),
        proposal=_proposal(),
    )

    assert result["status"] == "ready"
    assert result["fallback_used"] is False
    assert result["attempt_count"] == 1
    assert (
        result["patch"]["field_operations"][0]["operation"]
        == "increment"
    )
    assert result["patch"]["replace_named_entity_types"] == [
        "hotel"
    ]


def test_llm_revision_analyzer_retries_invalid_patch():
    invalid = _valid_payload()
    invalid["field_operations"] = [
        {
            "operation": "increment",
            "field": "days",
            "value": 2,
        },
        {
            "operation": "set",
            "field": "days",
            "value": 5,
        },
    ]

    client = FakeJSONClient(
        [invalid, _valid_payload()]
    )
    analyzer = LLMRevisionAnalyzer(
        client=client,
        max_attempts=2,
    )

    result = analyzer.analyze(
        trip_request=_trip_request(),
        change_request=_change_request(),
        proposal=_proposal(),
    )

    assert result["status"] == "ready"
    assert result["attempt_count"] == 2
    assert client.call_count == 2
    assert result["validation_errors"]
    assert "上一次 RevisionPatch" in client.user_prompts[1]


def test_llm_revision_analyzer_falls_back_to_rules():
    client = FakeJSONClient(
        [RuntimeError("network"), RuntimeError("network")]
    )
    analyzer = LLMRevisionAnalyzer(
        client=client,
        fallback_analyzer=(
            RuleBasedRevisionAnalyzer()
        ),
        max_attempts=2,
    )

    result = analyzer.analyze(
        trip_request=_trip_request(),
        change_request=_change_request(),
        proposal=_proposal(),
    )

    assert result["status"] == "ready"
    assert result["fallback_used"] is True
    assert result["validation_errors"]

    operations = result["patch"][
        "field_operations"
    ]
    assert {
        (item["field"], item["value"])
        for item in operations
    } == {
        ("days", 2),
        ("budget", 1000.0),
    }

    hotel = result["patch"][
        "named_constraint_additions"
    ][0]
    assert hotel["entity_name"] == "绿水青山酒店"


def test_rule_revision_analyzer_requests_clarification_when_nothing_understood():
    analyzer = RuleBasedRevisionAnalyzer()

    change = _change_request()
    change["raw_message"] = "换得更有感觉一点"

    result = analyzer.analyze(
        trip_request=_trip_request(),
        change_request=change,
        proposal=_proposal(),
    )

    assert (
        result["status"]
        == "clarification_required"
    )
    assert result["clarification_questions"]


def test_rule_revision_analyzer_blocks_ambiguous_flight_quality_even_with_other_changes():
    """
    即使“预算增加1000”可以确定执行，
    “飞机坐好一点”仍然会影响候选选择，必须先澄清，不能静默忽略。
    """

    analyzer = RuleBasedRevisionAnalyzer()
    change = _change_request()
    change["raw_message"] = "预算增加1000，飞机坐好一点"

    result = analyzer.analyze(
        trip_request=_trip_request(),
        change_request=change,
        proposal=_proposal(),
    )

    assert result["status"] == "clarification_required"
    assert result["patch"]["field_operations"]
    assert any(
        item["issue_type"] == "ambiguous_flight_quality"
        for item in result["patch"]["requirement_issues"]
    )
    assert result["clarification_questions"]
