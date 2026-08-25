from __future__ import annotations

from datetime import date

from app.schemas.trip_request_schema import TripRequest
from app.validators.trip_request_validator import (
    build_missing_info_message,
    build_missing_info_result,
    check_missing_trip_request_fields,
    validate_hard_constraint,
)


class SupportedCapabilityValidator:
    """让单元测试不依赖真实 data/mock。"""

    def validate(self, trip_request):
        return {
            "status": "supported",
            "supported": True,
            "reason": "test",
            "issues": [],
            "supported_destinations": ["成都"],
        }


class UnsupportedCapabilityValidator:
    """模拟 LLM 理解了目的地，但项目数据不支持。"""

    def validate(self, trip_request):
        return {
            "status": "unsupported",
            "supported": False,
            "reason": "test",
            "supported_destinations": ["成都"],
            "issues": [
                {
                    "issue_type": "unsupported_destination",
                    "message": "当前 Mock 不支持西安。",
                    "requires_clarification": True,
                    "suggested_question": "是否改为成都？",
                }
            ],
        }


def _complete_request(**overrides) -> TripRequest:
    """构造核心字段完整的单人请求。"""

    data = {
        "origin": "杭州",
        "destination": "成都",
        "start_date": date(2026, 7, 2),
        "days": 3,
        "people_count": 1,
        "room_count": 1,
    }
    data.update(overrides)
    return TripRequest(**data)


def test_validator_all_required_fields_present():
    result = build_missing_info_result(
        _complete_request(),
        capability_validator=SupportedCapabilityValidator(),
    )

    assert result["can_continue"] is True
    assert result["missing_fields"] == []
    assert result["status"] == "ready"


def test_budget_is_not_required():
    request = _complete_request(budget=None)
    assert check_missing_trip_request_fields(request) == []


def test_missing_start_date_requires_clarification():
    result = build_missing_info_result(
        _complete_request(start_date=None),
        capability_validator=SupportedCapabilityValidator(),
    )

    assert result["can_continue"] is False
    assert result["missing_fields"] == ["start_date"]
    assert "出发日期" in result["message"]


def test_multiple_missing_fields_from_dict():
    result = build_missing_info_result(
        {
            "origin": None,
            "destination": "成都",
            "start_date": None,
            "days": 3,
            "people_count": 1,
            "room_count": 1,
        },
        capability_validator=SupportedCapabilityValidator(),
    )

    assert result["missing_fields"] == ["origin", "start_date"]


def test_multiple_people_without_room_count_requires_clarification():
    result = build_missing_info_result(
        _complete_request(people_count=2, room_count=None),
        capability_validator=SupportedCapabilityValidator(),
    )

    assert result["can_continue"] is False
    assert result["missing_fields"] == ["room_count"]


def test_capability_scope_can_block_supported_schema():
    """LLM 能理解西安，不代表当前 Mock 能规划西安。"""

    result = build_missing_info_result(
        _complete_request(destination="西安"),
        capability_validator=UnsupportedCapabilityValidator(),
    )

    assert result["can_continue"] is False
    assert result["status"] == "unsupported_data_scope"
    assert result["capability_result"]["supported"] is False


def test_subjective_hard_constraint_is_not_executable():
    """安静度不能因为 LLM 误放进 hard_constraints 就被硬过滤。"""

    request = _complete_request(
        hard_constraints=[
            {
                "domain": "hotel",
                "field": "quiet_score",
                "operator": ">=",
                "value": 0.9,
                "evidence": "一定要安静",
                "source": "llm",
            }
        ]
    )

    issue = validate_hard_constraint(request.hard_constraints[0])

    assert issue is not None
    assert issue["issue_type"] == "subjective_hard_constraint"
    assert issue["requires_clarification"] is False


def test_objective_hard_constraint_is_allowed():
    request = _complete_request(
        hard_constraints=[
            {
                "domain": "flight",
                "field": "is_direct",
                "operator": "==",
                "value": True,
                "evidence": "必须直飞",
                "source": "rule",
            }
        ]
    )

    assert validate_hard_constraint(request.hard_constraints[0]) is None


def test_trip_budget_alias_is_canonicalized_before_validation():
    """
    LLM 偶发输出 trip.budget 时，Schema 必须在 MissingInfoCheck 之前
    将它收敛为 trip.total_budget；否则预算请求会被误报为不支持能力。
    """

    request = _complete_request(
        budget=1000,
        hard_constraints=[
            {
                "domain": "trip",
                "field": "budget",
                "operator": "<=",
                "value": 1000,
                "evidence": "预算1000元",
                "source": "llm",
            }
        ],
    )

    constraint = request.hard_constraints[0]

    assert constraint.field == "total_budget"
    assert validate_hard_constraint(constraint) is None
    assert build_missing_info_result(
        request,
        capability_validator=SupportedCapabilityValidator(),
    )["can_continue"] is True


def test_unmapped_core_requirement_is_not_silently_ignored():
    result = build_missing_info_result(
        _complete_request(
            unmapped_requirements=[
                {
                    "text": "每天早上必须游泳",
                    "category_guess": "fitness_facility",
                    "impact": "itinerary",
                    "confidence": 0.7,
                    "requires_clarification": True,
                }
            ]
        ),
        capability_validator=SupportedCapabilityValidator(),
    )

    assert result["can_continue"] is False
    assert any(
        issue["issue_type"] == "unmapped_requirement"
        for issue in result["blocking_issues"]
    )


def test_build_missing_info_message():
    assert build_missing_info_message([]) == "信息完整，可以继续旅行规划。"
    assert build_missing_info_message(["start_date"]) == (
        "还需要补充出发日期，才能继续规划。"
    )
    assert build_missing_info_message(["origin", "start_date"]) == (
        "还需要补充出发地和出发日期，才能继续规划。"
    )
