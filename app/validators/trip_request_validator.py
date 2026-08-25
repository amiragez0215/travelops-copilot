from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Mapping

from app.schemas.request_constraint_schema import HardConstraint
from app.schemas.trip_request_schema import TripRequest
from app.validators.request_capability_validator import (
    RequestCapabilityValidator,
)


@dataclass(frozen=True)
class RequiredFieldSpec:
    """
    必填字段的用户提示配置。

    MissingInfoCheckNode 负责判断缺什么，ClarifyNode 负责如何追问。
    将 label/question/reason 集中在这里，可以避免多个节点重复写文案。
    """

    field: str
    label: str
    question: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        """转成适合写入 TravelState 的 dict。"""

        return asdict(self)


REQUIRED_FIELD_SPECS: dict[str, RequiredFieldSpec] = {
    "origin": RequiredFieldSpec(
        field="origin",
        label="出发地",
        question="请补充你的出发城市，例如：杭州。",
        reason="航班查询需要知道从哪里出发。",
    ),
    "destination": RequiredFieldSpec(
        field="destination",
        label="目的地",
        question="请补充你的目的地城市，例如：成都。",
        reason="天气、酒店和攻略检索都需要目的地。",
    ),
    "start_date": RequiredFieldSpec(
        field="start_date",
        label="出发日期",
        question="请补充具体出发日期，例如：2026-07-02。",
        reason="天气、航班和酒店查询需要精确到具体日期。",
    ),
    "days": RequiredFieldSpec(
        field="days",
        label="旅行天数",
        question="请补充你计划旅行几天，例如：3天。",
        reason="住宿晚数、预算安排和每日行程都需要旅行天数。",
    ),
    "room_count": RequiredFieldSpec(
        field="room_count",
        label="房间数量",
        question="多人出行计划预订几间房？例如：2人1间房或2人2间房。",
        reason="酒店总价需要人数和房间数量。",
    ),
}


# ----------------------------------------------------------------------
# 当前项目允许执行的硬约束白名单
# ----------------------------------------------------------------------
# LLM 可以理解开放表达，但系统只执行这里明确支持的结构化约束。
SUPPORTED_HARD_CONSTRAINTS: dict[
    tuple[str, str],
    set[str],
] = {
    ("trip", "total_budget"): {"<="},
    ("flight", "is_direct"): {"=="},
    ("flight", "depart_time"): {">=", ">"},
    ("flight", "arrive_time"): {"<=", "<"},
    ("flight", "price"): {"<="},
    ("flight", "flight_no"): {"=="},
    ("hotel", "price_per_night"): {"<="},
    ("hotel", "hotel_id"): {"=="},
    ("hotel", "near_subway"): {"=="},
}

# 这些是连续、主观特征，不允许进入硬约束。
SUBJECTIVE_HARD_FIELDS = {
    ("hotel", "quiet"),
    ("hotel", "quiet_score"),
    ("hotel", "cleanliness"),
    ("hotel", "cleanliness_score"),
    ("hotel", "high_rating"),
    ("activity", "food"),
}


def build_missing_info_result(
    trip_request_data: TripRequest | Mapping[str, Any],
    *,
    capability_validator: RequestCapabilityValidator | None = None,
) -> dict[str, Any]:
    """
    执行 InputExtract 后的确定性请求校验。

    检查顺序：
        1. Pydantic 将 dict 还原为 TripRequest。
        2. 检查核心缺失字段。
        3. 检查日期、房间数、硬约束白名单和需求冲突。
        4. 核心字段完整时检查当前 Mock 数据能力范围。
        5. 汇总为 ClarifyNode 可以直接使用的结构。

    证据不足、航班无候选等后续业务问题不在这里处理。
    """

    trip_request = coerce_trip_request(trip_request_data)

    # 1. 检查核心字段和多人房间数量。
    missing_fields = check_missing_trip_request_fields(trip_request)

    missing_details = [
        REQUIRED_FIELD_SPECS[field].to_dict()
        for field in missing_fields
        if field in REQUIRED_FIELD_SPECS
    ]

    # 2. 检查 Schema 之外的确定性业务一致性。
    validation_issues = validate_trip_request_semantics(trip_request)

    # 3. 只有核心字段完整时才检查 Mock 能力范围，避免重复提示。
    if not missing_fields:
        active_capability_validator = (
            capability_validator
            or RequestCapabilityValidator()
        )
        capability_result = active_capability_validator.validate(
            trip_request.to_state_dict()
        )
    else:
        capability_result = {
            "status": "not_checked",
            "supported": True,
            "reason": "核心字段尚未完整。",
            "issues": [],
            "supported_destinations": [],
        }

    capability_issues = [
        dict(item)
        for item in capability_result.get("issues", [])
        if isinstance(item, Mapping)
    ]

    blocking_issues = [
        *[
            item
            for item in validation_issues
            if item.get("requires_clarification") is True
        ],
        *[
            item
            for item in capability_issues
            if item.get("requires_clarification") is True
        ],
    ]

    can_continue = not missing_fields and not blocking_issues

    status = _resolve_validation_status(
        missing_fields=missing_fields,
        validation_issues=validation_issues,
        capability_result=capability_result,
    )

    clarification_questions = [
        *[
            _question_from_missing_field(field)
            for field in missing_fields
        ],
        *[
            _question_from_issue(issue)
            for issue in blocking_issues
        ],
    ]

    return {
        "status": status,
        "required_fields": [
            "origin",
            "destination",
            "start_date",
            "days",
        ],
        "missing_fields": missing_fields,
        "missing_details": missing_details,
        "validation_issues": validation_issues,
        "blocking_issues": blocking_issues,
        "capability_result": capability_result,
        "clarification_questions": clarification_questions,
        "can_continue": can_continue,
        "message": build_validation_message(
            missing_fields=missing_fields,
            blocking_issues=blocking_issues,
            capability_result=capability_result,
        ),
        "trip_request_summary": {
            "origin": trip_request.origin,
            "destination": trip_request.destination,
            "start_date": (
                trip_request.start_date.isoformat()
                if trip_request.start_date
                else None
            ),
            "end_date": (
                trip_request.end_date.isoformat()
                if trip_request.end_date
                else None
            ),
            "partial_date": (
                trip_request.partial_date.to_state_dict()
                if trip_request.partial_date
                else None
            ),
            "days": trip_request.days,
            "budget": trip_request.budget,
            "people_count": trip_request.people_count,
            "room_count": trip_request.room_count,
            "named_constraints": [
                item.to_state_dict()
                for item in trip_request.named_constraints
            ],
        },
    }


def coerce_trip_request(
    trip_request_data: TripRequest | Mapping[str, Any],
) -> TripRequest:
    """把 State 中的 trip_request dict 重新校验为 TripRequest。"""

    if isinstance(trip_request_data, TripRequest):
        return trip_request_data

    if not isinstance(trip_request_data, Mapping):
        raise TypeError("trip_request 必须是 dict 或 TripRequest 对象")

    normalized_data = _normalize_empty_strings(
        dict(trip_request_data)
    )

    return TripRequest(**normalized_data)


def check_missing_trip_request_fields(
    trip_request: TripRequest,
) -> list[str]:
    """
    返回必须向用户补充的字段。

    当前 v1：
        - origin、destination、start_date、days 必填。
        - 单人默认 room_count=1。
        - 多人出行没有说明 room_count 时必须澄清。
        - budget 不是必填字段。
    """

    missing: list[str] = []

    if not _has_text(trip_request.origin):
        missing.append("origin")

    if not _has_text(trip_request.destination):
        missing.append("destination")

    if trip_request.start_date is None:
        missing.append("start_date")

    if trip_request.days is None:
        missing.append("days")

    if (
        trip_request.people_count > 1
        and trip_request.room_count is None
    ):
        missing.append("room_count")

    return missing


def validate_trip_request_semantics(
    trip_request: TripRequest,
) -> list[dict[str, Any]]:
    """
    检查日期、硬约束、开放需求和多人冲突。

    返回的问题都是结构化 dict，ClarifyNode 可以直接生成追问。
    """

    issues: list[dict[str, Any]] = []

    # 1. 日期范围必须正向。
    if (
        trip_request.start_date
        and trip_request.end_date
        and trip_request.end_date < trip_request.start_date
    ):
        issues.append(
            {
                "issue_type": "invalid_date_range",
                "message": "返程日期不能早于出发日期。",
                "requires_clarification": True,
                "suggested_question": "请重新确认出发日期和返程日期。",
            }
        )

    # 2. 明确日期范围和 days 必须一致。
    if (
        trip_request.start_date
        and trip_request.end_date
        and trip_request.days
    ):
        calculated_days = (
            trip_request.end_date
            - trip_request.start_date
        ).days + 1

        if calculated_days != trip_request.days:
            issues.append(
                {
                    "issue_type": "date_days_mismatch",
                    "message": (
                        f"日期范围对应 {calculated_days} 天，"
                        f"但用户需求中记录为 {trip_request.days} 天。"
                    ),
                    "requires_clarification": True,
                    "suggested_question": "请确认旅行日期范围或旅行天数。",
                    "details": {
                        "calculated_days": calculated_days,
                        "requested_days": trip_request.days,
                    },
                }
            )

    # 3. 房间数不能大于人数；这种输入通常是抽取错误或需要确认。
    if (
        trip_request.room_count is not None
        and trip_request.room_count > trip_request.people_count
    ):
        issues.append(
            {
                "issue_type": "room_count_exceeds_people_count",
                "message": "房间数量大于出行人数，请确认房间安排。",
                "requires_clarification": True,
                "suggested_question": "请确认出行人数和房间数量。",
            }
        )

    # 4. 将 LLM 已经发现的歧义和冲突加入统一问题列表。
    for issue in trip_request.requirement_issues:
        issues.append(issue.to_state_dict())

    # 5. 无法映射且会影响核心业务的要求不能静默忽略。
    for requirement in trip_request.unmapped_requirements:
        if not requirement.requires_clarification:
            continue

        issues.append(
            {
                "issue_type": "unmapped_requirement",
                "message": f"当前系统无法确定性处理该要求：{requirement.text}",
                "requires_clarification": True,
                "suggested_question": (
                    "这个要求会影响旅行选择，请换一种更具体、可验证的方式说明。"
                ),
                "details": requirement.to_state_dict(),
            }
        )

    # 6. 检查硬约束是否属于项目执行白名单。
    for constraint in trip_request.hard_constraints:
        issue = validate_hard_constraint(constraint)
        if issue:
            issues.append(issue)

    # 7. 当前 v1 不支持多人各自指定不同 required 酒店。
    required_hotels = [
        item
        for item in trip_request.named_constraints
        if item.entity_type == "hotel"
        and item.constraint_mode == "required"
    ]

    unique_required_hotels = {
        item.entity_name.strip().casefold()
        for item in required_hotels
        if item.entity_name.strip()
    }

    if len(unique_required_hotels) > 1:
        issues.append(
            {
                "issue_type": "multiple_required_hotels",
                "message": "当前 v1 全程只支持一家酒店，但请求中指定了多家必住酒店。",
                "requires_clarification": True,
                "suggested_question": "请选择一家酒店作为必住酒店，或将其他酒店改为备选。",
                "details": {
                    "hotel_names": [
                        item.entity_name
                        for item in required_hotels
                    ]
                },
            }
        )

    return _dedupe_issues(issues)


def validate_hard_constraint(
    constraint: HardConstraint,
) -> dict[str, Any] | None:
    """
    检查一条硬约束是否可由当前 Mock 数据确定性验证。

    主观字段会被拒绝并要求改成偏好，而不是偷偷当作过滤条件。
    """

    key = (constraint.domain, constraint.field)

    if key in SUBJECTIVE_HARD_FIELDS:
        return {
            "issue_type": "subjective_hard_constraint",
            "message": (
                f"{constraint.domain}.{constraint.field} 是主观连续属性，"
                "不能作为硬过滤条件；应改成高权重偏好。"
            ),
            "requires_clarification": False,
            "suggested_question": "",
            "details": constraint.to_state_dict(),
        }

    supported_operators = SUPPORTED_HARD_CONSTRAINTS.get(key)

    if supported_operators is None:
        return {
            "issue_type": "unsupported_hard_constraint",
            "message": (
                f"当前项目不能验证硬约束 {constraint.domain}.{constraint.field}。"
            ),
            "requires_clarification": True,
            "suggested_question": (
                "请将该要求改成偏好，或改成当前数据源能够验证的价格、时间、直飞或指定实体条件。"
            ),
            "details": constraint.to_state_dict(),
        }

    if constraint.operator not in supported_operators:
        return {
            "issue_type": "unsupported_constraint_operator",
            "message": (
                f"硬约束 {constraint.domain}.{constraint.field} 不支持操作符 "
                f"{constraint.operator}。"
            ),
            "requires_clarification": True,
            "suggested_question": "请重新说明这个限制条件。",
            "details": constraint.to_state_dict(),
        }

    # 时间字段统一要求 HH:MM。
    if constraint.field in {"depart_time", "arrive_time"}:
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", str(constraint.value)):
            return {
                "issue_type": "invalid_time_constraint",
                "message": "航班时间硬约束必须使用 HH:MM 格式。",
                "requires_clarification": True,
                "suggested_question": "请使用例如 08:00 或 20:30 的时间表达。",
                "details": constraint.to_state_dict(),
            }

    # 价格和预算字段必须是非负数。
    if constraint.field in {
        "total_budget",
        "price",
        "price_per_night",
    }:
        try:
            numeric_value = float(constraint.value)
        except (TypeError, ValueError):
            numeric_value = -1

        if numeric_value < 0:
            return {
                "issue_type": "invalid_numeric_constraint",
                "message": "价格或预算硬约束必须是非负数字。",
                "requires_clarification": True,
                "suggested_question": "请重新提供有效的金额限制。",
                "details": constraint.to_state_dict(),
            }

    return None


def build_missing_info_message(
    missing_fields: list[str],
) -> str:
    """兼容原测试的缺失字段消息入口。"""

    if not missing_fields:
        return "信息完整，可以继续旅行规划。"

    labels = [
        REQUIRED_FIELD_SPECS[field].label
        for field in missing_fields
        if field in REQUIRED_FIELD_SPECS
    ]

    if not labels:
        return "还需要补充必要信息，才能继续规划。"

    return f"还需要补充{_join_chinese_labels(labels)}，才能继续规划。"


def build_validation_message(
    *,
    missing_fields: list[str],
    blocking_issues: list[dict[str, Any]],
    capability_result: Mapping[str, Any],
) -> str:
    """生成 MissingInfoCheckNode 的总体说明。"""

    if missing_fields:
        return build_missing_info_message(missing_fields)

    if blocking_issues:
        first_message = str(
            blocking_issues[0].get("message")
            or "旅行需求存在需要确认的问题。"
        )
        return first_message

    if capability_result.get("supported") is False:
        return "当前模拟数据暂时无法支持这个旅行需求。"

    return "信息完整，可以继续旅行规划。"


def _resolve_validation_status(
    *,
    missing_fields: list[str],
    validation_issues: list[dict[str, Any]],
    capability_result: Mapping[str, Any],
) -> str:
    """根据缺字段、语义问题和能力范围生成稳定状态。"""

    if missing_fields:
        return "missing_required_fields"

    if capability_result.get("supported") is False:
        return "unsupported_data_scope"

    if any(
        issue.get("requires_clarification") is True
        for issue in validation_issues
    ):
        return "clarification_required"

    return "ready"


def _question_from_missing_field(field: str) -> dict[str, Any]:
    """把缺失字段配置转成 ClarifyNode 可直接使用的问题。"""

    spec = REQUIRED_FIELD_SPECS.get(field)

    if spec is None:
        return {
            "field": field,
            "label": field,
            "question": f"请补充 {field}。",
            "reason": "继续规划需要这个字段。",
            "example": "",
        }

    return {
        "field": spec.field,
        "label": spec.label,
        "question": spec.question,
        "reason": spec.reason,
        "example": "",
    }


def _question_from_issue(
    issue: Mapping[str, Any],
) -> dict[str, Any]:
    """把歧义、冲突或能力问题转换成结构化追问。"""

    return {
        "field": str(issue.get("issue_type") or "requirement_issue"),
        "label": "需求确认",
        "question": str(
            issue.get("suggested_question")
            or issue.get("message")
            or "请进一步说明这个旅行要求。"
        ),
        "reason": str(
            issue.get("message")
            or "继续规划前需要确认这个要求。"
        ),
        "example": "",
    }


def _normalize_empty_strings(
    data: dict[str, Any],
) -> dict[str, Any]:
    """顶层空字符串统一转成 None。"""

    normalized: dict[str, Any] = {}

    for key, value in data.items():
        if isinstance(value, str) and not value.strip():
            normalized[key] = None
        else:
            normalized[key] = value

    return normalized


def _has_text(value: str | None) -> bool:
    """判断字符串是否包含有效内容。"""

    return isinstance(value, str) and bool(value.strip())


def _join_chinese_labels(labels: list[str]) -> str:
    """把中文字段名拼成自然表达。"""

    if len(labels) == 1:
        return labels[0]

    if len(labels) == 2:
        return f"{labels[0]}和{labels[1]}"

    return "、".join(labels[:-1]) + f"和{labels[-1]}"


def _dedupe_issues(
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按 issue_type + message 去重，保持第一次出现顺序。"""

    seen: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []

    for issue in issues:
        key = (
            str(issue.get("issue_type") or ""),
            str(issue.get("message") or ""),
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(issue)

    return output
