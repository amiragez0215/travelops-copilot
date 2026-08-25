from __future__ import annotations

from typing import Any, Mapping

from app.validators.trip_request_validator import (
    REQUIRED_FIELD_SPECS,
    build_missing_info_message,
)


# ----------------------------------------------------------------------
# 缺失字段示例
# ----------------------------------------------------------------------
# ClarifyNode 面向用户追问时，不应该只说：
#   “请补充出发日期。”
#
# 更好的追问是：
#   “请补充你的出发日期，例如：2026-07-02。”
#
# 这些 example 会进入 questions，方便前端展示输入提示。
FIELD_EXAMPLES: dict[str, str] = {
    "origin": "例如：杭州",
    "destination": "例如：成都",
    "start_date": "例如：2026-07-02",
    "days": "例如：3天",
}


def build_clarification(
    missing_info_result: Mapping[str, Any] | None = None,
    missing_fields: list[str] | None = None,
    trip_request_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    构造 ClarifyNode 的 clarification 结构。

    所在层：
        app/formatters

    解决什么问题：
        MissingInfoCheckNode 只负责判断缺哪些字段。
        ClarifyNode 需要把这些缺失字段转成用户看得懂的追问。

    Args:
        missing_info_result:
            MissingInfoCheckNode 生成的详细结果。
            推荐优先使用这个字段，因为它包含 missing_details 和 message。

        missing_fields:
            兜底字段。
            如果没有 missing_info_result，也可以直接传缺失字段列表。

        trip_request_summary:
            当前已经抽取到的旅行需求摘要。
            方便前端或调试时知道用户已经提供了哪些信息。

    Returns:
        dict:
            适合写入 state["clarification"] 的结构。

    注意：
        这个函数不读 State，不写 State。
        它只是一个确定性格式化函数，方便单独测试。
    """

    resolved_missing_fields = _resolve_missing_fields(
        missing_info_result=missing_info_result,
        missing_fields=missing_fields,
    )

    resolved_summary = _resolve_trip_request_summary(
        missing_info_result=missing_info_result,
        trip_request_summary=trip_request_summary,
    )

    # MissingInfoCheckNode 可能不仅发现缺字段，还发现歧义、多人冲突或能力范围问题。
    # 优先使用 Validator 已经生成的 clarification_questions。
    questions = _resolve_clarification_questions(
        missing_info_result=missing_info_result,
        missing_fields=resolved_missing_fields,
    )

    can_continue = (
        missing_info_result.get("can_continue") is True
        if missing_info_result
        else not resolved_missing_fields and not questions
    )

    # 只有 Validator 明确允许继续时，才返回 ready。
    # 即使 missing_fields 为空，只要存在歧义或能力问题，仍然必须澄清。
    if can_continue:
        return {
            "type": "no_missing_info",
            "status": "ready",
            "message": "信息完整，无需澄清。",
            "missing_fields": [],
            "questions": [],
            "trip_request_summary": resolved_summary,
            "next_action": "continue",
            "available_actions": ["continue"],
        }

    message = _resolve_message(
        missing_info_result=missing_info_result,
        missing_fields=resolved_missing_fields,
    )

    clarification_type = _resolve_clarification_type(missing_info_result)

    return {
        "type": clarification_type,
        "status": "needs_clarification",
        "message": message,
        "missing_fields": resolved_missing_fields,
        "questions": questions,
        "trip_request_summary": resolved_summary,
        "next_action": "provide_missing_info",
        "available_actions": ["provide_missing_info", "cancel"],
    }


def build_clarification_final_response(
    clarification: Mapping[str, Any],
) -> dict[str, Any]:
    """
    构造给前端或 API 层使用的澄清响应。

    为什么 ClarifyNode 还写 final_response？

    你的整体流程是：
        ClarifyNode → FinalResponseNode → END

    所以最终项目里，FinalResponseNode 会统一格式化所有响应。
    但在开发早期，让 ClarifyNode 同时写一个可用的 final_response 有两个好处：

    1. 单独测试 ClarifyNode 时，可以直接看到用户会收到什么。
    2. 即使暂时还没实现 FinalResponseNode，接口也可以返回清晰的澄清问题。

    后续实现 FinalResponseNode 后，它可以基于 state["clarification"] 重新整理或覆盖 final_response。
    """

    status = clarification.get("status")

    if status == "ready":
        return {
            "status": "ready",
            "type": "no_missing_info",
            "message": clarification.get("message", "信息完整，无需澄清。"),
            "need_user_input": False,
            "clarification": dict(clarification),
            "available_actions": clarification.get("available_actions", ["continue"]),
        }

    clarification_type = str(clarification.get("type") or "")

    # 对普通缺字段和需求歧义，API 层统一使用 clarification_required。
    # unsupported_data_scope 保留独立类型，方便前端明确展示项目数据边界。
    response_type = (
        "unsupported_data_scope"
        if clarification_type == "unsupported_data_scope"
        else "clarification_required"
    )

    return {
        "status": "needs_clarification",
        "type": response_type,
        "message": clarification.get("message", "还需要补充信息，才能继续规划。"),
        "need_user_input": True,
        "clarification": dict(clarification),
        "available_actions": clarification.get(
            "available_actions",
            ["provide_missing_info", "cancel"],
        ),
    }


def _resolve_clarification_questions(
    missing_info_result: Mapping[str, Any] | None,
    missing_fields: list[str],
) -> list[dict[str, Any]]:
    """
    优先读取 Validator 生成的完整追问；没有时按缺失字段兜底。
    """

    if missing_info_result:
        questions = missing_info_result.get("clarification_questions")

        if isinstance(questions, list):
            return [
                dict(question)
                for question in questions
                if isinstance(question, Mapping)
            ]

    return [
        _build_question_for_field(field)
        for field in missing_fields
    ]


def _resolve_clarification_type(
    missing_info_result: Mapping[str, Any] | None,
) -> str:
    """
    根据 Validator 状态区分缺字段、歧义和数据能力范围问题。
    """

    status = (
        str(missing_info_result.get("status"))
        if missing_info_result
        else ""
    )

    if status == "unsupported_data_scope":
        return "unsupported_data_scope"

    if status == "clarification_required":
        return "requirement_clarification"

    return "missing_info"


def _resolve_missing_fields(
    missing_info_result: Mapping[str, Any] | None,
    missing_fields: list[str] | None,
) -> list[str]:
    """
    从 missing_info_result 或 missing_fields 中解析缺失字段。

    优先级：
    1. missing_info_result["missing_fields"]
    2. missing_fields 参数
    3. []
    """

    if missing_info_result:
        fields = missing_info_result.get("missing_fields")
        if isinstance(fields, list):
            return [str(field) for field in fields]

    if missing_fields:
        return [str(field) for field in missing_fields]

    return []


def _resolve_trip_request_summary(
    missing_info_result: Mapping[str, Any] | None,
    trip_request_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    解析当前 trip_request 摘要。

    missing_info_result 中通常已经包含 trip_request_summary。
    如果没有，则使用外部传入的 trip_request_summary。
    """

    if missing_info_result:
        summary = missing_info_result.get("trip_request_summary")
        if isinstance(summary, Mapping):
            return dict(summary)

    if isinstance(trip_request_summary, Mapping):
        return dict(trip_request_summary)

    return {}


def _resolve_message(
    missing_info_result: Mapping[str, Any] | None,
    missing_fields: list[str],
) -> str:
    """
    解析澄清消息。

    优先使用 MissingInfoCheckNode 已经生成的 message。
    如果没有，则用 build_missing_info_message 兜底。
    """

    if missing_info_result:
        message = missing_info_result.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

    return build_missing_info_message(missing_fields)


def _build_question_for_field(field: str) -> dict[str, Any]:
    """
    为某个缺失字段生成结构化追问。

    返回结构示例：
        {
            "field": "start_date",
            "label": "出发日期",
            "question": "请补充你的出发日期，例如：2026-07-02。",
            "reason": "天气、航班和酒店查询都需要明确日期。",
            "example": "例如：2026-07-02"
        }

    为什么要结构化 questions？

    因为后续前端可以根据 field 做定向输入框。
    Eval 也可以检查 ClarifyNode 是否正确指出缺失字段。
    """

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
        "example": FIELD_EXAMPLES.get(field, ""),
    }