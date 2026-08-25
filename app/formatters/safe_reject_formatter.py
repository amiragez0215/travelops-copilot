from __future__ import annotations

from typing import Any, Mapping


DEFAULT_PROJECT_BOUNDARY = {
    "no_real_payment": True,
    "no_real_booking": True,
    "no_real_email": True,
    "no_shopping_cart": True,
    "draft_only": True,
}


DEFAULT_SAFE_ALTERNATIVES = [
    {
        "type": "trip_proposal_draft",
        "message": "我可以继续帮你生成旅行方案草稿，但不会真实预订、付款或出票。",
    },
    {
        "type": "revise_request",
        "message": "你可以修改请求，例如改成“帮我规划一个可参考的行程方案”。",
    },
    {
        "type": "travel_checklist_text",
        "message": "我可以生成文字版出行准备建议，但不会购买商品或发送通知。",
    },
]


def build_safe_reject_final_response(
    safety_result: Mapping[str, Any] | None,
    safety_flags: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    把 SafetyCheckNode 的结果转换成用户可读的安全拒绝响应。
    """

    resolved_result = _normalize_safety_result(safety_result)
    resolved_flags = _resolve_flags(resolved_result, safety_flags)

    blocked_actions = _resolve_blocked_actions(resolved_result, resolved_flags)
    message = _build_user_message(resolved_result, resolved_flags)

    return {
        "status": "blocked",
        "type": "safe_reject",
        "message": message,
        "reason": resolved_result.get("reason"),
        "safe_mode": resolved_result.get("safe_mode", "reject_or_draft_only"),
        "blocked_actions": blocked_actions,
        "flags": [_public_flag(flag) for flag in resolved_flags],
        "safe_alternatives": DEFAULT_SAFE_ALTERNATIVES,
        "available_actions": ["revise_request", "draft_only_plan", "cancel"],
        "need_user_input": True,
        "project_boundary": resolved_result.get(
            "project_boundary",
            DEFAULT_PROJECT_BOUNDARY,
        ),
    }


def _normalize_safety_result(
    safety_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    缺少 safety_result 时，构造保守阻断结果。
    """

    if isinstance(safety_result, Mapping):
        return dict(safety_result)

    return {
        "blocked": True,
        "reason": "安全检查结果缺失，系统不会继续执行旅行规划。",
        "safe_mode": "reject_or_draft_only",
        "flags": [],
        "blocked_actions": ["unknown_safety_state"],
        "project_boundary": DEFAULT_PROJECT_BOUNDARY,
    }


def _resolve_flags(
    safety_result: Mapping[str, Any],
    safety_flags: list[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """
    优先使用 state["safety_flags"]，否则使用 safety_result["flags"]。
    """

    if safety_flags:
        return [dict(flag) for flag in safety_flags]

    flags_from_result = safety_result.get("flags")
    if isinstance(flags_from_result, list):
        return [
            dict(flag)
            for flag in flags_from_result
            if isinstance(flag, Mapping)
        ]

    return []


def _resolve_blocked_actions(
    safety_result: Mapping[str, Any],
    flags: list[dict[str, Any]],
) -> list[str]:
    """
    从 safety_result 或 flags 中提取被禁止动作。
    """

    blocked_actions = safety_result.get("blocked_actions")
    if isinstance(blocked_actions, list) and blocked_actions:
        return [str(action) for action in blocked_actions]

    categories = [
        str(flag.get("category"))
        for flag in flags
        if flag.get("action") == "block" and flag.get("category")
    ]

    return list(dict.fromkeys(categories))


def _build_user_message(
    safety_result: Mapping[str, Any],
    flags: list[dict[str, Any]],
) -> str:
    """
    生成用户可读的安全拒绝文案。
    """

    reason = safety_result.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "请求包含当前项目不支持的真实外部操作。"

    suggestions = [
        str(flag.get("suggestion"))
        for flag in flags
        if flag.get("suggestion")
    ]

    unique_suggestions = list(dict.fromkeys(suggestions))

    if unique_suggestions:
        suggestion_text = " ".join(unique_suggestions[:2])
    else:
        suggestion_text = "我可以帮你生成旅行方案草稿，但不会执行真实外部操作。"

    return f"{reason} {suggestion_text}"


def _public_flag(flag: Mapping[str, Any]) -> dict[str, Any]:
    """
    只保留适合返回给前端的安全字段。
    """

    return {
        "rule_id": flag.get("rule_id"),
        "category": flag.get("category"),
        "level": flag.get("level"),
        "action": flag.get("action"),
        "message": flag.get("message"),
        "suggestion": flag.get("suggestion"),
        "evidence": flag.get("evidence"),
    }