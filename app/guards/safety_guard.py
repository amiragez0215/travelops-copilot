from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


SafetyLevel = Literal["low", "medium", "high"]
SafetyAction = Literal["allow", "warn", "block"]


@dataclass(frozen=True)
class SafetyRule:
    """
    单条安全规则。
    """

    rule_id: str
    category: str
    level: SafetyLevel
    action: SafetyAction
    keywords: tuple[str, ...]
    message: str
    suggestion: str

    def to_dict(self) -> dict[str, Any]:
        """
        转成 dict，方便写入 State。
        """

        data = asdict(self)
        data["keywords"] = list(self.keywords)
        return data


SAFETY_RULES: tuple[SafetyRule, ...] = (
    SafetyRule(
        rule_id="real_payment_request",
        category="real_payment",
        level="high",
        action="block",
        keywords=(
            "直接付款",
            "帮我付款",
            "替我付款",
            "现在付款",
            "立即付款",
            "支付订单",
            "帮我支付",
            "扣款",
            "用我的信用卡",
            "用我信用卡",
            "支付宝付款",
            "微信支付",
            "pay for me",
            "make payment",
        ),
        message="请求包含真实支付动作，当前系统不能执行真实支付。",
        suggestion="我可以帮你生成旅行方案草稿，但不会真实付款。",
    ),
    SafetyRule(
        rule_id="real_booking_request",
        category="real_booking",
        level="high",
        action="block",
        keywords=(
            "直接预订",
            "马上预订",
            "立即预订",
            "直接下单",
            "帮我下单",
            "确认订单",
            "完成预订",
            "订好酒店",
            "订好机票",
            "帮我订好",
            "直接出票",
            "帮我出票",
            "锁定房间",
            "book it now",
            "confirm booking",
        ),
        message="请求包含真实预订或出票动作，当前系统不能执行真实预订。",
        suggestion="我可以帮你生成旅行方案草稿，但不会向外部平台下单或出票。",
    ),
    SafetyRule(
        rule_id="real_email_or_message_request",
        category="real_notification",
        level="high",
        action="block",
        keywords=(
            "发邮件给酒店",
            "发送邮件",
            "帮我发邮件",
            "给酒店打电话",
            "帮我打电话",
            "发送短信",
            "帮我通知酒店",
            "同步到日历",
            "send email",
            "send message",
            "call the hotel",
        ),
        message="请求包含真实邮件、短信、电话或日历动作，当前系统不能执行真实通知。",
        suggestion="我可以帮你生成文字提醒建议，但不会真实发送邮件、短信或电话。",
    ),
    SafetyRule(
        rule_id="shopping_cart_or_purchase_request",
        category="shopping_or_purchase",
        level="medium",
        action="block",
        keywords=(
            "加入购物车",
            "下单雨伞",
            "帮我买",
            "购买雨伞",
            "购买行李箱",
            "买一把伞",
            "下单商品",
            "购物车",
            "add to cart",
            "buy it",
        ),
        message="请求包含购物车或商品购买动作，当前项目不做购物车和商品购买。",
        suggestion="我可以给你生成出行清单建议，但不会购买商品。",
    ),
    SafetyRule(
        rule_id="illegal_or_fraud_request",
        category="illegal_or_fraud",
        level="high",
        action="block",
        keywords=(
            "伪造证件",
            "假身份证",
            "逃票",
            "绕过安检",
            "绕过验证码",
            "破解",
            "盗刷",
            "fake id",
            "bypass security",
        ),
        message="请求包含违法、欺诈或绕过安全机制的内容，系统不能协助。",
        suggestion="请遵守当地法律法规和平台规则。",
    ),
)


def evaluate_safety_request(
    raw_message: str,
    trip_request: Mapping[str, Any] | None = None,
    rules: tuple[SafetyRule, ...] = SAFETY_RULES,
) -> dict[str, Any]:
    """
    对用户请求进行安全检查。
    """

    combined_text = _build_combined_text(raw_message, trip_request)
    flags = _match_safety_rules(combined_text, rules)

    blocked = any(flag["action"] == "block" for flag in flags)

    return {
        "blocked": blocked,
        "reason": _build_block_reason(flags) if blocked else None,
        "safe_mode": "reject_or_draft_only" if blocked else "draft_only",
        "flags": flags,
        "allowed_actions": _build_allowed_actions(blocked),
        "blocked_actions": _build_blocked_actions(flags),
        "project_boundary": {
            "no_real_payment": True,
            "no_real_booking": True,
            "no_real_email": True,
            "no_shopping_cart": True,
            "draft_only": True,
        },
    }


def _build_combined_text(
    raw_message: str,
    trip_request: Mapping[str, Any] | None,
) -> str:
    """
    合并原始输入和 trip_request 中的约束文本。
    """

    pieces: list[str] = [raw_message or ""]

    if isinstance(trip_request, Mapping):
        raw_from_request = trip_request.get("raw_message")
        if isinstance(raw_from_request, str):
            pieces.append(raw_from_request)

        raw_constraints = trip_request.get("raw_constraints")
        if isinstance(raw_constraints, list):
            pieces.extend(str(item) for item in raw_constraints)

    return " ".join(piece for piece in pieces if piece)


def _match_safety_rules(
    text: str,
    rules: tuple[SafetyRule, ...],
) -> list[dict[str, Any]]:
    """
    遍历安全规则，返回命中的 flags。
    """

    flags: list[dict[str, Any]] = []

    for rule in rules:
        matched_keyword = _find_first_keyword(text, rule.keywords)
        if matched_keyword is None:
            continue

        flags.append(
            {
                "rule_id": rule.rule_id,
                "category": rule.category,
                "level": rule.level,
                "action": rule.action,
                "message": rule.message,
                "suggestion": rule.suggestion,
                "matched_keyword": matched_keyword,
                "evidence": _build_evidence_snippet(text, matched_keyword),
            }
        )

    return flags


def _find_first_keyword(
    text: str,
    keywords: tuple[str, ...],
) -> str | None:
    """
    返回第一个命中的关键词。
    """

    normalized_text = text.lower()
    compact_text = re.sub(r"\s+", "", normalized_text)

    for keyword in keywords:
        normalized_keyword = keyword.lower()
        compact_keyword = re.sub(r"\s+", "", normalized_keyword)

        if normalized_keyword in normalized_text:
            return keyword

        if compact_keyword in compact_text:
            return keyword

    return None


def _build_evidence_snippet(
    text: str,
    keyword: str,
    window: int = 18,
) -> str:
    """
    截取命中关键词附近的证据片段。
    """

    if not text:
        return ""

    index = text.lower().find(keyword.lower())

    if index == -1:
        return text[: window * 2]

    start = max(0, index - window)
    end = min(len(text), index + len(keyword) + window)

    return text[start:end]


def _build_block_reason(flags: list[dict[str, Any]]) -> str:
    """
    生成阻断原因。
    """

    categories = [
        str(flag["category"])
        for flag in flags
        if flag.get("action") == "block"
    ]

    unique_categories = list(dict.fromkeys(categories))

    if not unique_categories:
        return "请求包含当前系统不支持的动作。"

    return (
        f"请求包含当前项目不支持的动作：{'、'.join(unique_categories)}。"
        "当前系统只能生成旅行方案草稿，不能执行真实外部操作。"
    )


def _build_allowed_actions(blocked: bool) -> list[str]:
    """
    返回安全模式下允许的动作。
    """

    if blocked:
        return [
            "generate_safe_explanation",
            "offer_draft_only_plan",
            "revise_request",
            "cancel",
        ]

    return [
        "generate_trip_proposal",
        "query_mock_weather",
        "query_mock_flights",
        "query_mock_hotels",
        "retrieve_rag_evidence",
        "wait_for_user_decision",
    ]


def _build_blocked_actions(flags: list[dict[str, Any]]) -> list[str]:
    """
    返回被阻断的动作类别。
    """

    return list(
        dict.fromkeys(
            str(flag["category"])
            for flag in flags
            if flag.get("action") == "block"
        )
    )