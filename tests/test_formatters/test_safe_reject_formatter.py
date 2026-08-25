from app.formatters.safe_reject_formatter import build_safe_reject_final_response


def test_build_safe_reject_final_response_with_payment_flag():
    safety_result = {
        "blocked": True,
        "reason": "请求包含当前项目不支持的动作：real_payment。",
        "safe_mode": "reject_or_draft_only",
        "blocked_actions": ["real_payment"],
        "project_boundary": {
            "no_real_payment": True,
            "no_real_booking": True,
            "no_real_email": True,
            "no_shopping_cart": True,
            "draft_only": True,
        },
    }

    safety_flags = [
        {
            "rule_id": "real_payment_request",
            "category": "real_payment",
            "level": "high",
            "action": "block",
            "message": "请求包含真实支付动作，当前系统不能执行真实支付。",
            "suggestion": "我可以帮你生成旅行方案草稿，但不会真实付款。",
            "matched_keyword": "直接付款",
            "evidence": "帮我直接付款并预订酒店",
        }
    ]

    response = build_safe_reject_final_response(
        safety_result=safety_result,
        safety_flags=safety_flags,
    )

    assert response["status"] == "blocked"
    assert response["type"] == "safe_reject"
    assert response["need_user_input"] is True
    assert response["safe_mode"] == "reject_or_draft_only"
    assert response["blocked_actions"] == ["real_payment"]

    assert response["flags"][0]["category"] == "real_payment"
    assert "真实支付" in response["flags"][0]["message"]
    assert "真实付款" in response["message"]


def test_build_safe_reject_final_response_without_flags():
    response = build_safe_reject_final_response(
        safety_result={
            "blocked": True,
            "reason": "安全检查结果缺失，系统不会继续执行旅行规划。",
            "safe_mode": "reject_or_draft_only",
            "blocked_actions": ["unknown_safety_state"],
        },
        safety_flags=[],
    )

    assert response["status"] == "blocked"
    assert response["type"] == "safe_reject"
    assert response["blocked_actions"] == ["unknown_safety_state"]
    assert response["safe_alternatives"]
    assert "draft_only_plan" in response["available_actions"]


def test_build_safe_reject_final_response_with_none_result():
    response = build_safe_reject_final_response(
        safety_result=None,
        safety_flags=None,
    )

    assert response["status"] == "blocked"
    assert response["type"] == "safe_reject"
    assert response["blocked_actions"] == ["unknown_safety_state"]
    assert response["project_boundary"]["draft_only"] is True