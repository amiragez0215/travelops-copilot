from app.nodes.safe_reject_node import safe_reject_node


def test_safe_reject_node_success():
    """
    safety_result.blocked=True 时，SafeRejectNode 生成 final_response。
    """

    state = {
        "safety_result": {
            "blocked": True,
            "reason": "请求包含当前项目不支持的动作：real_payment。",
            "safe_mode": "reject_or_draft_only",
            "flags": [
                {
                    "rule_id": "real_payment_request",
                    "category": "real_payment",
                    "level": "high",
                    "action": "block",
                    "message": "请求包含真实支付动作，当前系统不能执行真实支付。",
                    "suggestion": "我可以帮你生成旅行方案和待审批草稿，但不会真实付款。",
                    "matched_keyword": "直接付款",
                    "evidence": "帮我直接付款并预订酒店",
                }
            ],
            "blocked_actions": ["real_payment"],
        },
        "safety_flags": [
            {
                "rule_id": "real_payment_request",
                "category": "real_payment",
                "level": "high",
                "action": "block",
                "message": "请求包含真实支付动作，当前系统不能执行真实支付。",
                "suggestion": "我可以帮你生成旅行方案和待审批草稿，但不会真实付款。",
                "matched_keyword": "直接付款",
                "evidence": "帮我直接付款并预订酒店",
            }
        ],
    }

    result = safe_reject_node(state)

    assert result["final_response"]["status"] == "blocked"
    assert result["final_response"]["type"] == "safe_reject"
    assert result["final_response"]["blocked_actions"] == ["real_payment"]
    assert result["itinerary_status"] == "rejected"

    assert result["trace"][0]["node_name"] == "safe_reject"
    assert result["trace"][0]["status"] == "success"


def test_safe_reject_node_missing_safety_result_returns_error():
    """
    没有 safety_result 时，节点返回兜底响应和 errors。
    """

    result = safe_reject_node({})

    assert result["final_response"]["status"] == "blocked"
    assert result["final_response"]["type"] == "safe_reject"
    assert result["itinerary_status"] == "rejected"

    assert result["errors"]
    assert result["errors"][0]["node"] == "safe_reject"
    assert result["trace"][0]["status"] == "failed"


def test_safe_reject_node_with_unblocked_result_returns_error():
    """
    blocked=False 时不应该进入 SafeRejectNode。
    """

    result = safe_reject_node(
        {
            "safety_result": {
                "blocked": False,
                "reason": None,
                "safe_mode": "draft_only",
                "flags": [],
            },
            "safety_flags": [],
        }
    )

    assert result["final_response"]["status"] == "blocked"
    assert result["errors"]
    assert result["errors"][0]["node"] == "safe_reject"
    assert result["trace"][0]["status"] == "failed"