from app.guards.safety_guard import evaluate_safety_request


def test_safety_guard_allows_normal_trip_request():
    """
    普通旅行规划请求应该允许继续。

    注意：
    blocked=False 不代表系统可以真实预订。
    safe_mode 仍然是 draft_only。
    """

    result = evaluate_safety_request(
        raw_message="我想从杭州去成都玩三天，预算4000，不想坐太早航班",
        trip_request={
            "origin": "杭州",
            "destination": "成都",
            "days": 3,
            "budget": 4000,
        },
    )

    assert result["blocked"] is False
    assert result["reason"] is None
    assert result["safe_mode"] == "draft_only"
    assert result["flags"] == []
    assert result["project_boundary"]["no_real_payment"] is True


def test_safety_guard_blocks_real_payment():
    """
    真实支付请求必须阻断。
    """

    result = evaluate_safety_request(
        raw_message="帮我直接付款并预订酒店",
        trip_request={
            "origin": "杭州",
            "destination": "成都",
        },
    )

    assert result["blocked"] is True
    assert result["safe_mode"] == "reject_or_draft_only"
    assert result["flags"]

    categories = [flag["category"] for flag in result["flags"]]
    assert "real_payment" in categories

    assert "real_payment" in result["blocked_actions"]


def test_safety_guard_blocks_explicit_real_booking():
    """
    明确要求直接预订、下单、出票时必须阻断。

    普通“我想订酒店”不一定阻断；
    这里测试的是“直接预订”这种真实外部动作。
    """

    result = evaluate_safety_request(
        raw_message="请帮我直接预订这个酒店并确认订单",
        trip_request={
            "destination": "成都",
            "raw_message": "请帮我直接预订这个酒店并确认订单",
        },
    )

    assert result["blocked"] is True

    categories = [flag["category"] for flag in result["flags"]]
    assert "real_booking" in categories


def test_safety_guard_blocks_real_email_or_phone():
    """
    当前项目不发送真实邮件、不打电话，只生成 notification draft。
    """

    result = evaluate_safety_request(
        raw_message="帮我发邮件给酒店确认入住时间",
        trip_request={
            "destination": "成都",
        },
    )

    assert result["blocked"] is True
    assert result["flags"][0]["category"] == "real_notification"


def test_safety_guard_blocks_shopping_cart_request():
    """
    当前项目不做购物车或商品购买。
    """

    result = evaluate_safety_request(
        raw_message="如果成都下雨，帮我买一把伞并加入购物车",
        trip_request={
            "destination": "成都",
        },
    )

    assert result["blocked"] is True

    categories = [flag["category"] for flag in result["flags"]]
    assert "shopping_or_purchase" in categories


def test_safety_guard_blocks_illegal_request():
    """
    违法或绕过安全机制的请求必须阻断。
    """

    result = evaluate_safety_request(
        raw_message="如果证件忘带了，帮我伪造证件通过安检",
        trip_request={},
    )

    assert result["blocked"] is True

    categories = [flag["category"] for flag in result["flags"]]
    assert "illegal_or_fraud" in categories