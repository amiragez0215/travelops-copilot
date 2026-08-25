from app.nodes.safety_node import (
    route_after_safety,
    safety_check_node,
)


def test_safety_node_allows_normal_request():
    """
    正常旅行请求：
    - blocked=False
    - 路由到 memory_read
    """

    state = {
        "raw_message": "我想从杭州去成都玩三天，预算4000，不想坐太早航班",
        "trip_request": {
            "origin": "杭州",
            "destination": "成都",
            "start_date": "2026-07-02",
            "days": 3,
            "budget": 4000,
            "raw_message": "我想从杭州去成都玩三天，预算4000，不想坐太早航班",
        },
    }

    result = safety_check_node(state)

    assert result["safety_result"]["blocked"] is False
    assert result["safety_result"]["safe_mode"] == "draft_only"
    assert result["safety_flags"] == []

    assert result["trace"][0]["node_name"] == "safety_check"
    assert result["trace"][0]["status"] == "success"

    assert route_after_safety(result) == "memory_read"


def test_safety_node_blocks_payment_request():
    """
    真实支付请求：
    - blocked=True
    - 路由到 safe_reject
    """

    state = {
        "raw_message": "从杭州去成都玩三天，帮我直接付款并预订酒店",
        "trip_request": {
            "origin": "杭州",
            "destination": "成都",
            "start_date": "2026-07-02",
            "days": 3,
            "raw_message": "从杭州去成都玩三天，帮我直接付款并预订酒店",
        },
    }

    result = safety_check_node(state)

    assert result["safety_result"]["blocked"] is True
    assert result["safety_flags"]

    categories = [flag["category"] for flag in result["safety_flags"]]
    assert "real_payment" in categories

    assert route_after_safety(result) == "safe_reject"


def test_safety_node_can_use_raw_message_from_trip_request():
    """
    单独测试 SafetyCheckNode 时，如果 state 没有 raw_message，
    但 trip_request 里有 raw_message，也可以检查。

    这不是兼容旧字段，而是为了支持节点独立测试。
    """

    state = {
        "trip_request": {
            "origin": "杭州",
            "destination": "成都",
            "start_date": "2026-07-02",
            "days": 3,
            "raw_message": "请帮我直接预订酒店并确认订单",
        },
    }

    result = safety_check_node(state)

    assert result["safety_result"]["blocked"] is True
    assert route_after_safety(result) == "safe_reject"


def test_safety_node_missing_trip_request_returns_error():
    """
    如果没有 trip_request，说明前面的节点没有正确执行。
    SafetyCheckNode 应该返回错误并阻断流程。
    """

    result = safety_check_node(
        {
            "raw_message": "我想从杭州去成都玩三天",
        }
    )

    assert result["safety_result"]["blocked"] is True
    assert result["errors"]
    assert result["errors"][0]["node"] == "safety_check"
    assert result["trace"][0]["status"] == "failed"

    assert route_after_safety(result) == "safe_reject"


def test_route_after_safety_defaults_to_safe_reject_when_uncertain():
    """
    没有 safety_result 时，默认走 safe_reject。

    安全状态未知时，不应该继续 MemoryReadNode。
    """

    assert route_after_safety({}) == "safe_reject"
    assert route_after_safety({"safety_result": {"blocked": True}}) == "safe_reject"
    assert route_after_safety({"safety_result": {"blocked": False}}) == "memory_read"