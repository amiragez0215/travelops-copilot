from app.nodes.clarify_node import clarify_node


def test_clarify_node_from_missing_info_result():
    """
    正常路径：
    MissingInfoCheckNode 已经写入 missing_info_result，
    ClarifyNode 基于它生成 clarification 和 final_response。
    """

    state = {
        "missing_info_result": {
            "status": "missing_required_fields",
            "missing_fields": ["start_date"],
            "missing_details": [
                {
                    "field": "start_date",
                    "label": "出发日期",
                    "question": "请补充你的出发日期，例如：2026-07-02。",
                    "reason": "天气、航班和酒店查询都需要明确日期。",
                }
            ],
            "can_continue": False,
            "message": "还需要补充出发日期，才能继续规划。",
            "trip_request_summary": {
                "origin": "杭州",
                "destination": "成都",
                "start_date": None,
                "days": 3,
                "budget": 4000,
            },
        }
    }

    result = clarify_node(state)

    assert result["clarification"]["status"] == "needs_clarification"
    assert result["clarification"]["missing_fields"] == ["start_date"]
    assert result["final_response"]["status"] == "needs_clarification"
    assert result["final_response"]["type"] == "clarification_required"
    assert result["itinerary_status"] == "collecting_info"

    assert result["trace"][0]["node_name"] == "clarify"
    assert result["trace"][0]["status"] == "success"


def test_clarify_node_can_fallback_from_trip_request():
    """
    如果 state 中没有 missing_info_result，
    但有 trip_request，ClarifyNode 可以重新调用 validator 生成缺失信息。
    """

    state = {
        "trip_request": {
            "origin": "杭州",
            "destination": "成都",
            "start_date": None,
            "days": 3,
            "budget": 4000,
        }
    }

    result = clarify_node(state)

    assert result["clarification"]["missing_fields"] == ["start_date"]
    assert "出发日期" in result["clarification"]["message"]
    assert result["final_response"]["need_user_input"] is True
    assert result["trace"][0]["status"] == "success"


def test_clarify_node_can_fallback_from_missing_fields_only():
    """
    如果只有 missing_fields，也可以生成澄清问题。
    """

    state = {
        "missing_fields": ["origin", "start_date"]
    }

    result = clarify_node(state)

    assert result["clarification"]["missing_fields"] == ["origin", "start_date"]

    fields = [
        question["field"]
        for question in result["clarification"]["questions"]
    ]
    assert fields == ["origin", "start_date"]

    assert result["final_response"]["status"] == "needs_clarification"


def test_clarify_node_returns_error_when_no_context():
    """
    如果既没有 missing_info_result，也没有 missing_fields，也没有 trip_request，
    ClarifyNode 应该返回 fallback response 和 errors，而不是直接抛异常。
    """

    result = clarify_node({})

    assert result["clarification"]["status"] == "needs_clarification"
    assert result["final_response"]["status"] == "needs_clarification"
    assert result["errors"]
    assert result["errors"][0]["node"] == "clarify"
    assert result["trace"][0]["status"] == "failed"