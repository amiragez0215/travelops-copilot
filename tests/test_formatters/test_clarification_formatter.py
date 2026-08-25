from app.formatters.clarification_formatter import (
    build_clarification,
    build_clarification_final_response,
)


def test_build_clarification_from_missing_info_result():
    """
    使用 MissingInfoCheckNode 的完整结果生成 clarification。
    """

    missing_info_result = {
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

    clarification = build_clarification(missing_info_result)

    assert clarification["status"] == "needs_clarification"
    assert clarification["type"] == "missing_info"
    assert clarification["missing_fields"] == ["start_date"]
    assert clarification["message"] == "还需要补充出发日期，才能继续规划。"

    assert len(clarification["questions"]) == 1
    assert clarification["questions"][0]["field"] == "start_date"
    assert clarification["questions"][0]["label"] == "出发日期"
    assert clarification["questions"][0]["example"] == "例如：2026-07-02"

    assert clarification["trip_request_summary"]["origin"] == "杭州"
    assert clarification["available_actions"] == ["provide_missing_info", "cancel"]


def test_build_clarification_from_missing_fields_only():
    """
    即使没有完整 missing_info_result，只传 missing_fields 也能生成澄清问题。
    """

    clarification = build_clarification(
        missing_fields=["origin", "start_date"],
        trip_request_summary={
            "destination": "成都",
            "days": 3,
        },
    )

    assert clarification["status"] == "needs_clarification"
    assert clarification["missing_fields"] == ["origin", "start_date"]

    fields = [question["field"] for question in clarification["questions"]]
    assert fields == ["origin", "start_date"]

    assert "出发地" in clarification["message"]
    assert "出发日期" in clarification["message"]


def test_build_clarification_when_no_missing_fields():
    """
    字段完整时，formatter 返回 ready。

    实际 workflow 中 ClarifyNode 不应该在这个场景被调用，
    但 formatter 本身要保持健壮。
    """

    clarification = build_clarification(missing_fields=[])

    assert clarification["status"] == "ready"
    assert clarification["type"] == "no_missing_info"
    assert clarification["message"] == "信息完整，无需澄清。"
    assert clarification["questions"] == []


def test_build_clarification_final_response():
    """
    clarification 可以转换成前端可用的 final_response。
    """

    clarification = build_clarification(missing_fields=["start_date"])
    final_response = build_clarification_final_response(clarification)

    assert final_response["status"] == "needs_clarification"
    assert final_response["type"] == "clarification_required"
    assert final_response["need_user_input"] is True
    assert final_response["clarification"]["missing_fields"] == ["start_date"]


def test_build_ready_final_response():
    """
    ready 类型 clarification 也能转换成 final_response。
    """

    clarification = build_clarification(missing_fields=[])
    final_response = build_clarification_final_response(clarification)

    assert final_response["status"] == "ready"
    assert final_response["type"] == "no_missing_info"
    assert final_response["need_user_input"] is False