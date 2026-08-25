from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from tests.fakes.plan_trip_workflow import (
    build_fake_runtime,
)


def _client(
    *,
    calls: list[str] | None = None,
) -> tuple[TestClient, list[str]]:
    """
    创建注入 Fake Graph Runtime 的 TestClient。

    Fake Graph 仍然使用真实 DecisionGate interrupt / resume，
    但不会调用真实 DeepSeek、MCP、Chroma 和业务数据库。
    """

    call_log = calls if calls is not None else []
    runtime = build_fake_runtime(
        calls=call_log
    )
    app = create_app(
        workflow_runtime=runtime
    )

    return TestClient(app), call_log


def _start(
    client: TestClient,
    thread_id: str,
) -> dict:
    """启动一条 Fake PlanTrip Thread 并返回 JSON。"""

    response = client.post(
        "/api/v1/travel/invoke",
        json={
            "user_id": "user_001",
            "message": "杭州到成都三日游。",
            "reference_date": "2026-07-01",
            "trip_session_id": thread_id,
        },
    )

    assert response.status_code == 200
    return response.json()


def test_frontend_entry_and_assets_are_served_by_the_same_fastapi_app():
    """前端不应需要第二个开发服务器，并且必须能从根路径加载全部本地资源。"""

    with _client()[0] as client:
        page = client.get("/")
        stylesheet = client.get("/assets/styles.css")
        script = client.get("/assets/app.js")

    assert page.status_code == 200
    assert "TravelOps Copilot" in page.text
    # 前端必须将面向用户的最终方案与管理员诊断区分为两个独立视图；
    # 这里检查静态入口的关键锚点，避免今后重构时又把 Trace 等内部信息
    # 混回用户展示区域。
    assert 'id="user-page"' in page.text
    assert 'id="manager-page"' in page.text
    assert "用户方案" in page.text
    assert "管理员面板" in page.text
    assert 'href="/assets/styles.css"' in page.text
    assert 'src="/assets/app.js"' in page.text
    assert stylesheet.status_code == 200
    assert "--navy" in stylesheet.text
    assert script.status_code == 200
    assert "submitPlan" in script.text
    assert "renderDailyPlan" in script.text
    assert "switchPage" in script.text
    # 修改方案不应再伪装成第三个审批按钮；页面应提供常驻 Composer，
    # 把自然语言修改要求提交给既有 request_changes API 合同。
    assert "inline-change-form" in script.text
    assert "submitInlineChange" in script.text


def test_invoke_returns_proposal_and_interrupt_payload():
    """
    POST /invoke 应运行完整 Graph 到 DecisionGate，
    并返回前端展示所需的 Proposal 和 Interrupt Payload。
    """

    with _client()[0] as client:
        body = _start(
            client,
            "api-invoke-001",
        )

        assert body["thread_id"] == (
            "api-invoke-001"
        )
        assert body["run_status"] == (
            "waiting_for_decision"
        )
        assert body["proposal"][
            "proposal_id"
        ] == "proposal_v1"
        # 用户页以这三个事实块组织最终方案；尤其 daily_plan 必须保留
        # 日期、主题和活动数组，不能在 API 层退化为扁平摘要。
        assert body["proposal"]["selected_flights"][
            "outbound"
        ]["flight_no"] == "MU1001"
        assert body["proposal"]["selected_hotel"][
            "name"
        ] == "测试酒店"
        assert body["proposal"]["daily_plan"][0][
            "activities"
        ][0]["title"] == "办理入住并在附近休息"
        assert body["verification"][
            "passed"
        ] is True
        assert body["interrupt"][
            "proposal_id"
        ] == "proposal_v1"
        assert body["interrupt"][
            "action_id"
        ] == "approve_proposal_v1"
        assert body["final_response"] is None


def test_approve_resume_completes_and_commit_runs_once():
    """
    POST /decision/{thread_id} 的 approve 应恢复同一 Thread，
    路由到 CommitDraft，再由 FinalResponse 结束。
    """

    calls: list[str] = []
    client, calls = _client(calls=calls)

    with client:
        started = _start(
            client,
            "api-approve-001",
        )

        response = client.post(
            "/api/v1/travel/decision/"
            "api-approve-001",
            json={
                "decision": "approve",
                "proposal_id": started[
                    "interrupt"
                ]["proposal_id"],
                "action_id": started[
                    "interrupt"
                ]["action_id"],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["run_status"] == (
            "completed"
        )
        assert body["final_response"][
            "status"
        ] == "approved_simulated"
        assert calls.count("commit_draft") == 1

        # Graph 已经完成，不再接受第二次 Resume。
        duplicate = client.post(
            "/api/v1/travel/decision/"
            "api-approve-001",
            json={
                "decision": "approve",
                "proposal_id": started[
                    "interrupt"
                ]["proposal_id"],
                "action_id": started[
                    "interrupt"
                ]["action_id"],
            },
        )
        assert duplicate.status_code == 409
        assert calls.count("commit_draft") == 1


def test_cancel_resume_completes_without_commit():
    """cancel 应进入 CancelNode，不应执行 CommitDraft。"""

    calls: list[str] = []
    client, calls = _client(calls=calls)

    with client:
        started = _start(
            client,
            "api-cancel-001",
        )

        response = client.post(
            "/api/v1/travel/decision/"
            "api-cancel-001",
            json={
                "decision": "cancel",
                "proposal_id": started[
                    "interrupt"
                ]["proposal_id"],
                "action_id": started[
                    "interrupt"
                ]["action_id"],
                "reason": "暂时取消。",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["run_status"] == (
            "completed"
        )
        assert body["final_response"][
            "status"
        ] == "cancelled"
        assert "cancel" in calls
        assert "commit_draft" not in calls


def test_request_changes_returns_second_interrupt_and_new_proposal():
    """
    request_changes 应完成 Revision Loop，
    返回 Proposal v2 的第二次 Human-in-the-loop 中断。
    """

    calls: list[str] = []
    client, calls = _client(calls=calls)

    with client:
        first = _start(
            client,
            "api-revision-001",
        )

        response = client.post(
            "/api/v1/travel/decision/"
            "api-revision-001",
            json={
                "decision": (
                    "request_changes"
                ),
                "proposal_id": first[
                    "interrupt"
                ]["proposal_id"],
                "action_id": first[
                    "interrupt"
                ]["action_id"],
                "change_request": (
                    "预算增加1000元。"
                ),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["run_status"] == (
            "waiting_for_decision"
        )
        assert body["proposal"][
            "proposal_id"
        ] == "proposal_v2"
        assert body["interrupt"][
            "proposal_id"
        ] == "proposal_v2"
        assert "revision_analyze" in calls
        assert "apply_revision" in calls
        assert calls.count("input_extract") == 1
        assert calls.count(
            "missing_info_check"
        ) == 2


def test_state_and_history_use_public_whitelist():
    """
    GET state/history 只能返回公开字段，
    不能直接暴露完整 TravelState 和 Checkpoint 私有对象。
    """

    with _client()[0] as client:
        _start(
            client,
            "api-state-001",
        )

        state_response = client.get(
            "/api/v1/travel/runs/"
            "api-state-001?include_trace=true"
        )
        assert state_response.status_code == 200

        state_body = state_response.json()
        assert state_body["run_status"] == (
            "waiting_for_decision"
        )
        assert "user_profile" not in state_body
        assert "evidence_pool" not in state_body
        assert "raw_hotel_results" not in (
            state_body
        )

        history_response = client.get(
            "/api/v1/travel/runs/"
            "api-state-001/history?limit=10"
        )
        assert history_response.status_code == 200

        history_body = history_response.json()
        assert history_body["count"] > 0
        assert "values" not in (
            history_body["items"][0]
        )
        assert "tasks" not in (
            history_body["items"][0]
        )


def test_unknown_thread_duplicate_start_and_invalid_decision_contract():
    """
    错误 thread_id 返回 404；重复启动返回 409；
    request_changes 缺少文本由 Pydantic 返回 422。
    """

    with _client()[0] as client:
        missing = client.get(
            "/api/v1/travel/runs/missing"
        )
        assert missing.status_code == 404

        _start(
            client,
            "api-conflict-001",
        )

        duplicate_start = client.post(
            "/api/v1/travel/invoke",
            json={
                "user_id": "user_001",
                "message": "重复启动。",
                "trip_session_id": (
                    "api-conflict-001"
                ),
            },
        )
        assert duplicate_start.status_code == 409

        invalid_decision = client.post(
            "/api/v1/travel/decision/"
            "api-conflict-001",
            json={
                "decision": (
                    "request_changes"
                ),
                "proposal_id": "proposal_v1",
                "action_id": (
                    "approve_proposal_v1"
                ),
                # 故意缺少 change_request。
            },
        )
        assert invalid_decision.status_code == 422
