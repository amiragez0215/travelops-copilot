from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.proposals.proposal_generator import LLMProposalGenerator
from app.verifiers.proposal_verifier import (
    ProposalVerifier,
    ProposalVerifierConfig,
)
from tests.test_proposals.test_proposal_generator import (
    SequenceJSONClient,
    _evidence_pool,
    _planning_context,
    _selection_result,
    _trip_request,
    _valid_draft,
    _weather_result,
)


def _generated_state(
    *,
    draft: dict[str, Any] | None = None,
    planning_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """使用真实 ProposalGenerator 逻辑构造一份可通过 Verifier 的 State。"""

    client = SequenceJSONClient([
        draft or _valid_draft()
    ])

    generator = LLMProposalGenerator(
        client=client,
        max_tokens=4096,
        max_attempts=1,
        max_evidence_items=20,
        max_evidence_chars=800,
    )

    trip_request = _trip_request()
    planning_context = planning_context or _planning_context()
    selection_result = _selection_result()
    weather_result = _weather_result()
    evidence_pool = _evidence_pool()
    evidence_result = {
        "status": "passed",
        "passed": True,
        "next_action": "continue",
    }

    generated = generator.generate(
        trip_request=trip_request,
        planning_context=planning_context,
        selection_result=selection_result,
        weather_result=weather_result,
        evidence_pool=evidence_pool,
        evidence_result=evidence_result,
    )

    outbound = deepcopy(
        selection_result[
            "selected_outbound_flight"
        ]
    )
    outbound["direction"] = "outbound"

    return_flight = deepcopy(
        selection_result[
            "selected_return_flight"
        ]
    )
    return_flight["direction"] = "return"

    return {
        "proposal": generated["proposal"],
        "proposal_result": generated["proposal_result"],
        "trip_request": trip_request,
        "planning_context": planning_context,
        "selection_result": selection_result,
        "weather_result": weather_result,
        "evidence_pool": evidence_pool,
        "evidence_result": evidence_result,
        "flight_candidates": [
            outbound,
            return_flight,
        ],
        "hotel_candidates": [
            deepcopy(
                selection_result[
                    "selected_hotel"
                ]
            )
        ],
        "retrieval_plan": {
            "max_repair_rounds": 1,
            "repair_exhausted": False,
        },
        "proposal_retry_count": 0,
        "budget_retry_count": 0,
        "rag_retry_count": 0,
    }


def _verify(
    state: dict[str, Any],
    *,
    max_proposal_repairs: int = 1,
    max_budget_repairs: int = 1,
) -> dict[str, Any]:
    """调用 Verifier 的统一测试入口。"""

    verifier = ProposalVerifier(
        ProposalVerifierConfig(
            max_proposal_repairs=(
                max_proposal_repairs
            ),
            max_budget_repairs=(
                max_budget_repairs
            ),
        )
    )

    return verifier.verify(
        proposal=state["proposal"],
        proposal_result=state["proposal_result"],
        trip_request=state["trip_request"],
        planning_context=state["planning_context"],
        selection_result=state["selection_result"],
        weather_result=state["weather_result"],
        evidence_pool=state["evidence_pool"],
        evidence_result=state["evidence_result"],
        flight_candidates=state["flight_candidates"],
        hotel_candidates=state["hotel_candidates"],
        proposal_retry_count=state.get(
            "proposal_retry_count",
            0,
        ),
        budget_retry_count=state.get(
            "budget_retry_count",
            0,
        ),
        retrieval_plan=state.get(
            "retrieval_plan"
        ),
        rag_retry_count=state.get(
            "rag_retry_count",
            0,
        ),
    )


def test_verifier_passes_valid_proposal_and_creates_approval_action():
    """完整合法 Proposal 应进入 DecisionGate，并创建 pending approval。"""

    output = _verify(
        _generated_state()
    )

    result = output[
        "verifier_result"
    ]

    assert result["status"] == "passed"
    assert result["passed"] is True
    assert result["next_action"] == "decision_gate"
    assert result["approval_required"] is True
    assert result["error_count"] == 0
    assert result["critical_count"] == 0

    assert output["proposal_feedback"] == {}
    assert len(output["pending_actions"]) == 1
    assert (
        output["pending_actions"][0][
            "action_type"
        ]
        == "approve_proposal"
    )


def test_locked_fact_tampering_routes_back_to_proposal():
    """修改最终酒店名称会破坏锁定事实和 proposal_id，应回 ProposalNode。"""

    state = _generated_state()
    state["proposal"][
        "selected_hotel"
    ]["name"] = "被篡改的酒店名称"

    output = _verify(state)
    result = output[
        "verifier_result"
    ]

    assert result["status"] == "repair_required"
    assert result["next_action"] == "proposal"
    assert output["proposal_feedback"]["issues"]

    issue_types = {
        item["issue_type"]
        for item in result["issues"]
    }

    assert "locked_fact_value_mismatch" in issue_types


def test_candidate_lineage_failure_routes_to_budget_optimize():
    """选中去程不在当前候选池时，应先重新执行 BudgetOptimize。"""

    state = _generated_state()

    # 只保留返程候选，模拟修改流程后旧 selection_result 已过期。
    state["flight_candidates"] = [
        item
        for item in state["flight_candidates"]
        if item["direction"] == "return"
    ]

    output = _verify(state)
    result = output[
        "verifier_result"
    ]

    assert result["next_action"] == "budget_optimize"
    assert result["repairable"] is True

    assert any(
        item["issue_type"]
        == "selected_candidate_not_in_ranked_pool"
        for item in result["issues"]
    )


def test_unknown_evidence_ref_routes_back_to_proposal():
    """最终 Proposal 引用 Evidence Pool 中不存在的 ID 时应重新生成。"""

    state = _generated_state()
    state["proposal"]["daily_plan"][0][
        "activities"
    ][1]["evidence_refs"] = [
        "invented_ref"
    ]

    output = _verify(state)
    result = output[
        "verifier_result"
    ]

    assert result["next_action"] == "proposal"

    assert any(
        item["issue_type"]
        == "unknown_evidence_ref"
        for item in result["issues"]
    )


def test_rain_day_all_outdoor_routes_back_to_proposal():
    """最终 Proposal 被后处理改成雨天全天户外时，Verifier 应独立发现。"""

    state = _generated_state()

    for activity in state["proposal"][
        "daily_plan"
    ][1]["activities"]:
        activity["indoor_outdoor"] = "outdoor"

    output = _verify(state)
    result = output[
        "verifier_result"
    ]

    assert result["next_action"] == "proposal"

    assert any(
        item["issue_type"]
        == "rain_day_all_outdoor"
        for item in result["issues"]
    )


def test_real_booking_claim_routes_back_to_proposal():
    """即使 execution_boundary 正确，正文声称已预订也必须被拦截。"""

    state = _generated_state()
    state["proposal"]["summary"] = (
        "已经为你预订成都酒店，并完成三日行程安排。"
    )

    output = _verify(state)
    result = output[
        "verifier_result"
    ]

    assert result["next_action"] == "proposal"

    assert any(
        item["issue_type"]
        == "real_booking_claim"
        for item in result["issues"]
    )


def test_proposal_repair_stops_after_retry_limit():
    """跨节点 Proposal 修复次数耗尽后应停止，防止无限循环。"""

    state = _generated_state()
    state["proposal"]["summary"] = (
        "已经为你预订成都酒店。"
    )
    state["proposal_retry_count"] = 1

    output = _verify(
        state,
        max_proposal_repairs=1,
    )
    result = output[
        "verifier_result"
    ]

    assert result["status"] == "failed"
    assert result["repairable"] is False
    assert result["next_action"] == "final_response"


def test_unpassed_evidence_routes_to_retrieval_plan():
    """Evidence 状态失效时，应先修复 RAG，而不是只重写 Proposal。"""

    state = _generated_state()
    state["evidence_result"] = {
        "status": "repair_required",
        "passed": False,
        "next_action": "repair",
    }

    output = _verify(state)
    result = output[
        "verifier_result"
    ]

    assert result["next_action"] == "retrieval_plan"

    assert any(
        item["issue_type"]
        == "evidence_grade_not_passed"
        for item in result["issues"]
    )


def test_proposal_id_tampering_is_detected():
    """proposal_id 必须与最终内容重新计算结果一致。"""

    state = _generated_state()
    state["proposal"]["proposal_id"] = (
        "proposal_tampered"
    )

    output = _verify(state)
    result = output[
        "verifier_result"
    ]

    assert result["next_action"] == "proposal"

    assert any(
        item["issue_type"]
        == "proposal_id_mismatch"
        for item in result["issues"]
    )


def test_tool_activity_source_keeps_proposal_id_consistent():
    """活动来源默认字段必须在生成 ID 前写入，避免规范化后哈希变化。"""

    planning_context = _planning_context()
    planning_context["activity_candidates"] = [
        {
            "activity_id": "CD-ACT-001",
            "name": "成都城市文化专题展",
            "city": "成都",
            "available_dates": ["2026-07-02"],
            "opening_hours": "10:00-18:00",
            "indoor_outdoor": "indoor",
            "tags": ["展览", "文化"],
        }
    ]

    draft = _valid_draft()
    draft["daily_plan"][0]["activities"][1].update(
        {
            "title": "成都城市文化专题展",
            "description": "使用本轮活动工具候选安排专题展。",
            "activity_type": "culture",
            "indoor_outdoor": "indoor",
            # 保留攻略证据以满足知识型活动的既有验证规则；本测试只隔离
            # activity_sources 规范化前后 proposal_id 是否保持一致。
            "evidence_refs": ["G02"],
            "tool_activity_id": "CD-ACT-001",
        }
    )

    state = _generated_state(
        draft=draft,
        planning_context=planning_context,
    )
    output = _verify(state)
    result = output["verifier_result"]

    # 这里验证完整 Generator -> TripProposal -> Verifier 链路；只检查最终
    # source 字段存在并不足以发现本次问题，因为 Pydantic 原本就会补默认值。
    assert result["passed"] is True, [
        (
            item["issue_type"],
            item.get("message"),
            item.get("path"),
        )
        for item in result["issues"]
    ]
    assert not any(
        item["issue_type"] == "proposal_id_mismatch"
        for item in result["issues"]
    )
