from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from app.nodes.proposal_node import (
    proposal_node,
    route_after_proposal,
)


class FakeProposalGenerator:
    """
    ProposalNode 单元测试使用的 Fake Generator。

    Node 测试只关心：
        - State 输入检查；
        - Generator 调用；
        - State 输出；
        - Conditional Edge。

    Prompt、天气适配和 Evidence 校验由 test_proposal_generator.py 单独测试。
    """

    model_id = "fake-proposal-model"

    def __init__(
        self,
        *,
        fail: bool = False,
    ) -> None:
        self.fail = fail
        self.call_count = 0
        self.last_kwargs: dict[str, Any] | None = None

    def generate(
        self,
        *,
        trip_request: Mapping[str, Any],
        planning_context: Mapping[str, Any],
        selection_result: Mapping[str, Any],
        weather_result: Mapping[str, Any],
        evidence_pool: Sequence[Mapping[str, Any]],
        evidence_result: Mapping[str, Any],
        verifier_feedback: Mapping[str, Any] | None = None,
        proposal_version: int = 1,
    ) -> dict[str, Any]:
        self.call_count += 1
        self.last_kwargs = {
            "trip_request": dict(trip_request),
            "planning_context": dict(planning_context),
            "selection_result": dict(selection_result),
            "weather_result": dict(weather_result),
            "evidence_pool": [dict(item) for item in evidence_pool],
            "evidence_result": dict(evidence_result),
            "verifier_feedback": dict(verifier_feedback or {}),
            "proposal_version": proposal_version,
        }

        if self.fail:
            raise RuntimeError("模拟 Proposal 生成失败")

        proposal = {
            "proposal_id": "proposal_test_001",
            "version": proposal_version,
            "status": "proposed",
            "summary": "成都三日旅行方案",
            "daily_plan": [
                {
                    "day_index": 1,
                    "date": "2026-07-02",
                    "activities": [],
                }
            ],
        }

        return {
            "proposal": proposal,
            "proposal_result": {
                "status": "generated",
                "generator_version": "test",
                "prompt_version": "test",
                "model_id": self.model_id,
                "attempt_count": 1,
                "max_attempts": 2,
                "evidence_input_count": len(evidence_pool),
                "evidence_prompt_count": len(evidence_pool),
                "used_evidence_count": 1,
                "used_evidence_refs": ["guide_001"],
                "locked_fact_hash": "hash",
                "proposal_id": "proposal_test_001",
                "daily_plan_count": 1,
                "validation_errors": [],
                "issues": [],
            },
        }


def _state() -> dict[str, Any]:
    """构造 ProposalNode 所需的最小完整 State。"""

    return {
        "trip_request": {
            "origin": "杭州",
            "destination": "成都",
            "start_date": "2026-07-02",
            "end_date": "2026-07-04",
            "days": 3,
            "people_count": 1,
            "room_count": 1,
        },
        "planning_context": {
            "request": {
                "origin": "杭州",
                "destination": "成都",
                "start_date": "2026-07-02",
                "end_date": "2026-07-04",
                "days": 3,
                "nights": 2,
                "people_count": 1,
                "room_count": 1,
            },
            "preference_weights": {},
            "named_constraints": [],
            "unmapped_requirements": [],
        },
        "selection_result": {
            "status": "feasible",
            "selected_combination": {
                "combination_id": "combo_001",
                "hotel_id": "hotel_001",
            },
            "selected_hotel_id": "hotel_001",
        },
        "weather_result": {
            "status": "ok",
            "daily": [],
        },
        "evidence_pool": [
            {
                "chunk_id": "guide_001",
                "category": "guides",
                "content": "成都攻略",
                "source": "guides/a.md",
                "metadata": {
                    "doc_type": "guide",
                },
            }
        ],
        "evidence_result": {
            "status": "passed",
            "passed": True,
            "next_action": "continue",
        },
    }


def test_proposal_node_success():
    """成功生成后应写入 proposal、current_proposal 和 proposed 状态。"""

    generator = FakeProposalGenerator()

    result = proposal_node(
        _state(),
        generator=generator,
    )

    assert generator.call_count == 1
    assert result["proposal"]["proposal_id"] == "proposal_test_001"
    assert result["current_proposal"] == result["proposal"]
    assert result["proposal_result"]["status"] == "generated"
    assert result["itinerary_status"] == "proposed"
    assert result["proposal_feedback"] == {}
    assert result["verifier_result"] == {}
    assert result["pending_actions"] == []
    assert result["trace"][0]["node_name"] == "proposal"
    assert result["trace"][0]["status"] == "success"
    assert route_after_proposal(result) == "verifier"


def test_proposal_node_uses_reranked_evidence_as_compatibility_fallback():
    """
    evidence_pool 缺失时可以读取 EvidenceGrade 写回的 reranked_evidence。
    """

    state = _state()
    state["reranked_evidence"] = state.pop("evidence_pool")

    generator = FakeProposalGenerator()

    result = proposal_node(
        state,
        generator=generator,
    )

    assert result["proposal_result"]["status"] == "generated"
    assert generator.last_kwargs is not None
    assert len(generator.last_kwargs["evidence_pool"]) == 1


def test_proposal_node_rejects_unpassed_evidence_before_generator_call():
    """EvidenceGrade 未通过时不应调用 ProposalGenerator。"""

    state = _state()
    state["evidence_result"] = {
        "status": "repair_required",
        "passed": False,
        "next_action": "repair",
    }

    generator = FakeProposalGenerator()

    result = proposal_node(
        state,
        generator=generator,
    )

    assert generator.call_count == 0
    assert result["proposal"] == {}
    assert result["proposal_result"]["status"] == "failed"
    assert result["errors"]
    assert route_after_proposal(result) == "final_response"


def test_proposal_node_rejects_non_feasible_selection():
    """没有可行航班酒店组合时不能生成 Proposal。"""

    state = _state()
    state["selection_result"] = {
        "status": "no_feasible_combination"
    }

    generator = FakeProposalGenerator()

    result = proposal_node(
        state,
        generator=generator,
    )

    assert generator.call_count == 0
    assert result["errors"]
    assert result["trace"][0]["status"] == "failed"


def test_proposal_node_handles_generator_failure():
    """模型或 Generator 异常应返回统一失败结构。"""

    generator = FakeProposalGenerator(
        fail=True
    )

    result = proposal_node(
        _state(),
        generator=generator,
    )

    assert result["proposal"] == {}
    assert result["proposal_result"]["status"] == "failed"
    assert result["errors"][0]["node"] == "proposal"
    assert result["trace"][0]["status"] == "failed"
    assert route_after_proposal(result) == "final_response"


def test_proposal_node_missing_required_state_returns_error():
    """缺少 PlanningContext 时采用安全失败，不初始化真实 LLM Client。"""

    state = _state()
    state.pop("planning_context")

    generator = FakeProposalGenerator()

    result = proposal_node(
        state,
        generator=generator,
    )

    assert generator.call_count == 0
    assert result["proposal_result"]["status"] == "failed"
    assert result["errors"]


def test_route_after_proposal_defaults_to_final_response():
    """Proposal 结果缺失或无效时，Conditional Edge 默认停止。"""

    assert route_after_proposal({}) == "final_response"

    assert route_after_proposal(
        {
            "proposal_result": {
                "status": "failed"
            }
        }
    ) == "final_response"


def test_proposal_node_passes_verifier_feedback_to_generator():
    """Verifier 跨节点反馈必须进入下一次 Proposal 生成。"""

    state = _state()
    state["proposal_feedback"] = {
        "source": "deterministic_proposal_verifier_v1",
        "issues": [
            {
                "issue_type": "rain_day_all_outdoor",
                "message": "雨天不能全天户外。",
            }
        ],
    }

    generator = FakeProposalGenerator()

    result = proposal_node(
        state,
        generator=generator,
    )

    assert result["proposal_result"]["status"] == "generated"
    assert generator.last_kwargs is not None
    assert (
        generator.last_kwargs["verifier_feedback"]["source"]
        == "deterministic_proposal_verifier_v1"
    )

    # 本轮成功使用反馈后必须清空，避免污染后续普通生成。
    assert result["proposal_feedback"] == {}



def test_proposal_node_initial_version_is_one():
    generator = FakeProposalGenerator()
    result = proposal_node(_state(), generator=generator)
    assert result["proposal"]["version"] == 1
    assert generator.last_kwargs["proposal_version"] == 1


def test_proposal_node_revision_version_is_base_plus_one():
    state = _state()
    state["workflow"] = "modify_trip"
    state["revision_result"] = {"status": "applied", "base_proposal_id": "proposal_old_001", "base_proposal_version": 1}
    generator = FakeProposalGenerator()
    result = proposal_node(state, generator=generator)
    assert result["proposal"]["version"] == 2
    assert generator.last_kwargs["proposal_version"] == 2


def test_proposal_node_verifier_regeneration_keeps_same_revision_version():
    state = _state()
    state["workflow"] = "modify_trip"
    state["revision_result"] = {"status": "applied", "base_proposal_id": "proposal_old_001", "base_proposal_version": 1}
    generator = FakeProposalGenerator()
    first = proposal_node(state, generator=generator)
    second = proposal_node(state, generator=generator)
    assert first["proposal"]["version"] == 2
    assert second["proposal"]["version"] == 2
