from __future__ import annotations

from copy import deepcopy

from app.nodes.verifier_node import (
    route_after_verifier,
    verifier_node,
)
from app.verifiers.proposal_verifier import (
    ProposalVerifier,
    ProposalVerifierConfig,
)
from tests.test_verifiers.test_proposal_verifier import (
    _generated_state,
)


def _verifier(
    *,
    max_proposal_repairs: int = 1,
    max_budget_repairs: int = 1,
) -> ProposalVerifier:
    """构造测试用确定性 Verifier。"""

    return ProposalVerifier(
        ProposalVerifierConfig(
            max_proposal_repairs=(
                max_proposal_repairs
            ),
            max_budget_repairs=(
                max_budget_repairs
            ),
        )
    )


def test_verifier_node_success_routes_to_decision_gate():
    """合法 Proposal 应写入 Verifier 结果和 pending approval。"""

    result = verifier_node(
        _generated_state(),
        verifier=_verifier(),
    )

    assert result["verifier_result"]["status"] == "passed"
    assert result["verifier_result"]["passed"] is True
    assert result["proposal_feedback"] == {}
    assert result["pending_actions"]
    assert result["trace"][0]["node_name"] == "verifier"
    assert result["trace"][0]["status"] == "success"
    assert route_after_verifier(result) == "decision_gate"


def test_verifier_node_proposal_repair_increments_retry_count():
    """Proposal 内容错误时应生成反馈并增加跨节点修复次数。"""

    state = _generated_state()
    state["proposal"]["summary"] = (
        "已经为你预订酒店。"
    )

    result = verifier_node(
        state,
        verifier=_verifier(),
    )

    assert (
        result["verifier_result"]["next_action"]
        == "proposal"
    )
    assert result["proposal_retry_count"] == 1
    assert result["proposal_feedback"]["issues"]
    assert result["pending_actions"] == []
    assert "errors" not in result
    assert result["trace"][0]["status"] == "degraded"
    assert route_after_verifier(result) == "proposal"


def test_verifier_node_budget_repair_increments_budget_retry_count():
    """选择结果不在当前候选池时应回 BudgetOptimize。"""

    state = _generated_state()
    state["flight_candidates"] = [
        item
        for item in state["flight_candidates"]
        if item["direction"] == "return"
    ]

    result = verifier_node(
        state,
        verifier=_verifier(),
    )

    assert (
        result["verifier_result"]["next_action"]
        == "budget_optimize"
    )
    assert result["budget_retry_count"] == 1
    assert result["proposal_feedback"] == {}
    assert "errors" not in result
    assert route_after_verifier(result) == "budget_optimize"


def test_verifier_node_retry_exhausted_routes_to_final_response():
    """Proposal 修复次数耗尽时不再循环。"""

    state = _generated_state()
    state["proposal"]["summary"] = (
        "已经为你预订酒店。"
    )
    state["proposal_retry_count"] = 1

    result = verifier_node(
        state,
        verifier=_verifier(
            max_proposal_repairs=1
        ),
    )

    assert result["verifier_result"]["status"] == "failed"
    assert (
        result["verifier_result"]["next_action"]
        == "final_response"
    )
    assert "proposal_retry_count" not in result
    assert route_after_verifier(result) == "final_response"


def test_verifier_node_missing_proposal_returns_program_error():
    """缺少 proposal 属于节点调用顺序错误，应写入 state.errors。"""

    state = _generated_state()
    state.pop("proposal")

    result = verifier_node(
        state,
        verifier=_verifier(),
    )

    assert result["verifier_result"]["status"] == "failed"
    assert result["errors"]
    assert result["errors"][0]["node"] == "verifier"
    assert result["trace"][0]["status"] == "failed"
    assert route_after_verifier(result) == "final_response"


def test_route_after_verifier_uses_safe_default():
    """缺失或非法 next_action 时默认进入 FinalResponse。"""

    assert route_after_verifier({}) == "final_response"

    assert route_after_verifier(
        {
            "verifier_result": {
                "next_action": "unknown"
            }
        }
    ) == "final_response"

    for expected in [
        "decision_gate",
        "proposal",
        "retrieval_plan",
        "budget_optimize",
    ]:
        assert route_after_verifier(
            {
                "verifier_result": {
                    "next_action": expected
                }
            }
        ) == expected
