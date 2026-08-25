from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


VerificationStatus = Literal[
    "passed",
    "repair_required",
    "failed",
]

VerificationNextAction = Literal[
    "decision_gate",
    "proposal",
    "retrieval_plan",
    "budget_optimize",
    "final_response",
]

VerificationSeverity = Literal[
    "warning",
    "error",
    "critical",
]

VerificationRepairTarget = Literal[
    "none",
    "proposal",
    "retrieval_plan",
    "budget_optimize",
    "final_response",
]


class VerificationIssue(BaseModel):
    """
    Verifier 发现的一条结构化问题。

    repair_target 不是错误类型，而是“这个问题应回到哪里修复”：

        proposal
            LLM 生成内容、Evidence 引用、天气适配或最终组装有问题。

        retrieval_plan
            当前 Evidence 状态本身不完整，需要重新规划并检索 RAG。

        budget_optimize
            选中组合与候选池、费用或总预算不一致。

        final_response
            State 缺失、上游合同损坏或重试次数耗尽，无法安全自动修复。
    """

    issue_type: str = Field(
        min_length=1,
        description="稳定问题类型，例如 locked_fact_mismatch。",
    )

    severity: VerificationSeverity = Field(
        description="warning 不阻断；error/critical 会阻断人工审批。",
    )

    repair_target: VerificationRepairTarget = Field(
        description="建议由哪个工作流阶段修复。",
    )

    check_name: str = Field(
        min_length=1,
        description="问题属于哪个检查组。",
    )

    message: str = Field(
        min_length=1,
        description="给 Trace、Eval 和修复 Prompt 阅读的问题说明。",
    )

    path: str | None = Field(
        default=None,
        description="问题在 Proposal 或 State 中的逻辑路径。",
    )

    expected: Any = Field(
        default=None,
        description="期望值；只保存适合 Trace 的小型值或摘要。",
    )

    actual: Any = Field(
        default=None,
        description="实际值；只保存适合 Trace 的小型值或摘要。",
    )

    details: dict[str, Any] = Field(
        default_factory=dict,
        description="问题附加信息。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class VerificationCheckResult(BaseModel):
    """一个检查组的执行摘要。"""

    check_name: str
    passed: bool
    issue_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    summary: str

    def to_state_dict(self) -> dict[str, Any]:
        """转成普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class ProposalVerificationResult(BaseModel):
    """
    VerifierNode 的完整输出。

    passed=True 只表示最终 Proposal 已通过独立规则审查，
    不表示系统已经预订、付款或执行真实外部动作。
    """

    status: VerificationStatus
    passed: bool
    repairable: bool
    next_action: VerificationNextAction

    strategy_version: str = "deterministic_proposal_verifier_v1"

    proposal_id: str | None = None
    expected_locked_fact_hash: str | None = None
    actual_locked_fact_hash: str | None = None

    proposal_retry_count: int = Field(ge=0)
    max_proposal_repairs: int = Field(ge=0)
    budget_retry_count: int = Field(ge=0)
    max_budget_repairs: int = Field(ge=0)

    approval_required: bool

    check_count: int = Field(ge=0)
    passed_check_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    critical_count: int = Field(ge=0)

    checks: dict[str, VerificationCheckResult] = Field(
        default_factory=dict,
        description="按检查组保存的结构化结果。",
    )

    issues: list[VerificationIssue] = Field(
        default_factory=list,
        description="Verifier 发现的问题。",
    )

    repair_feedback: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "下一节点可直接读取的修复信息。"
            "当 next_action=proposal 时会传入 Proposal Prompt。"
        ),
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()
