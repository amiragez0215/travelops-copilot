from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


DecisionType = Literal[
    "approve",
    "request_changes",
    "cancel",
]

DecisionNextAction = Literal[
    "commit_draft",
    "revision_analyze",
    "cancel",
    "decision_gate",
    "final_response",
]

DecisionGateStatus = Literal[
    "accepted",
    "invalid",
    "failed",
]


class DecisionActionOption(BaseModel):
    """
    DecisionGate 暴露给前端的一项可选人工操作。

    requires_text_input=True 表示：
        用户不能只点击按钮，还必须提交一段文本。

    当前只有 request_changes 需要修改说明。
    """

    decision: DecisionType
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    requires_text_input: bool = False


class DecisionInterruptPayload(BaseModel):
    """
    interrupt() 暂停工作流时返回给 API / 前端的 JSON Payload。

    这里只放人工决策真正需要的信息，不放完整 TravelState：
        - 避免泄漏内部 Trace、Prompt 和中间候选池；
        - 减少 Checkpoint 与 API Payload 的体积；
        - 让前端只依赖稳定的审批数据合同。
    """

    type: Literal["trip_proposal_decision"] = "trip_proposal_decision"

    action_id: str = Field(
        min_length=1,
        description="VerifierNode 创建的 pending approval action ID。",
    )

    proposal_id: str = Field(
        min_length=1,
        description="当前等待审批的 Proposal ID。",
    )

    proposal_version: int = Field(
        ge=1,
        description="当前 Proposal 版本，用于前端展示和过期决策检查。",
    )

    message: str = Field(min_length=1)

    available_actions: list[DecisionActionOption] = Field(
        min_length=3,
        description="approve / request_changes / cancel 三种人工操作。",
    )

    proposal_summary: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "给审批页面展示的精简方案摘要。"
            "它不是最终 Proposal 的替代品。"
        ),
    )

    pending_action: dict[str, Any] = Field(
        default_factory=dict,
        description="VerifierNode 创建的当前待审批动作摘要。",
    )

    validation_error: dict[str, Any] | None = Field(
        default=None,
        description=(
            "上一次人工输入不合法时的重新提示信息。"
            "第一次 interrupt 通常为 None。"
        ),
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转换成 interrupt() 可以安全序列化的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class DecisionResponse(BaseModel):
    """
    API 使用 Command(resume=...) 恢复工作流时提交的人工决定。

    proposal_id 和 action_id 都必须原样返回：
        - proposal_id 防止批准已经被修改过的旧 Proposal；
        - action_id 防止把一个审批结果错误应用到另一项待办动作。
    """

    decision: DecisionType

    proposal_id: str = Field(
        min_length=1,
        description="必须等于 interrupt payload 中的 proposal_id。",
    )

    action_id: str = Field(
        min_length=1,
        description="必须等于 interrupt payload 中的 action_id。",
    )

    change_request: str | None = Field(
        default=None,
        max_length=4000,
        description=(
            "request_changes 时必填的自然语言修改说明。"
            "后续 RevisionAnalyzeNode 会把它转换成 RevisionPatch。"
        ),
    )

    reason: str | None = Field(
        default=None,
        max_length=1000,
        description="批准或取消时可选的用户说明。",
    )

    client_request_id: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "前端可选的幂等请求标识。"
            "当前节点只记录它，未来 API 层可以据此防重复提交。"
        ),
    )

    @model_validator(mode="after")
    def validate_decision_payload(self) -> "DecisionResponse":
        """检查不同 decision 对附加字段的要求。"""

        change_request = (
            self.change_request.strip()
            if isinstance(self.change_request, str)
            else ""
        )

        # 1. 修改决定必须携带具体修改文本。
        if (
            self.decision == "request_changes"
            and not change_request
        ):
            raise ValueError(
                "decision=request_changes 时必须提供非空 change_request"
            )

        # 2. approve / cancel 不应误带修改文本，避免前后端语义冲突。
        if (
            self.decision in {"approve", "cancel"}
            and change_request
        ):
            raise ValueError(
                "approve 或 cancel 不能同时提交 change_request"
            )

        if change_request:
            self.change_request = change_request

        if isinstance(self.reason, str):
            stripped_reason = self.reason.strip()
            self.reason = stripped_reason or None

        return self


class DecisionGateResult(BaseModel):
    """
    DecisionGateNode 对一次 resume 输入的处理结果。

    status：
        accepted
            人工输入合法，已经路由到后续业务节点。

        invalid
            人工输入不合法，Graph 会回到 DecisionGate 再次 interrupt。

        failed
            上游 State 不满足审批前置条件，无法安全继续。
    """

    status: DecisionGateStatus
    strategy_version: str = "langgraph_hitl_decision_gate_v1"

    accepted: bool
    decision: DecisionType | None = None
    next_action: DecisionNextAction

    proposal_id: str | None = None
    proposal_version: int | None = Field(default=None, ge=1)
    action_id: str | None = None

    decision_event_id: str | None = None
    client_request_id: str | None = None

    attempt_count: int = Field(ge=0)
    decided_at: str | None = None

    change_request_provided: bool = False
    reason: str | None = None

    validation_errors: list[str] = Field(default_factory=list)

    def to_state_dict(self) -> dict[str, Any]:
        """转换成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class DecisionHistoryItem(BaseModel):
    """
    一条已经接受的人工决定审计记录。

    DecisionGateNode 只写入 State；
    真正持久化到业务数据库可以在 CommitDraftNode / CancelNode 中完成。
    """

    decision_event_id: str
    proposal_id: str
    proposal_version: int = Field(ge=1)
    action_id: str
    decision: DecisionType
    next_action: DecisionNextAction
    decided_at: str

    change_request: str | None = None
    reason: str | None = None
    client_request_id: str | None = None

    def to_state_dict(self) -> dict[str, Any]:
        """转换成适合 add reducer 追加的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()
