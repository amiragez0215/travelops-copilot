from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.decision_schema import DecisionResponse


RunStatus = Literal[
    "waiting_for_decision",
    "completed",
    "running",
    "failed",
]


class TravelInvokeRequest(BaseModel):
    """
    启动一次新的 PlanTripWorkflow 请求。

    trip_session_id：
        默认由服务端生成。
        测试、幂等调用或外部系统已经拥有会话 ID 时可以显式传入。
        如果该 ID 已经存在，API 返回 409，不会覆盖旧 Checkpoint。
    """

    user_id: str = Field(
        min_length=1,
        max_length=64,
        description="当前业务用户 ID。",
    )

    message: str = Field(
        min_length=1,
        max_length=10000,
        description="用户原始旅行规划自然语言。",
    )

    reference_date: date | None = Field(
        default=None,
        description=(
            "相对日期解析基准。"
            "不传时由 Workflow 使用服务器当前日期。"
        ),
    )

    trip_session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
        description=(
            "可选会话 ID；同时作为 LangGraph thread_id。"
        ),
    )

    @field_validator("user_id", "message")
    @classmethod
    def strip_required_text(
        cls,
        value: str,
    ) -> str:
        """去除首尾空白，并拒绝只包含空格的输入。"""

        normalized = value.strip()
        if not normalized:
            raise ValueError("字段不能为空白字符串")
        return normalized


class TravelDecisionRequest(DecisionResponse):
    """
    恢复 DecisionGate interrupt 的人工决定。

    继承项目已经冻结的 DecisionResponse，保证 API 和 Node
    使用完全相同的 approve / request_changes / cancel 数据合同。
    """


class VerificationSummary(BaseModel):
    """前端真正需要的 Verifier 结果摘要。"""

    status: str | None = None
    passed: bool | None = None
    proposal_id: str | None = None
    next_action: str | None = None
    warning_count: int | None = None
    error_count: int | None = None
    critical_count: int | None = None


class PublicTraceItem(BaseModel):
    """
    对外 Trace 摘要。

    不返回完整 Prompt、Evidence 正文、用户画像或内部堆栈。
    """

    node_name: str | None = None
    tool_name: str | None = None
    status: str | None = None
    latency_ms: int | None = None
    input_summary: str | None = None
    output_summary: str | None = None
    created_at: str | None = None


class TravelRunResponse(BaseModel):
    """
    invoke、decision/resume 和 state 查询共用的公开响应合同。

    run_status 说明：
        waiting_for_decision：
            Graph 已停在 DecisionGate interrupt。

        completed：
            Graph 已到 END；业务结果见 final_response。

        running：
            当前 Snapshot 仍有 next node，但没有 interrupt。
            同步 invoke 通常很少返回该状态，主要为兼容状态查询。

        failed：
            Graph 已结束但没有形成正常 final_response，或存在运行异常。
    """

    thread_id: str

    # Stage 5 的一次 Graph invoke / resume 标识。它不同于跨请求稳定的
    # thread_id：同一个 HITL 会话每次恢复都会产生新的 run_id，方便把
    # API 返回、离线 Eval 和受控 Trace 查询关联起来，且不泄漏 Trace 内容。
    run_id: str | None = None
    run_status: RunStatus

    workflow: str | None = None
    itinerary_status: str | None = None

    checkpoint_id: str | None = None
    next_nodes: list[str] = Field(default_factory=list)

    # waiting_for_decision 时用于前端展示完整旅行方案。
    proposal: dict[str, Any] | None = None

    # Verifier 只暴露必要摘要。
    verification: VerificationSummary | None = None

    # DecisionGate interrupt payload，包含 action_id、proposal_id 和按钮定义。
    interrupt: dict[str, Any] | None = None

    # Graph 到 END 后的统一公开业务响应。
    final_response: dict[str, Any] | None = None

    # 最近一次合法或非法人工决定摘要。
    decision_result: dict[str, Any] | None = None

    # 仅在 include_trace=true 时返回摘要化 Trace。
    trace: list[PublicTraceItem] = Field(
        default_factory=list
    )


class TravelRunHistoryItem(BaseModel):
    """一个不泄漏完整 State 的 Checkpoint 历史摘要。"""

    checkpoint_id: str | None = None
    created_at: str | None = None
    step: int | None = None
    source: str | None = None

    run_status: RunStatus
    next_nodes: list[str] = Field(default_factory=list)
    has_interrupt: bool = False

    itinerary_status: str | None = None
    proposal_id: str | None = None


class TravelRunHistoryResponse(BaseModel):
    """GET /runs/{thread_id}/history 的公开响应。"""

    thread_id: str
    count: int
    items: list[TravelRunHistoryItem]
