from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


CancelStatus = Literal[
    "cancelled",
    "already_cancelled",
    "failed",
]


class CancelResult(BaseModel):
    """
    CancelNode 对一次人工取消决定的结构化处理结果。

    status：
        cancelled
            当前 Proposal 第一次完成取消收尾。

        already_cancelled
            相同 decision_event_id 已经处理过；
            当前调用属于幂等重放，没有产生新的副作用。

        failed
            DecisionGate / Verifier / Proposal 状态不一致，
            CancelNode 无法安全确认取消。

    重要边界：
        - CancelNode 不删除 Proposal；
        - CancelNode 不删除 Checkpoint；
        - CancelNode 不删除已经存在的历史 TripDraft；
        - CancelNode 不执行真实酒店或航班取消；
        - 当前 v1 只取消“本次内部旅行方案审批流程”。
    """

    status: CancelStatus
    strategy_version: str = "deterministic_proposal_cancel_v1"

    # 相同取消事件被再次执行时为 True。
    idempotent_replay: bool = False

    proposal_id: str | None = None
    proposal_version: int | None = Field(default=None, ge=1)

    action_id: str | None = None
    decision_event_id: str | None = None

    cancelled_at: str | None = None
    reason: str | None = None

    # 当前取消只改变内部流程状态，不会删除已经生成的 Proposal 内容。
    proposal_deleted: bool = False

    # 当前项目没有真实预订，因此也不存在真实外部取消动作。
    real_booking_cancelled: bool = False
    real_payment_reversed: bool = False
    real_notification_sent: bool = False

    message: str
    issues: list[dict[str, Any]] = Field(default_factory=list)

    def to_state_dict(self) -> dict[str, Any]:
        """
        转换成适合写入 TravelState 的普通 dict。
        """

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()
