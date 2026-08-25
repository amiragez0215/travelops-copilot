from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


CommitDraftStatus = Literal[
    "committed",
    "already_committed",
    "failed",
]


class CommittedTripDraft(BaseModel):
    """
    CommitDraftNode 写入 trip_drafts 后返回的轻量摘要。

    注意：
        完整 Proposal 快照保存在 trip_versions.proposal_json 中；
        State 这里只保留前端和后续 FinalResponseNode 需要的核心字段，
        避免把同一份大型 Proposal 在 State 中重复保存多次。
    """

    trip_id: str = Field(min_length=1)
    trip_session_id: str | None = None
    user_id: str = Field(min_length=1)

    status: Literal["approved_simulated"] = "approved_simulated"
    current_version: int = Field(ge=1)
    current_proposal_id: str = Field(min_length=1)

    origin: str
    destination: str
    start_date: str
    end_date: str
    people_count: int = Field(ge=1)
    room_count: int = Field(ge=1)

    outbound_flight_id: str
    return_flight_id: str
    hotel_id: str

    known_subtotal: float = Field(ge=0)
    total_budget: float | None = Field(default=None, ge=0)
    remaining_budget: float | None = Field(default=None, ge=0)

    summary: str
    draft_only: bool = True

    created_at: str
    updated_at: str

    def to_state_dict(self) -> dict[str, Any]:
        """转换成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class CommittedTripVersion(BaseModel):
    """
    不可变 Proposal 版本摘要。

    完整 proposal_json、verifier_result_json 和 decision_result_json
    保存在 SQL 中；这里不重复返回完整 JSON。
    """

    version_id: str = Field(min_length=1)
    trip_id: str = Field(min_length=1)
    version_number: int = Field(ge=1)

    proposal_id: str = Field(min_length=1)
    locked_fact_hash: str = Field(min_length=1)
    proposal_schema_version: str
    generator_version: str

    created_by_decision_event_id: str = Field(min_length=1)
    created_at: str

    def to_state_dict(self) -> dict[str, Any]:
        """转换成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class CommittedApprovalRecord(BaseModel):
    """
    人工批准事件的业务数据库审计摘要。
    """

    approval_id: str = Field(min_length=1)
    decision_event_id: str = Field(min_length=1)

    user_id: str = Field(min_length=1)
    trip_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)

    proposal_id: str = Field(min_length=1)
    proposal_version: int = Field(ge=1)
    action_id: str = Field(min_length=1)

    decision: Literal["approve"] = "approve"
    reason: str | None = None
    client_request_id: str | None = None

    decided_at: str
    created_at: str

    def to_state_dict(self) -> dict[str, Any]:
        """转换成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class CommitDraftResult(BaseModel):
    """
    CommitDraftNode 的执行摘要。

    status：
        committed
            本次真正写入 TripDraft、TripVersion 和 ApprovalRecord。

        already_committed
            相同 decision_event_id 或 proposal_id 已经写入；
            本次属于幂等重放，没有创建重复记录。

        failed
            前置状态不满足或数据库事务失败。
    """

    status: CommitDraftStatus
    strategy_version: str = "transactional_trip_draft_commit_v1"

    idempotent_replay: bool = False
    records_written: int = Field(default=0, ge=0)

    trip_id: str | None = None
    version_id: str | None = None
    approval_id: str | None = None

    user_id: str | None = None
    proposal_id: str | None = None
    proposal_version: int | None = Field(default=None, ge=1)
    decision_event_id: str | None = None

    committed_at: str | None = None
    message: str
    issues: list[dict[str, Any]] = Field(default_factory=list)

    def to_state_dict(self) -> dict[str, Any]:
        """转换成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()
