from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base


Base = declarative_base()


def utc_now() -> datetime:
    """
    统一生成 UTC 时间。
    """

    return datetime.now(timezone.utc)


class TimestampMixin:
    """
    通用 created_at / updated_at 字段。
    """

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class User(Base, TimestampMixin):
    """
    用户基础信息表。

    当前用于 MemoryReadNode 判断用户是否存在。
    """

    __tablename__ = "users"

    user_id = Column(String(64), primary_key=True)
    name = Column(String(100), nullable=False)
    home_city = Column(String(100), nullable=True)


class UserProfile(Base, TimestampMixin):
    """
    用户长期偏好表。

    当前 MemoryReadNode 读取它。
    后续 UpdatePreferencesWorkflow 会更新它。

    偏好使用 JSON 字段，方便后续扩展新偏好，不需要频繁改表结构。
    """

    __tablename__ = "user_profiles"

    user_id = Column(String(64), ForeignKey("users.user_id"), primary_key=True)

    flight_preferences_json = Column(JSON, nullable=False, default=dict)
    hotel_preferences_json = Column(JSON, nullable=False, default=dict)
    activity_preferences_json = Column(JSON, nullable=False, default=dict)
    pace_preferences_json = Column(JSON, nullable=False, default=dict)
    diet_preferences_json = Column(JSON, nullable=False, default=dict)
    budget_preferences_json = Column(JSON, nullable=False, default=dict)
    risk_preferences_json = Column(JSON, nullable=False, default=dict)

    profile_notes_json = Column(JSON, nullable=False, default=list)


class TripHistory(Base):
    """
    旅行历史与反馈表。

    当前 MemoryReadNode 读取最近几条摘要。
    后续 UpdatePreferencesWorkflow 会写入新的反馈记录。
    """

    __tablename__ = "trip_history"

    history_id = Column(String(64), primary_key=True)
    user_id = Column(String(64), ForeignKey("users.user_id"), nullable=False)

    trip_id = Column(String(64), nullable=True)
    destination = Column(String(100), nullable=True)

    feedback_summary = Column(Text, nullable=False)
    preference_update_json = Column(JSON, nullable=False, default=dict)
    tags_json = Column(JSON, nullable=False, default=list)

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class TripDraft(Base, TimestampMixin):
    """
    已经经过 Verifier 和人工批准的内部模拟旅行草稿。

    这张表保存“当前版本”的轻量摘要，方便后续按用户、日期、目的地查询。
    完整 Proposal JSON 不放在这里，而是保存在 TripVersion 中。

    重要边界：
        - status=approved_simulated 只表示用户批准了内部草稿；
        - 不表示真实酒店或航班已经预订；
        - 不表示系统已经付款、出票或发送通知。
    """

    __tablename__ = "trip_drafts"

    trip_id = Column(String(64), primary_key=True)

    # LangGraph thread / trip session 的稳定业务标识。
    # 当前允许为空，方便旧调用链继续运行；未来 API 应主动传入。
    trip_session_id = Column(
        String(128),
        unique=True,
        nullable=True,
    )

    user_id = Column(
        String(64),
        ForeignKey("users.user_id"),
        nullable=False,
        index=True,
    )

    status = Column(
        String(32),
        nullable=False,
        default="approved_simulated",
    )

    current_version = Column(
        Integer,
        nullable=False,
        default=1,
    )

    current_proposal_id = Column(
        String(96),
        nullable=False,
    )

    origin = Column(String(100), nullable=False)
    destination = Column(String(100), nullable=False, index=True)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)

    people_count = Column(Integer, nullable=False, default=1)
    room_count = Column(Integer, nullable=False, default=1)

    outbound_flight_id = Column(String(96), nullable=False)
    return_flight_id = Column(String(96), nullable=False)
    hotel_id = Column(String(96), nullable=False)

    known_subtotal = Column(Numeric(12, 2), nullable=False)
    total_budget = Column(Numeric(12, 2), nullable=True)
    remaining_budget = Column(Numeric(12, 2), nullable=True)

    summary = Column(Text, nullable=False)

    # 永远为 True；显式保存是为了查询和审计时快速确认系统边界。
    draft_only = Column(Boolean, nullable=False, default=True)


class TripVersion(Base):
    """
    不可变旅行方案版本快照。

    每次批准一个新 Proposal 版本时新增一行；旧版本不会被覆盖。
    这样可以支持：
        - Proposal Version History；
        - 审批后审计；
        - 未来的回放和差异比较。
    """

    __tablename__ = "trip_versions"
    __table_args__ = (
        UniqueConstraint(
            "trip_id",
            "version_number",
            name="uq_trip_versions_trip_version",
        ),
    )

    version_id = Column(String(64), primary_key=True)

    trip_id = Column(
        String(64),
        ForeignKey("trip_drafts.trip_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    version_number = Column(Integer, nullable=False)

    # 一个 Proposal 只能保存为一个版本。
    proposal_id = Column(
        String(96),
        unique=True,
        nullable=False,
        index=True,
    )

    locked_fact_hash = Column(String(128), nullable=False)
    proposal_schema_version = Column(String(64), nullable=False)
    generator_version = Column(String(128), nullable=False)

    # 完整、已验证的不可变快照。
    proposal_json = Column(JSON, nullable=False)
    verifier_result_json = Column(JSON, nullable=False)
    decision_result_json = Column(JSON, nullable=False)

    created_by_decision_event_id = Column(
        String(96),
        nullable=False,
        index=True,
    )

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class ApprovalRecord(Base):
    """
    人工批准事件审计表。

    decision_event_id 由 DecisionGateNode 根据 Proposal、Action 和人工输入
    生成稳定哈希，是 CommitDraftNode 的主要幂等键。
    """

    __tablename__ = "approval_records"

    approval_id = Column(String(64), primary_key=True)

    decision_event_id = Column(
        String(96),
        unique=True,
        nullable=False,
        index=True,
    )

    user_id = Column(
        String(64),
        ForeignKey("users.user_id"),
        nullable=False,
        index=True,
    )

    trip_id = Column(
        String(64),
        ForeignKey("trip_drafts.trip_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    version_id = Column(
        String(64),
        ForeignKey("trip_versions.version_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    proposal_id = Column(
        String(96),
        nullable=False,
        index=True,
    )
    proposal_version = Column(Integer, nullable=False)
    action_id = Column(String(128), nullable=False)

    # 当前 CommitDraftNode 只接受 approve。
    decision = Column(String(32), nullable=False)

    reason = Column(Text, nullable=True)
    client_request_id = Column(String(200), nullable=True)

    decided_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
