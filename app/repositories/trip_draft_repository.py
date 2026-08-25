from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import (
    ApprovalRecord,
    TripDraft,
    TripVersion,
    User,
)


class TripDraftRepository:
    """
    TripDraft 持久化仓库。

    Repository 只负责 SQL 读写，不负责：
        - 校验 DecisionGate 业务前置条件；
        - 生成 Trip ID / Version ID；
        - 决定幂等策略；
        - 读写 LangGraph State。

    事务边界由 CommitDraftService 统一控制。
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_user(self, user_id: str) -> User | None:
        """按 user_id 读取用户。"""

        return self.session.get(User, user_id)

    def get_trip(self, trip_id: str) -> TripDraft | None:
        """按 trip_id 读取当前 TripDraft。"""

        return self.session.get(TripDraft, trip_id)

    def get_trip_by_session_id(
        self,
        trip_session_id: str,
    ) -> TripDraft | None:
        """按稳定旅行会话 ID 查找已经保存的 TripDraft。"""

        return (
            self.session.query(TripDraft)
            .filter(
                TripDraft.trip_session_id
                == trip_session_id
            )
            .one_or_none()
        )

    def get_version_by_proposal_id(
        self,
        proposal_id: str,
    ) -> TripVersion | None:
        """按 proposal_id 查找不可变版本快照。"""

        return (
            self.session.query(TripVersion)
            .filter(
                TripVersion.proposal_id
                == proposal_id
            )
            .one_or_none()
        )

    def get_version(
        self,
        version_id: str,
    ) -> TripVersion | None:
        """按 version_id 读取版本。"""

        return self.session.get(
            TripVersion,
            version_id,
        )

    def get_approval_by_decision_event_id(
        self,
        decision_event_id: str,
    ) -> ApprovalRecord | None:
        """
        按 DecisionGate 生成的稳定事件 ID 查找审批记录。

        这是 CommitDraft 的第一层幂等键。
        """

        return (
            self.session.query(ApprovalRecord)
            .filter(
                ApprovalRecord.decision_event_id
                == decision_event_id
            )
            .one_or_none()
        )

    def get_approval_by_proposal_id(
        self,
        proposal_id: str,
    ) -> ApprovalRecord | None:
        """
        按 proposal_id 查找审批记录。

        即使前端因为不同 client_request_id 生成了新的 decision_event_id，
        同一 Proposal 也不能被保存两次。
        """

        return (
            self.session.query(ApprovalRecord)
            .filter(
                ApprovalRecord.proposal_id
                == proposal_id
            )
            .one_or_none()
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_trip(self, trip: TripDraft) -> None:
        """把新的 TripDraft 加入当前事务。"""

        self.session.add(trip)

    def add_version(self, version: TripVersion) -> None:
        """把新的不可变 TripVersion 加入当前事务。"""

        self.session.add(version)

    def add_approval(
        self,
        approval: ApprovalRecord,
    ) -> None:
        """把人工批准记录加入当前事务。"""

        self.session.add(approval)

    def flush(self) -> None:
        """
        将当前事务写入数据库连接，但暂不提交。

        flush 可以提前触发唯一约束、外键和类型错误；
        任何失败仍可以由 Service 统一 rollback。
        """

        self.session.flush()

    def commit(self) -> None:
        """一次性提交当前事务中的所有写入。"""

        self.session.commit()

    def rollback(self) -> None:
        """回滚当前事务，防止部分表写入成功。"""

        self.session.rollback()
