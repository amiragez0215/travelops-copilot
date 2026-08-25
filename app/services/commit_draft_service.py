from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from sqlalchemy.exc import IntegrityError

from app.db.models import (
    ApprovalRecord,
    TripDraft,
    TripVersion,
)
from app.repositories.trip_draft_repository import (
    TripDraftRepository,
)
from app.schemas.commit_draft_schema import (
    CommitDraftResult,
    CommittedApprovalRecord,
    CommittedTripDraft,
    CommittedTripVersion,
)
from app.schemas.decision_schema import (
    DecisionGateResult,
)
from app.schemas.proposal_schema import TripProposal
from app.schemas.verifier_schema import (
    ProposalVerificationResult,
)


class CommitDraftError(RuntimeError):
    """CommitDraft 领域错误基类。"""


class CommitPreconditionError(CommitDraftError):
    """Decision / Verifier / Proposal 前置状态不满足。"""


class CommitConflictError(CommitDraftError):
    """同一 Trip 的版本或所属用户发生冲突。"""


class UserNotFoundError(CommitDraftError):
    """批准操作没有可关联的内部用户。"""


class CommitDraftService:
    """
    将已批准 Proposal 原子写入业务数据库。

    一次成功提交包含三张表：

        trip_drafts
            保存当前旅行草稿的可查询摘要和 current_version 指针。

        trip_versions
            保存完整、不可变的 Proposal / Verifier / Decision JSON 快照。

        approval_records
            保存人工批准事件，提供审计与幂等键。

    关键保证：
        1. 三张表在同一个 SQL 事务中提交；
        2. 相同 decision_event_id 重放不会重复写入；
        3. 即使 decision_event_id 不同，同一 proposal_id 也不会重复保存；
        4. 只接受 Verifier 已通过且 DecisionGate 已批准的 Proposal；
        5. 这里只保存内部模拟草稿，不执行真实预订和付款。
    """

    def __init__(
        self,
        repository: TripDraftRepository,
    ) -> None:
        self.repository = repository

    def commit(
        self,
        *,
        user_id: str,
        proposal: Mapping[str, Any],
        verifier_result: Mapping[str, Any],
        decision_result: Mapping[str, Any],
        pending_actions: Sequence[Mapping[str, Any]],
        trip_session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        校验并保存已批准的 Proposal。

        Args:
            user_id:
                业务数据库用户 ID。匿名规划可以生成 Proposal，
                但提交内部草稿时必须有可关联用户。

            proposal:
                ProposalNode 生成并经过 Verifier 的完整 TripProposal。

            verifier_result:
                VerifierNode 的最终结果，必须 passed=True。

            decision_result:
                DecisionGateNode 接受的人工决定，必须是 approve。

            pending_actions:
                当前审批动作列表。匹配 action_id 的动作必须已被标记 approved。

            trip_session_id:
                可选的稳定旅行会话 ID。未来 Revision Loop 中，
                同一 thread/session 的多个 Proposal 版本可以写入同一个 TripDraft。
        """

        # 1. 先在内存中完成全部数据合同和业务前置校验。
        #    前置校验失败时数据库不会产生任何写入。
        parsed_proposal = TripProposal.model_validate(
            dict(proposal)
        )

        parsed_verification = (
            ProposalVerificationResult.model_validate(
                dict(verifier_result)
            )
        )

        parsed_decision = (
            DecisionGateResult.model_validate(
                dict(decision_result)
            )
        )

        normalized_user_id = str(
            user_id or ""
        ).strip()

        if not normalized_user_id:
            raise UserNotFoundError(
                "CommitDraftNode 需要非空 user_id；"
                "匿名规划不能保存内部业务草稿。"
            )

        normalized_session_id = (
            str(trip_session_id).strip()
            if trip_session_id
            else None
        )

        self._validate_preconditions(
            proposal=parsed_proposal,
            verification=parsed_verification,
            decision=parsed_decision,
            pending_actions=pending_actions,
        )

        decision_event_id = str(
            parsed_decision.decision_event_id
            or ""
        )

        # 2. 第一层幂等检查：相同人工决定事件已经提交。
        existing_approval = (
            self.repository
            .get_approval_by_decision_event_id(
                decision_event_id
            )
        )

        if existing_approval is not None:
            return self._build_existing_output(
                approval=existing_approval,
                expected_user_id=normalized_user_id,
                reason=(
                    "相同 decision_event_id 已经提交，"
                    "本次返回已有草稿。"
                ),
            )

        # 3. 第二层幂等检查：同一 Proposal 已经保存。
        #    这能防止前端换了 client_request_id 后重复点击批准。
        existing_version = (
            self.repository
            .get_version_by_proposal_id(
                parsed_proposal.proposal_id
            )
        )

        if existing_version is not None:
            existing_by_proposal = (
                self.repository
                .get_approval_by_proposal_id(
                    parsed_proposal.proposal_id
                )
            )

            if existing_by_proposal is None:
                raise CommitConflictError(
                    "proposal_id 已存在 TripVersion，"
                    "但没有对应 ApprovalRecord，数据库状态不完整。"
                )

            return self._build_existing_output(
                approval=existing_by_proposal,
                expected_user_id=normalized_user_id,
                reason=(
                    "相同 proposal_id 已经保存，"
                    "本次未创建重复版本。"
                ),
            )

        # 4. 提交草稿必须能够关联一个真实内部用户。
        if self.repository.get_user(
            normalized_user_id
        ) is None:
            raise UserNotFoundError(
                f"数据库中不存在 user_id={normalized_user_id}，"
                "无法保存 TripDraft。"
            )

        # 5. 根据旅行会话和 Proposal 构造稳定业务 ID。
        trip_id = _build_trip_id(
            user_id=normalized_user_id,
            proposal_id=parsed_proposal.proposal_id,
            trip_session_id=normalized_session_id,
        )

        version_id = _build_version_id(
            trip_id=trip_id,
            proposal_id=parsed_proposal.proposal_id,
            proposal_version=parsed_proposal.version,
        )

        approval_id = _build_approval_id(
            decision_event_id
        )

        decided_at = _parse_iso_datetime(
            parsed_decision.decided_at
        )

        now = datetime.now(timezone.utc)

        # 6. 将 Proposal 中的锁定事实整理成可查询摘要。
        trip = self.repository.get_trip(
            trip_id
        )

        if trip is None and normalized_session_id:
            # 防御性检查：同一 session 不允许关联到另一条 trip_id。
            trip = (
                self.repository
                .get_trip_by_session_id(
                    normalized_session_id
                )
            )

        if trip is None:
            trip = self._build_new_trip(
                trip_id=trip_id,
                trip_session_id=normalized_session_id,
                user_id=normalized_user_id,
                proposal=parsed_proposal,
                now=now,
            )

            self.repository.add_trip(trip)

        else:
            self._update_existing_trip(
                trip=trip,
                expected_trip_id=trip_id,
                expected_user_id=normalized_user_id,
                proposal=parsed_proposal,
                now=now,
            )

        version = TripVersion(
            version_id=version_id,
            trip_id=trip.trip_id,
            version_number=parsed_proposal.version,
            proposal_id=parsed_proposal.proposal_id,
            locked_fact_hash=(
                parsed_proposal.locked_fact_hash
            ),
            proposal_schema_version=(
                parsed_proposal.schema_version
            ),
            generator_version=(
                parsed_proposal.generator_version
            ),
            proposal_json=(
                parsed_proposal.model_dump(
                    mode="json"
                )
            ),
            verifier_result_json=dict(
                verifier_result
            ),
            decision_result_json=dict(
                decision_result
            ),
            created_by_decision_event_id=(
                decision_event_id
            ),
            created_at=now,
        )

        approval = ApprovalRecord(
            approval_id=approval_id,
            decision_event_id=(
                decision_event_id
            ),
            user_id=normalized_user_id,
            trip_id=trip.trip_id,
            version_id=version_id,
            proposal_id=(
                parsed_proposal.proposal_id
            ),
            proposal_version=(
                parsed_proposal.version
            ),
            action_id=str(
                parsed_decision.action_id
            ),
            decision="approve",
            reason=parsed_decision.reason,
            client_request_id=(
                parsed_decision.client_request_id
            ),
            decided_at=decided_at,
            created_at=now,
        )

        # 7. 在同一个事务中按外键依赖顺序 flush。
        #
        #    flush 只把 SQL 发送到当前数据库事务，不会 commit。
        #    因此即使第三步 Approval 写入失败，后面的 rollback 仍会
        #    撤销前面已经 flush 的 TripDraft 和 TripVersion。
        try:
            # 7.1 先确保 trip_drafts 行存在。
            self.repository.flush()

            # 7.2 再写依赖 trip_drafts 的不可变版本。
            self.repository.add_version(
                version
            )
            self.repository.flush()

            # 7.3 最后写同时依赖 TripDraft 和 TripVersion 的审批记录。
            self.repository.add_approval(
                approval
            )
            self.repository.flush()

            # 7.4 三张表全部成功后才一次性提交事务。
            self.repository.commit()

        except IntegrityError:
            # 8. 并发重复请求可能在前面的查询后同时写入。
            #    唯一约束会保护数据库；回滚后再次查询已有结果。
            self.repository.rollback()

            concurrent_approval = (
                self.repository
                .get_approval_by_decision_event_id(
                    decision_event_id
                )
                or self.repository
                .get_approval_by_proposal_id(
                    parsed_proposal.proposal_id
                )
            )

            if concurrent_approval is not None:
                return self._build_existing_output(
                    approval=concurrent_approval,
                    expected_user_id=(
                        normalized_user_id
                    ),
                    reason=(
                        "并发请求已经完成同一 Proposal 的提交，"
                        "本次按幂等重放处理。"
                    ),
                )

            raise

        except Exception:
            # 9. 任何数据库异常都回滚整个事务，
            #    不允许出现有 TripDraft 但没有 ApprovalRecord 的半成品。
            self.repository.rollback()
            raise

        return self._build_output(
            trip=trip,
            version=version,
            approval=approval,
            status="committed",
            idempotent_replay=False,
            records_written=3,
            message=(
                "旅行方案已保存为内部模拟草稿；"
                "没有执行真实预订、付款或通知。"
            ),
        )

    # ==================================================================
    # Preconditions
    # ==================================================================

    def _validate_preconditions(
        self,
        *,
        proposal: TripProposal,
        verification: ProposalVerificationResult,
        decision: DecisionGateResult,
        pending_actions: Sequence[Mapping[str, Any]],
    ) -> None:
        """验证 CommitDraft 是否有权执行数据库副作用。"""

        # 1. 只有独立 Verifier 通过的 Proposal 才允许保存。
        if (
            verification.passed is not True
            or verification.status != "passed"
            or verification.next_action
            != "decision_gate"
        ):
            raise CommitPreconditionError(
                "Verifier 尚未通过，不能保存 TripDraft。"
            )

        # 2. DecisionGate 必须已经接受 approve，
        #    request_changes / cancel 不能触发 CommitDraft。
        if (
            decision.status != "accepted"
            or decision.accepted is not True
            or decision.decision != "approve"
            or decision.next_action
            != "commit_draft"
        ):
            raise CommitPreconditionError(
                "DecisionGate 没有接受 approve 决定，"
                "不能保存 TripDraft。"
            )

        if not decision.decision_event_id:
            raise CommitPreconditionError(
                "decision_result 缺少 decision_event_id，"
                "无法实现幂等提交。"
            )

        if not decision.action_id:
            raise CommitPreconditionError(
                "decision_result 缺少 action_id。"
            )

        if not decision.decided_at:
            raise CommitPreconditionError(
                "decision_result 缺少 decided_at。"
            )

        # 3. 人工批准必须仍然对应当前 Proposal 版本。
        if decision.proposal_id != proposal.proposal_id:
            raise CommitPreconditionError(
                "decision_result.proposal_id 与当前 Proposal 不一致。"
            )

        if decision.proposal_version != proposal.version:
            raise CommitPreconditionError(
                "decision_result.proposal_version 与当前 Proposal 不一致。"
            )

        if verification.proposal_id != proposal.proposal_id:
            raise CommitPreconditionError(
                "verifier_result.proposal_id 与当前 Proposal 不一致。"
            )

        # 4. 当前系统只能保存 draft-only 模拟草稿。
        boundary = proposal.execution_boundary

        if (
            boundary.draft_only is not True
            or boundary.real_booking_performed
            or boundary.real_payment_performed
            or boundary.real_notification_sent
        ):
            raise CommitPreconditionError(
                "Proposal execution_boundary 不符合内部模拟草稿边界。"
            )

        # 5. Pending Action 必须存在并已被 DecisionGate 标记 approved。
        matching_actions = [
            dict(item)
            for item in pending_actions
            if isinstance(item, Mapping)
            and str(item.get("action_id"))
            == str(decision.action_id)
        ]

        if len(matching_actions) != 1:
            raise CommitPreconditionError(
                "找不到唯一匹配的 approved pending action。"
            )

        action = matching_actions[0]

        if action.get("action_type") != "approve_proposal":
            raise CommitPreconditionError(
                "pending action 不是 approve_proposal。"
            )

        if action.get("status") not in {
            "approved",
            "committed",
        }:
            raise CommitPreconditionError(
                "pending action 尚未被人工批准。"
            )

        if (
            str(action.get("decision_event_id"))
            != str(decision.decision_event_id)
        ):
            raise CommitPreconditionError(
                "pending action 的 decision_event_id 与当前决定不一致。"
            )

    # ==================================================================
    # ORM Build / Update
    # ==================================================================

    def _build_new_trip(
        self,
        *,
        trip_id: str,
        trip_session_id: str | None,
        user_id: str,
        proposal: TripProposal,
        now: datetime,
    ) -> TripDraft:
        """根据 Proposal 锁定事实创建当前 TripDraft 行。"""

        overview = proposal.trip_overview
        outbound = proposal.selected_flights.outbound
        return_flight = (
            proposal.selected_flights.return_flight
        )
        hotel = proposal.selected_hotel
        budget = proposal.budget_summary

        known_subtotal = _money(
            budget.known_costs.get(
                "known_subtotal",
                0,
            )
        )

        return TripDraft(
            trip_id=trip_id,
            trip_session_id=trip_session_id,
            user_id=user_id,
            status="approved_simulated",
            current_version=proposal.version,
            current_proposal_id=(
                proposal.proposal_id
            ),
            origin=overview.origin,
            destination=overview.destination,
            start_date=overview.start_date,
            end_date=overview.end_date,
            people_count=overview.people_count,
            room_count=overview.room_count,
            outbound_flight_id=(
                outbound.flight_id
            ),
            return_flight_id=(
                return_flight.flight_id
            ),
            hotel_id=hotel.hotel_id,
            known_subtotal=known_subtotal,
            total_budget=_optional_money(
                budget.total_budget
            ),
            remaining_budget=_optional_money(
                budget.remaining_budget
            ),
            summary=proposal.summary,
            draft_only=True,
            created_at=now,
            updated_at=now,
        )

    def _update_existing_trip(
        self,
        *,
        trip: TripDraft,
        expected_trip_id: str,
        expected_user_id: str,
        proposal: TripProposal,
        now: datetime,
    ) -> None:
        """
        将同一旅行会话的较新 Proposal 设置为 current version。

        当前 v1 通常只在最终批准时提交一次；
        这个逻辑为未来“批准后再生成新版本”保留数据模型能力。
        """

        if trip.trip_id != expected_trip_id:
            raise CommitConflictError(
                "trip_session_id 已关联到另一条 TripDraft。"
            )

        if trip.user_id != expected_user_id:
            raise CommitConflictError(
                "同一 TripDraft 不能跨用户写入。"
            )

        if proposal.version <= trip.current_version:
            raise CommitConflictError(
                "新的 Proposal 版本必须高于当前已提交版本："
                f"current={trip.current_version}, "
                f"new={proposal.version}"
            )

        overview = proposal.trip_overview
        budget = proposal.budget_summary

        trip.status = "approved_simulated"
        trip.current_version = proposal.version
        trip.current_proposal_id = (
            proposal.proposal_id
        )
        trip.origin = overview.origin
        trip.destination = overview.destination
        trip.start_date = overview.start_date
        trip.end_date = overview.end_date
        trip.people_count = overview.people_count
        trip.room_count = overview.room_count
        trip.outbound_flight_id = (
            proposal.selected_flights.outbound.flight_id
        )
        trip.return_flight_id = (
            proposal.selected_flights.return_flight.flight_id
        )
        trip.hotel_id = (
            proposal.selected_hotel.hotel_id
        )
        trip.known_subtotal = _money(
            budget.known_costs.get(
                "known_subtotal",
                0,
            )
        )
        trip.total_budget = _optional_money(
            budget.total_budget
        )
        trip.remaining_budget = _optional_money(
            budget.remaining_budget
        )
        trip.summary = proposal.summary
        trip.draft_only = True
        trip.updated_at = now

    # ==================================================================
    # Output
    # ==================================================================

    def _build_existing_output(
        self,
        *,
        approval: ApprovalRecord,
        expected_user_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """将已经存在的提交转换成标准幂等结果。"""

        if approval.user_id != expected_user_id:
            raise CommitConflictError(
                "已有 approval 属于另一个用户，"
                "拒绝把该幂等结果返回给当前用户。"
            )

        trip = self.repository.get_trip(
            approval.trip_id
        )
        version = self.repository.get_version(
            approval.version_id
        )

        if trip is None or version is None:
            raise CommitConflictError(
                "已有 ApprovalRecord 缺少关联 TripDraft 或 TripVersion。"
            )

        return self._build_output(
            trip=trip,
            version=version,
            approval=approval,
            status="already_committed",
            idempotent_replay=True,
            records_written=0,
            message=reason,
        )

    def _build_output(
        self,
        *,
        trip: TripDraft,
        version: TripVersion,
        approval: ApprovalRecord,
        status: str,
        idempotent_replay: bool,
        records_written: int,
        message: str,
    ) -> dict[str, Any]:
        """把 ORM 对象转换成 Node 可写入 State 的轻量输出。"""

        trip_model = CommittedTripDraft(
            trip_id=trip.trip_id,
            trip_session_id=(
                trip.trip_session_id
            ),
            user_id=trip.user_id,
            status=trip.status,
            current_version=trip.current_version,
            current_proposal_id=(
                trip.current_proposal_id
            ),
            origin=trip.origin,
            destination=trip.destination,
            start_date=_date_to_iso(
                trip.start_date
            ),
            end_date=_date_to_iso(
                trip.end_date
            ),
            people_count=trip.people_count,
            room_count=trip.room_count,
            outbound_flight_id=(
                trip.outbound_flight_id
            ),
            return_flight_id=(
                trip.return_flight_id
            ),
            hotel_id=trip.hotel_id,
            known_subtotal=_decimal_to_float(
                trip.known_subtotal
            ),
            total_budget=(
                _optional_decimal_to_float(
                    trip.total_budget
                )
            ),
            remaining_budget=(
                _optional_decimal_to_float(
                    trip.remaining_budget
                )
            ),
            summary=trip.summary,
            draft_only=trip.draft_only,
            created_at=_datetime_to_iso(
                trip.created_at
            ),
            updated_at=_datetime_to_iso(
                trip.updated_at
            ),
        )

        version_model = CommittedTripVersion(
            version_id=version.version_id,
            trip_id=version.trip_id,
            version_number=(
                version.version_number
            ),
            proposal_id=version.proposal_id,
            locked_fact_hash=(
                version.locked_fact_hash
            ),
            proposal_schema_version=(
                version.proposal_schema_version
            ),
            generator_version=(
                version.generator_version
            ),
            created_by_decision_event_id=(
                version.created_by_decision_event_id
            ),
            created_at=_datetime_to_iso(
                version.created_at
            ),
        )

        approval_model = CommittedApprovalRecord(
            approval_id=approval.approval_id,
            decision_event_id=(
                approval.decision_event_id
            ),
            user_id=approval.user_id,
            trip_id=approval.trip_id,
            version_id=approval.version_id,
            proposal_id=approval.proposal_id,
            proposal_version=(
                approval.proposal_version
            ),
            action_id=approval.action_id,
            decision="approve",
            reason=approval.reason,
            client_request_id=(
                approval.client_request_id
            ),
            decided_at=_datetime_to_iso(
                approval.decided_at
            ),
            created_at=_datetime_to_iso(
                approval.created_at
            ),
        )

        result = CommitDraftResult(
            status=status,  # type: ignore[arg-type]
            idempotent_replay=(
                idempotent_replay
            ),
            records_written=records_written,
            trip_id=trip.trip_id,
            version_id=version.version_id,
            approval_id=approval.approval_id,
            user_id=trip.user_id,
            proposal_id=version.proposal_id,
            proposal_version=(
                version.version_number
            ),
            decision_event_id=(
                approval.decision_event_id
            ),
            committed_at=_datetime_to_iso(
                approval.created_at
            ),
            message=message,
            issues=[],
        )

        return {
            "trip_draft": (
                trip_model.to_state_dict()
            ),
            "trip_version": (
                version_model.to_state_dict()
            ),
            "approval_record": (
                approval_model.to_state_dict()
            ),
            "commit_result": (
                result.to_state_dict()
            ),
        }


# ======================================================================
# Stable IDs
# ======================================================================


def _build_trip_id(
    *,
    user_id: str,
    proposal_id: str,
    trip_session_id: str | None,
) -> str:
    """
    生成稳定 Trip ID。

    优先使用 trip_session_id：
        同一 LangGraph thread / trip session 的未来版本可以归入同一 Trip。

    没有 trip_session_id 时：
        使用 user_id + proposal_id，仍能保证重复提交得到同一 ID。
    """

    identity = (
        f"session:{trip_session_id}"
        if trip_session_id
        else f"proposal:{proposal_id}"
    )

    digest = hashlib.sha256(
        f"{user_id}|{identity}".encode(
            "utf-8"
        )
    ).hexdigest()[:24]

    return f"trip_{digest}"


def _build_version_id(
    *,
    trip_id: str,
    proposal_id: str,
    proposal_version: int,
) -> str:
    """根据 Trip、Proposal 和版本号生成稳定版本 ID。"""

    digest = hashlib.sha256(
        (
            f"{trip_id}|{proposal_id}|"
            f"{proposal_version}"
        ).encode("utf-8")
    ).hexdigest()[:24]

    return f"version_{digest}"


def _build_approval_id(
    decision_event_id: str,
) -> str:
    """根据 DecisionGate 事件生成稳定 Approval ID。"""

    digest = hashlib.sha256(
        decision_event_id.encode("utf-8")
    ).hexdigest()[:24]

    return f"approval_{digest}"


# ======================================================================
# Value Conversion
# ======================================================================


def _money(value: Any) -> Decimal:
    """
    将金额转换为两位小数 Decimal。

    使用 Decimal(str(value))，避免直接从二进制 float 构造 Decimal
    产生 0.1 -> 0.100000000000... 之类误差。
    """

    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CommitPreconditionError(
            f"无法解析金额：{value}"
        ) from exc

    if amount < 0:
        raise CommitPreconditionError(
            f"金额不能为负数：{value}"
        )

    return amount.quantize(
        Decimal("0.01")
    )


def _optional_money(
    value: Any,
) -> Decimal | None:
    """转换可空金额。"""

    if value is None:
        return None

    return _money(value)


def _parse_iso_datetime(
    value: str | None,
) -> datetime:
    """解析 DecisionGate 生成的 ISO 时间。"""

    if not value:
        raise CommitPreconditionError(
            "缺少 decided_at。"
        )

    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise CommitPreconditionError(
            f"decided_at 不是合法 ISO 时间：{value}"
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed


def _datetime_to_iso(
    value: datetime,
) -> str:
    """将数据库 datetime 转成 ISO 字符串。"""

    if value.tzinfo is None:
        value = value.replace(
            tzinfo=timezone.utc
        )

    return value.isoformat()


def _date_to_iso(value: date) -> str:
    """将数据库 Date 转成 YYYY-MM-DD。"""

    return value.isoformat()


def _decimal_to_float(
    value: Decimal,
) -> float:
    """将数据库 Decimal 转成 JSON 可序列化 float。"""

    return float(value)


def _optional_decimal_to_float(
    value: Decimal | None,
) -> float | None:
    """转换可空 Decimal。"""

    if value is None:
        return None

    return float(value)
