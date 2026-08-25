from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal

from pydantic import ValidationError

from app.agents.state import TravelState
from app.schemas.cancel_schema import CancelResult
from app.schemas.decision_schema import DecisionGateResult
from app.schemas.proposal_schema import TripProposal
from app.schemas.verifier_schema import ProposalVerificationResult


CancelDestination = Literal["final_response"]


def cancel_node(
    state: TravelState,
) -> dict[str, Any]:
    """
    CancelNode：完成用户取消当前旅行方案的内部收尾。

    读取 State：
        - proposal
        - verifier_result
        - decision_result
        - pending_actions
        - cancel_result，可选，用于幂等重放判断

    写入 State：
        - cancel_result
        - itinerary_status = "rejected"
        - pending_actions，补充取消已完成的信息
        - change_request，清空
        - proposal_feedback，清空
        - trace
        - errors，程序或状态异常时

    这个节点不做：
        - 不删除 Proposal；
        - 不删除 RAG Evidence；
        - 不删除 LangGraph Checkpoint；
        - 不删除已经存在的历史 TripDraft；
        - 不执行真实酒店或航班取消；
        - 不生成最终 API 文案。

    为什么 CancelNode 不写数据库？
        当前 v1 的取消发生在 CommitDraftNode 之前，
        用户只是拒绝当前内部 Proposal，并没有产生 TripDraft 业务记录。

        DecisionGateNode 已经把取消决定追加到：
            state["decision_history"]

        LangGraph Checkpointer 也保存了完整执行历史。

        如果未来产品要求把“未批准方案的取消行为”永久保存在业务数据库，
        可以再增加 cancellation_records 表；当前项目没有必要扩大范围。
    """

    node_name = "cancel"
    started_at = perf_counter()

    try:
        # 1. 解析并验证上游三个核心数据合同。
        #    CancelNode 不能只依赖 Graph 路由“应该正确”，
        #    执行状态变更前仍要再次检查当前 Proposal、Verifier 和人工决定。
        proposal = TripProposal.model_validate(
            _require_mapping(
                state.get("proposal"),
                field_name="proposal",
            )
        )

        verifier_result = (
            ProposalVerificationResult.model_validate(
                _require_mapping(
                    state.get("verifier_result"),
                    field_name="verifier_result",
                )
            )
        )

        decision_result = (
            DecisionGateResult.model_validate(
                _require_mapping(
                    state.get("decision_result"),
                    field_name="decision_result",
                )
            )
        )

        pending_actions = _require_mapping_list(
            state.get("pending_actions"),
            field_name="pending_actions",
        )

        # 2. 检查取消是否有权执行。
        _validate_cancel_preconditions(
            proposal=proposal,
            verifier_result=verifier_result,
            decision_result=decision_result,
            pending_actions=pending_actions,
        )

        decision_event_id = str(
            decision_result.decision_event_id
            or ""
        )

        # 3. 如果相同取消事件已经完成，返回幂等重放结果。
        #
        #    CancelNode 当前没有数据库副作用，重复执行本身也不会破坏数据；
        #    显式识别幂等重放可以让 Trace 和 FinalResponse 更清楚。
        existing_cancel_result = state.get(
            "cancel_result"
        )

        if _is_same_completed_cancel(
            existing_cancel_result,
            decision_event_id=decision_event_id,
        ):
            result = CancelResult(
                status="already_cancelled",
                idempotent_replay=True,
                proposal_id=proposal.proposal_id,
                proposal_version=proposal.version,
                action_id=decision_result.action_id,
                decision_event_id=decision_event_id,
                cancelled_at=(
                    decision_result.decided_at
                ),
                reason=decision_result.reason,
                message=(
                    "相同取消决定已经处理；"
                    "本次属于幂等重放，没有执行新的操作。"
                ),
                issues=[],
            )
        else:
            # 4. 使用 DecisionGate 的 decided_at 作为稳定取消时间。
            #    不使用 datetime.now() 重新生成业务时间，
            #    这样 Checkpoint 重放时结果保持一致。
            result = CancelResult(
                status="cancelled",
                idempotent_replay=False,
                proposal_id=proposal.proposal_id,
                proposal_version=proposal.version,
                action_id=decision_result.action_id,
                decision_event_id=decision_event_id,
                cancelled_at=(
                    decision_result.decided_at
                ),
                reason=decision_result.reason,
                message=(
                    "当前旅行方案已取消。"
                    "系统没有执行真实预订、付款、出票或外部通知。"
                ),
                issues=[],
            )

        # 5. DecisionGate 已经把 Pending Action 标记为 cancelled。
        #    当前节点只补充“取消收尾已完成”的可追踪字段。
        updated_pending_actions = (
            _mark_pending_action_cancel_finalized(
                pending_actions=pending_actions,
                action_id=str(
                    decision_result.action_id
                    or ""
                ),
                decision_event_id=(
                    decision_event_id
                ),
                finalized_at=(
                    decision_result.decided_at
                ),
            )
        )

        trace_item = _build_trace_item(
            status="success",
            started_at=started_at,
            input_summary=(
                f"proposal_id={proposal.proposal_id}, "
                f"decision_event_id={decision_event_id}"
            ),
            output_summary=(
                f"status={result.status}, "
                f"idempotent_replay={result.idempotent_replay}"
            ),
        )

        return {
            "cancel_result": (
                result.to_state_dict()
            ),

            # 6. State 级生命周期状态标记为 rejected。
            #    Proposal 本身仍然保留，便于最终响应、审计和调试。
            "itinerary_status": "rejected",

            "pending_actions": (
                updated_pending_actions
            ),

            # 7. cancel 分支不应携带上一轮修改请求或 Verifier 修复反馈。
            "change_request": {},
            "proposal_feedback": {},

            "trace": [trace_item],
        }

    except Exception as exc:
        # 8. 状态前置条件不满足时，不伪造取消成功。
        #    FinalResponseNode 后续根据 cancel_result.status=failed
        #    生成统一的失败响应。
        error_item = {
            "node": node_name,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        result = CancelResult(
            status="failed",
            idempotent_replay=False,
            proposal_id=_read_nested_string(
                state,
                "proposal",
                "proposal_id",
            ),
            proposal_version=(
                _read_nested_positive_int(
                    state,
                    "proposal",
                    "version",
                )
            ),
            action_id=_read_nested_string(
                state,
                "decision_result",
                "action_id",
            ),
            decision_event_id=(
                _read_nested_string(
                    state,
                    "decision_result",
                    "decision_event_id",
                )
            ),
            cancelled_at=None,
            reason=None,
            message=(
                "当前旅行方案取消失败，"
                "系统没有执行任何真实外部动作。"
            ),
            issues=[
                {
                    "type": (
                        exc.__class__.__name__
                    ),
                    "message": str(exc),
                }
            ],
        )

        trace_item = _build_trace_item(
            status="failed",
            started_at=started_at,
            input_summary=(
                "cancel precondition validation failed"
            ),
            output_summary=str(exc),
        )

        return {
            "cancel_result": (
                result.to_state_dict()
            ),
            "errors": [error_item],
            "trace": [trace_item],
        }


def route_after_cancel(
    state: TravelState,
) -> CancelDestination:
    """
    CancelNode 后固定进入 FinalResponseNode。

    成功时：
        FinalResponseNode 返回“方案已取消”。

    失败时：
        FinalResponseNode 返回“取消处理失败”。

    这个函数主要用于保持工作流代码可读；
    完整 Graph 也可以直接添加静态边：

        builder.add_edge("cancel", "final_response")
    """

    del state
    return "final_response"


# ======================================================================
# Preconditions
# ======================================================================


def _validate_cancel_preconditions(
    *,
    proposal: TripProposal,
    verifier_result: ProposalVerificationResult,
    decision_result: DecisionGateResult,
    pending_actions: Sequence[Mapping[str, Any]],
) -> None:
    """
    验证当前取消决定仍然对应同一个、已通过 Verifier 的 Proposal。

    这些检查和 DecisionGate 有少量重复，属于副作用边界前的防御式校验。
    即使未来 Graph 路由发生错误，CancelNode 也不会错误取消其他 Proposal。
    """

    # 1. 只有经过 Verifier 并进入 DecisionGate 的 Proposal 才能被取消。
    if (
        verifier_result.passed is not True
        or verifier_result.status != "passed"
        or verifier_result.next_action
        != "decision_gate"
    ):
        raise ValueError(
            "Verifier 尚未通过，不能确认当前取消决定。"
        )

    # 2. DecisionGate 必须已经接受 cancel。
    if (
        decision_result.status
        != "accepted"
        or decision_result.accepted is not True
        or decision_result.decision
        != "cancel"
        or decision_result.next_action
        != "cancel"
    ):
        raise ValueError(
            "DecisionGate 没有接受 cancel 决定。"
        )

    if not decision_result.decision_event_id:
        raise ValueError(
            "decision_result 缺少 decision_event_id。"
        )

    if not decision_result.action_id:
        raise ValueError(
            "decision_result 缺少 action_id。"
        )

    if not decision_result.decided_at:
        raise ValueError(
            "decision_result 缺少 decided_at。"
        )

    # 3. 取消决定必须仍然对应当前 Proposal ID 和版本。
    if (
        decision_result.proposal_id
        != proposal.proposal_id
    ):
        raise ValueError(
            "decision_result.proposal_id 与当前 Proposal 不一致。"
        )

    if (
        decision_result.proposal_version
        != proposal.version
    ):
        raise ValueError(
            "decision_result.proposal_version 与当前 Proposal 不一致。"
        )

    if (
        verifier_result.proposal_id
        != proposal.proposal_id
    ):
        raise ValueError(
            "verifier_result.proposal_id 与当前 Proposal 不一致。"
        )

    # 4. 当前项目只处理内部 draft-only 方案。
    boundary = proposal.execution_boundary

    if (
        boundary.draft_only is not True
        or boundary.real_booking_performed
        or boundary.real_payment_performed
        or boundary.real_notification_sent
    ):
        raise ValueError(
            "Proposal execution_boundary 不符合内部模拟方案边界。"
        )

    # 5. 找到唯一匹配的 Pending Action。
    matching_actions = [
        dict(item)
        for item in pending_actions
        if isinstance(item, Mapping)
        and str(item.get("action_id"))
        == str(decision_result.action_id)
    ]

    if len(matching_actions) != 1:
        raise ValueError(
            "找不到唯一匹配的 cancelled pending action。"
        )

    action = matching_actions[0]

    if (
        action.get("action_type")
        != "approve_proposal"
    ):
        raise ValueError(
            "pending action 不是 approve_proposal。"
        )

    # DecisionGateNode 在 cancel 分支会把状态更新为 cancelled。
    if action.get("status") != "cancelled":
        raise ValueError(
            "pending action 尚未被标记为 cancelled。"
        )

    if (
        str(action.get("decision_event_id"))
        != str(
            decision_result.decision_event_id
        )
    ):
        raise ValueError(
            "pending action 的 decision_event_id 与当前取消决定不一致。"
        )


# ======================================================================
# Idempotency / Pending Action
# ======================================================================


def _is_same_completed_cancel(
    value: Any,
    *,
    decision_event_id: str,
) -> bool:
    """
    判断 State 中是否已经保存同一个取消事件的成功结果。
    """

    if not isinstance(value, Mapping):
        return False

    return (
        value.get("status")
        in {"cancelled", "already_cancelled"}
        and str(
            value.get("decision_event_id")
            or ""
        )
        == decision_event_id
    )


def _mark_pending_action_cancel_finalized(
    *,
    pending_actions: Sequence[Mapping[str, Any]],
    action_id: str,
    decision_event_id: str,
    finalized_at: str | None,
) -> list[dict[str, Any]]:
    """
    返回更新后的 Pending Action 副本。

    status 继续保持 cancelled；额外字段表示 CancelNode 已完成内部收尾。
    """

    updated: list[dict[str, Any]] = []

    for raw_item in pending_actions:
        item = dict(raw_item)

        if str(item.get("action_id")) == action_id:
            item.update(
                {
                    "status": "cancelled",
                    "cancel_finalized": True,
                    "cancel_finalized_at": (
                        finalized_at
                    ),
                    "cancel_decision_event_id": (
                        decision_event_id
                    ),
                }
            )

        updated.append(item)

    return updated


# ======================================================================
# Defensive Helpers
# ======================================================================


def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    """读取必需 dict State。"""

    if not isinstance(value, Mapping):
        raise TypeError(
            f"state['{field_name}'] 必须是 dict"
        )

    return dict(value)


def _require_mapping_list(
    value: Any,
    *,
    field_name: str,
) -> list[dict[str, Any]]:
    """读取必需的 dict 列表 State。"""

    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes),
    ):
        raise TypeError(
            f"state['{field_name}'] 必须是 list"
        )

    result: list[dict[str, Any]] = []

    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"state['{field_name}'][{index}] 必须是 dict"
            )

        result.append(dict(item))

    return result


def _read_nested_string(
    state: Mapping[str, Any],
    outer_key: str,
    inner_key: str,
) -> str | None:
    """从嵌套 State 中安全读取字符串。"""

    outer = state.get(outer_key)

    if not isinstance(outer, Mapping):
        return None

    value = outer.get(inner_key)

    if value is None:
        return None

    normalized = str(value).strip()
    return normalized or None


def _read_nested_positive_int(
    state: Mapping[str, Any],
    outer_key: str,
    inner_key: str,
) -> int | None:
    """从嵌套 State 中安全读取正整数。"""

    outer = state.get(outer_key)

    if not isinstance(outer, Mapping):
        return None

    try:
        value = int(outer.get(inner_key))
    except (TypeError, ValueError):
        return None

    return value if value >= 1 else None


def _build_trace_item(
    *,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """构造统一节点 Trace。"""

    latency_ms = int(
        (perf_counter() - started_at)
        * 1000
    )

    return {
        "node_name": "cancel",
        "tool_name": (
            "deterministic_proposal_cancel"
        ),
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "latency_ms": latency_ms,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }
