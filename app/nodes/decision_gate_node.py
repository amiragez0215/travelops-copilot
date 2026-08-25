from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal

from langgraph.types import Command, interrupt
from pydantic import ValidationError

from app.agents.state import TravelState
from app.schemas.decision_schema import (
    DecisionGateResult,
    DecisionHistoryItem,
    DecisionInterruptPayload,
    DecisionResponse,
)


DecisionDestination = Literal[
    "commit_draft",
    "revision_analyze",
    "cancel",
    "decision_gate",
    "final_response",
]


# ----------------------------------------------------------------------
# DecisionGateNode
# ----------------------------------------------------------------------
# 这个节点与普通 Node 最大的区别：
#
#     普通 Node：
#         State -> 立即返回 State Update
#
#     DecisionGateNode：
#         State -> interrupt(payload) -> Graph 暂停
#               -> 人工以后通过 Command(resume=...) 提交决定
#               -> 节点从开头重新执行
#               -> interrupt() 返回人工提交的值
#               -> Command(update=..., goto=...) 动态路由
#
# 注意：
#     interrupt() 不能放进 broad try/except 中。
#     它内部会抛出 LangGraph 专用中断信号；
#     如果被普通 except 捕获，Graph 就无法正确暂停。
# ----------------------------------------------------------------------


def decision_gate_node(
    state: TravelState,
) -> Command[DecisionDestination]:
    """
    DecisionGateNode：暂停已通过 Verifier 的旅行方案，等待人工决定。

    读取 State：
        - proposal
        - verifier_result
        - pending_actions
        - decision_gate_error，可选
        - decision_attempt_count，可选

    人工恢复输入：
        DecisionResponse

    接受后的 State 更新：
        - decision
        - decision_result
        - decision_history
        - decision_attempt_count
        - change_request，request_changes 时
        - itinerary_status，request_changes 时
        - pending_actions
        - decision_gate_error，清空
        - trace

    动态路由：
        approve
            -> commit_draft

        request_changes
            -> revision_analyze

        cancel
            -> cancel

        非法人工输入
            -> decision_gate，再次 interrupt

        上游 State 损坏
            -> final_response

    重要边界：
        1. 本节点不写数据库；
        2. 本节点不保存 TripDraft；
        3. 本节点不删除方案；
        4. 本节点只记录人工决定并路由；
        5. 真正副作用放到 CommitDraftNode / CancelNode。
    """

    node_name = "decision_gate"
    started_at = perf_counter()

    # ------------------------------------------------------------------
    # 1. 在 interrupt 前只做无副作用的 State 前置条件检查。
    # ------------------------------------------------------------------
    # 如果前置 State 已损坏，不应该让用户审批一份无法验证的 Proposal。
    # 这里可以捕获普通校验异常，因为 interrupt 尚未执行。
    try:
        payload = build_decision_interrupt_payload(state)
    except (TypeError, ValueError, ValidationError) as exc:
        return _build_precondition_failure_command(
            state=state,
            error=exc,
            started_at=started_at,
        )

    # ------------------------------------------------------------------
    # 2. 真正的 Human-in-the-loop 暂停点。
    # ------------------------------------------------------------------
    # 第一次运行：
    #     interrupt() 暂停 Graph，并把 payload 返回给调用方。
    #
    # 使用相同 thread_id 执行：
    #     graph.invoke(Command(resume={...}), config=...)
    #
    # 之后节点从开头重新运行；LangGraph 会让本次 interrupt()
    # 直接返回 Command(resume=...) 中的值。
    #
    # 这里绝对不要套在 except Exception 中。
    raw_response = interrupt(
        payload.to_state_dict()
    )

    attempt_count = _safe_int(
        state.get("decision_attempt_count"),
        default=0,
    ) + 1

    # ------------------------------------------------------------------
    # 3. 校验人工恢复数据。
    # ------------------------------------------------------------------
    # 人工输入也必须经过 Pydantic 和过期 Proposal 检查，
    # 不能因为它来自人类就直接信任。
    try:
        response = DecisionResponse.model_validate(
            raw_response
        )

        _validate_response_against_current_state(
            response=response,
            payload=payload,
        )

    except (ValidationError, TypeError, ValueError) as exc:
        # 4. 非法输入不是程序异常，也不应该结束整个工作流。
        #    当前 Node 返回到自己；下一次执行会再次 interrupt，
        #    并把 validation_error 一并展示给前端。
        validation_error = _build_validation_error(
            error=exc,
            attempt_count=attempt_count,
        )

        result = DecisionGateResult(
            status="invalid",
            accepted=False,
            decision=None,
            next_action="decision_gate",
            proposal_id=payload.proposal_id,
            proposal_version=payload.proposal_version,
            action_id=payload.action_id,
            attempt_count=attempt_count,
            validation_errors=[
                validation_error["message"]
            ],
        )

        trace_item = _build_trace_item(
            status="degraded",
            started_at=started_at,
            input_summary=(
                f"proposal_id={payload.proposal_id}, "
                f"attempt={attempt_count}"
            ),
            output_summary=(
                "human decision rejected; "
                "goto=decision_gate"
            ),
        )

        return Command(
            update={
                "decision_result": (
                    result.to_state_dict()
                ),
                "decision_gate_error": (
                    validation_error
                ),
                "decision_attempt_count": (
                    attempt_count
                ),
                "trace": [trace_item],
            },
            goto="decision_gate",
        )

    # ------------------------------------------------------------------
    # 5. 合法输入：构造稳定审计记录并动态路由。
    # ------------------------------------------------------------------
    decided_at = datetime.now(
        timezone.utc
    ).isoformat()

    next_action = _decision_to_destination(
        response.decision
    )

    decision_event_id = _build_decision_event_id(
        response=response,
        proposal_version=(
            payload.proposal_version
        ),
    )

    result = DecisionGateResult(
        status="accepted",
        accepted=True,
        decision=response.decision,
        next_action=next_action,
        proposal_id=payload.proposal_id,
        proposal_version=payload.proposal_version,
        action_id=payload.action_id,
        decision_event_id=decision_event_id,
        client_request_id=(
            response.client_request_id
        ),
        attempt_count=attempt_count,
        decided_at=decided_at,
        change_request_provided=(
            response.decision
            == "request_changes"
        ),
        reason=response.reason,
        validation_errors=[],
    )

    history_item = DecisionHistoryItem(
        decision_event_id=decision_event_id,
        proposal_id=payload.proposal_id,
        proposal_version=payload.proposal_version,
        action_id=payload.action_id,
        decision=response.decision,
        next_action=next_action,
        decided_at=decided_at,
        change_request=(
            response.change_request
            if response.decision
            == "request_changes"
            else None
        ),
        reason=response.reason,
        client_request_id=(
            response.client_request_id
        ),
    )

    # 6. 将当前 pending approval 标记为用户已经处理。
    updated_pending_actions = (
        _mark_pending_action_decided(
            pending_actions=state.get(
                "pending_actions"
            ),
            action_id=payload.action_id,
            decision=response.decision,
            decision_event_id=(
                decision_event_id
            ),
            decided_at=decided_at,
            reason=response.reason,
        )
    )

    update: dict[str, Any] = {
        "decision": response.decision,
        "decision_result": (
            result.to_state_dict()
        ),
        # decision_history 使用 add reducer，
        # 每轮 Proposal 的人工决定都会被追加而不是覆盖。
        "decision_history": [
            history_item.to_state_dict()
        ],
        "decision_attempt_count": (
            attempt_count
        ),
        "decision_gate_error": {},
        "pending_actions": (
            updated_pending_actions
        ),
    }

    # 7. request_changes 只保存原始修改文本。
    #    RevisionAnalyzeNode 后面再负责把它转成结构化 RevisionPatch。
    if response.decision == "request_changes":
        update["change_request"] = {
            "raw_message": (
                response.change_request
            ),
            "base_proposal_id": (
                payload.proposal_id
            ),
            "base_proposal_version": (
                payload.proposal_version
            ),
            "submitted_at": decided_at,
        }

        update[
            "itinerary_status"
        ] = "revision_requested"

    else:
        # 8. approve / cancel 清空旧修改请求，避免上一轮残留。
        update["change_request"] = {}

    trace_item = _build_trace_item(
        status="success",
        started_at=started_at,
        input_summary=(
            f"proposal_id={payload.proposal_id}, "
            f"attempt={attempt_count}"
        ),
        output_summary=(
            f"decision={response.decision}, "
            f"goto={next_action}"
        ),
    )

    update["trace"] = [trace_item]

    # 9. Command 同时完成 State 更新与动态路由。
    #    最终组图时不要再给 DecisionGateNode 添加普通静态出边，
    #    否则 Command.goto 和静态边会同时执行。
    return Command(
        update=update,
        goto=next_action,
    )


# ======================================================================
# Interrupt Payload
# ======================================================================


def build_decision_interrupt_payload(
    state: TravelState,
) -> DecisionInterruptPayload:
    """
    根据已经通过 Verifier 的 State 构建精简人工审批 Payload。

    这是一个纯函数：
        - 不调用 interrupt；
        - 不写 State；
        - 不写数据库；
        - 可以独立单元测试。
    """

    proposal = _require_mapping(
        state.get("proposal"),
        field_name="proposal",
    )

    verifier_result = _require_mapping(
        state.get("verifier_result"),
        field_name="verifier_result",
    )

    # 1. 只有通过独立 Verifier 的 Proposal 才能进入人工审批。
    if verifier_result.get("passed") is not True:
        raise ValueError(
            "verifier_result.passed 必须为 True，"
            "未通过 Verifier 的方案不能进入 DecisionGate"
        )

    if (
        verifier_result.get("next_action")
        != "decision_gate"
    ):
        raise ValueError(
            "verifier_result.next_action 必须为 decision_gate"
        )

    if (
        verifier_result.get("approval_required")
        is not True
    ):
        raise ValueError(
            "verifier_result.approval_required 必须为 True"
        )

    proposal_id = str(
        proposal.get("proposal_id")
        or ""
    ).strip()

    if not proposal_id:
        raise ValueError(
            "proposal.proposal_id 不能为空"
        )

    proposal_version = _safe_int(
        proposal.get("version"),
        default=0,
    )

    if proposal_version < 1:
        raise ValueError(
            "proposal.version 必须大于等于 1"
        )

    # 2. 找到 VerifierNode 为当前 Proposal 创建的 pending approval。
    pending_action = _find_pending_approval_action(
        pending_actions=state.get(
            "pending_actions"
        ),
        proposal_id=proposal_id,
    )

    action_id = str(
        pending_action.get("action_id")
        or ""
    ).strip()

    if not action_id:
        raise ValueError(
            "pending approval 缺少 action_id"
        )

    decision_gate_error = state.get(
        "decision_gate_error"
    )

    validation_error = (
        dict(decision_gate_error)
        if isinstance(
            decision_gate_error,
            Mapping,
        )
        and decision_gate_error
        else None
    )

    return DecisionInterruptPayload(
        action_id=action_id,
        proposal_id=proposal_id,
        proposal_version=proposal_version,
        message=(
            "旅行方案已通过系统校验。"
            "请批准、提交修改意见或取消本次方案。"
        ),
        available_actions=[
            {
                "decision": "approve",
                "label": "同意方案",
                "description": (
                    "保存内部模拟行程草稿；"
                    "不会执行真实预订或付款。"
                ),
                "requires_text_input": False,
            },
            {
                "decision": "request_changes",
                "label": "修改方案",
                "description": (
                    "提交具体修改说明，"
                    "进入 RevisionAnalyzeNode。"
                ),
                "requires_text_input": True,
            },
            {
                "decision": "cancel",
                "label": "取消",
                "description": (
                    "取消当前方案，不执行任何真实外部动作。"
                ),
                "requires_text_input": False,
            },
        ],
        proposal_summary=(
            _build_proposal_summary(
                proposal
            )
        ),
        pending_action=(
            _sanitize_pending_action(
                pending_action
            )
        ),
        validation_error=validation_error,
    )


# ======================================================================
# Validation / Routing
# ======================================================================


def _validate_response_against_current_state(
    *,
    response: DecisionResponse,
    payload: DecisionInterruptPayload,
) -> None:
    """
    检查恢复输入是否仍然对应当前 Proposal 和 Pending Action。

    这是简单的乐观并发控制：
        用户打开 Proposal v1 的审批页面；
        另一个操作已经生成 Proposal v2；
        用户再提交 v1 的批准时，系统必须拒绝。
    """

    if response.proposal_id != payload.proposal_id:
        raise ValueError(
            "提交的 proposal_id 已过期或不匹配："
            f"expected={payload.proposal_id}, "
            f"actual={response.proposal_id}"
        )

    if response.action_id != payload.action_id:
        raise ValueError(
            "提交的 action_id 已过期或不匹配："
            f"expected={payload.action_id}, "
            f"actual={response.action_id}"
        )


def _decision_to_destination(
    decision: str,
) -> Literal[
    "commit_draft",
    "revision_analyze",
    "cancel",
]:
    """把人工决定转换成下一个业务节点。"""

    if decision == "approve":
        return "commit_draft"

    if decision == "request_changes":
        return "revision_analyze"

    if decision == "cancel":
        return "cancel"

    raise ValueError(
        f"unsupported decision: {decision}"
    )


# ======================================================================
# State Update Helpers
# ======================================================================


def _mark_pending_action_decided(
    *,
    pending_actions: Any,
    action_id: str,
    decision: str,
    decision_event_id: str,
    decided_at: str,
    reason: str | None,
) -> list[dict[str, Any]]:
    """
    返回更新后的 pending_actions 副本。

    本函数不会原地修改 State 中的旧列表，
    保持 LangGraph Node 的函数式更新习惯。
    """

    if not isinstance(
        pending_actions,
        Sequence,
    ) or isinstance(
        pending_actions,
        (str, bytes),
    ):
        return []

    status_by_decision = {
        "approve": "approved",
        "request_changes": (
            "changes_requested"
        ),
        "cancel": "cancelled",
    }

    updated: list[dict[str, Any]] = []

    for raw_item in pending_actions:
        if not isinstance(
            raw_item,
            Mapping,
        ):
            continue

        item = dict(raw_item)

        if str(item.get("action_id")) == action_id:
            item.update(
                {
                    "status": (
                        status_by_decision[
                            decision
                        ]
                    ),
                    "decision": decision,
                    "decision_event_id": (
                        decision_event_id
                    ),
                    "decided_at": decided_at,
                    "decision_reason": reason,
                }
            )

        updated.append(item)

    return updated


def _build_decision_event_id(
    *,
    response: DecisionResponse,
    proposal_version: int,
) -> str:
    """
    根据决定内容生成稳定事件 ID。

    当前节点没有数据库写入副作用，
    但稳定 ID 可以让后续 CommitDraftNode 做幂等保护。
    """

    payload = {
        "proposal_id": response.proposal_id,
        "proposal_version": proposal_version,
        "action_id": response.action_id,
        "decision": response.decision,
        "change_request": (
            response.change_request
        ),
        "reason": response.reason,
        "client_request_id": (
            response.client_request_id
        ),
    }

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()[:20]

    return f"decision_{digest}"


# ======================================================================
# Payload Summary Helpers
# ======================================================================


def _build_proposal_summary(
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从完整 Proposal 中提取审批页面真正需要的小型摘要。
    """

    trip = _read_mapping(
        proposal.get("trip_overview")
    )

    hotel = _read_mapping(
        proposal.get("selected_hotel")
    )

    flights = _read_mapping(
        proposal.get("selected_flights")
    )

    outbound = _read_mapping(
        flights.get("outbound")
    )

    return_flight = _read_mapping(
        flights.get("return_flight")
    )

    budget = _read_mapping(
        proposal.get("budget_summary")
    )

    known_costs = _read_mapping(
        budget.get("known_costs")
    )

    highlights = proposal.get(
        "highlights"
    )

    return {
        "summary": proposal.get("summary"),
        "origin": trip.get("origin"),
        "destination": trip.get("destination"),
        "start_date": trip.get("start_date"),
        "end_date": trip.get("end_date"),
        "days": trip.get("days"),
        "people_count": trip.get(
            "people_count"
        ),
        "hotel": {
            "hotel_id": hotel.get(
                "hotel_id"
            ),
            "name": hotel.get("name"),
            "estimated_total_price": (
                hotel.get(
                    "estimated_total_price"
                )
            ),
        },
        "outbound_flight": {
            "flight_id": outbound.get(
                "flight_id"
            ),
            "flight_no": outbound.get(
                "flight_no"
            ),
            "depart_time": outbound.get(
                "depart_time"
            ),
            "arrive_time": outbound.get(
                "arrive_time"
            ),
        },
        "return_flight": {
            "flight_id": return_flight.get(
                "flight_id"
            ),
            "flight_no": return_flight.get(
                "flight_no"
            ),
            "depart_time": return_flight.get(
                "depart_time"
            ),
            "arrive_time": return_flight.get(
                "arrive_time"
            ),
        },
        "known_subtotal": known_costs.get(
            "known_subtotal"
        ),
        "total_budget": budget.get(
            "total_budget"
        ),
        "remaining_budget": budget.get(
            "remaining_budget"
        ),
        "highlights": (
            [str(item) for item in highlights[:5]]
            if isinstance(highlights, list)
            else []
        ),
        "draft_only": _read_mapping(
            proposal.get(
                "execution_boundary"
            )
        ).get("draft_only", True),
    }


def _sanitize_pending_action(
    pending_action: Mapping[str, Any],
) -> dict[str, Any]:
    """只暴露审批页面需要的 Pending Action 字段。"""

    return {
        "action_id": pending_action.get(
            "action_id"
        ),
        "action_type": pending_action.get(
            "action_type"
        ),
        "description": pending_action.get(
            "description"
        ),
        "requires_approval": (
            pending_action.get(
                "requires_approval"
            )
        ),
        "status": pending_action.get(
            "status"
        ),
        "payload_summary": dict(
            _read_mapping(
                pending_action.get(
                    "payload_summary"
                )
            )
        ),
    }


def _find_pending_approval_action(
    *,
    pending_actions: Any,
    proposal_id: str,
) -> dict[str, Any]:
    """查找当前 Proposal 对应的待审批动作。"""

    if not isinstance(
        pending_actions,
        Sequence,
    ) or isinstance(
        pending_actions,
        (str, bytes),
    ):
        raise TypeError(
            "state['pending_actions'] 必须是 list"
        )

    matches: list[dict[str, Any]] = []

    for raw_item in pending_actions:
        if not isinstance(
            raw_item,
            Mapping,
        ):
            continue

        item = dict(raw_item)
        payload_summary = _read_mapping(
            item.get("payload_summary")
        )

        if (
            item.get("action_type")
            == "approve_proposal"
            and item.get("requires_approval")
            is True
            and item.get("status")
            == "pending"
            and payload_summary.get(
                "proposal_id"
            )
            == proposal_id
        ):
            matches.append(item)

    if len(matches) != 1:
        raise ValueError(
            "当前 Proposal 必须且只能存在一个 pending approve_proposal 动作，"
            f"实际数量：{len(matches)}"
        )

    return matches[0]


# ======================================================================
# Error / Trace Helpers
# ======================================================================


def _build_validation_error(
    *,
    error: Exception,
    attempt_count: int,
) -> dict[str, Any]:
    """把 Pydantic 或过期决定错误转换成前端可读结构。"""

    if isinstance(error, ValidationError):
        details = [
            {
                "path": ".".join(
                    str(part)
                    for part in item.get(
                        "loc",
                        (),
                    )
                ),
                "message": item.get("msg"),
                "type": item.get("type"),
            }
            for item in error.errors()
        ]

        message = "; ".join(
            str(item.get("message"))
            for item in details[:5]
        )
    else:
        details = []
        message = str(error)

    return {
        "error_type": (
            "invalid_human_decision"
        ),
        "message": message,
        "attempt_count": attempt_count,
        "details": details,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def _build_precondition_failure_command(
    *,
    state: TravelState,
    error: Exception,
    started_at: float,
) -> Command[Literal["final_response"]]:
    """
    上游审批前置 State 损坏时安全停止。

    这种错误发生在 interrupt 之前，
    所以不会把一份无法验证的方案交给人工审批。
    """

    attempt_count = _safe_int(
        state.get("decision_attempt_count"),
        default=0,
    )

    result = DecisionGateResult(
        status="failed",
        accepted=False,
        decision=None,
        next_action="final_response",
        attempt_count=attempt_count,
        validation_errors=[str(error)],
    )

    error_item = {
        "node": "decision_gate",
        "type": error.__class__.__name__,
        "message": str(error),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    trace_item = _build_trace_item(
        status="failed",
        started_at=started_at,
        input_summary=(
            "decision gate precondition failed"
        ),
        output_summary=str(error),
    )

    return Command(
        update={
            "decision_result": (
                result.to_state_dict()
            ),
            "decision_gate_error": {
                "error_type": (
                    "decision_gate_precondition_failed"
                ),
                "message": str(error),
            },
            "errors": [error_item],
            "trace": [trace_item],
        },
        goto="final_response",
    )


def _build_trace_item(
    *,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """生成统一 DecisionGate Trace。"""

    latency_ms = int(
        (
            perf_counter()
            - started_at
        )
        * 1000
    )

    return {
        "node_name": "decision_gate",
        "tool_name": (
            "langgraph_interrupt_command"
        ),
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "latency_ms": latency_ms,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ======================================================================
# Defensive Helpers
# ======================================================================


def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    """读取必需的 dict-like State 字段。"""

    if not isinstance(value, Mapping):
        raise TypeError(
            f"state['{field_name}'] 必须是 dict"
        )

    return dict(value)


def _read_mapping(
    value: Any,
) -> dict[str, Any]:
    """安全读取可选 Mapping。"""

    return (
        dict(value)
        if isinstance(value, Mapping)
        else {}
    )


def _safe_int(
    value: Any,
    *,
    default: int,
) -> int:
    """安全转换 int。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default
