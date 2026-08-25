from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from sqlalchemy.orm import Session

from app.agents.state import TravelState
from app.db.session import SessionLocal
from app.repositories.trip_draft_repository import (
    TripDraftRepository,
)
from app.services.commit_draft_service import (
    CommitDraftService,
)


SessionFactory = Callable[[], Session]


def commit_draft_node(
    state: TravelState,
    session_factory: SessionFactory | None = None,
) -> dict[str, Any]:
    """
    CommitDraftNode：把人工批准的 Proposal 保存为内部模拟旅行草稿。

    读取 State：
        - user_id
        - trip_session_id，可选
        - proposal
        - verifier_result
        - decision_result
        - pending_actions

    写入 State：
        - trip_draft
        - trip_version
        - approval_record
        - commit_result
        - itinerary_status = "approved_simulated"
        - pending_actions，匹配动作更新为 committed
        - trace
        - errors，提交失败时

    这个节点会执行数据库副作用，因此必须满足：
        1. Verifier 已通过；
        2. DecisionGate 已接受 approve；
        3. proposal_id / version / action_id 一致；
        4. Proposal 仍然是 draft_only；
        5. user_id 在业务数据库中存在。

    这个节点不做：
        - 不执行真实酒店或航班预订；
        - 不执行真实支付；
        - 不发送邮件、短信或通知；
        - 不更新长期偏好；
        - 不生成最终 API 文案。

    数据库事务：
        trip_drafts
        + trip_versions
        + approval_records

        三张表要么全部提交成功，要么全部 rollback。
    """

    node_name = "commit_draft"
    started_at = perf_counter()
    factory = session_factory or SessionLocal

    try:
        # 1. 读取 CommitDraft 需要的最小 State。
        user_id = str(
            state.get("user_id") or ""
        ).strip()

        proposal = _require_mapping(
            state.get("proposal"),
            field_name="proposal",
        )

        verifier_result = _require_mapping(
            state.get("verifier_result"),
            field_name="verifier_result",
        )

        decision_result = _require_mapping(
            state.get("decision_result"),
            field_name="decision_result",
        )

        pending_actions = _require_mapping_list(
            state.get("pending_actions"),
            field_name="pending_actions",
        )

        trip_session_id = state.get(
            "trip_session_id"
        )

        trip_session_id = (
            str(trip_session_id).strip()
            if trip_session_id
            else None
        )

        # 2. 每次节点调用创建一个短生命周期 SQLAlchemy Session。
        #    Service 在该 Session 中完成原子事务和幂等保护。
        with factory() as session:
            repository = TripDraftRepository(
                session
            )

            service = CommitDraftService(
                repository
            )

            output = service.commit(
                user_id=user_id,
                proposal=proposal,
                verifier_result=(
                    verifier_result
                ),
                decision_result=(
                    decision_result
                ),
                pending_actions=(
                    pending_actions
                ),
                trip_session_id=(
                    trip_session_id
                ),
            )

        commit_result = output[
            "commit_result"
        ]

        # 3. 数据库提交成功后，将 Pending Action 标记 committed。
        #    DecisionGate 已经标记 approved；这里表示副作用真正完成。
        updated_pending_actions = (
            _mark_pending_action_committed(
                pending_actions=(
                    pending_actions
                ),
                action_id=str(
                    decision_result.get(
                        "action_id"
                    )
                    or ""
                ),
                trip_id=str(
                    commit_result.get(
                        "trip_id"
                    )
                    or ""
                ),
                version_id=str(
                    commit_result.get(
                        "version_id"
                    )
                    or ""
                ),
                approval_id=str(
                    commit_result.get(
                        "approval_id"
                    )
                    or ""
                ),
                committed_at=(
                    commit_result.get(
                        "committed_at"
                    )
                ),
            )
        )

        trace_item = _build_trace_item(
            status="success",
            started_at=started_at,
            input_summary=(
                f"user_id={user_id}, "
                f"proposal_id={proposal.get('proposal_id')}, "
                f"decision_event_id="
                f"{decision_result.get('decision_event_id')}"
            ),
            output_summary=(
                f"status={commit_result.get('status')}, "
                f"trip_id={commit_result.get('trip_id')}, "
                f"version_id={commit_result.get('version_id')}, "
                f"idempotent_replay="
                f"{commit_result.get('idempotent_replay')}"
            ),
        )

        return {
            "trip_draft": output[
                "trip_draft"
            ],
            "trip_version": output[
                "trip_version"
            ],
            "approval_record": output[
                "approval_record"
            ],
            "commit_result": (
                commit_result
            ),
            "itinerary_status": (
                "approved_simulated"
            ),
            "pending_actions": (
                updated_pending_actions
            ),
            "trace": [trace_item],
        }

    except Exception as exc:
        # 4. CommitDraft 失败时不伪造成功结果。
        #    Service 已负责 rollback，Node 只记录错误和 Trace。
        error_item = {
            "node": node_name,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        trace_item = _build_trace_item(
            status="failed",
            started_at=started_at,
            input_summary=(
                "approved proposal commit failed"
            ),
            output_summary=str(exc),
        )

        return {
            "trip_draft": {},
            "trip_version": {},
            "approval_record": {},
            "commit_result": {
                "status": "failed",
                "strategy_version": (
                    "transactional_trip_draft_commit_v1"
                ),
                "idempotent_replay": False,
                "records_written": 0,
                "trip_id": None,
                "version_id": None,
                "approval_id": None,
                "user_id": state.get(
                    "user_id"
                ),
                "proposal_id": (
                    state.get("proposal", {})
                    .get("proposal_id")
                    if isinstance(
                        state.get("proposal"),
                        dict,
                    )
                    else None
                ),
                "proposal_version": None,
                "decision_event_id": (
                    state.get(
                        "decision_result",
                        {},
                    ).get(
                        "decision_event_id"
                    )
                    if isinstance(
                        state.get(
                            "decision_result"
                        ),
                        dict,
                    )
                    else None
                ),
                "committed_at": None,
                "message": (
                    "内部旅行草稿保存失败。"
                ),
                "issues": [
                    {
                        "type": (
                            exc.__class__.__name__
                        ),
                        "message": str(exc),
                    }
                ],
            },
            "errors": [error_item],
            "trace": [trace_item],
        }


def _mark_pending_action_committed(
    *,
    pending_actions: Sequence[Mapping[str, Any]],
    action_id: str,
    trip_id: str,
    version_id: str,
    approval_id: str,
    committed_at: Any,
) -> list[dict[str, Any]]:
    """
    返回更新后的 Pending Action 副本。

    不原地修改旧 State，保持 LangGraph Node 的函数式更新方式。
    """

    updated: list[dict[str, Any]] = []

    for raw_item in pending_actions:
        item = dict(raw_item)

        if str(item.get("action_id")) == action_id:
            item.update(
                {
                    "status": "committed",
                    "trip_id": trip_id,
                    "version_id": version_id,
                    "approval_id": approval_id,
                    "committed_at": committed_at,
                }
            )

        updated.append(item)

    return updated


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

    result = [
        dict(item)
        for item in value
        if isinstance(item, Mapping)
    ]

    if len(result) != len(value):
        raise TypeError(
            f"state['{field_name}'] 的每一项都必须是 dict"
        )

    return result


def _build_trace_item(
    *,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """生成 CommitDraftNode Trace。"""

    latency_ms = int(
        (
            perf_counter()
            - started_at
        )
        * 1000
    )

    return {
        "node_name": "commit_draft",
        "tool_name": (
            "sqlalchemy_transactional_commit"
        ),
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "latency_ms": latency_ms,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }
