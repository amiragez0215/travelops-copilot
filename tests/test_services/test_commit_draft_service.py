from __future__ import annotations

from copy import deepcopy

import pytest

from app.db.init_db import init_db
from app.db.models import (
    ApprovalRecord,
    TripDraft,
    TripVersion,
)
from app.db.seed_data import (
    DEFAULT_SEED_DATA,
    seed_database,
)
from app.db.session import (
    create_db_engine,
    create_session_factory,
)
from app.repositories.trip_draft_repository import (
    TripDraftRepository,
)
from app.services.commit_draft_service import (
    CommitConflictError,
    CommitDraftService,
    CommitPreconditionError,
    UserNotFoundError,
)
from tests.test_verifiers.test_proposal_verifier import (
    _generated_state,
    _verify,
)


def _session_factory():
    """创建带用户 Seed 的内存数据库。"""

    engine = create_db_engine(
        "sqlite:///:memory:"
    )
    init_db(engine)
    factory = create_session_factory(engine)

    with factory() as session:
        seed_database(
            session,
            DEFAULT_SEED_DATA,
            clear=True,
        )

    return factory


def _approved_context():
    """构造已经通过 Verifier 和 DecisionGate 的提交上下文。"""

    state = _generated_state()
    verified = _verify(state)

    proposal = state["proposal"]
    verifier_result = verified[
        "verifier_result"
    ]
    action = deepcopy(
        verified["pending_actions"][0]
    )

    decision_event_id = (
        "decision_test_commit_001"
    )

    action.update(
        {
            "status": "approved",
            "decision": "approve",
            "decision_event_id": (
                decision_event_id
            ),
            "decided_at": (
                "2026-07-24T12:00:00+00:00"
            ),
        }
    )

    decision_result = {
        "status": "accepted",
        "strategy_version": (
            "langgraph_hitl_decision_gate_v1"
        ),
        "accepted": True,
        "decision": "approve",
        "next_action": "commit_draft",
        "proposal_id": (
            proposal["proposal_id"]
        ),
        "proposal_version": (
            proposal["version"]
        ),
        "action_id": action["action_id"],
        "decision_event_id": (
            decision_event_id
        ),
        "client_request_id": (
            "client_request_001"
        ),
        "attempt_count": 1,
        "decided_at": (
            "2026-07-24T12:00:00+00:00"
        ),
        "change_request_provided": False,
        "reason": "同意当前方案",
        "validation_errors": [],
    }

    return {
        "proposal": proposal,
        "verifier_result": (
            verifier_result
        ),
        "decision_result": (
            decision_result
        ),
        "pending_actions": [action],
    }


def _service(session):
    """创建测试用 CommitDraftService。"""

    return CommitDraftService(
        TripDraftRepository(session)
    )


def test_commit_writes_trip_version_and_approval_atomically():
    """一次批准应写入三张表，并返回内部模拟草稿摘要。"""

    factory = _session_factory()
    context = _approved_context()

    with factory() as session:
        output = _service(session).commit(
            user_id="user_001",
            trip_session_id=(
                "trip_session_commit_001"
            ),
            **context,
        )

        assert (
            output["commit_result"]["status"]
            == "committed"
        )
        assert (
            output["commit_result"][
                "records_written"
            ]
            == 3
        )
        assert (
            output["trip_draft"]["status"]
            == "approved_simulated"
        )
        assert (
            output["trip_draft"]["draft_only"]
            is True
        )

        assert session.query(
            TripDraft
        ).count() == 1
        assert session.query(
            TripVersion
        ).count() == 1
        assert session.query(
            ApprovalRecord
        ).count() == 1

        stored_version = session.query(
            TripVersion
        ).one()

        # 完整 Proposal 只保存在不可变版本快照中。
        assert (
            stored_version.proposal_json[
                "proposal_id"
            ]
            == context["proposal"][
                "proposal_id"
            ]
        )


def test_same_decision_event_is_idempotent():
    """同一 decision_event_id 重放不能创建重复记录。"""

    factory = _session_factory()
    context = _approved_context()

    with factory() as session:
        service = _service(session)

        first = service.commit(
            user_id="user_001",
            trip_session_id=(
                "trip_session_commit_001"
            ),
            **context,
        )

        second = service.commit(
            user_id="user_001",
            trip_session_id=(
                "trip_session_commit_001"
            ),
            **context,
        )

        assert first["commit_result"][
            "status"
        ] == "committed"
        assert second["commit_result"][
            "status"
        ] == "already_committed"
        assert second["commit_result"][
            "idempotent_replay"
        ] is True
        assert second["commit_result"][
            "records_written"
        ] == 0

        assert session.query(
            TripDraft
        ).count() == 1
        assert session.query(
            TripVersion
        ).count() == 1
        assert session.query(
            ApprovalRecord
        ).count() == 1


def test_same_proposal_with_new_event_is_still_idempotent():
    """即使 decision_event_id 不同，同一 proposal_id 也只能保存一次。"""

    factory = _session_factory()
    context = _approved_context()

    with factory() as session:
        service = _service(session)

        service.commit(
            user_id="user_001",
            trip_session_id=(
                "trip_session_commit_001"
            ),
            **context,
        )

        replay = deepcopy(context)
        replay["decision_result"][
            "decision_event_id"
        ] = "decision_test_commit_002"
        replay["pending_actions"][0][
            "decision_event_id"
        ] = "decision_test_commit_002"

        output = service.commit(
            user_id="user_001",
            trip_session_id=(
                "trip_session_commit_001"
            ),
            **replay,
        )

        assert output["commit_result"][
            "status"
        ] == "already_committed"
        assert session.query(
            TripVersion
        ).count() == 1
        assert session.query(
            ApprovalRecord
        ).count() == 1


def test_unverified_proposal_is_rejected_without_writes():
    """Verifier 未通过时，三张业务表都不能写入。"""

    factory = _session_factory()
    context = _approved_context()
    context["verifier_result"][
        "passed"
    ] = False
    context["verifier_result"][
        "status"
    ] = "failed"
    context["verifier_result"][
        "next_action"
    ] = "final_response"

    with factory() as session:
        with pytest.raises(
            CommitPreconditionError
        ):
            _service(session).commit(
                user_id="user_001",
                **context,
            )

        assert session.query(
            TripDraft
        ).count() == 0
        assert session.query(
            TripVersion
        ).count() == 0
        assert session.query(
            ApprovalRecord
        ).count() == 0


def test_non_approve_decision_is_rejected():
    """request_changes / cancel 不能触发数据库提交。"""

    factory = _session_factory()
    context = _approved_context()
    context["decision_result"].update(
        {
            "decision": "cancel",
            "next_action": "cancel",
        }
    )

    with factory() as session:
        with pytest.raises(
            CommitPreconditionError
        ):
            _service(session).commit(
                user_id="user_001",
                **context,
            )

        assert session.query(
            TripDraft
        ).count() == 0


def test_unknown_user_is_rejected():
    """业务数据库不存在的用户不能创建带外键的 TripDraft。"""

    factory = _session_factory()
    context = _approved_context()

    with factory() as session:
        with pytest.raises(
            UserNotFoundError
        ):
            _service(session).commit(
                user_id="missing_user",
                **context,
            )

        assert session.query(
            TripDraft
        ).count() == 0



def test_newer_version_updates_current_trip_and_keeps_history():
    """同一 trip_session 的较新 Proposal 应新增版本并更新当前指针。"""

    factory = _session_factory()
    context = _approved_context()

    with factory() as session:
        service = _service(session)

        first = service.commit(
            user_id="user_001",
            trip_session_id=(
                "trip_session_commit_001"
            ),
            **context,
        )

        revised = deepcopy(context)
        revised["proposal"]["version"] = 2
        revised["proposal"][
            "proposal_id"
        ] = "proposal_revised_version_002"
        revised["proposal"][
            "summary"
        ] = "成都三日旅行方案第二版。"

        revised["verifier_result"][
            "proposal_id"
        ] = "proposal_revised_version_002"

        revised["decision_result"].update(
            {
                "proposal_id": (
                    "proposal_revised_version_002"
                ),
                "proposal_version": 2,
                "action_id": (
                    "approve_proposal_revised_version_002"
                ),
                "decision_event_id": (
                    "decision_revised_version_002"
                ),
            }
        )

        revised["pending_actions"][0].update(
            {
                "action_id": (
                    "approve_proposal_revised_version_002"
                ),
                "decision_event_id": (
                    "decision_revised_version_002"
                ),
            }
        )

        second = service.commit(
            user_id="user_001",
            trip_session_id=(
                "trip_session_commit_001"
            ),
            **revised,
        )

        assert first["trip_draft"][
            "trip_id"
        ] == second["trip_draft"][
            "trip_id"
        ]
        assert second["trip_draft"][
            "current_version"
        ] == 2
        assert second["trip_draft"][
            "current_proposal_id"
        ] == "proposal_revised_version_002"

        # 当前指针更新，但历史版本与审批记录不会被覆盖。
        assert session.query(
            TripDraft
        ).count() == 1
        assert session.query(
            TripVersion
        ).count() == 2
        assert session.query(
            ApprovalRecord
        ).count() == 2

def test_older_or_same_version_conflict_rolls_back():
    """同一 trip_session 的非递增版本不能覆盖当前版本。"""

    factory = _session_factory()
    context = _approved_context()

    with factory() as session:
        service = _service(session)

        service.commit(
            user_id="user_001",
            trip_session_id=(
                "trip_session_commit_001"
            ),
            **context,
        )

        conflict = deepcopy(context)
        conflict["proposal"][
            "proposal_id"
        ] = "proposal_conflicting_same_version"
        conflict["decision_result"].update(
            {
                "proposal_id": (
                    "proposal_conflicting_same_version"
                ),
                "decision_event_id": (
                    "decision_conflicting_version"
                ),
            }
        )
        conflict["verifier_result"][
            "proposal_id"
        ] = "proposal_conflicting_same_version"
        conflict["pending_actions"][0].update(
            {
                "decision_event_id": (
                    "decision_conflicting_version"
                ),
            }
        )

        with pytest.raises(
            CommitConflictError
        ):
            service.commit(
                user_id="user_001",
                trip_session_id=(
                    "trip_session_commit_001"
                ),
                **conflict,
            )

        # 冲突提交没有产生半成品。
        assert session.query(
            TripDraft
        ).count() == 1
        assert session.query(
            TripVersion
        ).count() == 1
        assert session.query(
            ApprovalRecord
        ).count() == 1
