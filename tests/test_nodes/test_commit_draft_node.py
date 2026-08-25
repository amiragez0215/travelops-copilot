from __future__ import annotations

from copy import deepcopy

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
from app.nodes.commit_draft_node import (
    commit_draft_node,
)
from tests.test_services.test_commit_draft_service import (
    _approved_context,
)


def _session_factory():
    """创建 Node 测试用内存数据库。"""

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


def _state():
    """构造 CommitDraftNode 所需的批准状态。"""

    context = _approved_context()

    return {
        "user_id": "user_001",
        "trip_session_id": (
            "trip_session_node_001"
        ),
        **context,
        "itinerary_status": "proposed",
    }


def test_commit_draft_node_success():
    """成功提交后应写回草稿、版本、审批和 Trace。"""

    factory = _session_factory()

    result = commit_draft_node(
        _state(),
        session_factory=factory,
    )

    assert result["commit_result"][
        "status"
    ] == "committed"
    assert result["itinerary_status"] == (
        "approved_simulated"
    )
    assert result["trip_draft"][
        "draft_only"
    ] is True
    assert result["pending_actions"][0][
        "status"
    ] == "committed"
    assert result["trace"][0][
        "node_name"
    ] == "commit_draft"
    assert result["trace"][0][
        "status"
    ] == "success"

    with factory() as session:
        assert session.query(
            TripDraft
        ).count() == 1
        assert session.query(
            TripVersion
        ).count() == 1
        assert session.query(
            ApprovalRecord
        ).count() == 1


def test_commit_draft_node_is_idempotent():
    """同一个 State 被 Checkpoint 重放时不能重复写数据库。"""

    factory = _session_factory()
    state = _state()

    first = commit_draft_node(
        state,
        session_factory=factory,
    )
    second = commit_draft_node(
        state,
        session_factory=factory,
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

    with factory() as session:
        assert session.query(
            TripDraft
        ).count() == 1
        assert session.query(
            TripVersion
        ).count() == 1
        assert session.query(
            ApprovalRecord
        ).count() == 1


def test_commit_draft_node_rejects_missing_decision_result():
    """缺少 DecisionGate 输出时应返回统一失败结构。"""

    factory = _session_factory()
    state = _state()
    state.pop("decision_result")

    result = commit_draft_node(
        state,
        session_factory=factory,
    )

    assert result["commit_result"][
        "status"
    ] == "failed"
    assert result["trip_draft"] == {}
    assert result["errors"]
    assert result["errors"][0][
        "node"
    ] == "commit_draft"

    with factory() as session:
        assert session.query(
            TripDraft
        ).count() == 0


def test_commit_draft_node_rejects_stale_proposal_id():
    """人工批准与当前 Proposal ID 不一致时不能写数据库。"""

    factory = _session_factory()
    state = _state()
    state["decision_result"] = deepcopy(
        state["decision_result"]
    )
    state["decision_result"][
        "proposal_id"
    ] = "stale_proposal"

    result = commit_draft_node(
        state,
        session_factory=factory,
    )

    assert result["commit_result"][
        "status"
    ] == "failed"
    assert "不一致" in result[
        "errors"
    ][0]["message"]


def test_commit_draft_node_does_not_create_final_response():
    """
    CommitDraft 只负责数据库副作用；
    用户可见响应仍由后续 FinalResponseNode 统一生成。
    """

    factory = _session_factory()

    result = commit_draft_node(
        _state(),
        session_factory=factory,
    )

    assert "final_response" not in result
