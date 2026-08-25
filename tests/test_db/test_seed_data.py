from app.db.init_db import init_db
from app.db.models import TripHistory, User, UserProfile
from app.db.seed_data import DEFAULT_SEED_DATA, seed_database
from app.db.session import create_db_engine, create_session_factory


def _build_test_session_factory():
    """
    创建测试用内存数据库。
    """

    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    return create_session_factory(engine)


def test_seed_database_inserts_users_profiles_and_history():
    """
    seed_database 应写入 users、user_profiles、trip_history。
    """

    session_factory = _build_test_session_factory()

    with session_factory() as session:
        seed_database(session, DEFAULT_SEED_DATA, clear=True)

        user = session.get(User, "user_001")
        profile = session.get(UserProfile, "user_001")

        assert user is not None
        assert profile is not None

        assert profile.activity_preferences_json["food"] == 0.75
        assert profile.pace_preferences_json["slow"] == 0.85
        assert profile.risk_preferences_json["weather_sensitive"] == 0.65

        histories = (
            session.query(TripHistory)
            .filter(TripHistory.user_id == "user_001")
            .all()
        )
        assert len(histories) == 2


def test_seed_database_is_idempotent():
    """
    重复运行 seed_database 不应产生重复数据。
    """

    session_factory = _build_test_session_factory()

    with session_factory() as session:
        seed_database(session, DEFAULT_SEED_DATA, clear=True)
        seed_database(session, DEFAULT_SEED_DATA, clear=False)

        assert session.query(User).count() == 2
        assert session.query(UserProfile).count() == 1
        assert session.query(TripHistory).count() == 2

def test_seed_clear_removes_committed_trip_tables():
    """
    seed_database(clear=True) 应先清理 CommitDraft 业务表，
    再清 users / profiles / history，避免外键冲突。
    """

    from app.db.models import (
        ApprovalRecord,
        TripDraft,
        TripVersion,
    )
    from app.repositories.trip_draft_repository import (
        TripDraftRepository,
    )
    from app.services.commit_draft_service import (
        CommitDraftService,
    )
    from tests.test_services.test_commit_draft_service import (
        _approved_context,
    )

    session_factory = _build_test_session_factory()

    with session_factory() as session:
        seed_database(
            session,
            DEFAULT_SEED_DATA,
            clear=True,
        )

        CommitDraftService(
            TripDraftRepository(session)
        ).commit(
            user_id="user_001",
            trip_session_id=(
                "trip_session_seed_clear"
            ),
            **_approved_context(),
        )

        assert session.query(
            TripDraft
        ).count() == 1

        # 再次 clear 不应因外键失败，并且会清除旧业务记录。
        seed_database(
            session,
            DEFAULT_SEED_DATA,
            clear=True,
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
