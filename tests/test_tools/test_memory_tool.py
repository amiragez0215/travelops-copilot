from app.db.init_db import init_db
from app.db.seed_data import DEFAULT_SEED_DATA, seed_database
from app.db.session import create_db_engine, create_session_factory
from app.tools.memory_tool import read_user_memory


def _build_seeded_session_factory():
    """
    创建并填充测试数据库。
    """

    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        seed_database(session, DEFAULT_SEED_DATA, clear=True)

    return session_factory


def test_read_user_memory_loads_seeded_profile():
    """
    user_001 有长期画像，应从 SQL 读取。
    """

    session_factory = _build_seeded_session_factory()

    with session_factory() as session:
        profile = read_user_memory(
            user_id="user_001",
            session=session,
            trip_request={
                "transport_preferences": [],
                "hotel_preferences": [],
                "travel_style": [],
            },
        )

    assert profile["user_id"] == "user_001"
    assert profile["user_found"] is True
    assert profile["profile_found"] is True
    assert profile["source"] == "sql:user_profiles"

    assert profile["flight_preferences"]["avoid_early_flight"] == 0.9
    assert profile["hotel_preferences"]["quiet"] == 0.85
    assert len(profile["recent_trip_history"]) == 2


def test_read_user_memory_merges_current_request_preferences():
    """
    本次输入偏好应该增强长期偏好。
    """

    session_factory = _build_seeded_session_factory()

    with session_factory() as session:
        profile = read_user_memory(
            user_id="user_001",
            session=session,
            trip_request={
                "transport_preferences": ["avoid_late_arrival"],
                "hotel_preferences": ["near_subway"],
                "travel_style": ["slow", "food"],
                "raw_constraints": ["避免太晚抵达"],
                "budget": 4000,
            },
        )

    assert profile["current_request"]["flight_preferences"]["avoid_late_arrival"] == 1.0
    assert profile["flight_preferences"]["avoid_late_arrival"] == 1.0
    assert profile["hotel_preferences"]["near_subway"] == 1.0
    assert profile["travel_style"]["slow"] == 1.0
    assert "activity_preferences" in profile
    assert "pace_preferences" in profile
    assert "risk_preferences" in profile
    assert "hard_constraints" not in profile
    assert profile["budget_preferences"]["current_trip_budget"] == 4000


def test_read_user_memory_user_without_profile_uses_default():
    """
    user_002 存在但没有画像，应使用默认画像。
    """

    session_factory = _build_seeded_session_factory()

    with session_factory() as session:
        profile = read_user_memory(
            user_id="user_002",
            session=session,
            trip_request={},
        )

    assert profile["user_id"] == "user_002"
    assert profile["user_found"] is True
    assert profile["profile_found"] is False
    assert profile["source"] == "default:no_profile"
    assert profile["flight_preferences"]["avoid_early_flight"] == 0.5


def test_read_user_memory_anonymous_uses_default():
    """
    没有 user_id 时，使用匿名默认画像。
    """

    session_factory = _build_seeded_session_factory()

    with session_factory() as session:
        profile = read_user_memory(
            user_id=None,
            session=session,
            trip_request={
                "hotel_preferences": ["quiet"],
            },
        )

    assert profile["user_id"] is None
    assert profile["user_found"] is False
    assert profile["profile_found"] is False
    assert profile["source"] == "default:anonymous"
    assert profile["hotel_preferences"]["quiet"] == 1.0

def test_read_user_memory_merges_preference_signals():
    """
    MemoryTool 应基于 preference_signals 合并本次偏好。
    """

    session_factory = _build_seeded_session_factory()

    with session_factory() as session:
        profile = read_user_memory(
            user_id="user_001",
            session=session,
            trip_request={
                "budget": 4000,
                "preference_signals": [
                    {
                        "domain": "hotel",
                        "key": "quiet",
                        "label": "安静",
                        "strength": 1.0,
                        "polarity": "positive",
                        "evidence": "一定要安静",
                        "source": "rule",
                    },
                    {
                        "domain": "hotel",
                        "key": "cleanliness",
                        "label": "干净",
                        "strength": 0.9,
                        "polarity": "positive",
                        "evidence": "干净的酒店",
                        "source": "rule",
                    },
                    {
                        "domain": "activity",
                        "key": "food",
                        "label": "美食",
                        "strength": 0.9,
                        "polarity": "positive",
                        "evidence": "喜欢美食",
                        "source": "rule",
                    },
                    {
                        "domain": "pace",
                        "key": "slow",
                        "label": "慢节奏",
                        "strength": 0.9,
                        "polarity": "positive",
                        "evidence": "行程不要太赶",
                        "source": "rule",
                    },
                ],
            },
        )

    assert profile["hotel_preferences"]["quiet"] == 1.0
    assert profile["hotel_preferences"]["cleanliness"] == 0.9
    assert profile["activity_preferences"]["food"] == 0.9
    assert profile["pace_preferences"]["slow"] == 0.9
    assert "hard_constraints" not in profile