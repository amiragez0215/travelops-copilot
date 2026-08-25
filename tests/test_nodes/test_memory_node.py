from app.db.init_db import init_db
from app.db.seed_data import DEFAULT_SEED_DATA, seed_database
from app.db.session import create_db_engine, create_session_factory
from app.nodes.memory_node import memory_read_node


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


def test_memory_read_node_success():
    """
    MemoryReadNode 应读取 SQL profile，并写入 user_profile。
    """

    session_factory = _build_seeded_session_factory()

    state = {
        "user_id": "user_001",
        "trip_request": {
            "origin": "杭州",
            "destination": "成都",
            "start_date": "2026-07-02",
            "days": 3,
            "budget": 4000,
            "transport_preferences": ["avoid_late_arrival"],
            "hotel_preferences": ["quiet"],
            "travel_style": ["slow"],
            "raw_constraints": ["避免太晚抵达"],
        },
    }

    result = memory_read_node(
        state,
        session_factory=session_factory,
    )

    profile = result["user_profile"]

    assert profile["user_id"] == "user_001"
    assert profile["profile_found"] is True
    assert profile["source"] == "sql:user_profiles"

    assert profile["flight_preferences"]["avoid_early_flight"] == 0.9
    assert profile["flight_preferences"]["avoid_late_arrival"] == 1.0
    assert profile["hotel_preferences"]["quiet"] == 1.0
    assert profile["travel_style"]["slow"] == 1.0

    assert result["trace"][0]["node_name"] == "memory_read"
    assert result["trace"][0]["tool_name"] == "memory_tool"
    assert result["trace"][0]["status"] == "success"


def test_memory_read_node_user_without_profile():
    """
    用户存在但没有画像时，不应该失败。
    """

    session_factory = _build_seeded_session_factory()

    state = {
        "user_id": "user_002",
        "trip_request": {
            "origin": "上海",
            "destination": "成都",
            "start_date": "2026-07-02",
            "days": 3,
        },
    }

    result = memory_read_node(
        state,
        session_factory=session_factory,
    )

    profile = result["user_profile"]

    assert profile["user_id"] == "user_002"
    assert profile["user_found"] is True
    assert profile["profile_found"] is False
    assert profile["source"] == "default:no_profile"
    assert result["trace"][0]["status"] == "success"


def test_memory_read_node_without_user_id_uses_anonymous_profile():
    """
    没有 user_id 时，也可以继续规划，只是没有长期记忆。
    """

    session_factory = _build_seeded_session_factory()

    state = {
        "trip_request": {
            "origin": "杭州",
            "destination": "成都",
            "start_date": "2026-07-02",
            "days": 3,
            "hotel_preferences": ["quiet"],
        },
    }

    result = memory_read_node(
        state,
        session_factory=session_factory,
    )

    profile = result["user_profile"]

    assert profile["user_id"] is None
    assert profile["source"] == "default:anonymous"
    assert profile["hotel_preferences"]["quiet"] == 1.0
    assert result["trace"][0]["status"] == "success"


def test_memory_read_node_invalid_trip_request_returns_fallback():
    """
    trip_request 类型错误时，节点返回 fallback profile 和 errors。
    """

    session_factory = _build_seeded_session_factory()

    result = memory_read_node(
        {
            "user_id": "user_001",
            "trip_request": "invalid",
        },
        session_factory=session_factory,
    )

    assert result["user_profile"]["source"] == "fallback:memory_read_error"
    assert result["errors"]
    assert result["errors"][0]["node"] == "memory_read"
    assert result["trace"][0]["status"] == "failed"

def test_memory_read_node_merges_preference_signals():
    """
    MemoryReadNode 应把 trip_request.preference_signals 合并进 user_profile。
    """

    session_factory = _build_seeded_session_factory()

    state = {
        "user_id": "user_001",
        "trip_request": {
            "origin": "杭州",
            "destination": "成都",
            "start_date": "2026-07-02",
            "days": 3,
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
                    "domain": "activity",
                    "key": "nature_scenery",
                    "label": "自然风景",
                    "strength": 0.85,
                    "polarity": "positive",
                    "evidence": "喜欢自然风景",
                    "source": "rule",
                },
            ],
        },
    }

    result = memory_read_node(
        state,
        session_factory=session_factory,
    )

    profile = result["user_profile"]

    assert profile["hotel_preferences"]["quiet"] == 1.0
    assert profile["activity_preferences"]["nature_scenery"] == 0.85
    assert "hard_constraints" not in profile
    assert result["trace"][0]["status"] == "success"