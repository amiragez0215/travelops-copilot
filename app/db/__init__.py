"""
db 包。

业务数据库当前保存：
- users / user_profiles / trip_history：用户与长期记忆；
- trip_drafts：批准后的当前模拟旅行草稿；
- trip_versions：不可变 Proposal 版本；
- approval_records：人工批准审计与幂等事件。
"""

from app.db.models import (
    ApprovalRecord,
    Base,
    TripDraft,
    TripHistory,
    TripVersion,
    User,
    UserProfile,
)
from app.db.session import (
    DEFAULT_DATABASE_URL,
    SessionLocal,
    create_db_engine,
    create_session_factory,
)

__all__ = [
    "Base",
    "User",
    "UserProfile",
    "TripHistory",
    "TripDraft",
    "TripVersion",
    "ApprovalRecord",
    "DEFAULT_DATABASE_URL",
    "SessionLocal",
    "create_db_engine",
    "create_session_factory",
]
