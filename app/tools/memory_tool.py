from __future__ import annotations

from typing import Any, Mapping

from sqlalchemy.orm import Session

from app.repositories.memory_repository import MemoryRepository


def read_user_memory(
    user_id: str | None,
    session: Session,
    trip_request: Mapping[str, Any] | None = None,
    history_limit: int = 3,
) -> dict[str, Any]:
    """
    读取用户记忆。

    Tool 层提供稳定业务能力。
    Node 调 Tool，Tool 调 Repository。
    """

    repository = MemoryRepository(session)

    return repository.read_user_memory(
        user_id=user_id,
        trip_request=trip_request,
        history_limit=history_limit,
    )