from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime
from sqlalchemy.orm import Session

from app.db.init_db import init_db
from app.db.models import (
    ApprovalRecord,
    TripDraft,
    TripHistory,
    TripVersion,
    User,
    UserProfile,
)
from app.db.session import SessionLocal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_FILE = PROJECT_ROOT / "data" / "seed" / "sql_seed.json"


DEFAULT_SEED_DATA: dict[str, Any] = {
    "users": [
        {
            "user_id": "user_001",
            "name": "林晓",
            "home_city": "杭州",
        },
        {
            "user_id": "user_002",
            "name": "测试用户无画像",
            "home_city": "上海",
        },
    ],
    "user_profiles": [
        {
            "user_id": "user_001",
            "flight_preferences_json": {
                "avoid_early_flight": 0.9,
                "prefer_direct": 0.75,
                "price_sensitive": 0.6,
                "avoid_late_arrival": 0.65,
            },
            "hotel_preferences_json": {
                "quiet": 0.85,
                "near_subway": 0.7,
                "high_rating": 0.65,
                "budget_friendly": 0.55,
                "cleanliness": 0.7,
            },
            "activity_preferences_json": {
                "food": 0.75,
                "nature_scenery": 0.55,
                "culture_history": 0.65,
                "city_walk": 0.6,
                "entertainment": 0.35,
                "fitness": 0.2,
                "adventure": 0.2,
                "photography": 0.35,
            },
            "pace_preferences_json": {
                "slow": 0.85,
                "low_walking_intensity": 0.65,
                "packed_schedule": 0.2,
            },
            "diet_preferences_json": {
                "spicy_tolerance": "medium",
                "likes": ["川菜", "小吃", "茶馆"],
                "avoid_foods": ["生冷海鲜"],
            },
            "budget_preferences_json": {
                "budget_sensitivity": 0.6,
                "preferred_hotel_price_max": 650,
                "preferred_total_budget_range": [3000, 4500],
            },
            "risk_preferences_json": {
                "weather_sensitive": 0.65,
                "avoid_crowds": 0.45,
                "risk_tolerance": 0.35,
            },
            "profile_notes_json": [
                "用户偏好安静住宿。",
                "用户不喜欢太早出发。",
                "用户喜欢慢节奏和本地生活体验。",
                "用户对美食和文化体验有一定兴趣。",
            ],
        }
    ],
    "trip_history": [
        {
            "history_id": "history_001",
            "user_id": "user_001",
            "trip_id": "trip_seed_001",
            "destination": "成都",
            "feedback_summary": "用户反馈喜欢安静酒店和慢节奏安排，不喜欢太早出发。",
            "preference_update_json": {
                "hotel_preferences.quiet": 0.9,
                "flight_preferences.avoid_early_flight": 0.95,
            },
            "tags_json": ["quiet_hotel", "slow_trip", "avoid_early_flight"],
            "created_at": "2026-06-20T09:30:00+08:00",
        },
        {
            "history_id": "history_002",
            "user_id": "user_001",
            "trip_id": "trip_seed_002",
            "destination": "苏州",
            "feedback_summary": "用户觉得行程过紧，之后更偏好每天一到两个主要活动。",
            "preference_update_json": {
                "travel_style.slow": 0.88,
            },
            "tags_json": ["too_tight", "prefer_slow"],
            "created_at": "2026-05-15T18:20:00+08:00",
        },
    ],
}


SEED_TABLES = [
    ("users", User, "user_id"),
    ("user_profiles", UserProfile, "user_id"),
    ("trip_history", TripHistory, "history_id"),
]


def load_seed_data(seed_file: Path | None = None) -> dict[str, Any]:
    """
    加载 seed 数据。

    如果 data/seed/sql_seed.json 存在，则优先使用文件。
    否则使用 DEFAULT_SEED_DATA。
    """

    target_file = seed_file or DEFAULT_SEED_FILE

    if target_file.exists():
        return json.loads(target_file.read_text(encoding="utf-8"))

    return DEFAULT_SEED_DATA


def seed_database(
    session: Session,
    seed_data: dict[str, Any] | None = None,
    clear: bool = False,
) -> None:
    """
    写入 seed 数据。

    clear=True 时先清空已有数据。
    """

    data = seed_data or load_seed_data()

    if clear:
        _clear_seed_tables(session)

    for key, model, pk_field in SEED_TABLES:
        for item in data.get(key, []):
            _upsert_model(session, model, pk_field, dict(item))

        # 每组父表/子表写完后立即 flush，确保 SQLite 开启外键后
        # users -> user_profiles -> trip_history 按确定顺序落库。
        session.flush()

    session.commit()


def _clear_seed_tables(session: Session) -> None:
    """
    按外键依赖顺序清空业务表和 seed 表。

    为什么要先清 TripDraft 相关表？

        CommitDraftNode 新增了：
            approval_records -> trip_versions -> trip_drafts -> users

        seed_database(clear=True) 如果直接删除 users，
        会因为已有 TripDraft 外键而失败。
    """

    # 1. 先删除 CommitDraft 产生的业务记录。
    for model in [
        ApprovalRecord,
        TripVersion,
        TripDraft,
    ]:
        session.query(model).delete()

    # 2. 再按原 Seed 表的反向依赖顺序删除。
    for _, model, _ in reversed(SEED_TABLES):
        session.query(model).delete()

    session.commit()


def _upsert_model(
    session: Session,
    model: type,
    pk_field: str,
    values: dict[str, Any],
) -> None:
    """
    根据主键 upsert 一条记录。
    """

    values = _normalize_seed_aliases(model, values)
    values = _coerce_model_values(model, values)

    if pk_field not in values:
        raise ValueError(f"seed item 缺少主键字段：{pk_field}")

    pk_value = values[pk_field]
    instance = session.get(model, pk_value)

    if instance is None:
        session.add(model(**values))
        return

    for key, value in values.items():
        setattr(instance, key, value)


def _normalize_seed_aliases(
    model: type,
    values: dict[str, Any],
) -> dict[str, Any]:
    """
    兼容旧 seed 字段名。

    例如：
    hotel_preferences -> hotel_preferences_json
    """

    normalized = dict(values)

    if model is UserProfile:
        alias_map = {
            "flight_preferences": "flight_preferences_json",
            "hotel_preferences": "hotel_preferences_json",
            "activity_preferences": "activity_preferences_json",
            "pace_preferences": "pace_preferences_json",
            "diet_preferences": "diet_preferences_json",
            "budget_preferences": "budget_preferences_json",
            "risk_preferences": "risk_preferences_json",
            "profile_notes": "profile_notes_json",
        }

        for old_key, new_key in alias_map.items():
            if old_key in normalized and new_key not in normalized:
                normalized[new_key] = normalized.pop(old_key)

    if model is TripHistory:
        alias_map = {
            "preference_update": "preference_update_json",
            "tags": "tags_json",
        }

        for old_key, new_key in alias_map.items():
            if old_key in normalized and new_key not in normalized:
                normalized[new_key] = normalized.pop(old_key)

    return normalized


def _coerce_model_values(
    model: type,
    values: dict[str, Any],
) -> dict[str, Any]:
    """
    过滤不存在的字段，并把 datetime 字符串转成 datetime。
    """

    coerced: dict[str, Any] = {}

    for key, value in values.items():
        column = model.__table__.columns.get(key)

        # 旧 seed 文件可能有多余字段，这里直接忽略。
        if column is None:
            continue

        if isinstance(column.type, DateTime) and isinstance(value, str):
            coerced[key] = _parse_datetime(value)
        else:
            coerced[key] = value

    return coerced


def _parse_datetime(value: str) -> datetime:
    """
    解析 ISO datetime 字符串。
    """

    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    """
    命令行入口。

    示例：
        python -m app.db.seed_data
        python -m app.db.seed_data --clear
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clear",
        action="store_true",
        help="写入前先清空 seed 表",
    )
    parser.add_argument(
        "--seed-file",
        type=str,
        default=None,
        help="自定义 seed JSON 文件路径",
    )
    args = parser.parse_args()

    init_db()

    seed_file = Path(args.seed_file) if args.seed_file else None

    with SessionLocal() as session:
        seed_database(
            session=session,
            seed_data=load_seed_data(seed_file),
            clear=args.clear,
        )

    print("Database seeded.")


if __name__ == "__main__":
    main()