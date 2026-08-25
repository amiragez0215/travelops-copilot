from __future__ import annotations

import argparse

from sqlalchemy.engine import Engine

from app.db.models import Base
from app.db.session import DEFAULT_DATABASE_URL, engine as default_engine


def init_db(engine: Engine | None = None) -> None:
    """
    创建所有数据库表。
    """

    target_engine = engine or default_engine
    Base.metadata.create_all(bind=target_engine)


def drop_db(engine: Engine | None = None) -> None:
    """
    删除所有数据库表。
    """

    target_engine = engine or default_engine
    Base.metadata.drop_all(bind=target_engine)


def reset_db(engine: Engine | None = None) -> None:
    """
    重建数据库。
    """

    target_engine = engine or default_engine
    drop_db(target_engine)
    init_db(target_engine)


def main() -> None:
    """
    命令行入口。

    示例：
        python -m app.db.init_db
        python -m app.db.init_db --reset
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="删除并重建所有表",
    )
    args = parser.parse_args()

    if args.reset:
        reset_db()
        print(f"Database reset: {DEFAULT_DATABASE_URL}")
        return

    init_db()
    print(f"Database initialized: {DEFAULT_DATABASE_URL}")


if __name__ == "__main__":
    main()