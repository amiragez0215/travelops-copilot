from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.common.config import settings


# 项目根目录：D:/agent/travelops-copilot
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Settings 会读取项目根目录 .env。
# 这里不再直接调用 os.getenv，否则 python-dotenv 没有显式加载时，
# DATABASE_URL 可能只使用硬编码默认值而忽略 .env。
DEFAULT_DATABASE_URL = settings.database_url


def create_db_engine(
    database_url: str | None = None,
    echo: bool = False,
) -> Engine:
    """
    创建 SQLAlchemy Engine。

    Args:
        database_url:
            可选数据库 URL。

            正式运行：
                不传，使用 Settings.database_url。

            单元测试：
                可以传 sqlite:///:memory: 或临时 SQLite 文件。

        echo:
            是否打印 SQL。开发排查数据库问题时可以开启，默认关闭。

    Returns:
        Engine:
            SQLAlchemy 的数据库连接池和 SQL 执行入口。

    当前项目默认数据库：
        data/db/travelops.db
    """

    # 1. 使用显式参数，或者回退到统一 Settings。
    url = database_url or DEFAULT_DATABASE_URL

    # 2. 如果是相对 SQLite 文件，先创建父目录。
    _ensure_sqlite_parent_dir(url)

    connect_args: dict[str, object] = {}
    engine_kwargs: dict[str, object] = {
        "echo": echo,
        "future": True,
    }

    if url.startswith("sqlite"):
        # FastAPI 的同步路由可能在线程池中执行，
        # 所以允许同一 SQLite 连接跨线程使用。
        connect_args["check_same_thread"] = False

    if url == "sqlite:///:memory:":
        # 内存 SQLite 默认每个连接拥有独立数据库。
        # StaticPool 保证测试中的所有 Session 共用同一个内存数据库。
        engine_kwargs["poolclass"] = StaticPool

    # 3. 这是创建数据库 Engine 的核心代码。
    engine = create_engine(
        url,
        connect_args=connect_args,
        **engine_kwargs,
    )

    # 4. SQLite 默认可能不强制执行外键。
    #    CommitDraft 会写 trip_drafts / trip_versions / approval_records，
    #    必须开启外键，避免产生孤立版本和审批记录。
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(
            dbapi_connection,
            connection_record,
        ) -> None:
            del connection_record

            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_session_factory(
    engine: Engine,
) -> sessionmaker:
    """
    根据 Engine 创建 SQLAlchemy Session 工厂。

    SessionLocal() 每调用一次，就创建一个短生命周期事务会话。
    Node / Service 完成后应及时关闭 Session。
    """

    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


def _ensure_sqlite_parent_dir(
    database_url: str,
) -> None:
    """
    为文件型 SQLite 数据库创建父目录。

    注意：
        sqlite:///./data/db/travelops.db 是相对于当前工作目录的路径。
        项目正常从根目录启动，因此会落在 data/db 下。
    """

    if database_url == "sqlite:///:memory:":
        return

    if not database_url.startswith("sqlite:///"):
        return

    db_path_text = database_url.replace(
        "sqlite:///",
        "",
        1,
    )

    if not db_path_text or db_path_text == ":memory:":
        return

    db_path = Path(db_path_text)

    # 相对路径以当前项目运行目录为基准。
    # 如果代码从其他目录启动，README / Docker 会显式设置工作目录。
    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# 正式业务数据库 Engine 和 Session Factory。
# 它们在应用进程中复用；每个数据库操作只创建独立 Session。
engine = create_db_engine()
SessionLocal = create_session_factory(engine)


def get_db_session() -> Generator[Session, None, None]:
    """
    FastAPI 依赖注入使用的数据库 Session。

    当前 Workflow Node 主要通过 Service / Repository 自己创建 Session，
    这个函数保留给后续业务查询 API 使用。
    """

    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()
