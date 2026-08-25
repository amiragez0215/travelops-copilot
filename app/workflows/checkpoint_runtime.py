from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

from app.common.config import settings


# 官方建议在创建 Checkpointer 前开启严格 MessagePack 类型限制。
# 这里显式同步 Settings 到真实环境变量，避免只在 .env 中声明却没有生效。
os.environ["LANGGRAPH_STRICT_MSGPACK"] = (
    "true"
    if settings.langgraph_strict_msgpack
    else "false"
)

# 延迟到安全环境变量设置之后再导入 SqliteSaver。
from langgraph.checkpoint.sqlite import SqliteSaver


if TYPE_CHECKING:  # pragma: no cover - 仅用于类型检查器
    from sqlite3 import Connection


PROJECT_ROOT = Path(__file__).resolve().parents[2]

_CHECKPOINTER: SqliteSaver | None = None
_CHECKPOINT_CONNECTION: sqlite3.Connection | None = None
_RUNTIME_LOCK = RLock()



def resolve_checkpoint_path(
    configured_path: str | Path | None = None,
) -> Path:
    """
    将配置中的 Checkpoint 路径解析成项目绝对路径。

    支持：
        ./data/db/langgraph_checkpoints.sqlite
        D:/agent/travelops-copilot/data/db/langgraph_checkpoints.sqlite
    """

    raw_path = Path(
        configured_path
        or settings.langgraph_checkpoint_db
    )

    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path

    return raw_path.resolve()



def create_sqlite_checkpointer(
    db_path: str | Path | None = None,
) -> tuple[
    SqliteSaver,
    sqlite3.Connection,
]:
    """
    创建同步 SQLite LangGraph Checkpointer。

    返回 connection 的原因：
        SqliteSaver 需要在整个 CompiledGraph 生命周期内保持连接打开；
        FastAPI shutdown 时必须显式 close()。

    当前适用范围：
        - 本地开发；
        - 单进程 FastAPI；
        - 面试演示和轻量测试。

    更高并发部署应切换到 PostgresSaver。
    """

    path = resolve_checkpoint_path(
        db_path
    )

    # 1. 自动创建 data/db 等父目录。
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 2. 建立 SQLite 连接。
    #    check_same_thread=False 允许 FastAPI 的工作线程复用同一连接；
    #    SqliteSaver 内部会使用锁保护同步访问。
    connection = sqlite3.connect(
        str(path),
        check_same_thread=False,
        timeout=(
            settings.langgraph_sqlite_timeout_seconds
        ),
    )

    # 3. WAL 让读取和写入更适合轻量服务并发；
    #    busy_timeout 避免短暂文件锁直接抛错。
    connection.execute(
        "PRAGMA journal_mode=WAL"
    )
    connection.execute(
        "PRAGMA busy_timeout=5000"
    )

    # 4. 这是创建持久化 Checkpointer 的核心代码。
    #    LangGraph Runtime 后续会自动调用：
    #        put / put_writes / get_tuple / list
    checkpointer = SqliteSaver(
        connection
    )

    return checkpointer, connection



def initialize_checkpointer(
    *,
    force_reload: bool = False,
) -> SqliteSaver:
    """
    初始化并缓存应用级 SQLite Checkpointer。

    正式主图编译时使用：
        graph = builder.compile(
            checkpointer=initialize_checkpointer()
        )
    """

    global _CHECKPOINTER
    global _CHECKPOINT_CONNECTION

    with _RUNTIME_LOCK:
        if (
            _CHECKPOINTER is not None
            and not force_reload
        ):
            return _CHECKPOINTER

        # 1. 强制重载前先关闭旧连接，避免文件句柄泄漏。
        _close_runtime_unlocked()

        # 2. 创建新的持久化 Checkpointer。
        (
            _CHECKPOINTER,
            _CHECKPOINT_CONNECTION,
        ) = create_sqlite_checkpointer()

        return _CHECKPOINTER



def get_checkpointer() -> SqliteSaver:
    """获取应用级 Checkpointer；未初始化时懒加载。"""

    return initialize_checkpointer(
        force_reload=False
    )



def get_checkpoint_connection() -> sqlite3.Connection | None:
    """
    只读获取当前 SQLite 连接。

    主要用于健康检查和测试，不应由业务节点直接执行 SQL。
    """

    return _CHECKPOINT_CONNECTION



def reset_checkpointer_runtime() -> None:
    """
    关闭 SQLite 连接并清除进程内引用。

    注意：
        只关闭连接，不删除磁盘 checkpoint 数据。
        服务重启后仍可用相同 thread_id 恢复旧 interrupt。
    """

    with _RUNTIME_LOCK:
        _close_runtime_unlocked()



def _close_runtime_unlocked() -> None:
    """在已经持有运行时锁的情况下关闭 Checkpointer 资源。"""

    global _CHECKPOINTER
    global _CHECKPOINT_CONNECTION

    if _CHECKPOINT_CONNECTION is not None:
        try:
            _CHECKPOINT_CONNECTION.close()
        except sqlite3.Error:
            # shutdown 阶段不应因为重复关闭连接阻断整个应用退出。
            pass

    _CHECKPOINT_CONNECTION = None
    _CHECKPOINTER = None
