from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import (
    approval_router,
    travel_router,
)
from app.common.config import settings
from app.db.init_db import init_db
from app.workflows.workflow_runtime import (
    initialize_workflow_runtime,
    reset_workflow_runtime,
)


# 前端是和 FastAPI 同进程发布的轻量单页应用，不引入 Node、React 或额外的
# Web 服务。这样面试演示时只需启动一条 uvicorn 命令，页面和后端 API 就能
# 使用同一个 origin 通信，也不需要额外处理 CORS。
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """
    管理 FastAPI 应用级长期资源。

    启动阶段：
        1. 创建不存在的业务数据库表；
        2. 初始化 SQLite Checkpointer；
        3. 编译并缓存一次完整 PlanTrip StateGraph；
        4. 可选预加载 HybridRetriever 和 Cross-Encoder。

    关闭阶段：
        1. 清理 CompiledGraph 进程内引用；
        2. 关闭 Checkpointer SQLite 连接；
        3. 清理 RAG / Reranker 进程内单例；
        4. 不删除任何磁盘数据库、Checkpoint、Chroma 或模型文件。

    为什么这些资源放在 lifespan 中？
        Graph、BM25、Embedding、Reranker 和 SQLite Checkpointer 都不应该
        在每个 HTTP 请求中重复创建。它们属于应用级运行时资源。
    """

    # 1. create_all() 只创建缺失表，不会删除已有业务数据。
    #    测试可以通过 DATABASE_INITIALIZE_ON_STARTUP=false 关闭。
    if settings.database_initialize_on_startup:
        init_db()

    # MCP Runtime 是应用级控制面；stdio Session 仅在启动期 Discovery 和
    # 单次 Tool Call 内短暂存在，不在 lifespan 中维持长连接。
    if not hasattr(app.state, "mcp_tool_runtime"):
        from app.mcp.tool_runtime import initialize_mcp_tool_runtime

        app.state.mcp_tool_runtime = initialize_mcp_tool_runtime()

    # 2. Workflow Runtime 是阶段一的核心资源。
    #    如果 create_app() 已经注入 Fake Runtime，不能在这里覆盖。
    if (
        settings.workflow_preload_on_startup
        and not hasattr(
            app.state,
            "plan_trip_runtime",
        )
    ):
        app.state.plan_trip_runtime = (
            initialize_workflow_runtime()
        )

    # 3. RAG 和 Reranker 使用延迟导入。
    #    关闭预加载后，仅启动 FastAPI 或运行 API 单元测试时，
    #    不会强制加载 Chroma、Embedding 和 Transformers 大模型。
    if (
        settings.rag_preload_on_startup
        and not hasattr(
            app.state,
            "hybrid_retriever",
        )
    ):
        from app.rag.runtime import (
            initialize_hybrid_retriever,
        )

        app.state.hybrid_retriever = (
            initialize_hybrid_retriever()
        )

    if (
        # ``preload`` 只描述何时加载，``enabled`` 才决定工作流是否使用
        # Cross-Encoder。两个条件都满足才允许在启动期导入 Transformers，
        # 否则默认前端请求始终保持在稳定、低延迟的 Hybrid RRF 路径。
        settings.rag_rerank_enabled
        and settings.rag_rerank_preload_on_startup
        and not hasattr(
            app.state,
            "reranker",
        )
    ):
        from app.rag.rerank_runtime import (
            initialize_reranker,
        )

        app.state.reranker = (
            initialize_reranker()
        )

    try:
        yield

    finally:
        # 4. 关闭 Graph Runtime 和 SQLite Checkpointer 连接。
        #    这只清理进程内对象，磁盘 Checkpoint 会继续保留，
        #    服务重启后仍可用相同 thread_id 恢复。
        reset_workflow_runtime()

        if hasattr(app.state, "mcp_tool_runtime"):
            from app.mcp.tool_runtime import reset_mcp_tool_runtime

            reset_mcp_tool_runtime()

        # 5. 只有当前 App 实际持有对应资源时才清理。
        #    测试中注入的 Fake Runtime 不会被当成真实全局 Runtime 处理。
        if hasattr(app.state, "reranker"):
            from app.rag.rerank_runtime import (
                reset_reranker_runtime,
            )

            reset_reranker_runtime()

        if hasattr(
            app.state,
            "hybrid_retriever",
        ):
            from app.rag.runtime import (
                reset_hybrid_retriever_runtime,
            )

            reset_hybrid_retriever_runtime()


def create_app(
    *,
    workflow_runtime: Any | None = None,
) -> FastAPI:
    """
    创建 FastAPI 应用实例。

    Args:
        workflow_runtime:
            可选的 PlanTripWorkflowRuntime 或测试替身。

            正式运行：
                不传，由 lifespan 创建真实 Runtime。

            API 单元测试：
                传入 Fake Runtime，避免加载真实 DeepSeek、MCP、Chroma。

    使用 Application Factory 的价值：
        - 测试可以创建隔离 App；
        - Docker / Uvicorn 仍可直接使用 app.main:app；
        - Runtime 依赖可以明确注入；
        - 不需要通过 monkeypatch 修改全局模块。
    """

    application = FastAPI(
        title="TravelOps-Copilot",
        description=(
            "LangGraph based travel planning Agent "
            "with MCP, Agentic RAG, Verifier and HITL."
        ),
        version="0.4.0",
        lifespan=lifespan,
    )

    # 1. 测试或特殊运行模式可以提前注入 Runtime。
    #    lifespan 会检查该字段是否已经存在，不会覆盖它。
    if workflow_runtime is not None:
        application.state.plan_trip_runtime = (
            workflow_runtime
        )

    # 静态资源只包含本项目自带的 HTML / CSS / JavaScript；页面不依赖 CDN，
    # 避免本地演示、内网环境或离线运行时出现样式/脚本加载失败。
    application.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIR),
        name="frontend-assets",
    )

    @application.get(
        "/",
        include_in_schema=False,
    )
    def frontend_index() -> FileResponse:
        """返回单页前端入口；业务数据仍只通过受控的 /api/v1/travel 获取。"""

        return FileResponse(
            FRONTEND_DIR / "index.html",
            media_type="text/html",
        )

    @application.get("/health")
    def health_check(
        request: Request,
    ) -> dict[str, object]:
        """
        返回不泄漏 API Key、Prompt 和本地模型路径的基础健康状态。
        """

        return {
            "status": "ok",
            "app_env": settings.app_env,
            "weather_mode": settings.weather_mode,
            # 当前正式主链使用的 TravelDataClient 模式。
            # mcp 表示 Weather/Flight/Hotel 节点会真实经过 MCP stdio。
            "travel_data_client_mode": (
                settings.travel_data_client_mode
            ),
            "input_extract_mode": (
                settings.input_extract_mode
            ),
            "rag_collection": (
                settings.rag_chroma_collection
            ),
            "workflow_runtime_ready": hasattr(
                request.app.state,
                "plan_trip_runtime",
            ),
            "mcp_tool_registry_ready": hasattr(request.app.state, "mcp_tool_runtime"),
            "mcp_tool_count": (
                len(request.app.state.mcp_tool_runtime.tool_registry)
                if hasattr(request.app.state, "mcp_tool_runtime") else 0
            ),
            "checkpoint_backend": "sqlite",
        }

    # 2. 启动、状态和 Checkpoint 历史接口。
    application.include_router(
        travel_router,
        prefix="/api/v1/travel",
        tags=["travel"],
    )

    # 3. Human-in-the-loop Decision Resume 接口。
    application.include_router(
        approval_router,
        prefix="/api/v1/travel",
        tags=["travel-approval"],
    )

    return application


# Uvicorn 默认入口：
#     uvicorn app.main:app --reload
app = create_app()
