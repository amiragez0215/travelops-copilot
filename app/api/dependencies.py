from __future__ import annotations

from fastapi import HTTPException, Request, status

from app.common.logger import get_logger
from app.workflows.workflow_runtime import (
    PlanTripWorkflowRuntime,
    initialize_workflow_runtime,
)


logger = get_logger("travelops.api.runtime")



def get_plan_trip_runtime(
    request: Request,
) -> PlanTripWorkflowRuntime:
    """
    从 FastAPI app.state 获取应用级 PlanTripWorkflowRuntime。

    正常情况：
        FastAPI lifespan 在启动阶段已经初始化：
            app.state.plan_trip_runtime

    测试或关闭预加载时：
        第一次 API 请求执行懒初始化。

    为什么不在每个 Route 中直接 build_plan_trip_graph()？
        - CompiledGraph 不应该每个请求重复构建；
        - SQLite Checkpointer 连接需要贯穿应用生命周期；
        - 同一 Runtime 才能稳定管理 Thread / Checkpoint；
        - RAG、Reranker 和 Graph 都应作为应用级资源复用。
    """

    runtime = getattr(
        request.app.state,
        "plan_trip_runtime",
        None,
    )

    if isinstance(
        runtime,
        PlanTripWorkflowRuntime,
    ):
        return runtime

    # 测试可能注入一个实现同样方法的 Fake Runtime，
    # 因此这里也接受结构化替身，而不是强制 isinstance。
    if runtime is not None:
        return runtime

    try:
        runtime = initialize_workflow_runtime()
    except Exception as exc:
        logger.exception(
            "PlanTrip Workflow Runtime 初始化失败"
        )
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "旅行规划运行时暂时不可用，"
                "请检查 Checkpointer 和 Graph 配置。"
            ),
        ) from exc

    request.app.state.plan_trip_runtime = (
        runtime
    )

    return runtime
