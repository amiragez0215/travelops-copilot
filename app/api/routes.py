from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)

from app.api.dependencies import (
    get_plan_trip_runtime,
)
from app.api.graph_response import (
    build_public_history_response,
    build_public_run_response,
)
from app.api.travel_schemas import (
    TravelInvokeRequest,
    TravelRunHistoryResponse,
    TravelRunResponse,
)
from app.common.config import settings
from app.common.logger import get_logger
from app.workflows.workflow_runtime import (
    PlanTripWorkflowRuntime,
    WorkflowRuntimeError,
    WorkflowThreadAlreadyExistsError,
    WorkflowThreadNotFoundError,
)


router = APIRouter()
logger = get_logger("travelops.api.travel")


@router.post(
    "/invoke",
    response_model=TravelRunResponse,
    summary="启动新的旅行规划 Graph",
)
def invoke_travel_agent(
    payload: TravelInvokeRequest,
    runtime: PlanTripWorkflowRuntime = Depends(
        get_plan_trip_runtime
    ),
) -> TravelRunResponse:
    """
    启动一次新的完整 PlanTripWorkflow。

    调用后 Graph 会一直运行到两种终点之一：

        1. DecisionGateNode interrupt()
            返回 run_status=waiting_for_decision，
            同时返回 Proposal 和 interrupt payload。

        2. FinalResponseNode → END
            返回 run_status=completed，
            业务结果位于 final_response。

    这个接口不会在审批前写 TripDraft；真正数据库提交只发生在：
        DecisionGate approve → CommitDraftNode
    """

    try:
        # 1. 这是启动完整 Agent 的核心调用。
        #    Runtime 内部会：
        #        - 构造 initial_state；
        #        - 生成/校验 thread_id；
        #        - graph.invoke()；
        #        - 从 Checkpointer 读取最新 Snapshot。
        execution = runtime.start_plan(
            user_id=payload.user_id,
            raw_message=payload.message,
            reference_date=(
                payload.reference_date.isoformat()
                if payload.reference_date
                is not None
                else None
            ),
            trip_session_id=(
                payload.trip_session_id
            ),
        )

    except WorkflowThreadAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        # build_initial_plan_trip_state() 的业务输入校验错误。
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except WorkflowRuntimeError as exc:
        logger.exception(
            "启动 PlanTripWorkflow 失败"
        )
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="旅行规划工作流启动失败。",
        ) from exc

    except Exception as exc:
        logger.exception(
            "未处理的 PlanTripWorkflow 启动异常"
        )
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="旅行规划服务发生内部错误。",
        ) from exc

    # 2. 只把公开字段返回前端，不返回完整 TravelState。
    return build_public_run_response(
        thread_id=execution.thread_id,
        graph_output=execution.output,
        snapshot=execution.snapshot,
        # Stage 5：安全返回本次运行关联 ID；完整 Run/Span 仍仅供 Runtime
        # 和离线 Eval 使用，不能通过普通 API 把内部 Trace 暴露给前端。
        run_id=execution.run_id,
        include_trace=False,
        max_trace_items=(
            settings.api_trace_max_items
        ),
    )


@router.get(
    "/runs/{thread_id}",
    response_model=TravelRunResponse,
    summary="查询旅行规划 Thread 的当前状态",
)
def get_travel_run(
    thread_id: str,
    include_trace: bool = Query(
        default=False,
        description=(
            "是否返回最近的节点 Trace 摘要。"
        ),
    ),
    runtime: PlanTripWorkflowRuntime = Depends(
        get_plan_trip_runtime
    ),
) -> TravelRunResponse:
    """
    根据 thread_id 查询最新 Checkpoint。

    该接口可以用于：
        - 页面刷新后恢复当前 Proposal；
        - 查看是否仍在等待人工决定；
        - 查询最终 final_response；
        - 调试当前 next nodes。
    """

    try:
        snapshot = runtime.get_snapshot(
            thread_id
        )

    except WorkflowThreadNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "读取 PlanTrip Thread 状态失败"
        )
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="读取旅行规划状态失败。",
        ) from exc

    return build_public_run_response(
        thread_id=thread_id,
        graph_output=None,
        snapshot=snapshot,
        include_trace=include_trace,
        max_trace_items=(
            settings.api_trace_max_items
        ),
    )


@router.get(
    "/runs/{thread_id}/history",
    response_model=TravelRunHistoryResponse,
    summary="查询旅行规划 Thread 的 Checkpoint 历史",
)
def get_travel_run_history(
    thread_id: str,
    limit: int | None = Query(
        default=None,
        ge=1,
        description=(
            "最多返回多少个 Checkpoint；"
            "不传时使用项目默认值。"
        ),
    ),
    runtime: PlanTripWorkflowRuntime = Depends(
        get_plan_trip_runtime
    ),
) -> TravelRunHistoryResponse:
    """
    返回经过脱敏的 Checkpoint 历史。

    历史接口不会返回完整 State，只返回：
        checkpoint_id、step、next nodes、是否 interrupt、proposal_id 等摘要。
    """

    resolved_limit = (
        limit
        if limit is not None
        else settings.api_history_default_limit
    )

    if resolved_limit > (
        settings.api_history_max_limit
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "limit 不能超过 "
                f"{settings.api_history_max_limit}"
            ),
        )

    try:
        snapshots = runtime.get_history(
            thread_id,
            limit=resolved_limit,
        )

    except WorkflowThreadNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "读取 PlanTrip Checkpoint 历史失败"
        )
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="读取旅行规划历史失败。",
        ) from exc

    return build_public_history_response(
        thread_id=thread_id,
        snapshots=snapshots,
    )
