from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from app.api.dependencies import (
    get_plan_trip_runtime,
)
from app.api.graph_response import (
    build_public_run_response,
)
from app.api.travel_schemas import (
    TravelDecisionRequest,
    TravelRunResponse,
)
from app.common.config import settings
from app.common.logger import get_logger
from app.workflows.workflow_runtime import (
    PlanTripWorkflowRuntime,
    WorkflowNotWaitingForDecisionError,
    WorkflowRuntimeError,
    WorkflowThreadNotFoundError,
)


router = APIRouter()
logger = get_logger("travelops.api.approval")


@router.post(
    "/decision/{thread_id}",
    response_model=TravelRunResponse,
    summary="恢复 Human-in-the-loop 并提交人工决定",
)
def submit_travel_decision(
    thread_id: str,
    payload: TravelDecisionRequest,
    runtime: PlanTripWorkflowRuntime = Depends(
        get_plan_trip_runtime
    ),
) -> TravelRunResponse:
    """
    使用同一个 thread_id 恢复停在 DecisionGateNode 的 Graph。

    支持三种人工决定：
        approve：
            → CommitDraftNode → FinalResponseNode

        request_changes：
            → RevisionAnalyzeNode → ApplyRevisionNode
            → 重新运行主链 → 第二次 interrupt

        cancel：
            → CancelNode → FinalResponseNode

    注意：
        API 不自行修改 TravelState，也不直接写数据库。
        它只把经过 Pydantic 校验的人工输入传给：
            Command(resume=...)
    """

    try:
        # 1. 这是 Human-in-the-loop Resume 的核心调用。
        execution = runtime.resume_decision(
            thread_id=thread_id,
            decision_payload=(
                payload.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            ),
        )

    except WorkflowThreadNotFoundError as exc:
        # 错误 thread_id 必须明确 404，不能偷偷创建新流程。
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except WorkflowNotWaitingForDecisionError as exc:
        # 已完成流程、未进入 interrupt 或重复提交 Decision 都属于 409。
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except WorkflowRuntimeError as exc:
        logger.exception(
            "恢复 PlanTripWorkflow 失败"
        )
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="旅行规划工作流恢复失败。",
        ) from exc

    except Exception as exc:
        logger.exception(
            "未处理的 Human-in-the-loop Resume 异常"
        )
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="人工决定提交失败。",
        ) from exc

    # 2. request_changes 可能产生新的 interrupt；approve/cancel 通常完成。
    return build_public_run_response(
        thread_id=execution.thread_id,
        graph_output=execution.output,
        snapshot=execution.snapshot,
        # resume 同样是一次独立 Run，run_id 可关联本次人工决定后的轨迹。
        run_id=execution.run_id,
        include_trace=False,
        max_trace_items=(
            settings.api_trace_max_items
        ),
    )
