from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from app.agents.state import TravelState
from app.rag.retrieval_planner import (
    RetrievalPlanner,
)


def retrieval_plan_node(
    state: TravelState,
    planner: RetrievalPlanner | None = None,
) -> dict[str, Any]:
    """
    RetrievalPlanNode：为行程知识和补充说明生成结构化检索任务。

    最终工作流中的位置：
        CandidateRankNode
        → BudgetOptimizeNode
        → RetrievalPlanNode

    读取 State：
        - trip_request
        - planning_context
        - weather_result，可选
        - selection_result，正常初次规划时应存在
        - raw_hotel_results，用来补齐选中酒店的 Mock 名称和区域
        - retrieval_feedback，repair 模式使用
        - rag_retry_count

    写入 State：
        - retrieval_plan
        - rag_retry_count，真正生成 repair 任务时递增
        - trace
        - errors，程序异常时

    这个节点不执行 BM25、Vector、Rerank，也不调用 LLM。
    """

    node_name = "retrieval_plan"
    started_at = perf_counter()
    if planner is None:
        # 默认 Planner 在应用级缓存；Agentic 模式不会在每个 Node 调用中
        # 重建模型客户端，rule 模式则返回原 RuleBasedRetrievalPlanner。
        from app.rag.agentic_runtime import get_default_retrieval_planner

        active_planner = get_default_retrieval_planner()
    else:
        active_planner = planner

    try:
        trip_request = state.get("trip_request")
        planning_context = state.get("planning_context")

        if not isinstance(trip_request, dict):
            raise TypeError(
                "state['trip_request'] 必须是 dict，请先运行 InputExtractNode"
            )

        if not isinstance(planning_context, dict):
            raise TypeError(
                "state['planning_context'] 必须是 dict，请先运行 PlanningContextNode"
            )

        weather_result = state.get("weather_result")
        selection_result = state.get("selection_result")
        raw_hotel_results = state.get("raw_hotel_results")
        retrieval_feedback = state.get("retrieval_feedback")
        rag_retry_count = _safe_int(state.get("rag_retry_count"), default=0)

        # 1. Planner 只生成任务计划；实际检索由 HybridRetrieveNode 执行。
        retrieval_plan = active_planner.plan(
            trip_request=trip_request,
            planning_context=planning_context,
            weather_result=(
                weather_result if isinstance(weather_result, dict) else {}
            ),
            selection_result=(
                selection_result if isinstance(selection_result, dict) else {}
            ),
            raw_hotel_results=(
                raw_hotel_results if isinstance(raw_hotel_results, dict) else {}
            ),
            retrieval_feedback=(
                retrieval_feedback if isinstance(retrieval_feedback, dict) else None
            ),
            rag_retry_count=rag_retry_count,
        )

        plan_dict = retrieval_plan.to_state_dict()
        update: dict[str, Any] = {
            "retrieval_plan": plan_dict,
        }

        # 2. 只有 repair 模式真正产生任务时，才增加重试次数。
        #    空 repair plan 或 repair_exhausted 不应消耗新的重试次数。
        if (
            retrieval_plan.mode == "repair"
            and retrieval_plan.tasks
            and not retrieval_plan.repair_exhausted
        ):
            update["rag_retry_count"] = rag_retry_count + 1

        trace_item = _build_trace_item(
            node_name=node_name,
            status="success",
            started_at=started_at,
            input_summary=_summarize_input(
                trip_request=trip_request,
                planning_context=planning_context,
                weather_result=weather_result,
                selection_result=selection_result,
                rag_retry_count=rag_retry_count,
            ),
            output_summary=_summarize_plan(plan_dict),
        )
        update["trace"] = [trace_item]

        return update

    except Exception as exc:
        error_item = {
            "node": node_name,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        trace_item = _build_trace_item(
            node_name=node_name,
            status="failed",
            started_at=started_at,
            input_summary="retrieval plan build failed",
            output_summary=str(exc),
        )

        return {
            "errors": [error_item],
            "trace": [trace_item],
        }


def _summarize_input(
    *,
    trip_request: dict[str, Any],
    planning_context: dict[str, Any],
    weather_result: Any,
    selection_result: Any,
    rag_retry_count: int,
) -> str:
    """生成不包含正文和大对象的输入摘要。"""

    request = planning_context.get("request")
    request = request if isinstance(request, dict) else {}

    risks = weather_result.get("risks") if isinstance(weather_result, dict) else []
    risk_count = len(risks) if isinstance(risks, list) else 0

    selected_hotel_id = None
    if isinstance(selection_result, dict):
        selected_hotel_id = selection_result.get("selected_hotel_id")
        combination = selection_result.get("selected_combination")
        if selected_hotel_id is None and isinstance(combination, dict):
            selected_hotel_id = combination.get("hotel_id")

    return (
        f"destination={request.get('destination') or trip_request.get('destination')}, "
        f"days={request.get('days') or trip_request.get('days')}, "
        f"selected_hotel_id={selected_hotel_id}, "
        f"weather_risk_count={risk_count}, "
        f"rag_retry_count={rag_retry_count}"
    )


def _summarize_plan(retrieval_plan: dict[str, Any]) -> str:
    """生成 RetrievalPlan 输出摘要。"""

    tasks = retrieval_plan.get("tasks")
    tasks = tasks if isinstance(tasks, list) else []
    categories = [
        str(task.get("category"))
        for task in tasks
        if isinstance(task, dict) and task.get("category")
    ]

    return (
        f"mode={retrieval_plan.get('mode')}, "
        f"planner={retrieval_plan.get('planner_meta', {}).get('planner_type', 'rule')}, "
        f"task_count={len(tasks)}, "
        f"categories={categories}, "
        f"repair_exhausted={retrieval_plan.get('repair_exhausted')}"
    )


def _safe_int(value: Any, default: int) -> int:
    """安全转换 int。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_trace_item(
    node_name: str,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """生成统一 Trace 记录。"""

    return {
        "node_name": node_name,
        "tool_name": "retrieval_planner",
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "latency_ms": int((perf_counter() - started_at) * 1000),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
