from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from app.agents.state import TravelState
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.runtime import get_default_hybrid_retriever


def hybrid_retrieve_node(
    state: TravelState,
    retriever: HybridRetriever | None = None,
) -> dict[str, Any]:
    """
    HybridRetrieveNode：执行 BM25 + Chroma Vector + RRF 混合检索。

    读取 State：
        - retrieval_plan

    写入 State：
        - retrieved_chunks
        - hybrid_retrieval_result
        - trace
        - errors（部分任务或整体失败时）

    生命周期说明：
        - BM25 在应用启动时构建一次并保存在 HybridRetriever 中。
        - Chroma 向量库持久化在磁盘。
        - 本节点每次只执行 query，不重新建立 BM25，也不重新编码文档。
    """

    node_name = "hybrid_retrieve"
    started_at = perf_counter()

    try:
        retrieval_plan = state.get("retrieval_plan")

        # 1. 必须先运行 RetrievalPlanNode，才能知道需要执行哪些检索任务。
        if not isinstance(retrieval_plan, dict):
            raise TypeError(
                "state['retrieval_plan'] 必须是 dict，请先运行 RetrievalPlanNode"
            )

        # 2. 单元测试可注入临时 retriever；正常运行复用应用启动时初始化的单例。
        active_retriever = retriever or get_default_hybrid_retriever()

        # 3. 执行 retrieval_plan 中所有任务。
        retrieval_output = active_retriever.retrieve_plan(retrieval_plan)
        retrieved_chunks = retrieval_output["retrieved_chunks"]
        retrieval_summary = retrieval_output["summary"]
        task_errors = retrieval_output["errors"]

        # 4. 生成 Trace 摘要，不把文档正文和完整 query 写入 Trace。
        trace_item = _build_trace_item(
            node_name=node_name,
            status=(
                "failed"
                if retrieval_summary["status"] == "failed"
                else "success"
            ),
            started_at=started_at,
            input_summary=_summarize_input(retrieval_plan),
            output_summary=_summarize_output(retrieval_summary),
        )

        update: dict[str, Any] = {
            "retrieved_chunks": retrieved_chunks,
            "hybrid_retrieval_result": retrieval_summary,
            "trace": [trace_item],
        }

        # 5. 某个 RetrievalTask 失败时，将任务错误转成统一 State errors。
        if task_errors:
            update["errors"] = [
                {
                    "node": node_name,
                    "task_id": error.get("task_id"),
                    "type": error.get("type", "RetrievalTaskError"),
                    "message": error.get("message", "RAG 检索任务执行失败"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                for error in task_errors
            ]

        return update

    except Exception as exc:
        # 6. 整个节点失败时，返回空结果和结构化失败摘要，让工作流能够进入错误处理。
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
            input_summary="hybrid retrieval failed",
            output_summary=str(exc),
        )

        return {
            "retrieved_chunks": [],
            "hybrid_retrieval_result": {
                "status": "failed",
                "strategy_version": "bm25_memory_chroma_rrf_v2",
                "corpus_size": 0,
                "task_count": 0,
                "successful_task_count": 0,
                "no_result_task_count": 0,
                "failed_task_count": 1,
                "total_retrieved_chunks": 0,
                "rrf_k": 60,
                "weights": {"bm25": 1.0, "vector": 1.0},
                "task_summaries": [],
            },
            "errors": [error_item],
            "trace": [trace_item],
        }


def _summarize_input(retrieval_plan: dict[str, Any]) -> str:
    """生成输入摘要，不记录完整 query 和 metadata。"""

    tasks = retrieval_plan.get("tasks", [])
    tasks = tasks if isinstance(tasks, list) else []
    categories = [
        task.get("category")
        for task in tasks
        if isinstance(task, dict)
    ]

    return (
        f"plan_id={retrieval_plan.get('plan_id')}, "
        f"mode={retrieval_plan.get('mode')}, "
        f"task_count={len(tasks)}, "
        f"categories={categories}"
    )


def _summarize_output(summary: dict[str, Any]) -> str:
    """生成输出摘要。"""

    return (
        f"status={summary.get('status')}, "
        f"corpus_size={summary.get('corpus_size')}, "
        f"successful_tasks={summary.get('successful_task_count')}, "
        f"no_result_tasks={summary.get('no_result_task_count')}, "
        f"failed_tasks={summary.get('failed_task_count')}, "
        f"retrieved_chunks={summary.get('total_retrieved_chunks')}"
    )


def _build_trace_item(
    node_name: str,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """生成统一 Trace 记录。"""

    latency_ms = int((perf_counter() - started_at) * 1000)

    return {
        "node_name": node_name,
        "tool_name": "bm25+chroma+rrf",
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "latency_ms": latency_ms,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
