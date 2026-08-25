from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from app.agents.state import TravelState
from app.common.config import settings
from app.rag.rerank_runtime import (
    get_default_reranker,
)
from app.rag.reranker import (
    CrossEncoderReranker,
)


def rerank_node(
    state: TravelState,
    reranker: CrossEncoderReranker | None = None,
) -> dict[str, Any]:
    """
    RerankNode：根据配置执行 Cross-Encoder 精排，或透传 Hybrid RRF 结果。

    读取 State：
        - retrieval_plan
        - retrieved_chunks

    写入 State：
        - reranked_evidence
        - rerank_result
        - trace
        - errors，发生不可恢复异常时写入

    这个节点不做：
        - 不重新执行 BM25。
        - 不重新执行 Chroma Vector Search。
        - 不执行 RRF。
        - 不判断最终证据是否足够。
        - 不生成旅行方案。

    当 ``RAG_RERANK_ENABLED=true``（或测试/评测代码显式传入 reranker）时，
    本节点执行下方所示的 Cross-Encoder 精排路径。默认关闭时，它不会加载
    Transformers 模型，而是把 Hybrid RRF（Variant C）的结果适配为统一的
    Evidence 结构后交给后续节点，兼顾低延迟和 Windows CPU 稳定性。

    精排开启时，本节点只做：

        RetrievalTask.semantic_query
        +
        RetrievedChunk.content
        ↓
        Cross-Encoder
        ↓
        相关性分数
        ↓
        按 task 精排
    """

    node_name = "rerank"
    started_at = perf_counter()

    try:
        retrieval_plan = state.get(
            "retrieval_plan"
        )

        retrieved_chunks = state.get(
            "retrieved_chunks"
        )

        if not isinstance(
            retrieval_plan,
            dict,
        ):
            raise TypeError(
                "state['retrieval_plan'] 必须是 dict，"
                "请先运行 RetrievalPlanNode"
            )

        if not isinstance(
            retrieved_chunks,
            list,
        ):
            raise TypeError(
                "state['retrieved_chunks'] 必须是 list，"
                "请先运行 HybridRetrieveNode"
            )

        # 1. 默认前端路径使用 Hybrid RRF（Variant C），而不是每次请求都
        #    加载并执行 CPU Cross-Encoder。这里故意在调用
        #    get_default_reranker() 之前判断，确保关闭开关时 Transformers
        #    根本不会被导入，避免 Windows CPU 的原生线程冲突。
        #
        #    测试或离线实验显式传入 reranker 时仍执行精排；这让 Variant D
        #    的实现和单元测试继续可用，而不会被默认交互配置干扰。
        if reranker is None and not settings.rag_rerank_enabled:
            output = _build_rrf_passthrough_output(
                retrieval_plan=retrieval_plan,
                retrieved_chunks=retrieved_chunks,
            )
            trace_item = _build_trace_item(
                node_name=node_name,
                status="success",
                started_at=started_at,
                input_summary=_summarize_input(
                    retrieval_plan=retrieval_plan,
                    retrieved_chunks=retrieved_chunks,
                ),
                output_summary=(
                    "status=ok, strategy=rrf_passthrough_v1, "
                    f"output_evidence={len(output['reranked_evidence'])}"
                ),
            )
            trace_item["tool_name"] = "rrf_passthrough"
            return {
                "reranked_evidence": output["reranked_evidence"],
                "rerank_result": output["summary"],
                "trace": [trace_item],
            }

        # 2. 测试时可以注入 Fake Reranker；高质量模式开启时才复用应用级
        #    Cross-Encoder 单例。
        active_reranker = (
            reranker
            or get_default_reranker()
        )

        # 3. 对 RetrievalPlan 中每个 task 的 chunks 独立精排。
        output = active_reranker.rerank_plan(
            retrieval_plan=(
                retrieval_plan
            ),
            retrieved_chunks=(
                retrieved_chunks
            ),
        )

        reranked_evidence = output[
            "reranked_evidence"
        ]

        rerank_result = output[
            "summary"
        ]

        task_errors = output[
            "errors"
        ]

        # 4. 根据精排结果决定 Trace 状态。
        if rerank_result["status"] == "failed":
            trace_status = "failed"

        elif rerank_result["status"] in {
            "degraded",
            "partial",
        }:
            trace_status = "degraded"

        else:
            trace_status = "success"

        trace_item = _build_trace_item(
            node_name=node_name,
            status=trace_status,
            started_at=started_at,
            input_summary=_summarize_input(
                retrieval_plan=(
                    retrieval_plan
                ),
                retrieved_chunks=(
                    retrieved_chunks
                ),
            ),
            output_summary=_summarize_output(
                rerank_result
            ),
        )

        update: dict[str, Any] = {
            "reranked_evidence":
                reranked_evidence,

            "rerank_result":
                rerank_result,

            "trace":
                [trace_item],
        }

        # 5. fallback 属于降级，不会阻止工作流继续。
        #    只有没有成功 fallback 的 task errors 才写入 State.errors。
        if task_errors:
            update["errors"] = [
                {
                    "node":
                        node_name,

                    "task_id":
                        error.get(
                            "task_id"
                        ),

                    "type":
                        error.get(
                            "type",
                            "RerankTaskError",
                        ),

                    "message":
                        error.get(
                            "message",
                            "Rerank 任务失败",
                        ),

                    "created_at":
                        datetime.now(
                            timezone.utc
                        ).isoformat(),
                }
                for error in task_errors
            ]

        return update

    except Exception as exc:
        error_item = {
            "node":
                node_name,

            "type":
                exc.__class__.__name__,

            "message":
                str(exc),

            "created_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        trace_item = _build_trace_item(
            node_name=node_name,
            status="failed",
            started_at=started_at,
            input_summary=(
                "rerank execution failed"
            ),
            output_summary=str(exc),
        )

        return {
            "reranked_evidence": [],

            "rerank_result": {
                "status": "failed",
                "strategy_version":
                    "cross_encoder_rerank_v1",
                "model_id": "unavailable",
                "task_count": 0,
                "successful_task_count": 0,
                "fallback_task_count": 0,
                "no_candidate_task_count": 0,
                "failed_task_count": 1,
                "input_chunk_count": 0,
                "output_evidence_count": 0,
                "top_k_per_task": 5,
                "min_hotel_evidence_per_hotel": 1,
                "fallback_used": False,
                "task_summaries": [],
                "issues": [],
            },

            "errors": [
                error_item
            ],

            "trace": [
                trace_item
            ],
        }


def _summarize_input(
    retrieval_plan: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """
    生成 RerankNode 输入摘要。

    不把完整 chunk 正文放进 Trace。
    """

    tasks = retrieval_plan.get(
        "tasks",
        [],
    )

    tasks = (
        tasks
        if isinstance(tasks, list)
        else []
    )

    task_ids = [
        task.get("task_id")
        for task in tasks
        if isinstance(task, dict)
    ]

    return (
        f"plan_id={retrieval_plan.get('plan_id')}, "
        f"task_count={len(tasks)}, "
        f"task_ids={task_ids}, "
        f"retrieved_chunk_count={len(retrieved_chunks)}"
    )


def _build_rrf_passthrough_output(
    retrieval_plan: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert Hybrid RRF chunks into the Evidence contract without a model.

    ``EvidenceGradeNode`` deliberately consumes one stable evidence schema no
    matter which ranking strategy produced it. When the optional Cross-Encoder
    is disabled, this adapter preserves every valid RRF result, its original
    rank and metadata, then fills the rerank-only fields with explicit ``None``
    / ``disabled`` values. It is not an error fallback: Variant C is the
    intentional default interaction path on local CPU deployments.
    """

    tasks = retrieval_plan.get("tasks", [])
    tasks = tasks if isinstance(tasks, list) else []
    category_by_task = {
        str(task.get("task_id")): str(task.get("category"))
        for task in tasks
        if isinstance(task, dict) and task.get("task_id")
    }

    # Group first so each task keeps its own RRF order. A globally sorted list
    # would interleave guide, hotel and safety chunks and make rerank_rank
    # meaningless to downstream debugging.
    chunks_by_task: dict[str, list[dict[str, Any]]] = {}
    for raw_chunk in retrieved_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        task_id = str(raw_chunk.get("task_id") or "")
        if task_id not in category_by_task:
            continue
        chunks_by_task.setdefault(task_id, []).append(dict(raw_chunk))

    evidence: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    no_candidate_task_count = 0

    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "")
        task_chunks = chunks_by_task.get(task_id, [])
        task_chunks.sort(
            key=lambda chunk: (
                _safe_rank(chunk.get("final_rank")),
                str(chunk.get("chunk_id") or ""),
            )
        )

        if not task_chunks:
            no_candidate_task_count += 1
            task_summaries.append(
                {
                    "task_id": task_id,
                    "category": category_by_task.get(task_id),
                    "status": "no_candidates",
                    "input_count": 0,
                    "output_count": 0,
                }
            )
            continue

        for rrf_rank, chunk in enumerate(task_chunks, start=1):
            evidence.append(
                {
                    "task_id": task_id,
                    "category": category_by_task[task_id],
                    "chunk_id": str(chunk.get("chunk_id") or ""),
                    "content": str(chunk.get("content") or ""),
                    "source": str(chunk.get("source") or ""),
                    "metadata": dict(chunk.get("metadata") or {}),
                    "bm25_rank": chunk.get("bm25_rank"),
                    "vector_rank": chunk.get("vector_rank"),
                    "original_fusion_rank": _safe_rank(chunk.get("final_rank")),
                    "fusion_score": float(chunk.get("fusion_score") or 0.0),
                    "retrieval_sources": list(chunk.get("retrieval_sources") or []),
                    "rerank_raw_score": None,
                    "rerank_score": None,
                    "rerank_rank": rrf_rank,
                    # 这是一个明确的策略标识，而不是把模型名称留空。
                    # EvidenceGrader 仅会接受这个精确标识 + 下方的
                    # selection_reason 组合，因此不能借由缺少分数的任意
                    # 伪造证据绕过校验。
                    "rerank_model": "rrf_passthrough",
                    "selection_reason": "rrf_passthrough",
                }
            )

        task_summaries.append(
            {
                "task_id": task_id,
                "category": category_by_task[task_id],
                "status": "rrf_passthrough",
                "input_count": len(task_chunks),
                "output_count": len(task_chunks),
            }
        )

    return {
        "reranked_evidence": evidence,
        "summary": {
            "status": "partial" if no_candidate_task_count else "ok",
            "strategy_version": "rrf_passthrough_v1",
            "model_id": "disabled",
            "task_count": len(tasks),
            "successful_task_count": len(tasks) - no_candidate_task_count,
            "fallback_task_count": 0,
            "no_candidate_task_count": no_candidate_task_count,
            "failed_task_count": 0,
            "input_chunk_count": len(retrieved_chunks),
            "output_evidence_count": len(evidence),
            # 不使用 0：该摘要仍需符合 RerankResult 的统一数据契约。真正的
            # 输出条数由 reranked_evidence 决定，RRF 透传不会据此截断候选。
            "top_k_per_task": settings.rag_rerank_top_k_per_task,
            "min_hotel_evidence_per_hotel": 0,
            "fallback_used": False,
            "task_summaries": task_summaries,
            "issues": [],
        },
    }


def _safe_rank(value: Any) -> int:
    """Return a sortable positive RRF rank without trusting malformed state."""

    try:
        rank = int(value)
    except (TypeError, ValueError):
        return 999999
    return rank if rank > 0 else 999999


def _summarize_output(
    rerank_result: dict[str, Any],
) -> str:
    """
    生成 RerankNode 输出摘要。
    """

    return (
        f"status={rerank_result.get('status')}, "
        f"model={rerank_result.get('model_id')}, "
        f"successful_tasks={rerank_result.get('successful_task_count')}, "
        f"fallback_tasks={rerank_result.get('fallback_task_count')}, "
        f"output_evidence={rerank_result.get('output_evidence_count')}"
    )


def _build_trace_item(
    node_name: str,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """
    生成统一 Trace 记录。
    """

    latency_ms = int(
        (
            perf_counter()
            - started_at
        )
        * 1000
    )

    return {
        "node_name":
            node_name,

        "tool_name":
            "cross_encoder_reranker",

        "input_summary":
            input_summary,

        "output_summary":
            output_summary,

        "status":
            status,

        "latency_ms":
            latency_ms,

        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }
