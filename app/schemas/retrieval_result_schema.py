from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """
    HybridRetrieveNode 输出的一条 RRF 融合检索结果。

    每个对象对应一个被 BM25、Vector，或两者共同召回的文档 chunk。
    """

    task_id: str = Field(
        description=(
            "当前检索结果所属的 RetrievalTask 唯一标识，"
            "用于区分同一次 HybridRetrieveNode 中的不同检索任务。"
        ),
    )

    chunk_id: str = Field(
        description=(
            "当前文档切片的唯一标识。"
            "用于 BM25 与 Vector 结果去重、融合、追踪和引用。"
        ),
    )

    content: str = Field(
        description=(
            "当前文档切片的正文内容，"
            "供后续 RerankNode、EvidenceGradeNode 和答案生成节点使用。"
        ),
    )

    source: str = Field(
        description=(
            "当前文档切片的来源标识，"
            "例如文件名、文档路径、数据源名称或知识库来源。"
        ),
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "当前文档切片的 metadata。"
            "可以包含 city、doc_type、scenario、section、risk_level、"
            "document_id、page 等过滤和追踪字段。"
        ),
    )

    bm25_score: float | None = Field(
        default=None,
        description=(
            "当前 chunk 在 BM25 稀疏关键词检索中的原始相关性分数。"
            "如果该 chunk 未被 BM25 召回，则为 None。"
        ),
    )

    vector_score: float | None = Field(
        default=None,
        description=(
            "当前 chunk 在稠密向量检索中的原始相似度分数。"
            "如果该 chunk 未被 Vector 检索召回，则为 None。"
            "具体数值含义取决于底层向量库返回的是相似度还是距离。"
        ),
    )

    bm25_rank: int | None = Field(
        default=None,
        ge=1,
        description=(
            "当前 chunk 在 BM25 检索结果中的排名，从 1 开始。"
            "如果未被 BM25 召回，则为 None。"
        ),
    )

    vector_rank: int | None = Field(
        default=None,
        ge=1,
        description=(
            "当前 chunk 在 Vector 检索结果中的排名，从 1 开始。"
            "如果未被 Vector 检索召回，则为 None。"
        ),
    )

    fusion_score: float = Field(
        ge=0,
        description=(
            "BM25 与 Vector 检索结果经过 RRF 或其他融合策略后得到的综合分数。"
            "分数越高，表示当前 chunk 的综合检索优先级越高。"
        ),
    )

    final_rank: int = Field(
        ge=1,
        description=(
            "当前 chunk 在完成 BM25、Vector 去重和融合后的最终排名，"
            "从 1 开始。"
        ),
    )

    retrieval_sources: list[Literal["bm25", "vector"]] = Field(
        default_factory=list,
        description=(
            "召回当前 chunk 的检索通道列表。"
            "可能为 ['bm25']、['vector']，"
            "或同时包含 ['bm25', 'vector']。"
        ),
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class HybridRetrievalResult(BaseModel):
    """
    HybridRetrieveNode 的运行摘要。

    用于记录本次混合检索的整体执行情况、任务统计、
    RRF 配置和各检索任务的运行结果。
    """

    status: Literal[
        "ok",
        "partial",
        "no_results",
        "no_tasks",
        "failed",
    ] = Field(
        description=(
            "HybridRetrieveNode 的整体运行状态。"
            "ok 表示全部任务成功；"
            "partial 表示部分任务成功；"
            "no_results 表示执行成功但没有召回结果；"
            "no_tasks 表示没有可执行的检索任务；"
            "failed 表示节点执行失败。"
        ),
    )

    strategy_version: str = Field(
        default="bm25_memory_chroma_rrf_v2",
        description=(
            "当前混合检索策略的版本标识。"
            "用于区分 BM25、向量库、融合方法和参数配置的不同实现版本。"
        ),
    )

    corpus_size: int = Field(
        ge=0,
        description=(
            "当前混合检索器可检索的文档 chunk 总数量。"
        ),
    )

    task_count: int = Field(
        ge=0,
        description=(
            "本次 HybridRetrieveNode 接收到的 RetrievalTask 总数量。"
        ),
    )

    successful_task_count: int = Field(
        ge=0,
        description=(
            "本次执行中成功完成并召回至少一条结果的检索任务数量。"
        ),
    )

    no_result_task_count: int = Field(
        ge=0,
        description=(
            "本次执行中正常完成但没有召回任何文档的检索任务数量。"
        ),
    )

    failed_task_count: int = Field(
        ge=0,
        description=(
            "本次执行中发生异常或未能完成检索的任务数量。"
        ),
    )

    total_retrieved_chunks: int = Field(
        ge=0,
        description=(
            "所有检索任务完成融合和去重后，"
            "最终输出的 RetrievedChunk 总数量。"
        ),
    )

    rrf_k: int = Field(
        ge=1,
        description=(
            "Reciprocal Rank Fusion 使用的排名平滑常数。"
            "该值用于降低排名靠后结果对融合分数的影响。"
        ),
    )

    weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "不同检索通道在融合阶段使用的权重配置。"
            "通常包含 bm25 和 vector，例如："
            "{'bm25': 1.0, 'vector': 1.0}。"
        ),
    )

    task_summaries: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "每个 RetrievalTask 的执行摘要列表。"
            "可以记录 task_id、查询文本、metadata_filter、"
            "BM25 召回数量、Vector 召回数量、融合结果数量、"
            "任务状态和错误信息等内容。"
        ),
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()