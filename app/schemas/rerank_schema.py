from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.retrieval_plan_schema import RetrievalCategory


class RerankedEvidence(BaseModel):
    """
    RerankNode 输出的一条精排证据。

    它保留 HybridRetrieveNode 原有的检索信息，
    同时增加 Cross-Encoder 的精排分数与排名。

    为什么保留原始 fusion_score？

        因为后续调试时，我们需要知道：

            RRF 原来把它排在第几名？
            Cross-Encoder 为什么把它升高或降低？

        所以 Rerank 不会覆盖原始检索信息，
        而是在原结果上增加精排信息。
    """

    task_id: str = Field(
        description="这条证据属于哪个 RetrievalTask。"
    )

    category: RetrievalCategory = Field(
        description="证据所属类别，例如 guides、hotel_reviews。"
    )

    chunk_id: str = Field(
        description="文档 chunk 的稳定唯一 ID。"
    )

    content: str = Field(
        description="文档 chunk 正文。"
    )

    source: str = Field(
        description="原始 Markdown 文档相对路径。"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "文档 metadata，例如 city、doc_type、hotel_id、section。"
        ),
    )

    bm25_rank: int | None = Field(
        default=None,
        description="该 chunk 在 BM25 中的排名。",
    )

    vector_rank: int | None = Field(
        default=None,
        description="该 chunk 在 Vector Search 中的排名。",
    )

    original_fusion_rank: int | None = Field(
        default=None,
        description="该 chunk 在 RRF 融合后的原始排名。",
    )

    fusion_score: float = Field(
        default=0.0,
        description="HybridRetrieveNode 生成的 RRF 分数。",
    )

    retrieval_sources: list[str] = Field(
        default_factory=list,
        description="该 chunk 是由 BM25、Vector 还是两者召回。",
    )

    rerank_raw_score: float | None = Field(
        default=None,
        description=(
            "Cross-Encoder 模型输出的原始 logit。"
            "这个值可能小于 0 或大于 1。"
        ),
    )

    rerank_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "通过 sigmoid 映射到 0 到 1 的相关性分数。"
            "分数越高，表示 Query 与当前 chunk 越相关。"
        ),
    )

    rerank_rank: int = Field(
        ge=1,
        description="当前 chunk 在所属 task 中的精排排名。",
    )

    rerank_model: str = Field(
        description="执行精排的模型或 fallback 策略名称。",
    )

    selection_reason: str = Field(
        default="top_rerank_score",
        description=(
            "这条证据被保留的原因。"
            "例如 top_rerank_score 或 hotel_coverage。"
        ),
    )

    def to_state_dict(self) -> dict[str, Any]:
        """
        转成可以写入 TravelState 的 dict。

        同时兼容 Pydantic v1 和 v2。
        """

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class RerankResult(BaseModel):
    """
    RerankNode 的执行摘要。

    它不保存文档正文，只保存任务统计和运行状态。
    """

    status: Literal[
        "ok",
        "partial",
        "degraded",
        "no_candidates",
        "no_tasks",
        "failed",
    ] = Field(
        description="本轮 Rerank 总体状态。"
    )

    strategy_version: str = Field(
        default="cross_encoder_rerank_v1",
        description="精排策略版本。",
    )

    model_id: str = Field(
        description="使用的 Cross-Encoder 模型名称。",
    )

    task_count: int = Field(
        ge=0,
        description="RetrievalPlan 中的任务数量。",
    )

    successful_task_count: int = Field(
        ge=0,
        description="成功执行 Cross-Encoder 精排的任务数量。",
    )

    fallback_task_count: int = Field(
        ge=0,
        description="因为模型异常而回退到 RRF 排名的任务数量。",
    )

    no_candidate_task_count: int = Field(
        ge=0,
        description="没有可精排候选的任务数量。",
    )

    failed_task_count: int = Field(
        ge=0,
        description="精排失败且没有成功 fallback 的任务数量。",
    )

    input_chunk_count: int = Field(
        ge=0,
        description="输入 RerankNode 的 retrieved_chunks 总数量。",
    )

    output_evidence_count: int = Field(
        ge=0,
        description="精排后保留的 evidence 总数量。",
    )

    top_k_per_task: int = Field(
        ge=1,
        description="每个任务默认保留的最大证据数量。",
    )

    min_hotel_evidence_per_hotel: int = Field(
        ge=0,
        description="每个候选酒店至少保留多少条酒店评价证据。",
    )

    fallback_used: bool = Field(
        description="本轮是否使用过 RRF fallback。",
    )

    task_summaries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="每个 RetrievalTask 的精排执行摘要。",
    )

    issues: list[dict[str, Any]] = Field(
        default_factory=list,
        description="精排过程中的降级、异常或覆盖问题。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """
        转成可以写入 TravelState 的 dict。
        """

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()