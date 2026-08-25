from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RetrievalCategory = Literal[
    "guides",
    "hotel_reviews",
    "safety_notices",
    "packing_checklists",
]

RAGDocumentType = Literal[
    "guide",
    "hotel_reviews",
    "safety_notice",
    "packing_checklist",
]


class RetrievalTask(BaseModel):
    """
    一条独立的 RAG 检索任务。

    RetrievalPlanNode 会根据用户需求拆出多条任务，例如：

    1. 查询成都三日游攻略。
    2. 查询候选酒店的住客评价。
    3. 查询成都雨天安全提醒。
    4. 查询雨天三日游出行清单。

    HybridRetrieveNode 后面会逐条执行这些任务。
    """

    task_id: str = Field(
        min_length=1,
        description="任务唯一标识，例如 guide_main、hotel_review_candidates。",
    )

    category: RetrievalCategory = Field(
        description=(
            "逻辑检索类别。"
            "使用复数名称，方便 coverage_requirements 按类别统计。"
        ),
    )

    doc_type: RAGDocumentType = Field(
        description=(
            "RAG 文档 metadata 中实际使用的 doc_type。"
            "例如 guide、hotel_reviews、safety_notice、packing_checklist。"
        ),
    )

    purpose: str = Field(
        min_length=1,
        description="这条任务解决什么问题，主要用于 trace 和调试。",
    )

    semantic_query: str = Field(
        min_length=1,
        description="给 Vector Search 使用的自然语言语义查询。",
    )

    keyword_query: str = Field(
        min_length=1,
        description="给 BM25 使用的关键词查询。",
    )

    metadata_filter: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "检索前使用的 metadata 过滤条件。"
            "例如 city、doc_type、hotel_ids、risk_level。"
        ),
    )

    required: bool = Field(
        default=True,
        description=(
            "这类证据是否必须存在。"
            "EvidenceGradeNode 会根据它判断是否需要修复检索。"
        ),
    )

    top_k: int = Field(
        default=8,
        ge=1,
        le=50,
        description="HybridRetrieveNode 初始召回的最大数量。",
    )

    min_required_chunks: int = Field(
        default=1,
        ge=0,
        le=20,
        description="EvidenceGradeNode 判断任务通过所需的最少证据数。",
    )

    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="任务优先级。1 最高，10 最低。",
    )

    reason: str = Field(
        default="",
        description="为什么生成这条检索任务。",
    )

    repair_of: str | None = Field(
        default=None,
        description=(
            "repair 模式下，记录这条任务用于修复哪个原任务。"
            "initial 模式下通常为 None。"
        ),
    )

    covers_need_ids: list[str] = Field(
        default_factory=list,
        description=(
            "这条检索任务服务的 Information Need IDs。"
            "规则模式可以为空，Agentic 模式用于语义覆盖和定向 repair。"
        ),
    )

    def to_state_dict(self) -> dict[str, Any]:
        """
        把 RetrievalTask 转成适合写入 TravelState 的 dict。

        同时兼容 Pydantic v1 和 v2。
        """

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class RetrievalPlan(BaseModel):
    """
    RetrievalPlanNode 的完整输出。

    这个结构只描述“应该如何检索”，不包含真正检索到的文档。

    后续调用关系：

        RetrievalPlanNode
        ↓
        retrieval_plan
        ↓
        HybridRetrieveNode
        ↓
        retrieved_chunks
        ↓
        RerankNode
        ↓
        reranked_evidence
        ↓
        EvidenceGradeNode
    """

    plan_id: str = Field(
        min_length=1,
        description="检索计划唯一标识。",
    )

    mode: Literal["initial", "repair"] = Field(
        default="initial",
        description="initial 为首次检索，repair 为证据不足后的修复检索。",
    )

    strategy_version: str = Field(
        default="rule_retrieval_plan_v1",
        description="检索规划策略版本，方便 Trace 和 Eval 对比。",
    )

    city: str = Field(
        min_length=1,
        description="本次 RAG 检索的目标城市。",
    )

    date_range: dict[str, str | None] = Field(
        default_factory=dict,
        description="旅行日期范围。",
    )

    tasks: list[RetrievalTask] = Field(
        default_factory=list,
        description="需要执行的检索任务列表。",
    )

    coverage_requirements: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "各类证据的覆盖要求。"
            "EvidenceGradeNode 会读取它检查证据是否足够。"
        ),
    )

    information_needs: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Agentic Planner 从用户需求拆出的稳定信息目标。"
            "repair 轮次必须沿用 initial 轮次的信息目标；是否为硬性要求"
            "由对应类别的 coverage_requirements 决定。"
        ),
    )

    planner_meta: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "规划器类型、模型、fallback 和 reason codes 等可观测元数据。"
        ),
    )

    max_repair_rounds: int = Field(
        default=1,
        ge=0,
        le=3,
        description="最多允许多少次修复检索。",
    )

    repair_exhausted: bool = Field(
        default=False,
        description="修复次数是否已经用完。",
    )

    notes: list[str] = Field(
        default_factory=list,
        description="检索计划说明和降级信息。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """
        把 RetrievalPlan 转成适合写入 TravelState 的 dict。
        """

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()
