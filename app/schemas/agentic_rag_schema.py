from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.retrieval_plan_schema import RetrievalCategory


class InformationNeed(BaseModel):
    """
    用户需求中需要由 RAG 证据回答的一项独立信息目标。

    是否 required 不属于 LLM 的决策范围。最终硬性要求由 Rule Plan 中
    对应类别的 RetrievalTask.required 决定，避免模型绕过确定性策略。
    """

    need_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    category: RetrievalCategory
    success_criteria: str = Field(
        min_length=1,
        description="什么样的证据才算真正覆盖这项需求。",
    )


class RetrievalIntentTask(BaseModel):
    """
    LLM 生成的语义检索意图。

    这里故意不包含 city、hotel_id、metadata_filter、top_k 等硬参数；
    它们由程序根据当前 TravelState 和现有规则计划补齐。
    """

    task_id: str = Field(min_length=1)
    category: RetrievalCategory
    covers_need_ids: list[str] = Field(min_length=1)
    semantic_query: str = Field(min_length=1, max_length=500)
    keyword_query: str = Field(min_length=1, max_length=300)
    purpose: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)


class RetrievalPlannerDecision(BaseModel):
    """LLM Retrieval Planner 的唯一结构化输出。"""

    mode: Literal["initial", "repair"] = "initial"
    information_needs: list[InformationNeed] = Field(default_factory=list)
    tasks: list[RetrievalIntentTask] = Field(default_factory=list)
    unsupported_needs: list[str] = Field(default_factory=list)


class NeedCoverage(BaseModel):
    """LLM Evidence Judge 对单个 Information Need 的覆盖结论。"""

    need_id: str = Field(min_length=1)
    status: Literal["supported", "partial", "missing"]
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    unmet_requirement_codes: list[
        Literal[
            "destination_not_supported",
            "activity_preference_not_supported",
            "complete_itinerary_not_preassembled",
            "trip_duration_not_explicit",
            "exact_date_not_explicit",
            "other",
        ]
    ] = Field(
        default_factory=list,
        description=(
            "partial/missing 的结构化原因；不得把 Proposal 负责的逐日编排、"
            "旅行天数或精确日期当作 Guides 偏好证据缺口。"
        ),
    )


class SemanticEvidenceDecision(BaseModel):
    """Evidence Judge 对本轮累计证据的语义充分性判断。"""

    sufficient: bool
    need_coverage: list[NeedCoverage] = Field(default_factory=list)
    conflict_codes: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    def missing_need_ids(self, required_need_ids: set[str]) -> list[str]:
        """返回 required 且尚未被完整支持的 need_id，保持模型输出顺序。"""

        return [
            item.need_id
            for item in self.need_coverage
            if item.need_id in required_need_ids
            and item.status != "supported"
        ]
