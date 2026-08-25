from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.retrieval_plan_schema import (
    RetrievalCategory,
)


class EvidenceIssue(BaseModel):
    """
    EvidenceGradeNode 发现的一条证据质量问题。

    severity：

        warning
            证据仍然可以使用，但质量发生降级。
            例如 Cross-Encoder 失败，使用 RRF fallback。

        error
            会导致 required evidence 无法通过。
            例如 hotel_id 不匹配、source 缺失、证据数量不足。
    """

    issue_type: str = Field(
        description=(
            "问题类型，例如 missing_chunks、invalid_metadata、"
            "missing_hotel_evidence。"
        )
    )

    severity: Literal[
        "warning",
        "error",
    ] = Field(
        description="问题严重程度。"
    )

    category: str | None = Field(
        default=None,
        description="问题所属证据类别。",
    )

    task_id: str | None = Field(
        default=None,
        description="问题所属 RetrievalTask。",
    )

    chunk_id: str | None = Field(
        default=None,
        description="问题对应的 chunk_id。",
    )

    message: str = Field(
        description="给开发者或 Trace 阅读的问题说明。"
    )

    details: dict[str, Any] = Field(
        default_factory=dict,
        description="问题附加信息。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """
        转成适合写入 TravelState 的 dict。
        """

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class EvidenceCoverage(BaseModel):
    """
    某一类证据的覆盖检查结果。

    例如：

        guides
        hotel_reviews
        safety_notices
        packing_checklists
    """

    category: RetrievalCategory = Field(
        description="证据类别。"
    )

    required: bool = Field(
        description="该类证据是否必须存在。"
    )

    required_chunks: int = Field(
        ge=0,
        description="至少需要多少个有效 chunk。",
    )

    actual_chunks: int = Field(
        ge=0,
        description="当前证据池中实际存在多少个有效 chunk。",
    )

    task_ids: list[str] = Field(
        default_factory=list,
        description="该类别对应的 RetrievalTask IDs。",
    )

    required_hotel_ids: list[str] = Field(
        default_factory=list,
        description="酒店评价任务要求覆盖的酒店 IDs。",
    )

    covered_hotel_ids: list[str] = Field(
        default_factory=list,
        description="当前证据池已经覆盖的酒店 IDs。",
    )

    missing_hotel_ids: list[str] = Field(
        default_factory=list,
        description="仍然缺少评价证据的酒店 IDs。",
    )

    required_risk_types: list[str] = Field(
        default_factory=list,
        description="天气结果要求覆盖的风险类型。",
    )

    covered_risk_types: list[str] = Field(
        default_factory=list,
        description="安全证据已经覆盖的风险类型。",
    )

    missing_risk_types: list[str] = Field(
        default_factory=list,
        description="仍然缺少安全证据的风险类型。",
    )

    passed: bool = Field(
        description="该类别是否通过证据覆盖检查。"
    )

    def to_state_dict(self) -> dict[str, Any]:
        """
        转成适合写入 TravelState 的 dict。
        """

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class EvidenceGradeResult(BaseModel):
    """
    EvidenceGradeNode 的完整输出。

    status 说明：

        passed
            所有 required 证据都满足要求。

        degraded
            Required 证据满足要求，但使用了 RRF fallback，
            或部分无效候选证据被丢弃。

        repair_required
            Required 证据不足，并且仍有修复次数。

        failed
            Required 证据不足，并且修复次数已经耗尽。
    """

    status: Literal[
        "passed",
        "degraded",
        "repair_required",
        "failed",
    ] = Field(
        description="证据质量检查状态。"
    )

    passed: bool = Field(
        description=(
            "Required 证据是否全部满足。"
            "degraded 状态下 passed 仍然可以为 True。"
        ),
    )

    repairable: bool = Field(
        description="当前证据不足是否仍然可以通过再次检索修复。",
    )

    next_action: Literal[
        "continue",
        "repair",
        "stop",
    ] = Field(
        description=(
            "continue：进入 ProposalNode；"
            "repair：返回 RetrievalPlanNode；"
            "stop：结束当前流程。"
        ),
    )

    strategy_version: str = Field(
        default="rule_evidence_grade_v1",
        description="证据评分策略版本。",
    )

    plan_id: str | None = Field(
        default=None,
        description="当前 RetrievalPlan ID。",
    )

    mode: Literal[
        "initial",
        "repair",
    ] = Field(
        description="当前是在检查首次检索还是修复检索。",
    )

    graded_evidence_count: int = Field(
        ge=0,
        description="当前累计有效证据数量。",
    )

    rejected_evidence_count: int = Field(
        ge=0,
        description="因为结构或 metadata 不合法而被拒绝的证据数量。",
    )

    required_category_count: int = Field(
        ge=0,
        description="必须通过的证据类别数量。",
    )

    passed_required_category_count: int = Field(
        ge=0,
        description="已经通过的必需证据类别数量。",
    )

    fallback_used: bool = Field(
        description="本轮是否使用了 RRF fallback 证据。",
    )

    coverage: dict[
        str,
        EvidenceCoverage,
    ] = Field(
        default_factory=dict,
        description="每一类证据的覆盖检查结果。",
    )

    issues: list[EvidenceIssue] = Field(
        default_factory=list,
        description="证据检查中发现的问题。",
    )

    retrieval_feedback: dict[str, Any] | None = Field(
        default=None,
        description=(
            "需要修复时返回给 RetrievalPlanNode 的结构化信息。"
        ),
    )

    semantic_review: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Agentic 模式下的 need-level 语义覆盖结论；"
            "规则模式或未调用 Judge 时为 None。"
        ),
    )

    def to_state_dict(self) -> dict[str, Any]:
        """
        转成适合写入 TravelState 的 dict。
        """

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()
