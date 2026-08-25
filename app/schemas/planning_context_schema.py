from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BudgetPlan(BaseModel):
    """
    PlanningContextNode 生成的预算安排策略。

    这里的数值不是“旅行真实花费预测”，而是：
        - 总预算硬上限；
        - 航班、住宿、餐饮活动的软预算目标；
        - 后续 CandidateRank 和 BudgetOptimize 使用的确定性配置。

    关键边界：
        hard_limits：可以用于硬过滤或组合可行性判断。
        soft_targets：只能用于价格适配评分，不能直接过滤候选。
    """

    mode: Literal["budget_limited", "no_budget_limit"] = Field(
        description="用户是否提供了本次旅行总预算。"
    )

    total_budget: float | None = Field(
        default=None,
        ge=0,
        description=(
            "用户给出的总预算。存在时由 BudgetOptimizeNode 作为组合硬上限；"
            "它不表示系统能够准确预测全部真实消费。"
        ),
    )

    hard_limits: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "预算相关的客观硬限制，例如 total_budget、"
            "max_hotel_price_per_night、max_transport_total。"
        ),
    )

    base_ratios: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "没有消费倾向时使用的默认启发式比例。"
            "当前为 transport=0.35、hotel=0.35、food_activity=0.30。"
        ),
    )

    adjusted_ratios: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "根据 spend_preferences 确定性调整并归一化后的比例。"
            "这些比例仍然只是软目标。"
        ),
    )

    soft_targets: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "按 adjusted_ratios 计算出的软预算目标，例如 transport_budget、"
            "hotel_budget、food_activity_budget 和 target_hotel_price_per_room_night。"
        ),
    )

    flexibility: dict[str, bool] = Field(
        default_factory=dict,
        description=(
            "预算跨类别调剂策略。某一类别可以超过软目标，"
            "只要最终组合不违反真正硬限制。"
        ),
    )

    allocation_source: Literal[
        "default_heuristic_v1",
        "user_adjusted_heuristic_v1",
        "no_budget",
    ] = Field(
        description="当前预算比例来自默认启发式还是用户消费倾向调整。"
    )

    allocation_reasons: list[str] = Field(
        default_factory=list,
        description="为什么提高或降低某类预算比例的可解释原因。",
    )

    nights: int = Field(
        ge=0,
        description="住宿晚数，一般等于 days - 1。",
    )

    people_count: int = Field(
        ge=1,
        description="出行人数。",
    )

    room_count: int = Field(
        ge=1,
        description=(
            "酒店房间数量。默认单人旅行时为 1；多人未说明房间数时，"
            "前置 Validator 会要求澄清。"
        ),
    )

    notes: list[str] = Field(
        default_factory=list,
        description="预算语义和使用边界说明。",
    )


class PlanningContext(BaseModel):
    """
    后续规划节点统一读取的结构化上下文。

    PlanningContextNode 的核心技术不是复杂算法，而是 Context Engineering：
        1. 将当前请求和长期记忆合并成统一偏好权重；
        2. 将主观偏好与客观硬约束彻底分离；
        3. 将消费倾向转换成可测试的预算软目标；
        4. 为 Weather、MCP 查询、CandidateRank、BudgetOptimize、RAG 和
           Proposal 提供同一份数据合同。
    """

    request: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "本次旅行的标准化基础信息，例如出发地、目的地、日期、天数、"
            "晚数、人数、房间数和总预算。"
        ),
    )

    hard_constraints: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict,
        description=(
            "按 trip、flight、hotel 分组的客观可验证硬约束。"
            "主观的安静、干净、方便等偏好不会进入这里。"
        ),
    )

    preference_weights: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "长期偏好和本次偏好合并后的 0—1 连续权重。"
            "CandidateRankNode 使用这些权重计算偏好匹配分。"
        ),
    )

    spend_preferences: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "用户对交通、酒店、餐饮活动的消费倾向。"
            "它们用于调整预算软目标，不直接成为硬限制。"
        ),
    )

    named_constraints: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "用户指定的酒店、航班、食物、景点或活动。"
            "酒店/航班最终还要在 Mock 数据中解析到真实候选。"
        ),
    )

    unmapped_requirements: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "系统已理解但当前没有确定性字段承载的开放要求。"
            "只影响表达风格的内容可给 Proposal 参考，核心未支持要求不能静默丢弃。"
        ),
    )

    diet_preferences: dict[str, Any] = Field(
        default_factory=dict,
        description="饮食偏好，例如辣度、喜欢的食物和忌口。",
    )

    budget_plan: BudgetPlan = Field(
        description="动态预算安排结果。",
    )

    context_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="给 Trace、Prompt 和调试使用的短摘要。",
    )

    source_meta: dict[str, Any] = Field(
        default_factory=dict,
        description="PlanningContext 构造策略和数据来源版本。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()
