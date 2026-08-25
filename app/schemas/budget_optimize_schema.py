from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


BudgetOptimizeStatus = Literal[
    "feasible",
    "no_feasible_combination",
    "no_candidates",
    "failed",
]

BudgetCheckStatus = Literal[
    "within_budget",
    "no_budget_limit",
    "no_feasible_combination",
    "no_candidates",
    "failed",
]

AdjustmentStatus = Literal[
    "not_needed",
    "need_user_change",
    "missing_candidates",
    "failed",
]


class KnownCostBreakdown(BaseModel):
    """
    由 Mock 候选能够确定的结构化费用。

    这里故意只保存：
        - 去程航班；
        - 返程航班；
        - 酒店。

    不包含：
        - 用户实际吃了什么；
        - 临时购物；
        - 实际打车次数；
        - 真实景点消费。

    因此 known_subtotal 不是“整趟旅行真实花费预测”，
    而是当前选中 Mock 航班和酒店的已知模拟报价合计。
    """

    outbound_flight: float = Field(
        ge=0,
        description="去程航班总价，已经乘以 people_count。",
    )

    return_flight: float = Field(
        ge=0,
        description="返程航班总价，已经乘以 people_count。",
    )

    transport_total: float = Field(
        ge=0,
        description="去程航班 + 返程航班。",
    )

    hotel: float = Field(
        ge=0,
        description="酒店每晚价格 × nights × room_count。",
    )

    known_subtotal: float = Field(
        ge=0,
        description="当前可确定的 Mock 航班与酒店费用合计。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class RemainingBudgetAllocation(BaseModel):
    """
    从总预算扣除 Mock 航班和酒店报价后的预算安排。

    这是一种预算规划，而不是对真实消费的预测。

    分配规则：
        1. 优先为 food_activity 保留 PlanningContext 中的软目标；
        2. 剩余金额归入 local_transport_and_buffer；
        3. 如果余额不足以达到餐饮活动软目标，记录 shortfall。
    """

    food_activity: float | None = Field(
        default=None,
        ge=0,
        description="建议为餐饮和活动保留的预算。",
    )

    local_transport_and_buffer: float | None = Field(
        default=None,
        ge=0,
        description="餐饮活动预算之外的市内交通和机动缓冲。",
    )

    food_activity_target: float | None = Field(
        default=None,
        ge=0,
        description="PlanningContext 计算出的餐饮活动软预算目标。",
    )

    food_activity_shortfall: float | None = Field(
        default=None,
        ge=0,
        description="当前余额距离餐饮活动软目标还差多少；不代表真实消费缺口。",
    )

    allocation_basis: Literal[
        "soft_target_then_buffer",
        "no_budget_limit",
    ] = Field(
        description="本次剩余预算采用的安排策略。"
    )

    note: str = Field(
        description="向后续 Proposal 和用户说明预算语义。"
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class BudgetCombinationOption(BaseModel):
    """
    一个完整的“去程航班 + 返程航班 + 酒店”组合。

    CandidateRankNode 只评价单个候选；
    BudgetOptimizeNode 才会把三个候选放在一起判断全局可行性。
    """

    combination_id: str = Field(
        description="由三个候选 ID 生成的稳定组合 ID。"
    )

    outbound_flight_id: str
    return_flight_id: str
    hotel_id: str

    outbound_rank: int = Field(
        ge=1,
        description="去程航班在 CandidateRankNode 中的组内排名。",
    )

    return_rank: int = Field(
        ge=1,
        description="返程航班在 CandidateRankNode 中的组内排名。",
    )

    hotel_rank: int = Field(
        ge=1,
        description="酒店在 CandidateRankNode 中的排名。",
    )

    combination_score: float = Field(
        ge=0,
        le=1,
        description="组合综合效用分，越高越适合当前用户。",
    )

    score_breakdown: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "组合得分拆分，包括 transport_quality、hotel_quality、"
            "food_activity_reserve_fit 和 component_weights。"
        ),
    )

    selected_costs: KnownCostBreakdown

    total_budget: float | None = Field(
        default=None,
        ge=0,
        description="用户总预算；未提供预算时为 None。",
    )

    remaining_budget: float | None = Field(
        default=None,
        ge=0,
        description="总预算减去 known_subtotal；无总预算时为 None。",
    )

    remaining_budget_allocation: RemainingBudgetAllocation

    soft_target_deviation: dict[str, float | None] = Field(
        default_factory=dict,
        description=(
            "实际组合金额与交通、酒店、餐饮活动软目标之间的差值。"
            "正数通常表示高于软目标，负数表示低于软目标。"
        ),
    )

    selection_reasons: list[str] = Field(
        default_factory=list,
        description="该组合排名靠前的主要原因。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class SelectionResult(BaseModel):
    """
    BudgetOptimizeNode 的权威选择结果。

    当 status=feasible 时，selected_* 字段保存最终锁定事实。
    后续 RetrievalPlanNode 和 ProposalNode 只能读取这些结构化结果，
    不能让 LLM 自行修改航班、酒店、价格或时间。
    """

    status: BudgetOptimizeStatus

    strategy_version: str = Field(
        default="deterministic_budget_optimize_v1",
        description="组合优化策略版本。",
    )

    budget_mode: Literal[
        "budget_limited",
        "no_budget_limit",
    ]

    selected_combination: BudgetCombinationOption | None = None

    selected_outbound_flight: dict[str, Any] = Field(
        default_factory=dict,
        description="完整的选中去程航班候选。",
    )

    selected_return_flight: dict[str, Any] = Field(
        default_factory=dict,
        description="完整的选中返程航班候选。",
    )

    selected_hotel: dict[str, Any] = Field(
        default_factory=dict,
        description="完整的选中酒店候选。",
    )

    selected_hotel_id: str | None = Field(
        default=None,
        description="给 RetrievalPlanNode 快速读取的酒店 ID。",
    )

    selected_costs: KnownCostBreakdown | None = None

    total_budget: float | None = Field(
        default=None,
        ge=0,
    )

    remaining_budget: float | None = Field(
        default=None,
        ge=0,
    )

    remaining_budget_allocation: RemainingBudgetAllocation | None = None

    alternatives: list[BudgetCombinationOption] = Field(
        default_factory=list,
        description="除最终选择外，得分最高的若干可行备选组合。",
    )

    notes: list[str] = Field(
        default_factory=list,
        description="选择边界和预算语义说明。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class BudgetCheckResult(BaseModel):
    """给后续节点读取的精简预算检查结果。"""

    status: BudgetCheckStatus

    budget_mode: Literal[
        "budget_limited",
        "no_budget_limit",
    ]

    total_budget: float | None = Field(
        default=None,
        ge=0,
    )

    selected_known_subtotal: float | None = Field(
        default=None,
        ge=0,
    )

    remaining_budget: float | None = Field(
        default=None,
        ge=0,
    )

    known_cost_note: str = Field(
        description=(
            "说明这里只计算 Mock 航班和酒店报价，"
            "不预测真实餐饮、活动和临时消费。"
        )
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class BudgetAdjustmentPlan(BaseModel):
    """
    没有可行组合时生成的确定性调整建议。

    它不会自动放松用户硬约束，也不会擅自提高预算；
    只把可解释的差距交给用户决定。
    """

    status: AdjustmentStatus

    reason_code: str | None = None

    current_total_budget: float | None = Field(
        default=None,
        ge=0,
    )

    minimum_known_subtotal: float | None = Field(
        default=None,
        ge=0,
        description="当前候选中最便宜的航班与酒店已知报价合计。",
    )

    required_budget_increase: float | None = Field(
        default=None,
        ge=0,
        description="若仅提高总预算，至少需要增加的金额。",
    )

    cheapest_combination: dict[str, Any] = Field(
        default_factory=dict,
        description="最便宜组合的 ID 和已知费用摘要。",
    )

    suggestions: list[str] = Field(
        default_factory=list,
        description="需要用户确认的调整方向。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class BudgetOptimizeResult(BaseModel):
    """BudgetOptimizeNode 的总体执行摘要。"""

    status: BudgetOptimizeStatus

    strategy_version: str = Field(
        default="deterministic_budget_optimize_v1",
    )

    budget_mode: Literal[
        "budget_limited",
        "no_budget_limit",
    ]

    input_counts: dict[str, int] = Field(
        default_factory=dict,
        description="输入的去程、返程和酒店候选数量。",
    )

    candidate_limits: dict[str, int] = Field(
        default_factory=dict,
        description="为控制组合数量，每组最多参与优化的候选数量。",
    )

    evaluated_combination_count: int = Field(
        ge=0,
        description="实际枚举并检查的组合数量。",
    )

    feasible_combination_count: int = Field(
        ge=0,
        description="满足所有组合级硬限制的组合数量。",
    )

    rejected_by_total_budget_count: int = Field(
        ge=0,
        description="因为 known_subtotal 超过总预算而被拒绝的组合数量。",
    )

    rejected_by_transport_limit_count: int = Field(
        ge=0,
        description="因为往返交通合计超过硬上限而被拒绝的组合数量。",
    )

    selected_combination_id: str | None = None

    selected_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    minimum_known_subtotal: float | None = Field(
        default=None,
        ge=0,
    )

    scoring_config: dict[str, Any] = Field(
        default_factory=dict,
        description="本轮组合评分参数，便于 Trace 和 Eval。",
    )

    issues: list[dict[str, Any]] = Field(
        default_factory=list,
        description="正常业务问题，例如所有组合都超过总预算。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()
