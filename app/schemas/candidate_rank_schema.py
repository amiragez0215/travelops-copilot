from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


CandidateRankStatus = Literal[
    "ok",
    "partial",
    "no_candidates",
    "failed",
]

CandidateType = Literal["flight", "hotel"]
FlightDirection = Literal["outbound", "return"]


class PreferenceGap(BaseModel):
    """
    一个高权重偏好在当前候选上没有得到充分满足的说明。

    CandidateRankNode 不会因为主观偏好分数较低就直接过滤候选。
    它会把这类差距记录下来，后续 ProposalNode 可以透明告诉用户：

        “当前没有完全符合安静要求的酒店，已选择候选中相对更安静的一家。”
    """

    preference_key: str = Field(
        description="标准偏好 key，例如 quiet、avoid_early_flight。"
    )

    preference_label: str = Field(
        description="给用户阅读的中文标签，例如 安静、避免早班机。"
    )

    user_weight: float = Field(
        ge=0,
        le=1,
        description="用户对该偏好的权重。",
    )

    candidate_feature_score: float = Field(
        ge=0,
        le=1,
        description="当前候选在该特征上的标准化得分。",
    )

    message: str = Field(
        description="偏好没有充分满足时的解释。"
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class RejectedCandidate(BaseModel):
    """
    被 CandidateRankNode 拒绝的候选。

    只有下面两类原因可以拒绝：

        1. 防御性数据检查失败，例如房间数不足。
        2. 不满足客观、可验证的真正硬约束。

    主观偏好分数低不会进入 rejected_candidates。
    """

    candidate_type: CandidateType = Field(
        description="被拒绝的是 flight 还是 hotel。"
    )

    candidate_id: str = Field(
        description="flight_id 或 hotel_id。"
    )

    direction: FlightDirection | None = Field(
        default=None,
        description="航班方向；酒店为 None。",
    )

    reason_code: str = Field(
        description="机器可读的拒绝原因代码。"
    )

    message: str = Field(
        description="给开发者和用户阅读的拒绝说明。"
    )

    constraint: dict[str, Any] | None = Field(
        default=None,
        description="触发拒绝的硬约束；非硬约束问题时为 None。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class RankedFlightCandidate(BaseModel):
    """
    经过硬约束过滤和确定性评分后的航班候选。

    该 Schema 保留 Mock Provider 返回的结构化事实，
    再增加 rank、total_score 和可解释评分信息。
    """

    flight_id: str
    flight_no: str | None = None
    airline: str | None = None

    departure_city: str | None = None
    arrival_city: str | None = None
    departure_airport: str | None = None
    arrival_airport: str | None = None

    depart_date: str | None = None
    depart_time: str | None = None
    arrive_date: str | None = None
    arrive_time: str | None = None

    duration_minutes: int = 0
    price: float = Field(
        ge=0,
        description="Mock Provider 返回的单人票价。",
    )
    total_price: float = Field(
        ge=0,
        description="price × people_count，用于后续组合预算。",
    )

    available_seats: int = 0
    cabin: str | None = None
    baggage: str | None = None
    is_direct: bool = True
    refundable: bool = False
    changeable: bool = False
    updated_at: str | None = None
    source: str | None = None

    query_direction: FlightDirection
    direction: FlightDirection

    rank: int = Field(
        ge=1,
        description="在 outbound 或 return 组内的最终排名。",
    )

    total_score: float = Field(
        ge=0,
        le=1,
        description="航班综合得分，越高越符合当前用户。",
    )

    score_breakdown: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "评分拆分，包括 preference_match、price_fit、flexibility、"
            "time_fit、direct_fit 和 component_weights。"
        ),
    )

    ranking_reasons: list[str] = Field(
        default_factory=list,
        description="该航班排名靠前的主要原因。",
    )

    preference_gaps: list[PreferenceGap] = Field(
        default_factory=list,
        description="高权重偏好没有充分满足的项目。",
    )

    hard_constraints_checked: list[dict[str, Any]] = Field(
        default_factory=list,
        description="该候选已经通过的航班硬约束。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class RankedHotelCandidate(BaseModel):
    """
    经过硬约束过滤和确定性评分后的酒店候选。

    酒店总分只由三部分组成：

        偏好匹配分
        + 价格适配分
        + 基础评分

    RAG 文档和 rerank_score 不参与这里的数值评分。
    """

    hotel_id: str
    name: str | None = None
    city: str | None = None
    district: str | None = None
    address: str | None = None

    price_per_night: float = Field(
        ge=0,
        description="每间每晚的 Mock 价格。",
    )

    estimated_total_price: float = Field(
        ge=0,
        description="price_per_night × nights × room_count。",
    )

    planned_nights: int = Field(
        ge=0,
        description="本次行程住宿晚数。",
    )

    planned_room_count: int = Field(
        ge=1,
        description="本次行程需要的房间数。",
    )

    rating: float = Field(
        ge=0,
        description="Mock Provider 返回的酒店基础评分，通常为 0—5。",
    )

    available_rooms: int = 0
    near_subway: bool = False
    distance_to_subway_meters: int = 999999
    quiet_score: float = Field(default=0.5, ge=0, le=1)
    cleanliness_score: float = Field(default=0.5, ge=0, le=1)
    tags: list[str] = Field(default_factory=list)
    amenities: list[str] = Field(default_factory=list)
    cancel_policy: str | None = None
    updated_at: str | None = None
    source: str | None = None

    rank: int = Field(
        ge=1,
        description="酒店候选最终排名。",
    )

    total_score: float = Field(
        ge=0,
        le=1,
        description="酒店综合得分，越高越符合当前用户。",
    )

    score_breakdown: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "评分拆分，包括 preference_match、price_fit、base_rating、"
            "feature_scores 和 component_weights。"
        ),
    )

    ranking_reasons: list[str] = Field(
        default_factory=list,
        description="该酒店排名靠前的主要原因。",
    )

    preference_gaps: list[PreferenceGap] = Field(
        default_factory=list,
        description="高权重偏好没有充分满足的项目。",
    )

    hard_constraints_checked: list[dict[str, Any]] = Field(
        default_factory=list,
        description="该候选已经通过的酒店硬约束。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class CandidateRankResult(BaseModel):
    """
    CandidateRankNode 的总体执行摘要。

    status：

        ok
            出发航班、返程航班和酒店都有可用排序结果。

        partial
            至少有一类候选，但某个必要类别为空。
            例如酒店有候选，但返程航班全部被硬约束过滤。

        no_candidates
            三类候选都为空。

        failed
            程序异常，不是正常业务无候选。
    """

    status: CandidateRankStatus

    strategy_version: str = Field(
        default="deterministic_candidate_rank_v1",
        description="候选排序策略版本。",
    )

    input_counts: dict[str, int] = Field(
        default_factory=dict,
        description="进入 CandidateRankNode 前的候选数量。",
    )

    output_counts: dict[str, int] = Field(
        default_factory=dict,
        description="硬过滤和评分后的候选数量。",
    )

    rejected_flights: list[RejectedCandidate] = Field(
        default_factory=list,
        description="被真正硬约束或防御性检查拒绝的航班。",
    )

    rejected_hotels: list[RejectedCandidate] = Field(
        default_factory=list,
        description="被真正硬约束或防御性检查拒绝的酒店。",
    )

    ignored_preference_keys: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "PlanningContext 中存在、但当前 CandidateRank v1 没有结构化特征可计算的偏好。"
        ),
    )

    ranking_config: dict[str, Any] = Field(
        default_factory=dict,
        description="本轮实际使用的评分参数，便于 Trace 和 Eval。",
    )

    issues: list[dict[str, Any]] = Field(
        default_factory=list,
        description="正常业务问题，例如没有返程航班候选。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()
