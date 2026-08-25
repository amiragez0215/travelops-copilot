from __future__ import annotations

from datetime import date as DateType
from typing import Any, Literal

from pydantic import BaseModel, Field


ProposalGenerationStatus = Literal[
    "generated",
    "failed",
]

ActivityPeriod = Literal[
    "morning",
    "afternoon",
    "evening",
    "full_day",
    "arrival",
    "departure",
]

ActivityType = Literal[
    "transfer",
    "check_in",
    "sightseeing",
    "culture",
    "food",
    "nature",
    "city_walk",
    "leisure",
    "shopping",
    "free_time",
    "other",
]

IndoorOutdoorType = Literal[
    "indoor",
    "outdoor",
    "mixed",
    "flexible",
    "transfer",
]

RequirementHandlingStatus = Literal[
    "included",
    "partially_included",
    "not_supported",
    "needs_confirmation",
    "not_applicable",
]


class ProposalActivityDraft(BaseModel):
    """
    LLM 为某一天生成的一项活动草稿。

    重要边界：
        - LLM 可以生成活动安排和说明；
        - LLM 不能修改航班、酒店、天气和预算锁定事实；
        - evidence_refs 只能引用本轮 EvidenceGrade 已验证的 chunk_id。
    """

    period: ActivityPeriod = Field(
        description="活动时段，例如 morning、afternoon、evening。"
    )

    title: str = Field(
        min_length=1,
        description="给用户展示的活动标题，例如‘博物馆与茶馆体验’。",
    )

    description: str = Field(
        min_length=1,
        description=(
            "活动内容和安排理由。不能编造证据库中不存在的精确营业时间、"
            "票价或实时状态。"
        ),
    )

    activity_type: ActivityType = Field(
        description=(
            "活动类型。transfer/check_in/free_time 等物流活动可以没有 RAG 引用；"
            "景点、美食、文化和自然活动应尽量有证据支持。"
        )
    )

    indoor_outdoor: IndoorOutdoorType = Field(
        description=(
            "活动的室内外属性。天气风险日会根据该字段检查是否做了合理调整。"
        )
    )

    evidence_refs: list[str] = Field(
        default_factory=list,
        description=(
            "支持这项活动的 Evidence chunk_id。"
            "只能使用 Prompt 中提供的 evidence_catalog.ref_id。"
        ),
    )

    tool_activity_id: str | None = Field(
        default=None,
        description=(
            "如果活动来自 search_city_activities，必须逐字复制候选的 activity_id；"
            "如果只来自 RAG 或属于自由时间则保持 null。"
        ),
    )

    practical_notes: list[str] = Field(
        default_factory=list,
        description="非实时的实用提醒，例如预留交通时间、注意防滑。",
    )


class ProposalDayDraft(BaseModel):
    """
    LLM 输出的一天行程草稿。

    date 必须与 TripRequest 的逐日日期完全一致。
    最终 Proposal 会由程序把 WeatherNode 的逐日天气事实注入本对象。
    """

    date: DateType = Field(
        description="该天日期，必须来自系统提供的 expected_dates。"
    )

    theme: str = Field(
        min_length=1,
        description="当天主题，例如‘雨天室内文化与美食’。",
    )

    activities: list[ProposalActivityDraft] = Field(
        min_length=1,
        description="当天至少一项活动。",
    )

    weather_adjustment: str | None = Field(
        default=None,
        description=(
            "当天存在 rain/heavy_rain/wind/heat 等风险时必须填写，"
            "说明如何调整室内外活动、时段或交通缓冲。"
        ),
    )

    day_notes: list[str] = Field(
        default_factory=list,
        description="当天补充说明。",
    )


class EvidenceBackedNoteDraft(BaseModel):
    """
    带证据引用的安全提醒或行李建议。
    """

    text: str = Field(
        min_length=1,
        description="提醒或建议正文。",
    )

    evidence_refs: list[str] = Field(
        default_factory=list,
        description="支持该建议的 Evidence chunk_id。",
    )


class HotelContextDraft(BaseModel):
    """
    LLM 根据 hotel_reviews RAG 生成的选中酒店补充说明。

    这些内容是非实时知识，不能改变酒店排序，也不能覆盖 Mock 结构化事实。
    """

    summary: str = Field(
        min_length=1,
        description="选中酒店的非实时补充说明。",
    )

    nearby_facilities: list[str] = Field(
        default_factory=list,
        description="RAG 文档明确提到的周边餐饮、超市或交通信息。",
    )

    cautions: list[str] = Field(
        default_factory=list,
        description="RAG 文档明确提到的局限或注意事项。",
    )

    evidence_refs: list[str] = Field(
        min_length=1,
        description="必须引用 hotel_reviews 类别的 Evidence chunk_id。",
    )


class RequirementHandlingDraft(BaseModel):
    """
    说明用户指定实体或未映射需求在方案中如何处理。

    requirement_key 由程序生成并提供给 LLM，
    用于避免模型遗漏用户要求或自行改写要求标识。
    """

    requirement_key: str = Field(
        min_length=1,
        description="程序提供的稳定要求标识。",
    )

    status: RequirementHandlingStatus = Field(
        description=(
            "included：已纳入；partially_included：部分满足；"
            "not_supported：当前证据或数据不足；"
            "needs_confirmation：需要用户确认；not_applicable：不适用于当前方案。"
        )
    )

    explanation: str = Field(
        min_length=1,
        description="如何处理该要求的说明。",
    )

    evidence_refs: list[str] = Field(
        default_factory=list,
        description="如果该要求由 RAG 支持，填写对应 Evidence chunk_id。",
    )


class ProposalDraft(BaseModel):
    """
    LLM 允许生成的全部字段。

    特意不包含：
        - 航班 ID、时间和价格；
        - 酒店 ID、名称和价格；
        - 每日天气事实；
        - BudgetOptimize 的费用结果。

    这些锁定事实由程序在 LLM 输出校验通过后确定性注入，
    从数据合同层面减少模型篡改事实的可能性。
    """

    summary: str = Field(
        min_length=1,
        description="旅行方案摘要。",
    )

    highlights: list[str] = Field(
        default_factory=list,
        description="方案主要亮点。",
    )

    daily_plan: list[ProposalDayDraft] = Field(
        min_length=1,
        description="逐日行程草稿。",
    )

    hotel_context: HotelContextDraft | None = Field(
        default=None,
        description=(
            "只有 evidence_pool 中存在选中酒店的 hotel_reviews 证据时才可填写。"
        ),
    )

    packing_tips: list[EvidenceBackedNoteDraft] = Field(
        default_factory=list,
        description="根据 packing_checklists 或安全证据生成的准备建议。",
    )

    safety_notes: list[EvidenceBackedNoteDraft] = Field(
        default_factory=list,
        description="根据天气风险和 safety_notices 生成的安全提醒。",
    )

    preference_alignment: list[str] = Field(
        default_factory=list,
        description="说明方案如何体现用户的主要偏好。",
    )

    requirement_handling: list[RequirementHandlingDraft] = Field(
        default_factory=list,
        description="指定实体和未映射需求的处理结果。",
    )

    limitations: list[str] = Field(
        default_factory=list,
        description="模型识别出的方案限制，不包含系统固定免责声明。",
    )


class TripOverview(BaseModel):
    """最终 Proposal 中的标准化旅行基础事实。"""

    origin: str
    destination: str
    start_date: DateType
    end_date: DateType
    days: int = Field(ge=1)
    nights: int = Field(ge=0)
    people_count: int = Field(ge=1)
    room_count: int = Field(ge=1)


class SelectedFlightFact(BaseModel):
    """
    从 selection_result 复制的锁定航班事实。

    ProposalNode 只做字段整理，不允许 LLM 修改这些值。
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

    duration_minutes: int = Field(default=0, ge=0)
    price: float = Field(ge=0)
    total_price: float = Field(ge=0)

    cabin: str | None = None
    baggage: str | None = None
    is_direct: bool = True
    refundable: bool = False
    changeable: bool = False

    rank: int | None = Field(default=None, ge=1)
    total_score: float | None = Field(default=None, ge=0, le=1)
    ranking_reasons: list[str] = Field(default_factory=list)
    preference_gaps: list[dict[str, Any]] = Field(default_factory=list)


class SelectedFlights(BaseModel):
    """去程和返程两个锁定航班。"""

    outbound: SelectedFlightFact
    return_flight: SelectedFlightFact


class SelectedHotelFact(BaseModel):
    """从 selection_result 复制的锁定酒店事实。"""

    hotel_id: str
    name: str | None = None
    city: str | None = None
    district: str | None = None
    address: str | None = None

    price_per_night: float = Field(ge=0)
    estimated_total_price: float = Field(ge=0)
    planned_nights: int = Field(ge=0)
    planned_room_count: int = Field(ge=1)

    rating: float = Field(ge=0)
    near_subway: bool = False
    distance_to_subway_meters: int = Field(default=999999, ge=0)
    quiet_score: float = Field(default=0.5, ge=0, le=1)
    cleanliness_score: float = Field(default=0.5, ge=0, le=1)

    tags: list[str] = Field(default_factory=list)
    amenities: list[str] = Field(default_factory=list)
    cancel_policy: str | None = None

    rank: int | None = Field(default=None, ge=1)
    total_score: float | None = Field(default=None, ge=0, le=1)
    ranking_reasons: list[str] = Field(default_factory=list)
    preference_gaps: list[dict[str, Any]] = Field(default_factory=list)


class DailyWeatherFact(BaseModel):
    """WeatherNode 为某一天提供的锁定天气事实。"""

    date: DateType
    data_status: Literal["available", "unavailable"]
    condition: str | None = None
    temperature_low: float | None = None
    temperature_high: float | None = None
    humidity: float | None = None
    wind: str | None = None
    precipitation_probability: float | None = None
    risk_types: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class WeatherOverview(BaseModel):
    """整段旅行的天气摘要和逐日天气事实。"""

    status: str
    summary: str
    daily: list[DailyWeatherFact]
    risk_types: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source: str | None = None


class ProposalDay(BaseModel):
    """最终逐日行程；天气由程序注入，活动由 LLM 生成。"""

    day_index: int = Field(ge=1)
    date: DateType
    theme: str
    weather: DailyWeatherFact
    activities: list[ProposalActivityDraft]
    weather_adjustment: str | None = None
    day_notes: list[str] = Field(default_factory=list)


class ProposalBudgetSummary(BaseModel):
    """
    最终方案中的预算摘要。

    只说明 Mock 航班和酒店已知报价，以及剩余预算安排；
    不预测实际餐饮、购物和临时消费。
    """

    budget_mode: Literal[
        "budget_limited",
        "no_budget_limit",
    ]

    total_budget: float | None = Field(default=None, ge=0)
    known_costs: dict[str, Any] = Field(default_factory=dict)
    remaining_budget: float | None = Field(default=None, ge=0)
    remaining_budget_allocation: dict[str, Any] | None = None
    note: str


class FinalHotelContext(BaseModel):
    """最终展示的选中酒店非实时补充知识。"""

    available: bool
    summary: str
    nearby_facilities: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    non_realtime_notice: str


class ProposalSourceRef(BaseModel):
    """最终方案真正使用到的一条证据来源。"""

    chunk_id: str
    category: str
    source: str
    title: str | None = None
    section: str | None = None
    doc_type: str | None = None
    hotel_id: str | None = None
    risk_type: str | list[str] | None = None
    rerank_score: float | None = Field(default=None, ge=0, le=1)


class ProposalActivitySourceRef(BaseModel):
    """最终行程实际采用的一条活动 MCP 结构化来源。"""

    activity_id: str
    name: str
    city: str
    available_dates: list[str] = Field(default_factory=list)
    opening_hours: str | None = None
    source: str = "mcp:search_city_activities"


class ExecutionBoundary(BaseModel):
    """明确声明当前系统没有执行真实外部动作。"""

    real_booking_performed: bool = False
    real_payment_performed: bool = False
    real_notification_sent: bool = False
    draft_only: bool = True


class TripProposal(BaseModel):
    """
    ProposalNode 的最终结构化输出。

    结构分为两部分：
        1. 锁定事实：程序从 Mock / BudgetOptimize / Weather 注入；
        2. 生成内容：LLM 在 RAG 证据和用户偏好范围内生成。
    """

    proposal_id: str
    version: int = Field(default=1, ge=1)
    status: Literal["proposed"] = "proposed"

    schema_version: str = "trip_proposal_v1"
    generator_version: str
    prompt_version: str
    model_id: str
    generated_at: str
    locked_fact_hash: str

    summary: str
    highlights: list[str] = Field(default_factory=list)

    trip_overview: TripOverview
    selected_flights: SelectedFlights
    selected_hotel: SelectedHotelFact
    weather_overview: WeatherOverview
    daily_plan: list[ProposalDay]
    budget_summary: ProposalBudgetSummary

    hotel_context: FinalHotelContext
    packing_tips: list[EvidenceBackedNoteDraft] = Field(default_factory=list)
    safety_notes: list[EvidenceBackedNoteDraft] = Field(default_factory=list)
    preference_alignment: list[str] = Field(default_factory=list)
    requirement_handling: list[RequirementHandlingDraft] = Field(
        default_factory=list
    )

    limitations: list[str] = Field(default_factory=list)
    sources: list[ProposalSourceRef] = Field(default_factory=list)
    activity_sources: list[ProposalActivitySourceRef] = Field(default_factory=list)
    execution_boundary: ExecutionBoundary = Field(
        default_factory=ExecutionBoundary
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class ProposalGenerationResult(BaseModel):
    """ProposalNode 的运行摘要，不保存完整 Prompt。"""

    status: ProposalGenerationStatus
    generator_version: str = "llm_locked_facts_rag_v2"
    prompt_version: str = "proposal_prompt_v2"
    model_id: str

    attempt_count: int = Field(ge=0)
    max_attempts: int = Field(ge=1)

    evidence_input_count: int = Field(ge=0)
    evidence_prompt_count: int = Field(ge=0)
    used_evidence_count: int = Field(ge=0)
    used_evidence_refs: list[str] = Field(default_factory=list)

    locked_fact_hash: str | None = None
    proposal_id: str | None = None
    daily_plan_count: int = Field(ge=0)

    validation_errors: list[str] = Field(default_factory=list)
    issues: list[dict[str, Any]] = Field(default_factory=list)

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()
