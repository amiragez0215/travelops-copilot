from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


ConstraintDomain = Literal["trip", "flight", "hotel"]
ConstraintOperator = Literal[
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "in",
    "not_in",
]
ConstraintSource = Literal["rule", "llm", "derived"]

SpendCategory = Literal["transport", "hotel", "food_activity"]
SpendDirection = Literal["increase", "decrease"]

NamedEntityType = Literal[
    "hotel",
    "flight",
    "food",
    "attraction",
    "activity",
]
NamedConstraintMode = Literal["required", "preferred", "avoid"]

ResolutionType = Literal[
    "explicit",
    "derived",
    "stable_world_knowledge",
    "ambiguous",
    "incomplete",
    "unsupported",
    "default",
]

RequirementImpact = Literal[
    "trip",
    "flight",
    "hotel",
    "budget",
    "itinerary",
    "proposal_style",
    "unknown",
]


# ----------------------------------------------------------------------
# HardConstraint 字段规范化
# ----------------------------------------------------------------------
# 用户请求层使用 TripRequest.budget；但组合筛选 / BudgetOptimize 层只认
# trip.total_budget。LLM 可能基于前者自然输出 trip.budget，因此必须在
# 结构化数据进入任意业务 Node 前统一为后者，避免让字段别名变成澄清问题。
#
# 这里故意按 (domain, field) 匹配，而不是把所有 budget 都改名：只有 trip
# 域的 budget 是“全程总预算”别名；未来出现其他 domain 时不会被误改写。
HARD_CONSTRAINT_FIELD_ALIASES: dict[tuple[str, str], str] = {
    ("trip", "budget"): "total_budget",
}


def canonicalize_hard_constraint_field(
    *,
    domain: str,
    field: str,
) -> str:
    """
    返回 HardConstraint.field 的唯一规范名称。

    这不是“放宽 Validator 白名单”。它只把已知、语义等价的历史/LLM
    别名改写为项目唯一字段；未知字段仍会由 Validator 明确拒绝，保证
    Closed-world Execution 的边界不被绕过。
    """

    normalized_field = str(field).strip()

    return HARD_CONSTRAINT_FIELD_ALIASES.get(
        (str(domain).strip(), normalized_field),
        normalized_field,
    )


class HardConstraint(BaseModel):
    """
    可由程序客观验证、且不满足时不能继续推荐该候选的硬约束。

    设计原则：
        1. 条件必须能够用 Mock 中的结构化字段验证。
        2. 条件必须有明确的布尔、数值、时间或实体含义。
        3. 主观偏好不能仅因为出现“一定、必须”就进入这里。

    合理示例：
        {
            "domain": "flight",
            "field": "is_direct",
            "operator": "==",
            "value": true,
            "evidence": "必须直飞",
            "source": "llm"
        }

        {
            "domain": "hotel",
            "field": "price_per_night",
            "operator": "<=",
            "value": 600,
            "evidence": "酒店每晚不能超过600元",
            "source": "rule"
        }
    """

    domain: ConstraintDomain = Field(
        description="硬约束作用域：trip、flight 或 hotel。"
    )

    field: str = Field(
        min_length=1,
        description=(
            "作用域内部的规范字段名。例如 flight 使用 is_direct、depart_time、"
            "arrive_time、price；hotel 使用 price_per_night、hotel_id、near_subway；"
            "trip 只使用 total_budget。用户总预算应优先写入顶层 budget 字段，"
            "程序会确定性地生成 trip.total_budget，不能输出 trip.budget。"
        ),
    )

    operator: ConstraintOperator = Field(
        description="比较操作符，例如 ==、<=、>=。"
    )

    value: Any = Field(
        description="约束值，例如 true、600、08:00、CA1234。"
    )

    evidence: str = Field(
        default="",
        description="用户原话证据。",
    )

    source: ConstraintSource = Field(
        default="llm",
        description="约束来自规则、LLM 或确定性推导。",
    )

    @model_validator(mode="after")
    def normalize_field_alias(self) -> "HardConstraint":
        """
        在 Pydantic 完成 domain / field 类型校验后规范化已知字段别名。

        放在 Schema 而不是某个 Extractor 中，能覆盖 LLM JSON、规则抽取、
        Hybrid 合并、Revision 和测试/工具直接构造的 HardConstraint；所有
        下游组件只需要处理 total_budget 这一种预算硬约束名称。
        """

        self.field = canonicalize_hard_constraint_field(
            domain=self.domain,
            field=self.field,
        )
        return self

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class SpendPreference(BaseModel):
    """
    用户希望把更多或更少预算放在哪一类消费上的信号。

    它不直接给出最终比例。
    PlanningContextBuilder 会从默认 35/35/30 出发，
    根据这些信号使用确定性公式调整并重新归一化。

    示例：
        {
            "category": "hotel",
            "direction": "increase",
            "strength": 0.9,
            "evidence": "住的地方好一点",
            "source": "llm"
        }
    """

    category: SpendCategory = Field(
        description="预算类别：交通、酒店、餐饮活动。"
    )

    direction: SpendDirection = Field(
        description="increase 表示提高预算倾向，decrease 表示降低。"
    )

    strength: float = Field(
        ge=0,
        le=1,
        description="消费倾向强度，范围 0 到 1。",
    )

    evidence: str = Field(
        default="",
        description="用户原话证据。",
    )

    source: ConstraintSource = Field(
        default="llm",
        description="信号来自规则、LLM 或确定性推导。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class NamedConstraint(BaseModel):
    """
    用户直接指定的酒店、航班、食物、景点或活动。

    例子：
        - “我只想住成都青羊静巷酒店”
        - “我想吃鸭血粉丝汤”
        - “我一定要去兵马俑”

    边界：
        酒店和航班必须在 Mock 候选中存在，才能进入结构化选择。
        食物、景点和活动必须有 RAG 证据，才能进入最终行程。
    """

    entity_type: NamedEntityType = Field(
        description="实体类型，例如 hotel、food、attraction。"
    )

    entity_name: str = Field(
        min_length=1,
        description="用户指定的实体名称。",
    )

    constraint_mode: NamedConstraintMode = Field(
        default="preferred",
        description=(
            "required 表示用户明确只接受该实体；"
            "preferred 表示优先考虑；avoid 表示希望避开。"
        ),
    )

    scope: str = Field(
        default="whole_trip",
        description="约束作用范围，例如 whole_trip、itinerary。",
    )

    traveler: str | None = Field(
        default=None,
        description=(
            "提出该要求的旅行者名称。当前 v1 只支持共享偏好；"
            "非空且出现多人冲突时需要 Clarify。"
        ),
    )

    evidence: str = Field(
        default="",
        description="用户原话证据。",
    )

    source: ConstraintSource = Field(
        default="llm",
        description="实体约束来自规则、LLM 或推导。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class UnmappedRequirement(BaseModel):
    """
    LLM 已经理解，但当前项目没有确定性字段或执行能力承载的要求。

    这类内容不能静默丢弃。
    如果只影响表达风格，可以给 ProposalNode 参考；
    如果会影响核心选择，则 Validator 应要求澄清或说明不支持。
    """

    text: str = Field(
        min_length=1,
        description="用户原始要求。",
    )

    category_guess: str = Field(
        default="unknown",
        description="LLM 对要求类型的粗略判断。",
    )

    impact: RequirementImpact = Field(
        default="unknown",
        description="该要求可能影响哪个业务部分。",
    )

    confidence: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="LLM 对分类结果的信心。",
    )

    requires_clarification: bool = Field(
        default=False,
        description="该未映射要求是否会阻止工作流继续。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class RequirementIssue(BaseModel):
    """
    抽取阶段发现的歧义、冲突或当前能力不支持问题。

    示例：
        {
            "issue_type": "ambiguous_destination",
            "message": "‘迪士尼所在城市’存在多个可能目的地。",
            "evidence": ["迪士尼所在的城市"],
            "candidate_values": ["上海", "香港", "东京"],
            "requires_clarification": true
        }
    """

    issue_type: str = Field(
        min_length=1,
        description="问题类型，例如 ambiguous_destination。",
    )

    message: str = Field(
        min_length=1,
        description="给用户或开发者阅读的问题说明。",
    )

    evidence: list[str] = Field(
        default_factory=list,
        description="触发问题的用户原话片段。",
    )

    candidate_values: list[str] = Field(
        default_factory=list,
        description="存在歧义时的候选值。",
    )

    requires_clarification: bool = Field(
        default=True,
        description="是否必须在继续规划前向用户澄清。",
    )

    suggested_question: str = Field(
        default="",
        description="建议 ClarifyNode 向用户提出的问题。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class FieldResolution(BaseModel):
    """
    记录核心字段是如何得到的，主要用于 Trace、Eval 和澄清。

    例如：
        “秦始皇兵马俑所在的城市” → 西安
        resolution_type = stable_world_knowledge
    """

    value: Any = Field(
        default=None,
        description="最终解析值。",
    )

    resolution_type: ResolutionType = Field(
        description="字段来自明确表达、推导、稳定常识、歧义或默认值。"
    )

    evidence: str = Field(
        default="",
        description="用户原话证据。",
    )

    needs_clarification: bool = Field(
        default=False,
        description="该字段是否仍需用户确认。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()


class PartialDate(BaseModel):
    """
    用户只提供了不完整日期时的结构化结果。

    示例：
        “7月出发”只知道 month=7，不能擅自补 day。
    """

    year: int | None = Field(
        default=None,
        ge=1900,
        le=2200,
        description="年份；用户未说且无法可靠确定时为 None。",
    )

    month: int | None = Field(
        default=None,
        ge=1,
        le=12,
        description="月份。",
    )

    day: int | None = Field(
        default=None,
        ge=1,
        le=31,
        description="日期；只说月份时为 None。",
    )

    precision: Literal["year", "month", "day"] = Field(
        description="当前日期信息精确到年、月或日。"
    )

    evidence: str = Field(
        default="",
        description="用户原话证据。",
    )

    def to_state_dict(self) -> dict[str, Any]:
        """转成适合写入 TravelState 的普通 dict。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()
