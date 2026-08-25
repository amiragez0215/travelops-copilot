from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.preference_schema import PreferenceSignal
from app.schemas.request_constraint_schema import (
    FieldResolution,
    HardConstraint,
    NamedConstraint,
    PartialDate,
    RequirementIssue,
    SpendPreference,
    UnmappedRequirement,
)


def _jsonable(value: Any) -> Any:
    """
    将 date、datetime、Pydantic 对象和嵌套容器转成 JSON 友好结构。

    TravelState、LangGraph Checkpoint、FastAPI Response 和 Eval 文件
    都更适合保存普通 dict/list/字符串，而不是 Python date 对象。
    """

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    if isinstance(value, BaseModel):
        if hasattr(value, "model_dump"):
            return _jsonable(value.model_dump(mode="json"))
        return _jsonable(value.dict())

    if isinstance(value, list):
        return [_jsonable(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }

    return value


class TripRequestDraft(BaseModel):
    """
    LLM Structured Output 的中间 Schema。

    为什么不让 LLM 直接生成完整 TripRequest？
        user_id、raw_message、extraction 等字段由程序掌控，不应该交给模型。
        模型只负责理解用户自然语言并输出旅行需求草稿。

    该 Schema 仍然只是“模型提取结果”。
    后面还需要：
        1. 与规则抽取结果合并。
        2. Pydantic 类型校验。
        3. TripRequestValidator 做日期、能力范围和冲突检查。
    """

    origin: str | None = Field(
        default=None,
        description="出发城市；用户未提供时为 null。",
    )

    destination: str | None = Field(
        default=None,
        description=(
            "目的地城市。可以使用稳定世界知识解析间接表达，"
            "例如‘兵马俑所在城市’解析为‘西安’；有歧义时必须为 null。"
        ),
    )

    start_date: date | None = Field(
        default=None,
        description="明确到具体日期的出发日期；只有月份时必须为 null。",
    )

    end_date: date | None = Field(
        default=None,
        description="明确的结束/返程日期；可由程序根据 start_date + days 推导。",
    )

    partial_date: PartialDate | None = Field(
        default=None,
        description="用户只提供到月份或年份时保存不完整日期信息。",
    )

    days: int | None = Field(
        default=None,
        ge=1,
        le=30,
        description="旅行天数。",
    )

    budget: float | None = Field(
        default=None,
        ge=0,
        description="用户总预算，单位为元。",
    )

    people_count: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="出行人数；用户未说时可以为 null，程序默认 1。",
    )

    room_count: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "需要的房间数量。单人默认 1；多人未说明时保持 null，后续澄清。"
        ),
    )

    preference_signals: list[PreferenceSignal] = Field(
        default_factory=list,
        description="所有主观偏好及其 0—1 强度。",
    )

    spend_preferences: list[SpendPreference] = Field(
        default_factory=list,
        description="用户希望提高或降低某类预算投入的信号。",
    )

    hard_constraints: list[HardConstraint] = Field(
        default_factory=list,
        description=(
            "仅包含客观、可验证、明确不可妥协的条件；"
            "不能把‘一定要安静’这类主观表达放进这里。用户总预算只能写入"
            "顶层 budget，不能额外生成 trip.budget 或 trip.total_budget。"
        ),
    )

    named_constraints: list[NamedConstraint] = Field(
        default_factory=list,
        description="用户明确指定的酒店、航班、食物、景点或活动。",
    )

    unmapped_requirements: list[UnmappedRequirement] = Field(
        default_factory=list,
        description="模型理解了、但当前固定字段无法承载的要求。",
    )

    requirement_issues: list[RequirementIssue] = Field(
        default_factory=list,
        description="歧义、冲突或必须澄清的问题。",
    )

    field_resolution: dict[str, FieldResolution] = Field(
        default_factory=dict,
        description="核心字段如何得到，用于 Trace、Eval 和澄清。",
    )

    extraction_confidence: float = Field(
        default=0.75,
        ge=0,
        le=1,
        description="LLM 对整体抽取结果的自评信心，只用于调试与 Eval。",
    )


class TripRequest(BaseModel):
    """
    InputExtractNode 输出的完整结构化旅行请求。

    该对象只描述“用户想要什么”，不包含天气、航班、酒店查询结果。

    核心设计：
        - 核心字段是固定 Schema。
        - 主观需求进入 preference_signals。
        - 客观不可妥协条件进入 hard_constraints。
        - 指定实体进入 named_constraints。
        - 无法映射的开放需求进入 unmapped_requirements。
        - 歧义和冲突进入 requirement_issues。

    这样可以实现：
        Open-world Capture（尽量捕获开放自然语言）
        +
        Closed-world Execution（只执行项目明确支持的能力）。
    """

    user_id: str | None = Field(
        default=None,
        description="当前用户 ID；匿名用户可以为空。",
    )

    origin: str | None = Field(
        default=None,
        description="出发城市。",
    )

    destination: str | None = Field(
        default=None,
        description="目的地城市。",
    )

    start_date: date | None = Field(
        default=None,
        description="出发日期；信息不完整时为 None。",
    )

    end_date: date | None = Field(
        default=None,
        description="结束/返程日期；可以由 start_date + days - 1 推导。",
    )

    partial_date: PartialDate | None = Field(
        default=None,
        description="只说到月份或年份时保存的不完整日期。",
    )

    days: int | None = Field(
        default=None,
        ge=1,
        le=30,
        description="旅行天数，当前限制 1—30 天。",
    )

    budget: float | None = Field(
        default=None,
        ge=0,
        description="总预算，单位元；不是必填字段。",
    )

    people_count: int = Field(
        default=1,
        ge=1,
        le=20,
        description="出行人数；用户未说明时默认 1。",
    )

    room_count: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "房间数量。单人默认 1；多人未说明房间数时由 Validator 要求澄清。"
        ),
    )

    # ------------------------------------------------------------------
    # 兼容性的轻量偏好索引
    # ------------------------------------------------------------------
    # 这些 list 便于日志和早期测试阅读。
    # 新代码的权威偏好来源仍然是 preference_signals。
    transport_preferences: list[str] = Field(
        default_factory=list,
        description="航班偏好 key 的轻量索引。",
    )

    hotel_preferences: list[str] = Field(
        default_factory=list,
        description="酒店偏好 key 的轻量索引。",
    )

    travel_style: list[str] = Field(
        default_factory=list,
        description="活动与节奏偏好 key 的轻量索引。",
    )

    raw_constraints: list[str] = Field(
        default_factory=list,
        description="从用户原话中保留的约束和偏好证据摘要。",
    )

    preference_signals: list[PreferenceSignal] = Field(
        default_factory=list,
        description="主观偏好及连续权重，后续用于 CandidateRank 评分。",
    )

    spend_preferences: list[SpendPreference] = Field(
        default_factory=list,
        description="动态预算分配信号。",
    )

    hard_constraints: list[HardConstraint] = Field(
        default_factory=list,
        description="CandidateRank/BudgetOptimize 可确定性验证的硬约束。",
    )

    named_constraints: list[NamedConstraint] = Field(
        default_factory=list,
        description="用户直接指定的酒店、航班、食物、景点或活动。",
    )

    unmapped_requirements: list[UnmappedRequirement] = Field(
        default_factory=list,
        description="当前固定字段无法承载、但不能静默丢弃的要求。",
    )

    requirement_issues: list[RequirementIssue] = Field(
        default_factory=list,
        description="抽取阶段发现的歧义、冲突或待澄清问题。",
    )

    field_resolution: dict[str, FieldResolution] = Field(
        default_factory=dict,
        description="核心字段的解析来源。",
    )

    raw_message: str = Field(
        default="",
        description="用户原始输入。",
    )

    extraction: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "抽取运行信息，例如 method、llm_used、fallback_used、confidence、notes。"
        ),
    )

    def required_missing_fields(self) -> list[str]:
        """
        返回 PlanTripWorkflow 最小核心缺失字段。

        注意：
            多人未给 room_count、歧义、能力范围等问题由 Validator 额外检查，
            不全部塞进这个简单方法。
        """

        missing: list[str] = []

        if not self.origin:
            missing.append("origin")

        if not self.destination:
            missing.append("destination")

        if not self.start_date:
            missing.append("start_date")

        if not self.days:
            missing.append("days")

        return missing

    def to_state_dict(self) -> dict[str, Any]:
        """
        转成可以安全写入 TravelState 的普通 dict。
        """

        if hasattr(self, "model_dump"):
            data = self.model_dump(mode="json")
        else:
            data = self.dict()

        return _jsonable(data)
