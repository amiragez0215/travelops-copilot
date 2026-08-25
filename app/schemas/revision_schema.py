from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.preference_schema import PreferenceSignal
from app.schemas.request_constraint_schema import (
    HardConstraint,
    NamedConstraint,
    RequirementIssue,
    SpendPreference,
    UnmappedRequirement,
)


RevisionFieldName = Literal[
    "origin",
    "destination",
    "start_date",
    "end_date",
    "days",
    "budget",
    "people_count",
    "room_count",
]

RevisionFieldOperationType = Literal[
    "set",
    "increment",
    "clear",
]

RevisionPlanStatus = Literal[
    "ready",
    "clarification_required",
    "failed",
]

RevisionApplyStatus = Literal[
    "applied",
    "failed",
]

RevisionDestination = Literal[
    "apply_revision",
    "final_response",
]

ApplyRevisionDestination = Literal[
    "missing_info_check",
    "final_response",
]


class RevisionFieldOperation(BaseModel):
    """
    对 TripRequest 核心字段执行的一条受控修改操作。

    为什么不让 LLM 直接输出一份新的完整 TripRequest？
        修改请求经常包含相对表达，例如：
            - “多待两天”
            - “预算增加1000”

        如果直接重新生成完整 TripRequest，模型可能把“增加1000”错误理解成
        “预算改成1000”。因此这里显式区分：

            set        直接设置新值；
            increment  在旧值基础上增加或减少；
            clear      清空字段，随后由 MissingInfoCheckNode 决定是否需要澄清。

    示例：
        {
            "operation": "increment",
            "field": "days",
            "value": 2,
            "evidence": "多待两天"
        }
    """

    operation: RevisionFieldOperationType
    field: RevisionFieldName
    value: Any = None
    evidence: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def validate_operation(self) -> "RevisionFieldOperation":
        """根据字段和操作类型检查 value 是否可执行。"""

        # 1. clear 不需要值；如果模型多传了值，直接拒绝，避免语义含糊。
        if self.operation == "clear":
            if self.value is not None:
                raise ValueError("operation=clear 时 value 必须为 null")
            return self

        # 2. increment 只允许用于明确的数值增量字段。
        if self.operation == "increment":
            if self.field not in {"days", "budget"}:
                raise ValueError(
                    "operation=increment 只允许用于 days 或 budget"
                )

            if isinstance(self.value, bool) or not isinstance(
                self.value,
                (int, float),
            ):
                raise ValueError("increment.value 必须是数值")

            if float(self.value) == 0:
                raise ValueError("increment.value 不能为 0")

            return self

        # 3. set 根据字段执行基础类型检查。
        if self.field in {"origin", "destination"}:
            if not isinstance(self.value, str) or not self.value.strip():
                raise ValueError(f"{self.field} 必须是非空字符串")

        elif self.field in {"start_date", "end_date"}:
            if isinstance(self.value, date):
                return self

            if not isinstance(self.value, str):
                raise ValueError(f"{self.field} 必须是 YYYY-MM-DD 字符串")

            try:
                date.fromisoformat(self.value)
            except ValueError as exc:
                raise ValueError(
                    f"{self.field} 必须是合法 YYYY-MM-DD 日期"
                ) from exc

        elif self.field in {"days", "people_count", "room_count"}:
            if isinstance(self.value, bool) or not isinstance(
                self.value,
                int,
            ):
                raise ValueError(f"{self.field} 必须是整数")

            if self.value < 1:
                raise ValueError(f"{self.field} 必须大于等于 1")

        elif self.field == "budget":
            if isinstance(self.value, bool) or not isinstance(
                self.value,
                (int, float),
            ):
                raise ValueError("budget 必须是数值")

            if float(self.value) < 0:
                raise ValueError("budget 不能小于 0")

        return self


class PreferenceSelector(BaseModel):
    """删除一条主观偏好时使用的最小定位信息。"""

    domain: str = Field(min_length=1)
    key: str = Field(min_length=1)


class SpendPreferenceSelector(BaseModel):
    """删除某一预算类别消费倾向时使用的定位信息。"""

    category: Literal[
        "transport",
        "hotel",
        "food_activity",
    ]


class HardConstraintSelector(BaseModel):
    """删除某一客观硬约束时使用的最小定位信息。"""

    domain: Literal["trip", "flight", "hotel"]
    field: str = Field(min_length=1)


class NamedConstraintSelector(BaseModel):
    """
    删除指定实体约束时使用的定位信息。

    entity_name 为空时，表示删除该 entity_type 下的全部匹配项。
    """

    entity_type: Literal[
        "hotel",
        "flight",
        "food",
        "attraction",
        "activity",
    ]
    entity_name: str | None = None


class RevisionPatch(BaseModel):
    """
    RevisionAnalyzeNode 输出的受控修改补丁。

    该结构故意只允许项目当前能够安全处理的修改类型，避免 LLM 输出任意
    Python 路径或任意 State 字段。

    主要能力：
        - 修改目的地、日期、天数、预算、人数和房间数；
        - 增删主观偏好；
        - 增删消费倾向；
        - 增删客观硬约束；
        - 替换或增删指定酒店、航班、食物、景点和活动；
        - 保存当前无法直接执行的开放需求。
    """

    field_operations: list[RevisionFieldOperation] = Field(
        default_factory=list,
        description="核心字段 set / increment / clear 操作。",
    )

    preference_upserts: list[PreferenceSignal] = Field(
        default_factory=list,
        description="按 domain + key 更新或新增主观偏好。",
    )

    preference_removals: list[PreferenceSelector] = Field(
        default_factory=list,
        description="需要删除的主观偏好。",
    )

    spend_preference_upserts: list[SpendPreference] = Field(
        default_factory=list,
        description="按 category 更新预算消费倾向。",
    )

    spend_preference_removals: list[SpendPreferenceSelector] = Field(
        default_factory=list,
        description="需要删除的预算消费倾向类别。",
    )

    hard_constraint_additions: list[HardConstraint] = Field(
        default_factory=list,
        description="新增或替换的客观硬约束。",
    )

    hard_constraint_removals: list[HardConstraintSelector] = Field(
        default_factory=list,
        description="需要删除的客观硬约束。",
    )

    named_constraint_additions: list[NamedConstraint] = Field(
        default_factory=list,
        description="新增的指定酒店、航班、食物、景点或活动。",
    )

    named_constraint_removals: list[NamedConstraintSelector] = Field(
        default_factory=list,
        description="需要删除的指定实体约束。",
    )

    replace_named_entity_types: list[
        Literal[
            "hotel",
            "flight",
            "food",
            "attraction",
            "activity",
        ]
    ] = Field(
        default_factory=list,
        description=(
            "先删除某一实体类型的旧约束，再添加新约束。"
            "例如“酒店换成绿水青山酒店”应包含 hotel。"
        ),
    )

    unmapped_requirement_additions: list[UnmappedRequirement] = Field(
        default_factory=list,
        description="能够理解但当前业务无法确定性执行的新增要求。",
    )

    requirement_issues: list[RequirementIssue] = Field(
        default_factory=list,
        description="会阻止应用补丁的歧义或冲突。",
    )

    summary: str = Field(
        default="",
        max_length=1000,
        description="对本次修改内容的简短说明。",
    )

    confidence: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="模型对补丁解析结果的自评信心，仅用于 Trace 和 Eval。",
    )

    @model_validator(mode="after")
    def validate_patch(self) -> "RevisionPatch":
        """防止同一核心字段被多条操作重复修改。"""

        touched_fields = [
            item.field
            for item in self.field_operations
        ]

        if len(touched_fields) != len(set(touched_fields)):
            raise ValueError("同一个核心字段只能出现一条 field_operation")

        return self

    def has_changes(self) -> bool:
        """判断补丁中是否至少包含一项可应用修改。"""

        return any(
            [
                self.field_operations,
                self.preference_upserts,
                self.preference_removals,
                self.spend_preference_upserts,
                self.spend_preference_removals,
                self.hard_constraint_additions,
                self.hard_constraint_removals,
                self.named_constraint_additions,
                self.named_constraint_removals,
                self.replace_named_entity_types,
                self.unmapped_requirement_additions,
            ]
        )

    def to_state_dict(self) -> dict[str, Any]:
        """转换成可以写入 TravelState 的普通 dict。"""

        return self.model_dump(mode="json")


class RevisionPlan(BaseModel):
    """
    RevisionAnalyzeNode 的完整输出。

    base_trip_request_hash 用于防止分析完成后，当前 TripRequest 已被其他请求
    修改，导致补丁应用到错误基线。
    """

    status: RevisionPlanStatus
    strategy_version: str = "llm_revision_patch_v1"

    base_proposal_id: str = Field(min_length=1)
    base_proposal_version: int = Field(ge=1)
    base_trip_request_hash: str = Field(min_length=64, max_length=64)

    raw_change_request: str = Field(min_length=1, max_length=4000)
    patch: RevisionPatch

    clarification_questions: list[str] = Field(default_factory=list)

    model_id: str
    attempt_count: int = Field(default=1, ge=0)
    fallback_used: bool = False
    validation_errors: list[str] = Field(default_factory=list)

    def to_state_dict(self) -> dict[str, Any]:
        """转换成可以写入 TravelState 的普通 dict。"""

        return self.model_dump(mode="json")


class RevisionApplyResult(BaseModel):
    """ApplyRevisionNode 应用补丁后的结构化执行摘要。"""

    status: RevisionApplyStatus
    strategy_version: str = "deterministic_revision_apply_v1"

    base_proposal_id: str | None = None
    base_proposal_version: int | None = None

    before_trip_request_hash: str | None = None
    after_trip_request_hash: str | None = None

    changed_fields: list[str] = Field(default_factory=list)
    invalidated_fields: list[str] = Field(default_factory=list)

    summary: str = ""
    applied_at: str | None = None
    issues: list[dict[str, Any]] = Field(default_factory=list)

    def to_state_dict(self) -> dict[str, Any]:
        """转换成可以写入 TravelState 的普通 dict。"""

        return self.model_dump(mode="json")
