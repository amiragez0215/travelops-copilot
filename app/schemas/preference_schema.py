from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


PreferenceDomain = Literal[
    "flight",
    "hotel",
    "activity",
    "pace",
    "diet",
    "budget",
    "risk",
]

PreferencePolarity = Literal["positive", "negative"]
PreferenceSource = Literal["rule", "llm", "memory", "merged"]


class PreferenceSignal(BaseModel):
    """
    用户本次输入中的一条“可评分偏好”。

    重要边界：
        PreferenceSignal 只负责表达用户更在意什么，后续用于加权评分。
        它不再携带 hard_constraint 字段。

    为什么删除 hard_constraint？
        “一定要安静”“必须干净”中的“一定、必须”常常只是表达强烈，
        但安静和干净本身是连续、主观属性，不适合直接过滤全部候选。

        真正可硬过滤的条件会使用独立的 HardConstraint，例如：
            - 必须直飞
            - 起飞不能早于 08:00
            - 酒店每晚不能超过 600 元
            - 只住某一家明确酒店

    示例：
        {
            "domain": "hotel",
            "key": "quiet",
            "label": "安静",
            "strength": 1.0,
            "polarity": "positive",
            "evidence": "一定要安静",
            "source": "llm"
        }
    """

    domain: PreferenceDomain = Field(
        description="偏好所属领域，例如 hotel、flight、activity、pace。"
    )

    key: str = Field(
        min_length=1,
        description="标准化偏好 key，例如 quiet、near_subway、food。",
    )

    label: str = Field(
        min_length=1,
        description="给用户或 Trace 阅读的中文标签，例如 安静、靠近地铁。",
    )

    strength: float = Field(
        ge=0,
        le=1,
        description=(
            "偏好强度，范围 0 到 1。"
            "‘一定、非常、特别’通常提高到接近 1.0，但不会自动变成硬约束。"
        ),
    )

    polarity: PreferencePolarity = Field(
        default="positive",
        description=(
            "positive 表示偏好某特征；negative 表示希望避免某特征。"
            "当前规则通常把负向表达标准化成 avoid_xxx 的 positive 偏好。"
        ),
    )

    evidence: str = Field(
        default="",
        description="用户原话证据，例如‘一定要安静’。",
    )

    source: PreferenceSource = Field(
        default="rule",
        description="偏好信号来自规则、LLM、长期记忆或合并结果。",
    )

    def to_state_dict(self) -> dict:
        """
        转成适合写入 TravelState 的普通 dict。

        Pydantic v1/v2 兼容。
        """

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")

        return self.dict()
