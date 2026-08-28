from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ToolCallSource = Literal["required_policy", "llm_selected"]


class ActivityToolArguments(BaseModel):
    """LLM 只负责填写活动语义，不允许重新决定城市和日期。"""

    interests: list[str] = Field(default_factory=list, max_length=12)
    indoor_preference: bool | None = None


class ExecutableToolCall(BaseModel):
    """经过 Policy 编译后、可以交给执行节点的单次调用。"""

    call_id: str = Field(min_length=1)
    requirement_key: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    server_id: str = Field(min_length=1)
    source: ToolCallSource
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(min_length=1)


class ToolPlan(BaseModel):
    """
    工具计划的权威 State 合同。

    required_policy_calls 始终由程序生成；LLM 只能补充活动调用。
    因此即使模型不可用，天气、航班和酒店仍然会进入执行计划。
    """

    status: Literal["planned", "degraded"] = "planned"
    planner_mode: Literal["rule", "llm"]
    model_id: str
    plan_retry_count: int = Field(default=0, ge=0)
    calls: list[ExecutableToolCall] = Field(default_factory=list)
    llm_selected_tools: list[str] = Field(default_factory=list)
    validation_issues: list[dict[str, Any]] = Field(default_factory=list)

    def to_state_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ToolCallResult(BaseModel):
    """一次真实或缓存 Tool 调用的统一结果。"""

    call_id: str
    requirement_key: str
    tool_name: str
    server_id: str
    source: ToolCallSource
    status: Literal["success", "failed", "cached"]
    arguments: dict[str, Any] = Field(default_factory=dict)
    data: Any | None = None
    latency_ms: int = Field(default=0, ge=0)
    transport: str
    error: dict[str, Any] | None = None


class ToolCheckResult(BaseModel):
    """ToolCheckNode 的路由决定，显式限制循环而不是依赖 Graph 总上限。"""

    status: Literal["passed", "degraded", "retry_execute", "replan", "failed"]
    next_action: Literal["candidate_rank", "tool_execute", "tool_plan", "final_response"]
    required_tools_passed: bool
    activity_status: Literal["not_requested", "success", "no_results", "degraded"]
    issues: list[dict[str, Any]] = Field(default_factory=list)

    def to_state_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
