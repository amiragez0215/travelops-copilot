"""
guards 包。

Guard 的职责：
- 在 workflow 继续执行前做边界检查。
- 例如安全边界、权限边界、项目范围边界。

Guard 和 Tool 的区别：
- Tool 通常执行查询、计算或写入能力。
- Guard 通常判断“能不能继续”。

SafetyCheckNode 会调用 SafetyGuard。
"""

from app.guards.safety_guard import (
    SAFETY_RULES,
    SafetyRule,
    evaluate_safety_request,
)

__all__ = [
    "SAFETY_RULES",
    "SafetyRule",
    "evaluate_safety_request",
]