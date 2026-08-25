"""
formatters 包。

Formatter 负责把结构化中间结果转成用户可读或 API 友好的结构。
"""

from app.formatters.clarification_formatter import (
    build_clarification,
    build_clarification_final_response,
)
from app.formatters.safe_reject_formatter import build_safe_reject_final_response
from app.formatters.final_response_formatter import build_final_response

__all__ = [
    "build_clarification",
    "build_clarification_final_response",
    "build_safe_reject_final_response",
    "build_final_response",
]