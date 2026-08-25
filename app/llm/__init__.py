"""LLM 基础客户端封装。"""

from app.llm.deepseek_json_client import (
    DeepSeekJSONClient,
    StructuredJSONClient,
)

__all__ = [
    "StructuredJSONClient",
    "DeepSeekJSONClient",
]
