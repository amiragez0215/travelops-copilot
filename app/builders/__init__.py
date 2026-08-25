"""
builders 包。

Builder 负责把已有结构化数据整理成更适合后续节点使用的上下文。
它不直接读写 TravelState，也不查数据库，不调用 LLM。
"""

from app.builders.planning_context_builder import build_planning_context

__all__ = ["build_planning_context"]