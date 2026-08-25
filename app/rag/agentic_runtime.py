from __future__ import annotations

from functools import lru_cache

from app.common.config import settings
from app.llm.deepseek_json_client import DeepSeekJSONClient
from app.rag.agentic_evidence_grader import AgenticEvidenceGrader
from app.rag.evidence_grader import EvidenceGrader, RuleBasedEvidenceGrader
from app.rag.llm_retrieval_planner import LLMRetrievalPlanner
from app.rag.retrieval_planner import RetrievalPlanner, RuleBasedRetrievalPlanner


@lru_cache(maxsize=1)
def _build_client() -> DeepSeekJSONClient:
    """构建 Planner/Judge 共用的结构化 JSON 客户端。"""

    if not settings.deepseek_api_key:
        raise ValueError("Agentic RAG 需要 DEEPSEEK_API_KEY")
    return DeepSeekJSONClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.llm_base_url,
        model_id=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        thinking_enabled=settings.llm_thinking_enabled,
    )


@lru_cache(maxsize=1)
def get_default_retrieval_planner() -> RetrievalPlanner:
    """
    返回应用级 Planner 单例。

    rule 模式或无 Key 时保持当前规则行为；agentic 模式下使用 LLM Planner，
    而 LLM Planner 内部仍持有同一个 Rule Planner 作为失败 fallback。
    """

    rule = RuleBasedRetrievalPlanner(
        max_repair_rounds=settings.agentic_rag_max_replans
    )
    if settings.rag_planner_mode != "agentic" or not settings.deepseek_api_key:
        return rule
    return LLMRetrievalPlanner(
        client=_build_client(),
        fallback_planner=rule,
        max_initial_tasks=settings.agentic_rag_max_initial_tasks,
        max_repair_tasks=settings.agentic_rag_max_repair_tasks,
        max_tokens=settings.agentic_rag_planner_max_tokens,
    )


@lru_cache(maxsize=1)
def get_default_evidence_grader() -> EvidenceGrader:
    """返回与当前 Planner 模式对应的 Rule 或 Agentic Evidence Grader。"""

    rule = RuleBasedEvidenceGrader()
    if (
        settings.rag_planner_mode != "agentic"
        or not settings.agentic_rag_semantic_judge_enabled
        or not settings.deepseek_api_key
    ):
        return rule
    return AgenticEvidenceGrader(
        client=_build_client(),
        rule_grader=rule,
        max_tokens=settings.agentic_rag_judge_max_tokens,
    )


def reset_agentic_rag_runtime() -> None:
    """测试或应用关闭时清除缓存的 Planner/Grader。"""

    get_default_retrieval_planner.cache_clear()
    get_default_evidence_grader.cache_clear()
    _build_client.cache_clear()
