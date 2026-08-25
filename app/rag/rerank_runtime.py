from __future__ import annotations

from threading import RLock

from app.common.config import settings
from app.rag.reranker import (
    CrossEncoderReranker,
    TransformersCrossEncoderScorer,
)


_DEFAULT_RERANKER: CrossEncoderReranker | None = None
_RUNTIME_LOCK = RLock()


def initialize_reranker(
    force_reload: bool = False,
) -> CrossEncoderReranker:
    """
    初始化应用级 Cross-Encoder Reranker 单例。

    正常生命周期：
        FastAPI 启动
        → 加载一次 tokenizer 和模型权重
        → 所有 RerankNode 请求复用同一个实例

    Args:
        force_reload:
            True 时忽略当前单例并重新加载模型。
            只用于开发阶段更换模型配置或运行资源检查。
    """

    global _DEFAULT_RERANKER

    # 1. 使用进程锁，避免并发请求同时加载同一个大型模型。
    with _RUNTIME_LOCK:
        # 2. 已经初始化且没有要求重载时，直接复用应用级单例。
        if _DEFAULT_RERANKER is not None and not force_reload:
            return _DEFAULT_RERANKER

        # 3. 根据统一 Settings 创建真实 Query-Passage 打分器。
        scorer = TransformersCrossEncoderScorer(
            model_id=settings.rag_rerank_model,
            device=settings.rag_rerank_device,
            batch_size=settings.rag_rerank_batch_size,
            max_length=settings.rag_rerank_max_length,
            cache_size=settings.rag_rerank_cache_size,
            use_fp16=settings.rag_rerank_use_fp16,
        )

        # 4. 创建任务级 Reranker；每个 RetrievalTask 独立精排。
        _DEFAULT_RERANKER = CrossEncoderReranker(
            scorer=scorer,
            top_k_per_task=settings.rag_rerank_top_k_per_task,
            min_hotel_evidence_per_hotel=settings.rag_rerank_min_per_hotel,
            fallback_on_error=settings.rag_rerank_fallback_on_error,
        )

        return _DEFAULT_RERANKER


def get_default_reranker() -> CrossEncoderReranker:
    """
    获取应用级 Reranker。

    如果 FastAPI 启动阶段没有预加载，这里会在第一次调用时懒加载。
    """

    return initialize_reranker(force_reload=False)


def reset_reranker_runtime() -> None:
    """
    清除当前进程中的 Reranker 引用。

    该操作不会删除本地模型目录，也不会影响 Chroma 数据。
    """

    global _DEFAULT_RERANKER

    with _RUNTIME_LOCK:
        _DEFAULT_RERANKER = None
