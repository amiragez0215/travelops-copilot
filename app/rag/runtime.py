from __future__ import annotations

from pathlib import Path
from threading import RLock

from app.common.config import settings
from app.rag.embeddings import create_default_embeddings
from app.rag.hybrid_retriever import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RETRIEVER: HybridRetriever | None = None
_RUNTIME_LOCK = RLock()


def initialize_hybrid_retriever(
    force_reload: bool = False,
    reset_collection: bool = False,
) -> HybridRetriever:
    """
    初始化应用级 HybridRetriever 单例。

    应用生命周期：
        FastAPI lifespan 启动
        → 加载和去重 Markdown
        → 应用内构建一次 BM25
        → 增量同步持久化 Chroma
        → 后续所有请求复用同一个对象

    Args:
        force_reload:
            True 时重新加载 Markdown，并重新构建进程内 BM25。
            文档管理脚本或测试时使用，普通请求不要使用。

        reset_collection:
            True 时删除旧 Chroma collection 后完整重建。
            更换 Embedding 模型或向量维度时使用。
    """

    global _DEFAULT_RETRIEVER

    # 1. 使用进程锁，避免并发启动时重复构建 BM25 或重复同步 Chroma。
    with _RUNTIME_LOCK:
        # 2. 已经初始化且不要求刷新时，直接复用整个对象。
        if _DEFAULT_RETRIEVER is not None and not force_reload:
            return _DEFAULT_RETRIEVER

        rag_docs_dir = _resolve_project_path(settings.rag_docs_dir)
        chroma_dir = _resolve_project_path(settings.rag_chroma_dir)

        # 3. 文档入库和在线 query 使用同一 Embedding 配置。
        embeddings = create_default_embeddings(
            model_name=settings.rag_embedding_model,
            device=settings.rag_embedding_device,
        )

        # 4. from_directory 完成：加载、SHA-256 去重、Chroma 增量同步和 BM25 一次构建。
        _DEFAULT_RETRIEVER = HybridRetriever.from_directory(
            root_dir=rag_docs_dir,
            persist_directory=chroma_dir,
            collection_name=settings.rag_chroma_collection,
            embeddings=embeddings,
            embedding_model_id=settings.rag_embedding_model,
            embedding_text_version=settings.rag_embedding_text_version,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            rrf_k=settings.rag_rrf_k,
            bm25_weight=settings.rag_bm25_weight,
            vector_weight=settings.rag_vector_weight,
            fetch_multiplier=settings.rag_fetch_multiplier,
            reset_collection=reset_collection,
        )

        return _DEFAULT_RETRIEVER


def get_default_hybrid_retriever() -> HybridRetriever:
    """
    获取应用级 HybridRetriever。

    FastAPI 正常启动时已经预加载；
    脚本直接调用 Node 时保留一次懒加载兜底。
    """

    if _DEFAULT_RETRIEVER is not None:
        return _DEFAULT_RETRIEVER

    return initialize_hybrid_retriever()


def reset_hybrid_retriever_runtime() -> None:
    """
    清空进程内单例。

    只释放 Python 对象，不删除 Chroma 磁盘数据。
    """

    global _DEFAULT_RETRIEVER

    with _RUNTIME_LOCK:
        _DEFAULT_RETRIEVER = None


def _resolve_project_path(value: str) -> Path:
    """将 .env 中的相对路径解析到项目根目录。"""

    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
