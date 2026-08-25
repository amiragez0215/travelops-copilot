"""
Agentic RAG 模块。

当前能力：
    - Markdown 加载与 SHA-256 去重
    - 应用生命周期内复用的 BM25
    - Chroma 持久化 Vector Store
    - BM25 + Vector + RRF Hybrid Retrieval
    - Cross-Encoder Rerank
    - Rule-based Evidence Grade 与 Repair Loop
"""

from app.rag.bm25_index import BM25Hit, InMemoryBM25Index
from app.rag.chroma_vector_store import (
    ChromaSyncReport,
    ChromaVectorStore,
    VectorHit,
)
from app.rag.document_loader import (
    DeduplicationResult,
    DuplicateDocument,
    deduplicate_documents,
    load_rag_documents,
)
from app.rag.evidence_grader import EvidenceGrader, RuleBasedEvidenceGrader
from app.rag.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from app.rag.rerank_runtime import (
    get_default_reranker,
    initialize_reranker,
    reset_reranker_runtime,
)
from app.rag.reranker import (
    CrossEncoderReranker,
    PairScore,
    PairScorer,
    TransformersCrossEncoderScorer,
    build_rerank_passage,
)
from app.rag.retrieval_planner import (
    RetrievalPlanner,
    RuleBasedRetrievalPlanner,
    build_retrieval_plan,
)
from app.rag.runtime import (
    get_default_hybrid_retriever,
    initialize_hybrid_retriever,
    reset_hybrid_retriever_runtime,
)


__all__ = [
    "BM25Hit",
    "InMemoryBM25Index",
    "VectorHit",
    "ChromaSyncReport",
    "ChromaVectorStore",
    "DuplicateDocument",
    "DeduplicationResult",
    "load_rag_documents",
    "deduplicate_documents",
    "RetrievalPlanner",
    "RuleBasedRetrievalPlanner",
    "build_retrieval_plan",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "initialize_hybrid_retriever",
    "get_default_hybrid_retriever",
    "reset_hybrid_retriever_runtime",
    "PairScore",
    "PairScorer",
    "TransformersCrossEncoderScorer",
    "CrossEncoderReranker",
    "build_rerank_passage",
    "initialize_reranker",
    "get_default_reranker",
    "reset_reranker_runtime",
    "EvidenceGrader",
    "RuleBasedEvidenceGrader",
]
