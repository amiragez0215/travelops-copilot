from __future__ import annotations

import math

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.nodes.hybrid_retrieve_node import hybrid_retrieve_node
from app.rag.bm25_index import InMemoryBM25Index
from app.rag.chroma_vector_store import ChromaVectorStore
from app.rag.hybrid_retriever import HybridRetriever, HybridStartupReport


class TestEmbeddings(Embeddings):
    vocabulary = ["成都", "美食", "攻略"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        values = [float(text.count(term)) for term in self.vocabulary]
        values.append(0.1)
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values]


def _retriever(tmp_path):
    documents = [
        Document(
            page_content="成都三日游可以体验美食和茶馆。",
            metadata={
                "chunk_id": "guide_001",
                "chunk_hash": "chunk-hash",
                "content_hash": "chunk-hash",
                "document_hash": "document-hash",
                "source": "guides/guide.md",
                "doc_type": "guide",
                "city": "成都",
                "section": "三日游",
            },
        )
    ]
    embeddings = TestEmbeddings()
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="travelops_node_test",
        embeddings=embeddings,
        embedding_model_id="test-v1",
    )
    sync = vector_store.sync_documents(documents)
    bm25 = InMemoryBM25Index(documents)

    return HybridRetriever(
        documents=documents,
        bm25_index=bm25,
        vector_store=vector_store,
        rag_docs_dir=tmp_path,
        startup_report=HybridStartupReport(
            source_document_count=1,
            active_chunk_count=1,
            duplicate_source_count=0,
            bm25_build_count=1,
            chroma_sync=sync.to_dict(),
        ),
    )


def _state():
    return {
        "retrieval_plan": {
            "plan_id": "rp_test",
            "mode": "initial",
            "strategy_version": "test",
            "city": "成都",
            "date_range": {"start": "2026-07-02", "end": "2026-07-04"},
            "tasks": [
                {
                    "task_id": "guide_main",
                    "category": "guides",
                    "doc_type": "guide",
                    "purpose": "攻略",
                    "semantic_query": "成都三日游美食",
                    "keyword_query": "成都 三日游 美食",
                    "metadata_filter": {"city": "成都", "doc_type": "guide"},
                    "required": True,
                    "top_k": 3,
                    "min_required_chunks": 1,
                    "priority": 1,
                    "reason": "测试",
                }
            ],
            "coverage_requirements": {},
            "max_repair_rounds": 1,
            "repair_exhausted": False,
            "notes": [],
        }
    }


def test_hybrid_retrieve_node_success(tmp_path):
    """Node 应写入 chunks、摘要和 trace。"""

    result = hybrid_retrieve_node(_state(), retriever=_retriever(tmp_path))

    assert result["retrieved_chunks"]
    assert result["hybrid_retrieval_result"]["status"] == "ok"
    assert result["trace"][0]["tool_name"] == "bm25+chroma+rrf"
    assert result["trace"][0]["status"] == "success"


def test_hybrid_retrieve_node_missing_plan_returns_error(tmp_path):
    """缺少 retrieval_plan 时返回结构化错误。"""

    result = hybrid_retrieve_node({}, retriever=_retriever(tmp_path))

    assert result["retrieved_chunks"] == []
    assert result["errors"][0]["node"] == "hybrid_retrieve"
    assert result["trace"][0]["status"] == "failed"


def test_hybrid_retrieve_node_handles_empty_tasks(tmp_path):
    """空任务计划返回 no_tasks，不视为程序异常。"""

    state = _state()
    state["retrieval_plan"]["tasks"] = []

    result = hybrid_retrieve_node(state, retriever=_retriever(tmp_path))

    assert result["retrieved_chunks"] == []
    assert result["hybrid_retrieval_result"]["status"] == "no_tasks"
    assert "errors" not in result
