from __future__ import annotations

import math

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag.bm25_index import InMemoryBM25Index
from app.rag.chroma_vector_store import ChromaVectorStore
from app.rag.hybrid_retriever import HybridRetriever, HybridStartupReport


class KeywordEmbeddings(Embeddings):
    vocabulary = [
        "成都",
        "美食",
        "自然",
        "酒店",
        "安静",
        "干净",
        "地铁",
        "雨天",
        "安全",
    ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        values = [float(text.count(term)) for term in self.vocabulary]
        values.append(0.1)
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values]


def _documents():
    return [
        Document(
            page_content="成都三日游可以安排本地美食、茶馆和自然风景。",
            metadata={
                "chunk_id": "guide_001",
                "chunk_hash": "guide-hash",
                "content_hash": "guide-hash",
                "document_hash": "guide-doc-hash",
                "source": "guides/guide.md",
                "doc_type": "guide",
                "city": "成都",
                "section": "三日游",
            },
        ),
        Document(
            page_content="成都青羊酒店夜间安静，房间干净，距离地铁较近。",
            metadata={
                "chunk_id": "hotel_001_chunk",
                "chunk_hash": "hotel1-hash",
                "content_hash": "hotel1-hash",
                "document_hash": "hotel1-doc-hash",
                "source": "hotel_reviews/h1.md",
                "doc_type": "hotel_reviews",
                "city": "成都",
                "hotel_id": "hotel_001",
                "section": "住客评价",
            },
        ),
        Document(
            page_content="成都春熙路酒店位置热闹，夜间可能存在街道噪音。",
            metadata={
                "chunk_id": "hotel_002_chunk",
                "chunk_hash": "hotel2-hash",
                "content_hash": "hotel2-hash",
                "document_hash": "hotel2-doc-hash",
                "source": "hotel_reviews/h2.md",
                "doc_type": "hotel_reviews",
                "city": "成都",
                "hotel_id": "hotel_002",
                "section": "住客评价",
            },
        ),
        Document(
            page_content="成都雨天道路湿滑，需要预留交通时间。",
            metadata={
                "chunk_id": "safety_001",
                "chunk_hash": "safety-hash",
                "content_hash": "safety-hash",
                "document_hash": "safety-doc-hash",
                "source": "safety_notices/rain.md",
                "doc_type": "safety_notice",
                "city": "成都",
                "risk_level": "medium",
                "section": "雨天安全",
            },
        ),
    ]


def _retriever(tmp_path):
    documents = _documents()
    embeddings = KeywordEmbeddings()
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="travelops_hybrid_test",
        embeddings=embeddings,
        embedding_model_id="keyword-v1",
    )
    sync_report = vector_store.sync_documents(documents)
    bm25 = InMemoryBM25Index(documents)

    return HybridRetriever(
        documents=documents,
        bm25_index=bm25,
        vector_store=vector_store,
        rag_docs_dir=tmp_path,
        startup_report=HybridStartupReport(
            source_document_count=4,
            active_chunk_count=4,
            duplicate_source_count=0,
            bm25_build_count=bm25.build_count,
            chroma_sync=sync_report.to_dict(),
        ),
    )


def _plan():
    return {
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
                "semantic_query": "成都美食和自然风景",
                "keyword_query": "成都 美食 自然",
                "metadata_filter": {"city": "成都", "doc_type": "guide"},
                "required": True,
                "top_k": 3,
                "min_required_chunks": 1,
                "priority": 1,
                "reason": "测试",
            },
            {
                "task_id": "hotel_review_candidates",
                "category": "hotel_reviews",
                "doc_type": "hotel_reviews",
                "purpose": "酒店评价",
                "semantic_query": "安静干净近地铁酒店",
                "keyword_query": "安静 干净 地铁",
                "metadata_filter": {
                    "city": "成都",
                    "doc_type": "hotel_reviews",
                    "hotel_ids": ["hotel_001"],
                },
                "required": True,
                "top_k": 3,
                "min_required_chunks": 1,
                "priority": 1,
                "reason": "测试",
            },
        ],
        "coverage_requirements": {},
        "max_repair_rounds": 1,
        "repair_exhausted": False,
        "notes": [],
    }


def test_hybrid_retriever_fuses_bm25_and_chroma_and_respects_metadata(tmp_path):
    """
    HybridRetriever 应使用同一 candidate_ids 执行两路检索，并用 RRF 融合。
    """

    retriever = _retriever(tmp_path)
    output = retriever.retrieve_plan(_plan())

    assert output["summary"]["status"] == "ok"
    assert output["retrieved_chunks"]

    hotel_chunks = [
        chunk
        for chunk in output["retrieved_chunks"]
        if chunk["task_id"] == "hotel_review_candidates"
    ]

    assert hotel_chunks
    assert all(chunk["metadata"]["hotel_id"] == "hotel_001" for chunk in hotel_chunks)
    assert any(
        set(chunk["retrieval_sources"]) == {"bm25", "vector"}
        for chunk in output["retrieved_chunks"]
    )


def test_hybrid_retriever_reuses_same_bm25_across_multiple_plans(tmp_path):
    """
    多次执行 retrieve_plan 不应重建 BM25。
    """

    retriever = _retriever(tmp_path)
    initial_build_count = retriever.bm25_index.build_count
    index_object_id = id(retriever.bm25_index._index)

    retriever.retrieve_plan(_plan())
    retriever.retrieve_plan(_plan())

    assert retriever.bm25_index.build_count == initial_build_count == 1
    assert id(retriever.bm25_index._index) == index_object_id


def test_delete_source_removes_chroma_and_rebuilds_bm25(tmp_path):
    """
    删除 source 后，Chroma 和 BM25 都必须移除同一批 chunks。
    """

    retriever = _retriever(tmp_path)
    before_build_count = retriever.bm25_index.build_count

    result = retriever.delete_source("hotel_reviews/h1.md", delete_source_file=False)

    assert result["deleted_vector_count"] == 1
    assert result["removed_memory_chunk_count"] == 1
    assert retriever.bm25_index.build_count == before_build_count + 1

    output = retriever.retrieve_task(_plan()["tasks"][1])
    assert output["chunks"] == []
