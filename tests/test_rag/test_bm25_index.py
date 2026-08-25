from __future__ import annotations

from langchain_core.documents import Document

from app.rag.bm25_index import InMemoryBM25Index


def _documents():
    return [
        Document(
            page_content="成都三日游可以安排美食、茶馆和城市公园。",
            metadata={"chunk_id": "guide_001", "source": "guides/a.md"},
        ),
        Document(
            page_content="酒店夜间安静，房间干净，距离地铁较近。",
            metadata={"chunk_id": "hotel_001", "source": "hotel_reviews/h1.md"},
        ),
    ]


def test_bm25_builds_once_and_reuses_during_queries():
    """
    多次 search 不应重建 BM25。
    """

    index = InMemoryBM25Index(_documents())
    initial_build_count = index.build_count
    internal_index_id = id(index._index)

    first = index.search("成都 美食", ["guide_001"], k=3)
    second = index.search("酒店 安静", ["hotel_001"], k=3)

    assert first[0].document.metadata["chunk_id"] == "guide_001"
    assert second[0].document.metadata["chunk_id"] == "hotel_001"
    assert index.build_count == initial_build_count == 1
    assert id(index._index) == internal_index_id


def test_bm25_rebuilds_only_after_explicit_corpus_change():
    """
    管理员删除或同步文档时显式 rebuild，build_count 才增加。
    """

    index = InMemoryBM25Index(_documents())
    index.rebuild(_documents()[:1])

    assert index.build_count == 2
    assert index.search("酒店 安静", ["hotel_001"], k=3) == []
