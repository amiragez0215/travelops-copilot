from __future__ import annotations

import math

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag.chroma_vector_store import ChromaVectorStore


class CountingEmbeddings(Embeddings):
    """
    测试用 Embeddings，同时统计文档编码和 query 编码次数。
    """

    vocabulary = ["成都", "美食", "酒店", "安静", "干净", "地铁"]

    def __init__(self):
        self.embed_documents_calls = 0
        self.embed_query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_calls += 1
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls += 1
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        values = [float(text.count(term)) for term in self.vocabulary]
        values.append(0.1)
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values]


def _doc(chunk_id, source, content, chunk_hash):
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "chunk_hash": chunk_hash,
            "content_hash": chunk_hash,
            "document_hash": f"doc-{source}",
            "source": source,
            "doc_type": "guide",
            "city": "成都",
            "section": "测试",
        },
    )


def test_chroma_sync_skips_unchanged_documents(tmp_path):
    """
    第二次同步相同 chunks 时，不应再次调用 embed_documents。
    """

    embeddings = CountingEmbeddings()
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="travelops_test_sync",
        embeddings=embeddings,
        embedding_model_id="counting-v1",
    )
    documents = [
        _doc("c1", "guides/a.md", "成都美食攻略", "hash-c1"),
        _doc("c2", "guides/b.md", "成都自然攻略", "hash-c2"),
    ]

    first = store.sync_documents(documents)
    calls_after_first = embeddings.embed_documents_calls
    second = store.sync_documents(documents)

    assert first.added_count == 2
    assert calls_after_first == 1
    assert second.unchanged_count == 2
    assert second.added_count == 0
    assert second.updated_count == 0
    assert embeddings.embed_documents_calls == calls_after_first


def test_chroma_updates_only_changed_chunk_and_queries_only_query(tmp_path):
    """
    chunk_hash 改变时只重算变化 chunk；在线查询只调用 embed_query。
    """

    embeddings = CountingEmbeddings()
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="travelops_test_update",
        embeddings=embeddings,
        embedding_model_id="counting-v1",
    )

    original = [_doc("c1", "guides/a.md", "成都美食攻略", "hash-v1")]
    store.sync_documents(original)

    changed = [_doc("c1", "guides/a.md", "成都美食与茶馆攻略", "hash-v2")]
    report = store.sync_documents(changed)
    document_calls_before_search = embeddings.embed_documents_calls

    hits = store.search("成都美食", candidate_ids=["c1"], k=3)

    assert report.updated_count == 1
    assert hits[0].document.metadata["chunk_id"] == "c1"
    assert embeddings.embed_documents_calls == document_calls_before_search
    assert embeddings.embed_query_calls == 1


def test_chroma_delete_by_source(tmp_path):
    """
    删除 source 后，对应全部向量记录应消失。
    """

    embeddings = CountingEmbeddings()
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="travelops_test_delete",
        embeddings=embeddings,
        embedding_model_id="counting-v1",
    )
    store.sync_documents(
        [
            _doc("c1", "guides/a.md", "成都美食攻略", "h1"),
            _doc("c2", "guides/a.md", "成都茶馆攻略", "h2"),
            _doc("c3", "guides/b.md", "成都自然攻略", "h3"),
        ]
    )

    deleted = store.delete_by_source("guides/a.md")

    assert deleted == 2
    assert store.count() == 1
    assert store.search("成都美食", ["c1", "c2"], k=3) == []


def test_chroma_reopens_persisted_collection_without_reembedding_documents(tmp_path):
    """
    新进程语义下重新创建 Store 时，应直接读取磁盘 collection；
    同步相同 chunks 不重新编码文档，查询只编码 query。
    """

    persist_directory = tmp_path / "chroma"
    documents = [_doc("c1", "guides/a.md", "成都美食攻略", "hash-c1")]

    first_embeddings = CountingEmbeddings()
    first_store = ChromaVectorStore(
        persist_directory=persist_directory,
        collection_name="travelops_test_persist",
        embeddings=first_embeddings,
        embedding_model_id="counting-v1",
    )
    first_store.sync_documents(documents)

    second_embeddings = CountingEmbeddings()
    reopened_store = ChromaVectorStore(
        persist_directory=persist_directory,
        collection_name="travelops_test_persist",
        embeddings=second_embeddings,
        embedding_model_id="counting-v1",
    )

    report = reopened_store.sync_documents(documents)
    hits = reopened_store.search("成都美食", ["c1"], k=3)

    assert report.unchanged_count == 1
    assert second_embeddings.embed_documents_calls == 0
    assert second_embeddings.embed_query_calls == 1
    assert hits[0].document.metadata["chunk_id"] == "c1"
