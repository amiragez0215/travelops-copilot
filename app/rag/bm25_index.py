from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class BM25Hit:
    """
    BM25 关键词检索的一条结果。
    """

    document: Document
    rank: int
    score: float


class InMemoryBM25Index:
    """
    应用生命周期内复用的 BM25 索引。

    设计目标：
        - 应用启动时根据去重后的 chunks 构建一次。
        - 每个 HybridRetrieveNode 只调用 search()，不重新构建索引。
        - 只有管理员同步/删除 RAG 文档时才显式 rebuild()。

    为什么 BM25 本轮不持久化？
        当前语料规模较小，BM25 构建只需要分词和统计，不需要调用昂贵的 Embedding 模型。
        将它保存在应用内存中，可以保持实现清晰，同时避免每次 workflow 请求重复构建。
    """

    def __init__(self, documents: list[Document]) -> None:
        """
        初始化并构建一次 BM25。
        """

        self.documents: list[Document] = []
        self._document_index_by_chunk_id: dict[str, int] = {}
        self._tokenized_corpus: list[list[str]] = []
        self._token_sets: list[set[str]] = []
        self._index: BM25Okapi | None = None
        self.build_count = 0

        # 1. 应用启动时第一次构建 BM25。
        self.rebuild(documents)

    def rebuild(self, documents: list[Document]) -> None:
        """
        使用新的同一批 chunks 重建 BM25。

        只在以下情况调用：
            - 应用首次启动。
            - RAG 文档同步后语料发生变化。
            - 删除文档后需要更新内存索引。
        """

        # 1. 保存新的文档顺序。
        self.documents = list(documents)

        # 2. 建立 chunk_id → 文档下标映射，供 metadata 候选过滤使用。
        self._document_index_by_chunk_id = {}

        for index, document in enumerate(self.documents):
            chunk_id = _chunk_id(document)

            if chunk_id in self._document_index_by_chunk_id:
                raise ValueError(f"BM25 语料存在重复 chunk_id: {chunk_id}")

            self._document_index_by_chunk_id[chunk_id] = index

        # 3. 对文档正文和重要 metadata 做中文分词。
        self._tokenized_corpus = [
            tokenize_for_bm25(_build_bm25_search_text(document))
            for document in self.documents
        ]

        # 4. 保存 token set，用于快速排除完全没有关键词交集的结果。
        self._token_sets = [set(tokens) for tokens in self._tokenized_corpus]

        # 5. 基于全部同源 chunks 创建一个 BM25Okapi 实例。
        self._index = (
            BM25Okapi(self._tokenized_corpus)
            if self._tokenized_corpus
            else None
        )

        # 6. 测试会检查正常查询不会增加 build_count，证明索引被复用。
        self.build_count += 1

    def search(
        self,
        query: str,
        candidate_ids: Iterable[str],
        k: int,
    ) -> list[BM25Hit]:
        """
        在 metadata 已筛选出的 candidate_ids 中执行 BM25 排序。

        Args:
            query:
                RetrievalTask.keyword_query。

            candidate_ids:
                metadata filter 匹配出的 chunk_id。

            k:
                最大返回数量。
        """

        # 1. 空语料、空查询或 k<=0 时直接返回。
        if self._index is None or k <= 0 or not query.strip():
            return []

        # 2. 只对当前 query 分词；不会重新分词所有文档。
        query_tokens = tokenize_for_bm25(query)

        if not query_tokens:
            return []

        # 3. 将候选 chunk_id 转成 BM25 语料中的文档下标。
        candidate_indices = [
            self._document_index_by_chunk_id[chunk_id]
            for chunk_id in dict.fromkeys(str(item) for item in candidate_ids)
            if chunk_id in self._document_index_by_chunk_id
        ]

        if not candidate_indices:
            return []

        # 4. 使用应用启动时已经建立的 BM25 索引计算 query 分数。
        scores = self._index.get_scores(query_tokens)
        query_token_set = set(query_tokens)
        ranked_candidates: list[tuple[int, float]] = []

        for document_index in candidate_indices:
            # 5. 没有任何关键词交集的文档不作为 BM25 命中。
            if not (query_token_set & self._token_sets[document_index]):
                continue

            ranked_candidates.append((document_index, float(scores[document_index])))

        # 6. 先按 BM25 分数降序，再按 chunk_id 保证结果稳定。
        ranked_candidates.sort(
            key=lambda item: (
                -item[1],
                _chunk_id(self.documents[item[0]]),
            )
        )

        # 7. 截取 top_k，并恢复文档正文与 metadata。
        return [
            BM25Hit(
                document=self.documents[document_index],
                rank=rank,
                score=score,
            )
            for rank, (document_index, score) in enumerate(
                ranked_candidates[:k],
                start=1,
            )
        ]


def tokenize_for_bm25(text: str) -> list[str]:
    """
    对中文、英文、数字和 hotel_id 混合文本进行 BM25 分词。
    """

    # 1. 统一英文大小写和首尾空白。
    normalized = text.lower().strip()
    tokens: list[str] = []

    # 2. 使用 jieba 对中文语句做基础切分。
    for piece in jieba.lcut(normalized, cut_all=False):
        # 3. 保留英文/数字/下划线 ID，也保留连续中文词。
        sub_tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", piece)

        for token in sub_tokens:
            cleaned = token.strip()

            if cleaned:
                tokens.append(cleaned)

    return tokens


def _build_bm25_search_text(document: Document) -> str:
    """
    把正文和重要 metadata 拼成 BM25 索引文本。

    这样 hotel_id、酒店名称、section、scenario 等精确字段也可以被关键词检索命中。
    """

    metadata = document.metadata
    metadata_terms = [
        metadata.get("title"),
        metadata.get("section"),
        metadata.get("city"),
        metadata.get("hotel_id"),
        metadata.get("doc_type"),
        metadata.get("scenario"),
        metadata.get("risk_type"),
        metadata.get("risk_level"),
        metadata.get("tags"),
    ]

    return " ".join(
        [
            document.page_content,
            *[
                _metadata_to_text(value)
                for value in metadata_terms
                if value is not None
            ],
        ]
    )


def _metadata_to_text(value: Any) -> str:
    """
    将 metadata 标量或数组转换成 BM25 可检索文本。
    """

    if isinstance(value, list):
        return " ".join(str(item) for item in value)

    return str(value)


def _chunk_id(document: Document) -> str:
    """
    读取并校验稳定 chunk_id。
    """

    chunk_id = document.metadata.get("chunk_id")

    if not chunk_id:
        raise ValueError("RAG Document 缺少 metadata.chunk_id")

    return str(chunk_id)
