from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.documents import Document


def metadata_matches(
    metadata: Mapping[str, Any],
    metadata_filter: Mapping[str, Any] | None,
) -> bool:
    """
    判断一条 Document metadata 是否满足 RetrievalTask.metadata_filter。

    当前支持：
        1. 标量相等：{"city": "成都"}
        2. 允许值列表：{"risk_level": ["medium", "high"]}
        3. hotel_ids 别名：过滤条件 hotel_ids 对应文档 metadata.hotel_id
        4. 文档 metadata 本身为数组时，只要与期望值存在交集即可
    """

    # 1. 没有过滤条件时，所有文档都匹配。
    if not metadata_filter:
        return True

    # 2. 每个过滤字段都必须满足；整体是 AND 关系。
    for filter_key, expected in metadata_filter.items():
        if expected is None:
            continue

        actual = _read_metadata_value(
            metadata=metadata,
            filter_key=filter_key,
        )

        if not _values_match(actual=actual, expected=expected):
            return False

    return True


def matching_chunk_ids(
    documents: list[Document],
    metadata_filter: Mapping[str, Any] | None,
) -> list[str]:
    """
    从当前语料中找出满足 metadata filter 的 chunk_id。

    HybridRetriever 会把这份 ID 列表同时交给：
        - BM25Index.search(candidate_ids=...)
        - ChromaVectorStore.search(candidate_ids=...)

    因此两个检索器在完全相同的候选集合中搜索。
    """

    result: list[str] = []

    # 1. 遍历应用启动时加载并去重后的同一批 chunks。
    for document in documents:
        if not metadata_matches(document.metadata, metadata_filter):
            continue

        chunk_id = document.metadata.get("chunk_id")

        if not chunk_id:
            raise ValueError("RAG Document 缺少 metadata.chunk_id")

        result.append(str(chunk_id))

    # 2. 保持文档原始顺序去重。
    return list(dict.fromkeys(result))


def _read_metadata_value(
    metadata: Mapping[str, Any],
    filter_key: str,
) -> Any:
    """
    处理检索计划字段和文档 metadata 字段之间的别名。
    """

    # RetrievalPlan 使用 hotel_ids 列表；每个酒店评价 chunk 使用单个 hotel_id。
    if filter_key == "hotel_ids":
        return metadata.get("hotel_id") or metadata.get("hotel_ids")

    return metadata.get(filter_key)


def _values_match(actual: Any, expected: Any) -> bool:
    """
    比较 metadata 实际值与过滤条件。

    例子：
        actual="medium"
        expected=["medium", "high"]
        → True

        actual=["rainy_city_trip", "general"]
        expected=["general"]
        → True
    """

    if actual is None:
        return False

    actual_values = actual if isinstance(actual, list) else [actual]
    expected_values = expected if isinstance(expected, list) else [expected]

    normalized_actual = {_normalize_scalar(value) for value in actual_values}
    normalized_expected = {_normalize_scalar(value) for value in expected_values}

    return bool(normalized_actual & normalized_expected)


def _normalize_scalar(value: Any) -> str:
    """
    将 metadata 标量标准化后比较。
    """

    return str(value).strip().casefold()
