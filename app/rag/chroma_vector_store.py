from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import chromadb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag.embedding_text import EMBEDDING_TEXT_VERSION, build_embedding_text


@dataclass(frozen=True)
class VectorHit:
    """
    Chroma Vector Search 的一条结果。
    """

    document: Document
    rank: int
    distance: float | None
    score: float | None


@dataclass(frozen=True)
class ChromaSyncReport:
    """
    Markdown chunks 与 Chroma 同步后的摘要。
    """

    collection_name: str
    desired_count: int
    existing_count_before: int
    added_count: int
    updated_count: int
    unchanged_count: int
    deleted_count: int
    final_count: int

    def to_dict(self) -> dict[str, Any]:
        """转成普通 dict，方便脚本打印和测试断言。"""

        return asdict(self)


class ChromaVectorStore:
    """
    持久化 Chroma 向量库封装。

    设计目标：
        1. 文档向量保存到磁盘，应用重启后直接加载。
        2. chunk_id 作为 Chroma record ID，保证更新和删除可控。
        3. chunk_hash 未变化时不重新 Embedding。
        4. 支持按 source、document_hash、chunk_id 删除。
        5. 查询时只编码 query，不重新编码文档。
    """

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str,
        embeddings: Embeddings,
        embedding_model_id: str,
        embedding_text_version: str = EMBEDDING_TEXT_VERSION,
    ) -> None:
        """
        创建或加载持久化 Chroma collection。

        Args:
            persist_directory:
                Chroma 数据库存储目录，例如 data/vector_store/chroma。

            collection_name:
                Collection 名称，例如 travelops_rag_v1。

            embeddings:
                LangChain Embeddings 对象。
                本类自行调用 embed_documents / embed_query，并把向量显式传给 Chroma。

            embedding_model_id:
                当前 Embedding 模型标识，用于防止用不同模型误读同一个 collection。
        """

        # 1. 创建持久化目录。
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.embeddings = embeddings
        self.embedding_model_id = embedding_model_id
        self.embedding_text_version = embedding_text_version

        # 2. PersistentClient 会自动把 Chroma 数据保存到本地，并在下次启动时加载。
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

        # 3. 创建 collection 时不绑定 Chroma 自带 Embedding Function。
        #    我们显式提供 embeddings，确保构建和查询都使用同一个项目 Embeddings 实现。
        self.collection: Any = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={
                "description": "TravelOps-Copilot RAG chunks",
                "embedding_model_id": embedding_model_id,
                "embedding_text_version": embedding_text_version,
            },
        )

        # 4. 如果 collection 已经存在，检查它最初使用的模型是否与当前配置一致。
        collection_metadata = self.collection.metadata or {}
        stored_model_id = collection_metadata.get("embedding_model_id")

        if stored_model_id and stored_model_id != embedding_model_id:
            raise ValueError(
                "Chroma collection 的 embedding_model_id 与当前配置不一致: "
                f"stored={stored_model_id}, current={embedding_model_id}. "
                "请更换 collection_name 或重建 collection。"
            )

    def sync_documents(
        self,
        documents: list[Document],
        delete_missing: bool = True,
    ) -> ChromaSyncReport:
        """
        将去重后的同一批 chunks 增量同步到 Chroma。

        同步规则：
            - 新 chunk_id：计算 Embedding 后 upsert。
            - 已存在但 chunk_hash 或 embedding_text_version 改变：重新编码并 upsert。
            - chunk_hash 未变化：跳过，不重新编码。
            - Chroma 中存在但当前 Markdown 已不存在：delete_missing=True 时删除。

        因此应用每次启动都可以安全调用 sync_documents()：
            没有文档变化时，不会重复生成文档向量。
        """

        # 1. 验证输入并建立 desired chunk_id → Document 映射。
        desired: dict[str, Document] = {}

        for document in documents:
            chunk_id = _required_metadata(document, "chunk_id")
            _required_metadata(document, "chunk_hash")
            _required_metadata(document, "document_hash")
            _required_metadata(document, "source")

            if chunk_id in desired:
                raise ValueError(f"同步语料中存在重复 chunk_id: {chunk_id}")

            desired[chunk_id] = document

        # 2. 分页读取 Chroma 中现有 record 的 metadata。
        existing_records = self._get_all_records(include=["metadatas"])
        existing_ids = [str(item) for item in existing_records.get("ids", [])]
        existing_metadatas = existing_records.get("metadatas") or []

        existing_hash_by_id: dict[str, str] = {}
        existing_embedding_text_version_by_id: dict[str, str] = {}

        for record_id, metadata in zip(existing_ids, existing_metadatas):
            metadata = metadata or {}
            existing_hash_by_id[record_id] = str(metadata.get("chunk_hash") or "")
            existing_embedding_text_version_by_id[record_id] = str(
                metadata.get("embedding_text_version") or ""
            )

        existing_id_set = set(existing_ids)
        desired_id_set = set(desired)

        # 3. 找出新增、更新、未变化和已经失效的 chunk。
        added_ids = sorted(desired_id_set - existing_id_set)
        common_ids = desired_id_set & existing_id_set

        updated_ids = sorted(
            chunk_id
            for chunk_id in common_ids
            if (
                existing_hash_by_id.get(chunk_id)
                != str(desired[chunk_id].metadata.get("chunk_hash") or "")
                or existing_embedding_text_version_by_id.get(chunk_id)
                != self.embedding_text_version
            )
        )

        unchanged_ids = sorted(common_ids - set(updated_ids))
        stale_ids = sorted(existing_id_set - desired_id_set) if delete_missing else []

        # 4. 只对新增、正文变化或 Embedding 文本格式变化的 chunks 编码。
        upsert_ids = added_ids + updated_ids

        if upsert_ids:
            upsert_documents = [desired[chunk_id] for chunk_id in upsert_ids]
            # Keep raw source bodies in Chroma for retrieval display and
            # Proposal citations; only the vectors use the enriched text.
            page_contents = [document.page_content for document in upsert_documents]
            embedding_texts = [
                build_embedding_text(document) for document in upsert_documents
            ]

            document_vectors = self.embeddings.embed_documents(embedding_texts)

            # 5. 使用 Chroma upsert：存在则更新，不存在则新增。
            self.collection.upsert(
                ids=upsert_ids,
                embeddings=document_vectors,
                documents=page_contents,
                metadatas=[
                    _sanitize_metadata_for_chroma(
                        {
                            **document.metadata,
                            "embedding_text_version": self.embedding_text_version,
                        }
                    )
                    for document in upsert_documents
                ],
            )

        # 6. 删除已经从源 Markdown 语料中消失的 chunks。
        if stale_ids:
            self.collection.delete(ids=stale_ids)

        final_count = self.collection.count()

        return ChromaSyncReport(
            collection_name=self.collection_name,
            desired_count=len(desired),
            existing_count_before=len(existing_ids),
            added_count=len(added_ids),
            updated_count=len(updated_ids),
            unchanged_count=len(unchanged_ids),
            deleted_count=len(stale_ids),
            final_count=final_count,
        )

    def search(
        self,
        query: str,
        candidate_ids: Iterable[str],
        k: int,
    ) -> list[VectorHit]:
        """
        在指定 candidate_ids 中执行 Chroma Vector Search。

        为什么不用每次根据 metadata 重新创建向量库？
            文档向量已经持久化在 Chroma 中。
            这里只把 metadata filter 转换后的 chunk_id 列表作为 ids 约束传给 Chroma。
            查询时只编码 semantic_query。
        """

        # 1. 规范化候选 ID，并去重。
        unique_candidate_ids = list(dict.fromkeys(str(item) for item in candidate_ids))

        if not unique_candidate_ids or k <= 0 or not query.strip():
            return []

        # 2. Chroma 中可能因为管理员删除而缺少某些候选 ID；先取交集避免无效查询。
        existing = self.collection.get(ids=unique_candidate_ids)
        existing_ids = [str(item) for item in existing.get("ids", [])]

        if not existing_ids:
            return []

        # 3. 只编码当前 semantic_query；不会重新编码任何文档。
        query_vector = self.embeddings.embed_query(query)

        # 4. 使用 ids 参数把向量搜索限制在 metadata 已匹配的候选 chunks 中。
        result = self.collection.query(
            query_embeddings=[query_vector],
            ids=existing_ids,
            n_results=min(k, len(existing_ids)),
            include=["documents", "metadatas", "distances"],
        )

        result_ids = (result.get("ids") or [[]])[0]
        result_documents = (result.get("documents") or [[]])[0]
        result_metadatas = (result.get("metadatas") or [[]])[0]
        result_distances = (result.get("distances") or [[]])[0]

        hits: list[VectorHit] = []

        # 5. 截取 top_k，并恢复文档正文与 metadata。
        for rank, (chunk_id, content, metadata, distance) in enumerate(
            zip(result_ids, result_documents, result_metadatas, result_distances),
            start=1,
        ):
            numeric_distance = float(distance) if distance is not None else None

            # 6. vector_score 只用于调试展示；RRF 真正使用的是 vector_rank。
            #    距离越小越相似，所以转换成越大越好的简单分数。
            score = (
                1.0 / (1.0 + max(numeric_distance, 0.0))
                if numeric_distance is not None
                else None
            )

            restored_metadata = dict(metadata or {})
            restored_metadata.setdefault("chunk_id", str(chunk_id))

            hits.append(
                VectorHit(
                    document=Document(
                        page_content=str(content or ""),
                        metadata=restored_metadata,
                    ),
                    rank=rank,
                    distance=numeric_distance,
                    score=score,
                )
            )

        return hits

    def delete_by_source(self, source: str) -> int:
        """
        删除某个 Markdown source 对应的全部向量、正文和 metadata。

        注意：
            如果源 Markdown 文件仍然存在，下一次应用启动同步时会重新加入。
            永久删除需要同时删除或移动源 Markdown 文件。
        """

        return self._delete_where({"source": source})

    def delete_by_document_hash(self, document_hash: str) -> int:
        """根据 SHA-256 document_hash 删除整篇文档的全部 chunks。"""

        return self._delete_where({"document_hash": document_hash})

    def delete_by_chunk_ids(self, chunk_ids: Iterable[str]) -> int:
        """根据 chunk_id 删除指定向量记录。"""

        ids = list(dict.fromkeys(str(item) for item in chunk_ids))

        if not ids:
            return 0

        existing = self.collection.get(ids=ids)
        existing_ids = [str(item) for item in existing.get("ids", [])]

        if not existing_ids:
            return 0

        self.collection.delete(ids=existing_ids)
        return len(existing_ids)

    def count(self) -> int:
        """返回当前 collection 中的 chunk 数量。"""

        return self.collection.count()

    def reset_collection(self) -> None:
        """
        删除整个 collection 并重新创建。

        这是破坏性操作，只用于开发阶段强制重建索引。
        """

        # 1. 删除 collection 中的全部向量、正文和 metadata。
        self.client.delete_collection(name=self.collection_name)

        # 2. 使用相同配置重新创建一个空 collection。
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={
                "description": "TravelOps-Copilot RAG chunks",
                "embedding_model_id": self.embedding_model_id,
            },
        )

    def _delete_where(self, where: Mapping[str, Any]) -> int:
        """
        先统计匹配记录，再使用 Chroma where 删除。
        """

        # 1. 使用相同 where 查询待删除 IDs，便于返回准确数量。
        matched = self.collection.get(where=dict(where))
        ids = [str(item) for item in matched.get("ids", [])]

        if not ids:
            return 0

        # 2. 根据 Chroma metadata filter 删除所有匹配记录。
        self.collection.delete(where=dict(where))
        return len(ids)

    def _get_all_records(
        self,
        include: list[str],
        batch_size: int = 500,
    ) -> dict[str, list[Any]]:
        """
        分页读取 collection 全部记录，避免将来语料变大时一次性 get 过多数据。
        """

        all_ids: list[Any] = []
        all_metadatas: list[Any] = []
        all_documents: list[Any] = []
        offset = 0

        while True:
            batch = self.collection.get(
                limit=batch_size,
                offset=offset,
                include=include,
            )

            batch_ids = batch.get("ids", [])

            if not batch_ids:
                break

            all_ids.extend(batch_ids)

            if "metadatas" in include:
                all_metadatas.extend(batch.get("metadatas") or [])

            if "documents" in include:
                all_documents.extend(batch.get("documents") or [])

            if len(batch_ids) < batch_size:
                break

            offset += batch_size

        result: dict[str, list[Any]] = {"ids": all_ids}

        if "metadatas" in include:
            result["metadatas"] = all_metadatas

        if "documents" in include:
            result["documents"] = all_documents

        return result


def _sanitize_metadata_for_chroma(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """
    将 LangChain Document metadata 转成 Chroma 支持的类型。

    当前 Chroma 支持字符串、整数、浮点数、布尔值，以及同类型的非空数组。
    dict 等复杂值会转换成 JSON 字符串。
    """

    import json

    sanitized: dict[str, Any] = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            sanitized[str(key)] = value
            continue

        if isinstance(value, (list, tuple)):
            cleaned = [item for item in value if item is not None]

            # Chroma 不允许空数组 metadata，因此直接跳过。
            if not cleaned:
                continue

            # 保证数组元素类型一致；否则保存成 JSON 字符串。
            element_types = {type(item) for item in cleaned}

            if len(element_types) == 1 and next(iter(element_types)) in {
                str,
                int,
                float,
                bool,
            }:
                sanitized[str(key)] = list(cleaned)
            else:
                sanitized[str(key)] = json.dumps(
                    cleaned,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            continue

        sanitized[str(key)] = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    return sanitized


def _required_metadata(document: Document, key: str) -> str:
    """读取必须存在的 metadata 字段。"""

    value = document.metadata.get(key)

    if value is None or not str(value).strip():
        raise ValueError(f"RAG Document 缺少 metadata.{key}")

    return str(value)
