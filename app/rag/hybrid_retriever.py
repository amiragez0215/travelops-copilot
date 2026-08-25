from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag.bm25_index import BM25Hit, InMemoryBM25Index
from app.rag.chroma_vector_store import (
    ChromaSyncReport,
    ChromaVectorStore,
    VectorHit,
)
from app.rag.embedding_text import EMBEDDING_TEXT_VERSION
from app.rag.document_loader import (
    DEFAULT_RAG_DOCS_DIR,
    DeduplicationResult,
    deduplicate_documents,
    load_rag_documents,
)
from app.rag.metadata_filter import matching_chunk_ids
from app.schemas.retrieval_plan_schema import RetrievalPlan, RetrievalTask
from app.schemas.retrieval_result_schema import (
    HybridRetrievalResult,
    RetrievedChunk,
)


@dataclass(frozen=True)
class HybridStartupReport:
    """
    应用启动时 RAG 索引初始化摘要。
    """

    source_document_count: int
    active_chunk_count: int
    duplicate_source_count: int
    bm25_build_count: int
    chroma_sync: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转成可打印 dict。"""

        return {
            "source_document_count": self.source_document_count,
            "active_chunk_count": self.active_chunk_count,
            "duplicate_source_count": self.duplicate_source_count,
            "bm25_build_count": self.bm25_build_count,
            "chroma_sync": self.chroma_sync,
        }


class HybridRetriever:
    """
    BM25 + Chroma Vector + RRF 混合检索器。

    生命周期设计：
        1. 应用启动时加载并切分 Markdown。
        2. 根据 SHA-256 document_hash 做文档级去重。
        3. 用去重后的同一批 chunks 构建一次内存 BM25。
        4. 将同一批 chunks 增量同步到持久化 Chroma。
        5. 后续每次 workflow 只执行 query，不重复构建 BM25，也不重复编码未变化文档。
    """

    def __init__(
        self,
        documents: list[Document],
        bm25_index: InMemoryBM25Index,
        vector_store: ChromaVectorStore,
        rag_docs_dir: str | Path,
        startup_report: HybridStartupReport,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        fetch_multiplier: int = 2,
    ) -> None:
        """
        Args:
            documents:
                去重后同时供 BM25 和 Chroma 使用的 chunks。

            bm25_index:
                应用启动时已经构建好的内存 BM25。

            vector_store:
                持久化 Chroma 向量库。

            rag_docs_dir:
                Markdown 根目录，用于删除源文件或重新同步。
        """

        if rrf_k <= 0:
            raise ValueError("rrf_k 必须大于 0")

        self.documents = list(documents)
        self.bm25_index = bm25_index
        self.vector_store = vector_store
        self.rag_docs_dir = Path(rag_docs_dir)
        self.startup_report = startup_report
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.fetch_multiplier = max(fetch_multiplier, 1)

    @classmethod
    def from_directory(
        cls,
        root_dir: str | Path | None,
        persist_directory: str | Path,
        collection_name: str,
        embeddings: Embeddings,
        embedding_model_id: str,
        embedding_text_version: str = EMBEDDING_TEXT_VERSION,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        fetch_multiplier: int = 2,
        reset_collection: bool = False,
    ) -> "HybridRetriever":
        """
        从 Markdown 目录创建应用级 HybridRetriever。

        这是应用启动时调用的核心工厂方法。
        """

        active_root = Path(root_dir) if root_dir else DEFAULT_RAG_DOCS_DIR

        # 1. 加载并切分所有 Markdown 文档。
        loaded_documents = load_rag_documents(
            root_dir=active_root,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # 2. 根据 SHA-256 document_hash 做文档级去重。
        dedup_result = deduplicate_documents(loaded_documents)
        active_documents = dedup_result.documents

        if not active_documents:
            raise ValueError("RAG 文档去重后没有可用 chunks")

        # 3. 创建或加载持久化 Chroma collection。
        vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embeddings=embeddings,
            embedding_model_id=embedding_model_id,
            embedding_text_version=embedding_text_version,
        )

        if reset_collection:
            vector_store.reset_collection()

        # 4. 增量同步 Chroma；未变化的 chunk 不会重新 Embedding。
        chroma_sync = vector_store.sync_documents(
            active_documents,
            delete_missing=True,
        )

        # 5. 根据与 Chroma 相同的 active_documents 构建一次 BM25。
        bm25_index = InMemoryBM25Index(active_documents)

        startup_report = HybridStartupReport(
            source_document_count=len(
                {str(document.metadata["source"]) for document in loaded_documents}
            ),
            active_chunk_count=len(active_documents),
            duplicate_source_count=len(dedup_result.duplicates),
            bm25_build_count=bm25_index.build_count,
            chroma_sync=chroma_sync.to_dict(),
        )

        return cls(
            documents=active_documents,
            bm25_index=bm25_index,
            vector_store=vector_store,
            rag_docs_dir=active_root,
            startup_report=startup_report,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            fetch_multiplier=fetch_multiplier,
        )

    def retrieve_task(
        self,
        task: RetrievalTask | Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        执行一条 RetrievalTask。

        处理步骤：
            1. 根据 metadata_filter 计算同一批 candidate chunk IDs。
            2. BM25 使用 keyword_query 搜索。
            3. Chroma 使用 semantic_query 搜索。
            4. RRF 根据两份排名融合。
            5. 截取 task.top_k。
        """

        # 1. 将普通 dict 校验为 RetrievalTask。
        task_model = task if isinstance(task, RetrievalTask) else RetrievalTask(**dict(task))

        # 2. 在同一批应用级 chunks 上执行 metadata 预过滤。
        candidate_ids = matching_chunk_ids(
            documents=self.documents,
            metadata_filter=task_model.metadata_filter,
        )

        if not candidate_ids:
            return {
                "chunks": [],
                "summary": {
                    "task_id": task_model.task_id,
                    "category": task_model.category,
                    "status": "no_results",
                    "metadata_matched_documents": 0,
                    "bm25_result_count": 0,
                    "vector_result_count": 0,
                    "fused_result_count": 0,
                },
            }

        # 3. 每个检索器先多取一些结果，再交给 RRF 融合。
        fetch_k = min(
            max(task_model.top_k * self.fetch_multiplier, task_model.top_k),
            len(candidate_ids),
        )

        # 4. BM25 使用 keyword_query；索引对象在应用启动时已构建并复用。
        bm25_hits = self.bm25_index.search(
            query=task_model.keyword_query,
            candidate_ids=candidate_ids,
            k=fetch_k,
        )

        # 5. Chroma 使用 semantic_query；只编码 query，文档向量从磁盘数据库读取。
        vector_hits = self.vector_store.search(
            query=task_model.semantic_query,
            candidate_ids=candidate_ids,
            k=fetch_k,
        )

        # 6. 用 RRF 融合 BM25 排名和 Vector 排名。
        fused_chunks = reciprocal_rank_fusion(
            task_id=task_model.task_id,
            bm25_hits=bm25_hits,
            vector_hits=vector_hits,
            rrf_k=self.rrf_k,
            bm25_weight=self.bm25_weight,
            vector_weight=self.vector_weight,
            top_k=task_model.top_k,
        )

        return {
            "chunks": [chunk.to_state_dict() for chunk in fused_chunks],
            "summary": {
                "task_id": task_model.task_id,
                "category": task_model.category,
                "status": "ok" if fused_chunks else "no_results",
                "metadata_matched_documents": len(candidate_ids),
                "bm25_result_count": len(bm25_hits),
                "vector_result_count": len(vector_hits),
                "fused_result_count": len(fused_chunks),
            },
        }

    def retrieve_plan(
        self,
        retrieval_plan: RetrievalPlan | Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        执行完整 RetrievalPlan，并允许单个 task 失败后继续其他任务。
        """

        # 1. 将 State 中的 dict 校验成 RetrievalPlan。
        plan_model = (
            retrieval_plan
            if isinstance(retrieval_plan, RetrievalPlan)
            else RetrievalPlan(**dict(retrieval_plan))
        )

        # 2. 空任务计划不是程序异常，例如 repair 已耗尽时会出现。
        if not plan_model.tasks:
            summary = HybridRetrievalResult(
                status="no_tasks",
                strategy_version="bm25_memory_chroma_rrf_v2",
                corpus_size=len(self.documents),
                task_count=0,
                successful_task_count=0,
                no_result_task_count=0,
                failed_task_count=0,
                total_retrieved_chunks=0,
                rrf_k=self.rrf_k,
                weights={"bm25": self.bm25_weight, "vector": self.vector_weight},
                task_summaries=[],
            )

            return {
                "retrieved_chunks": [],
                "summary": summary.to_state_dict(),
                "errors": [],
            }

        all_chunks: list[dict[str, Any]] = []
        task_summaries: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        successful_task_count = 0
        no_result_task_count = 0
        failed_task_count = 0

        # 3. 按 RetrievalPlanNode 生成的任务顺序逐条执行。
        for task in plan_model.tasks:
            try:
                task_result = self.retrieve_task(task)
                chunks = task_result["chunks"]
                summary = task_result["summary"]
                all_chunks.extend(chunks)
                task_summaries.append(summary)

                if chunks:
                    successful_task_count += 1
                else:
                    no_result_task_count += 1

            except Exception as exc:
                failed_task_count += 1
                errors.append(
                    {
                        "task_id": task.task_id,
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    }
                )
                task_summaries.append(
                    {
                        "task_id": task.task_id,
                        "category": task.category,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

        # 4. 根据任务执行情况计算总体状态。
        if successful_task_count == len(plan_model.tasks):
            status = "ok"
        elif successful_task_count > 0:
            status = "partial"
        elif failed_task_count > 0:
            status = "failed"
        else:
            status = "no_results"

        summary = HybridRetrievalResult(
            status=status,
            strategy_version="bm25_memory_chroma_rrf_v2",
            corpus_size=len(self.documents),
            task_count=len(plan_model.tasks),
            successful_task_count=successful_task_count,
            no_result_task_count=no_result_task_count,
            failed_task_count=failed_task_count,
            total_retrieved_chunks=len(all_chunks),
            rrf_k=self.rrf_k,
            weights={"bm25": self.bm25_weight, "vector": self.vector_weight},
            task_summaries=task_summaries,
        )

        return {
            "retrieved_chunks": all_chunks,
            "summary": summary.to_state_dict(),
            "errors": errors,
        }

    def sync_from_disk(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ) -> dict[str, Any]:
        """
        管理员修改 RAG Markdown 后，重新同步 Chroma 并重建一次 BM25。

        这不是每个 workflow 请求都会执行的函数。
        """

        # 1. 重新加载源 Markdown。
        loaded_documents = load_rag_documents(
            root_dir=self.rag_docs_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # 2. 使用相同 SHA-256 规则做文档级去重。
        dedup_result = deduplicate_documents(loaded_documents)
        active_documents = dedup_result.documents

        # 3. 增量同步 Chroma；只编码新增或变化 chunks。
        sync_report = self.vector_store.sync_documents(
            active_documents,
            delete_missing=True,
        )

        # 4. 替换应用内存语料，并仅在这次管理动作中重建 BM25。
        self.documents = active_documents
        self.bm25_index.rebuild(active_documents)

        return {
            "active_chunk_count": len(active_documents),
            "duplicate_source_count": len(dedup_result.duplicates),
            "bm25_build_count": self.bm25_index.build_count,
            "chroma_sync": sync_report.to_dict(),
        }

    def delete_source(
        self,
        source: str,
        delete_source_file: bool = False,
    ) -> dict[str, Any]:
        """
        删除某个 source 对应的 Chroma 向量，并同步更新内存 BM25。

        Args:
            source:
                相对于 data/rag_docs 的路径，例如 hotel_reviews/hotel_001.md。

            delete_source_file:
                False：只删 Chroma 和当前内存索引；应用重启同步时会重新加入。
                True：同时删除 Markdown 源文件，删除会跨应用重启保持。
        """

        # 1. 从 Chroma 持久化 collection 删除该 source 的全部 chunks。
        deleted_vector_count = self.vector_store.delete_by_source(source)

        # 2. 从应用内存语料中移除同一 source。
        remaining_documents = [
            document
            for document in self.documents
            if str(document.metadata.get("source")) != source
        ]

        removed_memory_count = len(self.documents) - len(remaining_documents)
        self.documents = remaining_documents

        # 3. 删除后重建一次 BM25，保证 BM25 和 Chroma 使用同一批 chunks。
        self.bm25_index.rebuild(remaining_documents)

        source_file_deleted = False

        # 4. 只有显式要求时才删除 Markdown 源文件。
        if delete_source_file:
            source_path = (self.rag_docs_dir / source).resolve()
            root_path = self.rag_docs_dir.resolve()

            # 5. 防止 ../ 路径越界删除 RAG 目录外的文件。
            if root_path not in source_path.parents:
                raise ValueError(f"非法 source 路径，不能删除 RAG 目录外文件: {source}")

            if source_path.exists():
                source_path.unlink()
                source_file_deleted = True

        return {
            "source": source,
            "deleted_vector_count": deleted_vector_count,
            "removed_memory_chunk_count": removed_memory_count,
            "source_file_deleted": source_file_deleted,
            "persistent_across_restart": delete_source_file,
            "bm25_build_count": self.bm25_index.build_count,
        }


def reciprocal_rank_fusion(
    task_id: str,
    bm25_hits: list[BM25Hit],
    vector_hits: list[VectorHit],
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
    top_k: int = 8,
) -> list[RetrievedChunk]:
    """
    使用加权 Reciprocal Rank Fusion 融合 BM25 和 Vector 排名。

    公式：
        score(d) = bm25_weight / (rrf_k + bm25_rank)
                 + vector_weight / (rrf_k + vector_rank)
    """

    fused: dict[str, dict[str, Any]] = {}

    # 1. 累加 BM25 排名贡献。
    for hit in bm25_hits:
        chunk_id = _chunk_id(hit.document)
        entry = fused.setdefault(chunk_id, _new_fusion_entry(hit.document))
        entry["bm25_rank"] = hit.rank
        entry["bm25_score"] = hit.score
        entry["fusion_score"] += bm25_weight / (rrf_k + hit.rank)
        entry["retrieval_sources"].append("bm25")

    # 2. 累加 Vector 排名贡献。
    for hit in vector_hits:
        chunk_id = _chunk_id(hit.document)
        entry = fused.setdefault(chunk_id, _new_fusion_entry(hit.document))
        entry["vector_rank"] = hit.rank
        entry["vector_score"] = hit.score
        entry["fusion_score"] += vector_weight / (rrf_k + hit.rank)
        entry["retrieval_sources"].append("vector")

    # 3. 按融合分数降序；分数相同时用最好单路排名和 chunk_id 保证稳定排序。
    ranked_entries = sorted(
        fused.values(),
        key=lambda entry: (
            -entry["fusion_score"],
            _best_rank(entry),
            entry["chunk_id"],
        ),
    )

    output: list[RetrievedChunk] = []

    # 4. 截取 top_k，并恢复文档正文与 metadata。
    for final_rank, entry in enumerate(ranked_entries[:top_k], start=1):
        output.append(
            RetrievedChunk(
                task_id=task_id,
                chunk_id=entry["chunk_id"],
                content=entry["document"].page_content,
                source=str(entry["document"].metadata.get("source", "unknown")),
                metadata=dict(entry["document"].metadata),
                bm25_score=(
                    round(entry["bm25_score"], 8)
                    if entry["bm25_score"] is not None
                    else None
                ),
                vector_score=(
                    round(entry["vector_score"], 8)
                    if entry["vector_score"] is not None
                    else None
                ),
                bm25_rank=entry["bm25_rank"],
                vector_rank=entry["vector_rank"],
                fusion_score=round(entry["fusion_score"], 8),
                final_rank=final_rank,
                retrieval_sources=list(dict.fromkeys(entry["retrieval_sources"])),
            )
        )

    return output


def _new_fusion_entry(document: Document) -> dict[str, Any]:
    """创建一条 RRF 临时记录。"""

    return {
        "chunk_id": _chunk_id(document),
        "document": document,
        "bm25_rank": None,
        "vector_rank": None,
        "bm25_score": None,
        "vector_score": None,
        "fusion_score": 0.0,
        "retrieval_sources": [],
    }


def _best_rank(entry: Mapping[str, Any]) -> int | float:
    """读取一条结果在两种检索器中的最好排名。"""

    ranks = [
        rank
        for rank in [entry.get("bm25_rank"), entry.get("vector_rank")]
        if isinstance(rank, int)
    ]

    return min(ranks) if ranks else math.inf


def _chunk_id(document: Document) -> str:
    """读取稳定 chunk_id。"""

    chunk_id = document.metadata.get("chunk_id")

    if not chunk_id:
        raise ValueError("RAG Document 缺少 metadata.chunk_id")

    return str(chunk_id)
