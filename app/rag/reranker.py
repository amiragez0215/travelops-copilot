from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.schemas.rerank_schema import (
    RerankedEvidence,
    RerankResult,
)
from app.schemas.retrieval_plan_schema import (
    RetrievalPlan,
    RetrievalTask,
)


@dataclass(frozen=True)
class PairScore:
    """
    Cross-Encoder 对一个 Query-Passage 文本对的评分结果。

    raw_score：
        模型直接输出的 logit。

    normalized_score：
        使用 sigmoid 映射到 0 到 1 后的分数。
    """

    raw_score: float
    normalized_score: float


class PairScorer(Protocol):
    """
    Query-Passage 打分器协议。

    CrossEncoderReranker 不直接依赖 transformers，
    而只依赖这个协议。

    正式运行：
        TransformersCrossEncoderScorer

    单元测试：
        FakePairScorer

    这样测试时不需要下载真实模型。
    """

    model_id: str

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[PairScore]:
        """
        对一批 Query-Passage 文本对进行评分。
        """
        ...


class TransformersCrossEncoderScorer:
    """
    基于 Hugging Face Transformers 的 Cross-Encoder 打分器。

    当前默认模型：
        BAAI/bge-reranker-v2-m3

    运行方式：

        Query 和 Passage 不是分别编码。

        而是作为一个文本对同时输入模型：

            [query, passage]

        模型直接判断两段文本的相关性。

    模型只加载一次：

        app/rag/rerank_runtime.py
        会在应用生命周期内复用同一个 scorer 和 model。
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        batch_size: int = 4,
        max_length: int = 512,
        cache_size: int = 2048,
        use_fp16: bool = False,
    ) -> None:
        """
        初始化 Cross-Encoder 模型。

        Args:
            model_id:
                Hugging Face 模型名或本地模型目录。

            device:
                cpu、cuda、mps 或 auto。

            batch_size:
                一次推理多少个 Query-Passage 文本对。

            max_length:
                Query 和 Passage 合并后的最大 token 长度。

            cache_size:
                进程内最多缓存多少个已经计算过的文本对分数。

            use_fp16:
                CUDA 环境下是否使用 FP16。
                CPU 环境不要开启。
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size 必须大于 0"
            )

        if max_length <= 0:
            raise ValueError(
                "max_length 必须大于 0"
            )

        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_size = max(cache_size, 0)

        # 1. 延迟导入大型依赖。
        #    普通单元测试如果注入 FakePairScorer，
        #    就不会执行这里，也不会加载真实模型。
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch

        # 2. 根据配置确定模型运行设备。
        if device == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = device

        self.device = resolved_device

        # 3. 加载 tokenizer。
        #    tokenizer 负责把 Query 和 Passage 转成 token IDs。
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id
        )

        # 4. 加载文本对分类模型。
        #    该模型的输出不是文档向量，而是 Query-Passage 相关性 logit。
        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(model_id)
        )

        # 5. 将模型移动到指定设备。
        self.model.to(self.device)

        # 6. CUDA 环境下可以选择 FP16 以减少显存和提高速度。
        if (
            use_fp16
            and self.device.startswith("cuda")
        ):
            self.model.half()

        # 7. 切换为推理模式。
        #    eval() 会关闭 dropout 等训练行为。
        self.model.eval()

        # 8. 创建进程内 LRU 缓存。
        #    修改工作流或重复查询时，相同 Query-Chunk 不必再次推理。
        self._score_cache: OrderedDict[
            str,
            PairScore,
        ] = OrderedDict()

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[PairScore]:
        """
        批量计算 Query-Passage 相关性。

        执行步骤：

            1. 检查缓存。
            2. 只收集没有计算过的文本对。
            3. 分 batch 执行 Cross-Encoder 推理。
            4. 对 raw logits 执行 sigmoid。
            5. 把分数写入 LRU 缓存。
            6. 按原始输入顺序返回结果。
        """

        if not pairs:
            return []

        results: list[PairScore | None] = [
            None
            for _ in pairs
        ]

        missing_pairs: list[
            tuple[str, str]
        ] = []

        missing_positions: list[int] = []

        # 1. 先检查每一个 Query-Passage 是否已经在缓存中。
        for position, (
            query,
            passage,
        ) in enumerate(pairs):

            cache_key = _pair_cache_key(
                model_id=self.model_id,
                query=query,
                passage=passage,
            )

            cached = self._score_cache.get(
                cache_key
            )

            if cached is not None:
                # 2. 命中缓存后，将记录移到末尾。
                #    OrderedDict 最前面保存最久没有使用的缓存项。
                self._score_cache.move_to_end(
                    cache_key
                )

                results[position] = cached
                continue

            missing_pairs.append(
                (query, passage)
            )

            missing_positions.append(
                position
            )

        # 3. 只对没有命中缓存的文本对执行模型推理。
        missing_scores = self._predict_missing_pairs(
            missing_pairs
        )

        if len(missing_scores) != len(
            missing_pairs
        ):
            raise RuntimeError(
                "Cross-Encoder 返回分数数量与输入文本对数量不一致"
            )

        # 4. 将新计算的分数放回原始位置，并写入缓存。
        for (
            original_position,
            pair,
            pair_score,
        ) in zip(
            missing_positions,
            missing_pairs,
            missing_scores,
        ):
            results[
                original_position
            ] = pair_score

            cache_key = _pair_cache_key(
                model_id=self.model_id,
                query=pair[0],
                passage=pair[1],
            )

            self._put_cache(
                cache_key=cache_key,
                score=pair_score,
            )

        # 5. 理论上每个位置现在都应该有分数。
        if any(
            result is None
            for result in results
        ):
            raise RuntimeError(
                "Cross-Encoder 内部评分结果不完整"
            )

        return [
            result
            for result in results
            if result is not None
        ]

    def _predict_missing_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[PairScore]:
        """
        对缓存中不存在的文本对执行真实模型推理。
        """

        if not pairs:
            return []

        all_scores: list[PairScore] = []

        # 1. 按 batch_size 分批处理，避免一次性占用过多内存。
        for start in range(
            0,
            len(pairs),
            self.batch_size,
        ):
            batch = pairs[
                start : start + self.batch_size
            ]

            queries = [
                query
                for query, _ in batch
            ]

            passages = [
                passage
                for _, passage in batch
            ]

            # 2. tokenizer 同时接收 queries 和 passages。
            #    Cross-Encoder 会让 Query 与 Passage 在同一个 Transformer
            #    中发生联合注意力，而不是分别编码。
            inputs = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # 3. 将 tokenizer 输出移动到模型所在设备。
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            # 4. inference_mode 表示当前只做推理，
            #    不构建梯度计算图，可以减少内存占用。
            with self._torch.inference_mode():
                outputs = self.model(
                    **inputs,
                    return_dict=True,
                )

            logits = outputs.logits

            # 5. 当前 BGE Reranker 每个文本对输出一个 logit。
            #    如果模型输出结构不同，立即报错，避免静默使用错误分数。
            if logits.ndim == 2:
                if logits.shape[1] != 1:
                    raise RuntimeError(
                        "当前 Reranker 期望每个文本对只输出一个 logit，"
                        f"实际 shape={tuple(logits.shape)}"
                    )

                logits = logits[:, 0]

            elif logits.ndim != 1:
                raise RuntimeError(
                    "无法识别 Reranker logits 形状："
                    f"{tuple(logits.shape)}"
                )

            raw_scores = (
                logits
                .float()
                .detach()
                .cpu()
            )

            # 6. 使用 sigmoid 将 logit 映射到 0 到 1。
            #    这个变换不会改变排序，只让分数更容易阅读。
            normalized_scores = (
                self._torch
                .sigmoid(raw_scores)
            )

            # 7. 将 Tensor 转成普通 Python float。
            for (
                raw_score,
                normalized_score,
            ) in zip(
                raw_scores.tolist(),
                normalized_scores.tolist(),
            ):
                all_scores.append(
                    PairScore(
                        raw_score=float(
                            raw_score
                        ),
                        normalized_score=float(
                            normalized_score
                        ),
                    )
                )

        return all_scores

    def _put_cache(
        self,
        cache_key: str,
        score: PairScore,
    ) -> None:
        """
        将评分写入进程内 LRU 缓存。
        """

        if self.cache_size <= 0:
            return

        # 1. 如果 key 已经存在，先删除旧位置。
        self._score_cache.pop(
            cache_key,
            None,
        )

        # 2. 把最新使用的项放到 OrderedDict 末尾。
        self._score_cache[
            cache_key
        ] = score

        # 3. 缓存超过上限时，删除最久没有使用的一项。
        while (
            len(self._score_cache)
            > self.cache_size
        ):
            self._score_cache.popitem(
                last=False
            )


class CrossEncoderReranker:
    """
    RAG Cross-Encoder 精排器。

    输入：
        retrieval_plan
        retrieved_chunks

    输出：
        reranked_evidence
        rerank_result
        errors

    核心规则：

        1. 每个 RetrievalTask 独立精排。
        2. 使用 task.semantic_query 与 chunk.content 构造文本对。
        3. Cross-Encoder 分数是主要排序依据。
        4. fusion_score 只作为分数相同时的辅助排序条件。
        5. 酒店评价需要尽量保留每个候选酒店的证据覆盖。
        6. 模型失败时可以回退到 RRF 原始排名。
    """

    def __init__(
        self,
        scorer: PairScorer,
        top_k_per_task: int = 5,
        min_hotel_evidence_per_hotel: int = 1,
        fallback_on_error: bool = True,
    ) -> None:
        """
        Args:
            scorer:
                Query-Passage 打分器。

            top_k_per_task:
                每个 RetrievalTask 默认保留多少条精排证据。

            min_hotel_evidence_per_hotel:
                hotel_reviews 任务中，每个候选酒店至少保留几条证据。

            fallback_on_error:
                Cross-Encoder 失败时，是否使用 RRF 原始顺序继续。
        """

        if top_k_per_task <= 0:
            raise ValueError(
                "top_k_per_task 必须大于 0"
            )

        if min_hotel_evidence_per_hotel < 0:
            raise ValueError(
                "min_hotel_evidence_per_hotel 不能小于 0"
            )

        self.scorer = scorer
        self.top_k_per_task = top_k_per_task
        self.min_hotel_evidence_per_hotel = (
            min_hotel_evidence_per_hotel
        )
        self.fallback_on_error = fallback_on_error

    def rerank_plan(
        self,
        retrieval_plan: RetrievalPlan | Mapping[str, Any],
        retrieved_chunks: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """
        对完整 RetrievalPlan 的检索结果执行精排。

        每个 task 使用自己的 semantic_query。

        例如：

            guide_main
                使用攻略语义 query 精排 guide chunks。

            hotel_review_candidates
                使用酒店评价语义 query 精排 hotel_review chunks。

            weather_safety
                使用天气安全语义 query 精排 safety chunks。
        """

        plan_model = (
            retrieval_plan
            if isinstance(
                retrieval_plan,
                RetrievalPlan,
            )
            else RetrievalPlan(
                **dict(retrieval_plan)
            )
        )

        normalized_chunks = [
            dict(chunk)
            for chunk in retrieved_chunks
            if isinstance(chunk, Mapping)
        ]

        # 1. 如果 retrieval_plan 没有任务，直接返回 no_tasks。
        if not plan_model.tasks:
            return self._empty_output(
                status="no_tasks",
                task_count=0,
                input_chunk_count=len(
                    normalized_chunks
                ),
            )

        # 2. 如果没有任何检索结果，返回 no_candidates。
        #    这不是程序异常，EvidenceGradeNode 后面会判断证据不足。
        if not normalized_chunks:
            return self._empty_output(
                status="no_candidates",
                task_count=len(
                    plan_model.tasks
                ),
                input_chunk_count=0,
            )

        # 3. 按 task_id 对 retrieved_chunks 分组。
        chunks_by_task: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        unknown_task_chunks: list[
            dict[str, Any]
        ] = []

        valid_task_ids = {
            task.task_id
            for task in plan_model.tasks
        }

        for chunk in normalized_chunks:
            task_id = str(
                chunk.get("task_id") or ""
            )

            if task_id not in valid_task_ids:
                unknown_task_chunks.append(
                    chunk
                )
                continue

            chunks_by_task.setdefault(
                task_id,
                [],
            ).append(chunk)

        reranked_evidence: list[
            dict[str, Any]
        ] = []

        task_summaries: list[
            dict[str, Any]
        ] = []

        issues: list[
            dict[str, Any]
        ] = []

        errors: list[
            dict[str, Any]
        ] = []

        successful_task_count = 0
        fallback_task_count = 0
        no_candidate_task_count = 0
        failed_task_count = 0

        # 4. 每个 RetrievalTask 独立执行精排。
        for task in plan_model.tasks:
            task_chunks = chunks_by_task.get(
                task.task_id,
                [],
            )

            if not task_chunks:
                no_candidate_task_count += 1

                task_summaries.append(
                    {
                        "task_id":
                            task.task_id,

                        "category":
                            task.category,

                        "status":
                            "no_candidates",

                        "input_count":
                            0,

                        "output_count":
                            0,
                    }
                )

                continue

            try:
                task_output = self._rerank_task(
                    task=task,
                    chunks=task_chunks,
                )

                reranked_evidence.extend(
                    task_output["evidence"]
                )

                task_summaries.append(
                    task_output["summary"]
                )

                successful_task_count += 1

            except Exception as exc:
                issue = {
                    "task_id":
                        task.task_id,

                    "type":
                        "reranker_failed",

                    "message":
                        str(exc),
                }

                issues.append(issue)

                # 5. 如果启用了 fallback，
                #    使用 HybridRetrieveNode 的 RRF 原始排名继续。
                if self.fallback_on_error:
                    fallback_output = (
                        self._fallback_task(
                            task=task,
                            chunks=task_chunks,
                            error_message=str(exc),
                        )
                    )

                    reranked_evidence.extend(
                        fallback_output[
                            "evidence"
                        ]
                    )

                    task_summaries.append(
                        fallback_output[
                            "summary"
                        ]
                    )

                    fallback_task_count += 1

                else:
                    failed_task_count += 1
                    errors.append(issue)

                    task_summaries.append(
                        {
                            "task_id":
                                task.task_id,

                            "category":
                                task.category,

                            "status":
                                "failed",

                            "input_count":
                                len(task_chunks),

                            "output_count":
                                0,

                            "error":
                                str(exc),
                        }
                    )

        # 6. 未知 task_id 的 chunks 不参与精排，但记录为 issue。
        if unknown_task_chunks:
            issues.append(
                {
                    "type":
                        "unknown_task_chunks",

                    "count":
                        len(
                            unknown_task_chunks
                        ),

                    "chunk_ids": [
                        chunk.get("chunk_id")
                        for chunk
                        in unknown_task_chunks
                    ],

                    "message":
                        "部分 retrieved_chunks 的 task_id "
                        "不存在于当前 retrieval_plan 中，已跳过。",
                }
            )

        # 7. 根据任务执行结果决定总体状态。
        if (
            failed_task_count > 0
            and not reranked_evidence
        ):
            status = "failed"

        elif fallback_task_count > 0:
            status = "degraded"

        elif (
            failed_task_count > 0
            or no_candidate_task_count > 0
            or unknown_task_chunks
        ):
            status = "partial"

        else:
            status = "ok"

        result = RerankResult(
            status=status,
            model_id=self.scorer.model_id,
            task_count=len(
                plan_model.tasks
            ),
            successful_task_count=(
                successful_task_count
            ),
            fallback_task_count=(
                fallback_task_count
            ),
            no_candidate_task_count=(
                no_candidate_task_count
            ),
            failed_task_count=(
                failed_task_count
            ),
            input_chunk_count=len(
                normalized_chunks
            ),
            output_evidence_count=len(
                reranked_evidence
            ),
            top_k_per_task=(
                self.top_k_per_task
            ),
            min_hotel_evidence_per_hotel=(
                self.min_hotel_evidence_per_hotel
            ),
            fallback_used=(
                fallback_task_count > 0
            ),
            task_summaries=task_summaries,
            issues=issues,
        )

        return {
            "reranked_evidence":
                reranked_evidence,

            "summary":
                result.to_state_dict(),

            "errors":
                errors,
        }

    def _rerank_task(
        self,
        task: RetrievalTask,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        对单个 RetrievalTask 的候选 chunks 进行精排。
        """

        # 1. 根据 chunk_id 去重。
        #    如果意外存在相同 chunk_id，保留 fusion_score 更高的一条。
        unique_chunks = _deduplicate_chunks(
            chunks
        )

        # 2. 为每个 chunk 构造 Cross-Encoder Passage。
        #
        #    Passage 不只是正文，还会加入：
        #    title、section、hotel_name、risk_type 等少量 metadata。
        #
        #    这样模型能够获得更完整的语义上下文。
        passages = [
            build_rerank_passage(chunk)
            for chunk in unique_chunks
        ]

        # 3. 为当前 task 构造 Query-Passage 文本对。
        #
        #    Rerank 使用 semantic_query，
        #    因为它是一条自然语言语义查询。
        pairs = [
            (
                task.semantic_query,
                passage,
            )
            for passage in passages
        ]

        # 4. 批量调用 Cross-Encoder 模型。
        pair_scores = self.scorer.score_pairs(
            pairs
        )

        if len(pair_scores) != len(
            unique_chunks
        ):
            raise RuntimeError(
                "Reranker 返回分数数量与候选 chunk 数量不一致"
            )

        scored_records: list[
            dict[str, Any]
        ] = []

        # 5. 将模型分数与原始检索结果合并。
        for chunk, pair_score in zip(
            unique_chunks,
            pair_scores,
        ):
            scored_records.append(
                {
                    "chunk":
                        chunk,

                    "raw_score":
                        pair_score.raw_score,

                    "normalized_score":
                        pair_score.normalized_score,

                    "selection_reason":
                        "top_rerank_score",
                }
            )

        # 6. 按 Cross-Encoder 分数降序排列。
        #
        #    如果分数相同：
        #    先比较 fusion_score，
        #    再比较 RRF 原始排名，
        #    最后比较 chunk_id，保证排序稳定。
        scored_records.sort(
            key=lambda record: (
                -record[
                    "normalized_score"
                ],
                -_safe_float(
                    record["chunk"].get(
                        "fusion_score"
                    ),
                    default=0.0,
                ),
                _original_rank(
                    record["chunk"]
                ),
                str(
                    record["chunk"].get(
                        "chunk_id"
                    )
                ),
            )
        )

        # 7. 根据 top_k 和酒店覆盖规则选择最终证据。
        selected_records, selection_meta = (
            self._select_records(
                task=task,
                ranked_records=scored_records,
            )
        )

        evidence: list[
            dict[str, Any]
        ] = []

        # 8. 截取最终结果，并恢复文档正文、metadata 和原始检索分数。
        for rerank_rank, record in enumerate(
            selected_records,
            start=1,
        ):
            chunk = record["chunk"]

            evidence_model = RerankedEvidence(
                task_id=task.task_id,
                category=task.category,
                chunk_id=str(
                    chunk["chunk_id"]
                ),
                content=str(
                    chunk.get("content") or ""
                ),
                source=str(
                    chunk.get("source")
                    or "unknown"
                ),
                metadata=dict(
                    chunk.get("metadata")
                    or {}
                ),
                bm25_rank=_safe_optional_int(
                    chunk.get("bm25_rank")
                ),
                vector_rank=_safe_optional_int(
                    chunk.get("vector_rank")
                ),
                original_fusion_rank=(
                    _safe_optional_int(
                        chunk.get(
                            "final_rank"
                        )
                    )
                ),
                fusion_score=_safe_float(
                    chunk.get(
                        "fusion_score"
                    ),
                    default=0.0,
                ),
                retrieval_sources=[
                    str(item)
                    for item in (
                        chunk.get(
                            "retrieval_sources"
                        )
                        or []
                    )
                ],
                rerank_raw_score=round(
                    float(
                        record["raw_score"]
                    ),
                    8,
                ),
                rerank_score=round(
                    float(
                        record[
                            "normalized_score"
                        ]
                    ),
                    8,
                ),
                rerank_rank=rerank_rank,
                rerank_model=(
                    self.scorer.model_id
                ),
                selection_reason=record[
                    "selection_reason"
                ],
            )

            evidence.append(
                evidence_model.to_state_dict()
            )

        selected_scores = [
            item["rerank_score"]
            for item in evidence
            if item["rerank_score"]
            is not None
        ]

        covered_hotel_ids = list(
            dict.fromkeys(
                str(
                    item["metadata"].get(
                        "hotel_id"
                    )
                )
                for item in evidence
                if item[
                    "metadata"
                ].get("hotel_id")
            )
        )

        required_hotel_ids = (
            selection_meta[
                "required_hotel_ids"
            ]
        )

        missing_hotel_ids = [
            hotel_id
            for hotel_id
            in required_hotel_ids
            if hotel_id
            not in covered_hotel_ids
        ]

        summary = {
            "task_id":
                task.task_id,

            "category":
                task.category,

            "status":
                "ok",

            "query_source":
                "semantic_query",

            "input_count":
                len(unique_chunks),

            "output_count":
                len(evidence),

            "effective_top_k":
                selection_meta[
                    "effective_top_k"
                ],

            "highest_score":
                (
                    max(selected_scores)
                    if selected_scores
                    else None
                ),

            "lowest_selected_score":
                (
                    min(selected_scores)
                    if selected_scores
                    else None
                ),

            "required_hotel_ids":
                required_hotel_ids,

            "covered_hotel_ids":
                covered_hotel_ids,

            "missing_hotel_ids":
                missing_hotel_ids,
        }

        return {
            "evidence": evidence,
            "summary": summary,
        }

    def _select_records(
        self,
        task: RetrievalTask,
        ranked_records: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        """
        根据 task 类型选择最终精排结果。

        普通任务：
            直接保留 rerank_score 最高的 top_k。

        酒店评价任务：
            先确保每个候选酒店至少保留一定数量的证据，
            然后再按全局 rerank_score 填充剩余名额。

        为什么酒店要做覆盖保护？

            如果有 8 家候选酒店，而只按全局分数取前 5 条，
            可能 5 条全部来自同一家酒店。

            这样 EvidenceGradeNode 无法判断其他选中/备选酒店，
            ProposalNode 也无法给出完整的酒店补充说明。
        """

        if not ranked_records:
            return [], {
                "effective_top_k": 0,
                "required_hotel_ids": [],
            }

        # 1. 默认 top_k 不超过 RetrievalTask 原始 top_k。
        base_top_k = min(
            self.top_k_per_task,
            task.top_k,
        )

        # 2. required task 至少保留 min_required_chunks 条证据。
        effective_top_k = max(
            base_top_k,
            task.min_required_chunks,
        )

        required_hotel_ids: list[str] = []

        if task.category == "hotel_reviews":
            hotel_ids = (
                task.metadata_filter.get(
                    "hotel_ids"
                )
            )

            if isinstance(hotel_ids, list):
                required_hotel_ids = list(
                    dict.fromkeys(
                        str(hotel_id)
                        for hotel_id
                        in hotel_ids
                        if hotel_id
                    )
                )

        selected: list[
            dict[str, Any]
        ] = []

        selected_chunk_ids: set[str] = set()

        # 3. 酒店评价任务先执行“每家酒店最低证据覆盖”。
        if (
            task.category
            == "hotel_reviews"
            and self.min_hotel_evidence_per_hotel
            > 0
        ):
            for hotel_id in required_hotel_ids:
                hotel_records = [
                    record
                    for record
                    in ranked_records
                    if str(
                        (
                            record["chunk"]
                            .get("metadata")
                            or {}
                        ).get("hotel_id")
                    )
                    == hotel_id
                ]

                for record in hotel_records[
                    :
                    self.min_hotel_evidence_per_hotel
                ]:
                    chunk_id = str(
                        record["chunk"][
                            "chunk_id"
                        ]
                    )

                    if (
                        chunk_id
                        in selected_chunk_ids
                    ):
                        continue

                    coverage_record = dict(
                        record
                    )

                    coverage_record[
                        "selection_reason"
                    ] = "hotel_coverage"

                    selected.append(
                        coverage_record
                    )

                    selected_chunk_ids.add(
                        chunk_id
                    )

        # 4. 如果酒店覆盖所需数量超过默认 top_k，
        #    自动扩大 effective_top_k。
        #
        #    例如 8 家酒店，每家至少 1 条证据，
        #    即使默认 top_k=5，也需要至少保留 8 条。
        effective_top_k = max(
            effective_top_k,
            len(selected),
        )

        effective_top_k = min(
            effective_top_k,
            len(ranked_records),
        )

        # 5. 按全局 rerank_score 顺序填充剩余名额。
        for record in ranked_records:
            if len(selected) >= effective_top_k:
                break

            chunk_id = str(
                record["chunk"]["chunk_id"]
            )

            if chunk_id in selected_chunk_ids:
                continue

            selected.append(record)
            selected_chunk_ids.add(
                chunk_id
            )

        # 6. 最终再按 rerank_score 排序。
        #    hotel_coverage 只影响是否保留，不强制改变相关性排名。
        selected.sort(
            key=lambda record: (
                -record[
                    "normalized_score"
                ],
                -_safe_float(
                    record["chunk"].get(
                        "fusion_score"
                    ),
                    default=0.0,
                ),
                _original_rank(
                    record["chunk"]
                ),
            )
        )

        return selected, {
            "effective_top_k":
                effective_top_k,

            "required_hotel_ids":
                required_hotel_ids,
        }

    def _fallback_task(
        self,
        task: RetrievalTask,
        chunks: list[dict[str, Any]],
        error_message: str,
    ) -> dict[str, Any]:
        """
        Cross-Encoder 失败时使用 RRF 原始排名作为降级结果。

        fallback 不会伪造 Cross-Encoder 分数：

            rerank_raw_score = None
            rerank_score = None

        后续 EvidenceGradeNode 可以通过 rerank_result.status=degraded
        知道当前证据没有经过真实精排。
        """

        unique_chunks = _deduplicate_chunks(
            chunks
        )

        # 1. 按 RRF 原始排名排列。
        unique_chunks.sort(
            key=lambda chunk: (
                _original_rank(chunk),
                -_safe_float(
                    chunk.get(
                        "fusion_score"
                    ),
                    default=0.0,
                ),
                str(
                    chunk.get(
                        "chunk_id"
                    )
                ),
            )
        )

        fallback_records = [
            {
                "chunk":
                    chunk,

                "raw_score":
                    None,

                "normalized_score":
                    None,

                "selection_reason":
                    "fallback_fusion_order",
            }
            for chunk in unique_chunks
        ]

        # 2. fallback 也要遵守酒店覆盖和 top_k 规则。
        selected_records = (
            self._select_fallback_records(
                task=task,
                ranked_records=(
                    fallback_records
                ),
            )
        )

        evidence: list[
            dict[str, Any]
        ] = []

        # 3. 截取 top_k，并恢复文档正文与 metadata。
        for rerank_rank, record in enumerate(
            selected_records,
            start=1,
        ):
            chunk = record["chunk"]

            evidence.append(
                RerankedEvidence(
                    task_id=task.task_id,
                    category=task.category,
                    chunk_id=str(
                        chunk["chunk_id"]
                    ),
                    content=str(
                        chunk.get(
                            "content"
                        )
                        or ""
                    ),
                    source=str(
                        chunk.get(
                            "source"
                        )
                        or "unknown"
                    ),
                    metadata=dict(
                        chunk.get(
                            "metadata"
                        )
                        or {}
                    ),
                    bm25_rank=(
                        _safe_optional_int(
                            chunk.get(
                                "bm25_rank"
                            )
                        )
                    ),
                    vector_rank=(
                        _safe_optional_int(
                            chunk.get(
                                "vector_rank"
                            )
                        )
                    ),
                    original_fusion_rank=(
                        _safe_optional_int(
                            chunk.get(
                                "final_rank"
                            )
                        )
                    ),
                    fusion_score=(
                        _safe_float(
                            chunk.get(
                                "fusion_score"
                            ),
                            default=0.0,
                        )
                    ),
                    retrieval_sources=[
                        str(item)
                        for item in (
                            chunk.get(
                                "retrieval_sources"
                            )
                            or []
                        )
                    ],
                    rerank_raw_score=None,
                    rerank_score=None,
                    rerank_rank=rerank_rank,
                    rerank_model=(
                        "fallback:rrf_order"
                    ),
                    selection_reason=(
                        record[
                            "selection_reason"
                        ]
                    ),
                ).to_state_dict()
            )

        return {
            "evidence":
                evidence,

            "summary": {
                "task_id":
                    task.task_id,

                "category":
                    task.category,

                "status":
                    "fallback",

                "input_count":
                    len(unique_chunks),

                "output_count":
                    len(evidence),

                "effective_top_k":
                    len(evidence),

                "fallback_reason":
                    error_message,
            },
        }

    def _select_fallback_records(
        self,
        task: RetrievalTask,
        ranked_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        在没有 Cross-Encoder 分数时，
        根据 RRF 顺序和酒店覆盖规则选择结果。
        """

        if not ranked_records:
            return []

        target_count = max(
            min(
                self.top_k_per_task,
                task.top_k,
            ),
            task.min_required_chunks,
        )

        selected: list[
            dict[str, Any]
        ] = []

        selected_ids: set[str] = set()

        hotel_ids = (
            task.metadata_filter.get(
                "hotel_ids"
            )
        )

        required_hotel_ids = (
            [
                str(item)
                for item in hotel_ids
                if item
            ]
            if isinstance(
                hotel_ids,
                list,
            )
            else []
        )

        # 1. 先确保酒店覆盖。
        if (
            task.category
            == "hotel_reviews"
            and self.min_hotel_evidence_per_hotel
            > 0
        ):
            for hotel_id in required_hotel_ids:
                matches = [
                    record
                    for record
                    in ranked_records
                    if str(
                        (
                            record["chunk"]
                            .get("metadata")
                            or {}
                        ).get("hotel_id")
                    )
                    == hotel_id
                ]

                for record in matches[
                    :
                    self.min_hotel_evidence_per_hotel
                ]:
                    chunk_id = str(
                        record["chunk"][
                            "chunk_id"
                        ]
                    )

                    if chunk_id in selected_ids:
                        continue

                    copied = dict(record)

                    copied[
                        "selection_reason"
                    ] = (
                        "fallback_hotel_coverage"
                    )

                    selected.append(copied)
                    selected_ids.add(
                        chunk_id
                    )

        target_count = max(
            target_count,
            len(selected),
        )

        target_count = min(
            target_count,
            len(ranked_records),
        )

        # 2. 再按 RRF 原始排名填充。
        for record in ranked_records:
            if len(selected) >= target_count:
                break

            chunk_id = str(
                record["chunk"]["chunk_id"]
            )

            if chunk_id in selected_ids:
                continue

            selected.append(record)
            selected_ids.add(chunk_id)

        # 3. 恢复成 RRF 排名顺序。
        selected.sort(
            key=lambda record: (
                _original_rank(
                    record["chunk"]
                ),
                -_safe_float(
                    record["chunk"].get(
                        "fusion_score"
                    ),
                    default=0.0,
                ),
            )
        )

        return selected

    def _empty_output(
        self,
        status: str,
        task_count: int,
        input_chunk_count: int,
    ) -> dict[str, Any]:
        """
        构造没有任务或没有候选时的标准输出。
        """

        result = RerankResult(
            status=status,  # type: ignore[arg-type]
            model_id=self.scorer.model_id,
            task_count=task_count,
            successful_task_count=0,
            fallback_task_count=0,
            no_candidate_task_count=(
                task_count
                if status
                == "no_candidates"
                else 0
            ),
            failed_task_count=0,
            input_chunk_count=(
                input_chunk_count
            ),
            output_evidence_count=0,
            top_k_per_task=(
                self.top_k_per_task
            ),
            min_hotel_evidence_per_hotel=(
                self.min_hotel_evidence_per_hotel
            ),
            fallback_used=False,
            task_summaries=[],
            issues=[],
        )

        return {
            "reranked_evidence": [],
            "summary": result.to_state_dict(),
            "errors": [],
        }


def build_rerank_passage(
    chunk: Mapping[str, Any],
) -> str:
    """
    为 Cross-Encoder 构造 Passage 文本。

    为什么不只使用 content？

        有些重要信息位于 metadata：

            酒店名称
            section 标题
            risk_type
            scenario
            doc_type

        把这些少量字段放进 Passage，
        可以帮助模型判断 Query 和当前 chunk 的关系。

    注意：
        不加入 chunk_id、source 路径和 hash。
        这些字段对语义相关性没有帮助。
    """

    metadata = chunk.get(
        "metadata"
    )

    metadata = (
        metadata
        if isinstance(
            metadata,
            Mapping,
        )
        else {}
    )

    parts: list[str] = []

    # 1. 加入文档标题。
    title = metadata.get("title")

    if title:
        parts.append(
            f"标题：{title}"
        )

    # 2. 加入当前 Markdown section。
    section = metadata.get("section")

    if section:
        parts.append(
            f"章节：{section}"
        )

    # 3. 酒店评价任务中加入酒店名称。
    hotel_name = metadata.get(
        "hotel_name"
    )

    if hotel_name:
        parts.append(
            f"酒店：{hotel_name}"
        )

    # 4. 安全文档中加入风险类型。
    risk_type = metadata.get(
        "risk_type"
    )

    if risk_type:
        parts.append(
            f"风险类型：{risk_type}"
        )

    # 5. 加入场景信息。
    scenario = metadata.get(
        "scenario"
    )

    if isinstance(scenario, list):
        scenario_text = "、".join(
            str(item)
            for item in scenario
        )
    else:
        scenario_text = (
            str(scenario)
            if scenario
            else ""
        )

    if scenario_text:
        parts.append(
            f"适用场景：{scenario_text}"
        )

    # 6. 最后加入真正的 chunk 正文。
    content = str(
        chunk.get("content")
        or ""
    ).strip()

    parts.append(
        f"正文：{content}"
    )

    return "\n".join(parts)


def _deduplicate_chunks(
    chunks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    根据 chunk_id 去重。

    同一 task 中如果出现重复 chunk_id，
    保留 fusion_score 更高的一条。
    """

    best_by_id: dict[
        str,
        dict[str, Any],
    ] = {}

    for raw_chunk in chunks:
        chunk = dict(raw_chunk)

        chunk_id = str(
            chunk.get("chunk_id")
            or ""
        )

        if not chunk_id:
            raise ValueError(
                "retrieved_chunk 缺少 chunk_id"
            )

        old = best_by_id.get(
            chunk_id
        )

        if old is None:
            best_by_id[
                chunk_id
            ] = chunk
            continue

        old_score = _safe_float(
            old.get("fusion_score"),
            default=0.0,
        )

        new_score = _safe_float(
            chunk.get("fusion_score"),
            default=0.0,
        )

        if new_score > old_score:
            best_by_id[
                chunk_id
            ] = chunk

    return list(
        best_by_id.values()
    )


def _original_rank(
    chunk: Mapping[str, Any],
) -> int:
    """
    读取 HybridRetrieveNode 生成的 RRF 原始排名。
    """

    value = chunk.get(
        "final_rank"
    )

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return 10**9


def _pair_cache_key(
    model_id: str,
    query: str,
    passage: str,
) -> str:
    """
    为 Query-Passage 评分生成 SHA-256 缓存 key。
    """

    payload = (
        f"{model_id}\n"
        f"{query}\n"
        f"{passage}"
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def _safe_float(
    value: Any,
    default: float,
) -> float:
    """
    安全转换 float。
    """

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _safe_optional_int(
    value: Any,
) -> int | None:
    """
    安全转换可空 int。
    """

    if value is None:
        return None

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return None