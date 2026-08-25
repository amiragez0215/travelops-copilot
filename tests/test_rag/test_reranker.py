from __future__ import annotations

from collections.abc import Sequence

from app.rag.reranker import (
    CrossEncoderReranker,
    PairScore,
)


class KeywordPairScorer:
    """
    测试用确定性 PairScorer。

    它不下载真实模型。

    评分规则：
        Passage 包含“最相关” → 0.95
        Passage 包含“较相关” → 0.75
        其他 → 0.20
    """

    model_id = "test-keyword-reranker"

    def __init__(self) -> None:
        self.received_pairs: list[
            tuple[str, str]
        ] = []

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[PairScore]:
        self.received_pairs.extend(
            pairs
        )

        output: list[PairScore] = []

        for _, passage in pairs:
            if "最相关" in passage:
                output.append(
                    PairScore(
                        raw_score=3.0,
                        normalized_score=0.95,
                    )
                )

            elif "较相关" in passage:
                output.append(
                    PairScore(
                        raw_score=1.0,
                        normalized_score=0.75,
                    )
                )

            else:
                output.append(
                    PairScore(
                        raw_score=-1.0,
                        normalized_score=0.20,
                    )
                )

        return output


class FailingPairScorer:
    """
    测试 Cross-Encoder 失败后的 fallback。
    """

    model_id = "test-failing-reranker"

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[PairScore]:
        raise RuntimeError(
            "模拟模型推理失败"
        )


def _retrieval_plan():
    """
    构造测试用 RetrievalPlan。
    """

    return {
        "plan_id": "rp_test",
        "mode": "initial",
        "strategy_version": "test",
        "city": "成都",

        "date_range": {
            "start": "2026-07-02",
            "end": "2026-07-04",
        },

        "tasks": [
            {
                "task_id": "guide_main",
                "category": "guides",
                "doc_type": "guide",
                "purpose": "攻略",

                "semantic_query":
                    "成都三日慢节奏美食旅行",

                "keyword_query":
                    "成都 三日游 美食",

                "metadata_filter": {
                    "city": "成都",
                    "doc_type": "guide",
                },

                "required": True,
                "top_k": 5,
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


def _guide_chunks():
    """
    构造测试用 Hybrid Retrieval 结果。

    注意：
        chunk_low 的 RRF 排名最高，
        但 Cross-Encoder 应将 chunk_best 排到第一。
    """

    return [
        {
            "task_id": "guide_main",
            "chunk_id": "chunk_low",
            "content": "这是相关性较低的普通内容。",
            "source": "guides/a.md",

            "metadata": {
                "title": "普通攻略",
                "section": "普通内容",
                "doc_type": "guide",
                "city": "成都",
            },

            "bm25_rank": 1,
            "vector_rank": 1,
            "fusion_score": 0.033,
            "final_rank": 1,

            "retrieval_sources": [
                "bm25",
                "vector",
            ],
        },
        {
            "task_id": "guide_main",
            "chunk_id": "chunk_best",
            "content": "这是最相关的成都美食慢节奏攻略。",
            "source": "guides/b.md",

            "metadata": {
                "title": "成都美食攻略",
                "section": "慢节奏路线",
                "doc_type": "guide",
                "city": "成都",
            },

            "bm25_rank": 3,
            "vector_rank": 2,
            "fusion_score": 0.031,
            "final_rank": 2,

            "retrieval_sources": [
                "bm25",
                "vector",
            ],
        },
        {
            "task_id": "guide_main",
            "chunk_id": "chunk_middle",
            "content": "这是较相关的成都旅行内容。",
            "source": "guides/c.md",

            "metadata": {
                "title": "成都旅行",
                "section": "路线",
                "doc_type": "guide",
                "city": "成都",
            },

            "bm25_rank": 2,
            "vector_rank": 3,
            "fusion_score": 0.030,
            "final_rank": 3,

            "retrieval_sources": [
                "bm25",
                "vector",
            ],
        },
    ]


def test_reranker_uses_cross_encoder_score_as_primary_order():
    """
    Rerank 后应以 Cross-Encoder 分数为主排序，
    而不是继续沿用 RRF 原始排名。
    """

    scorer = KeywordPairScorer()

    reranker = CrossEncoderReranker(
        scorer=scorer,
        top_k_per_task=3,
    )

    output = reranker.rerank_plan(
        retrieval_plan=_retrieval_plan(),
        retrieved_chunks=_guide_chunks(),
    )

    evidence = output[
        "reranked_evidence"
    ]

    assert output["summary"]["status"] == "ok"

    assert [
        item["chunk_id"]
        for item in evidence
    ] == [
        "chunk_best",
        "chunk_middle",
        "chunk_low",
    ]

    assert evidence[0]["rerank_score"] == 0.95
    assert evidence[0]["original_fusion_rank"] == 2
    assert evidence[0]["rerank_rank"] == 1

    # 每个文本对的 query 都来自 RetrievalTask.semantic_query。
    assert all(
        query
        == "成都三日慢节奏美食旅行"
        for query, _ in scorer.received_pairs
    )


def test_reranker_keeps_one_evidence_for_each_hotel():
    """
    酒店任务即使 top_k=1，
    也应确保每个候选酒店至少保留一条评价证据。
    """

    plan = _retrieval_plan()

    plan["tasks"] = [
        {
            "task_id":
                "hotel_review_candidates",

            "category":
                "hotel_reviews",

            "doc_type":
                "hotel_reviews",

            "purpose":
                "酒店评价",

            "semantic_query":
                "安静干净的成都酒店",

            "keyword_query":
                "安静 干净",

            "metadata_filter": {
                "city": "成都",
                "doc_type":
                    "hotel_reviews",

                "hotel_ids": [
                    "hotel_001",
                    "hotel_002",
                ],
            },

            "required": True,
            "top_k": 5,
            "min_required_chunks": 2,
            "priority": 1,
            "reason": "测试",
        }
    ]

    chunks = [
        {
            "task_id":
                "hotel_review_candidates",

            "chunk_id":
                "hotel_001_a",

            "content":
                "最相关：酒店一非常安静干净。",

            "source":
                "hotel_reviews/h1.md",

            "metadata": {
                "doc_type":
                    "hotel_reviews",

                "city":
                    "成都",

                "hotel_id":
                    "hotel_001",

                "hotel_name":
                    "酒店一",
            },

            "fusion_score": 0.033,
            "final_rank": 1,
            "retrieval_sources": [
                "bm25",
                "vector",
            ],
        },
        {
            "task_id":
                "hotel_review_candidates",

            "chunk_id":
                "hotel_001_b",

            "content":
                "较相关：酒店一交通方便。",

            "source":
                "hotel_reviews/h1.md",

            "metadata": {
                "doc_type":
                    "hotel_reviews",

                "city":
                    "成都",

                "hotel_id":
                    "hotel_001",

                "hotel_name":
                    "酒店一",
            },

            "fusion_score": 0.032,
            "final_rank": 2,
            "retrieval_sources": [
                "bm25",
            ],
        },
        {
            "task_id":
                "hotel_review_candidates",

            "chunk_id":
                "hotel_002_a",

            "content":
                "普通内容：酒店二清洁度尚可。",

            "source":
                "hotel_reviews/h2.md",

            "metadata": {
                "doc_type":
                    "hotel_reviews",

                "city":
                    "成都",

                "hotel_id":
                    "hotel_002",

                "hotel_name":
                    "酒店二",
            },

            "fusion_score": 0.030,
            "final_rank": 3,
            "retrieval_sources": [
                "vector",
            ],
        },
    ]

    reranker = CrossEncoderReranker(
        scorer=KeywordPairScorer(),
        top_k_per_task=1,
        min_hotel_evidence_per_hotel=1,
    )

    output = reranker.rerank_plan(
        retrieval_plan=plan,
        retrieved_chunks=chunks,
    )

    evidence = output[
        "reranked_evidence"
    ]

    # 默认 top_k=1，但两家酒店都需要覆盖，所以最终保留两条。
    assert len(evidence) == 2

    covered_hotel_ids = {
        item["metadata"]["hotel_id"]
        for item in evidence
    }

    assert covered_hotel_ids == {
        "hotel_001",
        "hotel_002",
    }


def test_reranker_falls_back_to_rrf_order():
    """
    Cross-Encoder 失败时，应回退到 RRF 原始排名。
    """

    reranker = CrossEncoderReranker(
        scorer=FailingPairScorer(),
        top_k_per_task=2,
        fallback_on_error=True,
    )

    output = reranker.rerank_plan(
        retrieval_plan=_retrieval_plan(),
        retrieved_chunks=_guide_chunks(),
    )

    evidence = output[
        "reranked_evidence"
    ]

    assert (
        output["summary"]["status"]
        == "degraded"
    )

    assert (
        output["summary"]["fallback_used"]
        is True
    )

    assert [
        item["chunk_id"]
        for item in evidence
    ] == [
        "chunk_low",
        "chunk_best",
    ]

    assert evidence[0]["rerank_score"] is None

    assert (
        evidence[0]["rerank_model"]
        == "fallback:rrf_order"
    )


def test_reranker_returns_no_candidates_for_empty_chunks():
    """
    空 retrieved_chunks 是业务状态，不是程序异常。
    """

    reranker = CrossEncoderReranker(
        scorer=KeywordPairScorer(),
    )

    output = reranker.rerank_plan(
        retrieval_plan=_retrieval_plan(),
        retrieved_chunks=[],
    )

    assert output["reranked_evidence"] == []

    assert (
        output["summary"]["status"]
        == "no_candidates"
    )

    assert output["errors"] == []