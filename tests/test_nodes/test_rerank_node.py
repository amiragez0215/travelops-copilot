import importlib
from collections.abc import Sequence

from app.nodes.rerank_node import (
    rerank_node,
)
from app.rag.reranker import (
    CrossEncoderReranker,
    PairScore,
)


# app.nodes 的 package namespace 也导出了同名 rerank_node 函数；使用
# import_module 明确拿到模块对象，才能在测试中替换其运行时依赖。
rerank_module = importlib.import_module("app.nodes.rerank_node")


class NodeTestScorer:
    """
    Node 测试使用的确定性 scorer。
    """

    model_id = "node-test-reranker"

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[PairScore]:
        return [
            PairScore(
                raw_score=float(index),
                normalized_score=(
                    0.9
                    if "推荐内容" in passage
                    else 0.3
                ),
            )
            for index, (_, passage)
            in enumerate(pairs)
        ]


def _state():
    """
    构造 RerankNode 所需 State。
    """

    return {
        "retrieval_plan": {
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
                        "成都三日美食攻略",

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
        },

        "retrieved_chunks": [
            {
                "task_id": "guide_main",
                "chunk_id": "chunk_001",
                "content": "普通内容。",
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
                "chunk_id": "chunk_002",
                "content": "推荐内容：成都美食与茶馆。",
                "source": "guides/b.md",

                "metadata": {
                    "title": "成都美食攻略",
                    "section": "茶馆",
                    "doc_type": "guide",
                    "city": "成都",
                },

                "bm25_rank": 2,
                "vector_rank": 2,
                "fusion_score": 0.031,
                "final_rank": 2,

                "retrieval_sources": [
                    "bm25",
                    "vector",
                ],
            },
        ],
    }


def _reranker():
    """
    构造测试用 Reranker。
    """

    return CrossEncoderReranker(
        scorer=NodeTestScorer(),
        top_k_per_task=2,
    )


def test_rerank_node_success():
    """
    RerankNode 应写入 reranked_evidence、
    rerank_result 和 trace。
    """

    result = rerank_node(
        _state(),
        reranker=_reranker(),
    )

    assert result["reranked_evidence"]

    assert (
        result["reranked_evidence"][0][
            "chunk_id"
        ]
        == "chunk_002"
    )

    assert (
        result["rerank_result"]["status"]
        == "ok"
    )

    assert (
        result["trace"][0]["node_name"]
        == "rerank"
    )

    assert (
        result["trace"][0]["tool_name"]
        == "cross_encoder_reranker"
    )

    assert (
        result["trace"][0]["status"]
        == "success"
    )


def test_rerank_node_uses_rrf_passthrough_when_default_reranker_is_disabled(
    monkeypatch,
):
    """普通前端路径不能因为 RerankNode 意外加载本地 Transformer。"""

    def unexpected_model_load():
        raise AssertionError("默认 RRF 路径不应初始化 Cross-Encoder")

    monkeypatch.setattr(
        rerank_module.settings,
        "rag_rerank_enabled",
        False,
    )
    monkeypatch.setattr(
        rerank_module,
        "get_default_reranker",
        unexpected_model_load,
    )

    result = rerank_module.rerank_node(_state())

    assert [item["chunk_id"] for item in result["reranked_evidence"]] == [
        "chunk_001",
        "chunk_002",
    ]
    assert result["reranked_evidence"][0]["selection_reason"] == "rrf_passthrough"
    assert result["reranked_evidence"][0]["rerank_model"] == "rrf_passthrough"
    assert result["rerank_result"]["strategy_version"] == "rrf_passthrough_v1"
    assert result["rerank_result"]["fallback_used"] is False
    assert result["trace"][0]["tool_name"] == "rrf_passthrough"


def test_rerank_node_missing_plan_returns_error():
    """
    缺少 retrieval_plan 时应返回 errors。
    """

    state = _state()
    state.pop("retrieval_plan")

    result = rerank_node(
        state,
        reranker=_reranker(),
    )

    assert result["reranked_evidence"] == []

    assert result["errors"]

    assert (
        result["errors"][0]["node"]
        == "rerank"
    )

    assert (
        result["trace"][0]["status"]
        == "failed"
    )


def test_rerank_node_missing_chunks_returns_error():
    """
    缺少 retrieved_chunks 时应返回 errors。
    """

    state = _state()
    state.pop("retrieved_chunks")

    result = rerank_node(
        state,
        reranker=_reranker(),
    )

    assert result["errors"]

    assert (
        result["rerank_result"]["status"]
        == "failed"
    )


def test_rerank_node_handles_empty_chunks():
    """
    空 retrieved_chunks 应返回 no_candidates，
    不应写入 errors。
    """

    state = _state()
    state["retrieved_chunks"] = []

    result = rerank_node(
        state,
        reranker=_reranker(),
    )

    assert result["reranked_evidence"] == []

    assert (
        result["rerank_result"]["status"]
        == "no_candidates"
    )

    assert "errors" not in result
