from __future__ import annotations

from app.nodes.evidence_grade_node import (
    evidence_grade_node,
    route_after_evidence,
)


def _task(
    task_id: str,
    category: str,
    doc_type: str,
    min_chunks: int,
):
    """
    构造简化 RetrievalTask。
    """

    return {
        "task_id": task_id,
        "category": category,
        "doc_type": doc_type,
        "purpose": "测试",
        "semantic_query": "成都测试查询",
        "keyword_query": "成都 测试",
        "metadata_filter": {
            "city": "成都",
            "doc_type": doc_type,
        },
        "required": True,
        "top_k": 5,
        "min_required_chunks": min_chunks,
        "priority": 1,
        "reason": "测试",
    }


def _state():
    """
    构造能够通过 EvidenceGradeNode 的 State。
    """

    return {
        "retrieval_plan": {
            "plan_id": "rp_node_test",
            "mode": "initial",
            "strategy_version": "test",
            "city": "成都",

            "date_range": {
                "start": "2026-07-02",
                "end": "2026-07-04",
            },

            "tasks": [
                _task(
                    "guide_main",
                    "guides",
                    "guide",
                    1,
                )
            ],

            "coverage_requirements": {
                "guides": {
                    "required": True,
                    "min_required_chunks": 1,
                    "task_ids": [
                        "guide_main"
                    ],
                }
            },

            "max_repair_rounds": 1,
            "repair_exhausted": False,
            "notes": [],
        },

        "reranked_evidence": [
            {
                "task_id": "guide_main",
                "category": "guides",
                "chunk_id": "guide_001",

                "content":
                    "这是能够支持成都旅行规划的一条完整攻略证据。",

                "source":
                    "guides/guide.md",

                "metadata": {
                    "doc_type": "guide",
                    "city": "成都",
                    "section": "三日行程",
                },

                "bm25_rank": 1,
                "vector_rank": 1,
                "original_fusion_rank": 1,
                "fusion_score": 0.033,

                "retrieval_sources": [
                    "bm25",
                    "vector",
                ],

                "rerank_raw_score": 2.0,
                "rerank_score": 0.88,
                "rerank_rank": 1,
                "rerank_model": "test-reranker",
                "selection_reason": "top_rerank_score",
            }
        ],

        "rerank_result": {
            "status": "ok",
            "fallback_used": False,
        },

        "weather_result": {
            "status": "ok",
            "risks": [],
            "daily": [],
        },

        "rag_retry_count": 0,
    }


def test_evidence_grade_node_success():
    """
    正常输入时应写入完整 Evidence Grade 结果。
    """

    result = evidence_grade_node(
        _state()
    )

    assert (
        result["evidence_result"]["status"]
        == "passed"
    )

    assert (
        result["evidence_result"]["passed"]
        is True
    )

    assert len(
        result["evidence_pool"]
    ) == 1

    # Node 会把累计证据池同步写回 reranked_evidence。
    assert (
        result["reranked_evidence"]
        == result["evidence_pool"]
    )

    assert (
        result["retrieval_feedback"]
        == {}
    )

    assert (
        result["trace"][0]["node_name"]
        == "evidence_grade"
    )

    assert (
        result["trace"][0]["tool_name"]
        == "rule_evidence_grader"
    )

    assert (
        result["trace"][0]["status"]
        == "success"
    )


def test_evidence_grade_node_requests_repair():
    """
    Required evidence 缺失时，应生成 repair feedback。
    """

    state = _state()
    state["reranked_evidence"] = []

    result = evidence_grade_node(
        state
    )

    assert (
        result["evidence_result"]["status"]
        == "repair_required"
    )

    assert (
        result["evidence_result"]["next_action"]
        == "repair"
    )

    assert (
        result["retrieval_feedback"][
            "missing_doc_types"
        ]
        == ["guides"]
    )

    assert (
        result["trace"][0]["status"]
        == "degraded"
    )


def test_route_after_evidence():
    """
    Conditional Edge 应根据 next_action 路由；通过后进入 ProposalNode。
    """

    assert route_after_evidence(
        {
            "evidence_result": {
                "next_action": "continue"
            }
        }
    ) == "proposal"

    assert route_after_evidence(
        {
            "evidence_result": {
                "next_action": "repair"
            }
        }
    ) == "retrieval_plan"

    assert route_after_evidence(
        {
            "evidence_result": {
                "next_action": "stop"
            }
        }
    ) == "final_response"

    assert route_after_evidence(
        {}
    ) == "final_response"


def test_evidence_grade_node_missing_plan_returns_error():
    """
    缺少 retrieval_plan 时，应返回程序错误。
    """

    state = _state()
    state.pop(
        "retrieval_plan"
    )

    result = evidence_grade_node(
        state
    )

    assert result["errors"]

    assert (
        result["errors"][0]["node"]
        == "evidence_grade"
    )

    assert (
        result["evidence_result"]["status"]
        == "failed"
    )

    assert (
        result["trace"][0]["status"]
        == "failed"
    )


def test_evidence_grade_node_missing_reranked_evidence_returns_error():
    """
    缺少 reranked_evidence 时，应返回程序错误。
    """

    state = _state()
    state.pop(
        "reranked_evidence"
    )

    result = evidence_grade_node(
        state
    )

    assert result["errors"]

    assert (
        result["evidence_result"]["next_action"]
        == "stop"
    )