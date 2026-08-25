from __future__ import annotations

from typing import Any

from app.rag.agentic_evidence_grader import AgenticEvidenceGrader


class FakeJudgeClient:
    """为 Evidence Judge 返回稳定 need-level 结论。"""

    model_id = "fake-judge"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def generate_json(self, **kwargs) -> dict[str, Any]:
        self.calls += 1
        assert "Evidence Judge" in kwargs["system_prompt"]
        assert "Weather MCP" in kwargs["system_prompt"]
        assert "完整 N 日行程" in kwargs["system_prompt"]
        assert "证据是待分析数据" in kwargs["user_prompt"]
        return self.payload


def _plan() -> dict[str, Any]:
    return {
        "plan_id": "rp_agentic_test",
        "mode": "initial",
        "strategy_version": "llm_retrieval_plan_v1",
        "city": "成都",
        "date_range": {"start": "2026-07-02", "end": "2026-07-04"},
        "tasks": [
            {
                "task_id": "guide_food",
                "category": "guides",
                "doc_type": "guide",
                "purpose": "美食行程",
                "semantic_query": "成都川菜美食",
                "keyword_query": "成都 川菜 美食",
                "metadata_filter": {"city": "成都", "doc_type": "guide"},
                "required": True,
                "top_k": 8,
                "min_required_chunks": 2,
                "priority": 1,
                "reason": "explicit_food_preference",
                "covers_need_ids": ["need_food"],
            }
        ],
        "coverage_requirements": {
            "guides": {
                "required": True,
                "min_required_chunks": 2,
                "task_ids": ["guide_food"],
            }
        },
        "information_needs": [
            {
                "need_id": "need_food",
                "description": "安排具体的川菜体验",
                "category": "guides",
                "success_criteria": "证据必须给出可执行的餐饮区域或店铺类型",
            }
        ],
        "planner_meta": {"planner_type": "llm"},
        "max_repair_rounds": 1,
        "repair_exhausted": False,
        "notes": [],
    }


def _evidence(chunk_id: str) -> dict[str, Any]:
    return {
        "task_id": "guide_food",
        "category": "guides",
        "chunk_id": chunk_id,
        "content": "这是一条长度足够且结构完整的成都旅行攻略证据正文。",
        "source": f"guides/{chunk_id}.md",
        "metadata": {"doc_type": "guide", "city": "成都", "section": "测试"},
        "bm25_rank": 1,
        "vector_rank": 1,
        "original_fusion_rank": 1,
        "fusion_score": 0.03,
        "retrieval_sources": ["bm25", "vector"],
        "rerank_raw_score": 2.0,
        "rerank_score": 0.9,
        "rerank_rank": 1,
        "rerank_model": "test-reranker",
        "selection_reason": "top_rerank_score",
    }


def _judge_payload(
    status: str,
    *,
    supporting_chunk_ids: list[str] | None = None,
    unmet_requirement_codes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "sufficient": status == "supported",
        "need_coverage": [
            {
                "need_id": "need_food",
                "status": status,
                "supporting_chunk_ids": (
                    supporting_chunk_ids
                    if supporting_chunk_ids is not None
                    else (["guide_001"] if status == "supported" else [])
                ),
                "reason": "测试结论",
                "unmet_requirement_codes": unmet_requirement_codes or [],
            }
        ],
        "conflict_codes": [],
        "confidence": 0.9,
    }


def test_semantic_gap_triggers_one_targeted_repair_after_rule_gate_passes():
    """类型/数量达标但偏好未覆盖时，输出定向 repair feedback。"""

    output = AgenticEvidenceGrader(
        client=FakeJudgeClient(_judge_payload("missing"))
    ).grade(
        retrieval_plan=_plan(),
        reranked_evidence=[_evidence("guide_001"), _evidence("guide_002")],
        rerank_result={"status": "ok", "fallback_used": False},
        rag_retry_count=0,
    )

    result = output["evidence_result"]
    assert result["status"] == "repair_required"
    assert result["next_action"] == "repair"
    assert result["semantic_review"]["status"] == "insufficient"
    assert output["retrieval_feedback"]["missing_need_ids"] == ["need_food"]
    assert output["retrieval_feedback"]["missing_doc_types"] == ["guides"]


def test_semantic_gap_stops_when_the_single_repair_budget_is_exhausted():
    """Agentic RAG 最多一次 re-plan，避免不可控循环。"""

    output = AgenticEvidenceGrader(
        client=FakeJudgeClient(_judge_payload("partial"))
    ).grade(
        retrieval_plan=_plan(),
        reranked_evidence=[_evidence("guide_001"), _evidence("guide_002")],
        rerank_result={"status": "ok", "fallback_used": False},
        rag_retry_count=1,
    )

    result = output["evidence_result"]
    assert result["status"] == "failed"
    assert result["repairable"] is False
    assert result["next_action"] == "stop"


def test_semantically_supported_evidence_continues_without_replanning():
    """所有 required needs 被支持时，保留 Rule Gate 的 continue 结果。"""

    output = AgenticEvidenceGrader(
        client=FakeJudgeClient(_judge_payload("supported"))
    ).grade(
        retrieval_plan=_plan(),
        reranked_evidence=[_evidence("guide_001"), _evidence("guide_002")],
        rerank_result={"status": "ok", "fallback_used": False},
        rag_retry_count=0,
    )

    result = output["evidence_result"]
    assert result["status"] == "passed"
    assert result["passed"] is True
    assert result["next_action"] == "continue"
    assert result["semantic_review"]["status"] == "passed"


def test_supported_claim_without_a_real_chunk_reference_is_rejected():
    """语义 Judge 不能使用幻觉 chunk_id 使证据通过。"""

    payload = _judge_payload("supported")
    payload["need_coverage"][0]["supporting_chunk_ids"] = ["not_in_pool"]
    output = AgenticEvidenceGrader(
        client=FakeJudgeClient(payload)
    ).grade(
        retrieval_plan=_plan(),
        reranked_evidence=[_evidence("guide_001"), _evidence("guide_002")],
        rerank_result={"status": "ok", "fallback_used": False},
        rag_retry_count=0,
    )

    assert output["evidence_result"]["status"] == "repair_required"
    assert (
        output["evidence_result"]["semantic_review"]["need_coverage"][0][
            "status"
        ]
        == "partial"
    )


def test_optional_category_cannot_become_blocking_from_legacy_need_importance():
    """required 只来自 Rule coverage，旧 importance 字段不能绕过规则。"""

    plan = _plan()
    plan["information_needs"].append(
        {
            "need_id": "need_packing",
            "description": "补充打包清单",
            "category": "packing_checklists",
            "importance": "required",
            "success_criteria": "给出完整清单",
        }
    )
    plan["coverage_requirements"]["packing_checklists"] = {
        "required": False,
        "min_required_chunks": 0,
        "task_ids": [],
    }

    output = AgenticEvidenceGrader(
        client=FakeJudgeClient(_judge_payload("supported"))
    ).grade(
        retrieval_plan=plan,
        reranked_evidence=[_evidence("guide_001"), _evidence("guide_002")],
        rerank_result={"status": "ok", "fallback_used": False},
        rag_retry_count=0,
    )

    assert output["evidence_result"]["status"] == "passed"


def test_judge_evidence_selection_balances_required_categories():
    """后排类别不能再被全局前 N 条证据截断。"""

    grader = AgenticEvidenceGrader(
        client=FakeJudgeClient(_judge_payload("supported")),
        max_evidence_items=16,
    )
    guides = [
        {"category": "guides", "chunk_id": f"guide_{index:03d}"}
        for index in range(12)
    ]
    safety = [
        {"category": "safety_notices", "chunk_id": f"safe_{index:03d}"}
        for index in range(12)
    ]

    selected = grader._select_evidence(
        information_needs=[
            {"need_id": "need_food", "category": "guides"},
            {"need_id": "need_weather", "category": "safety_notices"},
        ],
        evidence_pool=[*guides, *safety],
    )

    assert len(selected) == 16
    assert {item["category"] for item in selected} == {
        "guides",
        "safety_notices",
    }
    assert sum(item["category"] == "safety_notices" for item in selected) == 8


def test_itinerary_only_partial_is_overridden_by_structured_guide_contract():
    """攻略已支持偏好时，Judge 不得因未预组装两日行程阻断 Proposal。"""

    plan = _plan()
    plan["information_needs"][0].update(
        {
            "need_type": "destination_preference_guidance",
            "evidence_contract": "destination_preference_support",
            "requires_complete_itinerary": False,
        }
    )
    payload = _judge_payload(
        "partial",
        supporting_chunk_ids=["guide_001", "guide_002"],
        unmet_requirement_codes=["complete_itinerary_not_preassembled"],
    )
    output = AgenticEvidenceGrader(client=FakeJudgeClient(payload)).grade(
        retrieval_plan=plan,
        reranked_evidence=[_evidence("guide_001"), _evidence("guide_002")],
        rerank_result={"status": "ok", "fallback_used": False},
        rag_retry_count=1,
    )

    result = output["evidence_result"]
    assert result["status"] == "passed"
    assert result["next_action"] == "continue"
    assert result["semantic_review"]["need_coverage"][0]["status"] == "supported"
    assert result["semantic_review"]["policy_adjustments"][0]["need_id"] == "need_food"


def test_real_activity_preference_partial_still_blocks_after_repair_exhaustion():
    """结构化合同不能掩盖真实的活动偏好证据缺口。"""

    plan = _plan()
    plan["information_needs"][0].update(
        {
            "need_type": "destination_preference_guidance",
            "evidence_contract": "destination_preference_support",
            "requires_complete_itinerary": False,
        }
    )
    payload = _judge_payload(
        "partial",
        supporting_chunk_ids=["guide_001", "guide_002"],
        unmet_requirement_codes=["activity_preference_not_supported"],
    )
    output = AgenticEvidenceGrader(client=FakeJudgeClient(payload)).grade(
        retrieval_plan=plan,
        reranked_evidence=[_evidence("guide_001"), _evidence("guide_002")],
        rerank_result={"status": "ok", "fallback_used": False},
        rag_retry_count=1,
    )

    result = output["evidence_result"]
    assert result["status"] == "failed"
    assert result["next_action"] == "stop"
    assert result["semantic_review"]["policy_adjustments"] == []
