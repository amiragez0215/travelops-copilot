from __future__ import annotations

from app.rag.evidence_grader import RuleBasedEvidenceGrader


def _initial_plan():
    """构造与新工作流边界一致的初次检索计划。"""

    return {
        "plan_id": "rp_initial",
        "mode": "initial",
        "strategy_version": "test",
        "city": "成都",
        "date_range": {"start": "2026-07-02", "end": "2026-07-04"},
        "tasks": [
            {
                "task_id": "guide_main",
                "category": "guides",
                "doc_type": "guide",
                "purpose": "逐日攻略",
                "semantic_query": "成都三日雨天美食攻略",
                "keyword_query": "成都 三日游 雨天 美食",
                "metadata_filter": {"city": "成都", "doc_type": "guide"},
                "required": True,
                "top_k": 8,
                "min_required_chunks": 2,
                "priority": 1,
                "reason": "测试",
            },
            {
                "task_id": "selected_hotel_info",
                "category": "hotel_reviews",
                "doc_type": "hotel_reviews",
                "purpose": "选中酒店补充信息",
                "semantic_query": "成都青羊静巷酒店周边交通和餐饮",
                "keyword_query": "成都青羊静巷酒店 周边 交通 餐饮",
                "metadata_filter": {
                    "city": "成都",
                    "doc_type": "hotel_reviews",
                    "hotel_ids": ["hotel_001"],
                },
                "required": False,
                "top_k": 6,
                "min_required_chunks": 0,
                "priority": 3,
                "reason": "测试",
            },
            {
                "task_id": "weather_safety",
                "category": "safety_notices",
                "doc_type": "safety_notice",
                "purpose": "天气安全",
                "semantic_query": "成都雨天大风安全",
                "keyword_query": "成都 雨天 大风 安全",
                "metadata_filter": {
                    "city": "成都",
                    "doc_type": "safety_notice",
                    "risk_type": ["rain", "wind"],
                    "risk_level": ["medium", "high"],
                },
                "required": True,
                "top_k": 8,
                "min_required_chunks": 2,
                "priority": 1,
                "reason": "测试",
            },
            {
                "task_id": "packing_weather",
                "category": "packing_checklists",
                "doc_type": "packing_checklist",
                "purpose": "清单",
                "semantic_query": "三日雨天出行清单",
                "keyword_query": "三日 雨天 清单",
                "metadata_filter": {
                    "doc_type": "packing_checklist",
                    "scenario": ["rainy_city_trip", "general"],
                },
                "required": False,
                "top_k": 5,
                "min_required_chunks": 0,
                "priority": 4,
                "reason": "测试",
            },
        ],
        "coverage_requirements": {
            "guides": {
                "required": True,
                "min_required_chunks": 2,
                "task_ids": ["guide_main"],
            },
            "hotel_reviews": {
                "required": False,
                "min_required_chunks": 0,
                "task_ids": ["selected_hotel_info"],
            },
            "safety_notices": {
                "required": True,
                "min_required_chunks": 2,
                "task_ids": ["weather_safety"],
            },
            "packing_checklists": {
                "required": False,
                "min_required_chunks": 0,
                "task_ids": ["packing_weather"],
            },
        },
        "max_repair_rounds": 1,
        "repair_exhausted": False,
        "notes": [],
    }


def _evidence(
    task_id: str,
    category: str,
    chunk_id: str,
    doc_type: str,
    *,
    hotel_id: str | None = None,
    risk_type: str | None = None,
    risk_level: str | None = None,
    score: float | None = 0.9,
    rerank_model: str = "test-reranker",
):
    metadata = {
        "doc_type": doc_type,
        "city": "成都",
        "section": "测试章节",
    }
    if hotel_id:
        metadata["hotel_id"] = hotel_id
    if risk_type:
        metadata["risk_type"] = risk_type
    if risk_level:
        metadata["risk_level"] = risk_level

    return {
        "task_id": task_id,
        "category": category,
        "chunk_id": chunk_id,
        "content": "这是一条长度足够、能够支持后续旅行方案生成的测试证据正文。",
        "source": f"{category}/{chunk_id}.md",
        "metadata": metadata,
        "bm25_rank": 1,
        "vector_rank": 1,
        "original_fusion_rank": 1,
        "fusion_score": 0.03,
        "retrieval_sources": ["bm25", "vector"],
        "rerank_raw_score": 2.0 if score is not None else None,
        "rerank_score": score,
        "rerank_rank": 1,
        "rerank_model": rerank_model,
        "selection_reason": "top_rerank_score",
    }


def _required_evidence():
    return [
        _evidence("guide_main", "guides", "guide_001", "guide"),
        _evidence("guide_main", "guides", "guide_002", "guide"),
        _evidence(
            "weather_safety",
            "safety_notices",
            "safety_rain",
            "safety_notice",
            risk_type="rain",
            risk_level="medium",
        ),
        _evidence(
            "weather_safety",
            "safety_notices",
            "safety_wind",
            "safety_notice",
            risk_type="wind",
            risk_level="medium",
        ),
    ]


def _weather():
    return {
        "status": "ok",
        "risks": [
            {"risk_type": "rain", "level": "medium"},
            {"risk_type": "wind", "level": "medium"},
        ],
        "daily": [],
    }


def test_complete_required_evidence_passes_without_hotel_rag():
    """酒店补充资料和清单可缺失，但攻略和安全证据必须完整。"""

    output = RuleBasedEvidenceGrader().grade(
        retrieval_plan=_initial_plan(),
        reranked_evidence=_required_evidence(),
        weather_result=_weather(),
        rerank_result={"status": "ok", "fallback_used": False},
        rag_retry_count=0,
    )

    result = output["evidence_result"]
    assert result["status"] == "passed"
    assert result["passed"] is True
    assert result["next_action"] == "continue"
    assert result["coverage"]["hotel_reviews"]["required"] is False
    assert result["coverage"]["hotel_reviews"]["passed"] is True
    assert result["coverage"]["packing_checklists"]["passed"] is True


def test_missing_guide_chunk_requests_repair():
    evidence = _required_evidence()[1:]

    output = RuleBasedEvidenceGrader().grade(
        retrieval_plan=_initial_plan(),
        reranked_evidence=evidence,
        weather_result=_weather(),
        rag_retry_count=0,
    )

    result = output["evidence_result"]
    assert result["status"] == "repair_required"
    assert result["next_action"] == "repair"
    assert output["retrieval_feedback"]["missing_doc_types"] == ["guides"]


def test_missing_weather_risk_requests_targeted_repair():
    evidence = [
        item
        for item in _required_evidence()
        if item["metadata"].get("risk_type") != "wind"
    ]

    output = RuleBasedEvidenceGrader().grade(
        retrieval_plan=_initial_plan(),
        reranked_evidence=evidence,
        weather_result=_weather(),
        rag_retry_count=0,
    )

    feedback = output["retrieval_feedback"]
    assert "safety_notices" in feedback["missing_doc_types"]
    assert feedback["missing_risk_types"] == ["wind"]


def test_invalid_metadata_is_rejected_and_can_trigger_repair():
    evidence = _required_evidence()
    evidence[0]["metadata"]["city"] = "杭州"

    output = RuleBasedEvidenceGrader().grade(
        retrieval_plan=_initial_plan(),
        reranked_evidence=evidence,
        weather_result=_weather(),
        rag_retry_count=0,
    )

    result = output["evidence_result"]
    assert result["rejected_evidence_count"] == 1
    assert result["status"] == "repair_required"
    assert any(
        issue["issue_type"] == "metadata_filter_mismatch"
        for issue in result["issues"]
    )


def test_repair_mode_merges_old_and_new_evidence():
    """修复证据不能覆盖初次已经通过的证据池。"""

    old_pool = [
        _evidence("guide_main", "guides", "guide_001", "guide"),
        _evidence("guide_main", "guides", "guide_002", "guide"),
        _evidence(
            "weather_safety",
            "safety_notices",
            "safety_rain",
            "safety_notice",
            risk_type="rain",
            risk_level="medium",
        ),
    ]

    repair_plan = {
        "plan_id": "rp_repair",
        "mode": "repair",
        "strategy_version": "test",
        "city": "成都",
        "date_range": {"start": "2026-07-02", "end": "2026-07-04"},
        "tasks": [
            {
                "task_id": "repair_weather_safety",
                "category": "safety_notices",
                "doc_type": "safety_notice",
                "purpose": "补大风安全证据",
                "semantic_query": "成都大风安全",
                "keyword_query": "成都 大风 安全",
                "metadata_filter": {
                    "city": "成都",
                    "doc_type": "safety_notice",
                    "risk_type": ["wind"],
                    "risk_level": ["medium", "high"],
                },
                "required": True,
                "top_k": 5,
                "min_required_chunks": 1,
                "priority": 1,
                "reason": "修复",
                "repair_of": "weather_safety",
            }
        ],
        "coverage_requirements": {
            "safety_notices": {
                "required": True,
                "min_required_chunks": 1,
                "task_ids": ["repair_weather_safety"],
            }
        },
        "max_repair_rounds": 1,
        "repair_exhausted": False,
        "notes": [],
    }

    new_evidence = [
        _evidence(
            "repair_weather_safety",
            "safety_notices",
            "safety_wind",
            "safety_notice",
            risk_type="wind",
            risk_level="medium",
        )
    ]

    output = RuleBasedEvidenceGrader().grade(
        retrieval_plan=repair_plan,
        reranked_evidence=new_evidence,
        existing_evidence_pool=old_pool,
        existing_requirements=_initial_plan()["coverage_requirements"],
        weather_result=_weather(),
        rag_retry_count=1,
    )

    result = output["evidence_result"]
    assert result["status"] == "passed"
    assert len(output["evidence_pool"]) == 4
    assert result["coverage"]["safety_notices"]["missing_risk_types"] == []


def test_missing_required_evidence_stops_after_retry_limit():
    output = RuleBasedEvidenceGrader().grade(
        retrieval_plan=_initial_plan(),
        reranked_evidence=_required_evidence()[1:],
        weather_result=_weather(),
        rag_retry_count=1,
    )

    result = output["evidence_result"]
    assert result["status"] == "failed"
    assert result["repairable"] is False
    assert result["next_action"] == "stop"


def test_rrf_fallback_can_pass_as_degraded():
    evidence = _required_evidence()
    evidence[0]["rerank_score"] = None
    evidence[0]["rerank_raw_score"] = None
    evidence[0]["rerank_model"] = "fallback:rrf_order"

    output = RuleBasedEvidenceGrader(
        allow_rerank_fallback=True
    ).grade(
        retrieval_plan=_initial_plan(),
        reranked_evidence=evidence,
        weather_result=_weather(),
        rerank_result={"status": "degraded", "fallback_used": True},
        rag_retry_count=0,
    )

    result = output["evidence_result"]
    assert result["status"] == "degraded"
    assert result["passed"] is True
    assert result["next_action"] == "continue"


def test_default_rrf_passthrough_can_pass_without_a_rerank_score():
    """默认 Variant C 没有 Cross-Encoder 分数，但必须保留为有效证据。"""

    evidence = _required_evidence()
    evidence[0]["rerank_score"] = None
    evidence[0]["rerank_raw_score"] = None
    evidence[0]["rerank_model"] = "rrf_passthrough"
    evidence[0]["selection_reason"] = "rrf_passthrough"

    output = RuleBasedEvidenceGrader().grade(
        retrieval_plan=_initial_plan(),
        reranked_evidence=evidence,
        weather_result=_weather(),
        rerank_result={
            "status": "ok",
            "strategy_version": "rrf_passthrough_v1",
            "fallback_used": False,
        },
        rag_retry_count=0,
    )

    result = output["evidence_result"]
    assert result["status"] == "passed"
    assert result["passed"] is True
    assert result["fallback_used"] is False
