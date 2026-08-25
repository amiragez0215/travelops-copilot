from __future__ import annotations

from app.rag.retrieval_planner import (
    RuleBasedRetrievalPlanner,
    build_retrieval_plan,
)


def _trip_request():
    return {
        "origin": "杭州",
        "destination": "成都",
        "start_date": "2026-07-02",
        "end_date": "2026-07-04",
        "days": 3,
        "people_count": 1,
        "room_count": 1,
        "named_constraints": [
            {
                "entity_type": "food",
                "entity_name": "川菜",
                "constraint_mode": "preferred",
                "scope": "itinerary",
                "evidence": "想吃川菜",
                "source": "llm",
            }
        ],
    }


def _planning_context():
    return {
        "request": {
            "origin": "杭州",
            "destination": "成都",
            "start_date": "2026-07-02",
            "end_date": "2026-07-04",
            "days": 3,
            "nights": 2,
            "people_count": 1,
            "room_count": 1,
        },
        "preference_weights": {
            "flight": {},
            "hotel": {
                "quiet": 1.0,
                "cleanliness": 0.9,
                "near_subway": 0.8,
            },
            "activity": {
                "food": 0.9,
                "nature_scenery": 0.85,
            },
            "pace": {"slow": 0.9},
            "risk": {},
        },
        "named_constraints": _trip_request()["named_constraints"],
    }


def _weather_with_rain_and_wind():
    return {
        "status": "ok",
        "risks": [
            {"risk_type": "rain", "level": "medium"},
            {"risk_type": "wind", "level": "medium"},
        ],
        "daily": [],
    }


def _hotel_results():
    return {
        "status": "ok",
        "items": [
            {
                "hotel_id": "hotel_001",
                "name": "成都青羊静巷酒店",
                "city": "成都",
            },
            {
                "hotel_id": "hotel_002",
                "name": "成都春熙路酒店",
                "city": "成都",
            },
        ],
    }


def _selection_result():
    return {
        "status": "feasible",
        "selected_combination": {
            "hotel_id": "hotel_001",
            "outbound_flight_id": "flight_out_001",
            "return_flight_id": "flight_return_001",
        },
    }


def _task(plan, category):
    return next(item for item in plan["tasks"] if item["category"] == category)


def test_initial_plan_uses_selected_hotel_and_marks_hotel_rag_optional():
    """RAG 位于选择之后，只查询选中酒店，并且不把酒店资料作为评分前提。"""

    plan = build_retrieval_plan(
        trip_request=_trip_request(),
        planning_context=_planning_context(),
        weather_result=_weather_with_rain_and_wind(),
        selection_result=_selection_result(),
        raw_hotel_results=_hotel_results(),
    )

    categories = {item["category"] for item in plan["tasks"]}
    assert categories == {
        "guides",
        "hotel_reviews",
        "safety_notices",
        "packing_checklists",
    }

    guide = _task(plan, "guides")
    assert guide["required"] is True
    assert "成都" in guide["semantic_query"]
    assert "三日游" in guide["semantic_query"]
    assert "川菜" in guide["semantic_query"]
    assert "雨天室内景点" in guide["semantic_query"]

    hotel = _task(plan, "hotel_reviews")
    assert hotel["required"] is False
    assert hotel["min_required_chunks"] == 0
    assert hotel["metadata_filter"]["hotel_ids"] == ["hotel_001"]
    assert "成都青羊静巷酒店" in hotel["semantic_query"]

    safety = _task(plan, "safety_notices")
    assert safety["required"] is True
    assert set(safety["metadata_filter"]["risk_type"]) == {"rain", "wind"}

    packing = _task(plan, "packing_checklists")
    assert packing["required"] is False

    # Optional hotel RAG 不生成 required_hotel_ids。
    assert "required_hotel_ids" not in plan["coverage_requirements"]["hotel_reviews"]


def test_without_selection_skips_hotel_rag_even_when_raw_hotels_exist():
    """原始候选很多时也不能为全部酒店检索说明文档。"""

    plan = build_retrieval_plan(
        trip_request=_trip_request(),
        planning_context=_planning_context(),
        weather_result={"status": "ok", "risks": [], "daily": []},
        selection_result={},
        raw_hotel_results=_hotel_results(),
    )

    categories = {item["category"] for item in plan["tasks"]}
    assert "hotel_reviews" not in categories
    assert "safety_notices" not in categories
    assert "guides" in categories
    assert "packing_checklists" in categories
    assert any("没有 selection_result" in note for note in plan["notes"])


def test_weather_unavailable_does_not_fabricate_safety_requirement():
    plan = build_retrieval_plan(
        trip_request=_trip_request(),
        planning_context=_planning_context(),
        weather_result={"status": "unavailable", "risks": [], "daily": []},
        selection_result=_selection_result(),
        raw_hotel_results=_hotel_results(),
    )

    categories = {item["category"] for item in plan["tasks"]}
    assert "safety_notices" not in categories
    assert any("天气数据不可用" in note for note in plan["notes"])


def test_repair_plan_only_repairs_required_guide_gap():
    planner = RuleBasedRetrievalPlanner(max_repair_rounds=1)

    plan = planner.plan(
        trip_request=_trip_request(),
        planning_context=_planning_context(),
        weather_result=_weather_with_rain_and_wind(),
        selection_result=_selection_result(),
        raw_hotel_results=_hotel_results(),
        retrieval_feedback={"missing_doc_types": ["guides"]},
        rag_retry_count=0,
    ).to_state_dict()

    assert plan["mode"] == "repair"
    assert len(plan["tasks"]) == 1
    assert plan["tasks"][0]["task_id"] == "repair_guide_main"
    assert plan["tasks"][0]["repair_of"] == "guide_main"


def test_repair_plan_only_requests_missing_weather_risk():
    plan = build_retrieval_plan(
        trip_request=_trip_request(),
        planning_context=_planning_context(),
        weather_result=_weather_with_rain_and_wind(),
        selection_result=_selection_result(),
        raw_hotel_results=_hotel_results(),
        retrieval_feedback={
            "missing_doc_types": ["safety_notices"],
            "missing_risk_types": ["wind"],
        },
        rag_retry_count=0,
    )

    assert len(plan["tasks"]) == 1
    task = plan["tasks"][0]
    assert task["task_id"] == "repair_weather_safety"
    assert task["metadata_filter"]["risk_type"] == ["wind"]


def test_optional_selected_hotel_can_be_repaired_when_explicitly_requested():
    """正常不会因酒店资料缺失 repair，但保留手工补充能力。"""

    plan = build_retrieval_plan(
        trip_request=_trip_request(),
        planning_context=_planning_context(),
        weather_result={"status": "ok", "risks": [], "daily": []},
        selection_result=_selection_result(),
        raw_hotel_results=_hotel_results(),
        retrieval_feedback={
            "missing_doc_types": ["hotel_reviews"],
            "missing_hotel_ids": ["hotel_001"],
        },
        rag_retry_count=0,
    )

    assert len(plan["tasks"]) == 1
    assert plan["tasks"][0]["task_id"] == "repair_selected_hotel_info"
    assert plan["tasks"][0]["required"] is False


def test_repair_stops_after_retry_limit():
    planner = RuleBasedRetrievalPlanner(max_repair_rounds=1)

    plan = planner.plan(
        trip_request=_trip_request(),
        planning_context=_planning_context(),
        retrieval_feedback={"missing_doc_types": ["guides"]},
        rag_retry_count=1,
    ).to_state_dict()

    assert plan["mode"] == "repair"
    assert plan["repair_exhausted"] is True
    assert plan["tasks"] == []
