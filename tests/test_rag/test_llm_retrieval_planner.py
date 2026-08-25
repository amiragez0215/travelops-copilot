from __future__ import annotations

from typing import Any

from app.rag.llm_retrieval_planner import LLMRetrievalPlanner


class FakePlannerClient:
    """为 Planner 返回稳定 JSON，单测不访问真实模型。"""

    model_id = "fake-planner"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0
        self.last_user_prompt = ""

    def generate_json(self, **kwargs) -> dict[str, Any]:
        self.calls += 1
        self.last_user_prompt = kwargs["user_prompt"]
        assert "Information Needs" in kwargs["system_prompt"]
        assert "食品安全" in kwargs["system_prompt"]
        assert kwargs["max_tokens"] > 0
        return self.payload


def _trip_request() -> dict[str, Any]:
    return {
        "origin": "杭州",
        "destination": "成都",
        "start_date": "2026-07-02",
        "end_date": "2026-07-04",
        "days": 3,
        "people_count": 1,
        "raw_message": "去成都玩3天，想吃川菜，也想看历史文化，节奏慢一点",
    }


def _planning_context() -> dict[str, Any]:
    return {
        "request": {
            "destination": "成都",
            "start_date": "2026-07-02",
            "end_date": "2026-07-04",
            "days": 3,
        },
        "preference_weights": {
            "flight": {},
            "hotel": {},
            "activity": {"food": 0.9, "culture_history": 0.8},
            "pace": {"slow": 0.9},
            "risk": {},
        },
        "named_constraints": [],
    }


def _planner_payload() -> dict[str, Any]:
    return {
        "mode": "initial",
        "information_needs": [
            {
                "need_id": "need_food",
                "description": "找到能体验川菜的安排",
                "category": "guides",
                "success_criteria": "证据给出具体美食区域或餐饮体验",
            },
            {
                "need_id": "need_culture",
                "description": "找到成都历史文化体验",
                "category": "guides",
                "success_criteria": "证据给出文化场所和可执行建议",
            },
            {
                "need_id": "need_rain",
                "description": "降雨时的出行安全",
                "category": "safety_notices",
                "success_criteria": "证据覆盖雨天交通或防滑措施",
            },
        ],
        "tasks": [
            {
                "task_id": "food_task",
                "category": "guides",
                "covers_need_ids": ["need_food"],
                "semantic_query": "成都慢节奏川菜美食体验",
                "keyword_query": "成都 川菜 美食 慢节奏",
                "purpose": "找到美食行程证据",
                "reason_code": "explicit_food_preference",
            },
            {
                "task_id": "culture_task",
                "category": "guides",
                "covers_need_ids": ["need_culture"],
                "semantic_query": "成都历史文化博物馆古迹行程",
                "keyword_query": "成都 历史 文化 博物馆",
                "purpose": "找到文化行程证据",
                "reason_code": "explicit_culture_preference",
            },
            {
                "task_id": "rain_task",
                "category": "safety_notices",
                "covers_need_ids": ["need_rain"],
                "semantic_query": "成都降雨出行安全和交通提醒",
                "keyword_query": "成都 降雨 安全 交通",
                "purpose": "找到降雨安全证据",
                "reason_code": "weather_risk",
            },
        ],
        "unsupported_needs": [],
    }


def test_llm_planner_can_split_multiple_guide_needs_without_changing_filters():
    """语义任务可以同类多条，但 city/risk 等硬过滤仍来自规则。"""

    client = FakePlannerClient(_planner_payload())
    plan = LLMRetrievalPlanner(client=client).plan(
        trip_request=_trip_request(),
        planning_context=_planning_context(),
        weather_result={
            "status": "ok",
            "risks": [{"risk_type": "rain", "level": "medium"}],
        },
    )

    guides = [task for task in plan.tasks if task.category == "guides"]
    safety = next(task for task in plan.tasks if task.category == "safety_notices")

    assert len(guides) == 2
    assert {tuple(task.covers_need_ids) for task in guides} == {
        ("need_food",),
        ("need_culture",),
    }
    assert all(task.metadata_filter["city"] == "成都" for task in guides)
    assert safety.metadata_filter["risk_type"] == ["rain"]
    assert plan.coverage_requirements["guides"]["task_ids"] == [
        task.task_id for task in guides
    ]
    assert plan.coverage_requirements["guides"]["required"] is True
    assert plan.coverage_requirements["guides"]["min_required_chunks"] == 2
    assert all("importance" not in need for need in plan.information_needs)
    assert plan.planner_meta["planner_type"] == "llm"
    assert plan.max_repair_rounds == 1
    assert "category_policy_templates_this_round" in client.last_user_prompt
    assert "weather_safety" not in client.last_user_prompt
    assert "risk_type" in client.last_user_prompt
    assert client.calls == 1


def test_invalid_llm_plan_falls_back_to_the_existing_rule_plan():
    """模型输出无法通过 Schema 时，原规则 Planner 是唯一 fallback。"""

    plan = LLMRetrievalPlanner(
        client=FakePlannerClient({"tasks": "not-a-list"})
    ).plan(
        trip_request=_trip_request(),
        planning_context=_planning_context(),
    )

    assert plan.strategy_version == "rule_retrieval_plan_v2"
    assert plan.planner_meta["planner_type"] == "rule_fallback"
    assert plan.planner_meta["fallback_used"] is True


def test_required_category_gets_a_minimal_need_when_llm_omits_it():
    """模型不能通过遗漏 needs 让语义 Judge 空跑。"""

    payload = {
        "mode": "initial",
        "information_needs": [],
        "tasks": [],
        "unsupported_needs": [],
    }
    plan = LLMRetrievalPlanner(
        client=FakePlannerClient(payload)
    ).plan(
        trip_request=_trip_request(),
        planning_context=_planning_context(),
    )

    assert plan.planner_meta["planner_type"] == "llm"
    assert len(plan.information_needs) == 1
    need = plan.information_needs[0]
    assert need["need_id"] == "need_required_guides"
    assert need["category"] == "guides"
    assert need["description"]
    assert need["need_type"] == "destination_preference_guidance"
    assert need["requires_complete_itinerary"] is False
    assert "不要求包含旅行天数" in need["success_criteria"]
    assert plan.tasks[0].covers_need_ids == ["need_required_guides"]


def test_guides_itinerary_requirements_are_removed_deterministically():
    """LLM 即使要求两日成品行程，编译层也只保留目的地与活动偏好。"""

    payload = _planner_payload()
    payload["information_needs"][0]["description"] = (
        "北京两日游的完整逐日行程和美食活动"
    )
    payload["information_needs"][0]["success_criteria"] = (
        "必须明确组织成10月1日起的完整两日行程"
    )
    payload["tasks"][0]["semantic_query"] = (
        "北京 10月1日 两日游 完整逐日行程 美食活动"
    )
    payload["tasks"][0]["keyword_query"] = "北京 10月1日 两日游 行程 美食"

    request = _trip_request()
    request.update(
        {"destination": "北京", "start_date": "2026-10-01", "days": 2}
    )
    context = _planning_context()
    context["request"].update(
        {"destination": "北京", "start_date": "2026-10-01", "days": 2}
    )
    plan = LLMRetrievalPlanner(client=FakePlannerClient(payload)).plan(
        trip_request=request,
        planning_context=context,
    )

    need = next(item for item in plan.information_needs if item["need_id"] == "need_food")
    guide = next(task for task in plan.tasks if task.category == "guides")
    assert need["need_type"] == "destination_preference_guidance"
    assert need["requires_complete_itinerary"] is False
    assert "完整两日行程" not in need["success_criteria"]
    assert "10月1日" not in guide.semantic_query
    assert "两日游" not in guide.semantic_query
