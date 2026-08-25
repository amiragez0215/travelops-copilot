from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from app.proposals.proposal_generator import (
    LLMProposalGenerator,
    ProposalGenerationError,
    build_requirement_catalog,
)


class SequenceJSONClient:
    """
    按顺序返回预设 JSON 的 Fake StructuredJSONClient。

    它用于验证：
        - ProposalGenerator 不依赖真实 DeepSeek；
        - 首次输出失败后会带验证反馈重试；
        - 单元测试可以稳定复现模型行为。
    """

    model_id = "fake-proposal-model"

    def __init__(
        self,
        payloads: list[dict[str, Any] | Exception],
    ) -> None:
        self.payloads = list(payloads)
        self.call_count = 0
        self.user_prompts: list[str] = []

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        self.user_prompts.append(user_prompt)

        index = min(
            self.call_count,
            len(self.payloads) - 1,
        )
        self.call_count += 1

        item = self.payloads[index]

        if isinstance(item, Exception):
            raise item

        return deepcopy(item)


def _trip_request() -> dict[str, Any]:
    """构造 Proposal 所需的结构化旅行请求。"""

    return {
        "user_id": "user_001",
        "origin": "杭州",
        "destination": "成都",
        "start_date": "2026-07-02",
        "end_date": "2026-07-04",
        "days": 3,
        "budget": 4000,
        "people_count": 1,
        "room_count": 1,
        "named_constraints": [],
        "unmapped_requirements": [],
    }


def _planning_context() -> dict[str, Any]:
    """构造已经完成偏好融合和预算规划的 PlanningContext。"""

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
            "total_budget": 4000,
        },
        "preference_weights": {
            "flight": {
                "avoid_early_flight": 0.9,
            },
            "hotel": {
                "quiet": 1.0,
                "cleanliness": 0.9,
            },
            "activity": {
                "food": 0.9,
                "culture_history": 0.8,
            },
            "pace": {
                "slow": 0.85,
            },
            "risk": {},
        },
        "diet_preferences": {
            "likes": ["川菜", "小吃"],
        },
        "named_constraints": [],
        "unmapped_requirements": [],
        "context_summary": {
            "main_preferences": [
                "安静酒店",
                "美食",
                "文化历史",
                "慢节奏",
            ]
        },
    }


def _selection_result() -> dict[str, Any]:
    """构造 BudgetOptimizeNode 的可行组合结果。"""

    outbound = {
        "flight_id": "flight_out_001",
        "flight_no": "MU5203",
        "airline": "示例航空",
        "departure_city": "杭州",
        "arrival_city": "成都",
        "departure_airport": "杭州萧山国际机场",
        "arrival_airport": "成都双流国际机场",
        "depart_date": "2026-07-02",
        "depart_time": "09:30",
        "arrive_date": "2026-07-02",
        "arrive_time": "12:10",
        "duration_minutes": 160,
        "price": 720,
        "total_price": 720,
        "is_direct": True,
        "refundable": False,
        "changeable": True,
        "rank": 1,
        "total_score": 0.92,
        "ranking_reasons": ["出发时间符合偏好"],
        "preference_gaps": [],
    }

    return_flight = {
        "flight_id": "flight_return_001",
        "flight_no": "MU5204",
        "airline": "示例航空",
        "departure_city": "成都",
        "arrival_city": "杭州",
        "departure_airport": "成都双流国际机场",
        "arrival_airport": "杭州萧山国际机场",
        "depart_date": "2026-07-04",
        "depart_time": "16:00",
        "arrive_date": "2026-07-04",
        "arrive_time": "18:40",
        "duration_minutes": 160,
        "price": 680,
        "total_price": 680,
        "is_direct": True,
        "refundable": False,
        "changeable": True,
        "rank": 1,
        "total_score": 0.90,
        "ranking_reasons": ["返程时间较合适"],
        "preference_gaps": [],
    }

    hotel = {
        "hotel_id": "hotel_001",
        "name": "成都青羊静巷酒店",
        "city": "成都",
        "district": "青羊区",
        "address": "成都市青羊区示例路1号",
        "price_per_night": 580,
        "estimated_total_price": 1160,
        "planned_nights": 2,
        "planned_room_count": 1,
        "rating": 4.7,
        "near_subway": True,
        "distance_to_subway_meters": 350,
        "quiet_score": 0.90,
        "cleanliness_score": 0.92,
        "tags": ["安静", "近地铁"],
        "amenities": ["WiFi", "行李寄存"],
        "cancel_policy": "入住前24小时可免费取消",
        "rank": 1,
        "total_score": 0.94,
        "ranking_reasons": ["安静与清洁偏好匹配较好"],
        "preference_gaps": [],
    }

    selected_costs = {
        "outbound_flight": 720,
        "return_flight": 680,
        "transport_total": 1400,
        "hotel": 1160,
        "known_subtotal": 2560,
    }

    return {
        "status": "feasible",
        "strategy_version": "deterministic_budget_optimize_v1",
        "budget_mode": "budget_limited",
        "selected_combination": {
            "combination_id": "combo_001",
            "outbound_flight_id": "flight_out_001",
            "return_flight_id": "flight_return_001",
            "hotel_id": "hotel_001",
            "combination_score": 0.91,
            "selected_costs": selected_costs,
        },
        "selected_outbound_flight": outbound,
        "selected_return_flight": return_flight,
        "selected_hotel": hotel,
        "selected_hotel_id": "hotel_001",
        "selected_costs": selected_costs,
        "total_budget": 4000,
        "remaining_budget": 1440,
        "remaining_budget_allocation": {
            "food_activity": 1100,
            "local_transport_and_buffer": 340,
            "food_activity_target": 1100,
            "food_activity_shortfall": 0,
            "allocation_basis": "soft_target_then_buffer",
            "note": "这是预算安排，不是实际消费预测。",
        },
        "alternatives": [],
        "notes": [],
    }


def _weather_result() -> dict[str, Any]:
    """构造中间一天有雨的逐日天气。"""

    return {
        "status": "ok",
        "summary": "第一天多云，第二天小雨，第三天晴。",
        "daily": [
            {
                "date": "2026-07-02",
                "condition": "多云",
                "temperature_low": 23,
                "temperature_high": 30,
                "precipitation_probability": 20,
                "risk_tags": [],
                "warnings": [],
            },
            {
                "date": "2026-07-03",
                "condition": "小雨",
                "temperature_low": 22,
                "temperature_high": 27,
                "precipitation_probability": 75,
                "risk_tags": ["rain"],
                "warnings": ["建议携带雨具"],
            },
            {
                "date": "2026-07-04",
                "condition": "晴",
                "temperature_low": 24,
                "temperature_high": 31,
                "precipitation_probability": 10,
                "risk_tags": [],
                "warnings": [],
            },
        ],
        "risks": [
            {
                "risk_type": "rain",
                "level": "medium",
                "date": "2026-07-03",
                "message": "雨天减少长时间户外活动。",
            }
        ],
        "warnings": ["建议携带雨具"],
        "source": "mock_weather_provider",
    }


def _evidence_pool(
    *,
    include_hotel: bool = True,
) -> list[dict[str, Any]]:
    """构造 EvidenceGrade 已验证的累计证据池。"""

    items = [
        {
            "task_id": "guide_main",
            "category": "guides",
            "chunk_id": "guide_day_1",
            "content": "抵达成都后可安排人民公园、茶馆和本地小吃，适合慢节奏活动。",
            "source": "guides/chengdu_slow.md",
            "metadata": {
                "doc_type": "guide",
                "city": "成都",
                "title": "成都慢节奏旅行",
                "section": "公园与茶馆",
            },
            "fusion_score": 0.032,
            "rerank_score": 0.93,
            "rerank_rank": 1,
        },
        {
            "task_id": "guide_main",
            "category": "guides",
            "chunk_id": "guide_rain_day",
            "content": "雨天适合优先安排博物馆、展馆和茶馆，将公园活动调整到晴天。",
            "source": "guides/chengdu_rain.md",
            "metadata": {
                "doc_type": "guide",
                "city": "成都",
                "title": "成都雨天旅行",
                "section": "室内安排",
            },
            "fusion_score": 0.031,
            "rerank_score": 0.96,
            "rerank_rank": 1,
        },
        {
            "task_id": "guide_main",
            "category": "guides",
            "chunk_id": "guide_day_3",
            "content": "天气晴好时可以安排城市公园和街区漫步，返程日前预留交通时间。",
            "source": "guides/chengdu_walk.md",
            "metadata": {
                "doc_type": "guide",
                "city": "成都",
                "title": "成都城市漫步",
                "section": "晴天路线",
            },
            "fusion_score": 0.030,
            "rerank_score": 0.91,
            "rerank_rank": 2,
        },
        {
            "task_id": "weather_safety",
            "category": "safety_notices",
            "chunk_id": "safety_rain",
            "content": "雨天道路可能湿滑，前往机场和跨区域活动时应预留额外交通时间。",
            "source": "safety_notices/chengdu_rain.md",
            "metadata": {
                "doc_type": "safety_notice",
                "city": "成都",
                "title": "成都雨天安全提醒",
                "section": "道路与交通",
                "risk_type": "rain",
            },
            "fusion_score": 0.033,
            "rerank_score": 0.97,
            "rerank_rank": 1,
        },
        {
            "task_id": "packing_weather",
            "category": "packing_checklists",
            "chunk_id": "packing_rain",
            "content": "三日雨天城市旅行建议携带雨伞、防滑鞋、备用袜子和防水袋。",
            "source": "packing_checklists/rain_3days.md",
            "metadata": {
                "doc_type": "packing_checklist",
                "city": "general",
                "title": "三日雨天清单",
                "section": "雨天用品",
            },
            "fusion_score": 0.029,
            "rerank_score": 0.89,
            "rerank_rank": 1,
        },
    ]

    if include_hotel:
        items.append(
            {
                "task_id": "selected_hotel_info",
                "category": "hotel_reviews",
                "chunk_id": "hotel_001_context",
                "content": "酒店周边有便利店和小型餐饮，步行可到地铁站；靠近电梯的房间可能有走动声。",
                "source": "hotel_reviews/hotel_001.md",
                "metadata": {
                    "doc_type": "hotel_reviews",
                    "city": "成都",
                    "hotel_id": "hotel_001",
                    "hotel_name": "成都青羊静巷酒店",
                    "title": "成都青羊静巷酒店体验摘要",
                    "section": "周边与注意事项",
                },
                "fusion_score": 0.028,
                "rerank_score": 0.88,
                "rerank_rank": 1,
            }
        )

    return items


def _valid_draft(
    *,
    include_hotel: bool = True,
    requirement_handling: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """构造满足日期、天气和 Evidence 规则的 ProposalDraft。"""

    return {
        "summary": "成都三日慢节奏方案，结合天气安排文化、美食和城市漫步。",
        "highlights": [
            "中间雨天优先安排室内文化活动",
            "兼顾美食与慢节奏体验",
        ],
        "daily_plan": [
            {
                "date": "2026-07-02",
                "theme": "抵达与慢节奏城市体验",
                "activities": [
                    {
                        "period": "arrival",
                        "title": "抵达成都并前往酒店",
                        "description": "按锁定航班时间抵达后前往酒店办理入住。",
                        "activity_type": "transfer",
                        "indoor_outdoor": "transfer",
                        "evidence_refs": [],
                        "practical_notes": ["预留机场到酒店交通时间"],
                    },
                    {
                        "period": "afternoon",
                        "title": "公园与茶馆慢体验",
                        "description": "安排节奏舒缓的公园和茶馆活动。",
                        "activity_type": "leisure",
                        "indoor_outdoor": "mixed",
                        "evidence_refs": ["G02"],
                        "practical_notes": [],
                    },
                ],
                "weather_adjustment": None,
                "day_notes": [],
            },
            {
                "date": "2026-07-03",
                "theme": "雨天室内文化与美食",
                "activities": [
                    {
                        "period": "morning",
                        "title": "室内文化展馆",
                        "description": "利用雨天优先安排室内文化活动。",
                        "activity_type": "culture",
                        "indoor_outdoor": "indoor",
                        "evidence_refs": ["G01", "S01"],
                        "practical_notes": ["注意湿滑路面"],
                    },
                    {
                        "period": "afternoon",
                        "title": "茶馆与本地美食",
                        "description": "安排适合雨天的慢节奏茶馆和美食体验。",
                        "activity_type": "food",
                        "indoor_outdoor": "indoor",
                        "evidence_refs": ["G01"],
                        "practical_notes": [],
                    },
                ],
                "weather_adjustment": "将原本的户外公园活动调整到第三天，并增加交通缓冲。",
                "day_notes": ["随身携带雨具"],
            },
            {
                "date": "2026-07-04",
                "theme": "晴天城市漫步与返程",
                "activities": [
                    {
                        "period": "morning",
                        "title": "城市公园与街区漫步",
                        "description": "天气晴好时安排适度的户外漫步。",
                        "activity_type": "city_walk",
                        "indoor_outdoor": "outdoor",
                        "evidence_refs": ["G03"],
                        "practical_notes": ["控制步行强度"],
                    },
                    {
                        "period": "departure",
                        "title": "前往机场返程",
                        "description": "按锁定返程航班时间提前前往机场。",
                        "activity_type": "transfer",
                        "indoor_outdoor": "transfer",
                        "evidence_refs": [],
                        "practical_notes": ["预留返程交通时间"],
                    },
                ],
                "weather_adjustment": None,
                "day_notes": [],
            },
        ],
        "hotel_context": (
            {
                "summary": "酒店周边生活设施和交通较方便。",
                "nearby_facilities": ["便利店", "小型餐饮", "地铁站"],
                "cautions": ["靠近电梯的房间可能有走动声"],
                "evidence_refs": ["H01"],
            }
            if include_hotel
            else None
        ),
        "packing_tips": [
            {
                "text": "携带雨伞、防滑鞋和备用袜子。",
                "evidence_refs": ["P01"],
            }
        ],
        "safety_notes": [
            {
                "text": "雨天注意路面湿滑，并为跨区域交通预留时间。",
                "evidence_refs": ["S01"],
            }
        ],
        "preference_alignment": [
            "行程保持慢节奏",
            "安排本地美食和文化活动",
        ],
        "requirement_handling": requirement_handling or [],
        "limitations": ["景点开放情况需出行前再次确认"],
    }


def _generator(
    payloads: list[dict[str, Any] | Exception],
    *,
    max_attempts: int = 2,
) -> tuple[LLMProposalGenerator, SequenceJSONClient]:
    """创建测试用 ProposalGenerator。"""

    client = SequenceJSONClient(payloads)

    return (
        LLMProposalGenerator(
            client=client,
            max_tokens=4096,
            max_attempts=max_attempts,
            max_evidence_items=20,
            max_evidence_chars=800,
        ),
        client,
    )


def _generate(
    generator: LLMProposalGenerator,
    *,
    planning_context: dict[str, Any] | None = None,
    evidence_pool: list[dict[str, Any]] | None = None,
    proposal_version: int = 1,
) -> dict[str, Any]:
    """调用 Generator 的统一测试入口。"""

    return generator.generate(
        trip_request=_trip_request(),
        planning_context=(
            planning_context
            if planning_context is not None
            else _planning_context()
        ),
        selection_result=_selection_result(),
        weather_result=_weather_result(),
        evidence_pool=(
            evidence_pool
            if evidence_pool is not None
            else _evidence_pool()
        ),
        evidence_result={
            "status": "passed",
            "passed": True,
            "next_action": "continue",
        },
        proposal_version=proposal_version,
    )


def test_proposal_generator_assembles_locked_facts_and_weather():
    """
    LLM 只生成活动；最终航班、酒店、天气和预算必须来自锁定事实。
    """

    generator, client = _generator([
        _valid_draft()
    ])

    output = _generate(generator)
    proposal = output["proposal"]
    result = output["proposal_result"]

    assert result["status"] == "generated"
    assert result["attempt_count"] == 1
    assert client.call_count == 1

    # 航班和酒店全部来自 selection_result。
    assert (
        proposal["selected_flights"]["outbound"]["flight_id"]
        == "flight_out_001"
    )
    assert (
        proposal["selected_flights"]["return_flight"]["flight_id"]
        == "flight_return_001"
    )
    assert proposal["selected_hotel"]["hotel_id"] == "hotel_001"

    # 中间雨天的天气由程序注入，不是 LLM 自行生成。
    rainy_day = proposal["daily_plan"][1]
    assert rainy_day["date"] == "2026-07-03"
    assert rainy_day["weather"]["condition"] == "小雨"
    assert rainy_day["weather"]["risk_types"] == ["rain"]
    assert rainy_day["weather_adjustment"]

    # 预算只说明已知 Mock 成本和剩余预算安排。
    assert (
        proposal["budget_summary"]["known_costs"]["known_subtotal"]
        == 2560
    )
    assert proposal["budget_summary"]["remaining_budget"] == 1440
    assert "实际消费" in proposal["budget_summary"]["note"]

    # 当前系统永远是草稿模式，不会声称真实执行。
    assert (
        proposal["execution_boundary"]["real_booking_performed"]
        is False
    )
    assert proposal["execution_boundary"]["draft_only"] is True


def test_unknown_evidence_ref_triggers_bounded_retry():
    """
    模型编造 chunk_id 时，第一次输出应被拒绝，并带错误反馈重试。
    """

    invalid = _valid_draft()
    invalid["daily_plan"][0]["activities"][1]["evidence_refs"] = [
        "G99"
    ]

    generator, client = _generator([
        invalid,
        _valid_draft(),
    ])

    output = _generate(generator)

    assert output["proposal_result"]["attempt_count"] == 2
    assert client.call_count == 2
    assert output["proposal_result"]["validation_errors"]
    assert "上一次输出未通过验证" in client.user_prompts[1]
    assert "guides: G01, G02, G03" in client.user_prompts[1]
    assert "不得生成列表以外的引用" in client.user_prompts[1]


def test_rain_day_all_outdoor_is_rejected():
    """
    中间雨天如果所有活动都是纯户外，即使写了 adjustment 也不能通过。
    """

    invalid = _valid_draft()

    for activity in invalid["daily_plan"][1]["activities"]:
        activity["indoor_outdoor"] = "outdoor"

    generator, _ = _generator(
        [invalid],
        max_attempts=1,
    )

    with pytest.raises(ProposalGenerationError) as exc_info:
        _generate(generator)

    assert "未通过结构化校验" in str(exc_info.value)
    assert any(
        "不能把所有活动都安排为 outdoor" in item
        for item in exc_info.value.validation_errors
    )


def test_hotel_context_is_optional_when_no_hotel_evidence():
    """
    没有 hotel_reviews 时不能编造酒店补充知识，但主方案仍可生成。
    """

    generator, _ = _generator([
        _valid_draft(include_hotel=False)
    ])

    output = _generate(
        generator,
        evidence_pool=_evidence_pool(include_hotel=False),
    )

    hotel_context = output["proposal"]["hotel_context"]

    assert hotel_context["available"] is False
    assert hotel_context["evidence_refs"] == []
    assert "没有该酒店" in hotel_context["summary"]


def test_hotel_context_cannot_be_invented_without_hotel_evidence():
    """
    Evidence Pool 没有酒店文档时，模型不能自行生成周边餐厅或超市信息。
    """

    generator, _ = _generator(
        [_valid_draft(include_hotel=True)],
        max_attempts=1,
    )

    with pytest.raises(ProposalGenerationError) as exc_info:
        _generate(
            generator,
            evidence_pool=_evidence_pool(include_hotel=False),
        )

    assert any(
        "不能生成 hotel_context" in item
        for item in exc_info.value.validation_errors
    )


def test_named_and_unmapped_requirements_must_be_handled():
    """
    Proposal 必须逐项说明指定食物和表达风格要求，不能静默丢弃。
    """

    planning_context = _planning_context()
    planning_context["named_constraints"] = [
        {
            "entity_type": "food",
            "entity_name": "担担面",
            "constraint_mode": "preferred",
            "evidence": "想吃担担面",
        }
    ]
    planning_context["unmapped_requirements"] = [
        {
            "text": "希望旅程有一点电影感",
            "category_guess": "experience_style",
            "impact": "proposal_style",
        }
    ]

    catalog = build_requirement_catalog(
        planning_context=planning_context
    )

    handling = [
        {
            "requirement_key": item["requirement_key"],
            "status": (
                "not_supported"
                if item["requirement_type"] == "named_constraint"
                else "included"
            ),
            "explanation": (
                "当前证据没有明确提到担担面，不编造具体安排。"
                if item["requirement_type"] == "named_constraint"
                else "通过行程主题和文字表达体现电影感。"
            ),
            "evidence_refs": [],
        }
        for item in catalog
    ]

    generator, _ = _generator([
        _valid_draft(requirement_handling=handling)
    ])

    output = _generate(
        generator,
        planning_context=planning_context,
    )

    actual_keys = {
        item["requirement_key"]
        for item in output["proposal"]["requirement_handling"]
    }

    assert actual_keys == {
        item["requirement_key"]
        for item in catalog
    }


def test_requirement_omission_is_rejected():
    """有指定要求时，空 requirement_handling 不能通过。"""

    planning_context = _planning_context()
    planning_context["named_constraints"] = [
        {
            "entity_type": "food",
            "entity_name": "担担面",
            "constraint_mode": "preferred",
            "evidence": "想吃担担面",
        }
    ]

    generator, _ = _generator(
        [_valid_draft(requirement_handling=[])],
        max_attempts=1,
    )

    with pytest.raises(ProposalGenerationError) as exc_info:
        _generate(
            generator,
            planning_context=planning_context,
        )

    assert any(
        "必须完整覆盖 requirement_catalog" in item
        for item in exc_info.value.validation_errors
    )


def test_sources_only_include_evidence_actually_used():
    """最终 sources 不应把整个 Evidence Pool 全部暴露给用户。"""

    generator, _ = _generator([
        _valid_draft()
    ])

    output = _generate(generator)

    source_ids = {
        item["chunk_id"]
        for item in output["proposal"]["sources"]
    }

    assert source_ids == {
        "guide_day_1",
        "guide_rain_day",
        "guide_day_3",
        "safety_rain",
        "packing_rain",
        "hotel_001_context",
    }


def test_unpassed_evidence_is_rejected_before_llm_call():
    """EvidenceGrade 未通过时，ProposalGenerator 不得调用模型。"""

    generator, client = _generator([
        _valid_draft()
    ])

    with pytest.raises(ValueError):
        generator.generate(
            trip_request=_trip_request(),
            planning_context=_planning_context(),
            selection_result=_selection_result(),
            weather_result=_weather_result(),
            evidence_pool=_evidence_pool(),
            evidence_result={
                "status": "repair_required",
                "passed": False,
            },
        )

    assert client.call_count == 0


def test_verifier_feedback_is_included_in_repair_prompt():
    """跨节点 Verifier 反馈应进入新的 Proposal Prompt。"""

    generator, client = _generator([
        _valid_draft()
    ])

    output = generator.generate(
        trip_request=_trip_request(),
        planning_context=_planning_context(),
        selection_result=_selection_result(),
        weather_result=_weather_result(),
        evidence_pool=_evidence_pool(),
        evidence_result={
            "status": "passed",
            "passed": True,
            "next_action": "continue",
        },
        verifier_feedback={
            "source": "deterministic_proposal_verifier_v1",
            "instructions": [
                "修复雨天全天户外安排。"
            ],
        },
    )

    assert output["proposal_result"]["status"] == "generated"
    assert "verifier_feedback" in client.user_prompts[0]
    assert "修复雨天全天户外安排" in client.user_prompts[0]



def test_prompt_uses_short_aliases_and_final_proposal_restores_real_chunk_ids():
    generator, client = _generator([_valid_draft()])
    output = _generate(generator)
    prompt = client.user_prompts[0]
    assert '"ref_id": "G01"' in prompt
    assert '"ref_id": "S01"' in prompt
    assert '"ref_id": "H01"' in prompt
    assert '"ref_id": "P01"' in prompt
    used_refs = {ref for day in output["proposal"]["daily_plan"] for activity in day["activities"] for ref in activity["evidence_refs"]}
    assert "guide_rain_day" in used_refs
    assert "G01" not in used_refs


def test_generic_hotel_dinner_without_evidence_is_normalized_to_free_time():
    draft = _valid_draft()
    draft["daily_plan"][0]["activities"].append({
        "period": "evening", "title": "酒店周边晚餐与休息",
        "description": "在酒店周边自行用餐，之后返回酒店休息。",
        "activity_type": "food", "indoor_outdoor": "flexible",
        "evidence_refs": [], "practical_notes": [],
    })
    generator, _ = _generator([draft])
    output = _generate(generator)
    activity = output["proposal"]["daily_plan"][0]["activities"][-1]
    assert activity["activity_type"] == "free_time"
    assert activity["evidence_refs"] == []


def test_specific_food_activity_without_guide_is_still_rejected():
    draft = _valid_draft()
    draft["daily_plan"][0]["activities"].append({
        "period": "evening", "title": "特色担担面体验",
        "description": "安排具体的成都特色担担面体验。",
        "activity_type": "food", "indoor_outdoor": "indoor",
        "evidence_refs": [], "practical_notes": [],
    })
    generator, _ = _generator([draft], max_attempts=1)
    with pytest.raises(ProposalGenerationError) as exc_info:
        _generate(generator)
    assert any("必须引用至少一个 guides 证据" in item for item in exc_info.value.validation_errors)


def test_generator_uses_explicit_proposal_version():
    generator, _ = _generator([_valid_draft()])
    output = _generate(generator, proposal_version=2)
    assert output["proposal"]["version"] == 2


def test_tool_activity_id_can_ground_knowledge_activity_without_guide_ref():
    """活动 MCP 的稳定 ID 可以独立支撑具体活动，并写入最终来源。"""

    planning_context = _planning_context()
    planning_context["activity_candidates"] = [
        {
            "activity_id": "CD-ACT-001",
            "name": "成都城市文化专题展",
            "city": "成都",
            "available_dates": ["2026-07-02"],
            "opening_hours": "10:00-18:00",
            "indoor_outdoor": "indoor",
            "tags": ["展览", "文化"],
        }
    ]
    draft = _valid_draft()
    draft["daily_plan"][0]["activities"][1].update(
        {
            "title": "成都城市文化专题展",
            "description": "使用本轮活动工具候选安排专题展。",
            "activity_type": "culture",
            "indoor_outdoor": "indoor",
            "evidence_refs": [],
            "tool_activity_id": "CD-ACT-001",
        }
    )
    generator, _ = _generator([draft])

    result = _generate(generator, planning_context=planning_context)

    activity = result["proposal"]["daily_plan"][0]["activities"][1]
    assert activity["tool_activity_id"] == "CD-ACT-001"
    assert result["proposal"]["activity_sources"][0]["activity_id"] == "CD-ACT-001"
