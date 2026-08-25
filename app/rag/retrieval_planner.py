from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from app.schemas.retrieval_plan_schema import RetrievalPlan, RetrievalTask


# ----------------------------------------------------------------------
# PlanningContext 标准偏好 Key → RAG 查询词
# ----------------------------------------------------------------------
ACTIVITY_QUERY_TERMS = {
    "food": "美食 小吃 茶馆",
    "nature_scenery": "自然风景 公园",
    "culture_history": "文化 历史 博物馆 古迹",
    "city_walk": "城市漫步 街区",
    "entertainment": "娱乐 玩乐 夜生活",
    "fitness": "运动 锻炼",
    "adventure": "户外 冒险 徒步",
    "photography": "摄影 拍照 出片",
}

PACE_QUERY_TERMS = {
    "slow": "慢节奏 轻松",
    "low_walking_intensity": "低步行强度 少走路",
    "packed_schedule": "紧凑行程",
}

HOTEL_QUERY_TERMS = {
    "quiet": "安静 隔音 夜间环境",
    "cleanliness": "干净 清洁 卫生",
    "near_subway": "靠近地铁 交通便利",
    "high_rating": "整体体验 口碑",
    "budget_friendly": "性价比",
}

WEATHER_RISK_QUERY_TERMS = {
    "rain": "雨天 降雨 路面湿滑",
    "heavy_rain": "暴雨 强降雨 交通影响",
    "wind": "大风 强风 户外活动",
    "heat": "高温 防暑 正午户外",
}


class RetrievalPlanner(Protocol):
    """RetrievalPlanNode 依赖的检索规划器协议。"""

    def plan(
        self,
        trip_request: Mapping[str, Any],
        planning_context: Mapping[str, Any],
        weather_result: Mapping[str, Any] | None = None,
        selection_result: Mapping[str, Any] | None = None,
        raw_hotel_results: Mapping[str, Any] | None = None,
        retrieval_feedback: Mapping[str, Any] | None = None,
        rag_retry_count: int = 0,
    ) -> RetrievalPlan:
        ...


class RuleBasedRetrievalPlanner:
    """
    规则版 Agentic RAG 检索规划器。

    新的数据边界：
        - CandidateRank / BudgetOptimize 已经先完成酒店和航班选择。
        - RAG 不参与酒店或航班数值评分。
        - RAG 为逐日行程、天气调整、安全提醒、清单和选中酒店说明提供知识。

    因此初次任务规则是：
        1. guides 永远 required。
        2. 天气有风险时 safety_notices required。
        3. packing_checklists 永远 optional。
        4. 只有已经选中酒店时才查询 hotel_reviews，并且它是 optional。
    """

    def __init__(
        self,
        max_repair_rounds: int = 1,
    ) -> None:
        if max_repair_rounds < 0:
            raise ValueError("max_repair_rounds 不能小于 0")

        self.max_repair_rounds = max_repair_rounds

    def plan(
        self,
        trip_request: Mapping[str, Any],
        planning_context: Mapping[str, Any],
        weather_result: Mapping[str, Any] | None = None,
        selection_result: Mapping[str, Any] | None = None,
        raw_hotel_results: Mapping[str, Any] | None = None,
        retrieval_feedback: Mapping[str, Any] | None = None,
        rag_retry_count: int = 0,
    ) -> RetrievalPlan:
        """根据当前 State 生成 initial 或 repair RetrievalPlan。"""

        if not isinstance(trip_request, Mapping):
            raise TypeError("trip_request 必须是 dict-like 对象")

        if not isinstance(planning_context, Mapping):
            raise TypeError("planning_context 必须是 dict-like 对象")

        city = _read_destination(trip_request, planning_context)
        date_range = _read_date_range(trip_request, planning_context)
        weather = weather_result if isinstance(weather_result, Mapping) else {}
        selection = selection_result if isinstance(selection_result, Mapping) else {}
        hotels = raw_hotel_results if isinstance(raw_hotel_results, Mapping) else {}

        # 1. EvidenceGradeNode 给出了明确缺口时，进入局部 repair 模式。
        if _has_repair_request(retrieval_feedback):
            return self._build_repair_plan(
                city=city,
                date_range=date_range,
                trip_request=trip_request,
                planning_context=planning_context,
                weather_result=weather,
                selection_result=selection,
                raw_hotel_results=hotels,
                retrieval_feedback=retrieval_feedback or {},
                rag_retry_count=rag_retry_count,
            )

        # 2. 正常情况下构造完整 initial 检索计划。
        return self._build_initial_plan(
            city=city,
            date_range=date_range,
            trip_request=trip_request,
            planning_context=planning_context,
            weather_result=weather,
            selection_result=selection,
            raw_hotel_results=hotels,
        )

    def _build_initial_plan(
        self,
        *,
        city: str,
        date_range: dict[str, str | None],
        trip_request: Mapping[str, Any],
        planning_context: Mapping[str, Any],
        weather_result: Mapping[str, Any],
        selection_result: Mapping[str, Any],
        raw_hotel_results: Mapping[str, Any],
    ) -> RetrievalPlan:
        """构造首次完整检索计划。"""

        tasks: list[RetrievalTask] = []
        notes: list[str] = []

        days = _read_days(trip_request, planning_context)
        preference_weights = _read_preference_weights(planning_context)
        named_constraints = _read_named_constraints(planning_context, trip_request)
        weather_risks = _extract_weather_risk_types(weather_result)
        selected_hotel = _extract_selected_hotel(
            selection_result=selection_result,
            raw_hotel_results=raw_hotel_results,
        )

        # 1. 攻略证据是 ProposalNode 生成逐日行程的核心知识，因此必须检索。
        tasks.append(
            _build_guide_task(
                city=city,
                days=days,
                preference_weights=preference_weights,
                named_constraints=named_constraints,
                weather_risks=weather_risks,
            )
        )

        # 2. 只有 BudgetOptimize 已经选出酒店后，才查询该酒店的补充资料。
        #    酒店 RAG 只用于周边环境、交通体验、优点和限制的说明，
        #    不参与酒店数值评分，所以 required=False。
        if selected_hotel:
            tasks.append(
                _build_selected_hotel_task(
                    city=city,
                    hotel=selected_hotel,
                    preference_weights=preference_weights,
                )
            )
        else:
            notes.append(
                "当前没有 selection_result 中的选中酒店，因此跳过可选 hotel_reviews 任务。"
            )

        # 3. 只有天气存在风险时，安全知识才是必需证据。
        if weather_risks:
            tasks.append(
                _build_safety_notice_task(
                    city=city,
                    risk_types=weather_risks,
                )
            )
        else:
            weather_status = str(weather_result.get("status") or "")
            if weather_status in {"unavailable", "no_data", "failed"}:
                notes.append("天气数据不可用，本轮无法生成强制安全提醒任务。")
            else:
                notes.append("天气暂无标准风险，本轮不生成 safety_notices 任务。")

        # 4. 出行清单属于增强信息，不应因为缺少清单文档阻断旅行方案。
        tasks.append(
            _build_packing_task(
                city=city,
                days=days,
                risk_types=weather_risks,
            )
        )

        coverage_requirements = _build_coverage_requirements(tasks)

        plan_payload = {
            "mode": "initial",
            "city": city,
            "date_range": date_range,
            "tasks": [task.to_state_dict() for task in tasks],
        }

        return RetrievalPlan(
            plan_id=_build_plan_id(plan_payload),
            mode="initial",
            strategy_version="rule_retrieval_plan_v2",
            city=city,
            date_range=date_range,
            tasks=tasks,
            coverage_requirements=coverage_requirements,
            max_repair_rounds=self.max_repair_rounds,
            repair_exhausted=False,
            notes=notes,
        )

    def _build_repair_plan(
        self,
        *,
        city: str,
        date_range: dict[str, str | None],
        trip_request: Mapping[str, Any],
        planning_context: Mapping[str, Any],
        weather_result: Mapping[str, Any],
        selection_result: Mapping[str, Any],
        raw_hotel_results: Mapping[str, Any],
        retrieval_feedback: Mapping[str, Any],
        rag_retry_count: int,
    ) -> RetrievalPlan:
        """
        根据 EvidenceGradeNode 的缺口只生成必要的修复任务。

        当前 required 类别通常只有：
            - guides
            - 天气有风险时的 safety_notices

        hotel_reviews 和 packing_checklists 是 optional，正常情况下不会触发 repair。
        """

        # 1. 达到最大修复次数后返回空 repair plan，防止无限循环。
        if rag_retry_count >= self.max_repair_rounds:
            payload = {
                "mode": "repair",
                "city": city,
                "date_range": date_range,
                "tasks": [],
                "repair_exhausted": True,
            }

            return RetrievalPlan(
                plan_id=_build_plan_id(payload),
                mode="repair",
                strategy_version="rule_retrieval_plan_v2",
                city=city,
                date_range=date_range,
                tasks=[],
                coverage_requirements={},
                max_repair_rounds=self.max_repair_rounds,
                repair_exhausted=True,
                notes=["RAG 修复次数已经达到上限，不再生成新任务。"],
            )

        tasks: list[RetrievalTask] = []
        notes: list[str] = []

        days = _read_days(trip_request, planning_context)
        preference_weights = _read_preference_weights(planning_context)
        named_constraints = _read_named_constraints(planning_context, trip_request)
        weather_risks = _extract_weather_risk_types(weather_result)
        missing_doc_types = _normalize_missing_doc_types(
            retrieval_feedback.get("missing_doc_types")
        )
        missing_risk_types = _ensure_string_list(
            retrieval_feedback.get("missing_risk_types")
        )
        missing_hotel_ids = _ensure_string_list(
            retrieval_feedback.get("missing_hotel_ids")
        )

        # 2. 攻略证据不足时扩大召回数量并重新查询。
        if "guides" in missing_doc_types:
            guide_task = _build_guide_task(
                city=city,
                days=days,
                preference_weights=preference_weights,
                named_constraints=named_constraints,
                weather_risks=weather_risks,
            )
            guide_task.task_id = "repair_guide_main"
            guide_task.repair_of = "guide_main"
            guide_task.top_k = 12
            guide_task.reason = "EvidenceGradeNode 判断攻略证据不足。"
            tasks.append(guide_task)

        # 3. 安全证据不足时，只检索缺失的天气风险类型。
        if "safety_notices" in missing_doc_types or missing_risk_types:
            target_risks = missing_risk_types or weather_risks
            if target_risks:
                safety_task = _build_safety_notice_task(
                    city=city,
                    risk_types=target_risks,
                )
                safety_task.task_id = "repair_weather_safety"
                safety_task.repair_of = "weather_safety"
                safety_task.top_k = 10
                safety_task.reason = "EvidenceGradeNode 判断天气安全证据不足。"
                tasks.append(safety_task)

        # 4. optional 任务通常不触发 repair；保留这段仅支持手工或未来策略请求。
        if "hotel_reviews" in missing_doc_types or missing_hotel_ids:
            selected_hotel = _extract_selected_hotel(
                selection_result=selection_result,
                raw_hotel_results=raw_hotel_results,
            )

            if selected_hotel and (
                not missing_hotel_ids
                or str(selected_hotel.get("hotel_id")) in missing_hotel_ids
            ):
                hotel_task = _build_selected_hotel_task(
                    city=city,
                    hotel=selected_hotel,
                    preference_weights=preference_weights,
                )
                hotel_task.task_id = "repair_selected_hotel_info"
                hotel_task.repair_of = "selected_hotel_info"
                hotel_task.top_k = 8
                tasks.append(hotel_task)

        if "packing_checklists" in missing_doc_types:
            packing_task = _build_packing_task(
                city=city,
                days=days,
                risk_types=weather_risks,
            )
            packing_task.task_id = "repair_packing_checklist"
            packing_task.repair_of = "packing_weather"
            packing_task.top_k = 8
            tasks.append(packing_task)

        if not tasks:
            notes.append(
                "retrieval_feedback 没有生成可执行修复任务；可能只缺少 optional 证据。"
            )

        coverage_requirements = _build_coverage_requirements(tasks)
        payload = {
            "mode": "repair",
            "city": city,
            "date_range": date_range,
            "tasks": [task.to_state_dict() for task in tasks],
        }

        return RetrievalPlan(
            plan_id=_build_plan_id(payload),
            mode="repair",
            strategy_version="rule_retrieval_plan_v2",
            city=city,
            date_range=date_range,
            tasks=tasks,
            coverage_requirements=coverage_requirements,
            max_repair_rounds=self.max_repair_rounds,
            repair_exhausted=False,
            notes=notes,
        )


def build_retrieval_plan(
    trip_request: Mapping[str, Any],
    planning_context: Mapping[str, Any],
    weather_result: Mapping[str, Any] | None = None,
    selection_result: Mapping[str, Any] | None = None,
    raw_hotel_results: Mapping[str, Any] | None = None,
    retrieval_feedback: Mapping[str, Any] | None = None,
    rag_retry_count: int = 0,
    planner: RetrievalPlanner | None = None,
) -> dict[str, Any]:
    """Node、测试和脚本共用的 RetrievalPlan 构造入口。"""

    active_planner = planner or RuleBasedRetrievalPlanner()

    plan = active_planner.plan(
        trip_request=trip_request,
        planning_context=planning_context,
        weather_result=weather_result,
        selection_result=selection_result,
        raw_hotel_results=raw_hotel_results,
        retrieval_feedback=retrieval_feedback,
        rag_retry_count=rag_retry_count,
    )

    return plan.to_state_dict()


def _build_guide_task(
    *,
    city: str,
    days: int,
    preference_weights: Mapping[str, Any],
    named_constraints: Sequence[Mapping[str, Any]],
    weather_risks: list[str],
) -> RetrievalTask:
    """构造城市攻略与逐日行程知识任务。"""

    # 1. 只把权重较高的活动和节奏偏好写进 query，避免 query 过长。
    activity_terms = _top_weighted_terms(
        weights=_read_weight_bucket(preference_weights, "activity"),
        term_map=ACTIVITY_QUERY_TERMS,
        threshold=0.65,
        limit=4,
    )
    pace_terms = _top_weighted_terms(
        weights=_read_weight_bucket(preference_weights, "pace"),
        term_map=PACE_QUERY_TERMS,
        threshold=0.65,
        limit=2,
    )

    # 2. 指定食物、景点和活动需要进入攻略检索；没有 RAG 证据时不能编造。
    named_terms = [
        str(item.get("entity_name"))
        for item in named_constraints
        if item.get("entity_type") in {"food", "attraction", "activity"}
        and item.get("constraint_mode") != "avoid"
        and item.get("entity_name")
    ]

    # 3. 天气风险转换为“怎样调整行程”的检索词。
    weather_terms = _weather_planning_terms(weather_risks)

    semantic_query = _join_query_terms(
        [
            city,
            _days_label(days),
            *named_terms,
            *activity_terms,
            *pace_terms,
            *weather_terms,
            "逐日行程",
            "室内外合理安排",
        ]
    )
    keyword_query = _join_query_terms(
        [city, _days_label(days), *named_terms, *activity_terms, *pace_terms]
    )

    return RetrievalTask(
        task_id="guide_main",
        category="guides",
        doc_type="guide",
        purpose="检索目的地攻略、景点特点和逐日活动安排知识。",
        semantic_query=semantic_query,
        keyword_query=keyword_query,
        metadata_filter={
            "city": city,
            "doc_type": "guide",
        },
        required=True,
        top_k=8,
        min_required_chunks=2,
        priority=1,
        reason="ProposalNode 需要攻略证据生成有来源的逐日行程。",
    )


def _build_selected_hotel_task(
    *,
    city: str,
    hotel: Mapping[str, Any],
    preference_weights: Mapping[str, Any],
) -> RetrievalTask:
    """
    为 BudgetOptimize 已选酒店构造可选补充信息任务。

    这条任务的证据不进入酒店评分，
    只用于最终向用户说明区域、交通、周边设施、体验和限制。
    """

    hotel_id = str(hotel.get("hotel_id") or "").strip()
    hotel_name = str(hotel.get("name") or hotel.get("hotel_name") or "").strip()

    if not hotel_id:
        raise ValueError("selection_result 中的选中酒店缺少 hotel_id")

    hotel_terms = _top_weighted_terms(
        weights=_read_weight_bucket(preference_weights, "hotel"),
        term_map=HOTEL_QUERY_TERMS,
        threshold=0.60,
        limit=4,
    )

    semantic_query = _join_query_terms(
        [
            city,
            hotel_name,
            *hotel_terms,
            "周边餐饮",
            "超市便利店",
            "交通体验",
            "优点和限制",
        ]
    )
    keyword_query = _join_query_terms(
        [hotel_name, *hotel_terms, "周边 交通 餐饮 便利设施"]
    )

    return RetrievalTask(
        task_id="selected_hotel_info",
        category="hotel_reviews",
        doc_type="hotel_reviews",
        purpose="检索选中酒店的非实时补充信息和住宿体验说明。",
        semantic_query=semantic_query,
        keyword_query=keyword_query,
        metadata_filter={
            "city": city,
            "doc_type": "hotel_reviews",
            "hotel_ids": [hotel_id],
        },
        required=False,
        top_k=6,
        min_required_chunks=0,
        priority=3,
        reason="酒店已经由 Mock 数据完成选择，RAG 只补充说明，不参与评分。",
    )


def _build_safety_notice_task(
    *,
    city: str,
    risk_types: list[str],
) -> RetrievalTask:
    """构造天气风险安全知识任务。"""

    normalized_risks = list(dict.fromkeys(risk_types))
    risk_terms = _risk_query_terms(normalized_risks)

    return RetrievalTask(
        task_id="weather_safety",
        category="safety_notices",
        doc_type="safety_notice",
        purpose="检索与逐日天气风险对应的安全提醒和行程调整知识。",
        semantic_query=_join_query_terms(
            [city, *risk_terms, "旅行安全", "交通缓冲", "户外活动调整"]
        ),
        keyword_query=_join_query_terms(
            [city, *risk_terms, "出行安全", "交通影响"]
        ),
        metadata_filter={
            "city": city,
            "doc_type": "safety_notice",
            "risk_type": normalized_risks,
            "risk_level": ["medium", "high"],
        },
        required=True,
        top_k=max(6, len(normalized_risks) * 3),
        min_required_chunks=max(1, len(normalized_risks)),
        priority=1,
        reason="天气存在风险，Proposal 必须有相应安全提醒和替代安排。",
    )


def _build_packing_task(
    *,
    city: str,
    days: int,
    risk_types: list[str],
) -> RetrievalTask:
    """构造可选出行清单任务。"""

    risk_terms = _risk_query_terms(risk_types)

    if risk_types:
        semantic_query = _join_query_terms(
            [city, _days_label(days), *risk_terms, "出行准备清单", "随身用品"]
        )
        keyword_query = _join_query_terms(
            [_days_label(days), *risk_terms, "出行清单"]
        )
    else:
        semantic_query = _join_query_terms(
            [city, _days_label(days), "城市短途旅行", "出行准备清单"]
        )
        keyword_query = _join_query_terms(
            [_days_label(days), "城市旅行", "出行清单"]
        )

    scenarios = (
        ["rainy_city_trip", "general"]
        if any(risk in {"rain", "heavy_rain"} for risk in risk_types)
        else ["general"]
    )

    return RetrievalTask(
        task_id="packing_weather",
        category="packing_checklists",
        doc_type="packing_checklist",
        purpose="检索与旅行天数和天气相匹配的准备清单。",
        semantic_query=semantic_query,
        keyword_query=keyword_query,
        metadata_filter={
            "doc_type": "packing_checklist",
            "scenario": scenarios,
        },
        required=False,
        top_k=5,
        min_required_chunks=0,
        priority=4,
        reason="清单属于补充建议，缺失时不阻断旅行方案。",
    )


def _build_coverage_requirements(
    tasks: Sequence[RetrievalTask],
) -> dict[str, dict[str, Any]]:
    """从 RetrievalTasks 自动生成 EvidenceGrade 验收要求。"""

    result: dict[str, dict[str, Any]] = {}

    for task in tasks:
        category_result = result.setdefault(
            task.category,
            {
                "required": False,
                "min_required_chunks": 0,
                "task_ids": [],
            },
        )

        category_result["required"] = (
            bool(category_result["required"]) or task.required
        )
        category_result["min_required_chunks"] = max(
            int(category_result["min_required_chunks"]),
            task.min_required_chunks,
        )
        category_result["task_ids"].append(task.task_id)

        # 可选酒店任务不产生 required_hotel_ids，避免缺少酒店 RAG 阻断流程。
        if task.category == "hotel_reviews" and task.required:
            hotel_ids = task.metadata_filter.get("hotel_ids")
            if isinstance(hotel_ids, list):
                category_result["required_hotel_ids"] = [
                    str(hotel_id) for hotel_id in hotel_ids
                ]

    return result


def _extract_selected_hotel(
    *,
    selection_result: Mapping[str, Any],
    raw_hotel_results: Mapping[str, Any],
) -> dict[str, Any] | None:
    """
    从未来 BudgetOptimizeNode 输出中读取选中酒店，并补齐 Mock 详情。

    兼容以下结构：
        selection_result["selected_hotel"]
        selection_result["selected_hotel_id"]
        selection_result["selected_combination"]["hotel_id"]
        selection_result["selected_combination"]["hotel"]
    """

    selected_hotel = selection_result.get("selected_hotel")
    if isinstance(selected_hotel, Mapping) and selected_hotel.get("hotel_id"):
        base = dict(selected_hotel)
    else:
        combination = selection_result.get("selected_combination")
        combination = combination if isinstance(combination, Mapping) else {}

        combination_hotel = combination.get("hotel")
        if isinstance(combination_hotel, Mapping) and combination_hotel.get("hotel_id"):
            base = dict(combination_hotel)
        else:
            hotel_id = (
                selection_result.get("selected_hotel_id")
                or combination.get("hotel_id")
            )
            base = {"hotel_id": hotel_id} if hotel_id else {}

    hotel_id = str(base.get("hotel_id") or "").strip()
    if not hotel_id:
        return None

    # 1. 使用 raw_hotel_results 补齐酒店名称、区域等 Mock 事实。
    for item in _extract_hotel_items(raw_hotel_results):
        if str(item.get("hotel_id")) == hotel_id:
            return {**item, **base}

    return base


def _read_destination(
    trip_request: Mapping[str, Any],
    planning_context: Mapping[str, Any],
) -> str:
    """优先从 PlanningContext 标准化请求中读取目的地。"""

    request = planning_context.get("request")
    if isinstance(request, Mapping):
        value = request.get("destination")
        if value and str(value).strip():
            return str(value).strip()

    value = trip_request.get("destination")
    if value and str(value).strip():
        return str(value).strip()

    raise ValueError("无法生成 retrieval_plan：destination 不能为空")


def _read_date_range(
    trip_request: Mapping[str, Any],
    planning_context: Mapping[str, Any],
) -> dict[str, str | None]:
    """读取旅行日期范围。"""

    request = planning_context.get("request")
    request = request if isinstance(request, Mapping) else {}
    start = request.get("start_date") or trip_request.get("start_date")
    end = request.get("end_date") or trip_request.get("end_date") or start

    return {
        "start": str(start) if start else None,
        "end": str(end) if end else None,
    }


def _read_days(
    trip_request: Mapping[str, Any],
    planning_context: Mapping[str, Any],
) -> int:
    """读取旅行天数并保证至少为 1。"""

    request = planning_context.get("request")
    request = request if isinstance(request, Mapping) else {}
    value = request.get("days") or trip_request.get("days") or 1

    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return 1


def _read_preference_weights(
    planning_context: Mapping[str, Any],
) -> dict[str, Any]:
    """读取 PlanningContext 的连续偏好权重。"""

    value = planning_context.get("preference_weights")
    return dict(value) if isinstance(value, Mapping) else {}


def _read_named_constraints(
    planning_context: Mapping[str, Any],
    trip_request: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """优先读取 PlanningContext 中已经标准化的指定实体约束。"""

    value = planning_context.get("named_constraints")
    if not isinstance(value, list):
        value = trip_request.get("named_constraints")

    if not isinstance(value, list):
        return []

    return [dict(item) for item in value if isinstance(item, Mapping)]


def _extract_hotel_items(
    raw_hotel_results: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """从 HotelSearchNode 输出中读取合法酒店对象。"""

    items = raw_hotel_results.get("items")
    if not isinstance(items, list):
        return []

    return [
        dict(item)
        for item in items
        if isinstance(item, Mapping) and item.get("hotel_id")
    ]


def _extract_weather_risk_types(
    weather_result: Mapping[str, Any],
) -> list[str]:
    """从 risks 和 daily.risk_tags 中提取标准风险类型。"""

    risks: list[str] = []

    raw_risks = weather_result.get("risks")
    if isinstance(raw_risks, list):
        for item in raw_risks:
            if isinstance(item, Mapping) and item.get("risk_type"):
                risks.append(str(item["risk_type"]))

    daily = weather_result.get("daily")
    if isinstance(daily, list):
        for item in daily:
            if not isinstance(item, Mapping):
                continue
            tags = item.get("risk_tags")
            if isinstance(tags, list):
                risks.extend(str(tag) for tag in tags if tag)

    return list(dict.fromkeys(risks))


def _read_weight_bucket(
    preference_weights: Mapping[str, Any],
    bucket_name: str,
) -> dict[str, float]:
    """读取某个偏好桶并安全转换为 float。"""

    value = preference_weights.get(bucket_name)
    if not isinstance(value, Mapping):
        return {}

    output: dict[str, float] = {}
    for key, score in value.items():
        try:
            output[str(key)] = float(score)
        except (TypeError, ValueError):
            continue

    return output


def _top_weighted_terms(
    *,
    weights: Mapping[str, float],
    term_map: Mapping[str, str],
    threshold: float,
    limit: int,
) -> list[str]:
    """选择权重最高且达到阈值的 RAG 查询词。"""

    candidates = [
        (key, score)
        for key, score in weights.items()
        if score >= threshold and key in term_map
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)

    return [term_map[key] for key, _ in candidates[:limit]]


def _weather_planning_terms(risk_types: Sequence[str]) -> list[str]:
    """把天气风险转换成逐日活动调整查询词。"""

    terms: list[str] = []

    if any(risk in {"rain", "heavy_rain"} for risk in risk_types):
        terms.extend(["雨天室内景点", "雨天行程调整"])
    if "wind" in risk_types:
        terms.append("大风减少户外活动")
    if "heat" in risk_types:
        terms.append("高温减少正午户外")

    return terms


def _risk_query_terms(risk_types: Sequence[str]) -> list[str]:
    """把标准风险类型转换成中文检索短语。"""

    return list(
        dict.fromkeys(
            WEATHER_RISK_QUERY_TERMS[risk]
            for risk in risk_types
            if risk in WEATHER_RISK_QUERY_TERMS
        )
    )


def _days_label(days: int) -> str:
    """将天数转换成中文查询表达。"""

    mapping = {
        1: "一日游",
        2: "两日游",
        3: "三日游",
        4: "四日游",
        5: "五日游",
        6: "六日游",
        7: "七日游",
    }
    return mapping.get(days, f"{days}日游")


def _join_query_terms(terms: Sequence[str]) -> str:
    """清理、保持顺序去重并连接 Query 片段。"""

    cleaned = [str(term).strip() for term in terms if term and str(term).strip()]
    return " ".join(dict.fromkeys(cleaned))


def _has_repair_request(
    retrieval_feedback: Mapping[str, Any] | None,
) -> bool:
    """判断 EvidenceGradeNode 是否请求修复检索。"""

    if not isinstance(retrieval_feedback, Mapping):
        return False

    return any(
        bool(retrieval_feedback.get(field))
        for field in (
            "missing_doc_types",
            "missing_hotel_ids",
            "missing_risk_types",
            "repair_queries",
        )
    )


def _normalize_missing_doc_types(value: Any) -> list[str]:
    """将 doc_type 单数/复数写法统一为逻辑 category。"""

    aliases = {
        "guide": "guides",
        "guides": "guides",
        "hotel_review": "hotel_reviews",
        "hotel_reviews": "hotel_reviews",
        "safety_notice": "safety_notices",
        "safety_notices": "safety_notices",
        "packing_checklist": "packing_checklists",
        "packing_checklists": "packing_checklists",
    }

    return list(
        dict.fromkeys(
            aliases[item]
            for item in _ensure_string_list(value)
            if item in aliases
        )
    )


def _ensure_string_list(value: Any) -> list[str]:
    """把任意列表安全转换成非空字符串列表。"""

    if not isinstance(value, list):
        return []

    return [str(item) for item in value if item is not None and str(item).strip()]


def _build_plan_id(payload: Mapping[str, Any]) -> str:
    """根据计划内容生成稳定 SHA-256 plan_id。"""

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"rp_{digest}"
