from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from app.llm.deepseek_json_client import StructuredJSONClient
from app.rag.retrieval_planner import (
    RetrievalPlanner,
    RuleBasedRetrievalPlanner,
    _build_coverage_requirements,
)
from app.schemas.agentic_rag_schema import RetrievalPlannerDecision
from app.schemas.retrieval_plan_schema import RetrievalPlan, RetrievalTask


_CATEGORY_LIMITS = {
    "guides": 3,
    "hotel_reviews": 1,
    "safety_notices": 2,
    "packing_checklists": 1,
}


# Guides 只负责提供“目的地 + 用户活动偏好”的知识基础，不能要求知识库
# 预先写好与本次请求天数完全一致的成品行程。LLM 仍负责生成语义 Query，
# 但编译层会删除日期和时长表达，避免模型偶尔忽略 Prompt 后把“两日游完整
# 行程”变成 Evidence Judge 的硬性覆盖条件。Packing、Weather、Flight 等
# 需要日期/天数的模块不使用这个函数，因此它不会删除那些节点需要的事实。
_GUIDE_EXACT_DATE_PATTERN = re.compile(
    r"(?:\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?|\d{1,2}月\d{1,2}日)"
)
_GUIDE_DURATION_PATTERN = re.compile(
    r"(?:第\s*)?(?:\d+|[一二两三四五六七八九十]+)\s*(?:天|日)(?:游|行程)?"
)


def _normalize_guide_query(value: str) -> str:
    """从 Guides Query 中移除精确日期、旅行时长和成品行程措辞。"""

    normalized = _GUIDE_EXACT_DATE_PATTERN.sub(" ", str(value or ""))
    normalized = _GUIDE_DURATION_PATTERN.sub(" ", normalized)
    normalized = re.sub(
        r"(?:完整|具体|明确)?(?:的)?(?:逐日)?(?:旅行)?行程(?:安排|方案)?",
        "活动建议",
        normalized,
    )
    normalized = re.sub(r"逐日(?:安排|方案)", "活动建议", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" ,，;；。")
    return normalized


def _normalize_guide_needs(
    *,
    needs: list[dict[str, Any]],
    decision: RetrievalPlannerDecision,
    city: str,
) -> list[dict[str, Any]]:
    """
    为 Guides Need 注入确定性的证据合同，不采用 LLM 自由生成的成功标准。

    Query 中仍保留模型识别到的美食、文化、夜间活动、慢节奏等用户偏好；
    这里只删除天数和完整行程要求。Evidence Judge 因此判断“攻略是否支持
    这些偏好”，最终 Day 1/Day 2 的编排继续由 Proposal 独立完成。
    """

    guide_queries_by_need: dict[str, list[str]] = {}
    for intent in decision.tasks:
        if intent.category != "guides":
            continue
        query = _normalize_guide_query(intent.semantic_query)
        if not query:
            query = _normalize_guide_query(intent.keyword_query)
        if not query:
            continue
        for need_id in intent.covers_need_ids:
            guide_queries_by_need.setdefault(need_id, []).append(query)

    normalized_needs: list[dict[str, Any]] = []
    for raw_need in needs:
        need = dict(raw_need)
        if str(need.get("category")) != "guides":
            normalized_needs.append(need)
            continue

        need_id = str(need.get("need_id") or "")
        query_focus = list(
            dict.fromkeys(guide_queries_by_need.get(need_id, []))
        )
        if not query_focus:
            fallback_focus = _normalize_guide_query(
                str(need.get("description") or "")
            )
            if fallback_focus:
                query_focus = [fallback_focus]

        focus_text = "；".join(query_focus[:3])
        need["description"] = (
            f"{city}与用户明确活动偏好相关的事实和旅行建议"
            + (f"，检索重点：{focus_text}" if focus_text else "")
            + "。"
        )
        need["success_criteria"] = (
            "证据应直接支持上述目的地及用户活动偏好；"
            "不要求包含旅行天数、精确日期、完整逐日行程或最终旅行方案。"
        )
        # 这个结构化合同供 Evidence Judge 的程序路由使用。它明确表示
        # Guides 的职责是提供偏好证据，而不是提前完成 Proposal 的编排。
        need["need_type"] = "destination_preference_guidance"
        need["evidence_contract"] = "destination_preference_support"
        need["requires_complete_itinerary"] = False
        need["duration_is_coverage_condition"] = False
        normalized_needs.append(need)

    return normalized_needs


class LLMRetrievalPlanner(RetrievalPlanner):
    """
    使用 LLM 理解用户的信息需求，再生成受程序约束的 RetrievalPlan。

    LLM 只决定 Information Needs、检索类别和 query。真正执行前，本类会
    使用 RuleBasedRetrievalPlanner 生成的计划作为 policy template，补齐
    city、doc_type、hotel_id、risk_type、required、top_k 和 repair budget。

    这样正常路径拥有模型的语义理解能力，同时不会让模型改写权威事实。
    Rule Planner 也因此自然成为模型失败时的 baseline/fallback。
    """

    def __init__(
        self,
        *,
        client: StructuredJSONClient,
        fallback_planner: RuleBasedRetrievalPlanner | None = None,
        max_initial_tasks: int = 6,
        max_repair_tasks: int = 2,
        max_tokens: int = 3000,
    ) -> None:
        self.client = client
        self.fallback_planner = fallback_planner or RuleBasedRetrievalPlanner()
        self.max_initial_tasks = max_initial_tasks
        self.max_repair_tasks = max_repair_tasks
        self.max_tokens = max_tokens

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
        """
        生成 initial 或 repair 计划。

        关键步骤只有三个：
        1. 先由规则计划建立本轮允许的 category 和所有硬参数；
        2. LLM 生成 Information Needs 与语义 query；
        3. 将模型意图编译到规则模板，得到现有 Retriever 可直接执行的计划。
        """

        base_plan = self.fallback_planner.plan(
            trip_request=trip_request,
            planning_context=planning_context,
            weather_result=weather_result,
            selection_result=selection_result,
            raw_hotel_results=raw_hotel_results,
            retrieval_feedback=retrieval_feedback,
            rag_retry_count=rag_retry_count,
        )

        try:
            payload = self.client.generate_json(
                system_prompt=self._system_prompt(),
                user_prompt=self._user_prompt(
                    trip_request=trip_request,
                    planning_context=planning_context,
                    weather_result=weather_result,
                    selection_result=selection_result,
                    retrieval_feedback=retrieval_feedback,
                    rag_retry_count=rag_retry_count,
                    base_plan=base_plan,
                ),
                max_tokens=self.max_tokens,
            )
            decision = RetrievalPlannerDecision(**payload)
            plan = self._compile_decision(
                decision,
                base_plan,
                previous_information_needs=list(
                    (retrieval_feedback or {}).get("information_needs") or []
                ),
            )
            if not plan.tasks:
                raise ValueError("LLM Planner 没有生成可执行检索任务")
            return plan
        except Exception as exc:
            # Fallback 直接沿用已生成的规则计划，不再重试另一套复杂融合逻辑。
            base_plan.planner_meta = {
                "planner_type": "rule_fallback",
                "model": self.client.model_id,
                "fallback_used": True,
                "fallback_reason": str(exc),
            }
            return base_plan

    def _compile_decision(
        self,
        decision: RetrievalPlannerDecision,
        base_plan: RetrievalPlan,
        previous_information_needs: list[dict[str, Any]],
    ) -> RetrievalPlan:
        """
        把 LLM 的语义意图编译成安全的 RetrievalPlan。

        每个 category 使用规则计划中的首个任务作为硬参数模板。因此，即使
        LLM 为不同偏好生成多个 guides 任务，它们也只能在当前 city、日期和
        metadata 范围内检索。模型请求规则计划未开放的 category 会被忽略。
        """

        templates = {task.category: task for task in base_plan.tasks}
        generated_needs = [
            item.model_dump(mode="json") for item in decision.information_needs
        ]
        needs = list(
            previous_information_needs
            if base_plan.mode == "repair" and previous_information_needs
            else generated_needs
        )
        required_templates = [task for task in base_plan.tasks if task.required]

        # Rule Gate 要求的类别必须对应至少一个 required need。
        # 若模型遗漏，用规则任务的 purpose 补一个最小目标，
        # 避免 Evidence Judge 在空目标上“自动通过”。
        need_categories = {str(item.get("category")) for item in needs}
        for template in required_templates:
            if template.category in need_categories:
                continue
            needs.append(
                {
                    "need_id": f"need_required_{template.category}",
                    "description": template.purpose,
                    "category": template.category,
                    "success_criteria": (
                        f"证据必须能够支持：{template.purpose}"
                    ),
                }
            )
            need_categories.add(template.category)

        # LLM 的 Guides success_criteria 不参与最终硬路由。即使模型输出
        # “完整两日行程”，编译层也会将其重写为目的地偏好证据合同。
        # repair 轮次沿用的旧 Need 同样会再次规范化，避免旧错误跨轮保留。
        needs = _normalize_guide_needs(
            needs=needs,
            decision=decision,
            city=base_plan.city,
        )

        valid_need_ids = {item["need_id"] for item in needs}
        max_tasks = (
            self.max_repair_tasks
            if base_plan.mode == "repair"
            else self.max_initial_tasks
        )

        category_counts: Counter[str] = Counter()
        compiled: list[RetrievalTask] = []

        for intent in decision.tasks:
            template = templates.get(intent.category)
            if template is None:
                continue
            if category_counts[intent.category] >= _CATEGORY_LIMITS[intent.category]:
                continue

            covers = [
                need_id
                for need_id in intent.covers_need_ids
                if need_id in valid_need_ids
            ]
            if valid_need_ids and not covers:
                continue

            category_counts[intent.category] += 1
            task_number = category_counts[intent.category]
            # required 是类别级 Rule Gate：同类的第一条任务
            # 承载硬要求，其余语义子任务由 Judge 检查 need 覆盖。
            is_category_gate_task = template.required and task_number == 1
            compiled.append(
                RetrievalTask(
                    task_id=(
                        f"agentic_{base_plan.mode}_{intent.category}_{task_number}"
                    ),
                    category=template.category,
                    doc_type=template.doc_type,
                    purpose=intent.purpose,
                    semantic_query=(
                        (
                            _normalize_guide_query(intent.semantic_query)
                            or _normalize_guide_query(template.semantic_query)
                            or f"{base_plan.city} 活动 旅行建议"
                        )
                        if intent.category == "guides"
                        else intent.semantic_query.strip()
                    ),
                    keyword_query=(
                        (
                            _normalize_guide_query(intent.keyword_query)
                            or _normalize_guide_query(template.keyword_query)
                            or f"{base_plan.city} 活动 建议"
                        )
                        if intent.category == "guides"
                        else intent.keyword_query.strip()
                    ),
                    metadata_filter=dict(template.metadata_filter),
                    required=is_category_gate_task,
                    top_k=template.top_k,
                    min_required_chunks=(
                        template.min_required_chunks
                        if is_category_gate_task
                        else 0
                    ),
                    priority=template.priority,
                    reason=intent.reason_code,
                    repair_of=template.repair_of,
                    covers_need_ids=covers,
                )
            )

        # 模型不能通过遗漏任务来删除规则计划中的 required category。
        present_categories = {task.category for task in compiled}
        for template in required_templates:
            if template.category in present_categories:
                continue
            covers = [
                item["need_id"]
                for item in needs
                if item["category"] == template.category
            ]
            copied = template.model_copy(deep=True)
            copied.task_id = f"agentic_{base_plan.mode}_{template.category}_required"
            copied.covers_need_ids = covers
            compiled.append(copied)
            present_categories.add(template.category)

        # required category gate 优先，超出总预算时只裁 optional。
        compiled.sort(key=lambda task: (not task.required, task.priority))
        required_tasks = [task for task in compiled if task.required]
        optional_tasks = [task for task in compiled if not task.required]
        optional_budget = max(0, max_tasks - len(required_tasks))
        compiled = [*required_tasks, *optional_tasks[:optional_budget]]

        # Coverage 必须从最终真正执行的 tasks 派生。若复制 Rule Plan 后只替换
        # task_ids，未来新增 coverage 字段时很容易遗留融合前的旧值。
        coverage_requirements = _build_coverage_requirements(compiled)

        plan_payload = {
            "mode": base_plan.mode,
            "city": base_plan.city,
            "tasks": [task.model_dump(mode="json") for task in compiled],
        }
        plan_id = "rp_agentic_" + hashlib.sha256(
            json.dumps(plan_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

        return RetrievalPlan(
            plan_id=plan_id,
            mode=base_plan.mode,
            strategy_version="llm_retrieval_plan_v1",
            city=base_plan.city,
            date_range=base_plan.date_range,
            tasks=compiled,
            coverage_requirements=coverage_requirements,
            information_needs=needs,
            planner_meta={
                "planner_type": "llm",
                "model": self.client.model_id,
                "fallback_used": False,
                "task_count": len(compiled),
                "reason_codes": [task.reason for task in compiled if task.reason],
            },
            max_repair_rounds=base_plan.max_repair_rounds,
            repair_exhausted=base_plan.repair_exhausted,
            notes=[
                *base_plan.notes,
                *[f"暂不支持的信息需求：{item}" for item in decision.unsupported_needs],
            ],
        )

    def _system_prompt(self) -> str:
        schema = RetrievalPlannerDecision.model_json_schema()
        return (
            "你是 TravelOps 的受控 RAG 检索规划器。"
            "先把用户偏好拆成 Information Needs，再生成检索任务。\n"
            "只允许 guides、hotel_reviews、safety_notices、packing_checklists。\n"
            "一个用户明确目标只生成一个 Information Need；菜品、餐厅、"
            "美食街等只是同一美食偏好的候选证据，不得拆成多个必须同时满足"
            "的 Need。同一 Need 可以由多个不同 query 的任务共同覆盖。\n"
            "每个类别都必须严格遵守输入中的 category_policy_templates_this_round："
            "guides 只负责城市行程与用户活动偏好；hotel_reviews 只负责所选酒店；"
            "safety_notices 只负责当前天气风险与出行安全，不能用于食品安全、"
            "饮食卫生或过敏；packing_checklists 只负责天数和天气相关准备。\n"
            "不得因为用户提到某项活动偏好，就额外创造该活动的安全或打包需求。\n"
            "Information Need 不决定 required；硬性要求由程序根据类别注入。\n"
            "success_criteria 只描述生成答案所需的证据基础，不要要求知识库"
            "预先包含完整最终答案，也不要添加用户未明确要求的具体程度。\n"
            "guides 的 query 不得包含旅行天数、精确日期、完整 N 日行程或逐日"
            "安排，只能包含目的地与用户显式活动偏好；guides 只需提供目的地且"
            "匹配显式偏好的事实与建议，最终 N 日结构由"
            "Proposal 组合；safety_notices 只需覆盖当前 risk_type 和应对知识，"
            "精确日期天气由 Weather MCP 提供。\n"
            "不要输出城市、日期、hotel_id、metadata_filter、top_k 或循环次数，"
            "这些硬参数由程序注入。\n"
            "repair 模式只针对上一轮 missing needs 生成任务。\n"
            "只输出符合下列 JSON Schema 的对象：\n"
            + json.dumps(schema, ensure_ascii=False)
        )

    @staticmethod
    def _user_prompt(
        *,
        trip_request: Mapping[str, Any],
        planning_context: Mapping[str, Any],
        weather_result: Mapping[str, Any] | None,
        selection_result: Mapping[str, Any] | None,
        retrieval_feedback: Mapping[str, Any] | None,
        rag_retry_count: int,
        base_plan: RetrievalPlan,
    ) -> str:
        payload = {
            "mode": base_plan.mode,
            "raw_message": trip_request.get("raw_message"),
            "trip_request": dict(trip_request),
            "planning_context": dict(planning_context),
            "weather_risks": (weather_result or {}).get("risks", []),
            "selection_result": dict(selection_result or {}),
            "retrieval_feedback": dict(retrieval_feedback or {}),
            "rag_retry_count": rag_retry_count,
            # LLM 需要知道类别的业务边界，但仍不能输出或修改这些硬参数。
            # scope_constraints 只保留能说明范围的字段，例如天气风险、酒店 ID。
            "category_policy_templates_this_round": [
                {
                    "category": task.category,
                    "purpose": task.purpose,
                    "required": task.required,
                    "scope_constraints": {
                        key: value
                        for key, value in task.metadata_filter.items()
                        if key in {"risk_type", "hotel_ids", "scenario"}
                    },
                }
                for task in base_plan.tasks
            ],
        }
        return "请生成本轮检索计划：\n" + json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )
