from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Protocol

from pydantic import ValidationError

from app.common.config import settings
from app.llm.deepseek_json_client import (
    DeepSeekJSONClient,
    StructuredJSONClient,
)
from app.schemas.proposal_schema import (
    DailyWeatherFact,
    EvidenceBackedNoteDraft,
    ExecutionBoundary,
    FinalHotelContext,
    HotelContextDraft,
    ProposalBudgetSummary,
    ProposalDay,
    ProposalDraft,
    ProposalGenerationResult,
    ProposalSourceRef,
    SelectedFlightFact,
    SelectedFlights,
    SelectedHotelFact,
    TripOverview,
    TripProposal,
    WeatherOverview,
)


PROPOSAL_GENERATOR_VERSION = "llm_locked_facts_rag_v2"
PROPOSAL_PROMPT_VERSION = "proposal_prompt_v2"

# RAG 证据在 Prompt 中的稳定顺序。
# 先让模型看到攻略和安全知识，再看到酒店补充与行李清单。
EVIDENCE_CATEGORY_ORDER = {
    "guides": 1,
    "safety_notices": 2,
    "hotel_reviews": 3,
    "packing_checklists": 4,
}

# Prompt 内不再让 LLM 复制真实的长 chunk_id。
# 每一类证据使用稳定短别名：
#     G01：攻略 guide
#     S01：安全提醒 safety_notice
#     H01：酒店补充 hotel_review
#     P01：出行清单 packing_checklist
#
# 最终 Proposal 仍然保存真实 chunk_id；短别名只存在于 LLM Prompt 和
# ProposalDraft 校验阶段，用来降低模型抄错或自行拼接长 ID 的概率。
EVIDENCE_ALIAS_PREFIX = {
    "guides": "G",
    "safety_notices": "S",
    "hotel_reviews": "H",
    "packing_checklists": "P",
}

# 这些短语表示“泛化的自由安排”，而不是由攻略知识支持的具体活动。
# 如果模型没有给出 evidence_refs，却把这类内容标成 food / leisure / other，
# 程序会保守地归一化为 free_time，避免把“自行吃饭和休息”误判成
# 必须引用攻略的知识型活动。
GENERIC_FREE_TIME_PHRASES = (
    "酒店周边晚餐",
    "酒店周边用餐",
    "附近自行用餐",
    "自行用餐",
    "自由用餐",
    "晚餐与休息",
    "用餐与休息",
    "酒店休息",
    "返回酒店休息",
    "整理行李",
    "自由活动",
    "自行安排",
)

# 这些活动属于知识型活动，必须引用已经通过 EvidenceGrade 的 RAG 证据。
# 物流活动（接送、入住、自由活动）可以没有 RAG 引用。
KNOWLEDGE_ACTIVITY_TYPES = {
    "sightseeing",
    "culture",
    "food",
    "nature",
    "city_walk",
    "leisure",
    "shopping",
    "other",
}

# 出现这些天气风险时，逐日计划必须提供 weather_adjustment。
WEATHER_RISKS_REQUIRING_ADJUSTMENT = {
    "rain",
    "heavy_rain",
    "wind",
    "heat",
}

# 雨天不能把全天活动全部写成纯户外。
RAIN_RISK_TYPES = {
    "rain",
    "heavy_rain",
}


class ProposalGenerator(Protocol):
    """
    Proposal 生成器协议。

    ProposalNode 只依赖这个协议，不直接绑定 DeepSeek。
    单元测试可以注入 FakeProposalGenerator 或 Fake JSON Client。
    """

    model_id: str

    def generate(
        self,
        *,
        trip_request: Mapping[str, Any],
        planning_context: Mapping[str, Any],
        selection_result: Mapping[str, Any],
        weather_result: Mapping[str, Any],
        evidence_pool: Sequence[Mapping[str, Any]],
        evidence_result: Mapping[str, Any],
        verifier_feedback: Mapping[str, Any] | None = None,
        proposal_version: int = 1,
    ) -> dict[str, Any]:
        """生成最终 TripProposal 和 ProposalGenerationResult。"""
        ...


class ProposalGenerationError(RuntimeError):
    """
    Proposal 在限定尝试次数内仍无法通过 Schema 或业务校验。

    validation_errors 会被 ProposalNode 写入 proposal_result，
    方便 Trace 和后续 Eval 分析具体失败原因。
    """

    def __init__(
        self,
        message: str,
        *,
        validation_errors: Sequence[str] | None = None,
        attempt_count: int = 0,
    ) -> None:
        super().__init__(message)
        self.validation_errors = list(validation_errors or [])
        self.attempt_count = attempt_count


class LLMProposalGenerator:
    """
    使用 DeepSeek Structured JSON 生成旅行方案内容。

    最重要的架构原则：

        LLM 只生成：
            - 每日活动；
            - 天气调整说明；
            - 酒店补充知识；
            - 安全与行李建议；
            - 偏好和指定要求的处理说明。

        程序锁定并注入：
            - 去程和返程航班；
            - 酒店；
            - 航班与酒店价格；
            - 每日天气；
            - BudgetOptimize 结果；
            - Evidence 来源目录。

    因此即使模型试图在文字中修改航班或酒店，
    最终 TripProposal 仍然只会使用 selection_result 中的确定性事实。
    """

    def __init__(
        self,
        *,
        client: StructuredJSONClient,
        max_tokens: int = 8192,
        max_attempts: int = 2,
        max_evidence_items: int = 24,
        max_evidence_chars: int = 1200,
    ) -> None:
        """
        Args:
            client:
                结构化 JSON 模型客户端。

            max_tokens:
                Proposal Draft 的最大输出 Token 数。
                三日行程通常需要比 InputExtract 更大的输出空间。

            max_attempts:
                模型输出未通过 Pydantic 或业务校验时，最多尝试多少次。
                当前建议 2：首次生成 + 一次带错误反馈的修复。

            max_evidence_items:
                最多放入 Prompt 的证据 chunk 数量。
                防止 Evidence Pool 过大导致上下文膨胀。

            max_evidence_chars:
                每个证据 chunk 最多放入 Prompt 的字符数。
                完整正文仍保留在 State，本参数只控制 Prompt 长度。
        """

        if max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")

        if max_attempts <= 0:
            raise ValueError("max_attempts 必须大于 0")

        if max_evidence_items <= 0:
            raise ValueError("max_evidence_items 必须大于 0")

        if max_evidence_chars <= 0:
            raise ValueError("max_evidence_chars 必须大于 0")

        self.client = client
        self.model_id = client.model_id
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.max_evidence_items = max_evidence_items
        self.max_evidence_chars = max_evidence_chars

    def generate(
        self,
        *,
        trip_request: Mapping[str, Any],
        planning_context: Mapping[str, Any],
        selection_result: Mapping[str, Any],
        weather_result: Mapping[str, Any],
        evidence_pool: Sequence[Mapping[str, Any]],
        evidence_result: Mapping[str, Any],
        verifier_feedback: Mapping[str, Any] | None = None,
        proposal_version: int = 1,
    ) -> dict[str, Any]:
        """
        生成最终 Proposal。

        执行步骤：
            1. 从 State 构建锁定事实；
            2. 将 Evidence Pool 整理成受控证据目录；
            3. 构建 Pydantic JSON Schema Prompt；
            4. 调用 DeepSeek 生成 ProposalDraft；
            5. 校验日期、天气、证据引用和用户要求；
            6. 由程序注入航班、酒店、天气和预算；
            7. 生成可追溯的 sources 和 locked_fact_hash。
        """

        # 1. EvidenceGrade 必须已经允许工作流继续。
        #    ProposalNode 不应该使用未经验证或缺失的证据生成方案。
        if evidence_result.get("passed") is not True:
            raise ValueError(
                "evidence_result.passed 必须为 True，请先完成 EvidenceGradeNode"
            )

        # Proposal 版本号由工作流状态确定，不允许 LLM 自己生成。
        # 初次规划为 1；Revision 后由 ProposalNode 传入 base_version + 1。
        if proposal_version < 1:
            raise ValueError("proposal_version 必须大于等于 1")

        # 2. 构建全部锁定事实。
        #    这部分由程序生成，不交给 LLM 输出。
        locked_context = build_locked_proposal_context(
            trip_request=trip_request,
            planning_context=planning_context,
            selection_result=selection_result,
            weather_result=weather_result,
        )

        # 3. 选择并整理进入 Prompt 的证据，同时保留完整 Evidence Map。
        evidence_context = _build_evidence_context(
            evidence_pool=evidence_pool,
            selected_hotel_id=(
                locked_context["selected_hotel"].hotel_id
            ),
            max_items=self.max_evidence_items,
            max_chars=self.max_evidence_chars,
        )

        # 4. 把指定实体和未映射要求转成稳定 requirement_key。
        #    模型必须逐项说明如何处理，不能静默丢弃。
        requirement_catalog = build_requirement_catalog(
            planning_context=planning_context,
        )
        tool_activity_context = _build_tool_activity_context(planning_context)

        system_prompt = _build_system_prompt()
        base_user_prompt = _build_user_prompt(
            locked_context=locked_context,
            planning_context=planning_context,
            evidence_catalog=evidence_context["prompt_items"],
            evidence_reference_contract=evidence_context["reference_contract"],
            requirement_catalog=requirement_catalog,
            tool_activity_catalog=tool_activity_context["prompt_items"],
            verifier_feedback=(
                verifier_feedback
                if isinstance(verifier_feedback, Mapping)
                else {}
            ),
        )

        validation_errors: list[str] = []

        # 5. 使用有界内部重试。
        #    这是模型 JSON/Schema 修复，不是 LangGraph 的 Proposal-Verifier 循环。
        for attempt in range(1, self.max_attempts + 1):
            user_prompt = base_user_prompt

            if validation_errors:
                # 6. 第二次尝试只附加简短错误反馈，
                #    不把内部异常堆栈或完整 State 暴露给模型。
                user_prompt += (
                    "\n\n上一次输出未通过验证，请重新生成完整 JSON。"
                    "必须修复以下问题：\n- "
                    + "\n- ".join(validation_errors[-8:])
                    + "\n\n"
                    + _build_retry_guidance(evidence_context)
                )

            try:
                payload = self.client.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=self.max_tokens,
                )

                # 7. Pydantic 先验证字段类型、枚举和嵌套结构。
                draft = ProposalDraft(**payload)

                # 8. 保守修复“酒店周边晚餐与休息”等泛化活动的分类。
                normalized_draft = _normalize_generic_free_time_activities(draft)

                # 9. 使用短别名执行业务校验。
                _validate_proposal_draft(
                    draft=normalized_draft,
                    locked_context=locked_context,
                    evidence_context=evidence_context,
                    requirement_catalog=requirement_catalog,
                    tool_activity_context=tool_activity_context,
                )

                # 10. 将 G01/S01/H01/P01 映射回真实 chunk_id。
                resolved_draft = _resolve_draft_evidence_aliases(
                    draft=normalized_draft,
                    alias_to_chunk_id=evidence_context["alias_to_chunk_id"],
                )

                # 11. 程序注入锁定事实并组装最终 Proposal。
                proposal = _assemble_trip_proposal(
                    draft=resolved_draft,
                    locked_context=locked_context,
                    evidence_context=evidence_context,
                    model_id=self.model_id,
                    proposal_version=proposal_version,
                    tool_activity_context=tool_activity_context,
                )

                used_refs = _collect_draft_evidence_refs(resolved_draft)

                result = ProposalGenerationResult(
                    status="generated",
                    model_id=self.model_id,
                    attempt_count=attempt,
                    max_attempts=self.max_attempts,
                    evidence_input_count=len(evidence_context["all_items"]),
                    evidence_prompt_count=len(evidence_context["prompt_items"]),
                    used_evidence_count=len(used_refs),
                    used_evidence_refs=sorted(used_refs),
                    locked_fact_hash=proposal.locked_fact_hash,
                    proposal_id=proposal.proposal_id,
                    daily_plan_count=len(proposal.daily_plan),
                    validation_errors=validation_errors,
                    issues=[],
                )

                return {
                    "proposal": proposal.to_state_dict(),
                    "proposal_result": result.to_state_dict(),
                }

            except (ValidationError, ValueError, TypeError) as exc:
                # 10. Schema 或业务校验失败时记录简短原因，并尝试一次修复。
                validation_errors.append(
                    _short_error_message(exc)
                )

            except Exception as exc:
                # 11. API、网络或模型服务错误也允许在限定次数内重试。
                validation_errors.append(
                    f"模型调用失败：{_short_error_message(exc)}"
                )

        # 12. 达到上限仍失败时，不生成未经验证的 Proposal。
        raise ProposalGenerationError(
            "Proposal 在限定尝试次数内未通过结构化校验",
            validation_errors=validation_errors,
            attempt_count=self.max_attempts,
        )


@lru_cache(maxsize=1)
def build_default_proposal_generator() -> LLMProposalGenerator:
    """
    根据项目 Settings 创建并缓存默认 ProposalGenerator。

    缓存的对象包含 OpenAI-compatible Client，
    整个应用进程内可以复用连接配置，不必每次请求重新创建。
    """

    api_key = settings.deepseek_api_key or ""

    if not api_key.strip():
        raise ValueError(
            "DEEPSEEK_API_KEY 不能为空，ProposalNode 需要 LLM 生成行程内容"
        )

    client = DeepSeekJSONClient(
        api_key=api_key,
        base_url=settings.llm_base_url,
        model_id=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        thinking_enabled=settings.llm_thinking_enabled,
    )

    return LLMProposalGenerator(
        client=client,
        max_tokens=settings.proposal_max_tokens,
        max_attempts=settings.proposal_max_attempts,
        max_evidence_items=settings.proposal_max_evidence_items,
        max_evidence_chars=settings.proposal_max_evidence_chars,
    )


def build_locked_proposal_context(
    *,
    trip_request: Mapping[str, Any],
    planning_context: Mapping[str, Any],
    selection_result: Mapping[str, Any],
    weather_result: Mapping[str, Any],
) -> dict[str, Any]:
    """
    构建 Proposal 的锁定事实。

    这些事实全部来自确定性上游节点，
    不允许 LLM 在输出中重新生成或修改。
    """

    if selection_result.get("status") != "feasible":
        raise ValueError(
            "selection_result.status 必须为 feasible，请先完成 BudgetOptimizeNode"
        )

    request = planning_context.get("request")
    request = request if isinstance(request, Mapping) else {}

    # 1. 构建标准旅行日期和人数信息。
    trip_overview = _build_trip_overview(
        request=request,
        trip_request=trip_request,
    )

    # 2. 从 selection_result 读取最终选中的航班和酒店。
    selected_flights = _build_selected_flights(
        selection_result=selection_result,
    )

    selected_hotel = _build_selected_hotel(
        selection_result=selection_result,
    )

    # 3. 验证 selected_* 与 selected_combination 的 IDs 完全一致。
    _validate_selection_ids(
        selection_result=selection_result,
        selected_flights=selected_flights,
        selected_hotel=selected_hotel,
    )

    # 4. 按旅行日期构建逐日天气，缺失日期也会明确标记 unavailable。
    weather_overview = _build_weather_overview(
        weather_result=weather_result,
        trip_overview=trip_overview,
    )

    # 5. 预算摘要只复制 BudgetOptimize 的已知报价和剩余预算安排。
    budget_summary = _build_budget_summary(
        selection_result=selection_result,
    )

    locked_payload = {
        "trip_overview": _model_to_dict(trip_overview),
        "selected_flights": _model_to_dict(selected_flights),
        "selected_hotel": _model_to_dict(selected_hotel),
        "weather_overview": _model_to_dict(weather_overview),
        "budget_summary": _model_to_dict(budget_summary),
    }

    return {
        "trip_overview": trip_overview,
        "selected_flights": selected_flights,
        "selected_hotel": selected_hotel,
        "weather_overview": weather_overview,
        "budget_summary": budget_summary,
        "expected_dates": [
            item.date.isoformat()
            for item in weather_overview.daily
        ],
        "locked_fact_hash": _sha256_json(locked_payload),
        "prompt_locked_facts": locked_payload,
    }


def _build_trip_overview(
    *,
    request: Mapping[str, Any],
    trip_request: Mapping[str, Any],
) -> TripOverview:
    """从 PlanningContext.request 构建标准化旅行基础事实。"""

    origin = _required_text(
        request.get("origin") or trip_request.get("origin"),
        "origin",
    )

    destination = _required_text(
        request.get("destination") or trip_request.get("destination"),
        "destination",
    )

    start_date = _parse_required_date(
        request.get("start_date") or trip_request.get("start_date"),
        "start_date",
    )

    days = _safe_int(
        request.get("days") or trip_request.get("days"),
        default=0,
    )

    if days <= 0:
        raise ValueError("days 必须大于 0")

    end_date_value = request.get("end_date") or trip_request.get("end_date")
    end_date = (
        _parse_required_date(end_date_value, "end_date")
        if end_date_value
        else start_date + timedelta(days=days - 1)
    )

    expected_end_date = start_date + timedelta(days=days - 1)

    if end_date != expected_end_date:
        raise ValueError(
            "end_date 与 start_date + days - 1 不一致，不能生成 Proposal"
        )

    nights = _safe_int(
        request.get("nights"),
        default=max(days - 1, 0),
    )

    people_count = _safe_int(
        request.get("people_count") or trip_request.get("people_count"),
        default=1,
    )

    room_count = _safe_int(
        request.get("room_count") or trip_request.get("room_count"),
        default=1,
    )

    return TripOverview(
        origin=origin,
        destination=destination,
        start_date=start_date,
        end_date=end_date,
        days=days,
        nights=max(nights, 0),
        people_count=max(people_count, 1),
        room_count=max(room_count, 1),
    )


def _build_selected_flights(
    *,
    selection_result: Mapping[str, Any],
) -> SelectedFlights:
    """从 selection_result 构建去程和返程锁定航班事实。"""

    outbound = selection_result.get("selected_outbound_flight")
    return_flight = selection_result.get("selected_return_flight")

    if not isinstance(outbound, Mapping):
        raise ValueError("selection_result 缺少 selected_outbound_flight")

    if not isinstance(return_flight, Mapping):
        raise ValueError("selection_result 缺少 selected_return_flight")

    return SelectedFlights(
        outbound=_coerce_selected_flight(outbound),
        return_flight=_coerce_selected_flight(return_flight),
    )


def _coerce_selected_flight(
    flight: Mapping[str, Any],
) -> SelectedFlightFact:
    """
    将 CandidateRank 航班转换为 Proposal 所需的锁定字段。

    这里只做字段复制和安全数值转换，不改变候选事实。
    """

    return SelectedFlightFact(
        flight_id=_required_text(flight.get("flight_id"), "flight_id"),
        flight_no=_optional_text(flight.get("flight_no")),
        airline=_optional_text(flight.get("airline")),
        departure_city=_optional_text(flight.get("departure_city")),
        arrival_city=_optional_text(flight.get("arrival_city")),
        departure_airport=_optional_text(flight.get("departure_airport")),
        arrival_airport=_optional_text(flight.get("arrival_airport")),
        depart_date=_optional_text(flight.get("depart_date")),
        depart_time=_optional_text(flight.get("depart_time")),
        arrive_date=_optional_text(flight.get("arrive_date")),
        arrive_time=_optional_text(flight.get("arrive_time")),
        duration_minutes=max(
            _safe_int(flight.get("duration_minutes"), default=0),
            0,
        ),
        price=_safe_non_negative_float(flight.get("price"), "flight.price"),
        total_price=_safe_non_negative_float(
            flight.get("total_price", flight.get("price")),
            "flight.total_price",
        ),
        cabin=_optional_text(flight.get("cabin")),
        baggage=_optional_text(flight.get("baggage")),
        is_direct=bool(flight.get("is_direct", True)),
        refundable=bool(flight.get("refundable", False)),
        changeable=bool(flight.get("changeable", False)),
        rank=_safe_optional_positive_int(flight.get("rank")),
        total_score=_safe_optional_unit_float(flight.get("total_score")),
        ranking_reasons=_string_list(flight.get("ranking_reasons")),
        preference_gaps=_mapping_list(flight.get("preference_gaps")),
    )


def _build_selected_hotel(
    *,
    selection_result: Mapping[str, Any],
) -> SelectedHotelFact:
    """从 selection_result 构建锁定酒店事实。"""

    hotel = selection_result.get("selected_hotel")

    if not isinstance(hotel, Mapping):
        raise ValueError("selection_result 缺少 selected_hotel")

    return SelectedHotelFact(
        hotel_id=_required_text(hotel.get("hotel_id"), "hotel_id"),
        name=_optional_text(hotel.get("name")),
        city=_optional_text(hotel.get("city")),
        district=_optional_text(hotel.get("district")),
        address=_optional_text(hotel.get("address")),
        price_per_night=_safe_non_negative_float(
            hotel.get("price_per_night"),
            "hotel.price_per_night",
        ),
        estimated_total_price=_safe_non_negative_float(
            hotel.get("estimated_total_price"),
            "hotel.estimated_total_price",
        ),
        planned_nights=max(
            _safe_int(hotel.get("planned_nights"), default=0),
            0,
        ),
        planned_room_count=max(
            _safe_int(hotel.get("planned_room_count"), default=1),
            1,
        ),
        rating=_safe_non_negative_float(hotel.get("rating"), "hotel.rating"),
        near_subway=bool(hotel.get("near_subway", False)),
        distance_to_subway_meters=max(
            _safe_int(
                hotel.get("distance_to_subway_meters"),
                default=999999,
            ),
            0,
        ),
        quiet_score=_safe_unit_float(
            hotel.get("quiet_score", 0.5),
            "hotel.quiet_score",
        ),
        cleanliness_score=_safe_unit_float(
            hotel.get("cleanliness_score", 0.5),
            "hotel.cleanliness_score",
        ),
        tags=_string_list(hotel.get("tags")),
        amenities=_string_list(hotel.get("amenities")),
        cancel_policy=_optional_text(hotel.get("cancel_policy")),
        rank=_safe_optional_positive_int(hotel.get("rank")),
        total_score=_safe_optional_unit_float(hotel.get("total_score")),
        ranking_reasons=_string_list(hotel.get("ranking_reasons")),
        preference_gaps=_mapping_list(hotel.get("preference_gaps")),
    )


def _validate_selection_ids(
    *,
    selection_result: Mapping[str, Any],
    selected_flights: SelectedFlights,
    selected_hotel: SelectedHotelFact,
) -> None:
    """确认完整候选与 selected_combination 的 IDs 一致。"""

    combination = selection_result.get("selected_combination")

    if not isinstance(combination, Mapping):
        raise ValueError("selection_result 缺少 selected_combination")

    expected = {
        "outbound_flight_id": selected_flights.outbound.flight_id,
        "return_flight_id": selected_flights.return_flight.flight_id,
        "hotel_id": selected_hotel.hotel_id,
    }

    for field_name, actual_id in expected.items():
        combination_id = str(combination.get(field_name) or "").strip()

        if combination_id != actual_id:
            raise ValueError(
                f"selection_result.{field_name} 与完整候选 ID 不一致"
            )


def _build_weather_overview(
    *,
    weather_result: Mapping[str, Any],
    trip_overview: TripOverview,
) -> WeatherOverview:
    """
    将 WeatherNode 输出对齐到每一个旅行日期。

    即使某一天没有天气记录，也会生成 data_status=unavailable，
    避免 LLM 自行补充天气。
    """

    raw_daily = weather_result.get("daily")
    raw_daily = raw_daily if isinstance(raw_daily, list) else []

    daily_by_date: dict[str, Mapping[str, Any]] = {}

    for item in raw_daily:
        if not isinstance(item, Mapping):
            continue

        item_date = str(item.get("date") or "").strip()

        if item_date:
            daily_by_date[item_date] = item

    # 1. 顶层 risks 也按日期合并到逐日天气。
    risk_types_by_date: dict[str, list[str]] = defaultdict(list)
    risk_messages_by_date: dict[str, list[str]] = defaultdict(list)
    undated_risk_types: list[str] = []

    raw_risks = weather_result.get("risks")
    raw_risks = raw_risks if isinstance(raw_risks, list) else []

    for risk in raw_risks:
        if not isinstance(risk, Mapping):
            continue

        risk_type = str(risk.get("risk_type") or "").strip()
        risk_date = str(risk.get("date") or "").strip()
        message = str(risk.get("message") or "").strip()

        if risk_type:
            if risk_date:
                risk_types_by_date[risk_date].append(risk_type)
            else:
                undated_risk_types.append(risk_type)

        if risk_date and message:
            risk_messages_by_date[risk_date].append(message)

    daily_facts: list[DailyWeatherFact] = []

    for offset in range(trip_overview.days):
        current_date = trip_overview.start_date + timedelta(days=offset)
        date_key = current_date.isoformat()
        item = daily_by_date.get(date_key)

        if item is None:
            daily_facts.append(
                DailyWeatherFact(
                    date=current_date,
                    data_status="unavailable",
                    condition=None,
                    risk_types=[],
                    warnings=["该日期没有可用的 Mock 天气记录。"],
                )
            )
            continue

        risk_types = _dedupe_strings(
            [
                *_string_list(item.get("risk_tags")),
                *risk_types_by_date.get(date_key, []),
            ]
        )

        warnings = _dedupe_strings(
            [
                *_string_list(item.get("warnings")),
                *risk_messages_by_date.get(date_key, []),
            ]
        )

        daily_facts.append(
            DailyWeatherFact(
                date=current_date,
                data_status="available",
                condition=_optional_text(item.get("condition")),
                temperature_low=_safe_optional_float(
                    item.get("temperature_low")
                ),
                temperature_high=_safe_optional_float(
                    item.get("temperature_high")
                ),
                humidity=_safe_optional_float(item.get("humidity")),
                wind=_optional_text(item.get("wind")),
                precipitation_probability=_safe_optional_float(
                    item.get("precipitation_probability")
                ),
                risk_types=risk_types,
                warnings=warnings,
            )
        )

    all_risk_types = _dedupe_strings(
        [
            *undated_risk_types,
            *[
                risk_type
                for item in daily_facts
                for risk_type in item.risk_types
            ],
        ]
    )

    all_warnings = _dedupe_strings(
        [
            *_string_list(weather_result.get("warnings")),
            *[
                warning
                for item in daily_facts
                for warning in item.warnings
            ],
        ]
    )

    return WeatherOverview(
        status=str(weather_result.get("status") or "unknown"),
        summary=str(weather_result.get("summary") or "暂无天气摘要。"),
        daily=daily_facts,
        risk_types=all_risk_types,
        warnings=all_warnings,
        source=_optional_text(weather_result.get("source")),
    )


def _build_budget_summary(
    *,
    selection_result: Mapping[str, Any],
) -> ProposalBudgetSummary:
    """复制 BudgetOptimize 的预算结果并固定费用语义。"""

    budget_mode = str(selection_result.get("budget_mode") or "")

    if budget_mode not in {"budget_limited", "no_budget_limit"}:
        raise ValueError("selection_result.budget_mode 无效")

    known_costs = selection_result.get("selected_costs")
    known_costs = dict(known_costs) if isinstance(known_costs, Mapping) else {}

    if not known_costs:
        combination = selection_result.get("selected_combination")
        if isinstance(combination, Mapping):
            combination_costs = combination.get("selected_costs")
            if isinstance(combination_costs, Mapping):
                known_costs = dict(combination_costs)

    if not known_costs:
        raise ValueError("selection_result 缺少 selected_costs")

    allocation = selection_result.get("remaining_budget_allocation")
    allocation = dict(allocation) if isinstance(allocation, Mapping) else None

    return ProposalBudgetSummary(
        budget_mode=budget_mode,  # type: ignore[arg-type]
        total_budget=_safe_optional_float(selection_result.get("total_budget")),
        known_costs=known_costs,
        remaining_budget=_safe_optional_float(
            selection_result.get("remaining_budget")
        ),
        remaining_budget_allocation=allocation,
        note=(
            "预算摘要只包含 Mock 航班和酒店的已知模拟报价；"
            "剩余金额是餐饮、活动、市内交通和机动缓冲的安排建议，"
            "不是对实际消费的预测。"
        ),
    )


def _build_evidence_context(
    *,
    evidence_pool: Sequence[Mapping[str, Any]],
    selected_hotel_id: str,
    max_items: int,
    max_chars: int,
) -> dict[str, Any]:
    """
    将 Evidence Pool 整理为 Prompt 证据目录，并建立短别名映射。

    为什么需要短别名？

        真实 chunk_id 往往很长，例如：

            guide_chengdu_rainy_day_v1::05::001

        LLM 在长 JSON 中复制这类 ID 时，容易：
            - 抄错 section 序号；
            - 根据文件名自行猜测一个不存在的 ID；
            - 引用 Evidence Pool 中存在但本轮没有放入 Prompt 的 Chunk。

        因此 Prompt 中只暴露 G01/S01/H01/P01 等短别名。
        模型只输出短别名；校验通过后，程序再把短别名确定性映射回
        真实 chunk_id。最终 Proposal、Verifier 和数据库中仍保存真实 ID。

    选择策略：
        1. 过滤结构无效的证据；
        2. hotel_reviews 只保留当前选中酒店；
        3. 相同 chunk_id 只保留质量最高的一条；
        4. 每个非空类别先保留一条，保护证据多样性；
        5. 剩余名额按精排质量全局填充；
        6. 给最终进入 Prompt 的证据分配稳定短别名。
    """

    normalized: list[dict[str, Any]] = []
    for raw_item in evidence_pool:
        if not isinstance(raw_item, Mapping):
            continue
        item = dict(raw_item)
        chunk_id = str(item.get("chunk_id") or "").strip()
        category = str(item.get("category") or "").strip()
        content = str(item.get("content") or "").strip()
        source = str(item.get("source") or "").strip()
        metadata = item.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        if not chunk_id or not category or not content or not source:
            continue
        if category == "hotel_reviews":
            hotel_id = str(metadata.get("hotel_id") or "").strip()
            if hotel_id != selected_hotel_id:
                continue
        item["chunk_id"] = chunk_id
        item["category"] = category
        item["content"] = content
        item["source"] = source
        item["metadata"] = metadata
        normalized.append(item)

    best_by_chunk_id: dict[str, dict[str, Any]] = {}
    for item in normalized:
        old = best_by_chunk_id.get(item["chunk_id"])
        if old is None or _evidence_quality(item) > _evidence_quality(old):
            best_by_chunk_id[item["chunk_id"]] = item

    all_items = list(best_by_chunk_id.values())
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in all_items:
        grouped[item["category"]].append(item)
    for category_items in grouped.values():
        category_items.sort(key=_evidence_sort_key)

    selected: list[dict[str, Any]] = []
    selected_chunk_ids: set[str] = set()
    for category in sorted(grouped, key=lambda value: EVIDENCE_CATEGORY_ORDER.get(value, 99)):
        if len(selected) >= max_items:
            break
        first = grouped[category][0]
        selected.append(first)
        selected_chunk_ids.add(first["chunk_id"])

    remaining = sorted(
        [item for item in all_items if item["chunk_id"] not in selected_chunk_ids],
        key=_evidence_sort_key,
    )
    for item in remaining:
        if len(selected) >= max_items:
            break
        selected.append(item)
        selected_chunk_ids.add(item["chunk_id"])

    selected.sort(key=lambda item: (EVIDENCE_CATEGORY_ORDER.get(item["category"], 99), _evidence_sort_key(item)))

    category_counters: dict[str, int] = defaultdict(int)
    alias_to_chunk_id: dict[str, str] = {}
    chunk_id_to_alias: dict[str, str] = {}
    prompt_items: list[dict[str, Any]] = []
    prompt_by_ref: dict[str, dict[str, Any]] = {}
    refs_by_category: dict[str, set[str]] = defaultdict(set)

    for item in selected:
        category = item["category"]
        prefix = EVIDENCE_ALIAS_PREFIX.get(category, "E")
        category_counters[category] += 1
        alias = f"{prefix}{category_counters[category]:02d}"
        real_chunk_id = item["chunk_id"]
        alias_to_chunk_id[alias] = real_chunk_id
        chunk_id_to_alias[real_chunk_id] = alias
        prompt_by_ref[alias] = item
        refs_by_category[category].add(alias)
        prompt_items.append({
            "ref_id": alias,
            "category": category,
            "source": item["source"],
            "title": item["metadata"].get("title"),
            "section": item["metadata"].get("section"),
            "doc_type": item["metadata"].get("doc_type"),
            "hotel_id": item["metadata"].get("hotel_id"),
            "risk_type": item["metadata"].get("risk_type"),
            "content": _truncate_text(item["content"], max_chars),
        })

    by_ref = {item["chunk_id"]: item for item in all_items}
    reference_contract = {category: sorted(refs) for category, refs in refs_by_category.items()}
    return {
        "all_items": all_items,
        "prompt_items": prompt_items,
        "prompt_by_ref": prompt_by_ref,
        "refs_by_category": dict(refs_by_category),
        "allowed_refs": set(alias_to_chunk_id),
        "reference_contract": reference_contract,
        "alias_to_chunk_id": alias_to_chunk_id,
        "chunk_id_to_alias": chunk_id_to_alias,
        "by_ref": by_ref,
    }

def build_requirement_catalog(
    *,
    planning_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    构建 Proposal 和 Verifier 共用的特殊用户要求目录。

    公开这个入口的原因：
        ProposalGenerator 和 VerifierNode 必须使用完全相同的 requirement_key。
        如果两边分别复制生成逻辑，后续字段调整时容易产生不一致。
    """

    return _build_requirement_catalog(
        planning_context=planning_context,
    )


def _build_requirement_catalog(
    *,
    planning_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    为指定实体和未映射需求生成稳定 requirement_key。

    Proposal Draft 必须为每个 key 返回处理状态，
    从而避免模型静默忽略用户要求。
    """

    catalog: list[dict[str, Any]] = []

    named_constraints = planning_context.get("named_constraints")
    named_constraints = (
        named_constraints
        if isinstance(named_constraints, list)
        else []
    )

    for index, item in enumerate(named_constraints):
        if not isinstance(item, Mapping):
            continue

        entity_type = str(item.get("entity_type") or "unknown")
        entity_name = str(item.get("entity_name") or "").strip()

        if not entity_name:
            continue

        catalog.append(
            {
                "requirement_key": (
                    f"named:{entity_type}:"
                    f"{_short_hash(entity_name)}:{index}"
                ),
                "requirement_type": "named_constraint",
                "entity_type": entity_type,
                "entity_name": entity_name,
                "constraint_mode": item.get("constraint_mode", "preferred"),
                "text": item.get("evidence") or entity_name,
            }
        )

    unmapped_requirements = planning_context.get("unmapped_requirements")
    unmapped_requirements = (
        unmapped_requirements
        if isinstance(unmapped_requirements, list)
        else []
    )

    for index, item in enumerate(unmapped_requirements):
        if not isinstance(item, Mapping):
            continue

        text = str(item.get("text") or "").strip()

        if not text:
            continue

        catalog.append(
            {
                "requirement_key": (
                    f"unmapped:{_short_hash(text)}:{index}"
                ),
                "requirement_type": "unmapped_requirement",
                "category_guess": item.get("category_guess", "unknown"),
                "impact": item.get("impact", "unknown"),
                "text": text,
            }
        )

    return catalog


def _build_system_prompt() -> str:
    """构建 Proposal LLM 的固定系统规则。"""
    return (
        "你是 TravelOps-Copilot 的旅行方案生成器。"
        "你必须只输出一个合法 JSON object，并严格符合提供的 JSON Schema。\n\n"
        "你的职责：\n"
        "1. 根据逐日天气、用户偏好和 RAG 证据安排每日活动。\n"
        "2. 在雨天、大风或高温日期主动调整室内外活动和交通缓冲。\n"
        "3. 只能使用 evidence_catalog 中提供的短 ref_id，例如 G01、S01、H01、P01。\n"
        "4. 说明选中酒店的非实时补充信息，但不得用 RAG 改变酒店选择。\n"
        "5. 逐项处理 requirement_catalog 中的用户要求。\n\n"
        "6. 可以使用 tool_activity_catalog 中的结构化活动；使用时必须复制 tool_activity_id，"
        "并保持名称和可用日期一致。\n\n"
        "活动分类规则：\n"
        "1. 酒店休息、酒店周边自行用餐、自由选择晚餐、整理行李、等待返程、"
        "未指定具体地点的自由活动，必须使用 activity_type=free_time。\n"
        "2. free_time、transfer、check_in 可以不填写 evidence_refs。\n"
        "3. 只有明确依据攻略或活动工具中的具体景点、博物馆、公园、街区、菜品、餐饮区域或路线时，"
        "才使用 sightseeing、culture、food、nature、city_walk、leisure、shopping 或 other，"
        "并至少引用一个 G 开头的 guide ref。\n"
        "4. 如果没有合适攻略证据，不得把泛化的自行晚餐或休息包装成知识型活动。\n\n"
        "Evidence 引用规则：\n"
        "1. evidence_refs 只能逐字复制 allowed_evidence_refs 中的短 ID。\n"
        "2. 不得自行生成 G99、S99 或任何列表之外的 ID。\n"
        "3. 不得根据 source 文件名猜测真实 chunk_id。\n"
        "4. 没有合适证据时，删除该知识型活动，或把泛化安排改为 free_time。\n\n"
        "严格禁止：\n"
        "1. 不得重新选择或修改航班、酒店、价格、日期、天气和预算。\n"
        "2. 不得编造具体餐厅、景点、营业时间、票价或实时开放状态。\n"
        "3. 不得声称已经预订、付款、出票、发邮件或发送通知。\n"
        "4. 如果某个指定要求没有证据支持，必须标记 not_supported，不能猜测。\n"
        "5. 有天气风险的日期必须填写 weather_adjustment。\n"
        "6. 如果 verifier_feedback 非空，必须逐项修复其中问题，但仍不得修改 locked_facts。"
    )


def _build_tool_activity_context(
    planning_context: Mapping[str, Any],
) -> dict[str, Any]:
    """
    将 ToolExecute 写入的活动候选压缩为 Proposal 可见白名单。

    Prompt 不接收原始 MCP 响应，只接收经过 Provider 规范化的字段；Verifier
    仍能通过稳定 activity_id 将最终安排追溯到本轮工具结果。
    """

    raw_items = planning_context.get("activity_candidates")
    raw_items = raw_items if isinstance(raw_items, Sequence) else []
    prompt_items: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            continue
        activity_id = str(raw.get("activity_id") or "").strip()
        name = str(raw.get("name") or "").strip()
        city = str(raw.get("city") or "").strip()
        if not activity_id or not name or not city:
            continue
        item = {
            "activity_id": activity_id,
            "name": name,
            "city": city,
            "district": raw.get("district"),
            "category": raw.get("category"),
            "available_dates": list(raw.get("available_dates", [])),
            "opening_hours": raw.get("opening_hours"),
            "estimated_duration_minutes": raw.get("estimated_duration_minutes"),
            "estimated_price": raw.get("estimated_price"),
            "indoor_outdoor": raw.get("indoor_outdoor"),
            "reservation_required": raw.get("reservation_required"),
            "tags": list(raw.get("tags", [])),
        }
        prompt_items.append(item)
        by_id[activity_id] = item
    return {"prompt_items": prompt_items, "by_id": by_id}


def _build_user_prompt(
    *,
    locked_context: Mapping[str, Any],
    planning_context: Mapping[str, Any],
    evidence_catalog: Sequence[Mapping[str, Any]],
    evidence_reference_contract: Mapping[str, Sequence[str]],
    requirement_catalog: Sequence[Mapping[str, Any]],
    tool_activity_catalog: Sequence[Mapping[str, Any]],
    verifier_feedback: Mapping[str, Any] | None = None,
) -> str:
    """构建带 Evidence Alias 白名单的结构化 User Prompt。"""
    allowed_refs_by_category = {category: list(refs) for category, refs in evidence_reference_contract.items()}
    allowed_evidence_refs = sorted({ref for refs in allowed_refs_by_category.values() for ref in refs})
    prompt_payload = {
        "task": "generate_trip_proposal_draft",
        "prompt_version": PROPOSAL_PROMPT_VERSION,
        "important_boundary": "只生成 ProposalDraft 中的可生成内容；locked_facts 由程序注入最终 Proposal，不要在输出中重写。",
        "activity_type_contract": {
            "generic_free_time_examples": ["酒店休息", "酒店周边自行用餐", "自由选择晚餐", "整理行李", "等待返程", "未指定具体地点的自由活动"],
            "generic_activity_type": "free_time",
            "knowledge_activity_types": sorted(KNOWLEDGE_ACTIVITY_TYPES),
            "knowledge_activity_rule": "知识型活动必须引用至少一个 guides 类短别名；没有证据时不得编造具体地点。",
        },
        "evidence_reference_contract": {
            "allowed_evidence_refs": allowed_evidence_refs,
            "allowed_refs_by_category": allowed_refs_by_category,
            "copy_rule": "evidence_refs 只能逐字复制列表中的短 ID；不得自行构造任何 ID。",
        },
        "expected_dates": locked_context["expected_dates"],
        "locked_facts": locked_context["prompt_locked_facts"],
        "user_context": {
            "preference_weights": planning_context.get("preference_weights", {}),
            "diet_preferences": planning_context.get("diet_preferences", {}),
            "context_summary": planning_context.get("context_summary", {}),
            "selected_flight_preference_gaps": [*locked_context["selected_flights"].outbound.preference_gaps, *locked_context["selected_flights"].return_flight.preference_gaps],
            "selected_hotel_preference_gaps": locked_context["selected_hotel"].preference_gaps,
        },
        "requirement_catalog": list(requirement_catalog),
        "tool_activity_catalog": list(tool_activity_catalog),
        "verifier_feedback": dict(verifier_feedback or {}),
        "evidence_catalog": list(evidence_catalog),
        "output_json_schema": ProposalDraft.model_json_schema(),
    }
    return "请根据以下结构化上下文生成 ProposalDraft JSON：\n" + json.dumps(prompt_payload, ensure_ascii=False, indent=2, default=str)


def _build_retry_guidance(evidence_context: Mapping[str, Any]) -> str:
    """第二次尝试时列出唯一允许使用的短 Evidence Alias。"""
    contract = evidence_context.get("reference_contract")
    contract = contract if isinstance(contract, Mapping) else {}
    lines = ["本轮 Evidence 引用白名单如下："]
    for category in sorted(contract, key=lambda value: EVIDENCE_CATEGORY_ORDER.get(value, 99)):
        refs = contract.get(category)
        refs = list(refs) if isinstance(refs, Sequence) else []
        lines.append(f"- {category}: {', '.join(str(ref) for ref in refs) or '无'}")
    lines.extend([
        "只能逐字复制以上短 ID，不得生成列表以外的引用。",
        "如果某个知识型活动没有合适 Guide：删除该活动，或将泛化的自行用餐、酒店休息、整理行李改为 free_time。",
    ])
    return "\n".join(lines)


def _normalize_generic_free_time_activities(draft: ProposalDraft) -> ProposalDraft:
    """把没有证据的泛化自行用餐/休息活动保守归一化为 free_time。"""
    payload = draft.model_dump(mode="json")
    for day in payload.get("daily_plan", []):
        if not isinstance(day, dict):
            continue
        for activity in day.get("activities", []):
            if not isinstance(activity, dict):
                continue
            refs = activity.get("evidence_refs")
            refs = refs if isinstance(refs, list) else []
            if refs:
                continue
            if str(activity.get("activity_type") or "") not in {"food", "leisure", "other"}:
                continue
            text = str(activity.get("title") or "") + " " + str(activity.get("description") or "")
            if not any(phrase in text for phrase in GENERIC_FREE_TIME_PHRASES):
                continue
            activity["activity_type"] = "free_time"
            if activity.get("indoor_outdoor") == "outdoor":
                activity["indoor_outdoor"] = "flexible"
    return ProposalDraft(**payload)


def _resolve_draft_evidence_aliases(*, draft: ProposalDraft, alias_to_chunk_id: Mapping[str, str]) -> ProposalDraft:
    """将 ProposalDraft 中的短 Evidence Alias 确定性恢复为真实 chunk_id。"""
    payload = draft.model_dump(mode="json")
    def resolve(raw: Any) -> list[str]:
        refs = raw if isinstance(raw, list) else []
        return [alias_to_chunk_id[str(ref)] for ref in refs]
    for day in payload.get("daily_plan", []):
        if isinstance(day, dict):
            for activity in day.get("activities", []):
                if isinstance(activity, dict):
                    activity["evidence_refs"] = resolve(activity.get("evidence_refs"))
    hotel_context = payload.get("hotel_context")
    if isinstance(hotel_context, dict):
        hotel_context["evidence_refs"] = resolve(hotel_context.get("evidence_refs"))
    for field_name in ("packing_tips", "safety_notes", "requirement_handling"):
        for item in payload.get(field_name, []):
            if isinstance(item, dict):
                item["evidence_refs"] = resolve(item.get("evidence_refs"))
    return ProposalDraft(**payload)

def _validate_proposal_draft(
    *,
    draft: ProposalDraft,
    locked_context: Mapping[str, Any],
    evidence_context: Mapping[str, Any],
    requirement_catalog: Sequence[Mapping[str, Any]],
    tool_activity_context: Mapping[str, Any],
) -> None:
    """
    对 ProposalDraft 执行跨字段业务校验。

    这里是 ProposalNode 的可靠性核心：
        Pydantic 只能检查字段类型；
        本方法继续检查日期、天气、证据引用和用户要求是否一致。
    """

    expected_dates = list(locked_context["expected_dates"])
    actual_dates = [
        item.date.isoformat()
        for item in draft.daily_plan
    ]

    # 1. 每个旅行日期必须恰好出现一次，并保持顺序。
    if actual_dates != expected_dates:
        raise ValueError(
            "daily_plan 日期必须与 expected_dates 完全一致且顺序相同；"
            f"expected={expected_dates}, actual={actual_dates}"
        )

    allowed_refs = set(evidence_context["allowed_refs"])
    prompt_by_ref = evidence_context["prompt_by_ref"]
    refs_by_category = evidence_context["refs_by_category"]

    # 2. 在全局 unknown-ref 检查前先判断酒店知识是否存在。
    #    这样模型在没有酒店证据时编造 hotel_context，
    #    能得到更明确的业务错误，而不是只看到“未知引用”。
    hotel_review_refs = set(
        refs_by_category.get("hotel_reviews", set())
    )

    if not hotel_review_refs and draft.hotel_context is not None:
        raise ValueError(
            "当前没有选中酒店的 hotel_reviews 证据，不能生成 hotel_context"
        )

    used_refs = _collect_draft_evidence_refs(draft)
    unknown_refs = sorted(used_refs - allowed_refs)

    # 2. 所有证据引用必须来自 EvidenceGrade 已验证的 Evidence Pool。
    if unknown_refs:
        raise ValueError(
            "ProposalDraft 使用了未知 evidence_refs："
            + ", ".join(unknown_refs)
        )

    guide_refs = set(refs_by_category.get("guides", set()))
    activity_by_id = tool_activity_context.get("by_id", {})

    # 3. 知识型活动必须至少引用一个 guide 证据。
    for day in draft.daily_plan:
        for activity in day.activities:
            if activity.activity_type not in KNOWLEDGE_ACTIVITY_TYPES:
                continue

            tool_activity = (
                activity_by_id.get(activity.tool_activity_id)
                if activity.tool_activity_id
                else None
            )
            if activity.tool_activity_id and not isinstance(tool_activity, Mapping):
                raise ValueError(
                    f"{day.date.isoformat()} 使用了未知 tool_activity_id={activity.tool_activity_id}"
                )
            if isinstance(tool_activity, Mapping):
                if activity.title != str(tool_activity.get("name") or ""):
                    raise ValueError("工具活动标题必须与 activity_id 对应的候选名称完全一致")
                if day.date.isoformat() not in set(tool_activity.get("available_dates", [])):
                    raise ValueError("工具活动被安排在 available_dates 之外")

            if not set(activity.evidence_refs) & guide_refs and not tool_activity:
                raise ValueError(
                    f"{day.date.isoformat()} 的知识型活动“{activity.title}”"
                    "必须引用至少一个 guides 证据或有效 tool_activity_id"
                )

    weather_by_date = {
        item.date.isoformat(): item
        for item in locked_context["weather_overview"].daily
    }

    # 4. 天气风险日必须有调整；雨天不能把所有活动都安排成纯户外。
    for day in draft.daily_plan:
        weather = weather_by_date[day.date.isoformat()]
        risk_types = set(weather.risk_types)

        if risk_types & WEATHER_RISKS_REQUIRING_ADJUSTMENT:
            if not day.weather_adjustment or not day.weather_adjustment.strip():
                raise ValueError(
                    f"{day.date.isoformat()} 存在天气风险，"
                    "daily_plan.weather_adjustment 不能为空"
                )

        if risk_types & RAIN_RISK_TYPES:
            non_transfer_activities = [
                item
                for item in day.activities
                if item.indoor_outdoor != "transfer"
            ]

            if (
                non_transfer_activities
                and all(
                    item.indoor_outdoor == "outdoor"
                    for item in non_transfer_activities
                )
            ):
                raise ValueError(
                    f"{day.date.isoformat()} 是雨天，不能把所有活动都安排为 outdoor"
                )

    # 5. 有选中酒店 RAG 证据时必须生成酒店补充说明；
    #    没有酒店证据时禁止模型编造 hotel_context。
    if hotel_review_refs:
        if draft.hotel_context is None:
            raise ValueError(
                "存在选中酒店的 hotel_reviews 证据，hotel_context 不能为空"
            )

        hotel_context_refs = set(draft.hotel_context.evidence_refs)

        if not hotel_context_refs:
            raise ValueError("hotel_context 必须引用 hotel_reviews 证据")

        if not hotel_context_refs <= hotel_review_refs:
            raise ValueError(
                "hotel_context.evidence_refs 只能引用选中酒店的 hotel_reviews 证据"
            )
    elif draft.hotel_context is not None:
        raise ValueError(
            "当前没有选中酒店的 hotel_reviews 证据，不能生成 hotel_context"
        )

    all_weather_risks = set(
        locked_context["weather_overview"].risk_types
    )
    safety_refs = set(
        refs_by_category.get("safety_notices", set())
    )

    # 6. 有天气风险时必须输出安全提醒；
    #    如果 Evidence Pool 有 safety_notices，至少引用其中一条。
    if all_weather_risks:
        if not draft.safety_notes:
            raise ValueError("存在天气风险时 safety_notes 不能为空")

        safety_note_refs = {
            ref
            for note in draft.safety_notes
            for ref in note.evidence_refs
        }

        if safety_refs and not safety_note_refs & safety_refs:
            raise ValueError(
                "存在 safety_notices 证据时，安全提醒必须引用至少一条安全证据"
            )

    packing_refs = set(
        refs_by_category.get("packing_checklists", set())
    )

    # 7. 检索到行李清单时，应把它转化成至少一条准备建议。
    if packing_refs:
        if not draft.packing_tips:
            raise ValueError(
                "存在 packing_checklists 证据时 packing_tips 不能为空"
            )

        packing_tip_refs = {
            ref
            for note in draft.packing_tips
            for ref in note.evidence_refs
        }

        if not packing_tip_refs & packing_refs:
            raise ValueError(
                "packing_tips 必须引用至少一条 packing_checklists 证据"
            )

    # 8. 每个 requirement_key 必须恰好处理一次。
    expected_requirement_keys = {
        str(item["requirement_key"])
        for item in requirement_catalog
    }

    actual_requirement_keys = [
        item.requirement_key
        for item in draft.requirement_handling
    ]

    if len(actual_requirement_keys) != len(set(actual_requirement_keys)):
        raise ValueError("requirement_handling 中存在重复 requirement_key")

    if set(actual_requirement_keys) != expected_requirement_keys:
        missing = sorted(
            expected_requirement_keys - set(actual_requirement_keys)
        )
        extra = sorted(
            set(actual_requirement_keys) - expected_requirement_keys
        )
        raise ValueError(
            "requirement_handling 必须完整覆盖 requirement_catalog；"
            f"missing={missing}, extra={extra}"
        )

    # 9. requirement handling 的 evidence_refs 同样必须来自允许目录。
    for item in draft.requirement_handling:
        invalid_refs = set(item.evidence_refs) - allowed_refs
        if invalid_refs:
            raise ValueError(
                f"requirement_key={item.requirement_key} 使用了未知证据："
                + ", ".join(sorted(invalid_refs))
            )

    # 10. source metadata 的选中酒店 ID 再次执行防御性检查。
    selected_hotel_id = locked_context["selected_hotel"].hotel_id

    for ref in hotel_review_refs:
        metadata = prompt_by_ref[ref].get("metadata") or {}
        if str(metadata.get("hotel_id") or "") != selected_hotel_id:
            raise ValueError(
                "hotel_reviews Evidence 的 hotel_id 与选中酒店不一致"
            )


def _assemble_trip_proposal(
    *,
    draft: ProposalDraft,
    locked_context: Mapping[str, Any],
    evidence_context: Mapping[str, Any],
    model_id: str,
    proposal_version: int,
    tool_activity_context: Mapping[str, Any],
) -> TripProposal:
    """
    将 LLM Draft 与锁定事实组装成最终 TripProposal。

    这个函数不再调用 LLM，所有锁定字段都来自上游确定性结果。
    """

    weather_by_date = {
        item.date.isoformat(): item
        for item in locked_context["weather_overview"].daily
    }

    # 1. 将逐日天气事实注入 LLM 生成的活动草稿。
    final_days = [
        ProposalDay(
            day_index=index,
            date=day.date,
            theme=day.theme,
            weather=weather_by_date[day.date.isoformat()],
            activities=day.activities,
            weather_adjustment=day.weather_adjustment,
            day_notes=day.day_notes,
        )
        for index, day in enumerate(
            draft.daily_plan,
            start=1,
        )
    ]

    # 2. 酒店补充信息缺失时生成明确的非实时知识边界说明。
    final_hotel_context = _build_final_hotel_context(
        draft.hotel_context
    )

    used_refs = _collect_draft_evidence_refs(draft)
    used_tool_activity_ids = {
        activity.tool_activity_id
        for day in draft.daily_plan
        for activity in day.activities
        if activity.tool_activity_id
    }
    activity_sources = [
        {
            "activity_id": item["activity_id"],
            "name": item["name"],
            "city": item["city"],
            "available_dates": list(item.get("available_dates", [])),
            "opening_hours": item.get("opening_hours"),
            # 必须在计算 proposal_id 之前显式写入来源字段。
            # ProposalActivitySourceRef 虽然也为 source 提供了相同默认值，
            # 但那个默认值要等 TripProposal 完成 Pydantic 校验后才会补入。
            # 如果这里省略 source，生成器会对“未补全的数据”计算哈希，而
            # Verifier 会对“已补全的数据”重新计算哈希，最终必然得到不同的
            # proposal_id。显式写入可确保生成和校验使用完全相同的内容。
            "source": "mcp:search_city_activities",
        }
        for activity_id in sorted(used_tool_activity_ids)
        if isinstance((item := tool_activity_context.get("by_id", {}).get(activity_id)), Mapping)
    ]

    # 3. 最终 sources 只列出 Proposal 真正使用的证据。
    sources = _build_source_refs(
        used_refs=used_refs,
        evidence_by_ref=evidence_context["by_ref"],
    )

    # 4. 将固定限制与模型识别出的限制合并并去重。
    limitations = _dedupe_strings(
        [
            *draft.limitations,
            "天气、航班和酒店来自项目 Mock 数据，不代表真实供应商实时结果。",
            "RAG 攻略、酒店补充和安全知识是非实时资料，实际情况可能变化。",
            "预算只计算 Mock 航班和酒店已知报价；剩余预算是安排建议，不是实际消费预测。",
            "系统没有执行真实预订、付款、出票或通知。",
            *_preference_gap_messages(
                locked_context["selected_flights"],
                locked_context["selected_hotel"],
            ),
        ]
    )

    content_payload = {
        "summary": draft.summary,
        "highlights": draft.highlights,
        "daily_plan": [
            _model_to_dict(item)
            for item in final_days
        ],
        "hotel_context": _model_to_dict(final_hotel_context),
        "packing_tips": [
            _model_to_dict(item)
            for item in draft.packing_tips
        ],
        "safety_notes": [
            _model_to_dict(item)
            for item in draft.safety_notes
        ],
        "preference_alignment": draft.preference_alignment,
        "requirement_handling": [
            _model_to_dict(item)
            for item in draft.requirement_handling
        ],
        "limitations": limitations,
        "used_refs": sorted(used_refs),
        "activity_sources": activity_sources,
    }

    proposal_id = "proposal_" + _sha256_json(
        {
            "locked_fact_hash": locked_context["locked_fact_hash"],
            "proposal_version": proposal_version,
            "content": content_payload,
            "generator_version": PROPOSAL_GENERATOR_VERSION,
        }
    )[:20]

    return TripProposal(
        proposal_id=proposal_id,
        version=proposal_version,
        generator_version=PROPOSAL_GENERATOR_VERSION,
        prompt_version=PROPOSAL_PROMPT_VERSION,
        model_id=model_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        locked_fact_hash=locked_context["locked_fact_hash"],
        summary=draft.summary,
        highlights=draft.highlights,
        trip_overview=locked_context["trip_overview"],
        selected_flights=locked_context["selected_flights"],
        selected_hotel=locked_context["selected_hotel"],
        weather_overview=locked_context["weather_overview"],
        daily_plan=final_days,
        budget_summary=locked_context["budget_summary"],
        hotel_context=final_hotel_context,
        packing_tips=draft.packing_tips,
        safety_notes=draft.safety_notes,
        preference_alignment=draft.preference_alignment,
        requirement_handling=draft.requirement_handling,
        limitations=limitations,
        sources=sources,
        activity_sources=activity_sources,
        execution_boundary=ExecutionBoundary(),
    )


def _build_final_hotel_context(
    draft: HotelContextDraft | None,
) -> FinalHotelContext:
    """把可选 HotelContextDraft 转成最终展示结构。"""

    if draft is None:
        return FinalHotelContext(
            available=False,
            summary="当前知识库没有该酒店的非实时补充资料。",
            nearby_facilities=[],
            cautions=[],
            evidence_refs=[],
            non_realtime_notice=(
                "酒店选择仍只依据 Mock 结构化价格、房态、评分和偏好特征。"
            ),
        )

    return FinalHotelContext(
        available=True,
        summary=draft.summary,
        nearby_facilities=draft.nearby_facilities,
        cautions=draft.cautions,
        evidence_refs=draft.evidence_refs,
        non_realtime_notice=(
            "以上周边与体验信息来自非实时 RAG 文档，实际情况可能变化；"
            "它没有参与酒店数值评分。"
        ),
    )


def _build_source_refs(
    *,
    used_refs: set[str],
    evidence_by_ref: Mapping[str, Mapping[str, Any]],
) -> list[ProposalSourceRef]:
    """把实际使用的 chunk_id 转成可追溯来源列表。"""

    output: list[ProposalSourceRef] = []

    for ref in sorted(used_refs):
        item = evidence_by_ref.get(ref)

        if item is None:
            # 正常情况下前置校验已经拦截未知 ref。
            continue

        metadata = item.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}

        rerank_score = _safe_optional_float(item.get("rerank_score"))

        output.append(
            ProposalSourceRef(
                chunk_id=ref,
                category=str(item.get("category") or "unknown"),
                source=str(item.get("source") or "unknown"),
                title=_optional_text(metadata.get("title")),
                section=_optional_text(metadata.get("section")),
                doc_type=_optional_text(metadata.get("doc_type")),
                hotel_id=_optional_text(metadata.get("hotel_id")),
                risk_type=metadata.get("risk_type"),
                rerank_score=rerank_score,
            )
        )

    return output


def _collect_draft_evidence_refs(
    draft: ProposalDraft,
) -> set[str]:
    """收集 ProposalDraft 所有层级使用的 evidence_refs。"""

    refs: set[str] = set()

    for day in draft.daily_plan:
        for activity in day.activities:
            refs.update(activity.evidence_refs)

    if draft.hotel_context is not None:
        refs.update(draft.hotel_context.evidence_refs)

    for item in draft.packing_tips:
        refs.update(item.evidence_refs)

    for item in draft.safety_notes:
        refs.update(item.evidence_refs)

    for item in draft.requirement_handling:
        refs.update(item.evidence_refs)

    return {
        ref.strip()
        for ref in refs
        if isinstance(ref, str) and ref.strip()
    }


def _preference_gap_messages(
    selected_flights: SelectedFlights,
    selected_hotel: SelectedHotelFact,
) -> list[str]:
    """把 CandidateRank 的偏好差距转换成最终限制说明。"""

    messages: list[str] = []

    for gap in [
        *selected_flights.outbound.preference_gaps,
        *selected_flights.return_flight.preference_gaps,
        *selected_hotel.preference_gaps,
    ]:
        if not isinstance(gap, Mapping):
            continue

        message = str(gap.get("message") or "").strip()
        if message:
            messages.append(message)

    return _dedupe_strings(messages)


def _evidence_quality(item: Mapping[str, Any]) -> tuple[float, float]:
    """比较重复证据时，优先保留 rerank_score 更高的一条。"""

    rerank_score = _safe_optional_float(item.get("rerank_score"))
    fusion_score = _safe_optional_float(item.get("fusion_score"))

    return (
        rerank_score if rerank_score is not None else -1.0,
        fusion_score if fusion_score is not None else 0.0,
    )


def _evidence_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    """按精排分数、排名和 chunk_id 对证据稳定排序。"""

    rerank_score = _safe_optional_float(item.get("rerank_score"))
    rerank_rank = _safe_int(item.get("rerank_rank"), default=10**9)
    fusion_score = _safe_optional_float(item.get("fusion_score"))

    return (
        -(rerank_score if rerank_score is not None else -1.0),
        rerank_rank,
        -(fusion_score if fusion_score is not None else 0.0),
        str(item.get("chunk_id") or ""),
    )


def _short_error_message(exc: Exception, max_length: int = 900) -> str:
    """压缩异常文本，避免下一次 Prompt 被错误信息占满。"""

    text = " ".join(str(exc).split())

    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def _truncate_text(text: str, max_chars: int) -> str:
    """截断放入 Prompt 的 Evidence 正文。"""

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3] + "..."


def _sha256_json(value: Any) -> str:
    """对稳定 JSON 表示计算 SHA-256。"""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def _short_hash(text: str) -> str:
    """为要求生成短稳定标识。"""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:12]


def _model_to_dict(model: Any) -> dict[str, Any]:
    """兼容 Pydantic v1/v2 的 JSON dict 转换。"""

    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")

    return model.dict()


def _parse_required_date(value: Any, field_name: str) -> date:
    """解析必需 ISO 日期。"""

    if isinstance(value, date):
        return value

    if not value:
        raise ValueError(f"{field_name} 不能为空")

    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ValueError(
            f"{field_name} 不是合法 ISO 日期：{value}"
        ) from exc


def _required_text(value: Any, field_name: str) -> str:
    """读取必需文本。"""

    text = str(value or "").strip()

    if not text:
        raise ValueError(f"{field_name} 不能为空")

    return text


def _optional_text(value: Any) -> str | None:
    """读取可选文本。"""

    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _safe_non_negative_float(value: Any, field_name: str) -> float:
    """读取必须存在的非负金额或评分。"""

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是数字") from exc

    if number < 0:
        raise ValueError(f"{field_name} 不能小于 0")

    return number


def _safe_unit_float(value: Any, field_name: str) -> float:
    """读取 0—1 标准化分数。"""

    number = _safe_non_negative_float(value, field_name)

    if number > 1:
        raise ValueError(f"{field_name} 必须位于 0 到 1")

    return number


def _safe_optional_unit_float(value: Any) -> float | None:
    """读取可空 0—1 分数。"""

    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not 0 <= number <= 1:
        return None

    return number


def _safe_optional_float(value: Any) -> float | None:
    """安全读取可空 float。"""

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int) -> int:
    """安全读取 int。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_optional_positive_int(value: Any) -> int | None:
    """读取可空正整数。"""

    if value is None:
        return None

    try:
        number = int(value)
    except (TypeError, ValueError):
        return None

    return number if number >= 1 else None


def _string_list(value: Any) -> list[str]:
    """把任意 list 安全转换成去空字符串列表。"""

    if not isinstance(value, list):
        return []

    return [
        str(item).strip()
        for item in value
        if item is not None and str(item).strip()
    ]


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    """把任意 list 安全转换成 dict 列表。"""

    if not isinstance(value, list):
        return []

    return [
        dict(item)
        for item in value
        if isinstance(item, Mapping)
    ]


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    """保持顺序去重字符串。"""

    return list(
        dict.fromkeys(
            value.strip()
            for value in values
            if isinstance(value, str) and value.strip()
        )
    )
