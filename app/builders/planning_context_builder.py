from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Mapping, Sequence

from app.schemas.planning_context_schema import BudgetPlan, PlanningContext


# ----------------------------------------------------------------------
# 默认预算启发式
# ----------------------------------------------------------------------
# 这不是行业标准，也不是实际消费预测。
# 它只在用户没有表达消费倾向时，为价格适配评分提供一个初始软目标。
DEFAULT_BUDGET_RATIOS: dict[str, float] = {
    "transport": 0.35,
    "hotel": 0.35,
    "food_activity": 0.30,
}

# spend preference 强度最多让某一类别的基础权重上下变化 35%。
# 例如 hotel increase strength=1.0：hotel raw weight = 0.35 * 1.35。
# 最后还会把三个类别重新归一化，确保比例总和等于 1。
MAX_SPEND_RATIO_ADJUSTMENT = 0.35


PREFERENCE_LABELS = {
    "avoid_early_flight": "避免早班机",
    "avoid_late_arrival": "避免太晚抵达",
    "prefer_direct": "偏好直飞",
    "price_sensitive": "价格敏感",
    "quiet": "安静酒店",
    "cleanliness": "干净",
    "near_subway": "靠近地铁",
    "high_rating": "高评分",
    "budget_friendly": "预算友好",
    "food": "美食",
    "nature_scenery": "自然风景",
    "culture_history": "文化历史",
    "city_walk": "城市漫步",
    "entertainment": "玩乐娱乐",
    "fitness": "锻炼运动",
    "adventure": "冒险户外",
    "photography": "拍照摄影",
    "slow": "慢节奏",
    "low_walking_intensity": "低步行强度",
    "weather_sensitive": "天气敏感",
    "avoid_crowds": "避开人群",
}

SPEND_CATEGORY_LABELS = {
    "transport": "交通",
    "hotel": "住宿",
    "food_activity": "餐饮和活动",
}


def build_planning_context(
    trip_request: Mapping[str, Any],
    user_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    将旅行需求、长期偏好和本次偏好整理成统一 PlanningContext。

    这个函数是确定性的 Context Builder：
        - 不读数据库；
        - 不调用 LLM；
        - 不查询 Mock；
        - 不选择酒店和航班。

    Args:
        trip_request:
            InputExtractNode 已经完成 Pydantic 校验的旅行需求。

        user_profile:
            MemoryReadNode 返回的长期偏好和本次有效偏好。

    Returns:
        dict:
            可直接写入 state["planning_context"]。
    """

    if not isinstance(trip_request, Mapping):
        raise TypeError("trip_request 必须是 dict-like 对象")

    profile = user_profile if isinstance(user_profile, Mapping) else {}

    # 1. 统一基础旅行字段，避免后续每个节点重复计算 nights、room_count、end_date。
    request_summary = _build_request_summary(trip_request)

    # 2. 将长期偏好和本次 preference_signals 合并成 0—1 连续权重。
    preference_weights = _build_preference_weights(
        trip_request=trip_request,
        user_profile=profile,
    )

    # 3. 客观硬约束独立整理，不再从主观偏好推导硬约束。
    hard_constraints = _build_hard_constraints(trip_request)

    # 4. 消费倾向属于本次请求，用于调整预算软目标。
    spend_preferences = _read_mapping_list(
        trip_request.get("spend_preferences")
    )

    # 5. 根据默认启发式 + 用户消费倾向生成动态预算计划。
    budget_plan = _build_budget_plan(
        trip_request=trip_request,
        user_profile=profile,
        request_summary=request_summary,
        hard_constraints=hard_constraints,
        spend_preferences=spend_preferences,
    )

    named_constraints = _read_mapping_list(
        trip_request.get("named_constraints")
    )
    unmapped_requirements = _read_mapping_list(
        trip_request.get("unmapped_requirements")
    )

    # 6. 生成短摘要，供 Trace、Prompt 和调试使用；不保存大段原始输入。
    context_summary = _build_context_summary(
        request_summary=request_summary,
        preference_weights=preference_weights,
        hard_constraints=hard_constraints,
        budget_plan=budget_plan,
        named_constraints=named_constraints,
    )

    planning_context = PlanningContext(
        request=request_summary,
        hard_constraints=hard_constraints,
        preference_weights=preference_weights,
        spend_preferences=spend_preferences,
        named_constraints=named_constraints,
        unmapped_requirements=unmapped_requirements,
        diet_preferences=_read_diet_preferences(profile),
        budget_plan=budget_plan,
        context_summary=context_summary,
        source_meta={
            "builder_version": "planning_context_v2",
            "used_trip_request": True,
            "used_user_profile": bool(profile),
            "preference_source": "memory_effective_plus_current_signals",
            "hard_constraint_source": "trip_request.hard_constraints",
            "budget_policy": budget_plan.allocation_source,
        },
    )

    return planning_context.to_state_dict()


def _build_request_summary(
    trip_request: Mapping[str, Any],
) -> dict[str, Any]:
    """标准化基础旅行字段。"""

    days = _safe_int(trip_request.get("days"), default=0)
    people_count = max(
        _safe_int(trip_request.get("people_count"), default=1),
        1,
    )

    # 1. 单人旅行默认 1 间房。
    #    多人没有 room_count 的情况应在 MissingInfoCheckNode 阶段被澄清。
    room_count = _safe_optional_int(trip_request.get("room_count"))
    if room_count is None and people_count == 1:
        room_count = 1

    if room_count is None:
        # 防御式兜底：正常工作流不应走到这里。
        # 使用 1 只为避免 Builder 崩溃，Validator 仍会阻止流程继续。
        room_count = 1

    # 2. 三日游通常住宿两晚；一日游不需要住宿。
    nights = max(days - 1, 0)

    start_date = trip_request.get("start_date")
    end_date = trip_request.get("end_date")

    # 3. start_date + days 足够时，确定性推导 end_date。
    inferred_end_date = _infer_end_date(start_date=start_date, days=days)
    if not end_date and inferred_end_date:
        end_date = inferred_end_date

    return {
        "origin": trip_request.get("origin"),
        "destination": trip_request.get("destination"),
        "start_date": start_date,
        "end_date": end_date,
        "days": days,
        "nights": nights,
        "people_count": people_count,
        "room_count": max(room_count, 1),
        "total_budget": _safe_optional_float(trip_request.get("budget")),
    }


def _build_preference_weights(
    trip_request: Mapping[str, Any],
    user_profile: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """
    合并长期偏好和本次偏好。

    合并规则：
        1. 先读取 MemoryReadNode 的 effective preferences。
        2. 再读取本次 trip_request.preference_signals。
        3. 同名偏好取更大 strength。

    这里没有 soft/strong/hard 三档。
    所有主观偏好都统一为 0—1 连续权重。
    """

    weights = {
        "flight": _read_profile_weight_bucket(user_profile, "flight_preferences"),
        "hotel": _read_profile_weight_bucket(user_profile, "hotel_preferences"),
        "activity": _read_profile_weight_bucket(user_profile, "activity_preferences"),
        "pace": _read_profile_weight_bucket(user_profile, "pace_preferences"),
        "risk": _read_profile_weight_bucket(user_profile, "risk_preferences"),
    }

    # 1. 本次请求中的明确偏好覆盖较低的长期偏好。
    for signal in _iter_preference_signals(trip_request):
        domain = str(signal.get("domain") or "")
        key = str(signal.get("key") or "")
        strength = _clamp01(
            _safe_float(signal.get("strength"), default=1.0)
        )

        if domain not in weights or not key:
            continue

        weights[domain][key] = max(
            weights[domain].get(key, 0.0),
            strength,
        )

    # 2. 所有权重统一限制在 0—1。
    return {
        domain: _normalize_score_map(values)
        for domain, values in weights.items()
    }


def _build_hard_constraints(
    trip_request: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """
    将 trip_request.hard_constraints 按作用域分组。

    主观偏好不会进入这里。
    例如“一定要安静”只会在 preference_weights.hotel.quiet 中表现为高权重。
    """

    grouped: dict[str, list[dict[str, Any]]] = {
        "trip": [],
        "flight": [],
        "hotel": [],
    }

    # 1. 只保留 Validator 白名单能够确定性执行的客观硬约束。
    #    即使 LLM 偶尔错误地把 quiet_score 放进 hard_constraints，
    #    PlanningContext 也不会把它继续传给 CandidateRank 做硬过滤。
    from app.schemas.request_constraint_schema import HardConstraint
    from app.validators.trip_request_validator import validate_hard_constraint

    for item in _read_mapping_list(trip_request.get("hard_constraints")):
        try:
            constraint = HardConstraint(**item)
        except Exception:
            # Pydantic 通常已经在 TripRequest 阶段拦截坏结构；
            # 这里再做一次防御性忽略，避免 Builder 因脏 State 崩溃。
            continue

        if validate_hard_constraint(constraint) is not None:
            continue

        domain = constraint.domain
        normalized = constraint.to_state_dict()

        if normalized not in grouped[domain]:
            grouped[domain].append(normalized)

    # 2. 用户提供的总预算天然是组合层硬上限。
    total_budget = _safe_optional_float(trip_request.get("budget"))

    if total_budget is not None and not _contains_constraint(
        grouped["trip"],
        field="total_budget",
    ):
        grouped["trip"].append(
            {
                "domain": "trip",
                "field": "total_budget",
                "operator": "<=",
                "value": total_budget,
                "evidence": "用户本次旅行总预算",
                "source": "derived",
            }
        )

    return grouped


def _build_budget_plan(
    *,
    trip_request: Mapping[str, Any],
    user_profile: Mapping[str, Any],
    request_summary: Mapping[str, Any],
    hard_constraints: Mapping[str, Sequence[Mapping[str, Any]]],
    spend_preferences: list[dict[str, Any]],
) -> BudgetPlan:
    """
    生成动态预算安排。

    计算流程：
        1. 从 35/35/30 默认启发式开始。
        2. 将 spend_preferences 转成每类 -1 到 1 的净倾向。
        3. 使用固定 MAX_SPEND_RATIO_ADJUSTMENT 调整原始权重。
        4. 重新归一化，使三类比例总和等于 1。
        5. 有总预算时计算软目标金额。

    为什么由代码计算而不是让 LLM 直接输出比例？
        - 同样输入得到同样结果；
        - 容易单元测试和 Eval；
        - LLM 只负责理解“哪里想多花、哪里想省”，不负责随意给百分比。
    """

    total_budget = _safe_optional_float(trip_request.get("budget"))
    people_count = max(_safe_int(request_summary.get("people_count"), 1), 1)
    room_count = max(_safe_int(request_summary.get("room_count"), 1), 1)
    nights = max(_safe_int(request_summary.get("nights"), 0), 0)
    days = max(_safe_int(request_summary.get("days"), 1), 1)

    # 1. 根据用户消费表达调整默认比例。
    adjusted_ratios, allocation_reasons = _adjust_budget_ratios(
        spend_preferences
    )

    has_spend_preference = bool(spend_preferences)

    if total_budget is None:
        allocation_source = "no_budget"
    elif has_spend_preference:
        allocation_source = "user_adjusted_heuristic_v1"
    else:
        allocation_source = "default_heuristic_v1"

    # 2. 从客观硬约束中提取预算相关上限。
    max_hotel_price_per_night = _find_constraint_value(
        hard_constraints.get("hotel", []),
        field="price_per_night",
        operators={"<=", "<"},
    )
    max_transport_total = _find_constraint_value(
        hard_constraints.get("flight", []),
        field="price",
        operators={"<=", "<"},
    )

    budget_preferences = _read_budget_preferences(user_profile)
    preferred_hotel_price_max = _safe_optional_float(
        budget_preferences.get("preferred_hotel_price_max")
    )

    # 3. 没有总预算时，仍然保留长期酒店价格偏好作为软目标。
    if total_budget is None:
        return BudgetPlan(
            mode="no_budget_limit",
            total_budget=None,
            hard_limits={
                "total_budget": None,
                "max_hotel_price_per_night": max_hotel_price_per_night,
                "max_transport_total": max_transport_total,
            },
            base_ratios=dict(DEFAULT_BUDGET_RATIOS),
            adjusted_ratios=adjusted_ratios,
            soft_targets={
                "transport_budget": None,
                "hotel_budget": None,
                "food_activity_budget": None,
                "food_activity_budget_per_day": None,
                "target_hotel_price_per_room_night": preferred_hotel_price_max,
            },
            flexibility={
                "allow_cross_category_tradeoff": True,
                "hotel_can_exceed_target_if_total_ok": True,
                "flight_can_exceed_target_if_total_ok": True,
            },
            allocation_source=allocation_source,
            allocation_reasons=allocation_reasons,
            nights=nights,
            people_count=people_count,
            room_count=room_count,
            notes=[
                "用户没有提供总预算，因此不做总预算组合限制。",
                "消费比例只保留为解释性策略，不会产生具体金额软目标。",
                "长期 preferred_hotel_price_max 仅作为价格偏好，不是硬限制。",
            ],
        )

    # 4. 按动态比例计算三类软预算目标。
    transport_budget = round(
        total_budget * adjusted_ratios["transport"],
        2,
    )
    hotel_budget = round(
        total_budget * adjusted_ratios["hotel"],
        2,
    )
    food_activity_budget = round(
        total_budget * adjusted_ratios["food_activity"],
        2,
    )

    # 5. 酒店软目标按“房间数 × 晚数”计算每间每晚目标。
    room_nights = room_count * nights
    if room_nights > 0:
        target_hotel_price = round(hotel_budget / room_nights, 2)
    else:
        target_hotel_price = None

    # 6. 长期酒店价格偏好可以让软目标更保守，但不会进入 hard_limits。
    if preferred_hotel_price_max is not None:
        if target_hotel_price is None:
            target_hotel_price = preferred_hotel_price_max
        else:
            target_hotel_price = min(
                target_hotel_price,
                preferred_hotel_price_max,
            )

    food_activity_per_day = round(food_activity_budget / days, 2)

    notes = [
        "总预算只用于组合硬上限，不代表系统能够预测全部真实旅行消费。",
        "分项金额是 CandidateRank 和 BudgetOptimize 使用的软目标。",
        "航班或酒店可以超过单项软目标，只要最终组合没有超过真正硬限制。",
    ]

    if allocation_source == "default_heuristic_v1":
        notes.append("用户未表达消费倾向，使用 35/35/30 默认启发式。")
    else:
        notes.append("已根据用户消费倾向动态调整 35/35/30 默认启发式。")

    return BudgetPlan(
        mode="budget_limited",
        total_budget=total_budget,
        hard_limits={
            "total_budget": total_budget,
            "max_hotel_price_per_night": max_hotel_price_per_night,
            "max_transport_total": max_transport_total,
        },
        base_ratios=dict(DEFAULT_BUDGET_RATIOS),
        adjusted_ratios=adjusted_ratios,
        soft_targets={
            "transport_budget": transport_budget,
            "hotel_budget": hotel_budget,
            "food_activity_budget": food_activity_budget,
            "food_activity_budget_per_day": food_activity_per_day,
            "target_hotel_price_per_room_night": target_hotel_price,
        },
        flexibility={
            "allow_cross_category_tradeoff": True,
            "hotel_can_exceed_target_if_total_ok": True,
            "flight_can_exceed_target_if_total_ok": True,
        },
        allocation_source=allocation_source,
        allocation_reasons=allocation_reasons,
        nights=nights,
        people_count=people_count,
        room_count=room_count,
        notes=notes,
    )


def _adjust_budget_ratios(
    spend_preferences: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], list[str]]:
    """
    根据消费倾向调整默认预算比例。

    关键参数：
        MAX_SPEND_RATIO_ADJUSTMENT = 0.35

    它表示最强 increase/decrease 信号最多让某类别的“原始权重”
    在归一化前上下变化 35%，避免一句话导致预算比例极端失衡。
    """

    # 1. 每类消费先累计成 -1 到 1 的净倾向。
    net_signals = {
        category: 0.0
        for category in DEFAULT_BUDGET_RATIOS
    }
    evidence_by_category: dict[str, list[str]] = {
        category: []
        for category in DEFAULT_BUDGET_RATIOS
    }

    for item in spend_preferences:
        category = str(item.get("category") or "")
        direction = str(item.get("direction") or "")
        strength = _clamp01(_safe_float(item.get("strength"), 0.0))

        if category not in net_signals or direction not in {"increase", "decrease"}:
            continue

        signed_strength = strength if direction == "increase" else -strength
        net_signals[category] = _clamp(
            net_signals[category] + signed_strength,
            minimum=-1.0,
            maximum=1.0,
        )

        evidence = str(item.get("evidence") or "").strip()
        if evidence:
            evidence_by_category[category].append(evidence)

    # 2. 把净倾向转换为原始调整权重。
    raw_ratios: dict[str, float] = {}

    for category, base_ratio in DEFAULT_BUDGET_RATIOS.items():
        multiplier = 1.0 + (
            MAX_SPEND_RATIO_ADJUSTMENT * net_signals[category]
        )
        raw_ratios[category] = base_ratio * multiplier

    # 3. 重新归一化，确保最终比例之和严格为 1。
    total = sum(raw_ratios.values()) or 1.0
    adjusted_ratios = {
        category: round(value / total, 6)
        for category, value in raw_ratios.items()
    }

    # 4. 修正浮点四舍五入误差，让三个比例精确加总为 1。
    difference = round(1.0 - sum(adjusted_ratios.values()), 6)
    adjusted_ratios["food_activity"] = round(
        adjusted_ratios["food_activity"] + difference,
        6,
    )

    # 5. 生成人能看懂的调整原因。
    reasons: list[str] = []

    for category, signal in net_signals.items():
        if abs(signal) < 1e-9:
            continue

        direction_label = "提高" if signal > 0 else "降低"
        evidence_text = "；".join(dict.fromkeys(evidence_by_category[category]))
        reason = f"根据用户表达，{direction_label}{SPEND_CATEGORY_LABELS[category]}预算软目标。"

        if evidence_text:
            reason += f" 原话：{evidence_text}"

        reasons.append(reason)

    if not reasons:
        reasons.append("用户没有表达明确消费倾向，保留默认预算启发式。")

    return adjusted_ratios, reasons


def _build_context_summary(
    *,
    request_summary: Mapping[str, Any],
    preference_weights: Mapping[str, Mapping[str, float]],
    hard_constraints: Mapping[str, Sequence[Mapping[str, Any]]],
    budget_plan: BudgetPlan,
    named_constraints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """生成给 Trace 和后续 Prompt 使用的短摘要。"""

    start_date = request_summary.get("start_date")
    end_date = request_summary.get("end_date")

    if start_date and end_date:
        date_range = f"{start_date} 至 {end_date}"
    else:
        date_range = str(start_date or "")

    main_preferences = _extract_main_preference_labels(preference_weights)

    # 1. 硬约束摘要只保存字段和操作符，不保存整个原始大对象。
    hard_constraint_labels = [
        f"{domain}.{item.get('field')} {item.get('operator')} {item.get('value')}"
        for domain, items in hard_constraints.items()
        for item in items
    ]

    named_labels = [
        f"{item.get('entity_type')}:{item.get('entity_name')}"
        for item in named_constraints
        if item.get("entity_name")
    ]

    return {
        "destination": request_summary.get("destination"),
        "date_range": date_range,
        "days": request_summary.get("days"),
        "nights": request_summary.get("nights"),
        "people_count": request_summary.get("people_count"),
        "room_count": request_summary.get("room_count"),
        "budget_mode": budget_plan.mode,
        "allocation_source": budget_plan.allocation_source,
        "main_preferences": main_preferences,
        "hard_constraints": hard_constraint_labels,
        "named_constraints": named_labels,
    }


def _extract_main_preference_labels(
    preference_weights: Mapping[str, Mapping[str, float]],
    threshold: float = 0.7,
    max_items: int = 8,
) -> list[str]:
    """提取权重较高的主要偏好标签。"""

    items: list[tuple[str, float]] = []

    for bucket in preference_weights.values():
        for key, value in bucket.items():
            if value >= threshold:
                items.append((key, value))

    items.sort(key=lambda item: item[1], reverse=True)

    labels = [
        PREFERENCE_LABELS.get(key, key)
        for key, _ in items[:max_items]
    ]

    return list(dict.fromkeys(labels))


def _read_profile_weight_bucket(
    user_profile: Mapping[str, Any],
    bucket_name: str,
) -> dict[str, float]:
    """优先从 user_profile.effective 中读取某类偏好权重。"""

    effective = user_profile.get("effective")
    if isinstance(effective, Mapping):
        value = effective.get(bucket_name)
        if isinstance(value, Mapping):
            return _normalize_score_map(value)

    value = user_profile.get(bucket_name)
    if isinstance(value, Mapping):
        return _normalize_score_map(value)

    return {}


def _read_budget_preferences(
    user_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """读取长期预算偏好。"""

    effective = user_profile.get("effective")
    if isinstance(effective, Mapping):
        value = effective.get("budget_preferences")
        if isinstance(value, Mapping):
            return dict(value)

    value = user_profile.get("budget_preferences")
    return dict(value) if isinstance(value, Mapping) else {}


def _read_diet_preferences(
    user_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """读取饮食偏好；这类字段不强制全部转换成数值权重。"""

    effective = user_profile.get("effective")
    if isinstance(effective, Mapping):
        value = effective.get("diet_preferences")
        if isinstance(value, Mapping):
            return dict(value)

    value = user_profile.get("diet_preferences")
    return dict(value) if isinstance(value, Mapping) else {}


def _iter_preference_signals(
    trip_request: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """读取并过滤 trip_request.preference_signals。"""

    return _read_mapping_list(trip_request.get("preference_signals"))


def _read_mapping_list(value: Any) -> list[dict[str, Any]]:
    """只保留列表中的 dict-like 项。"""

    if not isinstance(value, list):
        return []

    return [dict(item) for item in value if isinstance(item, Mapping)]


def _normalize_score_map(
    value: Mapping[str, Any],
) -> dict[str, float]:
    """将偏好权重统一转换为 0—1 float。"""

    return {
        str(key): _clamp01(_safe_float(score, default=0.0))
        for key, score in value.items()
    }


def _contains_constraint(
    constraints: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> bool:
    """判断约束列表中是否已经包含某个字段。"""

    return any(str(item.get("field") or "") == field for item in constraints)


def _find_constraint_value(
    constraints: Sequence[Mapping[str, Any]],
    *,
    field: str,
    operators: set[str],
) -> float | None:
    """从约束列表中读取指定数值上限。"""

    values: list[float] = []

    for item in constraints:
        if str(item.get("field") or "") != field:
            continue
        if str(item.get("operator") or "") not in operators:
            continue

        value = _safe_optional_float(item.get("value"))
        if value is not None:
            values.append(value)

    # 多个上限同时存在时，取最严格的较小值。
    return min(values) if values else None


def _infer_end_date(
    start_date: Any,
    days: int,
) -> str | None:
    """根据 start_date + days 推导 end_date。"""

    if not start_date or days <= 0:
        return None

    try:
        start = date.fromisoformat(str(start_date)[:10])
    except ValueError:
        return None

    return (start + timedelta(days=days - 1)).isoformat()


def _safe_int(value: Any, default: int) -> int:
    """安全转换 int。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_optional_int(value: Any) -> int | None:
    """安全转换可空 int。"""

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float) -> float:
    """安全转换 float。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_optional_float(value: Any) -> float | None:
    """安全转换可空 float。"""

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(value: float) -> float:
    """把数值限制在 0 到 1。"""

    return _clamp(value, minimum=0.0, maximum=1.0)


def _clamp(value: float, *, minimum: float, maximum: float) -> float:
    """把数值限制在指定闭区间。"""

    return max(minimum, min(maximum, value))
