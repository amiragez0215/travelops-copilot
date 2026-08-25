from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Mapping

from sqlalchemy.orm import Session

from app.db.models import TripHistory, User, UserProfile


DEFAULT_LONG_TERM_PROFILE: dict[str, Any] = {
    "flight_preferences": {
        "avoid_early_flight": 0.5,
        "prefer_direct": 0.5,
        "price_sensitive": 0.5,
        "avoid_late_arrival": 0.5,
    },
    "hotel_preferences": {
        "quiet": 0.5,
        "cleanliness": 0.5,
        "near_subway": 0.5,
        "high_rating": 0.5,
        "budget_friendly": 0.5,
    },
    "activity_preferences": {
        "food": 0.5,
        "nature_scenery": 0.5,
        "culture_history": 0.5,
        "city_walk": 0.5,
        "entertainment": 0.5,
        "fitness": 0.3,
        "adventure": 0.3,
        "photography": 0.4,
    },
    "pace_preferences": {
        "slow": 0.5,
        "low_walking_intensity": 0.5,
        "packed_schedule": 0.3,
    },
    "diet_preferences": {
        "spicy_tolerance": "unknown",
        "likes": [],
        "avoid_foods": [],
    },
    "budget_preferences": {
        "budget_sensitivity": 0.5,
        "preferred_hotel_price_max": None,
        "preferred_total_budget_range": None,
    },
    "risk_preferences": {
        "weather_sensitive": 0.5,
        "avoid_crowds": 0.5,
        "risk_tolerance": 0.5,
    },
    "profile_notes": [],
}


class MemoryRepository:
    """
    用户记忆仓库。

    Repository 的职责：
        1. 从 SQL 读取长期用户画像。
        2. 从本次 trip_request 读取即时偏好。
        3. 将长期偏好和即时偏好合并成 effective preferences。

    Repository 不负责：
        - 不解析自然语言。
        - 不判断真正的硬约束。
        - 不读写 LangGraph State。
        - 不决定酒店或航班排序。

    特别说明：
        hard_constraints 已经从偏好记忆中移除。
        真正的客观硬约束来自 trip_request["hard_constraints"]，
        由 PlanningContextNode 整理并由 CandidateRankNode 执行。
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def read_user_memory(
        self,
        user_id: str | None,
        trip_request: Mapping[str, Any] | None = None,
        history_limit: int = 3,
    ) -> dict[str, Any]:
        """
        读取长期偏好，并合并本次请求偏好。

        Args:
            user_id:
                当前用户 ID。匿名用户可以传 None。

            trip_request:
                InputExtractNode 输出。当前只读取其中的偏好、预算和约束摘要。

            history_limit:
                最多读取多少条近期旅行历史。

        Returns:
            dict:
                MemoryReadNode 可以直接写入 state["user_profile"] 的结构。
        """

        # 1. 先从本次请求构造即时偏好。
        #    即使数据库用户不存在，这部分也能够正常工作。
        current_request = build_current_request_preferences(trip_request)

        # 2. 匿名用户没有 SQL 画像，使用默认长期偏好。
        if not user_id:
            return _build_memory_result(
                user_id=None,
                user_found=False,
                profile_found=False,
                source="default:anonymous",
                long_term=copy.deepcopy(DEFAULT_LONG_TERM_PROFILE),
                current_request=current_request,
                recent_trip_history=[],
            )

        # 3. 按 user_id 读取用户主体和画像。
        user = self.session.get(User, user_id)
        profile = self.session.get(UserProfile, user_id)

        # 4. ORM 对象转换成后续节点只需要的普通 dict。
        long_term = self._build_long_term_profile(profile)
        history = self._read_recent_trip_history(user_id, history_limit)

        # 5. 统一构造 user_profile 输出。
        return _build_memory_result(
            user_id=user_id,
            user_found=user is not None,
            profile_found=profile is not None,
            source="sql:user_profiles" if profile else "default:no_profile",
            long_term=long_term,
            current_request=current_request,
            recent_trip_history=history,
        )

    def _build_long_term_profile(
        self,
        profile: UserProfile | None,
    ) -> dict[str, Any]:
        """
        从 UserProfile ORM 对象构造长期偏好。

        如果用户还没有画像，返回一份深拷贝默认值，
        避免后续节点修改结果时污染模块级常量。
        """

        if profile is None:
            return copy.deepcopy(DEFAULT_LONG_TERM_PROFILE)

        return {
            "flight_preferences": _ensure_dict(profile.flight_preferences_json),
            "hotel_preferences": _ensure_dict(profile.hotel_preferences_json),
            "activity_preferences": _ensure_dict(profile.activity_preferences_json),
            "pace_preferences": _ensure_dict(profile.pace_preferences_json),
            "diet_preferences": _ensure_dict(profile.diet_preferences_json),
            "budget_preferences": _ensure_dict(profile.budget_preferences_json),
            "risk_preferences": _ensure_dict(profile.risk_preferences_json),
            "profile_notes": _ensure_list(profile.profile_notes_json),
        }

    def _read_recent_trip_history(
        self,
        user_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        读取最近几条旅行历史摘要。

        历史记录目前只用于给后续长期偏好更新提供上下文，
        不直接参与本轮酒店或航班评分。
        """

        rows = (
            self.session.query(TripHistory)
            .filter(TripHistory.user_id == user_id)
            .order_by(TripHistory.created_at.desc())
            .limit(limit)
            .all()
        )

        return [_trip_history_to_dict(row) for row in rows]


def build_current_request_preferences(
    trip_request: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    从本次 trip_request 提取即时偏好和请求级辅助信息。

    优先读取新的 preference_signals；
    如果旧测试或旧数据没有 preference_signals，
    再从 transport_preferences、hotel_preferences、travel_style 兜底。

    注意：
        spend_preferences 和 named_constraints 只属于本次请求，
        当前不会自动写入长期 user_profiles。
    """

    if not isinstance(trip_request, Mapping):
        return _empty_current_request()

    signals = trip_request.get("preference_signals")

    if isinstance(signals, list) and signals:
        result = _current_request_from_signals(
            signals=signals,
            trip_request=trip_request,
        )
    else:
        result = _current_request_from_legacy_lists(trip_request)

    # 1. 消费倾向单独保留，PlanningContextNode 后面使用它动态调整预算比例。
    result["spend_preferences"] = _ensure_mapping_list(
        trip_request.get("spend_preferences")
    )

    # 2. 指定酒店、食物、景点等实体约束也属于本次请求，不属于长期画像。
    result["named_constraints"] = _ensure_mapping_list(
        trip_request.get("named_constraints")
    )

    return result


def _empty_current_request() -> dict[str, Any]:
    """创建一个字段稳定的空本次偏好结构。"""

    return {
        "flight_preferences": {},
        "hotel_preferences": {},
        "activity_preferences": {},
        "pace_preferences": {},
        "risk_preferences": {},
        "preference_signals": [],
        "spend_preferences": [],
        "named_constraints": [],
        "constraints": [],
        "budget": None,
    }


def _current_request_from_signals(
    signals: list[Any],
    trip_request: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从 PreferenceSignal dict 列表构造当前偏好权重。

    同一个 domain.key 出现多次时取更高 strength。
    “一定、特别、非常”等词已经在 Extractor 中表现为更高 strength，
    不会在这里转成 hard constraint。
    """

    result = _empty_current_request()

    # 1. 遍历结构化偏好信号。
    for raw_signal in signals:
        if not isinstance(raw_signal, Mapping):
            continue

        domain = str(raw_signal.get("domain") or "")
        key = str(raw_signal.get("key") or "")
        strength = _clamp01(
            _safe_float(raw_signal.get("strength"), default=1.0)
        )

        if not domain or not key:
            continue

        # 2. 将 PreferenceSignal.domain 映射到用户画像中的偏好桶。
        target = _domain_to_preference_bucket(domain)

        if target is not None:
            result[target][key] = max(
                result[target].get(key, 0.0),
                strength,
            )

    # 3. 保留原始信号，方便 Trace、PlanningContext 和后续 Eval 使用。
    result["preference_signals"] = [
        dict(signal)
        for signal in signals
        if isinstance(signal, Mapping)
    ]
    result["constraints"] = _ensure_list(trip_request.get("raw_constraints"))
    result["budget"] = trip_request.get("budget")

    return result


def _current_request_from_legacy_lists(
    trip_request: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从旧版 list 字段构造当前偏好。

    这个方法只为兼容旧测试和历史数据；
    新代码应优先使用 preference_signals。
    """

    result = _empty_current_request()

    result["flight_preferences"] = _list_to_score_map(
        trip_request.get("transport_preferences"),
        score=1.0,
    )
    result["hotel_preferences"] = _list_to_score_map(
        trip_request.get("hotel_preferences"),
        score=1.0,
    )

    legacy_travel_style = _list_to_score_map(
        trip_request.get("travel_style"),
        score=1.0,
    )

    # 1. 旧 travel_style 中有节奏偏好，也有活动偏好。
    for key, value in legacy_travel_style.items():
        if key in {"slow", "low_walking_intensity", "packed_schedule"}:
            result["pace_preferences"][key] = value
        else:
            result["activity_preferences"][key] = value

    result["constraints"] = _ensure_list(trip_request.get("raw_constraints"))
    result["budget"] = trip_request.get("budget")

    return result


def _domain_to_preference_bucket(domain: str) -> str | None:
    """将 PreferenceSignal.domain 映射到 user_profile 字段。"""

    mapping = {
        "flight": "flight_preferences",
        "hotel": "hotel_preferences",
        "activity": "activity_preferences",
        "pace": "pace_preferences",
        "risk": "risk_preferences",
    }

    return mapping.get(domain)


def _build_memory_result(
    user_id: str | None,
    user_found: bool,
    profile_found: bool,
    source: str,
    long_term: dict[str, Any],
    current_request: dict[str, Any],
    recent_trip_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """构造 MemoryReadNode 的统一 user_profile 输出。"""

    # 1. 长期偏好和当前偏好合并成最终有效偏好。
    effective = build_effective_preferences(long_term, current_request)

    # 2. travel_style 作为旧字段兼容视图，合并活动和节奏偏好。
    legacy_travel_style = {
        **effective["activity_preferences"],
        **effective["pace_preferences"],
    }

    return {
        "user_id": user_id,
        "user_found": user_found,
        "profile_found": profile_found,
        "source": source,
        "long_term": long_term,
        "current_request": current_request,
        "effective": effective,
        "recent_trip_history": recent_trip_history,

        # 3. 扁平字段让 PlanningContextBuilder 更容易读取。
        "flight_preferences": effective["flight_preferences"],
        "hotel_preferences": effective["hotel_preferences"],
        "activity_preferences": effective["activity_preferences"],
        "pace_preferences": effective["pace_preferences"],
        "travel_style": legacy_travel_style,
        "diet_preferences": effective["diet_preferences"],
        "budget_preferences": effective["budget_preferences"],
        "risk_preferences": effective["risk_preferences"],
        "preference_signals": current_request.get("preference_signals", []),
        "spend_preferences": current_request.get("spend_preferences", []),
        "named_constraints": current_request.get("named_constraints", []),

        "memory_summary": build_memory_summary(
            source=source,
            profile_found=profile_found,
            recent_trip_history=recent_trip_history,
        ),
    }


def build_effective_preferences(
    long_term: Mapping[str, Any],
    current_request: Mapping[str, Any],
) -> dict[str, Any]:
    """
    合并长期偏好和本次偏好。

    数值偏好使用 max 合并：
        当前请求显式表达的高强度偏好能够覆盖较低的长期权重，
        但不会因为用户本次没有提到某项偏好就丢失长期记忆。
    """

    return {
        "flight_preferences": _merge_score_maps(
            _ensure_dict(long_term.get("flight_preferences")),
            _ensure_dict(current_request.get("flight_preferences")),
        ),
        "hotel_preferences": _merge_score_maps(
            _ensure_dict(long_term.get("hotel_preferences")),
            _ensure_dict(current_request.get("hotel_preferences")),
        ),
        "activity_preferences": _merge_score_maps(
            _ensure_dict(long_term.get("activity_preferences")),
            _ensure_dict(current_request.get("activity_preferences")),
        ),
        "pace_preferences": _merge_score_maps(
            _ensure_dict(long_term.get("pace_preferences")),
            _ensure_dict(current_request.get("pace_preferences")),
        ),
        "risk_preferences": _merge_score_maps(
            _ensure_dict(long_term.get("risk_preferences")),
            _ensure_dict(current_request.get("risk_preferences")),
        ),
        "diet_preferences": _ensure_dict(long_term.get("diet_preferences")),
        "budget_preferences": _merge_budget_preferences(
            _ensure_dict(long_term.get("budget_preferences")),
            current_request.get("budget"),
        ),
    }


def build_memory_summary(
    source: str,
    profile_found: bool,
    recent_trip_history: list[dict[str, Any]],
) -> str:
    """生成一段短记忆摘要，供 Trace 和后续 Prompt 使用。"""

    if not profile_found:
        return "未找到长期用户画像，使用默认偏好。"

    if not recent_trip_history:
        return f"已从 {source} 读取长期偏好，暂无近期旅行历史。"

    latest = recent_trip_history[0]
    return (
        f"已从 {source} 读取长期偏好；最近反馈："
        f"{latest.get('feedback_summary', '')}"
    )


def _merge_score_maps(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, float]:
    """合并两个 0—1 偏好权重 dict，同名项取更大值。"""

    merged: dict[str, float] = {}

    for key, value in base.items():
        merged[str(key)] = _clamp01(_safe_float(value, default=0.0))

    for key, value in override.items():
        normalized_key = str(key)
        merged[normalized_key] = max(
            merged.get(normalized_key, 0.0),
            _clamp01(_safe_float(value, default=1.0)),
        )

    return merged


def _merge_budget_preferences(
    long_term_budget: Mapping[str, Any],
    current_budget: Any,
) -> dict[str, Any]:
    """在长期预算偏好上补充当前旅行总预算。"""

    result = dict(long_term_budget)

    if current_budget is not None:
        result["current_trip_budget"] = current_budget

    return result


def _list_to_score_map(value: Any, score: float) -> dict[str, float]:
    """把旧版字符串列表转换成权重 dict。"""

    if not isinstance(value, list):
        return {}

    return {
        str(item): score
        for item in value
        if isinstance(item, str) and item.strip()
    }


def _trip_history_to_dict(row: TripHistory) -> dict[str, Any]:
    """将 TripHistory ORM 对象转成可序列化 dict。"""

    return {
        "history_id": row.history_id,
        "trip_id": row.trip_id,
        "destination": row.destination,
        "feedback_summary": row.feedback_summary,
        "preference_update": _ensure_dict(row.preference_update_json),
        "tags": _ensure_list(row.tags_json),
        "created_at": _datetime_to_string(row.created_at),
    }


def _ensure_dict(value: Any) -> dict[str, Any]:
    """确保返回普通 dict。"""

    return dict(value) if isinstance(value, Mapping) else {}


def _ensure_list(value: Any) -> list[Any]:
    """确保返回普通 list。"""

    return list(value) if isinstance(value, list) else []


def _ensure_mapping_list(value: Any) -> list[dict[str, Any]]:
    """只保留列表中的 dict-like 项。"""

    if not isinstance(value, list):
        return []

    return [dict(item) for item in value if isinstance(item, Mapping)]


def _safe_float(value: Any, default: float) -> float:
    """安全转换 float。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    """把偏好权重限制在 0 到 1。"""

    return max(0.0, min(1.0, value))


def _datetime_to_string(value: datetime | None) -> str | None:
    """datetime 转 ISO 字符串。"""

    if value is None:
        return None

    return value.isoformat()
