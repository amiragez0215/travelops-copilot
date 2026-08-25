from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from app.schemas.candidate_rank_schema import (
    CandidateRankResult,
    PreferenceGap,
    RankedFlightCandidate,
    RankedHotelCandidate,
    RejectedCandidate,
)


# ----------------------------------------------------------------------
# 用户偏好 key 与 CandidateRank 可计算特征之间的映射
# ----------------------------------------------------------------------
# 只有 Mock 中存在结构化字段的偏好才能进入确定性评分。
# activity、pace 等偏好会在后面的 RAG / Proposal 阶段使用，
# 不应该硬塞进酒店或航班分数。
HOTEL_PREFERENCE_LABELS = {
    "quiet": "安静",
    "cleanliness": "干净",
    "near_subway": "靠近地铁",
    "high_rating": "高评分",
    "budget_friendly": "价格友好",
}

FLIGHT_PREFERENCE_LABELS = {
    "avoid_early_flight": "避免早班机",
    "avoid_late_arrival": "避免太晚抵达",
    "prefer_direct": "偏好直飞",
    "price_sensitive": "价格敏感",
}

# 这些偏好在 preference_match 中有直接结构化特征。
HOTEL_DIRECT_FEATURE_KEYS = {
    "quiet",
    "cleanliness",
    "near_subway",
}

FLIGHT_DIRECT_FEATURE_KEYS = {
    "avoid_early_flight",
    "avoid_late_arrival",
    "prefer_direct",
}

# ----------------------------------------------------------------------
# 当前 v1 旅行产品边界
# ----------------------------------------------------------------------
# TravelOps-Copilot v1 只支持往返旅行：
#
#     outbound：出发地 → 目的地
#     return：目的地 → 出发地
#
# 因此交通预算软目标固定拆成两个航段。
#
# 注意：
#     航段数量必须来自产品需求，而不能根据实际查到的候选数量判断。
#     如果返程查询失败，仍然应该认为需要两个航段，
#     不能把整个往返预算全部分配给去程。
ROUND_TRIP_FLIGHT_LEG_COUNT = 2


@dataclass(frozen=True)
class CandidateRankConfig:
    """
    CandidateRanker 的确定性评分配置。

    这些参数是 v1 初始启发式，不是行业标准。
    它们集中放在一个配置对象中，便于：

        - 单元测试；
        - Eval 对比；
        - 后续统一调参；
        - 避免散落在多个函数里的魔法数字。
    """

    # 酒店总分的基础组件权重。
    hotel_preference_component_weight: float = 0.55
    hotel_price_component_weight: float = 0.25
    hotel_rating_component_weight: float = 0.20

    # 航班总分的基础组件权重。
    flight_preference_component_weight: float = 0.50
    flight_price_component_weight: float = 0.35
    flight_flexibility_component_weight: float = 0.15

    # 用户越重视“价格友好/价格敏感”，价格组件最多额外增加的原始权重。
    hotel_price_weight_bonus: float = 0.10
    hotel_rating_weight_bonus: float = 0.10
    flight_price_weight_bonus: float = 0.15

    # 价格超过软目标时的衰减速度。
    # 公式：1 / (1 + sensitivity × overrun_ratio)
    # sensitivity=2 时，超过目标 50% 的价格适配分为 0.5。
    price_overrun_sensitivity: float = 2.0

    # 重要偏好差距提示阈值。
    # 用户权重 >= 0.80，且候选特征分 < 0.65 时记录 preference_gap。
    preference_gap_weight_threshold: float = 0.80
    preference_gap_feature_threshold: float = 0.65

    # 时间舒适度的分段边界，单位为分钟。
    # 06:00 及更早视为明显早班；09:00 及以后得到满分。
    early_flight_bad_before_minutes: int = 6 * 60
    early_flight_good_after_minutes: int = 9 * 60

    # 20:00 及以前到达得到满分；24:00 及以后降为 0。
    late_arrival_good_before_minutes: int = 20 * 60
    late_arrival_bad_after_minutes: int = 24 * 60

    # preferred 指定酒店/航班在 preference_match 中使用的权重。
    named_preference_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """转成可写入 Trace / Eval 的普通 dict。"""

        return asdict(self)


class CandidateRanker:
    """
    对 Mock 查询得到的航班和酒店做确定性过滤、评分与排序。

    输入：
        planning_context
        raw_flight_results
        raw_hotel_results

    输出：
        flight_candidates
        hotel_candidates
        candidate_rank_result

    核心边界：

        1. 数据层无效候选应当已经由 Provider 过滤。
           本类只做少量防御性检查，例如 room_count 大于 available_rooms。

        2. 真正硬约束用于过滤。
           例如必须直飞、起飞不得早于 08:00、酒店每晚不超过 600。

        3. 主观偏好只用于 0—1 权重评分。
           “一定要安静”会让 quiet 权重接近 1.0，但不会直接过滤酒店。

        4. RAG 不参与航班或酒店分数。
           酒店周边和体验文档只在后续 Proposal 中作为补充说明。
    """

    def __init__(
        self,
        config: CandidateRankConfig | None = None,
    ) -> None:
        """
        Args:
            config:
                排序参数。未传入时使用 CandidateRankConfig 默认值。
        """

        self.config = config or CandidateRankConfig()

    def rank(
        self,
        planning_context: Mapping[str, Any],
        raw_flight_results: Mapping[str, Any],
        raw_hotel_results: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        执行完整候选排序。

        处理步骤：

            1. 读取请求、偏好、预算、硬约束和指定实体。
            2. 将出发航班、返程航班、酒店分别建立候选组。
            3. 先执行真正硬约束过滤。
            4. 对剩余候选计算确定性特征分。
            5. 计算总分并稳定排序。
            6. 生成拒绝原因、偏好差距和执行摘要。
        """

        if not isinstance(planning_context, Mapping):
            raise TypeError("planning_context 必须是 dict-like 对象")

        if not isinstance(raw_flight_results, Mapping):
            raise TypeError("raw_flight_results 必须是 dict-like 对象")

        if not isinstance(raw_hotel_results, Mapping):
            raise TypeError("raw_hotel_results 必须是 dict-like 对象")

        request = _read_mapping(planning_context.get("request"))
        preference_weights = _read_preference_weights(planning_context)
        hard_constraints = _read_grouped_constraints(planning_context)
        budget_plan = _read_mapping(planning_context.get("budget_plan"))
        named_constraints = _read_mapping_list(
            planning_context.get("named_constraints")
        )

        people_count = max(
            _safe_int(request.get("people_count"), default=1),
            1,
        )
        room_count = max(
            _safe_int(request.get("room_count"), default=1),
            1,
        )
        nights = max(
            _safe_int(request.get("nights"), default=0),
            0,
        )

        outbound = _read_candidate_list(
            raw_flight_results.get("outbound")
        )
        return_flights = _read_candidate_list(
            raw_flight_results.get("return")
        )
        hotels = _read_candidate_list(
            raw_hotel_results.get("items")
        )

        # 1. transport_budget 表示整个往返交通的软目标。
        #
        #    当前 v1 固定支持往返旅行，因此需要平均拆成：
        #
        #        去程参考目标
        #        返程参考目标
        #
        #    这里不能根据 outbound / return_flights 是否为空来判断航段数。
        #
        #    原因：
        #        候选列表为空只表示“工具没有查到结果”，
        #        不表示“用户不需要这个航段”。
        #
        #    例如往返交通软目标为 1500 元：
        #
        #        去程参考目标 = 750
        #        返程参考目标 = 750
        #
        #    这仍然只是评分参考，不是单航段硬上限。
        transport_target = _safe_optional_float(
            _read_mapping(budget_plan.get("soft_targets")).get(
                "transport_budget"
            )
        )

        per_leg_transport_target = (
            transport_target
            / ROUND_TRIP_FLIGHT_LEG_COUNT
            if transport_target is not None
            else None
        )

        hotel_target = _safe_optional_float(
            _read_mapping(budget_plan.get("soft_targets")).get(
                "target_hotel_price_per_room_night"
            )
        )

        # 2. 分别计算三组候选的价格范围。
        #    没有明确预算目标、但用户价格敏感时，
        #    可以使用组内相对价格做 affordability score。
        outbound_price_range = _price_range(outbound, field="price")
        return_price_range = _price_range(return_flights, field="price")
        hotel_price_range = _price_range(hotels, field="price_per_night")

        rejected_flights: list[RejectedCandidate] = []
        rejected_hotels: list[RejectedCandidate] = []

        ranked_outbound = self._rank_flight_group(
            candidates=outbound,
            direction="outbound",
            people_count=people_count,
            target_total_price=per_leg_transport_target,
            group_price_range=outbound_price_range,
            preference_weights=preference_weights.get("flight", {}),
            hard_constraints=hard_constraints.get("flight", []),
            named_constraints=named_constraints,
            rejected=rejected_flights,
        )

        ranked_return = self._rank_flight_group(
            candidates=return_flights,
            direction="return",
            people_count=people_count,
            target_total_price=per_leg_transport_target,
            group_price_range=return_price_range,
            preference_weights=preference_weights.get("flight", {}),
            hard_constraints=hard_constraints.get("flight", []),
            named_constraints=named_constraints,
            rejected=rejected_flights,
        )

        ranked_hotels = self._rank_hotels(
            candidates=hotels,
            room_count=room_count,
            nights=nights,
            target_price_per_room_night=hotel_target,
            group_price_range=hotel_price_range,
            preference_weights=preference_weights.get("hotel", {}),
            hard_constraints=hard_constraints.get("hotel", []),
            named_constraints=named_constraints,
            rejected=rejected_hotels,
        )

        flight_candidates = [
            *ranked_outbound,
            *ranked_return,
        ]

        # 3. 当前 v1 固定只支持往返旅行，因此返程始终是必需项。
        #
        #    不能依赖 raw_flight_results.return_date 判断，
        #    因为 return_date 属于 Provider 输出字段。
        #    即使 Provider 因异常漏掉该字段，产品需求仍然是往返旅行。
        return_required = True

        status, issues = _resolve_rank_status(
            return_required=return_required,
            input_outbound_count=len(outbound),
            input_return_count=len(return_flights),
            input_hotel_count=len(hotels),
            output_outbound_count=len(ranked_outbound),
            output_return_count=len(ranked_return),
            output_hotel_count=len(ranked_hotels),
        )

        ignored_preference_keys = {
            "flight": sorted(
                set(preference_weights.get("flight", {}))
                - FLIGHT_DIRECT_FEATURE_KEYS
                - {"price_sensitive"}
            ),
            "hotel": sorted(
                set(preference_weights.get("hotel", {}))
                - HOTEL_DIRECT_FEATURE_KEYS
                - {"high_rating", "budget_friendly"}
            ),
        }

        summary = CandidateRankResult(
            status=status,
            input_counts={
                "outbound_flights": len(outbound),
                "return_flights": len(return_flights),
                "hotels": len(hotels),
            },
            output_counts={
                "outbound_flights": len(ranked_outbound),
                "return_flights": len(ranked_return),
                "hotels": len(ranked_hotels),
            },
            rejected_flights=rejected_flights,
            rejected_hotels=rejected_hotels,
            ignored_preference_keys={
                domain: keys
                for domain, keys in ignored_preference_keys.items()
                if keys
            },
            ranking_config={
                "hotel_components": {
                    "preference_match": self.config.hotel_preference_component_weight,
                    "price_fit": self.config.hotel_price_component_weight,
                    "base_rating": self.config.hotel_rating_component_weight,
                },
                "flight_components": {
                    "preference_match": self.config.flight_preference_component_weight,
                    "price_fit": self.config.flight_price_component_weight,
                    "flexibility": self.config.flight_flexibility_component_weight,
                },
                "price_overrun_sensitivity": self.config.price_overrun_sensitivity,
                # 当前版本固定采用往返旅行，并把交通软目标平均分成两段。
                "trip_type":
                    "round_trip",

                "expected_flight_leg_count":
                    ROUND_TRIP_FLIGHT_LEG_COUNT,

                "per_leg_target_policy":"equal_split_round_trip_v1",
                "per_leg_transport_target": (
                    round(per_leg_transport_target, 2)
                    if per_leg_transport_target is not None
                    else None
                ),
                "hotel_price_target_per_room_night": hotel_target,
                "return_required": return_required,
            },
            issues=issues,
        )

        return {
            "flight_candidates": [
                item.to_state_dict()
                for item in flight_candidates
            ],
            "hotel_candidates": [
                item.to_state_dict()
                for item in ranked_hotels
            ],
            "candidate_rank_result": summary.to_state_dict(),
        }

    def _rank_flight_group(
        self,
        *,
        candidates: list[dict[str, Any]],
        direction: str,
        people_count: int,
        target_total_price: float | None,
        group_price_range: tuple[float, float] | None,
        preference_weights: Mapping[str, float],
        hard_constraints: list[dict[str, Any]],
        named_constraints: list[dict[str, Any]],
        rejected: list[RejectedCandidate],
    ) -> list[RankedFlightCandidate]:
        """
        对 outbound 或 return 航班独立过滤和排序。

        为什么分开排序？
            出发航班和返程航班不能放在同一排名中比较。
            BudgetOptimizeNode 后面会从两个独立候选池中各选一个。
        """

        scored: list[dict[str, Any]] = []

        price_sensitive = _clamp01(
            _safe_float(
                preference_weights.get("price_sensitive"),
                default=0.0,
            )
        )

        component_weights = self._flight_component_weights(
            price_sensitive=price_sensitive
        )

        for candidate in candidates:
            flight_id = str(candidate.get("flight_id") or "").strip()

            # 1. Provider 理论上已经过滤缺失 flight_id 的数据；
            #    这里再做一次防御性拒绝，避免坏 State 进入评分。
            if not flight_id:
                rejected.append(
                    RejectedCandidate(
                        candidate_type="flight",
                        candidate_id="unknown",
                        direction=direction,  # type: ignore[arg-type]
                        reason_code="missing_candidate_id",
                        message="航班缺少 flight_id，无法进入排序。",
                    )
                )
                continue

            passed, rejection = _check_hard_constraints(
                candidate=candidate,
                constraints=hard_constraints,
                candidate_type="flight",
                candidate_id=flight_id,
                direction=direction,
            )

            if not passed:
                rejected.append(rejection)
                continue

            named_passed, named_rejection = _check_named_constraints(
                candidate=candidate,
                candidate_type="flight",
                candidate_id=flight_id,
                direction=direction,
                named_constraints=named_constraints,
            )

            if not named_passed:
                rejected.append(named_rejection)
                continue

            total_price = round(
                _safe_float(candidate.get("price"), 0.0)
                * people_count,
                2,
            )

            # 2. 将原始时间、直飞和退改字段转成统一 0—1 特征。
            early_flight_fit = _early_departure_fit(
                candidate.get("depart_time"),
                config=self.config,
            )
            late_arrival_fit = _late_arrival_fit(
                candidate.get("arrive_time"),
                config=self.config,
            )
            direct_fit = 1.0 if bool(candidate.get("is_direct")) else 0.0
            flexibility_fit = _flight_flexibility_fit(candidate)

            feature_scores = {
                "avoid_early_flight": early_flight_fit,
                "avoid_late_arrival": late_arrival_fit,
                "prefer_direct": direct_fit,
            }

            # 3. preferred 指定航班属于偏好，不属于硬过滤。
            named_preference_score, has_named_preference = (
                _named_preference_score(
                    candidate=candidate,
                    candidate_type="flight",
                    named_constraints=named_constraints,
                )
            )

            preference_match = _weighted_preference_score(
                preference_weights=preference_weights,
                feature_scores=feature_scores,
                supported_keys=FLIGHT_DIRECT_FEATURE_KEYS,
                named_preference_score=(
                    named_preference_score
                    if has_named_preference
                    else None
                ),
                named_preference_weight=self.config.named_preference_weight,
            )

            # 4. 有软预算目标时，价格低于目标得到满分；
            #    超过目标时平滑衰减，但不会在这里直接过滤。
            price_fit = _price_fit_score(
                price=total_price,
                target=target_total_price,
                relative_price_range=(
                    _scale_price_range(
                        group_price_range,
                        multiplier=people_count,
                    )
                ),
                use_relative_when_no_target=(price_sensitive > 0),
                sensitivity=self.config.price_overrun_sensitivity,
            )

            total_score = _weighted_sum(
                {
                    "preference_match": preference_match,
                    "price_fit": price_fit,
                    "flexibility": flexibility_fit,
                },
                component_weights,
            )

            time_fit = _weighted_preference_score(
                preference_weights=preference_weights,
                feature_scores={
                    "avoid_early_flight": early_flight_fit,
                    "avoid_late_arrival": late_arrival_fit,
                },
                supported_keys={
                    "avoid_early_flight",
                    "avoid_late_arrival",
                },
                neutral_score=0.5,
            )

            gaps = _build_preference_gaps(
                preference_weights=preference_weights,
                feature_scores=feature_scores,
                labels=FLIGHT_PREFERENCE_LABELS,
                supported_keys=FLIGHT_DIRECT_FEATURE_KEYS,
                config=self.config,
            )

            reasons = _build_flight_reasons(
                preference_weights=preference_weights,
                early_flight_fit=early_flight_fit,
                late_arrival_fit=late_arrival_fit,
                direct_fit=direct_fit,
                price_fit=price_fit,
                flexibility_fit=flexibility_fit,
                target_total_price=target_total_price,
                total_price=total_price,
                named_preference_matched=(
                    has_named_preference
                    and named_preference_score >= 1.0
                ),
            )

            scored.append(
                {
                    "candidate": dict(candidate),
                    "direction": direction,
                    "total_price": total_price,
                    "total_score": total_score,
                    "score_breakdown": {
                        "preference_match": round(preference_match, 6),
                        "price_fit": round(price_fit, 6),
                        "flexibility": round(flexibility_fit, 6),
                        "time_fit": round(time_fit, 6),
                        "direct_fit": round(direct_fit, 6),
                        "feature_scores": {
                            key: round(value, 6)
                            for key, value in feature_scores.items()
                        },
                        "component_weights": {
                            key: round(value, 6)
                            for key, value in component_weights.items()
                        },
                    },
                    "ranking_reasons": reasons,
                    "preference_gaps": gaps,
                    "hard_constraints_checked": [
                        dict(item)
                        for item in hard_constraints
                    ],
                }
            )

        # 5. 先按综合分降序，再按总票价、起飞时间和 ID 稳定排序。
        scored.sort(
            key=lambda item: (
                -item["total_score"],
                item["total_price"],
                _time_sort_value(
                    item["candidate"].get("depart_time")
                ),
                str(item["candidate"].get("flight_id") or ""),
            )
        )

        output: list[RankedFlightCandidate] = []

        # 6. 写入组内 rank，并恢复完整航班结构。
        for rank, item in enumerate(scored, start=1):
            candidate = item["candidate"]

            output.append(
                RankedFlightCandidate(
                    flight_id=str(candidate["flight_id"]),
                    flight_no=_optional_text(candidate.get("flight_no")),
                    airline=_optional_text(candidate.get("airline")),
                    departure_city=_optional_text(candidate.get("departure_city")),
                    arrival_city=_optional_text(candidate.get("arrival_city")),
                    departure_airport=_optional_text(candidate.get("departure_airport")),
                    arrival_airport=_optional_text(candidate.get("arrival_airport")),
                    depart_date=_optional_text(candidate.get("depart_date")),
                    depart_time=_optional_text(candidate.get("depart_time")),
                    arrive_date=_optional_text(candidate.get("arrive_date")),
                    arrive_time=_optional_text(candidate.get("arrive_time")),
                    duration_minutes=max(
                        _safe_int(candidate.get("duration_minutes"), 0),
                        0,
                    ),
                    price=max(_safe_float(candidate.get("price"), 0.0), 0.0),
                    total_price=item["total_price"],
                    available_seats=max(
                        _safe_int(candidate.get("available_seats"), 0),
                        0,
                    ),
                    cabin=_optional_text(candidate.get("cabin")),
                    baggage=_optional_text(candidate.get("baggage")),
                    is_direct=bool(candidate.get("is_direct")),
                    refundable=bool(candidate.get("refundable")),
                    changeable=bool(candidate.get("changeable")),
                    updated_at=_optional_text(candidate.get("updated_at")),
                    source=_optional_text(candidate.get("source")),
                    query_direction=direction,  # type: ignore[arg-type]
                    direction=direction,  # type: ignore[arg-type]
                    rank=rank,
                    total_score=round(item["total_score"], 6),
                    score_breakdown=item["score_breakdown"],
                    ranking_reasons=item["ranking_reasons"],
                    preference_gaps=item["preference_gaps"],
                    hard_constraints_checked=item["hard_constraints_checked"],
                )
            )

        return output

    def _rank_hotels(
        self,
        *,
        candidates: list[dict[str, Any]],
        room_count: int,
        nights: int,
        target_price_per_room_night: float | None,
        group_price_range: tuple[float, float] | None,
        preference_weights: Mapping[str, float],
        hard_constraints: list[dict[str, Any]],
        named_constraints: list[dict[str, Any]],
        rejected: list[RejectedCandidate],
    ) -> list[RankedHotelCandidate]:
        """对酒店候选执行硬过滤、评分和排序。"""

        scored: list[dict[str, Any]] = []

        budget_friendly_weight = _clamp01(
            _safe_float(
                preference_weights.get("budget_friendly"),
                default=0.0,
            )
        )
        high_rating_weight = _clamp01(
            _safe_float(
                preference_weights.get("high_rating"),
                default=0.0,
            )
        )

        component_weights = self._hotel_component_weights(
            budget_friendly_weight=budget_friendly_weight,
            high_rating_weight=high_rating_weight,
        )

        for candidate in candidates:
            hotel_id = str(candidate.get("hotel_id") or "").strip()

            # 1. 防御性检查 hotel_id。
            if not hotel_id:
                rejected.append(
                    RejectedCandidate(
                        candidate_type="hotel",
                        candidate_id="unknown",
                        reason_code="missing_candidate_id",
                        message="酒店缺少 hotel_id，无法进入排序。",
                    )
                )
                continue

            # 2. Provider 当前只过滤 available_rooms > 0。
            #    CandidateRank 再检查是否足够覆盖本次 room_count。
            available_rooms = max(
                _safe_int(candidate.get("available_rooms"), default=0),
                0,
            )

            if available_rooms < room_count:
                rejected.append(
                    RejectedCandidate(
                        candidate_type="hotel",
                        candidate_id=hotel_id,
                        reason_code="insufficient_rooms",
                        message=(
                            f"酒店可用房间数为 {available_rooms}，"
                            f"少于本次需要的 {room_count} 间。"
                        ),
                    )
                )
                continue

            passed, rejection = _check_hard_constraints(
                candidate=candidate,
                constraints=hard_constraints,
                candidate_type="hotel",
                candidate_id=hotel_id,
                direction=None,
            )

            if not passed:
                rejected.append(rejection)
                continue

            named_passed, named_rejection = _check_named_constraints(
                candidate=candidate,
                candidate_type="hotel",
                candidate_id=hotel_id,
                direction=None,
                named_constraints=named_constraints,
            )

            if not named_passed:
                rejected.append(named_rejection)
                continue

            price_per_night = max(
                _safe_float(candidate.get("price_per_night"), 0.0),
                0.0,
            )

            estimated_total_price = round(
                price_per_night * nights * room_count,
                2,
            )

            # 3. 将 Mock 结构化字段统一转成 0—1 特征。
            feature_scores = {
                "quiet": _clamp01(
                    _safe_float(candidate.get("quiet_score"), 0.5)
                ),
                "cleanliness": _clamp01(
                    _safe_float(candidate.get("cleanliness_score"), 0.5)
                ),
                "near_subway": (
                    1.0 if bool(candidate.get("near_subway")) else 0.0
                ),
            }

            named_preference_score, has_named_preference = (
                _named_preference_score(
                    candidate=candidate,
                    candidate_type="hotel",
                    named_constraints=named_constraints,
                )
            )

            preference_match = _weighted_preference_score(
                preference_weights=preference_weights,
                feature_scores=feature_scores,
                supported_keys=HOTEL_DIRECT_FEATURE_KEYS,
                named_preference_score=(
                    named_preference_score
                    if has_named_preference
                    else None
                ),
                named_preference_weight=self.config.named_preference_weight,
            )

            price_fit = _price_fit_score(
                price=price_per_night,
                target=target_price_per_room_night,
                relative_price_range=group_price_range,
                use_relative_when_no_target=(budget_friendly_weight > 0),
                sensitivity=self.config.price_overrun_sensitivity,
            )

            base_rating = _clamp01(
                _safe_float(candidate.get("rating"), 0.0) / 5.0
            )

            # 4. 酒店总分严格只使用：偏好匹配 + 价格适配 + 基础评分。
            total_score = _weighted_sum(
                {
                    "preference_match": preference_match,
                    "price_fit": price_fit,
                    "base_rating": base_rating,
                },
                component_weights,
            )

            gaps = _build_preference_gaps(
                preference_weights=preference_weights,
                feature_scores=feature_scores,
                labels=HOTEL_PREFERENCE_LABELS,
                supported_keys=HOTEL_DIRECT_FEATURE_KEYS,
                config=self.config,
            )

            reasons = _build_hotel_reasons(
                preference_weights=preference_weights,
                feature_scores=feature_scores,
                price_fit=price_fit,
                base_rating=base_rating,
                target_price=target_price_per_room_night,
                actual_price=price_per_night,
                named_preference_matched=(
                    has_named_preference
                    and named_preference_score >= 1.0
                ),
            )

            scored.append(
                {
                    "candidate": dict(candidate),
                    "estimated_total_price": estimated_total_price,
                    "total_score": total_score,
                    "score_breakdown": {
                        "preference_match": round(preference_match, 6),
                        "price_fit": round(price_fit, 6),
                        "base_rating": round(base_rating, 6),
                        "feature_scores": {
                            key: round(value, 6)
                            for key, value in feature_scores.items()
                        },
                        "component_weights": {
                            key: round(value, 6)
                            for key, value in component_weights.items()
                        },
                    },
                    "ranking_reasons": reasons,
                    "preference_gaps": gaps,
                    "hard_constraints_checked": [
                        dict(item)
                        for item in hard_constraints
                    ],
                }
            )

        # 5. 总分相同时优先价格更低、基础评分更高的酒店。
        scored.sort(
            key=lambda item: (
                -item["total_score"],
                _safe_float(
                    item["candidate"].get("price_per_night"),
                    0.0,
                ),
                -_safe_float(
                    item["candidate"].get("rating"),
                    0.0,
                ),
                str(item["candidate"].get("hotel_id") or ""),
            )
        )

        output: list[RankedHotelCandidate] = []

        # 6. 写入最终 rank，并恢复完整酒店结构。
        for rank, item in enumerate(scored, start=1):
            candidate = item["candidate"]

            output.append(
                RankedHotelCandidate(
                    hotel_id=str(candidate["hotel_id"]),
                    name=_optional_text(candidate.get("name")),
                    city=_optional_text(candidate.get("city")),
                    district=_optional_text(candidate.get("district")),
                    address=_optional_text(candidate.get("address")),
                    price_per_night=max(
                        _safe_float(candidate.get("price_per_night"), 0.0),
                        0.0,
                    ),
                    estimated_total_price=item["estimated_total_price"],
                    planned_nights=nights,
                    planned_room_count=room_count,
                    rating=max(
                        _safe_float(candidate.get("rating"), 0.0),
                        0.0,
                    ),
                    available_rooms=max(
                        _safe_int(candidate.get("available_rooms"), 0),
                        0,
                    ),
                    near_subway=bool(candidate.get("near_subway")),
                    distance_to_subway_meters=max(
                        _safe_int(
                            candidate.get("distance_to_subway_meters"),
                            999999,
                        ),
                        0,
                    ),
                    quiet_score=_clamp01(
                        _safe_float(candidate.get("quiet_score"), 0.5)
                    ),
                    cleanliness_score=_clamp01(
                        _safe_float(candidate.get("cleanliness_score"), 0.5)
                    ),
                    tags=[
                        str(value)
                        for value in candidate.get("tags", [])
                    ],
                    amenities=[
                        str(value)
                        for value in candidate.get("amenities", [])
                    ],
                    cancel_policy=_optional_text(candidate.get("cancel_policy")),
                    updated_at=_optional_text(candidate.get("updated_at")),
                    source=_optional_text(candidate.get("source")),
                    rank=rank,
                    total_score=round(item["total_score"], 6),
                    score_breakdown=item["score_breakdown"],
                    ranking_reasons=item["ranking_reasons"],
                    preference_gaps=item["preference_gaps"],
                    hard_constraints_checked=item["hard_constraints_checked"],
                )
            )

        return output

    def _hotel_component_weights(
        self,
        *,
        budget_friendly_weight: float,
        high_rating_weight: float,
    ) -> dict[str, float]:
        """
        根据价格友好和高评分偏好，轻量调整酒店三类组件权重。

        注意：
            这不是新增第四类得分。
            最终仍然只有 preference_match、price_fit、base_rating 三部分。
        """

        raw = {
            "preference_match": self.config.hotel_preference_component_weight,
            "price_fit": (
                self.config.hotel_price_component_weight
                + self.config.hotel_price_weight_bonus * budget_friendly_weight
            ),
            "base_rating": (
                self.config.hotel_rating_component_weight
                + self.config.hotel_rating_weight_bonus * high_rating_weight
            ),
        }

        return _normalize_weights(raw)

    def _flight_component_weights(
        self,
        *,
        price_sensitive: float,
    ) -> dict[str, float]:
        """根据价格敏感度轻量提高航班价格组件权重。"""

        raw = {
            "preference_match": self.config.flight_preference_component_weight,
            "price_fit": (
                self.config.flight_price_component_weight
                + self.config.flight_price_weight_bonus * price_sensitive
            ),
            "flexibility": self.config.flight_flexibility_component_weight,
        }

        return _normalize_weights(raw)


# ======================================================================
# 硬约束与指定实体
# ======================================================================


def _check_hard_constraints(
    *,
    candidate: Mapping[str, Any],
    constraints: Sequence[Mapping[str, Any]],
    candidate_type: str,
    candidate_id: str,
    direction: str | None,
) -> tuple[bool, RejectedCandidate]:
    """
    检查一个候选是否满足该领域全部真正硬约束。

    返回：
        (True, dummy_rejection)
            全部通过。

        (False, rejection)
            至少一条硬约束不满足。
    """

    for constraint in constraints:
        field = str(constraint.get("field") or "")
        operator = str(constraint.get("operator") or "")
        expected = constraint.get("value")
        actual = candidate.get(field)

        if not _constraint_matches(
            field=field,
            actual=actual,
            operator=operator,
            expected=expected,
        ):
            evidence = str(constraint.get("evidence") or "").strip()
            evidence_text = f" 用户原话：{evidence}" if evidence else ""

            return False, RejectedCandidate(
                candidate_type=candidate_type,  # type: ignore[arg-type]
                candidate_id=candidate_id,
                direction=direction,  # type: ignore[arg-type]
                reason_code="hard_constraint_violation",
                message=(
                    f"候选不满足硬约束：{field} {operator} {expected}。"
                    f" 实际值：{actual}。{evidence_text}"
                ),
                constraint=dict(constraint),
            )

    # 调用方只会在 passed=False 时使用 rejection；
    # 这里构造一个不会写入输出的占位对象，保持返回类型简单。
    return True, RejectedCandidate(
        candidate_type=candidate_type,  # type: ignore[arg-type]
        candidate_id=candidate_id,
        direction=direction,  # type: ignore[arg-type]
        reason_code="passed",
        message="全部硬约束通过。",
    )


def _check_named_constraints(
    *,
    candidate: Mapping[str, Any],
    candidate_type: str,
    candidate_id: str,
    direction: str | None,
    named_constraints: Sequence[Mapping[str, Any]],
) -> tuple[bool, RejectedCandidate]:
    """
    检查 required / avoid 指定实体。

    preferred 指定实体不在这里过滤，后面作为 preference_match 的一部分加分。
    """

    relevant = [
        item
        for item in named_constraints
        if str(item.get("entity_type") or "") == candidate_type
    ]

    required_names = {
        _normalize_name(item.get("entity_name"))
        for item in relevant
        if item.get("constraint_mode") == "required"
        and _normalize_name(item.get("entity_name"))
    }

    avoided_names = {
        _normalize_name(item.get("entity_name"))
        for item in relevant
        if item.get("constraint_mode") == "avoid"
        and _normalize_name(item.get("entity_name"))
    }

    candidate_names = _candidate_names(
        candidate=candidate,
        candidate_type=candidate_type,
    )

    # 1. 存在 required 实体时，候选必须匹配其中一个。
    #    去重后的多个 required 名称在 v1 中视为允许集合；
    #    上游 Validator 已经对多家必住酒店做冲突检查。
    if required_names and not (candidate_names & required_names):
        return False, RejectedCandidate(
            candidate_type=candidate_type,  # type: ignore[arg-type]
            candidate_id=candidate_id,
            direction=direction,  # type: ignore[arg-type]
            reason_code="required_named_entity_mismatch",
            message=(
                f"候选不属于用户指定的必选实体："
                f"{', '.join(sorted(required_names))}。"
            ),
            constraint={
                "entity_type": candidate_type,
                "constraint_mode": "required",
                "entity_names": sorted(required_names),
            },
        )

    # 2. 候选命中 avoid 实体时直接拒绝。
    if avoided_names and (candidate_names & avoided_names):
        matched = sorted(candidate_names & avoided_names)

        return False, RejectedCandidate(
            candidate_type=candidate_type,  # type: ignore[arg-type]
            candidate_id=candidate_id,
            direction=direction,  # type: ignore[arg-type]
            reason_code="avoided_named_entity",
            message=f"候选命中用户希望避开的实体：{', '.join(matched)}。",
            constraint={
                "entity_type": candidate_type,
                "constraint_mode": "avoid",
                "entity_names": matched,
            },
        )

    return True, RejectedCandidate(
        candidate_type=candidate_type,  # type: ignore[arg-type]
        candidate_id=candidate_id,
        direction=direction,  # type: ignore[arg-type]
        reason_code="passed",
        message="指定实体约束通过。",
    )


def _constraint_matches(
    *,
    field: str,
    actual: Any,
    operator: str,
    expected: Any,
) -> bool:
    """执行一条白名单硬约束比较。"""

    if actual is None:
        return False

    # 1. 时间字段统一转成分钟比较，避免直接比较格式不统一的字符串。
    if field in {"depart_time", "arrive_time"}:
        actual_value = _parse_time_minutes(actual)
        expected_value = _parse_time_minutes(expected)

        if actual_value is None or expected_value is None:
            return False

    # 2. 价格字段统一转 float。
    elif field in {"price", "price_per_night", "total_budget"}:
        actual_value = _safe_optional_float(actual)
        expected_value = _safe_optional_float(expected)

        if actual_value is None or expected_value is None:
            return False

    # 3. 布尔字段统一解释常见真值/假值。
    elif field in {"is_direct", "near_subway"}:
        actual_value = _coerce_bool(actual)
        expected_value = _coerce_bool(expected)

        if actual_value is None or expected_value is None:
            return False

    # 4. ID、航班号等字符串忽略空格和大小写。
    else:
        actual_value = _normalize_name(actual)
        expected_value = _normalize_name(expected)

    if operator == "==":
        return actual_value == expected_value
    if operator == "!=":
        return actual_value != expected_value
    if operator == "<":
        return actual_value < expected_value
    if operator == "<=":
        return actual_value <= expected_value
    if operator == ">":
        return actual_value > expected_value
    if operator == ">=":
        return actual_value >= expected_value
    if operator == "in":
        return actual_value in _normalized_collection(expected)
    if operator == "not_in":
        return actual_value not in _normalized_collection(expected)

    return False


# ======================================================================
# 偏好与评分
# ======================================================================


def _weighted_preference_score(
    *,
    preference_weights: Mapping[str, float],
    feature_scores: Mapping[str, float],
    supported_keys: set[str],
    named_preference_score: float | None = None,
    named_preference_weight: float = 1.0,
    neutral_score: float = 0.5,
) -> float:
    """
    计算 0—1 加权偏好匹配分。

    公式：

        Σ(用户偏好权重 × 候选特征分)
        --------------------------------
                 Σ用户偏好权重

    没有可计算偏好时返回 neutral_score，
    避免“没有偏好”被错误解释为 0 分。
    """

    weighted_sum = 0.0
    total_weight = 0.0

    for key in supported_keys:
        weight = _clamp01(
            _safe_float(preference_weights.get(key), default=0.0)
        )

        if weight <= 0:
            continue

        feature_score = _clamp01(
            _safe_float(feature_scores.get(key), default=0.5)
        )

        weighted_sum += weight * feature_score
        total_weight += weight

    if named_preference_score is not None:
        weighted_sum += (
            named_preference_weight
            * _clamp01(named_preference_score)
        )
        total_weight += named_preference_weight

    if total_weight <= 0:
        return _clamp01(neutral_score)

    return _clamp01(weighted_sum / total_weight)


def _price_fit_score(
    *,
    price: float,
    target: float | None,
    relative_price_range: tuple[float, float] | None,
    use_relative_when_no_target: bool,
    sensitivity: float,
) -> float:
    """
    计算价格适配分。

    有软目标：
        price <= target
            → 1.0

        price > target
            → 1 / (1 + sensitivity × 超支比例)

    没有软目标：
        用户价格敏感
            → 使用候选组内相对价格，越便宜越高。

        用户不价格敏感
            → 返回中性 0.5。

    重要：
        这是软评分，不会在这里过滤超过目标的候选。
    """

    price = max(float(price), 0.0)

    if target is not None:
        target = max(float(target), 0.0)

        if target == 0:
            return 1.0 if price == 0 else 0.0

        if price <= target:
            return 1.0

        overrun_ratio = (price - target) / target
        return _clamp01(
            1.0 / (1.0 + sensitivity * overrun_ratio)
        )

    if use_relative_when_no_target and relative_price_range is not None:
        minimum, maximum = relative_price_range

        if maximum <= minimum:
            return 1.0

        return _clamp01(
            1.0 - ((price - minimum) / (maximum - minimum))
        )

    return 0.5


def _weighted_sum(
    scores: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """使用已归一化组件权重计算最终总分。"""

    return _clamp01(
        sum(
            _clamp01(_safe_float(scores.get(key), 0.0))
            * weight
            for key, weight in weights.items()
        )
    )


def _normalize_weights(
    raw_weights: Mapping[str, float],
) -> dict[str, float]:
    """把非负组件权重归一化为总和 1。"""

    cleaned = {
        key: max(float(value), 0.0)
        for key, value in raw_weights.items()
    }

    total = sum(cleaned.values())

    if total <= 0:
        raise ValueError("评分组件权重总和必须大于 0")

    return {
        key: value / total
        for key, value in cleaned.items()
    }


def _early_departure_fit(
    value: Any,
    *,
    config: CandidateRankConfig,
) -> float:
    """
    计算“避免早班机”的满足度。

    06:00 及更早为 0；09:00 及以后为 1；中间线性增长。
    """

    minutes = _parse_time_minutes(value)

    if minutes is None:
        return 0.5

    return _linear_fit(
        value=minutes,
        bad=config.early_flight_bad_before_minutes,
        good=config.early_flight_good_after_minutes,
        higher_is_better=True,
    )


def _late_arrival_fit(
    value: Any,
    *,
    config: CandidateRankConfig,
) -> float:
    """
    计算“避免太晚抵达”的满足度。

    20:00 及以前为 1；24:00 及以后为 0；中间线性下降。
    00:00—04:00 会被解释成次日凌晨，并加上 24 小时再比较。
    """

    minutes = _parse_time_minutes(value)

    if minutes is None:
        return 0.5

    if minutes <= 4 * 60:
        minutes += 24 * 60

    return _linear_fit(
        value=minutes,
        bad=config.late_arrival_bad_after_minutes,
        good=config.late_arrival_good_before_minutes,
        higher_is_better=False,
    )


def _linear_fit(
    *,
    value: float,
    bad: float,
    good: float,
    higher_is_better: bool,
) -> float:
    """在 bad 与 good 之间做线性 0—1 映射。"""

    if higher_is_better:
        if value <= bad:
            return 0.0
        if value >= good:
            return 1.0
        return _clamp01((value - bad) / (good - bad))

    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return _clamp01((bad - value) / (bad - good))


def _flight_flexibility_fit(
    candidate: Mapping[str, Any],
) -> float:
    """
    计算航班退改便利分。

    refundable 和 changeable 各占一半：

        可退 + 可改 → 1.0
        只可退或只可改 → 0.5
        都不支持 → 0.0
    """

    return (
        (0.5 if bool(candidate.get("refundable")) else 0.0)
        + (0.5 if bool(candidate.get("changeable")) else 0.0)
    )


def _named_preference_score(
    *,
    candidate: Mapping[str, Any],
    candidate_type: str,
    named_constraints: Sequence[Mapping[str, Any]],
) -> tuple[float, bool]:
    """计算 preferred 指定酒店或航班是否命中。"""

    preferred_names = {
        _normalize_name(item.get("entity_name"))
        for item in named_constraints
        if str(item.get("entity_type") or "") == candidate_type
        and item.get("constraint_mode") == "preferred"
        and _normalize_name(item.get("entity_name"))
    }

    if not preferred_names:
        return 0.0, False

    candidate_names = _candidate_names(
        candidate=candidate,
        candidate_type=candidate_type,
    )

    return (
        1.0 if candidate_names & preferred_names else 0.0,
        True,
    )


def _build_preference_gaps(
    *,
    preference_weights: Mapping[str, float],
    feature_scores: Mapping[str, float],
    labels: Mapping[str, str],
    supported_keys: set[str],
    config: CandidateRankConfig,
) -> list[PreferenceGap]:
    """为高权重但低满足度的主观偏好生成透明提示。"""

    gaps: list[PreferenceGap] = []

    for key in sorted(supported_keys):
        weight = _clamp01(
            _safe_float(preference_weights.get(key), default=0.0)
        )
        feature_score = _clamp01(
            _safe_float(feature_scores.get(key), default=0.5)
        )

        if weight < config.preference_gap_weight_threshold:
            continue

        if feature_score >= config.preference_gap_feature_threshold:
            continue

        label = labels.get(key, key)

        gaps.append(
            PreferenceGap(
                preference_key=key,
                preference_label=label,
                user_weight=round(weight, 6),
                candidate_feature_score=round(feature_score, 6),
                message=(
                    f"用户非常重视“{label}”，但当前候选在该特征上的"
                    f"得分只有 {feature_score:.2f}。"
                ),
            )
        )

    return gaps


# ======================================================================
# 可解释排序理由
# ======================================================================


def _build_hotel_reasons(
    *,
    preference_weights: Mapping[str, float],
    feature_scores: Mapping[str, float],
    price_fit: float,
    base_rating: float,
    target_price: float | None,
    actual_price: float,
    named_preference_matched: bool,
) -> list[str]:
    """生成酒店排序理由，最多保留四条。"""

    reasons: list[tuple[float, str]] = []

    if named_preference_matched:
        reasons.append((2.0, "命中用户优先指定的酒店。"))

    for key in HOTEL_DIRECT_FEATURE_KEYS:
        weight = _clamp01(_safe_float(preference_weights.get(key), 0.0))
        score = _clamp01(_safe_float(feature_scores.get(key), 0.5))

        if weight <= 0 or score < 0.70:
            continue

        label = HOTEL_PREFERENCE_LABELS.get(key, key)
        reasons.append(
            (
                weight * score,
                f"“{label}”偏好匹配较好（特征分 {score:.2f}）。",
            )
        )

    if target_price is not None:
        if actual_price <= target_price:
            message = (
                f"每晚价格 {actual_price:.0f} 元位于住宿软目标"
                f" {target_price:.0f} 元以内。"
            )
        else:
            message = (
                f"每晚价格高于住宿软目标，但价格适配分仍为 {price_fit:.2f}，"
                "不会被软目标直接过滤。"
            )

        reasons.append((price_fit, message))

    if base_rating >= 0.85:
        reasons.append(
            (base_rating, f"酒店基础评分较高（标准化分 {base_rating:.2f}）。")
        )

    reasons.sort(key=lambda item: -item[0])

    output = [message for _, message in reasons[:4]]

    if not output:
        output.append("当前候选在偏好、价格和基础评分之间较为均衡。")

    return output


def _build_flight_reasons(
    *,
    preference_weights: Mapping[str, float],
    early_flight_fit: float,
    late_arrival_fit: float,
    direct_fit: float,
    price_fit: float,
    flexibility_fit: float,
    target_total_price: float | None,
    total_price: float,
    named_preference_matched: bool,
) -> list[str]:
    """生成航班排序理由，最多保留四条。"""

    reasons: list[tuple[float, str]] = []

    if named_preference_matched:
        reasons.append((2.0, "命中用户优先指定的航班。"))

    feature_map = {
        "avoid_early_flight": early_flight_fit,
        "avoid_late_arrival": late_arrival_fit,
        "prefer_direct": direct_fit,
    }

    for key, score in feature_map.items():
        weight = _clamp01(_safe_float(preference_weights.get(key), 0.0))

        if weight <= 0 or score < 0.70:
            continue

        label = FLIGHT_PREFERENCE_LABELS.get(key, key)
        reasons.append(
            (
                weight * score,
                f"“{label}”偏好匹配较好（特征分 {score:.2f}）。",
            )
        )

    if target_total_price is not None:
        if total_price <= target_total_price:
            message = (
                f"本航段总票价 {total_price:.0f} 元位于交通软目标"
                f" {target_total_price:.0f} 元以内。"
            )
        else:
            message = (
                f"本航段价格高于交通软目标，但价格适配分为 {price_fit:.2f}，"
                "最终是否可选由组合预算决定。"
            )

        reasons.append((price_fit, message))

    if flexibility_fit >= 0.5:
        reasons.append(
            (
                flexibility_fit,
                f"退改便利度较好（特征分 {flexibility_fit:.2f}）。",
            )
        )

    reasons.sort(key=lambda item: -item[0])
    output = [message for _, message in reasons[:4]]

    if not output:
        output.append("当前航班在时间、价格和退改条件之间较为均衡。")

    return output


# ======================================================================
# 输入读取、状态判断与通用辅助函数
# ======================================================================


def _read_preference_weights(
    planning_context: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """读取 PlanningContext 中的分领域偏好权重。"""

    raw = planning_context.get("preference_weights")

    if not isinstance(raw, Mapping):
        return {}

    result: dict[str, dict[str, float]] = {}

    for domain, values in raw.items():
        if not isinstance(values, Mapping):
            continue

        result[str(domain)] = {
            str(key): _clamp01(_safe_float(value, 0.0))
            for key, value in values.items()
        }

    return result


def _read_grouped_constraints(
    planning_context: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """读取 trip / flight / hotel 三组硬约束。"""

    raw = planning_context.get("hard_constraints")

    if not isinstance(raw, Mapping):
        return {"trip": [], "flight": [], "hotel": []}

    return {
        domain: _read_mapping_list(raw.get(domain))
        for domain in ("trip", "flight", "hotel")
    }


def _resolve_rank_status(
    *,
    return_required: bool,
    input_outbound_count: int,
    input_return_count: int,
    input_hotel_count: int,
    output_outbound_count: int,
    output_return_count: int,
    output_hotel_count: int,
) -> tuple[str, list[dict[str, Any]]]:
    """根据三个候选池的输出数量生成业务状态。"""

    issues: list[dict[str, Any]] = []

    if output_outbound_count == 0:
        issues.append(
            {
                "issue_type": "no_outbound_flight_candidates",
                "message": "没有满足当前硬约束的出发航班。",
            }
        )

    if return_required and output_return_count == 0:
        issues.append(
            {
                "issue_type": "no_return_flight_candidates",
                "message": "没有满足当前硬约束的返程航班。",
            }
        )

    if output_hotel_count == 0:
        issues.append(
            {
                "issue_type": "no_hotel_candidates",
                "message": "没有满足当前房间数量或硬约束的酒店。",
            }
        )

    total_output = (
        output_outbound_count
        + output_return_count
        + output_hotel_count
    )

    if total_output == 0:
        return "no_candidates", issues

    all_required_present = (
        output_outbound_count > 0
        and output_hotel_count > 0
        and (not return_required or output_return_count > 0)
    )

    return ("ok" if all_required_present else "partial"), issues


def _candidate_names(
    *,
    candidate: Mapping[str, Any],
    candidate_type: str,
) -> set[str]:
    """生成指定实体匹配使用的候选名称集合。"""

    if candidate_type == "hotel":
        values = [
            candidate.get("hotel_id"),
            candidate.get("name"),
        ]
    else:
        airline = str(candidate.get("airline") or "")
        flight_no = str(candidate.get("flight_no") or "")

        values = [
            candidate.get("flight_id"),
            flight_no,
            f"{airline}{flight_no}",
        ]

    return {
        normalized
        for value in values
        if (normalized := _normalize_name(value))
    }


def _price_range(
    candidates: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> tuple[float, float] | None:
    """读取候选组的最小和最大合法价格。"""

    prices = [
        value
        for candidate in candidates
        if (
            value := _safe_optional_float(candidate.get(field))
        ) is not None
        and value >= 0
    ]

    if not prices:
        return None

    return min(prices), max(prices)


def _scale_price_range(
    price_range: tuple[float, float] | None,
    *,
    multiplier: int,
) -> tuple[float, float] | None:
    """将单人票价范围转换为本次 people_count 的总票价范围。"""

    if price_range is None:
        return None

    return (
        price_range[0] * multiplier,
        price_range[1] * multiplier,
    )


def _parse_time_minutes(value: Any) -> int | None:
    """把 HH:MM 转成从 00:00 开始的分钟数。"""

    text = str(value or "").strip()
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", text)

    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2))

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None

    return hour * 60 + minute


def _time_sort_value(value: Any) -> int:
    """排序时无法解析的时间放到最后。"""

    parsed = _parse_time_minutes(value)
    return parsed if parsed is not None else 10**9


def _normalized_collection(value: Any) -> set[Any]:
    """将 in / not_in 的右侧值标准化为集合。"""

    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]

    return {
        _normalize_name(item)
        for item in values
    }


def _coerce_bool(value: Any) -> bool | None:
    """解释常见布尔值。"""

    if isinstance(value, bool):
        return value

    normalized = str(value or "").strip().casefold()

    if normalized in {"true", "1", "yes", "是"}:
        return True
    if normalized in {"false", "0", "no", "否"}:
        return False

    return None


def _normalize_name(value: Any) -> str:
    """实体和 ID 比较时去空格并忽略大小写。"""

    return "".join(str(value or "").split()).casefold()


def _read_mapping(value: Any) -> dict[str, Any]:
    """读取 dict-like 值；非法值返回空 dict。"""

    return dict(value) if isinstance(value, Mapping) else {}


def _read_mapping_list(value: Any) -> list[dict[str, Any]]:
    """只保留列表中的 dict-like 项。"""

    if not isinstance(value, list):
        return []

    return [
        dict(item)
        for item in value
        if isinstance(item, Mapping)
    ]


def _read_candidate_list(value: Any) -> list[dict[str, Any]]:
    """读取候选列表，并复制每个 dict，避免修改上游 State。"""

    return _read_mapping_list(value)


def _optional_text(value: Any) -> str | None:
    """把非空值转换为字符串。"""

    text = str(value or "").strip()
    return text or None


def _safe_int(value: Any, default: int) -> int:
    """安全转换 int。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
    """把数值限制在 0—1。"""

    return max(0.0, min(1.0, float(value)))
