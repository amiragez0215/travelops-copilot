from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from app.schemas.budget_optimize_schema import (
    BudgetAdjustmentPlan,
    BudgetCheckResult,
    BudgetCombinationOption,
    BudgetOptimizeResult,
    KnownCostBreakdown,
    RemainingBudgetAllocation,
    SelectionResult,
)


# ----------------------------------------------------------------------
# BudgetOptimize 在 PlanningContext 缺少比例时使用的防御性默认值。
# ----------------------------------------------------------------------
# 正常流程会直接读取 planning_context.budget_plan.adjusted_ratios。
# 这里只是为了让手写测试 State 或旧数据不会导致优化器崩溃。
DEFAULT_COMBINATION_WEIGHTS = {
    "transport": 0.35,
    "hotel": 0.35,
    "food_activity": 0.30,
}


@dataclass(frozen=True)
class BudgetOptimizeConfig:
    """
    BudgetOptimizer 的确定性配置。

    为什么要限制每组候选数量？

        组合数量是：

            去程候选数 × 返程候选数 × 酒店候选数

        如果三组各有 100 个候选，就会产生 1,000,000 个组合。
        当前 Mock 数据很小，但仍然把限制显式写进配置，
        让代码具备可扩展边界，也方便后续 Eval 调整。
    """

    # 每组只取 CandidateRankNode 排名前 N 的候选参与组合。
    max_outbound_candidates: int = 10
    max_return_candidates: int = 10
    max_hotel_candidates: int = 10

    # selection_result 中最多保留多少个备选组合。
    max_alternatives: int = 3

    # 组合得分和比例统一保留的小数位数。
    score_round_digits: int = 6

    def to_dict(self) -> dict[str, Any]:
        """转成 Trace / Eval 可读的普通 dict。"""

        return asdict(self)


class BudgetOptimizer:
    """
    对“去程航班 + 返程航班 + 酒店”执行组合优化。

    输入：
        - planning_context
        - flight_candidates
        - hotel_candidates

    输出：
        - selection_result
        - budget_result
        - adjustment_plan
        - budget_optimize_result

    核心边界：

        1. CandidateRankNode 解决“单个候选好不好”。
        2. BudgetOptimizeNode 解决“组合在一起是否更适合用户”。
        3. total_budget 和 max_transport_total 属于组合级硬限制。
        4. transport / hotel / food_activity 比例只是效用权重和软目标。
        5. 不预测用户真实餐饮、购物或临时消费。
    """

    def __init__(
        self,
        config: BudgetOptimizeConfig | None = None,
    ) -> None:
        """
        Args:
            config:
                组合数量限制和输出数量配置。
                不传时使用 BudgetOptimizeConfig 默认值。
        """

        self.config = config or BudgetOptimizeConfig()
        self._validate_config()

    def optimize(
        self,
        *,
        planning_context: Mapping[str, Any],
        flight_candidates: Sequence[Mapping[str, Any]],
        hotel_candidates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """
        执行完整预算组合优化。

        处理步骤：

            1. 读取预算模式、硬限制、软目标和动态比例。
            2. 将航班按 outbound / return 分组。
            3. 每组只保留 CandidateRank 排名前 N 的候选。
            4. 枚举 outbound × return × hotel。
            5. 过滤超过总预算或交通硬上限的组合。
            6. 对可行组合计算综合效用分。
            7. 选择最高分组合并保存若干备选。
            8. 没有可行解时生成 adjustment_plan，而不是自动放松约束。
        """

        if not isinstance(planning_context, Mapping):
            raise TypeError("planning_context 必须是 dict-like 对象")

        if not isinstance(flight_candidates, Sequence) or isinstance(
            flight_candidates,
            (str, bytes),
        ):
            raise TypeError("flight_candidates 必须是候选列表")

        if not isinstance(hotel_candidates, Sequence) or isinstance(
            hotel_candidates,
            (str, bytes),
        ):
            raise TypeError("hotel_candidates 必须是候选列表")

        # 1. 标准化输入候选，只保留 dict-like 项。
        flights = [
            dict(item)
            for item in flight_candidates
            if isinstance(item, Mapping)
        ]
        hotels = [
            dict(item)
            for item in hotel_candidates
            if isinstance(item, Mapping)
        ]

        # 2. 按方向拆分航班。
        outbound = [
            item
            for item in flights
            if _flight_direction(item) == "outbound"
        ]
        return_flights = [
            item
            for item in flights
            if _flight_direction(item) == "return"
        ]

        # 3. 按 CandidateRank 的 rank 稳定排序后截取前 N。
        outbound = _limit_ranked_candidates(
            outbound,
            limit=self.config.max_outbound_candidates,
            id_field="flight_id",
        )
        return_flights = _limit_ranked_candidates(
            return_flights,
            limit=self.config.max_return_candidates,
            id_field="flight_id",
        )
        hotels = _limit_ranked_candidates(
            hotels,
            limit=self.config.max_hotel_candidates,
            id_field="hotel_id",
        )

        input_counts = {
            "outbound_flights": len(outbound),
            "return_flights": len(return_flights),
            "hotels": len(hotels),
        }

        budget_plan = _read_mapping(
            planning_context.get("budget_plan")
        )
        request = _read_mapping(
            planning_context.get("request")
        )

        budget_mode = _read_budget_mode(
            budget_plan
        )

        total_budget = _read_total_budget(
            budget_plan=budget_plan,
            request=request,
        )

        # 4. 任意必要候选组为空时，不存在完整旅行组合。
        if not outbound or not return_flights or not hotels:
            return self._build_no_candidates_output(
                input_counts=input_counts,
                budget_mode=budget_mode,
                total_budget=total_budget,
            )

        hard_limits = _read_mapping(
            budget_plan.get("hard_limits")
        )
        soft_targets = _read_mapping(
            budget_plan.get("soft_targets")
        )

        max_transport_total = _safe_optional_float(
            hard_limits.get("max_transport_total")
        )

        transport_target = _safe_optional_float(
            soft_targets.get("transport_budget")
        )
        hotel_target = _safe_optional_float(
            soft_targets.get("hotel_budget")
        )
        food_activity_target = _safe_optional_float(
            soft_targets.get("food_activity_budget")
        )

        # 5. adjusted_ratios 反映用户“住宿多投入、航班省一点”等消费倾向。
        #    它们在这里作为组合效用权重，而不是硬预算上限。
        score_weights = _resolve_combination_weights(
            budget_plan=budget_plan,
            include_food_activity=(
                budget_mode == "budget_limited"
                and total_budget is not None
                and food_activity_target is not None
            ),
        )

        feasible_options: list[
            tuple[
                BudgetCombinationOption,
                dict[str, Any],
                dict[str, Any],
                dict[str, Any],
            ]
        ] = []

        # all_cost_options 用于无可行解时找到当前最便宜的已知报价组合。
        all_cost_options: list[dict[str, Any]] = []

        evaluated_count = 0
        rejected_by_total_budget = 0
        rejected_by_transport_limit = 0

        # 6. 枚举三个候选组的笛卡尔积。
        for outbound_item, return_item, hotel_item in itertools.product(
            outbound,
            return_flights,
            hotels,
        ):
            evaluated_count += 1

            # 7. 计算当前组合能够确定的 Mock 费用。
            cost_breakdown = _build_known_cost_breakdown(
                outbound_flight=outbound_item,
                return_flight=return_item,
                hotel=hotel_item,
            )

            combination_id = _build_combination_id(
                outbound_flight_id=str(
                    outbound_item.get("flight_id") or ""
                ),
                return_flight_id=str(
                    return_item.get("flight_id") or ""
                ),
                hotel_id=str(
                    hotel_item.get("hotel_id") or ""
                ),
            )

            all_cost_options.append(
                {
                    "combination_id": combination_id,
                    "outbound_flight_id": str(
                        outbound_item.get("flight_id") or ""
                    ),
                    "return_flight_id": str(
                        return_item.get("flight_id") or ""
                    ),
                    "hotel_id": str(
                        hotel_item.get("hotel_id") or ""
                    ),
                    "known_subtotal": cost_breakdown.known_subtotal,
                    "transport_total": cost_breakdown.transport_total,
                    "selected_costs": cost_breakdown.to_state_dict(),
                }
            )

            # 8. max_transport_total 是往返交通合计的真正硬限制。
            if (
                max_transport_total is not None
                and cost_breakdown.transport_total
                > max_transport_total
            ):
                rejected_by_transport_limit += 1
                continue

            # 9. total_budget 是组合级硬上限。
            #    food_activity 等软目标不会在这里直接过滤组合。
            if (
                budget_mode == "budget_limited"
                and total_budget is not None
                and cost_breakdown.known_subtotal
                > total_budget
            ):
                rejected_by_total_budget += 1
                continue

            remaining_budget = (
                round(
                    total_budget
                    - cost_breakdown.known_subtotal,
                    2,
                )
                if total_budget is not None
                else None
            )

            # 10. 对剩余预算做“安排”，而不是预测真实消费。
            remaining_allocation = _build_remaining_budget_allocation(
                remaining_budget=remaining_budget,
                food_activity_target=food_activity_target,
            )

            # 11. 将单个候选分数组合成三个全局组件。
            transport_quality = round(
                (
                    _candidate_score(outbound_item)
                    + _candidate_score(return_item)
                )
                / 2.0,
                self.config.score_round_digits,
            )
            hotel_quality = round(
                _candidate_score(hotel_item),
                self.config.score_round_digits,
            )
            food_activity_reserve_fit = (
                _food_activity_reserve_fit(
                    remaining_budget=remaining_budget,
                    food_activity_target=food_activity_target,
                )
            )

            # 12. 使用动态消费比例计算组合效用分。
            combination_score = _weighted_combination_score(
                transport_quality=transport_quality,
                hotel_quality=hotel_quality,
                food_activity_reserve_fit=(
                    food_activity_reserve_fit
                ),
                weights=score_weights,
            )

            soft_target_deviation = {
                "transport": _difference_or_none(
                    cost_breakdown.transport_total,
                    transport_target,
                ),
                "hotel": _difference_or_none(
                    cost_breakdown.hotel,
                    hotel_target,
                ),
                "food_activity_reserve": _difference_or_none(
                    remaining_allocation.food_activity,
                    food_activity_target,
                ),
            }

            reasons = _build_selection_reasons(
                transport_quality=transport_quality,
                hotel_quality=hotel_quality,
                reserve_fit=food_activity_reserve_fit,
                known_subtotal=cost_breakdown.known_subtotal,
                total_budget=total_budget,
                remaining_budget=remaining_budget,
                food_activity_target=food_activity_target,
            )

            option = BudgetCombinationOption(
                combination_id=combination_id,
                outbound_flight_id=str(
                    outbound_item["flight_id"]
                ),
                return_flight_id=str(
                    return_item["flight_id"]
                ),
                hotel_id=str(
                    hotel_item["hotel_id"]
                ),
                outbound_rank=_candidate_rank(
                    outbound_item
                ),
                return_rank=_candidate_rank(
                    return_item
                ),
                hotel_rank=_candidate_rank(
                    hotel_item
                ),
                combination_score=round(
                    combination_score,
                    self.config.score_round_digits,
                ),
                score_breakdown={
                    "transport_quality": transport_quality,
                    "hotel_quality": hotel_quality,
                    "food_activity_reserve_fit": (
                        round(
                            food_activity_reserve_fit,
                            self.config.score_round_digits,
                        )
                        if food_activity_reserve_fit
                        is not None
                        else None
                    ),
                    "component_weights": {
                        key: round(
                            value,
                            self.config.score_round_digits,
                        )
                        for key, value in score_weights.items()
                    },
                },
                selected_costs=cost_breakdown,
                total_budget=total_budget,
                remaining_budget=remaining_budget,
                remaining_budget_allocation=(
                    remaining_allocation
                ),
                soft_target_deviation=(
                    soft_target_deviation
                ),
                selection_reasons=reasons,
            )

            feasible_options.append(
                (
                    option,
                    outbound_item,
                    return_item,
                    hotel_item,
                )
            )

        # 13. 没有任何可行组合时，生成可解释的调整建议。
        if not feasible_options:
            return self._build_no_feasible_output(
                budget_mode=budget_mode,
                total_budget=total_budget,
                input_counts=input_counts,
                evaluated_count=evaluated_count,
                rejected_by_total_budget=(
                    rejected_by_total_budget
                ),
                rejected_by_transport_limit=(
                    rejected_by_transport_limit
                ),
                all_cost_options=all_cost_options,
                max_transport_total=max_transport_total,
                score_weights=score_weights,
            )

        # 14. 先按组合分降序，再按已知费用、候选排名和 ID 稳定排序。
        feasible_options.sort(
            key=lambda item: (
                -item[0].combination_score,
                item[0].selected_costs.known_subtotal,
                (
                    item[0].outbound_rank
                    + item[0].return_rank
                    + item[0].hotel_rank
                ),
                item[0].combination_id,
            )
        )

        selected_option, selected_outbound, selected_return, selected_hotel = (
            feasible_options[0]
        )

        alternative_options = [
            item[0]
            for item in feasible_options[
                1 : 1 + self.config.max_alternatives
            ]
        ]

        selection_result = SelectionResult(
            status="feasible",
            budget_mode=budget_mode,
            selected_combination=selected_option,
            selected_outbound_flight=dict(
                selected_outbound
            ),
            selected_return_flight=dict(
                selected_return
            ),
            selected_hotel=dict(
                selected_hotel
            ),
            selected_hotel_id=str(
                selected_hotel["hotel_id"]
            ),
            selected_costs=(
                selected_option.selected_costs
            ),
            total_budget=total_budget,
            remaining_budget=(
                selected_option.remaining_budget
            ),
            remaining_budget_allocation=(
                selected_option.remaining_budget_allocation
            ),
            alternatives=alternative_options,
            notes=[
                "航班和酒店来自 Mock 结构化候选，并已由 CandidateRankNode 评分。",
                "餐饮、活动、市内交通和临时开支只做预算安排，不预测真实消费。",
                "RAG 不参与本节点的航班或酒店数值选择。",
            ],
        )

        budget_result = BudgetCheckResult(
            status=(
                "within_budget"
                if budget_mode == "budget_limited"
                else "no_budget_limit"
            ),
            budget_mode=budget_mode,
            total_budget=total_budget,
            selected_known_subtotal=(
                selected_option.selected_costs.known_subtotal
            ),
            remaining_budget=(
                selected_option.remaining_budget
            ),
            known_cost_note=(
                "当前只计算 Mock 航班和酒店的已知报价；"
                "剩余金额是预算安排，不是实际旅行消费预测。"
            ),
        )

        adjustment_plan = BudgetAdjustmentPlan(
            status="not_needed",
            suggestions=[],
        )

        minimum_known_subtotal = min(
            item["known_subtotal"]
            for item in all_cost_options
        )

        optimize_result = BudgetOptimizeResult(
            status="feasible",
            budget_mode=budget_mode,
            input_counts=input_counts,
            candidate_limits={
                "outbound_flights": (
                    self.config.max_outbound_candidates
                ),
                "return_flights": (
                    self.config.max_return_candidates
                ),
                "hotels": (
                    self.config.max_hotel_candidates
                ),
                "alternatives": (
                    self.config.max_alternatives
                ),
            },
            evaluated_combination_count=(
                evaluated_count
            ),
            feasible_combination_count=len(
                feasible_options
            ),
            rejected_by_total_budget_count=(
                rejected_by_total_budget
            ),
            rejected_by_transport_limit_count=(
                rejected_by_transport_limit
            ),
            selected_combination_id=(
                selected_option.combination_id
            ),
            selected_score=(
                selected_option.combination_score
            ),
            minimum_known_subtotal=(
                minimum_known_subtotal
            ),
            scoring_config={
                "combination_weights": (
                    score_weights
                ),
                "allocation_source": (
                    budget_plan.get(
                        "allocation_source"
                    )
                ),
                **self.config.to_dict(),
            },
            issues=[],
        )

        return {
            "selection_result": (
                selection_result.to_state_dict()
            ),
            "budget_result": (
                budget_result.to_state_dict()
            ),
            "adjustment_plan": (
                adjustment_plan.to_state_dict()
            ),
            "budget_optimize_result": (
                optimize_result.to_state_dict()
            ),
        }

    def _build_no_candidates_output(
        self,
        *,
        input_counts: Mapping[str, int],
        budget_mode: str,
        total_budget: float | None,
    ) -> dict[str, Any]:
        """构造缺少必要候选组时的业务输出。"""

        missing_groups = [
            group
            for group, count in input_counts.items()
            if count <= 0
        ]

        issues = [
            {
                "issue_type": "missing_candidate_group",
                "group": group,
                "message": f"缺少 {group} 候选，无法形成完整旅行组合。",
            }
            for group in missing_groups
        ]

        normalized_mode = (
            "no_budget_limit"
            if budget_mode == "no_budget_limit"
            else "budget_limited"
        )

        selection_result = SelectionResult(
            status="no_candidates",
            budget_mode=normalized_mode,
            total_budget=total_budget,
            notes=[
                "CandidateRankNode 没有提供完整的去程、返程和酒店候选。"
            ],
        )

        budget_result = BudgetCheckResult(
            status="no_candidates",
            budget_mode=normalized_mode,
            total_budget=total_budget,
            known_cost_note=(
                "候选组不完整，因此没有执行已知费用组合计算。"
            ),
        )

        adjustment_plan = BudgetAdjustmentPlan(
            status="missing_candidates",
            reason_code="missing_candidate_group",
            suggestions=[
                "调整硬约束或日期，以获得完整的航班和酒店候选。"
            ],
        )

        optimize_result = BudgetOptimizeResult(
            status="no_candidates",
            budget_mode=normalized_mode,
            input_counts=dict(input_counts),
            candidate_limits={
                "outbound_flights": (
                    self.config.max_outbound_candidates
                ),
                "return_flights": (
                    self.config.max_return_candidates
                ),
                "hotels": (
                    self.config.max_hotel_candidates
                ),
                "alternatives": (
                    self.config.max_alternatives
                ),
            },
            evaluated_combination_count=0,
            feasible_combination_count=0,
            rejected_by_total_budget_count=0,
            rejected_by_transport_limit_count=0,
            scoring_config=self.config.to_dict(),
            issues=issues,
        )

        return {
            "selection_result": selection_result.to_state_dict(),
            "budget_result": budget_result.to_state_dict(),
            "adjustment_plan": adjustment_plan.to_state_dict(),
            "budget_optimize_result": optimize_result.to_state_dict(),
        }

    def _build_no_feasible_output(
        self,
        *,
        budget_mode: str,
        total_budget: float | None,
        input_counts: Mapping[str, int],
        evaluated_count: int,
        rejected_by_total_budget: int,
        rejected_by_transport_limit: int,
        all_cost_options: Sequence[Mapping[str, Any]],
        max_transport_total: float | None,
        score_weights: Mapping[str, float],
    ) -> dict[str, Any]:
        """构造所有组合都违反组合级硬限制时的输出。"""

        # 1. 先找出满足交通硬上限的组合。
        #    如果一个组合连交通硬限制都不满足，单纯提高总预算也无法解决。
        transport_valid_options = [
            item
            for item in all_cost_options
            if (
                max_transport_total is None
                or _safe_float(
                    item.get("transport_total"),
                    default=float("inf"),
                )
                <= max_transport_total
            )
        ]

        cheapest_pool = (
            transport_valid_options
            or list(all_cost_options)
        )

        cheapest = min(
            cheapest_pool,
            key=lambda item: (
                _safe_float(
                    item.get("known_subtotal"),
                    default=float("inf"),
                ),
                str(item.get("combination_id") or ""),
            ),
        )

        minimum_known_subtotal = _safe_float(
            cheapest.get("known_subtotal"),
            default=0.0,
        )

        transport_limit_blocks_all = (
            max_transport_total is not None
            and not transport_valid_options
        )

        # 2. 只有存在满足交通硬限制的组合时，
        #    “提高总预算多少”才是一个真实可执行的建议。
        required_budget_increase = (
            round(
                max(
                    minimum_known_subtotal
                    - total_budget,
                    0.0,
                ),
                2,
            )
            if (
                total_budget is not None
                and not transport_limit_blocks_all
            )
            else None
        )

        suggestions: list[str] = []
        reason_code = "no_feasible_combination"

        # 3. 如果所有组合都被交通硬上限阻断，优先说明该客观条件。
        if transport_limit_blocks_all:
            reason_code = "transport_limit_blocks_all"
            suggestions.append(
                "放宽往返交通费用硬上限，或调整日期以获得更便宜的航班。"
            )

        # 4. 否则，如果满足其他硬限制的最便宜组合仍然超总预算，
        #    给出最小可解释预算差距。
        elif (
            total_budget is not None
            and required_budget_increase is not None
            and required_budget_increase > 0
        ):
            reason_code = "all_combinations_over_total_budget"
            suggestions.append(
                f"若保持当前候选和硬约束，总预算至少需要增加 "
                f"{required_budget_increase:.2f} 元。"
            )
            suggestions.append(
                "也可以调整日期、降低酒店价格偏好或放宽航班条件。"
            )

        if not suggestions:
            suggestions.append(
                "调整候选相关硬约束后重新查询和排序。"
            )

        issues = [
            {
                "issue_type": reason_code,
                "message": (
                    "当前没有同时满足组合级预算硬限制的"
                    "去程航班、返程航班和酒店组合。"
                ),
                "total_budget": total_budget,
                "max_transport_total": (
                    max_transport_total
                ),
                "minimum_known_subtotal": (
                    minimum_known_subtotal
                ),
            }
        ]

        normalized_mode = (
            "budget_limited"
            if budget_mode == "budget_limited"
            else "no_budget_limit"
        )

        selection_result = SelectionResult(
            status="no_feasible_combination",
            budget_mode=normalized_mode,
            total_budget=total_budget,
            notes=[
                "系统不会自动提高预算或放松用户硬约束。",
                "minimum_known_subtotal 只包含 Mock 航班和酒店报价。",
            ],
        )

        budget_result = BudgetCheckResult(
            status="no_feasible_combination",
            budget_mode=normalized_mode,
            total_budget=total_budget,
            selected_known_subtotal=None,
            remaining_budget=None,
            known_cost_note=(
                "当前没有可行组合；最便宜组合仍然只代表"
                "Mock 航班和酒店的已知报价。"
            ),
        )

        adjustment_plan = BudgetAdjustmentPlan(
            status="need_user_change",
            reason_code=reason_code,
            current_total_budget=total_budget,
            minimum_known_subtotal=(
                minimum_known_subtotal
            ),
            required_budget_increase=(
                required_budget_increase
            ),
            cheapest_combination={
                "combination_id": (
                    cheapest.get("combination_id")
                ),
                "outbound_flight_id": (
                    cheapest.get("outbound_flight_id")
                ),
                "return_flight_id": (
                    cheapest.get("return_flight_id")
                ),
                "hotel_id": (
                    cheapest.get("hotel_id")
                ),
                "selected_costs": (
                    cheapest.get("selected_costs")
                ),
            },
            suggestions=suggestions,
        )

        optimize_result = BudgetOptimizeResult(
            status="no_feasible_combination",
            budget_mode=normalized_mode,
            input_counts=dict(input_counts),
            candidate_limits={
                "outbound_flights": (
                    self.config.max_outbound_candidates
                ),
                "return_flights": (
                    self.config.max_return_candidates
                ),
                "hotels": (
                    self.config.max_hotel_candidates
                ),
                "alternatives": (
                    self.config.max_alternatives
                ),
            },
            evaluated_combination_count=(
                evaluated_count
            ),
            feasible_combination_count=0,
            rejected_by_total_budget_count=(
                rejected_by_total_budget
            ),
            rejected_by_transport_limit_count=(
                rejected_by_transport_limit
            ),
            minimum_known_subtotal=(
                minimum_known_subtotal
            ),
            scoring_config={
                "combination_weights": dict(
                    score_weights
                ),
                **self.config.to_dict(),
            },
            issues=issues,
        )

        return {
            "selection_result": selection_result.to_state_dict(),
            "budget_result": budget_result.to_state_dict(),
            "adjustment_plan": adjustment_plan.to_state_dict(),
            "budget_optimize_result": optimize_result.to_state_dict(),
        }

    def _validate_config(self) -> None:
        """在启动时尽早发现无效组合配置。"""

        for name in (
            "max_outbound_candidates",
            "max_return_candidates",
            "max_hotel_candidates",
        ):
            if getattr(self.config, name) <= 0:
                raise ValueError(
                    f"{name} 必须大于 0"
                )

        if self.config.max_alternatives < 0:
            raise ValueError(
                "max_alternatives 不能小于 0"
            )


# ======================================================================
# 输入读取和候选标准化
# ======================================================================


def _flight_direction(
    candidate: Mapping[str, Any],
) -> str:
    """兼容 direction 和 query_direction 两种字段名。"""

    return str(
        candidate.get("direction")
        or candidate.get("query_direction")
        or ""
    ).strip()


def _limit_ranked_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    id_field: str,
) -> list[dict[str, Any]]:
    """
    根据 CandidateRankNode 的 rank 截取前 N 个候选。

    同一个 ID 如果重复出现，只保留排名更靠前、得分更高的一条。
    """

    best_by_id: dict[str, dict[str, Any]] = {}

    # 1. 先按 rank、得分和价格稳定排序。
    ordered = sorted(
        (dict(item) for item in candidates),
        key=lambda item: (
            _candidate_rank(item),
            -_candidate_score(item),
            _candidate_cost_for_tie_break(item),
            str(item.get(id_field) or ""),
        ),
    )

    # 2. 按 ID 去重，保留第一条最佳记录。
    for item in ordered:
        candidate_id = str(
            item.get(id_field) or ""
        ).strip()

        if not candidate_id:
            continue

        best_by_id.setdefault(
            candidate_id,
            item,
        )

    # 3. 截取上限，控制笛卡尔积规模。
    return list(best_by_id.values())[:limit]


def _candidate_rank(
    candidate: Mapping[str, Any],
) -> int:
    """读取 CandidateRankNode 生成的排名。"""

    return max(
        _safe_int(
            candidate.get("rank"),
            default=10**9,
        ),
        1,
    )


def _candidate_score(
    candidate: Mapping[str, Any],
) -> float:
    """读取并限制单候选总分到 0—1。"""

    return _clamp01(
        _safe_float(
            candidate.get("total_score"),
            default=0.0,
        )
    )


def _candidate_cost_for_tie_break(
    candidate: Mapping[str, Any],
) -> float:
    """候选排序分数相同时使用的价格稳定条件。"""

    value = (
        candidate.get("total_price")
        if candidate.get("total_price")
        is not None
        else candidate.get(
            "estimated_total_price"
        )
    )

    return max(
        _safe_float(value, default=0.0),
        0.0,
    )


def _build_known_cost_breakdown(
    *,
    outbound_flight: Mapping[str, Any],
    return_flight: Mapping[str, Any],
    hotel: Mapping[str, Any],
) -> KnownCostBreakdown:
    """
    计算当前组合可以确定的 Mock 费用。

    CandidateRankNode 已经把人数和房间数乘入：
        flight.total_price
        hotel.estimated_total_price

    BudgetOptimizer 不再重复乘人数或晚数，避免费用被重复放大。
    """

    outbound_cost = max(
        _safe_float(
            outbound_flight.get(
                "total_price",
                outbound_flight.get("price"),
            ),
            default=0.0,
        ),
        0.0,
    )
    return_cost = max(
        _safe_float(
            return_flight.get(
                "total_price",
                return_flight.get("price"),
            ),
            default=0.0,
        ),
        0.0,
    )
    hotel_cost = max(
        _safe_float(
            hotel.get(
                "estimated_total_price"
            ),
            default=0.0,
        ),
        0.0,
    )

    transport_total = round(
        outbound_cost + return_cost,
        2,
    )
    known_subtotal = round(
        transport_total + hotel_cost,
        2,
    )

    return KnownCostBreakdown(
        outbound_flight=round(
            outbound_cost,
            2,
        ),
        return_flight=round(
            return_cost,
            2,
        ),
        transport_total=transport_total,
        hotel=round(hotel_cost, 2),
        known_subtotal=known_subtotal,
    )


# ======================================================================
# 预算策略和组合评分
# ======================================================================


def _read_budget_mode(
    budget_plan: Mapping[str, Any],
) -> str:
    """读取并标准化预算模式。"""

    return (
        "no_budget_limit"
        if budget_plan.get("mode")
        == "no_budget_limit"
        else "budget_limited"
    )


def _read_total_budget(
    *,
    budget_plan: Mapping[str, Any],
    request: Mapping[str, Any],
) -> float | None:
    """
    读取总预算。

    优先级：
        budget_plan.hard_limits.total_budget
        → budget_plan.total_budget
        → planning_context.request.total_budget
    """

    hard_limits = _read_mapping(
        budget_plan.get("hard_limits")
    )

    for value in (
        hard_limits.get("total_budget"),
        budget_plan.get("total_budget"),
        request.get("total_budget"),
    ):
        parsed = _safe_optional_float(value)

        if parsed is not None:
            return max(parsed, 0.0)

    return None


def _resolve_combination_weights(
    *,
    budget_plan: Mapping[str, Any],
    include_food_activity: bool,
) -> dict[str, float]:
    """
    把动态预算比例转成组合效用权重。

    有总预算：
        transport、hotel、food_activity 三项共同参与组合评分。

    没有总预算：
        无法计算“剩余餐饮活动预算适配度”，
        因此只在 transport 和 hotel 之间重新归一化。
    """

    adjusted = _read_mapping(
        budget_plan.get("adjusted_ratios")
    )

    raw = {
        "transport": max(
            _safe_float(
                adjusted.get("transport"),
                DEFAULT_COMBINATION_WEIGHTS[
                    "transport"
                ],
            ),
            0.0,
        ),
        "hotel": max(
            _safe_float(
                adjusted.get("hotel"),
                DEFAULT_COMBINATION_WEIGHTS[
                    "hotel"
                ],
            ),
            0.0,
        ),
    }

    if include_food_activity:
        raw["food_activity"] = max(
            _safe_float(
                adjusted.get("food_activity"),
                DEFAULT_COMBINATION_WEIGHTS[
                    "food_activity"
                ],
            ),
            0.0,
        )

    total = sum(raw.values()) or 1.0

    return {
        key: value / total
        for key, value in raw.items()
    }


def _food_activity_reserve_fit(
    *,
    remaining_budget: float | None,
    food_activity_target: float | None,
) -> float | None:
    """
    计算剩余预算对餐饮活动软目标的满足程度。

    公式：
        min(remaining_budget / food_activity_target, 1.0)

    这只是预算充足度，不是用户真实消费预测。
    """

    if (
        remaining_budget is None
        or food_activity_target is None
    ):
        return None

    if food_activity_target <= 0:
        return 1.0

    return _clamp01(
        remaining_budget
        / food_activity_target
    )


def _weighted_combination_score(
    *,
    transport_quality: float,
    hotel_quality: float,
    food_activity_reserve_fit: float | None,
    weights: Mapping[str, float],
) -> float:
    """使用动态消费比例计算组合综合效用分。"""

    score = (
        _clamp01(transport_quality)
        * _safe_float(
            weights.get("transport"),
            0.0,
        )
        + _clamp01(hotel_quality)
        * _safe_float(
            weights.get("hotel"),
            0.0,
        )
    )

    if (
        "food_activity" in weights
        and food_activity_reserve_fit
        is not None
    ):
        score += (
            _clamp01(
                food_activity_reserve_fit
            )
            * _safe_float(
                weights.get(
                    "food_activity"
                ),
                0.0,
            )
        )

    return _clamp01(score)


def _build_remaining_budget_allocation(
    *,
    remaining_budget: float | None,
    food_activity_target: float | None,
) -> RemainingBudgetAllocation:
    """
    将剩余预算安排到餐饮活动和市内交通缓冲。

    关键边界：
        - 这里只说明“建议如何保留预算”；
        - 不声称用户一定会花掉这些钱；
        - 不预测具体小吃、购物或打车金额。
    """

    if remaining_budget is None:
        return RemainingBudgetAllocation(
            food_activity=None,
            local_transport_and_buffer=None,
            food_activity_target=(
                food_activity_target
            ),
            food_activity_shortfall=None,
            allocation_basis=(
                "no_budget_limit"
            ),
            note=(
                "用户没有提供总预算，因此不生成具体剩余预算安排。"
            ),
        )

    remaining = max(
        round(remaining_budget, 2),
        0.0,
    )
    target = (
        max(
            round(food_activity_target, 2),
            0.0,
        )
        if food_activity_target is not None
        else 0.0
    )

    # 1. 先尽量满足餐饮活动软目标。
    food_activity = round(
        min(remaining, target),
        2,
    )

    # 2. 超出餐饮活动软目标的余额作为市内交通和机动缓冲。
    local_transport_and_buffer = round(
        remaining - food_activity,
        2,
    )

    # 3. 如果余额不足，记录与软目标的差距。
    shortfall = round(
        max(target - food_activity, 0.0),
        2,
    )

    return RemainingBudgetAllocation(
        food_activity=food_activity,
        local_transport_and_buffer=(
            local_transport_and_buffer
        ),
        food_activity_target=(
            target
            if food_activity_target is not None
            else None
        ),
        food_activity_shortfall=(
            shortfall
            if food_activity_target is not None
            else None
        ),
        allocation_basis=(
            "soft_target_then_buffer"
        ),
        note=(
            "这是总预算扣除 Mock 航班和酒店报价后的建议安排，"
            "不是对真实餐饮、交通或临时消费的预测。"
        ),
    )


def _build_selection_reasons(
    *,
    transport_quality: float,
    hotel_quality: float,
    reserve_fit: float | None,
    known_subtotal: float,
    total_budget: float | None,
    remaining_budget: float | None,
    food_activity_target: float | None,
) -> list[str]:
    """为每个组合生成简洁、可解释的选择原因。"""

    reasons = [
        f"往返航班组合质量分为 {transport_quality:.2f}。",
        f"酒店候选质量分为 {hotel_quality:.2f}。",
    ]

    if total_budget is not None and remaining_budget is not None:
        reasons.append(
            f"Mock 航班和酒店已知报价合计 {known_subtotal:.2f} 元，"
            f"在总预算内剩余 {remaining_budget:.2f} 元。"
        )

    if (
        reserve_fit is not None
        and food_activity_target is not None
    ):
        if reserve_fit >= 1.0:
            reasons.append(
                "剩余预算达到餐饮活动软目标。"
            )
        else:
            reasons.append(
                "剩余预算低于餐饮活动软目标，组合效用因此降低。"
            )

    return reasons


def _difference_or_none(
    actual: float | None,
    target: float | None,
) -> float | None:
    """计算实际值与软目标之差；任一缺失时返回 None。"""

    if actual is None or target is None:
        return None

    return round(
        float(actual) - float(target),
        2,
    )


# ======================================================================
# 稳定 ID 和通用辅助函数
# ======================================================================


def _build_combination_id(
    *,
    outbound_flight_id: str,
    return_flight_id: str,
    hotel_id: str,
) -> str:
    """
    根据三个候选 ID 生成稳定组合 ID。

    稳定 ID 便于：
        - Replay；
        - Eval；
        - ModifyTrip 比较前后组合；
        - Trace 追踪。
    """

    payload = (
        f"{outbound_flight_id}|"
        f"{return_flight_id}|"
        f"{hotel_id}"
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]

    return f"combo_{digest}"


def _read_mapping(
    value: Any,
) -> dict[str, Any]:
    """安全读取 dict-like 对象。"""

    return (
        dict(value)
        if isinstance(value, Mapping)
        else {}
    )


def _safe_int(
    value: Any,
    default: int,
) -> int:
    """安全转换 int。"""

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _safe_float(
    value: Any,
    default: float,
) -> float:
    """安全转换 float。"""

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _safe_optional_float(
    value: Any,
) -> float | None:
    """安全转换可空 float。"""

    if value is None:
        return None

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def _clamp01(
    value: float,
) -> float:
    """把数值限制在 0—1。"""

    return max(
        0.0,
        min(1.0, float(value)),
    )
