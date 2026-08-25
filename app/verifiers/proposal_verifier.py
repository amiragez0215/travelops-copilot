from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from pydantic import ValidationError

from app.proposals.proposal_generator import (
    KNOWLEDGE_ACTIVITY_TYPES,
    PROPOSAL_GENERATOR_VERSION,
    PROPOSAL_PROMPT_VERSION,
    RAIN_RISK_TYPES,
    WEATHER_RISKS_REQUIRING_ADJUSTMENT,
    build_locked_proposal_context,
    build_requirement_catalog,
)
from app.schemas.proposal_schema import TripProposal
from app.schemas.verifier_schema import (
    ProposalVerificationResult,
    VerificationCheckResult,
    VerificationIssue,
)


# ----------------------------------------------------------------------
# 用户可见文本中禁止出现的“已执行真实动作”声明。
# ----------------------------------------------------------------------
# 这些规则只匹配肯定完成式，例如“已经为你预订”。
# 固定免责声明“系统没有执行真实预订”不会命中。
UNSAFE_ACTION_CLAIM_PATTERNS: tuple[
    tuple[str, re.Pattern[str]],
    ...,
] = (
    (
        "real_booking_claim",
        re.compile(
            r"(?:已经|已)(?:为你|替你|帮你)?(?:完成|成功)?"
            r"(?:预订|预定|订好|订房|订票)"
            r"|(?:预订|预定|订房|订票)(?:已经)?成功"
        ),
    ),
    (
        "real_payment_claim",
        re.compile(
            r"(?:已经|已)(?:为你|替你|帮你)?(?:完成)?(?:付款|支付|扣款)"
            r"|(?:付款|支付|扣款)(?:已经)?成功"
        ),
    ),
    (
        "real_ticketing_claim",
        re.compile(
            r"(?:已经|已)(?:为你|替你|帮你)?(?:完成)?出票"
            r"|出票(?:已经)?成功"
        ),
    ),
    (
        "real_notification_claim",
        re.compile(
            r"(?:已经|已)(?:为你|替你|帮你)?(?:发送|发出)"
            r"(?:邮件|短信|通知)"
            r"|(?:邮件|短信|通知)(?:已经)?发送成功"
        ),
    ),
)


@dataclass(frozen=True)
class ProposalVerifierConfig:
    """
    ProposalVerifier 的确定性配置。

    Args:
        float_tolerance:
            金额和浮点分数比较容差。
            例如 0.01 表示金额允许一分钱以内的浮点误差。

        max_proposal_repairs:
            Verifier 发现 Proposal 内容问题后，最多允许回到 ProposalNode 几次。

        max_budget_repairs:
            Verifier 发现选择组合或预算状态过期后，最多允许重跑 BudgetOptimize 几次。

        max_issue_examples:
            最多写入 State 的问题数量，防止异常 Proposal 让 State 过大。
    """

    float_tolerance: float = 0.01
    max_proposal_repairs: int = 1
    max_budget_repairs: int = 1
    max_issue_examples: int = 50


class ProposalVerifier:
    """
    对最终 TripProposal 执行独立、确定性的工作流级审查。

    Verifier 与 Proposal 内部 `_validate_proposal_draft` 的区别：

        Proposal Draft Validator：
            检查 LLM 草稿是否有资格被组装。

        ProposalVerifier：
            不信任任何上游结果，重新检查最终完整 Proposal，
            包括程序注入的航班、酒店、天气、预算、来源和执行边界。

    本类不调用 LLM，因此：
        - 相同输入得到相同结果；
        - 容易单元测试；
        - 可以作为人工审批前的最后一道规则防线。
    """

    def __init__(
        self,
        config: ProposalVerifierConfig | None = None,
    ) -> None:
        self.config = config or ProposalVerifierConfig()

        if self.config.float_tolerance < 0:
            raise ValueError("float_tolerance 不能小于 0")

        if self.config.max_proposal_repairs < 0:
            raise ValueError("max_proposal_repairs 不能小于 0")

        if self.config.max_budget_repairs < 0:
            raise ValueError("max_budget_repairs 不能小于 0")

        if self.config.max_issue_examples <= 0:
            raise ValueError("max_issue_examples 必须大于 0")

    def verify(
        self,
        *,
        proposal: Mapping[str, Any],
        proposal_result: Mapping[str, Any],
        trip_request: Mapping[str, Any],
        planning_context: Mapping[str, Any],
        selection_result: Mapping[str, Any],
        weather_result: Mapping[str, Any],
        evidence_pool: Sequence[Mapping[str, Any]],
        evidence_result: Mapping[str, Any],
        flight_candidates: Sequence[Mapping[str, Any]],
        hotel_candidates: Sequence[Mapping[str, Any]],
        proposal_retry_count: int = 0,
        budget_retry_count: int = 0,
        retrieval_plan: Mapping[str, Any] | None = None,
        rag_retry_count: int = 0,
    ) -> dict[str, Any]:
        """
        验证最终 Proposal，并决定下一条 LangGraph 路由。

        检查组：
            1. 最终 Proposal Schema；
            2. ProposalGenerator 运行摘要；
            3. 候选来源和 selection_result；
            4. 锁定事实与 locked_fact_hash；
            5. 预算计算；
            6. 逐日日期与天气适配；
            7. Evidence 来源和引用；
            8. 用户特殊要求处理；
            9. 执行边界和危险声明；
            10. proposal_id 内容完整性。
        """

        issues: list[VerificationIssue] = []
        checks: dict[str, VerificationCheckResult] = {}

        parsed_proposal: TripProposal | None = None
        expected_locked_context: dict[str, Any] | None = None

        # 1. 先做最终 Schema 校验。
        #    如果 Schema 都无法解析，后续字段检查没有可靠对象可读。
        try:
            parsed_proposal = TripProposal(**dict(proposal))
        except ValidationError as exc:
            self._add_issue(
                issues,
                issue_type="proposal_schema_invalid",
                severity="error",
                repair_target="proposal",
                check_name="schema",
                message="最终 Proposal 未通过 TripProposal Pydantic Schema。",
                path="proposal",
                details={
                    "validation_error": self._short_text(str(exc)),
                },
            )

        self._record_check(
            checks=checks,
            issues=issues,
            check_name="schema",
            summary=(
                "最终 Proposal 通过 Pydantic Schema。"
                if parsed_proposal is not None
                else "最终 Proposal Schema 无效。"
            ),
        )

        # 2. 独立重建锁定事实。
        #    这里直接从 State 的确定性上游重新计算，而不是相信 Proposal 中的值。
        try:
            expected_locked_context = build_locked_proposal_context(
                trip_request=trip_request,
                planning_context=planning_context,
                selection_result=selection_result,
                weather_result=weather_result,
            )
        except Exception as exc:
            self._add_issue(
                issues,
                issue_type="locked_context_rebuild_failed",
                severity="critical",
                repair_target="final_response",
                check_name="locked_facts",
                message=(
                    "Verifier 无法从当前 State 重建锁定事实；"
                    "上游状态合同可能已经损坏。"
                ),
                path="state",
                details={
                    "error": self._short_text(str(exc)),
                },
            )

        # 3. 只有 Schema 和锁定上下文都存在时，才执行完整检查。
        if parsed_proposal is not None and expected_locked_context is not None:
            self._run_check(
                checks,
                issues,
                "proposal_result",
                "检查 ProposalGenerator 运行摘要是否与最终 Proposal 一致。",
                lambda: self._check_proposal_result(
                    proposal=parsed_proposal,
                    proposal_result=proposal_result,
                    issues=issues,
                ),
            )

            self._run_check(
                checks,
                issues,
                "selection_lineage",
                "检查选中航班和酒店是否来自 CandidateRank 候选池与 selection_result。",
                lambda: self._check_selection_lineage(
                    proposal=parsed_proposal,
                    selection_result=selection_result,
                    flight_candidates=flight_candidates,
                    hotel_candidates=hotel_candidates,
                    issues=issues,
                ),
            )

            self._run_check(
                checks,
                issues,
                "locked_facts",
                "检查航班、酒店、天气、日期和预算锁定事实。",
                lambda: self._check_locked_facts(
                    proposal=parsed_proposal,
                    expected_locked_context=expected_locked_context,
                    issues=issues,
                ),
            )

            self._run_check(
                checks,
                issues,
                "budget",
                "检查已知 Mock 费用、总预算和剩余预算计算。",
                lambda: self._check_budget_consistency(
                    proposal=parsed_proposal,
                    selection_result=selection_result,
                    issues=issues,
                ),
            )

            self._run_check(
                checks,
                issues,
                "itinerary_weather",
                "检查逐日日期、天气事实和风险日活动调整。",
                lambda: self._check_itinerary_weather(
                    proposal=parsed_proposal,
                    expected_locked_context=expected_locked_context,
                    evidence_pool=evidence_pool,
                    issues=issues,
                ),
            )

            self._run_check(
                checks,
                issues,
                "evidence_provenance",
                "检查 Evidence 白名单、最终 sources 和酒店/安全/清单来源。",
                lambda: self._check_evidence_provenance(
                    proposal=parsed_proposal,
                    evidence_pool=evidence_pool,
                    evidence_result=evidence_result,
                    issues=issues,
                ),
            )

            self._run_check(
                checks,
                issues,
                "tool_activity_grounding",
                "检查工具活动 ID、名称、城市和可用日期是否来自本轮 MCP 结果。",
                lambda: self._check_tool_activity_grounding(
                    proposal=parsed_proposal,
                    planning_context=planning_context,
                    issues=issues,
                ),
            )

            self._run_check(
                checks,
                issues,
                "requirements",
                "检查指定实体和未映射需求是否逐项处理。",
                lambda: self._check_requirement_handling(
                    proposal=parsed_proposal,
                    planning_context=planning_context,
                    evidence_pool=evidence_pool,
                    issues=issues,
                ),
            )

            self._run_check(
                checks,
                issues,
                "execution_boundary",
                "检查没有真实预订、付款、出票或通知声明。",
                lambda: self._check_execution_boundary(
                    proposal=parsed_proposal,
                    issues=issues,
                ),
            )

            self._run_check(
                checks,
                issues,
                "proposal_integrity",
                "根据最终内容重新计算 proposal_id。",
                lambda: self._check_proposal_integrity(
                    proposal=parsed_proposal,
                    issues=issues,
                ),
            )
        else:
            # 4. 没有可解析 Proposal 时，补齐未执行检查组摘要。
            for check_name, summary in {
                "proposal_result": "因 Proposal Schema 或锁定上下文无效而跳过。",
                "selection_lineage": "因 Proposal Schema 或锁定上下文无效而跳过。",
                "locked_facts": "因 Proposal Schema 或锁定上下文无效而跳过。",
                "budget": "因 Proposal Schema 或锁定上下文无效而跳过。",
                "itinerary_weather": "因 Proposal Schema 或锁定上下文无效而跳过。",
                "evidence_provenance": "因 Proposal Schema 或锁定上下文无效而跳过。",
                "tool_activity_grounding": "因 Proposal Schema 或锁定上下文无效而跳过。",
                "requirements": "因 Proposal Schema 或锁定上下文无效而跳过。",
                "execution_boundary": "因 Proposal Schema 或锁定上下文无效而跳过。",
                "proposal_integrity": "因 Proposal Schema 或锁定上下文无效而跳过。",
            }.items():
                checks.setdefault(
                    check_name,
                    VerificationCheckResult(
                        check_name=check_name,
                        passed=False,
                        issue_count=0,
                        warning_count=0,
                        error_count=0,
                        summary=summary,
                    ),
                )

        # 5. 根据问题类型、上游重试次数和修复优先级选择下一节点。
        next_action, repairable, status = self._decide_next_action(
            issues=issues,
            proposal_retry_count=proposal_retry_count,
            budget_retry_count=budget_retry_count,
            retrieval_plan=retrieval_plan,
            rag_retry_count=rag_retry_count,
        )

        repair_feedback = self._build_repair_feedback(
            issues=issues,
            next_action=next_action,
            proposal=(parsed_proposal or proposal),
            proposal_retry_count=proposal_retry_count,
        )

        warning_count = sum(
            issue.severity == "warning"
            for issue in issues
        )
        error_count = sum(
            issue.severity == "error"
            for issue in issues
        )
        critical_count = sum(
            issue.severity == "critical"
            for issue in issues
        )

        result = ProposalVerificationResult(
            status=status,
            passed=(next_action == "decision_gate"),
            repairable=repairable,
            next_action=next_action,
            proposal_id=(
                parsed_proposal.proposal_id
                if parsed_proposal is not None
                else str(proposal.get("proposal_id") or "") or None
            ),
            expected_locked_fact_hash=(
                str(expected_locked_context["locked_fact_hash"])
                if expected_locked_context is not None
                else None
            ),
            actual_locked_fact_hash=(
                parsed_proposal.locked_fact_hash
                if parsed_proposal is not None
                else str(proposal.get("locked_fact_hash") or "") or None
            ),
            proposal_retry_count=max(proposal_retry_count, 0),
            max_proposal_repairs=self.config.max_proposal_repairs,
            budget_retry_count=max(budget_retry_count, 0),
            max_budget_repairs=self.config.max_budget_repairs,
            approval_required=(next_action == "decision_gate"),
            check_count=len(checks),
            passed_check_count=sum(
                item.passed
                for item in checks.values()
            ),
            warning_count=warning_count,
            error_count=error_count,
            critical_count=critical_count,
            checks=checks,
            issues=issues[: self.config.max_issue_examples],
            repair_feedback=repair_feedback,
        )

        return {
            "verifier_result": result.to_state_dict(),
            "proposal_feedback": (
                repair_feedback
                if next_action == "proposal"
                else {}
            ),
            "pending_actions": (
                self._build_pending_actions(parsed_proposal)
                if next_action == "decision_gate"
                and parsed_proposal is not None
                else []
            ),
        }

    # ==================================================================
    # 各检查组
    # ==================================================================

    def _check_proposal_result(
        self,
        *,
        proposal: TripProposal,
        proposal_result: Mapping[str, Any],
        issues: list[VerificationIssue],
    ) -> None:
        """检查 ProposalGenerator 的运行摘要是否与最终 Proposal 对齐。"""

        if proposal_result.get("status") != "generated":
            self._add_issue(
                issues,
                issue_type="proposal_result_not_generated",
                severity="error",
                repair_target="proposal",
                check_name="proposal_result",
                message="proposal_result.status 不是 generated。",
                path="proposal_result.status",
                expected="generated",
                actual=proposal_result.get("status"),
            )

        expected_pairs = {
            "proposal_id": proposal.proposal_id,
            "locked_fact_hash": proposal.locked_fact_hash,
            "daily_plan_count": len(proposal.daily_plan),
            "model_id": proposal.model_id,
            "generator_version": proposal.generator_version,
            "prompt_version": proposal.prompt_version,
        }

        for field_name, expected_value in expected_pairs.items():
            actual_value = proposal_result.get(field_name)

            if actual_value != expected_value:
                self._add_issue(
                    issues,
                    issue_type="proposal_result_mismatch",
                    severity="error",
                    repair_target="proposal",
                    check_name="proposal_result",
                    message=f"proposal_result.{field_name} 与最终 Proposal 不一致。",
                    path=f"proposal_result.{field_name}",
                    expected=expected_value,
                    actual=actual_value,
                )

        used_refs = sorted(self._collect_proposal_evidence_refs(proposal))
        result_refs = sorted(
            str(item)
            for item in proposal_result.get("used_evidence_refs", [])
        )

        if result_refs != used_refs:
            self._add_issue(
                issues,
                issue_type="proposal_result_evidence_mismatch",
                severity="error",
                repair_target="proposal",
                check_name="proposal_result",
                message="proposal_result.used_evidence_refs 与最终 Proposal 使用的引用不一致。",
                path="proposal_result.used_evidence_refs",
                expected=used_refs,
                actual=result_refs,
            )

        if proposal_result.get("used_evidence_count") != len(used_refs):
            self._add_issue(
                issues,
                issue_type="proposal_result_evidence_count_mismatch",
                severity="error",
                repair_target="proposal",
                check_name="proposal_result",
                message="proposal_result.used_evidence_count 不正确。",
                path="proposal_result.used_evidence_count",
                expected=len(used_refs),
                actual=proposal_result.get("used_evidence_count"),
            )

    def _check_selection_lineage(
        self,
        *,
        proposal: TripProposal,
        selection_result: Mapping[str, Any],
        flight_candidates: Sequence[Mapping[str, Any]],
        hotel_candidates: Sequence[Mapping[str, Any]],
        issues: list[VerificationIssue],
    ) -> None:
        """确认最终航班和酒店仍来自 CandidateRank 与 BudgetOptimize。"""

        if selection_result.get("status") != "feasible":
            self._add_issue(
                issues,
                issue_type="selection_not_feasible",
                severity="critical",
                repair_target="budget_optimize",
                check_name="selection_lineage",
                message="selection_result 已不再是 feasible。",
                path="selection_result.status",
                expected="feasible",
                actual=selection_result.get("status"),
            )

        outbound_ids = {
            str(item.get("flight_id"))
            for item in flight_candidates
            if isinstance(item, Mapping)
            and item.get("direction") == "outbound"
            and item.get("flight_id")
        }
        return_ids = {
            str(item.get("flight_id"))
            for item in flight_candidates
            if isinstance(item, Mapping)
            and item.get("direction") == "return"
            and item.get("flight_id")
        }
        hotel_ids = {
            str(item.get("hotel_id"))
            for item in hotel_candidates
            if isinstance(item, Mapping)
            and item.get("hotel_id")
        }

        actual_ids = {
            "outbound_flight_id": proposal.selected_flights.outbound.flight_id,
            "return_flight_id": proposal.selected_flights.return_flight.flight_id,
            "hotel_id": proposal.selected_hotel.hotel_id,
        }

        membership_rules = {
            "outbound_flight_id": outbound_ids,
            "return_flight_id": return_ids,
            "hotel_id": hotel_ids,
        }

        for field_name, allowed_ids in membership_rules.items():
            if actual_ids[field_name] not in allowed_ids:
                self._add_issue(
                    issues,
                    issue_type="selected_candidate_not_in_ranked_pool",
                    severity="critical",
                    repair_target="budget_optimize",
                    check_name="selection_lineage",
                    message=f"{field_name} 不在 CandidateRank 候选池中。",
                    path=f"proposal.{field_name}",
                    expected=sorted(allowed_ids),
                    actual=actual_ids[field_name],
                )

        combination = selection_result.get("selected_combination")
        combination = combination if isinstance(combination, Mapping) else {}

        for field_name, actual_id in actual_ids.items():
            expected_id = str(combination.get(field_name) or "")

            if expected_id != actual_id:
                self._add_issue(
                    issues,
                    issue_type="selected_combination_id_mismatch",
                    severity="critical",
                    repair_target="budget_optimize",
                    check_name="selection_lineage",
                    message=f"Proposal 的 {field_name} 与 BudgetOptimize 结果不一致。",
                    path=f"selection_result.selected_combination.{field_name}",
                    expected=expected_id,
                    actual=actual_id,
                )

    def _check_locked_facts(
        self,
        *,
        proposal: TripProposal,
        expected_locked_context: Mapping[str, Any],
        issues: list[VerificationIssue],
    ) -> None:
        """逐项比较 Proposal 锁定事实和重新计算的权威值。"""

        expected_hash = str(expected_locked_context["locked_fact_hash"])

        if proposal.locked_fact_hash != expected_hash:
            self._add_issue(
                issues,
                issue_type="locked_fact_hash_mismatch",
                severity="critical",
                repair_target="proposal",
                check_name="locked_facts",
                message="Proposal.locked_fact_hash 与当前 State 重算结果不一致。",
                path="proposal.locked_fact_hash",
                expected=expected_hash,
                actual=proposal.locked_fact_hash,
            )

        comparisons = {
            "trip_overview": (
                self._model_to_dict(expected_locked_context["trip_overview"]),
                self._model_to_dict(proposal.trip_overview),
            ),
            "selected_flights": (
                self._model_to_dict(expected_locked_context["selected_flights"]),
                self._model_to_dict(proposal.selected_flights),
            ),
            "selected_hotel": (
                self._model_to_dict(expected_locked_context["selected_hotel"]),
                self._model_to_dict(proposal.selected_hotel),
            ),
            "weather_overview": (
                self._model_to_dict(expected_locked_context["weather_overview"]),
                self._model_to_dict(proposal.weather_overview),
            ),
            "budget_summary": (
                self._model_to_dict(expected_locked_context["budget_summary"]),
                self._model_to_dict(proposal.budget_summary),
            ),
        }

        for field_name, (expected_value, actual_value) in comparisons.items():
            if not self._json_equal(expected_value, actual_value):
                self._add_issue(
                    issues,
                    issue_type="locked_fact_value_mismatch",
                    severity="critical",
                    repair_target="proposal",
                    check_name="locked_facts",
                    message=f"Proposal.{field_name} 与当前权威 State 不一致。",
                    path=f"proposal.{field_name}",
                    expected=self._compact_value(expected_value),
                    actual=self._compact_value(actual_value),
                )

    def _check_budget_consistency(
        self,
        *,
        proposal: TripProposal,
        selection_result: Mapping[str, Any],
        issues: list[VerificationIssue],
    ) -> None:
        """独立检查费用加法和预算余额，避免只依赖上游 status。"""

        costs = proposal.budget_summary.known_costs

        outbound_cost = self._safe_float(costs.get("outbound_flight"))
        return_cost = self._safe_float(costs.get("return_flight"))
        transport_total = self._safe_float(costs.get("transport_total"))
        hotel_cost = self._safe_float(costs.get("hotel"))
        known_subtotal = self._safe_float(costs.get("known_subtotal"))

        expected_transport = outbound_cost + return_cost
        expected_subtotal = expected_transport + hotel_cost

        if not self._close(transport_total, expected_transport):
            self._add_issue(
                issues,
                issue_type="transport_cost_arithmetic_error",
                severity="critical",
                repair_target="budget_optimize",
                check_name="budget",
                message="transport_total 不等于去程与返程航班费用之和。",
                path="proposal.budget_summary.known_costs.transport_total",
                expected=round(expected_transport, 2),
                actual=transport_total,
            )

        if not self._close(known_subtotal, expected_subtotal):
            self._add_issue(
                issues,
                issue_type="known_subtotal_arithmetic_error",
                severity="critical",
                repair_target="budget_optimize",
                check_name="budget",
                message="known_subtotal 不等于往返交通和酒店费用之和。",
                path="proposal.budget_summary.known_costs.known_subtotal",
                expected=round(expected_subtotal, 2),
                actual=known_subtotal,
            )

        # 1. 已知费用必须与锁定航班和酒店本身一致。
        fact_costs = {
            "outbound_flight": proposal.selected_flights.outbound.total_price,
            "return_flight": proposal.selected_flights.return_flight.total_price,
            "hotel": proposal.selected_hotel.estimated_total_price,
        }

        for field_name, expected_value in fact_costs.items():
            actual_value = self._safe_float(costs.get(field_name))

            if not self._close(actual_value, expected_value):
                self._add_issue(
                    issues,
                    issue_type="known_cost_fact_mismatch",
                    severity="critical",
                    repair_target="budget_optimize",
                    check_name="budget",
                    message=f"预算中的 {field_name} 与选中事实价格不一致。",
                    path=f"proposal.budget_summary.known_costs.{field_name}",
                    expected=expected_value,
                    actual=actual_value,
                )

        # 2. budget_limited 模式必须满足总预算硬约束。
        if proposal.budget_summary.budget_mode == "budget_limited":
            total_budget = proposal.budget_summary.total_budget
            remaining_budget = proposal.budget_summary.remaining_budget

            if total_budget is None or remaining_budget is None:
                self._add_issue(
                    issues,
                    issue_type="limited_budget_fields_missing",
                    severity="critical",
                    repair_target="budget_optimize",
                    check_name="budget",
                    message="budget_limited 模式缺少 total_budget 或 remaining_budget。",
                    path="proposal.budget_summary",
                )
            else:
                if known_subtotal - total_budget > self.config.float_tolerance:
                    self._add_issue(
                        issues,
                        issue_type="total_budget_violation",
                        severity="critical",
                        repair_target="budget_optimize",
                        check_name="budget",
                        message="已知航班和酒店费用超过用户总预算。",
                        path="proposal.budget_summary.known_costs.known_subtotal",
                        expected=f"<= {total_budget}",
                        actual=known_subtotal,
                    )

                expected_remaining = total_budget - known_subtotal

                if not self._close(remaining_budget, expected_remaining):
                    self._add_issue(
                        issues,
                        issue_type="remaining_budget_mismatch",
                        severity="critical",
                        repair_target="budget_optimize",
                        check_name="budget",
                        message="remaining_budget 不等于 total_budget - known_subtotal。",
                        path="proposal.budget_summary.remaining_budget",
                        expected=round(expected_remaining, 2),
                        actual=remaining_budget,
                    )

                allocation = proposal.budget_summary.remaining_budget_allocation

                if isinstance(allocation, Mapping):
                    food_activity = self._safe_float(
                        allocation.get("food_activity"),
                        default=0.0,
                    )
                    buffer_amount = self._safe_float(
                        allocation.get("local_transport_and_buffer"),
                        default=0.0,
                    )

                    if not self._close(
                        food_activity + buffer_amount,
                        remaining_budget,
                    ):
                        self._add_issue(
                            issues,
                            issue_type="remaining_allocation_mismatch",
                            severity="critical",
                            repair_target="budget_optimize",
                            check_name="budget",
                            message=(
                                "food_activity 与 local_transport_and_buffer "
                                "之和不等于 remaining_budget。"
                            ),
                            path="proposal.budget_summary.remaining_budget_allocation",
                            expected=remaining_budget,
                            actual=round(food_activity + buffer_amount, 2),
                        )

        # 3. Proposal 预算和 selection_result 顶层余额也必须一致。
        selection_remaining = selection_result.get("remaining_budget")
        proposal_remaining = proposal.budget_summary.remaining_budget

        if (
            selection_remaining is not None
            or proposal_remaining is not None
        ) and not self._close(
            self._safe_float(selection_remaining, default=-1.0),
            self._safe_float(proposal_remaining, default=-2.0),
        ):
            self._add_issue(
                issues,
                issue_type="selection_budget_mismatch",
                severity="critical",
                repair_target="budget_optimize",
                check_name="budget",
                message="Proposal 剩余预算与 selection_result 不一致。",
                path="proposal.budget_summary.remaining_budget",
                expected=selection_remaining,
                actual=proposal_remaining,
            )

    def _check_itinerary_weather(
        self,
        *,
        proposal: TripProposal,
        expected_locked_context: Mapping[str, Any],
        evidence_pool: Sequence[Mapping[str, Any]],
        issues: list[VerificationIssue],
    ) -> None:
        """检查逐日行程是否真正使用了天气和攻略 Evidence。"""

        expected_dates = list(expected_locked_context["expected_dates"])
        actual_dates = [
            day.date.isoformat()
            for day in proposal.daily_plan
        ]

        if actual_dates != expected_dates:
            self._add_issue(
                issues,
                issue_type="daily_dates_mismatch",
                severity="error",
                repair_target="proposal",
                check_name="itinerary_weather",
                message="daily_plan 日期与旅行日期不完全一致。",
                path="proposal.daily_plan",
                expected=expected_dates,
                actual=actual_dates,
            )

        weather_by_date = {
            item.date.isoformat(): item
            for item in expected_locked_context["weather_overview"].daily
        }
        evidence_by_ref = self._evidence_by_ref(evidence_pool)

        for day in proposal.daily_plan:
            date_key = day.date.isoformat()
            expected_weather = weather_by_date.get(date_key)

            if expected_weather is None:
                continue

            if not self._json_equal(
                self._model_to_dict(expected_weather),
                self._model_to_dict(day.weather),
            ):
                self._add_issue(
                    issues,
                    issue_type="daily_weather_mismatch",
                    severity="error",
                    repair_target="proposal",
                    check_name="itinerary_weather",
                    message=f"{date_key} 的逐日天气与 WeatherNode 锁定事实不一致。",
                    path=f"proposal.daily_plan[{date_key}].weather",
                )

            risk_types = set(expected_weather.risk_types)

            if risk_types & WEATHER_RISKS_REQUIRING_ADJUSTMENT:
                if not day.weather_adjustment or not day.weather_adjustment.strip():
                    self._add_issue(
                        issues,
                        issue_type="weather_adjustment_missing",
                        severity="error",
                        repair_target="proposal",
                        check_name="itinerary_weather",
                        message=f"{date_key} 存在天气风险，但没有 weather_adjustment。",
                        path=f"proposal.daily_plan[{date_key}].weather_adjustment",
                        expected="非空字符串",
                        actual=day.weather_adjustment,
                    )

            if risk_types & RAIN_RISK_TYPES:
                non_transfer = [
                    item
                    for item in day.activities
                    if item.indoor_outdoor != "transfer"
                ]

                if non_transfer and all(
                    item.indoor_outdoor == "outdoor"
                    for item in non_transfer
                ):
                    self._add_issue(
                        issues,
                        issue_type="rain_day_all_outdoor",
                        severity="error",
                        repair_target="proposal",
                        check_name="itinerary_weather",
                        message=f"{date_key} 是雨天，但所有非交通活动都是 outdoor。",
                        path=f"proposal.daily_plan[{date_key}].activities",
                    )

            # 1. 知识型活动至少引用一个 guides Evidence。
            for activity in day.activities:
                if activity.activity_type not in KNOWLEDGE_ACTIVITY_TYPES:
                    continue

                categories = {
                    str(
                        evidence_by_ref.get(ref, {}).get("category")
                        or ""
                    )
                    for ref in activity.evidence_refs
                }

                if "guides" not in categories:
                    self._add_issue(
                        issues,
                        issue_type="knowledge_activity_missing_guide",
                        severity="error",
                        repair_target="proposal",
                        check_name="itinerary_weather",
                        message=(
                            f"{date_key} 的知识型活动“{activity.title}”"
                            "没有引用 guides Evidence。"
                        ),
                        path=f"proposal.daily_plan[{date_key}].activities",
                    )

    def _check_evidence_provenance(
        self,
        *,
        proposal: TripProposal,
        evidence_pool: Sequence[Mapping[str, Any]],
        evidence_result: Mapping[str, Any],
        issues: list[VerificationIssue],
    ) -> None:
        """检查所有引用、sources 和特殊类别 Evidence 的一致性。"""

        if evidence_result.get("passed") is not True:
            self._add_issue(
                issues,
                issue_type="evidence_grade_not_passed",
                severity="critical",
                repair_target="retrieval_plan",
                check_name="evidence_provenance",
                message="当前 evidence_result 已不再是 passed。",
                path="evidence_result.passed",
                expected=True,
                actual=evidence_result.get("passed"),
            )

        evidence_by_ref = self._evidence_by_ref(evidence_pool)
        allowed_refs = set(evidence_by_ref)
        used_refs = self._collect_proposal_evidence_refs(proposal)
        unknown_refs = sorted(used_refs - allowed_refs)

        if unknown_refs:
            self._add_issue(
                issues,
                issue_type="unknown_evidence_ref",
                severity="error",
                repair_target="proposal",
                check_name="evidence_provenance",
                message="Proposal 使用了 evidence_pool 中不存在的 chunk_id。",
                path="proposal.evidence_refs",
                expected="evidence_pool 中存在的 chunk_id",
                actual=unknown_refs,
            )

        source_ids = [
            item.chunk_id
            for item in proposal.sources
        ]

        if len(source_ids) != len(set(source_ids)):
            self._add_issue(
                issues,
                issue_type="duplicate_source_ref",
                severity="error",
                repair_target="proposal",
                check_name="evidence_provenance",
                message="proposal.sources 中存在重复 chunk_id。",
                path="proposal.sources",
            )

        source_set = set(source_ids)
        missing_sources = sorted(used_refs - source_set)
        extra_sources = sorted(source_set - used_refs)

        if missing_sources or extra_sources:
            self._add_issue(
                issues,
                issue_type="source_catalog_mismatch",
                severity="error",
                repair_target="proposal",
                check_name="evidence_provenance",
                message="proposal.sources 必须与实际使用的 evidence_refs 完全一致。",
                path="proposal.sources",
                expected=sorted(used_refs),
                actual=sorted(source_set),
                details={
                    "missing_sources": missing_sources,
                    "extra_sources": extra_sources,
                },
            )

        # 1. 逐条核对最终 source 的来源和 metadata。
        for source in proposal.sources:
            evidence = evidence_by_ref.get(source.chunk_id)

            if evidence is None:
                continue

            metadata = evidence.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}

            expected_fields = {
                "category": str(evidence.get("category") or "unknown"),
                "source": str(evidence.get("source") or "unknown"),
                "title": self._optional_text(metadata.get("title")),
                "section": self._optional_text(metadata.get("section")),
                "doc_type": self._optional_text(metadata.get("doc_type")),
                "hotel_id": self._optional_text(metadata.get("hotel_id")),
                "risk_type": metadata.get("risk_type"),
            }

            actual_fields = {
                "category": source.category,
                "source": source.source,
                "title": source.title,
                "section": source.section,
                "doc_type": source.doc_type,
                "hotel_id": source.hotel_id,
                "risk_type": source.risk_type,
            }

            if not self._json_equal(expected_fields, actual_fields):
                self._add_issue(
                    issues,
                    issue_type="source_metadata_mismatch",
                    severity="error",
                    repair_target="proposal",
                    check_name="evidence_provenance",
                    message=f"source={source.chunk_id} 与 evidence_pool metadata 不一致。",
                    path=f"proposal.sources[{source.chunk_id}]",
                    expected=self._compact_value(expected_fields),
                    actual=self._compact_value(actual_fields),
                )

            expected_score = evidence.get("rerank_score")
            actual_score = source.rerank_score

            # 2. rerank_score 的“是否存在”和具体数值都必须一致。
            score_presence_mismatch = (
                (expected_score is None)
                != (actual_score is None)
            )

            score_value_mismatch = (
                expected_score is not None
                and actual_score is not None
                and not self._close(
                    float(expected_score),
                    float(actual_score),
                )
            )

            if score_presence_mismatch or score_value_mismatch:
                self._add_issue(
                    issues,
                    issue_type="source_rerank_score_mismatch",
                    severity="error",
                    repair_target="proposal",
                    check_name="evidence_provenance",
                    message=f"source={source.chunk_id} 的 rerank_score 不一致。",
                    path=f"proposal.sources[{source.chunk_id}].rerank_score",
                    expected=expected_score,
                    actual=actual_score,
                )

        # 2. 酒店补充信息只能引用选中酒店的 hotel_reviews。
        selected_hotel_id = proposal.selected_hotel.hotel_id
        selected_hotel_refs = {
            ref
            for ref, item in evidence_by_ref.items()
            if item.get("category") == "hotel_reviews"
            and str(
                (item.get("metadata") or {}).get("hotel_id")
            ) == selected_hotel_id
        }

        hotel_context_refs = set(proposal.hotel_context.evidence_refs)

        if selected_hotel_refs:
            if not proposal.hotel_context.available:
                self._add_issue(
                    issues,
                    issue_type="hotel_context_not_generated",
                    severity="error",
                    repair_target="proposal",
                    check_name="evidence_provenance",
                    message="存在选中酒店的 RAG 证据，但 hotel_context.available=false。",
                    path="proposal.hotel_context.available",
                    expected=True,
                    actual=False,
                )

            if not hotel_context_refs or not hotel_context_refs <= selected_hotel_refs:
                self._add_issue(
                    issues,
                    issue_type="hotel_context_evidence_invalid",
                    severity="error",
                    repair_target="proposal",
                    check_name="evidence_provenance",
                    message="hotel_context 只能引用选中酒店的 hotel_reviews。",
                    path="proposal.hotel_context.evidence_refs",
                    expected=sorted(selected_hotel_refs),
                    actual=sorted(hotel_context_refs),
                )
        elif proposal.hotel_context.available or hotel_context_refs:
            self._add_issue(
                issues,
                issue_type="hotel_context_without_evidence",
                severity="error",
                repair_target="proposal",
                check_name="evidence_provenance",
                message="当前没有选中酒店 RAG 证据，不能生成可用 hotel_context。",
                path="proposal.hotel_context",
            )

        # 3. 有天气风险时必须有安全提醒；有安全证据时至少引用一条。
        all_risks = set(proposal.weather_overview.risk_types)
        safety_refs = {
            ref
            for ref, item in evidence_by_ref.items()
            if item.get("category") == "safety_notices"
        }

        if all_risks:
            if not proposal.safety_notes:
                self._add_issue(
                    issues,
                    issue_type="safety_notes_missing",
                    severity="error",
                    repair_target="proposal",
                    check_name="evidence_provenance",
                    message="存在天气风险，但 Proposal 没有 safety_notes。",
                    path="proposal.safety_notes",
                )

            used_safety_refs = {
                ref
                for note in proposal.safety_notes
                for ref in note.evidence_refs
            }

            if safety_refs and not used_safety_refs & safety_refs:
                self._add_issue(
                    issues,
                    issue_type="safety_evidence_not_used",
                    severity="error",
                    repair_target="proposal",
                    check_name="evidence_provenance",
                    message="存在 safety_notices Evidence，但安全提醒没有引用。",
                    path="proposal.safety_notes",
                    expected=sorted(safety_refs),
                    actual=sorted(used_safety_refs),
                )

        # 4. 有 packing_checklists 时必须形成至少一条准备建议。
        packing_refs = {
            ref
            for ref, item in evidence_by_ref.items()
            if item.get("category") == "packing_checklists"
        }

        if packing_refs:
            used_packing_refs = {
                ref
                for note in proposal.packing_tips
                for ref in note.evidence_refs
            }

            if not proposal.packing_tips or not used_packing_refs & packing_refs:
                self._add_issue(
                    issues,
                    issue_type="packing_evidence_not_used",
                    severity="error",
                    repair_target="proposal",
                    check_name="evidence_provenance",
                    message="存在 packing_checklists Evidence，但没有生成带引用的准备建议。",
                    path="proposal.packing_tips",
                    expected=sorted(packing_refs),
                    actual=sorted(used_packing_refs),
                )

    def _check_tool_activity_grounding(
        self,
        *,
        proposal: TripProposal,
        planning_context: Mapping[str, Any],
        issues: list[VerificationIssue],
    ) -> None:
        """防止 Proposal 编造 activity_id，或把真实候选安排到错误日期。"""

        raw_candidates = planning_context.get("activity_candidates")
        raw_candidates = raw_candidates if isinstance(raw_candidates, Sequence) else []
        by_id = {
            str(item.get("activity_id")): item
            for item in raw_candidates
            if isinstance(item, Mapping) and item.get("activity_id")
        }
        used_ids: set[str] = set()
        for day in proposal.daily_plan:
            for activity in day.activities:
                if not activity.tool_activity_id:
                    continue
                activity_id = activity.tool_activity_id
                used_ids.add(activity_id)
                candidate = by_id.get(activity_id)
                if not isinstance(candidate, Mapping):
                    self._add_issue(
                        issues,
                        issue_type="unknown_tool_activity_id",
                        severity="error",
                        repair_target="proposal",
                        check_name="tool_activity_grounding",
                        message="行程引用了本轮活动工具结果中不存在的 activity_id。",
                        path=f"proposal.daily_plan[{day.day_index}].activities",
                        actual=activity_id,
                    )
                    continue
                expected_date = day.date.isoformat()
                if expected_date not in set(candidate.get("available_dates", [])):
                    self._add_issue(
                        issues,
                        issue_type="tool_activity_date_mismatch",
                        severity="error",
                        repair_target="proposal",
                        check_name="tool_activity_grounding",
                        message="活动安排日期不在工具返回的 available_dates 中。",
                        path=f"proposal.daily_plan[{day.day_index}].activities",
                        expected=candidate.get("available_dates", []),
                        actual=expected_date,
                    )
                if activity.title != str(candidate.get("name") or ""):
                    self._add_issue(
                        issues,
                        issue_type="tool_activity_name_mismatch",
                        severity="error",
                        repair_target="proposal",
                        check_name="tool_activity_grounding",
                        message="活动标题与对应 activity_id 的工具事实不一致。",
                        path=f"proposal.daily_plan[{day.day_index}].activities",
                        expected=candidate.get("name"),
                        actual=activity.title,
                    )

        source_ids = {source.activity_id for source in proposal.activity_sources}
        if source_ids != used_ids:
            self._add_issue(
                issues,
                issue_type="tool_activity_sources_mismatch",
                severity="error",
                repair_target="proposal",
                check_name="tool_activity_grounding",
                message="Proposal.activity_sources 与 daily_plan 使用的活动工具 ID 不一致。",
                path="proposal.activity_sources",
                expected=sorted(used_ids),
                actual=sorted(source_ids),
            )

    def _check_requirement_handling(
        self,
        *,
        proposal: TripProposal,
        planning_context: Mapping[str, Any],
        evidence_pool: Sequence[Mapping[str, Any]],
        issues: list[VerificationIssue],
    ) -> None:
        """检查每个特殊用户要求是否恰好处理一次。"""

        catalog = build_requirement_catalog(
            planning_context=planning_context,
        )
        expected_keys = {
            str(item["requirement_key"])
            for item in catalog
        }
        actual_keys = [
            item.requirement_key
            for item in proposal.requirement_handling
        ]

        if len(actual_keys) != len(set(actual_keys)):
            self._add_issue(
                issues,
                issue_type="duplicate_requirement_key",
                severity="error",
                repair_target="proposal",
                check_name="requirements",
                message="requirement_handling 中存在重复 requirement_key。",
                path="proposal.requirement_handling",
            )

        actual_key_set = set(actual_keys)

        if actual_key_set != expected_keys:
            self._add_issue(
                issues,
                issue_type="requirement_catalog_not_covered",
                severity="error",
                repair_target="proposal",
                check_name="requirements",
                message="最终 Proposal 没有完整覆盖 requirement_catalog。",
                path="proposal.requirement_handling",
                expected=sorted(expected_keys),
                actual=sorted(actual_key_set),
                details={
                    "missing": sorted(expected_keys - actual_key_set),
                    "extra": sorted(actual_key_set - expected_keys),
                },
            )

        allowed_refs = set(self._evidence_by_ref(evidence_pool))

        for item in proposal.requirement_handling:
            unknown_refs = sorted(
                set(item.evidence_refs) - allowed_refs
            )

            if unknown_refs:
                self._add_issue(
                    issues,
                    issue_type="requirement_unknown_evidence_ref",
                    severity="error",
                    repair_target="proposal",
                    check_name="requirements",
                    message=(
                        f"requirement_key={item.requirement_key} "
                        "引用了未知 Evidence。"
                    ),
                    path=(
                        "proposal.requirement_handling."
                        f"{item.requirement_key}.evidence_refs"
                    ),
                    actual=unknown_refs,
                )

    def _check_execution_boundary(
        self,
        *,
        proposal: TripProposal,
        issues: list[VerificationIssue],
    ) -> None:
        """检查固定执行边界，并扫描用户可见文本中的危险完成式声明。"""

        boundary = proposal.execution_boundary

        expected_boundary = {
            "real_booking_performed": False,
            "real_payment_performed": False,
            "real_notification_sent": False,
            "draft_only": True,
        }
        actual_boundary = self._model_to_dict(boundary)

        if actual_boundary != expected_boundary:
            self._add_issue(
                issues,
                issue_type="execution_boundary_violation",
                severity="critical",
                repair_target="proposal",
                check_name="execution_boundary",
                message="Proposal.execution_boundary 违反项目 Draft-only 边界。",
                path="proposal.execution_boundary",
                expected=expected_boundary,
                actual=actual_boundary,
            )

        generated_texts = self._collect_generated_texts(proposal)

        for text_path, text in generated_texts:
            for issue_type, pattern in UNSAFE_ACTION_CLAIM_PATTERNS:
                match = pattern.search(text)

                if match is None:
                    continue

                self._add_issue(
                    issues,
                    issue_type=issue_type,
                    severity="critical",
                    repair_target="proposal",
                    check_name="execution_boundary",
                    message="用户可见文本声称系统已执行真实外部动作。",
                    path=text_path,
                    actual=match.group(0),
                    details={
                        "text_excerpt": self._short_text(text, 240),
                    },
                )

    def _check_proposal_integrity(
        self,
        *,
        proposal: TripProposal,
        issues: list[VerificationIssue],
    ) -> None:
        """根据最终 Proposal 内容重新计算稳定 proposal_id。"""

        if proposal.generator_version != PROPOSAL_GENERATOR_VERSION:
            self._add_issue(
                issues,
                issue_type="generator_version_mismatch",
                severity="error",
                repair_target="proposal",
                check_name="proposal_integrity",
                message="Proposal.generator_version 与当前生成器版本不一致。",
                path="proposal.generator_version",
                expected=PROPOSAL_GENERATOR_VERSION,
                actual=proposal.generator_version,
            )

        if proposal.prompt_version != PROPOSAL_PROMPT_VERSION:
            self._add_issue(
                issues,
                issue_type="prompt_version_mismatch",
                severity="warning",
                repair_target="none",
                check_name="proposal_integrity",
                message="Proposal.prompt_version 与当前 Prompt 版本不一致。",
                path="proposal.prompt_version",
                expected=PROPOSAL_PROMPT_VERSION,
                actual=proposal.prompt_version,
            )

        # 1. generated_at 至少必须是可解析的 ISO 时间。
        try:
            datetime.fromisoformat(
                proposal.generated_at.replace("Z", "+00:00")
            )
        except ValueError:
            self._add_issue(
                issues,
                issue_type="generated_at_invalid",
                severity="error",
                repair_target="proposal",
                check_name="proposal_integrity",
                message="Proposal.generated_at 不是合法 ISO 时间。",
                path="proposal.generated_at",
                actual=proposal.generated_at,
            )

        used_refs = sorted(
            self._collect_proposal_evidence_refs(proposal)
        )
        content_payload = {
            "summary": proposal.summary,
            "highlights": proposal.highlights,
            "daily_plan": [
                self._model_to_dict(item)
                for item in proposal.daily_plan
            ],
            "hotel_context": self._model_to_dict(
                proposal.hotel_context
            ),
            "packing_tips": [
                self._model_to_dict(item)
                for item in proposal.packing_tips
            ],
            "safety_notes": [
                self._model_to_dict(item)
                for item in proposal.safety_notes
            ],
            "preference_alignment": proposal.preference_alignment,
            "requirement_handling": [
                self._model_to_dict(item)
                for item in proposal.requirement_handling
            ],
            "limitations": proposal.limitations,
            "used_refs": used_refs,
            "activity_sources": [
                self._model_to_dict(item)
                for item in proposal.activity_sources
            ],
        }

        expected_id = "proposal_" + self._sha256_json(
            {
                "locked_fact_hash": proposal.locked_fact_hash,
                "proposal_version": proposal.version,
                "content": content_payload,
                "generator_version": PROPOSAL_GENERATOR_VERSION,
            }
        )[:20]

        if proposal.proposal_id != expected_id:
            self._add_issue(
                issues,
                issue_type="proposal_id_mismatch",
                severity="critical",
                repair_target="proposal",
                check_name="proposal_integrity",
                message="proposal_id 与最终 Proposal 内容重新计算结果不一致。",
                path="proposal.proposal_id",
                expected=expected_id,
                actual=proposal.proposal_id,
            )

    # ==================================================================
    # 路由与反馈
    # ==================================================================

    def _decide_next_action(
        self,
        *,
        issues: Sequence[VerificationIssue],
        proposal_retry_count: int,
        budget_retry_count: int,
        retrieval_plan: Mapping[str, Any] | None,
        rag_retry_count: int,
    ) -> tuple[str, bool, str]:
        """按上游优先级和重试保护决定下一节点。"""

        blocking = [
            issue
            for issue in issues
            if issue.severity in {"error", "critical"}
        ]

        if not blocking:
            return "decision_gate", False, "passed"

        targets = {
            issue.repair_target
            for issue in blocking
        }

        # 1. 无法自动修复的 State/程序合同问题优先停止。
        if "final_response" in targets:
            return "final_response", False, "failed"

        # 2. 选择组合无效时必须先回到 BudgetOptimize。
        #    如果先修 Proposal，RAG 和行程仍会基于错误组合。
        if "budget_optimize" in targets:
            if budget_retry_count < self.config.max_budget_repairs:
                return "budget_optimize", True, "repair_required"

            return "final_response", False, "failed"

        # 3. Evidence 状态失效时回到 RetrievalPlan / RAG repair。
        if "retrieval_plan" in targets:
            plan = retrieval_plan if isinstance(retrieval_plan, Mapping) else {}
            max_rag_repairs = self._safe_int(
                plan.get("max_repair_rounds"),
                default=1,
            )
            repair_exhausted = bool(
                plan.get("repair_exhausted", False)
            )

            if (
                not repair_exhausted
                and rag_retry_count < max_rag_repairs
            ):
                return "retrieval_plan", True, "repair_required"

            return "final_response", False, "failed"

        # 4. 模型内容或 Proposal 组装问题回到 ProposalNode。
        if "proposal" in targets:
            if proposal_retry_count < self.config.max_proposal_repairs:
                return "proposal", True, "repair_required"

            return "final_response", False, "failed"

        return "final_response", False, "failed"

    def _build_repair_feedback(
        self,
        *,
        issues: Sequence[VerificationIssue],
        next_action: str,
        proposal: Any,
        proposal_retry_count: int,
    ) -> dict[str, Any]:
        """把 Verifier 问题转换成下一轮可执行的结构化反馈。"""

        if next_action == "decision_gate":
            return {}

        if isinstance(proposal, TripProposal):
            proposal_id = proposal.proposal_id
        elif isinstance(proposal, Mapping):
            proposal_id = str(proposal.get("proposal_id") or "") or None
        else:
            proposal_id = None

        relevant = [
            issue
            for issue in issues
            if issue.severity in {"error", "critical"}
            and (
                issue.repair_target == next_action
                or (
                    next_action == "proposal"
                    and issue.repair_target == "proposal"
                )
            )
        ]

        return {
            "source": "deterministic_proposal_verifier_v1",
            "proposal_id": proposal_id,
            "next_action": next_action,
            "repair_round": (
                proposal_retry_count + 1
                if next_action == "proposal"
                else None
            ),
            "locked_facts_notice": (
                "不得修改 selection_result、WeatherNode 和 BudgetOptimize 提供的"
                "航班、酒店、天气、日期和预算锁定事实。"
            ),
            "issues": [
                {
                    "issue_type": issue.issue_type,
                    "check_name": issue.check_name,
                    "message": issue.message,
                    "path": issue.path,
                    "expected": issue.expected,
                    "actual": issue.actual,
                }
                for issue in relevant[:12]
            ],
            "instructions": [
                issue.message
                for issue in relevant[:12]
            ],
        }

    def _build_pending_actions(
        self,
        proposal: TripProposal,
    ) -> list[dict[str, Any]]:
        """Verifier 通过后创建等待用户确认的内部审批动作。"""

        return [
            {
                "action_id": (
                    "approve_" + proposal.proposal_id
                ),
                "action_type": "approve_proposal",
                "description": (
                    "确认当前旅行方案，并允许系统保存内部模拟行程草稿。"
                ),
                "requires_approval": True,
                "status": "pending",
                "payload_summary": {
                    "proposal_id": proposal.proposal_id,
                    "version": proposal.version,
                    "outbound_flight_id": (
                        proposal.selected_flights.outbound.flight_id
                    ),
                    "return_flight_id": (
                        proposal.selected_flights.return_flight.flight_id
                    ),
                    "hotel_id": proposal.selected_hotel.hotel_id,
                    "known_subtotal": (
                        proposal.budget_summary.known_costs.get(
                            "known_subtotal"
                        )
                    ),
                    "total_budget": proposal.budget_summary.total_budget,
                    "real_booking_performed": False,
                },
            }
        ]

    # ==================================================================
    # 检查框架和通用辅助函数
    # ==================================================================

    def _run_check(
        self,
        checks: dict[str, VerificationCheckResult],
        issues: list[VerificationIssue],
        check_name: str,
        summary: str,
        check_callable: Callable[[], None],
    ) -> None:
        """运行一个检查组，并自动生成摘要。"""

        before_count = len(issues)

        try:
            check_callable()
        except Exception as exc:
            # 检查函数本身抛异常说明 Verifier 无法可靠完成这一组检查。
            self._add_issue(
                issues,
                issue_type="verifier_check_failed",
                severity="critical",
                repair_target="final_response",
                check_name=check_name,
                message=f"Verifier 检查组 {check_name} 执行失败。",
                details={
                    "error": self._short_text(str(exc)),
                },
            )

        group_issues = issues[before_count:]
        blocking_count = sum(
            item.severity in {"error", "critical"}
            for item in group_issues
        )

        checks[check_name] = VerificationCheckResult(
            check_name=check_name,
            passed=(blocking_count == 0),
            issue_count=len(group_issues),
            warning_count=sum(
                item.severity == "warning"
                for item in group_issues
            ),
            error_count=blocking_count,
            summary=(
                summary
                if blocking_count == 0
                else f"{summary} 发现 {blocking_count} 个阻断问题。"
            ),
        )

    def _record_check(
        self,
        *,
        checks: dict[str, VerificationCheckResult],
        issues: Sequence[VerificationIssue],
        check_name: str,
        summary: str,
    ) -> None:
        """为已经在外层执行的检查生成摘要。"""

        group_issues = [
            item
            for item in issues
            if item.check_name == check_name
        ]
        blocking_count = sum(
            item.severity in {"error", "critical"}
            for item in group_issues
        )

        checks[check_name] = VerificationCheckResult(
            check_name=check_name,
            passed=(blocking_count == 0),
            issue_count=len(group_issues),
            warning_count=sum(
                item.severity == "warning"
                for item in group_issues
            ),
            error_count=blocking_count,
            summary=summary,
        )

    def _add_issue(
        self,
        issues: list[VerificationIssue],
        *,
        issue_type: str,
        severity: str,
        repair_target: str,
        check_name: str,
        message: str,
        path: str | None = None,
        expected: Any = None,
        actual: Any = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """添加一条结构化问题。"""

        issues.append(
            VerificationIssue(
                issue_type=issue_type,
                severity=severity,  # type: ignore[arg-type]
                repair_target=repair_target,  # type: ignore[arg-type]
                check_name=check_name,
                message=message,
                path=path,
                expected=expected,
                actual=actual,
                details=dict(details or {}),
            )
        )

    def _collect_proposal_evidence_refs(
        self,
        proposal: TripProposal,
    ) -> set[str]:
        """收集最终 Proposal 所有层级真正使用的 chunk_id。"""

        refs: set[str] = set()

        for day in proposal.daily_plan:
            for activity in day.activities:
                refs.update(activity.evidence_refs)

        refs.update(proposal.hotel_context.evidence_refs)

        for note in proposal.packing_tips:
            refs.update(note.evidence_refs)

        for note in proposal.safety_notes:
            refs.update(note.evidence_refs)

        for item in proposal.requirement_handling:
            refs.update(item.evidence_refs)

        return {
            ref.strip()
            for ref in refs
            if isinstance(ref, str)
            and ref.strip()
        }

    def _evidence_by_ref(
        self,
        evidence_pool: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """把 Evidence Pool 转成 chunk_id → evidence 映射。"""

        output: dict[str, dict[str, Any]] = {}

        for raw_item in evidence_pool:
            if not isinstance(raw_item, Mapping):
                continue

            chunk_id = str(raw_item.get("chunk_id") or "").strip()

            if not chunk_id:
                continue

            output[chunk_id] = dict(raw_item)

        return output

    def _collect_generated_texts(
        self,
        proposal: TripProposal,
    ) -> list[tuple[str, str]]:
        """收集需要扫描危险执行声明的 LLM 用户可见文本。"""

        texts: list[tuple[str, str]] = [
            ("proposal.summary", proposal.summary),
        ]

        texts.extend(
            (f"proposal.highlights[{index}]", text)
            for index, text in enumerate(proposal.highlights)
        )

        for day_index, day in enumerate(proposal.daily_plan):
            texts.append(
                (
                    f"proposal.daily_plan[{day_index}].theme",
                    day.theme,
                )
            )

            if day.weather_adjustment:
                texts.append(
                    (
                        f"proposal.daily_plan[{day_index}].weather_adjustment",
                        day.weather_adjustment,
                    )
                )

            texts.extend(
                (
                    f"proposal.daily_plan[{day_index}].day_notes[{index}]",
                    text,
                )
                for index, text in enumerate(day.day_notes)
            )

            for activity_index, activity in enumerate(day.activities):
                prefix = (
                    f"proposal.daily_plan[{day_index}]"
                    f".activities[{activity_index}]"
                )
                texts.append((f"{prefix}.title", activity.title))
                texts.append((f"{prefix}.description", activity.description))
                texts.extend(
                    (f"{prefix}.practical_notes[{index}]", text)
                    for index, text in enumerate(activity.practical_notes)
                )

        texts.append(
            ("proposal.hotel_context.summary", proposal.hotel_context.summary)
        )
        texts.extend(
            (f"proposal.hotel_context.nearby_facilities[{index}]", text)
            for index, text in enumerate(proposal.hotel_context.nearby_facilities)
        )
        texts.extend(
            (f"proposal.hotel_context.cautions[{index}]", text)
            for index, text in enumerate(proposal.hotel_context.cautions)
        )

        texts.extend(
            (f"proposal.packing_tips[{index}]", item.text)
            for index, item in enumerate(proposal.packing_tips)
        )
        texts.extend(
            (f"proposal.safety_notes[{index}]", item.text)
            for index, item in enumerate(proposal.safety_notes)
        )
        texts.extend(
            (f"proposal.preference_alignment[{index}]", text)
            for index, text in enumerate(proposal.preference_alignment)
        )
        texts.extend(
            (
                f"proposal.requirement_handling[{index}].explanation",
                item.explanation,
            )
            for index, item in enumerate(proposal.requirement_handling)
        )
        texts.extend(
            (f"proposal.limitations[{index}]", text)
            for index, text in enumerate(proposal.limitations)
        )

        return [
            (path, str(text))
            for path, text in texts
            if str(text).strip()
        ]

    def _model_to_dict(self, model: Any) -> dict[str, Any]:
        """把 Pydantic 模型转换成 JSON 兼容 dict。"""

        if hasattr(model, "model_dump"):
            return model.model_dump(mode="json")

        if hasattr(model, "dict"):
            return model.dict()

        if isinstance(model, Mapping):
            return dict(model)

        raise TypeError(f"无法转换为 dict：{type(model).__name__}")

    def _json_equal(self, left: Any, right: Any) -> bool:
        """通过规范化 JSON 比较嵌套结构。"""

        return self._canonical_json(left) == self._canonical_json(right)

    def _canonical_json(self, value: Any) -> str:
        """生成稳定 JSON 表示。"""

        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _sha256_json(self, value: Any) -> str:
        """对稳定 JSON 计算 SHA-256。"""

        return hashlib.sha256(
            self._canonical_json(value).encode("utf-8")
        ).hexdigest()

    def _close(self, left: float, right: float) -> bool:
        """按配置容差比较浮点数。"""

        return abs(left - right) <= self.config.float_tolerance

    def _safe_float(
        self,
        value: Any,
        default: float = 0.0,
    ) -> float:
        """安全转换 float。"""

        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _safe_int(
        self,
        value: Any,
        default: int = 0,
    ) -> int:
        """安全转换 int。"""

        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _optional_text(self, value: Any) -> str | None:
        """标准化可选文本。"""

        if value is None:
            return None

        text = str(value).strip()
        return text or None

    def _compact_value(self, value: Any) -> Any:
        """限制 expected/actual 大小，避免 VerifierResult 过大。"""

        text = self._canonical_json(value)

        if len(text) <= 900:
            return value

        return {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "preview": text[:700] + "...",
        }

    def _short_text(
        self,
        value: str,
        max_length: int = 600,
    ) -> str:
        """截断过长错误和文本。"""

        text = " ".join(str(value).split())

        if len(text) <= max_length:
            return text

        return text[:max_length] + "..."
