from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from time import perf_counter
from typing import Any, Literal, Mapping, Sequence

from app.agents.state import TravelState
from app.revisions.revision_analyzer import hash_trip_request
from app.schemas.revision_schema import (
    RevisionApplyResult,
    RevisionPlan,
)
from app.schemas.trip_request_schema import TripRequest


ApplyRevisionDestination = Literal[
    "missing_info_check",
    "final_response",
]


# ApplyRevision 后必须清空的派生 State。
# 这些字段都依赖旧 TripRequest、旧候选、旧 RAG 或旧 Proposal，继续保留会导致
# “新需求 + 旧结果”混用。
INVALIDATED_STATE_DEFAULTS: dict[str, Any] = {
    "missing_fields": [],
    "can_continue": False,
    "capability_result": {},
    "missing_info_result": {},
    "clarification": {},
    "safety_result": {},
    "safety_flags": [],
    "planning_context": {},
    "budget_plan": {},
    "weather_result": {},
    "tool_plan": {},
    "tool_execution_result": {},
    "tool_check_result": {},
    "tool_plan_retry_count": 0,
    "tool_execute_retry_count": 0,
    "activity_result": {},
    "activity_candidates": [],
    "activity_fetch_meta": {},
    "raw_flight_results": {},
    "raw_hotel_results": {},
    "weather_fetch_meta": {},
    "flight_fetch_meta": {},
    "hotel_fetch_meta": {},
    "flight_candidates": [],
    "hotel_candidates": [],
    "candidate_rank_result": {},
    "budget_result": {},
    "adjustment_plan": {},
    "budget_optimize_result": {},
    "selection_result": {},
    "retrieval_plan": {},
    "retrieved_chunks": [],
    "hybrid_retrieval_result": {},
    "reranked_evidence": [],
    "rerank_result": {},
    "evidence_pool": [],
    "evidence_requirements": {},
    "evidence_result": {},
    "retrieval_feedback": {},
    "proposal": {},
    "current_proposal": {},
    "proposal_result": {},
    "verifier_result": {},
    "proposal_feedback": {},
    "pending_actions": [],
    "decision": None,
    "decision_result": {},
    "decision_gate_error": {},
    "decision_attempt_count": 0,
    "trip_draft": {},
    "trip_version": {},
    "approval_record": {},
    "commit_result": {},
    "cancel_result": {},
    "final_response": {},
}


def apply_revision_node(
    state: TravelState,
) -> dict[str, Any]:
    """
    ApplyRevisionNode：把 RevisionPlan 确定性应用到当前 TripRequest。

    读取 State：
        - trip_request
        - proposal
        - change_request
        - revision_plan

    写入 State：
        - trip_request，修改后的新请求
        - revision_result，应用结果和字段差异
        - raw_message，本轮修改文本，供 SafetyCheckNode 重新检查
        - workflow = "modify_trip"
        - itinerary_status = "collecting_info"
        - 所有依赖旧请求的派生字段被清空
        - retry_count 重置
        - trace
        - errors，仅程序异常时

    这个节点不做：
        - 不调用 LLM；
        - 不查询 MCP；
        - 不执行 RAG；
        - 不尝试判断最终酒店或航班；
        - 不选择性复用旧候选。

    当前 v1 的策略是：
        只复用已经确认的 TripRequest 基线，应用补丁后重新执行完整 PlanTrip 主链。
        这样可以避免日期、目的地或偏好变化后仍然混用旧天气、旧候选和旧证据。
    """

    node_name = "apply_revision"
    started_at = perf_counter()

    try:
        current_request_data = _require_mapping(
            state.get("trip_request"),
            field_name="trip_request",
        )
        proposal = _require_mapping(
            state.get("proposal"),
            field_name="proposal",
        )
        change_request = _require_mapping(
            state.get("change_request"),
            field_name="change_request",
        )
        plan_data = _require_mapping(
            state.get("revision_plan"),
            field_name="revision_plan",
        )

        # 1. Pydantic 先验证 RevisionPlan 的结构和枚举。
        plan = RevisionPlan(**plan_data)

        if plan.status != "ready":
            raise ValueError(
                "revision_plan.status 必须为 ready"
            )

        # 2. 检查补丁仍然对应当前 Proposal，防止修改请求应用到新版本方案。
        _validate_plan_base(
            plan=plan,
            proposal=proposal,
            change_request=change_request,
        )

        # 3. 检查 TripRequest 基线 Hash，防止并发或旧状态补丁覆盖新请求。
        before_hash = hash_trip_request(
            current_request_data
        )

        if before_hash != plan.base_trip_request_hash:
            raise ValueError(
                "当前 trip_request 已经变化，RevisionPlan 基线已过期"
            )

        # 4. 先用 TripRequest Schema 规范化旧请求。
        current_request = TripRequest(
            **current_request_data
        ).to_state_dict()

        # 5. 在深拷贝上应用 Patch，不直接修改原 State 对象。
        updated_request = deepcopy(
            current_request
        )

        changed_fields: list[str] = []
        explicitly_touched_fields: set[str] = set()

        _apply_field_operations(
            request=updated_request,
            operations=(
                plan.patch.field_operations
            ),
            changed_fields=changed_fields,
            explicitly_touched_fields=(
                explicitly_touched_fields
            ),
        )

        # 6. 日期字段需要跨字段归一化。
        #    例如 days +2 后应重新计算 end_date。
        _normalize_revised_dates(
            request=updated_request,
            explicitly_touched_fields=(
                explicitly_touched_fields
            ),
            changed_fields=changed_fields,
        )

        # 7. 主观偏好按 domain + key 更新，删除操作先执行。
        updated_request[
            "preference_signals"
        ] = _merge_preferences(
            current=_mapping_list(
                updated_request.get(
                    "preference_signals"
                )
            ),
            removals=(
                plan.patch.preference_removals
            ),
            upserts=(
                plan.patch.preference_upserts
            ),
            changed_fields=changed_fields,
        )

        # 8. 同一预算类别只保留最新消费倾向。
        updated_request[
            "spend_preferences"
        ] = _merge_spend_preferences(
            current=_mapping_list(
                updated_request.get(
                    "spend_preferences"
                )
            ),
            removals=(
                plan.patch.spend_preference_removals
            ),
            upserts=(
                plan.patch.spend_preference_upserts
            ),
            changed_fields=changed_fields,
        )

        # 9. 客观硬约束按 domain + field 替换，避免同一字段存在冲突条件。
        updated_request[
            "hard_constraints"
        ] = _merge_hard_constraints(
            current=_mapping_list(
                updated_request.get(
                    "hard_constraints"
                )
            ),
            removals=(
                plan.patch.hard_constraint_removals
            ),
            additions=(
                plan.patch.hard_constraint_additions
            ),
            changed_fields=changed_fields,
        )

        # 10. “酒店换成X酒店”会先删除旧 hotel named constraint，再添加新项。
        updated_request[
            "named_constraints"
        ] = _merge_named_constraints(
            current=_mapping_list(
                updated_request.get(
                    "named_constraints"
                )
            ),
            replace_types=set(
                plan.patch.replace_named_entity_types
            ),
            removals=(
                plan.patch.named_constraint_removals
            ),
            additions=(
                plan.patch.named_constraint_additions
            ),
            changed_fields=changed_fields,
        )

        # 11. 无法映射的需求只追加去重，不让它们静默消失。
        updated_request[
            "unmapped_requirements"
        ] = _merge_unmapped_requirements(
            current=_mapping_list(
                updated_request.get(
                    "unmapped_requirements"
                )
            ),
            additions=(
                plan.patch.unmapped_requirement_additions
            ),
            changed_fields=changed_fields,
        )

        # 12. 新一轮 requirement_issues 以当前补丁为准，避免保留旧的已解决歧义。
        updated_request[
            "requirement_issues"
        ] = [
            item.to_state_dict()
            for item in plan.patch.requirement_issues
        ]

        if plan.patch.requirement_issues:
            changed_fields.append(
                "requirement_issues"
            )

        raw_change_message = str(
            change_request.get("raw_message")
            or ""
        ).strip()

        # 13. raw_message 只作为审计文本，不再交给 InputExtractNode 重解析。
        old_raw_message = str(
            updated_request.get("raw_message")
            or ""
        ).strip()
        updated_request["raw_message"] = (
            f"{old_raw_message}\n修改要求：{raw_change_message}"
            if old_raw_message
            else raw_change_message
        )

        # 14. 更新字段来源和抽取元信息，方便 Trace 与 Eval。
        _update_field_resolution(
            request=updated_request,
            plan=plan,
        )
        _update_extraction_metadata(
            request=updated_request,
            plan=plan,
        )

        # 15. 重新生成兼容性的轻量偏好索引和 raw_constraints。
        _rebuild_derived_request_indexes(
            updated_request
        )

        # 16. 最终再经过 TripRequest Pydantic，确保修改后数据合同仍然成立。
        revised_request = TripRequest(
            **updated_request
        ).to_state_dict()

        after_hash = hash_trip_request(
            revised_request
        )

        changed_fields = list(
            dict.fromkeys(changed_fields)
        )
        invalidated_fields = list(
            INVALIDATED_STATE_DEFAULTS.keys()
        )

        applied_at = datetime.now(
            timezone.utc
        ).isoformat()

        revision_result = RevisionApplyResult(
            status="applied",
            base_proposal_id=(
                plan.base_proposal_id
            ),
            base_proposal_version=(
                plan.base_proposal_version
            ),
            before_trip_request_hash=(
                before_hash
            ),
            after_trip_request_hash=(
                after_hash
            ),
            changed_fields=changed_fields,
            invalidated_fields=(
                invalidated_fields
            ),
            summary=(
                plan.patch.summary
                or "修改补丁已应用，准备重新规划。"
            ),
            applied_at=applied_at,
            issues=[],
        )

        # 17. 清空所有旧派生结果，再写入修改后的权威 TripRequest。
        update: dict[str, Any] = {
            key: deepcopy(value)
            for key, value
            in INVALIDATED_STATE_DEFAULTS.items()
        }

        update.update(
            {
                "trip_request": (
                    revised_request
                ),
                # 顶层 raw_message 使用本轮修改文本，
                # SafetyCheckNode 会重新检查是否包含真实预订、付款等越界要求。
                "raw_message": (
                    raw_change_message
                ),
                "workflow": "modify_trip",
                "itinerary_status": (
                    "collecting_info"
                ),
                "revision_result": (
                    revision_result.to_state_dict()
                ),
                # Patch 已经应用，原始 change_request 可以清空；
                # revision_plan 和 revision_result 仍保留审计信息。
                "change_request": {},
                "budget_retry_count": 0,
                "rag_retry_count": 0,
                "proposal_retry_count": 0,
            }
        )

        trace_item = _build_trace_item(
            status="success",
            started_at=started_at,
            input_summary=(
                f"proposal_id={plan.base_proposal_id}, "
                f"patch_operations="
                f"{len(plan.patch.field_operations)}"
            ),
            output_summary=(
                f"changed_fields={changed_fields}, "
                f"invalidated_count={len(invalidated_fields)}"
            ),
        )
        update["trace"] = [trace_item]

        return update

    except Exception as exc:
        error_item = {
            "node": node_name,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        result = RevisionApplyResult(
            status="failed",
            summary="修改补丁应用失败。",
            issues=[
                {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                }
            ],
        )

        trace_item = _build_trace_item(
            status="failed",
            started_at=started_at,
            input_summary=(
                "revision apply failed"
            ),
            output_summary=str(exc),
        )

        return {
            "revision_result": (
                result.to_state_dict()
            ),
            "errors": [error_item],
            "trace": [trace_item],
        }


def route_after_apply_revision(
    state: TravelState,
) -> ApplyRevisionDestination:
    """
    ApplyRevisionNode 后的 Conditional Edge。

    applied：
        → MissingInfoCheckNode，再重新执行完整 PlanTrip 主链。

    failed：
        → FinalResponseNode。
    """

    result = state.get("revision_result")

    if (
        isinstance(result, dict)
        and result.get("status") == "applied"
    ):
        return "missing_info_check"

    return "final_response"


# ======================================================================
# Core field operations
# ======================================================================


def _apply_field_operations(
    *,
    request: dict[str, Any],
    operations: Sequence[Any],
    changed_fields: list[str],
    explicitly_touched_fields: set[str],
) -> None:
    """按 RevisionFieldOperation 顺序修改核心字段。"""

    for operation in operations:
        field = operation.field
        explicitly_touched_fields.add(field)

        if operation.operation == "clear":
            request[field] = None

        elif operation.operation == "set":
            request[field] = operation.value

        elif operation.operation == "increment":
            old_value = request.get(field)

            if old_value is None:
                raise ValueError(
                    f"字段 {field} 当前为空，不能执行 increment"
                )

            if field == "days":
                request[field] = int(
                    old_value
                ) + int(operation.value)

                if request[field] < 1:
                    raise ValueError(
                        "修改后的 days 必须大于等于 1"
                    )

            elif field == "budget":
                request[field] = float(
                    old_value
                ) + float(operation.value)

                if request[field] < 0:
                    raise ValueError(
                        "修改后的 budget 不能小于 0"
                    )

        changed_fields.append(field)


def _normalize_revised_dates(
    *,
    request: dict[str, Any],
    explicitly_touched_fields: set[str],
    changed_fields: list[str],
) -> None:
    """
    保持 start_date、end_date 和 days 一致。

    规则：
        - start_date 或 days 改变，而 end_date 未明确修改：重新推导 end_date；
        - end_date 改变，而 days 未明确修改：重新推导 days；
        - days 和 end_date 同时明确修改：检查二者是否一致；
        - 清空 start_date 或 days 时，同时清空 end_date。
    """

    start_date_value = request.get(
        "start_date"
    )
    end_date_value = request.get(
        "end_date"
    )
    days_value = request.get("days")

    # 1. start_date 或 days 为空时，无法保留一个可验证的 end_date。
    if start_date_value is None or days_value is None:
        if request.get("end_date") is not None:
            request["end_date"] = None
            changed_fields.append("end_date")
        return

    start = _parse_iso_date(
        start_date_value,
        field_name="start_date",
    )
    days = int(days_value)

    inferred_end = start + timedelta(
        days=days - 1
    )

    end_touched = "end_date" in explicitly_touched_fields
    days_touched = "days" in explicitly_touched_fields

    # 2. end_date 明确修改，但 days 未修改：以日期范围为准重新计算 days。
    if end_touched and not days_touched:
        if end_date_value is None:
            return

        end = _parse_iso_date(
            end_date_value,
            field_name="end_date",
        )

        if end < start:
            raise ValueError(
                "修改后的 end_date 不能早于 start_date"
            )

        derived_days = (
            end - start
        ).days + 1
        request["days"] = derived_days
        changed_fields.append("days")
        return

    # 3. days 和 end_date 都明确修改时，两者必须一致。
    if end_touched and days_touched:
        if end_date_value is None:
            return

        end = _parse_iso_date(
            end_date_value,
            field_name="end_date",
        )

        if end != inferred_end:
            raise ValueError(
                "同时修改 days 和 end_date 时，两者计算结果不一致"
            )
        return

    # 4. start_date 或 days 变化时，使用明确天数推导返程日期。
    inferred_text = inferred_end.isoformat()

    if request.get("end_date") != inferred_text:
        request["end_date"] = inferred_text
        changed_fields.append("end_date")


# ======================================================================
# Collection merge helpers
# ======================================================================


def _merge_preferences(
    *,
    current: list[dict[str, Any]],
    removals: Sequence[Any],
    upserts: Sequence[Any],
    changed_fields: list[str],
) -> list[dict[str, Any]]:
    """按 domain + key 删除、更新和新增主观偏好。"""

    removal_keys = {
        (item.domain, item.key)
        for item in removals
    }

    merged = {
        (
            str(item.get("domain") or ""),
            str(item.get("key") or ""),
        ): dict(item)
        for item in current
        if item.get("domain")
        and item.get("key")
        and (
            str(item.get("domain")),
            str(item.get("key")),
        ) not in removal_keys
    }

    for item in upserts:
        merged[(item.domain, item.key)] = (
            item.to_state_dict()
        )

    result = list(merged.values())

    if result != current:
        changed_fields.append(
            "preference_signals"
        )

    return result


def _merge_spend_preferences(
    *,
    current: list[dict[str, Any]],
    removals: Sequence[Any],
    upserts: Sequence[Any],
    changed_fields: list[str],
) -> list[dict[str, Any]]:
    """同一预算 category 只保留最新消费倾向。"""

    removal_categories = {
        item.category
        for item in removals
    }

    merged = {
        str(item.get("category")): dict(item)
        for item in current
        if item.get("category")
        and str(item.get("category"))
        not in removal_categories
    }

    for item in upserts:
        merged[item.category] = (
            item.to_state_dict()
        )

    result = list(merged.values())

    if result != current:
        changed_fields.append(
            "spend_preferences"
        )

    return result


def _merge_hard_constraints(
    *,
    current: list[dict[str, Any]],
    removals: Sequence[Any],
    additions: Sequence[Any],
    changed_fields: list[str],
) -> list[dict[str, Any]]:
    """按 domain + field 替换客观硬约束。"""

    removal_keys = {
        (item.domain, item.field)
        for item in removals
    }

    merged = {
        (
            str(item.get("domain")),
            str(item.get("field")),
        ): dict(item)
        for item in current
        if item.get("domain")
        and item.get("field")
        and (
            str(item.get("domain")),
            str(item.get("field")),
        ) not in removal_keys
    }

    for item in additions:
        merged[(item.domain, item.field)] = (
            item.to_state_dict()
        )

    result = list(merged.values())

    if result != current:
        changed_fields.append(
            "hard_constraints"
        )

    return result


def _merge_named_constraints(
    *,
    current: list[dict[str, Any]],
    replace_types: set[str],
    removals: Sequence[Any],
    additions: Sequence[Any],
    changed_fields: list[str],
) -> list[dict[str, Any]]:
    """增删指定实体；replace_types 用于“换成另一家酒店”。"""

    def should_remove(item: Mapping[str, Any]) -> bool:
        entity_type = str(
            item.get("entity_type")
            or ""
        )
        entity_name = str(
            item.get("entity_name")
            or ""
        )

        if entity_type in replace_types:
            return True

        for selector in removals:
            if selector.entity_type != entity_type:
                continue

            if (
                selector.entity_name is None
                or selector.entity_name
                == entity_name
            ):
                return True

        return False

    remaining = [
        dict(item)
        for item in current
        if not should_remove(item)
    ]

    merged = {
        (
            str(item.get("entity_type")),
            str(item.get("entity_name")),
            str(item.get("constraint_mode")),
        ): dict(item)
        for item in remaining
        if item.get("entity_type")
        and item.get("entity_name")
    }

    for item in additions:
        key = (
            item.entity_type,
            item.entity_name,
            item.constraint_mode,
        )
        merged[key] = item.to_state_dict()

    result = list(merged.values())

    if result != current:
        changed_fields.append(
            "named_constraints"
        )

    return result


def _merge_unmapped_requirements(
    *,
    current: list[dict[str, Any]],
    additions: Sequence[Any],
    changed_fields: list[str],
) -> list[dict[str, Any]]:
    """按 text 去重追加未映射需求。"""

    merged = {
        str(item.get("text")): dict(item)
        for item in current
        if item.get("text")
    }

    for item in additions:
        merged[item.text] = (
            item.to_state_dict()
        )

    result = list(merged.values())

    if result != current:
        changed_fields.append(
            "unmapped_requirements"
        )

    return result


# ======================================================================
# Derived TripRequest metadata
# ======================================================================


def _update_field_resolution(
    *,
    request: dict[str, Any],
    plan: RevisionPlan,
) -> None:
    """为修改过的核心字段记录解析来源。"""

    resolution = request.get(
        "field_resolution"
    )
    resolution = (
        dict(resolution)
        if isinstance(resolution, dict)
        else {}
    )

    for operation in plan.patch.field_operations:
        resolution[operation.field] = {
            "value": request.get(
                operation.field
            ),
            "resolution_type": "explicit",
            "evidence": (
                operation.evidence
                or plan.raw_change_request
            ),
            "needs_clarification": False,
        }

    request["field_resolution"] = resolution


def _update_extraction_metadata(
    *,
    request: dict[str, Any],
    plan: RevisionPlan,
) -> None:
    """记录本轮 Revision Patch 的模型、回退和摘要。"""

    extraction = request.get("extraction")
    extraction = (
        dict(extraction)
        if isinstance(extraction, dict)
        else {}
    )

    previous_method = extraction.get(
        "method"
    )
    notes = extraction.get("notes")
    notes = (
        [str(item) for item in notes]
        if isinstance(notes, list)
        else []
    )

    notes.append(
        "已在已确认 TripRequest 基线上应用 RevisionPatch，"
        "没有重新解析完整历史对话。"
    )

    extraction.update(
        {
            "method": "revision_patch_v1",
            "previous_method": (
                previous_method
            ),
            "revision_model": (
                plan.model_id
            ),
            "revision_fallback_used": (
                plan.fallback_used
            ),
            "revision_attempt_count": (
                plan.attempt_count
            ),
            "revision_summary": (
                plan.patch.summary
            ),
            "notes": list(
                dict.fromkeys(notes)
            ),
        }
    )

    request["extraction"] = extraction


def _rebuild_derived_request_indexes(
    request: dict[str, Any],
) -> None:
    """
    根据 preference_signals 重建兼容性索引和 raw_constraints。

    新代码的权威来源仍然是 preference_signals；这些 list 只是方便日志和旧测试。
    """

    transport: list[str] = []
    hotels: list[str] = []
    styles: list[str] = []
    raw_constraints: list[str] = []

    for item in _mapping_list(
        request.get("preference_signals")
    ):
        domain = str(
            item.get("domain")
            or ""
        )
        key = str(
            item.get("key")
            or ""
        )

        if domain == "flight" and key:
            transport.append(key)
        elif domain == "hotel" and key:
            hotels.append(key)
        elif domain in {"activity", "pace"} and key:
            styles.append(key)

        evidence = str(
            item.get("evidence")
            or ""
        ).strip()
        if evidence:
            raw_constraints.append(evidence)

    for field_name in (
        "spend_preferences",
        "hard_constraints",
        "named_constraints",
    ):
        for item in _mapping_list(
            request.get(field_name)
        ):
            evidence = str(
                item.get("evidence")
                or ""
            ).strip()
            if evidence:
                raw_constraints.append(
                    evidence
                )

    request["transport_preferences"] = list(
        dict.fromkeys(transport)
    )
    request["hotel_preferences"] = list(
        dict.fromkeys(hotels)
    )
    request["travel_style"] = list(
        dict.fromkeys(styles)
    )
    request["raw_constraints"] = list(
        dict.fromkeys(raw_constraints)
    )


# ======================================================================
# Preconditions and common helpers
# ======================================================================


def _validate_plan_base(
    *,
    plan: RevisionPlan,
    proposal: Mapping[str, Any],
    change_request: Mapping[str, Any],
) -> None:
    """校验 Proposal ID / Version 三方一致。"""

    proposal_id = str(
        proposal.get("proposal_id")
        or ""
    )
    proposal_version = _safe_int(
        proposal.get("version"),
        default=0,
    )

    change_base_id = str(
        change_request.get(
            "base_proposal_id"
        )
        or ""
    )
    change_base_version = _safe_int(
        change_request.get(
            "base_proposal_version"
        ),
        default=0,
    )

    if not (
        proposal_id
        == plan.base_proposal_id
        == change_base_id
    ):
        raise ValueError(
            "RevisionPlan、change_request 与当前 Proposal ID 不一致"
        )

    if not (
        proposal_version
        == plan.base_proposal_version
        == change_base_version
    ):
        raise ValueError(
            "RevisionPlan、change_request 与当前 Proposal Version 不一致"
        )


def _parse_iso_date(
    value: Any,
    *,
    field_name: str,
) -> date:
    """把 date 或 YYYY-MM-DD 字符串转换成 date。"""

    if isinstance(value, date):
        return value

    try:
        return date.fromisoformat(
            str(value)
        )
    except ValueError as exc:
        raise ValueError(
            f"{field_name} 不是合法 YYYY-MM-DD 日期"
        ) from exc


def _mapping_list(
    value: Any,
) -> list[dict[str, Any]]:
    """把 list[Mapping] 安全转换为普通 dict 列表。"""

    if not isinstance(value, list):
        return []

    return [
        dict(item)
        for item in value
        if isinstance(item, Mapping)
    ]


def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    """读取必须存在的 dict State。"""

    if not isinstance(value, dict):
        raise TypeError(
            f"state['{field_name}'] 必须是 dict"
        )

    return dict(value)


def _safe_int(
    value: Any,
    default: int,
) -> int:
    """安全转换整数。"""

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _build_trace_item(
    *,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """生成统一 Trace 记录。"""

    return {
        "node_name": "apply_revision",
        "tool_name": "deterministic_revision_patch_applier",
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "latency_ms": int(
            (perf_counter() - started_at)
            * 1000
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }
