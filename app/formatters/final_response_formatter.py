from __future__ import annotations

from typing import Any, Mapping


# FinalResponse 只返回适合 API / 前端使用的信息。
# 不暴露：
#     - 原始候选池；
#     - 完整 user_profile；
#     - 内部 Trace；
#     - RAG 中间结果；
#     - Prompt；
#     - Checkpoint 细节。


def build_final_response(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    根据当前 TravelState 构造最终 API 响应。

    这是一个纯 Formatter：
        - 不调用 LLM；
        - 不查询数据库；
        - 不修改业务状态；
        - 不执行 LangGraph 路由；
        - 只把内部 State 映射成稳定、最小、可展示的响应。

    响应优先级：
        1. ClarifyNode / SafeRejectNode 已经生成的 final_response；
        2. CommitDraft 成功；
        3. Cancel 成功；
        4. RevisionAnalyze 需要澄清；
        5. RevisionApply 失败；
        6. CandidateRank / Budget / Evidence / Proposal / Verifier 失败；
        7. 兜底系统错误。
    """

    # 1. ClarifyNode 和 SafeRejectNode 已经拥有专用 Formatter。
    #    这里保留其内容，只补充会话元信息。
    existing = state.get("final_response")

    if isinstance(existing, Mapping) and existing:
        response = dict(existing)
        _attach_common_metadata(
            response=response,
            state=state,
        )
        return response

    commit_result = _mapping(
        state.get("commit_result")
    )
    cancel_result = _mapping(
        state.get("cancel_result")
    )
    revision_plan = _mapping(
        state.get("revision_plan")
    )
    revision_result = _mapping(
        state.get("revision_result")
    )

    # 2. Approve 分支：数据库已提交或属于幂等重放。
    if commit_result.get("status") in {
        "committed",
        "already_committed",
    }:
        response = _build_commit_response(
            state=state,
            commit_result=commit_result,
        )
        _attach_common_metadata(
            response=response,
            state=state,
        )
        return response

    # 3. Cancel 分支：当前内部方案审批流程已取消。
    if cancel_result.get("status") in {
        "cancelled",
        "already_cancelled",
    }:
        response = _build_cancel_response(
            state=state,
            cancel_result=cancel_result,
        )
        _attach_common_metadata(
            response=response,
            state=state,
        )
        return response

    # 4. 修改文本存在歧义时，向用户返回精确追问。
    if (
        revision_plan.get("status")
        == "clarification_required"
    ):
        response = _build_revision_clarification_response(
            revision_plan
        )
        _attach_common_metadata(
            response=response,
            state=state,
        )
        return response

    if revision_result.get("status") == "failed":
        response = {
            "status": "revision_failed",
            "type": "revision_error",
            "message": (
                revision_result.get("summary")
                or "修改请求未能安全应用，请重新说明修改内容。"
            ),
            "need_user_input": True,
            "available_actions": [
                "submit_revision",
                "cancel",
            ],
            "details": {
                "issues": _public_issues(
                    revision_result.get("issues")
                )
            },
        }
        _attach_common_metadata(
            response=response,
            state=state,
        )
        return response

    candidate_result = _mapping(
        state.get("candidate_rank_result")
    )

    if candidate_result.get("status") in {
        "partial",
        "no_candidates",
        "failed",
    }:
        response = {
            "status": "no_complete_candidates",
            "type": "candidate_unavailable",
            "message": (
                "当前条件下没有同时满足完整往返航班和酒店要求的候选。"
            ),
            "need_user_input": True,
            "available_actions": [
                "revise_request",
                "cancel",
            ],
            "details": {
                "issues": _public_issues(
                    candidate_result.get("issues")
                ),
                "output_counts": _mapping(
                    candidate_result.get(
                        "output_counts"
                    )
                ),
            },
        }
        _attach_common_metadata(
            response=response,
            state=state,
        )
        return response

    selection_result = _mapping(
        state.get("selection_result")
    )

    if selection_result.get("status") in {
        "no_feasible_combination",
        "no_candidates",
        "failed",
    }:
        response = _build_budget_failure_response(
            state=state,
            selection_result=selection_result,
        )
        _attach_common_metadata(
            response=response,
            state=state,
        )
        return response

    evidence_result = _mapping(
        state.get("evidence_result")
    )

    if evidence_result.get("status") == "failed":
        semantic_review = _mapping(
            evidence_result.get("semantic_review")
        )
        semantic_failed = (
            semantic_review.get("status") == "insufficient"
        )
        response = {
            "status": "insufficient_evidence",
            "type": "rag_evidence_failed",
            "message": (
                "已检索到相关资料，但部分必要信息未通过语义证据检查，"
                "系统没有生成未经支持的旅行方案。"
                if semantic_failed
                else "当前知识库缺少生成旅行方案所需的必要证据。"
            ),
            "need_user_input": True,
            "available_actions": [
                "revise_request",
                "cancel",
            ],
            "details": {
                "issues": _public_issues(
                    evidence_result.get("issues")
                ),
                "coverage": _public_coverage(
                    evidence_result.get("coverage")
                ),
                "semantic_review": _public_semantic_review(
                    semantic_review
                ),
            },
        }
        _attach_common_metadata(
            response=response,
            state=state,
        )
        return response

    proposal_result = _mapping(
        state.get("proposal_result")
    )

    if proposal_result.get("status") == "failed":
        response = {
            "status": "proposal_generation_failed",
            "type": "proposal_error",
            "message": (
                "旅行方案在限定重试次数内未通过结构和业务校验。"
            ),
            "need_user_input": True,
            "available_actions": [
                "retry",
                "revise_request",
                "cancel",
            ],
            "details": {
                "attempt_count": proposal_result.get(
                    "attempt_count"
                ),
                "validation_errors": _string_list(
                    proposal_result.get(
                        "validation_errors"
                    )
                )[:5],
            },
        }
        _attach_common_metadata(
            response=response,
            state=state,
        )
        return response

    verifier_result = _mapping(
        state.get("verifier_result")
    )

    if (
        verifier_result.get("status")
        in {"repair_required", "failed"}
        and verifier_result.get("next_action")
        == "final_response"
    ):
        response = {
            "status": "verification_failed",
            "type": "proposal_verification_error",
            "message": (
                "旅行方案未通过最终一致性校验，系统没有将其交给用户审批。"
            ),
            "need_user_input": True,
            "available_actions": [
                "retry",
                "revise_request",
                "cancel",
            ],
            "details": {
                "issues": _public_issues(
                    verifier_result.get("issues")
                ),
                "error_count": verifier_result.get(
                    "error_count"
                ),
                "critical_count": verifier_result.get(
                    "critical_count"
                ),
            },
        }
        _attach_common_metadata(
            response=response,
            state=state,
        )
        return response

    # 5. Commit / Cancel 失败需要单独说明，避免误报“已保存”或“已取消”。
    if commit_result.get("status") == "failed":
        response = {
            "status": "commit_failed",
            "type": "business_persistence_error",
            "message": (
                commit_result.get("message")
                or "方案已经批准，但内部旅行草稿保存失败。"
            ),
            "need_user_input": False,
            "available_actions": ["retry_commit"],
            "details": {
                "issues": _public_issues(
                    commit_result.get("issues")
                )
            },
        }
        _attach_common_metadata(
            response=response,
            state=state,
        )
        return response

    if cancel_result.get("status") == "failed":
        response = {
            "status": "cancel_failed",
            "type": "cancel_error",
            "message": (
                cancel_result.get("message")
                or "当前方案取消收尾失败。"
            ),
            "need_user_input": False,
            "available_actions": ["retry_cancel"],
            "details": {
                "issues": _public_issues(
                    cancel_result.get("issues")
                )
            },
        }
        _attach_common_metadata(
            response=response,
            state=state,
        )
        return response

    # 6. 最后只暴露精简错误，不返回内部 State 或堆栈。
    errors = state.get("errors")
    public_errors = _public_errors(errors)

    response = {
        "status": "failed",
        "type": "workflow_error",
        "message": (
            "旅行规划未能完成，请根据提示重试或修改请求。"
        ),
        "need_user_input": True,
        "available_actions": [
            "retry",
            "revise_request",
            "cancel",
        ],
        "details": {
            "errors": public_errors,
        },
    }
    _attach_common_metadata(
        response=response,
        state=state,
    )
    return response


def _build_commit_response(
    *,
    state: Mapping[str, Any],
    commit_result: Mapping[str, Any],
) -> dict[str, Any]:
    """构造批准并保存成功的最终响应。"""

    trip_draft = _mapping(
        state.get("trip_draft")
    )
    trip_version = _mapping(
        state.get("trip_version")
    )

    return {
        "status": "approved_simulated",
        "type": "trip_draft_saved",
        "message": (
            commit_result.get("message")
            or "旅行方案已保存为内部模拟草稿。"
        ),
        "need_user_input": False,
        "available_actions": [],
        "trip": {
            "trip_id": (
                commit_result.get("trip_id")
                or trip_draft.get("trip_id")
            ),
            "proposal_id": (
                commit_result.get("proposal_id")
                or trip_draft.get(
                    "current_proposal_id"
                )
            ),
            "version": (
                trip_version.get(
                    "version_number"
                )
                or commit_result.get(
                    "proposal_version"
                )
            ),
            "destination": trip_draft.get(
                "destination"
            ),
            "start_date": trip_draft.get(
                "start_date"
            ),
            "end_date": trip_draft.get(
                "end_date"
            ),
            "known_subtotal": trip_draft.get(
                "known_subtotal"
            ),
            "remaining_budget": trip_draft.get(
                "remaining_budget"
            ),
            "draft_only": True,
        },
        "execution_boundary": {
            "real_booking_performed": False,
            "real_payment_performed": False,
            "real_notification_sent": False,
            "draft_only": True,
        },
        "idempotent_replay": bool(
            commit_result.get(
                "idempotent_replay"
            )
        ),
    }


def _build_cancel_response(
    *,
    state: Mapping[str, Any],
    cancel_result: Mapping[str, Any],
) -> dict[str, Any]:
    """构造取消成功的最终响应。"""

    proposal = _mapping(
        state.get("proposal")
    )

    return {
        "status": "cancelled",
        "type": "trip_proposal_cancelled",
        "message": (
            cancel_result.get("message")
            or "当前旅行方案已取消。"
        ),
        "need_user_input": False,
        "available_actions": [],
        "proposal_id": (
            cancel_result.get("proposal_id")
            or proposal.get("proposal_id")
        ),
        "reason": cancel_result.get("reason"),
        "proposal_deleted": False,
        "real_booking_cancelled": False,
        "idempotent_replay": bool(
            cancel_result.get(
                "idempotent_replay"
            )
        ),
    }


def _build_revision_clarification_response(
    revision_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """构造修改请求歧义时的追问响应。"""

    questions = _string_list(
        revision_plan.get(
            "clarification_questions"
        )
    )

    return {
        "status": "revision_needs_clarification",
        "type": "revision_clarification",
        "message": (
            "修改请求中仍有无法安全确定的内容，请补充说明后再提交修改。"
        ),
        "need_user_input": True,
        "available_actions": [
            "submit_revision",
            "cancel",
        ],
        "proposal_id": revision_plan.get(
            "base_proposal_id"
        ),
        "questions": questions,
        "recognized_summary": _mapping(
            revision_plan.get("patch")
        ).get("summary"),
    }


def _build_budget_failure_response(
    *,
    state: Mapping[str, Any],
    selection_result: Mapping[str, Any],
) -> dict[str, Any]:
    """构造无预算可行组合响应。"""

    adjustment_plan = _mapping(
        state.get("adjustment_plan")
    )

    return {
        "status": str(
            selection_result.get("status")
            or "no_feasible_combination"
        ),
        "type": "budget_or_combination_unavailable",
        "message": (
            adjustment_plan.get("message")
            or adjustment_plan.get("reason")
            or "当前预算和约束下没有完整可行的航班酒店组合。"
        ),
        "need_user_input": True,
        "available_actions": [
            "increase_budget",
            "relax_constraints",
            "change_dates",
            "cancel",
        ],
        "details": {
            "current_total_budget": (
                adjustment_plan.get(
                    "current_total_budget"
                )
            ),
            "minimum_known_subtotal": (
                adjustment_plan.get(
                    "minimum_known_subtotal"
                )
            ),
            "required_budget_increase": (
                adjustment_plan.get(
                    "required_budget_increase"
                )
            ),
            "suggestions": _string_list(
                adjustment_plan.get(
                    "suggestions"
                )
            ),
        },
    }


def _attach_common_metadata(
    *,
    response: dict[str, Any],
    state: Mapping[str, Any],
) -> None:
    """补充稳定会话元信息，不覆盖已有业务字段。"""

    response.setdefault(
        "workflow",
        state.get("workflow"),
    )
    response.setdefault(
        "trip_session_id",
        state.get("trip_session_id"),
    )


def _mapping(
    value: Any,
) -> dict[str, Any]:
    """把 Mapping 安全复制为普通 dict。"""

    return (
        dict(value)
        if isinstance(value, Mapping)
        else {}
    )


def _string_list(
    value: Any,
) -> list[str]:
    """把列表安全转换为非空字符串列表。"""

    if not isinstance(value, list):
        return []

    return [
        str(item)
        for item in value
        if item is not None
        and str(item).strip()
    ]


def _public_issues(
    value: Any,
) -> list[dict[str, Any]]:
    """只返回适合前端的少量问题字段。"""

    if not isinstance(value, list):
        return []

    output: list[dict[str, Any]] = []

    for item in value[:10]:
        if not isinstance(item, Mapping):
            continue

        output.append(
            {
                "type": (
                    item.get("issue_type")
                    or item.get("type")
                ),
                "message": item.get(
                    "message"
                ),
                "category": item.get(
                    "category"
                ),
                "path": item.get("path"),
            }
        )

    return output


def _public_coverage(
    value: Any,
) -> dict[str, Any]:
    """压缩 Evidence coverage，只保留数量和 passed。"""

    if not isinstance(value, Mapping):
        return {}

    output: dict[str, Any] = {}

    for category, raw in value.items():
        if not isinstance(raw, Mapping):
            continue

        output[str(category)] = {
            "required": raw.get("required"),
            "required_chunks": raw.get(
                "required_chunks"
            ),
            "actual_chunks": raw.get(
                "actual_chunks"
            ),
            "passed": raw.get("passed"),
        }

    return output


def _public_semantic_review(
    value: Any,
) -> dict[str, Any]:
    """返回未通过的 need 和原因，不暴露 Prompt 或完整证据正文。"""

    if not isinstance(value, Mapping):
        return {}

    gaps: list[dict[str, Any]] = []
    raw_coverage = value.get("need_coverage")
    if isinstance(raw_coverage, list):
        for item in raw_coverage:
            if not isinstance(item, Mapping):
                continue
            status = str(item.get("status") or "")
            if status not in {"partial", "missing"}:
                continue
            gaps.append(
                {
                    "need_id": item.get("need_id"),
                    "status": status,
                    "reason": item.get("reason"),
                }
            )

    return {
        "status": value.get("status"),
        "gaps": gaps[:10],
    }


def _public_errors(
    value: Any,
) -> list[dict[str, Any]]:
    """只返回最后几条错误的节点、类型和消息。"""

    if not isinstance(value, list):
        return []

    output: list[dict[str, Any]] = []

    for item in value[-5:]:
        if not isinstance(item, Mapping):
            continue

        output.append(
            {
                "node": item.get("node"),
                "type": item.get("type"),
                "message": item.get(
                    "message"
                ),
            }
        )

    return output
