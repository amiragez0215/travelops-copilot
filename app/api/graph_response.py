from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.api.travel_schemas import (
    PublicTraceItem,
    TravelRunHistoryItem,
    TravelRunHistoryResponse,
    TravelRunResponse,
    VerificationSummary,
)
from app.workflows.snapshot_utils import (
    get_interrupt_values,
    snapshot_checkpoint_id,
    snapshot_exists,
    snapshot_next_nodes,
    snapshot_values,
)



def build_public_run_response(
    *,
    thread_id: str,
    graph_output: Any | None,
    snapshot: Any,
    run_id: str | None = None,
    include_trace: bool = False,
    max_trace_items: int = 50,
) -> TravelRunResponse:
    """
    将 LangGraph 内部运行结果转换成稳定的公开 API Response。

    这个函数是 Stage 1 的关键边界之一：

        内部 TravelState
            ↓
        只选择前端真正需要的字段
            ↓
        TravelRunResponse

    明确不返回：
        - user_profile；
        - raw_flight_results / raw_hotel_results；
        -完整 evidence_pool 正文；
        - Prompt；
        - Checkpointer 私有对象；
        - Python 异常堆栈。
    """

    # 1. Checkpointer 中的 snapshot.values 是当前 Thread 的权威状态。
    values = snapshot_values(snapshot)

    # 2. 启动 / Resume 刚刚命中 interrupt 时，优先从本次 invoke 输出读取；
    #    GET state 没有 invoke 输出时，再从 snapshot.tasks 中恢复。
    interrupts = get_interrupt_values(
        graph_output=graph_output,
        snapshot=snapshot,
    )

    next_nodes = snapshot_next_nodes(snapshot)
    final_response = _optional_mapping(
        values.get("final_response")
    )
    proposal = _optional_mapping(
        values.get("proposal")
    )
    verifier_result = _optional_mapping(
        values.get("verifier_result")
    )
    decision_result = _optional_mapping(
        values.get("decision_result")
    )

    # 3. 运行状态只根据 Runtime 的可观察事实判断，
    #    不让某个业务 Node 自己决定整个 Thread 是否已结束。
    run_status = _resolve_run_status(
        has_interrupt=bool(interrupts),
        next_nodes=next_nodes,
        final_response=final_response,
        state_values=values,
    )

    public_trace: list[PublicTraceItem] = []

    if include_trace:
        public_trace = _build_public_trace(
            values.get("trace"),
            max_items=max_trace_items,
        )

    verification = (
        _build_verification_summary(
            verifier_result
        )
        if verifier_result
        else None
    )

    return TravelRunResponse(
        thread_id=str(thread_id),
        # run_id 只在本次 invoke / resume 的同步响应中存在；GET state 没有
        # 产生新 Run，因此保持 None，不能误报一个历史 Run 为“当前正在跑”。
        run_id=str(run_id) if run_id else None,
        run_status=run_status,
        workflow=_optional_string(
            values.get("workflow")
        ),
        itinerary_status=_optional_string(
            values.get("itinerary_status")
        ),
        checkpoint_id=(
            snapshot_checkpoint_id(snapshot)
        ),
        next_nodes=next_nodes,
        proposal=proposal,
        verification=verification,
        interrupt=(
            interrupts[0]
            if interrupts
            else None
        ),
        final_response=final_response,
        decision_result=decision_result,
        trace=public_trace,
    )



def build_public_history_response(
    *,
    thread_id: str,
    snapshots: Sequence[Any],
) -> TravelRunHistoryResponse:
    """
    将 StateSnapshot 历史转换成不泄漏完整 State 的摘要列表。

    LangGraph 历史默认通常是最新 Snapshot 在前；
    本函数保持 Runtime 返回顺序，不自行重排。
    """

    items = [
        _build_history_item(snapshot)
        for snapshot in snapshots
        if snapshot_exists(snapshot)
    ]

    return TravelRunHistoryResponse(
        thread_id=str(thread_id),
        count=len(items),
        items=items,
    )



def _build_history_item(
    snapshot: Any,
) -> TravelRunHistoryItem:
    """构造单条公开 Checkpoint 摘要。"""

    values = snapshot_values(snapshot)
    interrupts = get_interrupt_values(
        snapshot=snapshot
    )
    next_nodes = snapshot_next_nodes(snapshot)
    final_response = _optional_mapping(
        values.get("final_response")
    )

    metadata = getattr(
        snapshot,
        "metadata",
        None,
    )
    metadata = (
        dict(metadata)
        if isinstance(metadata, Mapping)
        else {}
    )

    proposal = _optional_mapping(
        values.get("proposal")
    )

    return TravelRunHistoryItem(
        checkpoint_id=(
            snapshot_checkpoint_id(snapshot)
        ),
        created_at=_optional_string(
            getattr(
                snapshot,
                "created_at",
                None,
            )
        ),
        step=_optional_int(
            metadata.get("step")
        ),
        source=_optional_string(
            metadata.get("source")
        ),
        run_status=_resolve_run_status(
            has_interrupt=bool(interrupts),
            next_nodes=next_nodes,
            final_response=final_response,
            state_values=values,
        ),
        next_nodes=next_nodes,
        has_interrupt=bool(interrupts),
        itinerary_status=_optional_string(
            values.get("itinerary_status")
        ),
        proposal_id=(
            _optional_string(
                proposal.get("proposal_id")
            )
            if proposal
            else None
        ),
    )



def _resolve_run_status(
    *,
    has_interrupt: bool,
    next_nodes: Sequence[str],
    final_response: Mapping[str, Any] | None,
    state_values: Mapping[str, Any],
) -> str:
    """
    根据 Checkpoint 的可观察状态推导公开 run_status。

    优先级：
        interrupt > final_response > still has next > failed/completed
    """

    if has_interrupt:
        return "waiting_for_decision"

    if final_response:
        return "completed"

    if next_nodes:
        return "running"

    # Graph 已没有 next node，但也没有 final_response。
    # 如果存在程序 errors，公开为 failed；否则按 completed 处理。
    errors = state_values.get("errors")

    if isinstance(errors, list) and errors:
        return "failed"

    return "completed"



def _build_verification_summary(
    value: Mapping[str, Any],
) -> VerificationSummary:
    """只暴露审批页面和状态查询需要的 Verifier 摘要。"""

    return VerificationSummary(
        status=_optional_string(
            value.get("status")
        ),
        passed=_optional_bool(
            value.get("passed")
        ),
        proposal_id=_optional_string(
            value.get("proposal_id")
        ),
        next_action=_optional_string(
            value.get("next_action")
        ),
        warning_count=_optional_int(
            value.get("warning_count")
        ),
        error_count=_optional_int(
            value.get("error_count")
        ),
        critical_count=_optional_int(
            value.get("critical_count")
        ),
    )



def _build_public_trace(
    value: Any,
    *,
    max_items: int,
) -> list[PublicTraceItem]:
    """
    把内部 Trace 转为最近 N 条公开摘要。

    当前 Trace 本身已经是摘要，但这里再次使用字段白名单，
    防止未来某个 Node 给 Trace 增加 Prompt 或完整 Evidence 后被 API 误暴露。
    """

    if not isinstance(value, list):
        return []

    safe_limit = max(int(max_items), 0)

    if safe_limit == 0:
        return []

    selected = value[-safe_limit:]
    result: list[PublicTraceItem] = []

    for item in selected:
        if not isinstance(item, Mapping):
            continue

        result.append(
            PublicTraceItem(
                node_name=_optional_string(
                    item.get("node_name")
                    or item.get("node")
                ),
                tool_name=_optional_string(
                    item.get("tool_name")
                ),
                status=_optional_string(
                    item.get("status")
                ),
                latency_ms=_optional_int(
                    item.get("latency_ms")
                ),
                input_summary=_truncate(
                    item.get("input_summary"),
                    500,
                ),
                output_summary=_truncate(
                    item.get("output_summary"),
                    500,
                ),
                created_at=_optional_string(
                    item.get("created_at")
                ),
            )
        )

    return result



def _optional_mapping(
    value: Any,
) -> dict[str, Any] | None:
    """Mapping 非空时复制成普通 dict，否则返回 None。"""

    if isinstance(value, Mapping) and value:
        return dict(value)

    return None



def _optional_string(
    value: Any,
) -> str | None:
    """把非空值安全转换成字符串。"""

    if value is None:
        return None

    text = str(value).strip()
    return text or None



def _optional_int(
    value: Any,
) -> int | None:
    """安全转换可空整数。"""

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None



def _optional_bool(
    value: Any,
) -> bool | None:
    """只接受真正的 bool，避免字符串 'false' 被误判为 True。"""

    if isinstance(value, bool):
        return value

    return None



def _truncate(
    value: Any,
    max_chars: int,
) -> str | None:
    """将 Trace 文本限制在安全长度内。"""

    text = _optional_string(value)

    if text is None:
        return None

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."
