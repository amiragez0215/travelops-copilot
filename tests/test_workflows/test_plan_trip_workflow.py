from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import START
from langgraph.types import Command

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover - 兼容较早 LangGraph
    from langgraph.checkpoint.memory import (
        MemorySaver as InMemorySaver,
    )

from app.agents.state import TravelState
from app.nodes.decision_gate_node import (
    decision_gate_node,
)
from app.workflows.plan_trip_workflow import (
    PLAN_TRIP_NODE_NAMES,
    build_initial_plan_trip_state,
    build_plan_trip_config,
    build_plan_trip_graph,
    create_plan_trip_builder,
)


Node = Callable[[TravelState], Any]


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any]:
    """读取 graph.invoke 返回的第一个 Interrupt Payload。"""

    item = result["__interrupt__"][0]
    return getattr(item, "value", item)


def _recording_node(
    name: str,
    calls: list[str],
    update: dict[str, Any] | None = None,
) -> Node:
    """构造记录调用顺序的 Fake Node。"""

    def node(state: TravelState) -> dict[str, Any]:
        del state
        calls.append(name)
        return dict(update or {})

    return node


def _base_fake_nodes(calls: list[str]) -> dict[str, Node]:
    """为全部 Graph Node 提供默认 Fake，避免加载外部服务。"""

    return {
        name: _recording_node(name, calls)
        for name in PLAN_TRIP_NODE_NAMES
    }


def _normal_routes() -> dict[str, Callable[[TravelState], str]]:
    """构造一路进入正常审批分支的测试路由。"""

    return {
        "missing_info_check": (
            lambda state: (
                "safety_check"
                if state.get("can_continue") is True
                else "clarify"
            )
        ),
        "safety_check": (
            lambda state: (
                "safe_reject"
                if (
                    state.get("safety_result")
                    or {}
                ).get("blocked") is True
                else "memory_read"
            )
        ),
        "tool_check": (
            lambda state: (
                state.get("tool_check_result")
                or {}
            ).get("next_action", "candidate_rank")
        ),
        "candidate_rank": (
            lambda state: (
                "budget_optimize"
                if (
                    state.get("candidate_rank_result")
                    or {}
                ).get("status") == "ok"
                else "final_response"
            )
        ),
        "budget_optimize": (
            lambda state: (
                "retrieval_plan"
                if (
                    state.get("selection_result")
                    or {}
                ).get("status") == "feasible"
                else "final_response"
            )
        ),
        "evidence_grade": (
            lambda state: {
                "continue": "proposal",
                "repair": "retrieval_plan",
                "stop": "final_response",
            }.get(
                (
                    state.get("evidence_result")
                    or {}
                ).get("next_action"),
                "final_response",
            )
        ),
        "proposal": (
            lambda state: (
                "verifier"
                if (
                    state.get("proposal_result")
                    or {}
                ).get("status") == "generated"
                else "final_response"
            )
        ),
        "verifier": (
            lambda state: (
                state.get("verifier_result")
                or {}
            ).get(
                "next_action",
                "final_response",
            )
        ),
        "revision_analyze": (
            lambda state: (
                "apply_revision"
                if (
                    state.get("revision_plan")
                    or {}
                ).get("status") == "ready"
                else "final_response"
            )
        ),
        "apply_revision": (
            lambda state: (
                "missing_info_check"
                if (
                    state.get("revision_result")
                    or {}
                ).get("status") == "applied"
                else "final_response"
            )
        ),
    }


def _proposal(proposal_id: str, version: int) -> dict[str, Any]:
    """构造 DecisionGate Payload 所需的最小 Proposal。"""

    return {
        "proposal_id": proposal_id,
        "version": version,
        "summary": "测试旅行方案。",
        "highlights": ["天气适配", "预算可行"],
        "trip_overview": {
            "origin": "杭州",
            "destination": "成都",
            "start_date": "2026-07-02",
            "end_date": "2026-07-04",
            "days": 3,
            "people_count": 1,
        },
        "selected_flights": {
            "outbound": {
                "flight_id": "out_001",
                "flight_no": "MU1001",
                "depart_time": "09:00",
                "arrive_time": "11:30",
            },
            "return_flight": {
                "flight_id": "ret_001",
                "flight_no": "MU1002",
                "depart_time": "16:00",
                "arrive_time": "18:30",
            },
        },
        "selected_hotel": {
            "hotel_id": "hotel_001",
            "name": "测试酒店",
            "estimated_total_price": 1000,
        },
        "budget_summary": {
            "total_budget": 4000,
            "known_costs": {
                "known_subtotal": 2500,
            },
            "remaining_budget": 1500,
        },
        "execution_boundary": {
            "draft_only": True,
            "real_booking_performed": False,
            "real_payment_performed": False,
            "real_notification_sent": False,
        },
    }


def _verifier_update(
    proposal_id: str,
    version: int,
) -> dict[str, Any]:
    """构造 Verifier 通过和 Pending Action。"""

    return {
        "verifier_result": {
            "status": "passed",
            "passed": True,
            "repairable": False,
            "next_action": "decision_gate",
            "approval_required": True,
        },
        "pending_actions": [
            {
                "action_id": (
                    "approve_" + proposal_id
                ),
                "action_type": "approve_proposal",
                "description": "确认测试方案。",
                "requires_approval": True,
                "status": "pending",
                "payload_summary": {
                    "proposal_id": proposal_id,
                    "version": version,
                    "real_booking_performed": False,
                },
            }
        ],
    }


def _normal_fake_nodes(
    calls: list[str],
) -> dict[str, Node]:
    """构造可以一路运行到 DecisionGate 的 Fake Node 集合。"""

    nodes = _base_fake_nodes(calls)

    nodes["input_extract"] = _recording_node(
        "input_extract",
        calls,
        {
            "trip_request": {
                "origin": "杭州",
                "destination": "成都",
                "start_date": "2026-07-02",
                "end_date": "2026-07-04",
                "days": 3,
                "budget": 4000,
                "people_count": 1,
                "room_count": 1,
            }
        },
    )
    nodes["missing_info_check"] = _recording_node(
        "missing_info_check",
        calls,
        {
            "can_continue": True,
            "missing_fields": [],
        },
    )
    nodes["safety_check"] = _recording_node(
        "safety_check",
        calls,
        {
            "safety_result": {
                "blocked": False,
            }
        },
    )
    nodes["candidate_rank"] = _recording_node(
        "candidate_rank",
        calls,
        {
            "candidate_rank_result": {
                "status": "ok",
            }
        },
    )
    nodes["budget_optimize"] = _recording_node(
        "budget_optimize",
        calls,
        {
            "selection_result": {
                "status": "feasible",
            }
        },
    )
    nodes["evidence_grade"] = _recording_node(
        "evidence_grade",
        calls,
        {
            "evidence_result": {
                "status": "passed",
                "passed": True,
                "next_action": "continue",
            }
        },
    )

    def proposal_node(state: TravelState) -> dict[str, Any]:
        calls.append("proposal")
        revised = (
            state.get("revision_result")
            or {}
        ).get("status") == "applied"
        proposal_id = (
            "proposal_v2"
            if revised
            else "proposal_v1"
        )
        version = 2 if revised else 1
        return {
            "proposal": _proposal(
                proposal_id,
                version,
            ),
            "proposal_result": {
                "status": "generated",
            },
            "itinerary_status": "proposed",
        }

    nodes["proposal"] = proposal_node

    def verifier_node(state: TravelState) -> dict[str, Any]:
        calls.append("verifier")
        proposal = state["proposal"]
        return _verifier_update(
            proposal["proposal_id"],
            proposal["version"],
        )

    nodes["verifier"] = verifier_node

    # 使用真实 DecisionGateNode 测试 interrupt / resume / Command.goto。
    nodes["decision_gate"] = decision_gate_node

    def commit_node(state: TravelState) -> dict[str, Any]:
        calls.append("commit_draft")
        return {
            "commit_result": {
                "status": "committed",
            },
            "itinerary_status": (
                "approved_simulated"
            ),
        }

    nodes["commit_draft"] = commit_node

    def cancel_node(state: TravelState) -> dict[str, Any]:
        calls.append("cancel")
        return {
            "cancel_result": {
                "status": "cancelled",
            },
            "itinerary_status": "rejected",
        }

    nodes["cancel"] = cancel_node

    def final_response_node(state: TravelState) -> dict[str, Any]:
        calls.append("final_response")
        if (
            state.get("commit_result")
            or {}
        ).get("status") == "committed":
            status = "approved_simulated"
        elif (
            state.get("cancel_result")
            or {}
        ).get("status") == "cancelled":
            status = "cancelled"
        else:
            status = "completed"
        return {
            "final_response": {
                "status": status,
            }
        }

    nodes["final_response"] = final_response_node
    return nodes


def test_builder_registers_complete_graph_and_no_static_decision_edge():
    """Graph 应注册全部节点，并让 DecisionGate 只使用 Command.goto。"""

    calls: list[str] = []
    builder = create_plan_trip_builder(
        node_overrides=_base_fake_nodes(
            calls
        ),
        route_overrides=_normal_routes(),
    )

    assert set(PLAN_TRIP_NODE_NAMES) == set(
        builder.nodes
    )
    assert (
        START,
        "input_extract",
    ) in builder.edges
    assert (
        "final_response",
        "__end__",
    ) in builder.edges

    # DecisionGate 的去向由 Command.goto 决定，不能有普通静态出边。
    assert not any(
        source == "decision_gate"
        for source, _ in builder.edges
    )

    for source in (
        "missing_info_check",
        "safety_check",
        "tool_check",
        "candidate_rank",
        "budget_optimize",
        "evidence_grade",
        "proposal",
        "verifier",
        "revision_analyze",
        "apply_revision",
    ):
        assert source in builder.branches


def test_missing_info_branch_reaches_final_response():
    """缺字段分支应只执行 Clarify 和 FinalResponse。"""

    calls: list[str] = []
    nodes = _base_fake_nodes(calls)
    nodes["input_extract"] = _recording_node(
        "input_extract",
        calls,
        {"trip_request": {}},
    )
    nodes["missing_info_check"] = _recording_node(
        "missing_info_check",
        calls,
        {
            "can_continue": False,
            "missing_fields": [
                "start_date"
            ],
        },
    )
    nodes["clarify"] = _recording_node(
        "clarify",
        calls,
        {
            "final_response": {
                "status": (
                    "needs_clarification"
                )
            }
        },
    )
    nodes["final_response"] = _recording_node(
        "final_response",
        calls,
    )

    graph = build_plan_trip_graph(
        checkpointer=InMemorySaver(),
        node_overrides=nodes,
        route_overrides=_normal_routes(),
    )

    state = build_initial_plan_trip_state(
        user_id="user_001",
        raw_message="想去成都玩。",
        reference_date="2026-07-01",
        trip_session_id="clarify-thread",
    )

    result = graph.invoke(
        state,
        config=build_plan_trip_config(
            thread_id="clarify-thread"
        ),
    )

    assert calls == [
        "input_extract",
        "missing_info_check",
        "clarify",
        "final_response",
    ]
    assert (
        result["final_response"]["status"]
        == "needs_clarification"
    )


def test_normal_plan_interrupt_resume_approve_closed_loop():
    """完整正常主链应暂停审批，批准后提交并结束。"""

    calls: list[str] = []
    graph = build_plan_trip_graph(
        checkpointer=InMemorySaver(),
        node_overrides=_normal_fake_nodes(
            calls
        ),
        route_overrides=_normal_routes(),
    )

    thread_id = "approve-full-graph"
    config = build_plan_trip_config(
        thread_id=thread_id
    )
    initial_state = build_initial_plan_trip_state(
        user_id="user_001",
        raw_message="杭州到成都三日游。",
        reference_date="2026-07-01",
        trip_session_id=thread_id,
    )

    paused = graph.invoke(
        initial_state,
        config=config,
    )
    payload = _interrupt_payload(paused)

    assert payload["proposal_id"] == "proposal_v1"
    assert calls[-1] == "verifier"
    assert "commit_draft" not in calls

    final_state = graph.invoke(
        Command(
            resume={
                "decision": "approve",
                "proposal_id": payload[
                    "proposal_id"
                ],
                "action_id": payload[
                    "action_id"
                ],
            }
        ),
        config=config,
    )

    assert final_state["decision"] == "approve"
    assert (
        final_state["final_response"]["status"]
        == "approved_simulated"
    )
    assert calls[-2:] == [
        "commit_draft",
        "final_response",
    ]


def test_request_changes_reenters_main_chain_and_interrupts_new_proposal():
    """修改分支应应用 Patch 后从 MissingInfoCheck 重新跑完整主链。"""

    calls: list[str] = []
    nodes = _normal_fake_nodes(calls)

    nodes["revision_analyze"] = _recording_node(
        "revision_analyze",
        calls,
        {
            "revision_plan": {
                "status": "ready",
            }
        },
    )

    def apply_revision_node(
        state: TravelState,
    ) -> dict[str, Any]:
        calls.append("apply_revision")
        request = dict(
            state.get("trip_request")
            or {}
        )
        request["budget"] = 5000
        return {
            "trip_request": request,
            "revision_result": {
                "status": "applied",
            },
            "workflow": "modify_trip",
            "itinerary_status": (
                "collecting_info"
            ),
            "proposal": {},
            "proposal_result": {},
            "verifier_result": {},
            "pending_actions": [],
            "decision": None,
            "decision_result": {},
            "decision_attempt_count": 0,
        }

    nodes["apply_revision"] = (
        apply_revision_node
    )

    graph = build_plan_trip_graph(
        checkpointer=InMemorySaver(),
        node_overrides=nodes,
        route_overrides=_normal_routes(),
    )

    thread_id = "revision-full-graph"
    config = build_plan_trip_config(
        thread_id=thread_id
    )
    initial_state = build_initial_plan_trip_state(
        user_id="user_001",
        raw_message="杭州到成都三日游。",
        reference_date="2026-07-01",
        trip_session_id=thread_id,
    )

    first_paused = graph.invoke(
        initial_state,
        config=config,
    )
    first_payload = _interrupt_payload(
        first_paused
    )

    # 1. 用户提交修改后，同一次 resume 会继续 Revision Loop，
    #    直到新 Proposal 再次到达 DecisionGate。
    second_paused = graph.invoke(
        Command(
            resume={
                "decision": (
                    "request_changes"
                ),
                "proposal_id": first_payload[
                    "proposal_id"
                ],
                "action_id": first_payload[
                    "action_id"
                ],
                "change_request": (
                    "预算增加1000元。"
                ),
            }
        ),
        config=config,
    )
    second_payload = _interrupt_payload(
        second_paused
    )

    assert second_payload["proposal_id"] == (
        "proposal_v2"
    )
    assert "revision_analyze" in calls
    assert "apply_revision" in calls

    # ApplyRevision 后不会再次执行 InputExtract，
    # 而是从 MissingInfoCheck 重新进入主链。
    assert calls.count("input_extract") == 1
    assert calls.count(
        "missing_info_check"
    ) == 2
    assert calls.count("proposal") == 2
    assert calls.count("verifier") == 2

    # 2. 新 Proposal 再次接受人工批准。
    final_state = graph.invoke(
        Command(
            resume={
                "decision": "approve",
                "proposal_id": second_payload[
                    "proposal_id"
                ],
                "action_id": second_payload[
                    "action_id"
                ],
            }
        ),
        config=config,
    )

    assert (
        final_state["final_response"]["status"]
        == "approved_simulated"
    )
    assert len(
        final_state["decision_history"]
    ) == 2


def test_initial_state_and_config_share_session_identity():
    """trip_session_id 应与 Checkpointer thread_id 使用同一稳定值。"""

    state = build_initial_plan_trip_state(
        user_id="user_001",
        raw_message="测试请求",
        reference_date="2026-07-01",
        trip_session_id="trip-session-001",
    )
    config = build_plan_trip_config(
        thread_id=state[
            "trip_session_id"
        ],
        recursion_limit=80,
    )

    assert state["workflow"] == "plan_trip"
    assert state["rag_retry_count"] == 0
    assert (
        config["configurable"]["thread_id"]
        == "trip-session-001"
    )
    assert config["recursion_limit"] == 80
