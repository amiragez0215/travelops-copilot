from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
    build_plan_trip_graph,
)
from app.workflows.workflow_runtime import (
    PlanTripWorkflowRuntime,
)


FakeNode = Callable[[TravelState], Any]


def build_fake_plan_trip_graph(
    *,
    checkpointer: Any,
    calls: list[str] | None = None,
):
    """
    构建一个不调用 DeepSeek、MCP、Chroma 和业务数据库的完整 Fake Graph。

    这个 Fake Graph 保留真实的：
        - 完整 PlanTrip 节点拓扑；
        - Conditional Edge；
        - DecisionGateNode interrupt / resume；
        - request_changes 后重新进入主链；
        - approve / cancel 最终收尾。

    只把昂贵或有副作用的业务节点替换成确定性 Fake。
    因此它非常适合 API 和 Workflow Runtime 端到端测试。
    """

    call_log = calls if calls is not None else []

    graph = build_plan_trip_graph(
        checkpointer=checkpointer,
        node_overrides=_build_fake_nodes(
            call_log
        ),
        route_overrides=_build_fake_routes(),
    )

    return graph


def build_fake_runtime(
    *,
    calls: list[str] | None = None,
    checkpointer: Any | None = None,
) -> PlanTripWorkflowRuntime:
    """
    构建测试用 PlanTripWorkflowRuntime。

    默认使用 InMemorySaver；
    Durable Execution 测试可以显式传入 SQLite SqliteSaver。
    """

    active_checkpointer = (
        checkpointer
        if checkpointer is not None
        else InMemorySaver()
    )

    graph = build_fake_plan_trip_graph(
        checkpointer=active_checkpointer,
        calls=calls,
    )

    return PlanTripWorkflowRuntime(
        graph=graph,
        checkpointer=active_checkpointer,
        recursion_limit=100,
    )


def first_interrupt_payload(
    graph_output: Any,
) -> dict[str, Any]:
    """读取 graph.invoke() v1 返回值中的第一个 interrupt payload。"""

    raw_items = graph_output["__interrupt__"]
    item = raw_items[0]
    value = getattr(item, "value", item)
    return dict(value)


def _recording_node(
    name: str,
    calls: list[str],
    update: dict[str, Any] | None = None,
) -> FakeNode:
    """创建一个记录调用顺序并返回固定 State Update 的 Fake Node。"""

    def node(
        state: TravelState,
    ) -> dict[str, Any]:
        del state
        calls.append(name)
        return dict(update or {})

    return node


def _build_fake_nodes(
    calls: list[str],
) -> dict[str, FakeNode]:
    """
    为正式主图的所有节点创建测试替身。

    注意：DecisionGate 使用真实节点，
    这样测试的 interrupt / Command.goto 行为与正式项目一致。
    """

    nodes = {
        name: _recording_node(
            name,
            calls,
        )
        for name in PLAN_TRIP_NODE_NAMES
    }

    # ------------------------------------------------------------------
    # 1. 输入、校验和安全检查。
    # ------------------------------------------------------------------
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

    nodes["tool_check"] = _recording_node(
        "tool_check",
        calls,
        {
            "tool_check_result": {
                "status": "passed",
                "next_action": "candidate_rank",
            }
        },
    )

    # ------------------------------------------------------------------
    # 2. Candidate 和预算节点只返回路由所需的最小结果。
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. 首次生成 proposal_v1；ApplyRevision 后生成 proposal_v2。
    # ------------------------------------------------------------------
    def proposal_node(
        state: TravelState,
    ) -> dict[str, Any]:
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

    def verifier_node(
        state: TravelState,
    ) -> dict[str, Any]:
        calls.append("verifier")

        proposal = state["proposal"]
        proposal_id = proposal["proposal_id"]
        version = proposal["version"]

        return {
            "verifier_result": {
                "status": "passed",
                "passed": True,
                "repairable": False,
                "next_action": "decision_gate",
                "approval_required": True,
                "proposal_id": proposal_id,
                "warning_count": 0,
                "error_count": 0,
                "critical_count": 0,
            },
            "pending_actions": [
                {
                    "action_id": (
                        "approve_" + proposal_id
                    ),
                    "action_type": (
                        "approve_proposal"
                    ),
                    "description": (
                        "确认测试旅行方案。"
                    ),
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

    nodes["verifier"] = verifier_node

    # 4. 这里故意使用真实 DecisionGateNode，
    #    让 Fake Graph 仍然具有真正的 Human-in-the-loop 行为。
    nodes["decision_gate"] = decision_gate_node

    # ------------------------------------------------------------------
    # 5. 收尾节点不访问真实数据库。
    # ------------------------------------------------------------------
    def commit_node(
        state: TravelState,
    ) -> dict[str, Any]:
        calls.append("commit_draft")
        return {
            "commit_result": {
                "status": "committed",
                "idempotent_replay": False,
            },
            "itinerary_status": (
                "approved_simulated"
            ),
        }

    nodes["commit_draft"] = commit_node

    def cancel_node(
        state: TravelState,
    ) -> dict[str, Any]:
        calls.append("cancel")
        return {
            "cancel_result": {
                "status": "cancelled",
            },
            "itinerary_status": "rejected",
        }

    nodes["cancel"] = cancel_node

    # ------------------------------------------------------------------
    # 6. Revision Loop：分析修改并清理旧 Proposal 后重新进入主链。
    # ------------------------------------------------------------------
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

        trip_request = dict(
            state.get("trip_request")
            or {}
        )
        trip_request["budget"] = 5000

        return {
            "trip_request": trip_request,
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

    def final_response_node(
        state: TravelState,
    ) -> dict[str, Any]:
        calls.append("final_response")

        if (
            state.get("commit_result")
            or {}
        ).get("status") in {
            "committed",
            "already_committed",
        }:
            response_status = (
                "approved_simulated"
            )
        elif (
            state.get("cancel_result")
            or {}
        ).get("status") in {
            "cancelled",
            "already_cancelled",
        }:
            response_status = "cancelled"
        else:
            response_status = "completed"

        return {
            "final_response": {
                "status": response_status,
                "message": "Fake workflow completed.",
            }
        }

    nodes["final_response"] = (
        final_response_node
    )

    return nodes


def _build_fake_routes() -> dict[
    str,
    Callable[[TravelState], str],
]:
    """构造正常主链与 Revision Loop 使用的确定性路由。"""

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
            ).get("next_action", "final_response")
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


def _proposal(
    proposal_id: str,
    version: int,
) -> dict[str, Any]:
    """构造 DecisionGate 审批页面需要的最小旅行方案。"""

    return {
        "proposal_id": proposal_id,
        "version": version,
        "summary": "测试旅行方案。",
        "highlights": [
            "天气适配",
            "预算可行",
        ],
        "trip_overview": {
            "origin": "杭州",
            "destination": "成都",
            "start_date": "2026-07-02",
            "end_date": "2026-07-04",
            "days": 3,
            "nights": 2,
            "people_count": 1,
            "room_count": 1,
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
        # Fake API 回归不仅要覆盖审批中断，也要保留用户页真正需要的
        # daily_plan 结构。这样测试能及时发现“后端有逐日行程、前端却
        # 没有按天展示”的合同退化，而不会把嵌套活动简化成一行字符串。
        "daily_plan": [
            {
                "day_index": 1,
                "date": "2026-07-02",
                "theme": "抵达后的城市漫游",
                "weather": {
                    "condition": "多云",
                    "temperature_low": 23,
                    "temperature_high": 30,
                },
                "activities": [
                    {
                        "period": "afternoon",
                        "title": "办理入住并在附近休息",
                        "description": "抵达后预留缓冲时间，傍晚再开始轻松活动。",
                        "practical_notes": ["将大件行李寄存后再步行探索"],
                    },
                    {
                        "period": "evening",
                        "title": "街区美食体验",
                        "description": "选择交通方便的街区，安排晚餐和夜间散步。",
                    },
                ],
                "day_notes": ["当天行程以适应交通和休息为主。"],
            },
            {
                "day_index": 2,
                "date": "2026-07-03",
                "theme": "文化与城市生活",
                "weather": {"condition": "晴", "temperature_low": 24, "temperature_high": 32},
                "activities": [
                    {
                        "period": "morning",
                        "title": "博物馆或文化空间",
                        "description": "上午安排室内文化体验，避开午间高温。",
                    },
                    {
                        "period": "evening",
                        "title": "夜间餐饮与自由活动",
                        "description": "根据体力在酒店周边安排弹性晚间行程。",
                    },
                ],
            },
            {
                "day_index": 3,
                "date": "2026-07-04",
                "theme": "返程前的轻松收尾",
                "weather": {"condition": "小雨", "temperature_low": 22, "temperature_high": 28},
                "activities": [
                    {
                        "period": "morning",
                        "title": "室内早餐与整理行李",
                        "description": "优先选择酒店附近的室内安排，预留返程交通时间。",
                    },
                ],
                "weather_adjustment": "有雨时减少户外步行，并为前往机场预留交通缓冲。",
            },
        ],
        "execution_boundary": {
            "draft_only": True,
            "real_booking_performed": False,
            "real_payment_performed": False,
            "real_notification_sent": False,
        },
    }
