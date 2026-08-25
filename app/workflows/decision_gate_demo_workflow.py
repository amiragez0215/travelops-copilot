from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.state import TravelState
from app.nodes.decision_gate_node import (
    decision_gate_node,
)


def build_decision_gate_demo_graph(
    *,
    checkpointer: Any,
):
    """
    构建只用于学习和测试的 DecisionGate 最小闭环。

    这个 Demo 不是最终 PlanTripWorkflow。
    它只证明四件事：
        1. interrupt 能暂停；
        2. checkpointer 能保存状态；
        3. Command(resume=...) 能恢复；
        4. approve / request_changes / cancel 能动态路由。
    """

    builder = StateGraph(
        TravelState
    )

    # 1. DecisionGate 使用 Command.goto 动态路由，
    #    所以不要再给它添加普通静态出边。
    builder.add_node(
        "decision_gate",
        decision_gate_node,
    )

    # 2. 下面三个节点只是 Demo 终点。
    #    最终项目会替换成真正的：
    #        CommitDraftNode
    #        RevisionAnalyzeNode
    #        CancelNode
    builder.add_node(
        "commit_draft",
        _demo_commit_draft_node,
    )

    builder.add_node(
        "revision_analyze",
        _demo_revision_analyze_node,
    )

    builder.add_node(
        "cancel",
        _demo_cancel_node,
    )

    builder.add_node(
        "final_response",
        _demo_final_response_node,
    )

    # 3. 最小图从 DecisionGate 开始。
    builder.add_edge(
        START,
        "decision_gate",
    )

    # 4. Demo 下游节点结束后进入 END。
    builder.add_edge(
        "commit_draft",
        END,
    )

    builder.add_edge(
        "revision_analyze",
        END,
    )

    builder.add_edge(
        "cancel",
        END,
    )

    builder.add_edge(
        "final_response",
        END,
    )

    # 5. Checkpointer 是 interrupt / resume 的必要基础设施。
    return builder.compile(
        checkpointer=checkpointer
    )


def build_demo_decision_state() -> TravelState:
    """
    构造一个已通过 Verifier、等待人工审批的最小 State。
    """

    proposal_id = "proposal_demo_001"
    action_id = (
        "approve_" + proposal_id
    )

    return {
        "proposal": {
            "proposal_id": proposal_id,
            "version": 1,
            "summary": "成都三日慢节奏旅行方案。",
            "highlights": [
                "雨天安排室内文化活动",
                "选择安静且靠近地铁的酒店",
            ],
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
                    "flight_id": "flight_out_001",
                    "flight_no": "MU5203",
                    "depart_time": "09:30",
                    "arrive_time": "12:10",
                },
                "return_flight": {
                    "flight_id": "flight_return_001",
                    "flight_no": "MU5204",
                    "depart_time": "16:00",
                    "arrive_time": "18:40",
                },
            },
            "selected_hotel": {
                "hotel_id": "hotel_001",
                "name": "成都青羊静巷酒店",
                "estimated_total_price": 1160,
            },
            "budget_summary": {
                "total_budget": 4000,
                "known_costs": {
                    "known_subtotal": 2560,
                },
                "remaining_budget": 1440,
            },
            "execution_boundary": {
                "draft_only": True,
                "real_booking_performed": False,
                "real_payment_performed": False,
            },
        },
        "verifier_result": {
            "status": "passed",
            "passed": True,
            "repairable": False,
            "next_action": "decision_gate",
            "approval_required": True,
        },
        "pending_actions": [
            {
                "action_id": action_id,
                "action_type": (
                    "approve_proposal"
                ),
                "description": (
                    "确认当前旅行方案，并保存内部模拟草稿。"
                ),
                "requires_approval": True,
                "status": "pending",
                "payload_summary": {
                    "proposal_id": (
                        proposal_id
                    ),
                    "version": 1,
                    "real_booking_performed": (
                        False
                    ),
                },
            }
        ],
        "itinerary_status": "proposed",
        "decision_attempt_count": 0,
    }


def _demo_commit_draft_node(
    state: TravelState,
) -> dict[str, Any]:
    """Demo：模拟保存内部草稿。"""

    return {
        "itinerary_status": (
            "approved_simulated"
        ),
        "final_response": {
            "status": "approved_simulated",
            "message": (
                "方案已批准；Demo 只模拟保存，"
                "没有真实预订或付款。"
            ),
            "decision": state.get(
                "decision"
            ),
        },
    }


def _demo_revision_analyze_node(
    state: TravelState,
) -> dict[str, Any]:
    """Demo：表示下一步将分析修改文本。"""

    return {
        "final_response": {
            "status": "revision_requested",
            "message": (
                "已接收修改说明，下一步应进入 RevisionAnalyzeNode。"
            ),
            "change_request": state.get(
                "change_request"
            ),
        }
    }


def _demo_cancel_node(
    state: TravelState,
) -> dict[str, Any]:
    """Demo：模拟取消当前方案。"""

    return {
        "itinerary_status": "rejected",
        "final_response": {
            "status": "cancelled",
            "message": "当前旅行方案已取消。",
            "decision": state.get(
                "decision"
            ),
        },
    }


def _demo_final_response_node(
    state: TravelState,
) -> dict[str, Any]:
    """Demo：审批前置条件损坏时返回安全失败。"""

    return {
        "final_response": {
            "status": "failed",
            "message": (
                "DecisionGate 前置状态无效，"
                "未进入人工审批。"
            ),
            "decision_result": state.get(
                "decision_result"
            ),
        }
    }
