from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.agents.state import TravelState
from app.observability.harness import (
    build_default_prompt_registry,
    instrument_node,
)


# ======================================================================
# Public names and type aliases
# ======================================================================

# 普通 Node 的返回值可能是：
#   - dict：普通 State 局部更新；
#   - Command：DecisionGateNode 同时更新 State 并动态路由。
PlanTripNode = Callable[[TravelState], Any]
PlanTripRouter = Callable[[TravelState], str]


# 完整 PlanTripWorkflow 中注册的所有业务 Node。
#
# 使用固定顺序的原因：
#   1. 便于测试 Graph 是否漏注册节点；
#   2. 便于生成架构图和 README；
#   3. Node Override 测试可以检查是否提供了完整替身。
PLAN_TRIP_NODE_NAMES: tuple[str, ...] = (
    "input_extract",
    "missing_info_check",
    "clarify",
    "safety_check",
    "safe_reject",
    "memory_read",
    "planning_context",
    "tool_plan",
    "tool_execute",
    "tool_check",
    "candidate_rank",
    "budget_optimize",
    "retrieval_plan",
    "hybrid_retrieve",
    "rerank",
    "evidence_grade",
    "proposal",
    "verifier",
    "decision_gate",
    "commit_draft",
    "revision_analyze",
    "apply_revision",
    "cancel",
    "final_response",
)


# 需要 Conditional Edge 的源节点。
PLAN_TRIP_ROUTE_NAMES: tuple[str, ...] = (
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
)


# DecisionGateNode 使用 Command.goto 动态路由。
# destinations 只用于 Graph 可视化，不会改变运行行为。
DECISION_GATE_DESTINATIONS: tuple[str, ...] = (
    "commit_draft",
    "revision_analyze",
    "cancel",
    "decision_gate",
    "final_response",
)


# 当前主链大约需要 17 个 super-step；RAG repair、Verifier repair 和
# Revision Loop 会增加额外步骤。100 不是业务重试次数，而是 Graph 运行时
# 的最后一道防无限循环保护。
DEFAULT_PLAN_TRIP_RECURSION_LIMIT = 100


# ======================================================================
# Graph builder
# ======================================================================


def create_plan_trip_builder(
    *,
    node_overrides: Mapping[str, PlanTripNode] | None = None,
    route_overrides: Mapping[str, PlanTripRouter] | None = None,
) -> StateGraph:
    """
    创建尚未 compile 的完整 PlanTrip StateGraph Builder。

    Args:
        node_overrides:
            可选 Node 替身，主要用于 Workflow 端到端测试。

            正式运行：
                不传，加载 app.nodes 中的真实业务节点。

            测试：
                可以把 DeepSeek、MCP、Chroma、SQL 等节点替换成
                确定性的 Fake Node，只测试 Graph 的边和循环。

        route_overrides:
            可选路由函数替身，主要用于 Workflow 测试。

    Returns:
        StateGraph:
            已注册所有 Node、Edge 和 Conditional Edge，
            但尚未绑定 Checkpointer。

    为什么把“创建 Builder”和“编译 Graph”分开？
        - Builder 适合检查节点和边；
        - compile 阶段才安装 Checkpointer；
        - 测试可以单独断言 DecisionGate 没有普通静态出边。
    """

    # 1. 解析真实 Node / 测试替身。
    nodes = _resolve_nodes(node_overrides)
    routes = _resolve_routes(route_overrides)

    # 2. 阶段五只在 Graph 边界添加“观察包装器”，不修改 Node 的输入、
    #    返回值、路由和 State。没有 Runtime RunContext 的直接 Node/Graph
    #    单测会严格透传；Runtime 调用时才产生统一 Node Span。
    prompt_registry = build_default_prompt_registry()
    instrumented_nodes = {
        node_name: instrument_node(
            node_name=node_name,
            node=node,
            prompt_registry=prompt_registry,
        )
        for node_name, node in nodes.items()
    }

    # 3. TravelState 是所有节点共享的结构化状态合同。
    builder = StateGraph(TravelState)

    # ------------------------------------------------------------------
    # 4. 注册所有节点。
    # ------------------------------------------------------------------
    for node_name in PLAN_TRIP_NODE_NAMES:
        if node_name == "decision_gate":
            # DecisionGate 返回 Command.goto，不配置普通出边。
            # destinations 只让 Mermaid / Studio 能画出可能去向。
            builder.add_node(
                node_name,
                instrumented_nodes[node_name],
                destinations=DECISION_GATE_DESTINATIONS,
            )
        else:
            builder.add_node(
                node_name,
                instrumented_nodes[node_name],
            )

    # ==================================================================
    # 4. 输入理解与前置检查
    # ==================================================================

    builder.add_edge(
        START,
        "input_extract",
    )

    builder.add_edge(
        "input_extract",
        "missing_info_check",
    )

    builder.add_conditional_edges(
        "missing_info_check",
        routes["missing_info_check"],
        {
            "clarify": "clarify",
            "safety_check": "safety_check",
        },
    )

    # 缺字段、歧义或数据能力不足时，本轮返回澄清响应并结束。
    builder.add_edge(
        "clarify",
        "final_response",
    )

    builder.add_conditional_edges(
        "safety_check",
        routes["safety_check"],
        {
            "safe_reject": "safe_reject",
            "memory_read": "memory_read",
        },
    )

    # 真实预订、真实付款等越界请求进入安全拒绝响应。
    builder.add_edge(
        "safe_reject",
        "final_response",
    )

    # ==================================================================
    # 5. Context Engineering 与 MCP 数据获取
    # ==================================================================

    builder.add_edge(
        "memory_read",
        "planning_context",
    )

    # 统一工具编排：Policy 强制天气、往返航班、酒店；LLM 只选择活动。
    builder.add_edge(
        "planning_context",
        "tool_plan",
    )
    builder.add_edge(
        "tool_plan",
        "tool_execute",
    )
    builder.add_edge(
        "tool_execute",
        "tool_check",
    )

    builder.add_conditional_edges(
        "tool_check",
        routes["tool_check"],
        {
            "tool_plan": "tool_plan",
            "tool_execute": "tool_execute",
            "candidate_rank": "candidate_rank",
            "final_response": "final_response",
        },
    )

    # ==================================================================
    # 6. 确定性候选排序与预算组合优化
    # ==================================================================

    builder.add_conditional_edges(
        "candidate_rank",
        routes["candidate_rank"],
        {
            "budget_optimize": "budget_optimize",
            "final_response": "final_response",
        },
    )

    builder.add_conditional_edges(
        "budget_optimize",
        routes["budget_optimize"],
        {
            "retrieval_plan": "retrieval_plan",
            "final_response": "final_response",
        },
    )

    # ==================================================================
    # 7. Agentic RAG 主链与 Evidence Repair Loop
    # ==================================================================

    builder.add_edge(
        "retrieval_plan",
        "hybrid_retrieve",
    )
    builder.add_edge(
        "hybrid_retrieve",
        "rerank",
    )
    builder.add_edge(
        "rerank",
        "evidence_grade",
    )

    builder.add_conditional_edges(
        "evidence_grade",
        routes["evidence_grade"],
        {
            # Evidence 不足时只重新规划和执行 RAG，不重查 MCP。
            "retrieval_plan": "retrieval_plan",
            "proposal": "proposal",
            "final_response": "final_response",
        },
    )

    # ==================================================================
    # 8. Controlled Generation 与 Verifier Repair Loop
    # ==================================================================

    builder.add_conditional_edges(
        "proposal",
        routes["proposal"],
        {
            "verifier": "verifier",
            "final_response": "final_response",
        },
    )

    builder.add_conditional_edges(
        "verifier",
        routes["verifier"],
        {
            # LLM 内容或 Proposal 组装错误：重新生成 Proposal。
            "proposal": "proposal",

            # Evidence 状态失效：重新执行 RAG 链。
            "retrieval_plan": "retrieval_plan",

            # 组合或预算状态失效：重新做组合优化，再重新执行 RAG。
            "budget_optimize": "budget_optimize",

            # 完整 Proposal 通过：进入人工审批。
            "decision_gate": "decision_gate",

            # 无法自动修复或重试耗尽：安全结束。
            "final_response": "final_response",
        },
    )

    # ==================================================================
    # 9. Human-in-the-loop 与 Revision Loop
    # ==================================================================

    # DecisionGateNode 通过 Command.goto 动态进入：
    #   approve         -> commit_draft
    #   request_changes -> revision_analyze
    #   cancel          -> cancel
    #   invalid input   -> decision_gate
    #
    # 这里故意不添加任何 builder.add_edge("decision_gate", ...)。

    builder.add_conditional_edges(
        "revision_analyze",
        routes["revision_analyze"],
        {
            "apply_revision": "apply_revision",
            "final_response": "final_response",
        },
    )

    builder.add_conditional_edges(
        "apply_revision",
        routes["apply_revision"],
        {
            # ApplyRevision 已经生成新的标准 trip_request，
            # 所以从 MissingInfoCheck 重新进入主链，而不是再次 InputExtract。
            "missing_info_check": "missing_info_check",
            "final_response": "final_response",
        },
    )

    # ==================================================================
    # 10. 业务收尾与统一外部响应
    # ==================================================================

    builder.add_edge(
        "commit_draft",
        "final_response",
    )

    builder.add_edge(
        "cancel",
        "final_response",
    )

    builder.add_edge(
        "final_response",
        END,
    )

    return builder


def build_plan_trip_graph(
    *,
    checkpointer: Any,
    node_overrides: Mapping[str, PlanTripNode] | None = None,
    route_overrides: Mapping[str, PlanTripRouter] | None = None,
):
    """
    编译完整 PlanTripWorkflow。

    Args:
        checkpointer:
            LangGraph Checkpointer。

            正式运行建议使用项目现有 SQLite SqliteSaver；
            测试使用 InMemorySaver。

            本项目的 DecisionGateNode 使用 interrupt()，因此不允许
            在没有 Checkpointer 的情况下编译正式主图。

    Returns:
        CompiledStateGraph:
            可以 invoke / stream / get_state / get_state_history 的可执行图。

    调用时仍必须提供：
        config = {
            "configurable": {"thread_id": trip_session_id},
            "recursion_limit": 100,
        }
    """

    if checkpointer is None:
        raise ValueError(
            "PlanTripWorkflow 必须提供 checkpointer；"
            "DecisionGateNode 的 interrupt/resume 依赖持久化状态。"
        )

    builder = create_plan_trip_builder(
        node_overrides=node_overrides,
        route_overrides=route_overrides,
    )

    # Checkpointer 在 compile 阶段安装到整个 Graph Runtime，
    # 不是只服务 DecisionGateNode。
    return builder.compile(
        checkpointer=checkpointer
    )


def build_default_plan_trip_graph():
    """
    使用项目应用级 SQLite Checkpointer 构建正式主图。

    该函数采用懒加载，避免仅导入 workflow 模块时就打开数据库连接。
    后续 FastAPI lifespan 可以调用一次并把结果放到 app.state。
    """

    from app.workflows.checkpoint_runtime import (
        initialize_checkpointer,
    )

    return build_plan_trip_graph(
        checkpointer=initialize_checkpointer()
    )


# ======================================================================
# Initial State and Runnable Config helpers
# ======================================================================


def build_initial_plan_trip_state(
    *,
    user_id: str,
    raw_message: str,
    reference_date: str | None = None,
    trip_session_id: str | None = None,
) -> TravelState:
    """
    构造一次新旅行规划的最小初始 State。

    这里只放真正的输入和循环计数器，不提前制造下游节点输出。

    Args:
        user_id:
            当前业务用户 ID。

        raw_message:
            用户的原始旅行需求。

        reference_date:
            相对日期解析基准，格式 YYYY-MM-DD。
            不传时使用当前日期。

        trip_session_id:
            本次旅行规划会话 ID。
            推荐同时作为 LangGraph thread_id。
            不传时自动生成一个新 ID。
    """

    normalized_user_id = str(user_id).strip()
    normalized_message = str(raw_message).strip()

    if not normalized_user_id:
        raise ValueError("user_id 不能为空")

    if not normalized_message:
        raise ValueError("raw_message 不能为空")

    normalized_reference_date = (
        str(reference_date).strip()
        if reference_date is not None
        else date.today().isoformat()
    )

    # 1. 尽早验证日期格式，避免进入 InputExtract 后才报配置错误。
    try:
        date.fromisoformat(
            normalized_reference_date
        )
    except ValueError as exc:
        raise ValueError(
            "reference_date 必须是 YYYY-MM-DD"
        ) from exc

    session_id = (
        str(trip_session_id).strip()
        if trip_session_id is not None
        else f"trip_{uuid4().hex}"
    )

    if not session_id:
        raise ValueError("trip_session_id 不能为空")

    return {
        "user_id": normalized_user_id,
        "trip_session_id": session_id,
        "raw_message": normalized_message,
        "reference_date": normalized_reference_date,
        "workflow": "plan_trip",
        "itinerary_status": "collecting_info",

        # 2. 所有循环都从 0 开始；具体 Node 只在真实 repair 时递增。
        "budget_retry_count": 0,
        "tool_plan_retry_count": 0,
        "tool_execute_retry_count": 0,
        "tool_call_results": [],
        "rag_retry_count": 0,
        "proposal_retry_count": 0,
        "decision_attempt_count": 0,

        # 3. 追加式 Reducer 字段显式初始化为空列表，便于调试和测试。
        "decision_history": [],
        "trace": [],
        "errors": [],
    }


def build_plan_trip_config(
    *,
    thread_id: str,
    recursion_limit: int = DEFAULT_PLAN_TRIP_RECURSION_LIMIT,
) -> dict[str, Any]:
    """
    构造调用 CompiledGraph 时使用的 RunnableConfig。

    thread_id：
        Checkpointer 保存和恢复同一旅行会话的主键。

    recursion_limit：
        单次 invoke/resume 最多允许的 Graph super-step 数量。
        它只是最后一道无限循环保护，不替代各业务节点自己的 retry_count。
    """

    normalized_thread_id = str(
        thread_id
    ).strip()

    if not normalized_thread_id:
        raise ValueError("thread_id 不能为空")

    if recursion_limit < 1:
        raise ValueError(
            "recursion_limit 必须大于 0"
        )

    return {
        "configurable": {
            "thread_id": normalized_thread_id,
        },
        # 注意：recursion_limit 是 Config 顶层字段，不能放进 configurable。
        "recursion_limit": recursion_limit,
    }


# ======================================================================
# Dependency resolution for production and workflow tests
# ======================================================================


def _resolve_nodes(
    overrides: Mapping[str, PlanTripNode] | None,
) -> dict[str, PlanTripNode]:
    """合并真实节点和测试替身，并拒绝拼写错误的 Node 名。"""

    provided = dict(overrides or {})
    unknown = set(provided) - set(
        PLAN_TRIP_NODE_NAMES
    )

    if unknown:
        raise ValueError(
            "node_overrides 包含未知节点："
            + ", ".join(sorted(unknown))
        )

    missing = [
        name
        for name in PLAN_TRIP_NODE_NAMES
        if name not in provided
    ]

    if missing:
        defaults = _load_default_nodes()
        for name in missing:
            provided[name] = defaults[name]

    return provided


def _resolve_routes(
    overrides: Mapping[str, PlanTripRouter] | None,
) -> dict[str, PlanTripRouter]:
    """合并真实路由和测试替身，并拒绝未知路由名。"""
    """如果没有传入路由函数，或者只传入了一部分测试路由函数，_resolve_routes() 就用项目预先定义的真实路由函数补齐缺少的部分，最后返回一套完整的路由函数字典。"""

    provided = dict(overrides or {})
    unknown = set(provided) - set(
        PLAN_TRIP_ROUTE_NAMES
    )

    if unknown:
        raise ValueError(
            "route_overrides 包含未知路由："
            + ", ".join(sorted(unknown))
        )

    missing = [
        name
        for name in PLAN_TRIP_ROUTE_NAMES
        if name not in provided
    ]

    if missing:
        defaults = _load_default_routes()
        for name in missing:
            provided[name] = defaults[name]

    return provided


def _load_default_nodes() -> dict[str, PlanTripNode]:
    """
    延迟导入真实业务节点。

    延迟导入的意义：
        Workflow 结构测试可以注入全部 Fake Node，
        不需要加载 DeepSeek、MCP、Chroma 和本地模型。
    """

    from app.nodes import (
        apply_revision_node,
        budget_optimize_node,
        cancel_node,
        candidate_rank_node,
        clarify_node,
        commit_draft_node,
        decision_gate_node,
        evidence_grade_node,
        final_response_node,
        hybrid_retrieve_node,
        input_extract_node,
        memory_read_node,
        missing_info_check_node,
        planning_context_node,
        tool_check_node,
        tool_execute_node,
        tool_plan_node,
        proposal_node,
        rerank_node,
        retrieval_plan_node,
        revision_analyze_node,
        safe_reject_node,
        safety_check_node,
        verifier_node,
    )

    return {
        "input_extract": input_extract_node,
        "missing_info_check": missing_info_check_node,
        "clarify": clarify_node,
        "safety_check": safety_check_node,
        "safe_reject": safe_reject_node,
        "memory_read": memory_read_node,
        "planning_context": planning_context_node,
        "tool_plan": tool_plan_node,
        "tool_execute": tool_execute_node,
        "tool_check": tool_check_node,
        "candidate_rank": candidate_rank_node,
        "budget_optimize": budget_optimize_node,
        "retrieval_plan": retrieval_plan_node,
        "hybrid_retrieve": hybrid_retrieve_node,
        "rerank": rerank_node,
        "evidence_grade": evidence_grade_node,
        "proposal": proposal_node,
        "verifier": verifier_node,
        "decision_gate": decision_gate_node,
        "commit_draft": commit_draft_node,
        "revision_analyze": revision_analyze_node,
        "apply_revision": apply_revision_node,
        "cancel": cancel_node,
        "final_response": final_response_node,
    }


def _load_default_routes() -> dict[str, PlanTripRouter]:
    """延迟导入真实 Conditional Edge 路由函数。"""

    from app.nodes import (
        route_after_apply_revision,
        route_after_budget_optimize,
        route_after_candidate_rank,
        route_after_evidence,
        route_after_missing_info,
        route_after_proposal,
        route_after_revision_analyze,
        route_after_safety,
        route_after_tool_check,
        route_after_verifier,
    )

    return {
        "missing_info_check": (
            route_after_missing_info
        ),
        "safety_check": route_after_safety,
        "tool_check": route_after_tool_check,
        "candidate_rank": (
            route_after_candidate_rank
        ),
        "budget_optimize": (
            route_after_budget_optimize
        ),
        "evidence_grade": (
            route_after_evidence
        ),
        "proposal": route_after_proposal,
        "verifier": route_after_verifier,
        "revision_analyze": (
            route_after_revision_analyze
        ),
        "apply_revision": (
            route_after_apply_revision
        ),
    }
