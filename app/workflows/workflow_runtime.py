from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any

from app.common.config import settings
from app.observability.harness import (
    CheckpointManager,
    EvalHook,
    RunContext,
    TraceCollector,
    build_default_prompt_registry,
    use_run_context,
)
from app.workflows.checkpoint_runtime import (
    initialize_checkpointer,
    reset_checkpointer_runtime,
)

# checkpoint_runtime 会在导入 SqliteSaver 前设置严格序列化环境变量。
# 在它之后再导入 LangGraph Command，可以让运行时初始化顺序更清晰。
from langgraph.types import Command
from app.workflows.plan_trip_workflow import (
    build_initial_plan_trip_state,
    build_plan_trip_config,
    build_plan_trip_graph,
)
from app.workflows.snapshot_utils import snapshot_is_interrupted


class WorkflowRuntimeError(RuntimeError):
    """
    PlanTrip Graph 运行时异常的统一父类。

    API 层根据具体子类映射为：
        - 404：Thread 不存在；
        - 409：Thread 状态冲突；
        - 500：真正运行时异常。
    """


class WorkflowThreadNotFoundError(WorkflowRuntimeError):
    """指定 thread_id 在 Checkpointer 中不存在。"""


class WorkflowThreadAlreadyExistsError(WorkflowRuntimeError):
    """启动新流程时，客户端提交了已经存在的 thread_id。"""


class WorkflowNotWaitingForDecisionError(WorkflowRuntimeError):
    """指定 Thread 当前没有停在 interrupt，不能提交 Decision Resume。"""


@dataclass(frozen=True)
class WorkflowExecution:
    """
    一次 Graph invoke / resume 的标准运行结果。

    output：
        graph.invoke() 本次直接返回的值。
        如果命中 interrupt，里面通常包含 __interrupt__。

    snapshot：
        本次调用结束后从 Checkpointer 重新读取的权威 StateSnapshot。
        API 构造公开响应时优先读取 snapshot.values。

    run_id / trace_run：
        阶段五新增的统一观测结果。run_id 标识这一次 invoke 或 resume，
        而非整个 HITL 会话；同一 thread 的每次 resume 都有独立版本快照。
        Trace 仅包含哈希和摘要，绝不包含原始用户输入或 Prompt。
    """

    thread_id: str
    output: Any
    snapshot: Any
    run_id: str | None = None
    trace_run: dict[str, Any] | None = None


class PlanTripWorkflowRuntime:
    """
    完整 PlanTrip CompiledGraph 的应用级运行时封装。

    这个类不实现旅行规划业务，只负责五件事：
        1. 启动一条新的 Graph Thread；
        2. 使用 Command(resume=...) 恢复 Human-in-the-loop；
        3. 根据 thread_id 查询当前 StateSnapshot；
        4. 查询 Checkpoint 历史；
        5. 防止错误 thread_id 或错误状态偷偷创建新流程。

    为什么需要 Runtime，而不是让每个 FastAPI Route 直接调用 graph.invoke？
        - thread 是否存在的判断可以集中处理；
        - interrupt 前置状态检查可以集中处理；
        - config / recursion_limit 不会在多个接口中重复；
        - 测试可以注入 Fake Graph；
        - 后续 Trace、Replay、并发控制可以在同一层扩展。
    """

    def __init__(
        self,
        *,
        graph: Any,
        checkpointer: Any | None = None,
        recursion_limit: int | None = None,
        trace_collector: TraceCollector | None = None,
        eval_hook: EvalHook | None = None,
    ) -> None:
        if graph is None:
            raise ValueError("graph 不能为空")

        self.graph = graph
        self.checkpointer = checkpointer
        self.recursion_limit = (
            int(recursion_limit)
            if recursion_limit is not None
            else settings.workflow_recursion_limit
        )

        if self.recursion_limit < 1:
            raise ValueError(
                "workflow recursion_limit 必须大于 0"
            )

        # ------------------------------------------------------------------
        # Stage 5: 统一 Harness 的应用边界。
        #
        # Graph 仍负责状态机和 Checkpoint；TraceCollector 只记录运行证据；
        # EvalHook 只接收已结束的 Trace，不评分、不写业务数据库。
        # 这样新增可观测性不会改变审批、Commit 或路由行为。
        # ------------------------------------------------------------------
        self.prompt_registry = build_default_prompt_registry()
        self.trace_collector = trace_collector or TraceCollector()
        self.eval_hook = eval_hook or EvalHook()
        self.checkpoint_manager = CheckpointManager(
            graph=graph,
            checkpointer=checkpointer,
            config_builder=build_plan_trip_config,
        )

        # 当前 v1 使用同步 Graph + SQLite Saver。
        # 单进程 FastAPI 中用一把进程内锁串行化 invoke/resume，
        # 避免同一个 Thread 被两个 HTTP 请求同时恢复。
        #
        # CommitDraft 仍然保留数据库幂等保护，
        # 因为多进程部署时仅靠进程锁并不够。
        self._execution_lock = RLock()

    def start_plan(
        self,
        *,
        user_id: str,
        raw_message: str,
        reference_date: str | None = None,
        trip_session_id: str | None = None,
    ) -> WorkflowExecution:
        """
        启动一条新的 PlanTrip Graph Thread。

        核心流程：
            1. build_initial_plan_trip_state() 生成最小 State；
            2. trip_session_id 同时作为 LangGraph thread_id；
            3. 在 Checkpointer 中确认该 Thread 尚不存在；
            4. graph.invoke(initial_state, config)；
            5. 从 Checkpointer 重新读取最新 Snapshot。

        如果客户端重复使用已有 trip_session_id，返回冲突而不是：
            - 覆盖旧状态；
            - 从旧 Checkpoint 继续；
            - 创建行为不明确的新执行。
        """

        # 1. Initial State Helper 会统一校验 user_id、message 和 reference_date。
        initial_state = build_initial_plan_trip_state(
            user_id=user_id,
            raw_message=raw_message,
            reference_date=reference_date,
            trip_session_id=trip_session_id,
        )

        thread_id = initial_state[
            "trip_session_id"
        ]
        config = self._config(thread_id)

        with self._execution_lock:
            # 2. 显式拒绝已有 Thread，避免 start API 被当作 resume API 使用。
            if self._thread_exists_unlocked(
                thread_id
            ):
                raise WorkflowThreadAlreadyExistsError(
                    "trip_session_id 已存在，"
                    "请查询原流程或生成新的会话 ID。"
                )

            # 3. RunContext 包住“整次 Graph 调用”，而不是单个 HTTP 请求。
            #    Graph Builder 的 Node 包装器会从 ContextVar 读取它，形成：
            #        Run(start) -> Node Span -> Tool/MCP Span。
            #    这也让同一 thread 的后续 resume 获得独立 run_id 和版本快照。
            execution = self._invoke_with_harness(
                thread_id=thread_id,
                invocation_type="start",
                callback=lambda: self.graph.invoke(initial_state, config=config),
            )

        return execution

    def resume_decision(
        self,
        *,
        thread_id: str,
        decision_payload: Mapping[str, Any],
    ) -> WorkflowExecution:
        """
        使用人工决定恢复停在 DecisionGateNode 的 Graph。

        这个方法只允许恢复“当前仍有 interrupt”的 Thread。

        为什么要先检查 Thread 和 interrupt？
            如果直接对错误 thread_id 调用：
                graph.invoke(Command(resume=...), config)

            某些运行时版本可能返回难以理解的空线程结果。
            API 必须显式保证：
                - 404 的 Thread 不会偷偷创建；
                - 已完成 Thread 不会被重复 Resume；
                - 只有等待人工决定的 Thread 才能接受 Decision。
        """

        normalized_thread_id = self._normalize_thread_id(
            thread_id
        )
        config = self._config(
            normalized_thread_id
        )

        if not isinstance(
            decision_payload,
            Mapping,
        ):
            raise TypeError(
                "decision_payload 必须是 Mapping"
            )

        with self._execution_lock:
            # 1. 从持久化 Checkpoint 读取当前状态。
            snapshot_before = (
                self._get_existing_snapshot_unlocked(
                    normalized_thread_id
                )
            )

            # 2. 只有 DecisionGate interrupt 仍存在时才允许 resume。
            if not snapshot_is_interrupted(
                snapshot_before
            ):
                raise WorkflowNotWaitingForDecisionError(
                    "当前工作流没有等待人工决定，"
                    "可能已经完成、取消或仍处于其他状态。"
                )

            # 3. 这是 Human-in-the-loop 恢复的核心代码。
            #    Command.resume 中的 dict 会成为 DecisionGateNode 中：
            #        raw_response = interrupt(payload)
            #    的返回值。
            # 4. request_changes 可能在同一次 resume 内重新跑完整主链，
            #    并到达第二次 interrupt；approve/cancel 则通常直接 END。
            execution = self._invoke_with_harness(
                thread_id=normalized_thread_id,
                invocation_type="resume_decision",
                callback=lambda: self.graph.invoke(
                    Command(resume=dict(decision_payload)),
                    config=config,
                ),
            )

        return execution

    def get_snapshot(
        self,
        thread_id: str,
    ) -> Any:
        """
        获取指定 Thread 的最新 StateSnapshot。

        不存在时抛出 WorkflowThreadNotFoundError，
        由 FastAPI 转换成 404。
        """

        normalized_thread_id = self._normalize_thread_id(
            thread_id
        )

        with self._execution_lock:
            return self._get_existing_snapshot_unlocked(
                normalized_thread_id
            )

    def get_history(
        self,
        thread_id: str,
        *,
        limit: int | None = None,
    ) -> list[Any]:
        """
        返回指定 Thread 的 Checkpoint 历史，默认最新在前。

        Args:
            limit:
                只截取前 N 个 Snapshot。
                API 会对上限做限制，避免一次返回过多历史。
        """

        normalized_thread_id = self._normalize_thread_id(
            thread_id
        )
        with self._execution_lock:
            # 1. 先确认 Thread 存在，保证错误 ID 返回 404。
            self._get_existing_snapshot_unlocked(
                normalized_thread_id
            )

            # 2. get_state_history() 返回可迭代 Snapshot，
            #    LangGraph 官方约定通常按最新到最旧排序。
            iterator = self.checkpoint_manager.get_history(
                thread_id=normalized_thread_id,
                recursion_limit=self.recursion_limit,
            )

            history: list[Any] = []

            for snapshot in iterator:
                history.append(snapshot)

                if (
                    limit is not None
                    and len(history) >= limit
                ):
                    break

        return history

    def thread_exists(
        self,
        thread_id: str,
    ) -> bool:
        """公开检查某个 thread_id 是否已有持久化 Checkpoint。"""

        normalized_thread_id = self._normalize_thread_id(
            thread_id
        )

        with self._execution_lock:
            return self._thread_exists_unlocked(
                normalized_thread_id
            )

    def _get_existing_snapshot_unlocked(
        self,
        thread_id: str,
    ) -> Any:
        """读取已存在 Snapshot；调用方必须已经持有执行锁。"""

        if not self._thread_exists_unlocked(
            thread_id
        ):
            raise WorkflowThreadNotFoundError(
                f"不存在 thread_id={thread_id} 的旅行规划流程。"
            )

        return self.checkpoint_manager.get_state(
            thread_id=thread_id,
            recursion_limit=self.recursion_limit,
        )

    def _thread_exists_unlocked(
        self,
        thread_id: str,
    ) -> bool:
        """
        在已经持有执行锁时判断 Thread 是否存在。

        优先使用 Checkpointer.get_tuple()：
            它直接按 thread_id 查询最新 Checkpoint，语义最明确。

        测试注入 Fake Graph、没有 Checkpointer 时：
            回退到 graph.get_state() + snapshot_exists()。
        """

        return self.checkpoint_manager.exists(
            thread_id=thread_id,
            recursion_limit=self.recursion_limit,
        )

    def _config(
        self,
        thread_id: str,
    ) -> dict[str, Any]:
        """统一构造 Graph RunnableConfig。"""

        return self.checkpoint_manager.config(
            thread_id=thread_id,
            recursion_limit=self.recursion_limit,
        )

    def get_trace_run(self, run_id: str) -> dict[str, Any] | None:
        """读取一次已收集的统一 Run；供离线 Eval 或受控调试使用。"""

        return self.trace_collector.get_run(run_id)

    def get_trace_runs_for_thread(self, thread_id: str) -> list[dict[str, Any]]:
        """读取同一 HITL 会话所有 start/resume Run，不泄漏原始业务 State。"""

        normalized_thread_id = self._normalize_thread_id(thread_id)
        return self.trace_collector.list_thread_runs(normalized_thread_id)

    def get_eval_observations(self, thread_id: str | None = None) -> list[dict[str, Any]]:
        """让阶段四 Workflow Eval 或后续离线脚本消费 Run 观察记录。"""

        return self.eval_hook.list_observations(thread_id=thread_id)

    def _invoke_with_harness(
        self,
        *,
        thread_id: str,
        invocation_type: str,
        callback: Any,
    ) -> WorkflowExecution:
        """执行一次 Graph 调用并收尾 Trace；业务异常仍原样抛回 API / 调用者。

        重要的是顺序：先开始 Run，再安装 ContextVar，再调用 Graph；Graph 返回后
        从 Checkpoint 读取权威 Snapshot，最后才关闭 Run 并交给 EvalHook。这样
        Eval 永远不会拿到“节点仍在 running”的半截轨迹。
        """

        context = RunContext.create(
            thread_id=thread_id,
            workflow="plan_trip",
            invocation_type=invocation_type,
            registry=self.prompt_registry,
        )
        self.trace_collector.start_run(context)
        try:
            with use_run_context(context=context, collector=self.trace_collector):
                output = callback()
                snapshot = self.checkpoint_manager.get_state(
                    thread_id=thread_id,
                    recursion_limit=self.recursion_limit,
                )
        except Exception:
            run = self.trace_collector.finish_run(
                context,
                status="failed",
                error_code="graph_invoke_error",
            ).to_dict()
            self.eval_hook.capture(run)
            raise

        # interrupt 不是失败，而是已被设计的同步调用终点；此处统一标 success。
        run = self.trace_collector.finish_run(context, status="success").to_dict()
        self.eval_hook.capture(run)
        return WorkflowExecution(
            thread_id=thread_id,
            output=output,
            snapshot=snapshot,
            run_id=context.run_id,
            trace_run=run,
        )

    @staticmethod
    def _normalize_thread_id(
        thread_id: str,
    ) -> str:
        """拒绝空 thread_id，避免 Checkpointer 读取到不明确的线程。"""

        normalized = str(thread_id).strip()

        if not normalized:
            raise ValueError(
                "thread_id 不能为空"
            )

        return normalized


# ======================================================================
# Application-level singleton runtime
# ======================================================================

_WORKFLOW_RUNTIME: PlanTripWorkflowRuntime | None = None
_RUNTIME_LOCK = RLock()



def initialize_workflow_runtime(
    *,
    force_reload: bool = False,
) -> PlanTripWorkflowRuntime:
    """
    初始化应用级 PlanTripWorkflowRuntime 单例。

    正常 FastAPI 生命周期：
        startup：调用一次；
        request：所有接口复用同一个 CompiledGraph；
        shutdown：reset_workflow_runtime() 关闭 SQLite Checkpointer 连接。
    """

    global _WORKFLOW_RUNTIME

    with _RUNTIME_LOCK:
        if (
            _WORKFLOW_RUNTIME is not None
            and not force_reload
        ):
            return _WORKFLOW_RUNTIME

        # 1. 强制重载时，先关闭旧 Checkpointer 连接。
        if force_reload:
            _WORKFLOW_RUNTIME = None
            reset_checkpointer_runtime()

        # 2. 创建或复用应用级 SQLite Checkpointer。
        checkpointer = initialize_checkpointer(
            force_reload=False
        )

        # 3. 编译一次完整 PlanTrip StateGraph。
        #    这是 Runtime 初始化的核心代码；后续请求不再重复 build/compile。
        graph = build_plan_trip_graph(
            checkpointer=checkpointer
        )

        _WORKFLOW_RUNTIME = (
            PlanTripWorkflowRuntime(
                graph=graph,
                checkpointer=checkpointer,
                recursion_limit=(
                    settings.workflow_recursion_limit
                ),
            )
        )

        return _WORKFLOW_RUNTIME



def get_workflow_runtime() -> PlanTripWorkflowRuntime:
    """获取应用级 Runtime；未预加载时执行一次懒初始化。"""

    return initialize_workflow_runtime(
        force_reload=False
    )



def reset_workflow_runtime() -> None:
    """
    清除 CompiledGraph 引用并关闭 SQLite Checkpointer 连接。

    注意：
        这里只关闭连接，不删除磁盘中的 checkpoint 数据。
        服务重启后仍然可以用相同 thread_id 恢复中断流程。
    """

    global _WORKFLOW_RUNTIME

    with _RUNTIME_LOCK:
        _WORKFLOW_RUNTIME = None
        reset_checkpointer_runtime()
