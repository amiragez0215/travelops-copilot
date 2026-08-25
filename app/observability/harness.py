from __future__ import annotations

"""阶段五：为现有工作流提供轻量、可复现的 Run / Span Harness。

设计边界：
    - 不替代 LangGraph、MCP 或现有业务 Node；
    - 不把 Prompt、完整 Evidence、用户画像或异常堆栈写入 Trace；
    - Trace 只保存在进程内，适合本地 Demo、离线 Eval 与问题定位；
    - 本模块不执行历史重放或业务 Commit，保持 Trace 与工作流副作用边界清晰。

Run 是一次 ``graph.invoke`` 或 ``Command(resume=...)`` 调用；Span 是这次
调用中的一个节点、工具、检索、模型或数据库操作。这样一次跨 HTTP 的
HITL 会话会拥有同一个 thread_id，但 start / resume 的运行证据不会混淆。
"""

from collections import deque
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from subprocess import DEVNULL, TimeoutExpired, check_output
from threading import RLock
from time import perf_counter
from typing import Any, TypeVar
from uuid import uuid4

from app.common.config import settings


T = TypeVar("T")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ContextVar 让同步 LangGraph 节点不用改变函数签名，也能读取本次 Run。
# 它不像普通全局变量那样会把并发请求的 Span 错误地串到一起。
_CURRENT_RUN: ContextVar[RunContext | None] = ContextVar(
    "travelops_current_run", default=None
)
_CURRENT_COLLECTOR: ContextVar[TraceCollector | None] = ContextVar(
    "travelops_current_collector", default=None
)
_CURRENT_SPAN_ID: ContextVar[str | None] = ContextVar(
    "travelops_current_span_id", default=None
)


@dataclass(frozen=True)
class PromptDefinition:
    """一个已注册 Prompt 的不可变版本记录，不保存 Prompt 正文。"""

    prompt_name: str
    prompt_version: str
    template_hash: str


class PromptRegistry:
    """极简 Prompt 注册表：记录名称、版本和模板哈希，而不做 UI 或远端服务。"""

    def __init__(self) -> None:
        self._definitions: dict[str, PromptDefinition] = {}

    def register(
        self,
        *,
        prompt_name: str,
        prompt_version: str,
        template: str,
    ) -> PromptDefinition:
        """注册 Prompt 的版本指纹；模板只参与哈希，绝不进入 Trace 输出。"""

        name = _required_text(prompt_name, "prompt_name")
        version = _required_text(prompt_version, "prompt_version")
        definition = PromptDefinition(
            prompt_name=name,
            prompt_version=version,
            template_hash=_hash_value(template),
        )
        self._definitions[name] = definition
        return definition

    def get(self, prompt_name: str) -> PromptDefinition | None:
        """按名称读取已注册版本；未知 Prompt 返回 None，方便非 LLM 节点复用包装器。"""

        return self._definitions.get(str(prompt_name))

    def snapshot(self) -> list[dict[str, str]]:
        """输出稳定排序的公开快照，供 Run 版本信息和离线 Eval 使用。"""

        return [
            asdict(self._definitions[name])
            for name in sorted(self._definitions)
        ]


@dataclass(frozen=True)
class VersionMetadata:
    """一次 Run 必须固定记录的实现与数据版本，避免指标变化无法归因。"""

    git_commit: str | None
    llm_provider: str
    llm_model: str
    prompts: tuple[PromptDefinition, ...]
    embedding_model: str
    reranker_model: str
    bm25_tokenizer: str
    bm25_weight: float
    vector_weight: float
    chunk_size: int
    chunk_overlap: int
    corpus_version: str
    chroma_collection: str
    rrf_k: int
    fetch_multiplier: int
    retriever_version: str
    verifier_rule_version: str

    @classmethod
    def from_runtime(cls, registry: PromptRegistry) -> "VersionMetadata":
        """从 Settings 与受控常量创建快照，不能读取未版本化的运行时猜测值。"""

        return cls(
            git_commit=_read_git_commit(),
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            prompts=tuple(
                PromptDefinition(**item)
                for item in registry.snapshot()
            ),
            embedding_model=settings.rag_embedding_model,
            reranker_model=settings.rag_rerank_model,
            # 项目的 BM25 分词实现明确使用 jieba；将它写死在版本快照中，
            # 能避免后续替换 tokenizer 后仍错误比较两次 Eval。
            bm25_tokenizer="jieba.lcut(cut_all=false)",
            bm25_weight=settings.rag_bm25_weight,
            vector_weight=settings.rag_vector_weight,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            corpus_version=settings.rag_corpus_version,
            chroma_collection=settings.rag_chroma_collection,
            rrf_k=settings.rag_rrf_k,
            fetch_multiplier=settings.rag_fetch_multiplier,
            retriever_version="bm25_memory_chroma_rrf_v2",
            verifier_rule_version="deterministic_proposal_verifier_v1",
        )

    def to_dict(self) -> dict[str, Any]:
        """把 dataclass 与嵌套 PromptDefinition 转成 JSON 安全字典。"""

        payload = asdict(self)
        payload["prompts"] = [asdict(item) for item in self.prompts]
        return payload


@dataclass(frozen=True)
class RunContext:
    """一次同步 Graph 调用的身份、边界和不可变版本快照。"""

    run_id: str
    thread_id: str
    workflow: str
    invocation_type: str
    started_at: str
    versions: VersionMetadata

    @classmethod
    def create(
        cls,
        *,
        thread_id: str,
        workflow: str,
        invocation_type: str,
        registry: PromptRegistry,
    ) -> "RunContext":
        """创建 Run；run_id 与 thread_id 分离，才能区分同一会话的多次 resume。"""

        return cls(
            run_id=f"run_{uuid4().hex}",
            thread_id=_required_text(thread_id, "thread_id"),
            workflow=_required_text(workflow, "workflow"),
            invocation_type=_required_text(invocation_type, "invocation_type"),
            started_at=datetime.now(timezone.utc).isoformat(),
            versions=VersionMetadata.from_runtime(registry),
        )


@dataclass
class SpanRecord:
    """统一 Span Schema。只保存可审计摘要和哈希，不保存敏感原始输入。"""

    run_id: str
    thread_id: str
    span_id: str
    parent_span_id: str | None
    operation_type: str
    name: str
    model_or_tool: str | None
    prompt_name: str | None
    prompt_version: str | None
    retriever_version: str | None
    input_hash: str
    latency_ms: float | None
    token_usage: dict[str, int]
    status: str
    error_code: str | None
    started_at: str
    ended_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunRecord:
    """一个 Run 及其有序 Span 集合；它是 EvalHook 的唯一输入。"""

    context: RunContext
    spans: list[SpanRecord] = field(default_factory=list)
    status: str = "running"
    error_code: str | None = None
    ended_at: str | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.context.run_id,
            "thread_id": self.context.thread_id,
            "workflow": self.context.workflow,
            "invocation_type": self.context.invocation_type,
            "started_at": self.context.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "error_code": self.error_code,
            "latency_ms": self.latency_ms,
            "versions": self.context.versions.to_dict(),
            "spans": [span.to_dict() for span in self.spans],
        }


class TraceCollector:
    """进程内、线程安全的 Run / Span 收集器。

    保留数量有上限的原因是 Trace 只服务本地调试和离线评测，不应演变成
    没有生命周期管理的内存日志库。需要持久化 Trace / Replay 时进入阶段六。
    """

    def __init__(self, *, max_runs: int | None = None, max_spans_per_run: int | None = None) -> None:
        self.max_runs = max_runs if max_runs is not None else settings.trace_max_runs
        self.max_spans_per_run = (
            max_spans_per_run
            if max_spans_per_run is not None
            else settings.trace_max_spans_per_run
        )
        if self.max_runs < 1 or self.max_spans_per_run < 1:
            raise ValueError("TraceCollector 的容量必须大于 0")
        self._runs: deque[RunRecord] = deque(maxlen=self.max_runs)
        self._by_id: dict[str, RunRecord] = {}
        self._lock = RLock()

    def start_run(self, context: RunContext) -> RunRecord:
        """登记 Run；run_id 重复属于程序错误，不能静默覆盖旧证据。"""

        with self._lock:
            if context.run_id in self._by_id:
                raise ValueError(f"重复 run_id: {context.run_id}")
            # deque 自动淘汰前先清理索引，避免 get_run 返回已经不可见的 Run。
            if len(self._runs) == self.max_runs:
                expired = self._runs[0]
                self._by_id.pop(expired.context.run_id, None)
            record = RunRecord(context=context)
            # 与 Span 一样，monotonic timer 只存为私有运行时属性；ISO 时间
            # 负责跨系统审计，perf_counter 负责本进程内的精确耗时。
            setattr(record, "_started_perf_counter", perf_counter())
            self._runs.append(record)
            self._by_id[context.run_id] = record
            return record

    def start_span(
        self,
        *,
        context: RunContext,
        operation_type: str,
        name: str,
        input_value: Any,
        parent_span_id: str | None = None,
        model_or_tool: str | None = None,
        prompt: PromptDefinition | None = None,
        retriever_version: str | None = None,
    ) -> SpanRecord:
        """创建进行中的 Span。输入只写 SHA-256，原文永远不离开调用栈。"""

        with self._lock:
            record = self._require_record(context.run_id)
            if len(record.spans) >= self.max_spans_per_run:
                raise RuntimeError(
                    "单个 Run 的 Span 数达到上限；可能存在未受控循环。"
                )
            span = SpanRecord(
                run_id=context.run_id,
                thread_id=context.thread_id,
                span_id=f"span_{uuid4().hex}",
                parent_span_id=parent_span_id,
                operation_type=_required_text(operation_type, "operation_type"),
                name=_required_text(name, "name"),
                model_or_tool=model_or_tool,
                prompt_name=prompt.prompt_name if prompt else None,
                prompt_version=prompt.prompt_version if prompt else None,
                retriever_version=retriever_version,
                input_hash=_hash_value(input_value),
                latency_ms=None,
                token_usage={"input_tokens": 0, "output_tokens": 0},
                status="running",
                error_code=None,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            record.spans.append(span)
            return span

    def finish_span(
        self,
        span: SpanRecord,
        *,
        status: str = "success",
        error_code: str | None = None,
        model_or_tool: str | None = None,
        token_usage: Mapping[str, Any] | None = None,
    ) -> None:
        """完成 Span，并只接收已归一化的状态/计数，避免把异常正文写入 Trace。"""

        if span.status != "running":
            raise RuntimeError(f"Span {span.span_id} 已完成，不能二次结束")
        span.latency_ms = round((perf_counter() - _span_started_perf(span)) * 1000, 3)
        # ``perf_counter`` 不能从 ISO 时间恢复；真实开始时间保存在私有属性，
        # 这一行被下方属性读取以保持公开 schema 简洁。
        span.status = _normalize_status(status)
        span.error_code = error_code
        span.model_or_tool = model_or_tool or span.model_or_tool
        if token_usage:
            span.token_usage = {
                "input_tokens": int(token_usage.get("input_tokens") or 0),
                "output_tokens": int(token_usage.get("output_tokens") or 0),
            }
        span.ended_at = datetime.now(timezone.utc).isoformat()

    def finish_run(self, context: RunContext, *, status: str, error_code: str | None = None) -> RunRecord:
        """将 Run 标记为结束；在评测前必须结束，避免读取半截轨迹。"""

        with self._lock:
            record = self._require_record(context.run_id)
            record.status = _normalize_status(status)
            record.error_code = error_code
            record.ended_at = datetime.now(timezone.utc).isoformat()
            record.latency_ms = round(
                (perf_counter() - float(getattr(record, "_started_perf_counter", perf_counter()))) * 1000,
                3,
            )
            return record

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """返回深拷贝式 JSON 数据，调用方不能修改 Collector 内部对象。"""

        with self._lock:
            record = self._by_id.get(str(run_id))
            return record.to_dict() if record else None

    def list_thread_runs(self, thread_id: str) -> list[dict[str, Any]]:
        """按时间顺序返回同一 HITL 会话的 start / resume Run。"""

        normalized = _required_text(thread_id, "thread_id")
        with self._lock:
            return [
                record.to_dict()
                for record in self._runs
                if record.context.thread_id == normalized
            ]

    def _require_record(self, run_id: str) -> RunRecord:
        record = self._by_id.get(run_id)
        if record is None:
            raise KeyError(f"Trace 中不存在 run_id={run_id}")
        return record


class ToolExecutionError(RuntimeError):
    """ToolExecutor 对调用方暴露的安全异常；code 可进入 Trace，底层异常文本不可。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ToolExecutor:
    """统一工具边界：超时、异常映射和子 Span；它不替代 MCP Client / Server。"""

    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")

    def execute(
        self,
        *,
        name: str,
        operation_type: str,
        input_value: Any,
        callback: Callable[[], T],
        model_or_tool: str | None = None,
    ) -> T:
        """执行一次外部工具调用，并把它作为当前 Node Span 的子 Span。

        超时使用短生命周期线程：底层 Python 调用无法可靠强杀，因此超时后
        立即对主工作流返回 ``tool_timeout``；真正 MCP 调用仍保留其自身的
        transport timeout，形成双层保护。
        """

        context = _CURRENT_RUN.get()
        collector = _CURRENT_COLLECTOR.get()
        parent_span_id = _CURRENT_SPAN_ID.get()
        span = None
        if context and collector:
            span = collector.start_span(
                context=context,
                operation_type=operation_type,
                name=name,
                input_value=input_value,
                parent_span_id=parent_span_id,
                model_or_tool=model_or_tool or name,
            )
            _attach_span_perf_counter(span)
        try:
            result = self._invoke_with_timeout(callback)
        except ToolExecutionError as exc:
            if span and collector:
                collector.finish_span(span, status="failed", error_code=exc.code)
            raise
        except Exception as exc:
            # Trace 中只有稳定错误代码，具体异常由原有 Node 错误处理保留在
            # 内部 State，避免统一观测层意外收集供应商响应或敏感上下文。
            if span and collector:
                collector.finish_span(span, status="failed", error_code="tool_error")
            raise ToolExecutionError("tool_error") from exc
        else:
            if span and collector:
                collector.finish_span(span, status="success")
            return result

    def _invoke_with_timeout(self, callback: Callable[[], T]) -> T:
        if self.timeout_seconds is None:
            return callback()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="travelops-tool")
        future = executor.submit(callback)
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise ToolExecutionError("tool_timeout") from exc
        finally:
            # wait=False 使超时调用不会卡住 Graph；MCP Client 自身也必须负责
            # 关闭进程与连接，不能把资源清理责任转嫁给 Trace Harness。
            executor.shutdown(wait=False, cancel_futures=True)


class CheckpointManager:
    """对 LangGraph Checkpointer 的最小封装，统一 thread config 与安全查询语义。"""

    def __init__(self, *, graph: Any, checkpointer: Any, config_builder: Callable[..., dict[str, Any]]) -> None:
        self.graph = graph
        self.checkpointer = checkpointer
        self._config_builder = config_builder

    def config(self, *, thread_id: str, recursion_limit: int) -> dict[str, Any]:
        """唯一允许构建 Graph config 的入口，防止 start/resume 配置漂移。"""

        return self._config_builder(thread_id=thread_id, recursion_limit=recursion_limit)

    def exists(self, *, thread_id: str, recursion_limit: int) -> bool:
        """优先查询持久化 saver；Fake Graph 时使用 StateSnapshot 回退。"""

        config = self.config(thread_id=thread_id, recursion_limit=recursion_limit)
        if self.checkpointer is not None:
            return self.checkpointer.get_tuple(config) is not None
        return _snapshot_exists(self.graph.get_state(config))

    def get_state(self, *, thread_id: str, recursion_limit: int) -> Any:
        return self.graph.get_state(self.config(thread_id=thread_id, recursion_limit=recursion_limit))

    def get_history(self, *, thread_id: str, recursion_limit: int) -> Iterator[Any]:
        return self.graph.get_state_history(self.config(thread_id=thread_id, recursion_limit=recursion_limit))


class EvalHook:
    """把完成的 Run 转成离线 Eval 可消费的轻量观察记录，不执行任何评分。"""

    def __init__(self, *, max_observations: int | None = None) -> None:
        self.max_observations = max_observations or settings.trace_max_runs
        self._observations: deque[dict[str, Any]] = deque(maxlen=self.max_observations)
        self._lock = RLock()

    def capture(self, run: Mapping[str, Any]) -> dict[str, Any]:
        """保存路径、工具、状态、延迟、版本和完整 Span；输入原文不进入 Eval Hook。"""

        spans = list(run.get("spans") or [])
        observation = {
            "run_id": run.get("run_id"),
            "thread_id": run.get("thread_id"),
            "workflow": run.get("workflow"),
            "invocation_type": run.get("invocation_type"),
            "status": run.get("status"),
            # 路径评测关心所有 Graph Node；LLM / Retrieval / DB 只是 Node 的
            # 更细 operation_type，不能因为分类更具体就从 trajectory 丢失。
            "path": [span.get("name") for span in spans if span.get("operation_type") not in {"tool", "mcp"}],
            "tools": [span.get("model_or_tool") or span.get("name") for span in spans if span.get("operation_type") in {"tool", "mcp"}],
            # 不能把父 Node 与子 Tool Span 直接相加，否则会双重计算同一段
            # 调用时间。Run 的 wall-clock latency 才是一次请求的真实耗时。
            "latency_ms": float(run.get("latency_ms") or 0),
            "versions": run.get("versions") or {},
            "spans": spans,
        }
        with self._lock:
            self._observations.append(observation)
        return dict(observation)

    def list_observations(self, *, thread_id: str | None = None) -> list[dict[str, Any]]:
        """返回只读副本；可按 thread_id 将多次 HITL resume 串给离线 Eval。"""

        with self._lock:
            return [
                dict(item)
                for item in self._observations
                if thread_id is None or item["thread_id"] == thread_id
            ]


@contextmanager
def use_run_context(*, context: RunContext, collector: TraceCollector) -> Iterator[None]:
    """在一次 graph.invoke 边界内安装 Run Context，结束时无条件恢复上一个上下文。"""

    run_token = _CURRENT_RUN.set(context)
    collector_token = _CURRENT_COLLECTOR.set(collector)
    try:
        yield
    finally:
        _CURRENT_COLLECTOR.reset(collector_token)
        _CURRENT_RUN.reset(run_token)


def get_current_trace_collector() -> TraceCollector | None:
    """供 ToolExecutor 和测试读取当前 Collector；没有 Run 时返回 None。"""

    return _CURRENT_COLLECTOR.get()


def instrument_node(*, node_name: str, node: Callable[[Any], T], prompt_registry: PromptRegistry) -> Callable[[Any], T]:
    """把现有节点包装成 Node Span；没有 Run Context 时严格透传，确保单测不变。"""

    def wrapped(state: Any) -> T:
        context = _CURRENT_RUN.get()
        collector = _CURRENT_COLLECTOR.get()
        if context is None or collector is None:
            return node(state)

        prompt = _prompt_for_node(node_name, prompt_registry)
        span = collector.start_span(
            context=context,
            operation_type=_operation_type_for_node(node_name),
            name=node_name,
            input_value=state,
            parent_span_id=_CURRENT_SPAN_ID.get(),
            model_or_tool=_model_or_tool_for_node(node_name),
            prompt=prompt,
            retriever_version=(context.versions.retriever_version if node_name in {"retrieval_plan", "hybrid_retrieve", "rerank", "evidence_grade"} else None),
        )
        _attach_span_perf_counter(span)
        span_token = _CURRENT_SPAN_ID.set(span.span_id)
        try:
            result = node(state)
        except Exception as exc:
            collector.finish_span(span, status="failed", error_code="node_exception")
            raise
        else:
            status, error_code, tool_name, token_usage = _status_from_node_result(result)
            collector.finish_span(
                span,
                status=status,
                error_code=error_code,
                model_or_tool=tool_name,
                token_usage=token_usage,
            )
            return result
        finally:
            _CURRENT_SPAN_ID.reset(span_token)

    # 保留名字与 docstring，测试失败栈和 LangGraph 图检查会更易读。
    wrapped.__name__ = getattr(node, "__name__", f"instrumented_{node_name}")
    wrapped.__doc__ = getattr(node, "__doc__", None)
    return wrapped


def build_default_prompt_registry() -> PromptRegistry:
    """注册当前正式 Proposal Prompt 的实际源代码指纹。

    不从模型请求或 .env 读取 Prompt。这里以函数源码作为模板版本的输入，
    所以任何修改系统/用户 Prompt 拼装逻辑都会改变 template_hash。
    """

    from inspect import getsource
    from app.proposals.proposal_generator import PROPOSAL_PROMPT_VERSION, _build_system_prompt, _build_user_prompt

    registry = PromptRegistry()
    registry.register(
        prompt_name="proposal_generation",
        prompt_version=PROPOSAL_PROMPT_VERSION,
        template=getsource(_build_system_prompt) + getsource(_build_user_prompt),
    )
    return registry


def _status_from_node_result(result: Any) -> tuple[str, str | None, str | None, Mapping[str, Any] | None]:
    """复用节点已经产出的摘要 Trace 判断状态，避免 Harness 猜测业务是否失败。"""

    update = result if isinstance(result, Mapping) else getattr(result, "update", None)
    if not isinstance(update, Mapping):
        return "success", None, None, None
    traces = update.get("trace")
    if isinstance(traces, list) and traces and isinstance(traces[-1], Mapping):
        latest = traces[-1]
        status = str(latest.get("status") or "success")
        tool_name = latest.get("tool_name")
        return status, ("node_failed" if status == "failed" else None), (str(tool_name) if tool_name else None), latest.get("token_usage") if isinstance(latest.get("token_usage"), Mapping) else None
    if update.get("errors"):
        return "degraded", "node_reported_error", None, None
    return "success", None, None, None


def _operation_type_for_node(node_name: str) -> str:
    if node_name in {"hybrid_retrieve", "retrieval_plan", "rerank", "evidence_grade"}:
        return "retrieval"
    if node_name in {"proposal", "input_extract", "revision_analyze"}:
        return "llm"
    if node_name == "commit_draft":
        return "db"
    return "node"


def _model_or_tool_for_node(node_name: str) -> str | None:
    if node_name in {"proposal", "input_extract", "revision_analyze"}:
        return settings.llm_model
    if node_name == "hybrid_retrieve":
        return "bm25+chroma+rrf"
    if node_name == "rerank":
        return settings.rag_rerank_model
    return None


def _prompt_for_node(node_name: str, registry: PromptRegistry) -> PromptDefinition | None:
    return registry.get("proposal_generation") if node_name == "proposal" else None


def _hash_value(value: Any) -> str:
    """稳定计算任意输入的哈希；不把原文序列化到 Trace 或异常信息中。"""

    try:
        import json
        normalized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        normalized = repr(type(value))
    return "sha256:" + sha256(normalized.encode("utf-8")).hexdigest()


def _read_git_commit() -> str | None:
    """尽力读取当前代码提交；打包源码或非 Git 环境下明确返回 None 而非伪造版本。"""

    try:
        value = check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=DEVNULL, timeout=1, text=True)
    except (OSError, TimeoutExpired, Exception):
        return None
    commit = value.strip()
    return commit or None


def _required_text(value: Any, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    return normalized


def _normalize_status(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"running", "success", "degraded", "failed", "cancelled"}:
        return normalized
    # DecisionGate 的 interrupt 是设计内暂停，而不是节点错误。原有 Node
    # Trace 的 status 可能使用 waiting / interrupted 等业务词，统一映射为
    # 成功完成“暂停并交给人工”的节点职责。
    if normalized in {"waiting_for_decision", "waiting", "interrupted", "pending"}:
        return "success"
    return "failed"


# Span 的公开 schema 不包含 monotonic timestamp（它只能在本进程比较）。
# 将其挂在私有动态属性上，既能精确计算 latency，又不会污染 JSON 输出。
def _attach_span_perf_counter(span: SpanRecord) -> None:
    setattr(span, "_started_perf_counter", perf_counter())


def _span_started_perf(span: SpanRecord) -> float:
    return float(getattr(span, "_started_perf_counter", perf_counter()))


def _snapshot_exists(snapshot: Any) -> bool:
    """避免 Harness 反向导入 workflows 包造成循环依赖的最小 Snapshot 存在性检查。"""

    if snapshot is None:
        return False
    config = getattr(snapshot, "config", None)
    if isinstance(config, Mapping):
        configurable = config.get("configurable")
        if isinstance(configurable, Mapping) and configurable.get("checkpoint_id"):
            return True
    return bool(
        getattr(snapshot, "created_at", None)
        or getattr(snapshot, "metadata", None)
        or getattr(snapshot, "values", None)
        or getattr(snapshot, "tasks", None)
        or getattr(snapshot, "next", None)
    )
