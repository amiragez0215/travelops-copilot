"""阶段五的轻量可观测性 Harness 对外入口。

这里刻意不做一个新的 Agent 平台。模块只为现有 LangGraph Runtime
补齐统一 Run / Span、版本快照、工具包装和离线评测交接点。
"""

from app.observability.harness import (
    CheckpointManager,
    EvalHook,
    PromptDefinition,
    PromptRegistry,
    RunContext,
    ToolExecutionError,
    ToolExecutor,
    TraceCollector,
    VersionMetadata,
    get_current_trace_collector,
    instrument_node,
    use_run_context,
)

__all__ = [
    "CheckpointManager",
    "EvalHook",
    "PromptDefinition",
    "PromptRegistry",
    "RunContext",
    "ToolExecutionError",
    "ToolExecutor",
    "TraceCollector",
    "VersionMetadata",
    "get_current_trace_collector",
    "instrument_node",
    "use_run_context",
]
