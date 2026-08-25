from __future__ import annotations

import json
from time import sleep

import pytest

from app.observability.harness import (
    PromptRegistry,
    RunContext,
    ToolExecutionError,
    ToolExecutor,
    TraceCollector,
    instrument_node,
    use_run_context,
)
from app.workflows.snapshot_utils import get_interrupt_values
from tests.fakes.plan_trip_workflow import build_fake_runtime


def _context_and_collector():
    """创建不依赖真实 LLM/MCP 的最小 Run，专门测试 Harness 自身合同。"""

    registry = PromptRegistry()
    registry.register(
        prompt_name="test_prompt",
        prompt_version="v1",
        template="这只是测试模板，不应写入 trace。",
    )
    collector = TraceCollector(max_runs=5, max_spans_per_run=20)
    context = RunContext.create(
        thread_id="trace-unit-thread",
        workflow="plan_trip",
        invocation_type="start",
        registry=registry,
    )
    collector.start_run(context)
    return context, collector, registry


def test_instrumented_node_and_tool_form_parent_child_spans_without_raw_input():
    """阶段五最小闭环：一个 Node Span 下面必须挂上一个 Tool Span。"""

    context, collector, registry = _context_and_collector()

    def weather_like_node(state):
        # 此处故意把敏感文本放在工具输入里，断言证明 Trace 只有哈希。
        result = ToolExecutor(timeout_seconds=1).execute(
            name="get_weather",
            operation_type="tool",
            input_value={"private_message": state["private_message"]},
            callback=lambda: {"status": "ok"},
        )
        return {"weather_result": result}

    wrapped = instrument_node(
        node_name="weather",
        node=weather_like_node,
        prompt_registry=registry,
    )

    with use_run_context(context=context, collector=collector):
        assert wrapped({"private_message": "绝不能出现在 Trace 的原文"})["weather_result"]["status"] == "ok"

    run = collector.finish_run(context, status="success").to_dict()
    node_span, tool_span = run["spans"]
    assert node_span["operation_type"] == "node"
    assert tool_span["operation_type"] == "tool"
    assert tool_span["parent_span_id"] == node_span["span_id"]
    assert node_span["input_hash"].startswith("sha256:")
    assert tool_span["input_hash"].startswith("sha256:")
    assert "绝不能出现在" not in json.dumps(run, ensure_ascii=False)
    assert run["versions"]["corpus_version"] == "v5-agentic-challenge"
    assert run["versions"]["prompts"][0]["template_hash"].startswith("sha256:")


def test_tool_executor_maps_timeout_and_error_to_stable_codes():
    """超时和底层异常不能把不稳定的供应商异常文本变成可比较的 Trace 分类。"""

    timeout_executor = ToolExecutor(timeout_seconds=0.01)
    with pytest.raises(ToolExecutionError, match="tool_timeout"):
        timeout_executor.execute(
            name="slow_tool",
            operation_type="tool",
            input_value={"value": 1},
            callback=lambda: (sleep(0.05), "late")[1],
        )

    error_executor = ToolExecutor(timeout_seconds=1)
    with pytest.raises(ToolExecutionError, match="tool_error"):
        error_executor.execute(
            name="broken_tool",
            operation_type="tool",
            input_value={"value": 2},
            callback=lambda: (_ for _ in ()).throw(ValueError("provider secret detail")),
        )


def test_runtime_emits_separate_start_and_resume_runs_with_eval_ready_path():
    """同一 thread 的 start/resume 必须使用不同 run_id，且 EvalHook 能读取完整节点路径。"""

    runtime = build_fake_runtime()
    started = runtime.start_plan(
        user_id="user_001",
        raw_message="杭州到成都三日游；这段原文不得进入统一 Trace。",
        trip_session_id="trace-runtime-001",
    )

    assert started.run_id
    assert started.trace_run is not None
    assert started.trace_run["status"] == "success"
    assert started.trace_run["versions"]["retriever_version"] == "bm25_memory_chroma_rrf_v2"
    assert {span["name"] for span in started.trace_run["spans"]} >= {
        "input_extract",
        "proposal",
        "decision_gate",
    }
    assert "这段原文不得进入" not in json.dumps(started.trace_run, ensure_ascii=False)

    interrupt = get_interrupt_values(
        graph_output=started.output,
        snapshot=started.snapshot,
    )[0]
    resumed = runtime.resume_decision(
        thread_id="trace-runtime-001",
        decision_payload={
            "decision": "cancel",
            "proposal_id": interrupt["proposal_id"],
            "action_id": interrupt["action_id"],
        },
    )

    assert resumed.run_id and resumed.run_id != started.run_id
    runs = runtime.get_trace_runs_for_thread("trace-runtime-001")
    assert [run["invocation_type"] for run in runs] == ["start", "resume_decision"]
    observations = runtime.get_eval_observations("trace-runtime-001")
    assert observations[0]["path"][:3] == [
        "input_extract",
        "missing_info_check",
        "safety_check",
    ]
    assert observations[1]["path"] == ["decision_gate", "cancel", "final_response"]
