# 阶段五：轻量 Unified Trace Harness

## 目标与边界

本阶段把已有的分散 Node Trace 收拢为统一的 `Run / Span` 运行证据，但不引入新的商业 AgentOps 平台，也不改变现有 LangGraph、MCP、业务数据库或审批边界。

- `Run`：一次 `graph.invoke()` 或一次 `Command(resume=...)`。
- `Span`：Run 内的一个 Graph Node、Tool/MCP 调用、检索、LLM 或 DB 操作。
- 同一个 `thread_id` 可以有多个 `run_id`，例如 start、request_changes 后的 resume、approve 后的 resume。
- LangGraph Checkpoint 仍是工作流状态的权威来源；Run Trace 不是 Checkpoint，也不能用于阶段六前的 Replay。

## 运行链路

```text
PlanTripWorkflowRuntime
  -> RunContext（run_id、thread_id、版本快照）
  -> TraceCollector.start_run
  -> Graph Builder 的 instrument_node 包装器
       -> Node Span
       -> ToolExecutor
            -> Tool / MCP 子 Span
  -> TraceCollector.finish_run
  -> EvalHook.capture（离线评测观察记录）
```

`instrument_node` 在没有 `RunContext` 时完全透传原节点。因此既有 Node 单测和直接图测试不会被 Trace 行为干扰；只有 Runtime 调用才创建统一 Span。

## 统一 Schema

每个 Span 至少有以下字段：

```json
{
  "run_id": "run_xxx",
  "thread_id": "trip_session_xxx",
  "span_id": "span_xxx",
  "parent_span_id": "span_parent_or_null",
  "operation_type": "node | llm | retrieval | tool | mcp | db",
  "name": "hybrid_retrieve",
  "model_or_tool": "bm25+chroma+rrf",
  "prompt_name": null,
  "prompt_version": null,
  "retriever_version": "bm25_memory_chroma_rrf_v2",
  "input_hash": "sha256:...",
  "latency_ms": 12.3,
  "token_usage": {"input_tokens": 0, "output_tokens": 0},
  "status": "success",
  "error_code": null
}
```

Node 层仍保留原来的 `TravelState.trace`，因为它用于公开 API 的字段白名单摘要。统一 Harness 不会把完整 Span 注入 State，避免增大 Checkpoint 或误暴露内部信息。

## 可复现版本快照

每个 Run 固定写入：Git commit；LLM provider/model；Prompt name/version/template hash；Embedding/Reranker；BM25 tokenizer 和权重；Chunk size/overlap；Corpus version；Chroma collection；RRF；Retriever version；Verifier rule version。

`RAG_CORPUS_VERSION` 是显式配置。更新语料、Chunking 或 Chroma collection 时必须同步修改它并重跑 RAG Eval，否则不同 Run 的结果不可比较。

## ToolExecutor

天气、航班、酒店 Node 使用同一个 `ToolExecutor`。它统一创建 Tool/MCP 子 Span、执行超时、并将底层异常映射为稳定的 `tool_timeout` 或 `tool_error`。它不替换 MCP Client/Server；MCP 初始化、协议协商、stdio 生命周期和 transport timeout 仍在现有 MCP 实现中。

## 安全与限制

- Trace 只保存 SHA-256 输入哈希，不保存原始消息、Prompt、Evidence 正文、完整用户画像或异常栈。
- `run_id` 可通过启动/审批 API 的响应安全关联一次调用；完整 Trace 目前只由 Runtime 和离线 Eval 使用。
- Collector 是有上限的进程内内存缓冲，不保证重启后可用。
- `EvalHook` 只收集观察记录，不改业务结果、不写数据库、不自动评分。
- 当前求职版本在阶段五收尾；TraceCollector 的内存记录只用于受控调试和离线 Eval，不能被误称为跨进程 Replay 平台。

## 验证命令

```powershell
# 阶段五 Harness 单元与 Runtime 集成测试
python -m pytest tests\test_observability tests\test_workflows -q

# 与阶段四 Workflow Eval 联合回归
python -m pytest tests\test_eval tests\test_observability -q
```
