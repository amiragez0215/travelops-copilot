# 2026-08-20：阶段五 Unified Trace Harness 与 RAG 评测快照更新

## 比较基线

本文说明相对 Gitee `main` 在提交 `21f29cf`（`eval: publish RAG data alignment and workflow evaluation`）的变更。它不是新的产品承诺；真实预订、支付、实时库存、跨进程持久化 Trace 与 dry-run Replay 仍不在当前范围内。

## 代码与接口

### 1. 轻量 Unified Trace Harness

新增 `app/observability/harness.py` 及其测试。`PlanTripWorkflowRuntime` 在每次 `start_plan()` 或 `resume_decision()` 调用外围创建独立 `RunContext`，并由 Graph 节点包装器形成以下层级：

```text
Run(start / resume)
  -> Node Span
     -> Tool 或 MCP Span
```

Trace 固定记录 Git、LLM、Prompt 名称/版本/模板哈希、Embedding、Reranker、BM25、Chunking、Corpus、Chroma、RRF、Retriever 和 Verifier 的版本快照。每个 Span 只记录输入 SHA-256、状态、稳定错误码、耗时和 token 计数；不会记录 Prompt、Evidence 正文、完整用户画像、原始消息或异常栈。

`TraceCollector` 是容量受限的进程内缓冲区，`EvalHook` 只将已经结束的 Run 变成离线观察记录。二者不写业务数据库、不改变 LangGraph Checkpoint，也不能替代后续阶段的跨进程 Replay。

### 2. 外部工具调用边界

天气、航班、酒店 Node 改为经 `ToolExecutor` 执行外部查询。它负责创建 Tool/MCP 子 Span、遵循现有 MCP 超时配置，并把底层异常归一为 `tool_timeout` 或 `tool_error`。实际的 MCP stdio 初始化、协议协商、生命周期和数据查询仍由原有 Client Factory / Client 实现负责。

### 3. API 关联字段

`POST /api/v1/travel/invoke` 与 `POST /api/v1/travel/decision/{thread_id}` 的同步响应新增可选 `run_id`。同一个跨请求 HITL 会话仍使用稳定的 `thread_id`，但每次 start / resume 都有不同的 `run_id`。普通 API 不返回完整 Trace，避免暴露内部运行摘要。

### 4. 配置与文档

- 新增 `TRACE_MAX_RUNS`（默认 200）和 `TRACE_MAX_SPANS_PER_RUN`（默认 500）。
- 新增显式 `RAG_CORPUS_VERSION=v3-multi-city`，语料、Chunking 或 Chroma collection 改动时必须同步更新并重跑 RAG Eval。
- 更新当前设计、可靠性说明、README，并新增 [Trace Harness 设计](trace_harness_design.md)。
- `backups/` 仅保存本地恢复快照，已加入 Git 忽略规则，不会推送到公开仓库。

## RAG 数据与评测快照

Gold Dataset 仍是 120 条人工标注 Case（96 正例、24 负例）；当前生产语料快照从 607 调整为 592 Chunk。攻略、酒店评价、安全通知、行李清单的文本和部分 Gold 标注随之校准。评测固定使用同一数据集、Metadata Filter、Top-K 和生产检索组件，只改变检索变体。

| 指标（Top-5） | 旧报告：C Hybrid RRF | 当前：C Hybrid RRF | 当前：D Hybrid + Rerank |
| --- | ---: | ---: | ---: |
| Hit@5 | 0.9271 | 0.9583 | 1.0000 |
| Recall@5 | 0.8646 | 0.8158 | 0.8401 |
| MRR@5 | 0.7531 | 0.7823 | 0.8663 |
| nDCG@5 | 0.7560 | 0.7313 | 0.7983 |
| Metadata Accuracy | 1.0000 | 1.0000 | 1.0000 |
| Negative Case Accuracy | 1.0000 | 1.0000 | 1.0000 |
| Mean latency | 24.26 ms | 21.43 ms | 5229.97 ms |
| P95 latency | 38.92 ms | 35.20 ms | 9035.84 ms |

结论应按维度解读：C 的 Hit@5、MRR 和查询延迟改善，但 Recall@5 与 nDCG@5 没有同步提高，候选池 Recall@10 也从 0.9618 降至 0.9234。D 的排序质量最高，却带来秒级 CPU 延迟。因此当前数据不支持“所有检索指标都提升”或“Reranker 应默认开启”的说法。

Workflow Eval 的 32 条确定性轨迹仍保持路径、工具、Verifier、HITL、审批前无副作用、Revision 合并和 Commit 幂等均为 1.0000；当前报告的均值 / P95 延迟为 159.93 / 290.36 ms，旧报告为 70.09 / 151.59 ms。该差异应在同一机器、相同依赖和相同冷启动条件下进一步拆分，不应直接归因于单一组件。

## 验证边界

本次提交包含 Harness 单元/Runtime 集成测试、现有 Workflow Eval 结果和 RAG Eval 结果。全仓库测试应在提交前重新运行：

```powershell
python -m pytest -q
```

评测报告是本地 CPU、对应语料版本和模型版本下的快照，不是线上 SLA 或通用模型能力声明。
