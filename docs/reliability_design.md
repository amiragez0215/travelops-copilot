# 可靠性与安全设计

## 审批前无副作用

`DecisionGateNode` 只暂停和校验人工输入，不写业务数据库。`trip_drafts`、`trip_versions` 和 `approval_records` 只能由 `CommitDraftNode` 在合法 approve 后创建。

## Commit 幂等

审批请求由稳定的 `decision_event_id` 标识；提交服务同时使用数据库唯一约束和已有记录检查。重复 resume 或重复 approve 不应创建第二份 Draft、Version 或 Approval Record。

## Checkpoint 与 Thread

每个运行使用稳定的 `trip_session_id` 作为 LangGraph `thread_id`。Runtime 明确区分：

- 不存在 Thread：404，不隐式创建。
- 已完成或未暂停 Thread：409，不允许 resume。
- 等待 Decision 的 Thread：只允许使用同一 ID 恢复。

## 数据与证据边界

- Mock Provider 负责结构化事实；RAG 不覆盖其价格、库存和候选选择。
- Proposal Verifier 检查候选 ID、预算、城市、来源、越权表述和审批边界。
- RAG Eval 的 Metadata Accuracy 和 Negative Case Accuracy 用于验证跨城市、跨酒店、跨 doc_type 的 fail-closed 行为。

## 统一 Trace Harness（阶段五）

每次 `graph.invoke()` 或 `Command(resume=...)` 都会创建独立的 `RunContext`。同一个 `thread_id` 可以有多次 Run，因此能够区分首次规划、修改后的重新规划和最终 approve/cancel。Graph Builder 在不改变 Node 输入输出的前提下为每个节点记录 Span；天气、航班和酒店调用还会记录为其下的 Tool/MCP 子 Span。

每个 Run 固定快照以下版本：Git commit、LLM provider/model、Prompt 名称/版本/模板哈希、Embedding、Reranker、BM25 tokenizer/config、Chunk size/overlap、Corpus/Chroma、RRF、Retriever 及 Verifier 规则版本。Trace 不保存 Prompt、Evidence 正文、完整用户画像或原始输入，只保存输入 SHA-256、状态、耗时、结构化 token 计数和稳定错误码。

`EvalHook` 将已结束的 Run 转成路径、工具、延迟和版本观察记录，供离线 Workflow Eval 读取；它不做评分，也不写业务数据库。进程内 Collector 有容量上限，故它不是商业级 AgentOps 平台。

## 当前限制

当前求职版本在阶段五 Unified Trace Harness 收尾，不实现 dry-run Replay 或 Time Travel。TraceCollector 的内存记录不能被误称为可跨进程复放的平台；若以后转向 Agent Platform / Reliability 岗位，再以独立分支实现 Replay 即可。下一步优先做 GitHub CI、可复现运行说明和项目展示材料。
