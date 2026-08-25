# 架构说明

```text
FastAPI
  ├─ POST /invoke ------------------┐
  ├─ GET /runs/{thread_id}          │
  ├─ GET /runs/{thread_id}/history  │
  └─ POST /decision/{thread_id} ----┤
                                     v
                    PlanTripWorkflowRuntime
                                     |
                         Compiled LangGraph
                         /               \
              SQLite Checkpointer       Business services
                         |                       |
                     thread_id               SQLAlchemy / SQLite
                                                 |
                     ┌───────────────────────────┴─────────────────────────┐
                     v                                                     v
         TravelDataClient Factory                                  Agentic RAG
          local / MCP stdio                                BM25 + Chroma + RRF + Rerank
                     |                                                     |
         Weather / Flight / Hotel Mock                      EvidenceGrade -> Proposal
```

## 核心分层

- **API 层**：只处理输入、错误映射、公开响应和 `thread_id`；不直接修改 TravelState 或数据库。
- **Runtime 层**：创建一次 CompiledGraph，集中处理 start、resume、snapshot 和 history；阻止错误 Thread 或完成 Thread 被错误恢复。
- **Node 层**：每个节点只返回局部 State 更新；路由由 Graph 的静态/条件边或 `Command.goto` 控制。
- **数据客户端层**：业务节点只依赖 `TravelDataClient` Protocol，不关心数据来自 Local Provider 还是 MCP Server。
- **RAG 层**：在生产文档、Chunk、Metadata Filter 和检索组件上执行证据检索。
- **持久化层**：Checkpoint 持久化 Graph 运行；业务数据库持久化用户资料和审批后的内部草稿。

## 为什么需要两个 SQLite 边界

Checkpoint 回答“这个 Workflow 在哪里暂停、下一步该从哪里恢复”；业务数据库回答“用户批准了什么方案、保存了哪个版本”。两者不能互相替代：取消或修改流程会改变 Graph 状态，但只有 approve 能产生业务版本记录。
