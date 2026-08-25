# TravelOps-Copilot 当前实现设计

## 系统定位

TravelOps-Copilot 是一个受控的旅行规划 Workflow Agent，而不是开放式旅游问答或真实预订平台。外部旅行事实来自版本化 Mock 数据；RAG 只提供补充知识和可追溯证据；所有方案先经过 Verifier 与人工审批，批准后仅保存内部模拟草稿。

## 数据边界

| 层 | 责任 | 不负责 |
| --- | --- | --- |
| Mock Provider | 天气、航班、酒店候选和基础可用性事实 | 真实库存、真实价格、真实支付或预订。 |
| SQLAlchemy / SQLite | 用户、长期偏好、旅行历史、批准后的 Draft / Version / Approval 审计 | LangGraph 执行快照。 |
| SQLite Checkpointer | Graph 中断、恢复、状态查询和历史快照 | 业务审批记录或可替代的业务数据库。 |
| RAG | 攻略、酒店体验、安全提醒和行李清单的证据 | 航班/酒店候选数值评分或实时运营事实。 |

## 输入与候选决策

输入抽取采用 DeepSeek JSON Structured Output、Pydantic Schema、确定性 Validator 和 Rule-based Fallback 的组合。模型可以解析无歧义旅行表达，但不能编造日期、实时天气、航班价格、房态或预订结果。

候选数据经过以下边界处理：

- Provider 层只过滤城市、日期、座位/房量、非法价格、缺失 ID 等硬数据错误。
- CandidateRank 将用户偏好、价格适配和基础评分转换为确定性排序。
- BudgetOptimize 把总预算视为硬限制；分类预算是组合软目标，不预测真实消费。
- RAG 不参与航班和酒店的数值排序。

## RAG 与修复循环

`RetrievalPlan` 将任务划分为攻略、酒店评价、安全提醒和行李清单，并用 city、doc_type、hotel_id、risk_type 等 Metadata 限定业务边界。检索路径为：

```text
BM25 + Chroma Dense Retrieval -> RRF -> Cross-Encoder Rerank -> EvidenceGrade
```

Evidence 不足时，Graph 只回到 `RetrievalPlan` 重新检索，不重复调用天气、航班和酒店数据。Proposal 或 Verifier 发现证据、预算、引用或约束问题时，按问题归因回到 Proposal、RAG 或预算组合步骤；重试耗尽则安全结束。

## Human-in-the-loop 与提交边界

Verifier 通过后的 Proposal 会在 `DecisionGateNode` 中通过 `interrupt()` 暂停。调用方必须使用同一 `thread_id` 以 `Command(resume=...)` 提交：

- `approve`：进入 `CommitDraftNode`，幂等写入 Draft、Version 与 Approval Record。
- `request_changes`：进入 Revision Analyze / Apply Revision，重新进入主链并触发第二次人工确认。
- `cancel`：执行内部取消收尾，不写入旅行草稿。

提交前没有业务数据库副作用；`decision_event_id` 和数据库唯一约束共同防止重复 approve 产生多版本写入。

## 当前边界与下一步

当前支持：成都单目的地、往返、连续日期、单酒店、Mock 数据、可选 MCP stdio 调用、用户偏好读取、RAG 证据、Verifier、HITL、Revision 和 Commit 幂等。

当前不支持：真实预订支付、多城市行程、多段航班、每日换酒店、多人独立画像、跨进程持久化 Trace / Replay 平台或 Time Travel UI。项目已完成 Agent Workflow Eval 和轻量 Unified Trace Harness：固定 Case 验证路径、工具、Verifier、HITL、Revision、审批前无副作用和 Commit 幂等；每次 invoke/resume 记录独立 Run / Span 与版本快照。当前求职版本在阶段五收尾，下一步优先完成 GitHub CI、可复现运行说明和求职展示。
