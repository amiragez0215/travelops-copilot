# 评测设计

## 已完成：RAG Eval v1

评测直接复用生产检索组件：`InMemoryBM25Index`、`ChromaVectorStore`、`matching_chunk_ids`、RRF 与 `CrossEncoderReranker`。Gold Dataset 位于 `eval/rag_cases.jsonl`，运行前会校验 Gold Chunk 是否仍存在于当前生产语料。

固定变量：Gold Dataset、当前多城市语料（成都、北京、上海、杭州，共 607 个 Chunk）、Chunking、Metadata Filter 和 Top-K。四个 Variant 只改变检索组合；扩充城市或重新切块后应重新生成报告并记录 corpus version：

| Variant | BM25 | Dense | RRF | Reranker |
| --- | --- | --- | --- | --- |
| A | 是 | 否 | 否 | 否 |
| B | 否 | 是 | 否 | 否 |
| C | 是 | 是 | 是 | 否 |
| D | 是 | 是 | 是 | 是 |

输出指标包括 Hit@K、Recall@K、Precision@K、MRR@K、nDCG@K、Metadata Accuracy、Negative Case Accuracy、平均延迟和 P95。完整结果见 `eval/rag_report.md`；Bad Case 是回归资产，不应因修复而删除。

## 未完成：Agent Workflow Eval

下一阶段将用确定性 Case 验证完整轨迹，而不是只检查最终文本。目标覆盖：

- 预期路径和工具调用。
- Verifier 故障注入的检测率与误报。
- 审批前数据库零副作用。
- request_changes 的状态合并与第二次 interrupt。
- cancel 行为。
- Commit 幂等。
- Proposal Citation 相对同次 Evidence Pool 的 Grounded Citation Rate。

在该 Harness 完成前，不能把 RAG Eval 的高分误写成整个 Agent Workflow 已完成端到端评测。
