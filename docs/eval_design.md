# 评测设计

## 已完成：RAG Eval v1

评测直接复用生产检索组件：`InMemoryBM25Index`、`ChromaVectorStore`、`matching_chunk_ids`、RRF 与 `CrossEncoderReranker`。运行前会校验 Gold Chunk 是否仍存在于当前生产语料。

评测集不再承担单一职责：

- `eval/rag_cases.jsonl` 是 **Scope Regression Set**。它检查指定领域、城市和 metadata 下的证据是否可检索；安全查询使用多 Gold，避免相同 query/filter 被错误拆成彼此冲突的单 Gold case。
- `eval/rag_challenge_cases.jsonl` 是 **Semantic Holdout Set**。它用于比较检索策略在真实自然语言、多偏好、隐式表达和语义干扰下的表现，不能用 Scope Set 的标题匹配高分替代其结论。

在同一 suite 内固定 Gold Dataset、当前多城市语料、Chunking、Metadata Filter、Top-K 和 `embedding_text_version`。四个 Variant 只改变检索组合；扩充语料、重新切块或改变 Vector Embedding 输入格式后应重新生成报告并记录版本：

| Variant | BM25 | Dense | RRF | Reranker |
| --- | --- | --- | --- | --- |
| A | 是 | 否 | 否 | 否 |
| B | 否 | 是 | 否 | 否 |
| C | 是 | 是 | 是 | 否 |
| D | 是 | 是 | 是 | 是 |

输出指标包括 Hit@K、Recall@K、Precision@K、MRR@K、nDCG@K、Metadata Accuracy、Negative Case Accuracy、平均延迟和 P95。Scope 与 Semantic Holdout 应分别报告；Bad Case 是回归资产，不应因修复而删除。

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
