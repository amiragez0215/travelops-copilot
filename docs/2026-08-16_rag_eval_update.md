# 2026-08-16 RAG Eval 数据与检索评测更新

## 本次变更

本次更新完成了 RAG Eval 测试数据审计和 Query / Gold Chunk 对齐修复。修复目标是让单任务 RAG Eval 的输入 Query、metadata_filter、Gold Chunk 和文档主题对应同一个检索意图，避免用一条单任务 Query 同时评价 Guide 与 Safety 两类 Workflow 子任务。

主要调整包括：

- 修正上海亲子雨天 Guide、上海商务通勤 Guide、杭州慢游 Guide、北京文化场馆 Guide、杭州运河美食 Guide 的 Query 与 Gold Chunk 对齐关系。
- 修正上海亲子雨天行李清单 Query，使其明确包含亲子、雨天、基础物品和人流安全意图。
- 将原本同时包含 Guide 与 Safety Gold 的多意图 Case 改为单一 Guide 检索 Case；Safety 仍由独立 Safety Case 评测。多任务 Workflow 评测应在独立的 RetrievalPlan Eval 中验证，不能用一次单任务检索代替。
- 保留完整旧版 `rag_cases.jsonl` 备份，便于恢复和比较。

## 当前基线结果

当前默认检索方案保持不变：

```text
BM25 + Vector + RRF
fetch_multiplier = 2
不默认启用 Cross-Encoder Reranker
```

新版 120 Case（96 正例、24 负例、607 个 Chunk）结果：

| 指标 | 结果 |
| --- | ---: |
| Hit@5 | 0.9271 |
| Recall@5 | 0.8646 |
| Precision@5 | 0.2271 |
| MRR@5 | 0.7531 |
| nDCG@5 | 0.7560 |
| Metadata Accuracy | 1.0000 |
| Negative Case Accuracy | 1.0000 |

旧版相同 Hybrid RRF 的结果为 Hit@5=0.8438、Recall@5=0.7760、MRR@5=0.6724、nDCG@5=0.6742。该差异来自 Eval Query / Gold 对齐修复，不应表述为检索算法本身的无条件提升。

## 受控实验结论

- `fetch_multiplier=3` 没有整体优于默认 2 倍候选池，因此默认值保持 2。
- BGE Reranker 及 RRF–Reranker 线性融合没有超过纯 RRF，且 CPU 延迟明显更高，因此不进入默认路径。
- Plan / EvidenceGrade 的本地实验性拆分代码已恢复到 Gitee `origin/main` 版本；本次正式提交不包含该实验性改动。

## 验证

```text
RAG Eval 数据集校验：通过
全项目 pytest：310 passed
```

