# Bad Case Registry

当前已验证的 RAG Bad Case 位于 `eval/rag_bad_cases.md`，包括：

- 商务短住意图与“适合人群”章节词汇重叠不足，BM25 / RRF 在 Top-5 漏掉 Gold，而 Cross-Encoder 可恢复。
- Dense Retrieval 对“住客不满意的地方”和“可能的不足”映射不足。
- 多意图的长者低步行需求被 Dense 过度压缩为宽泛的三日游相似性。

处理原则：保留 Case、记录检索结果和版本、提出单一可验证假设、只改变一个变量、重新运行完整消融，并检查目标 Case 改善是否伤害总体指标。不要把 Bad Case 删除或用单次手工演示替代回归验证。
