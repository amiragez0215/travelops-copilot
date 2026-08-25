# PlanTrip Workflow

## 路径

```text
START
 -> input_extract
 -> missing_info_check
    -> clarify -> final_response -> END
    -> safety_check
       -> safe_reject -> final_response -> END
       -> memory_read -> planning_context
          -> weather -> flight_search -> hotel_search
          -> candidate_rank -> budget_optimize
          -> retrieval_plan -> hybrid_retrieve -> rerank -> evidence_grade
             -> retrieval_plan              (证据修复)
             -> proposal -> verifier
                -> proposal                 (方案修复)
                -> retrieval_plan           (证据修复)
                -> budget_optimize          (组合修复)
                -> decision_gate interrupt
                   -> approve -> commit_draft -> final_response -> END
                   -> request_changes -> revision_analyze -> apply_revision
                                      -> missing_info_check
                   -> cancel -> cancel -> final_response -> END
```

## 状态与路由原则

- 缺字段、输入歧义、超出 Mock 能力或无候选，不假装完成方案；进入 Clarify 或安全结束。
- 真实预订、付款等请求不进入工具或提交节点；进入 SafeReject。
- 检索不足只重跑 RAG 链，避免不必要重复调用工具。
- Proposal / Verifier 修复按问题类型回退，不同时修改所有变量。
- `DecisionGate` 没有普通静态出边；只有合法人工输入能通过 `Command.goto` 进入 approve、修改或取消分支。
- `request_changes` 产生新的标准 `trip_request`，再从 MissingInfoCheck 回到主链，而不是将旧 Proposal 直接覆盖。
