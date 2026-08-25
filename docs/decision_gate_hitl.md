# DecisionGateNode 与 Human-in-the-loop

## 最小闭环

```text
已通过 Verifier 的 State
↓
DecisionGateNode
↓
interrupt(payload)
↓
Checkpointer 按 thread_id 保存 Graph State
↓
API / 前端收到 Interrupt Payload
↓
用户 approve / request_changes / cancel
↓
Command(resume=decision_payload)
↓
使用同一个 thread_id 重新调用 Graph
↓
DecisionGateNode 从开头重新执行
↓
interrupt() 返回人工输入
↓
Pydantic + stale proposal 检查
↓
Command(update=..., goto=...)
```

## 四个关键对象

### interrupt

在 Node 中动态暂停 Graph，把 JSON Payload 暴露给调用方。

### Checkpointer

保存每个 Graph super-step 的 State 快照。Human-in-the-loop 必须配置 Checkpointer。

### thread_id

Checkpoint 的持久指针。恢复时必须使用同一个 thread_id；新 thread_id 表示新工作流。

### Command

两种使用方式：

```python
# 调用方恢复 Graph
Command(resume={...})

# Node 更新 State 并动态路由
Command(update={...}, goto="commit_draft")
```

## Interrupt 规则

1. 不要用 broad try/except 包裹 interrupt。
2. interrupt 前不要执行非幂等副作用。
3. 恢复后 Node 会从开头重新执行。
4. Payload 必须是 JSON 可序列化类型。
5. 人工输入错误时，不要在一个 Node 内 while True 重复 interrupt；应更新错误 State 并回环到 Node。

## 当前项目的三条人工分支

```text
approve
→ CommitDraftNode

request_changes
→ RevisionAnalyzeNode

cancel
→ CancelNode
```

DecisionGateNode 本身不写数据库、不真实预订、不真实付款。
