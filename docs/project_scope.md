# 项目范围与非目标

## 已支持

- 以结构化自然语言输入启动单目的地往返旅行规划。
- 当前 Mock 数据覆盖下的目的地、航线、日期和指定酒店能力检查。
- 天气、航班、酒店候选的 Local 或 MCP stdio 查询。
- 用户长期偏好与旅行历史读取。
- 候选排序、预算组合、RAG 证据检索、Proposal Verifier。
- LangGraph Checkpoint、Human-in-the-loop 的 approve / request_changes / cancel。
- 批准后的模拟草稿、不可变版本和审批审计记录。

## 明确非目标

- 不调用真实航班、酒店、支付或预订服务。
- 不声称已出票、已付款、已预订或已发送通知。
- 不支持多城市、多段交通、分段住宿或多人独立画像。
- 不把 RAG 文档当作实时价格、库存、开放时间或运营状态来源。
- 不把 Checkpoint 当成业务数据库，也不把业务数据库当成 Graph 恢复机制。

## 已知证据归属边界

- `search_city_activities` 等活动工具可提供按日期、城市和用户偏好筛选的结构化候选；其可用性和参数有效性由 `ToolCheck` 验证，候选会传入 Proposal。
- `EvidenceGrade` 当前只审查 RAG 的静态 `evidence_pool`，不会把活动工具候选当作某个 Information Need 的语义支持证据。
- 因而，若用户将“限时活动”等日期敏感活动设为必需要求，而版本化 Guides 语料中没有对应静态证据，系统会以 `insufficient_evidence` 保守停止，不生成未经支持的方案。这不是工具执行失败，也不代表系统会虚构活动。
- 将 ToolCheck 与 EvidenceGrade 的跨来源证据归属统一为结构化 Need 属性，是后续版本的架构演进项；当前范围不通过降低 RAG 审查门槛来规避这一边界。

## 数据与安全约束

- `data/mock/` 是模拟结构化事实源。
- `data/rag_docs/` 是版本化的补充知识语料；易变旅行信息需要在实际出行前核验。
- `.env`、本地模型、SQLite、Checkpoint、Chroma 和未审计 staging 数据不进入版本控制。
- 系统拒绝真实预订、支付等越界请求，并给出安全的最终响应。
