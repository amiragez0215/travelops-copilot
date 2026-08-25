from __future__ import annotations

import os


# ======================================================================
# Test environment isolation
# ======================================================================
# 这些环境变量必须在测试模块导入 app.common.config 之前设置。
# Settings 使用 lru_cache，因此测试进程启动后不应再依赖动态修改 .env。

# 1. InputExtract 单元测试默认使用规则模式，绝不调用真实 DeepSeek。
os.environ["INPUT_EXTRACT_MODE"] = "rule"

# 2. FastAPI lifespan 不加载真实 Chroma、Embedding 和 Reranker。
os.environ["RAG_PRELOAD_ON_STARTUP"] = "false"
os.environ["RAG_RERANK_PRELOAD_ON_STARTUP"] = "false"

# 2.1 Planner/Judge 单测必须显式注入 Fake 才能走 Agentic。
#     其他 Node 回归保持规则模式，避免读取本机 .env 中的 Key。
os.environ["RAG_PLANNER_MODE"] = "rule"

# 3. API 单元测试会注入 Fake Workflow Runtime，
#    不应自动创建正式 CompiledGraph 和项目 Checkpoint 连接。
os.environ["WORKFLOW_PRELOAD_ON_STARTUP"] = "false"

# 4. API 测试使用临时数据库或 Fake Node，
#    不应在导入 App 时修改本地 data/db/travelops.db。
os.environ["DATABASE_INITIALIZE_ON_STARTUP"] = "false"

# 5. 新应用启用严格 Checkpoint MessagePack 类型限制。
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"
# 6. 普通单元测试默认不启动真实 MCP 子进程。
#    正式主链通过本地 .env 的 TRAVEL_DATA_CLIENT_MODE=mcp 使用 MCP。
os.environ["TRAVEL_DATA_CLIENT_MODE"] = "local"

# 7. 普通回归不访问真实模型；ToolPlan 的原生 Tool Calling 单测会注入
#    FakeNativeToolCallingClient，并显式验证 LLM 分支。
os.environ["TOOL_PLAN_MODE"] = "rule"
