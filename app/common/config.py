from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


# ======================================================================
# Windows CPU inference stability
# ======================================================================
# Embedding 与 Cross-Encoder 都会间接导入 PyTorch、MKL 和 Hugging Face
# Tokenizers。Windows 上让这些运行库各自创建默认线程池时，曾在同一
# FastAPI worker 内触发原生访问冲突（0xC0000005），而不是可被 Python
# try/except 捕获的普通异常。
#
# 这段代码必须位于任何 RAG / Transformers 模块导入之前：底层运行库会在
# 首次 import 时读取这些变量。setdefault 不会覆盖用户在系统环境或部署
# 脚本中显式设置的值；非 Windows 平台也不改变原有线程策略。
if os.name == "nt":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Settings(BaseSettings):
    """
    TravelOps-Copilot 的统一配置。

    所有字段都可以通过项目根目录的 .env 覆盖。
    业务模块不要到处直接调用 os.getenv，而应统一读取 settings。

    这样做的作用：
        1. 配置类型可以由 Pydantic 校验；
        2. FastAPI、LangGraph、RAG 和数据库使用同一套配置来源；
        3. 测试可以通过环境变量关闭真实模型和持久化资源；
        4. README 可以明确列出项目全部运行参数。
    """

    # ==================================================================
    # 1. Application / Business Database
    # ==================================================================

    app_env: str = "dev"

    # 业务数据库保存：
    #     users / user_profiles / trip_history
    #     trip_drafts / trip_versions / approval_records
    database_url: str = (
        "sqlite:///./data/db/travelops.db"
    )

    # FastAPI 启动时自动创建尚不存在的业务表。
    # SQLAlchemy create_all() 不会删除现有数据。
    database_initialize_on_startup: bool = True

    weather_mode: str = "mock"

    # ==================================================================
    # 2. Travel Data Client / MCP
    # ==================================================================

    # 正式运行默认使用 mcp：
    #     Node -> MCPTravelDataClient -> MCP Server -> Tool -> Mock Provider
    #
    # 单元测试通过 tests/conftest.py 强制使用 local，
    # 或者在调用 Node 时显式注入 LocalTravelDataClient。
    travel_data_client_mode: Literal[
        "local",
        "mcp",
    ] = "mcp"

    # 当前 MCP Server 仍然是项目内模块，
    # MCPTravelDataClient 会通过当前 Python 解释器执行：
    #
    #     python -m app.mcp.travel_data_server
    mcp_travel_data_server_module: str = (
        "app.mcp.travel_data_server"
    )

    # 一次完整 stdio 连接、initialize 和 Tool 调用的超时时间。
    # 这是最小必要保护，防止 MCP Server 异常时主 Graph 永久等待。
    mcp_travel_data_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=300,
    )

    # ==================================================================
    # 3. LangGraph Runtime / Persistence / Human-in-the-loop
    # ==================================================================

    # 与业务数据库分开保存 LangGraph Checkpoint。
    # Checkpoint 保存 Graph State、执行位置和 interrupt；
    # 它不替代 TripDraft 业务持久化。
    langgraph_checkpoint_db: str = (
        "./data/db/langgraph_checkpoints.sqlite"
    )

    # 官方建议新应用限制 MessagePack 反序列化类型。
    # checkpoint_runtime.py 会在创建 SqliteSaver 前把它同步到环境变量。
    langgraph_strict_msgpack: bool = True

    # SQLite 连接等待数据库锁释放的最长时间。
    langgraph_sqlite_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=300,
    )

    # 是否在 FastAPI lifespan 启动阶段构建一次 CompiledGraph。
    # false 时，第一次 API 请求执行懒初始化。
    workflow_preload_on_startup: bool = True

    # 单次 invoke / resume 最大 Graph super-step 数量。
    # 这是运行时最后一道防无限循环保护，
    # 不替代 rag/proposal/budget retry_count。
    workflow_recursion_limit: int = Field(
        default=100,
        ge=10,
        le=1000,
    )

    # 公开 API 最多返回多少条经过字段白名单处理的 Trace 摘要。
    api_trace_max_items: int = Field(
        default=50,
        ge=0,
        le=500,
    )

    # Checkpoint 历史接口默认和最大返回数量。
    api_history_default_limit: int = Field(
        default=20,
        ge=1,
        le=200,
    )
    api_history_max_limit: int = Field(
        default=100,
        ge=1,
        le=500,
    )

    # ==================================================================
    # 3.1 Lightweight Trace Harness (Stage 5)
    # ==================================================================

    # Trace Harness 是进程内的轻量观测层，不是持久化 AgentOps 平台。
    # 限制条数可以防止本地长时间演示时 Trace 无边界占用内存。
    trace_max_runs: int = Field(
        default=200,
        ge=1,
        le=10000,
    )
    trace_max_spans_per_run: int = Field(
        default=500,
        ge=1,
        le=10000,
    )

    # ==================================================================
    # 4. Input Extraction / LLM
    # ==================================================================

    input_extract_mode: Literal[
        "rule",
        "hybrid",
        "llm",
    ] = "llm"

    input_extract_max_tokens: int = Field(
        default=4096,
        ge=512,
        le=32768,
    )

    # ToolPlan 使用原生 LLM tool_calls 选择唯一的可选活动工具。
    # rule 模式只用于离线测试和无 API Key 的确定性降级；四个必选业务
    # 需求（往返航班、酒店、天气）在两种模式下都由 Policy 强制加入。
    tool_plan_mode: Literal["rule", "llm"] = "llm"
    tool_plan_max_tokens: int = Field(default=1024, ge=256, le=4096)
    tool_plan_max_retries: int = Field(default=1, ge=0, le=2)
    tool_execute_max_retries: int = Field(default=1, ge=0, le=2)
    activity_tool_max_results: int = Field(default=12, ge=1, le=30)

    llm_provider: str = "deepseek"
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    llm_thinking_enabled: bool = False
    llm_temperature: float = Field(
        default=0.1,
        ge=0,
        le=2,
    )
    llm_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
    )
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
    )

    deepseek_api_key: str | None = None
    tavily_api_key: str | None = None

    # ==================================================================
    # 5. RAG Documents / Chroma / Embedding
    # ==================================================================

    rag_docs_dir: str = "./data/rag_docs"
    rag_chroma_dir: str = (
        "./data/vector_store/chroma"
    )
    rag_chroma_collection: str = (
        "travelops_rag_v1"
    )

    # 这是人工维护的数据集版本标签，必须与 RAG Eval 报告使用同一含义。
    # 更新语料、Chunking 或 Chroma collection 时应同步更新，才能解释指标变化。
    # The active fixture is intentionally bounded to a seven-day, three-city
    # window.  Keeping the version in settings lets traces and RAG Eval reports
    # identify which corpus was used instead of silently mixing old Gold IDs
    # with the rebuilt Markdown documents.
    rag_corpus_version: str = "v5-agentic-challenge"

    rag_embedding_model: str = (
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    rag_embedding_device: str = "cpu"

    rag_chunk_size: int = Field(
        default=600,
        ge=100,
    )
    rag_chunk_overlap: int = Field(
        default=100,
        ge=0,
    )

    rag_rrf_k: int = Field(
        default=60,
        ge=1,
    )
    rag_bm25_weight: float = Field(
        default=1.0,
        gt=0,
    )
    rag_vector_weight: float = Field(
        default=1.0,
        gt=0,
    )
    rag_fetch_multiplier: int = Field(
        # 默认交互路径不使用 Cross-Encoder，因此无需扩大候选池给第二阶段
        # 精排；使用 1 可以降低 Chroma 查询和 State 传递的开销。
        default=1,
        ge=1,
    )

    # 启动时同步 Chroma 并构建一次内存 BM25。
    # false 时在第一次 HybridRetrieveNode 调用时懒加载。
    rag_preload_on_startup: bool = True

    # ==================================================================
    # 5.1 Agentic RAG planning / semantic evidence
    # ==================================================================

    # rule 保持原有确定性规划；agentic 使用 LLM 理解 Information Needs，
    # 失败时自动回退 RuleBasedRetrievalPlanner。
    rag_planner_mode: Literal[
        "rule",
        "agentic",
    ] = "agentic"

    agentic_rag_semantic_judge_enabled: bool = True
    agentic_rag_max_replans: int = Field(default=1, ge=0, le=1)
    agentic_rag_max_initial_tasks: int = Field(default=6, ge=1, le=8)
    agentic_rag_max_repair_tasks: int = Field(default=2, ge=1, le=4)
    agentic_rag_planner_max_tokens: int = Field(default=3000, ge=512, le=8192)
    agentic_rag_judge_max_tokens: int = Field(default=2000, ge=512, le=8192)

    # ==================================================================
    # 6. Cross-Encoder Rerank
    # ==================================================================

    rag_rerank_model: str = (
        "BAAI/bge-reranker-v2-m3"
    )
    rag_rerank_device: str = "cpu"
    rag_rerank_batch_size: int = Field(
        default=4,
        ge=1,
    )
    rag_rerank_max_length: int = Field(
        default=512,
        ge=64,
    )
    rag_rerank_top_k_per_task: int = Field(
        default=5,
        ge=1,
    )
    rag_rerank_min_per_hotel: int = Field(
        default=1,
        ge=0,
    )
    rag_rerank_cache_size: int = Field(
        default=2048,
        ge=0,
    )
    rag_rerank_use_fp16: bool = False
    rag_rerank_fallback_on_error: bool = True

    # 默认交互路径使用 BM25 + Vector + RRF（Variant C）。
    #
    # false 时 RerankNode 不会导入或加载 Cross-Encoder，而是将 RRF 排名
    # 直接标准化为 Evidence 交给 EvidenceGradeNode。这样本地前端演示保留
    # 快速、稳定的检索闭环；离线 RAG Eval 仍可显式运行 Variant D。
    #
    # true 时才启用 BGE Cross-Encoder，适合高质量证据筛选或完成线程稳定性
    # 验证后的部署环境。它与 rag_rerank_preload_on_startup 是两个独立开关。
    rag_rerank_enabled: bool = False

    # 仅在 rag_rerank_enabled=true 时此开关才生效。默认 false，避免本地
    # 前端演示在启动阶段加载不走默认路径的 Cross-Encoder。
    rag_rerank_preload_on_startup: bool = False

    # ==================================================================
    # 7. Proposal Generation
    # ==================================================================

    proposal_max_tokens: int = Field(
        default=8192,
        ge=1024,
        le=32768,
    )

    # ProposalGenerator 内部：首次生成 + 带验证错误的修复尝试。
    proposal_max_attempts: int = Field(
        default=2,
        ge=1,
        le=3,
    )

    proposal_max_evidence_items: int = Field(
        default=24,
        ge=1,
        le=100,
    )
    proposal_max_evidence_chars: int = Field(
        default=1200,
        ge=100,
        le=10000,
    )

    # ==================================================================
    # 7. Proposal Verifier / Repair Loop
    # ==================================================================

    verifier_float_tolerance: float = Field(
        default=0.01,
        ge=0,
        le=1,
    )
    verifier_max_proposal_repairs: int = Field(
        default=1,
        ge=0,
        le=3,
    )
    verifier_max_budget_repairs: int = Field(
        default=1,
        ge=0,
        le=3,
    )
    verifier_max_issue_examples: int = Field(
        default=50,
        ge=1,
        le=200,
    )

    # ==================================================================
    # 9. Revision Analysis
    # ==================================================================

    revision_max_tokens: int = Field(
        default=4096,
        ge=512,
        le=16384,
    )
    revision_max_attempts: int = Field(
        default=2,
        ge=1,
        le=3,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """创建并缓存应用级 Settings 单例。"""

    return Settings()


settings = get_settings()
