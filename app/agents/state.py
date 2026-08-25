from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal, TypedDict


class TravelState(TypedDict, total=False):
    """
    TravelOps-Copilot 的 LangGraph 共享状态。

    使用原则：
        1. State 只保存跨节点共享的数据。
        2. 每个 Node 返回局部更新，不直接修改原始 state。
        3. 大型 Prompt、ORM 对象和完整模型对象不能放进 State。
        4. trace 和 errors 使用 add reducer，在并行或连续节点间追加。
        5. total=False 表示字段由对应节点执行后再出现，不需要一次全部初始化。
    """

    # ==================================================================
    # 1. 基础输入
    # ==================================================================

    user_id: str
    """
    当前用户 ID。

    示例：
        "user_001"
    """

    trip_session_id: str
    """
    一次旅行规划会话的稳定 ID。

    推荐同时作为 LangGraph thread_id 使用。

    它连接两类状态：
        - LangGraph Checkpointer 中的执行线程；
        - CommitDraftNode 保存到业务数据库的 TripDraft。

    示例：
        "trip_session_user001_a81f"

    注意：
        user_id 表示“是谁”；
        trip_session_id 表示“这个用户的哪一次旅行规划”。
    """

    raw_message: str
    """
    用户本轮原始自然语言输入。

    示例：
        "2026年7月2日从杭州去成都玩三天，预算4000，住宿好一点，航班别太贵。"
    """

    reference_date: str
    """
    相对日期解析基准，格式 YYYY-MM-DD。

    示例：
        "2026-07-01"
    """

    workflow: Literal[
        "plan_trip",
        "modify_trip",
        "update_preferences",
        "clarify",
        "unsupported",
    ]
    """
    当前工作流类型。

    示例：
        "plan_trip"
    """

    # ==================================================================
    # 2. InputExtractNode：结构化旅行需求
    # ==================================================================

    trip_request: dict[str, Any]
    """
    InputExtractNode 的输出。

    最终抽取链路：
        DeepSeek JSON Structured Output
        → Pydantic Schema
        → 规则 Validator
        → LLM 失败时 Rule-based fallback

    示例：
        {
            "user_id": "user_001",
            "origin": "杭州",
            "destination": "成都",
            "start_date": "2026-07-02",
            "end_date": "2026-07-04",
            "partial_date": null,
            "days": 3,
            "budget": 4000.0,
            "people_count": 1,
            "room_count": 1,

            "preference_signals": [
                {
                    "domain": "hotel",
                    "key": "quiet",
                    "label": "安静",
                    "strength": 1.0,
                    "polarity": "positive",
                    "evidence": "一定要安静",
                    "source": "merged"
                },
                {
                    "domain": "activity",
                    "key": "food",
                    "label": "美食",
                    "strength": 0.9,
                    "polarity": "positive",
                    "evidence": "想多吃当地美食",
                    "source": "llm"
                }
            ],

            "spend_preferences": [
                {
                    "category": "hotel",
                    "direction": "increase",
                    "strength": 0.9,
                    "evidence": "住宿好一点",
                    "source": "llm"
                },
                {
                    "category": "transport",
                    "direction": "decrease",
                    "strength": 0.8,
                    "evidence": "航班不用很贵",
                    "source": "llm"
                }
            ],

            "hard_constraints": [
                {
                    "domain": "flight",
                    "field": "is_direct",
                    "operator": "==",
                    "value": true,
                    "evidence": "必须直飞",
                    "source": "llm"
                }
            ],

            "named_constraints": [
                {
                    "entity_type": "food",
                    "entity_name": "鸭血粉丝汤",
                    "constraint_mode": "preferred",
                    "scope": "itinerary",
                    "traveler": null,
                    "evidence": "想吃正宗鸭血粉丝汤",
                    "source": "llm"
                }
            ],

            "unmapped_requirements": [],
            "requirement_issues": [],

            "field_resolution": {
                "destination": {
                    "value": "成都",
                    "resolution_type": "explicit",
                    "evidence": "去成都",
                    "needs_clarification": false
                }
            },

            "transport_preferences": ["avoid_early_flight"],
            "hotel_preferences": ["quiet"],
            "travel_style": ["food"],
            "raw_constraints": ["一定要安静", "住宿好一点"],
            "raw_message": "...",

            "extraction": {
                "method": "hybrid_llm_rule_v2",
                "model": "deepseek-v4-flash",
                "confidence": 0.9,
                "llm_used": true,
                "fallback_used": false,
                "missing_fields": [],
                "notes": []
            }
        }

    关键边界：
        - 主观偏好只使用 0—1 strength，不自动变成硬约束。
        - hard_constraints 只能保存客观、可验证条件。
        - LLM 可以用稳定知识解析“兵马俑所在城市→西安”，但不能编造动态事实。
    """

    # ==================================================================
    # 3. MissingInfo / Capability / Clarify
    # ==================================================================

    missing_fields: list[str]
    """
    缺失核心字段列表。

    示例：
        ["start_date"]
    """

    can_continue: bool
    """
    当前输入是否能够进入后续安全检查。

    示例：
        false
    """

    capability_result: dict[str, Any]
    """
    当前 Mock 数据能力是否覆盖该请求。

    示例：
        {
            "status": "unsupported_data_scope",
            "supported": false,
            "reason": "当前模拟数据不支持目的地西安。",
            "issues": [
                {
                    "issue_type": "unsupported_destination",
                    "message": "缺少西安的酒店或天气 Mock 数据。",
                    "requires_clarification": true
                }
            ],
            "supported_destinations": ["成都"]
        }
    """

    missing_info_result: dict[str, Any]
    """
    MissingInfoCheckNode 的完整确定性校验结果。

    示例：
        {
            "status": "missing_required_fields",
            "missing_fields": ["start_date"],
            "validation_issues": [],
            "blocking_issues": [],
            "capability_result": {"status": "not_checked", "supported": true},
            "can_continue": false,
            "message": "还需要补充出发日期，才能继续规划。",
            "clarification_questions": [
                {
                    "field": "start_date",
                    "question": "请补充具体出发日期，例如2026-07-02。"
                }
            ]
        }
    """

    clarification: dict[str, Any]
    """
    ClarifyNode 输出的用户追问。

    示例：
        {
            "type": "missing_info",
            "status": "needs_clarification",
            "message": "还需要补充出发日期，才能继续规划。",
            "missing_fields": ["start_date"],
            "questions": [...],
            "next_action": "provide_missing_info"
        }
    """

    # ==================================================================
    # 4. SafetyCheck / SafeReject
    # ==================================================================

    safety_result: dict[str, Any]
    """
    安全和项目边界检查结果。

    示例：
        {
            "blocked": true,
            "reason": "请求包含真实预订动作。",
            "safe_mode": "reject_or_draft_only",
            "flags": [...],
            "blocked_actions": ["real_booking"]
        }
    """

    safety_flags: list[dict[str, Any]]
    """
    命中的具体安全规则。

    示例：
        [
            {
                "rule_id": "real_booking_request",
                "category": "real_booking",
                "level": "high",
                "action": "block",
                "matched_keyword": "直接预订"
            }
        ]
    """

    # ==================================================================
    # 5. MemoryReadNode
    # ==================================================================

    user_profile: dict[str, Any]
    """
    长期偏好与本次偏好合并结果。

    示例：
        {
            "user_id": "user_001",
            "source": "sql:user_profiles",
            "long_term": {
                "hotel_preferences": {"quiet": 0.85, "near_subway": 0.7},
                "flight_preferences": {"avoid_early_flight": 0.9},
                "activity_preferences": {"food": 0.75},
                "pace_preferences": {"slow": 0.85}
            },
            "current_request": {
                "hotel_preferences": {"quiet": 1.0},
                "spend_preferences": [
                    {"category": "hotel", "direction": "increase", "strength": 0.9}
                ]
            },
            "effective": {
                "hotel_preferences": {"quiet": 1.0, "near_subway": 0.7},
                "flight_preferences": {"avoid_early_flight": 0.9},
                "activity_preferences": {"food": 0.75},
                "pace_preferences": {"slow": 0.85}
            },
            "recent_trip_history": [],
            "memory_summary": "已从SQL读取长期偏好。"
        }

    注意：
        hard_constraints 不属于长期偏好，不会保存在 user_profile.effective 中。
    """

    memory_update: dict[str, Any]
    """
    UpdatePreferencesWorkflow 未来写入长期记忆的 patch。

    示例：
        {
            "preference_patch": {"hotel_preferences": {"quiet": 0.95}},
            "reason": "用户明确表示以后都优先安静酒店。"
        }
    """

    # ==================================================================
    # 6. PlanningContextNode / Budget Strategy
    # ==================================================================

    planning_context: dict[str, Any]
    """
    后续所有规划节点统一读取的 Context Engineering 结果。

    示例：
        {
            "request": {
                "origin": "杭州",
                "destination": "成都",
                "start_date": "2026-07-02",
                "end_date": "2026-07-04",
                "days": 3,
                "nights": 2,
                "people_count": 1,
                "room_count": 1,
                "total_budget": 4000.0
            },

            "hard_constraints": {
                "trip": [
                    {
                        "field": "total_budget",
                        "operator": "<=",
                        "value": 4000.0
                    }
                ],
                "flight": [
                    {
                        "field": "is_direct",
                        "operator": "==",
                        "value": true
                    }
                ],
                "hotel": []
            },

            "preference_weights": {
                "flight": {"avoid_early_flight": 0.9},
                "hotel": {"quiet": 1.0, "cleanliness": 0.9},
                "activity": {"food": 0.9},
                "pace": {"slow": 0.85},
                "risk": {}
            },

            "spend_preferences": [
                {"category": "hotel", "direction": "increase", "strength": 0.9}
            ],
            "named_constraints": [],
            "unmapped_requirements": [],
            "diet_preferences": {},
            "budget_plan": {...},
            "context_summary": {...},
            "source_meta": {"builder_version": "planning_context_v2"}
        }
    """

    budget_plan: dict[str, Any]
    """
    PlanningContextNode 同步写到 State 顶层的预算策略。

    示例：
        {
            "mode": "budget_limited",
            "total_budget": 4000.0,
            "hard_limits": {
                "total_budget": 4000.0,
                "max_hotel_price_per_night": null,
                "max_transport_total": null
            },
            "base_ratios": {
                "transport": 0.35,
                "hotel": 0.35,
                "food_activity": 0.30
            },
            "adjusted_ratios": {
                "transport": 0.23,
                "hotel": 0.42,
                "food_activity": 0.35
            },
            "soft_targets": {
                "transport_budget": 920.0,
                "hotel_budget": 1680.0,
                "food_activity_budget": 1400.0,
                "food_activity_budget_per_day": 466.67,
                "target_hotel_price_per_room_night": 650.0
            },
            "allocation_source": "user_adjusted_heuristic_v1",
            "allocation_reasons": ["根据用户表达，提高住宿预算软目标。"],
            "room_count": 1,
            "nights": 2
        }

    注意：
        soft_targets 只参与评分；总预算和显式金额上限才是硬限制。
    """

    # ==================================================================
    # 7. MCP / Mock 外部结构化数据
    # ==================================================================

    tool_plan: dict[str, Any]
    """Policy 必选调用与 LLM 可选活动调用合并后的统一执行计划。"""

    tool_execution_result: dict[str, Any]
    """ToolExecuteNode 的执行摘要，不替代各工具原有 fetch_meta。"""

    tool_check_result: dict[str, Any]
    """ToolCheckNode 的通过、重试、重新规划或失败路由决定。"""

    tool_plan_retry_count: int
    tool_execute_retry_count: int

    activity_result: dict[str, Any]
    """活动 MCP 工具原始结果；未选择时 status=skipped。"""

    activity_candidates: list[dict[str, Any]]
    """供 Proposal 使用的结构化活动事实，不进入 CandidateRank/BudgetOptimize。"""

    activity_fetch_meta: dict[str, Any]
    """活动工具调用签名、协议、状态和降级信息。"""

    weather_result: dict[str, Any]
    """
    WeatherNode 通过 MCP 查询 Mock 得到的逐日天气。

    示例：
        {
            "status": "ok",
            "city": "成都",
            "daily": [
                {
                    "date": "2026-07-03",
                    "condition": "小雨",
                    "risk_tags": ["rain"]
                }
            ],
            "risks": [
                {"risk_type": "rain", "level": "medium", "date": "2026-07-03"}
            ],
            "source": "mock_weather_provider"
        }
    """

    raw_flight_results: dict[str, Any]
    """
    FlightSearchNode 的原始可用航班结果，还没有按用户偏好排序。

    示例：
        {
            "status": "ok",
            "outbound": [
                {
                    "flight_id": "F001",
                    "depart_time": "10:00",
                    "arrive_time": "12:45",
                    "price": 720.0,
                    "is_direct": true,
                    "available_seats": 4
                }
            ],
            "return": [...]
        }
    """

    raw_hotel_results: dict[str, Any]
    """
    HotelSearchNode 的原始可用酒店结果，还没有按用户偏好排序。

    示例：
        {
            "status": "ok",
            "items": [
                {
                    "hotel_id": "H001",
                    "name": "成都青羊静巷酒店",
                    "price_per_night": 580.0,
                    "rating": 4.7,
                    "quiet_score": 0.90,
                    "cleanliness_score": 0.92,
                    "near_subway": true,
                    "available_rooms": 4
                }
            ]
        }
    """

    weather_fetch_meta: dict[str, Any]
    """
    天气查询签名，用于 ModifyTrip 判断是否需要刷新。

    示例：
        {"tool_name": "get_weather", "query_signature": {"city": "成都", "start_date": "2026-07-02"}}
    """

    flight_fetch_meta: dict[str, Any]
    """
    航班查询签名。

    示例：
        {"tool_name": "search_flights", "query_signature": {"origin": "杭州", "destination": "成都"}}
    """

    hotel_fetch_meta: dict[str, Any]
    """
    酒店查询签名。

    示例：
        {"tool_name": "search_hotels", "query_signature": {"city": "成都", "nights": 2, "people_count": 1}}
    """

    # ==================================================================
    # 8. CandidateRank / BudgetOptimize（后续实现）
    # ==================================================================

    flight_candidates: list[dict[str, Any]]
    """
    CandidateRankNode 排序后的航班。

    示例：
        [
            {
                "flight_id": "F001",
                "direction": "outbound",
                "score_breakdown": {
                    "time_fit": 0.95,
                    "price_fit": 0.88,
                    "direct_fit": 1.0,
                    "change_policy_fit": 0.7
                },
                "total_score": 0.91,
                "ranking_reasons": ["出发时间合适", "价格接近交通软目标"]
            }
        ]
    """

    hotel_candidates: list[dict[str, Any]]
    """
    CandidateRankNode 排序后的酒店。

    酒店总分只由：偏好匹配分 + 价格适配分 + 基础评分组成，RAG 不参与评分。

    示例：
        [
            {
                "hotel_id": "H001",
                "score_breakdown": {
                    "preference_match": 0.92,
                    "price_fit": 0.85,
                    "base_rating": 0.94
                },
                "total_score": 0.90,
                "preference_gaps": []
            }
        ]
    """

    candidate_rank_result: dict[str, Any]
    """
    CandidateRankNode 的总体摘要和被硬约束拒绝的候选。

    示例：
        {
            "status": "ok",
            "rejected_flights": [{"flight_id": "F002", "reason": "不满足必须直飞"}],
            "rejected_hotels": [],
            "flight_count": 4,
            "hotel_count": 5
        }
    """

    selection_result: dict[str, Any]
    """
    BudgetOptimizeNode 的权威组合选择结果。

    示例：
        {
            "status": "feasible",
            "selected_combination": {
                "outbound_flight_id": "F001",
                "return_flight_id": "F101",
                "hotel_id": "H001"
            },
            "selected_hotel": {
                "hotel_id": "H001",
                "name": "成都青羊静巷酒店"
            },
            "selected_costs": {
                "outbound_flight": 720.0,
                "return_flight": 680.0,
                "hotel": 1160.0,
                "subtotal": 2560.0
            },
            "remaining_budget": 1440.0,
            "remaining_budget_allocation": {
                "food_activity": 1100.0,
                "local_transport_and_buffer": 340.0
            },
            "combination_score": 0.91
        }

    注意：
        remaining_budget_allocation 是预算安排，不是实际消费预测。
    """

    budget_result: dict[str, Any]
    """
    预算组合检查摘要；后续可作为 selection_result 中预算部分的兼容视图。

    示例：
        {"status": "within_budget", "selected_subtotal": 2560.0, "remaining_budget": 1440.0}
    """

    adjustment_plan: dict[str, Any]
    """
    没有预算可行组合时的确定性调整建议。

    示例：
        {"status": "need_user_change", "suggestions": ["提高预算", "降低酒店价格偏好"]}
    """

    budget_optimize_result: dict[str, Any]
    """
    BudgetOptimizeNode 的运行摘要。

    示例：
        {
            "status": "feasible",
            "strategy_version": "deterministic_budget_optimize_v1",
            "evaluated_combination_count": 18,
            "feasible_combination_count": 7,
            "rejected_combination_count": 11,
            "selected_combination_id": "combo_001"
        }
    """

    # ==================================================================
    # 9. Agentic RAG
    # ==================================================================

    retrieval_plan: dict[str, Any]
    """
    RetrievalPlanNode 输出的任务计划。

    新工作流中，它在 BudgetOptimize 之后执行。

    示例：
        {
            "plan_id": "rp_xxx",
            "mode": "initial",
            "strategy_version": "rule_retrieval_plan_v2",
            "city": "成都",
            "tasks": [
                {
                    "task_id": "guide_main",
                    "category": "guides",
                    "doc_type": "guide",
                    "semantic_query": "成都 三日游 美食 雨天室内景点 逐日行程",
                    "keyword_query": "成都 三日游 美食",
                    "metadata_filter": {"city": "成都", "doc_type": "guide"},
                    "required": true,
                    "top_k": 8,
                    "min_required_chunks": 2
                },
                {
                    "task_id": "selected_hotel_info",
                    "category": "hotel_reviews",
                    "metadata_filter": {"hotel_ids": ["H001"]},
                    "required": false,
                    "min_required_chunks": 0
                }
            ],
            "coverage_requirements": {
                "guides": {"required": true, "min_required_chunks": 2},
                "hotel_reviews": {"required": false, "min_required_chunks": 0}
            }
        }
    """

    retrieved_chunks: list[dict[str, Any]]
    """
    HybridRetrieveNode 的 BM25 + Chroma Vector + RRF 融合结果。

    示例：
        [
            {
                "task_id": "guide_main",
                "chunk_id": "guide_cd::02::001",
                "content": "雨天可以优先安排博物馆和茶馆……",
                "source": "guides/guide_chengdu_rainy_day.md",
                "metadata": {"doc_type": "guide", "city": "成都", "section": "雨天安排"},
                "bm25_rank": 2,
                "vector_rank": 1,
                "fusion_score": 0.0325,
                "final_rank": 1
            }
        ]
    """

    hybrid_retrieval_result: dict[str, Any]
    """
    HybridRetrieveNode 运行摘要。

    示例：
        {"status": "ok", "task_count": 3, "total_retrieved_chunks": 18, "rrf_k": 60}
    """

    reranked_evidence: list[dict[str, Any]]
    """
    Cross-Encoder 精排后的证据；EvidenceGrade 通过后会被替换为累计 evidence_pool。

    示例：
        [
            {
                "task_id": "guide_main",
                "category": "guides",
                "chunk_id": "guide_cd::02::001",
                "rerank_score": 0.94,
                "rerank_rank": 1,
                "source": "guides/guide_chengdu_rainy_day.md",
                "metadata": {...}
            }
        ]
    """

    rerank_result: dict[str, Any]
    """
    RerankNode 运行摘要。

    示例：
        {"status": "ok", "model_id": "./models/bge-reranker-v2-m3", "fallback_used": false}
    """

    evidence_pool: list[dict[str, Any]]
    """
    initial 证据和 repair 新证据合并后的累计证据池。

    去重键：
        category + chunk_id
    """

    evidence_requirements: dict[str, Any]
    """
    initial RetrievalPlan 保存的完整证据验收要求。

    示例：
        {
            "guides": {"required": true, "min_required_chunks": 2},
            "safety_notices": {"required": true, "min_required_chunks": 1},
            "hotel_reviews": {"required": false, "min_required_chunks": 0}
        }
    """

    evidence_result: dict[str, Any]
    """
    EvidenceGradeNode 的质量检查结果。

    示例：
        {
            "status": "passed",
            "passed": true,
            "repairable": false,
            "next_action": "continue",
            "coverage": {...},
            "issues": [],
            "retrieval_feedback": null
        }
    """

    retrieval_feedback: dict[str, Any]
    """
    EvidenceGradeNode 返回给 RetrievalPlanNode 的修复信息。

    示例：
        {
            "missing_doc_types": ["guides"],
            "missing_hotel_ids": [],
            "missing_risk_types": ["wind"],
            "reason": "攻略和大风安全证据不足。"
        }
    """

    # ==================================================================
    # 10. Proposal / Verifier / Human Decision（后续实现）
    # ==================================================================

    proposal: dict[str, Any]
    """
    ProposalNode 输出的结构化旅行方案。

    锁定事实来自 Mock 和 selection_result；LLM 只能生成逐日活动、解释和建议。

    示例：
        {
            "proposal_id": "P001",
            "selected_flights": {"outbound_flight_id": "F001", "return_flight_id": "F101"},
            "selected_hotel": {"hotel_id": "H001", "name": "成都青羊静巷酒店"},
            "daily_plan": [
                {
                    "date": "2026-07-03",
                    "weather": "小雨",
                    "activities": ["室内博物馆", "茶馆和美食体验"],
                    "weather_adjustment": "将户外公园调整到晴天。",
                    "evidence_refs": ["guide_cd::02::001"]
                }
            ],
            "budget_summary": {"selected_subtotal": 2560.0, "remaining_budget": 1440.0}
        }
    """

    proposal_result: dict[str, Any]
    """
    ProposalNode 的生成运行摘要。

    示例：
        {
            "status": "generated",
            "model_id": "deepseek-v4-flash",
            "attempt_count": 1,
            "used_evidence_refs": [
                "guide_rain_day",
                "safety_rain"
            ],
            "locked_fact_hash": "...",
            "proposal_id": "proposal_xxx",
            "daily_plan_count": 3
        }
    """

    proposal_feedback: dict[str, Any]
    """
    VerifierNode 返回给 ProposalNode 的跨节点修复反馈。

    它与 ProposalGenerator 内部 validation_errors 不同：

        validation_errors
            同一次 ProposalNode 调用中的 JSON/业务校验错误。

        proposal_feedback
            最终 TripProposal 经过独立 Verifier 后发现的问题，
            需要重新运行 ProposalNode。

    示例：
        {
            "source": "deterministic_proposal_verifier_v1",
            "next_action": "proposal",
            "repair_round": 1,
            "issues": [
                {
                    "issue_type": "rain_day_all_outdoor",
                    "path": "proposal.daily_plan[2026-07-03].activities",
                    "message": "雨天不能把所有活动都安排为 outdoor。"
                }
            ],
            "instructions": [
                "雨天不能把所有活动都安排为 outdoor。"
            ]
        }
    """

    itinerary_status: Literal[
        "collecting_info",
        "proposed",
        "revision_requested",
        "revised",
        "approved_simulated",
        "rejected",
    ]
    """
    当前方案状态。

    示例：
        "proposed"
    """

    current_proposal: dict[str, Any]
    """
    ModifyTripWorkflow 加载的当前方案版本。

    示例：
        {"proposal_id": "P001", "version": 1, "status": "proposed"}
    """

    verifier_result: dict[str, Any]
    """
    VerifierNode 对最终 TripProposal 的独立确定性审查结果。

    示例：
        {
            "status": "passed",
            "passed": true,
            "repairable": false,
            "next_action": "decision_gate",
            "proposal_id": "proposal_xxx",
            "expected_locked_fact_hash": "...",
            "actual_locked_fact_hash": "...",
            "approval_required": true,
            "check_count": 10,
            "passed_check_count": 10,
            "warning_count": 0,
            "error_count": 0,
            "critical_count": 0,
            "checks": {
                "locked_facts": {
                    "passed": true,
                    "issue_count": 0
                },
                "evidence_provenance": {
                    "passed": true,
                    "issue_count": 0
                }
            },
            "issues": [],
            "repair_feedback": {}
        }

    next_action 可能为：
        - decision_gate：通过，等待用户批准；
        - proposal：修复 LLM 内容或 Proposal 组装；
        - retrieval_plan：重新执行 RAG repair；
        - budget_optimize：重新选择预算组合；
        - final_response：无法自动修复或重试耗尽。
    """

    pending_actions: list[dict[str, Any]]
    """
    等待人工确认的内部动作。

    示例：
        [{"action_type": "approve_proposal", "status": "pending", "requires_approval": true}]
    """

    decision: Literal["approve", "request_changes", "cancel"] | None
    """
    DecisionGateNode 已接受的用户决策。

    该字段只有在人工 Resume Payload 通过 Pydantic 和过期方案检查后才写入。

    示例：
        "request_changes"
    """

    decision_result: dict[str, Any]
    """
    DecisionGateNode 对最近一次人工输入的结构化处理结果。

    示例：
        {
            "status": "accepted",
            "accepted": true,
            "decision": "approve",
            "next_action": "commit_draft",
            "proposal_id": "proposal_001",
            "proposal_version": 1,
            "action_id": "approve_proposal_001",
            "decision_event_id": "decision_abcd1234",
            "attempt_count": 1,
            "decided_at": "2026-07-23T10:00:00+00:00"
        }

    非法人工输入示例：
        {
            "status": "invalid",
            "accepted": false,
            "next_action": "decision_gate",
            "validation_errors": [
                "提交的 proposal_id 已过期或不匹配"
            ]
        }
    """

    decision_history: Annotated[list[dict[str, Any]], add]
    """
    已接受人工决定的追加式审计历史。

    使用 add reducer 的原因：
        一次旅行方案可能经历多轮：

            request_changes
            → 新 Proposal
            → approve

        每一次合法人工决定都需要保留，不能被后一次覆盖。

    示例：
        [
            {
                "decision_event_id": "decision_001",
                "proposal_id": "proposal_v1",
                "decision": "request_changes",
                "next_action": "revision_analyze"
            },
            {
                "decision_event_id": "decision_002",
                "proposal_id": "proposal_v2",
                "decision": "approve",
                "next_action": "commit_draft"
            }
        ]
    """

    decision_gate_error: dict[str, Any]
    """
    最近一次非法 Human-in-the-loop 输入的重新提示信息。

    DecisionGateNode 回到自身后，下一次 interrupt payload 会携带该字段，
    让前端向用户说明为什么需要重新提交。

    示例：
        {
            "error_type": "invalid_human_decision",
            "message": "提交的 proposal_id 已过期或不匹配",
            "attempt_count": 1
        }
    """

    decision_attempt_count: int
    """
    当前 Proposal 在 DecisionGate 中收到的人工提交次数。

    包括：
        - 非法提交；
        - 最终被接受的提交。

    它不是自动重试次数，因为每一次都需要真实人工输入。

    示例：
        2
    """

    change_request: dict[str, Any]
    """
    DecisionGateNode 保存的原始修改请求。

    示例：
        {
            "raw_message": "酒店换成绿水青山酒店，多待两天，预算增加1000元。",
            "base_proposal_id": "proposal_v1",
            "base_proposal_version": 1,
            "submitted_at": "2026-07-25T10:00:00+00:00"
        }
    """

    revision_plan: dict[str, Any]
    """
    RevisionAnalyzeNode 输出的受控 RevisionPatch。

    它只描述“如何修改当前 TripRequest”，不会重新生成完整请求。

    示例：
        {
            "status": "ready",
            "base_proposal_id": "proposal_v1",
            "base_trip_request_hash": "...",
            "patch": {
                "field_operations": [
                    {"operation": "increment", "field": "days", "value": 2},
                    {"operation": "increment", "field": "budget", "value": 1000}
                ],
                "replace_named_entity_types": ["hotel"],
                "named_constraint_additions": [
                    {
                        "entity_type": "hotel",
                        "entity_name": "绿水青山酒店",
                        "constraint_mode": "required"
                    }
                ]
            }
        }
    """

    revision_result: dict[str, Any]
    """
    ApplyRevisionNode 的确定性应用结果。

    示例：
        {
            "status": "applied",
            "before_trip_request_hash": "...",
            "after_trip_request_hash": "...",
            "changed_fields": ["days", "end_date", "budget", "named_constraints"],
            "invalidated_fields": ["planning_context", "weather_result", "proposal"]
        }
    """

    # ==================================================================
    # 11. CommitDraft / Cancel 业务收尾
    # ==================================================================

    trip_draft: dict[str, Any]
    """
    CommitDraftNode 保存后的当前内部模拟旅行草稿摘要。

    完整 Proposal JSON 不在 State 中重复保存，
    而是保存在 SQL trip_versions.proposal_json。

    示例：
        {
            "trip_id": "trip_abc",
            "trip_session_id": "trip_session_user001_a81f",
            "status": "approved_simulated",
            "current_version": 1,
            "current_proposal_id": "proposal_001",
            "destination": "成都",
            "known_subtotal": 2560.0,
            "draft_only": true
        }
    """

    trip_version: dict[str, Any]
    """
    本次批准产生的不可变 Proposal 版本摘要。

    完整 proposal / verifier / decision 快照保存在 SQL trip_versions 表。

    示例：
        {
            "version_id": "version_def",
            "trip_id": "trip_abc",
            "version_number": 1,
            "proposal_id": "proposal_001",
            "created_by_decision_event_id": "decision_xyz"
        }
    """

    approval_record: dict[str, Any]
    """
    本次人工批准在业务数据库中的审计记录摘要。

    示例：
        {
            "approval_id": "approval_xyz",
            "decision_event_id": "decision_xyz",
            "trip_id": "trip_abc",
            "proposal_id": "proposal_001",
            "decision": "approve"
        }
    """

    commit_result: dict[str, Any]
    """
    CommitDraftNode 的事务和幂等执行摘要。

    示例：
        {
            "status": "committed",
            "idempotent_replay": false,
            "records_written": 3,
            "trip_id": "trip_abc",
            "version_id": "version_def",
            "approval_id": "approval_xyz"
        }
    """

    cancel_result: dict[str, Any]
    """
    CancelNode 对当前 Proposal 取消收尾的结构化结果。

    示例：
        {
            "status": "cancelled",
            "idempotent_replay": false,
            "proposal_id": "proposal_001",
            "decision_event_id": "decision_cancel_xyz",
            "proposal_deleted": false,
            "real_booking_cancelled": false,
            "message": "当前旅行方案已取消。"
        }

    注意：
        CancelNode 不删除 Proposal，也不执行真实预订取消。
        State 通过 itinerary_status="rejected" 表示当前流程已取消。
    """

    # ==================================================================
    # 12. 循环保护、最终响应与可观测性
    # ==================================================================

    budget_retry_count: int
    """
    预算调整循环次数。

    示例：
        0
    """

    rag_retry_count: int
    """
    RAG repair 已执行次数。

    示例：
        1
    """

    proposal_retry_count: int
    """
    Proposal / Verifier 修复次数。

    示例：
        0
    """

    final_response: dict[str, Any]
    """
    最终 API 响应；不能直接返回完整 TravelState。

    示例：
        {
            "status": "decision_required",
            "message": "旅行方案已生成，请批准、修改或取消。",
            "proposal": {...},
            "available_actions": ["approve", "request_changes", "cancel"]
        }
    """

    trace: Annotated[list[dict[str, Any]], add]
    """
    节点和工具 Trace。add reducer 会把多个节点的列表追加起来。

    示例：
        [
            {
                "node_name": "retrieval_plan",
                "tool_name": "retrieval_planner",
                "status": "success",
                "latency_ms": 2,
                "input_summary": "destination=成都...",
                "output_summary": "task_count=3..."
            }
        ]
    """

    errors: Annotated[list[dict[str, Any]], add]
    """
    程序异常列表。正常业务分支，例如缺字段或无可行候选，不一定是 error。

    示例：
        [
            {
                "node": "weather",
                "type": "FileNotFoundError",
                "message": "weather_mock.json 不存在",
                "created_at": "2026-07-20T12:00:00+00:00"
            }
        ]
    """
