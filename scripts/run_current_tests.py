from __future__ import annotations

import subprocess
import sys


TEST_TARGETS = [
    "tests/test_extractors/test_preference_extractor.py",
    "tests/test_extractors/test_constraint_extractor.py",
    "tests/test_extractors/test_hybrid_trip_request_extractor.py",
    "tests/test_extractors/test_trip_request_extractor.py",
    "tests/test_nodes/test_input_extract_node.py",
    "tests/test_validators/test_trip_request_validator.py",
    "tests/test_validators/test_request_capability_validator.py",
    "tests/test_nodes/test_missing_info_node.py",
    "tests/test_formatters/test_clarification_formatter.py",
    "tests/test_nodes/test_clarify_node.py",
    "tests/test_guards/test_safety_guard.py",
    "tests/test_nodes/test_safety_node.py",
    "tests/test_formatters/test_safe_reject_formatter.py",
    "tests/test_nodes/test_safe_reject_node.py",
    "tests/test_db/test_seed_data.py",
    "tests/test_tools/test_memory_tool.py",
    "tests/test_nodes/test_memory_node.py",
    "tests/test_builders/test_planning_context_builder.py",
    "tests/test_nodes/test_planning_context_node.py",
    "tests/test_providers/test_mock_providers.py",
    "tests/test_mcp/test_travel_data_tools.py",
    "tests/test_mcp/test_client_factory.py",
    "tests/test_nodes/test_data_fetch_nodes.py",

    "tests/test_ranking/test_candidate_ranker.py",
    "tests/test_nodes/test_candidate_rank_node.py",
    "tests/test_optimization/test_budget_optimizer.py",
    "tests/test_nodes/test_budget_optimize_node.py",

    "tests/test_rag/test_retrieval_planner.py",
    "tests/test_nodes/test_retrieval_plan_node.py",
    "tests/test_rag/test_document_loader.py",
    "tests/test_rag/test_bm25_index.py",
    "tests/test_rag/test_chroma_vector_store.py",
    "tests/test_rag/test_hybrid_retriever.py",
    "tests/test_nodes/test_hybrid_retrieve_node.py",
    "tests/test_rag/test_reranker.py",
    "tests/test_nodes/test_rerank_node.py",
    "tests/test_rag/test_evidence_grader.py",
    "tests/test_nodes/test_evidence_grade_node.py",

    "tests/test_proposals/test_proposal_generator.py",
    "tests/test_nodes/test_proposal_node.py",
    "tests/test_verifiers/test_proposal_verifier.py",
    "tests/test_nodes/test_verifier_node.py",
    "tests/test_nodes/test_decision_gate_node.py",
    "tests/test_workflows/test_decision_gate_hitl.py",
    "tests/test_workflows/test_plan_trip_workflow.py",
    "tests/test_workflows/test_workflow_runtime.py",
    "tests/test_api/test_graph_response.py",
    "tests/test_api/test_travel_routes.py",

    "tests/test_services/test_commit_draft_service.py",
    "tests/test_nodes/test_commit_draft_node.py",
    "tests/test_nodes/test_cancel_node.py",
    "tests/test_revisions/test_revision_analyzer.py",
    "tests/test_nodes/test_revision_analyze_node.py",
    "tests/test_nodes/test_apply_revision_node.py",
    "tests/test_formatters/test_final_response_formatter.py",
    "tests/test_nodes/test_final_response_node.py",
]


def main() -> int:
    """运行当前已经完成的节点、服务、工具和数据库测试。"""

    command = [
        sys.executable,
        "-m",
        "pytest",
        *TEST_TARGETS,
        "-q",
    ]

    print("Running:")
    print(" ".join(command))

    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
