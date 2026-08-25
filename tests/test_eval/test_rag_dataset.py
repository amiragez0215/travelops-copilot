from __future__ import annotations

import json
from pathlib import Path

from app.rag.document_loader import load_rag_documents
from app.rag.bm25_index import InMemoryBM25Index
from app.rag.hybrid_retriever import HybridRetriever, HybridStartupReport
from eval.rag_eval import load_cases, validate_cases_against_corpus


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_gold_dataset_has_required_coverage_and_unique_case_ids():
    cases = load_cases(PROJECT_ROOT / "eval" / "rag_cases.jsonl")

    # This is a scope-regression set, not a semantic leaderboard. Its safety
    # cases intentionally group all evidence for one query under multi-Gold.
    assert len(cases) == 108
    assert sum(not case.expected_no_results for case in cases) == 84
    assert sum(case.expected_no_results for case in cases) == 24
    assert len({case.case_id for case in cases}) == len(cases)
    assert {
        "guides",
        "hotel_reviews",
        "safety_notices",
        "packing_checklists",
    }.issubset({case.category for case in cases})
    assert {"北京", "上海", "杭州"}.issubset(
        {str(case.metadata_filter.get("city")) for case in cases}
    )
    safety_cases = [case for case in cases if case.category == "safety_notices"]
    assert len(safety_cases) == 12
    assert all(len(case.gold_chunk_ids) == 2 for case in safety_cases)
    signatures = [
        (
            case.semantic_query,
            case.keyword_query,
            json.dumps(case.metadata_filter, ensure_ascii=False, sort_keys=True),
        )
        for case in cases
        if not case.expected_no_results
    ]
    assert len(signatures) == len(set(signatures))


def test_gold_chunk_ids_match_the_current_corpus_without_loading_models():
    documents = load_rag_documents(PROJECT_ROOT / "data" / "rag_docs")
    retriever = HybridRetriever(
        documents=documents,
        bm25_index=InMemoryBM25Index(documents),
        vector_store=None,  # type: ignore[arg-type]
        rag_docs_dir=PROJECT_ROOT / "data" / "rag_docs",
        startup_report=HybridStartupReport(0, len(documents), 0, 1, {}),
    )

    validate_cases_against_corpus(
        load_cases(PROJECT_ROOT / "eval" / "rag_cases.jsonl"),
        retriever,
    )


def test_semantic_holdout_has_need_level_groups_and_semantic_decoys():
    """Semantic Holdout 必须真正覆盖多意图，而不是 Scope Set 的重复。"""

    cases = load_cases(PROJECT_ROOT / "eval" / "rag_challenge_cases.jsonl")

    assert len(cases) == 42
    assert sum(not case.expected_no_results for case in cases) == 39
    assert sum(case.expected_no_results for case in cases) == 3
    assert sum(case.query_mode == "combined" for case in cases) == 9
    assert sum(case.query_mode == "split_need" for case in cases) == 18
    assert sum(case.query_mode == "implicit" for case in cases) == 3
    assert all(case.challenge_group for case in cases)
    assert all(case.need_ids for case in cases)
    assert any(len(case.gold_chunk_ids) >= 3 for case in cases)
    assert any(case.decoy_chunk_ids for case in cases)


def test_agentic_challenge_gold_and_decoy_ids_match_current_corpus():
    """Gold 和干扰 chunk 都必须是可追溯的真实语料。"""

    documents = load_rag_documents(PROJECT_ROOT / "data" / "rag_docs")
    retriever = HybridRetriever(
        documents=documents,
        bm25_index=InMemoryBM25Index(documents),
        vector_store=None,  # type: ignore[arg-type]
        rag_docs_dir=PROJECT_ROOT / "data" / "rag_docs",
        startup_report=HybridStartupReport(0, len(documents), 0, 1, {}),
    )

    validate_cases_against_corpus(
        load_cases(PROJECT_ROOT / "eval" / "rag_challenge_cases.jsonl"),
        retriever,
    )
    assert len(documents) == 222
