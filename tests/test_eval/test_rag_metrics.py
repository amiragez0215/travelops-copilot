from __future__ import annotations

import pytest

from eval.rag_metrics import (
    average,
    grounded_citation_rate,
    hit_at_k,
    metadata_accuracy,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_rank_metrics_distinguish_recall_precision_and_first_rank():
    ranked = ["noise", "gold_a", "gold_b"]
    gold = ["gold_a", "gold_b"]

    assert hit_at_k(ranked, gold, 1) == 0.0
    assert hit_at_k(ranked, gold, 2) == 1.0
    assert recall_at_k(ranked, gold, 2) == 0.5
    assert precision_at_k(ranked, gold, 2) == 0.5
    assert reciprocal_rank_at_k(ranked, gold, 3) == 0.5


def test_ndcg_rewards_putting_highest_grade_first():
    relevance = {"strong": 2, "weak": 1}

    assert ndcg_at_k(["strong", "weak"], relevance, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["weak", "strong"], relevance, 2) < 1.0


def test_negative_cases_are_not_folded_into_positive_ranking_metrics():
    assert hit_at_k([], [], 5) is None
    assert recall_at_k([], [], 5) is None
    assert precision_at_k([], [], 5) is None
    assert reciprocal_rank_at_k([], [], 5) is None
    assert ndcg_at_k([], {}, 5) is None


@pytest.mark.parametrize(
    "metric,args",
    [
        (hit_at_k, ([], [], 0)),
        (recall_at_k, ([], [], 0)),
        (precision_at_k, ([], [], 0)),
        (reciprocal_rank_at_k, ([], [], 0)),
        (ndcg_at_k, ([], {}, 0)),
    ],
)
def test_rank_metrics_reject_non_positive_k_even_for_negative_cases(metric, args):
    with pytest.raises(ValueError, match="k must be greater than 0"):
        metric(*args)


def test_metadata_accuracy_supports_hotel_ids_alias_and_negative_cases():
    result = [{"city": "成都", "doc_type": "hotel_reviews", "hotel_id": "hotel_001"}]
    assert metadata_accuracy(
        result,
        {"city": "成都", "doc_type": "hotel_reviews", "hotel_ids": ["hotel_001"]},
    ) == 1.0
    assert metadata_accuracy([], {"city": "上海"}, expected_no_results=True) == 1.0
    assert metadata_accuracy([], {"city": "成都"}) == 0.0


def test_grounded_citation_rate_and_summary_helpers():
    assert grounded_citation_rate(["chunk_a", "chunk_b"], ["chunk_a"]) == 0.5
    assert grounded_citation_rate([], ["chunk_a"]) is None
    assert average([1.0, None, 0.0]) == 0.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)


def test_percentile_rejects_invalid_range_even_when_values_are_empty():
    with pytest.raises(
        ValueError,
        match="percentile_value must be between 0 and 100",
    ):
        percentile([], 101)
