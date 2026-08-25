from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from statistics import mean
from typing import Any


def hit_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float | None:
    """Return 1 when one relevant item appears in the first ``k`` results."""

    _positive_k(k)
    relevant = _normalized_set(relevant_ids)
    if not relevant:
        return None
    return float(bool(relevant.intersection(_top_k(ranked_ids, k))))


def recall_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float | None:
    """Measure the fraction of all relevant chunks retrieved in the first ``k``."""

    _positive_k(k)
    relevant = _normalized_set(relevant_ids)
    if not relevant:
        return None
    return len(relevant.intersection(_top_k(ranked_ids, k))) / len(relevant)


def precision_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float | None:
    """Measure the fraction of the first ``k`` ranks that are relevant."""

    _positive_k(k)
    relevant = _normalized_set(relevant_ids)
    if not relevant:
        return None
    return len(relevant.intersection(_top_k(ranked_ids, k))) / _positive_k(k)


def reciprocal_rank_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float | None:
    """Return the reciprocal rank of the first relevant result."""

    _positive_k(k)
    relevant = _normalized_set(relevant_ids)
    if not relevant:
        return None
    for rank, item_id in enumerate(_top_k(ranked_ids, k), start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, int | float],
    k: int,
) -> float | None:
    """Compute normalized DCG for graded relevance labels."""

    _positive_k(k)
    normalized = {
        str(item_id).strip().lower(): max(float(grade), 0.0)
        for item_id, grade in relevance.items()
        if str(item_id).strip()
    }
    positive_grades = [grade for grade in normalized.values() if grade > 0]
    if not positive_grades:
        return None

    actual_grades = [
        normalized.get(item_id, 0.0)
        for item_id in _top_k(ranked_ids, k)
    ]
    ideal_grades = sorted(positive_grades, reverse=True)[:_positive_k(k)]
    ideal_dcg = _dcg(ideal_grades)
    if ideal_dcg == 0:
        return None
    return _dcg(actual_grades) / ideal_dcg


def metadata_accuracy(
    result_metadata: Sequence[Mapping[str, Any]],
    metadata_filter: Mapping[str, Any],
    *,
    expected_no_results: bool = False,
) -> float:
    """Measure whether returned results obey the requested metadata filter."""

    if not result_metadata:
        return 1.0 if expected_no_results else 0.0
    matched = sum(
        1
        for metadata in result_metadata
        if _metadata_matches(metadata, metadata_filter)
    )
    return matched / len(result_metadata)


def grounded_citation_rate(
    cited_chunk_ids: Iterable[str],
    evidence_chunk_ids: Iterable[str],
) -> float | None:
    """Measure proposal citations that can be traced to their evidence pool."""

    citations = list(dict.fromkeys(_normalized_values(cited_chunk_ids)))
    if not citations:
        return None
    evidence = _normalized_set(evidence_chunk_ids)
    return sum(item_id in evidence for item_id in citations) / len(citations)


def average(values: Iterable[float | None]) -> float | None:
    """Average numeric values while treating ``None`` as not applicable."""

    numeric = [float(value) for value in values if value is not None]
    return mean(numeric) if numeric else None


def percentile(values: Iterable[float], percentile_value: float) -> float | None:
    """Compute a linearly interpolated percentile without third-party packages."""

    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile_value must be between 0 and 100")
    numeric = sorted(float(value) for value in values)
    if not numeric:
        return None
    if len(numeric) == 1:
        return numeric[0]

    position = (len(numeric) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return numeric[lower]
    return numeric[lower] + (numeric[upper] - numeric[lower]) * (position - lower)


def _dcg(grades: Sequence[float]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def _metadata_matches(
    metadata: Mapping[str, Any],
    expected_filter: Mapping[str, Any],
) -> bool:
    for key, expected in expected_filter.items():
        actual = metadata.get("hotel_id") if key == "hotel_ids" else metadata.get(key)
        expected_values = _normalized_set(
            expected if isinstance(expected, (list, tuple, set)) else [expected]
        )
        actual_values = _normalized_set(
            actual if isinstance(actual, (list, tuple, set)) else [actual]
        )
        if expected_values and not expected_values.intersection(actual_values):
            return False
    return True


def _top_k(ranked_ids: Sequence[str], k: int) -> list[str]:
    return list(dict.fromkeys(_normalized_values(ranked_ids)))[:_positive_k(k)]


def _positive_k(k: int) -> int:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    return k


def _normalized_set(values: Iterable[Any]) -> set[str]:
    return set(_normalized_values(values))


def _normalized_values(values: Iterable[Any]) -> list[str]:
    return [str(value).strip().lower() for value in values if str(value).strip()]
