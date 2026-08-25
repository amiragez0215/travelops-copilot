from __future__ import annotations

from collections.abc import Iterable, Sequence

from eval.rag_metrics import average, percentile


def ordered_path_contains(actual: Sequence[str], expected: Sequence[str]) -> float:
    """Return 1 when ``expected`` is an ordered subsequence of ``actual``."""

    cursor = 0
    for item in actual:
        if cursor < len(expected) and item == expected[cursor]:
            cursor += 1
    return float(cursor == len(expected))


def set_precision(actual: Iterable[str], expected: Iterable[str]) -> float | None:
    """Precision for expected tool calls; empty/empty is a correct no-tool path."""

    actual_set = set(actual)
    expected_set = set(expected)
    if not actual_set and not expected_set:
        return 1.0
    if not actual_set:
        return 0.0
    return len(actual_set & expected_set) / len(actual_set)


def set_recall(actual: Iterable[str], expected: Iterable[str]) -> float | None:
    """Recall for expected tool calls; empty/empty is a correct no-tool path."""

    actual_set = set(actual)
    expected_set = set(expected)
    if not actual_set and not expected_set:
        return 1.0
    if not expected_set:
        return 0.0
    return len(actual_set & expected_set) / len(expected_set)


def boolean_rate(values: Iterable[bool | float]) -> float | None:
    """Average pass rate for boolean or 0/1 observations."""

    return average(float(value) for value in values)


def latency_summary(values: Iterable[float]) -> dict[str, float | None]:
    """Return mean and P95 latency in milliseconds."""

    numeric = list(values)
    return {
        "mean_latency_ms": average(numeric),
        "p95_latency_ms": percentile(numeric, 95),
    }
