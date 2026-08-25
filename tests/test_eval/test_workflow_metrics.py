from __future__ import annotations

from eval.workflow_metrics import (
    latency_summary,
    ordered_path_contains,
    set_precision,
    set_recall,
)


def test_ordered_path_requires_the_expected_sequence_not_just_membership():
    actual = ["input_extract", "weather", "flight_search", "hotel_search"]

    assert ordered_path_contains(actual, ["weather", "hotel_search"]) == 1.0
    assert ordered_path_contains(actual, ["hotel_search", "weather"]) == 0.0


def test_tool_precision_recall_handles_correct_no_tool_paths():
    assert set_precision([], []) == 1.0
    assert set_recall([], []) == 1.0
    assert set_precision(["get_weather", "search_hotels"], ["get_weather"]) == 0.5
    assert set_recall(["get_weather"], ["get_weather", "search_hotels"]) == 0.5


def test_latency_summary_reports_mean_and_p95():
    summary = latency_summary([1.0, 2.0, 3.0, 4.0])

    assert summary["mean_latency_ms"] == 2.5
    assert summary["p95_latency_ms"] == 3.85
