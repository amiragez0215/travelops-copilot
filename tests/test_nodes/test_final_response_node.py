from __future__ import annotations

from app.nodes.final_response_node import (
    final_response_node,
)


def test_final_response_node_builds_commit_response_and_trace():
    result = final_response_node(
        {
            "commit_result": {
                "status": "committed",
                "message": "保存成功。",
                "trip_id": "trip_001",
                "proposal_id": "proposal_001",
                "proposal_version": 1,
            },
            "trip_draft": {
                "trip_id": "trip_001",
                "destination": "成都",
            },
            "trip_version": {
                "version_number": 1
            },
        }
    )

    assert (
        result["final_response"]["status"]
        == "approved_simulated"
    )
    assert result["trace"][0]["node_name"] == "final_response"
    assert result["trace"][0]["status"] == "success"


def test_final_response_node_builds_cancel_response():
    result = final_response_node(
        {
            "cancel_result": {
                "status": "cancelled",
                "message": "已取消。",
                "proposal_id": "proposal_001",
            }
        }
    )

    assert result["final_response"]["status"] == "cancelled"
