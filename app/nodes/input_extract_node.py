from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from app.agents.state import TravelState
from app.extractors.hybrid_trip_request_extractor import (
    build_default_trip_request_extractor,
)
from app.extractors.trip_request_extractor import (
    TripRequestExtractor,
    coerce_reference_date,
)
from app.schemas.trip_request_schema import TripRequest


def input_extract_node(
    state: TravelState,
    extractor: TripRequestExtractor | None = None,
) -> dict[str, Any]:
    """
    InputExtractNode：把用户自然语言输入抽取成结构化 trip_request。
    """

    node_name = "input_extract"
    started_at = perf_counter()
    # 1. 测试可以注入 Fake/Rule Extractor；正常运行根据 .env 选择 rule/hybrid。
    extractor = extractor or build_default_trip_request_extractor()

    # 当前 State 只使用 raw_message 作为用户输入字段。
    raw_message = str(state.get("raw_message") or "").strip()

    # user_id 可以为空，匿名用户后面会走默认画像。
    user_id = state.get("user_id")
    user_id = str(user_id) if user_id is not None else None

    # reference_date 用于解析“明天、后天、7月2日”。
    reference_date = coerce_reference_date(state.get("reference_date"))

    try:
        if not raw_message:
            raise ValueError("raw_message 不能为空")

        # 调用 extractor，Node 不关心具体抽取规则。
        trip_request = extractor.extract(
            message=raw_message,
            user_id=user_id,
            reference_date=reference_date,
        )

        trip_request_dict = trip_request.to_state_dict()

        trace_item = _build_trace_item(
            node_name=node_name,
            status="success",
            started_at=started_at,
            input_summary=_shorten(raw_message),
            output_summary=(
                f"origin={trip_request.origin}, "
                f"destination={trip_request.destination}, "
                f"start_date={trip_request_dict.get('start_date')}, "
                f"days={trip_request.days}, "
                f"budget={trip_request.budget}, "
                f"method={trip_request.extraction.get('method')}"
            ),
        )

        return {
            "raw_message": raw_message,
            "trip_request": trip_request_dict,
            "trace": [trace_item],
        }

    except Exception as exc:
        # 抽取失败时仍返回 fallback trip_request，便于 FinalResponse 和 trace 排查。
        fallback_request = TripRequest(
            user_id=user_id,
            raw_message=raw_message,
            extraction={
                "method": "input_extract_failed",
                "confidence": 0.0,
                "error": str(exc),
            },
        )

        error_item = {
            "node": node_name,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "raw_message": raw_message,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        trace_item = _build_trace_item(
            node_name=node_name,
            status="failed",
            started_at=started_at,
            input_summary=_shorten(raw_message),
            output_summary=str(exc),
        )

        return {
            "raw_message": raw_message,
            "trip_request": fallback_request.to_state_dict(),
            "errors": [error_item],
            "trace": [trace_item],
        }


def _build_trace_item(
    node_name: str,
    status: str,
    started_at: float,
    input_summary: str,
    output_summary: str,
) -> dict[str, Any]:
    """
    生成统一 trace item。
    """

    latency_ms = int((perf_counter() - started_at) * 1000)

    return {
        "node_name": node_name,
        "tool_name": None,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "latency_ms": latency_ms,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _shorten(text: str, max_len: int = 100) -> str:
    """
    截断过长文本，避免 trace 太大。
    """

    if len(text) <= max_len:
        return text

    return text[: max_len - 3] + "..."