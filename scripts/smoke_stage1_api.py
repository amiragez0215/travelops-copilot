from __future__ import annotations

import argparse
import json
from typing import Any
from uuid import uuid4

import httpx


def _print_json(
    title: str,
    value: Any,
) -> None:
    """以易读格式打印一次 API 返回。"""

    print(f"\n=== {title} ===")
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    """
    通过真实 HTTP 接口验证阶段一闭环。

    使用前先启动服务：
        uvicorn app.main:app --reload

    示例：
        python scripts/smoke_stage1_api.py --decision approve

        python scripts/smoke_stage1_api.py \
            --decision request_changes \
            --change-request "预算增加1000元"

    这个脚本不会绕过 FastAPI 直接调用 Graph，
    因此可以同时验证：
        - invoke API；
        - interrupt payload；
        - thread_id；
        - Command(resume=...)；
        - approve / request_changes / cancel；
        - FinalResponse 或第二次 interrupt。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--user-id",
        default="user_001",
    )
    parser.add_argument(
        "--message",
        default=(
            "2026年7月2日从杭州去成都玩三天，"
            "预算4000，想住安静一点。"
        ),
    )
    parser.add_argument(
        "--reference-date",
        default="2026-07-01",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
    )
    parser.add_argument(
        "--decision",
        choices=[
            "approve",
            "request_changes",
            "cancel",
        ],
        default=None,
    )
    parser.add_argument(
        "--change-request",
        default=None,
    )
    parser.add_argument(
        "--reason",
        default=None,
    )
    args = parser.parse_args()

    thread_id = (
        args.thread_id
        or f"smoke_{uuid4().hex[:12]}"
    )

    with httpx.Client(
        base_url=args.base_url,
        timeout=300.0,
    ) as client:
        # 1. 通过 FastAPI 启动一条全新 Graph Thread。
        start_response = client.post(
            "/api/v1/travel/invoke",
            json={
                "user_id": args.user_id,
                "message": args.message,
                "reference_date": (
                    args.reference_date
                ),
                "trip_session_id": thread_id,
            },
        )
        start_response.raise_for_status()
        start_body = start_response.json()
        _print_json(
            "Invoke Response",
            start_body,
        )

        # 2. 没有要求自动提交决定时，到这里结束。
        #    你可以复制 response 中的 thread_id、proposal_id 和 action_id，
        #    再用 Postman 或 curl 手工恢复。
        if args.decision is None:
            return

        if (
            start_body.get("run_status")
            != "waiting_for_decision"
        ):
            raise RuntimeError(
                "当前 Graph 没有停在 DecisionGate，"
                "不能提交人工决定。"
            )

        interrupt_payload = (
            start_body.get("interrupt")
            or {}
        )

        decision_payload: dict[str, Any] = {
            "decision": args.decision,
            "proposal_id": (
                interrupt_payload[
                    "proposal_id"
                ]
            ),
            "action_id": (
                interrupt_payload["action_id"]
            ),
        }

        if args.change_request:
            decision_payload[
                "change_request"
            ] = args.change_request

        if args.reason:
            decision_payload["reason"] = (
                args.reason
            )

        # 3. 使用同一个 thread_id 提交 Command(resume=...)。
        decision_response = client.post(
            "/api/v1/travel/decision/"
            f"{thread_id}",
            json=decision_payload,
        )
        decision_response.raise_for_status()
        decision_body = (
            decision_response.json()
        )
        _print_json(
            "Decision Resume Response",
            decision_body,
        )

        # 4. 再通过 State API 读取 Checkpointer 中的最新权威状态。
        state_response = client.get(
            "/api/v1/travel/runs/"
            f"{thread_id}",
            params={
                "include_trace": "true"
            },
        )
        state_response.raise_for_status()
        _print_json(
            "Latest Thread State",
            state_response.json(),
        )


if __name__ == "__main__":
    main()
