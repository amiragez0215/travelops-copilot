from __future__ import annotations

import json
from typing import Any, Protocol


class StructuredJSONClient(Protocol):
    """
    结构化 JSON 模型客户端协议。

    Extractor 只依赖这个协议，不绑定 DeepSeek 或 OpenAI SDK。
    单元测试可以注入 FakeStructuredJSONClient，避免真实网络请求。
    """

    model_id: str

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        """调用模型并返回已经解析的 JSON object。"""
        ...


class NativeToolCallingClient(Protocol):
    """
    原生 Tool Calling 客户端协议。

    与 generate_json 不同，这个协议要求模型真正返回 API 的 tool_calls，
    使测试能够区分“普通 JSON 规划”与真实 Function/Tool Calling。
    """

    model_id: str

    def generate_tool_calls(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """返回标准化后的 call_id、tool_name 和 arguments。"""
        ...


class DeepSeekJSONClient:
    """
    基于 DeepSeek OpenAI-compatible Chat Completions 的 JSON 客户端。

    这个类只负责：
        1. 调用模型。
        2. 要求模型输出 JSON object。
        3. 解析 JSON 字符串。

    它不负责：
        - 旅行字段业务校验。
        - 日期一致性校验。
        - 判断目的地是否有 Mock 数据。

    上述职责分别由 Pydantic Schema 和 Validator 完成。
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_id: str,
        temperature: float = 0.1,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        thinking_enabled: bool = False,
    ) -> None:
        """
        Args:
            api_key:
                DeepSeek API Key。

            base_url:
                DeepSeek OpenAI-compatible API 地址。

            model_id:
                .env 中配置的模型名称。

            temperature:
                非思考模式下的生成温度；抽取任务建议使用低温度。

            timeout_seconds:
                单次 API 请求超时时间。

            max_retries:
                OpenAI SDK 在连接或服务异常时的重试次数。

            thinking_enabled:
                是否开启思考模式。结构化抽取默认关闭，减少延迟和输出不确定性。
        """

        if not api_key.strip():
            raise ValueError("DEEPSEEK_API_KEY 不能为空")

        # 1. 延迟导入 OpenAI SDK。
        #    规则模式和单元测试不需要安装或初始化真实客户端。
        from openai import OpenAI

        self.model_id = model_id
        self.temperature = temperature
        self.thinking_enabled = thinking_enabled

        # 2. OpenAI-compatible SDK 统一处理认证、超时和重试。
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        """
        调用 DeepSeek JSON Output，并返回 dict。

        DeepSeek JSON 模式只能保证返回合法 JSON，
        不能保证字段完全符合业务 Schema；因此调用者仍必须使用 Pydantic 校验。
        """

        request_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "response_format": {
                "type": "json_object",
            },
            "max_tokens": max_tokens,
            "stream": False,
        }

        # 1. 思考模式和 temperature 分开处理。
        #    抽取任务默认关闭思考模式，并使用低 temperature。
        if self.thinking_enabled:
            request_kwargs["extra_body"] = {
                "thinking": {
                    "type": "enabled",
                }
            }
        else:
            request_kwargs["temperature"] = self.temperature
            request_kwargs["extra_body"] = {
                "thinking": {
                    "type": "disabled",
                }
            }

        # 2. 发起 OpenAI-compatible Chat Completion 请求。
        # self.client        → API 客户端
        # chat.completions   → 对话生成接口
        # create()           → 发起一次模型请求
        response = self.client.chat.completions.create(
            **request_kwargs
        )

        content = response.choices[0].message.content

        if not isinstance(content, str) or not content.strip():
            raise ValueError("DeepSeek 返回了空 JSON 内容")

        # 3. 清理模型偶尔附带的 Markdown 代码围栏。
        normalized = _strip_json_code_fence(content)

        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "DeepSeek 返回内容不是合法 JSON object"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError("DeepSeek Structured Output 根节点必须是 JSON object")

        return parsed

    def generate_tool_calls(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """
        使用 OpenAI-compatible 原生 tools 参数生成 Tool Call。

        当前只向模型暴露可选活动工具；天气、航班和酒店不在这里暴露，
        因为它们由应用 Policy 强制执行，不能被模型遗漏或改写参数。
        """

        request_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.thinking_enabled:
            request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            request_kwargs["temperature"] = self.temperature
            request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        response = self.client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message
        raw_calls = getattr(message, "tool_calls", None) or []
        normalized: list[dict[str, Any]] = []

        for index, raw_call in enumerate(raw_calls):
            function = getattr(raw_call, "function", None)
            tool_name = getattr(function, "name", None)
            raw_arguments = getattr(function, "arguments", None)
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            try:
                arguments = json.loads(raw_arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError("DeepSeek Tool Call arguments 不是合法 JSON") from exc
            if not isinstance(arguments, dict):
                raise ValueError("DeepSeek Tool Call arguments 必须是 JSON object")
            normalized.append(
                {
                    "call_id": str(getattr(raw_call, "id", None) or f"llm_call_{index + 1}"),
                    "tool_name": tool_name.strip(),
                    "arguments": arguments,
                }
            )

        return normalized


def _strip_json_code_fence(text: str) -> str:
    """
    去掉模型可能附带的 ```json ... ``` 外层代码围栏。

    正常 JSON Output 不应该产生围栏，但保留这个防御性处理可以提高健壮性。
    """

    normalized = text.strip()

    if normalized.startswith("```json"):
        normalized = normalized[len("```json") :]
    elif normalized.startswith("```"):
        normalized = normalized[len("```") :]

    if normalized.endswith("```"):
        normalized = normalized[: -len("```")]

    return normalized.strip()
