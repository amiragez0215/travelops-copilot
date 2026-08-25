from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any, Mapping

from app.extractors.trip_request_extractor import (
    RuleBasedTripRequestExtractor,
    TripRequestExtractor,
)
from app.llm.deepseek_json_client import (
    DeepSeekJSONClient,
    StructuredJSONClient,
)
from app.schemas.preference_schema import PreferenceSignal
from app.schemas.trip_request_schema import (
    TripRequest,
    TripRequestDraft,
)


class LLMTripRequestExtractor:
    """
    使用 DeepSeek JSON Output 理解开放自然语言的旅行需求抽取器。

    LLM 适合处理：
        - 间接目的地：“兵马俑所在的城市” → 西安。
        - 消费倾向：“住宿好一点，航班省一点”。
        - 指定实体：“想住某酒店”“想吃鸭血粉丝汤”。
        - 复杂否定、条件、歧义和无法映射需求。

    LLM 不负责提供：
        - 实时天气、航班、酒店价格和房态。
        - Mock 数据中不存在的酒店或航班。
        - 用户没有说出的具体日期。
    """

    def __init__(
        self,
        client: StructuredJSONClient,
        max_tokens: int = 4096,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens

    def extract(
        self,
        message: str,
        user_id: str | None = None,
        reference_date: date | None = None,
    ) -> TripRequest:
        """
        调用 LLM 生成 TripRequestDraft，再转成完整 TripRequest。

        Pydantic 会校验日期、数字范围、枚举值和嵌套结构。
        更复杂的业务一致性由 TripRequestValidator 继续检查。
        """

        reference_date = reference_date or date.today()
        schema = _draft_json_schema()

        # 1. System Prompt 明确模型的知识边界和硬约束边界。
        system_prompt = _build_system_prompt(schema)

        # 2. User Prompt 提供参考日期，避免“明天、7月”缺少解析基准。
        user_prompt = (
            "请把下面的旅行需求解析为 JSON。\n"
            f"参考日期：{reference_date.isoformat()}\n"
            f"用户输入：{message.strip()}"
        )

        # 3. 调用 JSON 客户端。JSON 合法并不代表业务合法，下一步还要 Pydantic。
        payload = self.client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens,
        )

        # 4. Pydantic 将模型 JSON 校验为稳定的 TripRequestDraft。
        draft = TripRequestDraft(**payload)

        # 5. 根据明确日期与 days 补齐可确定推导的字段。
        start_date, end_date, days = _normalize_date_fields(
            start_date=draft.start_date,
            end_date=draft.end_date,
            days=draft.days,
        )

        people_count = draft.people_count or 1

        # 6. 单人默认一间房；多人未说明房间数时保持 None，交给 Validator 澄清。
        room_count = (
            draft.room_count
            if draft.room_count is not None
            else 1
            if people_count == 1
            else None
        )

        preference_signals = _dedupe_preference_signals(
            draft.preference_signals
        )

        (
            transport_preferences,
            hotel_preferences,
            travel_style,
            raw_constraints,
        ) = _derive_preference_lists(preference_signals)

        raw_constraints = _dedupe_strings(
            [
                *raw_constraints,
                *[
                    item.evidence
                    for item in draft.spend_preferences
                    if item.evidence
                ],
                *[
                    item.evidence
                    for item in draft.hard_constraints
                    if item.evidence
                ],
                *[
                    item.evidence
                    for item in draft.named_constraints
                    if item.evidence
                ],
            ]
        )

        trip_request = TripRequest(
            user_id=user_id,
            origin=_clean_optional_text(draft.origin),
            destination=_clean_optional_text(draft.destination),
            start_date=start_date,
            end_date=end_date,
            partial_date=draft.partial_date,
            days=days,
            budget=draft.budget,
            people_count=people_count,
            room_count=room_count,
            transport_preferences=transport_preferences,
            hotel_preferences=hotel_preferences,
            travel_style=travel_style,
            raw_constraints=raw_constraints,
            preference_signals=preference_signals,
            spend_preferences=draft.spend_preferences,
            hard_constraints=draft.hard_constraints,
            named_constraints=draft.named_constraints,
            unmapped_requirements=draft.unmapped_requirements,
            requirement_issues=draft.requirement_issues,
            field_resolution=draft.field_resolution,
            raw_message=message.strip(),
            extraction={
                "method": "llm_json_v1",
                "model": self.client.model_id,
                "confidence": draft.extraction_confidence,
                "llm_used": True,
                "fallback_used": False,
                "notes": [
                    "LLM 负责开放自然语言理解，结构由 Pydantic 校验。",
                    "动态事实仍必须来自 MCP Mock Provider。",
                ],
            },
        )

        trip_request.extraction[
            "missing_fields"
        ] = trip_request.required_missing_fields()

        return trip_request


class HybridTripRequestExtractor:
    """
    最终 InputExtract 使用的 LLM-first 抽取器。

    工作顺序：
        1. 正常路径只调用 LLM，避免规则和模型重复理解同一段用户意图。
        2. LLM 输出继续经过 Pydantic、日期归一化和后续业务 Validator。
        3. 只有模型不可用或结构化输出失败时，才调用规则抽取器降级。

    RuleBasedTripRequestExtractor 仍然保留，但职责仅是：
        - 无 API Key 时提供离线能力；
        - LLM 超时、JSON/Pydantic 失败时保证工作流可以继续澄清；
        - 为 rule 模式和确定性测试提供 baseline。

    这个类保留原名称以兼容现有导入；它不再合并两套语义结果。

    最终链路：
        LLM Structured Output
        → Pydantic Schema
        → 确定性字段归一化
        → MissingInfo / Validator

        LLM failure
        → Rule-based Fallback
    """

    def __init__(
        self,
        *,
        rule_extractor: RuleBasedTripRequestExtractor | None = None,
        llm_extractor: LLMTripRequestExtractor | None = None,
    ) -> None:
        self.rule_extractor = rule_extractor or RuleBasedTripRequestExtractor()
        self.llm_extractor = llm_extractor

    def extract(
        self,
        message: str,
        user_id: str | None = None,
        reference_date: date | None = None,
    ) -> TripRequest:
        """优先执行 LLM；只有 LLM 不可用或失败时才执行规则 fallback。"""

        # 没有配置模型客户端时无需先尝试一次必然失败的模型调用。
        if self.llm_extractor is None:
            rule_request = self.rule_extractor.extract(
                message=message,
                user_id=user_id,
                reference_date=reference_date,
            )
            return _mark_rule_fallback(
                rule_request,
                reason="未配置 DEEPSEEK_API_KEY 或 INPUT_EXTRACT_MODE=rule。",
            )

        try:
            # 正常路径只信任一份结构化语义结果。日期、数字关系和能力范围
            # 仍会在 LLMTripRequestExtractor 及后续 Validator 中确定性校验。
            return self.llm_extractor.extract(
                message=message,
                user_id=user_id,
                reference_date=reference_date,
            )
        except Exception as exc:
            # API 超时、JSON 解析或 Pydantic 失败时才运行规则抽取。
            # fallback 的目标是保持可用，并不声称拥有与 LLM 相同的理解能力。
            rule_request = self.rule_extractor.extract(
                message=message,
                user_id=user_id,
                reference_date=reference_date,
            )
            return _mark_rule_fallback(
                rule_request,
                reason=f"LLM 抽取失败：{exc}",
            )


def build_default_trip_request_extractor() -> TripRequestExtractor:
    """
    根据 .env 创建 InputExtractNode 默认使用的抽取器。

    INPUT_EXTRACT_MODE：
        rule
            只使用规则，适合离线测试。

        hybrid
            兼容旧配置；行为与 llm 相同，使用 LLM-first + Rule fallback。

        llm
            默认模式；LLM 成功时直接使用模型结果，失败时规则 fallback。
    """

    # 1. 延迟导入 Settings，避免测试模块导入时就初始化外部客户端。
    from app.common.config import settings

    rule_extractor = RuleBasedTripRequestExtractor()
    mode = settings.input_extract_mode.strip().lower()

    if mode == "rule":
        return rule_extractor

    llm_extractor: LLMTripRequestExtractor | None = None

    # 2. 只有配置了 Key 才创建 DeepSeek Client。
    if settings.deepseek_api_key:
        client = DeepSeekJSONClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.llm_base_url,
            model_id=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            thinking_enabled=settings.llm_thinking_enabled,
        )

        llm_extractor = LLMTripRequestExtractor(
            client=client,
            max_tokens=settings.input_extract_max_tokens,
        )

    return HybridTripRequestExtractor(
        rule_extractor=rule_extractor,
        llm_extractor=llm_extractor,
    )


def _mark_rule_fallback(
    request: TripRequest,
    *,
    reason: str,
) -> TripRequest:
    """标记本次抽取使用了规则 fallback。"""

    metadata = dict(request.extraction)
    notes = list(metadata.get("notes") or [])
    notes.append(reason)

    metadata.update(
        {
            "method": "rule_fallback_v2",
            "llm_used": False,
            "fallback_used": True,
            "notes": notes,
            "missing_fields": request.required_missing_fields(),
        }
    )

    request.extraction = metadata
    return request


def _normalize_date_fields(
    *,
    start_date: date | None,
    end_date: date | None,
    days: int | None,
) -> tuple[date | None, date | None, int | None]:
    """使用确定性规则补齐可推导的日期和天数。"""

    if start_date and end_date and end_date >= start_date and days is None:
        days = (end_date - start_date).days + 1

    if start_date and days and end_date is None:
        end_date = start_date + timedelta(days=days - 1)

    if end_date and days and start_date is None:
        start_date = end_date - timedelta(days=days - 1)

    return start_date, end_date, days


def _dedupe_preference_signals(
    signals: list[PreferenceSignal],
) -> list[PreferenceSignal]:
    """LLM 单独输出中同一 domain.key 只保留最高强度。"""

    best: dict[tuple[str, str], PreferenceSignal] = {}
    for signal in signals:
        key = (signal.domain, signal.key)
        old = best.get(key)
        if old is None or signal.strength > old.strength:
            best[key] = signal
    return list(best.values())


def _derive_preference_lists(
    preference_signals: list[PreferenceSignal],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """从权威 preference_signals 派生兼容性的轻量 list 字段。"""

    transport_preferences: list[str] = []
    hotel_preferences: list[str] = []
    travel_style: list[str] = []
    evidence: list[str] = []

    for signal in preference_signals:
        if signal.domain == "flight":
            transport_preferences.append(signal.key)
        elif signal.domain == "hotel":
            hotel_preferences.append(signal.key)
        elif signal.domain in {"activity", "pace"}:
            travel_style.append(signal.key)

        if signal.evidence:
            evidence.append(signal.evidence)

    return (
        _dedupe_strings(transport_preferences),
        _dedupe_strings(hotel_preferences),
        _dedupe_strings(travel_style),
        _dedupe_strings(evidence),
    )


def _draft_json_schema() -> dict[str, Any]:
    """获取 TripRequestDraft JSON Schema，兼容 Pydantic v1/v2。"""

    if hasattr(TripRequestDraft, "model_json_schema"):
        return TripRequestDraft.model_json_schema()

    return TripRequestDraft.schema()


def _build_system_prompt(schema: Mapping[str, Any]) -> str:
    """构造 DeepSeek 旅行需求抽取 Prompt。"""

    return f"""
你是 TravelOps-Copilot 的旅行需求抽取器。
你必须只输出一个合法 JSON object，不要输出 Markdown，不要输出解释。
输出必须尽量符合下面的 JSON Schema：
{json.dumps(schema, ensure_ascii=False)}

抽取规则：
1. 可以使用稳定的通用世界知识解析无歧义的间接目的地。
   例如“秦始皇兵马俑所在的城市”可以解析为“西安”。
2. 有多个合理答案时不要猜，例如“迪士尼所在的城市”；destination 设为 null，
   并在 requirement_issues 中添加 ambiguous_destination。
3. 不得编造实时天气、酒店、航班、价格、房态或营业状态。
4. 用户只说“7月出发”时，start_date 必须为 null，并写 partial_date.month=7。
5. “一定要安静”“必须干净”属于高强度主观偏好，写入 preference_signals，
   strength 可以为 1.0，但不能写入 hard_constraints。
6. hard_constraints 只允许客观可验证条件，例如必须直飞、不能早于某时间、
   酒店每晚不能超过某金额、指定明确酒店或航班。字段只能使用下列规范名称：
   - flight：is_direct、depart_time、arrive_time、price、flight_no；
   - hotel：price_per_night、hotel_id、near_subway；
   - trip：total_budget。
7. 用户说“预算 / 总预算 / 不超过 X 元 / 控制在 X 元以内”时，只填写顶层
   budget=X；不要额外生成 hard_constraints 中的 trip.budget 或
   trip.total_budget。程序会把顶层 budget 确定性转换成总预算硬上限。
8. 用户希望住宿、交通、餐饮活动多花或少花预算时，写入 spend_preferences。
9. 指定酒店、航班、食物、景点或活动时，写入 named_constraints。
10. 理解了但当前字段无法承载的要求必须写入 unmapped_requirements，不能丢弃。
11. 多人提出相互冲突的酒店或偏好时，写入 requirement_issues 并要求澄清。
12. field_resolution 说明 origin、destination、start_date 等字段来自 explicit、derived、
    stable_world_knowledge、ambiguous、incomplete 或 default。
13. 尽量把负向表达标准化成正向 avoid key。
    例如“不想坐早班机”应输出 key=avoid_early_flight、polarity=positive。

示例输入：
“我想去秦始皇兵马俑所在的城市玩3天，7月出发，住宿好一点，航班不用很贵。”

示例 JSON 重点：
{{
  "destination": "西安",
  "start_date": null,
  "partial_date": {{
    "year": null,
    "month": 7,
    "day": null,
    "precision": "month",
    "evidence": "7月出发"
  }},
  "days": 3,
  "spend_preferences": [
    {{
      "category": "hotel",
      "direction": "increase",
      "strength": 0.9,
      "evidence": "住宿好一点",
      "source": "llm"
    }},
    {{
      "category": "transport",
      "direction": "decrease",
      "strength": 0.8,
      "evidence": "航班不用很贵",
      "source": "llm"
    }}
  ],
  "field_resolution": {{
    "destination": {{
      "value": "西安",
      "resolution_type": "stable_world_knowledge",
      "evidence": "秦始皇兵马俑所在的城市",
      "needs_clarification": false
    }},
    "start_date": {{
      "value": null,
      "resolution_type": "incomplete",
      "evidence": "7月出发",
      "needs_clarification": true
    }}
  }}
}}
""".strip()


def _clean_optional_text(value: str | None) -> str | None:
    """去掉字符串两端空白；空字符串转成 None。"""

    if value is None:
        return None

    normalized = value.strip()
    return normalized or None


def _dedupe_strings(items: list[str]) -> list[str]:
    """保持第一次出现顺序的字符串去重。"""

    return list(dict.fromkeys(item for item in items if item))
