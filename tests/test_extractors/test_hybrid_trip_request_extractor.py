from __future__ import annotations

from datetime import date
from typing import Any

from app.extractors.hybrid_trip_request_extractor import (
    HybridTripRequestExtractor,
    LLMTripRequestExtractor,
)
from app.extractors.trip_request_extractor import RuleBasedTripRequestExtractor


class FakeJSONClient:
    """返回预设 JSON 的离线模型客户端。"""

    model_id = "fake-deepseek"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0
        self.system_prompts: list[str] = []

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        self.calls += 1
        self.system_prompts.append(system_prompt)
        assert "稳定的通用世界知识" in system_prompt
        assert "参考日期" in user_prompt
        assert max_tokens > 0
        return self.payload


class FailingJSONClient(FakeJSONClient):
    """模拟 DeepSeek 超时或非法输出。"""

    def generate_json(self, **kwargs) -> dict[str, Any]:
        raise RuntimeError("模拟 LLM 服务失败")


class CountingRuleExtractor:
    """记录规则抽取调用次数，用于验证 LLM-first 路由。"""

    def __init__(self) -> None:
        self.calls = 0
        self.delegate = RuleBasedTripRequestExtractor()

    def extract(self, **kwargs):
        self.calls += 1
        return self.delegate.extract(**kwargs)


def test_llm_can_resolve_stable_world_knowledge_but_not_invent_day():
    """LLM 可以解析兵马俑→西安，但“7月出发”仍需澄清具体日期。"""

    client = FakeJSONClient(
        {
            "origin": "杭州",
            "destination": "西安",
            "start_date": None,
            "end_date": None,
            "partial_date": {
                "year": 2026,
                "month": 7,
                "day": None,
                "precision": "month",
                "evidence": "7月出发",
            },
            "days": 3,
            "people_count": 1,
            "preference_signals": [],
            "spend_preferences": [],
            "hard_constraints": [],
            "named_constraints": [
                {
                    "entity_type": "attraction",
                    "entity_name": "秦始皇兵马俑",
                    "constraint_mode": "preferred",
                    "scope": "itinerary",
                    "evidence": "秦始皇兵马俑所在的城市",
                    "source": "llm",
                }
            ],
            "field_resolution": {
                "destination": {
                    "value": "西安",
                    "resolution_type": "stable_world_knowledge",
                    "evidence": "秦始皇兵马俑所在的城市",
                    "needs_clarification": False,
                },
                "start_date": {
                    "value": None,
                    "resolution_type": "incomplete",
                    "evidence": "7月出发",
                    "needs_clarification": True,
                },
            },
            "extraction_confidence": 0.92,
        }
    )

    rule = CountingRuleExtractor()
    extractor = HybridTripRequestExtractor(
        rule_extractor=rule,
        llm_extractor=LLMTripRequestExtractor(client=client),
    )

    data = extractor.extract(
        message="我从杭州去秦始皇兵马俑所在的城市玩3天，7月出发",
        user_id="user_001",
        reference_date=date(2026, 6, 1),
    ).to_state_dict()

    assert data["destination"] == "西安"
    assert data["start_date"] is None
    assert data["partial_date"]["month"] == 7
    assert data["field_resolution"]["destination"]["resolution_type"] == (
        "stable_world_knowledge"
    )
    assert data["extraction"]["method"] == "llm_json_v1"
    assert client.calls == 1
    assert rule.calls == 0


def test_llm_success_is_not_merged_with_rule_result():
    """LLM 成功时直接返回模型结果，规则不在后台补字段。"""

    client = FakeJSONClient(
        {
            "origin": "杭州",
            "destination": "成都",
            "start_date": None,
            "days": 3,
            "people_count": 1,
            "preference_signals": [],
            "spend_preferences": [
                {
                    "category": "hotel",
                    "direction": "increase",
                    "strength": 0.9,
                    "evidence": "住宿安排好一点",
                    "source": "llm",
                }
            ],
            "hard_constraints": [],
            "named_constraints": [],
            "unmapped_requirements": [
                {
                    "text": "希望整个旅程像电影一样",
                    "category_guess": "experience_style",
                    "impact": "proposal_style",
                    "confidence": 0.7,
                    "requires_clarification": False,
                }
            ],
            "field_resolution": {},
            "extraction_confidence": 0.85,
        }
    )

    rule = CountingRuleExtractor()
    extractor = HybridTripRequestExtractor(
        rule_extractor=rule,
        llm_extractor=LLMTripRequestExtractor(client=client),
    )

    data = extractor.extract(
        message=(
            "2026年7月2日从杭州去成都玩3天，"
            "住宿安排好一点，希望整个旅程像电影一样"
        ),
        reference_date=date(2026, 7, 1),
    ).to_state_dict()

    # 这个 payload 故意没给日期；若仍在做 hybrid merge，
    # 规则会从 raw message 补出 2026-07-02。
    assert data["start_date"] is None
    assert data["end_date"] is None
    assert data["spend_preferences"][0]["category"] == "hotel"
    assert data["unmapped_requirements"][0]["impact"] == "proposal_style"
    assert data["extraction"]["method"] == "llm_json_v1"
    assert rule.calls == 0


def test_hybrid_extractor_falls_back_to_rules_when_llm_fails():
    """模型异常不能让 InputExtract 整体失败。"""

    extractor = HybridTripRequestExtractor(
        rule_extractor=RuleBasedTripRequestExtractor(),
        llm_extractor=LLMTripRequestExtractor(
            client=FailingJSONClient({})
        ),
    )

    data = extractor.extract(
        message="2026年7月2日从杭州去成都玩3天，预算4000",
        reference_date=date(2026, 7, 1),
    ).to_state_dict()

    assert data["destination"] == "成都"
    assert data["start_date"] == "2026-07-02"
    assert data["extraction"]["fallback_used"] is True
    assert data["extraction"]["method"] == "rule_fallback_v2"


def test_llm_budget_alias_is_normalized_to_the_canonical_constraint_field():
    """
    模拟真实线上情况：LLM 已正确抽到顶层 budget，却又生成 trip.budget。
    Schema 规范化必须保留预算语义、改写字段名，并让后续 Validator 可执行。
    """

    client = FakeJSONClient(
        {
            "origin": "北京",
            "destination": "杭州",
            "start_date": "2026-10-06",
            "days": 2,
            "budget": 1000,
            "people_count": 1,
            "preference_signals": [],
            "spend_preferences": [],
            "hard_constraints": [
                {
                    "domain": "trip",
                    "field": "budget",
                    "operator": "<=",
                    "value": 1000,
                    "evidence": "预算1000元",
                    "source": "llm",
                }
            ],
            "named_constraints": [],
            "field_resolution": {},
            "extraction_confidence": 0.9,
        }
    )

    data = HybridTripRequestExtractor(
        rule_extractor=RuleBasedTripRequestExtractor(),
        llm_extractor=LLMTripRequestExtractor(client=client),
    ).extract(
        message="明天从北京去杭州玩两天，预算1000元",
        reference_date=date(2026, 10, 5),
    ).to_state_dict()

    assert data["budget"] == 1000
    assert data["hard_constraints"] == [
        {
            "domain": "trip",
            "field": "total_budget",
            "operator": "<=",
            "value": 1000,
            "evidence": "预算1000元",
            "source": "llm",
        }
    ]
    # Prompt 是第一道预防，Schema alias 是第二道防线；两者都要可回归验证。
    assert "只填写顶层" in client.system_prompts[0]
    assert "不要额外生成 hard_constraints" in client.system_prompts[0]


def test_llm_result_is_authoritative_without_rule_conflict_noise():
    """LLM 成功时不再运行第二套规则制造伪冲突。"""

    client = FakeJSONClient(
        {
            "origin": "上海",
            "destination": "成都",
            "start_date": "2026-07-02",
            "days": 3,
            "people_count": 1,
            "preference_signals": [],
            "spend_preferences": [],
            "hard_constraints": [],
            "named_constraints": [],
            "field_resolution": {},
            "extraction_confidence": 0.8,
        }
    )

    rule = CountingRuleExtractor()
    extractor = HybridTripRequestExtractor(
        rule_extractor=rule,
        llm_extractor=LLMTripRequestExtractor(client=client),
    )

    data = extractor.extract(
        message="2026年7月2日从杭州去成都玩3天",
        reference_date=date(2026, 7, 1),
    ).to_state_dict()

    assert data["origin"] == "上海"
    assert data["requirement_issues"] == []
    assert rule.calls == 0
