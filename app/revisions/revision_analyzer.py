from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import date
from typing import Any, Mapping, Protocol

from pydantic import ValidationError

from app.common.config import settings
from app.extractors.constraint_extractor import RuleBasedConstraintExtractor
from app.extractors.preference_extractor import RuleBasedPreferenceExtractor
from app.llm.deepseek_json_client import (
    DeepSeekJSONClient,
    StructuredJSONClient,
)
from app.schemas.request_constraint_schema import (
    NamedConstraint,
    RequirementIssue,
    UnmappedRequirement,
)
from app.schemas.revision_schema import (
    RevisionFieldOperation,
    RevisionPatch,
    RevisionPlan,
)


class RevisionAnalyzer(Protocol):
    """RevisionAnalyzeNode 依赖的最小分析器协议。"""

    def analyze(
        self,
        *,
        trip_request: Mapping[str, Any],
        change_request: Mapping[str, Any],
        proposal: Mapping[str, Any],
    ) -> dict[str, Any]:
        """把修改文本转换成受控 RevisionPlan。"""
        ...


class RuleBasedRevisionAnalyzer:
    """
    规则版修改分析器。

    它不是为了覆盖所有自然语言，而是作为 LLM 不可用时的安全 fallback，
    只处理少量高确定性的常见修改：

        - 多待 / 少待 N 天；
        - 预算增加 / 减少 N 元；
        - 预算改成 N 元；
        - 酒店换成某个明确名称；
        - 已有 PreferenceExtractor 能识别的偏好；
        - 已有 ConstraintExtractor 能识别的客观约束和消费倾向。

    无法确定性理解的内容会进入 unmapped_requirements，而不是被静默忽略。
    """

    model_id = "rule_revision_fallback_v1"

    def __init__(self) -> None:
        self.preference_extractor = RuleBasedPreferenceExtractor()
        self.constraint_extractor = RuleBasedConstraintExtractor()

    def analyze(
        self,
        *,
        trip_request: Mapping[str, Any],
        change_request: Mapping[str, Any],
        proposal: Mapping[str, Any],
    ) -> dict[str, Any]:
        """从修改文本构造一个最小 RevisionPlan。"""

        raw_message = _required_change_message(change_request)
        base_proposal_id, base_proposal_version = _resolve_base_proposal(
            change_request=change_request,
            proposal=proposal,
        )

        field_operations = _extract_rule_field_operations(
            raw_message,
        )

        preference_upserts = self.preference_extractor.extract(
            raw_message
        )

        constraints = self.constraint_extractor.extract(
            raw_message
        )

        named_additions = list(
            constraints.named_constraints
        )
        replace_types: list[str] = []

        # 1. 额外支持“把酒店换成 X 酒店”。
        replacement_hotel = _extract_replacement_hotel(
            raw_message
        )

        if replacement_hotel is not None:
            named_additions = [
                item
                for item in named_additions
                if item.entity_type != "hotel"
            ]
            named_additions.append(
                replacement_hotel
            )
            replace_types.append("hotel")

        unmapped = _extract_rule_unmapped_requirements(
            raw_message=raw_message,
            field_operations=field_operations,
            preference_count=len(preference_upserts),
            named_count=len(named_additions),
        )

        # 2. 对会影响候选选择、但表达仍然不够明确的内容生成阻断澄清。
        #    例如“飞机坐好一点”可能指直飞、时间、舱位或退改条件，
        #    不能由规则 fallback 擅自选择一种解释。
        ambiguity_issues = _extract_rule_ambiguity_issues(
            raw_message
        )

        patch = RevisionPatch(
            field_operations=field_operations,
            preference_upserts=preference_upserts,
            spend_preference_upserts=(
                constraints.spend_preferences
            ),
            hard_constraint_additions=(
                constraints.hard_constraints
            ),
            named_constraint_additions=named_additions,
            replace_named_entity_types=replace_types,
            unmapped_requirement_additions=unmapped,
            requirement_issues=ambiguity_issues,
            summary="规则 fallback 解析了修改请求中的可确定内容。",
            confidence=0.55,
        )

        clarification_questions: list[str] = [
            item.suggested_question
            or item.message
            for item in patch.requirement_issues
            if item.requires_clarification
        ]

        # 3. 如果规则没有识别出任何可应用修改，不能假装修改成功。
        if not patch.has_changes() and not clarification_questions:
            issue = RequirementIssue(
                issue_type="revision_not_understood",
                message="当前修改说明无法被可靠解析。",
                evidence=[raw_message],
                requires_clarification=True,
                suggested_question=(
                    "请明确说明要修改的字段，例如：预算增加1000元、"
                    "多待2天、酒店换成某某酒店。"
                ),
            )
            patch.requirement_issues.append(issue)
            clarification_questions.append(
                issue.suggested_question
            )

        status = (
            "clarification_required"
            if clarification_questions
            else "ready"
        )

        plan = RevisionPlan(
            status=status,
            strategy_version="rule_revision_patch_v1",
            base_proposal_id=base_proposal_id,
            base_proposal_version=base_proposal_version,
            base_trip_request_hash=hash_trip_request(
                trip_request
            ),
            raw_change_request=raw_message,
            patch=patch,
            clarification_questions=clarification_questions,
            model_id=self.model_id,
            attempt_count=1,
            fallback_used=True,
            validation_errors=[],
        )

        return plan.to_state_dict()


class LLMRevisionAnalyzer:
    """
    使用 DeepSeek Structured JSON 生成 RevisionPatch。

    设计边界：
        1. 模型只输出修改补丁，不重新生成完整 TripRequest；
        2. 相对修改必须用 increment 表达；
        3. 补丁只能使用 RevisionPatch Schema 允许的字段；
        4. Pydantic 或业务校验失败时进行有限重试；
        5. 达到重试上限后回退 RuleBasedRevisionAnalyzer。
    """

    def __init__(
        self,
        *,
        client: StructuredJSONClient,
        fallback_analyzer: RevisionAnalyzer | None = None,
        max_tokens: int = 4096,
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts 必须大于等于 1")

        self.client = client
        self.fallback_analyzer = (
            fallback_analyzer
            or RuleBasedRevisionAnalyzer()
        )
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts

    def analyze(
        self,
        *,
        trip_request: Mapping[str, Any],
        change_request: Mapping[str, Any],
        proposal: Mapping[str, Any],
    ) -> dict[str, Any]:
        """调用模型生成补丁，并在失败时安全回退规则版。"""

        raw_message = _required_change_message(change_request)
        base_proposal_id, base_proposal_version = _resolve_base_proposal(
            change_request=change_request,
            proposal=proposal,
        )
        base_hash = hash_trip_request(trip_request)

        system_prompt = _build_revision_system_prompt()
        base_user_prompt = _build_revision_user_prompt(
            trip_request=trip_request,
            raw_message=raw_message,
        )

        validation_errors: list[str] = []

        # 1. 首次生成失败时，把简短校验错误反馈给模型再试一次。
        for attempt in range(1, self.max_attempts + 1):
            user_prompt = base_user_prompt

            if validation_errors:
                user_prompt += (
                    "\n\n上一次 RevisionPatch 未通过校验。"
                    "请重新输出完整 JSON，并修复：\n- "
                    + "\n- ".join(validation_errors[-8:])
                )

            try:
                payload = self.client.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=self.max_tokens,
                )

                patch = RevisionPatch(**payload)
                _validate_revision_patch_against_base(
                    patch=patch,
                    trip_request=trip_request,
                )

                clarification_questions = [
                    item.suggested_question
                    or item.message
                    for item in patch.requirement_issues
                    if item.requires_clarification
                ]

                # UnmappedRequirement 也允许显式声明必须澄清。
                # 这样模型即使没有创建 RequirementIssue，也不会让关键歧义静默通过。
                clarification_questions.extend(
                    f"请进一步说明：{item.text}"
                    for item in patch.unmapped_requirement_additions
                    if item.requires_clarification
                )
                clarification_questions = list(
                    dict.fromkeys(clarification_questions)
                )

                if not patch.has_changes() and not clarification_questions:
                    raise ValueError(
                        "RevisionPatch 没有任何可应用修改，也没有澄清问题"
                    )

                status = (
                    "clarification_required"
                    if clarification_questions
                    else "ready"
                )

                plan = RevisionPlan(
                    status=status,
                    strategy_version="llm_revision_patch_v1",
                    base_proposal_id=base_proposal_id,
                    base_proposal_version=base_proposal_version,
                    base_trip_request_hash=base_hash,
                    raw_change_request=raw_message,
                    patch=patch,
                    clarification_questions=(
                        clarification_questions
                    ),
                    model_id=self.client.model_id,
                    attempt_count=attempt,
                    fallback_used=False,
                    validation_errors=(
                        validation_errors
                    ),
                )

                return plan.to_state_dict()

            except (
                ValidationError,
                ValueError,
                TypeError,
            ) as exc:
                validation_errors.append(
                    _short_error_message(exc)
                )

            except Exception as exc:
                # 2. API、网络或模型错误也只在限定次数内重试。
                validation_errors.append(
                    "模型调用失败："
                    f"{_short_error_message(exc)}"
                )

        # 3. LLM 多次失败后使用规则 fallback，保证常见修改仍可继续。
        fallback_output = self.fallback_analyzer.analyze(
            trip_request=trip_request,
            change_request=change_request,
            proposal=proposal,
        )

        fallback_output[
            "validation_errors"
        ] = validation_errors
        fallback_output["attempt_count"] = (
            self.max_attempts
        )
        fallback_output["fallback_used"] = True

        return fallback_output


def build_default_revision_analyzer() -> RevisionAnalyzer:
    """
    根据当前配置创建默认 Revision Analyzer。

    没有配置 DeepSeek Key 时直接使用规则 fallback，便于离线开发和测试。
    """

    if not settings.deepseek_api_key:
        return RuleBasedRevisionAnalyzer()

    client = DeepSeekJSONClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.llm_base_url,
        model_id=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        thinking_enabled=settings.llm_thinking_enabled,
    )

    return LLMRevisionAnalyzer(
        client=client,
        fallback_analyzer=(
            RuleBasedRevisionAnalyzer()
        ),
        max_tokens=settings.revision_max_tokens,
        max_attempts=settings.revision_max_attempts,
    )


def hash_trip_request(
    trip_request: Mapping[str, Any],
) -> str:
    """
    为当前 TripRequest 计算稳定 SHA-256。

    ApplyRevisionNode 会重新计算并比较该值，防止补丁应用到已经变化的旧请求。
    """

    payload = json.dumps(
        _jsonable_mapping(trip_request),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


# ======================================================================
# Prompt
# ======================================================================


def _build_revision_system_prompt() -> str:
    """构造 RevisionPatch 的严格系统提示词。"""

    schema = RevisionPatch.model_json_schema()

    return (
        "你是 TravelOps-Copilot 的修改请求分析器。\n"
        "只输出一个符合 JSON Schema 的 JSON object，不要输出 Markdown。\n\n"
        "任务不是重新生成完整 TripRequest，而是生成受控 RevisionPatch。\n"
        "必须遵守：\n"
        "1. ‘预算增加1000’使用 field_operations.increment budget=1000，"
        "不能设置 budget=1000。\n"
        "2. ‘多待两天’使用 increment days=2。\n"
        "3. ‘改成5天’使用 set days=5。\n"
        "4. ‘酒店换成X酒店’应先在 replace_named_entity_types 中加入 hotel，"
        "再添加 required hotel named constraint。\n"
        "5. 主观要求如‘更安静、更干净’进入 preference_upserts，"
        "不自动变成硬约束。\n"
        "6. 只有可结构化验证的条件进入 hard_constraint_additions。\n"
        "7. 无法执行但不阻断核心规划的要求进入 unmapped_requirement_additions。\n"
        "8. 真正存在歧义、无法安全应用时进入 requirement_issues，"
        "并设置 requires_clarification=true。\n"
        "9. 不得修改任何未被用户提及的字段。\n"
        "10. 不得提供天气、航班、酒店房态等动态事实。\n\n"
        "RevisionPatch JSON Schema：\n"
        + json.dumps(
            schema,
            ensure_ascii=False,
        )
    )


def _build_revision_user_prompt(
    *,
    trip_request: Mapping[str, Any],
    raw_message: str,
) -> str:
    """提供当前请求基线和用户本轮修改文本。"""

    # 只把修改分析需要的请求字段发送给模型，避免加入工具结果和内部 Trace。
    compact_request = deepcopy(
        dict(trip_request)
    )
    compact_request.pop("extraction", None)

    return (
        "当前已经确认的 TripRequest：\n"
        + json.dumps(
            _jsonable_mapping(compact_request),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n\n用户本轮修改要求：\n"
        + raw_message
    )


# ======================================================================
# Rule fallback helpers
# ======================================================================


def _extract_rule_field_operations(
    text: str,
) -> list[RevisionFieldOperation]:
    """抽取常见天数和预算修改。"""

    compact = re.sub(r"\s+", "", text)
    output: list[RevisionFieldOperation] = []

    # 1. “多待两天 / 延长2天”。
    days_increase = re.search(
        r"(?:多待|多玩|延长|增加)([一二两三四五六七八九十\d]+)天",
        compact,
    )
    days_decrease = re.search(
        r"(?:少待|少玩|缩短|减少)([一二两三四五六七八九十\d]+)天",
        compact,
    )
    days_set = re.search(
        r"(?:改成|调整为|变成|总共玩|一共玩)([一二两三四五六七八九十\d]+)天",
        compact,
    )

    if days_set:
        output.append(
            RevisionFieldOperation(
                operation="set",
                field="days",
                value=_parse_small_number(
                    days_set.group(1)
                ),
                evidence=days_set.group(0),
            )
        )
    elif days_increase:
        output.append(
            RevisionFieldOperation(
                operation="increment",
                field="days",
                value=_parse_small_number(
                    days_increase.group(1)
                ),
                evidence=days_increase.group(0),
            )
        )
    elif days_decrease:
        output.append(
            RevisionFieldOperation(
                operation="increment",
                field="days",
                value=-_parse_small_number(
                    days_decrease.group(1)
                ),
                evidence=days_decrease.group(0),
            )
        )

    # 2. “预算增加1000 / 减少500 / 改成5000”。
    budget_set = re.search(
        r"预算(?:改成|调整为|调整到|变成|设为)(\d+(?:\.\d+)?)",
        compact,
    )
    budget_increase = re.search(
        r"预算(?:增加|加|提高)(\d+(?:\.\d+)?)",
        compact,
    )
    budget_decrease = re.search(
        r"预算(?:减少|减|降低)(\d+(?:\.\d+)?)",
        compact,
    )

    if budget_set:
        output.append(
            RevisionFieldOperation(
                operation="set",
                field="budget",
                value=float(
                    budget_set.group(1)
                ),
                evidence=budget_set.group(0),
            )
        )
    elif budget_increase:
        output.append(
            RevisionFieldOperation(
                operation="increment",
                field="budget",
                value=float(
                    budget_increase.group(1)
                ),
                evidence=(
                    budget_increase.group(0)
                ),
            )
        )
    elif budget_decrease:
        output.append(
            RevisionFieldOperation(
                operation="increment",
                field="budget",
                value=-float(
                    budget_decrease.group(1)
                ),
                evidence=(
                    budget_decrease.group(0)
                ),
            )
        )

    return output


def _extract_replacement_hotel(
    text: str,
) -> NamedConstraint | None:
    """识别“酒店换成某某酒店”这类明确替换。"""

    compact = re.sub(r"\s+", "", text)
    match = re.search(
        r"(?:酒店|住宿)?(?:换成|改成|换为|改为)"
        r"(?P<name>[^，。；]{2,30}?(?:酒店|宾馆|客栈|民宿))",
        compact,
    )

    if not match:
        return None

    return NamedConstraint(
        entity_type="hotel",
        entity_name=match.group("name"),
        constraint_mode="required",
        scope="whole_trip",
        evidence=match.group(0),
        source="rule",
    )


def _extract_rule_ambiguity_issues(
    raw_message: str,
) -> list[RequirementIssue]:
    """
    识别会改变航班选择、但缺少可执行标准的模糊修改。

    这些要求不能只作为 Proposal 文案参考，因为它们会直接影响候选排序。
    当前规则 fallback 选择澄清，而不是擅自把“好一点”解释成某个特征。
    """

    compact = re.sub(r"\s+", "", raw_message)
    issues: list[RequirementIssue] = []

    if any(
        pattern in compact
        for pattern in (
            "飞机坐好一点",
            "航班好一点",
        )
    ):
        issues.append(
            RequirementIssue(
                issue_type="ambiguous_flight_quality",
                message=(
                    "‘飞机好一点’无法确定是指直飞、时间更合适、"
                    "退改更灵活，还是其他条件。"
                ),
                evidence=[raw_message],
                requires_clarification=True,
                suggested_question=(
                    "你说的‘飞机好一点’主要是希望直飞、"
                    "起降时间更合适，还是退改更灵活？"
                ),
            )
        )

    if any(
        pattern in compact
        for pattern in (
            "返回时间延后一点",
            "返程时间延后一点",
            "换个返回时间",
        )
    ):
        issues.append(
            RequirementIssue(
                issue_type="ambiguous_return_time",
                message=(
                    "返程时间需要一个可验证的时间范围，"
                    "‘延后一点’没有明确下限。"
                ),
                evidence=[raw_message],
                requires_clarification=True,
                suggested_question=(
                    "你希望返程航班最早几点起飞？"
                    "例如：不早于16:00。"
                ),
            )
        )

    return issues


def _extract_rule_unmapped_requirements(
    *,
    raw_message: str,
    field_operations: list[RevisionFieldOperation],
    preference_count: int,
    named_count: int,
) -> list[UnmappedRequirement]:
    """
    保存规则 fallback 能识别为“只影响展示风格”的非阻断要求。

    当前最小版本不尝试用规则猜测开放式体验风格，因此默认返回空列表。
    如果整段修改完全没有被识别，上层会生成统一澄清问题。
    """

    del raw_message, field_operations, preference_count, named_count
    return []


def _parse_small_number(value: str) -> int:
    """解析 1—30 范围内的中文小数字或阿拉伯数字。"""

    if value.isdigit():
        return int(value)

    mapping = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    if value in mapping:
        return mapping[value]

    if value.startswith("十"):
        return 10 + mapping.get(
            value[1:],
            0,
        )

    if "十" in value:
        left, right = value.split(
            "十",
            maxsplit=1,
        )
        return (
            mapping.get(left, 0) * 10
            + mapping.get(right, 0)
        )

    raise ValueError(
        f"无法解析数字：{value}"
    )


# ======================================================================
# Validation and common helpers
# ======================================================================


def _validate_revision_patch_against_base(
    *,
    patch: RevisionPatch,
    trip_request: Mapping[str, Any],
) -> None:
    """检查相对修改是否有可用基线。"""

    for operation in patch.field_operations:
        if operation.operation != "increment":
            continue

        current_value = trip_request.get(
            operation.field
        )

        if current_value is None:
            raise ValueError(
                f"字段 {operation.field} 当前为空，"
                "不能执行 increment，必须先给出明确新值"
            )


def _required_change_message(
    change_request: Mapping[str, Any],
) -> str:
    """读取 DecisionGate 保存的非空修改文本。"""

    raw_message = str(
        change_request.get("raw_message")
        or ""
    ).strip()

    if not raw_message:
        raise ValueError(
            "change_request.raw_message 不能为空"
        )

    return raw_message


def _resolve_base_proposal(
    *,
    change_request: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> tuple[str, int]:
    """确认修改请求仍然对应当前 Proposal。"""

    proposal_id = str(
        proposal.get("proposal_id")
        or ""
    ).strip()
    proposal_version = _safe_int(
        proposal.get("version"),
        default=0,
    )

    base_id = str(
        change_request.get(
            "base_proposal_id"
        )
        or ""
    ).strip()
    base_version = _safe_int(
        change_request.get(
            "base_proposal_version"
        ),
        default=0,
    )

    if not proposal_id or proposal_version < 1:
        raise ValueError(
            "当前 proposal_id 或 version 无效"
        )

    if base_id != proposal_id:
        raise ValueError(
            "change_request.base_proposal_id 与当前 Proposal 不一致"
        )

    if base_version != proposal_version:
        raise ValueError(
            "change_request.base_proposal_version 与当前 Proposal 不一致"
        )

    return proposal_id, proposal_version


def _jsonable_mapping(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """将日期等值转换为稳定 JSON 结构。"""

    return json.loads(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            default=str,
        )
    )


def _short_error_message(
    exc: Exception,
    max_length: int = 1000,
) -> str:
    """把异常压缩成适合反馈给模型和 Trace 的文本。"""

    text = str(exc).strip()

    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def _safe_int(
    value: Any,
    default: int,
) -> int:
    """安全转换整数。"""

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return default
