from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from app.rag.metadata_filter import metadata_matches
from app.schemas.evidence_grade_schema import (
    EvidenceCoverage,
    EvidenceGradeResult,
    EvidenceIssue,
)
from app.schemas.retrieval_plan_schema import (
    RetrievalPlan,
    RetrievalTask,
)


# 当前 WeatherNode 和安全 RAG 文档支持的标准天气风险。
SUPPORTED_WEATHER_RISK_TYPES = {
    "rain",
    "heavy_rain",
    "wind",
    "heat",
}


# Evidence Pool 输出时使用的稳定类别顺序。
CATEGORY_ORDER = {
    "guides": 1,
    "hotel_reviews": 2,
    "safety_notices": 3,
    "packing_checklists": 4,
}


class EvidenceGrader(Protocol):
    """
    Evidence Grader 协议。

    EvidenceGradeNode 依赖这个协议，
    测试或未来 LLM Evidence Judge 可以替换实现。
    """

    def grade(
        self,
        retrieval_plan: Mapping[str, Any],
        reranked_evidence: Sequence[Mapping[str, Any]],
        existing_evidence_pool: Sequence[Mapping[str, Any]] | None = None,
        existing_requirements: Mapping[str, Any] | None = None,
        weather_result: Mapping[str, Any] | None = None,
        rerank_result: Mapping[str, Any] | None = None,
        rag_retry_count: int = 0,
    ) -> dict[str, Any]:
        """
        对当前证据执行完整质量检查。
        """
        ...


class RuleBasedEvidenceGrader:
    """
    规则版证据评分器。

    主要检查五个方面：

        1. 结构合法性
            chunk_id、source、content、metadata 是否存在。

        2. 任务一致性
            task_id、category 和 doc_type 是否与 RetrievalTask 一致。

        3. Metadata Filter
            city、hotel_id、scenario、risk_level 是否符合检索计划。

        4. 覆盖完整性
            required 类别是否达到 min_required_chunks。

        5. 特殊业务覆盖
            每个候选酒店是否有评价证据。
            每个天气风险是否有安全提醒证据。

    这个类不判断：

        - 旅行方案是否正确。
        - 酒店是否最终值得选择。
        - 预算是否满足。
        - LLM 是否编造内容。

    上述内容分别由 CandidateRankNode、BudgetOptimizeNode 和
    VerifierNode 负责。
    """

    def __init__(
        self,
        allow_rerank_fallback: bool = True,
        min_content_chars: int = 10,
        max_issue_examples: int = 50,
    ) -> None:
        """
        Args:
            allow_rerank_fallback:
                Cross-Encoder 失败时，是否允许 RRF fallback 证据继续参与。

                True：
                    fallback 证据可以使用，但最终状态为 degraded。

                False：
                    fallback 证据会被判定为无效。

            min_content_chars:
                一条证据正文的最少字符数。

                这个参数只防止空正文或极短无意义文本。
                它不是判断内容语义质量的阈值。

            max_issue_examples:
                最多在 evidence_result.issues 中保存多少条问题。

                防止大量坏 chunk 让 State 和 Trace 变得过大。
        """

        if min_content_chars < 1:
            raise ValueError(
                "min_content_chars 必须大于等于 1"
            )

        if max_issue_examples < 1:
            raise ValueError(
                "max_issue_examples 必须大于等于 1"
            )

        self.allow_rerank_fallback = (
            allow_rerank_fallback
        )

        self.min_content_chars = (
            min_content_chars
        )

        self.max_issue_examples = (
            max_issue_examples
        )

    def grade(
        self,
        retrieval_plan: Mapping[str, Any],
        reranked_evidence: Sequence[Mapping[str, Any]],
        existing_evidence_pool: Sequence[Mapping[str, Any]] | None = None,
        existing_requirements: Mapping[str, Any] | None = None,
        weather_result: Mapping[str, Any] | None = None,
        rerank_result: Mapping[str, Any] | None = None,
        rag_retry_count: int = 0,
    ) -> dict[str, Any]:
        """
        对当前 RAG 证据执行质量检查。

        整体步骤：

            1. 校验并解析 RetrievalPlan。
            2. 读取或创建初始证据要求。
            3. 校验本轮 reranked_evidence。
            4. 初次模式重置证据池，修复模式合并证据池。
            5. 按 category 检查证据覆盖。
            6. 检查酒店 ID 和天气风险覆盖。
            7. 决定通过、降级、修复或失败。
            8. 生成 RetrievalPlanNode 可读取的 retrieval_feedback。
        """

        # 1. 将普通 dict 转换为 Pydantic RetrievalPlan。
        #    这样后续可以稳定读取 tasks、mode、max_repair_rounds。
        plan = RetrievalPlan(
            **dict(retrieval_plan)
        )

        latest_evidence = [
            dict(item)
            for item in reranked_evidence
            if isinstance(item, Mapping)
        ]

        old_pool = [
            dict(item)
            for item in (
                existing_evidence_pool
                or []
            )
            if isinstance(item, Mapping)
        ]

        # 2. 确定本次使用的“完整证据要求”。
        #
        #    initial：
        #        当前 plan 的 coverage_requirements 是完整要求。
        #
        #    repair：
        #        优先使用 initial 阶段保存的 existing_requirements，
        #        防止只检查当前修复任务而忘记其他类别。
        requirements = (
            self._resolve_requirements(
                plan=plan,
                existing_requirements=(
                    existing_requirements
                ),
            )
        )

        # 3. 校验当前 RerankNode 刚刚输出的证据。
        #
        #    这里只校验 latest_evidence。
        #    existing_evidence_pool 中的证据已经在前一轮通过过结构校验。
        (
            valid_latest_evidence,
            rejected_count,
            validation_issues,
            fallback_task_ids,
        ) = self._validate_latest_evidence(
            plan=plan,
            evidence=latest_evidence,
        )

        # 4. 根据 initial / repair 模式创建累计 evidence pool。
        #
        #    initial：
        #        开始一轮新的旅行检索，旧证据不能继续复用。
        #
        #    repair：
        #        将旧证据和本轮新证据合并。
        if plan.mode == "initial":
            base_pool: list[
                dict[str, Any]
            ] = []
        else:
            base_pool = old_pool

        evidence_pool = (
            self._merge_evidence_pool(
                old_evidence=base_pool,
                new_evidence=(
                    valid_latest_evidence
                ),
            )
        )

        # 5. 根据完整 requirements 检查证据覆盖。
        coverage, coverage_issues = (
            self._grade_coverage(
                requirements=requirements,
                evidence_pool=evidence_pool,
                weather_result=(
                    weather_result
                    or {}
                ),
            )
        )

        issues = [
            *validation_issues,
            *coverage_issues,
        ]

        # 6. 检查 Rerank 是否使用过 fallback。
        #
        #    fallback 证据仍可使用，但需要把状态标记为 degraded，
        #    让后续 Trace 和 Eval 知道没有正常完成 Cross-Encoder 精排。
        rerank_fallback_used = (
            bool(fallback_task_ids)
            or bool(
                (
                    rerank_result
                    or {}
                ).get("fallback_used")
            )
        )

        if rerank_fallback_used:
            issues.append(
                EvidenceIssue(
                    issue_type=(
                        "rerank_fallback_used"
                    ),
                    severity="warning",
                    message=(
                        "部分证据使用了 RRF 原始排名 fallback，"
                        "没有经过正常 Cross-Encoder 精排。"
                    ),
                    details={
                        "task_ids": sorted(
                            fallback_task_ids
                        )
                    },
                )
            )

        # 7. 统计 required category 是否全部通过。
        required_coverages = [
            item
            for item in coverage.values()
            if item.required
        ]

        passed_required_coverages = [
            item
            for item
            in required_coverages
            if item.passed
        ]

        required_passed = (
            len(required_coverages)
            ==
            len(passed_required_coverages)
        )

        # 8. 判断是否还有修复次数。
        #
        #    RetrievalPlanNode 在真正生成 repair plan 时，
        #    会将 rag_retry_count 增加 1。
        #
        #    默认最大修复次数为 1：
        #
        #        初次 grade：
        #            rag_retry_count = 0
        #            可以 repair
        #
        #        repair 后再次 grade：
        #            rag_retry_count = 1
        #            不允许再次 repair
        can_repair = (
            not required_passed
            and not plan.repair_exhausted
            and rag_retry_count
            < plan.max_repair_rounds
        )

        # 9. 如果 required evidence 不足，生成结构化修复反馈。
        retrieval_feedback = (
            self._build_retrieval_feedback(
                plan=plan,
                coverage=coverage,
                issues=issues,
            )
            if not required_passed
            else None
        )

        # 10. 决定 EvidenceGradeNode 的最终状态。
        if required_passed:
            # Required 证据都满足，但发生了 fallback 或坏证据被拒绝，
            # 状态为 degraded，工作流仍然可以继续。
            if (
                rerank_fallback_used
                or rejected_count > 0
            ):
                status = "degraded"
            else:
                status = "passed"

            passed = True
            repairable = False
            next_action = "continue"

        elif can_repair:
            status = "repair_required"
            passed = False
            repairable = True
            next_action = "repair"

        else:
            status = "failed"
            passed = False
            repairable = False
            next_action = "stop"

        # 11. 只保留限定数量的问题，避免 State 过大。
        issues = issues[
            : self.max_issue_examples
        ]

        grade_result = EvidenceGradeResult(
            status=status,
            passed=passed,
            repairable=repairable,
            next_action=next_action,
            plan_id=plan.plan_id,
            mode=plan.mode,
            graded_evidence_count=len(
                evidence_pool
            ),
            rejected_evidence_count=(
                rejected_count
            ),
            required_category_count=len(
                required_coverages
            ),
            passed_required_category_count=len(
                passed_required_coverages
            ),
            fallback_used=(
                rerank_fallback_used
            ),
            coverage=coverage,
            issues=issues,
            retrieval_feedback=(
                retrieval_feedback
            ),
        )

        return {
            "evidence_pool":
                evidence_pool,

            "evidence_requirements":
                requirements,

            "evidence_result":
                grade_result.to_state_dict(),

            "retrieval_feedback":
                (
                    retrieval_feedback
                    or {}
                ),
        }

    def _resolve_requirements(
        self,
        plan: RetrievalPlan,
        existing_requirements: Mapping[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        """
        确定当前证据检查使用的完整覆盖要求。
        """

        # 1. initial 模式必须使用当前 plan 的完整 requirements。
        if plan.mode == "initial":
            requirements = (
                self._normalize_requirements(
                    plan.coverage_requirements
                )
            )

        # 2. repair 模式优先使用初次检索保存的 requirements。
        elif existing_requirements:
            requirements = (
                self._normalize_requirements(
                    existing_requirements
                )
            )

        # 3. 单独测试 repair plan 时可能没有旧 requirements，
        #    这时只能使用当前 repair plan 的要求。
        else:
            requirements = (
                self._normalize_requirements(
                    plan.coverage_requirements
                )
            )

        # 4. 某些手写 RetrievalPlan 可能没有 coverage_requirements。
        #    根据 tasks 补充最小要求，避免 EvidenceGrade 静默通过。
        for task in plan.tasks:
            category_requirement = (
                requirements.setdefault(
                    task.category,
                    {
                        "required": False,
                        "min_required_chunks": 0,
                        "task_ids": [],
                    },
                )
            )

            category_requirement[
                "required"
            ] = (
                bool(
                    category_requirement.get(
                        "required"
                    )
                )
                or task.required
            )

            category_requirement[
                "min_required_chunks"
            ] = max(
                _safe_int(
                    category_requirement.get(
                        "min_required_chunks"
                    ),
                    default=0,
                ),
                task.min_required_chunks,
            )

            task_ids = (
                category_requirement.setdefault(
                    "task_ids",
                    [],
                )
            )

            if task.task_id not in task_ids:
                task_ids.append(
                    task.task_id
                )

            # 5. 酒店评价任务记录必须覆盖的 hotel_ids。
            if task.category == "hotel_reviews":
                hotel_ids = (
                    task.metadata_filter.get(
                        "hotel_ids"
                    )
                )

                if isinstance(hotel_ids, list):
                    existing_hotel_ids = (
                        category_requirement.setdefault(
                            "required_hotel_ids",
                            [],
                        )
                    )

                    for hotel_id in hotel_ids:
                        hotel_id = str(
                            hotel_id
                        )

                        if (
                            hotel_id
                            not in existing_hotel_ids
                        ):
                            existing_hotel_ids.append(
                                hotel_id
                            )

        return requirements

    def _validate_latest_evidence(
        self,
        plan: RetrievalPlan,
        evidence: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        int,
        list[EvidenceIssue],
        set[str],
    ]:
        """
        校验当前 RerankNode 输出的证据结构和 metadata。
        """

        valid: list[
            dict[str, Any]
        ] = []

        issues: list[
            EvidenceIssue
        ] = []

        rejected_count = 0

        fallback_task_ids: set[
            str
        ] = set()

        task_by_id = {
            task.task_id: task
            for task in plan.tasks
        }

        # 1. 逐条检查 evidence。
        for item in evidence:
            task_id = str(
                item.get("task_id")
                or ""
            )

            chunk_id = str(
                item.get("chunk_id")
                or ""
            )

            task = task_by_id.get(
                task_id
            )

            # 2. task_id 必须属于当前 RetrievalPlan。
            if task is None:
                rejected_count += 1

                issues.append(
                    EvidenceIssue(
                        issue_type=(
                            "unknown_task_id"
                        ),
                        severity="error",
                        task_id=task_id or None,
                        chunk_id=chunk_id or None,
                        message=(
                            "证据 task_id 不存在于当前 retrieval_plan，"
                            "已拒绝该证据。"
                        ),
                    )
                )

                continue

            # 3. chunk_id 必须存在。
            if not chunk_id:
                rejected_count += 1

                issues.append(
                    EvidenceIssue(
                        issue_type=(
                            "missing_chunk_id"
                        ),
                        severity="error",
                        category=task.category,
                        task_id=task_id,
                        message=(
                            "证据缺少 chunk_id，无法追踪来源。"
                        ),
                    )
                )

                continue

            source = str(
                item.get("source")
                or ""
            ).strip()

            # 4. source 必须存在。
            if not source:
                rejected_count += 1

                issues.append(
                    EvidenceIssue(
                        issue_type=(
                            "missing_source"
                        ),
                        severity="error",
                        category=task.category,
                        task_id=task_id,
                        chunk_id=chunk_id,
                        message=(
                            "证据缺少 source，无法进行来源追溯。"
                        ),
                    )
                )

                continue

            content = str(
                item.get("content")
                or ""
            ).strip()

            # 5. 正文不能是空文本或无意义极短文本。
            if len(content) < self.min_content_chars:
                rejected_count += 1

                issues.append(
                    EvidenceIssue(
                        issue_type=(
                            "content_too_short"
                        ),
                        severity="error",
                        category=task.category,
                        task_id=task_id,
                        chunk_id=chunk_id,
                        message=(
                            "证据正文过短，无法支持后续方案生成。"
                        ),
                        details={
                            "content_length":
                                len(content),

                            "min_content_chars":
                                self.min_content_chars,
                        },
                    )
                )

                continue

            metadata = item.get(
                "metadata"
            )

            # 6. metadata 必须是 dict。
            if not isinstance(
                metadata,
                Mapping,
            ):
                rejected_count += 1

                issues.append(
                    EvidenceIssue(
                        issue_type=(
                            "invalid_metadata"
                        ),
                        severity="error",
                        category=task.category,
                        task_id=task_id,
                        chunk_id=chunk_id,
                        message=(
                            "证据 metadata 缺失或不是 dict。"
                        ),
                    )
                )

                continue

            metadata = dict(
                metadata
            )

            # 7. evidence.category 必须与 RetrievalTask.category 一致。
            evidence_category = str(
                item.get("category")
                or ""
            )

            if evidence_category != task.category:
                rejected_count += 1

                issues.append(
                    EvidenceIssue(
                        issue_type=(
                            "category_mismatch"
                        ),
                        severity="error",
                        category=evidence_category or None,
                        task_id=task_id,
                        chunk_id=chunk_id,
                        message=(
                            "证据 category 与 RetrievalTask.category 不一致。"
                        ),
                        details={
                            "expected":
                                task.category,

                            "actual":
                                evidence_category,
                        },
                    )
                )

                continue

            # 8. metadata.doc_type 必须与 RetrievalTask.doc_type 一致。
            actual_doc_type = str(
                metadata.get("doc_type")
                or ""
            )

            if actual_doc_type != task.doc_type:
                rejected_count += 1

                issues.append(
                    EvidenceIssue(
                        issue_type=(
                            "doc_type_mismatch"
                        ),
                        severity="error",
                        category=task.category,
                        task_id=task_id,
                        chunk_id=chunk_id,
                        message=(
                            "证据 doc_type 与 RetrievalTask.doc_type 不一致。"
                        ),
                        details={
                            "expected":
                                task.doc_type,

                            "actual":
                                actual_doc_type,
                        },
                    )
                )

                continue

            # 9. 验证 city、hotel_id、scenario、risk_level 等过滤条件。
            #
            #    这一步相当于重新确认：
            #    HybridRetrieveNode 没有把 metadata 不匹配的文档带进来。
            if not metadata_matches(
                metadata=metadata,
                metadata_filter=(
                    task.metadata_filter
                ),
            ):
                rejected_count += 1

                issues.append(
                    EvidenceIssue(
                        issue_type=(
                            "metadata_filter_mismatch"
                        ),
                        severity="error",
                        category=task.category,
                        task_id=task_id,
                        chunk_id=chunk_id,
                        message=(
                            "证据 metadata 不满足 RetrievalTask 的过滤条件。"
                        ),
                        details={
                            "metadata_filter":
                                task.metadata_filter,

                            "actual_metadata":
                                metadata,
                        },
                    )
                )

                continue

            rerank_score = item.get(
                "rerank_score"
            )

            rerank_model = str(
                item.get("rerank_model")
                or ""
            )

            is_fallback = (
                rerank_model.startswith(
                    "fallback:"
                )
            )

            # RRF passthrough 是产品默认的 Variant C，而不是 Cross-Encoder
            # 故障后的 fallback。它刻意没有伪造 rerank_score：排序依据是
            # HybridRetrieveNode 已产出的 final_rank / fusion_score。
            #
            # 这里要求 model 和 selection_reason 两个字段同时精确匹配，才
            # 放行无分数证据。这样既支持默认快速路径，也仍能拒绝任何误删
            # rerank_score、但没有可靠排序来源的普通证据。
            is_rrf_passthrough = (
                rerank_model == "rrf_passthrough"
                and str(item.get("selection_reason") or "")
                == "rrf_passthrough"
            )

            # 10. 正常 Cross-Encoder 结果必须有 0—1 的 rerank_score。
            if rerank_score is not None:
                try:
                    numeric_score = float(
                        rerank_score
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    numeric_score = -1.0

                if not (
                    0.0
                    <= numeric_score
                    <= 1.0
                ):
                    rejected_count += 1

                    issues.append(
                        EvidenceIssue(
                            issue_type=(
                                "invalid_rerank_score"
                            ),
                            severity="error",
                            category=task.category,
                            task_id=task_id,
                            chunk_id=chunk_id,
                            message=(
                                "rerank_score 必须位于 0 到 1 之间。"
                            ),
                            details={
                                "rerank_score":
                                    rerank_score
                            },
                        )
                    )

                    continue

            # 11. 没有 rerank_score 时，只允许两种明确、可追溯的 RRF 来源：
            #
            #    - fallback:*：Cross-Encoder 实际失败后产生的降级结果；
            #    - rrf_passthrough：配置明确关闭 Cross-Encoder 时的 Variant C。
            #
            # 两者都不编造 Cross-Encoder 分数；前者会把工作流标为 degraded，
            # 后者是正常默认策略，不会被误报为模型故障。
            elif is_fallback:
                if not self.allow_rerank_fallback:
                    rejected_count += 1

                    issues.append(
                        EvidenceIssue(
                            issue_type=(
                                "fallback_not_allowed"
                            ),
                            severity="error",
                            category=task.category,
                            task_id=task_id,
                            chunk_id=chunk_id,
                            message=(
                                "当前 EvidenceGrader 不允许使用 RRF fallback。"
                            ),
                        )
                    )

                    continue

                fallback_task_ids.add(
                    task_id
                )

            elif is_rrf_passthrough:
                # Variant C 已由上方严格标识，保留 RRF 原始顺序并继续参与
                # coverage / provenance 校验；不写 fallback_task_ids，避免把
                # 用户主动选择的默认快速模式误显示成异常降级。
                pass

            else:
                rejected_count += 1

                issues.append(
                    EvidenceIssue(
                        issue_type=(
                            "missing_rerank_score"
                        ),
                        severity="error",
                        category=task.category,
                        task_id=task_id,
                        chunk_id=chunk_id,
                        message=(
                            "证据没有 rerank_score，"
                            "同时也不是合法的 RRF fallback。"
                        ),
                    )
                )

                continue

            # 12. 所有检查通过后，将证据加入有效列表。
            valid.append(item)

        return (
            valid,
            rejected_count,
            issues,
            fallback_task_ids,
        )

    def _merge_evidence_pool(
        self,
        old_evidence: list[dict[str, Any]],
        new_evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        合并旧证据池和本轮新证据。

        去重 Key：

            category + chunk_id

        为什么不使用 task_id + chunk_id？

            修复任务的 task_id 可能是：

                repair_hotel_reviews

            初始任务的 task_id 可能是：

                hotel_review_candidates

            它们可能检索到同一个 chunk。
            使用 category + chunk_id 可以避免修复任务造成重复证据。
        """

        merged: dict[
            tuple[str, str],
            dict[str, Any],
        ] = {}

        # 1. 先放入旧证据。
        for item in old_evidence:
            key = _evidence_key(
                item
            )

            if key is None:
                continue

            merged[key] = item

        # 2. 再放入本轮新证据。
        for item in new_evidence:
            key = _evidence_key(
                item
            )

            if key is None:
                continue

            old = merged.get(key)

            # 3. 如果是全新证据，直接加入。
            if old is None:
                merged[key] = item
                continue

            # 4. 如果证据已经存在，保留质量更高的一条。
            #
            #    优先比较：
            #        rerank_score
            #
            #    如果没有 Cross-Encoder 分数：
            #        比较 fusion_score
            if _evidence_quality(
                item
            ) > _evidence_quality(old):
                merged[key] = item

        evidence_pool = list(
            merged.values()
        )

        # 5. 按 category 和 rerank 排名稳定排序。
        evidence_pool.sort(
            key=lambda item: (
                CATEGORY_ORDER.get(
                    str(
                        item.get("category")
                    ),
                    99,
                ),
                _evidence_sort_score(
                    item
                ),
                _safe_int(
                    item.get("rerank_rank"),
                    default=10**9,
                ),
                str(
                    item.get("chunk_id")
                ),
            )
        )

        return evidence_pool

    def _grade_coverage(
        self,
        requirements: Mapping[str, Any],
        evidence_pool: list[dict[str, Any]],
        weather_result: Mapping[str, Any],
    ) -> tuple[
        dict[str, EvidenceCoverage],
        list[EvidenceIssue],
    ]:
        """
        按 category 检查证据数量、酒店覆盖和天气风险覆盖。
        """

        coverage: dict[
            str,
            EvidenceCoverage,
        ] = {}

        issues: list[
            EvidenceIssue
        ] = []

        # 1. 从 WeatherNode 结果中读取本次必须覆盖的风险。
        required_weather_risks = (
            _extract_weather_risk_types(
                weather_result
            )
        )

        # 2. 逐个检查 requirements 中的类别。
        for category, raw_requirement in requirements.items():
            if not isinstance(
                raw_requirement,
                Mapping,
            ):
                continue

            required = bool(
                raw_requirement.get(
                    "required",
                    False,
                )
            )

            required_chunks = (
                _safe_int(
                    raw_requirement.get(
                        "min_required_chunks"
                    ),
                    default=0,
                )
            )

            task_ids = _ensure_string_list(
                raw_requirement.get(
                    "task_ids"
                )
            )

            category_evidence = [
                item
                for item in evidence_pool
                if item.get("category")
                == category
            ]

            # 3. 按 chunk_id 去重后统计真实数量。
            unique_chunk_ids = {
                str(
                    item.get("chunk_id")
                )
                for item in category_evidence
                if item.get("chunk_id")
            }

            actual_chunks = len(
                unique_chunk_ids
            )

            required_hotel_ids = (
                _ensure_string_list(
                    raw_requirement.get(
                        "required_hotel_ids"
                    )
                )
            )

            covered_hotel_ids: list[
                str
            ] = []

            # 4. 酒店评价从 metadata.hotel_id 中计算覆盖情况。
            if category == "hotel_reviews":
                covered_hotel_ids = list(
                    dict.fromkeys(
                        str(
                            (
                                item.get(
                                    "metadata"
                                )
                                or {}
                            ).get(
                                "hotel_id"
                            )
                        )
                        for item in category_evidence
                        if (
                            item.get(
                                "metadata"
                            )
                            or {}
                        ).get("hotel_id")
                    )
                )

            missing_hotel_ids = [
                hotel_id
                for hotel_id
                in required_hotel_ids
                if hotel_id
                not in covered_hotel_ids
            ]

            required_risk_types: list[
                str
            ] = []

            covered_risk_types: list[
                str
            ] = []

            # 5. safety_notices 需要覆盖 WeatherNode 中的风险类型。
            if (
                category
                == "safety_notices"
                and required
            ):
                required_risk_types = (
                    required_weather_risks
                )

                covered_risk_types = list(
                    dict.fromkeys(
                        risk_type
                        for item
                        in category_evidence
                        for risk_type
                        in _metadata_risk_types(
                            item.get(
                                "metadata"
                            )
                            or {}
                        )
                    )
                )

            missing_risk_types = [
                risk_type
                for risk_type
                in required_risk_types
                if risk_type
                not in covered_risk_types
            ]

            # 6. Required 类别必须同时满足：
            #
            #    - chunk 数量
            #    - hotel_id 覆盖
            #    - weather risk 覆盖
            #
            #    Optional 类别即使没有证据，也不阻断流程。
            if required:
                category_passed = (
                    actual_chunks
                    >= required_chunks
                    and not missing_hotel_ids
                    and not missing_risk_types
                )
            else:
                category_passed = True

            coverage_item = EvidenceCoverage(
                category=category,  # type: ignore[arg-type]
                required=required,
                required_chunks=(
                    required_chunks
                ),
                actual_chunks=(
                    actual_chunks
                ),
                task_ids=task_ids,
                required_hotel_ids=(
                    required_hotel_ids
                ),
                covered_hotel_ids=(
                    covered_hotel_ids
                ),
                missing_hotel_ids=(
                    missing_hotel_ids
                ),
                required_risk_types=(
                    required_risk_types
                ),
                covered_risk_types=(
                    covered_risk_types
                ),
                missing_risk_types=(
                    missing_risk_types
                ),
                passed=category_passed,
            )

            coverage[
                category
            ] = coverage_item

            # 7. 生成便于 Trace 和 repair 的问题记录。
            if (
                required
                and actual_chunks
                < required_chunks
            ):
                issues.append(
                    EvidenceIssue(
                        issue_type=(
                            "missing_chunks"
                        ),
                        severity="error",
                        category=category,
                        message=(
                            f"{category} 证据数量不足："
                            f"需要 {required_chunks} 条，"
                            f"当前只有 {actual_chunks} 条。"
                        ),
                        details={
                            "required_chunks":
                                required_chunks,

                            "actual_chunks":
                                actual_chunks,
                        },
                    )
                )

            for hotel_id in missing_hotel_ids:
                issues.append(
                    EvidenceIssue(
                        issue_type=(
                            "missing_hotel_evidence"
                        ),
                        severity="error",
                        category=category,
                        message=(
                            f"候选酒店 {hotel_id} 缺少评价证据。"
                        ),
                        details={
                            "hotel_id":
                                hotel_id
                        },
                    )
                )

            for risk_type in missing_risk_types:
                issues.append(
                    EvidenceIssue(
                        issue_type=(
                            "missing_risk_evidence"
                        ),
                        severity="error",
                        category=category,
                        message=(
                            f"天气风险 {risk_type} 缺少安全提醒证据。"
                        ),
                        details={
                            "risk_type":
                                risk_type
                        },
                    )
                )

        return coverage, issues

    def _build_retrieval_feedback(
        self,
        plan: RetrievalPlan,
        coverage: Mapping[
            str,
            EvidenceCoverage,
        ],
        issues: Sequence[EvidenceIssue],
    ) -> dict[str, Any]:
        """
        构造 RetrievalPlanNode repair 模式能够直接读取的反馈。
        """

        missing_doc_types: list[
            str
        ] = []

        missing_hotel_ids: list[
            str
        ] = []

        missing_risk_types: list[
            str
        ] = []

        # 1. 收集所有没有通过的 required category。
        for category, item in coverage.items():
            if not item.required or item.passed:
                continue

            missing_doc_types.append(
                category
            )

            missing_hotel_ids.extend(
                item.missing_hotel_ids
            )

            missing_risk_types.extend(
                item.missing_risk_types
            )

        # 2. 从 error issues 生成简短 reason。
        reason_parts = [
            issue.message
            for issue in issues
            if issue.severity == "error"
        ]

        return {
            "missing_doc_types":
                list(
                    dict.fromkeys(
                        missing_doc_types
                    )
                ),

            "missing_hotel_ids":
                list(
                    dict.fromkeys(
                        missing_hotel_ids
                    )
                ),

            "missing_risk_types":
                list(
                    dict.fromkeys(
                        missing_risk_types
                    )
                ),

            "repair_queries":
                [],

            "source_plan_id":
                plan.plan_id,

            "requested_by":
                "rule_evidence_grade_v1",

            "reason":
                " ".join(
                    reason_parts[:5]
                )
                or "Required RAG 证据不足。",
        }

    def _normalize_requirements(
        self,
        value: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """
        将 coverage_requirements 转为普通可修改 dict。
        """

        result: dict[
            str,
            dict[str, Any],
        ] = {}

        for category, raw in value.items():
            if not isinstance(
                raw,
                Mapping,
            ):
                continue

            result[str(category)] = {
                "required":
                    bool(
                        raw.get(
                            "required",
                            False,
                        )
                    ),

                "min_required_chunks":
                    _safe_int(
                        raw.get(
                            "min_required_chunks"
                        ),
                        default=0,
                    ),

                "task_ids":
                    _ensure_string_list(
                        raw.get(
                            "task_ids"
                        )
                    ),

                "required_hotel_ids":
                    _ensure_string_list(
                        raw.get(
                            "required_hotel_ids"
                        )
                    ),
            }

        return result


def _extract_weather_risk_types(
    weather_result: Mapping[str, Any],
) -> list[str]:
    """
    从 WeatherNode 输出中提取标准天气风险类型。
    """

    result: list[str] = []

    # 1. 读取 weather_result.risks。
    raw_risks = weather_result.get(
        "risks"
    )

    if isinstance(raw_risks, list):
        for item in raw_risks:
            if not isinstance(
                item,
                Mapping,
            ):
                continue

            risk_type = str(
                item.get("risk_type")
                or ""
            )

            if (
                risk_type
                in SUPPORTED_WEATHER_RISK_TYPES
            ):
                result.append(
                    risk_type
                )

    # 2. 同时读取 daily[*].risk_tags。
    daily = weather_result.get(
        "daily"
    )

    if isinstance(daily, list):
        for item in daily:
            if not isinstance(
                item,
                Mapping,
            ):
                continue

            risk_tags = item.get(
                "risk_tags"
            )

            if not isinstance(
                risk_tags,
                list,
            ):
                continue

            for risk_type in risk_tags:
                risk_type = str(
                    risk_type
                )

                if (
                    risk_type
                    in SUPPORTED_WEATHER_RISK_TYPES
                ):
                    result.append(
                        risk_type
                    )

    return list(
        dict.fromkeys(result)
    )


def _metadata_risk_types(
    metadata: Mapping[str, Any],
) -> list[str]:
    """
    从安全文档 metadata 中读取 risk_type。

    同时支持：

        risk_type: rain

    和：

        risk_type:
          - rain
          - wind
    """

    value = metadata.get(
        "risk_type"
    )

    if isinstance(value, list):
        values = value
    elif value is None:
        values = []
    else:
        values = [value]

    return [
        str(item)
        for item in values
        if str(item)
        in SUPPORTED_WEATHER_RISK_TYPES
    ]


def _evidence_key(
    item: Mapping[str, Any],
) -> tuple[str, str] | None:
    """
    生成 Evidence Pool 去重 Key。
    """

    category = str(
        item.get("category")
        or ""
    )

    chunk_id = str(
        item.get("chunk_id")
        or ""
    )

    if not category or not chunk_id:
        return None

    return category, chunk_id


def _evidence_quality(
    item: Mapping[str, Any],
) -> tuple[float, float]:
    """
    用于判断两条重复证据哪一条质量更高。

    优先级：
        rerank_score
        fusion_score

    fallback 没有 rerank_score 时使用 -1，
    因此正常 Cross-Encoder 结果会优先保留。
    """

    rerank_score = item.get(
        "rerank_score"
    )

    if rerank_score is None:
        normalized_rerank = -1.0
    else:
        normalized_rerank = _safe_float(
            rerank_score,
            default=-1.0,
        )

    fusion_score = _safe_float(
        item.get("fusion_score"),
        default=0.0,
    )

    return (
        normalized_rerank,
        fusion_score,
    )


def _evidence_sort_score(
    item: Mapping[str, Any],
) -> float:
    """
    Evidence Pool 排序使用的分数。

    负号用于在 sorted() 中实现降序。
    """

    rerank_score = item.get(
        "rerank_score"
    )

    if rerank_score is not None:
        return -_safe_float(
            rerank_score,
            default=0.0,
        )

    return -_safe_float(
        item.get("fusion_score"),
        default=0.0,
    )


def _ensure_string_list(
    value: Any,
) -> list[str]:
    """
    把任意列表安全转换成字符串列表。
    """

    if not isinstance(
        value,
        list,
    ):
        return []

    return [
        str(item)
        for item in value
        if item is not None
        and str(item).strip()
    ]


def _safe_int(
    value: Any,
    default: int,
) -> int:
    """
    安全转换 int。
    """

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _safe_float(
    value: Any,
    default: float,
) -> float:
    """
    安全转换 float。
    """

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default
