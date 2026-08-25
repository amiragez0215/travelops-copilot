from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.common.config import settings
from app.rag.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from app.rag.metadata_filter import matching_chunk_ids
from app.rag.reranker import build_rerank_passage
from app.rag.rerank_runtime import initialize_reranker
from app.rag.runtime import initialize_hybrid_retriever
from app.schemas.retrieval_plan_schema import RetrievalPlan, RetrievalTask
from eval.rag_metrics import (
    average,
    hit_at_k,
    metadata_accuracy,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


VARIANT_BM25 = "A_bm25_only"
VARIANT_VECTOR = "B_vector_only"
VARIANT_HYBRID = "C_bm25_vector_rrf"
VARIANT_RERANK = "D_bm25_vector_rrf_rerank"
VARIANT_RERANK_BLEND_025 = "E_rrf_rerank_blend_025"
VARIANT_RERANK_BLEND_050 = "F_rrf_rerank_blend_050"
VARIANT_RERANK_BLEND_075 = "G_rrf_rerank_blend_075"
DEFAULT_VARIANTS = (
    VARIANT_BM25,
    VARIANT_VECTOR,
    VARIANT_HYBRID,
    VARIANT_RERANK,
    VARIANT_RERANK_BLEND_025,
    VARIANT_RERANK_BLEND_050,
    VARIANT_RERANK_BLEND_075,
)
DEFAULT_K_VALUES = (1, 3, 5)


@dataclass(frozen=True)
class RAGEvalCase:
    case_id: str
    category: str
    semantic_query: str
    keyword_query: str
    metadata_filter: dict[str, Any]
    gold_chunk_ids: list[str]
    relevance: dict[str, float]
    expected_doc_types: list[str]
    expected_no_results: bool = False
    top_k: int = 5
    # Challenge Set 的可选分组标注。baseline JSONL 不需要这些字段，
    # 因此保持默认值以兼容原 120 条 Gold Cases。
    challenge_group: str | None = None
    query_mode: str = "baseline"
    raw_message: str | None = None
    need_ids: list[str] = field(default_factory=list)
    decoy_chunk_ids: list[str] = field(default_factory=list)

    @property
    def doc_type(self) -> str:
        # Multi-intent Gold cases may legitimately span guide + safety (or
        # another pair).  The reranker API accepts one primary task type, so
        # use the first labelled type for that stage while the metadata filter
        # and final metrics still evaluate all Gold chunks.
        if not self.expected_doc_types:
            raise ValueError(f"{self.case_id}: expected_doc_types is empty")
        return self.expected_doc_types[0]


@dataclass(frozen=True)
class RetrievalOutcome:
    """Final results plus the RRF candidate pool used by a reranker."""

    chunks: list[dict[str, Any]]
    candidate_chunks: list[dict[str, Any]]


def load_cases(path: str | Path) -> list[RAGEvalCase]:
    """Load and validate the hand-labelled JSONL Gold Dataset."""

    cases: list[RAGEvalCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        case = RAGEvalCase(
            case_id=_required_text(raw, "case_id", line_number),
            category=_required_text(raw, "category", line_number),
            semantic_query=_required_text(raw, "semantic_query", line_number),
            keyword_query=_required_text(raw, "keyword_query", line_number),
            metadata_filter=dict(raw.get("metadata_filter") or {}),
            gold_chunk_ids=[str(item) for item in raw.get("gold_chunk_ids") or []],
            relevance={str(key): float(value) for key, value in (raw.get("relevance") or {}).items()},
            expected_doc_types=[str(item) for item in raw.get("expected_doc_types") or []],
            expected_no_results=bool(raw.get("expected_no_results", False)),
            top_k=int(raw.get("top_k", 5)),
            challenge_group=(
                str(raw["challenge_group"]).strip()
                if raw.get("challenge_group")
                else None
            ),
            query_mode=str(raw.get("query_mode") or "baseline").strip(),
            raw_message=(
                str(raw["raw_message"]).strip()
                if raw.get("raw_message")
                else None
            ),
            need_ids=[str(item) for item in raw.get("need_ids") or []],
            decoy_chunk_ids=[
                str(item) for item in raw.get("decoy_chunk_ids") or []
            ],
        )
        if case.case_id in seen_ids:
            raise ValueError(f"duplicate RAG Eval case_id: {case.case_id}")
        if case.top_k <= 0:
            raise ValueError(f"{case.case_id}: top_k must be positive")
        if case.expected_no_results and case.gold_chunk_ids:
            raise ValueError(f"{case.case_id}: negative cases cannot define gold_chunk_ids")
        if not case.expected_no_results and not case.gold_chunk_ids:
            raise ValueError(f"{case.case_id}: positive cases require gold_chunk_ids")
        if set(case.gold_chunk_ids) - set(case.relevance):
            raise ValueError(f"{case.case_id}: each Gold chunk requires a relevance label")
        if not case.expected_doc_types:
            raise ValueError(f"{case.case_id}: expected_doc_types is required")
        if set(case.gold_chunk_ids).intersection(case.decoy_chunk_ids):
            raise ValueError(
                f"{case.case_id}: Gold chunks and semantic decoys must not overlap"
            )
        seen_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError("RAG Eval dataset is empty")
    return cases


def validate_cases_against_corpus(
    cases: Sequence[RAGEvalCase],
    retriever: HybridRetriever,
) -> None:
    """Fail early when a Gold label no longer matches the versioned corpus."""

    by_chunk_id = {
        str(document.metadata.get("chunk_id")): document
        for document in retriever.documents
    }
    for case in cases:
        for chunk_id in [*case.gold_chunk_ids, *case.decoy_chunk_ids]:
            document = by_chunk_id.get(chunk_id)
            if document is None:
                label = "Gold" if chunk_id in case.gold_chunk_ids else "decoy"
                raise ValueError(
                    f"{case.case_id}: unknown {label} chunk_id {chunk_id}"
                )
            if chunk_id not in case.gold_chunk_ids:
                continue
            actual_type = str(document.metadata.get("doc_type"))
            if actual_type not in case.expected_doc_types:
                raise ValueError(
                    f"{case.case_id}: Gold {chunk_id} has doc_type={actual_type}, "
                    f"expected one of {case.expected_doc_types}"
                )


def run_ablation(
    cases: Sequence[RAGEvalCase],
    *,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    fetch_multiplier: int | None = None,
) -> dict[str, Any]:
    """Run the four retrieval variants against the production corpus and models."""

    unknown = set(variants) - set(DEFAULT_VARIANTS)
    if unknown:
        raise ValueError(f"unknown RAG Eval variants: {sorted(unknown)}")

    startup_started = time.perf_counter()
    retriever = initialize_hybrid_retriever(force_reload=True)
    if fetch_multiplier is not None:
        if fetch_multiplier <= 0:
            raise ValueError("fetch_multiplier must be positive")
        retriever.fetch_multiplier = fetch_multiplier
    rerank_variants = {
        VARIANT_RERANK,
        VARIANT_RERANK_BLEND_025,
        VARIANT_RERANK_BLEND_050,
        VARIANT_RERANK_BLEND_075,
    }
    reranker = initialize_reranker() if rerank_variants.intersection(variants) else None
    startup_ms = (time.perf_counter() - startup_started) * 1000
    validate_cases_against_corpus(cases, retriever)

    # Warm up the local embedding model and Cross-Encoder outside measured cases.
    _warm_up(retriever, reranker)

    observations: list[dict[str, Any]] = []
    for variant in variants:
        for case in cases:
            started = time.perf_counter()
            outcome = _retrieve_variant(
                retriever=retriever,
                reranker=reranker,
                case=case,
                variant=variant,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            observations.append(
                _observe_case(
                    case,
                    variant,
                    outcome.chunks,
                    latency_ms,
                    k_values,
                    candidate_chunks=outcome.candidate_chunks,
                )
            )

    report = {
        "schema_version": "rag_eval_v1",
        "dataset_case_count": len(cases),
        "positive_case_count": sum(not case.expected_no_results for case in cases),
        "negative_case_count": sum(case.expected_no_results for case in cases),
        "dataset_slices": {
            "query_modes": dict(
                sorted(Counter(case.query_mode for case in cases).items())
            ),
            "challenge_group_count": len(
                {
                    case.challenge_group
                    for case in cases
                    if case.challenge_group
                }
            ),
        },
        "k_values": list(k_values),
        "startup_ms": round(startup_ms, 2),
        "runtime": {
            "corpus_chunk_count": len(retriever.documents),
            # Read the version from the same application settings used by the
            # production retriever so an evaluation report cannot claim a stale
            # corpus version after a fixture migration.
            "corpus_version": settings.rag_corpus_version,
            "embedding_model": settings.rag_embedding_model,
            "reranker_model": settings.rag_rerank_model,
            "rrf_k": settings.rag_rrf_k,
            "fetch_multiplier": retriever.fetch_multiplier,
            "device": settings.rag_embedding_device,
        },
        "variants": _summarize_variants(observations, variants, k_values),
        "query_mode_metrics": _summarize_query_modes(observations, variants),
        "observations": observations,
    }
    report["bad_cases"] = _find_bad_cases(observations)
    report["citation_metric"] = {
        "status": "not_measured",
        "reason": (
            "Grounded Citation Rate requires final Proposal.sources and the exact "
            "Evidence Pool from an end-to-end workflow run. It is implemented in "
            "eval.rag_metrics but intentionally not fabricated from retrieval output."
        ),
    }
    return report


def write_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Persist machine-readable results and a reviewable Markdown report."""

    output_json = Path(json_path)
    output_markdown = Path(markdown_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise, auditable experiment report from measured output."""

    lines = [
        "# TravelOps RAG Eval Report",
        "",
        "## Scope",
        "",
        f"- Dataset: {report['dataset_case_count']} cases "
        f"({report['positive_case_count']} positive, {report['negative_case_count']} negative).",
        "- Query modes: "
        + ", ".join(
            f"{name}={count}"
            for name, count in report.get("dataset_slices", {})
            .get("query_modes", {})
            .items()
        ),
        f"- Corpus: {report['runtime']['corpus_chunk_count']} chunks, "
        f"version {report['runtime']['corpus_version']}.",
        f"- Startup excluded from query timing: {report['startup_ms']} ms.",
        f"- Embedding: `{report['runtime']['embedding_model']}` on "
        f"`{report['runtime']['device']}`.",
        f"- Reranker: `{report['runtime']['reranker_model']}`.",
        "",
        "## Ablation Results",
        "",
        "| Variant | Hit@5 | Recall@5 | Precision@5 | MRR@5 | nDCG@5 | Metadata | Negative accuracy | Mean ms | P95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in report["variants"].items():
        lines.append(
            "| {name} | {hit} | {recall} | {precision} | {mrr} | {ndcg} | {metadata} | {negative} | {mean_ms} | {p95_ms} |".format(
                name=name,
                hit=_display(summary["metrics"].get("hit_at_5")),
                recall=_display(summary["metrics"].get("recall_at_5")),
                precision=_display(summary["metrics"].get("precision_at_5")),
                mrr=_display(summary["metrics"].get("mrr_at_5")),
                ndcg=_display(summary["metrics"].get("ndcg_at_5")),
                metadata=_display(summary["metadata_accuracy"]),
                negative=_display(summary["negative_case_accuracy"]),
                mean_ms=_display(summary["mean_latency_ms"]),
                p95_ms=_display(summary["p95_latency_ms"]),
            )
        )

    query_mode_metrics = report.get("query_mode_metrics") or {}
    if query_mode_metrics:
        lines.extend(
            [
                "",
                "## Challenge Query Modes",
                "",
                "`combined` 代表未拆分的多偏好 query，`split_need` 代表 "
                "Agentic Planner 可以生成的单 need query。Decoy Hit@5 越低越好。",
                "",
                "| Variant | Query mode | Positive | Hit@5 | Recall@5 | Decoy Hit@5 | Negative accuracy |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for variant, modes in query_mode_metrics.items():
            for mode, summary in modes.items():
                lines.append(
                    "| {variant} | {mode} | {positive} | {hit} | {recall} | "
                    "{decoy} | {negative} |".format(
                        variant=variant,
                        mode=mode,
                        positive=summary["positive_case_count"],
                        hit=_display(summary["hit_at_5"]),
                        recall=_display(summary["recall_at_5"]),
                        decoy=_display(summary["decoy_hit_at_5"]),
                        negative=_display(summary["negative_case_accuracy"]),
                    )
                )

    lines.extend(
        [
            "",
            "## Candidate-Pool Diagnostics",
            "",
            "These metrics use the pre-rerank RRF Top-10 candidate pool. They separate first-stage recall from final Top-5 ranking quality.",
            "",
            "| Variant | Candidate Hit@10 | Candidate Recall@10 |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, summary in report["variants"].items():
        lines.append(
            "| {name} | {hit} | {recall} |".format(
                name=name,
                hit=_display(summary.get("candidate_hit_at_10")),
                recall=_display(summary.get("candidate_recall_at_10")),
            )
        )

    lines.extend(["", "## Bad Cases", ""])
    bad_cases = report.get("bad_cases") or []
    if not bad_cases:
        lines.append("No retrieval failure met the current Bad Case threshold.")
    else:
        for item in bad_cases:
            lines.extend(
                [
                    f"### {item['case_id']} ({item['variant']})",
                    "",
                    f"- Reason: {item['reason']}",
                    f"- Top results: {', '.join(item['ranked_chunk_ids'][:5]) or '(none)'}",
                    f"- Gold: {', '.join(item['gold_chunk_ids']) or '(negative case)'}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "Grounded Citation Rate is not reported here because this runner evaluates retrieval. "
            "It must be measured from `TripProposal.sources` against the same run's Evidence Pool; "
            "treating retrieved chunks themselves as citations would create a meaningless 100% score.",
            "",
        ]
    )
    return "\n".join(lines)


def _retrieve_variant(
    *,
    retriever: HybridRetriever,
    reranker: Any | None,
    case: RAGEvalCase,
    variant: str,
) -> RetrievalOutcome:
    candidate_ids = matching_chunk_ids(retriever.documents, case.metadata_filter)
    if not candidate_ids:
        return RetrievalOutcome(chunks=[], candidate_chunks=[])

    fetch_k = min(max(case.top_k * retriever.fetch_multiplier, case.top_k), len(candidate_ids))
    bm25_hits = retriever.bm25_index.search(case.keyword_query, candidate_ids, fetch_k)
    vector_hits = retriever.vector_store.search(case.semantic_query, candidate_ids, fetch_k)

    if variant == VARIANT_BM25:
        fused = reciprocal_rank_fusion(
            task_id=case.case_id,
            bm25_hits=bm25_hits,
            vector_hits=[],
            rrf_k=retriever.rrf_k,
            top_k=case.top_k,
        )
        chunks = [item.to_state_dict() for item in fused]
        return RetrievalOutcome(chunks=chunks, candidate_chunks=chunks)

    if variant == VARIANT_VECTOR:
        fused = reciprocal_rank_fusion(
            task_id=case.case_id,
            bm25_hits=[],
            vector_hits=vector_hits,
            rrf_k=retriever.rrf_k,
            top_k=case.top_k,
        )
        chunks = [item.to_state_dict() for item in fused]
        return RetrievalOutcome(chunks=chunks, candidate_chunks=chunks)

    fused = reciprocal_rank_fusion(
        task_id=case.case_id,
        bm25_hits=bm25_hits,
        vector_hits=vector_hits,
        rrf_k=retriever.rrf_k,
        bm25_weight=retriever.bm25_weight,
        vector_weight=retriever.vector_weight,
        top_k=fetch_k,
    )
    fused_dicts = [item.to_state_dict() for item in fused]
    if variant == VARIANT_HYBRID:
        return RetrievalOutcome(
            chunks=fused_dicts[:case.top_k],
            candidate_chunks=fused_dicts,
        )
    if reranker is None:
        raise RuntimeError("Reranker was not initialized for a rerank variant")

    task = _task_for_case(case, top_k=fetch_k)
    plan = RetrievalPlan(
        plan_id=f"eval_{case.case_id}",
        city=str(case.metadata_filter.get("city") or "general"),
        tasks=[task],
    )
    if variant == VARIANT_RERANK:
        result = reranker.rerank_plan(plan, fused_dicts)
        return RetrievalOutcome(
            chunks=list(result.get("reranked_evidence") or [])[:case.top_k],
            candidate_chunks=fused_dicts,
        )

    alpha_by_variant = {
        VARIANT_RERANK_BLEND_025: 0.25,
        VARIANT_RERANK_BLEND_050: 0.50,
        VARIANT_RERANK_BLEND_075: 0.75,
    }
    alpha = alpha_by_variant.get(variant)
    if alpha is None:
        raise ValueError(f"unknown RAG Eval variant: {variant}")
    return RetrievalOutcome(
        chunks=_blend_rerank_with_rrf(
            reranker=reranker,
            query=case.semantic_query,
            candidates=fused_dicts,
            top_k=case.top_k,
            alpha=alpha,
        ),
        candidate_chunks=fused_dicts,
    )


def _task_for_case(case: RAGEvalCase, *, top_k: int) -> RetrievalTask:
    category_by_doc_type = {
        "guide": "guides",
        "hotel_reviews": "hotel_reviews",
        "safety_notice": "safety_notices",
        "packing_checklist": "packing_checklists",
    }
    primary_doc_type = case.doc_type
    return RetrievalTask(
        task_id=case.case_id,
        category=category_by_doc_type[primary_doc_type],
        doc_type=primary_doc_type,
        purpose="offline_rag_eval",
        semantic_query=case.semantic_query,
        keyword_query=case.keyword_query,
        metadata_filter=case.metadata_filter,
        top_k=top_k,
    )


def _blend_rerank_with_rrf(
    *,
    reranker: Any,
    query: str,
    candidates: Sequence[Mapping[str, Any]],
    top_k: int,
    alpha: float,
) -> list[dict[str, Any]]:
    """Blend normalized Cross-Encoder and RRF scores for offline ablation only."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    records = [dict(candidate) for candidate in candidates]
    pair_scores = reranker.scorer.score_pairs(
        [(query, build_rerank_passage(candidate)) for candidate in records]
    )
    if len(pair_scores) != len(records):
        raise RuntimeError("Cross-Encoder score count does not match candidate count")

    fusion_scores = [float(record.get("fusion_score") or 0.0) for record in records]
    low, high = min(fusion_scores, default=0.0), max(fusion_scores, default=0.0)
    score_range = high - low
    for record, pair_score in zip(records, pair_scores):
        fusion_score = float(record.get("fusion_score") or 0.0)
        normalized_fusion = 1.0 if score_range == 0 else (fusion_score - low) / score_range
        rerank_score = float(pair_score.normalized_score)
        record["rerank_score"] = round(rerank_score, 8)
        record["rerank_raw_score"] = round(float(pair_score.raw_score), 8)
        record["blend_alpha"] = alpha
        record["blended_score"] = round(alpha * rerank_score + (1.0 - alpha) * normalized_fusion, 8)
    records.sort(
        key=lambda record: (
            -float(record["blended_score"]),
            -float(record.get("fusion_score") or 0.0),
            int(record.get("final_rank") or 0),
            str(record.get("chunk_id") or ""),
        )
    )
    return records[:top_k]


def _observe_case(
    case: RAGEvalCase,
    variant: str,
    chunks: Sequence[Mapping[str, Any]],
    latency_ms: float,
    k_values: Sequence[int],
    *,
    candidate_chunks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ranked_ids = [str(chunk.get("chunk_id") or "") for chunk in chunks]
    candidate_ids = [str(chunk.get("chunk_id") or "") for chunk in candidate_chunks]
    ranked_metadata = [dict(chunk.get("metadata") or {}) for chunk in chunks]
    metrics: dict[str, float | None] = {}
    for k in k_values:
        metrics[f"hit_at_{k}"] = hit_at_k(ranked_ids, case.gold_chunk_ids, k)
        metrics[f"recall_at_{k}"] = recall_at_k(ranked_ids, case.gold_chunk_ids, k)
        metrics[f"precision_at_{k}"] = precision_at_k(ranked_ids, case.gold_chunk_ids, k)
        metrics[f"mrr_at_{k}"] = reciprocal_rank_at_k(ranked_ids, case.gold_chunk_ids, k)
        metrics[f"ndcg_at_{k}"] = ndcg_at_k(ranked_ids, case.relevance, k)

    negative_accuracy = None
    if case.expected_no_results:
        negative_accuracy = float(not ranked_ids)
    return {
        "case_id": case.case_id,
        "category": case.category,
        "challenge_group": case.challenge_group,
        "query_mode": case.query_mode,
        "need_ids": case.need_ids,
        "variant": variant,
        "expected_no_results": case.expected_no_results,
        "gold_chunk_ids": case.gold_chunk_ids,
        "decoy_chunk_ids": case.decoy_chunk_ids,
        "ranked_chunk_ids": ranked_ids,
        "candidate_chunk_ids": candidate_ids,
        "metrics": metrics,
        "metadata_accuracy": metadata_accuracy(
            ranked_metadata,
            case.metadata_filter,
            expected_no_results=case.expected_no_results,
        ),
        "negative_case_accuracy": negative_accuracy,
        "candidate_metrics": {
            "candidate_hit_at_10": hit_at_k(candidate_ids, case.gold_chunk_ids, 10),
            "candidate_recall_at_10": recall_at_k(candidate_ids, case.gold_chunk_ids, 10),
        },
        "decoy_hit_at_5": (
            float(bool(set(ranked_ids[:5]).intersection(case.decoy_chunk_ids)))
            if case.decoy_chunk_ids
            else None
        ),
        "latency_ms": round(latency_ms, 3),
    }


def _summarize_variants(
    observations: Sequence[Mapping[str, Any]],
    variants: Sequence[str],
    k_values: Sequence[int],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for variant in variants:
        rows = [row for row in observations if row["variant"] == variant]
        positive_rows = [row for row in rows if not row["expected_no_results"]]
        negative_rows = [row for row in rows if row["expected_no_results"]]
        metrics = {
            f"{name}_at_{k}": average(row["metrics"][f"{name}_at_{k}"] for row in positive_rows)
            for name in ("hit", "recall", "precision", "mrr", "ndcg")
            for k in k_values
        }
        summaries[variant] = {
            "positive_case_count": len(positive_rows),
            "negative_case_count": len(negative_rows),
            "metrics": metrics,
            "metadata_accuracy": average(row["metadata_accuracy"] for row in rows),
            "negative_case_accuracy": average(row["negative_case_accuracy"] for row in negative_rows),
            "candidate_hit_at_10": average(
                row["candidate_metrics"]["candidate_hit_at_10"]
                for row in positive_rows
            ),
            "candidate_recall_at_10": average(
                row["candidate_metrics"]["candidate_recall_at_10"]
                for row in positive_rows
            ),
            "mean_latency_ms": average(row["latency_ms"] for row in rows),
            "p95_latency_ms": percentile((row["latency_ms"] for row in rows), 95),
        }
    return summaries


def _summarize_query_modes(
    observations: Sequence[Mapping[str, Any]],
    variants: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """按 Challenge Set 的 query_mode 汇总，直接比较 combined 与 split。"""

    if not any(row.get("query_mode") != "baseline" for row in observations):
        return {}

    output: dict[str, dict[str, Any]] = {}
    for variant in variants:
        variant_rows = [
            row for row in observations if row["variant"] == variant
        ]
        mode_summaries: dict[str, Any] = {}
        for mode in sorted({str(row["query_mode"]) for row in variant_rows}):
            rows = [row for row in variant_rows if row["query_mode"] == mode]
            positive = [row for row in rows if not row["expected_no_results"]]
            negative = [row for row in rows if row["expected_no_results"]]
            mode_summaries[mode] = {
                "case_count": len(rows),
                "positive_case_count": len(positive),
                "hit_at_5": average(
                    row["metrics"].get("hit_at_5") for row in positive
                ),
                "recall_at_5": average(
                    row["metrics"].get("recall_at_5") for row in positive
                ),
                "decoy_hit_at_5": average(
                    row.get("decoy_hit_at_5") for row in positive
                ),
                "negative_case_accuracy": average(
                    row.get("negative_case_accuracy") for row in negative
                ),
            }
        output[variant] = mode_summaries
    return output


def _find_bad_cases(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in observations:
        if row["expected_no_results"]:
            if row["negative_case_accuracy"] == 1:
                continue
            reason = "metadata filter returned results for a negative case"
        elif row["metrics"].get("hit_at_5") == 1:
            continue
        else:
            reason = "no Gold chunk appeared in Top-5"
        failures.append(
            {
                "case_id": row["case_id"],
                "variant": row["variant"],
                "reason": reason,
                "gold_chunk_ids": row["gold_chunk_ids"],
                "ranked_chunk_ids": row["ranked_chunk_ids"],
            }
        )
    return failures[:5]


def _warm_up(retriever: HybridRetriever, reranker: Any | None) -> None:
    candidate_ids = [str(document.metadata["chunk_id"]) for document in retriever.documents[:8]]
    if candidate_ids:
        retriever.vector_store.search("成都旅行 RAG 预热", candidate_ids, min(3, len(candidate_ids)))
    if reranker is not None:
        # The model is loaded lazily when the first pair is scored.
        reranker.scorer.score_pairs([("成都旅行预热", "成都旅行评测预热文本")])


def _required_text(raw: Mapping[str, Any], key: str, line_number: int) -> str:
    value = str(raw.get(key) or "").strip()
    if not value:
        raise ValueError(f"line {line_number}: {key} is required")
    return value


def _display(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"
