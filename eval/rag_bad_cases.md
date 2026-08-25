# RAG Eval Bad Case Registry

This registry records retrieval failures from production-component RAG
ablations. Preserving a failure is more useful than silently replacing it with
a later aggregate score.

## Current evaluation scope

- Corpus: 222 chunks (v5-agentic-challenge), embedding text
  `v2_metadata_enriched`.
- Scope Regression Set: 108 cases, 84 positive and 24 negative; no current
  variant has a Top-5 miss.
- Semantic Holdout Set: 42 cases, 39 positive and 3 negative. MiniLM is the
  source of the retained semantic failures; BGE Base has no Top-5 miss in the
  current run, but its Decoy metrics remain part of the comparison.
- Metadata Accuracy and Negative Case Accuracy: 1.0000 for all current variants.
- Current source of truth: [MiniLM Scope Report](rag_report.md),
  [MiniLM Semantic Report](rag_challenge_report.md),
  [BGE Scope Report](rag_bge_base_zh_v15_scope_report.md), and
  [BGE Semantic Report](rag_bge_base_zh_v15_semantic_report.md).

## Observed retrieval failures

### Semantic Holdout — MiniLM B_vector_only

The metadata-enriched MiniLM Vector index fixed the prior scope misses, but it
still misses natural-language preference cases such as:

- `rag_challenge_hgh_split_culture` — Gold
  `guide_hgh_agentic_preferences_v1::02::001` is absent from Top-5.
- `rag_challenge_hgh_split_low_walking` — Gold
  `guide_hgh_agentic_preferences_v1::03::001` is absent from Top-5.
- `rag_challenge_sha_combined_family_rain_photo` — both Gold chunks are absent
  from Top-5.

The full, versioned ranked IDs remain in
[`rag_challenge_results.json`](results/rag_challenge_results.json). These are
regression assets for future embedding-text and model comparisons.

## Interpretation

The retained MiniLM failures are semantic-ranking failures, not Metadata-filter
failures: results obey city/doc-type constraints but are broad travel-topic or
deliberately similar decoy chunks rather than direct evidence. The BGE Base
comparison removes these Top-5 misses on the same Holdout, demonstrating that
the issue was primarily first-stage embedding discrimination rather than
metadata filtering or Gold-label validity.

Do not fix this registry by deleting the cases. A future change should first
identify whether the cause is query wording, chunking, embedding drift, fusion,
reranking, or the Gold label; then add the selected case to regression checks
and rerun all four controlled variants.
