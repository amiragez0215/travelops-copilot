# RAG Eval Bad Case Registry

This registry records the failures from the current production-component RAG
ablation. Preserving a failure is more useful than silently replacing it with a
later aggregate score.

## Current evaluation scope

- Corpus: 222 chunks (v5-agentic-challenge).
- Dataset: 120 baseline cases: 96 positive and 24 negative.
- Variants: BM25 only, Vector only, BM25 + Vector + RRF, and Hybrid + BGE Rerank.
- Metadata Accuracy: 1.0000 for all variants.
- Negative Case Accuracy: 1.0000 for all variants.
- Current source of truth: [RAG Eval Report](rag_report.md).

## Observed retrieval failures

### rag_scope_guides_pek_03 — B_vector_only

- Query: 北京展览预约应该怎么安排和注意什么
- Metadata filter: city=北京, doc_type=guide
- Failure: the Gold Chunk did not appear in Top-5.
- Gold: guide_pek_scope_v1::03::001
- Vector Top-5: guide_pek_scope_v1::01::001, guide_pek_scope_v1::08::001,
  guide_pek_agentic_decoys_v1::01::001,
  guide_pek_agentic_preferences_v1::03::001,
  guide_pek_agentic_decoys_v1::03::001

### rag_scope_guides_pek_07 — B_vector_only

- Query: 北京夜间活动应该怎么安排和注意什么
- Metadata filter: city=北京, doc_type=guide
- Failure: the Gold Chunk did not appear in Top-5.
- Gold: guide_pek_scope_v1::07::001
- Vector Top-5: guide_pek_agentic_decoys_v1::01::001,
  guide_pek_agentic_decoys_v1::03::001,
  guide_pek_agentic_preferences_v1::05::001,
  guide_pek_scope_v1::01::001,
  guide_pek_agentic_preferences_v1::03::001

## Interpretation

The two failures are semantic-ranking failures, not Metadata-filter failures:
the retrieved chunks obey the Beijing Guides filter but are broad travel-topic
or deliberately similar decoy chunks rather than direct evidence. They explain
why Vector only has Hit@5=0.9792, MRR@5=0.7005, and nDCG@5=0.7707 in the
current report, while BM25 preserves exact-term ranking for this corpus.

Do not fix this registry by deleting the cases. A future change should first
identify whether the cause is query wording, chunking, embedding drift, fusion,
reranking, or the Gold label; then add the selected case to regression checks
and rerun all four controlled variants.
