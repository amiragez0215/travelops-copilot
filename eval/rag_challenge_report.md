# TravelOps RAG Eval Report

## Scope

- Dataset: 42 cases (39 positive, 3 negative).
- Query modes: combined=9, implicit=3, multi_attribute=6, multi_risk=3, negative=3, split_need=18
- Corpus: 222 chunks, version v5-agentic-challenge.
- Startup excluded from query timing: 4924.28 ms.
- Embedding: `./models/paraphrase-multilingual-MiniLM-L12-v2` on `cpu`.
- Embedding text format: `v2_metadata_enriched`.
- Reranker: `./models/bge-reranker-v2-m3`.

## Ablation Results

| Variant | Hit@5 | Recall@5 | Precision@5 | MRR@5 | nDCG@5 | Metadata | Negative accuracy | Mean ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_bm25_only | 1.0000 | 0.8932 | 0.2974 | 0.8974 | 0.8484 | 1.0000 | 1.0000 | 31.0867 | 38.5715 |
| B_vector_only | 0.8205 | 0.6880 | 0.2615 | 0.5252 | 0.5196 | 1.0000 | 1.0000 | 30.7705 | 38.9446 |
| C_bm25_vector_rrf | 1.0000 | 0.9038 | 0.3128 | 0.7201 | 0.7217 | 1.0000 | 1.0000 | 31.2369 | 37.2676 |
| D_bm25_vector_rrf_rerank | 1.0000 | 0.9338 | 0.3282 | 0.9274 | 0.9049 | 1.0000 | 1.0000 | 4043.7245 | 5436.4448 |

## Challenge Query Modes

`combined` 代表未拆分的多偏好 query，`split_need` 代表 Agentic Planner 可以生成的单 need query。Decoy Hit@5 越低越好。

| Variant | Query mode | Positive | Hit@5 | Recall@5 | Decoy Hit@5 | Negative accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A_bm25_only | combined | 9 | 1.0000 | 0.8889 | 0.8889 | N/A |
| A_bm25_only | implicit | 3 | 1.0000 | 0.4444 | 0.6667 | N/A |
| A_bm25_only | multi_attribute | 6 | 1.0000 | 1.0000 | N/A | N/A |
| A_bm25_only | multi_risk | 3 | 1.0000 | 0.5000 | 1.0000 | N/A |
| A_bm25_only | negative | 0 | N/A | N/A | N/A | 1.0000 |
| A_bm25_only | split_need | 18 | 1.0000 | 1.0000 | 1.0000 | N/A |
| B_vector_only | combined | 9 | 0.8889 | 0.5000 | 0.5556 | N/A |
| B_vector_only | implicit | 3 | 1.0000 | 0.4444 | 0.0000 | N/A |
| B_vector_only | multi_attribute | 6 | 1.0000 | 1.0000 | N/A | N/A |
| B_vector_only | multi_risk | 3 | 1.0000 | 1.0000 | 0.0000 | N/A |
| B_vector_only | negative | 0 | N/A | N/A | N/A | 1.0000 |
| B_vector_only | split_need | 18 | 0.6667 | 0.6667 | 0.2222 | N/A |
| C_bm25_vector_rrf | combined | 9 | 1.0000 | 0.8333 | 0.6667 | N/A |
| C_bm25_vector_rrf | implicit | 3 | 1.0000 | 0.3333 | 0.3333 | N/A |
| C_bm25_vector_rrf | multi_attribute | 6 | 1.0000 | 1.0000 | N/A | N/A |
| C_bm25_vector_rrf | multi_risk | 3 | 1.0000 | 0.9167 | 1.0000 | N/A |
| C_bm25_vector_rrf | negative | 0 | N/A | N/A | N/A | 1.0000 |
| C_bm25_vector_rrf | split_need | 18 | 1.0000 | 1.0000 | 0.7222 | N/A |
| D_bm25_vector_rrf_rerank | combined | 9 | 1.0000 | 0.9630 | 0.5556 | N/A |
| D_bm25_vector_rrf_rerank | implicit | 3 | 1.0000 | 0.3333 | 0.3333 | N/A |
| D_bm25_vector_rrf_rerank | multi_attribute | 6 | 1.0000 | 1.0000 | N/A | N/A |
| D_bm25_vector_rrf_rerank | multi_risk | 3 | 1.0000 | 0.9167 | 1.0000 | N/A |
| D_bm25_vector_rrf_rerank | negative | 0 | N/A | N/A | N/A | 1.0000 |
| D_bm25_vector_rrf_rerank | split_need | 18 | 1.0000 | 1.0000 | 0.7222 | N/A |

## Candidate-Pool Diagnostics

These metrics use the pre-rerank RRF Top-10 candidate pool. They separate first-stage recall from final Top-5 ranking quality.

| Variant | Candidate Hit@10 | Candidate Recall@10 |
| --- | ---: | ---: |
| A_bm25_only | 1.0000 | 0.9017 |
| B_vector_only | 0.8205 | 0.7222 |
| C_bm25_vector_rrf | 1.0000 | 0.9423 |
| D_bm25_vector_rrf_rerank | 1.0000 | 0.9423 |

## Bad Cases

### rag_challenge_hgh_split_culture (B_vector_only)

- Reason: no Gold chunk appeared in Top-5
- Top results: guide_hgh_scope_v1::01::001, guide_hgh_scope_v1::05::001, guide_hgh_scope_v1::07::001, guide_hgh_scope_v1::04::001, guide_hgh_scope_v1::03::001
- Gold: guide_hgh_agentic_preferences_v1::02::001

### rag_challenge_hgh_split_low_walking (B_vector_only)

- Reason: no Gold chunk appeared in Top-5
- Top results: guide_hgh_agentic_preferences_v1::05::001, guide_hgh_scope_v1::07::001, guide_hgh_scope_v1::02::001, guide_hgh_scope_v1::08::001, guide_hgh_agentic_decoys_v1::02::001
- Gold: guide_hgh_agentic_preferences_v1::03::001

### rag_challenge_hgh_split_photography (B_vector_only)

- Reason: no Gold chunk appeared in Top-5
- Top results: guide_hgh_scope_v1::03::001, guide_hgh_scope_v1::06::001, guide_hgh_scope_v1::07::001, guide_hgh_scope_v1::01::001, guide_hgh_scope_v1::02::001
- Gold: guide_hgh_agentic_preferences_v1::06::001

### rag_challenge_sha_combined_family_rain_photo (B_vector_only)

- Reason: no Gold chunk appeared in Top-5
- Top results: guide_sha_agentic_decoys_v1::03::001, guide_sha_scope_v1::06::001, guide_sha_agentic_decoys_v1::04::001, guide_sha_scope_v1::05::001, guide_sha_scope_v1::08::001
- Gold: guide_sha_agentic_preferences_v1::04::001, guide_sha_agentic_preferences_v1::06::001

### rag_challenge_sha_split_low_walking (B_vector_only)

- Reason: no Gold chunk appeared in Top-5
- Top results: guide_sha_scope_v1::08::001, guide_sha_agentic_preferences_v1::05::001, guide_sha_scope_v1::05::001, guide_sha_scope_v1::06::001, guide_sha_agentic_decoys_v1::03::001
- Gold: guide_sha_agentic_preferences_v1::03::001

## Interpretation Boundary

Grounded Citation Rate is not reported here because this runner evaluates retrieval. It must be measured from `TripProposal.sources` against the same run's Evidence Pool; treating retrieved chunks themselves as citations would create a meaningless 100% score.
