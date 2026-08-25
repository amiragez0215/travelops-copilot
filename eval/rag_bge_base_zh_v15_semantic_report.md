# TravelOps RAG Eval Report

## Scope

- Dataset: 42 cases (39 positive, 3 negative).
- Query modes: combined=9, implicit=3, multi_attribute=6, multi_risk=3, negative=3, split_need=18
- Corpus: 222 chunks, version v5-agentic-challenge.
- Startup excluded from query timing: 4771.06 ms.
- Embedding: `models/bge-base-zh-v1.5` on `cpu`.
- Embedding text format: `v2_metadata_enriched`.
- Reranker: `./models/bge-reranker-v2-m3`.

## Ablation Results

| Variant | Hit@5 | Recall@5 | Precision@5 | MRR@5 | nDCG@5 | Metadata | Negative accuracy | Mean ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_bm25_only | 1.0000 | 0.8932 | 0.2974 | 0.8974 | 0.8484 | 1.0000 | 1.0000 | 88.2259 | 110.8347 |
| B_vector_only | 1.0000 | 0.9017 | 0.3026 | 0.9573 | 0.8913 | 1.0000 | 1.0000 | 79.9958 | 98.1791 |
| C_bm25_vector_rrf | 1.0000 | 0.9103 | 0.3077 | 0.9487 | 0.9133 | 1.0000 | 1.0000 | 77.4194 | 94.2096 |
| D_bm25_vector_rrf_rerank | 1.0000 | 0.9188 | 0.3128 | 0.9316 | 0.9047 | 1.0000 | 1.0000 | 4076.6568 | 5332.7655 |

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
| B_vector_only | combined | 9 | 1.0000 | 0.8889 | 0.3333 | N/A |
| B_vector_only | implicit | 3 | 1.0000 | 0.5556 | 0.0000 | N/A |
| B_vector_only | multi_attribute | 6 | 1.0000 | 1.0000 | N/A | N/A |
| B_vector_only | multi_risk | 3 | 1.0000 | 0.5000 | 1.0000 | N/A |
| B_vector_only | negative | 0 | N/A | N/A | N/A | 1.0000 |
| B_vector_only | split_need | 18 | 1.0000 | 1.0000 | 0.3889 | N/A |
| C_bm25_vector_rrf | combined | 9 | 1.0000 | 0.9259 | 0.6667 | N/A |
| C_bm25_vector_rrf | implicit | 3 | 1.0000 | 0.5556 | 0.3333 | N/A |
| C_bm25_vector_rrf | multi_attribute | 6 | 1.0000 | 1.0000 | N/A | N/A |
| C_bm25_vector_rrf | multi_risk | 3 | 1.0000 | 0.5000 | 1.0000 | N/A |
| C_bm25_vector_rrf | negative | 0 | N/A | N/A | N/A | 1.0000 |
| C_bm25_vector_rrf | split_need | 18 | 1.0000 | 1.0000 | 0.8889 | N/A |
| D_bm25_vector_rrf_rerank | combined | 9 | 1.0000 | 0.9259 | 0.7778 | N/A |
| D_bm25_vector_rrf_rerank | implicit | 3 | 1.0000 | 0.6667 | 0.6667 | N/A |
| D_bm25_vector_rrf_rerank | multi_attribute | 6 | 1.0000 | 1.0000 | N/A | N/A |
| D_bm25_vector_rrf_rerank | multi_risk | 3 | 1.0000 | 0.5000 | 1.0000 | N/A |
| D_bm25_vector_rrf_rerank | negative | 0 | N/A | N/A | N/A | 1.0000 |
| D_bm25_vector_rrf_rerank | split_need | 18 | 1.0000 | 1.0000 | 0.8889 | N/A |

## Candidate-Pool Diagnostics

These metrics use the pre-rerank RRF Top-10 candidate pool. They separate first-stage recall from final Top-5 ranking quality.

| Variant | Candidate Hit@10 | Candidate Recall@10 |
| --- | ---: | ---: |
| A_bm25_only | 1.0000 | 0.9017 |
| B_vector_only | 1.0000 | 0.9188 |
| C_bm25_vector_rrf | 1.0000 | 0.9188 |
| D_bm25_vector_rrf_rerank | 1.0000 | 0.9188 |

## Bad Cases

No retrieval failure met the current Bad Case threshold.
## Interpretation Boundary

Grounded Citation Rate is not reported here because this runner evaluates retrieval. It must be measured from `TripProposal.sources` against the same run's Evidence Pool; treating retrieved chunks themselves as citations would create a meaningless 100% score.
