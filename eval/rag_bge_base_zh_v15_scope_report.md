# TravelOps RAG Eval Report

## Scope

- Dataset: 108 cases (84 positive, 24 negative).
- Query modes: baseline=108
- Corpus: 222 chunks, version v5-agentic-challenge.
- Startup excluded from query timing: 60596.13 ms.
- Embedding: `models/bge-base-zh-v1.5` on `cpu`.
- Embedding text format: `v2_metadata_enriched`.
- Reranker: `./models/bge-reranker-v2-m3`.

## Ablation Results

| Variant | Hit@5 | Recall@5 | Precision@5 | MRR@5 | nDCG@5 | Metadata | Negative accuracy | Mean ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_bm25_only | 1.0000 | 0.9940 | 0.2262 | 0.9464 | 0.9586 | 1.0000 | 1.0000 | 63.8907 | 96.6421 |
| B_vector_only | 1.0000 | 0.9940 | 0.2262 | 0.9405 | 0.9519 | 1.0000 | 1.0000 | 57.3724 | 81.6982 |
| C_bm25_vector_rrf | 1.0000 | 0.9940 | 0.2262 | 0.9554 | 0.9626 | 1.0000 | 1.0000 | 57.1072 | 81.4157 |
| D_bm25_vector_rrf_rerank | 1.0000 | 0.9940 | 0.2262 | 0.9583 | 0.9636 | 1.0000 | 1.0000 | 3059.8395 | 4363.3925 |

## Candidate-Pool Diagnostics

These metrics use the pre-rerank RRF Top-10 candidate pool. They separate first-stage recall from final Top-5 ranking quality.

| Variant | Candidate Hit@10 | Candidate Recall@10 |
| --- | ---: | ---: |
| A_bm25_only | 1.0000 | 0.9940 |
| B_vector_only | 1.0000 | 0.9940 |
| C_bm25_vector_rrf | 1.0000 | 0.9940 |
| D_bm25_vector_rrf_rerank | 1.0000 | 0.9940 |

## Bad Cases

No retrieval failure met the current Bad Case threshold.
## Interpretation Boundary

Grounded Citation Rate is not reported here because this runner evaluates retrieval. It must be measured from `TripProposal.sources` against the same run's Evidence Pool; treating retrieved chunks themselves as citations would create a meaningless 100% score.
