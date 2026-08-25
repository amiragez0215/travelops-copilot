# TravelOps RAG Eval Report

## Scope

- Dataset: 120 cases (96 positive, 24 negative).
- Query modes: baseline=120
- Corpus: 222 chunks, version v5-agentic-challenge.
- Startup excluded from query timing: 5199.38 ms.
- Embedding: `./models/paraphrase-multilingual-MiniLM-L12-v2` on `cpu`.
- Reranker: `./models/bge-reranker-v2-m3`.

## Ablation Results

| Variant | Hit@5 | Recall@5 | Precision@5 | MRR@5 | nDCG@5 | Metadata | Negative accuracy | Mean ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_bm25_only | 1.0000 | 1.0000 | 0.2000 | 0.8187 | 0.8627 | 1.0000 | 1.0000 | 23.8662 | 32.0590 |
| B_vector_only | 0.9792 | 0.9792 | 0.1958 | 0.7005 | 0.7707 | 1.0000 | 1.0000 | 23.1212 | 30.6114 |
| C_bm25_vector_rrf | 1.0000 | 1.0000 | 0.2000 | 0.7549 | 0.8167 | 1.0000 | 1.0000 | 24.4301 | 33.2403 |
| D_bm25_vector_rrf_rerank | 1.0000 | 1.0000 | 0.2000 | 0.8592 | 0.8945 | 1.0000 | 1.0000 | 2671.1939 | 4226.7146 |

## Candidate-Pool Diagnostics

These metrics use the pre-rerank RRF Top-10 candidate pool. They separate first-stage recall from final Top-5 ranking quality.

| Variant | Candidate Hit@10 | Candidate Recall@10 |
| --- | ---: | ---: |
| A_bm25_only | 1.0000 | 1.0000 |
| B_vector_only | 0.9792 | 0.9792 |
| C_bm25_vector_rrf | 1.0000 | 1.0000 |
| D_bm25_vector_rrf_rerank | 1.0000 | 1.0000 |

## Bad Cases

### rag_scope_guides_pek_03 (B_vector_only)

- Reason: no Gold chunk appeared in Top-5
- Top results: guide_pek_scope_v1::01::001, guide_pek_scope_v1::08::001, guide_pek_agentic_decoys_v1::01::001, guide_pek_agentic_preferences_v1::03::001, guide_pek_agentic_decoys_v1::03::001
- Gold: guide_pek_scope_v1::03::001

### rag_scope_guides_pek_07 (B_vector_only)

- Reason: no Gold chunk appeared in Top-5
- Top results: guide_pek_agentic_decoys_v1::01::001, guide_pek_agentic_decoys_v1::03::001, guide_pek_agentic_preferences_v1::05::001, guide_pek_scope_v1::01::001, guide_pek_agentic_preferences_v1::03::001
- Gold: guide_pek_scope_v1::07::001

## Interpretation Boundary

Grounded Citation Rate is not reported here because this runner evaluates retrieval. It must be measured from `TripProposal.sources` against the same run's Evidence Pool; treating retrieved chunks themselves as citations would create a meaningless 100% score.
