# TravelOps RAG Eval Report

## Scope

- Dataset: 120 cases (96 positive, 24 negative).
- Query modes: baseline=120
- Corpus: 222 chunks, version v5-agentic-challenge.
- Startup excluded from query timing: 2916.72 ms.
- Embedding: `./models/paraphrase-multilingual-MiniLM-L12-v2` on `cpu`.
- Reranker: `./models/bge-reranker-v2-m3`.

## Ablation Results

| Variant | Hit@5 | Recall@5 | Precision@5 | MRR@5 | nDCG@5 | Metadata | Negative accuracy | Mean ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C_bm25_vector_rrf | 1.0000 | 1.0000 | 0.2000 | 0.7549 | 0.8167 | 1.0000 | 1.0000 | 21.1172 | 27.6978 |

## Candidate-Pool Diagnostics

These metrics use the pre-rerank RRF Top-10 candidate pool. They separate first-stage recall from final Top-5 ranking quality.

| Variant | Candidate Hit@10 | Candidate Recall@10 |
| --- | ---: | ---: |
| C_bm25_vector_rrf | 1.0000 | 1.0000 |

## Bad Cases

No retrieval failure met the current Bad Case threshold.
## Interpretation Boundary

Grounded Citation Rate is not reported here because this runner evaluates retrieval. It must be measured from `TripProposal.sources` against the same run's Evidence Pool; treating retrieved chunks themselves as citations would create a meaningless 100% score.
