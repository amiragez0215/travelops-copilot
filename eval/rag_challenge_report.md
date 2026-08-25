# TravelOps RAG Eval Report

## Scope

- Dataset: 42 cases (39 positive, 3 negative).
- Query modes: combined=9, implicit=3, multi_attribute=6, multi_risk=3, negative=3, split_need=18
- Corpus: 222 chunks, version v5-agentic-challenge.
- Startup excluded from query timing: 3045.45 ms.
- Embedding: `./models/paraphrase-multilingual-MiniLM-L12-v2` on `cpu`.
- Reranker: `./models/bge-reranker-v2-m3`.

## Ablation Results

| Variant | Hit@5 | Recall@5 | Precision@5 | MRR@5 | nDCG@5 | Metadata | Negative accuracy | Mean ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C_bm25_vector_rrf | 1.0000 | 0.9188 | 0.3231 | 0.7842 | 0.7940 | 1.0000 | 1.0000 | 26.9267 | 32.0547 |

## Challenge Query Modes

`combined` 代表未拆分的多偏好 query，`split_need` 代表 Agentic Planner 可以生成的单 need query。Decoy Hit@5 越低越好。

| Variant | Query mode | Positive | Hit@5 | Recall@5 | Decoy Hit@5 | Negative accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C_bm25_vector_rrf | combined | 9 | 1.0000 | 0.8333 | 0.8889 | N/A |
| C_bm25_vector_rrf | implicit | 3 | 1.0000 | 0.4444 | 0.6667 | N/A |
| C_bm25_vector_rrf | multi_attribute | 6 | 1.0000 | 1.0000 | N/A | N/A |
| C_bm25_vector_rrf | multi_risk | 3 | 1.0000 | 1.0000 | 0.3333 | N/A |
| C_bm25_vector_rrf | negative | 0 | N/A | N/A | N/A | 1.0000 |
| C_bm25_vector_rrf | split_need | 18 | 1.0000 | 1.0000 | 0.8333 | N/A |

## Candidate-Pool Diagnostics

These metrics use the pre-rerank RRF Top-10 candidate pool. They separate first-stage recall from final Top-5 ranking quality.

| Variant | Candidate Hit@10 | Candidate Recall@10 |
| --- | ---: | ---: |
| C_bm25_vector_rrf | 1.0000 | 0.9487 |

## Bad Cases

No retrieval failure met the current Bad Case threshold.
## Interpretation Boundary

Grounded Citation Rate is not reported here because this runner evaluates retrieval. It must be measured from `TripProposal.sources` against the same run's Evidence Pool; treating retrieved chunks themselves as citations would create a meaningless 100% score.
