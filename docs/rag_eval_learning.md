# TravelOps RAG Eval Learning Guide

## 1. What RAG Eval proves

RAG Eval does not ask whether a program raises an exception. It asks whether a
retrieval system returns the right evidence, ranks it high enough, obeys its
metadata constraints, and does so within an acceptable latency budget.

In TravelOps, the chain is:

```text
Trip request -> RetrievalPlan -> metadata filter -> BM25 + Dense -> RRF
-> Cross-Encoder rerank -> EvidenceGrade -> Proposal
```

Each arrow can fail independently. A plausible final answer is not proof that
the evidence chain was correct; an LLM can answer from general knowledge even
when the retrieval results were wrong.

## 2. The Gold Dataset

`eval/rag_cases.jsonl` is a hand-labelled dataset, not generated test input.
Every case records two query forms, the metadata filter, stable Gold chunk IDs,
and graded relevance labels.

```json
{
  "semantic_query": "成都下雨时适合安排哪些室内文化活动",
  "keyword_query": "成都 雨天 室内 博物馆",
  "metadata_filter": {"city": "成都", "doc_type": "guide"},
  "gold_chunk_ids": ["guide_chengdu_rainy_day_v1::03::001"],
  "relevance": {"guide_chengdu_rainy_day_v1::03::001": 2}
}
```

Use binary relevance when a chunk is either usable or unusable. Use graded
relevance when one chunk directly answers the question (`2`) and another only
provides useful context (`1`). Negative cases deliberately request a city,
hotel, or filter absent from the corpus. They verify fail-closed filtering.

## 3. Metrics you must be able to explain

| Metric | Question answered | TravelOps example |
| --- | --- | --- |
| Hit@K | Did Top-K include any Gold chunk? | Did a rain query find at least one rain guide? |
| Recall@K | How much of all Gold evidence was retrieved? | Did it find both the outdoor risk and backup plan chunks? |
| Precision@K | How much of Top-K was relevant? | Did Hybrid retrieval add noise to improve recall? |
| MRR@K | How early was the first correct result? | Was the exact hotel review ranked first? |
| nDCG@K | Did the ranking respect graded relevance? | Did the direct answer rank above a merely related chunk? |
| Metadata Accuracy | Did returned chunks obey the filter? | Did a Chengdu hotel query stay within the requested hotel ID? |
| Mean / P95 latency | What is normal and tail query cost? | Is Cross-Encoder quality worth its CPU cost? |
| Grounded Citation Rate | Are Proposal citations in that run's Evidence Pool? | Did `TripProposal.sources` invent a source ID? |

Grounded Citation Rate belongs to end-to-end Proposal evaluation. Do not report
100% merely because retrieval chunks are drawn from the corpus: that measures
source traceability, not whether the final answer cited evidence correctly.

## 4. Controlled ablation

The runner compares exactly four variants:

```text
A  BM25 only
B  Dense Vector only
C  BM25 + Dense + Reciprocal Rank Fusion
D  BM25 + Dense + RRF + BGE Cross-Encoder Reranker
```

The Gold Dataset, corpus version, chunking, filters, Top-K, and metrics must
remain fixed. Otherwise a metric change cannot be attributed to the component
being evaluated. That is the purpose of an ablation experiment.

Expected hypotheses for this project:

- BM25 is strong for hotel IDs, place names, and exact risk terms.
- Dense retrieval is strong for paraphrases and natural-language intent.
- RRF can improve recall when the two retrievers make different mistakes.
- A Cross-Encoder can improve MRR/nDCG by reading the query and passage
  together, but costs more latency on this CPU-only environment.

Hypotheses are not results. The generated report is the evidence that accepts,
rejects, or refines them.

## 5. How to investigate a Bad Case

For every failure, first preserve the query, filter, Gold label, retrieved IDs,
configuration, and latency. Then classify the root cause:

```text
Query wording | Metadata filter | Missing corpus content | Chunking
BM25 tokenization | Dense semantic drift | RRF fusion | Reranker
Gold label quality
```

Fix only the identified cause, add the case to regression evaluation, and rerun
all four variants. Do not change several variables and claim a single cause.

## 6. Commands and study sequence

```powershell
python -m pytest tests/test_eval -q
python -m eval.run_rag_eval --max-cases 4
python -m eval.run_rag_eval
```

Study in this order:

1. Read one Gold Case and find its Markdown source and stable chunk ID.
2. Hand-calculate Hit@3, Recall@3, Precision@3, and MRR for one ranked list.
3. Compare BM25 and Dense for an exact hotel ID and a paraphrased rain query.
4. Read the RRF formula in `app/rag/hybrid_retriever.py` and explain why it
   fuses ranks rather than raw scores.
5. Inspect a reranker Bad Case and explain its quality/latency tradeoff.
6. Run a full Proposal workflow later and measure citation grounding against
   its Evidence Pool.

## 7. Interview-ready explanation

"I evaluated RAG as a retrieval system rather than only judging generated
answers. I built a manually labelled Gold Dataset with positive and negative
cases, controlled corpus and filter variables across BM25, Dense, RRF, and
Cross-Encoder ablations, and compared Recall, Precision, MRR, nDCG, metadata
accuracy, and P95 latency. I used Bad Cases to determine whether a regression
came from the query, metadata, corpus, chunking, retriever, or reranker."
