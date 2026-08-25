# TravelOps RAG Eval Report

## Scope

- Dataset: 120 cases (96 positive, 24 negative).
- Corpus: 607 chunks, version v3-multi-city.
- Startup excluded from query timing: 4432.13 ms.
- Embedding: `./models/paraphrase-multilingual-MiniLM-L12-v2` on `cpu`.
- Reranker: `./models/bge-reranker-v2-m3`.

## Ablation Results

| Variant | Hit@5 | Recall@5 | Precision@5 | MRR@5 | nDCG@5 | Metadata | Negative accuracy | Mean ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C_bm25_vector_rrf | 0.9271 | 0.8646 | 0.2271 | 0.7531 | 0.7560 | 1.0000 | 1.0000 | 26.1472 | 38.8163 |

## Candidate-Pool Diagnostics

These metrics use the pre-rerank RRF Top-10 candidate pool. They separate first-stage recall from final Top-5 ranking quality.

| Variant | Candidate Hit@10 | Candidate Recall@10 |
| --- | ---: | ---: |
| C_bm25_vector_rrf | 1.0000 | 0.9618 |

## Bad Cases

### rag_hotel_gaoxin_business_017 (C_bm25_vector_rrf)

- Reason: no Gold chunk appeared in Top-5
- Top results: hotel_review_chengdu_gaoxin_008_v1::02::001, hotel_review_chengdu_gaoxin_008_v1::07::001, hotel_review_chengdu_gaoxin_008_v1::01::001, hotel_review_chengdu_gaoxin_008_v1::04::001, hotel_review_chengdu_gaoxin_008_v1::05::001
- Gold: hotel_review_chengdu_gaoxin_008_v1::08::001

### rag_v2_hotel_new_compare (C_bm25_vector_rrf)

- Reason: no Gold chunk appeared in Top-5
- Top results: hotel_review_chengdu_jinniu_013_v2::01::001, hotel_review_chengdu_chenghua_014_v2::03::001, hotel_review_chengdu_chenghua_014_v2::11::001, hotel_review_chengdu_longquanyi_016_v2::02::001, hotel_review_chengdu_jinniu_013_v2::06::001
- Gold: hotel_review_chengdu_jinniu_013_v2::04::001, hotel_review_chengdu_chenghua_014_v2::04::001, hotel_review_chengdu_longquanyi_016_v2::04::001

### rag_city_shanghai_guide_weather (C_bm25_vector_rrf)

- Reason: no Gold chunk appeared in Top-5
- Top results: guide_shanghai_business_transit_v3::07::001, guide_shanghai_business_hongqiao_v1::05::001, guide_shanghai_business_hongqiao_v1::09::001, guide_shanghai_family_rainy_v3::06::001, guide_shanghai_bund_museum_v1::09::001
- Gold: guide_shanghai_business_transit_v3::04::001, guide_shanghai_business_transit_v3::08::001

### rag_city_hangzhou_packing (C_bm25_vector_rrf)

- Reason: no Gold chunk appeared in Top-5
- Top results: packing_checklist_hangzhou_outdoor_rain_v3::08::001, packing_checklist_hangzhou_outdoor_rain_v3::07::001, packing_checklist_hangzhou_outdoor_rain_v3::05::001, packing_hangzhou_city_v1::01::001, packing_checklist_hangzhou_outdoor_rain_v3::02::001
- Gold: packing_checklist_hangzhou_outdoor_rain_v3::01::001, packing_checklist_hangzhou_outdoor_rain_v3::06::001

### rag_city_beijing_safety_01 (C_bm25_vector_rrf)

- Reason: no Gold chunk appeared in Top-5
- Top results: safety_notice_beijing_heat_101_v1::02::001, safety_notice_beijing_heat_101_v1::03::001, safety_notice_beijing_heat_101_v1::05::001, safety_notice_beijing_heat_101_v1::04::001, safety_notice_beijing_heat_101_v1::06::001
- Gold: safety_notice_beijing_heat_101_v1::01::001

## Interpretation Boundary

Grounded Citation Rate is not reported here because this runner evaluates retrieval. It must be measured from `TripProposal.sources` against the same run's Evidence Pool; treating retrieved chunks themselves as citations would create a meaningless 100% score.
