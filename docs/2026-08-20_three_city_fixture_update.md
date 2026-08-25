# 三城固定窗口数据更新

本次数据迁移把可运行的 Mock / RAG 试验范围固定为 2026-10-01 至
2026-10-07、杭州/北京/上海三座城市，最长旅行时长为 7 天。日期范围是
结构化库存的能力边界；城市攻略、酒店评价和安全提醒是与具体日期无关的
可复用知识。

## Mock 数据

- 航班：6 条有向路线 × 7 个日期 × 每天 4 班，共 168 条；每条路线都同时
  生成正向和反向记录，因此出发城市不再固定为杭州。
- 酒店：每城 11 家，包含经济型 2、中档型 2、高档型 2、地铁/商圈便利型
  2、安静型 2 和明显不足的备选 1 家。字段显式保留 `hotel_type`、
  `near_subway`、`quiet_score`、`cleanliness_score`，供候选排序和负面案例使用。
- 天气：每城 7 天，共 21 条；分配晴、多云、小雨、高温、大风、强降雨等状态，
  风险标签严格使用 `rain`、`heat`、`wind`、`heavy_rain`。

## RAG 语料与 Gold Dataset

活动语料只保留三城，按攻略、酒店评价、行李清单和四类安全风险拆成具有不同
主题词的 Markdown 文档。`v5-agentic-challenge` 为每城新增一份 need-specific
多偏好攻略、一份近义干扰攻略，并把酒店评价从每城 1 家扩充到每城 5 家。
当前 39 份文档产生 222 个 active chunks，所有 Gold/decoy Chunk ID 都通过语料一致性校验。

`eval/rag_cases.jsonl` 仍是 120 条稳定 baseline：96 条正例覆盖三城的攻略、酒店、
行李和风险证据，24 条负例覆盖未知城市、未知酒店 ID 和不匹配风险 metadata。
新增 `eval/rag_challenge_cases.jsonl` 42 条（39 正例、3 负例），专门测试 combined/split
query、多偏好 guides、含蓄表达、多属性酒店、多风险 safety 与 semantic decoys。
生成脚本为 `scripts/rebuild_scoped_data.py`，语料与两套 Case 可重复生成。

## 评测口径

baseline 用于保持历史可比性；challenge 用于证明 Agentic 拆分是否真正提高 need-level
召回。当前 Hybrid RRF 在 challenge 上的 split-need Recall@5 为 1.0000，combined 为
0.8333，implicit 为 0.4444，multi-risk 为 1.0000；同时 Metadata Accuracy 和
Negative Accuracy 均为 1.0000。详见 `eval/rag_challenge_report.md`。
