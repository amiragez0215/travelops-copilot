> **历史说明**：本文件保留早期 Day 1 学习记录，里面的 `TripIntent`、`WeatherTool`、`PreTripCheckWorkflow` 等内容已经不是当前实现。最新冻结设计请以 `docs/current_design_v2.md` 和当前代码为准。

# Day 1：Schema、TravelState、WeatherTool

## 今天完成了什么

今天新增了 TripIntent、TripProposal 相关 schema、TravelState 和 WeatherTool，并为 WeatherTool 编写了单元测试。WeatherTool 当前使用 mock 模式读取 data/mock/weather_mock.json，不调用真实天气 API。

## 模块数据流

用户输入会先经过 IntentExtractNode，得到 TripIntent。后续 WeatherQueryNode 从 TravelState 中读取 intent，取出 destination、date_start、date_end，然后调用 get_weather。get_weather 读取 mock 天气数据，返回 summary、daily、risks、warnings 和 source。WeatherQueryNode 再把结果写入 TravelState["weather"]。后续 ItineraryBuildNode、VerifierNode、PreTripCheckWorkflow、Trace 和 Eval 会使用 weather 信息。

## 模块属于哪一层

TripIntent 和 TripProposal 属于 app/schemas，因为它们定义项目的数据结构。TravelState 属于 app/agents，因为它是 LangGraph 工作流共享状态。WeatherTool 属于 app/tools，因为它是可独立测试的确定性工具函数。WeatherTool 的测试放在 tests/test_tools，因为工具层应该先独立测试，再接入 node 和 workflow。

## 核心函数的输入、输出、副作用

get_weather 的输入包括 city、date_start、date_end、mode 和 mock_file。输出包括 city、date_range、summary、daily、risks、warnings 和 source。它没有写数据库，没有修改 TravelState，没有调用真实 API，只读取 mock JSON 文件。

## 今天的关键设计调整

最初的 WeatherTool 为了兼容多种 mock JSON 格式，写了较多格式适配逻辑。后来发现这会让 Day 1 的学习重点变模糊。更合理的做法是先冻结 data/mock/weather_mock.json 的数据契约，再让 WeatherTool 按固定结构读取数据。后续如果接真实天气 API，也应该由 real weather adapter 把外部格式转换成项目内部统一 WeatherInfo，而不是让 workflow 关心外部格式。

## 关于 TripProposal 的调整

TripProposal 不应该只包含航班、酒店、每日计划和预算，也应该包含 weather 信息。天气摘要和风险是用户需要看到的内容，也会被 Verifier、PreTripCheckWorkflow 和 Eval 使用。因此后续会给 TripProposal 增加 weather 字段。

## 我今天学到的项目原则

如果一个工具函数里出现大量格式兼容代码，要优先思考是不是数据 contract 没定义好。项目开发不是写万能函数，而是先定义稳定的数据结构，再围绕数据结构实现工具、测试和 workflow。

## 还不确定的问题

- WeatherTool 是否应该长期使用 mock JSON，还是后续增加数据库缓存。
- 真实天气 API adapter 应该如何设计。
- WeatherInfo schema 应该放在 trip_schema.py，还是单独建 weather_schema.py。

## 为什么 TripIntent 不直接用 dict？


TripIntent 是用户需求的数据合同。
它可以约束字段类型、默认值、范围，也方便后面 LLM structured output、ClarifyNode、Verifier 和 Eval 使用。

如果直接用 dict，问题是：

字段名可能不统一
类型可能不统一
缺失字段不好检查
后面 workflow 很难稳定组合
## 为什么 TravelState 用 TypedDict，而不是 Pydantic BaseModel？
核心原因是：

TravelState 是 LangGraph 节点之间流动的工作状态，更适合用 dict-like 的结构。

LangGraph 里的 node 通常是这样工作的：

def weather_node(state: TravelState) -> dict:
    ...
    return {"weather": weather_result}

也就是说，每个节点返回的是“局部更新”。

TypedDict 的好处是：

1. 形式上仍然是 dict，适合 LangGraph 的状态合并方式。
2. 有类型提示，写代码时 IDE 能提示字段。
3. 比 Pydantic BaseModel 更轻，不需要每个节点都重新实例化整个对象。
4. 适合不断被多个节点追加字段的 workflow 状态。

Pydantic 更适合：

外部输入
结构化输出
API 请求响应
工具返回结果 schema
最终 TripProposal

TypedDict 更适合：

LangGraph 内部流转状态

所以可以简单记：

Schema 给边界用。
State 给流程内部用。
