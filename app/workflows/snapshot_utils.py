from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any



def coerce_graph_output(
    output: Any,
) -> dict[str, Any]:
    """
    将 LangGraph invoke() 的返回值统一转换成普通 dict。

    当前项目默认使用 LangGraph v1 风格输出：
        graph.invoke(...) -> dict

    新版本 LangGraph 也可能通过 GraphOutput 对象暴露：
        output.value
        output.interrupts

    API 层不应该到处判断版本差异，因此在这里统一兼容。

    Args:
        output:
            graph.invoke() / graph.stream() 的单次最终返回值。

    Returns:
        dict[str, Any]:
            可以安全交给 API Formatter 的普通字典。
    """

    # 1. 当前项目最常见的情况：invoke() 直接返回 Mapping。
    if isinstance(output, Mapping):
        return dict(output)

    # 2. 兼容 LangGraph v2 GraphOutput：真正的 State 位于 value。
    value = getattr(output, "value", None)
    if isinstance(value, Mapping):
        result = dict(value)

        # v2 的 interrupts 不是放在 value 中；为了让后续逻辑统一，
        # 这里把它映射成和 v1 一样的 __interrupt__ 字段。
        interrupts = getattr(output, "interrupts", None)
        if interrupts:
            result["__interrupt__"] = list(interrupts)

        return result

    # 3. 不认识的返回类型说明运行时或调用方式发生了变化。
    #    这里不静默返回空字典，否则 API 会误把失败当作完成。
    raise TypeError(
        "无法识别 LangGraph 返回值类型："
        f"{type(output).__name__}"
    )



def snapshot_values(
    snapshot: Any,
) -> dict[str, Any]:
    """
    从 StateSnapshot 中读取当前 TravelState。

    StateSnapshot 是 LangGraph Runtime 对一个 Checkpoint 的公开视图。
    当前 API 只读取 values，不直接依赖 Checkpointer 私有数据库结构。
    """

    values = getattr(snapshot, "values", None)

    if isinstance(values, Mapping):
        return dict(values)

    return {}



def snapshot_next_nodes(
    snapshot: Any,
) -> list[str]:
    """
    返回当前 Checkpoint 之后准备执行的节点名称。

    示例：
        ["decision_gate"]
        ["commit_draft"]
        []  # Graph 已结束
    """

    raw_next = getattr(snapshot, "next", None)

    if not isinstance(raw_next, Sequence) or isinstance(
        raw_next,
        (str, bytes),
    ):
        return []

    return [
        str(item)
        for item in raw_next
        if item is not None
    ]



def snapshot_checkpoint_id(
    snapshot: Any,
) -> str | None:
    """
    读取 StateSnapshot 对应的 checkpoint_id。

    checkpoint_id 只用于调试、历史查看和以后 Replay；
    Human-in-the-loop 恢复仍然主要依赖 thread_id。
    """

    config = getattr(snapshot, "config", None)

    if not isinstance(config, Mapping):
        return None

    configurable = config.get("configurable")

    if not isinstance(configurable, Mapping):
        return None

    checkpoint_id = configurable.get("checkpoint_id")

    if checkpoint_id is None:
        return None

    return str(checkpoint_id)



def snapshot_exists(
    snapshot: Any,
) -> bool:
    """
    判断 graph.get_state(config) 是否真的读取到了一个已存在 Thread。

    为什么不能只看 snapshot.values？
        理论上某些 Graph 可以保存空 State；同时不同 LangGraph 版本对
        “不存在 Thread”返回的空 StateSnapshot 细节可能不同。

    因此当前综合检查：
        - checkpoint_id；
        - created_at；
        - metadata；
        - values；
        - tasks；
        - next。
    """

    if snapshot_checkpoint_id(snapshot):
        return True

    if getattr(snapshot, "created_at", None):
        return True

    if getattr(snapshot, "metadata", None):
        return True

    if snapshot_values(snapshot):
        return True

    if getattr(snapshot, "tasks", None):
        return True

    if snapshot_next_nodes(snapshot):
        return True

    return False



def interrupt_values_from_output(
    output: Any,
) -> list[dict[str, Any]]:
    """
    从 graph.invoke() 返回值中提取 interrupt payload。

    第一次执行到 DecisionGateNode 时，payload 通常位于：
        result["__interrupt__"]

    每一项可能是：
        - Interrupt 对象，真正 JSON 在 .value；
        - 已经是普通 dict。
    """

    normalized = coerce_graph_output(output)
    raw_items = normalized.get("__interrupt__")

    if not isinstance(raw_items, Sequence) or isinstance(
        raw_items,
        (str, bytes),
    ):
        return []

    return _normalize_interrupt_items(raw_items)



def interrupt_values_from_snapshot(
    snapshot: Any,
) -> list[dict[str, Any]]:
    """
    从 StateSnapshot.tasks 中提取仍在等待 Resume 的 interrupt payload。

    这个函数主要服务 GET /runs/{thread_id}：
        查询接口没有本次 graph.invoke() 返回值，
        只能从持久化 Checkpoint 恢复当前 interrupt 信息。
    """

    tasks = getattr(snapshot, "tasks", None)

    if not isinstance(tasks, Sequence) or isinstance(
        tasks,
        (str, bytes),
    ):
        return []

    raw_items: list[Any] = []

    # 1. 每个 PregelTask 都可能携带 0 到多个 interrupts。
    for task in tasks:
        task_interrupts = getattr(
            task,
            "interrupts",
            None,
        )

        if not isinstance(
            task_interrupts,
            Sequence,
        ) or isinstance(
            task_interrupts,
            (str, bytes),
        ):
            continue

        raw_items.extend(task_interrupts)

    return _normalize_interrupt_items(raw_items)



def get_interrupt_values(
    *,
    graph_output: Any | None = None,
    snapshot: Any | None = None,
) -> list[dict[str, Any]]:
    """
    统一读取 interrupt payload。

    优先级：
        1. 本次 invoke() 返回值；
        2. 持久化 StateSnapshot。

    这样启动接口、Resume 接口和状态查询接口可以共用同一套逻辑。
    """

    if graph_output is not None:
        output_values = interrupt_values_from_output(
            graph_output
        )
        if output_values:
            return output_values

    if snapshot is not None:
        return interrupt_values_from_snapshot(
            snapshot
        )

    return []



def snapshot_is_interrupted(
    snapshot: Any,
) -> bool:
    """
    判断当前 Thread 是否正在等待 Human-in-the-loop Resume。
    """

    return bool(
        interrupt_values_from_snapshot(snapshot)
    )



def _normalize_interrupt_items(
    items: Sequence[Any],
) -> list[dict[str, Any]]:
    """
    把 Interrupt 对象或普通 Mapping 统一转成 JSON-like dict。
    """

    result: list[dict[str, Any]] = []

    for item in items:
        value = getattr(item, "value", item)

        if isinstance(value, Mapping):
            result.append(dict(value))
            continue

        # 当前 DecisionGate Payload 必须是 dict。
        # 如果以后其他节点使用字符串 interrupt，仍然以稳定字典暴露给 API。
        result.append({"value": value})

    return result
