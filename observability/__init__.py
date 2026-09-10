"""
可观测子系统：事件模型 + 事件总线 + 落盘 + 指标。

一次 publish 有四条出口：内存环形缓冲（断线重放）、JSONL 落盘、订阅者扇出
（WebSocket 实时推送）、指标聚合（P50/P95、失败率、重试次数）。
`Span` 上下文管理器负责记录耗时并还原调用树（span_id / parent_span_id）。
"""
from observability.bus import EventBus, event_bus
from observability.events import ALL_EVENT_TYPES, EventType, Span, TraceEvent, new_span_id
from observability.metrics import MetricsRegistry, metrics
from observability.store import EventStore, event_store

__all__ = [
    "EventBus", "event_bus",
    "ALL_EVENT_TYPES", "EventType", "Span", "TraceEvent", "new_span_id",
    "MetricsRegistry", "metrics",
    "EventStore", "event_store",
]
