"""
可观测事件模型：把"发生了什么"变成可排序、可关联、可重放的结构化数据。

三个关键字段的用途：

| 字段 | 解决什么 |
|---|---|
| `event_id`（会话内单调递增） | 断线重放：前端记下最后收到的 id，重连时带上，服务端补发之后的事件 |
| `trace_id` / `span_id` / `parent_span_id` | 调用树：多 Agent 的调用是树形的（会话 → 子 Agent → 工具），平铺日志看不出层级，有 span 才能还原"这个工具是哪个子 Agent 调的、那次检索整体花了多久" |
| `duration_ms` | 性能定位：用户看到"正在查数据库"转了 40 秒，是数据库慢还是模型慢，没有耗时就分不出来 |
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator


class EventType:
    """事件类型的集中定义（字符串常量，避免拼写漂移）。"""

    # --- 基础事件（前端按这些名字分发）---
    SESSION_CREATED = "session_created"
    TOOL_START = "tool_start"
    ASSISTANT_CALL = "assistant_call"
    TASK_RESULT = "task_result"
    ERROR = "error"

    # --- 工具生命周期与耗时 ---
    TOOL_END = "tool_end"
    TOOL_ERROR = "tool_error"

    # --- 沙箱 ---
    SANDBOX_READY = "sandbox_ready"
    SANDBOX_DENIED = "sandbox_denied"
    SANDBOX_DESTROYED = "sandbox_destroyed"

    # --- 安全 ---
    PERMISSION_DENIED = "permission_denied"
    PATH_DENIED = "path_denied"
    SQL_REJECTED = "sql_rejected"
    SQL_EXECUTED = "sql_executed"

    # --- 人工审批 ---
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_RESOLVED = "approval_resolved"

    # --- 并发 ---
    QUEUED = "queued"
    RETRY = "retry"
    CIRCUIT_OPEN = "circuit_open"


ALL_EVENT_TYPES = frozenset(
    v for k, v in vars(EventType).items() if not k.startswith("_") and isinstance(v, str))


@dataclass(slots=True)
class TraceEvent:
    """一条可观测事件。"""

    type: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    event_id: int = 0
    ts: float = field(default_factory=time.time)
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str | None = None
    duration_ms: int = 0
    level: str = "info"                      # info / warn / error

    def to_payload(self) -> dict:
        """转成前端/落盘用的字典。

        `type` 固定为 "monitor_event"，`event` 才是具体事件名。前端的
        `handleSocketMessage` 靠这个约定分发。
        """
        payload = {
            "type": "monitor_event",
            "event": self.type,
            "message": self.message,
            "data": self.data,
            "timestamp": self.ts,
            "event_id": self.event_id,
            "trace_id": self.trace_id or self.session_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "duration_ms": self.duration_ms,
            "level": self.level,
        }
        return payload

    def to_jsonl(self) -> str:
        import json
        return json.dumps(self.to_payload(), ensure_ascii=False, default=str)


def new_span_id() -> str:
    return uuid.uuid4().hex[:16]


class Span:
    """上下文管理器：自动记录一段代码的耗时并发出配对事件。

    用法::

        with Span(bus, "tool:db_query", session_id=sid, event_type=EventType.TOOL_START):
            do_work()

    退出时发 `tool_end`（异常时发 `tool_error`）事件，带 duration_ms。
    进入时把自己压入会话的 span 栈，子 span 取栈顶作为 parent_span_id，
    调用树就是这么还原出来的。
    """

    __slots__ = ("bus", "name", "session_id", "span_id", "parent_id", "start", "data",
                 "event_type", "end_event_type", "_error")

    def __init__(self, bus, name: str, *, session_id: str, data: dict | None = None,
                 event_type: str = EventType.TOOL_START,
                 end_event_type: str = EventType.TOOL_END, parent_span_id: str | None = None):
        self.bus = bus
        self.name = name
        self.session_id = session_id
        self.span_id = new_span_id()
        self.parent_id = parent_span_id if parent_span_id is not None else bus.current_span(session_id)
        self.start = 0.0
        self.data = data or {}
        self.event_type = event_type
        self.end_event_type = end_event_type
        self._error: str | None = None

    def __enter__(self) -> "Span":
        self.start = time.time()
        self.bus.push_span(self.session_id, self.span_id)
        self.bus.publish(
            self.event_type, self.name, session_id=self.session_id,
            data=self.data, span_id=self.span_id, parent_span_id=self.parent_id)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration = int((time.time() - self.start) * 1000)
        self.bus.pop_span(self.session_id, self.span_id)
        if exc_type is not None:
            self.bus.publish(
                EventType.TOOL_ERROR, f"{self.name} 执行失败：{exc}", session_id=self.session_id,
                data={**self.data, "error": str(exc)}, span_id=self.span_id,
                parent_span_id=self.parent_id, duration_ms=duration, level="error")
            return False                          # 不吞异常
        self.bus.publish(
            self.end_event_type, f"{self.name} 完成", session_id=self.session_id,
            data=self.data, span_id=self.span_id, parent_span_id=self.parent_id,
            duration_ms=duration)
        try:
            self.bus.record_metric(self.name, duration, ok=True)
        except Exception:
            pass
        return False


def iter_session_events(events: Iterator[dict]) -> Iterator[dict]:  # pragma: no cover
    """占位：便于将来接 OTel / LangSmith 时的适配入口。"""
    yield from events
