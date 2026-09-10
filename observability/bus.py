"""
事件总线：采集方只管 publish，不需要知道下游有谁。

三条出口：
1. 内存环形缓冲（每会话一个 deque），支撑断线重放。前端记住最后收到的 `event_id`，
   重连时带上，服务端从缓冲里补发。缓冲上限默认 2000 条，超出后最老的事件被挤掉，
   补不到的部分回退到落盘文件里读。
2. JSONL 落盘，用于事后复盘、审计、性能分析。内存会丢，磁盘不会。
3. 订阅者扇出，WebSocket 实时推送。

publish 可能从任意线程被调用（工具里开了线程、或走同步执行路径），而订阅者是异步
回调，所以要处理在哪个 loop 上调度协程：

* 同一个 loop：直接 `loop.create_task()`，无额外开销；
* 不同 loop / 不同线程：必须用 `asyncio.run_coroutine_threadsafe()`。协程只能在创建
  它的那个 loop 里跑，用错方法会报 "coroutine was attached to a different loop"；
* 没有 loop（纯同步上下文，比如脚本模式）：只写缓冲和落盘，实时推送不生效，不抛异常。

另外 `builtins.runtime.stream_writer`（deepagents 脚本模式注入的流式输出函数）
是保底通道，让脱离 FastAPI 直接跑脚本时也能看到事件。
"""
from __future__ import annotations

import asyncio
import builtins
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Awaitable, Callable, Optional

from observability.events import EventType, TraceEvent, new_span_id
from observability.metrics import metrics
from observability.store import event_store

logger = logging.getLogger(__name__)

Subscriber = Callable[[dict, str], Awaitable[None]]


class EventBus:
    """会话级事件总线（进程内单例）。"""

    def __init__(self, *, buffer_size: int = 2000, persist: bool = True) -> None:
        self._buffer_size = buffer_size
        self._persist = persist
        self._buffers: dict[str, deque[TraceEvent]] = defaultdict(
            lambda: deque(maxlen=self._buffer_size))
        self._counters: dict[str, int] = defaultdict(int)
        self._subscribers: list[Subscriber] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._spans: dict[str, list[str]] = defaultdict(list)
        self._lock = threading.RLock()
        self._stats = {"published": 0, "persisted": 0, "dispatched": 0, "dropped": 0}
        self._runtime = getattr(builtins, "runtime", None)

    # ------------------------------------------------------------------
    # 装配
    # ------------------------------------------------------------------
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """绑定主事件循环（FastAPI 的 startup 里调用）。

        不要在模块导入时取 loop：那一刻 uvicorn 的 loop 还没起来，取到的可能是马上
        会被替换掉的临时 loop，之后所有跨线程投递都会投进那个死掉的 loop，表现为
        消息静默丢失。
        """
        self._loop = loop
        logger.info("事件总线已绑定事件循环 id=%s", id(loop))

    def subscribe(self, callback: Subscriber) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def set_runtime(self, runtime: Any) -> None:
        self._runtime = runtime

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop

    # ------------------------------------------------------------------
    # 发布
    # ------------------------------------------------------------------
    def publish(self, event_type: str, message: str = "", *, session_id: str | None = None,
                data: dict[str, Any] | None = None, span_id: str = "",
                parent_span_id: str | None = None, duration_ms: int = 0,
                level: str = "info") -> TraceEvent:
        """发布一条事件。同步方法，可从任意线程调用。"""
        sid = session_id or ""
        if not sid:
            try:
                from api.context import get_thread_context
                sid = get_thread_context() or ""
            except Exception:
                sid = ""

        with self._lock:
            self._counters[sid] += 1
            event = TraceEvent(
                type=event_type,
                message=message,
                data=data or {},
                session_id=sid,
                event_id=self._counters[sid],
                trace_id=sid,
                span_id=span_id or new_span_id(),
                parent_span_id=parent_span_id,
                duration_ms=duration_ms,
                level=level,
            )
            self._buffers[sid].append(event)
            self._stats["published"] += 1

        if self._persist:
            try:
                event_store.append(sid, event.to_jsonl())
                self._stats["persisted"] += 1
            except Exception:
                logger.exception("事件落盘失败（已忽略）")

        self._dispatch(event)
        self._stream_writer(event)
        return event

    def _dispatch(self, event: TraceEvent) -> None:
        """把事件扇出给所有异步订阅者。"""
        if not self._subscribers:
            return
        payload = event.to_payload()
        loop = self._loop
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None

        if loop is None:
            self._stats["dropped"] += 1          # 无 loop：只落盘，不实时推
            return

        for callback in list(self._subscribers):
            try:
                coro = callback(payload, event.session_id)
            except Exception:
                logger.exception("订阅者调用失败")
                continue
            try:
                if current is loop:
                    loop.create_task(coro)        # 同 loop：直接创建任务
                else:
                    asyncio.run_coroutine_threadsafe(coro, loop)   # 跨 loop/跨线程：线程安全
                self._stats["dispatched"] += 1
            except Exception:
                self._stats["dropped"] += 1
                logger.exception("事件投递失败（已忽略）")

    def _stream_writer(self, event: TraceEvent) -> None:
        """把事件转给脚本模式注入的流式输出函数（没有就跳过）。"""
        writer = self._runtime
        if writer is None:
            writer = getattr(builtins, "runtime", None)
        if writer is None or not hasattr(writer, "stream_writer"):
            return
        try:
            writer.stream_writer(event.to_payload())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 重放
    # ------------------------------------------------------------------
    def replay(self, session_id: str, *, after_event_id: int = 0, limit: int = 1000) -> list[dict]:
        """补发 `after_event_id` 之后的事件。内存缓冲优先，不够再读落盘文件。"""
        with self._lock:
            buffered = [e for e in self._buffers.get(session_id, ())
                        if e.event_id > after_event_id]
        if len(buffered) >= limit:
            return [e.to_payload() for e in buffered[:limit]]

        earliest = buffered[0].event_id if buffered else None
        # 缓冲为空、或最早一条不是 after_event_id+1（说明已被环形覆盖）时，从磁盘补齐
        if not buffered or (earliest is not None and earliest > after_event_id + 1):
            from_disk = event_store.read(session_id, after_event_id=after_event_id, limit=limit)
            seen = {e.event_id for e in buffered}
            merged = [d for d in from_disk if d.get("event_id") not in seen]
            return (merged + [e.to_payload() for e in buffered])[:limit]
        return [e.to_payload() for e in buffered]

    def last_event_id(self, session_id: str) -> int:
        with self._lock:
            buf = self._buffers.get(session_id)
            return buf[-1].event_id if buf else 0

    # ------------------------------------------------------------------
    # span 栈：还原调用树
    # ------------------------------------------------------------------
    def current_span(self, session_id: str) -> str | None:
        with self._lock:
            stack = self._spans.get(session_id)
            return stack[-1] if stack else None

    def push_span(self, session_id: str, span_id: str) -> None:
        with self._lock:
            self._spans[session_id].append(span_id)

    def pop_span(self, session_id: str, span_id: str) -> None:
        with self._lock:
            stack = self._spans.get(session_id)
            if not stack:
                return
            try:
                stack.remove(span_id)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # 指标
    # ------------------------------------------------------------------
    def record_metric(self, name: str, duration_ms: int, *, ok: bool = True,
                      retried: bool = False, queue_wait_ms: int = 0) -> None:
        metrics.record(name, duration_ms, ok=ok, retried=retried, queue_wait_ms=queue_wait_ms)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            return {**self._stats, "sessions": len(self._buffers),
                    "subscribers": len(self._subscribers),
                    "loop_bound": self._loop is not None}

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            self._buffers.pop(session_id, None)
            self._spans.pop(session_id, None)


#: 全局单例
event_bus = EventBus()
