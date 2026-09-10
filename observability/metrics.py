"""
指标聚合：按操作名统计调用次数、成功/失败数、总耗时、P50/P95、最大耗时。

分位数用固定窗口的样本环形缓冲算（每个操作保留最近 512 次耗时），不引入
Prometheus 这类重依赖。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

_WINDOW = 512          # 每个操作保留最近 512 次耗时的样本，用于算分位数


@dataclass
class MetricEntry:
    count: int = 0
    errors: int = 0
    total_ms: int = 0
    max_ms: int = 0
    retries: int = 0
    queue_wait_ms: int = 0
    samples: deque = field(default_factory=lambda: deque(maxlen=_WINDOW))
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def percentile(self, p: float) -> int:
        """窗口内耗时的第 p 百分位（p 取 0-100），样本为空返回 0。"""
        if not self.samples:
            return 0
        ordered = sorted(self.samples)
        idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
        return ordered[idx]

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "errors": self.errors,
            "error_rate": round(self.errors / self.count, 4) if self.count else 0.0,
            "avg_ms": int(self.total_ms / self.count) if self.count else 0,
            "p50_ms": self.percentile(50),
            "p95_ms": self.percentile(95),
            "max_ms": self.max_ms,
            "retries": self.retries,
            "samples": len(self.samples),
        }


class MetricsRegistry:
    """线程安全的指标注册表。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, MetricEntry] = {}

    def record(self, name: str, duration_ms: int, *, ok: bool = True, retried: bool = False,
               queue_wait_ms: int = 0) -> None:
        with self._lock:
            e = self._entries.get(name)
            if e is None:
                e = MetricEntry()
                self._entries[name] = e
            e.count += 1
            if not ok:
                e.errors += 1
            if retried:
                e.retries += 1
            e.total_ms += max(0, int(duration_ms))
            e.max_ms = max(e.max_ms, int(duration_ms))
            e.queue_wait_ms += max(0, int(queue_wait_ms))
            e.samples.append(max(0, int(duration_ms)))
            e.last_seen = time.time()

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {k: v.to_dict() for k, v in sorted(self._entries.items())}

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()

    def summary_line(self, name: str) -> str:
        d = self.snapshot().get(name)
        if not d:
            return f"{name}: 无数据"
        return (f"{name}: 调用 {d['count']} 次，失败 {d['errors']} 次，"
                f"P50 {d['p50_ms']}ms，P95 {d['p95_ms']}ms，最大 {d['max_ms']}ms")


metrics = MetricsRegistry()
