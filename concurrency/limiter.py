"""
并发限制器：给会打下游的动作装闸门。

并发有三层来源：多用户会话、单任务内的多个子 Agent、单轮内的多个工具调用。它们最终
都汇聚到同一批下游：LLM API、Tavily、RAGFlow、MySQL。不限制的后果是级联失败：下游
限流 → 重试 → 流量被放大 → 更多 429 → 重试耗尽 → 工具返回错误字符串 → 用户感知到的
是"答案质量下降"，而不是一个明确的报错。

三个层次：

1. `session_limiter`   全局同时执行的会话数（保护整机）
2. `subagent_limiter`  单会话内同时跑的子 Agent 数（保护单任务不失控）
3. `downstream_limiter` 按下游服务分别限流（保护外部依赖，最常见的是 LLM API）

两个实现细节：

* 信号量显式记录排队耗时。"排队等了 8 秒"和"执行花了 8 秒"是不同的问题，前者说明
  闸门太紧或下游太慢，后者说明任务本身重。
* 获取/释放必须成对且异常安全。用 async context manager 表达，任何异常路径都会走
  `finally` 释放，避免泄漏一个名额导致容量永久少 1。
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

logger = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class LimiterStats:
    acquired: int = 0
    waited: int = 0
    total_wait_ms: int = 0
    rejected: int = 0
    max_wait_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "acquired": self.acquired,
            "waited": self.waited,
            "avg_wait_ms": int(self.total_wait_ms / self.waited) if self.waited else 0,
            "max_wait_ms": self.max_wait_ms,
            "rejected": self.rejected,
        }


class Limiter:
    """带排队超时与统计的信号量封装。"""

    def __init__(self, name: str, capacity: int, *, queue_timeout_sec: float = 0):
        self.name = name
        self.capacity = max(1, capacity)
        self._sem = asyncio.Semaphore(self.capacity)
        self._queue_timeout = queue_timeout_sec
        self.stats = LimiterStats()

    @property
    def available(self) -> int:
        return self._sem._value  # noqa: SLF001 只读观测用

    @property
    def in_use(self) -> int:
        return self.capacity - self.available

    @asynccontextmanager
    async def acquire(self, *, session_id: str = "") -> AsyncIterator[int]:
        """获取一个名额，yield 出排队等待的毫秒数；超时抛 LimiterTimeout。"""
        started = time.time()
        if self._sem.locked():
            self.stats.waited += 1
            logger.debug("[%s] 已达上限 %s，等待中…", self.name, self.capacity)
        acquired_ok = False
        try:
            if self._queue_timeout > 0:
                await asyncio.wait_for(self._sem.acquire(), timeout=self._queue_timeout)
            else:
                await self._sem.acquire()
            acquired_ok = True
        except asyncio.TimeoutError:
            self.stats.rejected += 1
            wait_ms = int((time.time() - started) * 1000)
            logger.warning("[%s] 排队超时（%sms），拒绝请求 session=%s",
                           self.name, wait_ms, session_id)
            raise LimiterTimeout(self.name, wait_ms, session_id) from None

        wait_ms = int((time.time() - started) * 1000)
        self.stats.acquired += 1
        self.stats.total_wait_ms += wait_ms
        self.stats.max_wait_ms = max(self.stats.max_wait_ms, wait_ms)
        if wait_ms > 50:
            logger.info("[%s] 排队 %sms 后获得名额 session=%s", self.name, wait_ms, session_id)
        try:
            yield wait_ms
        finally:
            if acquired_ok:
                self._sem.release()

    def to_dict(self) -> dict:
        return {"name": self.name, "capacity": self.capacity,
                "in_use": self.in_use, **self.stats.to_dict()}


class LimiterTimeout(RuntimeError):
    """排队等待超时：快速失败，不无限堆积。"""

    def __init__(self, name: str, wait_ms: int, session_id: str = ""):
        super().__init__(f"{name} 排队 {wait_ms}ms 仍未获得执行名额，请稍后重试")
        self.name = name
        self.wait_ms = wait_ms
        self.session_id = session_id


# ---------------------------------------------------------------------------
# 全局限流器实例
# ---------------------------------------------------------------------------
session_limiter = Limiter("session", _env_int("MAX_CONCURRENT_SESSIONS", 8),
                          queue_timeout_sec=_env_int("SESSION_QUEUE_TIMEOUT_SEC", 120))
subagent_limiter = Limiter("subagent", _env_int("MAX_CONCURRENT_SUBAGENTS", 3),
                           queue_timeout_sec=_env_int("SUBAGENT_QUEUE_TIMEOUT_SEC", 60))

#: 按下游服务分别限流：各下游承受能力不同，不能共用一个全局池
DOWNSTREAM_CAPACITY = {
    "llm": _env_int("MAX_CONCURRENT_LLM", 8),
    "tavily": _env_int("MAX_CONCURRENT_TAVILY", 4),
    "ragflow": _env_int("MAX_CONCURRENT_RAGFLOW", 4),
    "mysql": _env_int("MAX_CONCURRENT_MYSQL", 8),
}
downstream_limiters: dict[str, Limiter] = {
    name: Limiter(f"downstream:{name}", cap, queue_timeout_sec=60)
    for name, cap in DOWNSTREAM_CAPACITY.items()
}


def limiter_for(service: str) -> Limiter:
    """拿到某个下游服务的限流器；未配置的服务按默认上限补一个，避免漏配等于不限流。"""
    if service not in downstream_limiters:
        downstream_limiters[service] = Limiter(
            f"downstream:{service}", _env_int(f"MAX_CONCURRENT_{service.upper()}", 4),
            queue_timeout_sec=60)
    return downstream_limiters[service]


def all_stats() -> dict:
    return {
        "session": session_limiter.to_dict(),
        "subagent": subagent_limiter.to_dict(),
        "downstream": {k: v.to_dict() for k, v in downstream_limiters.items()},
    }
