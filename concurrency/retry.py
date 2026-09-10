"""
重试、退避与熔断。

* 重试：同一个动作再试一次，适合瞬时失败（网络抖动、429 限流）。
* 退避：重试之间要等，而且要越等越久。固定间隔重试等于给已经过载的下游继续加压。
  指数退避 + 抖动（jitter）才能打散重试风暴：没有抖动，同一批请求会在几乎同一时刻
  重试，形成共振。
* 熔断：连续失败到阈值就直接拒绝，不再打下游。重试处理偶发失败，熔断处理"下游已经
  挂了，再打只会拖慢自己"。

`is_retryable` 显式区分两类失败：重试业务性失败（比如"没有可用的表"）只是浪费一次
调用，重试鉴权失败更是错上加错。所以默认不重试，只有明确是超时 / 连接错误 / 限流
才重试。

配合 `limiter.py` 使用：限流挡住入口流量，退避安抚下游，熔断快速失败。
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, TypeVar

from observability.events import EventType
from observability.metrics import metrics

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: 明确"值得重试"的异常类型名（子串匹配，避免强依赖各家 SDK 的异常类）
RETRYABLE_MARKERS = (
    "timeout", "timedout", "connection", "connectionreset", "connectionerror",
    "temporarily", "unavailable", "ratelimit", "rate_limit", "toomanyrequests",
    "429", "500", "502", "503", "504", "ssl", "eof", "broken pipe",
)
#: 明确"不该重试"的标记
NON_RETRYABLE_MARKERS = (
    "unauthorized", "forbidden", "authentication", "invalid api key", "not found",
    "permission", "quota exceeded", "bad request", "400", "401", "403", "404",
)


def is_retryable(exc: BaseException) -> bool:
    """判断一个异常是否值得重试。默认不重试，只有明确命中的才重试。"""
    text = f"{type(exc).__name__} {exc}".lower()
    if any(m in text for m in NON_RETRYABLE_MARKERS):
        return False
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    return any(m in text for m in RETRYABLE_MARKERS)


@dataclass
class RetryPolicy:
    attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    multiplier: float = 2.0
    jitter: float = 0.3          # 抖动比例：delay * (1 ± jitter)

    def delay_for(self, attempt: int) -> float:
        """第 attempt 次重试前该等多久：指数增长、封顶 max_delay、乘 ±jitter 的随机抖动。"""
        raw = min(self.max_delay, self.base_delay * (self.multiplier ** (attempt - 1)))
        return max(0.0, raw * (1 + random.uniform(-self.jitter, self.jitter)))


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    session_id: str = "",
    operation: str = "",
    circuit: "CircuitBreaker | None" = None,
    retryable: Callable[[BaseException], bool] = is_retryable,
) -> T:
    """带指数退避的重试执行器。

    不可重试的异常立刻抛出；重试耗尽时抛最后一次的异常；熔断打开时抛
    `CircuitOpenError`。
    """
    policy = policy or RetryPolicy()
    op = operation or getattr(fn, "__name__", "operation")

    if circuit is not None and circuit.is_open():
        metrics.record(op, 0, ok=False, retried=False)
        raise CircuitOpenError(circuit.name, circuit.remaining_cooldown())

    last_exc: BaseException | None = None
    for attempt in range(1, policy.attempts + 1):
        started = time.time()
        try:
            result = await fn()
            metrics.record(op, int((time.time() - started) * 1000), ok=True,
                           retried=attempt > 1)
            if circuit is not None:
                circuit.on_success()
            return result
        except Exception as exc:                     # noqa: BLE001 需要判断是否可重试
            last_exc = exc
            metrics.record(op, int((time.time() - started) * 1000), ok=False,
                           retried=attempt > 1)
            if circuit is not None:
                circuit.on_failure()

            if not retryable(exc) or attempt >= policy.attempts:
                raise
            delay = policy.delay_for(attempt)
            logger.warning("[%s] 第 %s/%s 次失败（%s），%.2fs 后重试",
                           op, attempt, policy.attempts, type(exc).__name__, delay)
            _emit_retry(session_id, op, attempt, delay, exc)
            if circuit is not None and circuit.is_open():
                raise CircuitOpenError(circuit.name, circuit.remaining_cooldown()) from exc
            await asyncio.sleep(delay)

    raise last_exc if last_exc else RuntimeError(f"{op} 重试耗尽")


def _emit_retry(session_id: str, op: str, attempt: int, delay: float, exc: BaseException) -> None:
    try:
        from observability.bus import event_bus
        event_bus.publish(
            EventType.RETRY, f"{op} 第 {attempt} 次重试（{delay:.1f}s 后）",
            session_id=session_id,
            data={"operation": op, "attempt": attempt, "delay": round(delay, 2),
                  "error": f"{type(exc).__name__}: {exc}"[:200]},
            level="warn")
    except Exception:
        pass


class CircuitOpenError(RuntimeError):
    """熔断打开：短期内直接拒绝，不再打下游。"""

    def __init__(self, name: str, cooldown: float):
        super().__init__(f"{name} 熔断打开，{cooldown:.1f}s 内拒绝请求")
        self.name = name
        self.cooldown = cooldown


@dataclass
class CircuitBreaker:
    """三态熔断器：CLOSED（正常）→ OPEN（拒绝）→ HALF_OPEN（试探）。

    状态迁移：CLOSED 下连续失败到 failure_threshold 就进 OPEN；OPEN 持续
    recovery_timeout 后转 HALF_OPEN；HALF_OPEN 期间最多放 half_open_max_calls 个
    试探请求，成功回 CLOSED，再失败立刻回 OPEN。

    冷却时间一到就完全放开会把刚恢复的下游重新打挂，所以半开只放少量试探请求。
    """

    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 1

    _failures: int = 0
    _opened_at: float = 0.0
    _half_open_calls: int = 0
    _state: str = "closed"

    def is_open(self) -> bool:
        """是否拒绝请求。OPEN 到期会自动转 HALF_OPEN，并在半开名额内放行。"""
        if self._state == "open":
            if time.time() - self._opened_at >= self.recovery_timeout:
                self._state = "half_open"
                self._half_open_calls = 0
                return False
            return True
        if self._state == "half_open":
            if self._half_open_calls >= self.half_open_max_calls:
                return True
            self._half_open_calls += 1
            return False
        return False

    def on_success(self) -> None:
        self._failures = 0
        if self._state != "closed":
            logger.info("[%s] 熔断恢复 -> closed", self.name)
        self._state = "closed"

    def on_failure(self) -> None:
        """记一次失败：半开下失败直接回 OPEN，CLOSED 下累计到阈值才开。"""
        self._failures += 1
        if self._state == "half_open":
            self._open()
            return
        if self._failures >= self.failure_threshold and self._state == "closed":
            self._open()

    def _open(self) -> None:
        """进入 OPEN 并开始计时，同时往事件总线发一条 circuit_open。"""
        self._state = "open"
        self._opened_at = time.time()
        logger.warning("[%s] 熔断打开（连续失败 %s 次），%.0fs 内直接拒绝",
                       self.name, self._failures, self.recovery_timeout)
        try:
            from observability.bus import event_bus
            event_bus.publish(EventType.CIRCUIT_OPEN, f"{self.name} 熔断打开",
                              data={"failures": self._failures,
                                    "cooldown_sec": self.recovery_timeout}, level="error")
        except Exception:
            pass

    def remaining_cooldown(self) -> float:
        """距离 OPEN 到期还有多少秒，已过期返回 0。"""
        return max(0.0, self.recovery_timeout - (time.time() - self._opened_at))

    @property
    def state(self) -> str:
        return self._state

    def to_dict(self) -> dict:
        return {"name": self.name, "state": self._state, "failures": self._failures,
                "recovery_timeout": self.recovery_timeout}


#: 各下游的熔断器
breakers: dict[str, CircuitBreaker] = {}


def breaker_for(service: str) -> CircuitBreaker:
    if service not in breakers:
        breakers[service] = CircuitBreaker(service)
    return breakers[service]


def breakers_snapshot() -> dict:
    return {k: v.to_dict() for k, v in breakers.items()}
