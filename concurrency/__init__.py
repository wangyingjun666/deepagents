"""
并发治理：限流（背压）+ 重试退避 + 熔断 + 连接池。

    limiter.py   信号量闸门：全局会话数、单会话子 Agent 数、按下游服务分别限流
    retry.py     指数退避重试 + 三态熔断（CLOSED / OPEN / HALF_OPEN）
    db_pool.py   MySQL 连接池 + 只读账号池

典型链路：

    请求进来 → session_limiter 排队 → 子 Agent 并发 → subagent_limiter 闸门
             → 调下游 → downstream_limiter 排队 → with_retry 退避重试
             → 连续失败 → breaker 熔断快速失败 → 事件总线记录 → 指标聚合
"""
from concurrency.db_pool import ConnectionPools, is_readonly_account_configured, pools
from concurrency.limiter import (
    Limiter,
    LimiterTimeout,
    all_stats as limiter_stats,
    downstream_limiters,
    limiter_for,
    session_limiter,
    subagent_limiter,
)
from concurrency.retry import (
    CircuitBreaker,
    CircuitOpenError,
    RetryPolicy,
    breaker_for,
    breakers_snapshot,
    is_retryable,
    with_retry,
)

__all__ = [
    "ConnectionPools", "pools", "is_readonly_account_configured",
    "Limiter", "LimiterTimeout", "limiter_stats", "downstream_limiters", "limiter_for",
    "session_limiter", "subagent_limiter",
    "CircuitBreaker", "CircuitOpenError", "RetryPolicy", "breaker_for",
    "breakers_snapshot", "is_retryable", "with_retry",
]
