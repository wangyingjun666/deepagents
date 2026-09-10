"""
数据库连接池 + 只读执行封装。

每次工具调用都新建连接会有两个问题：建连本身很贵（TCP 握手 + MySQL 鉴权握手），
高频调用下延迟明显；MySQL 的 `max_connections` 是硬上限，超了整个服务都会
`Too many connections`，不只是变慢。

这里做三件事：

* 池化：用 `mysql.connector.pooling.MySQLConnectionPool` 复用连接，池满时等待而不是
  无限建连（`pool_size` 就是背压点）。
* 只读账号：读连接走 `MYSQL_RO_USER/MYSQL_RO_PASSWORD`。应用层的 SQL 白名单可能有
  bug，但只有 SELECT 权限的数据库账号是数据库自己保证的，这是最后一道防线。
* 按会话记账：记录每个会话借了多少次连接、执行了多少条 SQL，便于定位是谁把数据库
  打爆了。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    from mysql.connector.pooling import MySQLConnectionPool
except ImportError:  # pragma: no cover
    mysql = None
    MySQLConnectionPool = None
    MySQLError = Exception


def _pool_config(readonly: bool) -> dict:
    """构造连接池配置。

    `use_pure=True`：mysql-connector-python 9.x 的 C 扩展在当前环境会抛
    `RuntimeError: Failed raising error`，走纯 Python 实现可以绕开，代价是性能略低。
    """
    if readonly:
        user = os.getenv("MYSQL_RO_USER") or os.getenv("MYSQL_USER")
        password = os.getenv("MYSQL_RO_PASSWORD") or os.getenv("MYSQL_PASSWORD")
    else:
        user = os.getenv("MYSQL_USER")
        password = os.getenv("MYSQL_PASSWORD")

    config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": user,
        "password": password,
        "database": os.getenv("MYSQL_DATABASE"),
        "charset": os.getenv("MYSQL_CHARSET", "utf8mb4"),
        "collation": os.getenv("MYSQL_COLLATION", "utf8mb4_unicode_ci"),
        "autocommit": True,
        "sql_mode": os.getenv("MYSQL_SQL_MODE", "TRADITIONAL"),
        "use_pure": True,
        "connection_timeout": int(os.getenv("MYSQL_CONNECT_TIMEOUT", "10")),
        "read_timeout": int(os.getenv("MYSQL_READ_TIMEOUT", "30")),
    }
    return {k: v for k, v in config.items() if v is not None}


@dataclass
class PoolStats:
    created: int = 0
    borrowed: int = 0
    errors: int = 0
    total_wait_ms: int = 0
    per_session_queries: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pool_created": self.created,
            "borrowed": self.borrowed,
            "errors": self.errors,
            "avg_wait_ms": int(self.total_wait_ms / self.borrowed) if self.borrowed else 0,
            "sessions": len(self.per_session_queries),
        }


class ConnectionPools:
    """读写两套连接池的懒加载持有者。"""

    def __init__(self) -> None:
        self._pools: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.stats = PoolStats()
        self._session_counts: dict[str, int] = {}

    def get(self, *, readonly: bool = True):
        """取（或首次创建）只读 / 读写连接池。只读用 "ro"，读写用 "rw"，两套独立。"""
        if MySQLConnectionPool is None:
            raise RuntimeError("未安装 mysql-connector-python")
        key = "ro" if readonly else "rw"
        with self._lock:
            pool = self._pools.get(key)
            if pool is None:
                cfg = _pool_config(readonly)
                if not cfg.get("user") or not cfg.get("database"):
                    raise RuntimeError("数据库核心配置缺失（MYSQL_USER / MYSQL_DATABASE）")
                pool = MySQLConnectionPool(
                    pool_name=f"deep_search_{key}",
                    pool_size=int(os.getenv("MYSQL_POOL_SIZE", "8")),
                    pool_reset_session=True,
                    **cfg,
                )
                self._pools[key] = pool
                self.stats.created += 1
                logger.info("已创建数据库连接池[%s] size=%s", key,
                            os.getenv("MYSQL_POOL_SIZE", "8"))
            return pool

    def borrow(self, *, readonly: bool = True, session_id: str = ""):
        """借一条连接，配 with 使用，退出时自动归还。传 session_id 会记一次借用。"""
        started = time.time()
        pool = self.get(readonly=readonly)
        try:
            conn = pool.get_connection()
        except Exception:
            self.stats.errors += 1
            raise
        self.stats.borrowed += 1
        self.stats.total_wait_ms += int((time.time() - started) * 1000)
        if session_id:
            with self._lock:
                self._session_counts[session_id] = self._session_counts.get(session_id, 0) + 1
        return conn

    def session_queries(self, session_id: str) -> int:
        """该会话至今借过多少次连接。"""
        with self._lock:
            return self._session_counts.get(session_id, 0)

    def describe(self) -> dict:
        return {**self.stats.to_dict(), "pools": list(self._pools)}


pools = ConnectionPools()


def is_readonly_account_configured() -> bool:
    """是否配置了独立的只读账号。返回 False 时"只读"只靠应用层保证。"""
    return bool(os.getenv("MYSQL_RO_USER"))
