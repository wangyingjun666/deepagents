"""
沙箱配置：后端选择、资源上限、网络开关集中到一处，全部可用环境变量覆盖。

默认值取安全优先：

* `SANDBOX_BACKEND=auto` 有 Docker 用 Docker，没有降级到进程级
* `SANDBOX_NETWORK=0` 默认断网。联网检索走宿主侧 Tavily 工具，容器不需要出口；
  断网后即使容器内跑到恶意代码也发不出数据
* `SANDBOX_CPUS` / `SANDBOX_MEMORY_MB` / `SANDBOX_PIDS_LIMIT` 单会话资源上限
* `SANDBOX_READ_ONLY_ROOTFS=1` 根文件系统只读，只有 /workspace 可写、/tmp 用 tmpfs
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from sandbox.base import SandboxSpec


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class SandboxSettings:
    """沙箱全局设置。"""

    backend: str = "auto"                 # docker | process | auto
    image: str = "deep-search-sandbox:latest"
    network: bool = False
    read_only_rootfs: bool = True
    cpus: float = 1.0
    memory_mb: int = 512
    pids_limit: int = 128
    tmpfs_mb: int = 64
    exec_timeout_sec: int = 60
    render_timeout_sec: int = 180
    idle_ttl_sec: int = 900               # 空闲多久回收容器
    max_concurrent: int = 4               # 全局同时存活的沙箱数（背压）
    startup_timeout_sec: int = 60         # 容器启动等待上限
    container_prefix: str = "ds-sbx-"
    keep_alive: bool = False              # 调试用：进程退出不销毁容器
    registry_prefix: str = ""             # 私有镜像仓库前缀（内网部署用）

    _fallback_reason: str = ""
    _lock: threading.Lock = None          # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    # ---------------- 降级记录 ----------------
    def record_fallback(self, reason: str) -> None:
        """记录一次"想用容器但退回了进程级"的事件，同时写审计。"""
        with self._lock:
            self._fallback_reason = reason
        try:
            from security.audit import audit
            audit.record(action="sandbox:backend_fallback", decision="allow",
                         target="process", reason=reason[:300])
        except Exception:
            pass

    @property
    def fallback_reason(self) -> str:
        with self._lock:
            return self._fallback_reason

    # ---------------- 规格 ----------------
    def build_spec(self) -> SandboxSpec:
        return SandboxSpec(
            cpus=self.cpus,
            memory_mb=self.memory_mb,
            pids_limit=self.pids_limit,
            network_enabled=self.network,
            tmpfs_mb=self.tmpfs_mb,
            exec_timeout_sec=self.exec_timeout_sec,
            idle_ttl_sec=self.idle_ttl_sec,
        )

    def describe(self) -> dict:
        return {
            "backend": self.backend,
            "image": self.image,
            "network_enabled": self.network,
            "read_only_rootfs": self.read_only_rootfs,
            "cpus": self.cpus,
            "memory_mb": self.memory_mb,
            "pids_limit": self.pids_limit,
            "max_concurrent": self.max_concurrent,
            "fallback_reason": self.fallback_reason,
        }


def _load() -> SandboxSettings:
    return SandboxSettings(
        backend=os.getenv("SANDBOX_BACKEND", "auto").strip().lower(),
        image=os.getenv("SANDBOX_IMAGE", "deep-search-sandbox:latest"),
        network=_env_bool("SANDBOX_NETWORK", False),
        read_only_rootfs=_env_bool("SANDBOX_READ_ONLY_ROOTFS", True),
        cpus=_env_float("SANDBOX_CPUS", 1.0),
        memory_mb=_env_int("SANDBOX_MEMORY_MB", 512),
        pids_limit=_env_int("SANDBOX_PIDS_LIMIT", 128),
        tmpfs_mb=_env_int("SANDBOX_TMPFS_MB", 64),
        exec_timeout_sec=_env_int("SANDBOX_EXEC_TIMEOUT_SEC", 60),
        render_timeout_sec=_env_int("SANDBOX_RENDER_TIMEOUT_SEC", 180),
        idle_ttl_sec=_env_int("SANDBOX_IDLE_TTL_SEC", 900),
        max_concurrent=_env_int("SANDBOX_MAX_CONCURRENT", 4),
        startup_timeout_sec=_env_int("SANDBOX_STARTUP_TIMEOUT_SEC", 60),
        container_prefix=os.getenv("SANDBOX_CONTAINER_PREFIX", "ds-sbx-"),
        keep_alive=_env_bool("SANDBOX_KEEP_ALIVE", False),
        registry_prefix=os.getenv("SANDBOX_REGISTRY_PREFIX", ""),
    )


sandbox_settings = _load()
