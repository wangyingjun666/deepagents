"""
沙箱管理器：会话与隔离环境的生命周期、并发上限、空闲回收。

- 生命周期：会话开始 acquire，结束 release，用 async context manager 表达，
  异常路径也会回收。
- 并发闸门：每个沙箱约占 1 核 + 512MB，不限流的话多用户同时提问会打爆宿主机。
  用 asyncio.Semaphore 拦截，超过 SANDBOX_MAX_CONCURRENT 的请求排队等名额。
- 空闲回收：除显式 release 外，后台 reaper 按 idle_ttl_sec 回收被遗忘的沙箱
  （页面关闭、WebSocket 断开但任务已跑完）。
- 同一会话复用同一个沙箱，per-session asyncio.Lock 防止并发请求创建出两个容器。
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from sandbox.base import SandboxBackend, SandboxError, build_backend
from sandbox.config import sandbox_settings

logger = logging.getLogger(__name__)


class SandboxManager:
    """全局沙箱注册表 + 并发闸门 + 空闲回收。"""

    def __init__(self) -> None:
        self._sandboxes: dict[str, SandboxBackend] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(sandbox_settings.max_concurrent)
        self._reaper: asyncio.Task | None = None
        self._probe_result = None
        self._stats = {"created": 0, "reused": 0, "destroyed": 0, "waited": 0}

    # ------------------------------------------------------------------
    # 启动期探测：确认实际跑在哪个后端
    # ------------------------------------------------------------------
    def probe(self) -> dict:
        """探测沙箱运行时就绪情况（同步，启动时调用一次）。"""
        from sandbox.base import build_backend as _  # noqa: F401
        info: dict = {"configured": sandbox_settings.backend}
        try:
            from sandbox.docker_backend import DockerSandboxBackend
            probe = DockerSandboxBackend.probe()
            info.update({
                "docker_available": probe.ok,
                "docker_detail": probe.stdout or probe.stderr,
            })
            if not probe.ok:
                sandbox_settings.record_fallback(probe.stderr)
                info["effective"] = "process"
            else:
                info["effective"] = "docker" if sandbox_settings.backend != "process" else "process"
        except Exception as exc:
            info.update({"docker_available": False, "docker_detail": str(exc), "effective": "process"})
        if sandbox_settings.backend == "process":
            info["effective"] = "process"
        info["spec"] = sandbox_settings.describe()
        self._probe_result = info
        return info

    @property
    def probe_result(self) -> dict:
        return self._probe_result or {}

    # ------------------------------------------------------------------
    async def _get_lock(self, session_id: str) -> asyncio.Lock:
        async with self._guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[session_id] = lock
            return lock

    @asynccontextmanager
    async def acquire(self, session_id: str, *, workspace: str | Path,
                      uploads: str | Path | None = None) -> AsyncIterator[SandboxBackend]:
        """获取或复用会话沙箱。退出上下文时不销毁，留在池里复用，
        销毁交给 release() 或空闲 reaper。"""
        lock = await self._get_lock(session_id)
        async with lock:
            sandbox = self._sandboxes.get(session_id)
            if sandbox is not None and sandbox.started:
                self._stats["reused"] += 1
                yield sandbox
                return

            # 并发闸门：拿不到名额就在这里排队
            if self._semaphore.locked():
                self._stats["waited"] += 1
                logger.info("沙箱并发已达上限 %s，会话 %s 排队等待",
                            sandbox_settings.max_concurrent, session_id)
            await self._semaphore.acquire()
            try:
                sandbox = build_backend(session_id, workspace=Path(workspace),
                                        uploads=Path(uploads) if uploads else None,
                                        spec=sandbox_settings.build_spec())
                await sandbox.start()
                self._sandboxes[session_id] = sandbox
                self._stats["created"] += 1
                self._ensure_reaper()
                yield sandbox
            except Exception:
                self._semaphore.release()
                raise

    async def release(self, session_id: str, *, keep: bool = False) -> None:
        """结束会话：销毁沙箱并归还并发名额。"""
        async with self._guard:
            sandbox = self._sandboxes.pop(session_id, None)
            self._locks.pop(session_id, None)
        if sandbox is None:
            return
        if not keep:
            try:
                await sandbox.stop()
            except Exception:
                logger.exception("销毁沙箱失败 session=%s", session_id)
        self._stats["destroyed"] += 1
        try:
            self._semaphore.release()
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # 空闲回收
    # ------------------------------------------------------------------
    def _ensure_reaper(self) -> None:
        if self._reaper is None or self._reaper.done():
            try:
                self._reaper = asyncio.get_running_loop().create_task(self._reap_loop())
            except RuntimeError:
                pass

    async def _reap_loop(self) -> None:
        ttl = sandbox_settings.idle_ttl_sec
        while True:
            await asyncio.sleep(max(30, ttl // 4))
            now = time.time()
            stale = [sid for sid, sb in list(self._sandboxes.items())
                     if now - getattr(sb, "last_used", now) > ttl]
            for sid in stale:
                logger.info("回收空闲沙箱 session=%s（闲置超过 %ss）", sid, ttl)
                await self.release(sid)

    async def shutdown(self) -> None:
        """进程退出时清理，避免留下孤儿容器/进程。"""
        if self._reaper:
            self._reaper.cancel()
        for sid in list(self._sandboxes):
            await self.release(sid)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            **self._stats,
            "alive": len(self._sandboxes),
            "max_concurrent": sandbox_settings.max_concurrent,
            "sessions": list(self._sandboxes),
            "backend": (next(iter(self._sandboxes.values())).name
                        if self._sandboxes else self.probe_result.get("effective", "unknown")),
        }

    def describe(self, session_id: str) -> dict:
        sb = self._sandboxes.get(session_id)
        return sb.meta if sb else {}

    def is_isolated(self, session_id: str) -> bool:
        sb = self._sandboxes.get(session_id)
        return bool(sb and sb.meta.get("isolated"))


#: 全局单例
sandbox_manager = SandboxManager()
