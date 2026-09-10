"""
沙箱抽象层：定义"会话执行环境"的统一契约。

模型触发的副作用（写文件、读上传件、渲染 PDF、执行受控命令）都走这里，不在
FastAPI 进程里裸跑。两个实现：`DockerSandboxBackend`（每会话一个容器）和
`ProcessSandboxBackend`（每会话一个受 Job Object 约束的子进程）。上层只依赖本
模块接口，切换靠 `SANDBOX_BACKEND`。

接口只接受相对路径（相对会话工作区）。宿主侧由 `security.path_guard` 拦截绝对
路径，容器内再校验一次。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


class SandboxError(RuntimeError):
    """沙箱层面的执行失败（容器起不来、超时、内部错误）。"""

    def __init__(self, message: str, *, code: str = "sandbox_error"):
        super().__init__(message)
        self.code = code


class SandboxPathError(SandboxError):
    """沙箱内的路径越界/非法（与宿主 path_guard 相互独立）。"""

    def __init__(self, message: str):
        super().__init__(message, code="sandbox_path_denied")


@dataclass(slots=True)
class ExecResult:
    """一次沙箱内操作的执行结果。

    沙箱内的失败以返回值表达，不抛异常：上层把 stderr 直接喂回模型自我纠正，
    不必打断 Agent 循环。
    """

    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: int = 0
    error_code: str = ""
    backend: str = ""
    isolated: bool = False          # 是否处于真隔离环境（容器/受限进程）

    @property
    def output(self) -> str:
        return self.stdout if self.ok else (self.stderr or self.stdout)

    def __str__(self) -> str:  # 方便直接拼进 ToolMessage
        return self.output


@dataclass(slots=True)
class SandboxSpec:
    """沙箱的资源与网络策略，集中定义便于审计。"""

    cpus: float = 1.0
    memory_mb: int = 512
    pids_limit: int = 128
    disk_quota_mb: int = 1024
    network_enabled: bool = False          # 默认断网
    workspace_read_only: bool = False      # 工作区可写（只挂会话目录）
    tmpfs_mb: int = 64
    exec_timeout_sec: int = 60
    idle_ttl_sec: int = 900

    def to_dict(self) -> dict:
        return {
            "cpus": self.cpus,
            "memory_mb": self.memory_mb,
            "pids_limit": self.pids_limit,
            "network_enabled": self.network_enabled,
            "workspace_read_only": self.workspace_read_only,
            "exec_timeout_sec": self.exec_timeout_sec,
            "idle_ttl_sec": self.idle_ttl_sec,
        }


class SandboxBackend(abc.ABC):
    """会话沙箱后端契约。"""

    name: str = "abstract"

    def __init__(self, session_id: str, *, workspace: Path, uploads: Path | None = None,
                 spec: SandboxSpec | None = None):
        self.session_id = session_id
        self.workspace = Path(workspace)
        self.uploads = Path(uploads) if uploads else None
        self.spec = spec or SandboxSpec()
        self._started = False
        self._meta: dict = {}

    # ---------------- 生命周期 ----------------
    @property
    def started(self) -> bool:
        return self._started

    @property
    def meta(self) -> dict:
        """运行时可观测信息（容器 id / 进程 pid / 镜像版本），用于审计与前端展示。"""
        return dict(self._meta)

    @abc.abstractmethod
    async def start(self) -> None:
        """创建并启动隔离环境。幂等。"""

    @abc.abstractmethod
    async def stop(self) -> None:
        """销毁隔离环境并回收资源。幂等。"""

    async def __aenter__(self) -> "SandboxBackend":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    # ---------------- 能力 ----------------
    @abc.abstractmethod
    async def _invoke(self, op: str, payload: dict, *, timeout: int | None = None) -> ExecResult:
        """执行一次沙箱内操作（内部协议，业务层不直接调用）。"""

    # ---------------- 业务操作（统一入口） ----------------
    async def write_file(self, rel_path: str, content: str) -> ExecResult:
        return await self._invoke("write_file", {"path": rel_path, "content": content})

    async def read_file(self, rel_path: str, *, max_bytes: int = 2_000_000) -> ExecResult:
        return await self._invoke("read_file", {"path": rel_path, "max_bytes": max_bytes})

    async def list_files(self, rel_dir: str = ".") -> ExecResult:
        return await self._invoke("list_files", {"path": rel_dir})

    async def render_pdf(self, rel_md: str, rel_pdf: str | None = None) -> ExecResult:
        """容器内 LibreOffice headless 渲染 md -> pdf。"""
        return await self._invoke("md_to_pdf", {"md": rel_md, "pdf": rel_pdf},
                                  timeout=max(self.spec.exec_timeout_sec, 120))

    async def exec(self, argv: Iterable[str], *, timeout: int | None = None) -> ExecResult:
        """执行受控命令，仅允许白名单内的可执行文件（见 worker/ops.py）。"""
        return await self._invoke("exec", {"argv": list(argv)}, timeout=timeout)

    # ---------------- 宿主侧只读映射 ----------------
    def host_path(self, rel_path: str) -> Path:
        """把会话内相对路径映射回宿主路径，仅供宿主只读用途（前端列文件、下载）。

        写操作一律走沙箱，不走这里。
        """
        from security.path_guard import guard_path, PathMode
        return guard_path(rel_path, self.workspace, mode=PathMode.WRITE_LOCAL)


def build_backend(session_id: str, *, workspace: Path, uploads: Path | None = None,
                  spec: SandboxSpec | None = None) -> SandboxBackend:
    """按配置构造沙箱后端（Docker 优先，不可用时自动降级为进程级）。"""
    from sandbox.config import sandbox_settings

    backend_name = sandbox_settings.backend
    if backend_name in ("docker", "auto"):
        try:
            from sandbox.docker_backend import DockerSandboxBackend
            probe = DockerSandboxBackend.probe()
            if probe.ok:
                return DockerSandboxBackend(session_id, workspace=workspace,
                                            uploads=uploads, spec=spec)
            if backend_name == "docker":
                raise SandboxError(f"Docker 不可用：{probe.stderr or probe.stdout}", code="docker_unavailable")
            sandbox_settings.record_fallback(probe.stderr or probe.stdout)
        except ImportError as exc:  # pragma: no cover
            if backend_name == "docker":
                raise SandboxError(f"缺少 docker SDK：{exc}", code="docker_sdk_missing") from exc
            sandbox_settings.record_fallback(str(exc))

    from sandbox.process_backend import ProcessSandboxBackend
    return ProcessSandboxBackend(session_id, workspace=workspace, uploads=uploads, spec=spec)
