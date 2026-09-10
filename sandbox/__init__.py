"""
沙箱子系统：会话执行环境的隔离与生命周期管理。

    base.py            抽象契约（SandboxBackend / Spec / ExecResult）
    config.py          策略配置（后端选择、资源上限、网络开关）
    docker_backend.py  容器后端
    process_backend.py 进程后端（Job Object 资源约束）
    manager.py         生命周期 + 并发闸门 + 空闲回收
    worker/ops.py      沙箱内执行器
    worker/Dockerfile  沙箱镜像定义
    build_image.py     镜像构建脚本

    from sandbox import sandbox_manager
    async with sandbox_manager.acquire(session_id, workspace=ws, uploads=up) as sb:
        await sb.write_file("report.md", content)
        await sb.render_pdf("report.md")
"""
from sandbox.base import (
    ExecResult,
    SandboxBackend,
    SandboxError,
    SandboxPathError,
    SandboxSpec,
    build_backend,
)
from sandbox.config import sandbox_settings
from sandbox.manager import SandboxManager, sandbox_manager

__all__ = [
    "ExecResult", "SandboxBackend", "SandboxError", "SandboxPathError", "SandboxSpec",
    "build_backend", "sandbox_settings", "SandboxManager", "sandbox_manager",
]
