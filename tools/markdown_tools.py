"""Markdown 生成工具：只在会话沙箱内写文件。

写入链路：path_guard 校验 -> 能力校验（FS_WRITE_SESSION）-> 沙箱执行
（容器只挂载本会话目录；进程级后端是受限子进程，内部再校验一次）。
任何一层拒绝都返回可读错误文本给模型，不抛异常。工具返回值就是模型的输入，
抛异常会打断整轮 Agent 循环；返回文本模型还能自己纠正。
沙箱不可用时直接拒绝，不降级回直接写宿主磁盘。
"""
import asyncio
import concurrent.futures
import logging
import time
from pathlib import Path

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from langchain_core.tools import tool

from api.context import get_sandbox_context, get_session_context, get_thread_context
from api.monitor import monitor
from security.audit import audit
from security.path_guard import PathMode, PathSecurityError, guard_path, sanitize_filename
from security.permissions import Capability, requires

logger = logging.getLogger(__name__)


def run_async(coro):
    """在同步工具里驱动异步沙箱操作。

    工具保持同步函数：LangChain 的 `@tool` 支持同步函数，deepagents 在异步执行
    路径下会把同步工具丢进线程池，没必要为了接沙箱把所有工具改成 async。

    不能无脑用 `asyncio.run`：工具可能已经在事件循环里被调用，会报
    "asyncio.run() cannot be called from a running event loop"。所以分两路：
    已经在 loop 里就另开线程跑，否则直接 run。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def audit_event(action: str, decision: str, session_id: str, target: str,
                reason: str = "", **extra) -> None:
    try:
        audit.record(action=action, decision=decision, session_id=session_id,
                     target=target, reason=reason, **extra)
    except Exception:
        logger.exception("审计写入失败（已忽略）")


@tool
@requires(Capability.FS_WRITE_SESSION)
def generate_markdown(
        content: Annotated[str, "要写入Markdown文档的文本内容"],
        filename: Annotated[str, "Markdown文档的文件名（不包含扩展名或包含.md）"],
        path: Annotated[str, "文件保存的相对路径（留空表示写在会话工作目录根下）"] = ""
):
    """根据提供的文本内容，生成对应的Markdown(.md)文件。文件会写入当前会话的隔离工作目录。"""
    started = time.time()
    monitor.report_tool("Markdown文档生成工具", {"filename": filename})

    # 1. 文件名收敛：把任意字符串变成安全的单层 .md 文件名
    # 这里是"换后缀"而不是"补后缀"：模型给 `xx.exe` 时补成 `xx.exe.md` 也安全，
    # 但会留下可疑的文件名。统一收敛成 .md，产物类型可控。
    safe_name = sanitize_filename(filename or "untitled")
    if safe_name.lower().endswith(".md"):
        pass
    elif Path(safe_name).suffix:
        safe_name = str(Path(safe_name).with_suffix(".md"))
    else:
        safe_name = safe_name + ".md"

    session_dir = get_session_context()
    sandbox = get_sandbox_context()
    thread_id = get_thread_context() or ""

    # 2. 宿主侧路径校验（第一道闸门）
    # 不只拼路径：绝对路径、穿越、Windows 保留名、后缀不合法都在这里被拒。
    rel_path = safe_name
    if path and path not in (".", "./"):
        try:
            parent = guard_path(path, session_dir, mode=PathMode.WRITE)
            sub = Path(parent).relative_to(Path(session_dir)).as_posix()
            rel_path = safe_name if sub == "." else f"{sub}/{safe_name}"
        except PathSecurityError as exc:
            monitor.report_denied("path", f"路径被拒绝：{exc.reason}",
                                  {"candidate": path, "reason": exc.reason})
            audit_event("generate_markdown", "deny", thread_id, str(path), exc.reason)
            return (f"【路径被拒绝】{exc.reason}：{path}\n"
                    f"请使用会话工作目录内的相对路径（例如直接给文件名）。")

    # 3. 沙箱不可用时不降级
    if sandbox is None:
        audit_event("generate_markdown", "deny", thread_id, rel_path, "sandbox_unavailable")
        return "【执行环境不可用】当前会话没有隔离沙箱，已拒绝写入（不会降级为直接访问宿主磁盘）。"

    # 4. 交给沙箱执行
    try:
        result = run_async(sandbox.write_file(rel_path, content))
    except Exception as exc:                      # 沙箱通信层面异常
        logger.exception("沙箱写入失败")
        audit_event("generate_markdown", "error", thread_id, rel_path, str(exc)[:200])
        return f"生成Markdown文件失败：{exc}"

    duration = int((time.time() - started) * 1000)
    audit_event("generate_markdown", "allow" if result.ok else "error",
                thread_id, rel_path, result.error_code or "",
                backend=result.backend, isolated=result.isolated, bytes=len(content))
    monitor.report_tool_end("Markdown文档生成工具", duration_ms=duration,
                            result_preview=result.output)

    if not result.ok:
        return f"生成Markdown文件失败：{result.output}"

    where = "容器隔离环境" if result.isolated else "会话工作目录"
    return f"Markdown文件 '{rel_path}' 已成功生成并保存到{where}。"
