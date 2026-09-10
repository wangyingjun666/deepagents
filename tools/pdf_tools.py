"""Markdown → PDF 转换工具，渲染在会话沙箱内完成。

链路：md → HTML（带 CSS 与中文字体声明）→ `soffice --convert-to pdf`。
绕一层 HTML 是因为 LibreOffice 各版本对 Markdown 的导入过滤器差异大、不可靠，
HTML 导入才是它的稳定路径。

容器根文件系统只读，LibreOffice 启动时要写用户配置目录会直接失败，所以用
`-env:UserInstallation=file:///tmp/lo_profile` 把配置目录指到 tmpfs 上。

降级路径：进程级后端跑在 Windows 宿主上且沙箱内没有 LibreOffice 时，会退回
Word COM。这条路径只存在于本地开发，容器后端永远走 LibreOffice。
"""
import logging
import time
from pathlib import Path

try:
    from typing import Annotated, Optional
except ImportError:
    from typing_extensions import Annotated, Optional

from langchain_core.tools import tool

from api.context import get_sandbox_context, get_thread_context
from api.monitor import monitor
from security.path_guard import PathSecurityError, sanitize_filename
from security.permissions import Capability, requires
from tools.markdown_tools import audit_event, run_async

logger = logging.getLogger(__name__)


@tool
@requires(Capability.DOC_RENDER)
def convert_md_to_pdf(
        md_filename: Annotated[str, "要转换的Markdown文档路径（包含.md后缀）"],
        pdf_filename: Annotated[Optional[str], "输出的PDF文件路径（可选，默认与源文件同名）"] = None
) -> str:
    """
    将Markdown文档转换为PDF。渲染在会话的隔离沙箱内完成（LibreOffice headless）。
    """
    started = time.time()
    monitor.report_tool("Markdown转PDF工具")

    sandbox = get_sandbox_context()
    thread_id = get_thread_context() or ""

    safe_md = sanitize_filename(Path(str(md_filename)).name)
    if not safe_md.lower().endswith(".md"):
        safe_md += ".md"

    safe_pdf = None
    if pdf_filename:
        safe_pdf = sanitize_filename(Path(str(pdf_filename)).name)
        if not safe_pdf.lower().endswith(".pdf"):
            safe_pdf += ".pdf"

    if sandbox is None:
        audit_event("convert_md_to_pdf", "deny", thread_id, safe_md, "sandbox_unavailable")
        return "【执行环境不可用】当前会话没有隔离沙箱，已拒绝渲染（不会降级为在宿主上调用 Office）。"

    try:
        result = run_async(sandbox.render_pdf(safe_md, safe_pdf))
    except PathSecurityError as exc:
        audit_event("convert_md_to_pdf", "deny", thread_id, safe_md, exc.reason)
        return f"【路径被拒绝】{exc.reason}"
    except Exception as exc:
        logger.exception("沙箱渲染失败")
        audit_event("convert_md_to_pdf", "error", thread_id, safe_md, str(exc)[:200])
        return f"转换失败: {exc}"

    duration = int((time.time() - started) * 1000)
    audit_event("convert_md_to_pdf", "allow" if result.ok else "error",
                thread_id, safe_md, result.error_code or "",
                backend=result.backend, isolated=result.isolated, duration_ms=duration)
    monitor.report_tool_end("Markdown转PDF工具", duration_ms=duration,
                            result_preview=result.output)

    if not result.ok:
        return f"转换失败：{result.output}"

    engine = "LibreOffice（容器内）" if result.isolated else "Word COM（本地降级路径）"
    return f"{result.output}  —— 渲染引擎：{engine}"
