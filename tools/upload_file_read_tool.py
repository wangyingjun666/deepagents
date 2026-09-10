"""文件内容读取工具，读取在会话沙箱内完成。

容器后端里上传件以只读挂载出现在 `/uploads`，工作区是读写挂载 `/workspace`；
读取先在工作区找，找不到再退到上传区，两条路都在沙箱内。
`../../` 这类穿越在宿主侧就被路径守卫拒绝，不会传进沙箱。
沙箱镜像只装了 markdown 解析，docx/pdf/xlsx 会明确提示需要在受控解析服务中处理，
不会静默失败。解析不可信文件的风险不放进宿主进程。
"""
import logging

from typing import Annotated, Optional

from langchain_core.tools import tool

from api.context import get_sandbox_context, get_thread_context
from api.monitor import monitor
from security.audit import audit
from security.permissions import Capability, requires
from tools.markdown_tools import audit_event, run_async

logger = logging.getLogger(__name__)

MAX_READ_BYTES = 2_000_000


@tool
@requires(Capability.FS_READ_SESSION)
def read_file_content(
        filename: Annotated[str, "要读取的文件名或路径（支持 .md, .txt, .json, .csv, .log）"],
        instruction: Annotated[str, "对提取内容的具体指令（例如：'提取摘要', '统计数据'）"] = "提取全部内容"
) -> str:
    """读取指定文件的内容。支持 Markdown/文本类格式（.md/.txt/.json/.csv/.log），
    读取在会话的隔离沙箱内完成。上传的文件可以先用文件名直接读取。"""
    monitor.report_tool("文件内容读取工具", {"filename": filename, "instruction": instruction})

    sandbox = get_sandbox_context()
    thread_id = get_thread_context() or ""

    if sandbox is None:
        audit_event("read_file_content", "deny", thread_id, str(filename), "sandbox_unavailable")
        return "【执行环境不可用】当前会话没有隔离沙箱，已拒绝读取。"

    # 先在工作区找，找不到再退到只读上传区（上传件会先复制进工作区，这里只是兜底）
    candidates = [filename, f"uploads/{filename}"]
    last_error = ""
    for candidate in candidates:
        try:
            result = run_async(sandbox.read_file(candidate, max_bytes=MAX_READ_BYTES))
        except Exception as exc:
            logger.exception("沙箱读取失败")
            audit_event("read_file_content", "error", thread_id, str(filename), str(exc)[:200])
            return f"读取文件出错: {exc}"

        if result.ok:
            audit_event("read_file_content", "allow", thread_id, str(candidate), "",
                        backend=result.backend, isolated=result.isolated,
                        bytes=len(result.stdout))
            return result.stdout

        last_error = result.output
        # 权限/路径类错误没有重试意义，直接返回
        if result.error_code in ("sandbox_path_denied", "suffix_not_allowed", "read_only_mount"):
            break

    audit_event("read_file_content", "error", thread_id, str(filename), last_error[:200])
    hint = ""
    if any(str(filename).lower().endswith(x) for x in (".docx", ".pdf", ".xlsx", ".xls")):
        hint = ("\n提示：该格式需要专门的解析服务，当前沙箱只放行文本类格式"
                "（.md/.txt/.json/.csv/.log），以避免在进程内解析不可信文件。")
    return f"错误：无法读取 '{filename}'。({last_error}){hint}"
