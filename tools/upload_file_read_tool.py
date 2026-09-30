"""文件内容读取工具，读取与解析都在会话沙箱内完成。

容器后端里上传件以只读挂载出现在 `/uploads`，工作区是读写挂载 `/workspace`；
读取先在工作区找，找不到再退到上传区，两条路都在沙箱内。
`../../` 这类穿越在宿主侧就被路径守卫拒绝，不会传进沙箱。

按后缀分两条路：
  文本类（.md/.txt/.json/.csv/.log）      -> 沙箱 read_file
  文档类（.docx/.pdf/.xlsx/.xls）          -> 沙箱 extract_document（python-docx/pypdf/pandas）
解析不可信文件的风险不放进宿主进程；沙箱缺解析库时返回结构化提示，不静默失败。
"""
import logging

from typing import Annotated

from langchain_core.tools import tool

from api.context import get_sandbox_context, get_thread_context
from api.monitor import monitor
from security.audit import audit
from security.permissions import Capability, requires
from tools.markdown_tools import audit_event, run_async

logger = logging.getLogger(__name__)

MAX_READ_BYTES = 2_000_000
MAX_EXTRACT_CHARS = 200_000

#: 走 extract_document 的文档格式，与沙箱 worker/ops.py 的 DOCUMENT_SUFFIXES 保持一致
DOC_SUFFIXES = (".docx", ".pdf", ".xlsx", ".xls")

#: 没有重试意义的错误码：换一个候选路径结果也一样，直接返回
_NO_RETRY_CODES = ("sandbox_path_denied", "suffix_not_allowed", "read_only_mount",
                   "unsupported_suffix", "missing_dependency")


@tool
@requires(Capability.FS_READ_SESSION)
def read_file_content(
        filename: Annotated[str, "要读取的文件名或路径（文本类支持 .md/.txt/.json/.csv/.log，"
                                  "文档类支持 .docx/.pdf/.xlsx/.xls）"],
        instruction: Annotated[str, "对提取内容的具体指令（例如：'提取摘要', '统计数据'）"] = "提取全部内容"
) -> str:
    """读取指定文件的内容。支持 Markdown/文本类格式（.md/.txt/.json/.csv/.log），
    以及文档格式（.docx/.pdf/.xlsx/.xls，解析成纯文本返回）。
    读取与解析都在会话的隔离沙箱内完成。上传的文件可以直接用文件名读取。"""
    monitor.report_tool("文件内容读取工具", {"filename": filename, "instruction": instruction})

    sandbox = get_sandbox_context()
    thread_id = get_thread_context() or ""

    if sandbox is None:
        audit_event("read_file_content", "deny", thread_id, str(filename), "sandbox_unavailable")
        return "【执行环境不可用】当前会话没有隔离沙箱，已拒绝读取。"

    is_document = str(filename).lower().endswith(DOC_SUFFIXES)

    # 先在工作区找，找不到再退到只读上传区（上传件会先复制进工作区，这里只是兜底）
    candidates = [filename, f"uploads/{filename}"]
    last_error = ""
    for candidate in candidates:
        try:
            if is_document:
                coro = sandbox.extract_document(candidate, max_chars=MAX_EXTRACT_CHARS)
            else:
                coro = sandbox.read_file(candidate, max_bytes=MAX_READ_BYTES)
            result = run_async(coro)
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
        # 权限/路径/格式/缺库类错误没有重试意义，直接返回
        if result.error_code in _NO_RETRY_CODES:
            break

    audit_event("read_file_content", "error", thread_id, str(filename), last_error[:200])
    return f"错误：无法读取 '{filename}'。({last_error})"
