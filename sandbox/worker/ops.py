#!/usr/bin/env python3
"""
沙箱内操作执行器 —— 跑在容器里 / 受限子进程里的那一半。

协议
----
stdin 收一行 JSON：  {"op": "...", ...参数}
stdout 回一行 JSON：  {"ok": bool, "stdout": str, "stderr": str, "exit_code": int, ...}

一律返回结构化 JSON，不靠退出码传业务错误：调用方（宿主）要能区分"沙箱没起来"
（框架错误）和"操作被沙箱拒绝"（安全事件）。

路径校验
--------
宿主 `security.path_guard` 已经拦过一遍。这里再拦一次是纵深防御：两段代码的
信任边界不同，宿主守卫防"模型骗过宿主"，容器守卫防"宿主代码写错或有人绕过宿主
直接调沙箱"。容器里的策略是挂载点级别的硬约束（/workspace 可写、/uploads 只读）。

命令白名单
----------
`exec` 只放行极小的可执行文件集合，永远 shell=False（不做字符串拼接，没有 shell
注入面）。`python3` 由环境变量 `SANDBOX_ALLOW_CODE_EXEC` 单独控制，默认关闭。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 工作区/上传区根路径可配置：
#   容器后端 -> /workspace、/uploads（Dockerfile 里定的挂载点）
#   进程级后端 -> 宿主机上的会话目录与上传目录（由 process_backend 注入环境变量）
WORKSPACE = os.getenv("SANDBOX_WORKSPACE", "/workspace")
UPLOADS = os.getenv("SANDBOX_UPLOADS", "/uploads")

def _load_guard():
    """加载沙箱内路径守卫。

    两条加载路径对应两种部署形态：
      * 容器后端：`path_guard.py` 被 COPY 到 /opt/sandbox，与 ops.py 同目录，直接 import
      * 进程级后端：ops.py 跑在宿主上，用的是仓库根目录的 `security/path_guard.py`
    """
    try:
        from path_guard import guard_within_sandbox as g   # 容器内
        return g
    except ImportError:
        pass
    try:
        root = Path(__file__).resolve().parents[2]         # deep_search/
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from security.path_guard import guard_within_sandbox as g
        return g
    except Exception:
        return None


_guard = _load_guard()


def guard_within_sandbox(rel_path: str, *, allow_uploads: bool = False) -> str:
    if _guard is not None:
        return _guard(rel_path, allow_uploads=allow_uploads,
                      workspace=WORKSPACE, uploads=UPLOADS)
    base = UPLOADS if allow_uploads else WORKSPACE
    return os.path.realpath(os.path.join(base, str(rel_path).lstrip("/")))


MAX_WRITE_BYTES = 8 * 1024 * 1024

#: 沙箱内允许写入的后缀白名单，与宿主守卫的默认值一致（纵深防御）。
WRITE_SUFFIXES = {
    s.strip().lower()
    for s in os.getenv("SANDBOX_WRITE_SUFFIXES", ".md,.txt,.json,.csv,.html").split(",")
    if s.strip()
}

#: 允许执行的命令名白名单，全是不带写能力的只读工具。
#: 实际路径用 shutil.which 在运行时解析：容器（Linux）和进程级后端（Windows 宿主）
#: 里同一个工具的落点不同，写死绝对路径会在另一平台上失效。
ALLOWED_COMMAND_NAMES = {
    "ls", "cat", "head", "tail", "wc", "grep", "find", "file", "stat", "sort", "more",
    "findstr",          # Windows 上的 grep
    "where",            # Windows 上的 which
}


def _resolve_command(name: str) -> str | None:
    """把白名单里的命令名解析成当前平台上的可执行文件路径。"""
    if name == "python3":
        return sys.executable
    if name not in ALLOWED_COMMAND_NAMES:
        return None
    return shutil.which(name)


ALLOWED_COMMANDS = {name: [] for name in ALLOWED_COMMAND_NAMES}
if os.getenv("SANDBOX_ALLOW_CODE_EXEC", "0").strip().lower() in ("1", "true", "yes"):
    ALLOWED_COMMANDS["python3"] = []


class OpError(Exception):
    def __init__(self, message: str, code: str = "op_error"):
        super().__init__(message)
        self.code = code


def _ok(stdout: str = "", **extra) -> dict:
    return {"ok": True, "stdout": stdout, "stderr": "", "exit_code": 0, **extra}


def _err(message: str, code: str = "op_error", exit_code: int = 1) -> dict:
    return {"ok": False, "stdout": "", "stderr": message, "exit_code": exit_code, "error_code": code}


# --------------------------------------------------------------------------
# 各操作实现
# --------------------------------------------------------------------------
def op_write_file(params: dict) -> dict:
    rel = str(params.get("path", ""))
    content = params.get("content", "")
    if not isinstance(content, str):
        return _err("content 必须是字符串", "bad_params")
    if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
        return _err(f"内容超过单文件上限 {MAX_WRITE_BYTES} 字节", "file_too_large")

    suffix = Path(rel.replace("\\", "/")).suffix.lower()
    if suffix and suffix not in WRITE_SUFFIXES:
        return _err(f"后缀 {suffix!r} 不在沙箱写入白名单内：{sorted(WRITE_SUFFIXES)}",
                    "suffix_not_allowed")

    # 上传目录是只读挂载，显式拒绝写入，否则会落到工作区里的同名子目录
    if rel.replace("\\", "/").lstrip("/").startswith("uploads/"):
        return _err("上传目录是只读挂载，不允许写入", "read_only_mount")

    try:
        target = Path(guard_within_sandbox(rel))
    except Exception as exc:
        return _err(f"路径被沙箱拒绝：{exc}", "sandbox_path_denied")

    if str(target).replace("\\", "/").startswith(str(UPLOADS).replace("\\", "/")):
        return _err("上传目录是只读挂载，不允许写入", "read_only_mount")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return _err(f"写入失败：{exc}", "write_failed")
    return _ok(f"已写入 {target}（{len(content.encode('utf-8'))} 字节）",
               path=str(target), bytes=len(content.encode("utf-8")))


def op_read_file(params: dict) -> dict:
    rel = str(params.get("path", ""))
    max_bytes = int(params.get("max_bytes", 2_000_000))
    try:
        target = Path(guard_within_sandbox(rel, allow_uploads=True))
    except Exception as exc:
        return _err(f"路径被沙箱拒绝：{exc}", "sandbox_path_denied")

    if not target.exists():
        return _err(f"文件不存在：{rel}", "not_found")
    if target.is_dir():
        return _err(f"{rel} 是目录不是文件", "is_directory")

    size = target.stat().st_size
    if size > max_bytes:
        return _err(f"文件大小 {size} 字节超过读取上限 {max_bytes}", "file_too_large")
    try:
        return _ok(target.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        return _err(f"读取失败：{exc}", "read_failed")


def op_list_files(params: dict) -> dict:
    rel = str(params.get("path", ".") or ".")
    try:
        target = Path(guard_within_sandbox(rel, allow_uploads=True))
    except Exception as exc:
        return _err(f"路径被沙箱拒绝：{exc}", "sandbox_path_denied")
    if not target.exists():
        return _err(f"目录不存在：{rel}", "not_found")

    items = []
    for p in sorted(target.rglob("*")):
        try:
            items.append({
                "name": p.name,
                "rel": str(p).replace(WORKSPACE + "/", "").replace(UPLOADS + "/", "uploads/"),
                "size": p.stat().st_size if p.is_file() else 0,
                "is_dir": p.is_dir(),
            })
        except OSError:
            continue
    return _ok(json.dumps(items, ensure_ascii=False), count=len(items), items=items)


def op_md_to_pdf(params: dict) -> dict:
    """把工作区里的 .md 渲染成 .pdf。

    链路：markdown -> HTML（带 CSS 与中文字体）-> LibreOffice headless -> PDF。
    绕 HTML 是因为 LibreOffice 的 Markdown 导入过滤器版本差异大、不可靠，HTML
    导入才是稳定路径。整个渲染在容器里完成，宿主不依赖 Windows + Office。
    """
    rel_md = str(params.get("md", ""))
    rel_pdf = params.get("pdf")

    try:
        md_path = Path(guard_within_sandbox(rel_md))
    except Exception as exc:
        return _err(f"路径被沙箱拒绝：{exc}", "sandbox_path_denied")
    if not md_path.exists():
        return _err(f"源文件不存在：{rel_md}", "not_found")

    pdf_name = Path(str(rel_pdf)).name if rel_pdf else md_path.with_suffix(".pdf").name
    if not pdf_name.lower().endswith(".pdf"):
        pdf_name += ".pdf"
    pdf_path = md_path.parent / pdf_name

    try:
        import markdown as md_lib
    except ImportError:
        return _err("沙箱镜像缺少 markdown 库", "missing_dependency")

    html = md_lib.markdown(md_path.read_text(encoding="utf-8"),
                           extensions=["tables", "fenced_code", "sane_lists"])
    html_doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
        @page {{ size: A4; margin: 18mm 16mm; }}
        body {{ font-family: "Noto Sans CJK SC","WenQuanYi Zen Hei",sans-serif; font-size: 11pt; line-height: 1.65; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #333; padding: 6px; }}
        th {{ background: #eef2f8; }}
        pre, code {{ font-family: monospace; background: #f4f4f4; }}
    </style></head><body>
    {html}
    </body></html>"""
    html_path = md_path.with_suffix(".render.html")
    html_path.write_text(html_doc, encoding="utf-8")

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        # 进程级后端跑在 Windows 宿主机上，没有 LibreOffice 时退回 Word COM。
        # 容器后端永远走 LibreOffice，不走这条降级路径。
        return _render_with_word_com(html_path, pdf_path)

    # 只读根文件系统下必须把 LibreOffice 的用户配置目录指到 tmpfs，否则启动即失败
    profile = f"file://{os.getenv('SANDBOX_LO_PROFILE', '/tmp/lo_profile')}"
    cmd = [soffice, "--headless", "--norestore", "--invisible", "--nolockcheck",
           f"-env:UserInstallation={profile}",
           "--convert-to", "pdf:writer_pdf_Export",
           "--outdir", str(md_path.parent), str(html_path)]

    proc = subprocess.run(cmd, capture_output=True, timeout=150, shell=False)
    produced = md_path.parent / (html_path.stem + ".pdf")
    if not produced.exists():
        # LibreOffice 可能把输出名定为 html 的 stem，兜底改名成期望的文件名
        candidates = list(md_path.parent.glob("*.pdf"))
        if pdf_path.exists():
            produced = pdf_path
        elif candidates:
            produced = max(candidates, key=lambda p: p.stat().st_mtime)
        else:
            return _err(f"PDF 渲染失败：{proc.stderr.decode('utf-8', 'replace')[:400]}", "render_failed")

    if produced != pdf_path:
        produced.replace(pdf_path)

    return _ok(f"已生成 PDF：{pdf_path.name}（{pdf_path.stat().st_size} 字节）",
               path=str(pdf_path), bytes=pdf_path.stat().st_size)


def _render_with_word_com(html_path: Path, pdf_path: Path) -> dict:
    """Windows 兜底渲染：调用本机 Word COM，只有进程级后端会走到这里。"""
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        return _err("沙箱内既没有 LibreOffice 也没有 Word COM（pywin32）", "missing_dependency")

    word = None
    pythoncom.CoInitialize()
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = False
        doc = word.Documents.Open(str(html_path.resolve()))
        doc.SaveAs(str(pdf_path.resolve()), FileFormat=17)   # wdFormatPDF
        doc.Close(SaveChanges=0)
    except Exception as exc:
        return _err(f"Word COM 渲染失败：{exc}", "render_failed")
    finally:
        try:
            if word:
                word.Quit()
        finally:
            pythoncom.CoUninitialize()

    if not pdf_path.exists():
        return _err("Word COM 未产出 PDF", "render_failed")
    return _ok(f"已生成 PDF（Word COM 降级路径）：{pdf_path.name}",
               path=str(pdf_path), bytes=pdf_path.stat().st_size)


def op_exec(params: dict) -> dict:
    """执行白名单内的只读命令。永远 shell=False，不做字符串拼接。"""
    argv = params.get("argv") or []
    if not isinstance(argv, list) or not argv:
        return _err("argv 必须是非空数组", "bad_params")

    name = str(argv[0])
    if name not in ALLOWED_COMMANDS:
        return _err(f"命令 {name!r} 不在沙箱白名单内：{sorted(ALLOWED_COMMANDS)}", "command_denied")

    binary = _resolve_command(name)
    if not binary:
        return _err(f"命令 {name!r} 在当前平台不可用（未找到可执行文件）", "command_unavailable")

    resolved = [binary, *[str(a) for a in argv[1:]]]
    timeout = int(params.get("timeout", 30))
    try:
        proc = subprocess.run(resolved, capture_output=True, timeout=timeout,
                              shell=False, cwd=WORKSPACE)
    except subprocess.TimeoutExpired:
        return _err(f"命令执行超时（{timeout}s）", "timeout")
    except OSError as exc:
        return _err(f"命令执行失败：{exc}", "exec_failed")

    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    if proc.returncode != 0:
        return {"ok": False, "stdout": out, "stderr": err,
                "exit_code": proc.returncode, "error_code": "nonzero_exit"}
    return _ok(out, stderr=err)


def _probe_write(path: str) -> bool:
    try:
        p = Path(path)
        p.write_text("probe", encoding="utf-8")
        p.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _probe_read(path: str) -> bool:
    try:
        os.listdir(path)
        return True
    except Exception:
        return False


def _system_root() -> str:
    """当前平台的系统根目录，用来探测根文件系统是否真的不可写。"""
    if os.name == "nt":
        return os.environ.get("SystemRoot", r"C:\Windows")
    return "/etc"


def op_health(params: dict) -> dict:
    """沙箱自检：报告隔离能力与运行时信息，前端与审计都用它。

    报告里的值都是当场探出来的：根文件系统真只读则 `system_root_writable` 为
    False，容器真断网则 `network` 为 blocked。
    """
    report = {
        "backend_platform": os.name,
        "workspace_writable": _probe_write(os.path.join(WORKSPACE, ".probe")),
        "uploads_readable": _probe_read(UPLOADS),
        "uploads_writable": _probe_write(os.path.join(UPLOADS, ".probe")),
        "system_root_writable": _probe_write(os.path.join(_system_root(), ".sandbox_probe")),
        "uid": os.getuid() if hasattr(os, "getuid") else -1,
        "network": _probe_network(),
        "soffice": bool(shutil.which("soffice") or shutil.which("libreoffice")),
        "code_exec_enabled": "python3" in ALLOWED_COMMANDS,
        "python": sys.version.split()[0],
    }
    return _ok(json.dumps(report, ensure_ascii=False), health=report)


def _probe_network() -> str:
    import socket
    try:
        socket.setdefaulttimeout(2)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("1.1.1.1", 53))
        return "reachable"
    except Exception:
        return "blocked"


OPERATIONS = {
    "write_file": op_write_file,
    "read_file": op_read_file,
    "list_files": op_list_files,
    "md_to_pdf": op_md_to_pdf,
    "exec": op_exec,
    "health": op_health,
}


def main() -> int:
    started = time.time()
    raw = sys.stdin.readline()
    if not raw.strip():
        print(json.dumps(_err("未收到操作指令", "bad_request"), ensure_ascii=False))
        return 2
    try:
        request = json.loads(raw)
        op = request.get("op")
        handler = OPERATIONS.get(op)
        if handler is None:
            result = _err(f"未知操作 {op!r}，支持：{sorted(OPERATIONS)}", "unknown_op")
        else:
            result = handler(request)
    except json.JSONDecodeError as exc:
        result = _err(f"请求不是合法 JSON：{exc}", "bad_json")
    except Exception as exc:  # 内部异常也要保证输出是结构化 JSON
        result = _err(f"沙箱内部错误：{type(exc).__name__}: {exc}", "internal_error")

    result.setdefault("duration_ms", int((time.time() - started) * 1000))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
