"""
路径安全守卫：所有文件读写的唯一入口，读侧写侧统一硬校验。

策略是拒绝型（deny-by-default）：解析结果不在允许的根目录之下就直接拒绝，
不透传、不降级为原路径。输入视为完全不可信（LLM 生成的 tool_call 参数、
HTTP 查询串），要防的绕过手法：

1. 相对路径穿越            `../../etc/passwd`、`..\\..\\windows\\win.ini`
2. 绝对路径越权            `D:\\secrets\\x.md`、`/etc/shadow`
3. 符号链接逃逸            `session_x/link -> C:\\Windows`，写 `link/system32/x`
4. Windows 盘符相对路径    `C:foo`（不是绝对路径，但会跳到 C 盘当前目录）
5. UNC / 设备路径          `\\\\server\\share\\x`、`\\\\.\\PhysicalDrive0`
6. NTFS 备用数据流(ADS)    `report.md:evil.exe`（写到同名文件的数据流里）
7. 保留设备名              `CON`、`NUL`、`COM1`、`LPT1`（写入会劫持设备）
8. 结尾点/空格             `report.md.`、`report.md `（Win32 会归一化掉，绕过后缀校验）
9. 前缀伪造                `/data/output_evil` 冒充 `/data/output` 的子目录
10. 编码/Unicode 变体       NFD 形式、控制字符、零宽字符、超长路径

纵深防御：先做语法层拒绝（上述绝大多数），再用 realpath 展开软链接后判包含关系，
最后容器内同名守卫（`sandbox/worker/ops.py`）再复核一次。任一层拒绝都中止操作并写审计。
"""
from __future__ import annotations

import os
import re
import unicodedata
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Iterable, Sequence

MAX_PATH_LEN = 240                     # 留出余量，避免触发 Windows MAX_PATH 边界
MAX_FILENAME_LEN = 120

# Windows 保留设备名（写入会劫持到设备，读取行为诡异）
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

# 允许写入的后缀（白名单，防"落一个 .exe/.bat 到工作区"这类事）
DEFAULT_WRITE_SUFFIXES = frozenset({".md", ".txt", ".json", ".csv", ".html"})
# 允许读取的后缀
DEFAULT_READ_SUFFIXES = frozenset({
    ".md", ".txt", ".json", ".csv", ".log",
    ".docx", ".pdf", ".xlsx", ".xls",
})

_ILLEGAL_CHARS = re.compile(r'[\x00-\x1f\x7f<>:"|?*]')
_ZERO_WIDTH = re.compile(r"[​-‏‪-‮⁠﻿]")


class PathMode(str, Enum):
    """校验模式：读 / 写 / 宿主侧写映射。"""

    READ = "read"
    WRITE = "write"
    WRITE_LOCAL = "write_local"    # 宿主侧只读映射用的别名，语义同 WRITE


class PathSecurityError(PermissionError):
    """路径违反安全策略：一律拒绝，不降级、不透传。"""

    def __init__(self, message: str, *, reason: str, candidate: str = ""):
        super().__init__(message)
        self.reason = reason          # 机器可读的拒绝原因，进审计日志
        self.candidate = candidate


def _reject(reason: str, candidate: str, detail: str) -> PathSecurityError:
    return PathSecurityError(f"路径被安全策略拒绝（{reason}）：{detail}", reason=reason,
                             candidate=candidate)


def sanitize_filename(name: str) -> str:
    """把任意字符串收敛成一个安全的单层文件名（不含目录分隔符）。"""
    name = unicodedata.normalize("NFC", str(name)).strip()
    name = _ZERO_WIDTH.sub("", name)
    name = _ILLEGAL_CHARS.sub("_", name.replace("/", "_").replace("\\", "_"))
    name = name.strip(" .")                     # 去掉结尾的点与空格（Win32 会归一化）
    name = name[:MAX_FILENAME_LEN]
    stem = Path(name).stem.upper()
    if stem in _WINDOWS_RESERVED:
        name = f"_{name}"
    return name or "untitled"


def _check_component_syntax(part: str, candidate: str) -> None:
    """逐个路径分量做语法层检查。"""
    if not part or part in (".", ".."):
        return
    if _ZERO_WIDTH.search(part):
        raise _reject("zero_width_char", candidate, f"分量包含零宽字符：{part!r}")
    if _ILLEGAL_CHARS.search(part):
        raise _reject("illegal_char", candidate, f"分量包含非法字符：{part!r}")
    if part != part.strip(" ."):
        raise _reject("trailing_dot_space", candidate, f"分量以点或空格结尾：{part!r}")
    if part.split(".")[0].upper() in _WINDOWS_RESERVED:
        raise _reject("reserved_device_name", candidate, f"分量是 Windows 保留设备名：{part!r}")
    if len(part) > MAX_FILENAME_LEN:
        raise _reject("name_too_long", candidate, f"分量过长（{len(part)} 字符）：{part[:40]}...")


def _normalize_candidate(candidate: str) -> str:
    """把各种奇怪写法归一化成后续可比较的形式。"""
    if candidate is None:
        raise _reject("empty_path", "", "路径为空")
    raw = str(candidate)

    if "\x00" in raw:
        raise _reject("nul_byte", raw, "路径包含 NUL 字节")

    # Unicode 归一化：NFD 等变体折叠成 NFC，避免"看起来一样但比较不等"的绕过
    norm = unicodedata.normalize("NFC", raw).strip()
    if not norm:
        raise _reject("empty_path", raw, "路径为空")

    if len(norm) > MAX_PATH_LEN:
        raise _reject("path_too_long", norm, f"路径长度 {len(norm)} 超过上限 {MAX_PATH_LEN}")

    # 统一分隔符，便于后续判定
    return norm.replace("\\", "/")


def _strip_virtual_prefixes(path_str: str) -> str:
    """剥离模型幻觉出来的虚拟工作目录前缀（sandbox 风格）。

    这些前缀（/workspace、/mnt/data、/home/user）在容器里是真实目录，宿主上是
    虚构的，剥离后按相对路径处理。剥离只对虚拟前缀生效，不会让 `/etc/passwd`
    变成相对路径。
    """
    for prefix in ("/workspace", "/mnt/data", "/home/user", "/sandbox"):
        if path_str == prefix:
            return "."
        if path_str.startswith(prefix + "/"):
            return path_str[len(prefix) + 1:]
    return path_str


def _is_absolute_like(path_str: str) -> bool:
    """判断是否属于绝对路径或驱动器相对路径，这两类一律拒绝。"""
    if path_str.startswith("//"):            # UNC / 设备路径 \\server\share
        return True
    if path_str.startswith("/"):             # Unix 风格根路径 /etc/passwd
        return True
    pw = PureWindowsPath(path_str)
    if pw.drive:                             # D:\x  或  C:foo（驱动器相对，会跳盘）
        return True
    return False


def guard_path(
    candidate: str,
    root: str | os.PathLike,
    *,
    mode: PathMode = PathMode.READ,
    extra_roots: Sequence[str | os.PathLike] = (),
    allowed_suffixes: Iterable[str] | None = None,
    must_exist: bool = False,
    reject_symlink: bool = True,
) -> Path:
    """把不可信路径解析成保证落在允许根目录下的宿主绝对路径。

    Args:
        candidate: 来自模型/HTTP 的不可信路径（只允许相对路径）。
        root: 主根目录（会话工作区）。
        mode: 读 / 写。写模式只允许落在 root 内；读模式还允许 extra_roots。
        extra_roots: 读模式下的附加只读根（例如上传目录）。
        allowed_suffixes: 后缀白名单，None 表示用模式默认值。
        must_exist: 是否要求目标已存在（读文件时用）。
        reject_symlink: 是否拒绝路径中出现的任何符号链接。

    Returns:
        已通过校验的宿主绝对 Path。

    Raises:
        PathSecurityError: 任何一条策略不通过。
    """
    path_str = _normalize_candidate(candidate)

    # ---- 1. 剥离虚拟前缀后，绝对路径一律拒绝 ----
    path_str = _strip_virtual_prefixes(path_str)
    if _is_absolute_like(path_str):
        raise _reject("absolute_path_denied", candidate,
                      f"只接受工作区内的相对路径，收到：{candidate!r}")

    # ---- 2. 逐分量语法检查 ----
    parts = [p for p in path_str.split("/") if p not in ("", ".")]
    if not parts:
        parts = []
    depth = 0
    for part in parts:
        _check_component_syntax(part, candidate)
        if part == "..":
            depth -= 1
            if depth < 0:
                raise _reject("parent_traversal", candidate,
                              f"路径向上越出工作区：{candidate!r}")
        else:
            depth += 1

    # ---- 3. 后缀白名单 ----
    if mode in (PathMode.WRITE, PathMode.WRITE_LOCAL):
        suffixes = set(allowed_suffixes) if allowed_suffixes is not None else DEFAULT_WRITE_SUFFIXES
    else:
        suffixes = set(allowed_suffixes) if allowed_suffixes is not None else DEFAULT_READ_SUFFIXES
    if parts and suffixes:
        suffix = Path(parts[-1]).suffix.lower()
        if suffix and suffix not in suffixes:
            raise _reject("suffix_not_allowed", candidate,
                          f"后缀 {suffix!r} 不在白名单内：{sorted(suffixes)}")

    # ---- 4. 拼接并做真实路径（realpath）层校验 ----
    root_real = Path(os.path.realpath(str(root)))
    target = Path(os.path.realpath(str(root_real / Path(*parts)))) if parts else root_real

    allowed_roots = [root_real]
    if mode == PathMode.READ:
        allowed_roots += [Path(os.path.realpath(str(r))) for r in extra_roots]

    if not _is_within(target, allowed_roots):
        raise _reject("outside_allowed_root", candidate,
                      f"解析后落在允许目录之外：{target}")

    # ---- 5. 符号链接检查（realpath 之后仍存在的链接 = 指向内部的链接）----
    if reject_symlink:
        probe = root_real
        for part in parts:
            probe = probe / part
            try:
                if probe.is_symlink():
                    raise _reject("symlink_denied", candidate,
                                  f"路径中包含符号链接：{probe}")
            except OSError:
                break

    if must_exist and not target.exists():
        raise FileNotFoundError(f"文件不存在：{target}")

    return target


def _is_within(target: Path, roots: Sequence[Path]) -> bool:
    """按路径分量判断包含关系。

    不能用裸字符串 startswith：`D:/data/output_evil` 会被 `D:/data/output` 的
    前缀匹配误判成子目录，必须补上分隔符才算真子路径。

    Windows 另外两个坑：`str(Path)` 用反斜杠，不统一分隔符会漏判；NTFS 大小写
    不敏感，`C:\\Temp` 与 `c:\\temp` 是同一个目录，不归一小写会误判为越界。
    """
    def key(p: Path | str) -> str:
        s = str(p).replace("\\", "/").rstrip("/")
        return s.lower() if os.name == "nt" else s

    t = key(target)
    for r in roots:
        rk = key(r)
        if t == rk or t.startswith(rk + "/"):
            return True
    return False


def guard_within_sandbox(rel_path: str, *, allow_uploads: bool = False,
                         workspace: str = "/workspace", uploads: str = "/uploads") -> str:
    """沙箱内部使用的相对路径校验（与宿主守卫相互独立，构成纵深防御）。

    容器里工作区固定挂在 /workspace、上传件只读挂在 /uploads；进程级后端则通过
    `workspace` / `uploads` 参数传入本机的会话目录。这里保证解析结果不逃出这两个根，
    即便宿主守卫被绕过这一层仍会拒绝。
    """
    p = _normalize_candidate(rel_path)
    p = _strip_virtual_prefixes(p)
    if _is_absolute_like(p):
        raise _reject("absolute_path_denied", rel_path, "沙箱内只接受相对路径")

    parts = [x for x in p.split("/") if x not in ("", ".")]
    depth = 0
    for part in parts:
        _check_component_syntax(part, rel_path)
        depth = depth - 1 if part == ".." else depth + 1
        if depth < 0:
            raise _reject("parent_traversal", rel_path, "沙箱内路径越出挂载点")

    use_uploads = allow_uploads and parts and parts[0] == "uploads"
    base = uploads if use_uploads else workspace
    if use_uploads:
        parts = parts[1:]
    joined = os.path.realpath(os.path.join(base, *parts)) if parts else os.path.realpath(base)
    if not _is_within(Path(joined), [Path(os.path.realpath(base))]):
        raise _reject("outside_allowed_root", rel_path, "沙箱内解析越界")
    return joined
