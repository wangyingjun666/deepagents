"""路径解析工具，是 `security.path_guard` 的一层兼容包装。

保留这个文件是为了让 `resolve_path(filename, session_dir)` 这个调用名不变，
但语义是拒绝型的（deny-by-default）：会话外的绝对路径、`../` 穿越、符号链接
逃逸、Windows 保留名/ADS/结尾点一律拒绝，抛 `PathSecurityError`，绝不透传、
不降级为原路径。

它不负责猜模型想写到哪。路径不合法就把错误返回给模型让它纠正，
工具的返回文本就是对模型最好的反馈。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

from security.path_guard import (
    PathMode,
    PathSecurityError,
    guard_path,
    sanitize_filename,
)


def resolve_path(filename: str, session_dir: Optional[str] = None,
                 *, mode: PathMode = PathMode.WRITE,
                 extra_roots: Sequence[str] = ()) -> str:
    """把路径解析成**保证落在会话目录内**的宿主绝对路径。

    Args:
        filename: 待解析路径（只接受相对路径；容器风格的 `/workspace/...` 会被剥离前缀）。
        session_dir: 会话工作目录；为空时退回当前工作目录（脚本/测试场景）。
        mode: 读写模式，决定后缀白名单。
        extra_roots: 读模式下的附加只读根（例如上传目录）。

    Raises:
        PathSecurityError: 路径不合法或越出允许范围。

    Note:
        路径越界时不返回原路径。调用方把异常信息作为工具结果返回给模型，
        或让上层统一兜底，无论如何都不能"照常执行"。
    """
    root = session_dir or os.getcwd()
    target = guard_path(filename, root, mode=mode, extra_roots=extra_roots)
    return str(target)


def resolve_read_path(filename: str, session_dir: Optional[str] = None,
                      uploads_dir: Optional[str] = None) -> str:
    """读取场景的路径解析：允许读会话目录，也允许读上传目录（只读）。"""
    extra = [uploads_dir] if uploads_dir else []
    return resolve_path(filename, session_dir, mode=PathMode.READ, extra_roots=extra)


# 保留这些导出名，避免 import 失败
__all__ = ["resolve_path", "resolve_read_path", "PathSecurityError", "sanitize_filename"]
