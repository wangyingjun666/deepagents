"""
事件落盘：JSONL 追加写入，按会话分文件。

事件只活在内存里的话，事后想查"那个失败的会话卡在哪一步"就没有依据，所以需要
持久化。

格式选 JSONL：追加写不用读全文件、可流式处理、`grep`/`jq` 直接可用、单行损坏不影响
其余行。

写入策略是追加 + flush，不做 fsync：事件量大，fsync 会拖慢主流程。审计日志才需要
fsync。
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterable


class EventStore:
    """按会话落 JSONL 的事件存储。"""

    def __init__(self, root: str | os.PathLike | None = None):
        base = Path(root or os.getenv("TRACE_DIR", "logs/traces"))
        if not base.is_absolute():
            base = Path(__file__).resolve().parents[1] / base
        self.root = base
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handles: dict[str, object] = {}

    def _path(self, session_id: str) -> Path:
        """会话 ID 里可能有路径分隔符等字符，先替换掉再拼文件名，并截断到 80 字符。"""
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(session_id))[:80]
        return self.root / f"trace_{safe or 'unknown'}.jsonl"

    def append(self, session_id: str, line: str) -> None:
        """追加一行。文件句柄按会话缓存，避免每次重开。"""
        with self._lock:
            fh = self._handles.get(session_id)
            if fh is None:
                fh = self._path(session_id).open("a", encoding="utf-8")
                self._handles[session_id] = fh
            fh.write(line + "\n")
            fh.flush()

    def read(self, session_id: str, *, after_event_id: int = 0, limit: int = 5000) -> list[dict]:
        """按行读回 event_id 大于游标的事件，最多 limit 条。"""
        path = self._path(session_id)
        if not path.exists():
            return []
        out: list[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue                       # 半截行（进程被杀）：跳过这一行
                if int(obj.get("event_id", 0)) > after_event_id:
                    out.append(obj)
                    if len(out) >= limit:
                        break
        return out

    def sessions(self) -> list[str]:
        """已落盘的所有会话 ID（从文件名反推），按字典序返回。"""
        return sorted(p.stem.replace("trace_", "", 1) for p in self.root.glob("trace_*.jsonl"))

    def close(self) -> None:
        """关闭所有缓存的文件句柄（进程退出前调用）。"""
        with self._lock:
            for fh in self._handles.values():
                try:
                    fh.close()
                except Exception:
                    pass
            self._handles.clear()


event_store = EventStore()
