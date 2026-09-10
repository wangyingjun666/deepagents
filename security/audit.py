"""
审计日志：所有安全决策的不可抵赖记录。

用哈希链而非普通日志：每条记录的哈希把上一条的哈希算进去，
`hash_n = SHA256(prev_hash || canonical_json(record_n))`。任何一条被改动或
删除，其后所有记录的哈希都对不上，`verify_chain()` 能定位到篡改位置。

记录范围：能力校验拒绝、路径校验拒绝（含原始候选路径）、SQL 策略的拒绝与放行
（放行记改写后的 SQL 与哈希）、沙箱生命周期（创建 / 销毁 / 降级）、人工审批的
请求与决议。

写入是追加 + 立即 fsync：宁可慢一点，也不能因进程崩溃丢掉最后一条。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

GENESIS_HASH = "0" * 64


def _canonical(obj: dict) -> str:
    """稳定序列化：键排序 + 无多余空白，保证同样内容算出同样哈希。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass
class AuditRecord:
    seq: int
    ts: float
    action: str
    decision: str                 # allow / deny / error
    session_id: str = ""
    user_id: str = ""
    role: str = ""
    target: str = ""
    reason: str = ""
    extra: dict = field(default_factory=dict)
    prev_hash: str = GENESIS_HASH
    hash: str = ""

    def payload(self) -> dict:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "action": self.action,
            "decision": self.decision,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "role": self.role,
            "target": self.target,
            "reason": self.reason,
            "extra": self.extra,
        }

    def compute_hash(self) -> str:
        return hashlib.sha256((self.prev_hash + _canonical(self.payload())).encode("utf-8")).hexdigest()


class AuditLog:
    """按天分文件的哈希链审计日志（线程安全）。"""

    def __init__(self, directory: str | os.PathLike | None = None):
        root = Path(directory or os.getenv("AUDIT_DIR", "logs/audit"))
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[1] / root
        self.directory = root
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._seq = 0
        self._last_hash = GENESIS_HASH
        self._last_day = ""
        self._recover_tail()

    # ---------------- 内部：恢复链尾 ----------------
    def _day_file(self, day: str | None = None) -> Path:
        day = day or time.strftime("%Y%m%d")
        return self.directory / f"audit-{day}.jsonl"

    def _recover_tail(self) -> None:
        """进程重启后从当天文件尾部恢复 seq 与 last_hash，链条才不会断。"""
        f = self._day_file()
        if not f.exists():
            return
        last: dict | None = None
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line)
                    except json.JSONDecodeError:
                        continue
        if last:
            self._seq = int(last.get("seq", 0))
            self._last_hash = last.get("hash", GENESIS_HASH)
        self._last_day = time.strftime("%Y%m%d")

    # ---------------- 写入 ----------------
    def record(self, *, action: str, decision: str, session_id: str = "", user_id: str = "",
               role: str = "", target: str = "", reason: str = "", **extra: Any) -> AuditRecord:
        with self._lock:
            day = time.strftime("%Y%m%d")
            if day != self._last_day:            # 跨天/首次：重置链头（按天独立成链）
                self._last_day = day
                self._seq = 0
                self._last_hash = GENESIS_HASH

            rec = AuditRecord(
                seq=self._seq + 1,
                ts=time.time(),
                action=action,
                decision=decision,
                session_id=session_id,
                user_id=user_id,
                role=role,
                target=target,
                reason=reason,
                extra=extra or {},
                prev_hash=self._last_hash,
            )
            rec.hash = rec.compute_hash()
            self._seq = rec.seq
            self._last_hash = rec.hash

            line = json.dumps({**rec.payload(), "prev_hash": rec.prev_hash, "hash": rec.hash},
                              ensure_ascii=False)
            with self._day_file(day).open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())            # 崩溃也不丢最后一条
            return rec

    # ---------------- 校验 ----------------
    def verify_chain(self, day: str | None = None) -> tuple[bool, str]:
        """校验哈希链完整性，返回 (是否完整, 说明)。"""
        f = self._day_file(day)
        if not f.exists():
            return True, "无审计文件"
        prev = GENESIS_HASH
        expected_seq = 0
        with f.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    return False, f"第 {lineno} 行不是合法 JSON"
                expected_seq += 1
                if obj.get("prev_hash") != prev:
                    return False, f"第 {lineno} 行 prev_hash 与上一条不匹配（疑似删除/插入）"
                payload = {k: obj.get(k) for k in
                           ("seq", "ts", "action", "decision", "session_id", "user_id",
                            "role", "target", "reason", "extra")}
                recalc = hashlib.sha256((prev + _canonical(payload)).encode("utf-8")).hexdigest()
                if recalc != obj.get("hash"):
                    return False, f"第 {lineno} 行内容被篡改（哈希不匹配）"
                if obj.get("seq") != expected_seq:
                    return False, f"第 {lineno} 行序号不连续（期望 {expected_seq}）"
                prev = obj["hash"]
        return True, f"共 {expected_seq} 条记录，链完整"

    # ---------------- 读取 ----------------
    def tail(self, limit: int = 200, day: str | None = None) -> list[dict]:
        f = self._day_file(day)
        if not f.exists():
            return []
        lines = f.read_text(encoding="utf-8").splitlines()
        return [json.loads(x) for x in lines[-limit:] if x.strip()]

    def query(self, *, session_id: str | None = None, decision: str | None = None,
              action_prefix: str | None = None, limit: int = 200) -> list[dict]:
        out: list[dict] = []
        for rec in reversed(self.tail(limit=10000)):
            if session_id and rec.get("session_id") != session_id:
                continue
            if decision and rec.get("decision") != decision:
                continue
            if action_prefix and not str(rec.get("action", "")).startswith(action_prefix):
                continue
            out.append(rec)
            if len(out) >= limit:
                break
        return out


#: 全局单例
audit = AuditLog()


def _now() -> float:
    return time.time()
