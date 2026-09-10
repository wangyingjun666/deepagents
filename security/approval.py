"""
人工审批（Human-in-the-Loop）：高风险动作执行前挂起，等人确认。

能力校验、路径守卫、SQL 白名单都是自动判定，覆盖不了"这条查询是不是用户真要的
数据"这类问题，所以对有副作用且不可逆的动作改为中断执行、交给人决定。

deepagents 的 `create_deep_agent(interrupt_on={...})` 在指定工具调用前抛
LangGraph 中断，图状态由 checkpointer 保存（用持久化 checkpointer 时中断可以
跨越进程重启，这也是选用 SQLite 的原因），用户决定后用 `Command(resume=...)`
恢复。本模块负责中间那段：把中断请求变成用户看得懂的问题，等人回答，超时兜底。

超时按拒绝处理（fail-closed）。超时若默认放行，攻击者只要让请求超时即可绕过审批。
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

Decision = Literal["approve", "reject", "edit"]


@dataclass
class ApprovalRequest:
    """一次待审批请求。"""

    approval_id: str
    session_id: str
    tool_name: str
    args: dict[str, Any]
    reason: str = ""
    created_at: float = field(default_factory=time.time)
    timeout_sec: int = 120
    status: str = "pending"                # pending / approved / rejected / timeout
    decided_by: str = ""
    decided_at: float = 0.0
    _future: asyncio.Future | None = None

    def to_payload(self) -> dict:
        return {
            "approval_id": self.approval_id,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "args": _redact(self.args),
            "reason": self.reason,
            "created_at": self.created_at,
            "timeout_sec": self.timeout_sec,
            "status": self.status,
        }

    @property
    def remaining(self) -> float:
        return max(0.0, self.timeout_sec - (time.time() - self.created_at))


_SENSITIVE_KEYS = ("password", "token", "secret", "key", "authorization")


def _redact(args: dict) -> dict:
    """审批界面不该回显密钥类参数。"""
    out = {}
    for k, v in (args or {}).items():
        if any(s in str(k).lower() for s in _SENSITIVE_KEYS):
            out[k] = "***"
        elif isinstance(v, str) and len(v) > 800:
            out[k] = v[:800] + f"...(截断，共 {len(v)} 字符)"
        else:
            out[k] = v
    return out


class ApprovalCenter:
    """待审批请求的注册表 + 等待/决议协调。"""

    def __init__(self) -> None:
        self._pending: dict[str, ApprovalRequest] = {}
        self._history: list[dict] = []
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    async def request(self, *, session_id: str, tool_name: str, args: dict,
                      reason: str = "", timeout_sec: int | None = None) -> ApprovalRequest:
        """登记一次审批请求并等待用户决议（超时按拒绝处理）。

        调用方（`main_agent` 的中断处理）await 这个方法，Agent 执行在此挂起。
        """
        from concurrency.limiter import _env_int  # 复用环境变量读取
        timeout = timeout_sec or _env_int("APPROVAL_TIMEOUT_SEC", 120)

        req = ApprovalRequest(
            approval_id=uuid.uuid4().hex[:12],
            session_id=session_id,
            tool_name=tool_name,
            args=args,
            reason=reason,
            timeout_sec=timeout,
            _future=asyncio.get_running_loop().create_future(),
        )
        async with self._lock:
            self._pending[req.approval_id] = req

        self._publish(req, "approval_required", f"操作 {tool_name} 需要你确认")
        self._audit(req, "pending")

        try:
            await asyncio.wait_for(req._future, timeout=timeout)
        except asyncio.TimeoutError:
            req.status = "timeout"
            # 超时 = 拒绝（fail-closed）
            self._publish(req, "approval_resolved",
                          f"审批超时（{timeout}s），按拒绝处理", level="warn")
        except asyncio.CancelledError:
            req.status = "rejected"
            req.decided_by = "system:cancelled"
            self._publish(req, "approval_resolved", "会话被取消，按拒绝处理", level="warn")
            raise
        finally:
            async with self._lock:
                self._pending.pop(req.approval_id, None)
            self._history.append({**req.to_payload(), "decided_at": req.decided_at,
                                  "decided_by": req.decided_by})
            self._audit(req, req.status)
        return req

    async def resolve(self, approval_id: str, decision: Decision, *, by: str = "user",
                      edited_args: dict | None = None) -> bool:
        """用户做出决议。返回是否成功（请求可能已超时/不存在）。"""
        async with self._lock:
            req = self._pending.get(approval_id)
        if req is None or req._future is None or req._future.done():
            return False

        req.status = {"approve": "approved", "reject": "rejected", "edit": "approved"}[decision]
        req.decided_by = by
        req.decided_at = time.time()
        if decision == "edit" and edited_args:
            req.args = {**req.args, **edited_args}
        req._future.set_result(req)
        self._publish(req, "approval_resolved",
                      f"用户{'同意' if decision != 'reject' else '拒绝'}了 {req.tool_name}")
        return True

    # ------------------------------------------------------------------
    def list_pending(self, session_id: str = "") -> list[dict]:
        return [r.to_payload() for r in self._pending.values()
                if not session_id or r.session_id == session_id]

    def history(self, limit: int = 100) -> list[dict]:
        return self._history[-limit:]

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    # ------------------------------------------------------------------
    def _publish(self, req: ApprovalRequest, event_type: str, message: str,
                 level: str = "info") -> None:
        try:
            from observability.bus import event_bus
            event_bus.publish(event_type, message, session_id=req.session_id,
                              data={**req.to_payload(), "status": req.status}, level=level)
        except Exception:
            pass

    def _audit(self, req: ApprovalRequest, decision: str) -> None:
        try:
            from security.audit import audit
            audit.record(action="approval", decision=decision, session_id=req.session_id,
                         target=req.tool_name, reason=req.reason,
                         approval_id=req.approval_id, decided_by=req.decided_by,
                         args_digest=str(_redact(req.args))[:300])
        except Exception:
            pass


#: 全局单例
approval_center = ApprovalCenter()


#: 哪些工具在什么条件下需要人工审批，集中定义
def needs_approval(tool_name: str, args: dict | None = None) -> tuple[bool, str]:
    """判定一个工具调用是否需要人工审批。

    规则偏保守：带写操作或不可逆的动作都要求确认。
    """
    import os
    if os.getenv("APPROVAL_ENABLED", "1").strip().lower() not in ("1", "true", "yes"):
        return False, ""

    write_tools = {"execute_sql_query"}
    if tool_name in write_tools:
        return True, "该操作会直接对数据库执行语句，且结果不可撤销"

    if tool_name == "generate_markdown" and args:
        # TODO: 覆盖已有文件时要求确认，避免冲掉用户已有内容
        return False, ""
    return False, ""
