"""
权限路由：把"这个动作能不能做"从 prompt 约束变成运行时强制。

静态挂工具只解决了"这个子 Agent 有哪些工具"，解决不了按用户鉴权（所有会话共用
同一批工具）和运行时判定（工具一旦挂上在任何上下文都能调用）。这里补上：
Role 经路由表得到 Capability 集合，绑定到 SessionPrincipal，工具调用时过 `verify()`。

设计要点
--------
* 默认拒绝：principal 未建立或能力缺失时一律拒绝，不做宽松兜底。
* 能力粒度按"动作 × 数据域"切分，同一工具在不同能力下行为不同。例如
  `execute_sql_query` 需要 `DB_QUERY`，只有 `DB_QUERY_WRITE` 才放行非 SELECT。
* 静态路由 + 动态校验：子 Agent 只挂自己的工具，每次调用仍过一次 `verify()`。
* 环境变量 `SECURITY_MODE`（strict / standard / off）允许本地开发放宽，
  默认 strict，降级会打审计。
"""
from __future__ import annotations

import contextvars
import functools
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, FrozenSet, Iterable

logger = logging.getLogger(__name__)


class Capability(str, Enum):
    """系统里的全部能力，按"数据域 × 动作"切分。"""

    # 文件系统：会话工作区
    FS_READ_SESSION = "fs:read:session"
    FS_WRITE_SESSION = "fs:write:session"
    # 文件系统：用户上传件（只读）
    FS_READ_UPLOAD = "fs:read:upload"
    # 数据库
    DB_QUERY = "db:query"                    # 只读查询
    DB_QUERY_WRITE = "db:query:write"        # 允许写语句（默认不授予任何角色）
    # 外部数据源
    NET_SEARCH = "net:search"                # 联网检索
    KB_QUERY = "kb:query"                    # 内部知识库检索
    # 交付物
    DOC_RENDER = "doc:render"                # 文档生成 / PDF 渲染
    # 危险能力
    CODE_EXEC = "code:exec"                  # 沙箱内执行命令（默认不授予）
    SANDBOX_ADMIN = "sandbox:admin"          # 管理沙箱生命周期


class Role(str, Enum):
    """会话角色，决定能力路由结果。"""

    GUEST = "guest"      # 匿名/未登录：只能问，不能落文档
    USER = "user"        # 普通用户：检索 + 生成文档
    ANALYST = "analyst"  # 分析人员：额外允许查数据库明细
    ADMIN = "admin"      # 管理员：全部能力（含代码执行）


#: 角色 → 能力的路由表（唯一事实来源，审计按这张表复述）
ROLE_CAPABILITIES: dict[Role, FrozenSet[Capability]] = {
    Role.GUEST: frozenset({
        Capability.NET_SEARCH,
        Capability.KB_QUERY,
    }),
    Role.USER: frozenset({
        Capability.FS_READ_SESSION,
        Capability.FS_WRITE_SESSION,
        Capability.FS_READ_UPLOAD,
        Capability.NET_SEARCH,
        Capability.KB_QUERY,
        Capability.DB_QUERY,
        Capability.DOC_RENDER,
    }),
    Role.ANALYST: frozenset({
        Capability.FS_READ_SESSION,
        Capability.FS_WRITE_SESSION,
        Capability.FS_READ_UPLOAD,
        Capability.NET_SEARCH,
        Capability.KB_QUERY,
        Capability.DB_QUERY,
        Capability.DOC_RENDER,
    }),
    Role.ADMIN: frozenset(Capability),
}


class PermissionDenied(PermissionError):
    """能力校验未通过。工具层会转成可读错误文本回灌给模型。"""

    def __init__(self, capability: Capability, *, principal: "SessionPrincipal | None"):
        who = principal.user_id if principal else "<anonymous>"
        super().__init__(f"当前会话（{who}）不具备能力 {capability.value}，操作被拒绝")
        self.capability = capability
        self.principal = principal


@dataclass(slots=True)
class SessionPrincipal:
    """一次会话的安全主体：谁能做什么。"""

    session_id: str
    user_id: str = "anonymous"
    role: Role = Role.USER
    capabilities: FrozenSet[Capability] = field(default_factory=frozenset)
    scopes: tuple[str, ...] = ()          # 附加数据域约束（预留：多租户行级过滤）

    @classmethod
    def build(cls, session_id: str, *, user_id: str = "anonymous",
              role: Role | str | None = None) -> "SessionPrincipal":
        role = Role(role or os.getenv("DEFAULT_ROLE", Role.USER.value))
        return cls(
            session_id=session_id,
            user_id=user_id,
            role=role,
            capabilities=ROLE_CAPABILITIES.get(role, frozenset()),
        )

    def has(self, capability: Capability) -> bool:
        if security_mode() == "off":
            return True
        return capability in self.capabilities

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "role": self.role.value,
            "capabilities": sorted(c.value for c in self.capabilities),
        }


# --------------------------------------------------------------------------
# 上下文绑定：能力跟协程走，不跟进程走（ContextVar 语义）。并发会话各持一份。
# --------------------------------------------------------------------------
_principal_ctx: contextvars.ContextVar[SessionPrincipal | None] = contextvars.ContextVar(
    "security_principal", default=None)


def set_principal(principal: SessionPrincipal):
    return _principal_ctx.set(principal)


def reset_principal(token) -> None:
    _principal_ctx.reset(token)


def current_principal() -> SessionPrincipal | None:
    return _principal_ctx.get()


def security_mode() -> str:
    return os.getenv("SECURITY_MODE", "strict").strip().lower()


def verify(capability: Capability, *, reason: str = "") -> SessionPrincipal:
    """运行时能力校验，所有受控动作都必须先过这里。"""
    principal = current_principal()
    if security_mode() == "strict" and principal is None:
        raise PermissionDenied(capability, principal=None)
    if principal is None:                      # standard 模式下允许匿名
        principal = SessionPrincipal.build("unknown")
    if not principal.has(capability):
        _audit_denied(capability, principal, reason)
        raise PermissionDenied(capability, principal=principal)
    return principal


def _audit_denied(capability: Capability, principal: SessionPrincipal, reason: str) -> None:
    logger.warning("权限拒绝 session=%s user=%s role=%s capability=%s reason=%s",
                   principal.session_id, principal.user_id, principal.role.value,
                   capability.value, reason or "-")
    try:
        from security.audit import audit
        audit.record(
            action=f"capability:{capability.value}",
            decision="deny",
            session_id=principal.session_id,
            user_id=principal.user_id,
            role=principal.role.value,
            reason=reason or "capability_not_granted",
        )
    except Exception:                          # 审计失败不能影响判定
        logger.exception("审计写入失败（已忽略）")


def requires(capability: Capability):
    """工具装饰器：把能力校验挂在函数入口。

    用法::

        @tool
        @requires(Capability.DB_QUERY)
        def execute_sql_query(query: str) -> str: ...

    拒绝时不抛异常（会打断 Agent 循环），改为返回一句可读文本，让模型换个路径。
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                verify(capability, reason=f"tool:{func.__name__}")
            except PermissionDenied as exc:
                return f"【权限拒绝】{exc}"
            return func(*args, **kwargs)

        wrapper.__required_capability__ = capability  # 供自省/测试使用
        return wrapper

    return decorator


def capabilities_for(role: Role | str) -> FrozenSet[Capability]:
    return ROLE_CAPABILITIES.get(Role(role), frozenset())


def describe_routing() -> dict[str, list[str]]:
    """把路由表导出成可展示结构，供前端权限面板和审计使用。"""
    return {role.value: sorted(c.value for c in caps) for role, caps in ROLE_CAPABILITIES.items()}
