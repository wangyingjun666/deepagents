"""
安全治理层：路径守卫 + SQL 策略 + 权限路由 + 哈希链审计。

一次工具调用的判定顺序：permissions.verify → path_guard.guard_path →
sandbox 执行 → audit.record。任何一层拒绝都返回可读错误文本给模型（不抛异常
打断 Agent 循环），并写审计。
"""
from security.audit import AuditLog, audit
from security.path_guard import (
    PathMode,
    PathSecurityError,
    guard_path,
    guard_within_sandbox,
    sanitize_filename,
)
from security.permissions import (
    Capability,
    PermissionDenied,
    Role,
    ROLE_CAPABILITIES,
    SessionPrincipal,
    capabilities_for,
    current_principal,
    describe_routing,
    requires,
    reset_principal,
    security_mode,
    set_principal,
    verify,
)
from security.sql_guard import (
    SqlDecision,
    SqlPolicyViolation,
    validate_sql,
)

__all__ = [
    "AuditLog", "audit",
    "PathMode", "PathSecurityError", "guard_path", "guard_within_sandbox", "sanitize_filename",
    "Capability", "PermissionDenied", "Role", "ROLE_CAPABILITIES", "SessionPrincipal",
    "capabilities_for", "current_principal", "describe_routing", "requires",
    "reset_principal", "security_mode", "set_principal", "verify",
    "SqlDecision", "SqlPolicyViolation", "validate_sql",
]
