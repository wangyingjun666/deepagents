# -*- coding: utf-8 -*-
"""
安全与沙箱的单元测试（不依赖 Docker、不依赖数据库、不依赖网络）。

覆盖范围
--------
1. 路径守卫：16 种非法输入的**具体拒绝原因码**
2. 权限路由：角色 → 能力映射、运行时校验、拒绝审计
3. SQL 策略：9 类拒绝 + 白名单 + LIMIT 改写 + 超时注入
4. 哈希链审计：链完整性校验

运行：
    python tests/test_security_and_sandbox.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from security import (  # noqa: E402
    Capability,
    PathMode,
    PathSecurityError,
    Role,
    SessionPrincipal,
    SqlPolicyViolation,
    audit,
    guard_path,
    reset_principal,
    set_principal,
    validate_sql,
    verify,
)
from security.permissions import PermissionDenied  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(("  [OK]   " if cond else "  [FAIL] ") + name + (f"  -> {detail}" if detail else ""))


# --------------------------------------------------------------------------
def test_path_guard() -> None:
    print("\n=== 1. 路径守卫（拒绝型） ===")
    root = Path(tempfile.mkdtemp(prefix="pg_"))
    ws = root / "session_x"
    ws.mkdir(parents=True, exist_ok=True)
    (root / "secret.txt").write_text("secret", encoding="utf-8")

    deny_cases = [
        ("../secret.txt", "parent_traversal"),
        ("../../Windows/win.ini", "parent_traversal"),
        ("a/../../b.md", "parent_traversal"),
        ("D:/secrets/x.md", "absolute_path_denied"),
        ("/etc/passwd", "absolute_path_denied"),
        ("//server/share/x.md", "absolute_path_denied"),
        ("C:foo.md", "absolute_path_denied"),
        ("a\x00b.md", "nul_byte"),
        ("CON.md", "reserved_device_name"),
        ("report.md.", "trailing_dot_space"),
        ("evil.exe", "suffix_not_allowed"),
        ("x" * 200 + ".md", "name_too_long"),
        ("", "empty_path"),
    ]
    for candidate, expect in deny_cases:
        try:
            guard_path(candidate, ws, mode=PathMode.WRITE)
            check(f"拒绝 {candidate[:30]!r}", False, "本应拒绝但通过了")
        except PathSecurityError as exc:
            check(f"拒绝 {candidate[:30]!r}", exc.reason == expect,
                  f"期望 {expect}，实际 {exc.reason}")

    ok = guard_path("sub/report.md", ws, mode=PathMode.WRITE)
    check("放行合法相对路径", str(ok).endswith("report.md"), str(ok))

    ok = guard_path("/workspace/report.md", ws, mode=PathMode.WRITE)
    check("虚拟前缀被剥离后放行", str(ok).endswith("report.md"), str(ok))

    # 前缀伪造：session_x_evil 不应被认作 session_x 的子目录
    evil = root / "session_x_evil"
    evil.mkdir(exist_ok=True)
    try:
        guard_path(str(evil / "x.md"), ws, mode=PathMode.WRITE)
        check("前缀伪造被拒", False, "本应拒绝")
    except PathSecurityError as exc:
        check("前缀伪造被拒", True, exc.reason)

    # 大小写归一（Windows 上 C:\Temp 与 c:\temp 是同一目录）
    try:
        p = guard_path("Report.MD", ws, mode=PathMode.WRITE)
        check("大小写后缀白名单按小写判定", str(p).endswith("Report.MD"), str(p))
    except PathSecurityError as exc:
        check("大小写后缀白名单按小写判定", False, exc.reason)


# --------------------------------------------------------------------------
def test_permissions() -> None:
    print("\n=== 2. 权限路由 ===")
    guest = SessionPrincipal.build("s1", user_id="访客", role=Role.GUEST)
    token = set_principal(guest)
    try:
        try:
            verify(Capability.DB_QUERY)
            check("guest 查库被拒", False, "本应拒绝")
        except PermissionDenied:
            check("guest 查库被拒", True)
        verify(Capability.NET_SEARCH)
        check("guest 联网检索放行", True)
        try:
            verify(Capability.FS_WRITE_SESSION)
            check("guest 写文件被拒", False, "本应拒绝")
        except PermissionDenied:
            check("guest 写文件被拒", True)
    finally:
        reset_principal(token)

    admin = SessionPrincipal.build("s2", user_id="admin", role=Role.ADMIN)
    token = set_principal(admin)
    try:
        verify(Capability.CODE_EXEC)
        check("admin 代码执行放行", True)
    except PermissionDenied:
        check("admin 代码执行放行", False)
    finally:
        reset_principal(token)

    from security import describe_routing
    check("角色路由表完整（4 个角色）", len(describe_routing()) == 4,
          str(list(describe_routing())))


# --------------------------------------------------------------------------
def test_sql_guard() -> None:
    print("\n=== 3. SQL 策略守卫 ===")
    d = validate_sql("select id, name from products where id = 1")
    check("合法 SELECT 放行并包裹 LIMIT",
          d.statement_type == "SELECT" and "LIMIT 200" in d.sql, d.sql[:70])
    check("超时提示已注入", "MAX_EXECUTION_TIME(5000)" in d.sql)

    d = validate_sql("select * from t limit 9999")
    check("已有 LIMIT 被外层收紧", "LIMIT 200" in d.sql and "_guard_limit" in d.sql)

    deny_cases = [
        ("delete from products", "statement_type_denied"),
        ("drop table products", "statement_type_denied"),
        ("update products set price=1", "statement_type_denied"),
        ("insert into t values (1)", "statement_type_denied"),
        ("select 1; delete from products", "multiple_statements"),
        ("select * from t into outfile '/tmp/x'", "forbidden_construct"),
        ("select load_file('/etc/passwd')", "forbidden_construct"),
        ("select sleep(10)", "forbidden_construct"),
        ("select * from t for update", "forbidden_construct"),
        ("select * from mysql.user", "forbidden_construct"),
        ("select * from information_schema.tables", "system_schema_denied"),
        ("", "empty_sql"),
    ]
    for sql, expect in deny_cases:
        try:
            validate_sql(sql)
            check(f"拒绝 {sql[:34]!r}", False, "本应拒绝")
        except SqlPolicyViolation as exc:
            check(f"拒绝 {sql[:34]!r}", exc.reason.startswith(expect), f"reason={exc.reason}")

    d = validate_sql("select * from products", allowed_tables=["products"])
    check("表白名单放行", "products" in d.tables, str(d.tables))
    try:
        validate_sql("select * from orders", allowed_tables=["products"])
        check("白名单外表被拦", False, "本应拒绝")
    except SqlPolicyViolation as exc:
        check("白名单外表被拦", exc.reason == "table_not_allowed")

    # 注释不能用来藏黑名单关键字
    try:
        validate_sql("select 1 /* 无害注释 */ ; delete from t")
        check("注释藏分号被拦", False, "本应拒绝")
    except SqlPolicyViolation as exc:
        check("注释藏分号被拦", exc.reason in ("multiple_statements",), exc.reason)


# --------------------------------------------------------------------------
def test_audit() -> None:
    print("\n=== 4. 哈希链审计 ===")
    ok, detail = audit.verify_chain()
    check("审计链完整", ok, detail)
    check("审计有记录", len(audit.tail(limit=5)) > 0)


def main() -> int:
    test_path_guard()
    test_permissions()
    test_sql_guard()
    test_audit()
    print(f"\n=== 单元测试结果：{len(PASS)} 通过 / {len(FAIL)} 失败 ===")
    for f in FAIL:
        print("   - 失败：", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
