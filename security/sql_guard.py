"""
SQL 策略守卫：把 `execute_sql_query` 从"模型给什么都执行"变成受策略约束。

裸的 `execute_sql_query(query)` 把模型字符串直接送进 `cursor.execute()`，没有语句
类型限制（写出 `UPDATE`/`DELETE` 就会真执行）、没有强制 LIMIT、没有表名白名单
（`information_schema`、`mysql.user` 都能查）、没有执行超时，且连接是
`autocommit=True`，写操作立即生效无回滚余地。安全性完全寄托在 prompt 约束和
数据库账号权限上，本模块把这几层补上。

四道闸门
--------
1. 语句类型白名单：只放行 SELECT / WITH / SHOW / EXPLAIN / DESCRIBE。
2. 危险构造黑名单：类型合法也拒绝 `INTO OUTFILE`、`LOAD_FILE`、
   `SLEEP`/`BENCHMARK`（DoS）、`FOR UPDATE`（加锁）、系统库元数据探测等。
3. 强制行数上限：用子查询包一层钉死上限，模型自己写的 LIMIT 也一并覆盖
   （外层 LIMIT 与内层取较小值）。
4. 强制执行超时：注入 MySQL 8 的 `MAX_EXECUTION_TIME` 优化器提示，让数据库侧
   自己掐断慢查询，比客户端超时可靠，因为它能真正中断执行。

所有被拒绝的语句都进审计，含原始 SQL，便于复盘模型的乱写行为。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import sqlparse
from sqlparse import sql as S
from sqlparse import tokens as T

MAX_QUERY_LEN = 4000
MAX_ROWS_HARD_CAP = 1000

#: 允许的语句类型（sqlparse 的 statement.get_type() 返回值）
ALLOWED_STATEMENT_TYPES = frozenset({"SELECT", "SHOW", "EXPLAIN", "DESCRIBE", "DESC"})

#: 危险构造黑名单：正则 → 拒绝原因
_FORBIDDEN_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bINTO\s+(OUTFILE|DUMPFILE)\b", re.I), "写文件到数据库服务器磁盘"),
    (re.compile(r"\bLOAD_FILE\s*\(", re.I), "读取数据库服务器本地文件"),
    (re.compile(r"\bLOAD\s+DATA\b", re.I), "批量导入数据"),
    (re.compile(r"\b(SLEEP|BENCHMARK)\s*\(", re.I), "可被用于拖慢/打爆数据库的延时函数"),
    (re.compile(r"\bGET_LOCK\s*\(", re.I), "获取命名锁，可能造成阻塞"),
    (re.compile(r"\bFOR\s+UPDATE\b", re.I), "行锁，长事务会阻塞其他会话"),
    (re.compile(r"\bLOCK\s+IN\s+SHARE\s+MODE\b", re.I), "共享锁"),
    (re.compile(r"\bINTO\s+@", re.I), "会话变量赋值"),
    (re.compile(r"\bSET\s+@", re.I), "会话变量赋值"),
    (re.compile(r"\bmysql\s*\.\s*(user|db|global_priv)\b", re.I), "读取 MySQL 账号表"),
    (re.compile(r"\bperformance_schema\b", re.I), "探测服务器运行时信息"),
)

#: 一律拒绝的库/表前缀（元数据探测面）
_DENIED_SCHEMA_PREFIXES = ("information_schema.", "mysql.", "performance_schema.", "sys.")


class SqlPolicyViolation(PermissionError):
    """SQL 未通过策略校验。"""

    def __init__(self, message: str, *, reason: str, sql: str = ""):
        super().__init__(message)
        self.reason = reason
        self.sql = sql


@dataclass(slots=True)
class SqlDecision:
    """校验结果：改写后的 SQL 与决策元信息（一起写审计）。"""

    sql: str
    statement_type: str
    tables: list[str] = field(default_factory=list)
    row_limit: int = 0
    timeout_ms: int = 0
    injected_limit: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "statement_type": self.statement_type,
            "tables": self.tables,
            "row_limit": self.row_limit,
            "timeout_ms": self.timeout_ms,
            "injected_limit": self.injected_limit,
        }


def _strip_comments(sql: str) -> str:
    """去掉注释后再做黑名单匹配。

    否则 `SELECT 1 /* */ ; DELETE FROM t` 这类写法能靠注释逃过一部分检查。
    sqlparse 的 strip_comments 一并处理 `--`、`#`、`/* */` 三种 MySQL 注释。
    """
    return sqlparse.format(sql, strip_comments=True).strip()


def _extract_tables(parsed: S.Statement) -> list[str]:
    """从语句里抽取被引用的表名（FROM / JOIN 之后出现的标识符）。"""
    tables: list[str] = []
    expect = False
    for token in parsed.flatten():
        if token.ttype in (T.Keyword, T.Keyword.DML, T.Keyword.DDL):
            if token.normalized.upper() in ("FROM", "JOIN", "INTO", "UPDATE"):
                expect = True
                continue
        if expect:
            if token.ttype in (T.Whitespace, T.Comment, T.Newline, T.Punctuation):
                if token.ttype is not T.Whitespace and token.ttype is not T.Newline:
                    pass
                continue
            if token.ttype in (T.Name, T.String.Symbol, T.Name.Placeholder, T.Keyword):
                name = token.value.strip("`\"' ")
                if name and name.upper() not in ("SELECT", "WHERE", "ON", "AS", "USING"):
                    tables.append(name)
                expect = False
    # 去重保序
    seen: dict[str, None] = {}
    for t in tables:
        seen.setdefault(t, None)
    return list(seen)


def _has_top_level_limit(parsed: S.Statement) -> bool:
    return any(
        tok.ttype is T.Keyword and tok.normalized.upper() == "LIMIT"
        for tok in parsed.flatten()
    )


def validate_sql(
    query: str,
    *,
    allowed_tables: Sequence[str] | None = None,
    max_rows: int = 200,
    timeout_ms: int = 5000,
    allow_write: bool = False,
) -> SqlDecision:
    """校验并改写一条 SQL。不通过则抛 `SqlPolicyViolation`。

    Args:
        query: 模型给出的原始 SQL。
        allowed_tables: 表白名单；None 表示不限制（仍受元数据库前缀限制）。
        max_rows: 强制行数上限。
        timeout_ms: 强制执行超时（毫秒），通过优化器提示注入。
        allow_write: 是否放行写语句，只有持有 `DB_QUERY_WRITE` 能力的会话才应为 True。
    """
    if not query or not query.strip():
        raise SqlPolicyViolation("SQL 为空", reason="empty_sql", sql=query)
    if len(query) > MAX_QUERY_LEN:
        raise SqlPolicyViolation(
            f"SQL 长度 {len(query)} 超过上限 {MAX_QUERY_LEN}", reason="too_long", sql=query[:200])

    cleaned = _strip_comments(query)

    # ---- 闸门 1：必须单条语句（禁止 `SELECT 1; DROP TABLE t` 这种夹带）----
    statements = [s for s in sqlparse.parse(cleaned) if str(s).strip()]
    if len(statements) != 1:
        raise SqlPolicyViolation(
            f"只允许单条 SQL 语句，解析到 {len(statements)} 条", reason="multiple_statements", sql=cleaned)

    parsed = statements[0]
    stmt_type = (parsed.get_type() or "UNKNOWN").upper()
    # 递归 CTE 会被 sqlparse 标成 UNKNOWN，按首个关键字识别为 WITH
    if stmt_type == "UNKNOWN" and cleaned.lstrip().upper().startswith("WITH"):
        stmt_type = "SELECT"

    if stmt_type not in ALLOWED_STATEMENT_TYPES:
        if not (allow_write and stmt_type in {"INSERT", "UPDATE", "DELETE"}):
            raise SqlPolicyViolation(
                f"语句类型 {stmt_type} 不被允许（只读会话仅放行 "
                f"{'/'.join(sorted(ALLOWED_STATEMENT_TYPES))}）",
                reason=f"statement_type_denied:{stmt_type}", sql=cleaned)

    # ---- 闸门 2：危险构造黑名单 ----
    for pattern, why in _FORBIDDEN_PATTERNS:
        if pattern.search(cleaned):
            raise SqlPolicyViolation(f"SQL 包含被禁止的构造（{why}）",
                                     reason=f"forbidden_construct:{why}", sql=cleaned)

    # ---- 闸门 3：元数据库探测 ----
    lowered = cleaned.lower()
    for prefix in _DENIED_SCHEMA_PREFIXES:
        if prefix in lowered:
            raise SqlPolicyViolation(f"不允许访问系统库 {prefix.rstrip('.')}",
                                     reason="system_schema_denied", sql=cleaned)

    tables = _extract_tables(parsed)
    if allowed_tables:
        allow_set = {t.lower() for t in allowed_tables}
        illegal = [t for t in tables if t.lower() not in allow_set]
        if illegal:
            raise SqlPolicyViolation(
                f"访问了白名单之外的表：{', '.join(illegal)}",
                reason="table_not_allowed", sql=cleaned)

    # ---- 闸门 4：强制行数上限 ----
    max_rows = max(1, min(int(max_rows), MAX_ROWS_HARD_CAP))
    injected = False
    final_sql = cleaned
    if stmt_type in ("SELECT", "WITH") and not allow_write:
        if _has_top_level_limit(parsed):
            # 已经写了 LIMIT：不能简单追加，也不能信任它小于上限。
            # 用子查询包一层，外层 LIMIT 与内层取较小值。
            final_sql = f"SELECT * FROM (\n{cleaned.rstrip(';')}\n) AS _guard_limit\nLIMIT {max_rows}"
            injected = True
        else:
            final_sql = f"SELECT * FROM (\n{cleaned.rstrip(';')}\n) AS _guard_limit\nLIMIT {max_rows}"
            injected = True

    # ---- 闸门 5：强制执行超时（MySQL 8 优化器提示，数据库侧自己中断）----
    timeout_ms = max(100, int(timeout_ms))
    final_sql = f"/*+ MAX_EXECUTION_TIME({timeout_ms}) */ {final_sql}"

    return SqlDecision(
        sql=final_sql,
        statement_type=stmt_type,
        tables=tables,
        row_limit=max_rows,
        timeout_ms=timeout_ms,
        injected_limit=injected,
    )


def list_sql_tables_result(table_names: Iterable[str]) -> str:
    """把表名列表渲染成模型可读文本。"""
    names = [str(t) for t in table_names]
    return f"可用的表有：{', '.join(names)}" if names else "没有可用的表"


def audit_payload(query: str, decision: SqlDecision | None, *, session_id: str) -> dict:
    """构造审计记录体。"""
    return {
        "action": "sql_execute",
        "session_id": session_id,
        "sql_preview": (decision.sql if decision else query)[:500],
        "sql_hash": __import__("hashlib").sha256(
            (decision.sql if decision else query).encode("utf-8")).hexdigest()[:16],
        **(decision.to_dict() if decision else {"statement_type": "REJECTED"}),
    }
