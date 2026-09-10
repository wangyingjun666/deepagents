"""数据库只读查询工具。

SQL 只允许 SELECT/SHOW/EXPLAIN，执行前过 sql_guard（语句白名单、系统库拒绝、
强制 LIMIT、超时提示），执行走连接池，放行和拒绝都写审计。
审批中断（interrupt_on）放在 Agent 层，工具层不做。工具里拿不到等待用户确认的
交互通道，硬做会阻塞线程；交给 LangGraph 的中断机制，图状态有 checkpoint，
确认后用 `Command(resume=...)` 恢复。
"""
import logging
import time

from dotenv import load_dotenv
from langchain_core.tools import tool

from api.context import get_thread_context
from api.monitor import monitor
from concurrency.db_pool import is_readonly_account_configured, pools
from concurrency.limiter import limiter_for
from concurrency.retry import breaker_for, is_retryable, with_retry
from security.audit import audit
from security.permissions import Capability, requires
from security.sql_guard import SqlPolicyViolation, validate_sql
from tools.markdown_tools import run_async

load_dotenv()
logger = logging.getLogger(__name__)

#: 表名白名单（环境变量配置，逗号分隔；为空表示不额外限制，仍受系统库拒绝保护）
import os  # noqa: E402

ALLOWED_TABLES = [t.strip() for t in os.getenv("SQL_ALLOWED_TABLES", "").split(",") if t.strip()]
MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "200"))
STATEMENT_TIMEOUT_MS = int(os.getenv("SQL_TIMEOUT_MS", "5000"))


def _audit_sql(decision: str, *, sql: str, reason: str = "", extra: dict | None = None) -> None:
    try:
        audit.record(action="sql_execute", decision=decision,
                     session_id=get_thread_context() or "",
                     target=sql[:200], reason=reason, **(extra or {}))
    except Exception:
        logger.exception("审计写入失败（已忽略）")


def _rows_to_csv(cursor) -> str:
    """把结果集渲染成 CSV（表头 + 逗号分隔，模型对这种格式最熟悉）。"""
    description = cursor.description
    if not description:
        return ""
    header = ",".join(str(d[0]) for d in description)
    rows = [",".join(map(str, row)) for row in cursor.fetchall()]
    return "\n".join([header, *rows])


def _query(sql: str) -> str:
    """执行一条**已经过策略校验**的 SQL（连接池 + 超时保护）。

    这里只管执行，策略判定都在调用方完成，单测策略时不需要真连数据库。
    """
    conn = pools.borrow(readonly=True, session_id=get_thread_context() or "")
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            return _rows_to_csv(cursor)
    finally:
        conn.close()                     # 归还到池，不是真正关闭


@tool
@requires(Capability.DB_QUERY)
def list_sql_tables() -> str:
    """
    查询当前库中所有可用的表！
    作用：为了模型识别有哪些可用的表！方便进行后续的自定义sql查询
    :return: 有表： 可用的表有：表1,表2,表3....  没有表: 没有可用的表   出现异常：查询出现异常：异常信息
    """
    monitor.report_tool(tool_name="数据库表名查询工具：list_sql_tables", args={})

    async def _run() -> str:
        return await __import__("asyncio").to_thread(_query, "SHOW TABLES")

    try:
        raw = run_async(_run())
    except Exception as exc:
        _audit_sql("error", sql="SHOW TABLES", reason=str(exc)[:200])
        return f"查询出现异常：{exc}"

    names = [line for line in raw.splitlines()[1:] if line.strip()]
    _audit_sql("allow", sql="SHOW TABLES", extra={"table_count": len(names)})
    if not names:
        return "没有可用的表"
    return f"可用的表有：{', '.join(names)}"


@tool
@requires(Capability.DB_QUERY)
def get_table_data(table_name) -> str:
    """
    查询指定表名的数据！当前工具调用之前，必须先调用list_sql_tables完成表名的校验！
    此工具的作用：1.可以完成单表数据的查询 2. 可以为多表查询提供表结果信息（列名&数据格式）
    :param table_name: 表名
    :return: csv格式的数据（模拟表格数据格式），至多100条
    """
    monitor.report_tool(tool_name="数据库表数据查询工具：get_table_data", args={"table_name": table_name})

    # 表名不可信：只允许字母数字下划线，杜绝 "users; drop ..." 这类拼接注入
    raw_name = str(table_name or "").strip()
    if not raw_name.replace("_", "").replace("-", "").isalnum():
        _audit_sql("deny", sql=f"table:{raw_name}", reason="illegal_table_name")
        return "【已拒绝】表名只能包含字母、数字、下划线，请先用 list_sql_tables 获取准确表名。"

    sql = f"SELECT * FROM `{raw_name}` LIMIT 100"
    try:
        decision = validate_sql(sql, allowed_tables=ALLOWED_TABLES or None,
                                max_rows=min(100, MAX_ROWS), timeout_ms=STATEMENT_TIMEOUT_MS)
    except SqlPolicyViolation as exc:
        _audit_sql("deny", sql=sql, reason=exc.reason)
        monitor.report_denied("sql", f"SQL 被策略拒绝：{exc.reason}")
        return f"【已拒绝】{exc}"

    async def _run() -> str:
        return await __import__("asyncio").to_thread(_query, decision.sql)

    try:
        csv = run_async(_run())
    except Exception as exc:
        _audit_sql("error", sql=decision.sql, reason=str(exc)[:200])
        return f"查询出现异常：{exc}"

    _audit_sql("allow", sql=decision.sql, extra=decision.to_dict())
    if not csv.strip():
        return f"数据表：{raw_name}为空没有数据！"
    return csv


@tool
@requires(Capability.DB_QUERY)
def execute_sql_query(query) -> str:
    """
    执行自定义查询sql语句！切记：执行之前，需要通过执行 list_sql_tables明确表名！
    执行get_table_data明确表结构和数据格式！
    仅支持只读查询（SELECT/SHOW/EXPLAIN），会自动附加行数上限与执行超时。
    :param query: 要执行的自定义sql语句
    :return: csv格式的数据（模拟表格数据格式），至多若干条（默认 200）
    """
    monitor.report_tool(tool_name="数据库表数据查询工具：execute_sql_query", args={"query": query})

    # 闸门 1：SQL 策略校验与改写
    try:
        decision = validate_sql(query, allowed_tables=ALLOWED_TABLES or None,
                                max_rows=MAX_ROWS, timeout_ms=STATEMENT_TIMEOUT_MS)
    except SqlPolicyViolation as exc:
        _audit_sql("deny", sql=str(query)[:500], reason=exc.reason)
        monitor.report_denied("sql", f"SQL 被策略拒绝：{exc.reason}",
                              {"sql": str(query)[:300], "reason": exc.reason})
        return (f"【已拒绝】{exc}\n"
                f"本工具只允许只读查询（SELECT / SHOW / EXPLAIN / DESCRIBE），"
                f"且不允许访问系统库或写文件类构造。")

    # 闸门 2：只读账号若未配置则显式告警
    if not is_readonly_account_configured():
        logger.warning("未配置 MYSQL_RO_USER，数据库侧只读约束缺失，仅依赖应用层 SQL 白名单")

    # 闸门 3：限流 + 重试退避 + 熔断
    limiter = limiter_for("mysql")
    breaker = breaker_for("mysql")

    async def _guarded() -> str:
        async with limiter.acquire(session_id=get_thread_context() or "") as wait_ms:
            if wait_ms > 200:
                monitor._emit("queued", f"数据库查询排队 {wait_ms}ms", {"wait_ms": wait_ms})
            return await __import__("asyncio").to_thread(_query, decision.sql)

    try:
        csv = run_async(with_retry(_guarded, session_id=get_thread_context() or "",
                                   operation="db:execute_sql_query", circuit=breaker))
    except Exception as exc:
        _audit_sql("error", sql=decision.sql, reason=f"{type(exc).__name__}: {exc}"[:200])
        return f"查询出现异常：{exc}"

    # 闸门 4：审计（放行也记，含改写后的 SQL）
    _audit_sql("allow", sql=decision.sql, extra=decision.to_dict())
    monitor._emit("sql_executed", f"SQL 执行完成（{decision.statement_type}）",
                  {**decision.to_dict(), "sql": decision.sql[:400]})

    if not csv.strip():
        return f"执行自定义SQL语句查询没有结果，sql为：{query}！"
    return csv


@requires(Capability.DB_QUERY)
def db_health() -> dict:
    """数据库侧自检：连接池状态、是否配置只读账号。"""
    return {
        **pools.describe(),
        "readonly_account_configured": is_readonly_account_configured(),
        "allowed_tables": ALLOWED_TABLES or "未配置（仅受系统库拒绝保护）",
        "max_rows": MAX_ROWS,
        "statement_timeout_ms": STATEMENT_TIMEOUT_MS,
    }
