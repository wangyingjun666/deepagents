# -*- coding: utf-8 -*-
"""端到端集成测试：沙箱 + 安全 + 工具 + 可观测 + 并发。

不依赖 Docker（无 Docker 时走进程级后端）/ 数据库 / 外部 API。
运行：python tests/test_integration.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(("  [OK]   " if cond else "  [FAIL] ") + name + (f"  -> {detail}" if detail else ""))


async def main() -> int:
    from api.context import (reset_sandbox_context, reset_security_context,
                             reset_session_context, set_sandbox_context,
                             set_security_context, set_session_context, set_thread_context)
    from concurrency.limiter import session_limiter
    from concurrency.retry import RetryPolicy, breaker_for, with_retry
    from observability.bus import event_bus
    from observability.metrics import metrics
    from sandbox import sandbox_manager
    from security.permissions import Role, SessionPrincipal
    from tools.markdown_tools import generate_markdown
    from tools.pdf_tools import convert_md_to_pdf
    from tools.upload_file_read_tool import read_file_content

    print("\n=== 1. 沙箱后端探测 ===")
    probe = sandbox_manager.probe()
    print(f"  配置={probe.get('configured')}  生效={probe.get('effective')}  "
          f"Docker可用={probe.get('docker_available')}")
    print(f"  说明：{str(probe.get('docker_detail', ''))[:100]}")
    check("探测返回生效后端", "effective" in probe)

    tmp = Path(tempfile.mkdtemp(prefix="integ_"))
    ws = tmp / "output" / "session_t1"
    up = tmp / "updated" / "session_t1"
    up.mkdir(parents=True, exist_ok=True)
    (up / "上传资料.md").write_text("# 上传的资料\n这是用户上传的内容。", encoding="utf-8")

    print("\n=== 2. 工具链路（沙箱内写 / 读 / 渲染） ===")
    async with sandbox_manager.acquire("t1", workspace=ws, uploads=up) as sb:
        meta = sb.meta
        print(f"  后端={meta.get('backend')}  真隔离={meta.get('isolated')}  "
              f"网络强制={meta.get('network_enforced')}  文件强制={meta.get('fs_enforced')}")
        print(f"  自检={meta.get('health')}")

        tokens = [set_session_context(str(ws)), set_thread_context("t1"),
                  set_sandbox_context(sb),
                  set_security_context(SessionPrincipal.build("t1", user_id="张三",
                                                              role=Role.USER))]
        try:
            r = generate_markdown.invoke({
                "content": "# 报告\n\n中文内容\n\n| A | B |\n|---|---|\n| 1 | 2 |",
                "filename": "测试报告"})
            check("generate_markdown 走沙箱成功", "已成功生成" in r, r[:140])

            r = read_file_content.invoke({"filename": "测试报告.md"})
            check("read_file_content 读回内容", "中文内容" in r, r[:140])

            r = read_file_content.invoke({"filename": "../../../etc/passwd"})
            check("读取穿越路径被拒", ("错误" in r or "拒绝" in r), r[:140])

            r = convert_md_to_pdf.invoke({"md_filename": "测试报告.md"})
            check("convert_md_to_pdf 渲染成功", "成功" in r or "已生成" in r, r[:180])

            r = generate_markdown.invoke({"content": "x", "filename": "evil.exe"})
            check("非 md 后缀被强制收敛", "evil.md" in r and ".exe" not in r, r[:140])

            r = generate_markdown.invoke({"content": "x", "filename": "a", "path": "D:/tmp"})
            check("绝对路径被拒", "拒绝" in r, r[:140])

            r = generate_markdown.invoke({"content": "x", "filename": "a", "path": "../../.."})
            check("穿越路径被拒", "拒绝" in r, r[:140])
        finally:
            reset_security_context(tokens[3])
            reset_sandbox_context(tokens[2])
            reset_session_context(tokens[0], tokens[1])

    print("\n=== 3. 权限路由端到端（guest 不该能写文件） ===")
    async with sandbox_manager.acquire("t1", workspace=ws, uploads=up) as sb:
        tokens = [set_session_context(str(ws)), set_thread_context("t1"), set_sandbox_context(sb),
                  set_security_context(SessionPrincipal.build("t1", user_id="访客",
                                                              role=Role.GUEST))]
        try:
            r = generate_markdown.invoke({"content": "x", "filename": "guest"})
            check("guest 写文件被拒", "权限拒绝" in r, r[:140])
        finally:
            reset_security_context(tokens[3])
            reset_sandbox_context(tokens[2])
            reset_session_context(tokens[0], tokens[1])

    await sandbox_manager.release("t1")

    print("\n=== 4. 事件总线与指标 ===")
    stats = event_bus.stats()
    check("事件已发布", stats["published"] > 0, str(stats))
    check("事件已落盘", stats["persisted"] > 0, f"persisted={stats['persisted']}")
    replay = event_bus.replay("t1")
    check("事件可重放", len(replay) > 0, f"{len(replay)} 条")
    ids = [e.get("event_id") for e in replay]
    check("事件 ID 单调递增无空洞", ids == list(range(ids[0], ids[0] + len(ids))), str(ids[:6]))
    check("指标已聚合", len(metrics.snapshot()) > 0, str(list(metrics.snapshot())[:3]))

    print("\n=== 5. 数据库工具的策略拦截（无库也要能拦） ===")
    tokens = [set_thread_context("t1"),
              set_security_context(SessionPrincipal.build("t1", role=Role.USER))]
    try:
        from tools.db_tools import db_health, execute_sql_query
        for sql, label in [("delete from products", "DELETE"),
                           ("select sleep(10)", "SLEEP"),
                           ("select * from information_schema.tables", "系统库"),
                           ("select 1; drop table t", "多语句")]:
            r = execute_sql_query.invoke({"query": sql})
            check(f"{label} 被 SQL 策略拦截", "已拒绝" in r, r[:120])
        h = db_health()
        check("数据库自检可调用", "readonly_account_configured" in h, f"max_rows={h.get('max_rows')}")
    finally:
        reset_security_context(tokens[1])

    print("\n=== 6. 并发限流 / 重试退避 / 熔断 ===")

    async def _acquire():
        async with session_limiter.acquire(session_id="x") as wait_ms:
            return wait_ms

    waits = await asyncio.gather(*[_acquire() for _ in range(3)])
    check("信号量并发放行", len(waits) == 3, str(waits))

    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("模拟超时")
        return "ok"

    got = await with_retry(flaky, policy=RetryPolicy(attempts=5, base_delay=0.01),
                           operation="test:flaky")
    check("重试退避生效", got == "ok" and calls["n"] == 3, f"调用 {calls['n']} 次")

    async def fatal():
        raise ValueError("invalid api key")

    try:
        await with_retry(fatal, policy=RetryPolicy(attempts=3, base_delay=0.01),
                         operation="test:fatal")
        check("不可重试异常直接抛出", False, "本应抛出")
    except ValueError:
        check("不可重试异常直接抛出", True)

    breaker = breaker_for("test")
    for _ in range(5):
        breaker.on_failure()
    check("熔断器打开", breaker.is_open() and breaker.state == "open", breaker.state)

    print(f"\n=== 集成测试结果：{len(PASS)} 通过 / {len(FAIL)} 失败 ===")
    for f in FAIL:
        print("   - 失败：", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
