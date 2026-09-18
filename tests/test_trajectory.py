# -*- coding: utf-8 -*-
"""
轨迹日志的单元测试（不依赖网络、不依赖真实大模型）。

覆盖范围
--------
1. 记录：模型的输入输出、工具的输入输出都能落盘
2. 读回：load / summarize / frames 的形状正确
3. 容错：文件中混入坏行时，其余步骤仍能读出
4. 分叉：前缀复制、按 step 截断、新会话带 fork_of 标记
5. 回放：按顺序吐出录制的响应；耗尽后返回收尾消息而不是抛异常
6. 消息序列化：dump 与 load 往返一致（含 tool_calls）
7. 截断：设了字符上限时超长字段被裁剪并留标记
8. 隔离：轨迹读写失败不会把调用方带崩

运行：
    python tests/test_trajectory.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_TMP = tempfile.mkdtemp(prefix="trajectory_test_")
os.environ["TRAJECTORY_DIR"] = _TMP

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402

from observability.trajectory import (  # noqa: E402
    TrajectoryRecorder,
    TrajectoryStore,
    build_replay_model,
    dump_message,
    load_message,
    trajectory_enabled,
)

PASS: list[str] = []
FAIL: list[str] = []


def check(cond, name):
    (PASS if cond else FAIL).append(name)
    print(("  [OK]   " if cond else "  [FAIL] ") + name)


def _fake_llm_result(message, usage=None):
    gen = type("G", (), {"message": message, "text": getattr(message, "content", "")})()
    return type("R", (), {"generations": [[gen]], "llm_output": {"token_usage": usage or {}}})()


def _record_basic(session_id="s1"):
    rec = TrajectoryRecorder(session_id)
    rec.on_chat_model_start(
        {"name": "ChatOpenAI"},
        [[SystemMessage(content="你是助手"), HumanMessage(content="查阿莫西林库存")]],
        run_id="r1")
    rec.on_llm_end(_fake_llm_result(
        AIMessage(content="", tool_calls=[
            {"name": "execute_sql_query", "args": {"q": "select 1"}, "id": "c1"}]),
        usage={"total_tokens": 42}), run_id="r1")
    rec.on_tool_start({"name": "execute_sql_query"}, "select 1", run_id="r2")
    rec.on_tool_end("drug,stock\n阿莫西林,120", run_id="r2")
    return rec


# ---------------------------------------------------------------------------

def test_records_all_kinds():
    print("\n[1] 记录：模型与工具的输入输出")
    _record_basic("s1")
    store = TrajectoryStore()
    steps = store.load("s1")
    kinds = [s.kind for s in steps]
    check(kinds == ["llm_input", "llm_output", "tool_input", "tool_output"],
          f"四类步骤齐全（实际 {kinds}）")
    check(store.exists("s1"), "轨迹文件已创建")


def test_input_snapshot_is_complete():
    print("\n[2] 输入快照记下了模型看到的原文")
    steps = TrajectoryStore().load("s1")
    llm_in = [s for s in steps if s.kind == "llm_input"][0]
    flat = json.dumps(llm_in.payload, ensure_ascii=False)
    check("你是助手" in flat, "系统提示词被记录")
    check("查阿莫西林库存" in flat, "用户消息被记录")
    check(llm_in.payload.get("model") == "ChatOpenAI", "模型名被记录")


def test_tool_calls_recorded():
    print("\n[3] 模型返回的 tool_calls 被记录")
    steps = TrajectoryStore().load("s1")
    llm_out = [s for s in steps if s.kind == "llm_output"][0]
    calls = llm_out.result["generations"][0].get("tool_calls") or []
    check(len(calls) == 1 and calls[0]["name"] == "execute_sql_query",
          "tool_calls 完整")
    check(llm_out.result.get("usage", {}).get("total_tokens") == 42, "token 用量被记录")


def test_summarize_and_frames():
    print("\n[4] 概览与时间线")
    store = TrajectoryStore()
    summary = store.summarize("s1")
    check(summary["steps"] == 4, f"步数正确（实际 {summary['steps']}）")
    check(summary["llm_calls"] == 1, "LLM 调用次数正确")
    check(summary["tool_calls"] == 1, "工具调用次数正确")

    frames = store.frames("s1")
    check(len(frames) == 4, "时间线帧数一致")
    tool_frame = [f for f in frames if f["kind"] == "tool_input"][0]
    check(tool_frame.get("tool") == "execute_sql_query", "时间线带出工具名")

    deep = store.frames("s1", with_payload=True)
    check("payload" in deep[0] and deep[0]["payload"], "with_payload 时带完整载荷")


def test_bad_line_tolerated():
    print("\n[5] 容错：混入坏行")
    path = Path(_TMP) / "s1.jsonl"
    original = path.read_text(encoding="utf-8")
    path.write_text(original + "{这不是合法 json}\n", encoding="utf-8")
    steps = TrajectoryStore().load("s1")
    check(len(steps) == 4, f"坏行被跳过，其余仍可读（实际 {len(steps)} 步）")
    path.write_text(original, encoding="utf-8")


def test_fork_prefix():
    print("\n[6] 分叉：前缀复制")
    store = TrajectoryStore()
    new_id = store.fork("s1", new_session_id="s1-fork")
    check(store.exists(new_id), "新轨迹已创建")
    steps = store.load(new_id)
    check(len(steps) == 5, f"4 步正文 + 1 条 meta（实际 {len(steps)}）")
    meta = [s for s in steps if s.kind == "meta"][0]
    check(meta.meta.get("fork_of") == "s1", "记录了来源轨迹")

    # 按 step 截断
    cut = store.fork("s1", upto_step="2", new_session_id="s1-cut")
    check(len(store.load(cut)) == 3, "按 step 截断到前缀（2 步 + meta）")


def test_replay_order_and_exhaustion():
    print("\n[7] 回放：顺序与耗尽")
    model = build_replay_model("s1")
    first = model.invoke([HumanMessage(content="任意")])
    check(isinstance(first, AIMessage), "返回的是 AIMessage")
    check(bool(first.tool_calls), "第一条带着当初的 tool_calls")

    second = model.invoke([])
    check(isinstance(second, AIMessage), "耗尽后仍返回消息")
    check("回放" in str(second.content), f"耗尽时给收尾提示（实际 {second.content!r}）")
    check(model.bind_tools([]) is not None, "bind_tools 可用（agent 构图会调它）")


def test_message_roundtrip():
    print("\n[8] 消息序列化往返")
    cases = [
        SystemMessage(content="系统"),
        HumanMessage(content="用户"),
        AIMessage(content="回答"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {"a": 1}, "id": "x"}]),
        ToolMessage(content="结果", tool_call_id="x", name="t"),
    ]
    for msg in cases:
        back = load_message(dump_message(msg))
        check(type(back) is type(msg), f"{type(msg).__name__} 类型保持")
        if isinstance(msg, AIMessage) and msg.tool_calls:
            check(back.tool_calls and back.tool_calls[0]["name"] == "t", "tool_calls 保住")


def test_field_clipping():
    print("\n[9] 字段截断")
    long_text = "甲" * 500
    os.environ["TRAJECTORY_MAX_FIELD_CHARS"] = "100"
    rec = TrajectoryRecorder("s-clip")
    rec.on_chat_model_start({"name": "m"}, [[HumanMessage(content=long_text)]], run_id="r")
    os.environ["TRAJECTORY_MAX_FIELD_CHARS"] = "0"

    got = TrajectoryStore().load("s-clip")[0]
    content = got.payload["messages"][0][0]["content"]
    check(len(content) < 500, f"超长内容被裁剪（{len(content)} 字符）")
    check("已截断" in content, "裁剪处留了标记")


def test_write_failure_isolated():
    print("\n[10] 轨迹写失败不影响调用方")
    rec = TrajectoryRecorder("s-broken")
    rec._path = Path(_TMP) / "no_such_dir" / "x.jsonl"   # 父目录不存在
    try:
        rec.on_tool_start({"name": "t"}, "in", run_id="r")
        survived = True
    except Exception as exc:
        survived = False
        print("     抛出：", exc)
    check(survived, "写失败被吞掉，调用方不受影响")


def test_enabled_flag():
    print("\n[11] 开关")
    os.environ["TRAJECTORY_ENABLED"] = "0"
    check(trajectory_enabled() is False, "读到 0 时关闭")
    os.environ["TRAJECTORY_ENABLED"] = "1"
    check(trajectory_enabled() is True, "读到 1 时开启")


def main():
    test_records_all_kinds()
    test_input_snapshot_is_complete()
    test_tool_calls_recorded()
    test_summarize_and_frames()
    test_bad_line_tolerated()
    test_fork_prefix()
    test_replay_order_and_exhaustion()
    test_message_roundtrip()
    test_field_clipping()
    test_write_failure_isolated()
    test_enabled_flag()

    shutil.rmtree(_TMP, ignore_errors=True)
    print(f"\n=== 单元测试结果：{len(PASS)} 通过 / {len(FAIL)} 失败 ===")
    for f in FAIL:
        print("   - 失败：", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
