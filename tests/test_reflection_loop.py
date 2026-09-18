# -*- coding: utf-8 -*-
"""
反思循环的接线测试（把图和反思都替换成假的，不依赖网络、不依赖真实大模型）。

验的是 `_stream_graph` 里那层循环的行为，不是反思本身的判断逻辑
（那部分在 test_reflection.py）：

1. 反思判充分 → 只跑一轮，不追加
2. 反思判不充分 → 追加，总轮数 = 1 + max_rounds
3. 反思环节报错（fail-open）→ 只跑一轮，已产出的结果不受影响
4. 追加那轮没产出回答 → 保留上一轮的结果
5. 反思每轮都判不充分 → 到上限停下并发出告警事件
6. 反思拿到的是原始任务，不是上一轮的回答

运行：
    python tests/test_reflection_loop.py
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OPENAI_API_KEY", "dummy-for-test")

with contextlib.redirect_stdout(io.StringIO()):
    import agent.main_agent as ma  # noqa: E402
    from agent.reflection import ReflectionConfig, ReflectionResult  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(cond, name):
    (PASS if cond else FAIL).append(name)
    print(("  [OK]   " if cond else "  [FAIL] ") + name)


class Harness:
    """把 _stream_graph 依赖的外部件全部换掉，只留循环骨架。"""

    def __init__(self, answers, reflections, max_rounds=2):
        self.answers = list(answers)
        self.reflections = list(reflections)
        self.max_rounds = max_rounds
        self.rounds = 0
        self.reflect_inputs = []
        self.events = []
        self._saved = {}

    def __enter__(self):
        self._saved = {
            "get_main_agent": ma.get_main_agent,
            "_run_graph_once": ma._run_graph_once,
            "reflect_on_answer": ma.reflect_on_answer,
            "load_reflection_config": ma.load_reflection_config,
            "_emit": ma.monitor._emit,
        }

        ma.get_main_agent = lambda: object()          # 不真的构图

        async def fake_run(agent, payload, config, run_context, session_id):
            self.rounds += 1
            return self.answers.pop(0) if self.answers else ""

        ma._run_graph_once = fake_run

        def fake_reflect(task, answer, *, llm_call=None, config=None):
            self.reflect_inputs.append((task, answer))
            if self.reflections:
                return self.reflections.pop(0)
            return ReflectionResult(sufficient=True, reason="默认充分")

        ma.reflect_on_answer = fake_reflect
        ma.load_reflection_config = lambda: ReflectionConfig(
            enabled=True, max_rounds=self.max_rounds)
        ma.monitor._emit = lambda *a, **kw: self.events.append((a, kw))
        return self

    def __exit__(self, *exc):
        ma.get_main_agent = self._saved["get_main_agent"]
        ma._run_graph_once = self._saved["_run_graph_once"]
        ma.reflect_on_answer = self._saved["reflect_on_answer"]
        ma.load_reflection_config = self._saved["load_reflection_config"]
        ma.monitor._emit = self._saved["_emit"]
        return False


def run_session():
    asyncio.run(ma._stream_graph("查销量前五并生成报告", "sess-1", "output/session_sess-1",
                                 "", "tester"))


def insufficient(query="补齐第四到第五名"):
    return ReflectionResult(sufficient=False, missing=["只查到前三名"],
                            next_query=query, reason="发现缺口")


def sufficient():
    return ReflectionResult(sufficient=True, reason="回答已覆盖任务要求")


# ---------------------------------------------------------------------------

def test_sufficient_runs_once():
    print("\n[1] 反思判充分 → 只跑一轮")
    with Harness(answers=["第一版报告"], reflections=[sufficient()]) as h:
        run_session()
    check(h.rounds == 1, f"只跑了一轮（实际 {h.rounds}）")
    check(len(h.reflect_inputs) == 1, "反思被调用一次")


def test_insufficient_appends_rounds():
    print("\n[2] 反思判不充分 → 追加轮次")
    with Harness(answers=["第一版", "第二版", "第三版"],
                 reflections=[insufficient(), insufficient()],
                 max_rounds=2) as h:
        run_session()
    check(h.rounds == 3, f"1 首轮 + 2 追加 = 3（实际 {h.rounds}）")
    check(len(h.reflect_inputs) == 2, "每次追加前都先反思")


def test_reflect_error_fails_open():
    print("\n[3] 反思报错 → 不追加，已产出的结果不受影响")
    err = ReflectionResult(sufficient=True, reason="反思异常，按充分处理: timeout")
    with Harness(answers=["第一版报告"], reflections=[err]) as h:
        run_session()
    check(h.rounds == 1, f"只跑了一轮（实际 {h.rounds}）")
    emitted = [a[0] for a, _ in h.events if a]
    check(any(str(e) == "reflection" for e in emitted), "反思事件已上报")


def test_empty_followup_keeps_previous():
    print("\n[4] 追加轮没产出 → 保留上一轮结果")
    with Harness(answers=["第一版报告", ""],           # 第二轮返回空
                 reflections=[insufficient(), sufficient()],
                 max_rounds=2) as h:
        run_session()
    check(h.rounds == 2, f"追加了一轮（实际 {h.rounds}）")
    # 第二轮拿不到回答时应回落到第一版；再反思时传给模型的就是第一版
    check(h.reflect_inputs[1][1] == "第一版报告",
          f"传给第二轮反思的是首轮结果（实际 {h.reflect_inputs[1][1]!r}）")


def test_bounded_by_max_rounds():
    print("\n[5] 一直判不充分 → 到上限停下并告警")
    with Harness(answers=["v1", "v2", "v3", "v4", "v5"],
                 reflections=[insufficient(), insufficient(), insufficient(),
                              insufficient(), insufficient()],
                 max_rounds=2) as h:
        run_session()
    check(h.rounds == 3, f"轮数被 max_rounds 卡住（实际 {h.rounds}）")
    exhausted = [a[2] for a, _ in h.events
                 if len(a) > 2 and isinstance(a[2], dict) and a[2].get("exhausted")]
    check(len(exhausted) == 1, "发出了「已达上限」的告警事件")


def test_reflect_sees_original_task():
    print("\n[6] 反思拿到的是原始任务")
    with Harness(answers=["第一版"], reflections=[sufficient()]) as h:
        run_session()
    task, _ = h.reflect_inputs[0]
    check(task == "查销量前五并生成报告", f"任务原文一致（实际 {task!r}）")


def main():
    test_sufficient_runs_once()
    test_insufficient_appends_rounds()
    test_reflect_error_fails_open()
    test_empty_followup_keeps_previous()
    test_bounded_by_max_rounds()
    test_reflect_sees_original_task()

    print(f"\n=== 接线测试结果：{len(PASS)} 通过 / {len(FAIL)} 失败 ===")
    for f in FAIL:
        print("   - 失败：", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
