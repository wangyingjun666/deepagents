# -*- coding: utf-8 -*-
"""
结果反思的单元测试（不依赖网络、不依赖真实大模型）。

覆盖范围
--------
1. 解析：标准 JSON、带代码块包裹、脏字段都能处理
2. 缺字段兜底：模型少给 sufficient 时按「充分」处理，不触发无谓重跑
3. fail-open：反思调用抛异常、返回非 JSON，一律按「充分」处理
4. 判不充分但没给补充方向时，等同充分（再跑也没方向）
5. 有界：max_rounds 为 0 或配置钳制
6. 追加指令的措辞：包含缺口条目、包含检索问题、且明确要求不重写
7. 配置从环境变量装配并钳制到合法区间

运行：
    python tests/test_reflection.py
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# agent.prompts 在导入期会把提示词打印到 stdout，这里屏蔽掉，免得盖住测试结果
with contextlib.redirect_stdout(io.StringIO()):
    from agent.reflection import (  # noqa: E402
        MAX_MISSING_ITEMS,
        ReflectionConfig,
        ReflectionResult,
        build_followup_message,
        load_reflection_config,
        parse_reflection_response,
        reflect_on_answer,
    )

PASS: list[str] = []
FAIL: list[str] = []


def check(cond, name):
    (PASS if cond else FAIL).append(name)
    print(("  [OK]   " if cond else "  [FAIL] ") + name)


def _fake_llm(payload: str):
    """返回一个固定响应的 llm_call。"""
    return lambda prompt: payload


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def test_parse_plain_json():
    print("\n[1] 解析：标准 JSON")
    raw = '{"sufficient": false, "missing": ["缺少价格"], "next_query": "查价格"}'
    got = parse_reflection_response(raw)
    check(got is not None, "能解析")
    check(got["sufficient"] is False, "sufficient 解析为 False")
    check(got["missing"] == ["缺少价格"], "missing 解析正确")
    check(got["next_query"] == "查价格", "next_query 解析正确")


def test_parse_fenced_json():
    print("\n[2] 解析：带 ```json 包裹")
    raw = '```json\n{"sufficient": true, "missing": [], "next_query": ""}\n```'
    got = parse_reflection_response(raw)
    check(got is not None and got["sufficient"] is True, "剥掉代码块后能解析")


def test_parse_dirty_fields():
    print("\n[3] 解析：字段脏数据")
    raw = '{"sufficient": false, "missing": "只有一条", "next_query": null}'
    got = parse_reflection_response(raw)
    check(got is not None, "能解析")
    check(got["missing"] == ["只有一条"], "字符串型 missing 被包成列表")
    check(got["next_query"] == "", "null 的 next_query 归一成空串")


def test_parse_missing_sufficient_defaults_true():
    print("\n[4] 缺 sufficient 字段时按「充分」处理")
    got = parse_reflection_response('{"missing": ["x"], "next_query": "y"}')
    check(got is not None and got["sufficient"] is True,
          "少写字段不会触发无谓的一轮重跑")


def test_parse_garbage_returns_none():
    print("\n[5] 解析：非 JSON")
    check(parse_reflection_response("我觉得还行") is None, "非 JSON 返回 None")
    check(parse_reflection_response("[1, 2]") is None, "顶层是列表也返回 None")
    check(parse_reflection_response("") is None, "空串返回 None")


# ---------------------------------------------------------------------------
# fail-open
# ---------------------------------------------------------------------------

def test_llm_exception_is_fail_open():
    print("\n[6] fail-open：反思调用抛异常")
    def boom(prompt):
        raise RuntimeError("模型超时")

    r = reflect_on_answer("任务", "回答", llm_call=boom,
                          config=ReflectionConfig(enabled=True, max_rounds=2))
    check(r.sufficient is True, "异常时判定为「充分」")
    check(r.needs_another_round() is False, "不会触发重跑")
    check("异常" in r.reason, "原因里记录了异常")


def test_non_json_response_is_fail_open():
    print("\n[7] fail-open：返回不是 JSON")
    r = reflect_on_answer("任务", "回答", llm_call=_fake_llm("我不确定"),
                          config=ReflectionConfig(enabled=True, max_rounds=2))
    check(r.sufficient is True, "解析不了时判定为「充分」")
    check(r.needs_another_round() is False, "不会触发重跑")


def test_empty_answer_skips():
    print("\n[8] 本轮没有产出回答时不反思")
    called = {"n": 0}

    def spy(prompt):
        called["n"] += 1
        return "{}"

    r = reflect_on_answer("任务", "   ", llm_call=spy,
                          config=ReflectionConfig(enabled=True, max_rounds=2))
    check(r.sufficient is True, "空回答判充分")
    check(called["n"] == 0, "压根没有调用模型")


# ---------------------------------------------------------------------------
# 是否值得再跑一轮
# ---------------------------------------------------------------------------

def test_insufficient_without_query_is_final():
    print("\n[9] 判不充分但没给补充方向 → 等同充分")
    raw = '{"sufficient": false, "missing": ["缺东西"], "next_query": ""}'
    r = reflect_on_answer("任务", "回答", llm_call=_fake_llm(raw),
                          config=ReflectionConfig(enabled=True, max_rounds=2))
    check(r.sufficient is True, "被回落成充分")
    check(r.needs_another_round() is False, "不会用空问题去重跑")


def test_insufficient_with_query_triggers_round():
    print("\n[10] 判不充分且给了方向 → 应该再跑一轮")
    raw = '{"sufficient": false, "missing": ["只查到前三名"], "next_query": "补齐第四到第五名"}'
    r = reflect_on_answer("查销量前五", "前三名是...", llm_call=_fake_llm(raw),
                          config=ReflectionConfig(enabled=True, max_rounds=2))
    check(r.sufficient is False, "判定为不充分")
    check(r.needs_another_round() is True, "会触发追加一轮")
    check(r.missing == ["只查到前三名"], "缺口被保留")


def test_missing_items_capped():
    print("\n[11] 缺口条目数有上限")
    items = [f"缺口{i}" for i in range(20)]
    raw = '{"sufficient": false, "missing": %s, "next_query": "继续查"}' % (
        "[" + ",".join('"%s"' % i for i in items) + "]")
    r = reflect_on_answer("任务", "回答", llm_call=_fake_llm(raw),
                          config=ReflectionConfig(enabled=True, max_rounds=2))
    check(len(r.missing) <= MAX_MISSING_ITEMS,
          f"缺口被截到 {MAX_MISSING_ITEMS} 条以内（实际 {len(r.missing)}）")


# ---------------------------------------------------------------------------
# 配置与开关
# ---------------------------------------------------------------------------

def test_disabled_short_circuits():
    print("\n[12] 关闭时直接返回充分，不调用模型")
    def spy(prompt):
        raise AssertionError("不应该被调用")

    r = reflect_on_answer("任务", "回答", llm_call=spy,
                          config=ReflectionConfig(enabled=False, max_rounds=2))
    check(r.sufficient is True, "关闭时判充分")

    r2 = reflect_on_answer("任务", "回答", llm_call=spy,
                           config=ReflectionConfig(enabled=True, max_rounds=0))
    check(r2.sufficient is True, "max_rounds=0 等同关闭")


def test_config_from_env_and_clamped():
    print("\n[13] 配置从环境变量装配并钳制")
    os.environ["REFLECTION_ENABLED"] = "0"
    os.environ["REFLECTION_MAX_ROUNDS"] = "99"
    cfg = load_reflection_config()
    check(cfg.enabled is False, "enabled 读到 0")
    check(cfg.max_rounds == 5, f"max_rounds 被钳到 5（实际 {cfg.max_rounds}）")

    os.environ["REFLECTION_MAX_ROUNDS"] = "-3"
    check(load_reflection_config().max_rounds == 0, "负数被钳到 0")

    os.environ["REFLECTION_MAX_ROUNDS"] = "abc"
    check(load_reflection_config().max_rounds == 2, "非数字回退默认值 2")

    os.environ["REFLECTION_ENABLED"] = "1"
    os.environ["REFLECTION_MAX_ROUNDS"] = "2"


# ---------------------------------------------------------------------------
# 追加指令
# ---------------------------------------------------------------------------

def test_followup_message_content():
    print("\n[14] 追加指令的内容")
    res = ReflectionResult(sufficient=False,
                           missing=["只覆盖了前三名", "没有给出价格"],
                           next_query="补齐第四到第五名并补充价格")
    msg = build_followup_message(res, 1)
    check("只覆盖了前三名" in msg, "包含缺口条目")
    check("没有给出价格" in msg, "包含全部缺口条目")
    check("补齐第四到第五名并补充价格" in msg, "包含本轮检索问题")
    check("不要重写" in msg, "明确要求不重写已完成的部分")
    check("不要编造" in msg, "要求查不到就说明查不到")
    check("第 1 轮" in msg, "标注了轮次")


def test_followup_message_without_missing():
    print("\n[15] 没有 missing 明细时也能拼出指令")
    res = ReflectionResult(sufficient=False, missing=[], next_query="再查一次X")
    msg = build_followup_message(res, 2)
    check("再查一次X" in msg, "检索问题仍然在")
    check("第 2 轮" in msg, "轮次正确")


def main():
    test_parse_plain_json()
    test_parse_fenced_json()
    test_parse_dirty_fields()
    test_parse_missing_sufficient_defaults_true()
    test_parse_garbage_returns_none()
    test_llm_exception_is_fail_open()
    test_non_json_response_is_fail_open()
    test_empty_answer_skips()
    test_insufficient_without_query_is_final()
    test_insufficient_with_query_triggers_round()
    test_missing_items_capped()
    test_disabled_short_circuits()
    test_config_from_env_and_clamped()
    test_followup_message_content()
    test_followup_message_without_missing()

    print(f"\n=== 单元测试结果：{len(PASS)} 通过 / {len(FAIL)} 失败 ===")
    for f in FAIL:
        print("   - 失败：", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
