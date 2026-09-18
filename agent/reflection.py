"""
回答充分性反思：交付前核一遍结果有没有答全，没答全就带着缺口再跑一轮。

要解决的问题：
    主智能体拿到子智能体结果后直接成文，中间没有人回头核对"用户要的是不是都齐了"。
    最常见的漏项是只答了一半——问"销量前五名"只查到前三、要求对比两个型号只覆盖了一个。
    这类漏项不会报错，交付出去才被发现。

做法：
    成文后跑一次反思，把原始任务和最终回答一起交给模型，输出
        {"sufficient": bool, "missing": [...], "next_query": "..."}
    不充分就把 missing 和 next_query 拼成一条追加指令，再跑一轮。
    轮数有上限，超了按当前结果交付。

三条约束：
    1. 有界。反思一直判"不充分"的话，没有上限会一路烧穿 recursion_limit。
    2. fail-open。反思自己出问题（超时、返回解析不了）一律当成"充分"。
       宁可交付一份可能不完整的答案，也不能因为反思环节挂了把已经拿到的结果丢掉。
    3. 可注入。llm_call 可替换，单测不依赖真实模型。
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from agent.prompts import reflection_content

logger = logging.getLogger(__name__)

LOG_TAG = "[反思]"

#: 交给模型判断的回答长度上限，超出截断。反思关心的是覆盖面，不需要全文。
MAX_ANSWER_CHARS = 6000

#: 追加指令里回显的缺口条目上限，多了会把下一轮的问句撑得很长。
MAX_MISSING_ITEMS = 6


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class ReflectionConfig:
    """反思阈值配置，全部可用环境变量覆盖。"""

    enabled: bool = True        # 总开关，关掉就退回"一次成文"
    max_rounds: int = 2         # 最多追加几轮，0 等于关闭
    llm_model: str = ""         # 反思用模型，空则复用默认


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("%s 环境变量 %s=%r 不是整数，回退默认值 %s", LOG_TAG, name, raw, default)
        return default


def load_reflection_config() -> ReflectionConfig:
    """从环境变量装配反思配置。"""
    return ReflectionConfig(
        enabled=_env_bool("REFLECTION_ENABLED", True),
        # 上限卡在 0~5：再多没有实际收益，只会成倍放大 token 消耗和时延
        max_rounds=min(5, max(0, _env_int("REFLECTION_MAX_ROUNDS", 2))),
        llm_model=os.getenv("REFLECTION_LLM_MODEL") or "",
    )


# ---------------------------------------------------------------------------
# 返回值
# ---------------------------------------------------------------------------

@dataclass
class ReflectionResult:
    """一次反思的结论。

    sufficient 为 True 有两种来源：模型确实认为够，或者反思环节本身没跑成
    （fail-open）。调用方不需要区分，两种情况都按"可以交付"处理。
    """

    sufficient: bool = True
    missing: List[str] = field(default_factory=list)
    next_query: str = ""
    reason: str = ""

    def needs_another_round(self) -> bool:
        """是否值得再跑一轮：判不充分，且给出了可执行的补充方向。"""
        return (not self.sufficient) and bool(self.next_query.strip())


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def _strip_code_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _as_str_list(value: Any) -> List[str]:
    """把模型返回的字段规整成去重、去空、保序的字符串列表。"""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    result: List[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def parse_reflection_response(raw: str) -> Optional[dict]:
    """解析反思模型的 JSON 返回，无法解析返回 None。"""
    cleaned = _strip_code_fence(raw)
    if not cleaned:
        return None
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    # sufficient 只在模型明确给了布尔值时才采信，缺字段一律当"够"，
    # 避免模型少写一个字段就触发一轮没必要的重跑
    sufficient = data.get("sufficient")
    if not isinstance(sufficient, bool):
        sufficient = True

    return {
        "sufficient": sufficient,
        "missing": _as_str_list(data.get("missing")),
        "next_query": str(data.get("next_query") or "").strip(),
    }


def _default_llm_call(prompt: str, callbacks: Optional[List[Any]] = None) -> str:
    """默认的反思调用：走项目统一的大模型入口。

    callbacks 透传进来是为了让反思这几次调用也进轨迹 —— 否则排障时会看到
    "两次模型调用之间凭空多了一轮"，而那一轮恰恰是判断要不要重跑的。
    """
    from agent.llm import model

    if callbacks:
        response = model.invoke(prompt, config={"callbacks": list(callbacks)})
    else:
        response = model.invoke(prompt)
    return getattr(response, "content", "") or ""


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def reflect_on_answer(
        task: str,
        answer: str,
        *,
        llm_call: Optional[Callable[[str], str]] = None,
        config: Optional[ReflectionConfig] = None,
        callbacks: Optional[List[Any]] = None,
) -> ReflectionResult:
    """判断当前回答是否充分回答了原始任务。

    :param task: 用户最初的任务描述
    :param answer: 本轮产出的最终回答
    :param llm_call: 反思生成函数，签名 (prompt) -> str；默认走项目大模型
    :param config: 配置，默认从环境变量读
    :param callbacks: 透传给模型调用的回调（通常是轨迹记录器）；注入了 llm_call 时忽略
    :return: ReflectionResult；任何失败都返回"充分"，不会抛异常
    """
    cfg = config or load_reflection_config()
    result = ReflectionResult()

    if not cfg.enabled or cfg.max_rounds <= 0:
        result.reason = "反思未启用"
        return result

    if not (answer or "").strip():
        result.reason = "本轮没有产出回答，无需反思"
        return result

    try:
        prompt = reflection_content["user_prompt"].format(
            task=task,
            answer=(answer or "").strip()[:MAX_ANSWER_CHARS],
        )
        # 自己注入的 llm_call 由调用方负责，默认路径才需要把回调带下去
        raw = llm_call(prompt) if llm_call else _default_llm_call(prompt, callbacks)
        parsed = parse_reflection_response(raw)

        if parsed is None:
            # fail-open：解析不了就按"够"处理，绝不能因为反思挂了丢结果
            result.reason = "反思返回无法解析为 JSON，按充分处理"
            logger.warning("%s 解析失败，本轮按充分处理 | 原始返回: %s",
                           LOG_TAG, str(raw)[:300])
            return result

        result.sufficient = parsed["sufficient"]
        result.missing = parsed["missing"][:MAX_MISSING_ITEMS]
        result.next_query = parsed["next_query"]

        if result.sufficient:
            result.reason = "回答已覆盖任务要求"
        elif not result.next_query:
            # 判了不充分却没给下一步，再跑也没有方向，等同充分
            result.sufficient = True
            result.reason = "判为不充分但未给出补充方向，按充分处理"
        else:
            result.reason = "发现缺口：" + "；".join(result.missing or ["未说明"])

        logger.info("%s 结论=%s | %s", LOG_TAG,
                    "充分" if result.sufficient else "需补充", result.reason)
        return result

    except Exception as exc:
        result.sufficient = True
        result.reason = f"反思异常，按充分处理: {exc}"
        logger.warning("%s 执行异常，本轮按充分处理：%s", LOG_TAG, exc, exc_info=True)
        return result


def build_followup_message(result: ReflectionResult, round_no: int) -> str:
    """把反思结论拼成下一轮的追加指令。

    措辞上刻意要求"只补缺口，不要重写已完成的报告"——否则模型容易把
    上一轮给出的报告推倒重来，既浪费 token，也会把已经核对过的内容改坏。
    """
    lines = [
        f"【补充检索指令 · 第 {round_no} 轮】",
        "你上一轮给出的结果经核对后仍存在缺口，请针对下列内容补充检索与核实：",
    ]
    for idx, item in enumerate(result.missing, start=1):
        lines.append(f"  {idx}. {item}")

    lines.append("")
    lines.append(f"本轮需要查清的问题：{result.next_query}")
    lines.append("")
    lines.append("要求：")
    lines.append("1. 只补充上述缺口，不要重写已经完成的部分。")
    lines.append("2. 如果检索后确认某项确实查不到，明确说明查不到，不要编造。")
    lines.append("3. 补齐后重新给出一份完整结果。")
    return "\n".join(lines)
