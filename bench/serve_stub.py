# -*- coding: utf-8 -*-
"""
压测专用的启动器：用「固定延迟的桩模型」代替真实大模型，把服务跑起来。

为什么要替换模型
----------------
真实链路里一次深度搜索要调十几次大模型，端到端耗时基本由对面那台推理服务决定，
压出来的 P95 反映的是 OpenAI/DeepSeek 当时的快慢，跟自己写的代码没关系。

压测关心的是**本项目自己那段开销**：并发闸门排队、上下文绑定、沙箱申请与回收、
图调度、事件总线落盘与扇出、WebSocket 推送。把模型换成固定延迟的桩之后，
剩下这些就都被单独量出来了。

用法：
    python bench/serve_stub.py                       # 模型调用固定 300ms
    python bench/serve_stub.py --model-latency-ms 800
    python bench/serve_stub.py --port 8000 --profile deep

    --profile simple : 桩模型直接给答复（一轮模型调用）
    --profile deep   : 桩模型先委派子 Agent，再给答复（多轮 + 工具调用，更接近真实深度搜索）
"""

import argparse
import asyncio
import io
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def build_stub_model(latency_ms: int, profile: str):
    """造一个够用的假模型：能绑工具、能同步/异步返回，每次调用固定睡 latency_ms。"""
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class StubChatModel(BaseChatModel):
        """固定延迟桩模型。

        latency_ms / profile 写成 pydantic 字段，BaseChatModel 本身就是 pydantic 模型，
        这样实例化时能直接当参数传。
        """

        latency_ms: int = 300
        profile: str = "simple"

        @property
        def _llm_type(self) -> str:
            return "stub-chat-model"

        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
            # 真实模型绑完工具还是自己，桩模型照做即可：压测不关心工具 schema
            return self

        def _reply(self, messages):
            """按 profile 决定这一轮返回什么。"""
            return AIMessage(content=(
                "已完成检索与汇总。结论：该型号支持双面打印，"
                "建议按手册第 3 章步骤配置。"
                "（压测桩模型输出，内容无实际含义）"
            ))

        def _generate(self, messages, stop=None, run_manager: CallbackManagerForLLMRun = None,  # noqa: ANN001
                      **kwargs):  # noqa: ANN003
            time.sleep(self.latency_ms / 1000.0)
            return ChatResult(generations=[ChatGeneration(message=self._reply(messages))])

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN003
            # 真实推理是 IO 等待，用 sleep 而不是阻塞，才能让出事件循环、
            # 让并发请求真的重叠在一起——否则压测量的是"串行有多快"，没有意义
            await asyncio.sleep(self.latency_ms / 1000.0)
            return ChatResult(generations=[ChatGeneration(message=self._reply(messages))])

    return StubChatModel(latency_ms=latency_ms, profile=profile)


def install_stub(latency_ms: int, profile: str) -> None:
    """在 agent.main_agent 被导入之前把 agent.llm.model 换掉。

    main_agent 里写的是 `from agent.llm import model`，导入那一刻就把名字绑死了，
    所以必须先 import agent.llm、替换、再让 api.server 去导入 main_agent。

    注意 agent.llm 在模块加载时就会 init_chat_model，没有 Key 会直接抛异常——哪怕
    这个模型对象后面根本不会被用到。所以在导入之前先塞一个占位 Key 进去（纯本地
    字符串，不会发起任何网络请求）。
    """
    os.environ.setdefault("OPENAI_API_KEY", "sk-bench-stub-not-used")
    import agent.llm as llm_module

    llm_module.model = build_stub_model(latency_ms, profile)
    print(f"[Stub] 已替换大模型：固定延迟 {latency_ms}ms，profile={profile}")


def main() -> int:
    ap = argparse.ArgumentParser(description="压测用启动器（桩模型）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model-latency-ms", type=int, default=300)
    ap.add_argument("--profile", default="simple", choices=["simple", "deep"])
    ap.add_argument("--disable-checkpointer", dest="disable_checkpointer",
                    action="store_true", default=True,
                    help="跳过持久化 checkpointer（默认开，见下方说明）")
    ap.add_argument("--keep-checkpointer", dest="disable_checkpointer",
                    action="store_false",
                    help="保留持久化 checkpointer（当前代码下会因同步/异步不兼容而报错）")
    args = ap.parse_args()

    # 压测环境的默认配置：本机没有 Docker / MySQL / 外部 Key 时也能整条链路跑起来
    os.environ.setdefault("SANDBOX_BACKEND", "process")   # 无 Docker，走进程级后端
    os.environ.setdefault("SECURITY_MODE", "standard")    # 无安全主体时允许匿名
    os.environ.setdefault("APPROVAL_ENABLED", "1")
    os.environ.setdefault("CHECKPOINT_DB", "logs/bench_checkpoints.sqlite")
    os.environ.setdefault("MEMORY_DB", "logs/bench_memory.sqlite")
    os.environ.setdefault("TRACE_DIR", "logs/trace")
    os.environ.setdefault("AUDIT_DIR", "logs/audit")

    install_stub(args.model_latency_ms, args.profile)

    if args.disable_checkpointer:
        # 当前 main_agent 用的是同步 SqliteSaver，而图是用 astream 异步跑的，
        # LangGraph 会直接抛 "The SqliteSaver does not support async methods"，
        # 任务根本执行不下去。压测时先把它摘掉（checkpointer=None，图仍能跑，只是
        # 不落盘、不支持跨进程中断恢复）。
        #
        # 注意：这意味着压测结果里**不含 checkpoint 落盘开销**，真实数字会比这里高。
        import agent.main_agent as ma
        ma._checkpointer_cm = None
        print("[Bench] 已关闭持久化 checkpointer（同步 SqliteSaver 与异步执行不兼容）")

    import uvicorn
    from api.server import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
