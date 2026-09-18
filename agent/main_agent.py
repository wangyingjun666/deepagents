"""
主智能体：编排、会话生命周期、沙箱、权限与人工审批的汇合点。

一次任务执行的链路：
    ① 并发闸门（session_limiter）：全局会话数上限，超了排队
    ② 准备会话资源：工作目录、上传件目录
    ③ 建立安全主体（principal）：这个会话是谁、什么角色、有哪些能力
    ④ 启动沙箱：容器（真隔离）或受限进程（兜底），失败降级写审计
    ⑤ 绑定上下文：session_dir / thread_id / sandbox / principal 全部进 ContextVar
    ⑥ 执行图：astream 流式跑，逐 chunk 上报事件（工具、子 Agent、最终结果）
    ⑦ 中断-审批循环：需要人工确认的调用先挂起，等决议后用 Command 恢复
    ⑧ finally 清理：reset 上下文、释放沙箱、归还并发名额
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from langchain_core.messages import AIMessage

from langgraph.checkpoint.memory import InMemorySaver

# main_agent tool导入
from tools.markdown_tools import generate_markdown
from tools.pdf_tools import convert_md_to_pdf
from tools.upload_file_read_tool import read_file_content

from deepagents import create_deep_agent

from agent.llm import model
from agent.memory import (
    MEMORY_SOURCES,
    MemoryContext,
    build_backend,
    build_memory_store,
    ensure_memory_seed,
)
from agent.reflection import (
    build_followup_message,
    load_reflection_config,
    reflect_on_answer,
)
from agent.prompts import main_agent_content
from agent.subagents.database_query_agent import database_query_agent
from agent.subagents.knowledge_base_agent import knowledge_base_agent
from agent.subagents.network_search_agent import network_search_agent

from api.context import (
    reset_sandbox_context,
    reset_security_context,
    reset_session_context,
    set_sandbox_context,
    set_security_context,
    set_session_context,
    set_thread_context,
)
from api.monitor import monitor
from concurrency.limiter import LimiterTimeout, session_limiter
from observability.bus import event_bus
from observability.events import EventType
from observability.trajectory import TrajectoryRecorder, trajectory_enabled
from sandbox import sandbox_manager
from security.approval import approval_center, needs_approval
from security.audit import audit
from security.permissions import Role, SessionPrincipal

logger = logging.getLogger(__name__)

#: 需要人工审批的工具，交给框架的 HumanInTheLoopMiddleware 处理。
#: 只有有副作用且不可逆的动作才需要，查询类不需要（审批疲劳会让机制失效）。
INTERRUPT_ON = {
    "execute_sql_query": {"allowed_decisions": ["approve", "reject"]},
}


def build_checkpointer():
    """构造持久化 checkpointer（同步 SqliteSaver）。

    `InMemorySaver` 重启即丢、多 worker 各存一份，而且撑不住中断恢复：
    审批要等用户确认，可能几十秒甚至跨进程重启，图状态必须能存下来。
    """
    import os
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    path = os.getenv("CHECKPOINT_DB", "data/checkpoints.sqlite")
    db_path = Path(path)
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parents[1] / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 用同步 SqliteSaver 配 check_same_thread=False：LangGraph 的异步执行会把同步
    # checkpointer 的调用丢到线程池里跑，连接会跨线程使用，而 SQLite 默认禁止跨线程
    # 复用连接，必须显式放开。写并发由 LangGraph 的调用顺序保证，不会产生竞态。
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


# 没有 SQLite 依赖时退回内存版，但明确告警（会影响中断恢复）
try:
    _checkpointer_cm = build_checkpointer()
    CHECKPOINTER_PERSISTENT = True
except Exception as exc:  # pragma: no cover
    logger.warning("持久化 checkpointer 初始化失败（%s），退回内存版：中断恢复只能在同一进程内生效", exc)
    _checkpointer_cm = None
    CHECKPOINTER_PERSISTENT = False


# 长期记忆库：起不来就降级成「不启用记忆」，但不能拖垮整个服务
try:
    _memory_store = build_memory_store()
    ensure_memory_seed(_memory_store)
    MEMORY_ENABLED = True
except Exception as exc:  # pragma: no cover
    logger.warning("长期记忆初始化失败（%s），本次启动不启用跨会话记忆", exc)
    _memory_store = None
    MEMORY_ENABLED = False


def _make_agent():
    """延迟创建图，避免在模块导入期就初始化 checkpointer 与模型。"""
    checkpointer = _checkpointer_cm

    # 长期记忆三件套缺一不可：
    #   store   —— 持久化底座（LangGraph BaseStore）
    #   backend —— 把 /memories/ 前缀路由到 store，其余路径行为不变
    #   memory  —— 告诉 MemoryMiddleware 该把哪个文件读进系统提示词
    memory_kwargs = {}
    if _memory_store is not None:
        memory_kwargs = {
            "store": _memory_store,
            "backend": build_backend,
            "memory": MEMORY_SOURCES,
        }

    return create_deep_agent(
        model=model,
        system_prompt=main_agent_content["system_prompt"],
        tools=[generate_markdown, convert_md_to_pdf, read_file_content],
        checkpointer=checkpointer,
        subagents=[
            database_query_agent,
            network_search_agent,
            knowledge_base_agent,
        ],
        interrupt_on=INTERRUPT_ON,
        context_schema=MemoryContext,
        **memory_kwargs,
    ).with_config({"recursion_limit": int(__import__("os").getenv("AGENT_RECURSION_LIMIT", "40"))})


# recursion_limit 通过 AGENT_RECURSION_LIMIT 配置，默认 40。框架默认是 1000，
# 收紧是因为一次深度搜索的合理步数有限，放开到 1000 意味着跑飞的循环能烧掉 1000 步 token。
# 实测低于 20 会切断正常的"三段检索 + 成文"流程。

project_root_path = Path(__file__).parents[1].resolve()

_main_agent = None


def get_main_agent():
    """惰性获取图实例。"""
    global _main_agent
    if _main_agent is None:
        _main_agent = _make_agent()
    return _main_agent


# 保留模块级名字，首次访问时才构建图
class _LazyAgent:
    def __getattr__(self, item):
        return getattr(get_main_agent(), item)


main_agent = _LazyAgent()


async def run_deep_agent(task_query, session_id, *, user_id: str = "anonymous",
                         role: str | Role | None = None):
    """流式异步执行主智能体，执行过程中的事件全部经 monitor 上报。

    task_query: 前端提问的问题
    session_id: 每个前端会话对应的标识

    外层负责并发闸门，真正的工作在 `_run_session` 里。
    """
    principal = SessionPrincipal.build(session_id, user_id=user_id, role=role)

    # ---- ① 并发闸门：拿不到名额就在这里排队 ----
    try:
        async with session_limiter.acquire(session_id=session_id) as wait_ms:
            if wait_ms > 100:
                monitor._emit(EventType.QUEUED,
                              f"会话排队 {wait_ms}ms 后开始执行", {"wait_ms": wait_ms})
                event_bus.record_metric("session:queue", wait_ms)
            await _run_session(task_query, session_id, principal)
    except LimiterTimeout as exc:
        monitor._emit(EventType.ERROR, str(exc), {"reason": "limiter_timeout"}, level="error")
        audit.record(action="session_start", decision="deny", session_id=session_id,
                     user_id=principal.user_id, reason="queue_timeout")


async def _run_session(task_query: str, session_id: str, principal: SessionPrincipal) -> None:
    """单个会话的完整生命周期（含沙箱与上下文清理）。"""
    print(f"当前会话的main_agent开始执行了！ 会话id:{session_id}")

    # ---- ② 准备会话资源 ----
    session_dir = project_root_path / "output" / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_dir_str = str(session_dir).replace("\\", "/")

    uploads_dir = project_root_path / "updated" / f"session_{session_id}"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    relative_session_dir_str = str(session_dir.relative_to(project_root_path)).replace("\\", "/")

    updated_info_prompt = ""
    files = [f.name for f in uploads_dir.iterdir() if f.is_file()] if uploads_dir.exists() else []
    if files:
        for filename in files:
            # copy2 保留原文件的修改时间与权限等元数据
            shutil.copy2(uploads_dir / filename, session_dir / filename)
        updated_info_prompt = ("\n    [已上传文件] 已加载到工作目录:\n"
                               + "\n".join([f"    - {f}" for f in files])
                               + "\n    请优先使用工具（read_file_content）读取并参考这些文件。")

    # ---- ③④⑤ 沙箱 + 上下文绑定 ----
    tokens = {}
    sandbox_meta = {}
    try:
        async with sandbox_manager.acquire(session_id, workspace=session_dir,
                                           uploads=uploads_dir) as sandbox:
            sandbox_meta = sandbox.meta

            tokens["session"] = set_session_context(session_dir_str)
            tokens["thread"] = set_thread_context(session_id)
            tokens["sandbox"] = set_sandbox_context(sandbox)
            tokens["security"] = set_security_context(principal)

            monitor.report_session_dir(session_dir_str)
            monitor.report_sandbox(sandbox_meta)
            audit.record(action="session_start", decision="allow", session_id=session_id,
                         user_id=principal.user_id, role=principal.role.value,
                         target=str(session_dir), backend=sandbox_meta.get("backend", ""),
                         isolated=bool(sandbox_meta.get("isolated")))

            await _stream_graph(task_query, session_id, relative_session_dir_str,
                                updated_info_prompt, principal.user_id)
    except Exception as exc:
        logger.exception("会话执行失败")
        monitor._emit(EventType.ERROR, f"执行主智能发生异常信息：{exc}",
                      {"session_id": session_id}, level="error")
        audit.record(action="session_run", decision="error", session_id=session_id,
                     reason=f"{type(exc).__name__}: {exc}"[:300])
    finally:
        reset_security_context(tokens.get("security"))
        reset_sandbox_context(tokens.get("sandbox"))
        reset_session_context(tokens.get("session"), tokens.get("thread"))
        await sandbox_manager.release(session_id)


async def _stream_graph(task_query: str, session_id: str, workdir: str,
                        updated_info_prompt: str, user_id: str = "anonymous") -> None:
    """执行图并处理流式输出 + 中断审批循环。"""
    from langgraph.types import Command

    agent = get_main_agent()

    path_instruction = f"""
    【工作环境指令】
    工作目录: {workdir}
    {updated_info_prompt}

    规则：
    1. 新生成文件必须保存到工作目录：'{workdir}/filename'
    2. 读取已上传的文件时，请直接将文件名（例如：'开篇.txt'）作为 filename 参数传入（read_file_content）读取工具，不要带上任何目录前缀。
    3. 使用相对路径，禁止使用绝对路径
    4. 若存在上传文件，请先分析内容
    """

    config = {"configurable": {"thread_id": session_id}}
    payload = {"messages": [{"role": "user", "content": task_query + path_instruction}]}

    # 轨迹记录：把模型收到的完整输入和输出落盘，供事后回放与分叉。
    # 挂不上就跳过，不影响本次执行。
    if trajectory_enabled():
        try:
            config["callbacks"] = [TrajectoryRecorder(
                session_id, meta={"user_id": user_id, "task": task_query[:200]})]
        except Exception as exc:  # pragma: no cover
            logger.warning("轨迹记录器创建失败（%s），本次不记录轨迹", exc)

    # 长期记忆的命名空间靠 context 里的 user_id 决定，不能走 ContextVar：
    # 中间件钩子可能在线程池里执行，ContextVar 传不过去，而 context 是本次调用显式带的。
    run_context = MemoryContext(user_id=user_id)

    # 新用户第一次进来时命名空间是空的，MemoryMiddleware 对不存在的文件是静默跳过的，
    # 于是「功能装好了但什么都没发生」。这里补一次种子，让记忆从第一轮就可读。
    if MEMORY_ENABLED:
        try:
            ensure_memory_seed(_memory_store, user_id=user_id)
        except Exception as exc:  # pragma: no cover
            logger.warning("长期记忆种子写入失败（%s），本次跳过，不影响本次执行", exc)

    answer = await _run_graph_once(agent, payload, config, run_context, session_id)

    # 反思重规划：核对原始任务里的要求是否都覆盖到了，有缺口就带着缺口再跑一轮。
    # 轮数有上限，超了按当前结果交付；反思自身出错一律按「充分」处理。
    reflect_cfg = load_reflection_config()
    if reflect_cfg.enabled and reflect_cfg.max_rounds > 0:
        for round_no in range(1, reflect_cfg.max_rounds + 1):
            result = reflect_on_answer(task_query, answer, config=reflect_cfg,
                                       callbacks=config.get("callbacks"))
            monitor._emit(
                EventType.REFLECTION,
                f"第 {round_no} 轮反思：{result.reason}",
                {"round": round_no, "sufficient": result.sufficient,
                 "missing": result.missing, "next_query": result.next_query})
            if not result.needs_another_round():
                break

            logger.info("反思判定存在缺口，追加第 %s 轮检索", round_no)
            followup = build_followup_message(result, round_no)
            payload = {"messages": [{"role": "user", "content": followup}]}
            # 这一轮拿不到新回答时保留上一轮的结果，不让反思把已有的产出弄丢
            answer = await _run_graph_once(agent, payload, config, run_context,
                                           session_id) or answer
        else:
            monitor._emit(EventType.REFLECTION,
                          f"反思轮次已达上限（{reflect_cfg.max_rounds}），按当前结果交付",
                          {"exhausted": True}, level="warn")


async def _run_graph_once(agent, payload, config, run_context, session_id: str) -> str:
    """跑一遍图（含中断-审批循环），返回这一轮产出的最终回答。

    payload 为 Command 时表示从断点恢复，属于同一次调用，context 要与首次保持一致。
    """
    from langgraph.types import Command

    latest_answer = ""
    max_resumes = int(__import__("os").getenv("MAX_APPROVAL_ROUNDS", "5"))
    for round_no in range(max_resumes + 1):
        interrupted = False
        async for chunk in agent.astream(payload, config=config, context=run_context):
            # 中断信号由 LangGraph 以特殊 key 返回
            if "__interrupt__" in chunk:
                interrupted = True
                await _handle_interrupt(chunk["__interrupt__"], session_id)
                decision = _last_decision()
                payload = Command(resume=decision)
                break

            for node_name, state in chunk.items():
                if not state or "messages" not in state:
                    continue
                messages = state["messages"]
                if not messages or not isinstance(messages, list):
                    continue
                last_msg = messages[-1]
                if node_name == "model":
                    produced = _report_model_step(last_msg)
                    if produced:
                        latest_answer = produced
        if not interrupted:
            break
    else:
        monitor._emit(EventType.ERROR, "审批轮次超过上限，任务终止", level="warn")

    return latest_answer


def _report_model_step(last_msg) -> str:
    """把一轮模型的产出翻译成事件，返回最终回答文本（纯工具调用轮返回空串）。"""
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        for tool_call in last_msg.tool_calls:
            # tool_call 结构：
            # {name: "task", args: {subagent_type: 子智能体名字, description: 描述}}
            if tool_call["name"] == "task":
                monitor.report_assistant(
                    tool_call["args"]["subagent_type"],
                    {"description": tool_call["args"]["description"]})
        return ""

    content = getattr(last_msg, "content", None)
    if content:
        print(f"主智能体执行结果，最终结果：{str(content)[:100]}")
        monitor.report_task_result(content)
        return str(content)
    return ""


_pending_decision: dict[str, dict] = {}


def _last_decision() -> dict:
    """取出最近一次审批决议（由 `_handle_interrupt` 写入）。"""
    return _pending_decision.pop("resume", {"decisions": [{"type": "reject"}]})


async def _handle_interrupt(interrupts, session_id: str) -> None:
    """处理一次人工审批中断：把请求发到前端，等待决议，转成 resume 指令。"""
    for item in interrupts if isinstance(interrupts, (list, tuple)) else [interrupts]:
        value = getattr(item, "value", item)
        action_requests = {}
        if isinstance(value, dict):
            action_requests = value.get("action_requests") or value.get("requests") or {}

        if isinstance(action_requests, list) and action_requests:
            for req in action_requests:
                tool_name = req.get("name") or req.get("tool") or "unknown"
                args = req.get("args") or {}
                need, reason = needs_approval(tool_name, args)
                if not need:
                    reason = req.get("description") or "该操作需要人工确认后执行"
                approval = await approval_center.request(
                    session_id=session_id, tool_name=tool_name, args=args, reason=reason)
                decision = "approve" if approval.status == "approved" else "reject"
                _pending_decision["resume"] = {"decisions": [{"type": decision}]}
                monitor._emit(EventType.APPROVAL_RESOLVED,
                              f"{tool_name}：{approval.status}",
                              {"approval_id": approval.approval_id, "decision": decision})
                return
        else:
            # 没有结构化 action 信息时也要给用户一个决策点，默认拒绝更安全
            approval = await approval_center.request(
                session_id=session_id, tool_name="unknown", args={},
                reason="检测到需要人工确认的操作")
            decision = "approve" if approval.status == "approved" else "reject"
            _pending_decision["resume"] = {"decisions": [{"type": decision}]}
            return
