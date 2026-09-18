"""
轨迹日志：把模型每一次调用"看到什么、返回什么"按顺序原样落盘，用于事后回放与分叉。

和 event_bus 的分工
-------------------
event_bus 面向当下：事件是薄的（谁在干什么），实时推给前端、顺手聚合指标。
trajectory 面向事后：记的是模型收到的完整输入（系统提示词、压缩后的历史、
工具 schema）和完整输出，体积大得多，但能拿来重放。

为什么非要把完整输入记下来：
    排障时经常要回答"模型当时到底看到了什么"。事件日志只能告诉你它调了哪个工具，
    回答不了"它是不是压根没看到那段上下文"。把输入快照留下来，这个问题才有答案。

回放与分叉
----------
ReplayModel 按录制顺序吐出当初的响应，不打真实 API，于是同一条轨迹可以反复重跑，
用来验证"改了代码之后，同样的输入还会不会走出同样的路"。
fork 把某条轨迹的前缀复制成一条新轨迹，在分叉点之后换参数继续跑，两条直接对比。

挂载方式：
    config = {"configurable": {"thread_id": sid}, "callbacks": [TrajectoryRecorder(sid)]}
回调是同步的，LangChain 在异步链路里会把它丢到线程池执行，不会阻塞事件循环。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

#: 落盘目录，可用 TRAJECTORY_DIR 覆盖；关了就没轨迹
DEFAULT_TRAJECTORY_DIR = "logs/trajectory"

#: 单个字段的字符上限，0 表示不截断。默认不截断，需要时再收紧。
DEFAULT_MAX_FIELD_CHARS = 0


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
        logger.warning("环境变量 %s=%r 不是整数，回退默认值 %s", name, raw, default)
        return default


def trajectory_enabled() -> bool:
    return _env_bool("TRAJECTORY_ENABLED", True)


def _trajectory_root() -> Path:
    raw = os.getenv("TRAJECTORY_DIR", DEFAULT_TRAJECTORY_DIR)
    path = Path(raw)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_session_id(session_id: str) -> str:
    keep = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(session_id))
    return keep[:80] or "unknown"


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------

def _clip(value: Any, limit: int) -> Any:
    """按字符上限裁剪长文本，裁剪后留下标记，避免看起来像内容本身就短。"""
    if not limit or not isinstance(value, str):
        return value
    if len(value) <= limit:
        return value
    return f"{value[:limit]}…[已截断 {len(value) - limit} 字符]"


def dump_message(message: Any, limit: int = DEFAULT_MAX_FIELD_CHARS) -> Dict[str, Any]:
    """把一条消息压成可 JSON 化的字典。不认识的类型退化成字符串。"""
    try:
        from langchain_core.messages import BaseMessage
    except ImportError:  # pragma: no cover
        BaseMessage = ()  # type: ignore

    if not isinstance(message, BaseMessage):
        return {"raw": _clip(str(message), limit)}

    content = message.content
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, default=str)

    dumped: Dict[str, Any] = {
        "type": message.__class__.__name__,
        "content": _clip(content, limit),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        dumped["tool_calls"] = [
            {"name": c.get("name"), "args": c.get("args"), "id": c.get("id")}
            for c in tool_calls
        ]
    for attr in ("tool_call_id", "name"):
        value = getattr(message, attr, None)
        if value:
            dumped[attr] = value
    return dumped


def load_message(data: Dict[str, Any]) -> Any:
    """dump_message 的逆操作，用于回放时重建消息。"""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    if "raw" in data and "type" not in data:
        return HumanMessage(content=data["raw"])

    kind = data.get("type") or ""
    content = data.get("content") or ""
    if kind.startswith("System"):
        return SystemMessage(content=content)
    if kind.startswith("Tool"):
        return ToolMessage(content=content,
                           tool_call_id=data.get("tool_call_id") or "replay",
                           name=data.get("name") or "tool")
    if kind.startswith("AI"):
        kwargs: Dict[str, Any] = {}
        if data.get("tool_calls"):
            kwargs["tool_calls"] = data["tool_calls"]
        return AIMessage(content=content, **kwargs)
    return HumanMessage(content=content)


# ---------------------------------------------------------------------------
# 轨迹记录
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryStep:
    """轨迹上的一步。"""

    step_id: str
    kind: str                       # llm | tool | error
    ts: float
    duration_ms: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class TrajectoryRecorder:
    """挂在 agent 调用上的回调，把每一步原样写进 JSONL。

    刻意不继承 BaseCallbackHandler：那个基类在 import 期就把 langchain_core 拉起来，
    而本模块的读取侧（TrajectoryStore）在排障脚本里常常不装 langchain 也要能用。
    接口形状保持一致，用的时候挂到 config["callbacks"] 里即可。
    """

    #: 让 LangChain 把它当回调处理
    ignore_llm = False
    ignore_chat_model = False
    ignore_chain = True
    ignore_agent = True
    ignore_retriever = True
    raise_error = False
    run_inline = False

    def __init__(self, session_id: str, *, fork_of: Optional[str] = None,
                 meta: Optional[Dict[str, Any]] = None) -> None:
        self.session_id = session_id
        self.fork_of = fork_of
        self._started: Dict[str, float] = {}
        self._seq = 0
        self._lock = threading.Lock()
        self._limit = _env_int("TRAJECTORY_MAX_FIELD_CHARS", DEFAULT_MAX_FIELD_CHARS)
        self._path = _trajectory_root() / f"{_safe_session_id(session_id)}.jsonl"
        self._header_written = self._path.exists()
        if meta:
            self._append({"step_id": "0", "kind": "meta", "ts": time.time(),
                          "meta": dict(meta, fork_of=fork_of, session_id=session_id)})

    # -- 内部 ---------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    def _append(self, record: Dict[str, Any]) -> None:
        try:
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    fh.flush()
        except Exception as exc:  # 轨迹写失败不能影响主链路
            logger.warning("轨迹写入失败（%s）：%s", self._path, exc)

    def _next_step_id(self) -> str:
        self._seq += 1
        return str(self._seq)

    def _tick(self, run_id: Any) -> int:
        started = self._started.pop(str(run_id), None)
        return int((time.time() - started) * 1000) if started else 0

    # -- LLM ----------------------------------------------------------------

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        """记录模型收到的完整输入。messages 是 batch 的二维列表，这里逐批展开。"""
        self._started[str(run_id)] = time.time()
        batches = []
        for batch in messages or []:
            batches.append([dump_message(m, self._limit) for m in batch])
        self._append({
            "step_id": self._next_step_id(),
            "kind": "llm_input",
            "ts": time.time(),
            "payload": {
                "model": (serialized or {}).get("name") or kwargs.get("name"),
                "messages": batches,
            },
            "meta": {"run_id": str(run_id), "tags": kwargs.get("tags")},
        })

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        """非对话模型的入口，保持同样的记录形状，便于统一读取。"""
        self._started.setdefault(str(run_id), time.time())
        self._append({
            "step_id": self._next_step_id(),
            "kind": "llm_input",
            "ts": time.time(),
            "payload": {
                "model": (serialized or {}).get("name") or kwargs.get("name"),
                "messages": [[{"type": "Prompt", "content": _clip(p, self._limit)}]
                             for p in (prompts or [])],
            },
            "meta": {"run_id": str(run_id)},
        })

    def on_llm_end(self, response, *, run_id, **kwargs):
        generations = []
        for batch in getattr(response, "generations", []) or []:
            for gen in batch:
                message = getattr(gen, "message", None)
                generations.append(dump_message(message, self._limit)
                                   if message is not None
                                   else {"text": _clip(getattr(gen, "text", ""), self._limit)})
        usage = getattr(response, "llm_output", None) or {}
        self._append({
            "step_id": self._next_step_id(),
            "kind": "llm_output",
            "ts": time.time(),
            "duration_ms": self._tick(run_id),
            "result": {"generations": generations,
                       "usage": usage.get("token_usage") or usage.get("usage") or {}},
            "meta": {"run_id": str(run_id)},
        })

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._append({
            "step_id": self._next_step_id(),
            "kind": "llm_error",
            "ts": time.time(),
            "duration_ms": self._tick(run_id),
            "result": {"error": f"{type(error).__name__}: {error}"},
            "meta": {"run_id": str(run_id)},
        })

    # -- 工具 ---------------------------------------------------------------

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self._started[str(run_id)] = time.time()
        self._append({
            "step_id": self._next_step_id(),
            "kind": "tool_input",
            "ts": time.time(),
            "payload": {
                "tool": (serialized or {}).get("name") or kwargs.get("name"),
                "input": _clip(input_str, self._limit),
                "inputs": kwargs.get("inputs"),
            },
            "meta": {"run_id": str(run_id), "parent_run_id": str(kwargs.get("parent_run_id"))},
        })

    def on_tool_end(self, output, *, run_id, **kwargs):
        text = output if isinstance(output, str) else json.dumps(
            getattr(output, "content", output), ensure_ascii=False, default=str)
        self._append({
            "step_id": self._next_step_id(),
            "kind": "tool_output",
            "ts": time.time(),
            "duration_ms": self._tick(run_id),
            "result": {"output": _clip(text, self._limit)},
            "meta": {"run_id": str(run_id)},
        })

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._append({
            "step_id": self._next_step_id(),
            "kind": "tool_error",
            "ts": time.time(),
            "duration_ms": self._tick(run_id),
            "result": {"error": f"{type(error).__name__}: {error}"},
            "meta": {"run_id": str(run_id)},
        })


# ---------------------------------------------------------------------------
# 读取 / 分叉
# ---------------------------------------------------------------------------

class TrajectoryStore:
    """按会话读写轨迹。"""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else _trajectory_root()

    def _path(self, session_id: str) -> Path:
        return self.root / f"{_safe_session_id(session_id)}.jsonl"

    def exists(self, session_id: str) -> bool:
        return self._path(session_id).exists()

    def sessions(self) -> List[str]:
        return sorted(p.stem for p in self.root.glob("*.jsonl"))

    def load(self, session_id: str) -> List[TrajectoryStep]:
        """读出一条轨迹的全部步骤，坏行跳过，不影响其余。"""
        path = self._path(session_id)
        if not path.exists():
            return []
        steps: List[TrajectoryStep] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                steps.append(TrajectoryStep(
                    step_id=str(data.get("step_id", "")),
                    kind=str(data.get("kind", "")),
                    ts=float(data.get("ts") or 0.0),
                    duration_ms=int(data.get("duration_ms") or 0),
                    payload=data.get("payload") or {},
                    result=data.get("result") or {},
                    meta=data.get("meta") or {},
                ))
        steps.sort(key=lambda s: (s.ts, s.step_id))
        return steps

    def summarize(self, session_id: str) -> Dict[str, Any]:
        """给前端时间线用的概览。"""
        steps = self.load(session_id)
        if not steps:
            return {"session_id": session_id, "steps": 0}

        kinds: Dict[str, int] = {}
        for s in steps:
            kinds[s.kind] = kinds.get(s.kind, 0) + 1

        llm_out = [s for s in steps if s.kind == "llm_output"]
        return {
            "session_id": session_id,
            "steps": len(steps),
            "kinds": kinds,
            "llm_calls": len(llm_out),
            "tool_calls": kinds.get("tool_input", 0),
            "llm_total_ms": sum(s.duration_ms for s in llm_out),
            "started_at": steps[0].ts,
            "ended_at": steps[-1].ts,
            "span_ms": int((steps[-1].ts - steps[0].ts) * 1000),
        }

    def frames(self, session_id: str, *, with_payload: bool = False) -> List[Dict[str, Any]]:
        """把轨迹压成前端可以直接渲染的时间线。"""
        out = []
        for s in self.load(session_id):
            frame = {"step_id": s.step_id, "kind": s.kind, "ts": s.ts,
                     "duration_ms": s.duration_ms}
            if with_payload:
                frame["payload"] = s.payload
                frame["result"] = s.result
            elif s.kind == "llm_output":
                gens = (s.result or {}).get("generations") or [{}]
                preview = (gens[0].get("content") or gens[0].get("text") or "")
                frame["preview"] = str(preview)[:160]
            elif s.kind == "tool_input":
                frame["tool"] = (s.payload or {}).get("tool")
            out.append(frame)
        return out

    def fork(self, session_id: str, *, upto_step: Optional[str] = None,
             new_session_id: Optional[str] = None) -> str:
        """把一条轨迹的前缀复制成一条新轨迹，返回新会话 id。

        复制到 upto_step 为止（含）；不给则整条复制。分叉点之后怎么跑由调用方决定。
        """
        src = self._path(session_id)
        if not src.exists():
            raise FileNotFoundError(f"轨迹不存在：{session_id}")

        new_id = new_session_id or f"{_safe_session_id(session_id)}-fork-{uuid.uuid4().hex[:8]}"
        dst = self._path(new_id)

        kept: List[str] = []
        with open(src, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                kept.append(json.dumps(data, ensure_ascii=False))
                if upto_step is not None and str(data.get("step_id")) == str(upto_step):
                    break

        header = json.dumps({"step_id": "0", "kind": "meta", "ts": time.time(),
                             "meta": {"session_id": new_id, "fork_of": session_id,
                                      "forked_at_step": upto_step}}, ensure_ascii=False)
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(header + "\n")
            for line in kept:
                fh.write(line + "\n")
        logger.info("轨迹已分叉：%s → %s（%s 步）", session_id, new_id, len(kept))
        return new_id


# ---------------------------------------------------------------------------
# 回放
# ---------------------------------------------------------------------------

def recorded_responses(session_id: str, root: Optional[Path] = None) -> List[Any]:
    """取出一条轨迹里录制到的模型响应，按时间顺序还原成消息对象。"""
    store = TrajectoryStore(root)
    out = []
    for step in store.load(session_id):
        if step.kind != "llm_output":
            continue
        for gen in (step.result or {}).get("generations") or []:
            if "type" in gen:
                out.append(load_message(gen))
    return out


def build_replay_model(session_id: str, *, root: Optional[Path] = None, **kwargs):
    """构造一个按录制顺序返回响应的假模型，用于确定性回放。"""
    # 模块级 __getattr__ 只管 module.attr 访问，函数体里的裸名字取不到，
    # 所以这里显式构造一次
    return _replay_model_class()(responses=recorded_responses(session_id, root), **kwargs)


def _replay_model_class():
    """延迟构造 ReplayModel 类，避免没装 langchain 时连读取侧都 import 不了。"""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import Field

    class ReplayModel(BaseChatModel):
        """按录制顺序吐出响应的假模型。

        走完录制的响应后返回一句收尾话，而不是抛异常 —— 回放时更关心
        "同样的输入会不会走出同样的路"，多跑一步就崩会打断整条链路。
        """

        responses: List[Any] = Field(default_factory=list)
        cursor: int = 0
        exhausted_text: str = "（回放已到录制末尾，没有更多响应）"

        @property
        def _llm_type(self) -> str:
            return "trajectory-replay"

        def bind_tools(self, tools, **kwargs):  # noqa: D102
            # 回放不需要真的绑定工具：响应里已经带着当初的 tool_calls
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> Any:
            if self.cursor < len(self.responses):
                message = self.responses[self.cursor]
                self.cursor += 1
            else:
                message = AIMessage(content=self.exhausted_text)
            return ChatResult(generations=[ChatGeneration(message=message)])

    return ReplayModel


def __getattr__(name: str):
    """模块级惰性属性，让 ReplayModel 可以 `from trajectory import ReplayModel` 直接用。"""
    if name == "ReplayModel":
        return _replay_model_class()
    raise AttributeError(name)
