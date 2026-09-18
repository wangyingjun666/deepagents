"""
长期记忆（跨会话）：把值得沉淀的信息存下来，下一轮自动注入系统提示词。

框架的 MemoryMiddleware 本身只做两件事——读文件、把内容拼进 system prompt。
「记忆能不能活过这个会话」完全取决于它背后的 backend：

    StateBackend（create_deep_agent 的默认值）  图内内存态，会话一结束就没了
    StoreBackend                              落在 LangGraph 的 BaseStore 上，真正的跨会话持久化

所以这里用 CompositeBackend 做路由：**只有 /memories/ 前缀走 StoreBackend**，
其余路径仍然是原来的 StateBackend。现有文件工具（ls / read_file / write_file / glob …）
的行为一点不变，长期记忆只是多插了一条持久化通道。

记忆范围按用户隔离，走框架原生的 context 机制（context_schema -> runtime.context）。
这里没有用 ContextVar：中间件的钩子有可能被丢到线程池里执行，ContextVar 未必跟得过去，
而 runtime.context 是跟着这一次调用显式传进来的，不依赖上下文传播。

注意：MemoryMiddleware 只挂在主 Agent 上（框架 graph.py 里子 Agent 的中间件栈没有它），
子 Agent 想共享长期记忆，需要在自己的 subagent 定义里单独加一个 MemoryMiddleware。
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend

logger = logging.getLogger(__name__)

#: 长期记忆的虚拟目录前缀，命中这个前缀的路径才会被持久化
MEMORY_DIR = "/memories"

#: 记忆文件（遵循 AGENTS.md 约定，MemoryMiddleware 就是按这个约定读的）
MEMORY_FILE = f"{MEMORY_DIR}/AGENTS.md"

#: 交给 create_deep_agent 的 memory= 参数
MEMORY_SOURCES = [MEMORY_FILE]

#: 命名空间组件允许的字符。
#: 注意这里比框架 backends/store.py 里的校验更严：那边的正则是允许「.」的，
#: 但真正落库的 LangGraph BaseStore 不收（Namespace labels cannot contain periods），
#: 照抄它那份会写出一个到运行期才炸的坑，所以这里直接把点也剔掉。
_NAMESPACE_SAFE_RE = re.compile(r"[^A-Za-z0-9\-_@+:~]")


@dataclass
class MemoryContext:
    """跟着单次调用传进图的上下文，目前只用来决定记忆的命名空间。"""

    user_id: str = "anonymous"


# 首次启动时写入的记忆骨架。给模型一个「该记什么」的范例，
# 否则它会倾向于把整段对话原样抄进来，记忆很快就会被废话撑爆。
MEMORY_SEED = """# 长期记忆

> 这个文件是跨会话保留的。只记「下次还用得上」的结论，不要抄原始对话。

## 用户偏好
- （例：回答先给结论再给依据；表格优先用 Markdown）

## 长期有效的事实
- （例：常用设备型号 HAK 180；默认知识库是 company_db）

## 已确认的结论
- （例：XX 报表的口径以财务口径为准，不是运营口径）

## 待跟进
- （例：用户还在等 XX 的报价，下次问起要接着上次的进度）
"""


def _sanitize_component(value: str, default: str) -> str:
    """把任意字符串洗成合法的命名空间组件，洗空了就退回默认值。

    用户标识是外部输入，直接当命名空间用的话，一个带「/」或「.」的 user_id
    就能让 StoreBackend 在写记忆时抛 ValueError，把整个请求打挂。
    """
    cleaned = _NAMESPACE_SAFE_RE.sub("_", str(value or ""))
    cleaned = re.sub(r"_{2,}", "_", cleaned).strip("_")
    return cleaned[:64] or default


def memory_namespace(context: Any = None) -> tuple[str, ...]:
    """解析记忆命名空间：按用户隔离，用户维拿不到就用公共空间。

    不同用户之间互不可见；user_id 为 anonymous / 缺失时落到公共空间，
    避免把一堆匿名会话混在各自的空记忆里。
    """
    user_id = getattr(context, "user_id", None)
    if not user_id or str(user_id) == "anonymous":
        return ("memories", "shared")
    return ("memories", _sanitize_component(str(user_id), "shared"))


def build_backend(runtime: Any) -> CompositeBackend:
    """backend 工厂：/memories/ 走持久 store，其余路径维持原有内存态行为。

    create_deep_agent 的 backend 参数同时接受实例和工厂（BackendFactory = (runtime) -> backend），
    StoreBackend / StateBackend 都需要 runtime，所以这里必须写成工厂，不能传实例。
    """
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={
            MEMORY_DIR + "/": StoreBackend(
                runtime,
                namespace=lambda _ctx: memory_namespace(getattr(runtime, "context", None)),
            )
        },
    )


def _resolve_db_path() -> Path:
    """记忆库文件路径，默认落在 logs/memory/ 下（logs/ 已在 .gitignore 里）。"""
    raw = os.getenv("MEMORY_DB", "logs/memory/memory.sqlite")
    path = Path(raw)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_memory_store():
    """构造长期记忆的 store（SQLite 版 LangGraph BaseStore）。

    create_deep_agent 的 store 参数要求请求级常驻，所以不能用
    `SqliteStore.from_conn_string(...)` 那个 contextmanager 版本——它出了 with 就关连接。
    这里自己开连接并保持打开：

    - check_same_thread=False：异步图会把同步调用丢进线程池，连接会跨线程使用；
    - isolation_level=None：SqliteStore 的迁移脚本自己管事务，连接必须处于 autocommit，
      否则第二步 BEGIN 会撞上「cannot start a transaction within a transaction」。
    """
    from langgraph.store.sqlite import SqliteStore

    db_path = _resolve_db_path()
    conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    # 建表 + 跑迁移，SqliteStore 不会自动做，必须显式调一次
    store = SqliteStore(conn)
    store.setup()
    logger.info("长期记忆库就绪：%s", db_path)
    return store


class _StoreRuntime:
    """给 StoreBackend 用的最小 runtime，只在图外读写记忆时用得上。

    StoreBackend 实际只依赖 runtime.store（真正读写）和 runtime.state/config（解析命名空间），
    给它一份静态的就够了，不需要真的把图跑起来。
    """

    def __init__(self, store, user_id: str = "anonymous") -> None:
        self.store = store
        self.state: dict = {}
        self.config: dict = {"metadata": {}}
        self.context = MemoryContext(user_id=user_id)


def ensure_memory_seed(store=None, *, user_id: str = "anonymous", force: bool = False) -> bool:
    """确保该用户的记忆文件存在，不存在就写入骨架。

    MemoryMiddleware 对「文件不存在」是静默跳过的，不报错也不提示，
    装完看不到任何效果很容易以为功能没生效，所以这里主动种一份初始内容。

    记忆是按用户分命名空间的，所以这个种子也得按用户种：只种 shared 的话，
    真实用户第一次进来命名空间是空的，模型拿不到「该记什么」的示范。

    :param user_id: 记忆归属的用户，决定写进哪个命名空间
    :param force: True 则无视已有内容强行覆盖（正常流程不要用）
    :return: True 表示这次真的写入了种子；False 表示已有内容、没动它
    """
    store = store or get_memory_store()
    # 必须走 CompositeBackend，不能直接拿 StoreBackend 写：
    # 前缀 /memories/ 是 CompositeBackend 负责剥掉的，直接写会把 /memories 当成 key 的一部分，
    # 存进去的路径和 MemoryMiddleware 读的路径对不上，表现为「种子写成功了但读出来是空的」。
    backend = build_backend(_StoreRuntime(store, user_id=user_id))

    responses = backend.download_files([MEMORY_FILE])
    existing = responses[0] if responses else None
    if existing is not None and existing.error is None and existing.content and not force:
        return False

    result = backend.write(MEMORY_FILE, MEMORY_SEED)
    if getattr(result, "error", None):
        logger.warning("写入记忆种子失败：%s", result.error)
        return False
    logger.info("已写入长期记忆种子：%s", MEMORY_FILE)
    return True


_memory_store = None


def get_memory_store():
    """惰性获取记忆 store 单例。"""
    global _memory_store
    if _memory_store is None:
        _memory_store = build_memory_store()
    return _memory_store
