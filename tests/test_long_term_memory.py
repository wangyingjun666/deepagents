# -*- coding: utf-8 -*-
"""
长期记忆的单元测试（不依赖 Docker、不依赖网络、不依赖真实大模型）。

覆盖范围
--------
1. 跨会话持久：写进 /memories/ 的内容，换一个连接、换一个 store 还能读到
2. 路由隔离：非 /memories/ 前缀的路径仍然是原来的内存态行为，不会意外落盘
3. 提示词注入：MemoryMiddleware 确实把记忆文件读进了 memory_contents
4. 按用户隔离：A 用户的记忆 B 用户看不到，anonymous 落到公共命名空间
5. 种子写入：首次写入骨架，第二次不覆盖已有内容
6. 命名空间清洗：非法字符的 user_id 不会把 StoreBackend 弄崩

运行：
    python tests/test_long_term_memory.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 记忆库落到临时目录，别污染仓库里的 logs/
_TMP_DIR = tempfile.mkdtemp(prefix="memory_test_")
os.environ["MEMORY_DB"] = str(Path(_TMP_DIR) / "memory.sqlite")

from agent.memory import (  # noqa: E402
    MEMORY_FILE,
    MEMORY_SEED,
    MEMORY_SOURCES,
    MemoryContext,
    build_backend,
    build_memory_store,
    ensure_memory_seed,
    memory_namespace,
)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(f"{name} :: {detail}")
        print(f"  [FAIL] {name} :: {detail}")


class FakeRuntime:
    """够 Middleware / Backend 用的最小 runtime。

    MemoryMiddleware 会用到 runtime.stream_writer；StoreBackend 用到 runtime.store / state / config。
    """

    def __init__(self, store, user_id: str = "anonymous") -> None:
        self.store = store
        self.state: dict = {}
        self.config: dict = {"metadata": {}}
        self.context = MemoryContext(user_id=user_id)
        self.stream_writer = lambda *a, **kw: None


def fresh_store(tag: str):
    """每次调用都开一个新连接 + 新 store，同一个 tag 指向同一个库文件。

    每个用例用自己的 tag：共用一个库的话，前一个用例写进去的记忆会被后一个用例读到，
    看起来就像隔离失效了（这个坑第一版测试就踩过）。
    """
    os.environ["MEMORY_DB"] = str(Path(_TMP_DIR) / f"{tag}.sqlite")
    return build_memory_store()


# --------------------------------------------------------------------------
def test_cross_session_persistence() -> None:
    print("\n=== 1. 跨会话持久 ===")

    writer = fresh_store("persist")
    rt = FakeRuntime(writer, user_id="u-alice")
    build_backend(rt).write(MEMORY_FILE, "用户偏好：先给结论再给依据。")

    # 换一个连接、换一个 store 实例 —— 等价于服务重启后的下一个会话
    reader = fresh_store("persist")
    rt2 = FakeRuntime(reader, user_id="u-alice")
    resp = build_backend(rt2).download_files([MEMORY_FILE])[0]

    check("重启后仍能读到记忆", resp.error is None and bool(resp.content),
          f"error={resp.error}")
    check("记忆内容一致", (resp.content or b"").decode("utf-8") == "用户偏好：先给结论再给依据。",
          repr(resp.content))

    # /memories/ 下的其他文件也该持久，不只是那一个固定文件名
    build_backend(rt).write("/memories/notes.md", "随手记：报表口径以财务为准")
    resp2 = build_backend(FakeRuntime(reader, user_id="u-alice")).download_files(["/memories/notes.md"])[0]
    check("同前缀的其他文件也持久", resp2.error is None and bool(resp2.content), f"error={resp2.error}")


def test_non_memory_paths_stay_ephemeral() -> None:
    print("\n=== 2. 路由隔离：非记忆路径不落盘 ===")

    writer = fresh_store("ephemeral")
    rt = FakeRuntime(writer, user_id="u-alice")
    build_backend(rt).write("/temp.txt", "这是临时文件")

    reader = fresh_store("ephemeral")
    resp = build_backend(FakeRuntime(reader, user_id="u-alice")).download_files(["/temp.txt"])[0]

    check("临时文件不会跨会话保留", resp.error is not None or not resp.content,
          f"error={resp.error} content={resp.content}")


def test_middleware_injects_memory() -> None:
    print("\n=== 3. 记忆被注入提示词 ===")

    from deepagents.middleware.memory import MemoryMiddleware

    store = fresh_store("middleware")
    rt = FakeRuntime(store, user_id="u-bob")
    build_backend(rt).write(MEMORY_FILE, "长期事实：常用设备是 HAK 180。")

    middleware = MemoryMiddleware(backend=build_backend, sources=MEMORY_SOURCES)
    out = middleware.before_agent({"messages": []}, rt, {}) or {}

    contents = out.get("memory_contents") or {}
    check("memory_contents 里有记忆文件", MEMORY_FILE in contents, str(list(contents)))
    check("记忆正文被读出来", "HAK 180" in str(contents.get(MEMORY_FILE, "")),
          str(contents)[:200])

    # 文件不存在时不能报错（MemoryMiddleware 对 file_not_found 是静默跳过的）
    out2 = middleware.before_agent({"messages": []}, FakeRuntime(fresh_store("middleware"), user_id="u-nobody"), {}) or {}
    check("无记忆时不报错", isinstance(out2, dict))


def test_per_user_isolation() -> None:
    print("\n=== 4. 按用户隔离 ===")

    store = fresh_store("isolation")
    build_backend(FakeRuntime(store, user_id="u-alice")).write(MEMORY_FILE, "alice 的私人记忆")

    bob = build_backend(FakeRuntime(store, user_id="u-bob")).download_files([MEMORY_FILE])[0]
    check("B 用户看不到 A 用户的记忆", bob.error is not None or not bob.content,
          f"error={bob.error}")

    alice = build_backend(FakeRuntime(store, user_id="u-alice")).download_files([MEMORY_FILE])[0]
    check("A 用户自己能读到", alice.error is None and bool(alice.content), f"error={alice.error}")

    check("命名空间按用户切分",
          memory_namespace(MemoryContext(user_id="u-alice")) == ("memories", "u-alice"),
          str(memory_namespace(MemoryContext(user_id="u-alice"))))
    check("anonymous 落公共空间",
          memory_namespace(MemoryContext(user_id="anonymous")) == ("memories", "shared"),
          str(memory_namespace(MemoryContext(user_id="anonymous"))))
    check("user_id 缺失也落公共空间",
          memory_namespace(None) == ("memories", "shared"), str(memory_namespace(None)))


def test_namespace_sanitizing() -> None:
    print("\n=== 5. 命名空间清洗 ===")

    # StoreBackend 对命名空间组件有字符校验，非法字符会被它直接抛 ValueError，
    # 所以必须在传进去之前洗掉，否则一个奇怪的用户名就能把整个请求打挂
    weird = memory_namespace(MemoryContext(user_id="张三/../$(rm -rf)"))
    # 点号是重点：框架自己的校验放行「.」，但底层 BaseStore 不收，只按框架那份抄会在运行期炸
    check("非法字符被替换",
          all("/" not in p and "$" not in p and "." not in p for p in weird), str(weird))

    store = fresh_store("sanitize")
    try:
        backend = build_backend(FakeRuntime(store, user_id="张三/../$(rm -rf)"))
        backend.write(MEMORY_FILE, "脏 user_id 也要能写")
        ok = True
    except Exception as exc:  # noqa: BLE001
        ok = False
        print("     异常：", exc)
    check("脏 user_id 不会写崩", ok)

    empty = memory_namespace(MemoryContext(user_id="!!!"))
    check("洗空了退回 shared", empty == ("memories", "shared"), str(empty))


def test_seed() -> None:
    print("\n=== 6. 记忆种子 ===")

    store = fresh_store("seed")

    first = ensure_memory_seed(store)
    check("首次写入种子", first is True)

    rt = FakeRuntime(store, user_id="anonymous")
    resp = build_backend(rt).download_files([MEMORY_FILE])[0]
    check("种子内容可读", resp.error is None and "长期记忆" in (resp.content or b"").decode("utf-8"),
          repr(resp.content)[:120])

    # 模拟模型/用户往记忆里补了一条（更新走 edit，不是 write）
    backend = build_backend(rt)
    edited = backend.edit(MEMORY_FILE, "## 待跟进", "## 待跟进\n- 用户还在等 XX 的报价")
    check("记忆可以被编辑追加", edited.error is None, str(edited.error))

    # 第二次不能被骨架覆盖，否则用户攒的记忆每重启一次就没了
    second = ensure_memory_seed(store)
    after = backend.download_files([MEMORY_FILE])[0]
    check("已有内容时不覆盖", second is False)
    check("已有内容保留", "用户还在等 XX 的报价" in (after.content or b"").decode("utf-8"),
          repr(after.content))


def test_existing_memory_is_not_clobbered_by_write() -> None:
    print("\n=== 7. 记忆文件不会被 write 覆盖 ===")

    store = fresh_store("clobber")
    rt = FakeRuntime(store, user_id="u-carol")
    backend = build_backend(rt)
    backend.write(MEMORY_FILE, "原始记忆")

    # write 对已存在的文件是拒绝的（框架约定：先 read 再 edit），
    # 这条保证了模型手里一个走神的 write 不会把攒了半年的记忆一把抹掉
    again = backend.write(MEMORY_FILE, "覆盖掉")
    check("write 拒绝覆盖已有记忆", again.error is not None, str(again.error))
    check("原内容没被动", backend.download_files([MEMORY_FILE])[0].content == "原始记忆".encode("utf-8"),
          repr(backend.download_files([MEMORY_FILE])[0].content))


def test_end_to_end_injection_and_isolation() -> None:
    """真起一张图跑一遍：记忆要进到模型真正收到的 system 消息里，且不串用户。

    前面几个用例是拿 MemoryMiddleware / Backend 单测的，
    这个用例补上「整条链路真的通」——中间任何一处命名空间或路径对不上，这里就会露馅。
    """
    print("\n=== 8. 端到端：记忆注入 + 用户隔离 ===")

    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from deepagents import create_deep_agent

    seen: list[str] = []

    class _RecordingModel(GenericFakeChatModel):
        """记录模型真正收到的消息（含被中间件改写过的 system 提示词）。"""

        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN003
            seen.append(" ".join(str(getattr(m, "content", "")) for m in messages))
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    store = fresh_store("e2e")
    for user, secret in (("u-x", "紫色犀牛"), ("u-y", "蓝色鲸鱼")):
        ensure_memory_seed(store, user_id=user)
        build_backend(FakeRuntime(store, user_id=user)).edit(
            MEMORY_FILE, "## 待跟进", f"## 待跟进\n- 暗号：{secret}"
        )

    # 一个用户一条模型消息就够了：回复里没有工具调用，图跑一轮就结束
    model = _RecordingModel(messages=iter([AIMessage(content="好的")] * 4))
    agent = create_deep_agent(
        model=model,
        memory=MEMORY_SOURCES,
        backend=build_backend,
        store=store,
        context_schema=MemoryContext,
    )

    import asyncio

    for user, mine, theirs in (("u-x", "紫色犀牛", "蓝色鲸鱼"), ("u-y", "蓝色鲸鱼", "紫色犀牛")):
        seen.clear()
        asyncio.run(
            agent.ainvoke(
                {"messages": [{"role": "user", "content": "你好"}]},
                config={"configurable": {"thread_id": f"thread-{user}"}},
                context=MemoryContext(user_id=user),
            )
        )
        sent = seen[0] if seen else ""
        check(f"{user} 的记忆进了 system 提示词", mine in sent, f"len={len(sent)}")
        check(f"{user} 看不到另一个用户的记忆", theirs not in sent, "发生串号")
        check(f"{user} 记忆带 agent_memory 标记", "agent_memory" in sent, "缺少标记")


def test_seed_template_has_sections() -> None:
    print("\n=== 9. 种子模板结构 ===")
    for section in ("用户偏好", "长期有效的事实", "已确认的结论", "待跟进"):
        check(f"种子含「{section}」", section in MEMORY_SEED)


# --------------------------------------------------------------------------
def main() -> int:
    test_cross_session_persistence()
    test_non_memory_paths_stay_ephemeral()
    test_middleware_injects_memory()
    test_per_user_isolation()
    test_namespace_sanitizing()
    test_seed()
    test_existing_memory_is_not_clobbered_by_write()
    test_end_to_end_injection_and_isolation()
    test_seed_template_has_sections()
    print(f"\n=== 单元测试结果：{len(PASS)} 通过 / {len(FAIL)} 失败 ===")
    for f in FAIL:
        print("   - 失败：", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
