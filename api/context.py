"""
会话上下文：把"当前是谁在执行"绑定到协程上。

四个变量：
* `session_dir` —— 当前会话工作目录（宿主视角，仅用于展示与只读映射）
* `thread_id`   —— 当前会话标识，用于把事件路由到对应的 WebSocket
* `sandbox`     —— 当前会话的隔离执行环境句柄（文件读写与 PDF 渲染都走它）
* `principal`   —— 当前会话的安全主体（用户 / 角色 / 能力集合），见 security/permissions.py
"""
from contextvars import ContextVar, Token
from typing import Optional

# =================================================================================================
# ContextVar：协程级隔离
# =================================================================================================
# asyncio 里并发请求跑在同一个线程上，所以：
#   - 全局变量会被并发请求互相覆盖（A 的文件写进 B 的目录）；
#   - threading.local 按线程隔离，同一线程内失效；
#   - ContextVar 按 asyncio Task 隔离，同一个请求链路（含任意深度的调用）里
#     get() 拿到的都是本请求的值。
# 隔离单元是协程，不是线程也不是进程。
# =================================================================================================


# ContextVar 的变量名只是标识符，值存在当前 Context 环境里。

# 当前会话的文件该落在哪。工具函数在任意深度都能直接取到，不必层层传参。
_session_dir_ctx: ContextVar[Optional[str]] = ContextVar("session_dir", default=None)

# 当前是谁在执行任务。Agent 打日志或经 WebSocket 发消息时据此路由到正确的会话。
_thread_id_ctx: ContextVar[Optional[str]] = ContextVar("thread_id", default=None)

# 当前会话的隔离执行环境。工具不直接写宿主磁盘，而是把操作交给沙箱执行。
_sandbox_ctx: ContextVar[Optional[object]] = ContextVar("sandbox", default=None)


def set_session_context(path: str) -> Token:
    """
    设置当前请求链路的会话目录。
    通常在 Agent 开始执行任务前调用。

    Returns:
        Token: 返回一个 Token 对象，后续可用它来恢复(reset)变量状态。
    """
    return _session_dir_ctx.set(path)


def get_session_context() -> Optional[str]:
    """
    获取当前请求链路的会话目录。
    可以在任何深层调用的工具函数中直接使用，无需层层传递参数。
    """
    return _session_dir_ctx.get()


def set_thread_context(thread_id: str) -> Token:
    """
    设置当前请求链路的 Thread ID。
    """
    return _thread_id_ctx.set(thread_id)


def get_thread_context() -> Optional[str]:
    """
    获取当前请求链路的 Thread ID。
    """
    return _thread_id_ctx.get()


def reset_session_context(session_token, thread_token=None):
    """
    清理/重置上下文，通常在请求处理结束 (finally 块) 中调用。

    用 token 回滚而不是 `set(None)`：token 是栈式的，嵌套场景下不会破坏外层值；
    `set(None)` 是无条件覆盖，会把外层也抹掉。
    """
    if session_token is not None:
        _session_dir_ctx.reset(session_token)
    if thread_token is not None:
        _thread_id_ctx.reset(thread_token)


# 沙箱上下文：副作用只在隔离环境里发生
def set_sandbox_context(sandbox):
    return _sandbox_ctx.set(sandbox)


def get_sandbox_context():
    return _sandbox_ctx.get()


def reset_sandbox_context(token) -> None:
    if token is not None:
        _sandbox_ctx.reset(token)


def require_sandbox():
    """取当前会话沙箱。取不到直接抛异常，不降级成直接写宿主磁盘。

    隔离环境不可用时正确的行为是失败，而不是绕过隔离。
    """
    sb = _sandbox_ctx.get()
    if sb is None:
        raise RuntimeError("当前上下文没有沙箱，拒绝执行文件操作（不会降级为直接访问宿主）")
    return sb


# 安全主体：和沙箱一样按协程隔离
def set_security_context(principal):
    from security.permissions import set_principal
    return set_principal(principal)


def get_security_context():
    from security.permissions import current_principal
    return current_principal()


def reset_security_context(token) -> None:
    from security.permissions import reset_principal
    reset_principal(token)


def describe() -> dict:
    """当前协程的上下文快照（调试和事件 payload 用）。"""
    principal = get_security_context()
    sb = _sandbox_ctx.get()
    return {
        "session_dir": _session_dir_ctx.get(),
        "thread_id": _thread_id_ctx.get(),
        "sandbox": getattr(sb, "name", None),
        "sandbox_isolated": bool(getattr(sb, "meta", {}).get("isolated")),
        "principal": principal.to_dict() if principal else None,
    }
