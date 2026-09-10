"""
监控上报层：工具与 Agent 的统一埋点入口。

对外 API 是 `monitor.report_tool(...)` / `report_assistant(...)` /
`report_task_result(...)` / `report_session_dir(...)`，签名和语义保持稳定。

内部把事件交给 `observability.event_bus`，由总线统一负责缓冲、落盘、扇出、指标：
内存环形缓冲支持断线重放（前端带 last_event_id 重连），JSONL 落盘供事后复盘，
订阅者扇出做 WebSocket 实时推送，指标聚合出 P50/P95 和失败率。
新增消费方只需要 `event_bus.subscribe(...)`，业务代码不用改。
"""
import asyncio
import datetime
import logging
from typing import Any, Dict, Optional

from fastapi import WebSocket

from api.context import get_thread_context
from observability.bus import event_bus
from observability.events import EventType, Span

logger = logging.getLogger(__name__)

# 尝试导入全局运行时（用于脚本模式下的流式输出）
try:
    import builtins
except ImportError:
    builtins = None


class ToolMonitor:
    """工具监控单例，工具执行过程中上报进度和状态。

    使用示例:
    from api.monitor import monitor

    def my_tool(arg1):
        monitor.report_tool("my_tool", {"arg1": arg1})
        ...
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolMonitor, cls).__new__(cls)
            cls._instance.websocket_manager = None    # 预留给 FastAPI WebSocketManager
        return cls._instance

    def set_websocket_manager(self, manager):
        """注入 FastAPI 的 WebSocket 管理器（不是新建）。

        总线绑定事件循环后需要知道往哪个连接管理器投递，这样 monitor 只依赖
        "能发送"这个能力，不依赖具体实现。
        """
        self.websocket_manager = manager
        # 把连接管理器注册成总线的订阅者
        event_bus.subscribe(manager.send_to_thread)
        logger.info("[Monitor] WebSocket 管理器已注册为事件订阅者")

    # ------------------------------------------------------------------
    def _emit(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None,
              *, duration_ms: int = 0, level: str = "info") -> None:
        """内部发送方法：构造事件并交给总线。

        方法名保留 `_emit`，既有调用点（`monitor._emit("error", ...)`）不用改。
        """
        payload = {
            "type": "monitor_event",          # 前端协议字段
            "event": event_type,
            "message": message,
            "data": data or {},
            "timestamp": datetime.datetime.now().isoformat(),
        }
        # 走总线：由它负责缓冲、落盘、扇出、指标
        event_bus.publish(event_type, message, session_id=get_thread_context() or "",
                          data=payload["data"], duration_ms=duration_ms, level=level)
        # 控制台保底输出，方便本地调试
        print(f"\n[Monitor:{event_type}] {message}")

    # ------------------------------------------------------------------
    def report_tool(self, tool_name: str, args: Dict[str, Any] = None):
        """报告工具开始执行。"""
        self._emit(EventType.TOOL_START, f"开始执行工具: {tool_name}",
                   {"tool_name": tool_name, "args": args})

    def report_tool_end(self, tool_name: str, *, duration_ms: int = 0, result_preview: str = ""):
        """报告工具执行完成（带耗时，用于性能定位）。"""
        self._emit(EventType.TOOL_END, f"工具执行完成: {tool_name}（{duration_ms}ms）",
                   {"tool_name": tool_name, "result_preview": result_preview[:300]},
                   duration_ms=duration_ms)
        event_bus.record_metric(f"tool:{tool_name}", duration_ms, ok=True)

    def report_assistant(self, assistant_name: str, args: Dict[str, Any] = None):
        """报告正在调用的子智能体进度。"""
        self._emit(EventType.ASSISTANT_CALL, f"正在调用助手: {assistant_name}",
                   {"assistant_name": assistant_name, "args": args})

    def report_task_result(self, result: str):
        """报告任务最终结果。"""
        self._emit(EventType.TASK_RESULT, "任务执行完成", {"result": result})

    def report_session_dir(self, path: str):
        """报告任务工作目录。"""
        self._emit(EventType.SESSION_CREATED, f"工作目录已创建: {path}", {"path": path})

    def report_sandbox(self, meta: dict):
        """报告沙箱就绪，前端据此显示当前会话是否处于真隔离环境。"""
        isolated = bool(meta.get("isolated"))
        self._emit(
            EventType.SANDBOX_READY,
            f"执行环境就绪：{meta.get('backend')}"
            + ("（容器隔离）" if isolated else "（进程级，非安全边界）"),
            meta, level="info" if isolated else "warn")

    def report_denied(self, kind: str, message: str, data: dict | None = None):
        """报告一次安全拒绝：权限 / 路径 / SQL 三类。"""
        mapping = {
            "permission": EventType.PERMISSION_DENIED,
            "path": EventType.PATH_DENIED,
            "sql": EventType.SQL_REJECTED,
        }
        self._emit(mapping.get(kind, EventType.ERROR), message, data or {}, level="warn")

    # ------------------------------------------------------------------
    def span(self, name: str, *, session_id: str = "", data: dict | None = None) -> Span:
        """创建一个计时 span，自动记录耗时与调用树关系。"""
        return Span(event_bus, name, session_id=session_id or (get_thread_context() or ""),
                    data=data)


# 全局单例实例
monitor = ToolMonitor()


class ConnectionManager:
    """WebSocket 连接管理：会话标识 → 连接 的映射 + 定向发送。"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        # 延迟绑定 loop，防止初始化时 loop 不一致
        self.loop = None

    def set_loop(self, loop):
        """显式设置事件循环，在 FastAPI 的 startup 事件里调用。

        不能在模块导入时取 loop：那一刻 uvicorn 的 loop 还没起来，取到的可能是
        一个马上被替换的临时 loop，之后所有跨线程投递都会投进死循环器，
        表现为消息静默丢失。
        """
        self.loop = loop
        monitor.set_websocket_manager(self)
        event_bus.bind_loop(loop)
        print(f"[Monitor] ConnectionManager manually bound to loop: {id(self.loop)}")

    async def connect(self, websocket: WebSocket, thread_id: str):
        await websocket.accept()
        print(f"存储当前会话id:{thread_id}对应的:{websocket}")
        self.active_connections[thread_id] = websocket
        print(f"Client connected: {thread_id}")

    def disconnect(self, websocket: WebSocket, thread_id: str):
        if thread_id in self.active_connections:
            del self.active_connections[thread_id]
        print(f"Client disconnected: {thread_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def send_to_thread(self, message: dict, thread_id: str):
        """定向推送：只发给这个会话对应的那一个连接。"""
        websocket = self.active_connections.get(thread_id)
        if websocket is None:
            return                                 # 会话没连或已断开，静默跳过
        try:
            await websocket.send_json(message)
        except Exception as exc:
            # 推送失败只记日志，不向上抛，避免可观测系统拖垮业务
            print(f"[Monitor] 推送失败 thread={thread_id}: {exc}")
            self.active_connections.pop(thread_id, None)

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


manager = ConnectionManager()
