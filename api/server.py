"""
FastAPI 服务层：任务入口、文件接口、WebSocket 实时通道、审批与观测接口。

接口清单
--------
* `POST /api/task`                  —— 提交任务，后台跑 Agent，立即返回 thread_id
* `POST /api/upload`                —— 上传文件到 `updated/session_{thread_id}`
* `GET  /api/files`                 —— 列出 output 目录下的生成文件
* `GET  /api/download`              —— 下载 output 目录内的文件
* `GET  /api/trace/{session_id}`    —— 拉取会话的结构化事件（支持增量）
* `GET  /api/trace/{session_id}/spans` —— 事件还原成调用树
* `GET  /api/trajectory`            —— 已落盘的轨迹列表
* `GET  /api/trajectory/{sid}`      —— 会话轨迹（模型看到什么、返回什么）
* `POST /api/trajectory/{sid}/fork` —— 轨迹分叉，用于换参数重跑并对比
* `GET  /api/metrics`               —— P50/P95、失败率、限流与熔断状态
* `GET  /api/audit`                 —— 审计日志查询（哈希链）
* `GET  /api/system`                —— 沙箱后端、隔离自检、安全策略总览
* `GET  /api/approvals`             —— 当前待审批列表
* `POST /api/approvals/{approval_id}` —— 人工审批决议
* `GET  /api/approvals/history`     —— 审批历史
* `WS   /ws/{thread_id}`            —— 实时事件通道，支持 `?after_event_id=N` 重连补发

`/api/files` 与 `/api/download` 统一走 `security.path_guard` 做拒绝型校验，
含后缀白名单与符号链接检查。任务入口接收 `user_id` / `role` 建立安全主体并做能力路由。
"""
import asyncio
import shutil
import sys
import uuid
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import Body, FastAPI, File, Form, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import agent runner and monitor
# 注意：agent.main_agent 导入时会初始化 main_agent，耗时约几秒
from agent.main_agent import run_deep_agent
from api.monitor import manager, monitor
from concurrency.db_pool import is_readonly_account_configured
from concurrency.limiter import all_stats as limiter_stats
from concurrency.retry import breakers_snapshot
from observability.bus import event_bus
from observability.metrics import metrics
from observability.store import event_store
from observability.trajectory import TrajectoryStore
from sandbox import sandbox_manager
from security.approval import approval_center
from security.audit import audit
from security.path_guard import PathMode, PathSecurityError, guard_path
from security.permissions import describe_routing, security_mode

app = FastAPI(title="DeepAgents API")

# 挂载输出目录，前端可访问生成的静态文件
output_dir = project_root / "output"
output_dir.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(output_dir)), name="outputs")

# 上传目录
updated_dir = project_root / "updated"
updated_dir.mkdir(exist_ok=True)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskRequest(BaseModel):
    query: str
    thread_id: str = None
    user_id: str = "anonymous"
    role: Optional[str] = None


class ApprovalDecision(BaseModel):
    decision: str                     # approve / reject
    by: str = "user"
    edited_args: Optional[dict] = None


@app.on_event("startup")
async def startup_event():
    """启动时把当前事件循环绑定到 WebSocket 管理器和事件总线。

    后台线程靠 run_coroutine_threadsafe 投递消息，必须拿到正确的 loop。
    """
    loop = asyncio.get_running_loop()
    manager.set_loop(loop)             # 内部会一并 event_bus.bind_loop(loop)
    probe = sandbox_manager.probe()
    print(f"[Server] WebSocket Manager bound to loop: {id(loop)}")
    print(f"[Server] 沙箱后端：{probe.get('effective')}  "
          f"（配置={probe.get('configured')}，Docker可用={probe.get('docker_available')}）")
    if probe.get("effective") != "docker":
        print(f"[Server] 注意：当前不是容器隔离，原因：{probe.get('docker_detail', '')[:200]}")


@app.on_event("shutdown")
async def shutdown_event():
    """退出时清理沙箱（避免留下孤儿容器/进程），并关闭事件落盘句柄。"""
    await sandbox_manager.shutdown()
    event_store.close()


@app.post("/api/task")
async def run_task(request: TaskRequest):
    thread_id = request.thread_id or str(uuid.uuid4())

    # 后台执行，不阻塞请求；实时推送由 main_agent 负责，
    # 并发上限由 session_limiter 控制（超限排队而非直接失败）
    asyncio.create_task(run_deep_agent(request.query, thread_id,
                                       user_id=request.user_id, role=request.role))

    return {"status": "started", "thread_id": thread_id}


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...), thread_id: str = Form(...)):
    """文件上传：存到 `updated/session_{thread_id}`，供 Agent 在后续任务中读取。"""
    from security.path_guard import sanitize_filename

    target_dir = updated_dir / f"session_{thread_id}"
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    rejected = []
    for file in files:
        # 文件名是不可信输入：收敛成安全的单层文件名，
        # 否则 "../../x.py" 这类名字会写到目录外面
        safe_name = sanitize_filename(file.filename or "unnamed")
        file_path = target_dir / safe_name
        try:
            if not file_path.resolve().is_relative_to(target_dir.resolve()):
                rejected.append(file.filename)
                continue
        except OSError:
            rejected.append(file.filename)
            continue
        # copyfileobj 流式复制，避免把大文件一次性读进内存
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(safe_name)

    audit.record(action="file_upload", decision="allow" if saved_files else "deny",
                 session_id=thread_id, target=str(target_dir),
                 reason=f"accepted={saved_files} rejected={rejected}")
    return {"status": "uploaded", "files": saved_files, "rejected": rejected}


def _guard_output_path(path: str):
    """把用户传入的路径收敛到 output 目录内（拒绝型校验）。

    显式传 `allowed_suffixes=()` 关掉后缀白名单：这两个接口是只读的，路径已经
    限制在 output 目录内，再限制后缀只会误伤生成的图片、zip 附件。
    """
    return guard_path(path, output_dir, mode=PathMode.READ,
                      allowed_suffixes=(), reject_symlink=False)


@app.get("/api/download")
async def download_file(path: str):
    """文件下载。安全检查走 path_guard 做拒绝型校验，越界直接拒绝，不做尽力解析。"""
    try:
        abs_path = _guard_output_path(path)
    except PathSecurityError as exc:
        audit.record(action="file_download", decision="deny", target=path, reason=exc.reason)
        return {"error": f"拒绝访问：{exc.reason}"}
    except Exception:
        return {"error": "无效的路径参数"}

    if not abs_path.exists():
        return {"error": "文件不存在"}

    audit.record(action="file_download", decision="allow", target=str(abs_path))
    return FileResponse(abs_path, filename=abs_path.name)


@app.get("/api/files")
async def list_files(path: str):
    """文件列表：列出目录下的生成文件及其元数据（大小、时间）。路径经 path_guard 校验。"""
    print(f"[DEBUG] 请求文件列表: {path}")

    try:
        abs_path = _guard_output_path(path)
    except PathSecurityError as exc:
        print(f"[ERROR] 拒绝访问: {exc}")
        audit.record(action="file_list", decision="deny", target=path, reason=exc.reason)
        return {"error": f"拒绝访问: {exc.reason}"}
    except Exception as exc:
        print(f"[ERROR] 路径解析失败: {exc}")
        return {"error": f"路径无效: {exc}"}

    if not abs_path.exists():
        return {"error": "目录不存在"}

    files = []
    try:
        for file_path in abs_path.rglob("*"):
            if file_path.is_file():
                # 符号链接指向外部时跳过，避免暴露 output 之外的文件
                try:
                    if not file_path.resolve().is_relative_to(output_dir.resolve()):
                        continue
                except OSError:
                    continue
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "type": "file",
                    "path": str(file_path),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                })
    except Exception as e:
        print(f"[ERROR] 遍历文件失败: {e}")
        return {"error": str(e)}

    files.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    print(f"[DEBUG] 找到 {len(files)} 个文件")
    return {"files": files}


# ==========================================================================
# 可观测接口
# ==========================================================================
@app.get("/api/trace/{session_id}")
async def get_trace(session_id: str, after_event_id: int = Query(0),
                    limit: int = Query(1000)):
    """拉取会话的结构化事件（增量）。前端重连时用它补齐断线期间的事件。"""
    events = event_bus.replay(session_id, after_event_id=after_event_id, limit=limit)
    return {"session_id": session_id, "count": len(events),
            "last_event_id": event_bus.last_event_id(session_id), "events": events}


@app.get("/api/trace/{session_id}/spans")
async def get_spans(session_id: str):
    """把事件还原成调用树，用于回答"这个子 Agent 花了多久"这类问题。"""
    events = event_bus.replay(session_id, limit=5000)
    by_id = {e.get("span_id"): e for e in events}
    roots = []
    for e in events:
        parent = e.get("parent_span_id")
        if not parent or parent not in by_id:
            roots.append({"event": e, "children": []})
    # 只做一层聚合，够用于上面的问题
    return {"session_id": session_id, "total": len(events),
            "roots": len(roots), "events": events}


@app.get("/api/trajectory")
async def list_trajectories():
    """列出已经落盘的轨迹会话。"""
    sessions = TrajectoryStore().sessions()
    return {"count": len(sessions), "sessions": sessions}


@app.get("/api/trajectory/{session_id}")
async def get_trajectory(session_id: str, with_payload: bool = Query(False)):
    """拉取会话的完整轨迹。

    with_payload=true 时连模型收到的完整输入一起返回，用于回答
    "它当时到底看到了什么"；默认只给时间线骨架，体积小得多。
    """
    store = TrajectoryStore()
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail=f"没有这条轨迹：{session_id}")
    return {"session_id": session_id,
            "summary": store.summarize(session_id),
            "frames": store.frames(session_id, with_payload=with_payload)}


@app.post("/api/trajectory/{session_id}/fork")
async def fork_trajectory(session_id: str, upto_step: Optional[str] = Body(None, embed=True)):
    """把一条轨迹的前缀分叉成新轨迹，在分叉点之后换参数重跑并对比。"""
    try:
        new_id = TrajectoryStore().fork(session_id, upto_step=upto_step)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"session_id": session_id, "forked": new_id, "upto_step": upto_step}


@app.get("/api/metrics")
async def get_metrics():
    """性能指标：按操作聚合的调用次数、失败率、P50/P95。"""
    return {
        "operations": metrics.snapshot(),
        "limiters": limiter_stats(),
        "breakers": breakers_snapshot(),
        "bus": event_bus.stats(),
    }


@app.get("/api/audit")
async def get_audit(session_id: str = "", decision: str = "", limit: int = Query(200)):
    """审计日志查询（哈希链）。"""
    ok, detail = audit.verify_chain()
    return {"chain_ok": ok, "chain_detail": detail,
            "records": audit.query(session_id=session_id or None,
                                   decision=decision or None, limit=limit)}


@app.get("/api/system")
async def get_system():
    """系统状态总览：沙箱隔离能力、安全策略、依赖配置。"""
    probe = sandbox_manager.probe_result or sandbox_manager.probe()
    return {
        "sandbox": {**probe, "runtime": sandbox_manager.stats()},
        "security": {
            "mode": security_mode(),
            "role_routing": describe_routing(),
            "database_readonly_account": is_readonly_account_configured(),
        },
        "observability": {"bus": event_bus.stats(), "persistent_trace": True,
                          "trace_dir": str(event_store.root)},
        "connections": manager.connection_count,
    }


# ==========================================================================
# 人工审批接口（人在回路）
# ==========================================================================
@app.get("/api/approvals")
async def list_approvals(session_id: str = ""):
    """当前待审批的操作。"""
    return {"pending": approval_center.list_pending(session_id), "count": approval_center.pending_count}


@app.post("/api/approvals/{approval_id}")
async def decide_approval(approval_id: str, body: ApprovalDecision):
    """对一次待审批操作做出决议。超时按拒绝处理（fail-closed）。

    审批如果默认放行，攻击者只要让请求超时就能绕过。
    """
    if body.decision not in ("approve", "reject", "edit"):
        return {"ok": False, "error": "decision 必须是 approve / reject / edit"}
    ok = await approval_center.resolve(approval_id, body.decision, by=body.by,
                                       edited_args=body.edited_args)
    return {"ok": ok, "approval_id": approval_id, "decision": body.decision}


@app.get("/api/approvals/history")
async def approval_history(limit: int = Query(100)):
    return {"history": approval_center.history(limit)}


# 浏览器请求 ws://localhost:8000/ws/thread_123 时，FastAPI 在事件循环里实例化
# WebSocket 对象（封装 TCP 连接、握手信息、send_text/receive_text），
# 按路由匹配注入到下面的处理函数。
@app.websocket("/ws/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str,
                             after_event_id: int = Query(0)):
    """WebSocket 实时通道。

    1. 接受连接并注册到 `monitor.manager`，按 `thread_id` 做会话级消息隔离。
    2. 带了 `?after_event_id=N` 时，从事件总线缓冲/落盘文件补发断线期间的事件。
    3. 进入消息循环，收到前端心跳就回 pong 并带上当前游标 `last_event_id`。
    """
    await manager.connect(websocket, thread_id)

    try:
        # 重连补发：补齐 after_event_id 之后的事件
        if after_event_id or True:
            missed = event_bus.replay(thread_id, after_event_id=after_event_id, limit=500)
            if missed:
                await websocket.send_json({
                    "type": "monitor_event",
                    "event": "replay",
                    "message": f"补发 {len(missed)} 条断线期间的事件",
                    "data": {"count": len(missed), "events": missed,
                             "last_event_id": event_bus.last_event_id(thread_id)},
                })

        while True:
            # 监听前端消息 (通常是 ping 心跳)
            data = await websocket.receive_text()
            # 回复 pong
            await websocket.send_json({
                "type": "pong",
                "message": f"服务端已收到: {data}",
                "last_event_id": event_bus.last_event_id(thread_id),
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket, thread_id)
        print(f"[WebSocket] 客户端已断开: {thread_id}")

    except Exception as e:
        print(f"[WebSocket] 连接异常: {e}")
        manager.disconnect(websocket, thread_id)


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
