"""
Docker 沙箱后端 —— 每个会话一个容器，namespace 级隔离。

容器参数（在 `start()` 里显式声明）

| 参数                                        | 作用                                      |
|---------------------------------------------|-------------------------------------------|
| `--network none`                            | 容器内代码断网；检索走宿主侧工具           |
| `--read-only` 根文件系统                    | 防止篡改镜像内文件、写后门                 |
| `--tmpfs /tmp` (nosuid)                     | 临时目录可用但不可特权，进程退出即消失     |
| `--user 1000:1000`                          | 容器内非 root                              |
| `--cap-drop ALL`                            | 丢掉全部 Linux capabilities                |
| `no-new-privileges`                         | 禁止 setuid 提权                           |
| `--pids-limit 128`                          | 防 fork 炸弹                               |
| `--cpus 1 --memory 512m --memory-swap 512m` | CPU/内存上限（memory-swap 等于内存 = 禁 swap） |
| `/workspace` 只挂会话目录（rw）             | 只能访问本会话文件                         |
| `/uploads` 只读挂载                         | 上传件不可改                               |

通信方式：一次操作一次 `docker exec`，请求/响应走 exec 的 stdin/stdout。
单次调用无状态，容器重启或挂掉不会留下半截连接，同一条 JSON 喂给 ops.py
即可重放。
"""
from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from sandbox.base import ExecResult, SandboxBackend, SandboxError, SandboxSpec
from sandbox.config import sandbox_settings

logger = logging.getLogger(__name__)

#: Windows 命名管道"对端已关闭"的错误码。
#: 109 = ERROR_BROKEN_PIPE，232 = ERROR_NO_DATA，233 = ERROR_PIPE_NOT_CONNECTED
_PIPE_ENDED_WINERRORS = frozenset({109, 232, 233})


def _is_pipe_ended(exc: BaseException) -> bool:
    """判断异常是不是"命名管道对端关闭"，即正常的流结束信号。

    POSIX 上 socket 读到末尾返回空字节，`if not chunk: break` 就能退出。
    Windows 命名管道相反：`win32file.ReadFile` 在对端关闭时抛异常。
    docker-py 的 `NpipeSocket.recv` 原样透传这个异常
    （docker/transport/npipesocket.py:121-122），没翻译成 EOF 语义；
    而且抛出的 `pywintypes.error` 继承自 `Exception` 而非 `OSError`
    （MRO: error -> Exception -> BaseException），`except OSError` 拦不住，
    只能按 winerror 字段识别。
    """
    winerror = getattr(exc, "winerror", None)
    if winerror in _PIPE_ENDED_WINERRORS:
        return True
    if isinstance(exc, OSError) and exc.errno in (errno.EPIPE, errno.ECONNRESET):
        return True
    return False

try:
    import docker
    from docker.errors import APIError, DockerException, ImageNotFound, NotFound
except ImportError:  # pragma: no cover
    docker = None


def _demux(raw: bytes) -> tuple[str, str]:
    """拆开 Docker exec 的多路复用流。

    非 tty 的 exec 把 stdout/stderr 交替封装成
    `[stream_type(1B)][000][size(4B big-endian)][payload]`，
    不解帧的话字符串里会混进二进制头。
    """
    out, err = bytearray(), bytearray()
    i = 0
    while i + 8 <= len(raw):
        stream = raw[i]
        size = int.from_bytes(raw[i + 4:i + 8], "big")
        i += 8
        chunk = raw[i:i + size]
        i += size
        (out if stream == 1 else err).extend(chunk)
    return out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


class DockerSandboxBackend(SandboxBackend):
    """Docker 容器沙箱后端。"""

    name = "docker"

    def __init__(self, session_id: str, *, workspace: Path, uploads: Path | None = None,
                 spec: SandboxSpec | None = None):
        super().__init__(session_id, workspace=workspace, uploads=uploads, spec=spec)
        self._client = None
        self._container = None
        self._container_name = f"{sandbox_settings.container_prefix}{session_id}"
        self._last_used = time.time()
        self._health: dict = {}

    # ------------------------------------------------------------------
    # 可用性探测
    # ------------------------------------------------------------------
    @classmethod
    def probe(cls) -> ExecResult:
        """探测 Docker 守护进程与沙箱镜像是否就绪（启动时调一次，结果进审计）。"""
        if docker is None:
            return ExecResult(ok=False, stderr="未安装 docker SDK（pip install docker）",
                              error_code="sdk_missing", backend="docker")
        started = time.time()
        try:
            client = docker.from_env()
            version = client.version()
        except DockerException as exc:
            return ExecResult(ok=False, stderr=f"Docker 守护进程不可达：{exc}",
                              error_code="daemon_unreachable", backend="docker")
        try:
            client.images.get(sandbox_settings.image)
        except ImageNotFound:
            return ExecResult(
                ok=False,
                stderr=(f"沙箱镜像 {sandbox_settings.image} 不存在，"
                        f"请先执行：python sandbox/build_image.py"),
                error_code="image_missing", backend="docker")
        except DockerException as exc:
            return ExecResult(ok=False, stderr=f"查询镜像失败：{exc}",
                              error_code="image_query_failed", backend="docker")

        return ExecResult(
            ok=True,
            stdout=f"Docker {version.get('Version')} / 镜像 {sandbox_settings.image} 就绪",
            duration_ms=int((time.time() - started) * 1000),
            backend="docker", isolated=True)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    async def start(self) -> None:
        if self._started:
            return
        await asyncio.to_thread(self._start_sync)
        self._started = True

    def _start_sync(self) -> None:
        client = self._get_client()
        self.workspace.mkdir(parents=True, exist_ok=True)

        # 清掉残留容器（进程崩溃/重部署），保证幂等
        self._remove_existing(client)

        volumes = {
            str(self.workspace.resolve()): {"bind": "/workspace", "mode": "rw"},
        }
        if self.uploads and Path(self.uploads).exists():
            volumes[str(Path(self.uploads).resolve())] = {"bind": "/uploads", "mode": "ro"}
        else:
            # 没有上传件也要保证 /uploads 存在，否则 ops.py 的 health 自检会报错
            empty = self.workspace.parent / f".uploads_empty_{self.session_id}"
            empty.mkdir(parents=True, exist_ok=True)
            volumes[str(empty.resolve())] = {"bind": "/uploads", "mode": "ro"}

        kwargs: dict[str, Any] = dict(
            image=sandbox_settings.image,
            command=["sleep", "infinity"],
            name=self._container_name,
            detach=True,
            user="1000:1000",
            working_dir="/workspace",
            volumes=volumes,
            read_only=sandbox_settings.read_only_rootfs,
            tmpfs={"/tmp": f"rw,nosuid,size={sandbox_settings.tmpfs_mb}m"},
            mem_limit=f"{sandbox_settings.memory_mb}m",
            memswap_limit=f"{sandbox_settings.memory_mb}m",   # 禁用 swap：内存硬上限
            nano_cpus=int(sandbox_settings.cpus * 1e9),
            pids_limit=sandbox_settings.pids_limit,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            environment={
                "SANDBOX_SESSION_ID": self.session_id,
                "SANDBOX_LO_PROFILE": "/tmp/lo_profile",
                "SANDBOX_ALLOW_CODE_EXEC": "1" if os.getenv("SANDBOX_ALLOW_CODE_EXEC", "0") in ("1", "true") else "0",
                "HOME": "/tmp",
            },
            ulimits=[{"Name": "nofile", "Soft": 1024, "Hard": 2048}],
            labels={"app": "deep-search-sandbox", "session": self.session_id},
            network_mode="none" if not self.spec.network_enabled else "bridge",
        )
        try:
            self._container = client.containers.run(**kwargs)
        except DockerException as exc:
            raise SandboxError(f"沙箱容器启动失败：{exc}", code="container_start_failed") from exc

        self._wait_running()

        # 启动后立刻自检，实际探测隔离是否生效
        health = self._invoke_sync("health", {}, timeout=30)
        self._health = health.meta if hasattr(health, "meta") else {}
        if health.ok:
            try:
                self._health = json.loads(health.stdout)
            except json.JSONDecodeError:
                self._health = {}
        self._meta = {
            "backend": "docker",
            "container_id": self._container.id[:12],
            "container_name": self._container_name,
            "image": sandbox_settings.image,
            "isolated": True,
            "network_enforced": not self.spec.network_enabled,   # --network none 是内核级断网
            "fs_enforced": "mount-namespace",                    # 只挂载本会话目录
            "spec": self.spec.to_dict(),
            "health": self._health,
        }
        self._audit("create", "allow", target=self._container_name,
                    container_id=self._container.id[:12], health=self._health)

    def _wait_running(self, timeout: int | None = None) -> None:
        timeout = timeout or sandbox_settings.startup_timeout_sec
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._container.reload()
            status = self._container.status
            if status == "running":
                return
            if status in ("exited", "dead"):
                logs = self._container.logs(tail=40).decode("utf-8", "replace")
                raise SandboxError(f"沙箱容器提前退出：{logs}", code="container_died")
            time.sleep(0.3)
        raise SandboxError(f"沙箱容器在 {timeout}s 内未进入 running", code="container_timeout")

    def _remove_existing(self, client) -> None:
        try:
            stale = client.containers.get(self._container_name)
            stale.remove(force=True)
        except NotFound:
            pass
        except DockerException:
            pass

    async def stop(self) -> None:
        if not self._started or self._container is None:
            self._started = False
            return
        await asyncio.to_thread(self._stop_sync)

    def _stop_sync(self) -> None:
        try:
            if sandbox_settings.keep_alive:
                self._audit("stop", "allow", target=self._container_name, reason="keep_alive=true")
                return
            self._container.remove(force=True)
            self._audit("destroy", "allow", target=self._container_name)
        except Exception as exc:                       # 销毁失败不能影响主流程
            self._audit("destroy", "error", target=self._container_name, reason=str(exc)[:200])
        finally:
            self._container = None
            self._started = False

    # ------------------------------------------------------------------
    # 操作执行
    # ------------------------------------------------------------------
    async def _invoke(self, op: str, payload: dict, *, timeout: int | None = None) -> ExecResult:
        if not self._started or self._container is None:
            raise SandboxError("沙箱未启动", code="not_started")
        timeout = timeout or self.spec.exec_timeout_sec
        self._last_used = time.time()
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._invoke_sync, op, payload, timeout),
                                          timeout=timeout + 10)
        except asyncio.TimeoutError:
            return ExecResult(ok=False, stderr=f"沙箱操作超时（{timeout}s）",
                              error_code="timeout", backend="docker", isolated=True)
        except (APIError, DockerException) as exc:
            return ExecResult(ok=False, stderr=f"沙箱通信失败：{exc}",
                              error_code="docker_api_error", backend="docker", isolated=True)

    def _invoke_sync(self, op: str, payload: dict, timeout: int) -> ExecResult:
        started = time.time()
        # 结尾的 \n 是协议的一部分：ops.py 用 readline() 收请求，读到换行才返回。
        # 不能靠 shutdown 来催它，见下方 sendall 处的说明。
        request = (json.dumps({"op": op, **payload}, ensure_ascii=False) + "\n").encode("utf-8")

        container = self._container
        container.reload()
        if container.status != "running":
            return ExecResult(ok=False, stderr="沙箱容器已停止，请重新创建会话",
                              error_code="container_not_running", backend="docker")

        api = self._get_client().api
        exec_id = api.exec_create(
            container.id,
            cmd=["python", "/opt/sandbox/ops.py"],
            stdin=True, stdout=True, stderr=True, tty=False,
            user="1000:1000", workdir="/workspace",
        )["Id"]

        stream = api.exec_start(exec_id, detach=False, socket=True, tty=False)
        sock = getattr(stream, "_sock", stream)
        try:
            # 不要用 shutdown(SHUT_WR) 催 ops.py 的 readline 返回。
            # Windows 命名管道没有半关闭语义：docker-py 的
            # NpipeSocket.shutdown() 就是 close()（docker/transport/npipesocket.py:203），
            # 关掉后紧接着的 recv 会抛
            # RuntimeError('Can not reuse socket after connection was closed.')。
            # unix socket 支持 SHUT_WR，所以这个坑只在 Windows 上出现。
            # 请求末尾的 \n 已经足够让 readline() 返回，不需要 EOF。
            sock.sendall(request)
            raw = bytearray()
            while True:
                try:
                    chunk = sock.recv(65536)
                except Exception as exc:
                    # Windows 命名管道读到末尾抛异常，见 _is_pipe_ended
                    if _is_pipe_ended(exc):
                        break
                    raise
                if not chunk:
                    break                          # POSIX：读到末尾是空字节
                raw.extend(chunk)
        except Exception as exc:
            # 真读不下来了。不往外抛：下面的 exec_inspect 仍能拿到退出码，
            # 已读到的部分照常解析，一个字都没读到则走 bad_sandbox_response 分支。
            # 让 Agent 收到可读的失败反馈，比撞传输层异常好排查。
            logger.warning("沙箱响应读取中断：%s", exc)
        finally:
            try:
                stream.close()
            except Exception:
                pass

        exit_code = api.exec_inspect(exec_id).get("ExitCode", -1)
        stdout_text, stderr_text = _demux(bytes(raw))
        duration = int((time.time() - started) * 1000)

        # 业务层错误也走 JSON，解析不出 JSON 才算传输层故障
        try:
            result = json.loads(stdout_text.strip().splitlines()[-1]) if stdout_text.strip() else {}
        except (json.JSONDecodeError, IndexError):
            return ExecResult(ok=False, stderr=(stderr_text or stdout_text)[:1000] or "沙箱无输出",
                              exit_code=exit_code, duration_ms=duration,
                              error_code="bad_sandbox_response", backend="docker", isolated=True)

        return ExecResult(
            ok=bool(result.get("ok")),
            stdout=str(result.get("stdout", "")),
            stderr=str(result.get("stderr", "")),
            exit_code=int(result.get("exit_code", exit_code)),
            duration_ms=duration,
            error_code=str(result.get("error_code", "")),
            backend="docker",
            isolated=True,
        )

    # ------------------------------------------------------------------
    def _audit(self, action: str, decision: str, *, target: str, reason: str = "", **extra) -> None:
        try:
            from security.audit import audit
            audit.record(action=f"sandbox:{action}", decision=decision,
                         session_id=self.session_id, target=target, reason=reason, **extra)
        except Exception:
            pass

    @property
    def last_used(self) -> float:
        return self._last_used
