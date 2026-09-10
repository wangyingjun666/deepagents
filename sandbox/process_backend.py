"""
进程级沙箱后端 —— 没有 Docker 时的本地兜底。

每个会话一个受 Windows Job Object 约束的子进程，硬性限制：

* `JOB_OBJECT_LIMIT_PROCESS_MEMORY` / `JOB_OBJECT_LIMIT_JOB_MEMORY` —— 内存上限，超了直接杀进程
* `JOB_OBJECT_LIMIT_ACTIVE_PROCESS` —— 进程数上限，挡 fork 炸弹
* `JOB_OBJECT_CPU_RATE_CONTROL`（HARD_CAP）—— CPU 占用率硬上限
* `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` —— 句柄关闭即全部退出，不留孤儿进程

加上工作目录锁在会话目录、环境变量最小化、`ops.py` 内部再跑一遍路径守卫，
构成"资源 + 路径 + 环境"三层约束。

不是安全沙箱，与容器后端的差距：

| 维度 | Docker 后端 | 进程级后端 |
|---|---|---|
| 文件系统 | 挂载命名空间隔离，根文件系统只读 | 只有 `ops.py` 内部的路径检查，同用户身份下没有内核级拦阻 |
| 网络 | `--network none`，内核级断网 | 无法限制，子进程可自由联网 |
| 权限 | 非 root + cap-drop ALL + no-new-privileges | 与宿主同权限 |
| 逃逸影响 | 需突破 namespace + cap + seccomp | 拿到宿主用户权限 |

定位是开发机兜底：没有 Docker 也能跑通全链路。生产部署用 Docker 后端，所以
`SANDBOX_BACKEND=auto` 优先探测 Docker，失败才降级，每次降级都写审计。

（不用 `subprocess` + `resource.setrlimit` 是因为 `setrlimit` 属 POSIX，Windows
没有；Windows 上对应的原语就是 Job Object。）
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from sandbox.base import ExecResult, SandboxBackend, SandboxError, SandboxSpec
from sandbox.config import sandbox_settings

_OPS_SCRIPT = Path(__file__).resolve().parent / "worker" / "ops.py"


class ProcessSandboxBackend(SandboxBackend):
    """基于受约束子进程的沙箱后端。"""

    name = "process"

    def __init__(self, session_id: str, *, workspace: Path, uploads: Path | None = None,
                 spec: SandboxSpec | None = None):
        super().__init__(session_id, workspace=workspace, uploads=uploads, spec=spec)
        self._job = None
        self._last_used = time.time()
        self._health: dict = {}
        self._job_error: str = ""

    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._started:
            return
        await asyncio.to_thread(self._start_sync)
        self._started = True

    def _start_sync(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        if self.uploads:
            Path(self.uploads).mkdir(parents=True, exist_ok=True)

        self._job_error = self._create_job_object()

        health = self._invoke_sync("health", {}, timeout=30)
        try:
            self._health = json.loads(health.stdout) if health.ok else {}
        except json.JSONDecodeError:
            self._health = {}

        self._meta = {
            "backend": "process",
            "pid_namespace": f"job:{self.session_id}",
            "isolated": False,                     # 非真隔离
            "job_object": self._job is not None,
            "job_error": self._job_error,
            "network_enforced": False,             # Job Object 管不了网络
            "fs_enforced": "app-level-only",       # 文件隔离靠 ops.py 内部守卫，非内核级
            "spec": self.spec.to_dict(),
            "health": self._health,
        }
        self._audit("create", "allow",
                    target=str(self.workspace),
                    backend="process",
                    job_object=bool(self._job),
                    reason=self._job_error or "process-level fallback (NOT a security boundary)")

    # ---------------- Windows Job Object ----------------
    def _create_job_object(self) -> str:
        """给这个会话创建资源受限的 Job Object。失败时返回原因，不阻断启动。"""
        if os.name != "nt":
            return "非 Windows 平台未创建 Job Object（POSIX 下应改用 cgroup/rlimit）"
        try:
            import win32api  # noqa: F401
            import win32job
        except ImportError:
            return "未安装 pywin32，无法创建 Job Object，仅做路径约束"

        try:
            job = win32job.CreateJobObject(None, f"ds_sbx_{self.session_id}")
            info = win32job.QueryInformationJobObject(
                job, win32job.JobObjectExtendedLimitInformation)

            flags = (win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                     | win32job.JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                     | win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
                     | win32job.JOB_OBJECT_LIMIT_JOB_MEMORY)
            info["BasicLimitInformation"]["LimitFlags"] = (
                info["BasicLimitInformation"].get("LimitFlags", 0) | flags)
            info["BasicLimitInformation"]["ActiveProcessLimit"] = self.spec.pids_limit
            info["ProcessMemoryLimit"] = self.spec.memory_mb * 1024 * 1024
            info["JobMemoryLimit"] = self.spec.memory_mb * 1024 * 1024
            win32job.SetInformationJobObject(
                job, win32job.JobObjectExtendedLimitInformation, info)

            # CPU 硬上限：CpuRate 单位是万分之一，1.0 核 => 10000
            try:
                cpu_info = {
                    "ControlFlags": (win32job.JOB_OBJECT_CPU_RATE_CONTROL_ENABLE
                                     | win32job.JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP),
                    "CpuRate": max(1, int(self.spec.cpus * 10000)),
                }
                win32job.SetInformationJobObject(
                    job, win32job.JobObjectCpuRateControlInformation, cpu_info)
            except Exception as exc:               # 老系统不支持 CPU 速率控制
                self._job_error = f"CPU 速率控制不可用：{exc}"

            self._job = job
            return ""
        except Exception as exc:
            return f"Job Object 创建失败：{exc}"

    def _assign_to_job(self, pid: int) -> None:
        """把刚启动的子进程纳入 Job Object。

        先启动后纳管，中间有极短的未纳管窗口。用 CREATE_SUSPENDED 创建再纳管
        （`win32api.CreateProcess`）能消除这个窗口，但复杂度高不少。这里跑的是
        自己的 ops.py，不是任意不可信代码，窗口只影响"极端瞬时内存峰值可能不被
        记账"，不构成安全风险。
        """
        if self._job is None:
            return
        try:
            import win32api
            import win32con
            import win32job
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, pid)
            win32job.AssignProcessToJobObject(self._job, handle)
            win32api.CloseHandle(handle)
        except Exception as exc:
            self._job_error = f"进程纳管失败：{exc}"

    # ------------------------------------------------------------------
    async def stop(self) -> None:
        self._started = False
        self._audit("destroy", "allow", target=str(self.workspace), backend="process")
        if self._job is not None:
            try:
                import win32api
                win32api.CloseHandle(self._job)    # KILL_ON_JOB_CLOSE 连带清理子进程
            except Exception:
                pass
            self._job = None

    # ------------------------------------------------------------------
    async def _invoke(self, op: str, payload: dict, *, timeout: int | None = None) -> ExecResult:
        timeout = timeout or self.spec.exec_timeout_sec
        self._last_used = time.time()
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._invoke_sync, op, payload, timeout), timeout=timeout + 10)
        except asyncio.TimeoutError:
            return ExecResult(ok=False, stderr=f"沙箱操作超时（{timeout}s）",
                              error_code="timeout", backend="process", isolated=False)

    def _build_env(self) -> dict:
        """最小环境变量集，不把宿主环境（含各种 API Key）泄进子进程。"""
        keep = ("PATH", "SYSTEMROOT", "SystemRoot", "TEMP", "TMP", "WINDIR",
                "PATHEXT", "COMSPEC", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE")
        env = {k: v for k, v in os.environ.items() if k in keep}
        env.update({
            "SANDBOX_WORKSPACE": str(Path(self.workspace).resolve()),
            "SANDBOX_UPLOADS": str(Path(self.uploads).resolve()) if self.uploads else str(self.workspace),
            "SANDBOX_SESSION_ID": self.session_id,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        return env

    def _invoke_sync(self, op: str, payload: dict, timeout: int) -> ExecResult:
        started = time.time()
        request = json.dumps({"op": op, **payload}, ensure_ascii=False)

        try:
            proc = subprocess.Popen(
                [sys.executable, str(_OPS_SCRIPT)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=str(Path(self.workspace).resolve()),
                env=self._build_env(),
                shell=False,
                text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
        except OSError as exc:
            return ExecResult(ok=False, stderr=f"沙箱子进程启动失败：{exc}",
                              error_code="spawn_failed", backend="process")

        self._assign_to_job(proc.pid)

        try:
            out, err = proc.communicate(request, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return ExecResult(ok=False, stderr=f"沙箱操作超时（{timeout}s）",
                              error_code="timeout",
                              duration_ms=int((time.time() - started) * 1000),
                              backend="process")

        duration = int((time.time() - started) * 1000)
        try:
            result = json.loads(out.strip().splitlines()[-1]) if out.strip() else {}
        except (json.JSONDecodeError, IndexError):
            return ExecResult(ok=False, stderr=(err or out)[:1000] or "沙箱无输出",
                              exit_code=proc.returncode, duration_ms=duration,
                              error_code="bad_sandbox_response", backend="process")

        return ExecResult(
            ok=bool(result.get("ok")),
            stdout=str(result.get("stdout", "")),
            stderr=str(result.get("stderr", "")),
            exit_code=int(result.get("exit_code", proc.returncode)),
            duration_ms=duration,
            error_code=str(result.get("error_code", "")),
            backend="process",
            isolated=False,
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
