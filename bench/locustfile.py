# -*- coding: utf-8 -*-
"""
DeepAgents 深度搜索的压测脚本。

压什么
------
项目里 `POST /api/task` 是"提交即返回"——请求只负责把任务丢进后台，立刻回一个
thread_id，真正干活在图里异步跑。所以只压这个接口量出来的几十毫秒毫无意义，
**用户感知的时延是"提交 → 拿到最终结果"**。

本脚本对每个虚拟用户模拟一次完整的深度搜索：
    1. POST /api/task 提交任务，拿到 thread_id
    2. 轮询 GET /api/trace/{thread_id}，直到出现 task_result / error 事件
    3. 把「提交→完成」的耗时单独记成一条 TASK/deep_search_e2e 请求

这样 Locust 报表里会同时出现两类数据：
    POST /api/task        —— 接口本身的响应（亚秒级，看的是 Web 层）
    TASK deep_search_e2e  —— 端到端任务时延（看的是整条链路，**P95 主要看这一行**）
    GET /api/trace/...    —— 轮询开销，合并成一行，方便确认它没有喧宾夺主

关于模型
--------
配合 `bench/serve_stub.py` 使用：那里的模型是固定延迟的桩，所以量出来的时延是
「本项目编排 + 沙箱 + 可观测 + 并发闸门」的开销，不含真实推理时间。换成真实 Key
直接压正式服务（`python main.py`）时，端到端 P95 会把模型耗时一起算进去。

用法：
    python bench/serve_stub.py --port 8011 --model-latency-ms 300
    locust -f bench/locustfile.py --headless -u 8 -r 8 -t 60s --host http://127.0.0.1:8011
"""

import io
import os
import sys
import time

from locust import HttpUser, task, constant

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# 任务最长等多久算失败（秒）。深度搜索含多轮模型调用，给得比单次调用宽裕些
TASK_TIMEOUT_SEC = float(os.environ.get("TASK_TIMEOUT_SEC", "180"))
# 轮询间隔（秒）。太密会把服务端自己的 /api/trace 压成瓶颈，太疏又会低估精度
POLL_INTERVAL_SEC = float(os.environ.get("POLL_INTERVAL_SEC", "0.2"))

QUERIES = [
    "帮我查一下公司有多少员工",
    "统计各部门的人数分布",
    "查一下最近三个月的订单总额",
    "整理一份公司组织架构说明",
    "查一下员工平均薪资是多少",
    "找出入职时间最长的十位员工",
    "统计每个产品的销量排名",
    "查一下公司现有的产品线有哪些",
]


class DeepSearchUser(HttpUser):
    """一个虚拟用户 = 一次完整的深度搜索会话。"""

    # 0 等待：拿到结果立刻发起下一次，用并发数直接表示"同时有多少个会话在跑"
    wait_time = constant(0)

    def _wait_for_result(self, thread_id: str) -> tuple[bool, int]:
        """轮询到任务结束。返回 (是否成功, 轮询次数)。"""
        deadline = time.perf_counter() + TASK_TIMEOUT_SEC
        polls = 0
        while time.perf_counter() < deadline:
            time.sleep(POLL_INTERVAL_SEC)
            polls += 1
            # 用 name 把带 id 的路径合并成一行，否则报表会被几千个不同 URL 撑爆
            with self.client.get(f"/api/trace/{thread_id}", name="/api/trace/[id]",
                                 catch_response=True) as resp:
                if resp.status_code != 200:
                    resp.failure(f"HTTP {resp.status_code}")
                    continue
                try:
                    events = resp.json().get("events", [])
                except Exception as exc:  # noqa: BLE001
                    resp.failure(f"响应不是合法 JSON: {exc}")
                    continue
                resp.success()
                kinds = {e.get("event") for e in events}
                if "task_result" in kinds:
                    return True, polls
                if "error" in kinds:
                    return False, polls
        return False, polls

    @task
    def deep_search(self):
        query = QUERIES[time.perf_counter_ns() % len(QUERIES)]
        started = time.perf_counter()

        with self.client.post("/api/task",
                              json={"query": query, "user_id": "loadtest"},
                              catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            try:
                thread_id = resp.json().get("thread_id")
            except Exception as exc:  # noqa: BLE001
                resp.failure(f"响应不是合法 JSON: {exc}")
                return
            if not thread_id:
                resp.failure("响应里没有 thread_id")
                return
            resp.success()

        ok, polls = self._wait_for_result(thread_id)
        elapsed_ms = (time.perf_counter() - started) * 1000

        # 端到端时延单独立一行：这才是用户实际等待的时间
        self.environment.events.request.fire(
            request_type="TASK",
            name="deep_search_e2e",
            response_time=elapsed_ms,
            response_length=0,
            exception=None if ok else RuntimeError(
                f"任务未在 {TASK_TIMEOUT_SEC}s 内完成（轮询 {polls} 次）"),
            context={},
        )
