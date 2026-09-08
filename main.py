"""
DeepAgents 深度搜索项目 · 一键启动入口（后端服务）

PyCharm 用法：
    右键 main.py -> Run 'main'  即可启动后端服务（默认端口 8000）。

启动后：
    后端接口文档：http://localhost:8000/docs
    前端页面    ：在 ui 目录执行 `npm run dev` 后访问 http://localhost:5173

说明：
    首次启动会导入并构建智能体（main_agent），可能需要几秒到几十秒，
    看到 "Uvicorn running on http://0.0.0.0:8000" 即启动成功。
"""
import sys
from pathlib import Path

# 保证从任意位置运行时都能正确导入本项目模块
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from api.server import app  # noqa: E402  (构建 main_agent，耗时数秒)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
