# DeepAgents 深度搜索系统

基于 DeepAgents 框架的多智能体深度搜索示例。1 个主智能体调度 3 个子智能体
（联网搜索 / MySQL 查询 / RAGFlow 知识库），结果生成 Markdown 与 PDF，
Vue3 前端通过 WebSocket 实时显示执行进度。

在基础的多智能体编排之外，项目重点做了三件事：

- **执行过程可观测**：会话级上下文标识贯穿工具调用与子 Agent 委派，事件带
  `event_id` / `trace_id` / `span_id` / `parent_span_id` / `duration_ms`，
  落盘 JSONL 并可断线重放，另有按操作名聚合的 P50/P95 指标。
- **并发与容错**：会话 / 子 Agent / 按下游服务三层信号量闸门，指数退避重试
  带抖动，三态熔断，MySQL 连接池读写分离。
- **执行环境隔离**：模型触发的文件读写与文档渲染全部跑在**每会话一个的
  Docker 容器**里（`--network none` / 只读根文件系统 / 非 root / cap-drop ALL /
  cgroup 限额 / 只挂载本会话目录），宿主进程不直接落盘。无 Docker 环境自动
  退回 Windows Job Object 进程级后端，降级会写审计。

## 环境

- Python 3.10+、Node.js 18+、MySQL
- 一个 OpenAI 兼容的大模型 Key（示例用 DeepSeek）
- Tavily Key（可选：联网搜索）；RAGFlow（可选：知识库）
- Docker（可选但推荐：不装则沙箱走进程级兜底）

## 快速开始

```bash
# 1. 导入数据库（建库 + 示例数据）
mysql -u root -p < sql/company_data.sql

# 2. 配置（复制后填写）
copy .env.example .env        # Windows
cp .env.example .env          # Linux / macOS

# 3. 安装后端依赖
pip install -r requirements.txt

# 4. 构建沙箱镜像（可选，未构建时自动走进程级后端）
python sandbox/build_image.py

# 5. 启动后端
python main.py                # 接口文档 http://localhost:8000/docs
```

启动日志会打印实际生效的沙箱后端：

```
[Server] 沙箱后端：docker  （配置=auto，Docker可用=True）
```

另开一个终端启动前端：

```bash
cd ui
npm install
npm run dev                   # 页面 http://localhost:5173
```

页面里可以试：

```
从网络查询2026年新能源汽车销量前5名，保存到md文档
查询 company_db 数据库里阿莫西林的总库存，生成一个markdown报告
```

缺 Tavily / RAGFlow Key 不影响启动，对应子智能体会提示不可用。

## 目录结构

```
agent/          主智能体、子智能体、LLM 客户端
api/            FastAPI 服务、会话上下文（ContextVar）、事件总线出口
tools/          工具实现（数据库 / Markdown / PDF / 联网 / 知识库 / 上传件读取）
security/       路径守卫、SQL 守卫、能力路由、人工审批、哈希链审计
sandbox/        沙箱后端（Docker / 进程级可插拔）、生命周期管理、容器内执行器
observability/  事件模型、总线、落盘、指标
concurrency/    信号量闸门、重试与熔断、连接池
prompt/         提示词配置
sql/            建库与示例数据
ui/             Vue3 前端
tests/          安全沙箱单元测试 + 端到端集成测试
```

## 主要接口

| 接口 | 作用 |
| --- | --- |
| `POST /api/task` | 提交任务，返回会话 ID |
| `POST /api/upload` | 上传附件（落在只读挂载的 uploads 目录） |
| `GET /api/files` / `GET /api/download` | 列出 / 下载会话产物 |
| `GET /api/approvals` / `POST /api/approvals/{id}` | 高风险操作的人工审批 |
| `GET /api/trace/{session_id}` | 拉取某会话的完整事件流 |
| `GET /api/metrics` | 调用次数、失败率、P50/P95 |
| `GET /api/audit` | 审计记录，含哈希链完整性校验 |
| `GET /api/system` | 沙箱后端、隔离自检、安全策略总览 |
| `WS /ws/{session_id}?after_event_id=N` | 实时事件流，带游标可断线重放 |

## 安全设计

- **路径**：拒绝型守卫。绝对路径、`../` 穿越、符号链接逃逸、Windows 保留设备名、
  NTFS 数据流、结尾点空格等一律拒绝；容器内会再校验一次，纵深防御。
- **数据库**：SQL 语句类型白名单 + 危险构造黑名单 + 系统库拒绝 + 表名白名单 +
  强制行数上限 + 超时注入。建议另行配置只读账号作为数据库侧硬边界。
- **权限**：`Capability` 枚举与 `Role → Capability` 路由表，`@requires` 装饰器在
  每次调用前校验。默认拒绝。
- **审批**：不可逆操作执行前挂起，前端确认后恢复；超时按拒绝处理。
- **审计**：`hash_n = SHA256(prev_hash ‖ 记录)`，`verify_chain()` 可检出篡改。

## 测试

```bash
python tests/test_security_and_sandbox.py   # 42 项，不需要 Docker / 数据库
python tests/test_integration.py            # 23 项，无 Docker 时自动走进程级后端
```

## 说明

纯学习交流项目。二次开发请遵守 DeepAgents / LangChain / FastAPI / Vue 等框架 License。
