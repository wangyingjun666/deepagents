# DeepAgents 深度搜索系统

基于 DeepAgents 框架的多智能体深度搜索示例：1 个主智能体调度 3 个子智能体
（联网搜索 / MySQL 查询 / RAGFlow 知识库），结果生成 Markdown / PDF，
Vue3 前端 + WebSocket 实时进度。

## 环境

- Python 3.10+、Node.js 18+、MySQL
- 一个 OpenAI 兼容的大模型 Key（示例用 DeepSeek）
- Tavily Key（可选：联网搜索）；RAGFlow（可选：知识库）

## 快速开始

```bash
# 1. 导入数据库（建库 + 示例数据）
mysql -u root -p < sql/company_data.sql

# 2. 配置（复制后填写）
copy .env.example .env    # 填 OPENAI_API_KEY 和 MYSQL_PASSWORD

# 3. 安装后端依赖
pip install -r requirements.txt

# 4. 启动后端
python main.py            # 接口文档 http://localhost:8000/docs

# 5. 另开一个终端启动前端
cd ui
npm install
npm run dev               # 页面 http://localhost:5173
```

在页面输入示例：`从网络查询2026年新能源汽车销量前5名，保存到md文档`
或 `查询pharma_db数据库里阿莫西林的总库存，生成一个markdown报告`。

缺 Tavily / RAGFlow Key 不影响启动，对应子智能体会提示不可用。

## .env 必填项

| 配置 | 说明 |
| --- | --- |
| OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL | 大模型 |
| MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE / MYSQL_HOST / MYSQL_PORT | MySQL（先导入 sql/company_data.sql） |

## 说明

纯学习交流项目。二次开发请遵守 DeepAgents / LangChain / FastAPI / Vue 等框架 License。
