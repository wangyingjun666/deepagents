# 目标： 创建文档解析子智能体（第三路信息源域：用户上传的文档）
# 与网络搜索（公网信息）、数据库查询（内部结构化数据）数据源正交、互相独立，
# 主 Agent 可在同一轮并行委派三路，任何一路都不依赖其他路的产出。
from agent.subagents import build_subagent
from tools.upload_file_read_tool import read_file_content

document_parse_agent = build_subagent("doc_parse", [read_file_content])
