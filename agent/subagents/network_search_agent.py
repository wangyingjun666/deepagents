# 目标： 创建网络搜索子智能体（第一路信息源域：公网信息）
# 方式1： dict -> deepagents  方式2： compiledSubAgent -> langchain langgraph
from agent.subagents import build_subagent
from tools.tavily_tool import internet_search

network_search_agent = build_subagent("tavily", [internet_search])
