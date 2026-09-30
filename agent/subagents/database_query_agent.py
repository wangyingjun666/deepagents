# 目标： 创建数据库查询子智能体（第二路信息源域：企业内部结构化数据）
from agent.subagents import build_subagent
from tools.db_tools import execute_sql_query, get_table_data, list_sql_tables

database_query_agent = build_subagent("db", [list_sql_tables, get_table_data, execute_sql_query])
