"""子智能体注册处：每一路一个文件，公共构造收敛在这里。

子 Agent 走 deepagents 的 dict 协议（name / description / system_prompt / tools）：
  - description   给主 Agent 做委派路由的依据
  - system_prompt 给子 Agent 自己用，各路上下文互相隔离
两者都集中在 prompt/prompts.yml 的 sub_agents 段，代码里只声明"这一路挂哪些工具"。

新增一路子 Agent 的固定三步：
  1. prompts.yml 的 sub_agents 下加一段（name/description/system_prompt）
  2. 本目录新建 xxx_agent.py，调 build_subagent("<key>", [工具...])
  3. agent/main_agent.py 的 subagents 列表里挂载
"""
from agent.prompts import sub_agents_content


def build_subagent(spec_key: str, tools: list) -> dict:
    """按 prompts.yml 里 sub_agents.<spec_key> 的定义构造子 Agent。"""
    spec = sub_agents_content[spec_key]
    return {
        "name": spec["name"],
        "description": spec["description"],
        "system_prompt": spec["system_prompt"],
        "tools": tools,
    }
