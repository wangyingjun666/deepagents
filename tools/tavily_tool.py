# 定义一个网络搜索的工具！
# ======================== 导入核心依赖 ========================
# 类型注解：增强代码提示和静态检查能力
from typing import Literal
# LangChain 工具装饰器：将普通函数转为 Agent 可调用的工具
from langchain_core.tools import tool

# 系统/第三方依赖
import os  # 系统路径/环境变量处理
from dotenv import load_dotenv  # 加载 .env 文件中的环境变量

# 自定义模块：工具调用埋点监控（需确保 api 模块可导入）
from api.monitor import monitor

# ======================== 初始化配置 ========================
# 加载项目根目录的 .env 文件，读取环境变量（如 TAVILY_API_KEY）
load_dotenv()

# 步骤1： 定义一个TavilyClient对象
# 注意：改为惰性创建（懒加载）——没有配置 TAVILY_API_KEY 时不让程序启动崩溃，
#       而是等真正调用搜索工具时返回友好提示。
_tavily_client = None


def _get_tavily_client():
    """惰性创建 Tavily 客户端；没有 key 或创建失败时返回 None。"""
    global _tavily_client
    if _tavily_client is not None:
        return _tavily_client

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("[Tavily] 未配置 TAVILY_API_KEY，联网搜索不可用。")
        return None

    try:
        from tavily import TavilyClient
        _tavily_client = TavilyClient(api_key=api_key)
        return _tavily_client
    except Exception as e:
        print(f"[Tavily] 客户端初始化失败: {e}")
        return None


# 步骤2： 定义一个网络搜索工具
@tool
def internet_search(
        query: str,
        topic: Literal["news", "finance", "general"] = "general",
        max_results: int = 5,
        include_raw_content: bool = False
):
    """
    根据用户问题，进行网络信息收集！
    注意：主要搜索公开的网络信息！如果指定查询数据库或者rag不能使用此工具！
    :param query: 用户的查询信息
    :param topic: 查询的类型
    :param max_results: 返回的最大条数
    :param include_raw_content: 是否返回原内容 False 精简 True 详细
    :return:
    """
    # 每次调用工具，都都会向前端推进调用进度！
    # 参数1： 工具的名字  参数2： 就是调用工具的参数信息
    monitor.report_tool(tool_name="网络搜索工具",
                        args={"query": query, "topic": topic, "max_results": max_results,
                              "include_raw_content": include_raw_content})

    client = _get_tavily_client()
    if client is None:
        return ("网络搜索不可用：缺少有效的 TAVILY_API_KEY。"
                "请到 https://app.tavily.com 申请免费 key 并填写到 .env 中。")

    try:
        return client.search(query=query, topic=topic,
                             max_results=max_results, include_raw_content=include_raw_content)
    except Exception as e:
        return f"网络搜索失败：{e}"
