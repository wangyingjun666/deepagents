from dotenv import load_dotenv,find_dotenv
import os
from langchain.chat_models import init_chat_model

# 加载配置文件
# find_dotenv() 确保找到 .env文件 递归查询当前项目文件夹
load_dotenv(find_dotenv())

# 大模型统一入口：读取 .env 中的 LLM_MODEL（当前指向 DeepSeek）
# DeepSeek 兼容 OpenAI 接口，因此仍使用 model_provider="openai"
model = init_chat_model(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    model_provider="openai"
)
