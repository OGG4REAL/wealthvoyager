from camel.types import ModelType, ModelPlatformType
from camel.configs import ChatGPTConfig
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 模型配置
MODEL_CONFIG = {
    "model_platform": ModelPlatformType.DEEPSEEK,
    "model_type": "deepseek-chat",
    "model_config_dict": {
        "temperature": 0.7,
        "max_tokens": 2000
    },
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "url": "https://api.deepseek.com/v1/"
} 