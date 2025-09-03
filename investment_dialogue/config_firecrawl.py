import os
from typing import Dict, Any

class Config:
    # Firecrawl API 配置
    FIRECRAWL_API_KEY: str = "fc-018b0eb1333c4ccf97460d3dd764c0d2"
    
    # OpenAI API 配置
    OPENAI_API_KEY: str = "sk-HCkRhThj35aWxnYND50c64BbC026434f95E6538bE92f1cC5"
    OPENAI_API_BASE: str = "https://api.shubiaobiao.cn/v1"
    
    # LLM 配置
    LLM_CONFIG: Dict[str, Any] = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.2,
        "max_tokens": 1000,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    }
    
    # 重试配置
    RETRY_CONFIG: Dict[str, Any] = {
        "max_retries": 3,
        "initial_delay": 2,
        "max_delay": 10,
        "exponential_base": 2
    }
    
    # 环境变量配置
    ENV_VARS: Dict[str, str] = {
        "NODE_TLS_REJECT_UNAUTHORIZED": "0",
        "PATH": os.environ.get("PATH", "")
    } 