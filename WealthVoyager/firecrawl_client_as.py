from mcp_openai import MCPClient, MCPClientConfig, MCPServerConfig, LLMClientConfig, LLMRequestConfig
import asyncio
from typing import List, Dict, Any, Optional
import sys
import os
import anyio

# 由于config.py会被迁移到本目录，直接import
from config_firecrawl import Config

class FirecrawlClient:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.client = None
        self.exit_stack = None
        self._closing = False
        self._connect_task = None
    
    async def initialize(self):
        """初始化 MCP 客户端"""
        try:
            # 配置 MCP 服务器
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "../firecrawl_test/scripts/start_server.sh")
            
            if not os.path.exists(script_path):
                raise Exception(f"启动脚本不存在: {script_path}")
            
            print("正在配置服务器...")
            mcp_server_config = MCPServerConfig(
                command="/bin/bash",
                args=[script_path],
                env={
                    "FIRECRAWL_API_KEY": self.config.FIRECRAWL_API_KEY,
                    **self.config.ENV_VARS
                }
            )

            # 配置 MCP 客户端，只注册 deep_research 工具
            mcp_client_config = MCPClientConfig(
                mcpServers={
                    "firecrawl": mcp_server_config
                }
            )

            # === DEBUG: 打印并截断所有工具的 description ===
            if hasattr(mcp_server_config, 'tools'):
                for tool in mcp_server_config.tools:
                    desc = getattr(tool, 'description', '')
                    print(f"[DEBUG] tool: {getattr(tool, 'name', 'unknown')}, description length: {len(desc)}")
                    print(f"[DEBUG] description preview: {desc[:100]}")
                    if isinstance(desc, str) and len(desc) > 256:
                        tool.description = desc[:256]
                        print(f"[DEBUG] description truncated to 256 chars")

            # 配置 LLM 客户端
            llm_client_config = LLMClientConfig(
                api_key=self.config.OPENAI_API_KEY,
                base_url=self.config.OPENAI_API_BASE
            )

            # 配置 LLM 请求
            llm_request_config = LLMRequestConfig(
                **self.config.LLM_CONFIG
            )

            print("正在创建客户端...")
            # 创建客户端
            self.client = MCPClient(
                mpc_client_config=mcp_client_config,
                llm_client_config=llm_client_config,
                llm_request_config=llm_request_config
            )

            # === DEBUG: 创建后再次打印 self.client.tools ===
            if hasattr(self.client, 'tools'):
                for tool in self.client.tools:
                    desc = getattr(tool, 'description', '')
                    print(f"[DEBUG][AFTER CREATE] tool: {getattr(tool, 'name', 'unknown')}, description length: {len(desc)}")
                    print(f"[DEBUG][AFTER CREATE] description preview: {desc[:100]}")

            print("正在连接到服务器...")
            # 设置连接超时
            try:
                async with anyio.create_task_group() as tg:
                    self._connect_task = tg.start_soon(self._connect_to_server)
                    await asyncio.sleep(5)  # 给服务器一些启动时间
                    
                    try:
                        await asyncio.wait_for(
                            self._wait_for_connection(),
                            timeout=25  # 剩余25秒超时
                        )
                        print("服务器连接成功！")
                    except asyncio.TimeoutError:
                        raise Exception("连接服务器超时，请检查服务器是否正常运行")
            except Exception as e:
                if not isinstance(e, Exception) or "连接服务器超时" not in str(e):
                    print(f"连接错误: {type(e).__name__}: {str(e)}")
                raise
            
            # 截断所有工具的 description，防止超长报错（冗余保险）
            if hasattr(self.client, "tools"):
                for tool in self.client.tools:
                    if hasattr(tool, "description") and isinstance(tool.description, str):
                        tool.description = tool.description[:256]
        except Exception as e:
            print(f"\n初始化错误: {str(e)}")
            print(f"详细错误信息: {type(e).__name__}: {str(e)}")
            await self.close()
            raise
    
    async def _connect_to_server(self):
        """连接到服务器的内部方法"""
        try:
            await self.client.connect_to_server("firecrawl")
            self.exit_stack = self.client.exit_stack
        except Exception as e:
            print(f"连接过程中出错: {type(e).__name__}: {str(e)}")
            raise
    
    async def _wait_for_connection(self):
        """等待连接完成"""
        while not self.exit_stack:
            await asyncio.sleep(0.1)
    
    async def retry_with_exponential_backoff(self, func):
        """使用指数退避的重试机制"""
        delay = self.config.RETRY_CONFIG["initial_delay"]
        max_retries = self.config.RETRY_CONFIG["max_retries"]
        max_delay = self.config.RETRY_CONFIG["max_delay"]
        exponential_base = self.config.RETRY_CONFIG["exponential_base"]
        
        for retry_count in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if retry_count == max_retries - 1:
                    raise e
                
                print(f"\n重试 {retry_count + 1}/{max_retries}，等待 {delay} 秒...")
                await asyncio.sleep(delay)
                delay = min(delay * exponential_base, max_delay)
        
        raise Exception("达到最大重试次数")

    async def process_query(self, query: str) -> str:
        """处理用户查询"""
        if not self.client or not self.exit_stack:
            raise Exception("客户端未初始化或未连接")
        
        if self._closing:
            raise Exception("客户端正在关闭")
        
        messages = [{"role": "user", "content": query}]
        
        try:
            print("\n正在发送查询到 MCP 服务器...")
            # 使用重试机制处理消息
            messages = await self.retry_with_exponential_backoff(
                lambda: self.client.process_messages(messages)
            )
            print("\nMCP 服务器返回的消息数量:", len(messages))
            
            # 提取助手的回复
            for message in messages:
                if message["role"] == "assistant" and message.get("content"):
                    return message["content"]
            
            return "未获取到回复"
            
        except Exception as e:
            print(f"\n错误: {str(e)}")
            print(f"详细错误信息: {type(e).__name__}: {str(e)}")
            return "处理查询时发生错误"

    async def close(self):
        """关闭客户端连接"""
        if self._closing:
            return
            
        self._closing = True
        try:
            if self._connect_task and not self._connect_task.done():
                self._connect_task.cancel()
            
            if self.exit_stack:
                async with anyio.create_task_group() as tg:
                    await tg.start(self.exit_stack.aclose)
                print("客户端连接已关闭")
        except Exception as e:
            print(f"关闭连接时出错: {str(e)}")
        finally:
            self.exit_stack = None
            self.client = None
            self._closing = False

    async def debug_deep_research(self, query: str, maxDepth=2, timeLimit=60, maxUrls=10):
        """
        调用firecrawl_deep_research工具，打印参数和原始内容，便于debug
        """
        if not self.client or not self.exit_stack or not hasattr(self.client, "session"):
            raise Exception("客户端未初始化或未连接")
        tool_name = "firecrawl_deep_research"
        tool_args = {
            "query": query,
            "maxDepth": maxDepth,
            "timeLimit": timeLimit,
            "maxUrls": maxUrls
        }
        print("\n=== 调用function tool参数如下 ===")
        import json
        print(json.dumps({"tool_name": tool_name, "tool_args": tool_args}, ensure_ascii=False, indent=2))
        result = await self.client.session.call_tool(tool_name, tool_args)
        print("\n=== MCP抓取到的原始内容如下（前1000字） ===")
        print(str(result)[:1000])
        print(f"\n=== 原始内容总长度: {len(str(result))} ===")
        return result 