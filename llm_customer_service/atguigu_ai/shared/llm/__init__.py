# -*- coding: utf-8 -*-
"""
llm - 统一LLM客户端模块

通过LangChain框架提供统一的LLM接口，支持多种类型：
- openai: OpenAI API / vLLM / 其他OpenAI兼容服务
- qwen: 阿里云DashScope通义千问API
- azure: Azure OpenAI
- anthropic: Anthropic Claude
"""

from atguigu_ai.shared.llm.base_client import LLMClient, LLMResponse
from atguigu_ai.shared.llm.langchain_client import LangChainClient

__all__ = [
    "LLMClient",
    "LLMResponse",
    "LangChainClient",
    "create_llm_client",
]


def create_llm_client(
    type: str = "openai",
    **kwargs,
) -> LLMClient:
    """创建LLM客户端实例
    
    工厂函数，创建统一的LLM客户端。
    
    参数：
        type: LLM类型 (openai/qwen/azure/anthropic)
            - openai: OpenAI API或兼容服务(如vLLM)
            - qwen: 阿里云DashScope通义千问
            - azure: Azure OpenAI
            - anthropic: Anthropic Claude
        **kwargs: 客户端配置参数
            - model: 模型名称
            - api_key: API密钥
            - api_base: 自定义API基础URL（用于vLLM等）
            - temperature: 温度参数
            - max_tokens: 最大生成Token数
            - timeout: 请求超时(秒)
            - enable_thinking: 启用深度思考模式
        
    返回：
        LLMClient实例
        
    示例：
        # OpenAI API
        >>> client = create_llm_client(
        ...     type="openai",
        ...     model="gpt-4",
        ...     api_key="sk-xxx",
        ... )
        
        # 阿里云DashScope (支持thinking)
        >>> client = create_llm_client(
        ...     type="qwen",
        ...     model="qwen-plus",
        ...     api_key="sk-xxx",
        ...     enable_thinking=True,
        ... )
        
        # vLLM自部署 (OpenAI兼容接口)
        >>> client = create_llm_client(
        ...     type="openai",
        ...     model="Qwen/Qwen3-8B",
        ...     api_base="http://localhost:8000/v1",
        ...     api_key="EMPTY",
        ...     enable_thinking=True,
        ... )
        
        # 使用
        >>> response = await client.complete([
        ...     {"role": "user", "content": "你好"}
        ... ])
    """
    return LangChainClient(type=type, **kwargs)
