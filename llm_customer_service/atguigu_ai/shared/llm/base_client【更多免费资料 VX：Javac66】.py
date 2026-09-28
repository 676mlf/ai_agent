# -*- coding: utf-8 -*-
"""
base_client - LLM客户端基类

定义LLM客户端的抽象接口，所有具体客户端实现都继承此基类。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Text


@dataclass
class LLMResponse:
    """LLM响应数据类
    
    封装LLM API的响应结果。
    
    属性：
        content: 生成的文本内容
        model: 使用的模型名称
        usage: Token使用统计
        latency: 响应延迟(秒)
        raw_response: 原始API响应(用于调试)
        metadata: 额外元数据(如思考过程等)
    """
    content: str
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    latency: float = 0.0
    raw_response: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def prompt_tokens(self) -> int:
        """提示词Token数"""
        return self.usage.get("prompt_tokens", 0)
    
    @property
    def completion_tokens(self) -> int:
        """生成Token数"""
        return self.usage.get("completion_tokens", 0)
    
    @property
    def total_tokens(self) -> int:
        """总Token数"""
        return self.usage.get("total_tokens", 0)
    
    @property
    def thinking_content(self) -> Optional[str]:
        """思考过程内容（如果启用了thinking模式）"""
        return self.metadata.get("thinking_content")


class LLMClient(ABC):
    """LLM客户端抽象基类
    
    定义LLM客户端的统一接口，所有具体实现都必须实现这些方法。
    
    属性：
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大生成Token数
        timeout: 请求超时(秒)
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 30,
        **kwargs: Any,
    ) -> None:
        """初始化LLM客户端
        
        参数：
            model: 模型名称
            api_key: API密钥
            api_base: API基础URL(可选)
            temperature: 温度参数
            max_tokens: 最大生成Token数
            timeout: 请求超时(秒)
            **kwargs: 额外配置参数
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_config = kwargs
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """异步生成文本补全
        
        核心方法，发送消息列表给LLM并获取响应。
        
        参数：
            messages: 消息列表，格式为[{"role": "user/assistant/system", "content": "..."}]
            **kwargs: 额外的API参数
            
        返回：
            LLMResponse对象
            
        异常：
            LLMConnectionError: 连接失败
            LLMTimeoutError: 请求超时
            LLMResponseError: 响应解析失败
        """
        pass
    
    @abstractmethod
    def complete_sync(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """同步生成文本补全
        
        同步版本的complete方法，用于非异步上下文。
        
        参数：
            messages: 消息列表
            **kwargs: 额外的API参数
            
        返回：
            LLMResponse对象
        """
        pass
    
    def validate(self) -> bool:
        """验证客户端配置
        
        检查API密钥等配置是否有效。
        
        返回：
            配置有效返回True，否则返回False
        """
        return bool(self.api_key and self.model)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
