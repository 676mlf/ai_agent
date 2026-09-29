# -*- coding: utf-8 -*-
"""
langchain_client - 统一LLM客户端

通过LangChain框架集成多种LLM后端，提供统一的接口。
支持的类型：
- openai: OpenAI API / vLLM / 其他OpenAI兼容服务
- qwen: 阿里云DashScope通义千问API
- azure: Azure OpenAI
- anthropic: Anthropic Claude
"""

import time
from typing import Any, Dict, List, Optional

from atguigu_ai.shared.llm.base_client import LLMClient, LLMResponse
from atguigu_ai.shared.proxy import bypass_proxy_for_domestic_hosts
from atguigu_ai.shared.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)


class LangChainClient(LLMClient):
    """统一LLM客户端
    
    通过LangChain框架封装LLM调用，支持多种后端模型。
    
    使用示例：
        # OpenAI API
        >>> client = LangChainClient(
        ...     type="openai",
        ...     model="gpt-4",
        ...     api_key="sk-xxx",
        ... )
        
        # 阿里云DashScope (支持thinking模式)
        >>> client = LangChainClient(
        ...     type="qwen",
        ...     model="qwen-plus",
        ...     api_key="sk-xxx",
        ...     enable_thinking=True,
        ... )
        
        # vLLM自部署 (OpenAI兼容接口，支持thinking模式)
        >>> client = LangChainClient(
        ...     type="openai",
        ...     model="Qwen/Qwen3-8B",
        ...     api_base="http://localhost:8000/v1",
        ...     api_key="EMPTY",
        ...     enable_thinking=True,  # vLLM需要启动时加 --enable-reasoning
        ... )
    """
    
    # 支持的类型
    SUPPORTED_TYPES = ["openai", "qwen", "azure", "anthropic"]
    
    def __init__(
        self,
        type: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: str = "",
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 30,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> None:
        """初始化LLM客户端
        
        参数：
            type: LLM类型 (openai/qwen/azure/anthropic)
                - openai: OpenAI API或兼容服务(如vLLM)
                - qwen: 阿里云DashScope通义千问
                - azure: Azure OpenAI
                - anthropic: Anthropic Claude
            model: 模型名称
            api_key: API密钥
            api_base: 自定义API基础URL（用于vLLM等自部署服务）
            temperature: 温度参数
            max_tokens: 最大生成Token数
            timeout: 请求超时(秒)
            enable_thinking: 启用深度思考/推理模式
                - qwen类型：DashScope原生支持
                - openai类型+api_base：通过chat_template_kwargs传递(vLLM)
            **kwargs: 额外配置参数（如azure_endpoint, api_version等）
        """
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )
        
        self.type = type.lower()
        self.enable_thinking = enable_thinking
        self._llm = None
        
        # 验证类型
        if self.type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"不支持的LLM类型: {self.type}。"
                f"支持的类型: {', '.join(self.SUPPORTED_TYPES)}"
            )
    
    def _get_llm(self):
        """获取LangChain LLM实例(懒加载)"""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm
    
    def _create_llm(self):
        """创建LangChain LLM实例
        
        根据type创建对应的LangChain Chat模型对象。
        """
        # 国内模型服务与本地服务直连，避免系统代理(如 Clash/v2rayN)
        # 引起 ProxyError('Unable to connect to proxy', SSLEOFError())
        bypass_proxy_for_domestic_hosts()
        
        if self.type == "openai":
            return self._create_openai_llm()
        elif self.type == "qwen":
            return self._create_qwen_llm()
        elif self.type == "azure":
            return self._create_azure_llm()
        elif self.type == "anthropic":
            return self._create_anthropic_llm()
        else:
            return self._create_openai_llm()
    
    def _create_openai_llm(self):
        """创建ChatOpenAI实例
        
        支持：
        - OpenAI官方API
        - vLLM等OpenAI兼容服务（通过api_base配置）
        - vLLM的thinking模式（通过extra_body传递chat_template_kwargs）
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "使用OpenAI类型需要安装langchain-openai包。"
                "请运行: pip install langchain-openai -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
        
        llm_kwargs = {
            "model": self.model,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        
        # 自定义API地址（用于vLLM等）
        if self.api_base:
            llm_kwargs["base_url"] = self.api_base
        
        # vLLM thinking模式支持
        # 当使用自定义api_base且启用thinking时，通过extra_body传递配置
        # vLLM需要启动时加 --enable-reasoning --reasoning-parser qwen3
        if self.api_base and self.enable_thinking:
            llm_kwargs["model_kwargs"] = {
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": True}
                }
            }
        
        return ChatOpenAI(**llm_kwargs)
    
    def _create_qwen_llm(self):
        """创建ChatTongyi实例（阿里云DashScope通义千问）
        
        原生支持enable_thinking深度思考模式。
        """
        try:
            from langchain_community.chat_models import ChatTongyi
        except ImportError:
            raise ImportError(
                "使用Qwen类型需要安装langchain-community包。"
                "请运行: pip install langchain-community dashscope -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
        
        llm_kwargs = {
            "model": self.model,
            "dashscope_api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # DashScope原生支持thinking模式
        if self.enable_thinking:
            llm_kwargs["model_kwargs"] = {"enable_thinking": True}
        
        return ChatTongyi(**llm_kwargs)
    
    def _create_azure_llm(self):
        """创建AzureChatOpenAI实例"""
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ImportError(
                "使用Azure类型需要安装langchain-openai包。"
                "请运行: pip install langchain-openai -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
        
        llm_kwargs = {
            "model": self.model,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # Azure特定配置
        if "azure_endpoint" in self.extra_config:
            llm_kwargs["azure_endpoint"] = self.extra_config["azure_endpoint"]
        if "api_version" in self.extra_config:
            llm_kwargs["api_version"] = self.extra_config["api_version"]
        
        return AzureChatOpenAI(**llm_kwargs)
    
    def _create_anthropic_llm(self):
        """创建ChatAnthropic实例"""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "使用Anthropic类型需要安装langchain-anthropic包。"
                "请运行: pip install langchain-anthropic -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
        
        return ChatAnthropic(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List:
        """转换消息格式为LangChain格式
        
        参数：
            messages: 标准消息列表 [{"role": "user", "content": "..."}]
            
        返回：
            LangChain消息对象列表
        """
        try:
            from langchain_core.messages import (
                AIMessage,
                HumanMessage,
                SystemMessage,
            )
        except ImportError:
            raise ImportError(
                "需要安装langchain-core包。"
                "请运行: pip install langchain-core -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
        
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                result.append(SystemMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            else:  # user
                result.append(HumanMessage(content=content))
        
        return result
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """异步生成文本补全
        
        参数：
            messages: 消息列表 [{"role": "user", "content": "..."}]
            **kwargs: 额外API参数
            
        返回：
            LLMResponse对象
        """
        llm = self._get_llm()
        langchain_messages = self._convert_messages(messages)
        
        start_time = time.time()
        
        try:
            response = await llm.ainvoke(langchain_messages)
        except Exception as e:
            self._handle_error(e)
        
        latency = time.time() - start_time
        
        return self._parse_response(response, latency)
    
    def complete_sync(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """同步生成文本补全
        
        参数：
            messages: 消息列表 [{"role": "user", "content": "..."}]
            **kwargs: 额外API参数
            
        返回：
            LLMResponse对象
        """
        llm = self._get_llm()
        langchain_messages = self._convert_messages(messages)
        
        start_time = time.time()
        
        try:
            response = llm.invoke(langchain_messages)
        except Exception as e:
            self._handle_error(e)
        
        latency = time.time() - start_time
        
        return self._parse_response(response, latency)
    
    def _parse_response(self, response: Any, latency: float) -> LLMResponse:
        """解析LangChain响应
        
        参数：
            response: LangChain响应对象
            latency: 请求延迟(秒)
            
        返回：
            LLMResponse对象
        """
        try:
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 提取Token用量信息
            usage = {}
            if hasattr(response, 'response_metadata'):
                resp_metadata = response.response_metadata
                if 'token_usage' in resp_metadata:
                    token_usage = resp_metadata['token_usage']
                    usage = {
                        "prompt_tokens": token_usage.get('prompt_tokens', 0),
                        "completion_tokens": token_usage.get('completion_tokens', 0),
                        "total_tokens": token_usage.get('total_tokens', 0),
                    }
            
            # 提取thinking内容（深度思考/推理内容）
            metadata = {}
            if hasattr(response, 'additional_kwargs'):
                additional = response.additional_kwargs
                # Qwen/vLLM的推理内容可能在不同字段
                for key in ['reasoning_content', 'thinking_content', 'thinking']:
                    if key in additional:
                        metadata["thinking_content"] = additional[key]
                        break
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency=latency,
                raw_response=response,
                metadata=metadata,
            )
        except Exception as e:
            raise LLMResponseError(f"解析响应失败: {e}")
    
    def _handle_error(self, error: Exception) -> None:
        """处理API错误
        
        将底层异常转换为语义化的LLM异常，并附带可操作的排查提示。
        
        参数：
            error: 原始异常
        """
        error_message = str(error)
        error_type = type(error).__name__
        lowered = error_message.lower()
        # 异常链上的类型名也要判断（如 ProxyError 包装的 SSLEOFError）
        chain_types = {error_type}
        cause = error.__cause__ or error.__context__
        while cause is not None:
            chain_types.add(type(cause).__name__)
            cause = cause.__cause__ or cause.__context__
        chain_text = " ".join(chain_types).lower()
        
        # 1) 代理问题：开启系统代理(如 Clash)后连不上国内模型服务
        if "proxyerror" in lowered or "proxy" in chain_text or "ssl" in chain_text:
            raise LLMConnectionError(
                f"网络代理异常({error_type}): {error_message}\n"
                f"提示: 检测到请求经过了系统代理且握手失败。请关闭系统代理，"
                f"或在 .env / 环境变量中把模型服务域名加入 NO_PROXY"
                f"(例如 NO_PROXY=api.siliconflow.cn,localhost,127.0.0.1)。"
            )
        
        # 2) 认证失败：API Key 无效 / 与服务商不匹配
        if (
            "invalid_api_key" in lowered
            or "invalidapikey" in lowered
            or "incorrect api key" in lowered
            or "status_code: 401" in lowered
            or "401" in lowered and ("api-key" in lowered or "api_key" in lowered or "apikey" in lowered)
            or "authentication" in lowered
            or "unauthorized" in lowered
        ):
            raise LLMAuthenticationError(
                f"认证失败({self.type}/{self.model}): {error_message}\n"
                f"提示: 请确认 api_key 与服务商匹配"
                f"(硅基流动的 key 只能用于 api.siliconflow.cn，"
                f"阿里百炼的 key 只能用于 dashscope.aliyuncs.com)，"
                f"且 key 未过期、未充值完。"
            )
        
        # 3) 模型名不存在（常见于换了服务商但没改模型名）
        if "model_not_found" in lowered or "invalid model" in lowered or "does not exist" in lowered:
            raise LLMResponseError(
                f"模型不可用({self.model}): {error_message}\n"
                f"提示: 模型名必须是该服务商模型列表中存在的完整名称，"
                f"例如硅基流动写作 Qwen/Qwen3-30B-A3B-Instruct-2507。"
            )
        
        # 4) 其余常规错误
        if "timeout" in lowered or "timed out" in lowered:
            raise LLMTimeoutError(f"请求超时: {error_message}")
        elif "rate" in lowered or "429" in lowered:
            raise LLMRateLimitError(f"速率限制: {error_message}")
        elif "auth" in lowered or "key" in lowered:
            raise LLMAuthenticationError(f"认证失败: {error_message}")
        else:
            raise LLMConnectionError(f"请求失败({error_type}): {error_message}")
