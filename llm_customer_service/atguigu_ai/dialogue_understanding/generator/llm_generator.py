# -*- coding: utf-8 -*-
"""
LLM命令生成器

使用LLM将用户输入转换为命令。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from atguigu_ai.dialogue_understanding.generator.base_generator import (
    CommandGenerator,
    GeneratorConfig,
    GenerationResult,
)
from atguigu_ai.dialogue_understanding.generator.prompt_builder import PromptBuilder
from atguigu_ai.dialogue_understanding.generator.command_parser import (
    CommandParser,
    ParseResult,
)
from atguigu_ai.dialogue_understanding.commands.answer_commands import CannotHandleCommand
from atguigu_ai.dialogue_understanding.commands.error_commands import (
    ErrorCommand,
    InternalErrorCommand,
)
from atguigu_ai.shared.llm import create_llm_client
from atguigu_ai.shared.llm.base_client import LLMClient, LLMResponse
from atguigu_ai.shared.exceptions import LLMException

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker
    from atguigu_ai.core.domain import Domain
    from atguigu_ai.dialogue_understanding.commands.base import Command

logger = logging.getLogger(__name__)


@dataclass
class LLMGeneratorConfig(GeneratorConfig):
    """LLM生成器配置。
    
    Attributes:
        type: LLM类型 (openai, qwen, azure, anthropic)
        model: 模型名称
        api_key: API密钥
        api_base: 自定义API基础URL（用于vLLM等）
        temperature: 采样温度
        max_tokens: 最大生成token数
        timeout: 超时时间（秒）
        enable_thinking: 启用深度思考模式
    """
    type: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 256
    timeout: float = 30.0
    enable_thinking: bool = False


class LLMCommandGenerator(CommandGenerator):
    """LLM命令生成器。
    
    使用大语言模型将用户输入转换为对话系统命令。
    
    工作流程：
    1. 构建Prompt（包含上下文、槽位、Flow信息）
    2. 调用LLM生成命令文本
    3. 解析命令文本为命令对象
    """
    
    def __init__(
        self,
        config: Optional[LLMGeneratorConfig] = None,
        llm_client: Optional[LLMClient] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        command_parser: Optional[CommandParser] = None,
    ):
        """初始化LLM命令生成器。
        
        Args:
            config: 生成器配置
            llm_client: LLM客户端（如果不提供，则根据config创建）
            prompt_builder: Prompt构建器
            command_parser: 命令解析器
        """
        self.config = config or LLMGeneratorConfig()
        
        # 初始化LLM客户端
        if llm_client:
            self._llm_client = llm_client
        else:
            self._llm_client = None  # 延迟初始化
        
        # 初始化Prompt构建器
        self.prompt_builder = prompt_builder or PromptBuilder(
            max_history_turns=self.config.max_history_turns,
            include_slots=self.config.include_slots,
            include_flows=self.config.include_flows,
        )
        
        # 初始化命令解析器
        self.command_parser = command_parser or CommandParser()
    
    @property
    def llm_client(self) -> LLMClient:
        """获取LLM客户端（延迟初始化）。"""
        if self._llm_client is None:
            self._llm_client = create_llm_client(
                type=self.config.type,
                model=self.config.model,
                api_key=self.config.api_key,
                api_base=self.config.api_base,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                enable_thinking=self.config.enable_thinking,
            )
        return self._llm_client
    
    async def generate(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        flows: Optional[List[Any]] = None,
    ) -> GenerationResult:
        """使用LLM生成命令。
        
        Args:
            tracker: 对话状态追踪器
            domain: Domain定义
            flows: 可用的Flow列表
            
        Returns:
            生成结果
        """
        result = GenerationResult()
        
        # 检查是否有用户消息
        if not tracker.latest_message:
            logger.warning("No user message to process")
            result.commands = [CannotHandleCommand(reason="no_user_message")]
            return result
        
        try:
            # 1. 构建Prompt
            messages = self.prompt_builder.build_messages(tracker, domain, flows)
            result.prompt = self._format_messages_for_log(messages)
            
            logger.debug(f"Generated prompt with {len(messages)} messages")
            
            # 2. 调用LLM
            llm_response = await self.llm_client.complete(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            result.raw_output = llm_response.content
            result.metadata["llm_response"] = {
                "model": llm_response.model,
                "usage": llm_response.usage,
                "latency": llm_response.latency,
            }
            
            logger.info(f"LLM response: {llm_response.content}")
            
            # 3. 解析命令
            parse_result = self.command_parser.parse(llm_response.content)
            result.commands = parse_result.commands
            
            if parse_result.errors:
                result.metadata["parse_errors"] = parse_result.errors
                logger.warning(f"Parse errors: {parse_result.errors}")
            
            # 4. 如果没有解析出命令，返回cannot_handle
            if not result.commands:
                logger.warning("No commands parsed from LLM output")
                result.commands = [CannotHandleCommand(reason="parse_failed")]
            
        except LLMException as e:
            logger.error(f"LLM error: {e}")
            result.commands = [InternalErrorCommand(
                exception_type=e.__class__.__name__,
                exception_message=str(e),
            )]
            result.metadata["error"] = str(e)
            
        except Exception as e:
            logger.error(f"Unexpected error in command generation: {e}")
            result.commands = [ErrorCommand(
                error_type="generation_error",
                message=str(e),
            )]
            result.metadata["error"] = str(e)
        
        return result
    
    def _format_messages_for_log(self, messages: List[Dict[str, str]]) -> str:
        """格式化消息用于日志。"""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # 截断长内容
            if len(content) > 500:
                content = content[:500] + "..."
            parts.append(f"[{role}]: {content}")
        return "\n".join(parts)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMCommandGenerator":
        """从配置字典创建生成器。
        
        Args:
            config: 配置字典
            
        Returns:
            LLMCommandGenerator实例
        """
        generator_config = LLMGeneratorConfig(
            provider=config.get("provider", "openai"),
            model=config.get("model", "gpt-4o-mini"),
            api_key=config.get("api_key"),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 256),
            timeout=config.get("timeout", 30.0),
            max_history_turns=config.get("max_history_turns", 5),
            include_slots=config.get("include_slots", True),
            include_flows=config.get("include_flows", True),
        )
        return cls(config=generator_config)


# 导出
__all__ = [
    "LLMCommandGenerator",
    "LLMGeneratorConfig",
]
