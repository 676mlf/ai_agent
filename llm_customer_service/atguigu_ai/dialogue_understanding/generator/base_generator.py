# -*- coding: utf-8 -*-
"""
命令生成器基类

定义命令生成器的抽象接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker
    from atguigu_ai.core.domain import Domain
    from atguigu_ai.dialogue_understanding.commands.base import Command


@dataclass
class GeneratorConfig:
    """生成器配置。
    
    Attributes:
        max_history_turns: 发送给LLM的最大历史轮数
        include_slots: 是否包含槽位信息
        include_flows: 是否包含Flow信息
        temperature: LLM采样温度
    """
    max_history_turns: int = 5
    include_slots: bool = True
    include_flows: bool = True
    temperature: float = 0.0


@dataclass
class GenerationResult:
    """生成结果。
    
    Attributes:
        commands: 生成的命令列表
        raw_output: LLM的原始输出
        prompt: 发送给LLM的提示词
        metadata: 额外的元数据
    """
    commands: List["Command"] = field(default_factory=list)
    raw_output: str = ""
    prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """是否成功生成了命令。"""
        return len(self.commands) > 0
    
    @property
    def first_command(self) -> Optional["Command"]:
        """返回第一个命令。"""
        return self.commands[0] if self.commands else None


class CommandGenerator(ABC):
    """命令生成器抽象基类。
    
    命令生成器负责将用户输入转换为系统可执行的命令。
    不同的实现可以使用不同的策略（LLM、规则、NLU等）。
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        """初始化生成器。
        
        Args:
            config: 生成器配置
        """
        self.config = config or GeneratorConfig()
    
    @abstractmethod
    async def generate(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        flows: Optional[List[Any]] = None,
    ) -> GenerationResult:
        """生成命令。
        
        Args:
            tracker: 对话状态追踪器
            domain: Domain定义
            flows: 可用的Flow列表
            
        Returns:
            生成结果
        """
        raise NotImplementedError()
    
    def generate_sync(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        flows: Optional[List[Any]] = None,
    ) -> GenerationResult:
        """同步版本的generate方法。
        
        Args:
            tracker: 对话状态追踪器
            domain: Domain定义
            flows: 可用的Flow列表
            
        Returns:
            生成结果
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.generate(tracker, domain, flows))
    
    @property
    def name(self) -> str:
        """返回生成器名称。"""
        return self.__class__.__name__


# 导出
__all__ = [
    "CommandGenerator",
    "GeneratorConfig",
    "GenerationResult",
]
