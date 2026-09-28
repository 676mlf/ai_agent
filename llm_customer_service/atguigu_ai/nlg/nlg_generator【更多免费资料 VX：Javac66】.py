# -*- coding: utf-8 -*-
"""
NLG生成器基类

定义自然语言生成的抽象接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker
    from atguigu_ai.core.domain import Domain


@dataclass
class NLGConfig:
    """NLG配置。
    
    Attributes:
        use_llm: 是否使用LLM生成
        llm_type: LLM类型 (openai/qwen/azure/anthropic)
        llm_model: LLM模型
    """
    use_llm: bool = False
    llm_type: str = "openai"
    llm_model: str = "gpt-4o-mini"


@dataclass
class NLGResponse:
    """NLG响应。
    
    Attributes:
        text: 生成的文本
        buttons: 按钮列表
        image: 图片URL
        custom: 自定义数据
        metadata: 元数据
    """
    text: str = ""
    buttons: List[Dict[str, Any]] = field(default_factory=list)
    image: Optional[str] = None
    custom: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def as_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        result = {"text": self.text}
        if self.buttons:
            result["buttons"] = self.buttons
        if self.image:
            result["image"] = self.image
        if self.custom:
            result["custom"] = self.custom
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class NLGGenerator(ABC):
    """NLG生成器抽象基类。
    
    负责根据动作名称或模板生成机器人回复。
    """
    
    def __init__(self, config: Optional[NLGConfig] = None):
        """初始化生成器。
        
        Args:
            config: NLG配置
        """
        self.config = config or NLGConfig()
    
    @abstractmethod
    async def generate(
        self,
        utter_action: str,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> NLGResponse:
        """生成回复。
        
        Args:
            utter_action: 响应动作名称（如utter_greet）
            tracker: 对话状态追踪器
            domain: Domain定义
            **kwargs: 额外参数
            
        Returns:
            NLG响应
        """
        raise NotImplementedError()
    
    def generate_sync(
        self,
        utter_action: str,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> NLGResponse:
        """同步版本的生成方法。"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.generate(utter_action, tracker, domain, **kwargs)
        )


# 导出
__all__ = [
    "NLGGenerator",
    "NLGConfig",
    "NLGResponse",
]
