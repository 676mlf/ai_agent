# -*- coding: utf-8 -*-
"""
策略基类

定义所有策略的抽象接口和通用功能。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker
    from atguigu_ai.core.domain import Domain
    from atguigu_ai.dialogue_understanding.flow import FlowsList


@dataclass
class PolicyConfig:
    """策略配置。
    
    Attributes:
        priority: 策略优先级，数值越大优先级越高
        max_history: 考虑的最大历史轮数
    """
    priority: int = 1
    max_history: Optional[int] = None


@dataclass
class PolicyPrediction:
    """策略预测结果。
    
    Attributes:
        action: 预测的动作名称
        confidence: 预测置信度 (0.0 - 1.0)
        events: 附带的事件列表
        metadata: 额外的元数据
        policy_name: 产生此预测的策略名称
    """
    action: Optional[str] = None
    confidence: float = 0.0
    events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    policy_name: str = ""
    
    @property
    def is_abstain(self) -> bool:
        """策略是否放弃预测。"""
        return self.action is None or self.confidence == 0.0
    
    @classmethod
    def abstain(cls, policy_name: str = "") -> "PolicyPrediction":
        """创建一个放弃预测的结果。"""
        return cls(action=None, confidence=0.0, policy_name=policy_name)


class Policy(ABC):
    """策略基类。
    
    策略负责根据当前对话状态预测下一步应该执行的动作。
    不同的策略实现不同的决策逻辑：
    - FlowPolicy: 基于Flow定义执行
    - EnterpriseSearchPolicy: 基于知识库检索回答
    """
    
    # 默认优先级
    DEFAULT_PRIORITY = 1
    
    def __init__(
        self,
        config: Optional[PolicyConfig] = None,
        **kwargs: Any,
    ):
        """初始化策略。
        
        Args:
            config: 策略配置
            **kwargs: 额外配置
        """
        self.config = config or PolicyConfig()
        self._name = self.__class__.__name__
    
    @property
    def name(self) -> str:
        """策略名称。"""
        return self._name
    
    @property
    def priority(self) -> int:
        """策略优先级。"""
        return self.config.priority
    
    @abstractmethod
    async def predict(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        flows: Optional["FlowsList"] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """预测下一步动作。
        
        Args:
            tracker: 对话状态追踪器
            domain: Domain定义
            flows: Flow列表
            **kwargs: 额外参数
            
        Returns:
            预测结果
        """
        raise NotImplementedError()
    
    def predict_sync(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        flows: Optional["FlowsList"] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """同步版本的预测方法。"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.predict(tracker, domain, flows, **kwargs)
        )
    
    def should_predict(
        self,
        tracker: "DialogueStateTracker",
    ) -> bool:
        """检查策略是否应该进行预测。
        
        子类可以覆盖此方法以实现条件性预测。
        
        Args:
            tracker: 对话状态追踪器
            
        Returns:
            是否应该预测
        """
        return True
    
    def does_support_stack_frame(self, frame: Optional[Any] = None) -> bool:
        """检查策略是否支持处理当前栈帧。
        
        子类可以覆盖此方法以声明支持特定类型的栈帧。
        PolicyEnsemble可以使用此方法路由请求到合适的策略。
        
        Args:
            frame: 要检查的栈帧，如果为None则检查是否支持任何栈帧
            
        Returns:
            是否支持处理该栈帧
        """
        return True
    
    def train(
        self,
        training_data: Any,
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> None:
        """训练策略。
        
        大多数策略不需要训练，可以保持默认实现。
        
        Args:
            training_data: 训练数据
            domain: Domain定义
            **kwargs: 额外参数
        """
        pass
    
    def persist(self, path: str) -> None:
        """持久化策略到文件。
        
        Args:
            path: 保存路径
        """
        pass
    
    @classmethod
    def load(cls, path: str) -> "Policy":
        """从文件加载策略。
        
        Args:
            path: 加载路径
            
        Returns:
            策略实例
        """
        return cls()


# 导出
__all__ = [
    "Policy",
    "PolicyPrediction",
    "PolicyConfig",
]
