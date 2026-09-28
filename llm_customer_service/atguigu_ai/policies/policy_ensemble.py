# -*- coding: utf-8 -*-
"""
策略集成器

管理多个策略，按优先级选择最佳预测。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

from atguigu_ai.policies.base_policy import Policy, PolicyPrediction
from atguigu_ai.shared.constants import ACTION_LISTEN

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker
    from atguigu_ai.core.domain import Domain
    from atguigu_ai.dialogue_understanding.flow import FlowsList

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """集成器配置。
    
    Attributes:
        fallback_action: 所有策略都放弃时的默认动作
        min_confidence: 最小置信度阈值
    """
    fallback_action: str = ACTION_LISTEN
    min_confidence: float = 0.0


class PolicyEnsemble:
    """策略集成器。
    
    管理多个策略，按优先级顺序执行预测，选择最佳结果。
    
    策略选择逻辑：
    1. 按优先级从高到低遍历策略
    2. 选择第一个非放弃且置信度最高的预测
    3. 如果所有策略都放弃，返回默认动作
    """
    
    def __init__(
        self,
        policies: Optional[List[Policy]] = None,
        config: Optional[EnsembleConfig] = None,
    ):
        """初始化策略集成器。
        
        Args:
            policies: 策略列表
            config: 集成器配置
        """
        self.policies = policies or []
        self.config = config or EnsembleConfig()
        
        # 按优先级排序（从高到低）
        self._sort_policies()
    
    def _sort_policies(self) -> None:
        """按优先级排序策略。"""
        self.policies.sort(key=lambda p: p.priority, reverse=True)
    
    def add_policy(self, policy: Policy) -> None:
        """添加策略。
        
        Args:
            policy: 要添加的策略
        """
        self.policies.append(policy)
        self._sort_policies()
    
    def remove_policy(self, policy_name: str) -> Optional[Policy]:
        """移除策略。
        
        Args:
            policy_name: 策略名称
            
        Returns:
            被移除的策略，如果不存在则返回None
        """
        for i, policy in enumerate(self.policies):
            if policy.name == policy_name:
                return self.policies.pop(i)
        return None
    
    async def predict(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        flows: Optional["FlowsList"] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """使用策略集成进行预测。
        
        按优先级顺序尝试每个策略，返回最佳预测。
        
        Args:
            tracker: 对话状态追踪器
            domain: Domain定义
            flows: Flow列表
            **kwargs: 额外参数
            
        Returns:
            最佳预测结果
        """
        best_prediction: Optional[PolicyPrediction] = None
        all_predictions: List[PolicyPrediction] = []
        
        for policy in self.policies:
            # 检查策略是否应该预测
            if not policy.should_predict(tracker):
                logger.debug(f"Policy {policy.name} skipped (should_predict=False)")
                continue
            
            try:
                prediction = await policy.predict(
                    tracker, domain, flows, **kwargs
                )
                all_predictions.append(prediction)
                
                logger.debug(
                    f"Policy {policy.name} predicted: "
                    f"action={prediction.action}, confidence={prediction.confidence}"
                )
                
                # 如果策略给出了非放弃的预测
                if not prediction.is_abstain:
                    # 检查置信度是否满足阈值
                    if prediction.confidence >= self.config.min_confidence:
                        # 选择置信度最高的
                        if best_prediction is None or \
                           prediction.confidence > best_prediction.confidence:
                            best_prediction = prediction
                        
                        # 如果置信度为1.0，直接返回
                        if prediction.confidence >= 1.0:
                            break
                            
            except Exception as e:
                logger.error(f"Policy {policy.name} error: {e}")
                continue
        
        # 如果有有效预测，返回最佳预测
        if best_prediction is not None:
            logger.info(
                f"Selected prediction from {best_prediction.policy_name}: "
                f"action={best_prediction.action}"
            )
            return best_prediction
        
        # 所有策略都放弃，返回默认动作
        logger.debug("All policies abstained, using fallback action")
        return PolicyPrediction(
            action=self.config.fallback_action,
            confidence=0.0,
            policy_name="PolicyEnsemble",
            metadata={"fallback": True},
        )
    
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
    
    def get_policy(self, name: str) -> Optional[Policy]:
        """根据名称获取策略。
        
        Args:
            name: 策略名称
            
        Returns:
            策略实例，如果不存在则返回None
        """
        for policy in self.policies:
            if policy.name == name:
                return policy
        return None
    
    @property
    def policy_names(self) -> List[str]:
        """获取所有策略名称。"""
        return [p.name for p in self.policies]
    
    def train_all(
        self,
        training_data: Any,
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> None:
        """训练所有策略。
        
        Args:
            training_data: 训练数据
            domain: Domain定义
            **kwargs: 额外参数
        """
        for policy in self.policies:
            try:
                policy.train(training_data, domain, **kwargs)
                logger.info(f"Trained policy: {policy.name}")
            except Exception as e:
                logger.error(f"Failed to train {policy.name}: {e}")


# 便捷函数

def create_default_ensemble() -> PolicyEnsemble:
    """创建默认的策略集成器。
    
    包含FlowPolicy和EnterpriseSearchPolicy。
    
    Returns:
        PolicyEnsemble实例
    """
    from atguigu_ai.policies.flow_policy import FlowPolicy
    from atguigu_ai.policies.enterprise_search_policy import EnterpriseSearchPolicy
    
    return PolicyEnsemble(policies=[
        FlowPolicy(),
        EnterpriseSearchPolicy(),
    ])


# 导出
__all__ = [
    "PolicyEnsemble",
    "EnsembleConfig",
    "create_default_ensemble",
]
