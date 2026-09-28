# -*- coding: utf-8 -*-
"""
消息处理器

负责处理用户消息的完整流程。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from atguigu_ai.agent.actions import get_action, ActionResult
from atguigu_ai.dialogue_understanding.generator import LLMCommandGenerator
from atguigu_ai.dialogue_understanding.processor import CommandProcessor
from atguigu_ai.policies import PolicyEnsemble, FlowPolicy, EnterpriseSearchPolicy
from atguigu_ai.core.tracker import DialogueStateTracker, UserMessage, BotMessage
from atguigu_ai.shared.constants import ACTION_LISTEN

if TYPE_CHECKING:
    from atguigu_ai.core.domain import Domain
    from atguigu_ai.dialogue_understanding.flow import FlowsList

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """处理器配置。
    
    Attributes:
        max_actions_per_turn: 每轮最大动作数
        enable_command_generation: 是否启用命令生成
    """
    max_actions_per_turn: int = 10
    enable_command_generation: bool = True


@dataclass
class MessageResponse:
    """消息响应。
    
    Attributes:
        messages: 机器人回复消息列表
        events: 产生的事件列表
        metadata: 额外元数据
    """
    messages: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, text: str, **kwargs: Any) -> None:
        """添加回复消息。"""
        message = {"text": text}
        message.update(kwargs)
        self.messages.append(message)


class MessageProcessor:
    """消息处理器。
    
    负责处理用户消息的完整流程：
    1. 接收用户消息
    2. 使用命令生成器生成命令
    3. 使用命令处理器处理命令
    4. 使用策略确定下一步动作
    5. 执行动作并返回响应
    """
    
    def __init__(
        self,
        domain: Optional["Domain"] = None,
        flows: Optional["FlowsList"] = None,
        policy_ensemble: Optional[PolicyEnsemble] = None,
        command_generator: Optional[LLMCommandGenerator] = None,
        config: Optional[ProcessorConfig] = None,
    ):
        """初始化消息处理器。
        
        Args:
            domain: Domain定义
            flows: Flow列表
            policy_ensemble: 策略集成器
            command_generator: 命令生成器
            config: 处理器配置
        """
        self.domain = domain
        self.flows = flows
        self.config = config or ProcessorConfig()
        
        # 初始化策略集成器
        if policy_ensemble:
            self.policy_ensemble = policy_ensemble
        else:
            self.policy_ensemble = PolicyEnsemble(policies=[
                FlowPolicy(flows=flows),
                EnterpriseSearchPolicy(),
            ])
        
        # 初始化命令生成器
        self.command_generator = command_generator
        
        # 初始化命令处理器
        self.command_processor = CommandProcessor(
            domain=domain,
            flows=flows.flows if flows else [],
        )
    
    async def process_message(
        self,
        message: str,
        tracker: DialogueStateTracker,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MessageResponse:
        """处理用户消息。
        
        Args:
            message: 用户消息文本
            tracker: 对话状态追踪器
            metadata: 消息元数据
            
        Returns:
            处理响应
        """
        response = MessageResponse()
        
        # 1. 创建用户消息并更新tracker
        user_message = UserMessage(
            text=message,
            sender_id=tracker.sender_id,
            metadata=metadata or {},
        )
        tracker.update_with_message(user_message)
        
        logger.info(f"Processing message from {tracker.sender_id}: {message}")
        
        try:
            # 2. 使用命令生成器生成命令（如果启用）
            if self.config.enable_command_generation and self.command_generator:
                generation_result = await self.command_generator.generate(
                    tracker, self.domain, self.flows.flows if self.flows else []
                )
                
                if generation_result.commands:
                    # 3. 使用命令处理器处理命令
                    process_result = self.command_processor.process(
                        generation_result.commands, tracker
                    )
                    response.events.extend(process_result.events)
                    response.metadata["commands"] = [
                        cmd.as_dict() for cmd in generation_result.commands
                    ]
            
            # 4. 使用策略确定下一步动作
            prediction = await self.policy_ensemble.predict(
                tracker, self.domain, self.flows
            )
            
            response.metadata["policy"] = prediction.policy_name
            response.metadata["action"] = prediction.action
            response.metadata["confidence"] = prediction.confidence
            
            # 5. 执行动作循环
            action_count = 0
            current_action = prediction.action
            
            while current_action and current_action != ACTION_LISTEN:
                if action_count >= self.config.max_actions_per_turn:
                    logger.warning("Max actions per turn reached")
                    break
                
                # 执行动作
                action_result = await self._execute_action(
                    current_action, tracker, prediction.metadata
                )
                
                # 收集响应
                for resp in action_result.responses:
                    response.messages.append(resp)
                    # 添加机器人消息到tracker
                    bot_message = BotMessage(
                        text=resp.get("text", ""),
                        data=resp,
                    )
                    tracker.add_bot_message(bot_message)
                
                response.events.extend(action_result.events)
                
                action_count += 1
                
                # 获取下一个动作
                prediction = await self.policy_ensemble.predict(
                    tracker, self.domain, self.flows
                )
                current_action = prediction.action
            
            logger.info(
                f"Processed message, generated {len(response.messages)} responses"
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            response.add_message("抱歉，处理您的消息时出现了问题。")
            response.metadata["error"] = str(e)
        
        return response
    
    async def _execute_action(
        self,
        action_name: str,
        tracker: DialogueStateTracker,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        """执行动作。
        
        Args:
            action_name: 动作名称
            tracker: 对话状态追踪器
            metadata: 动作元数据
            
        Returns:
            动作执行结果
        """
        logger.debug(f"Executing action: {action_name}")
        
        # 获取动作
        action = get_action(action_name)
        
        if action is None:
            logger.warning(f"Action not found: {action_name}")
            return ActionResult(success=False)
        
        # 合并元数据到kwargs
        kwargs = metadata or {}
        
        # 如果是闲聊动作，传递LLM配置
        if action_name == "action_chitchat_response" and self.command_generator:
            config = self.command_generator.config
            kwargs["llm_config"] = {
                "type": config.type,
                "model": config.model,
                "api_key": config.api_key,
                "api_base": config.api_base,
                "enable_thinking": config.enable_thinking,
            }
        
        # 执行动作
        try:
            result = await action.run(tracker, self.domain, **kwargs)
            tracker.latest_action_name = action_name
            return result
        except Exception as e:
            logger.error(f"Action {action_name} failed: {e}")
            return ActionResult(success=False)
    
    def set_domain(self, domain: "Domain") -> None:
        """设置Domain。"""
        self.domain = domain
        self.command_processor.set_domain(domain)
    
    def set_flows(self, flows: "FlowsList") -> None:
        """设置Flows。"""
        self.flows = flows
        self.command_processor.set_flows(flows.flows if flows else [])
        
        # 更新FlowPolicy
        for policy in self.policy_ensemble.policies:
            if isinstance(policy, FlowPolicy):
                policy.set_flows(flows)


# 导出
__all__ = [
    "MessageProcessor",
    "ProcessorConfig",
    "MessageResponse",
]
