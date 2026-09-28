# -*- coding: utf-8 -*-
"""
图状态定义

定义 LangGraph 消息处理图的状态结构。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class MessageProcessingState(TypedDict, total=False):
    """消息处理图的状态定义。
    
    这是 LangGraph StateGraph 的核心状态结构，包含：
    - 核心对话状态（tracker, domain, flows）
    - 输入输出数据
    - 流程控制字段
    - 中间结果缓存
    - 组件引用（用于节点访问）
    
    注意：为了兼容 LangGraph 的运行时类型解析，复杂对象类型使用 Any。
    
    Attributes:
        tracker: 对话状态追踪器 (DialogueStateTracker)
        domain: Domain定义
        flows: Flow列表 (FlowsList)
        input_message: 用户输入消息
        metadata: 消息元数据
        final_responses: 累积的响应列表
        is_finished: 是否已完成处理
        action_count: 已执行的动作计数
        max_actions: 最大动作数限制
        current_commands: 当前生成的命令结果 (GenerationResult)
        current_prediction: 当前策略预测结果 (PolicyPrediction)
        current_action_result: 当前动作执行结果 (ActionResult)
        node_history: 执行过的节点历史
        error: 错误信息
        _command_generator: 命令生成器引用 (LLMCommandGenerator)
        _command_processor: 命令处理器引用 (CommandProcessor)
        _policy_ensemble: 策略集成器引用 (PolicyEnsemble)
    """
    # 核心状态（使用 Any 以兼容 LangGraph 运行时类型解析）
    tracker: Any  # DialogueStateTracker
    domain: Any  # Optional[Domain]
    flows: Any  # Optional[FlowsList]
    
    # 输入输出
    input_message: str
    metadata: Dict[str, Any]
    final_responses: List[Dict[str, Any]]
    
    # 流程控制
    is_finished: bool
    action_count: int
    max_actions: int
    
    # 中间结果
    current_commands: Any  # Optional[GenerationResult]
    current_prediction: Any  # Optional[PolicyPrediction]
    current_action_result: Any  # Optional[ActionResult]
    
    # 调试信息
    node_history: List[str]
    error: Optional[str]
    
    # 组件引用（内部使用，以 _ 开头）
    _command_generator: Any  # Optional[LLMCommandGenerator]
    _command_processor: Any  # Optional[CommandProcessor]
    _policy_ensemble: Any  # Optional[PolicyEnsemble]


def create_initial_state(
    tracker: Any,
    input_message: str,
    domain: Any = None,
    flows: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_actions: int = 10,
    command_generator: Any = None,
    command_processor: Any = None,
    policy_ensemble: Any = None,
) -> MessageProcessingState:
    """创建初始状态。
    
    Args:
        tracker: 对话状态追踪器 (DialogueStateTracker)
        input_message: 用户输入消息
        domain: Domain定义
        flows: Flow列表 (FlowsList)
        metadata: 消息元数据
        max_actions: 最大动作数限制
        command_generator: 命令生成器 (LLMCommandGenerator)
        command_processor: 命令处理器 (CommandProcessor)
        policy_ensemble: 策略集成器 (PolicyEnsemble)
        
    Returns:
        初始化的状态字典
    """
    return MessageProcessingState(
        # 核心状态
        tracker=tracker,
        domain=domain,
        flows=flows,
        # 输入输出
        input_message=input_message,
        metadata=metadata or {},
        final_responses=[],
        # 流程控制
        is_finished=False,
        action_count=0,
        max_actions=max_actions,
        # 中间结果
        current_commands=None,
        current_prediction=None,
        current_action_result=None,
        # 调试信息
        node_history=[],
        error=None,
        # 组件引用
        _command_generator=command_generator,
        _command_processor=command_processor,
        _policy_ensemble=policy_ensemble,
    )


# 导出
__all__ = [
    "MessageProcessingState",
    "create_initial_state",
]
