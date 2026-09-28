# -*- coding: utf-8 -*-
"""
图式编排模块

基于 LangGraph 实现的消息处理图，用于编排 LLMCommandGenerator、
CommandProcessor、PolicyEnsemble、Action 等核心组件的执行流程。

主要组件：
- MessageProcessingState: 图状态定义
- build_message_processing_graph: 图构建器
- 节点函数: understand_node, policy_node, action_node, response_node, guard_node
- 条件边: should_execute_action, should_continue
"""

from atguigu_ai.agent.graph.state import (
    MessageProcessingState,
    create_initial_state,
)
from atguigu_ai.agent.graph.builder import (
    build_message_processing_graph,
    get_message_processing_graph,
    reset_graph_instance,
)
from atguigu_ai.agent.graph.edges import (
    should_execute_action,
    should_continue,
)
from atguigu_ai.agent.graph.nodes import (
    understand_node,
    policy_node,
    action_node,
    response_node,
    guard_node,
)

__all__ = [
    # 状态
    "MessageProcessingState",
    "create_initial_state",
    # 构建器
    "build_message_processing_graph",
    "get_message_processing_graph",
    "reset_graph_instance",
    # 边
    "should_execute_action",
    "should_continue",
    # 节点
    "understand_node",
    "policy_node",
    "action_node",
    "response_node",
    "guard_node",
]
