# -*- coding: utf-8 -*-
"""
图构建器

负责构建 LangGraph 消息处理图。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from atguigu_ai.agent.graph.state import MessageProcessingState
from atguigu_ai.agent.graph.nodes import (
    understand_node,
    policy_node,
    action_node,
    response_node,
    guard_node,
)
from atguigu_ai.agent.graph.edges import (
    should_execute_action,
    should_continue,
)

if TYPE_CHECKING:
    from atguigu_ai.dialogue_understanding.generator import LLMCommandGenerator
    from atguigu_ai.dialogue_understanding.processor import CommandProcessor
    from atguigu_ai.policies import PolicyEnsemble

logger = logging.getLogger(__name__)


def build_message_processing_graph() -> CompiledStateGraph:
    """构建消息处理图。
    
    构建一个 LangGraph StateGraph，用于编排消息处理流程：
    
    图结构:
    
        START → understand → policy → [route] → action → guard → [route] → ...
                                        ↓                           ↓
                                     response ← ← ← ← ← ← ← ← ← ← ←
                                        ↓
                                       END
    
    Returns:
        编译后的图
    """
    logger.info("构建消息处理图...")
    
    # 创建状态图
    graph = StateGraph(MessageProcessingState)
    
    # 添加节点
    graph.add_node("understand", understand_node)
    graph.add_node("policy", policy_node)
    graph.add_node("action", action_node)
    graph.add_node("guard", guard_node)
    graph.add_node("response", response_node)
    
    # 设置入口边
    graph.add_edge(START, "understand")
    
    # understand → policy
    graph.add_edge("understand", "policy")
    
    # policy → [条件边] → action 或 response
    graph.add_conditional_edges(
        "policy",
        should_execute_action,
        {
            "action": "action",
            "response": "response",
        }
    )
    
    # action → guard
    graph.add_edge("action", "guard")
    
    # guard → [条件边] → policy 或 response
    graph.add_conditional_edges(
        "guard",
        should_continue,
        {
            "policy": "policy",
            "response": "response",
        }
    )
    
    # response → END
    graph.add_edge("response", END)
    
    # 编译图
    compiled_graph = graph.compile()
    
    logger.info("消息处理图构建完成")
    
    return compiled_graph


# 全局图实例（惰性初始化）
_graph_instance: CompiledStateGraph | None = None


def get_message_processing_graph() -> CompiledStateGraph:
    """获取消息处理图单例。
    
    返回全局共享的图实例，避免重复构建。
    
    Returns:
        编译后的图
    """
    global _graph_instance
    
    if _graph_instance is None:
        _graph_instance = build_message_processing_graph()
    
    return _graph_instance


def reset_graph_instance() -> None:
    """重置图实例。
    
    主要用于测试场景，强制重新构建图。
    """
    global _graph_instance
    _graph_instance = None


# 导出
__all__ = [
    "build_message_processing_graph",
    "get_message_processing_graph",
    "reset_graph_instance",
]
