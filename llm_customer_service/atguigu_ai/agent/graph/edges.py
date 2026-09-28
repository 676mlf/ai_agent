# -*- coding: utf-8 -*-
"""
条件边路由函数

定义 LangGraph 图中的条件边路由逻辑。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal

from atguigu_ai.shared.constants import ACTION_LISTEN

logger = logging.getLogger(__name__)


def should_execute_action(state: Dict[str, Any]) -> Literal["action", "response"]:
    """决定是执行动作还是返回响应。
    
    在 policy_node 之后调用，根据预测结果决定下一步：
    - 如果 is_finished 为 True，或动作为 action_listen，则跳转到 response_node
    - 否则跳转到 action_node 执行动作
    
    Args:
        state: 当前图状态 (MessageProcessingState)
        
    Returns:
        下一个节点名称
    """
    is_finished = state.get("is_finished", False)
    current_prediction = state.get("current_prediction")
    
    # 检查是否已完成
    if is_finished:
        logger.debug("[should_execute_action] 已完成，跳转到 response")
        return "response"
    
    # 检查动作是否为 action_listen
    if current_prediction:
        action = current_prediction.action
        if action == ACTION_LISTEN or action is None:
            logger.debug(f"[should_execute_action] 动作为 {action}，跳转到 response")
            return "response"
    else:
        # 没有预测结果，跳转到响应
        logger.debug("[should_execute_action] 没有预测结果，跳转到 response")
        return "response"
    
    logger.debug("[should_execute_action] 跳转到 action 执行动作")
    return "action"


def should_continue(state: Dict[str, Any]) -> Literal["policy", "response"]:
    """决定是继续循环还是结束。
    
    在 guard_node 之后调用，根据状态决定是否继续：
    - 如果 is_finished 为 True，或达到最大动作数，则跳转到 response_node
    - 否则跳转回 policy_node 继续决策
    
    Args:
        state: 当前图状态 (MessageProcessingState)
        
    Returns:
        下一个节点名称
    """
    is_finished = state.get("is_finished", False)
    action_count = state.get("action_count", 0)
    max_actions = state.get("max_actions", 10)
    
    logger.debug(
        f"[should_continue] is_finished={is_finished}, "
        f"action_count={action_count}/{max_actions}"
    )
    
    # 检查是否已完成
    if is_finished:
        logger.debug("[should_continue] 已完成，跳转到 response")
        return "response"
    
    # 检查是否达到最大动作数
    if action_count >= max_actions:
        logger.debug(
            f"[should_continue] 达到最大动作数 ({action_count}/{max_actions})，"
            "跳转到 response"
        )
        return "response"
    
    logger.debug("[should_continue] 继续循环，跳转到 policy")
    return "policy"


# 导出
__all__ = [
    "should_execute_action",
    "should_continue",
]
