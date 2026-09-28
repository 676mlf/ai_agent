# -*- coding: utf-8 -*-
"""
保护节点

负责检查循环次数，防止无限循环。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from atguigu_ai.agent.graph.state import MessageProcessingState

logger = logging.getLogger(__name__)


async def guard_node(state: "MessageProcessingState") -> Dict[str, Any]:
    """保护节点：检查循环次数。
    
    该节点在每次动作执行后检查是否达到最大循环次数，
    如果达到则强制终止处理。
    
    Args:
        state: 当前图状态
        
    Returns:
        状态更新字典
    """
    action_count = state.get("action_count", 0)
    max_actions = state.get("max_actions", 10)
    
    if action_count >= max_actions:
        logger.warning(
            f"[guard_node] 达到最大动作数限制 ({action_count}/{max_actions})，强制终止"
        )
        return {
            "is_finished": True,
            "error": f"达到最大动作数限制: {max_actions}",
            "node_history": state.get("node_history", []) + ["guard"],
        }
    
    logger.debug(f"[guard_node] 动作计数: {action_count}/{max_actions}")
    
    return {
        "node_history": state.get("node_history", []) + ["guard"],
    }


# 导出
__all__ = ["guard_node"]
