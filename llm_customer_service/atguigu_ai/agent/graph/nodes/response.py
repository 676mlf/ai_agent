# -*- coding: utf-8 -*-
"""
响应节点

负责收集最终响应并标记处理完成。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from atguigu_ai.agent.graph.state import MessageProcessingState

logger = logging.getLogger(__name__)


async def response_node(state: "MessageProcessingState") -> Dict[str, Any]:
    """响应节点：收集响应并完成处理。
    
    这是消息处理流程的最后一个节点，负责：
    1. 确认处理已完成
    2. 记录最终状态
    
    Args:
        state: 当前图状态
        
    Returns:
        状态更新字典
    """
    final_responses = state.get("final_responses", [])
    action_count = state.get("action_count", 0)
    error = state.get("error")
    
    logger.info(
        f"[response_node] 处理完成, "
        f"共 {len(final_responses)} 个响应, "
        f"执行了 {action_count} 个动作"
    )
    
    if error:
        logger.warning(f"[response_node] 处理过程中出现错误: {error}")
    
    return {
        "is_finished": True,
        "node_history": state.get("node_history", []) + ["response"],
    }


# 导出
__all__ = ["response_node"]
