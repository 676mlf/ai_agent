# -*- coding: utf-8 -*-
"""
策略节点

负责调用 PolicyEnsemble 预测下一个动作。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, TYPE_CHECKING

from atguigu_ai.policies.base_policy import PolicyPrediction
from atguigu_ai.shared.constants import ACTION_LISTEN

if TYPE_CHECKING:
    from atguigu_ai.agent.graph.state import MessageProcessingState

logger = logging.getLogger(__name__)


async def policy_node(state: "MessageProcessingState") -> Dict[str, Any]:
    """策略节点：预测下一个动作。
    
    该节点调用 PolicyEnsemble.predict() 来决定系统应该执行什么动作。
    如果 CommandProcessor 已经确定了 next_action，优先使用该动作（仅第一轮）。
    如果预测的动作是 action_listen，则标记处理完成。
    
    Args:
        state: 当前图状态
        
    Returns:
        状态更新字典
    """
    tracker = state["tracker"]
    domain = state.get("domain")
    flows = state.get("flows")
    policy_ensemble = state.get("_policy_ensemble")
    process_result = state.get("process_result")
    action_count = state.get("action_count", 0)
    
    logger.debug("[policy_node] 开始策略预测")
    
    # 默认预测结果
    current_prediction = None
    is_finished = True  # 如果没有策略集成器，默认结束
    
    try:
        # 优先使用 CommandProcessor 确定的 next_action（仅第一轮，action_count == 0）
        if action_count == 0 and process_result and process_result.next_action:
            next_action = process_result.next_action
            # 跳过 action_run_flow_* 类型的动作，这些由 FlowPolicy 处理
            if not next_action.startswith("action_run_flow_"):
                current_prediction = PolicyPrediction(
                    action=next_action,
                    confidence=1.0,
                    policy_name="CommandProcessor",
                    metadata=process_result.metadata,
                )
                is_finished = (next_action == ACTION_LISTEN)
                
                logger.info(
                    f"[policy_node] 使用 CommandProcessor 动作: {next_action}, "
                    f"是否结束: {is_finished}"
                )
                
                return {
                    "current_prediction": current_prediction,
                    "is_finished": is_finished,
                    "node_history": state.get("node_history", []) + ["policy"],
                }
        
        # 否则使用 PolicyEnsemble 预测
        if policy_ensemble:
            prediction = await policy_ensemble.predict(tracker, domain, flows)
            current_prediction = prediction
            
            # 检查是否应该结束（action_listen 表示等待用户输入）
            is_finished = (prediction.action == ACTION_LISTEN or prediction.action is None)
            
            logger.info(
                f"[policy_node] 预测动作: {prediction.action}, "
                f"置信度: {prediction.confidence:.2f}, "
                f"是否结束: {is_finished}"
            )
        else:
            logger.warning("[policy_node] 未配置策略集成器，跳过预测")
            
    except Exception as e:
        logger.error(f"[policy_node] 预测失败: {e}")
        return {
            "current_prediction": None,
            "is_finished": True,
            "error": str(e),
            "node_history": state.get("node_history", []) + ["policy"],
        }
    
    return {
        "current_prediction": current_prediction,
        "is_finished": is_finished,
        "node_history": state.get("node_history", []) + ["policy"],
    }


# 导出
__all__ = ["policy_node"]
