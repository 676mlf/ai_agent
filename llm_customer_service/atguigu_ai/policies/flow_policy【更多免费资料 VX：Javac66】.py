# -*- coding: utf-8 -*-
"""
Flow策略

基于Flow定义执行对话流程的策略。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from atguigu_ai.policies.base_policy import Policy, PolicyConfig, PolicyPrediction
from atguigu_ai.dialogue_understanding.flow import FlowExecutor, FlowsList
from atguigu_ai.dialogue_understanding.stack.stack_frame import FlowStackFrame
from atguigu_ai.shared.constants import ACTION_LISTEN

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker
    from atguigu_ai.core.domain import Domain

logger = logging.getLogger(__name__)


@dataclass
class FlowPolicyConfig(PolicyConfig):
    """Flow策略配置。
    
    Attributes:
        priority: 策略优先级（FlowPolicy优先级最高）
        max_steps_per_turn: 每轮最大执行步数（防止死循环）
    """
    priority: int = 100  # 高优先级
    max_steps_per_turn: int = 50


class FlowPolicy(Policy):
    """Flow策略。
    
    基于Flow定义执行对话流程。当对话栈中有活动的Flow时，
    此策略会根据Flow步骤决定下一步动作。
    
    工作流程：
    1. 检查是否有活动的Flow
    2. 获取当前步骤
    3. 执行步骤并确定动作
    4. 更新对话状态
    """
    
    DEFAULT_PRIORITY = 100
    
    def __init__(
        self,
        config: Optional[FlowPolicyConfig] = None,
        flows: Optional[FlowsList] = None,
        **kwargs: Any,
    ):
        """初始化Flow策略。
        
        Args:
            config: 策略配置
            flows: Flow列表
            **kwargs: 额外参数
        """
        super().__init__(config or FlowPolicyConfig(), **kwargs)
        self.flows = flows or FlowsList()
        self.executor = FlowExecutor(flows=self.flows)
    
    def should_predict(
        self,
        tracker: "DialogueStateTracker",
    ) -> bool:
        """检查是否应该预测。
        
        只有当对话栈中有活动Flow时才进行预测。
        """
        return tracker.active_flow is not None
    
    async def predict(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        flows: Optional[FlowsList] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """预测下一步动作。
        
        Args:
            tracker: 对话状态追踪器
            domain: Domain定义
            flows: Flow列表（覆盖初始化时的flows）
            **kwargs: 额外参数
            
        Returns:
            预测结果
        """
        # 如果没有活动Flow，放弃预测
        if not self.should_predict(tracker):
            logger.debug("No active flow, abstaining")
            return PolicyPrediction.abstain(self.name)
        
        # 使用提供的flows或初始化时的flows
        if flows:
            self.executor.set_flows(flows)
        
        # 检查是否有正在完成的 flow（上一轮执行了最后一个 action）
        flow_frame = tracker.dialogue_stack.top_flow_frame()
        if flow_frame and flow_frame.completing:
            completed_flow = flow_frame.flow_id
            logger.debug(f"Flow {completed_flow} completing, triggering action_flow_completed")
            # 重置 scoped slots
            self._reset_scoped_slots(tracker, completed_flow)
            # 结束 flow
            tracker.end_flow()
            return PolicyPrediction(
                action="action_flow_completed",
                confidence=1.0,
                events=[],
                policy_name=self.name,
                metadata={
                    "flow_completed": True,
                    "completed_flow": completed_flow,
                },
            )
        
        # 执行Flow（循环处理，直到需要用户输入或执行动作）
        try:
            max_iterations = self.config.max_steps_per_turn
            all_events = []
            
            for _ in range(max_iterations):
                result = self.executor.execute_next_step(tracker)
                all_events.extend(result.events)
                
                # 优先检查是否需要收集槽位（collect 步骤会同时设置 action 和 slot_to_collect）
                # 必须先处理 slot_to_collect，否则会走 action 分支而丢失 metadata 信息
                if result.slot_to_collect:
                    logger.debug(f"Collecting slot: {result.slot_to_collect}")
                    
                    # 更新FlowStackFrame的slot_to_collect属性
                    flow_frame = tracker.dialogue_stack.top_flow_frame()
                    if flow_frame:
                        flow_frame.slot_to_collect = result.slot_to_collect
                    
                    # 使用 collect 步骤指定的 action，或默认的 utter_ask_xxx
                    action = result.action or f"utter_ask_{result.slot_to_collect}"
                    
                    # 构建 metadata，包含 fallback_action 以支持 action_ask_xxx 降级
                    prediction_metadata = {
                        "slot_to_collect": result.slot_to_collect,
                        "next_step_id": result.next_step_id,
                    }
                    # 传递 fallback_action（如果存在）
                    if result.metadata.get("fallback_action"):
                        prediction_metadata["fallback_action"] = result.metadata["fallback_action"]
                    
                    return PolicyPrediction(
                        action=action,
                        confidence=1.0,
                        events=all_events,
                        policy_name=self.name,
                        metadata=prediction_metadata,
                    )
                
                # 如果有动作需要执行（非 collect 步骤的 action）
                if result.action:
                    logger.debug(f"Flow action: {result.action}")
                    
                    # 检查是否是最后一个动作（next 是 END）
                    is_final_action = (
                        result.flow_completed or 
                        (isinstance(result.next_step_id, str) and result.next_step_id.upper() == "END")
                    )
                    
                    if is_final_action:
                        completed_flow = tracker.active_flow
                        logger.debug(f"Flow {completed_flow} will complete after action")
                        # 设置completing标志，下一轮predict时触发action_flow_completed
                        flow_frame = tracker.dialogue_stack.top_flow_frame()
                        if flow_frame:
                            flow_frame.completing = True
                            # 清除 slot_to_collect，因为 flow 即将完成
                            flow_frame.slot_to_collect = None
                    elif result.next_step_id:
                        # 更新步骤
                        self.executor.advance_flow(tracker, result.next_step_id)
                        
                        # 检查下一步是否是 collect 步骤，如果是则预设 slot_to_collect
                        # 这样 understand_node 在下一轮可以获取到 current_slot 信息
                        self._preset_slot_to_collect_if_needed(
                            tracker, result.next_step_id
                        )
                    
                    return PolicyPrediction(
                        action=result.action,
                        confidence=1.0,
                        events=all_events,
                        policy_name=self.name,
                        metadata=self._build_action_metadata(
                            tracker, result.next_step_id, is_final_action
                        ),
                    )
                
                # 如果Flow已完成且没有动作，触发 action_flow_completed
                if result.flow_completed:
                    completed_flow = tracker.active_flow
                    logger.debug(f"Flow {completed_flow} completed")
                    # 在 end_flow 之前重置槽位
                    if completed_flow:
                        self._reset_scoped_slots(tracker, completed_flow)
                    tracker.end_flow()
                    return PolicyPrediction(
                        action="action_flow_completed",
                        confidence=1.0,
                        events=all_events,
                        policy_name=self.name,
                        metadata={
                            "flow_completed": True,
                            "completed_flow": completed_flow,
                        },
                    )
                
                # 如果有下一步但没有动作（slot已填充的情况），推进并继续执行
                if result.next_step_id:
                    logger.debug(f"Slot filled, advancing to step: {result.next_step_id}")
                    # 清除slot_to_collect，因为slot已填充
                    flow_frame = tracker.dialogue_stack.top_flow_frame()
                    if flow_frame:
                        flow_frame.slot_to_collect = None
                    self.executor.advance_flow(tracker, result.next_step_id)
                    continue
                
                # 没有动作也没有下一步，等待用户输入
                break
            
            return PolicyPrediction(
                action=ACTION_LISTEN,
                confidence=1.0,
                events=all_events,
                policy_name=self.name,
            )
            
        except Exception as e:
            logger.error(f"Error executing flow: {e}")
            return PolicyPrediction(
                action="action_default_fallback",
                confidence=0.5,
                policy_name=self.name,
                metadata={"error": str(e)},
            )
    
    def _build_action_metadata(
        self,
        tracker: "DialogueStateTracker",
        next_step_id: Optional[str],
        flow_completed: bool,
    ) -> Dict[str, Any]:
        """构建 action 步骤的 metadata。
        
        Args:
            tracker: 对话状态追踪器
            next_step_id: 下一步骤 ID
            flow_completed: Flow 是否已完成
            
        Returns:
            metadata 字典
        """
        metadata: Dict[str, Any] = {
            "next_step_id": next_step_id,
            "flow_completed": flow_completed,
        }
        
        # 注意：不在这里添加 slot_to_collect
        # slot_to_collect 只应该在 collect 步骤返回时添加到 metadata
        # 如果在 action 步骤也添加，会导致 action_node 误以为当前是 collect 步骤而提前终止
        
        return metadata
    
    def _preset_slot_to_collect_if_needed(
        self,
        tracker: "DialogueStateTracker",
        next_step_id: str,
    ) -> None:
        """预设 slot_to_collect（如果下一步是 collect 步骤）。
        
        当执行完 action 步骤后推进到 collect 步骤时，
        预先设置 slot_to_collect，这样下一轮 understand_node 
        可以在 prompt 中告诉 LLM 当前正在收集哪个槽位。
        
        Args:
            tracker: 对话状态追踪器
            next_step_id: 下一步骤 ID
        """
        flow_id = tracker.active_flow
        if not flow_id:
            return
        
        flow = self.executor.flows.get_flow(flow_id)
        if not flow:
            return
        
        next_step = flow.get_step(next_step_id)
        if not next_step:
            return
        
        # 检查下一步是否是 collect 步骤
        from atguigu_ai.dialogue_understanding.flow.flow import StepType
        if next_step.step_type == StepType.COLLECT and next_step.collect:
            flow_frame = tracker.dialogue_stack.top_flow_frame()
            if flow_frame:
                flow_frame.slot_to_collect = next_step.collect
                logger.debug(
                    f"预设 slot_to_collect: {next_step.collect} "
                    f"（下一步: {next_step_id}）"
                )
    
    def _reset_scoped_slots(
        self,
        tracker: "DialogueStateTracker",
        flow_id: str,
    ) -> None:
        """重置 flow 作用域内的槽位。
        
        当 flow 结束时，重置在该 flow 中收集的槽位（除非配置了 reset_after_flow_ends=False
        或者槽位在 persisted_slots 列表中）。
        
        这确保了不同 flow 之间的槽位隔离，避免一个 flow 设置的槽位被下一个 flow 错误地复用。
        
        Args:
            tracker: 对话状态追踪器
            flow_id: 结束的 flow ID
        """
        flow = self.executor.flows.get_flow(flow_id)
        if not flow:
            logger.warning(f"[FlowPolicy] 未找到 flow: {flow_id}，跳过槽位重置")
            return
        
        logger.debug(f"[FlowPolicy] Flow {flow_id} 结束，开始重置 scoped slots")
        
        persisted_slots = set(flow.persisted_slots)
        not_resettable_slots = set()
        
        # 遍历 flow 中的 collect 步骤
        from atguigu_ai.dialogue_understanding.flow.flow import StepType
        for step in flow.steps:
            if step.step_type == StepType.COLLECT and step.collect:
                slot_name = step.collect
                # 检查是否需要重置
                if step.reset_after_flow_ends and slot_name not in persisted_slots:
                    # 重置槽位
                    if slot_name in tracker.slots:
                        old_value = tracker.slots[slot_name].value
                        tracker.slots[slot_name].reset()
                        logger.debug(f"[FlowPolicy] 重置槽位 {slot_name}: {old_value} -> None")
                    else:
                        logger.debug(f"[FlowPolicy] 槽位 {slot_name} 不存在，跳过重置")
                else:
                    not_resettable_slots.add(slot_name)
                    logger.debug(
                        f"[FlowPolicy] 槽位 {slot_name} 不需要重置 "
                        f"(reset_after_flow_ends={step.reset_after_flow_ends}, "
                        f"persisted={slot_name in persisted_slots})"
                    )
        
        # 重置 set_slot 步骤设置的槽位（除非在 not_resettable_slots 或 persisted_slots 中）
        for step in flow.steps:
            if step.step_type == StepType.SET_SLOT and step.slot_name:
                slot_name = step.slot_name
                if slot_name not in not_resettable_slots and slot_name not in persisted_slots:
                    if slot_name in tracker.slots:
                        old_value = tracker.slots[slot_name].value
                        tracker.slots[slot_name].reset()
                        logger.debug(f"[FlowPolicy] 重置槽位 {slot_name} (set_slot): {old_value} -> None")
    
    def set_flows(self, flows: FlowsList) -> None:
        """设置Flow列表。
        
        Args:
            flows: Flow列表
        """
        self.flows = flows
        self.executor.set_flows(flows)


# 导出
__all__ = [
    "FlowPolicy",
    "FlowPolicyConfig",
]
