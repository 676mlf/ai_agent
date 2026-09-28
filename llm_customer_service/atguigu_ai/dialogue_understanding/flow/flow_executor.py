# -*- coding: utf-8 -*-
"""
Flow执行器

负责执行Flow中的步骤。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from atguigu_ai.dialogue_understanding.flow.flow import Flow, FlowStep, FlowsList, StepType
from atguigu_ai.dialogue_understanding.stack.stack_frame import FlowStackFrame

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker
    from atguigu_ai.core.domain import Domain

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """执行结果。
    
    Attributes:
        action: 要执行的动作
        slot_to_collect: 要收集的槽位
        events: 产生的事件
        flow_completed: Flow是否已完成
        next_step_id: 下一个步骤ID
    """
    action: Optional[str] = None
    slot_to_collect: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    flow_completed: bool = False
    next_step_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FlowExecutor:
    """Flow执行器。
    
    负责执行Flow中的步骤，管理Flow的状态转换。
    
    工作流程：
    1. 获取当前Flow和步骤
    2. 根据步骤类型执行相应操作
    3. 确定下一个步骤
    4. 更新状态
    """
    
    def __init__(
        self,
        flows: Optional[FlowsList] = None,
        domain: Optional["Domain"] = None,
    ):
        """初始化执行器。
        
        Args:
            flows: Flow列表
            domain: Domain定义
        """
        self.flows = flows or FlowsList()
        self.domain = domain
    
    def execute_next_step(
        self,
        tracker: "DialogueStateTracker",
    ) -> ExecutionResult:
        """执行下一个步骤。
        
        Args:
            tracker: 对话状态追踪器
            
        Returns:
            执行结果
        """
        result = ExecutionResult()
        
        # 获取当前Flow
        flow_id = tracker.active_flow
        if not flow_id:
            logger.debug("No active flow")
            result.flow_completed = True
            return result
        
        flow = self.flows.get_flow(flow_id)
        if not flow:
            logger.warning(f"Flow not found: {flow_id}")
            result.flow_completed = True
            return result
        
        # 获取当前步骤
        current_step_id = self._get_current_step_id(tracker, flow)
        current_step = flow.get_step(current_step_id)
        
        if not current_step:
            logger.debug(f"Step not found: {current_step_id}, flow completed")
            result.flow_completed = True
            return result
        
        logger.debug(f"Executing step: {current_step_id} in flow {flow_id}")
        
        # 根据步骤类型执行
        if current_step.step_type == StepType.ACTION:
            result = self._execute_action_step(current_step, tracker, flow)
        
        elif current_step.step_type == StepType.COLLECT:
            result = self._execute_collect_step(current_step, tracker, flow)
        
        elif current_step.step_type == StepType.SET_SLOT:
            result = self._execute_set_slot_step(current_step, tracker, flow)
        
        elif current_step.step_type == StepType.CONDITION:
            result = self._execute_condition_step(current_step, tracker, flow)
        
        elif current_step.step_type == StepType.LINK:
            result = self._execute_link_step(current_step, tracker, flow)
        
        elif current_step.step_type == StepType.CALL:
            result = self._execute_call_step(current_step, tracker, flow)
        
        elif current_step.step_type == StepType.END:
            result.flow_completed = True
        
        return result
    
    def _get_current_step_id(
        self,
        tracker: "DialogueStateTracker",
        flow: Flow,
    ) -> str:
        """获取当前步骤ID。
        
        Args:
            tracker: 对话状态追踪器
            flow: 当前Flow
            
        Returns:
            当前步骤ID
        """
        # 从dialogue_stack获取当前步骤
        flow_frame = tracker.dialogue_stack.find_flow_frame(flow.id)
        if flow_frame:
            step_id = flow_frame.step_id
            # 如果 step_id 是 "start" 或 "START"，返回 flow 的第一个步骤
            if step_id.lower() == "start":
                first_step = flow.get_first_step()
                return first_step.id if first_step else step_id
            return step_id
        
        # 默认返回第一个步骤
        first_step = flow.get_first_step()
        return first_step.id if first_step else "START"
    
    def _execute_action_step(
        self,
        step: FlowStep,
        tracker: "DialogueStateTracker",
        flow: Flow,
    ) -> ExecutionResult:
        """执行动作步骤。"""
        result = ExecutionResult()
        result.action = step.action
        result.next_step_id = step.next
        
        # 检查是否到达结束
        if step.next is None or step.next == "end":
            result.flow_completed = True
        
        return result
    
    def _execute_collect_step(
        self,
        step: FlowStep,
        tracker: "DialogueStateTracker",
        flow: Flow,
    ) -> ExecutionResult:
        """执行收集步骤。
        
        collect步骤行为：
        - 不指定action：默认先调用 utter_ask_{slot_name}，找不到再调用 action_ask_{slot_name}
        - 显式指定action：直接使用指定的动作（可以是utter_xxx或action_xxx）
        - ask_before_filling: 是否在LLM预填充后仍询问用户确认
          - 语义：第一次进入此步骤时清空槽位并询问，用户填充后继续执行
          - 避免死循环：通过检查 slot_to_collect 判断是否已经询问过
        """
        result = ExecutionResult()
        
        slot_name = step.collect
        if not slot_name:
            result.next_step_id = self._resolve_next_step(step.next, tracker)
            return result
        
        # 检查槽位是否已填充
        current_value = tracker.get_slot(slot_name)
        
        # 获取 flow_frame 以检查是否正在收集此槽位
        flow_frame = tracker.dialogue_stack.top_flow_frame()
        currently_collecting = flow_frame.slot_to_collect if flow_frame else None
        
        # 判断是否需要询问用户
        need_ask = False
        if current_value is None:
            # 槽位未填充，需要询问
            need_ask = True
        elif step.ask_before_filling and currently_collecting != slot_name:
            # ask_before_filling 为 True，且我们还没开始收集这个槽位
            # 说明是第一次进入此步骤，需要清空槽位并询问
            need_ask = True
            # 清空槽位，让用户重新输入
            tracker.set_slot(slot_name, None)
            logger.debug(
                f"Slot {slot_name} has value but ask_before_filling=True and "
                f"not currently collecting, clearing slot and asking user"
            )
        elif step.ask_before_filling and currently_collecting == slot_name:
            # 我们已经在收集这个槽位，用户刚刚填充了值
            # 不需要再次询问，应该继续执行
            logger.debug(
                f"Slot {slot_name} filled by user (was collecting), proceeding"
            )
        
        if need_ask:
            # 需要收集槽位
            result.slot_to_collect = slot_name
            
            # 确定询问动作
            if step.action:
                # 显式指定了action，直接使用
                result.action = step.action
            else:
                # 未指定action，使用默认降级策略：
                # 先尝试 utter_ask_{slot_name}，找不到再调用 action_ask_{slot_name}
                result.action = f"utter_ask_{slot_name}"
                result.metadata["fallback_action"] = f"action_ask_{slot_name}"
        else:
            # 槽位已填充且不需要确认，处理条件分支并跳到下一步
            # step.next 可能是字符串或条件列表
            resolved_next = self._resolve_next_step(step.next, tracker)
            
            # 检查是否有嵌套的动作需要执行
            nested_action = self._get_nested_action(step.next, tracker)
            if nested_action:
                result.action = nested_action["action"]
                result.next_step_id = nested_action.get("next")
                if result.next_step_id:
                    result.next_step_id = self._resolve_next_step(
                        result.next_step_id, tracker
                    )
                logger.debug(
                    f"Slot {slot_name} filled, executing nested action: {result.action}"
                )
            else:
                result.next_step_id = resolved_next
                logger.debug(f"Slot {slot_name} already filled, moving to: {resolved_next}")
        
        # 检查是否流程结束
        next_val = self._resolve_next_step(step.next, tracker)
        if next_val is None or (isinstance(next_val, str) and next_val.upper() == "END"):
            if not result.action:  # 只有在没有动作要执行时才标记完成
                result.flow_completed = True
        
        return result
    
    def _execute_set_slot_step(
        self,
        step: FlowStep,
        tracker: "DialogueStateTracker",
        flow: Flow,
    ) -> ExecutionResult:
        """执行设置槽位步骤。"""
        result = ExecutionResult()
        
        if step.slot_name and step.slot_value is not None:
            tracker.set_slot(step.slot_name, step.slot_value)
            result.events.append({
                "event": "slot_set",
                "name": step.slot_name,
                "value": step.slot_value,
            })
        
        result.next_step_id = step.next
        
        if step.next is None or step.next == "end":
            result.flow_completed = True
        
        return result
    
    def _execute_condition_step(
        self,
        step: FlowStep,
        tracker: "DialogueStateTracker",
        flow: Flow,
    ) -> ExecutionResult:
        """执行条件步骤。"""
        result = ExecutionResult()
        
        # 评估条件
        condition_met = self._evaluate_condition(step.condition, tracker)
        
        if condition_met:
            result.next_step_id = step.then
        else:
            result.next_step_id = step.else_
        
        if result.next_step_id is None or result.next_step_id == "end":
            result.flow_completed = True
        
        return result
    
    def _execute_link_step(
        self,
        step: FlowStep,
        tracker: "DialogueStateTracker",
        flow: Flow,
    ) -> ExecutionResult:
        """执行链接步骤（切换到另一个Flow）。"""
        result = ExecutionResult()
        
        if step.flow_id:
            # 结束当前Flow
            tracker.end_flow()
            # 启动新Flow
            tracker.start_flow(step.flow_id)
            result.events.append({
                "event": "flow_switched",
                "from_flow": flow.id,
                "to_flow": step.flow_id,
            })
        
        result.flow_completed = True
        return result
    
    def _execute_call_step(
        self,
        step: FlowStep,
        tracker: "DialogueStateTracker",
        flow: Flow,
    ) -> ExecutionResult:
        """执行调用步骤（调用子Flow，完成后返回）。"""
        result = ExecutionResult()
        
        if step.flow_id:
            # 压入当前Flow状态
            tracker.start_flow(step.flow_id)
            result.events.append({
                "event": "subflow_called",
                "parent_flow": flow.id,
                "child_flow": step.flow_id,
            })
            result.metadata["return_step"] = step.next
        
        return result
    
    def _evaluate_condition(
        self,
        condition: Optional[str],
        tracker: "DialogueStateTracker",
    ) -> bool:
        """评估条件表达式。
        
        支持简单的槽位检查条件。
        
        Args:
            condition: 条件表达式
            tracker: 对话状态追踪器
            
        Returns:
            条件是否满足
        """
        if not condition:
            return True
        
        # 简单的槽位检查
        # 格式: slot_name == value 或 slot_name != value 或 slot_name
        # 也支持 slots.slot_name 格式
        condition = condition.strip()
        
        # 移除 "slots." 前缀（如果有）
        condition = condition.replace("slots.", "")
        
        # 检查相等
        if "==" in condition:
            parts = condition.split("==")
            if len(parts) == 2:
                slot_name = parts[0].strip()
                expected_value = parts[1].strip().strip('"\'')
                actual_value = tracker.get_slot(slot_name)
                return str(actual_value) == expected_value
        
        # 检查不相等
        if "!=" in condition:
            parts = condition.split("!=")
            if len(parts) == 2:
                slot_name = parts[0].strip()
                expected_value = parts[1].strip().strip('"\'')
                actual_value = tracker.get_slot(slot_name)
                return str(actual_value) != expected_value
        
        # 检查槽位值是否为真（truthy）
        # 当条件只是槽位名时（如 "if: slots.set_receive_info"），检查槽位值是否为真
        slot_value = tracker.get_slot(condition)
        # 处理字符串形式的布尔值
        if isinstance(slot_value, str):
            if slot_value.lower() == "true":
                return True
            elif slot_value.lower() == "false":
                return False
        # 返回布尔值（None、False、空字符串等都返回 False）
        return bool(slot_value)
    
    def _resolve_next_step(
        self,
        next_value: Any,
        tracker: "DialogueStateTracker",
    ) -> Optional[str]:
        """解析下一步骤ID。
        
        处理 next 字段可能是字符串或条件列表的情况。
        
        Args:
            next_value: next 字段的值（可能是字符串或条件列表）
            tracker: 对话状态追踪器
            
        Returns:
            解析后的下一步骤ID
        """
        if next_value is None:
            return None
        
        # 如果是字符串，直接返回
        if isinstance(next_value, str):
            return next_value
        
        # 如果是列表，评估条件找到正确的下一步
        if isinstance(next_value, list):
            for branch in next_value:
                if not isinstance(branch, dict):
                    continue
                
                # 处理 if-then 分支
                if "if" in branch:
                    condition = branch["if"]
                    if self._evaluate_condition(condition, tracker):
                        then_value = branch.get("then")
                        # then 可能是字符串或嵌套的步骤列表
                        if isinstance(then_value, str):
                            return then_value
                        elif isinstance(then_value, list) and then_value:
                            # 如果是步骤列表，返回第一个步骤的 action 或 id
                            first_step = then_value[0]
                            if isinstance(first_step, dict):
                                # 返回一个标记，表示需要执行嵌套步骤
                                return f"__nested__{id(then_value)}"
                        return then_value
                
                # 处理 else 分支
                if "else" in branch:
                    else_value = branch["else"]
                    return else_value if isinstance(else_value, str) else None
        
        return None
    
    def _get_nested_action(
        self,
        next_value: Any,
        tracker: "DialogueStateTracker",
    ) -> Optional[Dict[str, Any]]:
        """获取嵌套步骤中的动作。
        
        当 next 是条件列表且 then 包含嵌套步骤时，返回第一个动作。
        
        Args:
            next_value: next 字段的值
            tracker: 对话状态追踪器
            
        Returns:
            嵌套步骤中的动作信息，包含 action 和 next
        """
        if not isinstance(next_value, list):
            return None
        
        for branch in next_value:
            if not isinstance(branch, dict):
                continue
            
            # 处理 if-then 分支
            if "if" in branch:
                condition = branch["if"]
                if self._evaluate_condition(condition, tracker):
                    then_value = branch.get("then")
                    if isinstance(then_value, list) and then_value:
                        first_step = then_value[0]
                        if isinstance(first_step, dict) and "action" in first_step:
                            return {
                                "action": first_step["action"],
                                "next": first_step.get("next"),
                            }
                    return None
            
            # 处理 else 分支
            if "else" in branch:
                return None
        
        return None
    
    def advance_flow(
        self,
        tracker: "DialogueStateTracker",
        next_step_id: str,
    ) -> None:
        """推进Flow到下一步。
        
        Args:
            tracker: 对话状态追踪器
            next_step_id: 下一步ID
        """
        # 更新栈顶FlowStackFrame的step_id
        flow_frame = tracker.dialogue_stack.top_flow_frame()
        if flow_frame:
            flow_frame.step_id = next_step_id
    
    def set_flows(self, flows: FlowsList) -> None:
        """设置Flow列表。
        
        Args:
            flows: Flow列表
        """
        self.flows = flows


# 导出
__all__ = [
    "FlowExecutor",
    "ExecutionResult",
]
