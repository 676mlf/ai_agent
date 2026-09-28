# -*- coding: utf-8 -*-
"""
Flow定义

定义对话流程的数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union


class StepType(str, Enum):
    """步骤类型枚举。"""
    ACTION = "action"
    COLLECT = "collect"
    LINK = "link"
    SET_SLOT = "set_slot"
    CONDITION = "condition"
    END = "end"
    CALL = "call"


@dataclass
class FlowStep:
    """Flow步骤。
    
    表示Flow中的一个执行步骤。
    
    Attributes:
        id: 步骤ID
        step_type: 步骤类型
        action: 要执行的动作
            - 当step_type为ACTION时：要执行的动作
            - 当step_type为COLLECT时：可选，显式指定询问动作（utter_xxx或action_xxx）
        collect: 要收集的槽位（当step_type为collect时）
        ask_before_filling: 是否在LLM预填充后仍询问用户确认（collect步骤可选）
        reset_after_flow_ends: flow结束时是否重置该槽位（collect步骤，默认True）
        next: 下一个步骤ID
        condition: 条件表达式（当step_type为condition时）
        then: 条件为真时的下一步
        else_: 条件为假时的下一步
        slot_name: 槽位名（当step_type为set_slot时）
        slot_value: 槽位值
        flow_id: 要调用的Flow ID（当step_type为call或link时）
        description: 步骤描述
        metadata: 额外元数据
    """
    id: str
    step_type: StepType = StepType.ACTION
    action: Optional[str] = None
    collect: Optional[str] = None
    ask_before_filling: bool = False
    reset_after_flow_ends: bool = True
    next: Optional[str] = None
    condition: Optional[str] = None
    then: Optional[str] = None
    else_: Optional[str] = None
    slot_name: Optional[str] = None
    slot_value: Any = None
    flow_id: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, step_id: str, data: Dict[str, Any]) -> "FlowStep":
        """从字典创建步骤。
        
        Args:
            step_id: 步骤ID
            data: 步骤配置字典
            
        Returns:
            FlowStep实例
        """
        # 确定步骤类型
        step_type = StepType.ACTION
        if "collect" in data:
            step_type = StepType.COLLECT
        elif "link" in data:
            step_type = StepType.LINK
        elif "set_slot" in data or "set_slots" in data:
            step_type = StepType.SET_SLOT
        elif "if" in data or "condition" in data:
            step_type = StepType.CONDITION
        elif "call" in data:
            step_type = StepType.CALL
        elif data.get("id") == "end" or data.get("action") == "end":
            step_type = StepType.END
        
        # 提取字段
        action = data.get("action")
        collect = data.get("collect")
        ask_before_filling = data.get("ask_before_filling", False)
        reset_after_flow_ends = data.get("reset_after_flow_ends", True)
        next_step = data.get("next")
        condition = data.get("if") or data.get("condition")
        then = data.get("then")
        else_ = data.get("else")
        
        # set_slot处理
        slot_name = None
        slot_value = None
        if "set_slot" in data:
            set_slot = data["set_slot"]
            if isinstance(set_slot, dict):
                slot_name = list(set_slot.keys())[0] if set_slot else None
                slot_value = set_slot.get(slot_name) if slot_name else None
        
        # link/call处理
        flow_id = data.get("link") or data.get("call")
        
        return cls(
            id=step_id,
            step_type=step_type,
            action=action,
            collect=collect,
            ask_before_filling=ask_before_filling,
            reset_after_flow_ends=reset_after_flow_ends,
            next=next_step,
            condition=condition,
            then=then,
            else_=else_,
            slot_name=slot_name,
            slot_value=slot_value,
            flow_id=flow_id,
            description=data.get("description"),
            metadata=data.get("metadata", {}),
        )
    
    def as_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        data = {"id": self.id}
        
        if self.action:
            data["action"] = self.action
        if self.collect:
            data["collect"] = self.collect
        if self.ask_before_filling:
            data["ask_before_filling"] = self.ask_before_filling
        if not self.reset_after_flow_ends:
            # 只有非默认值（False）才序列化
            data["reset_after_flow_ends"] = self.reset_after_flow_ends
        if self.next:
            data["next"] = self.next
        if self.condition:
            data["condition"] = self.condition
        if self.then:
            data["then"] = self.then
        if self.else_:
            data["else"] = self.else_
        if self.slot_name:
            data["set_slot"] = {self.slot_name: self.slot_value}
        if self.flow_id:
            data["link"] = self.flow_id
        if self.description:
            data["description"] = self.description
        if self.metadata:
            data["metadata"] = self.metadata
        
        return data
    
    def is_end(self) -> bool:
        """检查是否是结束步骤。"""
        return self.step_type == StepType.END or self.id == "end"


@dataclass
class Flow:
    """对话流程。
    
    Flow定义了一个完整的对话流程，包含多个步骤。
    
    Attributes:
        id: Flow ID
        description: Flow描述
        steps: 步骤列表
        name: Flow显示名称
        slot_initial_values: 槽位初始值
        persisted_slots: flow结束后仍保留的槽位列表
        metadata: 额外元数据
    """
    id: str
    description: str = ""
    steps: List[FlowStep] = field(default_factory=list)
    name: Optional[str] = None
    slot_initial_values: Dict[str, Any] = field(default_factory=dict)
    persisted_slots: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理。"""
        # 构建步骤索引
        self._step_index: Dict[str, FlowStep] = {}
        for step in self.steps:
            self._step_index[step.id] = step
    
    @classmethod
    def from_dict(cls, flow_id: str, data: Dict[str, Any]) -> "Flow":
        """从字典创建Flow。
        
        Args:
            flow_id: Flow ID
            data: Flow配置字典
            
        Returns:
            Flow实例
        """
        # 解析步骤
        steps = []
        steps_data = data.get("steps", [])
        
        for i, step_data in enumerate(steps_data):
            # 确定步骤ID
            if isinstance(step_data, dict):
                step_id = step_data.get("id", f"step_{i}")
                step = FlowStep.from_dict(step_id, step_data)
            else:
                # 简单的动作名
                step = FlowStep(
                    id=f"step_{i}",
                    step_type=StepType.ACTION,
                    action=str(step_data),
                )
            
            # 自动设置next（如果没有指定）
            if step.next is None and i < len(steps_data) - 1:
                next_step_data = steps_data[i + 1]
                if isinstance(next_step_data, dict):
                    step.next = next_step_data.get("id", f"step_{i + 1}")
                else:
                    step.next = f"step_{i + 1}"
            
            steps.append(step)
        
        return cls(
            id=flow_id,
            description=data.get("description", ""),
            steps=steps,
            name=data.get("name"),
            slot_initial_values=data.get("slot_initial_values", {}),
            persisted_slots=data.get("persisted_slots", []),
            metadata=data.get("metadata", {}),
        )
    
    def as_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        data = {
            "id": self.id,
            "description": self.description,
            "steps": [step.as_dict() for step in self.steps],
        }
        
        if self.name:
            data["name"] = self.name
        if self.slot_initial_values:
            data["slot_initial_values"] = self.slot_initial_values
        if self.persisted_slots:
            data["persisted_slots"] = self.persisted_slots
        if self.metadata:
            data["metadata"] = self.metadata
        
        return data
    
    def get_step(self, step_id: str) -> Optional[FlowStep]:
        """获取指定ID的步骤。
        
        Args:
            step_id: 步骤ID
            
        Returns:
            步骤，如果不存在则返回None
        """
        return self._step_index.get(step_id)
    
    def get_first_step(self) -> Optional[FlowStep]:
        """获取第一个步骤。"""
        return self.steps[0] if self.steps else None
    
    def get_next_step(self, current_step_id: str) -> Optional[FlowStep]:
        """获取下一个步骤。
        
        Args:
            current_step_id: 当前步骤ID
            
        Returns:
            下一个步骤，如果没有则返回None
        """
        current_step = self.get_step(current_step_id)
        if current_step and current_step.next:
            return self.get_step(current_step.next)
        return None
    
    def get_collect_steps(self) -> List[FlowStep]:
        """获取所有收集信息的步骤。"""
        return [step for step in self.steps if step.step_type == StepType.COLLECT]
    
    def get_slots_to_collect(self) -> List[str]:
        """获取需要收集的所有槽位名。"""
        return [
            step.collect
            for step in self.steps
            if step.collect
        ]
    
    def __iter__(self) -> Iterator[FlowStep]:
        """迭代所有步骤。"""
        return iter(self.steps)
    
    def __len__(self) -> int:
        """返回步骤数量。"""
        return len(self.steps)


@dataclass
class FlowsList:
    """Flow列表容器。
    
    管理多个Flow的集合。
    
    Attributes:
        flows: Flow列表
    """
    flows: List[Flow] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后处理。"""
        self._flow_index: Dict[str, Flow] = {}
        for flow in self.flows:
            self._flow_index[flow.id] = flow
    
    @property
    def flow_ids(self) -> List[str]:
        """获取所有Flow ID。"""
        return list(self._flow_index.keys())
    
    def get_flow(self, flow_id: str) -> Optional[Flow]:
        """获取指定ID的Flow。
        
        Args:
            flow_id: Flow ID
            
        Returns:
            Flow，如果不存在则返回None
        """
        return self._flow_index.get(flow_id)
    
    def add_flow(self, flow: Flow) -> None:
        """添加Flow。
        
        Args:
            flow: 要添加的Flow
        """
        self.flows.append(flow)
        self._flow_index[flow.id] = flow
    
    def remove_flow(self, flow_id: str) -> Optional[Flow]:
        """移除Flow。
        
        Args:
            flow_id: Flow ID
            
        Returns:
            被移除的Flow，如果不存在则返回None
        """
        flow = self._flow_index.pop(flow_id, None)
        if flow:
            self.flows.remove(flow)
        return flow
    
    def has_flow(self, flow_id: str) -> bool:
        """检查是否存在指定Flow。"""
        return flow_id in self._flow_index
    
    def __iter__(self) -> Iterator[Flow]:
        """迭代所有Flow。"""
        return iter(self.flows)
    
    def __len__(self) -> int:
        """返回Flow数量。"""
        return len(self.flows)
    
    def __contains__(self, flow_id: str) -> bool:
        """检查Flow是否存在。"""
        return flow_id in self._flow_index


# 导出
__all__ = [
    "Flow",
    "FlowStep",
    "FlowsList",
    "StepType",
]
