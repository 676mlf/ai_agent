# -*- coding: utf-8 -*-
"""
Flow相关命令

包含Flow启动、取消等命令。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from atguigu_ai.dialogue_understanding.commands.base import Command, register_command

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker


@register_command
@dataclass
class StartFlowCommand(Command):
    """启动Flow命令。
    
    用于启动指定的对话流程。当LLM识别到用户意图与某个Flow匹配时，
    会生成此命令来启动相应的Flow。
    
    设计说明：
        StartFlowCommand 是"即时数据命令"，其 run() 方法直接操作 tracker
        设置 active_flow，以便 FlowPolicy 能立即接管执行 Flow 步骤。
        这与"动作触发命令"（如 CancelFlowCommand）不同，后者通过
        设置 next_action 触发对应的 Action 来执行操作。
    
    Attributes:
        flow: 要启动的Flow ID
    """
    
    flow: str
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "start_flow"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StartFlowCommand":
        """从字典创建命令。
        
        Args:
            data: 包含flow字段的字典
            
        Returns:
            StartFlowCommand实例
            
        Raises:
            ValueError: 如果缺少flow字段
        """
        try:
            return StartFlowCommand(flow=data["flow"])
        except KeyError as e:
            raise ValueError(f"Missing required field 'flow' for StartFlowCommand") from e
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行启动Flow命令。
        
        在tracker上启动指定的Flow。
        
        Args:
            tracker: 对话状态追踪器
            flows: 可用的Flow列表
            
        Returns:
            产生的事件列表
        """
        # 检查Flow是否存在
        if flows is not None:
            flow_ids = getattr(flows, 'flow_ids', [])
            if hasattr(flows, '__iter__') and not isinstance(flows, str):
                flow_ids = [f.id if hasattr(f, 'id') else str(f) for f in flows]
            if self.flow not in flow_ids:
                # Flow不存在，返回空事件列表
                return []
        
        # 启动Flow
        tracker.start_flow(self.flow)
        
        return [{
            "event": "flow_started",
            "flow_id": self.flow,
            "timestamp": None,  # 由tracker设置
        }]
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        return f"start flow {self.flow}"
    
    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。"""
        # 支持多种格式:
        # - start flow <flow_name>
        # - StartFlow(<flow_name>)
        return r"""(?:start\s+flow\s+['"`]?([a-zA-Z0-9_-]+)['"`]?|StartFlow\(['"]?([a-zA-Z0-9_-]+)['"]?\))"""
    
    @classmethod
    def _from_regex_match(cls, match: re.Match) -> Optional["StartFlowCommand"]:
        """从正则匹配创建命令。"""
        # 第一个分组是 start flow 格式，第二个是 StartFlow() 格式
        flow = match.group(1) or match.group(2)
        if flow:
            return StartFlowCommand(flow=flow.strip())
        return None
    
    def __hash__(self) -> int:
        return hash(self.flow)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StartFlowCommand):
            return False
        return other.flow == self.flow


@register_command
@dataclass
class CancelFlowCommand(Command):
    """取消Flow命令。
    
    用于取消当前正在执行的Flow。
    
    Attributes:
        flow: 要取消的Flow ID，如果为None则取消当前活动的Flow
    """
    
    flow: Optional[str] = None
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "cancel_flow"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CancelFlowCommand":
        """从字典创建命令。"""
        return CancelFlowCommand(flow=data.get("flow"))
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行取消Flow命令。
        
        注意：此方法只返回事件标记，实际的取消操作由 ActionCancelFlow 执行。
        
        Args:
            tracker: 对话状态追踪器
            flows: 可用的Flow列表
            
        Returns:
            产生的事件列表
        """
        flow_id = self.flow or tracker.active_flow
        if flow_id:
            # 不直接取消，让 ActionCancelFlow 来执行
            return [{
                "event": "cancel_flow_requested",
                "flow_id": flow_id,
                "timestamp": None,
            }]
        return []
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        if self.flow:
            return f"cancel flow {self.flow}"
        return "cancel flow"
    
    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。"""
        return r"""(?:cancel\s+flow(?:\s+['"`]?([a-zA-Z0-9_-]+)['"`]?)?|CancelFlow\((?:['"]?([a-zA-Z0-9_-]*)['"]?)?\))"""
    
    @classmethod
    def _from_regex_match(cls, match: re.Match) -> Optional["CancelFlowCommand"]:
        """从正则匹配创建命令。"""
        flow = match.group(1) or match.group(2)
        return CancelFlowCommand(flow=flow.strip() if flow else None)


@register_command
@dataclass
class ChangeFlowCommand(Command):
    """切换Flow命令。
    
    用于从当前Flow切换到另一个Flow。
    
    设计说明：
        ChangeFlowCommand 是"动作触发命令"，其 run() 方法只返回事件标记，
        实际的切换操作由 ActionChangeFlow 执行。这样设计确保了
        Command 只声明意图，Action 执行实际操作的原则。
    
    Attributes:
        flow: 要切换到的Flow ID
    """
    
    flow: str
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "change_flow"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChangeFlowCommand":
        """从字典创建命令。"""
        try:
            return ChangeFlowCommand(flow=data["flow"])
        except KeyError as e:
            raise ValueError(f"Missing required field 'flow' for ChangeFlowCommand") from e
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行切换Flow命令。
        
        注意：此方法只返回事件标记，实际的切换操作由 ActionChangeFlow 执行。
        """
        old_flow = tracker.active_flow
        return [{
            "event": "change_flow_requested",
            "old_flow_id": old_flow,
            "new_flow_id": self.flow,
            "timestamp": None,
        }]
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        return f"change flow {self.flow}"
    
    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。"""
        return r"""change\s+flow\s+['"`]?([a-zA-Z0-9_-]+)['"`]?"""
    
    @classmethod
    def _from_regex_match(cls, match: re.Match) -> Optional["ChangeFlowCommand"]:
        """从正则匹配创建命令。"""
        flow = match.group(1)
        if flow:
            return ChangeFlowCommand(flow=flow.strip())
        return None


# 导出
__all__ = [
    "StartFlowCommand",
    "CancelFlowCommand",
    "ChangeFlowCommand",
]
