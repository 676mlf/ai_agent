# -*- coding: utf-8 -*-
"""
会话相关命令

包含会话管理、澄清、人工转接等命令。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from atguigu_ai.dialogue_understanding.commands.base import Command, register_command

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker


@register_command
@dataclass
class SessionStartCommand(Command):
    """会话开始命令。
    
    用于标记新会话的开始，初始化对话状态。
    """
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "session_start"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionStartCommand":
        """从字典创建命令。"""
        return SessionStartCommand()
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行会话开始命令。
        
        注意：此方法只返回事件标记，实际的初始化操作由 ActionSessionStart 执行。
        
        Args:
            tracker: 对话状态追踪器
            flows: 可用的Flow列表
            
        Returns:
            产生的事件列表
        """
        # 不直接重启，让 ActionSessionStart 来执行
        return [{
            "event": "session_start_requested",
            "timestamp": None,
        }]
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        return "session_start"
    
    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。"""
        return r"""^session_start$"""
    
    @classmethod
    def _from_regex_match(cls, match: re.Match) -> Optional["SessionStartCommand"]:
        """从正则匹配创建命令。"""
        return SessionStartCommand()


@register_command
@dataclass
class ClarifyCommand(Command):
    """澄清命令。
    
    当用户输入不清晰或需要更多信息时，使用此命令请求澄清。
    
    Attributes:
        question: 澄清问题（可选）
        options: 供用户选择的选项列表（可选）
    """
    
    question: Optional[str] = None
    options: List[str] = field(default_factory=list)
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "clarify"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClarifyCommand":
        """从字典创建命令。"""
        return ClarifyCommand(
            question=data.get("question"),
            options=data.get("options", []),
        )
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行澄清命令。
        
        标记当前轮次需要向用户请求澄清。
        
        Args:
            tracker: 对话状态追踪器
            flows: 可用的Flow列表
            
        Returns:
            产生的事件列表
        """
        return [{
            "event": "clarification_requested",
            "question": self.question,
            "options": self.options,
            "timestamp": None,
        }]
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        if self.question:
            return f'clarify("{self.question}")'
        return "clarify"
    
    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。"""
        return r"""^clarify(?:\(['"]?(.*)['"]?\))?$"""
    
    @classmethod
    def _from_regex_match(cls, match: re.Match) -> Optional["ClarifyCommand"]:
        """从正则匹配创建命令。"""
        question = match.group(1)
        return ClarifyCommand(question=question.strip() if question else None)


@register_command
@dataclass
class HumanHandoffCommand(Command):
    """人工转接命令。
    
    当系统无法处理用户请求或用户明确要求时，
    将对话转接给人工客服。
    
    Attributes:
        reason: 转接原因
    """
    
    reason: Optional[str] = None
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "human_handoff"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanHandoffCommand":
        """从字典创建命令。"""
        return HumanHandoffCommand(reason=data.get("reason"))
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行人工转接命令。
        
        直接压入 HumanHandoffStackFrame，Policy 会检测该栈帧并执行对应操作。
        
        Args:
            tracker: 对话状态追踪器
            flows: 可用的Flow列表
            
        Returns:
            产生的事件列表
        """
        # 直接压入 HumanHandoffStackFrame，Policy 会在 predict 时检测并处理
        from atguigu_ai.dialogue_understanding.stack.stack_frame import HumanHandoffStackFrame
        tracker.dialogue_stack.push(HumanHandoffStackFrame(reason=self.reason))
        
        return [{
            "event": "human_handoff_requested",
            "reason": self.reason,
            "timestamp": None,
        }]
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        if self.reason:
            return f'human_handoff("{self.reason}")'
        return "human_handoff"
    
    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。"""
        return r"""^human_handoff(?:\(['"]?(.*)['"]?\))?$"""
    
    @classmethod
    def _from_regex_match(cls, match: re.Match) -> Optional["HumanHandoffCommand"]:
        """从正则匹配创建命令。"""
        reason = match.group(1)
        return HumanHandoffCommand(reason=reason.strip() if reason else None)


@register_command
@dataclass
class RestartCommand(Command):
    """重启命令。
    
    重置当前对话状态，清空所有槽位和Flow。
    """
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "restart"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RestartCommand":
        """从字典创建命令。"""
        return RestartCommand()
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行重启命令。
        
        注意：此方法只返回事件标记，实际的重启操作由 ActionRestart 执行。
        """
        # 不直接重启，让 ActionRestart 来执行
        return [{
            "event": "restart_requested",
            "timestamp": None,
        }]
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        return "restart"
    
    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。"""
        return r"""^restart$"""
    
    @classmethod
    def _from_regex_match(cls, match: re.Match) -> Optional["RestartCommand"]:
        """从正则匹配创建命令。"""
        return RestartCommand()


@register_command
@dataclass
class NoopCommand(Command):
    """空操作命令。
    
    不执行任何操作。用于特定场景下需要显式表示"不做任何事"。
    """
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "noop"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NoopCommand":
        """从字典创建命令。"""
        return NoopCommand()
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行空操作命令。"""
        return []
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        return "noop"
    
    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。"""
        return r"""^noop$"""
    
    @classmethod
    def _from_regex_match(cls, match: re.Match) -> Optional["NoopCommand"]:
        """从正则匹配创建命令。"""
        return NoopCommand()


# 导出
__all__ = [
    "SessionStartCommand",
    "ClarifyCommand",
    "HumanHandoffCommand",
    "RestartCommand",
    "NoopCommand",
]
