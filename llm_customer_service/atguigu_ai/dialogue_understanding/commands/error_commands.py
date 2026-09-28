# -*- coding: utf-8 -*-
"""
错误相关命令

用于处理系统错误和异常情况的命令。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from atguigu_ai.dialogue_understanding.commands.base import Command, register_command
from atguigu_ai.shared.constants import DegradationReason

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker


@register_command
@dataclass
class ErrorCommand(Command):
    """错误命令。
    
    当系统遇到错误时生成此命令。
    用于记录和处理各种错误情况。
    
    Attributes:
        error_type: 错误类型
        message: 错误消息
    """
    
    error_type: str = "unknown_error"
    message: Optional[str] = None
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "error"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorCommand":
        """从字典创建命令。"""
        return ErrorCommand(
            error_type=data.get("error_type", "unknown_error"),
            message=data.get("message"),
        )
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行错误命令。
        
        记录错误事件。
        
        Args:
            tracker: 对话状态追踪器
            flows: 可用的Flow列表
            
        Returns:
            产生的事件列表
        """
        return [{
            "event": "error_occurred",
            "error_type": self.error_type,
            "message": self.message,
            "degradation_reason": DegradationReason.INTERNAL_ERROR,
            "timestamp": None,
        }]
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        if self.message:
            return f'error("{self.error_type}", "{self.message}")'
        return f'error("{self.error_type}")'
    
    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。"""
        return r"""^error\(['"]?([^'"]+)['"]?(?:,\s*['"]?([^'"]+)['"]?)?\)$"""
    
    @classmethod
    def _from_regex_match(cls, match: re.Match) -> Optional["ErrorCommand"]:
        """从正则匹配创建命令。"""
        error_type = match.group(1)
        message = match.group(2)
        return ErrorCommand(
            error_type=error_type.strip() if error_type else "unknown_error",
            message=message.strip() if message else None,
        )


@register_command
@dataclass
class InternalErrorCommand(Command):
    """内部错误命令。
    
    当系统遇到内部错误（如LLM调用失败、超时等）时生成此命令。
    
    Attributes:
        exception_type: 异常类型名
        exception_message: 异常消息
    """
    
    exception_type: str = "InternalError"
    exception_message: Optional[str] = None
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "internal_error"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InternalErrorCommand":
        """从字典创建命令。"""
        return InternalErrorCommand(
            exception_type=data.get("exception_type", "InternalError"),
            exception_message=data.get("exception_message"),
        )
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行内部错误命令。"""
        return [{
            "event": "internal_error",
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "degradation_reason": DegradationReason.INTERNAL_ERROR,
            "timestamp": None,
        }]
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        if self.exception_message:
            return f'internal_error("{self.exception_type}", "{self.exception_message}")'
        return f'internal_error("{self.exception_type}")'


@register_command
@dataclass
class ParseErrorCommand(Command):
    """解析错误命令。
    
    当命令解析失败时生成此命令。
    
    Attributes:
        raw_text: 原始文本
        error_message: 解析错误消息
    """
    
    raw_text: str = ""
    error_message: Optional[str] = None
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "parse_error"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParseErrorCommand":
        """从字典创建命令。"""
        return ParseErrorCommand(
            raw_text=data.get("raw_text", ""),
            error_message=data.get("error_message"),
        )
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行解析错误命令。"""
        return [{
            "event": "parse_error",
            "raw_text": self.raw_text,
            "error_message": self.error_message,
            "timestamp": None,
        }]
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        return f'parse_error("{self.raw_text}")'


# 导出
__all__ = [
    "ErrorCommand",
    "InternalErrorCommand",
    "ParseErrorCommand",
]
