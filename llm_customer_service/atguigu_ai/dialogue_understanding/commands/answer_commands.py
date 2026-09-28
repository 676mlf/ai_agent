# -*- coding: utf-8 -*-
"""
回答相关命令

包含各种类型的回答命令，用于处理不同场景的用户输入。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from atguigu_ai.dialogue_understanding.commands.base import Command, register_command
from atguigu_ai.shared.constants import DegradationReason

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker


@register_command
@dataclass
class ChitChatAnswerCommand(Command):
    """闲聊回答命令。
    
    用于处理用户的闲聊输入，这些输入不属于任何业务Flow。
    系统会使用LLM生成自然的闲聊回复。
    
    执行时直接压入ChitChatStackFrame，由EnterpriseSearchPolicy检测并生成响应。
    """
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "chitchat"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChitChatAnswerCommand":
        """从字典创建命令。"""
        return ChitChatAnswerCommand()
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行闲聊回答命令。
        
        直接压入ChitChatStackFrame，由EnterpriseSearchPolicy检测并生成响应。
        
        Args:
            tracker: 对话状态追踪器
            flows: 可用的Flow列表
            
        Returns:
            产生的事件列表
        """
        from atguigu_ai.dialogue_understanding.stack.stack_frame import ChitChatStackFrame
        
        tracker.dialogue_stack.push(ChitChatStackFrame())
        return [{
            "event": "chitchat_triggered",
            "degradation_reason": DegradationReason.CHITCHAT,
            "timestamp": None,
        }]
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        return "chitchat"
    
    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。"""
        return r"""^(?:chitchat|ChitChat\(\))$"""
    
    @classmethod
    def _from_regex_match(cls, match: re.Match) -> Optional["ChitChatAnswerCommand"]:
        """从正则匹配创建命令。"""
        return ChitChatAnswerCommand()


@register_command
@dataclass
class CannotHandleCommand(Command):
    """无法处理命令。
    
    当系统无法理解或处理用户输入时使用。
    执行时直接压入CannotHandleStackFrame，由EnterpriseSearchPolicy检测并生成降级响应。
    
    Attributes:
        reason: 无法处理的原因
    """
    
    reason: str = DegradationReason.DEFAULT
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "cannot_handle"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CannotHandleCommand":
        """从字典创建命令。"""
        return CannotHandleCommand(
            reason=data.get("reason", DegradationReason.DEFAULT)
        )
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行无法处理命令。
        
        直接压入CannotHandleStackFrame，由EnterpriseSearchPolicy检测并生成降级响应。
        
        Args:
            tracker: 对话状态追踪器
            flows: 可用的Flow列表
            
        Returns:
            产生的事件列表
        """
        from atguigu_ai.dialogue_understanding.stack.stack_frame import CannotHandleStackFrame
        
        tracker.dialogue_stack.push(CannotHandleStackFrame(reason=self.reason))
        return [{
            "event": "cannot_handle",
            "reason": self.reason,
            "degradation_reason": DegradationReason.CANNOT_HANDLE,
            "timestamp": None,
        }]
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        if self.reason and self.reason != DegradationReason.DEFAULT:
            return f'cannot_handle("{self.reason}")'
        return "cannot_handle"
    
    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。"""
        return r"""^(?:cannot_handle(?:\(['"]?(.*)['"]?\))?|CannotHandle\((?:['"]?(.*)['"]?)?\))$"""
    
    @classmethod
    def _from_regex_match(cls, match: re.Match) -> Optional["CannotHandleCommand"]:
        """从正则匹配创建命令。"""
        reason = match.group(1) or match.group(2) or DegradationReason.DEFAULT
        return CannotHandleCommand(reason=reason.strip() if reason else DegradationReason.DEFAULT)


@register_command
@dataclass
class KnowledgeAnswerCommand(Command):
    """知识库回答命令。
    
    用于触发基于知识库检索的回答（RAG）。
    执行时直接压入SearchStackFrame，由EnterpriseSearchPolicy检测并执行检索。
    
    Attributes:
        query: 用于知识库检索的查询（可选，默认使用用户最新消息）
    """
    
    query: Optional[str] = None
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "knowledge_answer"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeAnswerCommand":
        """从字典创建命令。"""
        return KnowledgeAnswerCommand(query=data.get("query"))
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行知识库回答命令。
        
        直接压入SearchStackFrame，由EnterpriseSearchPolicy检测并执行检索。
        不存储query，检索时从latest_message获取。
        
        Args:
            tracker: 对话状态追踪器
            flows: 可用的Flow列表
            
        Returns:
            产生的事件列表
        """
        from atguigu_ai.dialogue_understanding.stack.stack_frame import SearchStackFrame
        
        tracker.dialogue_stack.push(SearchStackFrame())
        
        query = self.query
        if not query and tracker.latest_message:
            query = tracker.latest_message.text
        
        return [{
            "event": "knowledge_search_triggered",
            "query": query,
            "degradation_reason": DegradationReason.NO_RELEVANT_ANSWER,
            "timestamp": None,
        }]
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        if self.query:
            return f'knowledge_answer("{self.query}")'
        return "knowledge_answer"
    
    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。"""
        return r"""^(?:knowledge_answer(?:\(['"]?(.*)['"]?\))?|SearchAndReply\((?:['"]?(.*)['"]?)?\))$"""
    
    @classmethod
    def _from_regex_match(cls, match: re.Match) -> Optional["KnowledgeAnswerCommand"]:
        """从正则匹配创建命令。"""
        query = match.group(1) or match.group(2)
        return KnowledgeAnswerCommand(query=query.strip() if query else None)


@register_command
@dataclass
class FreeFormAnswerCommand(Command):
    """自由回答命令。
    
    用于生成不受Flow约束的自由回答。
    """
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "free_form_answer"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FreeFormAnswerCommand":
        """从字典创建命令。"""
        return FreeFormAnswerCommand()
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行自由回答命令。"""
        return [{
            "event": "free_form_answer_triggered",
            "timestamp": None,
        }]
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        return "free_form_answer"
    
    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。"""
        return r"""^free_form_answer$"""
    
    @classmethod
    def _from_regex_match(cls, match: re.Match) -> Optional["FreeFormAnswerCommand"]:
        """从正则匹配创建命令。"""
        return FreeFormAnswerCommand()


# 导出
__all__ = [
    "ChitChatAnswerCommand",
    "CannotHandleCommand",
    "KnowledgeAnswerCommand",
    "FreeFormAnswerCommand",
]
