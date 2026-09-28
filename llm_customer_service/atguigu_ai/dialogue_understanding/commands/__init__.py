# -*- coding: utf-8 -*-
"""
命令系统模块

定义对话系统的所有命令类型。命令是本架构中的核心概念，
表示对话系统可以执行的原子操作。

命令类型：
- Flow相关: StartFlowCommand, CancelFlowCommand
- 槽位相关: SetSlotCommand
- 回答相关: ChitChatAnswerCommand, CannotHandleCommand
- 会话相关: SessionStartCommand, ClarifyCommand
- 系统相关: ErrorCommand, HumanHandoffCommand
"""

from atguigu_ai.dialogue_understanding.commands.base import Command
from atguigu_ai.dialogue_understanding.commands.flow_commands import (
    StartFlowCommand,
    CancelFlowCommand,
)
from atguigu_ai.dialogue_understanding.commands.slot_commands import SetSlotCommand
from atguigu_ai.dialogue_understanding.commands.answer_commands import (
    ChitChatAnswerCommand,
    CannotHandleCommand,
    KnowledgeAnswerCommand,
)
from atguigu_ai.dialogue_understanding.commands.session_commands import (
    SessionStartCommand,
    ClarifyCommand,
    HumanHandoffCommand,
)
from atguigu_ai.dialogue_understanding.commands.error_commands import ErrorCommand

__all__ = [
    # Base
    "Command",
    # Flow commands
    "StartFlowCommand",
    "CancelFlowCommand",
    # Slot commands
    "SetSlotCommand",
    # Answer commands
    "ChitChatAnswerCommand",
    "CannotHandleCommand",
    "KnowledgeAnswerCommand",
    # Session commands
    "SessionStartCommand",
    "ClarifyCommand",
    "HumanHandoffCommand",
    # Error commands
    "ErrorCommand",
]
