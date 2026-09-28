# -*- coding: utf-8 -*-
"""
对话理解模块 (Dialogue Understanding / DU)

本架构的核心模块，负责：
- 命令生成：使用LLM将用户输入转换为命令
- 命令处理：执行命令并更新对话状态
- Flow管理：管理对话流程的执行
- 对话栈：管理多轮对话的上下文

核心概念：
- Command: 对话系统的原子操作，如StartFlow, SetSlot等
- Generator: 使用LLM生成命令
- Processor: 处理并执行命令
- Flow: 定义对话流程
- DialogueStack: 管理对话上下文栈
"""

from atguigu_ai.dialogue_understanding.commands import (
    Command,
    StartFlowCommand,
    SetSlotCommand,
    CancelFlowCommand,
    ChitChatAnswerCommand,
    CannotHandleCommand,
    ClarifyCommand,
    HumanHandoffCommand,
    ErrorCommand,
    SessionStartCommand,
)

__all__ = [
    # Commands
    "Command",
    "StartFlowCommand",
    "SetSlotCommand",
    "CancelFlowCommand",
    "ChitChatAnswerCommand",
    "CannotHandleCommand",
    "ClarifyCommand",
    "HumanHandoffCommand",
    "ErrorCommand",
    "SessionStartCommand",
]
