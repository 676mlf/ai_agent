# -*- coding: utf-8 -*-
"""
对话栈模块

管理对话上下文的栈结构，支持多轮对话和Flow嵌套。
"""

from atguigu_ai.dialogue_understanding.stack.dialogue_stack import DialogueStack
from atguigu_ai.dialogue_understanding.stack.stack_frame import (
    StackFrame,
    FlowStackFrame,
    SearchStackFrame,
    ChitChatStackFrame,
)

__all__ = [
    "DialogueStack",
    "StackFrame",
    "FlowStackFrame",
    "SearchStackFrame",
    "ChitChatStackFrame",
]
