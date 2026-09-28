# -*- coding: utf-8 -*-
"""
命令处理器

负责执行命令并更新对话状态。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from atguigu_ai.dialogue_understanding.commands.base import Command
from atguigu_ai.dialogue_understanding.commands.flow_commands import (
    StartFlowCommand,
    CancelFlowCommand,
    ChangeFlowCommand,
)
from atguigu_ai.dialogue_understanding.commands.slot_commands import SetSlotCommand
from atguigu_ai.dialogue_understanding.commands.answer_commands import (
    ChitChatAnswerCommand,
    CannotHandleCommand,
    KnowledgeAnswerCommand,
)
from atguigu_ai.dialogue_understanding.commands.session_commands import (
    ClarifyCommand,
    HumanHandoffCommand,
    SessionStartCommand,
    RestartCommand,
)
from atguigu_ai.dialogue_understanding.stack.dialogue_stack import DialogueStack
from atguigu_ai.dialogue_understanding.stack.stack_frame import (
    FlowStackFrame,
    SearchStackFrame,
    ChitChatStackFrame,
    CannotHandleStackFrame,
)

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker
    from atguigu_ai.core.domain import Domain

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """处理器配置。
    
    Attributes:
        allow_parallel_commands: 是否允许并行执行多个命令
        validate_flows: 是否验证Flow存在性
        validate_slots: 是否验证槽位存在性
    """
    allow_parallel_commands: bool = True
    validate_flows: bool = True
    validate_slots: bool = True


@dataclass
class ProcessResult:
    """处理结果。
    
    Attributes:
        events: 产生的事件列表
        commands_executed: 成功执行的命令数
        errors: 错误列表
        next_action: 下一步应该执行的动作
        response_type: 响应类型（flow, chitchat, knowledge, cannot_handle等）
    """
    events: List[Dict[str, Any]] = field(default_factory=list)
    commands_executed: int = 0
    errors: List[str] = field(default_factory=list)
    next_action: Optional[str] = None
    response_type: str = "none"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """是否成功处理了命令。"""
        return self.commands_executed > 0 and len(self.errors) == 0


class CommandProcessor:
    """命令处理器。
    
    负责执行命令并更新对话状态。这是对话理解模块的核心组件之一。
    
    处理流程：
    1. 接收命令列表
    2. 按顺序执行每个命令
    3. 更新对话状态（Tracker, Stack）
    4. 返回产生的事件和下一步动作
    """
    
    def __init__(
        self,
        config: Optional[ProcessorConfig] = None,
        domain: Optional["Domain"] = None,
        flows: Optional[List[Any]] = None,
    ):
        """初始化处理器。
        
        Args:
            config: 处理器配置
            domain: Domain定义
            flows: 可用的Flow列表
        """
        self.config = config or ProcessorConfig()
        self.domain = domain
        self.flows = flows or []
        self._flow_ids = set(
            getattr(f, 'id', str(f)) for f in self.flows
        ) if self.flows else set()
    
    def process(
        self,
        commands: List[Command],
        tracker: "DialogueStateTracker",
    ) -> ProcessResult:
        """处理命令列表。
        
        Args:
            commands: 要处理的命令列表
            tracker: 对话状态追踪器
            
        Returns:
            处理结果
        """
        result = ProcessResult()
        
        if not commands:
            logger.debug("No commands to process")
            return result
        
        logger.debug(f"Processing {len(commands)} commands")
        
        # 基于 force_slot_filling 机制，过滤 collect 步骤中的无效命令
        commands = self._filter_commands_during_collect(commands, tracker)
        
        if not commands:
            logger.debug("All commands filtered out during collect step")
            return result
        
        for command in commands:
            try:
                # 执行命令
                events = self._execute_command(command, tracker)
                result.events.extend(events)
                result.commands_executed += 1
                
                # 确定响应类型
                self._update_response_type(command, result)
                
            except Exception as e:
                error_msg = f"Failed to execute {command.command_name()}: {e}"
                result.errors.append(error_msg)
                logger.error(error_msg)
        
        # 确定下一步动作
        self._determine_next_action(tracker, result)
        
        return result
    
    def _execute_command(
        self,
        command: Command,
        tracker: "DialogueStateTracker",
    ) -> List[Dict[str, Any]]:
        """执行单个命令。
        
        Args:
            command: 要执行的命令
            tracker: 对话状态追踪器
            
        Returns:
            产生的事件列表
        """
        # 使用命令自身的run方法
        events = command.run(tracker, self.flows)
        
        # 记录命令到tracker
        tracker.add_commands([command.as_dict()])
        
        return events
    
    def _update_response_type(
        self,
        command: Command,
        result: ProcessResult,
    ) -> None:
        """更新响应类型。
        
        Args:
            command: 执行的命令
            result: 处理结果
        """
        if isinstance(command, StartFlowCommand):
            result.response_type = "flow"
        elif isinstance(command, CancelFlowCommand):
            result.response_type = "cancel_flow"
        elif isinstance(command, ChangeFlowCommand):
            result.response_type = "change_flow"
            result.metadata["target_flow"] = command.flow
        elif isinstance(command, SessionStartCommand):
            result.response_type = "session_start"
        elif isinstance(command, RestartCommand):
            result.response_type = "restart"
        elif isinstance(command, ChitChatAnswerCommand):
            result.response_type = "chitchat"
        elif isinstance(command, KnowledgeAnswerCommand):
            result.response_type = "knowledge"
        elif isinstance(command, CannotHandleCommand):
            result.response_type = "cannot_handle"
        elif isinstance(command, ClarifyCommand):
            result.response_type = "clarify"
        elif isinstance(command, HumanHandoffCommand):
            result.response_type = "human_handoff"
        # SetSlotCommand 不改变响应类型
    
    def _determine_next_action(
        self,
        tracker: "DialogueStateTracker",
        result: ProcessResult,
    ) -> None:
        """确定下一步动作。
        
        根据当前状态和处理结果，决定下一步应该执行什么动作。
        
        注意：对于已经在 Command.run() 中压入栈帧的 Command 类型
        （chitchat, knowledge, cannot_handle, human_handoff），
        不设置 next_action，让 Policy 通过检测栈帧来决定动作。
        
        Args:
            tracker: 对话状态追踪器
            result: 处理结果
        """
        # 根据响应类型决定下一步
        if result.response_type == "flow":
            # Flow已启动，执行Flow的下一步
            if tracker.active_flow:
                result.next_action = f"action_run_flow_{tracker.active_flow}"
            else:
                result.next_action = "action_listen"
        
        elif result.response_type == "cancel_flow":
            result.next_action = "action_cancel_flow"
        
        elif result.response_type == "change_flow":
            result.next_action = "action_change_flow"
        
        elif result.response_type == "session_start":
            result.next_action = "action_session_start"
        
        elif result.response_type == "restart":
            result.next_action = "action_restart"
        
        elif result.response_type == "clarify":
            # ClarifyCommand 没有压入栈帧，需要显式设置 next_action
            result.next_action = "action_clarify"
        
        # 以下类型的 Command 已经在 run() 中压入了栈帧，
        # 不设置 next_action，让 Policy 检测栈帧来决定动作：
        # - chitchat → ChitChatStackFrame → Policy 返回 action_send_text
        # - knowledge → SearchStackFrame → Policy 返回 action_send_text
        # - cannot_handle → CannotHandleStackFrame → Policy 返回 action_send_text
        # - human_handoff → HumanHandoffStackFrame → Policy 返回 action_send_text
        
        elif result.response_type in ("chitchat", "knowledge", "cannot_handle", "human_handoff"):
            # 不设置 next_action，让 Policy 通过栈帧检测来决定
            pass
        
        else:
            # 默认情况处理
            # 如果存在活跃 flow，不设置 next_action，让 FlowPolicy 决定下一步
            # 这对于按钮点击 SetSlots 等情况尤为重要
            if tracker.active_flow:
                # 不设置 next_action，让 FlowPolicy 处理 flow 的下一步
                pass
            else:
                # 没有活跃 flow，等待用户输入
                result.next_action = "action_listen"
    
    def validate_command(self, command: Command) -> bool:
        """验证命令是否有效。
        
        Args:
            command: 要验证的命令
            
        Returns:
            是否有效
        """
        # 验证StartFlowCommand的flow是否存在
        if isinstance(command, StartFlowCommand):
            if self.config.validate_flows and self._flow_ids:
                return command.flow in self._flow_ids
        
        # 验证SetSlotCommand的槽位是否存在
        if isinstance(command, SetSlotCommand):
            if self.config.validate_slots and self.domain:
                return command.name in self.domain.slots
        
        return True
    
    def filter_valid_commands(self, commands: List[Command]) -> List[Command]:
        """过滤出有效的命令。
        
        Args:
            commands: 命令列表
            
        Returns:
            有效的命令列表
        """
        return [cmd for cmd in commands if self.validate_command(cmd)]
    
    def _get_current_slot_to_collect(
        self,
        tracker: "DialogueStateTracker",
    ) -> Optional[str]:
        """获取当前正在收集的槽位名称。
        
        从 DialogueStack 的 FlowStackFrame 中获取 slot_to_collect。
        
        Args:
            tracker: 对话状态追踪器
            
        Returns:
            当前正在收集的槽位名，如果不在 collect 步骤则返回 None
        """
        if not hasattr(tracker, 'dialogue_stack'):
            return None
        
        flow_frame = tracker.dialogue_stack.top_flow_frame()
        if flow_frame and hasattr(flow_frame, 'slot_to_collect'):
            return flow_frame.slot_to_collect
        
        return None
    
    def _filter_commands_during_collect(
        self,
        commands: List[Command],
        tracker: "DialogueStateTracker",
    ) -> List[Command]:
        """基于 force_slot_filling 机制，过滤 collect 步骤中的无效命令。
        
        当处于 collect 步骤（正在收集某个槽位）时：
        1. 只保留设置当前槽位的 SetSlotCommand
        2. 丢弃其他 SetSlotCommand（防止 LLM 同时设置多个槽位）
        3. 丢弃 StartFlowCommand（防止 LLM 错误触发新流程）
        4. 保留其他命令（如 CancelFlowCommand）
        
        Args:
            commands: 原始命令列表
            tracker: 对话状态追踪器
            
        Returns:
            过滤后的命令列表
        """
        slot_to_collect = self._get_current_slot_to_collect(tracker)
        
        if not slot_to_collect:
            # 不在 collect 步骤，不过滤
            return commands
        
        logger.debug(f"[force_slot_filling] 当前正在收集槽位: {slot_to_collect}")
        
        filtered = []
        for command in commands:
            if isinstance(command, SetSlotCommand):
                if command.name == slot_to_collect:
                    filtered.append(command)
                    logger.debug(f"[force_slot_filling] 保留 SetSlotCommand: {command.name}={command.value}")
                else:
                    logger.warning(
                        f"[force_slot_filling] 忽略非当前槽位的设置: {command.name}, "
                        f"当前正在收集: {slot_to_collect}"
                    )
            elif isinstance(command, StartFlowCommand):
                # collect 步骤中禁止启动新流程（防止 LLM 误判用户意图）
                logger.warning(
                    f"[force_slot_filling] 忽略 collect 步骤中的 StartFlowCommand: {command.flow}"
                )
            else:
                # 保留其他命令（如 CancelFlowCommand、ChitChatAnswerCommand 等）
                filtered.append(command)
        
        return filtered
    
    def set_domain(self, domain: "Domain") -> None:
        """设置Domain。
        
        Args:
            domain: Domain定义
        """
        self.domain = domain
    
    def set_flows(self, flows: List[Any]) -> None:
        """设置可用的Flows。
        
        Args:
            flows: Flow列表
        """
        self.flows = flows
        self._flow_ids = set(
            getattr(f, 'id', str(f)) for f in flows
        ) if flows else set()


# 便捷函数

def process_commands(
    commands: List[Command],
    tracker: "DialogueStateTracker",
    domain: Optional["Domain"] = None,
    flows: Optional[List[Any]] = None,
) -> ProcessResult:
    """便捷函数：处理命令。
    
    Args:
        commands: 命令列表
        tracker: 对话状态追踪器
        domain: Domain定义
        flows: Flow列表
        
    Returns:
        处理结果
    """
    processor = CommandProcessor(domain=domain, flows=flows)
    return processor.process(commands, tracker)


# 导出
__all__ = [
    "CommandProcessor",
    "ProcessorConfig",
    "ProcessResult",
    "process_commands",
]
