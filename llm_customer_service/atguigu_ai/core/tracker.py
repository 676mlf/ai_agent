# -*- coding: utf-8 -*-
"""
tracker - 对话状态追踪器

管理对话过程中的状态信息，包括槽位值、对话历史、活跃Flow等。
Tracker是对话系统的核心数据结构，记录完整的对话上下文。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Text, Union
from copy import deepcopy

from atguigu_ai.shared.constants import (
    ACTION_LISTEN,
    ACTION_SESSION_START,
    DEFAULT_SENDER_ID,
)
from atguigu_ai.core.slots import Slot, create_slot
from atguigu_ai.dialogue_understanding.stack.dialogue_stack import DialogueStack
from atguigu_ai.dialogue_understanding.stack.stack_frame import FlowStackFrame


@dataclass
class UserMessage:
    """用户消息
    
    封装用户发送的消息及其元数据。
    
    属性：
        text: 消息文本
        sender_id: 发送者ID
        timestamp: 时间戳
        input_channel: 输入通道名称
        metadata: 额外元数据
    """
    text: str
    sender_id: str = DEFAULT_SENDER_ID
    timestamp: float = field(default_factory=time.time)
    input_channel: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "text": self.text,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "input_channel": self.input_channel,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserMessage":
        """从字典创建用户消息"""
        return cls(
            text=data.get("text", ""),
            sender_id=data.get("sender_id", DEFAULT_SENDER_ID),
            timestamp=data.get("timestamp", time.time()),
            input_channel=data.get("input_channel"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BotMessage:
    """Bot响应消息
    
    封装Bot发送的响应消息。
    
    属性：
        text: 消息文本
        data: 结构化数据(按钮、图片等)
        timestamp: 时间戳
        metadata: 额外元数据
    """
    text: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "text": self.text,
            "data": self.data,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BotMessage":
        """从字典创建Bot消息"""
        return cls(
            text=data.get("text"),
            data=data.get("data", {}),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DialogueTurn:
    """对话轮次
    
    一个完整的对话轮次，包含用户消息和Bot响应。
    
    属性：
        user_message: 用户消息
        bot_messages: Bot响应消息列表
        commands: 生成的命令列表
        action_name: 执行的动作名称
        timestamp: 轮次时间戳
    """
    user_message: Optional[UserMessage] = None
    bot_messages: List[BotMessage] = field(default_factory=list)
    commands: List[Dict[str, Any]] = field(default_factory=list)
    action_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "user_message": self.user_message.to_dict() if self.user_message else None,
            "bot_messages": [m.to_dict() for m in self.bot_messages],
            "commands": self.commands,
            "action_name": self.action_name,
            "timestamp": self.timestamp,
        }


class DialogueStateTracker:
    """对话状态追踪器
    
    管理单个用户会话的完整对话状态。
    
    核心功能：
    - 记录对话历史(用户消息和Bot响应)
    - 管理槽位状态
    - 跟踪活跃的Flow（通过dialogue_stack）
    - 支持状态序列化和反序列化
    
    状态管理：
        使用统一的 dialogue_stack 管理所有对话上下文（Flow、搜索、闲聊等）。
        active_flow 是从 dialogue_stack 派生的计算属性。
    
    属性：
        sender_id: 会话ID(通常是用户ID)
        slots: 槽位状态字典
        dialogue_turns: 对话轮次历史
        dialogue_stack: 对话栈（唯一状态源）
        active_flow: 当前活跃的Flow名称（从dialogue_stack计算）
        latest_message: 最新的用户消息
        latest_action_name: 最新执行的动作
    """
    
    def __init__(
        self,
        sender_id: str = DEFAULT_SENDER_ID,
        slots: Optional[Dict[str, Slot]] = None,
        max_turns: int = 100,
    ) -> None:
        """初始化对话状态追踪器
        
        参数：
            sender_id: 会话ID
            slots: 初始槽位字典
            max_turns: 最大保留轮次数
        """
        self.sender_id = sender_id
        self.slots: Dict[str, Slot] = slots or {}
        self.max_turns = max_turns
        
        # 对话历史
        self.dialogue_turns: List[DialogueTurn] = []
        
        # 当前轮次(正在进行中)
        self._current_turn: Optional[DialogueTurn] = None
        
        # 统一的对话栈 - 唯一的状态管理
        self.dialogue_stack: DialogueStack = DialogueStack()
        
        # Flow历史记录（用于追溯）
        self.flow_history: List[Dict[str, Any]] = []
        
        # 最新状态
        self.latest_message: Optional[UserMessage] = None
        self.latest_action_name: str = ACTION_LISTEN
        
        # 元数据
        self.followup_action: Optional[str] = None
        self.paused: bool = False
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
    
    @property
    def active_flow(self) -> Optional[str]:
        """当前活跃的Flow名称（从dialogue_stack计算）"""
        frame = self.dialogue_stack.active_flow_frame()
        return frame.flow_id if frame else None
    
    def update_with_message(self, message: UserMessage) -> None:
        """使用新的用户消息更新状态
        
        开始新的对话轮次。
        
        参数：
            message: 用户消息
        """
        # 保存之前的轮次
        if self._current_turn is not None:
            self._save_current_turn()
        
        # 开始新轮次
        self._current_turn = DialogueTurn(user_message=message)
        self.latest_message = message
        # 重置 latest_action_name，表示等待新的动作
        self.latest_action_name = ACTION_LISTEN
        self.updated_at = time.time()
    
    def add_bot_message(self, message: BotMessage) -> None:
        """添加Bot响应消息
        
        参数：
            message: Bot消息
        """
        if self._current_turn is None:
            self._current_turn = DialogueTurn()
        
        self._current_turn.bot_messages.append(message)
        self.updated_at = time.time()
    
    def set_slot(
        self,
        slot_name: str,
        value: Any,
        create_if_missing: bool = True,
    ) -> None:
        """设置槽位值
        
        参数：
            slot_name: 槽位名称
            value: 槽位值
            create_if_missing: 槽位不存在时是否创建
        """
        if slot_name in self.slots:
            self.slots[slot_name].value = value
        elif create_if_missing:
            # 创建新槽位时 initial_value 应该是 None，这样 reset() 才能正确重置
            self.slots[slot_name] = create_slot(name=slot_name, initial_value=None)
            self.slots[slot_name].value = value
        
        self.updated_at = time.time()
    
    def get_slot(self, slot_name: str) -> Any:
        """获取槽位值
        
        参数：
            slot_name: 槽位名称
            
        返回：
            槽位值，不存在返回None
        """
        slot = self.slots.get(slot_name)
        return slot.value if slot else None
    
    def get_all_slots(self) -> Dict[str, Any]:
        """获取所有槽位的值
        
        返回：
            槽位名称到值的映射字典
        """
        return {name: slot.value for name, slot in self.slots.items()}
    
    def reset_slots(self) -> None:
        """重置所有槽位为初始值"""
        for slot in self.slots.values():
            slot.reset()
        self.updated_at = time.time()
    
    def set_latest_action(self, action_name: str) -> None:
        """设置最新执行的动作
        
        参数：
            action_name: 动作名称
        """
        self.latest_action_name = action_name
        if self._current_turn:
            self._current_turn.action_name = action_name
        self.updated_at = time.time()
    
    def add_commands(self, commands: List[Dict[str, Any]]) -> None:
        """添加生成的命令到当前轮次
        
        参数：
            commands: 命令列表
        """
        if self._current_turn is None:
            self._current_turn = DialogueTurn()
        
        self._current_turn.commands.extend(commands)
        self.updated_at = time.time()
    
    def start_flow(self, flow_name: str, step_id: str = "START") -> None:
        """开始执行Flow
        
        将FlowStackFrame压入dialogue_stack。
        
        参数：
            flow_name: Flow名称
            step_id: 起始步骤ID
        """
        # 压入FlowStackFrame到dialogue_stack
        self.dialogue_stack.push_flow(flow_name, step_id)
        
        # 记录到历史
        self.flow_history.append({
            "flow_name": flow_name,
            "started_at": time.time(),
            "ended_at": None,
            "completed": False,
        })
        self.updated_at = time.time()
    
    def end_flow(self) -> Optional[str]:
        """结束当前Flow
        
        从dialogue_stack弹出栈顶的FlowStackFrame。
        
        返回：
            结束的Flow名称，栈为空返回None
        """
        # 获取栈顶Flow帧
        flow_frame = self.dialogue_stack.top_flow_frame()
        if flow_frame is None:
            return None
        
        flow_name = flow_frame.flow_id
        
        # 弹出到该Flow（包括它上面的所有帧）
        self.dialogue_stack.pop_to_flow(flow_name)
        # 再弹出Flow本身
        self.dialogue_stack.pop()
        
        # 更新flow_history中对应项的结束信息
        for hist in reversed(self.flow_history):
            if hist["flow_name"] == flow_name and hist["ended_at"] is None:
                hist["ended_at"] = time.time()
                hist["completed"] = True
                break
        
        self.updated_at = time.time()
        return flow_name
    
    def cancel_flow(self) -> None:
        """取消所有活跃的Flow
        
        清空dialogue_stack。
        """
        self.dialogue_stack.clear()
        self.updated_at = time.time()
    
    def record_pattern(self, pattern_type: str, completed: bool = True) -> None:
        """记录内置 Pattern 的执行历史
        
        将内置 Pattern（如 chitchat、search、cannot_handle 等）的执行记录
        添加到 flow_history 中，以便在 Inspect 页面统一展示。
        
        参数：
            pattern_type: Pattern 类型，如 "chitchat"、"search"、"cannot_handle"、
                         "completed"、"human_handoff"
            completed: 是否执行完成，默认为 True
        """
        current_time = time.time()
        self.flow_history.append({
            "flow_name": f"pattern_{pattern_type}",
            "started_at": current_time,
            "ended_at": current_time if completed else None,
            "completed": completed,
        })
        self.updated_at = current_time
    
    def get_conversation_history(
        self,
        max_turns: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """获取对话历史
        
        参数：
            max_turns: 最大返回轮次数
            
        返回：
            对话轮次列表
        """
        turns = self.dialogue_turns[:]
        if self._current_turn:
            turns.append(self._current_turn)
        
        if max_turns:
            turns = turns[-max_turns:]
        
        return [turn.to_dict() for turn in turns]
    
    def get_messages_for_llm(
        self,
        max_turns: int = 10,
    ) -> List[Dict[str, str]]:
        """获取用于LLM的消息历史
        
        将对话历史转换为LLM消息格式。
        
        参数：
            max_turns: 最大轮次数
            
        返回：
            LLM消息列表
        """
        messages = []
        turns = self.dialogue_turns[-max_turns:] if max_turns else self.dialogue_turns
        
        for turn in turns:
            if turn.user_message:
                messages.append({
                    "role": "user",
                    "content": turn.user_message.text,
                })
            
            for bot_msg in turn.bot_messages:
                if bot_msg.text:
                    messages.append({
                        "role": "assistant",
                        "content": bot_msg.text,
                    })
        
        # 添加当前轮次的用户消息
        if self._current_turn and self._current_turn.user_message:
            messages.append({
                "role": "user",
                "content": self._current_turn.user_message.text,
            })
        
        return messages
    
    def _save_current_turn(self) -> None:
        """保存当前轮次到历史"""
        if self._current_turn:
            self.dialogue_turns.append(self._current_turn)
            
            # 限制历史长度
            if len(self.dialogue_turns) > self.max_turns:
                self.dialogue_turns = self.dialogue_turns[-self.max_turns:]
            
            self._current_turn = None
    
    def finalize_turn(self) -> None:
        """完成当前轮次"""
        self._save_current_turn()
        self.updated_at = time.time()
    
    def restart(self) -> None:
        """重启会话，清除所有状态"""
        self.dialogue_turns.clear()
        self._current_turn = None
        self.reset_slots()
        self.dialogue_stack.clear()
        self.latest_message = None
        self.latest_action_name = ACTION_LISTEN
        self.followup_action = None
        self.paused = False
        self.updated_at = time.time()
    
    def current_state(self) -> Dict[str, Any]:
        """获取当前完整状态
        
        返回：
            状态字典
        """
        return {
            "sender_id": self.sender_id,
            "slots": self.get_all_slots(),
            "active_flow": self.active_flow,
            "dialogue_stack": self.dialogue_stack.as_dict(),
            "latest_message": self.latest_message.to_dict() if self.latest_message else None,
            "latest_action_name": self.latest_action_name,
            "paused": self.paused,
            "dialogue_turns_count": len(self.dialogue_turns),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典
        
        用于持久化存储。
        
        返回：
            完整状态字典
        """
        # 确保当前轮次被保存
        self._save_current_turn()
        
        return {
            "sender_id": self.sender_id,
            "slots": {name: slot.to_dict() for name, slot in self.slots.items()},
            "dialogue_turns": [turn.to_dict() for turn in self.dialogue_turns],
            "dialogue_stack": self.dialogue_stack.as_dict(),
            "flow_history": self.flow_history,
            "latest_action_name": self.latest_action_name,
            "latest_message": self.latest_message.to_dict() if self.latest_message else None,
            "paused": self.paused,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        domain_slots: Optional[Dict[str, Slot]] = None,
    ) -> "DialogueStateTracker":
        """从字典反序列化
        
        参数：
            data: 状态字典
            domain_slots: Domain定义的槽位(用于恢复槽位类型)
            
        返回：
            DialogueStateTracker实例
        """
        # 恢复槽位
        slots = {}
        for slot_name, slot_data in data.get("slots", {}).items():
            if domain_slots and slot_name in domain_slots:
                # 使用Domain定义的槽位类型
                slot = deepcopy(domain_slots[slot_name])
                slot._value = slot_data.get("value")
            else:
                # 从数据恢复
                slot = Slot.from_dict(slot_data)
            slots[slot_name] = slot
        
        tracker = cls(
            sender_id=data.get("sender_id", DEFAULT_SENDER_ID),
            slots=slots,
        )
        
        # 恢复dialogue_stack
        if "dialogue_stack" in data:
            tracker.dialogue_stack = DialogueStack.from_dict(data["dialogue_stack"])
        
        # 恢复其他状态
        tracker.flow_history = data.get("flow_history", [])
        tracker.latest_action_name = data.get("latest_action_name", ACTION_LISTEN)
        tracker.paused = data.get("paused", False)
        tracker.created_at = data.get("created_at", time.time())
        tracker.updated_at = data.get("updated_at", time.time())
        
        # 恢复 latest_message
        if data.get("latest_message"):
            tracker.latest_message = UserMessage.from_dict(data["latest_message"])
        
        # 恢复对话历史(简化处理，只恢复基本信息)
        for turn_data in data.get("dialogue_turns", []):
            turn = DialogueTurn(
                timestamp=turn_data.get("timestamp", time.time()),
                action_name=turn_data.get("action_name"),
                commands=turn_data.get("commands", []),
            )
            
            if turn_data.get("user_message"):
                turn.user_message = UserMessage.from_dict(turn_data["user_message"])
            
            for bot_msg_data in turn_data.get("bot_messages", []):
                turn.bot_messages.append(BotMessage.from_dict(bot_msg_data))
            
            tracker.dialogue_turns.append(turn)
        
        return tracker
    
    def copy(self) -> "DialogueStateTracker":
        """创建Tracker的深拷贝
        
        返回：
            新的DialogueStateTracker实例
        """
        return DialogueStateTracker.from_dict(self.to_dict())
    
    def __repr__(self) -> str:
        return (
            f"DialogueStateTracker(sender_id={self.sender_id}, "
            f"slots={len(self.slots)}, turns={len(self.dialogue_turns)}, "
            f"active_flow={self.active_flow})"
        )
