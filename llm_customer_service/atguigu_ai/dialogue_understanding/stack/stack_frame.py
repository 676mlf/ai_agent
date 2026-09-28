# -*- coding: utf-8 -*-
"""
栈帧定义

定义对话栈中的各种帧类型。
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type
import uuid


def generate_frame_id() -> str:
    """生成唯一的帧ID。"""
    return uuid.uuid4().hex[:8]


# 帧类型注册表
_FRAME_TYPE_REGISTRY: Dict[str, Type["StackFrame"]] = {}


def register_frame_type(cls: Type["StackFrame"]) -> Type["StackFrame"]:
    """栈帧类型注册装饰器。"""
    _FRAME_TYPE_REGISTRY[cls.frame_type()] = cls
    return cls


class FrameState(str, Enum):
    """帧状态枚举。"""
    ACTIVE = "active"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class FlowFrameType(str, Enum):
    """Flow帧类型枚举。"""
    REGULAR = "regular"
    INTERRUPT = "interrupt"
    LINK = "link"


@dataclass
class StackFrame(ABC):
    """栈帧基类。
    
    栈帧表示对话栈中的一个条目，记录对话上下文的一个状态点。
    
    Attributes:
        frame_id: 帧的唯一标识
        state: 帧的当前状态
    """
    
    frame_id: str = field(default_factory=generate_frame_id)
    state: FrameState = FrameState.ACTIVE
    
    @classmethod
    @abstractmethod
    def frame_type(cls) -> str:
        """返回帧类型标识。"""
        raise NotImplementedError()
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StackFrame":
        """从字典创建帧实例。"""
        raise NotImplementedError()
    
    def as_dict(self) -> Dict[str, Any]:
        """将帧转换为字典。"""
        data = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Enum):
                data[f.name] = value.value
            else:
                data[f.name] = value
        data["type"] = self.frame_type()
        return data
    
    def is_active(self) -> bool:
        """检查帧是否处于活动状态。"""
        return self.state == FrameState.ACTIVE
    
    def is_completed(self) -> bool:
        """检查帧是否已完成。"""
        return self.state == FrameState.COMPLETED
    
    def complete(self) -> None:
        """将帧标记为已完成。"""
        self.state = FrameState.COMPLETED
    
    def interrupt(self) -> None:
        """将帧标记为已中断。"""
        self.state = FrameState.INTERRUPTED
    
    def cancel(self) -> None:
        """将帧标记为已取消。"""
        self.state = FrameState.CANCELLED


@register_frame_type
@dataclass
class FlowStackFrame(StackFrame):
    """Flow栈帧。
    
    表示一个正在执行的Flow。
    
    Attributes:
        flow_id: Flow的ID
        step_id: 当前步骤ID
        flow_frame_type: Flow帧类型（regular, interrupt, link）
        slot_to_collect: 当前正在收集的槽位名称
        completing: Flow是否正在完成中（action执行完后需要触发action_flow_completed）
    """
    
    flow_id: str = ""
    step_id: str = "START"
    flow_frame_type: FlowFrameType = FlowFrameType.REGULAR
    slot_to_collect: Optional[str] = None
    completing: bool = False
    
    @classmethod
    def frame_type(cls) -> str:
        """返回帧类型标识。"""
        return "flow"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlowStackFrame":
        """从字典创建帧实例。"""
        state = data.get("state", FrameState.ACTIVE.value)
        if isinstance(state, str):
            state = FrameState(state)
        
        # 支持 flow_frame_type 和旧的 frame_type 字段
        flow_frame_type = data.get("flow_frame_type") or data.get("frame_type", FlowFrameType.REGULAR.value)
        if isinstance(flow_frame_type, str):
            flow_frame_type = FlowFrameType(flow_frame_type)
        
        return FlowStackFrame(
            frame_id=data.get("frame_id", generate_frame_id()),
            state=state,
            flow_id=data.get("flow_id", ""),
            step_id=data.get("step_id", "START"),
            flow_frame_type=flow_frame_type,
            slot_to_collect=data.get("slot_to_collect"),
            completing=data.get("completing", False),
        )
    
    def advance_to_step(self, step_id: str) -> None:
        """前进到指定步骤。"""
        self.step_id = step_id
    
    def is_interrupt(self) -> bool:
        """检查是否是中断帧。"""
        return self.flow_frame_type == FlowFrameType.INTERRUPT


@register_frame_type
@dataclass
class SearchStackFrame(StackFrame):
    """搜索栈帧。
    
    表示正在执行知识库搜索（RAG）。
    不存储query，从latest_message获取。
    """
    
    @classmethod
    def frame_type(cls) -> str:
        """返回帧类型标识。"""
        return "search"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchStackFrame":
        """从字典创建帧实例。"""
        state = data.get("state", FrameState.ACTIVE.value)
        if isinstance(state, str):
            state = FrameState(state)
        
        return SearchStackFrame(
            frame_id=data.get("frame_id", generate_frame_id()),
            state=state,
        )


@register_frame_type
@dataclass
class ChitChatStackFrame(StackFrame):
    """闲聊栈帧。
    
    表示正在处理闲聊。
    """
    
    @classmethod
    def frame_type(cls) -> str:
        """返回帧类型标识。"""
        return "chitchat"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChitChatStackFrame":
        """从字典创建帧实例。"""
        state = data.get("state", FrameState.ACTIVE.value)
        if isinstance(state, str):
            state = FrameState(state)
        
        return ChitChatStackFrame(
            frame_id=data.get("frame_id", generate_frame_id()),
            state=state,
        )


@register_frame_type
@dataclass
class CannotHandleStackFrame(StackFrame):
    """无法处理栈帧。
    
    表示系统无法处理用户请求。
    
    Attributes:
        reason: 无法处理的原因
    """
    
    reason: str = ""
    
    @classmethod
    def frame_type(cls) -> str:
        """返回帧类型标识。"""
        return "cannot_handle"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CannotHandleStackFrame":
        """从字典创建帧实例。"""
        state = data.get("state", FrameState.ACTIVE.value)
        if isinstance(state, str):
            state = FrameState(state)
        
        return CannotHandleStackFrame(
            frame_id=data.get("frame_id", generate_frame_id()),
            state=state,
            reason=data.get("reason", ""),
        )


@register_frame_type
@dataclass
class CompletedStackFrame(StackFrame):
    """完成栈帧。
    
    表示所有Flow已完成，系统处于空闲状态。
    由 FlowPolicy 在 Flow 完成后自动压入。
    由 EnterpriseSearchPolicy 处理，生成询问用户是否还有其他需求的响应。
    
    Attributes:
        previous_flow_name: 刚完成的Flow名称
    """
    
    previous_flow_name: str = ""
    
    @classmethod
    def frame_type(cls) -> str:
        """返回帧类型标识。"""
        return "completed"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompletedStackFrame":
        """从字典创建帧实例。"""
        state = data.get("state", FrameState.ACTIVE.value)
        if isinstance(state, str):
            state = FrameState(state)
        
        return CompletedStackFrame(
            frame_id=data.get("frame_id", generate_frame_id()),
            state=state,
            previous_flow_name=data.get("previous_flow_name", ""),
        )


@register_frame_type
@dataclass
class HumanHandoffStackFrame(StackFrame):
    """人工转接栈帧。
    
    表示需要将对话转接给人工客服。
    由 ActionHumanHandoff 压入，由 EnterpriseSearchPolicy 处理。
    
    Attributes:
        reason: 转接原因
    """
    
    reason: str = ""
    
    @classmethod
    def frame_type(cls) -> str:
        """返回帧类型标识。"""
        return "human_handoff"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanHandoffStackFrame":
        """从字典创建帧实例。"""
        state = data.get("state", FrameState.ACTIVE.value)
        if isinstance(state, str):
            state = FrameState(state)
        
        return HumanHandoffStackFrame(
            frame_id=data.get("frame_id", generate_frame_id()),
            state=state,
            reason=data.get("reason", ""),
        )


def create_frame_from_dict(data: Dict[str, Any]) -> StackFrame:
    """从字典创建栈帧。
    
    根据type字段确定帧类型，然后创建对应的帧实例。
    
    Args:
        data: 包含帧数据的字典
        
    Returns:
        栈帧实例
        
    Raises:
        ValueError: 如果帧类型未知
    """
    frame_type = data.get("type")
    if not frame_type:
        raise ValueError("Missing 'type' field in frame data")
    
    frame_cls = _FRAME_TYPE_REGISTRY.get(frame_type)
    if frame_cls is None:
        raise ValueError(f"Unknown frame type: {frame_type}")
    
    return frame_cls.from_dict(data)


# 导出
__all__ = [
    "StackFrame",
    "FlowStackFrame",
    "SearchStackFrame",
    "ChitChatStackFrame",
    "CannotHandleStackFrame",
    "CompletedStackFrame",
    "HumanHandoffStackFrame",
    "FrameState",
    "FlowFrameType",
    "generate_frame_id",
    "register_frame_type",
    "create_frame_from_dict",
]
