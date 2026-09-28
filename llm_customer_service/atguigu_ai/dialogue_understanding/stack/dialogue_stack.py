# -*- coding: utf-8 -*-
"""
对话栈

管理对话上下文的栈结构，支持Flow嵌套、中断和恢复。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Type

from atguigu_ai.dialogue_understanding.stack.stack_frame import (
    StackFrame,
    FlowStackFrame,
    FrameState,
    create_frame_from_dict,
)


@dataclass
class DialogueStack:
    """对话栈。
    
    管理对话过程中的上下文栈，支持：
    - Flow嵌套执行
    - 中断和恢复
    - 模式触发
    
    栈采用后进先出(LIFO)结构，栈顶是当前活动的帧。
    
    Attributes:
        frames: 栈帧列表，最后一个是栈顶
    """
    
    frames: List[StackFrame] = field(default_factory=list)
    
    def push(self, frame: StackFrame) -> None:
        """将帧压入栈顶。
        
        Args:
            frame: 要压入的栈帧
        """
        self.frames.append(frame)
    
    def pop(self) -> Optional[StackFrame]:
        """弹出栈顶帧。
        
        Returns:
            栈顶帧，如果栈为空则返回None
        """
        if self.frames:
            return self.frames.pop()
        return None
    
    def top(self) -> Optional[StackFrame]:
        """获取栈顶帧（不弹出）。
        
        Returns:
            栈顶帧，如果栈为空则返回None
        """
        if self.frames:
            return self.frames[-1]
        return None
    
    def is_empty(self) -> bool:
        """检查栈是否为空。"""
        return len(self.frames) == 0
    
    def size(self) -> int:
        """返回栈的大小。"""
        return len(self.frames)
    
    def clear(self) -> None:
        """清空栈。"""
        self.frames.clear()
    
    def __len__(self) -> int:
        """返回栈的大小。"""
        return len(self.frames)
    
    def __iter__(self) -> Iterator[StackFrame]:
        """从栈顶到栈底迭代。"""
        return reversed(self.frames).__iter__()
    
    def bottom_up(self) -> Iterator[StackFrame]:
        """从栈底到栈顶迭代。"""
        return iter(self.frames)
    
    # ========== Flow相关操作 ==========
    
    def top_flow_frame(self) -> Optional[FlowStackFrame]:
        """获取栈顶的Flow帧。
        
        从栈顶开始查找第一个FlowStackFrame。
        
        Returns:
            Flow帧，如果没有则返回None
        """
        for frame in self:
            if isinstance(frame, FlowStackFrame):
                return frame
        return None
    
    def active_flow_frame(self) -> Optional[FlowStackFrame]:
        """获取当前活动的Flow帧。
        
        返回栈顶的活动状态的Flow帧。
        
        Returns:
            活动的Flow帧，如果没有则返回None
        """
        for frame in self:
            if isinstance(frame, FlowStackFrame) and frame.is_active():
                return frame
        return None
    
    def push_flow(self, flow_id: str, step_id: str = "START") -> FlowStackFrame:
        """压入新的Flow帧。
        
        Args:
            flow_id: Flow ID
            step_id: 起始步骤ID
            
        Returns:
            新创建的Flow帧
        """
        frame = FlowStackFrame(flow_id=flow_id, step_id=step_id)
        self.push(frame)
        return frame
    
    def find_flow_frame(self, flow_id: str) -> Optional[FlowStackFrame]:
        """查找指定Flow的帧。
        
        Args:
            flow_id: Flow ID
            
        Returns:
            找到的帧，如果不存在则返回None
        """
        for frame in self:
            if isinstance(frame, FlowStackFrame) and frame.flow_id == flow_id:
                return frame
        return None
    
    def has_flow(self, flow_id: str) -> bool:
        """检查栈中是否包含指定Flow。"""
        return self.find_flow_frame(flow_id) is not None
    
    def get_all_flow_ids(self) -> List[str]:
        """获取栈中所有Flow的ID列表。
        
        Returns:
            Flow ID列表，从栈顶到栈底排序
        """
        return [
            frame.flow_id
            for frame in self
            if isinstance(frame, FlowStackFrame)
        ]
    
    def pop_to_flow(self, flow_id: str) -> List[StackFrame]:
        """弹出栈帧直到到达指定Flow。
        
        弹出目标Flow之上的所有帧（不包括目标Flow本身）。
        
        Args:
            flow_id: 目标Flow ID
            
        Returns:
            被弹出的帧列表
        """
        popped = []
        while self.frames:
            top = self.top()
            if isinstance(top, FlowStackFrame) and top.flow_id == flow_id:
                break
            popped.append(self.pop())
        return popped
    
    def interrupt_top_flow(self) -> Optional[FlowStackFrame]:
        """中断栈顶的Flow。
        
        Returns:
            被中断的Flow帧，如果没有则返回None
        """
        flow_frame = self.top_flow_frame()
        if flow_frame:
            flow_frame.interrupt()
        return flow_frame
    
    # ========== 帧查找操作 ==========
    
    def find_frame(self, frame_id: str) -> Optional[StackFrame]:
        """根据帧ID查找帧。
        
        Args:
            frame_id: 帧ID
            
        Returns:
            找到的帧，如果不存在则返回None
        """
        for frame in self:
            if frame.frame_id == frame_id:
                return frame
        return None
    
    def find_frames_of_type(self, frame_type: Type[StackFrame]) -> List[StackFrame]:
        """查找指定类型的所有帧。
        
        Args:
            frame_type: 帧类型
            
        Returns:
            匹配的帧列表
        """
        return [frame for frame in self if isinstance(frame, frame_type)]
    
    def remove_frame(self, frame_id: str) -> Optional[StackFrame]:
        """移除指定ID的帧。
        
        Args:
            frame_id: 帧ID
            
        Returns:
            被移除的帧，如果不存在则返回None
        """
        for i, frame in enumerate(self.frames):
            if frame.frame_id == frame_id:
                return self.frames.pop(i)
        return None
    
    # ========== 序列化 ==========
    
    def as_dict(self) -> Dict[str, Any]:
        """将栈转换为字典。
        
        Returns:
            包含所有帧的字典
        """
        return {
            "frames": [frame.as_dict() for frame in self.frames]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueStack":
        """从字典创建栈。
        
        Args:
            data: 包含帧数据的字典
            
        Returns:
            DialogueStack实例
        """
        frames = []
        for frame_data in data.get("frames", []):
            try:
                frame = create_frame_from_dict(frame_data)
                frames.append(frame)
            except ValueError:
                # 跳过无法解析的帧
                continue
        
        stack = cls()
        stack.frames = frames
        return stack
    
    def copy(self) -> "DialogueStack":
        """创建栈的副本。
        
        Returns:
            栈的深拷贝
        """
        return DialogueStack.from_dict(self.as_dict())
    
    def __repr__(self) -> str:
        """返回栈的字符串表示。"""
        if self.is_empty():
            return "DialogueStack(empty)"
        
        frame_strs = []
        for frame in self:
            if isinstance(frame, FlowStackFrame):
                frame_strs.append(f"Flow({frame.flow_id}@{frame.step_id})")
            else:
                frame_strs.append(f"{frame.frame_type()}")
        
        return f"DialogueStack([{' > '.join(frame_strs)}])"


# 导出
__all__ = ["DialogueStack"]
