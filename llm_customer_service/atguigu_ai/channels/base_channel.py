# -*- coding: utf-8 -*-
"""
通道基类

定义输入和输出通道的抽象接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Awaitable
import time
import uuid


@dataclass
class UserMessage:
    """用户消息。
    
    Attributes:
        text: 消息文本
        sender_id: 发送者ID
        input_channel: 输入通道名称
        message_id: 消息ID
        timestamp: 时间戳
        metadata: 元数据
    """
    text: str
    sender_id: str = "default"
    input_channel: str = "default"
    message_id: Optional[str] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class BotResponse:
    """机器人响应。
    
    Attributes:
        text: 响应文本
        recipient_id: 接收者ID
        buttons: 按钮列表
        image: 图片URL
        attachment: 附件
        custom: 自定义数据
    """
    text: str = ""
    recipient_id: str = ""
    buttons: List[Dict[str, Any]] = field(default_factory=list)
    image: Optional[str] = None
    attachment: Optional[str] = None
    custom: Optional[Dict[str, Any]] = None


class OutputChannel(ABC):
    """输出通道抽象基类。
    
    负责发送消息给用户。
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """通道名称。"""
        raise NotImplementedError()
    
    @abstractmethod
    async def send_response(
        self,
        recipient_id: str,
        message: Dict[str, Any],
    ) -> None:
        """发送响应消息。
        
        Args:
            recipient_id: 接收者ID
            message: 消息内容
        """
        raise NotImplementedError()
    
    async def send_text_message(
        self,
        recipient_id: str,
        text: str,
        **kwargs: Any,
    ) -> None:
        """发送文本消息。
        
        Args:
            recipient_id: 接收者ID
            text: 消息文本
            **kwargs: 额外参数
        """
        await self.send_response(recipient_id, {"text": text, **kwargs})
    
    async def send_text_with_buttons(
        self,
        recipient_id: str,
        text: str,
        buttons: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """发送带按钮的文本消息。"""
        await self.send_response(
            recipient_id,
            {"text": text, "buttons": buttons, **kwargs},
        )
    
    async def send_image_url(
        self,
        recipient_id: str,
        image: str,
        **kwargs: Any,
    ) -> None:
        """发送图片消息。"""
        await self.send_response(recipient_id, {"image": image, **kwargs})
    
    async def send_custom_json(
        self,
        recipient_id: str,
        json_message: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """发送自定义JSON消息。"""
        await self.send_response(recipient_id, {"custom": json_message, **kwargs})


class CollectingOutputChannel(OutputChannel):
    """收集输出通道。
    
    将所有发送的消息收集到列表中，用于测试或同步处理。
    """
    
    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []
    
    @property
    def name(self) -> str:
        return "collector"
    
    async def send_response(
        self,
        recipient_id: str,
        message: Dict[str, Any],
    ) -> None:
        """收集消息。"""
        self.messages.append({
            "recipient_id": recipient_id,
            **message,
        })
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """获取所有收集的消息。"""
        return self.messages
    
    def clear(self) -> None:
        """清空收集的消息。"""
        self.messages.clear()


class InputChannel(ABC):
    """输入通道抽象基类。
    
    负责接收用户消息并传递给Agent处理。
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """通道名称。"""
        raise NotImplementedError()
    
    def get_output_channel(self) -> Optional[OutputChannel]:
        """获取关联的输出通道。"""
        return None
    
    def get_metadata(self, request: Any) -> Optional[Dict[str, Any]]:
        """从请求中提取元数据。"""
        return None
    
    @classmethod
    def from_credentials(
        cls,
        credentials: Optional[Dict[str, Any]],
    ) -> "InputChannel":
        """从认证配置创建通道。"""
        return cls()


# 消息处理回调类型
MessageHandler = Callable[[UserMessage], Awaitable[Any]]


# 导出
__all__ = [
    "InputChannel",
    "OutputChannel",
    "UserMessage",
    "BotResponse",
    "CollectingOutputChannel",
    "MessageHandler",
]
