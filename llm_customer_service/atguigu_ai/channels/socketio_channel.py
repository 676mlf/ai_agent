# -*- coding: utf-8 -*-
"""
SocketIO通道

基于Socket.IO的实时通道实现。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Awaitable, TYPE_CHECKING

from atguigu_ai.channels.base_channel import (
    InputChannel,
    OutputChannel,
    UserMessage,
)

if TYPE_CHECKING:
    from socketio import AsyncServer

logger = logging.getLogger(__name__)


class SocketIOOutputChannel(OutputChannel):
    """SocketIO输出通道。
    
    通过Socket.IO实时发送消息。
    """
    
    def __init__(
        self,
        sio: "AsyncServer",
        bot_message_evt: str = "bot_uttered",
    ) -> None:
        """初始化。
        
        Args:
            sio: Socket.IO服务器
            bot_message_evt: 机器人消息事件名
        """
        self.sio = sio
        self.bot_message_evt = bot_message_evt
    
    @property
    def name(self) -> str:
        return "socketio"
    
    async def send_response(
        self,
        recipient_id: str,
        message: Dict[str, Any],
    ) -> None:
        """发送消息。"""
        await self.sio.emit(
            self.bot_message_evt,
            message,
            room=recipient_id,
        )


class SocketIOChannel(InputChannel):
    """Socket.IO通道。
    
    提供基于WebSocket的实时双向通信。
    
    事件：
    - connect: 客户端连接
    - disconnect: 客户端断开
    - user_uttered: 用户发送消息
    - bot_uttered: 机器人回复消息
    - session_request: 请求会话
    """
    
    def __init__(
        self,
        user_message_evt: str = "user_uttered",
        bot_message_evt: str = "bot_uttered",
        session_persistence: bool = True,
        socketio_path: str = "/socket.io",
    ):
        """初始化Socket.IO通道。
        
        Args:
            user_message_evt: 用户消息事件名
            bot_message_evt: 机器人消息事件名
            session_persistence: 是否保持会话
            socketio_path: Socket.IO路径
        """
        self.user_message_evt = user_message_evt
        self.bot_message_evt = bot_message_evt
        self.session_persistence = session_persistence
        self.socketio_path = socketio_path
        self.sio: Optional["AsyncServer"] = None
    
    @property
    def name(self) -> str:
        return "socketio"
    
    def get_output_channel(self) -> Optional[OutputChannel]:
        if self.sio:
            return SocketIOOutputChannel(self.sio, self.bot_message_evt)
        return None
    
    def create_sio_server(self) -> "AsyncServer":
        """创建Socket.IO服务器。"""
        import socketio
        
        self.sio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins="*",
        )
        return self.sio
    
    def register_handlers(
        self,
        sio: "AsyncServer",
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ) -> None:
        """注册Socket.IO事件处理器。
        
        Args:
            sio: Socket.IO服务器
            on_new_message: 消息处理回调
        """
        self.sio = sio
        
        @sio.event
        async def connect(sid: str, environ: Dict[str, Any]) -> bool:
            """客户端连接。"""
            logger.info(f"Client connected: {sid}")
            return True
        
        @sio.event
        async def disconnect(sid: str) -> None:
            """客户端断开。"""
            logger.info(f"Client disconnected: {sid}")
        
        @sio.on("session_request")
        async def session_request(sid: str, data: Dict[str, Any]) -> None:
            """会话请求。"""
            session_id = data.get("session_id", sid)
            logger.info(f"Session request from {sid}: {session_id}")
            
            # 将客户端加入会话房间
            await sio.enter_room(sid, session_id)
            await sio.emit("session_confirm", {"session_id": session_id}, room=sid)
        
        @sio.on(self.user_message_evt)
        async def handle_message(sid: str, data: Dict[str, Any]) -> None:
            """处理用户消息。"""
            try:
                text = data.get("message", data.get("text", ""))
                sender_id = data.get("session_id", data.get("sender_id", sid))
                metadata = data.get("metadata", {})
                
                if not text:
                    return
                
                logger.debug(f"Received message from {sender_id}: {text}")
                
                # 创建用户消息
                user_message = UserMessage(
                    text=text,
                    sender_id=sender_id,
                    input_channel=self.name,
                    metadata=metadata,
                )
                
                # 处理消息
                await on_new_message(user_message)
                
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                await sio.emit(
                    self.bot_message_evt,
                    {"text": "抱歉，处理消息时出现错误。"},
                    room=sid,
                )
    
    def create_app(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ) -> Any:
        """创建ASGI应用。
        
        Args:
            on_new_message: 消息处理回调
            
        Returns:
            ASGI应用
        """
        import socketio
        
        sio = self.create_sio_server()
        self.register_handlers(sio, on_new_message)
        
        # 创建ASGI应用
        app = socketio.ASGIApp(sio)
        return app


# 导出
__all__ = [
    "SocketIOChannel",
    "SocketIOOutputChannel",
]
