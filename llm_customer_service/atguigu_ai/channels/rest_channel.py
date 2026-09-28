# -*- coding: utf-8 -*-
"""
REST通道

基于HTTP REST API的通道实现。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Callable, Awaitable, TYPE_CHECKING

from atguigu_ai.channels.base_channel import (
    InputChannel,
    OutputChannel,
    CollectingOutputChannel,
    UserMessage,
)

if TYPE_CHECKING:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class RestOutputChannel(OutputChannel):
    """REST输出通道。
    
    收集响应消息，等待一次性返回。
    """
    
    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []
    
    @property
    def name(self) -> str:
        return "rest"
    
    async def send_response(
        self,
        recipient_id: str,
        message: Dict[str, Any],
    ) -> None:
        """收集消息。"""
        self.messages.append(message)
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """获取所有消息。"""
        return self.messages


class RestChannel(InputChannel):
    """REST API通道。
    
    提供HTTP端点接收用户消息。
    
    端点：
    - POST /webhooks/rest/webhook: 发送消息
    - GET /webhooks/rest/: 健康检查
    """
    
    @property
    def name(self) -> str:
        return "rest"
    
    def get_output_channel(self) -> OutputChannel:
        return RestOutputChannel()
    
    def create_routes(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ) -> Any:
        """创建FastAPI路由。
        
        Args:
            on_new_message: 消息处理回调
            
        Returns:
            FastAPI Router
        """
        from fastapi import APIRouter, Request
        from fastapi.responses import JSONResponse
        
        router = APIRouter(prefix="/webhooks/rest", tags=["rest"])
        
        @router.get("/")
        async def health_check():
            """健康检查。"""
            return {"status": "ok", "channel": self.name}
        
        @router.post("/webhook")
        async def receive_message(request: Request):
            """接收用户消息。"""
            try:
                data = await request.json()
                
                sender_id = data.get("sender", "default")
                text = data.get("message", data.get("text", ""))
                metadata = data.get("metadata", {})
                
                if not text:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "No message text provided"},
                    )
                
                # 创建用户消息
                user_message = UserMessage(
                    text=text,
                    sender_id=sender_id,
                    input_channel=self.name,
                    metadata=metadata,
                )
                
                # 处理消息
                output_channel = RestOutputChannel()
                
                # 调用处理回调
                await on_new_message(user_message)
                
                return JSONResponse(content=output_channel.get_messages())
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)},
                )
        
        return router


# 导出
__all__ = [
    "RestChannel",
    "RestOutputChannel",
]
