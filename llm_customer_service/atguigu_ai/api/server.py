# -*- coding: utf-8 -*-
"""
FastAPI应用服务器

提供对话系统的Web服务接口，包括：
- REST API端点
- WebSocket实时通信
- 健康检查
- CORS支持
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from atguigu_ai.shared.constants import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
)
from atguigu_ai.channels.base_channel import UserMessage

if TYPE_CHECKING:
    from atguigu_ai.agent.agent import Agent

logger = logging.getLogger(__name__)


# Pydantic模型定义
class MessageRequest(BaseModel):
    """消息请求模型。"""
    sender: str = "user"
    message: str
    metadata: Optional[Dict[str, Any]] = None


class MessageResponse(BaseModel):
    """消息响应模型。"""
    recipient_id: str
    text: Optional[str] = None
    buttons: Optional[List[Dict[str, Any]]] = None
    image: Optional[str] = None
    custom: Optional[Dict[str, Any]] = None


class SessionInfo(BaseModel):
    """会话信息模型。"""
    session_id: str
    slots: Dict[str, Any]
    latest_message: Optional[Dict[str, Any]] = None
    events_count: int


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str
    version: str
    agent_ready: bool


class AtguiguServer:
    """FastAPI应用服务器。
    
    管理Agent实例和Web服务。
    """
    
    def __init__(
        self,
        agent: Optional[Agent] = None,
        cors_origins: Optional[List[str]] = None,
        enable_inspect: bool = True,
    ):
        """初始化服务器。
        
        Args:
            agent: Agent实例
            cors_origins: CORS允许的源列表
            enable_inspect: 是否启用调试页面
        """
        self.agent = agent
        self.cors_origins = cors_origins or ["*"]
        self.enable_inspect = enable_inspect
        
        # WebSocket连接管理
        self._ws_connections: Dict[str, List[WebSocket]] = {}
        
        # 创建FastAPI应用
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        """创建FastAPI应用。"""
        app = FastAPI(
            title="Atguigu AI",
            description="教学版对话系统API",
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )
        
        # 添加CORS中间件
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 注册路由
        self._register_routes(app)
        
        return app
    
    def _register_routes(self, app: FastAPI) -> None:
        """注册API路由。"""
        
        @app.get("/", response_model=HealthResponse)
        async def root():
            """根路径健康检查。"""
            from atguigu_ai import __version__
            return {
                "status": "ok",
                "version": __version__,
                "agent_ready": self.agent is not None,
            }
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """健康检查端点。"""
            from atguigu_ai import __version__
            return {
                "status": "ok",
                "version": __version__,
                "agent_ready": self.agent is not None,
            }
        
        @app.post("/api/messages", response_model=List[MessageResponse])
        async def send_message(request: MessageRequest):
            """发送消息到对话系统。
            
            Args:
                request: 消息请求
                
            Returns:
                Bot响应列表
            """
            if not self.agent:
                raise HTTPException(status_code=503, detail="Agent not ready")
            
            try:
                # 创建用户消息
                user_message = UserMessage(
                    text=request.message,
                    sender_id=request.sender,
                    input_channel="rest",
                    metadata=request.metadata or {},
                )
                
                # 处理消息
                response = await self.agent.handle_message(
                    message=user_message.text,
                    sender_id=user_message.sender_id,
                    metadata=user_message.metadata,
                )
                
                # 转换响应格式 - response 是 MessageResponse 对象
                result = []
                for msg in response.messages:
                    result.append({
                        "recipient_id": request.sender,
                        "text": msg.get("text"),
                        "buttons": msg.get("buttons"),
                        "image": msg.get("image"),
                        "custom": msg.get("custom"),
                    })
                
                # 广播到WebSocket连接
                await self._broadcast_to_session(
                    request.sender,
                    {"type": "bot_response", "data": result},
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/sessions/{session_id}", response_model=SessionInfo)
        async def get_session(session_id: str):
            """获取会话状态。
            
            Args:
                session_id: 会话ID
                
            Returns:
                会话信息
            """
            if not self.agent:
                raise HTTPException(status_code=503, detail="Agent not ready")
            
            try:
                tracker = await self.agent.get_tracker(session_id)
                if not tracker:
                    raise HTTPException(status_code=404, detail="Session not found")
                
                return {
                    "session_id": session_id,
                    "slots": tracker.get_all_slots(),
                    "latest_message": tracker.latest_message,
                    "events_count": len(tracker.dialogue_turns),
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting session: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/sessions/{session_id}/reset")
        async def reset_session(session_id: str):
            """重置会话。
            
            Args:
                session_id: 会话ID
                
            Returns:
                重置结果
            """
            if not self.agent:
                raise HTTPException(status_code=503, detail="Agent not ready")
            
            try:
                await self.agent.reset_tracker(session_id)
                return {"status": "ok", "message": f"Session {session_id} reset"}
                
            except Exception as e:
                logger.error(f"Error resetting session: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/domain")
        async def get_domain():
            """获取domain配置。"""
            if not self.agent:
                raise HTTPException(status_code=503, detail="Agent not ready")
            
            return self.agent.domain.as_dict()
        
        @app.get("/api/flows")
        async def get_flows():
            """获取所有Flow定义。"""
            if not self.agent:
                raise HTTPException(status_code=503, detail="Agent not ready")
            
            flows_data = []
            if self.agent.flows:
                for flow in self.agent.flows.flows:  # flows 是列表
                    steps_data = []
                    for step in flow.steps:
                        step_dict = {
                            "id": step.id,
                            "action": step.action,
                            "next": step.next,
                            "collect": step.collect,
                            "set_slots": step.set_slots if hasattr(step, 'set_slots') else None,
                            "step_type": step.step_type.value if hasattr(step.step_type, 'value') else str(step.step_type),
                        }
                        steps_data.append(step_dict)
                    
                    flows_data.append({
                        "id": flow.id,
                        "description": flow.description,
                        "steps": steps_data,
                    })
            
            return flows_data
        
        @app.get("/api/tracker/{session_id}/full")
        async def get_full_tracker(session_id: str):
            """获取完整的Tracker状态。
            
            包括：slots, events, flow_stack, messages 等。
            """
            if not self.agent:
                raise HTTPException(status_code=503, detail="Agent not ready")
            
            try:
                tracker = await self.agent.get_tracker(session_id)
                if not tracker:
                    raise HTTPException(status_code=404, detail="Session not found")
                
                # 构建完整的 tracker 状态
                events = []
                for turn in tracker.dialogue_turns:
                    if turn.user_message:
                        events.append({
                            "event": "user",
                            "text": turn.user_message.text,
                            "timestamp": turn.user_message.timestamp,
                        })
                    for bot_msg in turn.bot_messages:
                        events.append({
                            "event": "bot",
                            "text": bot_msg.text,
                            "timestamp": getattr(bot_msg, 'timestamp', None),
                        })
                
                # Flow 历史
                flow_stack = []
                for frame in tracker.dialogue_stack.frames:
                    from atguigu_ai.dialogue_understanding.stack.stack_frame import FlowStackFrame
                    if isinstance(frame, FlowStackFrame):
                        flow_stack.append({
                            "flow_id": frame.flow_id,
                            "step_id": frame.step_id,
                            "frame_id": frame.frame_id,
                        })
                
                # Flow 执行历史（包括已完成的）
                flow_history = []
                for hist in tracker.flow_history:
                    flow_history.append({
                        "flow_id": hist.get("flow_name", ""),
                        "started_at": hist.get("started_at", ""),
                        "ended_at": hist.get("ended_at"),
                        "completed": hist.get("completed", False),
                    })
                
                return {
                    "sender_id": session_id,
                    "slots": tracker.get_all_slots(),
                    "events": events,
                    "flow_stack": flow_stack,
                    "flow_history": flow_history,
                    "active_flow": tracker.active_flow,
                    "latest_action": tracker.latest_action_name,
                    "latest_message": {
                        "text": tracker.latest_message.text if tracker.latest_message else None,
                        "timestamp": tracker.latest_message.timestamp if tracker.latest_message else None,
                    } if tracker.latest_message else None,
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting full tracker: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.websocket("/api/stream")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket实时消息流。
            
            支持的消息类型：
            - connect: 连接并指定session_id
            - message: 发送消息
            - ping: 心跳
            """
            await websocket.accept()
            session_id = None
            
            try:
                while True:
                    data = await websocket.receive_json()
                    msg_type = data.get("type", "message")
                    
                    if msg_type == "connect":
                        session_id = data.get("session_id", "default")
                        self._add_ws_connection(session_id, websocket)
                        await websocket.send_json({
                            "type": "connected",
                            "session_id": session_id,
                        })
                    
                    elif msg_type == "message":
                        if not self.agent:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Agent not ready",
                            })
                            continue
                        
                        text = data.get("message", data.get("text", ""))
                        sender_id = data.get("sender_id", session_id or "default")
                        
                        try:
                            response = await self.agent.handle_message(
                                message=text,
                                sender_id=sender_id,
                            )
                            
                            # response 是 MessageResponse 对象，需要取出 messages 列表
                            await websocket.send_json({
                                "type": "bot_response",
                                "data": response.messages,
                            })
                            
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": str(e),
                            })
                    
                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                        
            except WebSocketDisconnect:
                if session_id:
                    self._remove_ws_connection(session_id, websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if session_id:
                    self._remove_ws_connection(session_id, websocket)
        
        # 调试页面
        if self.enable_inspect:
            @app.get("/inspect", response_class=HTMLResponse)
            async def inspect_page():
                """调试页面。"""
                return self._get_inspect_html()
    
    def _add_ws_connection(self, session_id: str, ws: WebSocket) -> None:
        """添加WebSocket连接。"""
        if session_id not in self._ws_connections:
            self._ws_connections[session_id] = []
        self._ws_connections[session_id].append(ws)
    
    def _remove_ws_connection(self, session_id: str, ws: WebSocket) -> None:
        """移除WebSocket连接。"""
        if session_id in self._ws_connections:
            if ws in self._ws_connections[session_id]:
                self._ws_connections[session_id].remove(ws)
            if not self._ws_connections[session_id]:
                del self._ws_connections[session_id]
    
    async def _broadcast_to_session(
        self,
        session_id: str,
        message: Dict[str, Any],
    ) -> None:
        """广播消息到会话的所有连接。"""
        if session_id not in self._ws_connections:
            return
        
        disconnected = []
        for ws in self._ws_connections[session_id]:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
        
        # 清理断开的连接
        for ws in disconnected:
            self._remove_ws_connection(session_id, ws)
    
    def _get_inspect_html(self) -> str:
        """获取调试页面HTML。"""
        import os
        template_path = os.path.join(
            os.path.dirname(__file__), "templates", "inspect.html"
        )
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def set_agent(self, agent: Agent) -> None:
        """设置Agent实例。"""
        self.agent = agent
    
    def run(
        self,
        host: str = DEFAULT_SERVER_HOST,
        port: int = DEFAULT_SERVER_PORT,
    ) -> None:
        """运行服务器。
        
        Args:
            host: 监听地址
            port: 监听端口
        """
        import uvicorn
        
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
        )


def create_app(
    agent: Optional[Agent] = None,
    cors_origins: Optional[List[str]] = None,
    enable_inspect: bool = True,
) -> FastAPI:
    """创建FastAPI应用。
    
    工厂函数，便于在不同场景下创建应用。
    
    Args:
        agent: Agent实例
        cors_origins: CORS允许的源列表
        enable_inspect: 是否启用调试页面
        
    Returns:
        FastAPI应用实例
    """
    server = AtguiguServer(
        agent=agent,
        cors_origins=cors_origins,
        enable_inspect=enable_inspect,
    )
    return server.app


# 导出
__all__ = [
    "AtguiguServer",
    "create_app",
    "MessageRequest",
    "MessageResponse",
    "SessionInfo",
    "HealthResponse",
]
