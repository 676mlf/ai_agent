# -*- coding: utf-8 -*-
"""
Inspect代理

为开发调试提供实时状态查看功能。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Awaitable, TYPE_CHECKING

from atguigu_ai.channels.base_channel import InputChannel, UserMessage

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker
    from atguigu_ai.agent.message_processor import MessageProcessor

logger = logging.getLogger(__name__)


class TrackerStream:
    """Tracker状态流。
    
    通过WebSocket向连接的客户端广播Tracker状态更新。
    """
    
    def __init__(
        self,
        get_tracker_state: Callable[[str], Awaitable[str]],
    ):
        """初始化。
        
        Args:
            get_tracker_state: 获取Tracker状态的回调
        """
        self.get_tracker_state = get_tracker_state
        self._clients: Set[Any] = set()
    
    def add_client(self, websocket: Any) -> None:
        """添加客户端。"""
        self._clients.add(websocket)
    
    def remove_client(self, websocket: Any) -> None:
        """移除客户端。"""
        self._clients.discard(websocket)
    
    async def broadcast(self, message: str) -> None:
        """广播消息给所有客户端。"""
        if not self._clients:
            return
        
        tasks = []
        for client in self._clients.copy():
            try:
                tasks.append(self._send(client, message))
            except Exception:
                self._clients.discard(client)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send(self, websocket: Any, message: str) -> None:
        """发送消息给单个客户端。"""
        try:
            await websocket.send_text(message)
        except Exception:
            self._clients.discard(websocket)


class InspectProxy(InputChannel):
    """Inspect代理通道。
    
    包装底层通道，提供对话状态的实时查看功能。
    
    功能：
    - 包装任意输入通道
    - 提供/inspect.html页面查看对话状态
    - 通过WebSocket实时推送状态更新
    """
    
    def __init__(
        self,
        underlying_channel: InputChannel,
        processor: Optional["MessageProcessor"] = None,
    ):
        """初始化Inspect代理。
        
        Args:
            underlying_channel: 被包装的底层通道
            processor: 消息处理器
        """
        self.underlying = underlying_channel
        self.processor = processor
        self.tracker_stream = TrackerStream(get_tracker_state=self._get_tracker_state)
    
    @property
    def name(self) -> str:
        return f"inspect_{self.underlying.name}"
    
    def set_processor(self, processor: "MessageProcessor") -> None:
        """设置消息处理器。"""
        self.processor = processor
    
    async def _get_tracker_state(self, sender_id: str) -> str:
        """获取Tracker状态的JSON字符串。"""
        if not self.processor:
            return "{}"
        
        # 获取tracker
        tracker = await self.processor.domain.tracker_store.retrieve(sender_id)
        if not tracker:
            return "{}"
        
        # 转换为字典
        state = tracker.current_state()
        return json.dumps(state, ensure_ascii=False, default=str)
    
    async def on_tracker_updated(self, tracker: "DialogueStateTracker") -> None:
        """Tracker更新时广播状态。"""
        try:
            state = tracker.current_state()
            state_json = json.dumps(state, ensure_ascii=False, default=str)
            await self.tracker_stream.broadcast(state_json)
        except Exception as e:
            logger.error(f"Failed to broadcast tracker state: {e}")
    
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
        from fastapi import APIRouter, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
        
        router = APIRouter(tags=["inspect"])
        
        # 包装消息处理回调，添加状态广播
        async def wrapped_handler(message: UserMessage) -> Any:
            result = await on_new_message(message)
            # 广播状态更新由钩子处理
            return result
        
        # 添加底层通道的路由
        if hasattr(self.underlying, 'create_routes'):
            underlying_router = self.underlying.create_routes(wrapped_handler)
            router.include_router(underlying_router)
        
        @router.get("/inspect.html", response_class=HTMLResponse)
        async def inspect_page():
            """Inspect页面。"""
            return self._get_inspect_html()
        
        @router.websocket("/tracker_stream")
        async def tracker_stream(websocket: WebSocket):
            """Tracker状态WebSocket流。"""
            await websocket.accept()
            self.tracker_stream.add_client(websocket)
            
            try:
                while True:
                    # 接收客户端消息
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # 处理获取状态请求
                    if message.get("action") == "retrieve":
                        sender_id = message.get("sender_id", "default")
                        state = await self._get_tracker_state(sender_id)
                        await websocket.send_text(state)
                        
            except WebSocketDisconnect:
                pass
            finally:
                self.tracker_stream.remove_client(websocket)
        
        return router
    
    def _get_inspect_html(self) -> str:
        """返回Inspect页面HTML。"""
        return """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Atguigu AI - 对话调试器</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
        .container { display: flex; height: 100vh; }
        .panel { flex: 1; padding: 20px; overflow-y: auto; }
        .panel-left { background: white; border-right: 1px solid #ddd; }
        .panel-right { background: #fafafa; }
        h2 { margin-bottom: 15px; color: #333; }
        .chat-box { height: calc(100vh - 200px); overflow-y: auto; border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #fff; margin-bottom: 15px; }
        .message { margin-bottom: 10px; padding: 10px 15px; border-radius: 18px; max-width: 80%; }
        .user-message { background: #007bff; color: white; margin-left: auto; }
        .bot-message { background: #e9ecef; color: #333; }
        .input-area { display: flex; gap: 10px; }
        .input-area input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 25px; font-size: 14px; }
        .input-area button { padding: 12px 25px; background: #007bff; color: white; border: none; border-radius: 25px; cursor: pointer; }
        .input-area button:hover { background: #0056b3; }
        .state-section { margin-bottom: 20px; }
        .state-section h3 { margin-bottom: 10px; color: #666; font-size: 14px; text-transform: uppercase; }
        .state-content { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; }
        .slot-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }
        .slot-name { font-weight: 500; }
        .slot-value { color: #007bff; }
        pre { font-size: 12px; overflow-x: auto; white-space: pre-wrap; }
        .status { padding: 5px 10px; border-radius: 4px; font-size: 12px; margin-bottom: 15px; }
        .status.connected { background: #d4edda; color: #155724; }
        .status.disconnected { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <div class="panel panel-left">
            <h2>对话</h2>
            <div id="status" class="status disconnected">未连接</div>
            <div id="chat-box" class="chat-box"></div>
            <div class="input-area">
                <input type="text" id="message-input" placeholder="输入消息..." onkeypress="if(event.key==='Enter')sendMessage()">
                <button onclick="sendMessage()">发送</button>
            </div>
        </div>
        <div class="panel panel-right">
            <h2>对话状态</h2>
            <div class="state-section">
                <h3>槽位</h3>
                <div id="slots" class="state-content">暂无数据</div>
            </div>
            <div class="state-section">
                <h3>活动Flow</h3>
                <div id="flow" class="state-content">暂无</div>
            </div>
            <div class="state-section">
                <h3>原始状态</h3>
                <div class="state-content"><pre id="raw-state">{}</pre></div>
            </div>
        </div>
    </div>
    
    <script>
        const senderId = 'inspect_user_' + Date.now();
        let ws = null;
        
        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/tracker_stream`);
            
            ws.onopen = () => {
                document.getElementById('status').className = 'status connected';
                document.getElementById('status').textContent = '已连接';
                ws.send(JSON.stringify({action: 'retrieve', sender_id: senderId}));
            };
            
            ws.onclose = () => {
                document.getElementById('status').className = 'status disconnected';
                document.getElementById('status').textContent = '已断开 - 重连中...';
                setTimeout(connect, 3000);
            };
            
            ws.onmessage = (event) => {
                try {
                    const state = JSON.parse(event.data);
                    updateState(state);
                } catch (e) {
                    console.error('Parse error:', e);
                }
            };
        }
        
        function updateState(state) {
            // 更新槽位
            const slotsDiv = document.getElementById('slots');
            if (state.slots && Object.keys(state.slots).length > 0) {
                slotsDiv.innerHTML = Object.entries(state.slots)
                    .map(([name, value]) => `<div class="slot-item"><span class="slot-name">${name}</span><span class="slot-value">${JSON.stringify(value)}</span></div>`)
                    .join('');
            } else {
                slotsDiv.textContent = '暂无槽位';
            }
            
            // 更新Flow
            const flowDiv = document.getElementById('flow');
            flowDiv.textContent = state.active_flow || '暂无活动Flow';
            
            // 更新原始状态
            document.getElementById('raw-state').textContent = JSON.stringify(state, null, 2);
        }
        
        function addMessage(text, isUser) {
            const chatBox = document.getElementById('chat-box');
            const div = document.createElement('div');
            div.className = 'message ' + (isUser ? 'user-message' : 'bot-message');
            div.textContent = text;
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        async function sendMessage() {
            const input = document.getElementById('message-input');
            const text = input.value.trim();
            if (!text) return;
            
            addMessage(text, true);
            input.value = '';
            
            try {
                const response = await fetch('/webhooks/rest/webhook', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({sender: senderId, message: text})
                });
                const messages = await response.json();
                messages.forEach(msg => {
                    if (msg.text) addMessage(msg.text, false);
                });
            } catch (e) {
                addMessage('发送失败: ' + e.message, false);
            }
        }
        
        connect();
    </script>
</body>
</html>
"""


# 导出
__all__ = [
    "InspectProxy",
    "TrackerStream",
]
