# -*- coding: utf-8 -*-
"""
控制台通道

用于命令行交互的通道。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Awaitable

from atguigu_ai.channels.base_channel import (
    InputChannel,
    OutputChannel,
    UserMessage,
)

logger = logging.getLogger(__name__)


class ConsoleOutputChannel(OutputChannel):
    """控制台输出通道。
    
    将消息打印到控制台。
    """
    
    @property
    def name(self) -> str:
        return "console"
    
    async def send_response(
        self,
        recipient_id: str,
        message: Dict[str, Any],
    ) -> None:
        """打印消息到控制台。"""
        text = message.get("text", "")
        if text:
            print(f"Bot: {text}")
        
        # 处理按钮
        buttons = message.get("buttons", [])
        if buttons:
            print("选项:")
            for i, button in enumerate(buttons, 1):
                title = button.get("title", button.get("text", ""))
                print(f"  {i}. {title}")
        
        # 处理图片
        image = message.get("image")
        if image:
            print(f"[图片: {image}]")


class ConsoleChannel(InputChannel):
    """控制台输入通道。
    
    从命令行读取用户输入。
    """
    
    def __init__(
        self,
        sender_id: str = "console_user",
        exit_commands: Optional[list] = None,
    ):
        """初始化。
        
        Args:
            sender_id: 发送者ID
            exit_commands: 退出命令列表
        """
        self.sender_id = sender_id
        self.exit_commands = exit_commands or ["exit", "quit", "bye", "/quit", "/exit"]
    
    @property
    def name(self) -> str:
        return "console"
    
    def get_output_channel(self) -> OutputChannel:
        return ConsoleOutputChannel()
    
    async def run_interactive(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ) -> None:
        """运行交互式控制台。
        
        Args:
            on_new_message: 消息处理回调
        """
        print("=" * 50)
        print("Atguigu AI 对话系统")
        print("输入 'exit' 或 'quit' 退出")
        print("=" * 50)
        print()
        
        while True:
            try:
                # 获取用户输入
                user_input = await self._get_user_input()
                
                if not user_input:
                    continue
                
                # 检查退出命令
                if user_input.lower() in self.exit_commands:
                    print("再见！")
                    break
                
                # 创建用户消息
                user_message = UserMessage(
                    text=user_input,
                    sender_id=self.sender_id,
                    input_channel=self.name,
                )
                
                # 处理消息
                await on_new_message(user_message)
                print()
                
            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"错误: {e}")
    
    async def _get_user_input(self) -> str:
        """获取用户输入（异步）。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: input("You: ").strip())


# 导出
__all__ = [
    "ConsoleChannel",
    "ConsoleOutputChannel",
]
