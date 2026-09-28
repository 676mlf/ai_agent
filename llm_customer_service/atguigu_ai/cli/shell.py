# -*- coding: utf-8 -*-
"""
交互式Shell命令

提供命令行交互式对话测试。
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)


class InteractiveShell:
    """交互式对话Shell。"""
    
    def __init__(self, agent, sender_id: str = "shell_user"):
        """初始化Shell。
        
        Args:
            agent: Agent实例
            sender_id: 发送者ID
        """
        self.agent = agent
        self.sender_id = sender_id
        self.running = True
    
    async def run(self) -> None:
        """运行交互式会话。"""
        self._print_welcome()
        
        while self.running:
            try:
                # 获取用户输入
                user_input = self._get_input()
                
                if user_input is None:
                    # EOF (Ctrl+D)
                    click.echo("\n再见!")
                    break
                
                user_input = user_input.strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue
                
                # 处理对话消息
                await self._handle_message(user_input)
                
            except KeyboardInterrupt:
                click.echo("\n使用 /quit 或 Ctrl+D 退出")
            except Exception as e:
                click.echo(f"错误: {e}", err=True)
    
    def _print_welcome(self) -> None:
        """打印欢迎信息。"""
        click.echo()
        click.echo("=" * 50)
        click.echo("Atguigu AI - 交互式对话")
        click.echo("=" * 50)
        click.echo()
        click.echo("输入消息开始对话，或使用以下命令:")
        click.echo("  /help    - 显示帮助信息")
        click.echo("  /reset   - 重置对话")
        click.echo("  /slots   - 显示当前槽位值")
        click.echo("  /quit    - 退出")
        click.echo()
    
    def _get_input(self) -> Optional[str]:
        """获取用户输入。"""
        try:
            return input("You: ")
        except EOFError:
            return None
    
    def _handle_command(self, command: str) -> None:
        """处理Shell命令。"""
        cmd = command.lower().split()[0]
        
        if cmd in ["/quit", "/exit", "/q"]:
            self.running = False
            click.echo("再见!")
        
        elif cmd in ["/help", "/h", "/?"]:
            self._print_help()
        
        elif cmd == "/reset":
            asyncio.get_event_loop().run_until_complete(
                self._reset_conversation()
            )
        
        elif cmd == "/slots":
            asyncio.get_event_loop().run_until_complete(
                self._show_slots()
            )
        
        elif cmd == "/history":
            asyncio.get_event_loop().run_until_complete(
                self._show_history()
            )
        
        elif cmd == "/debug":
            self._toggle_debug()
        
        else:
            click.echo(f"未知命令: {cmd}")
            click.echo("使用 /help 查看可用命令")
    
    def _print_help(self) -> None:
        """打印帮助信息。"""
        click.echo()
        click.echo("可用命令:")
        click.echo("  /help, /h, /?  - 显示帮助信息")
        click.echo("  /reset         - 重置对话，清除会话状态")
        click.echo("  /slots         - 显示当前所有槽位值")
        click.echo("  /history       - 显示对话历史")
        click.echo("  /debug         - 切换调试模式")
        click.echo("  /quit, /exit   - 退出程序")
        click.echo()
    
    async def _reset_conversation(self) -> None:
        """重置对话。"""
        try:
            await self.agent.reset_tracker(self.sender_id)
            click.echo("对话已重置")
        except Exception as e:
            click.echo(f"重置失败: {e}", err=True)
    
    async def _show_slots(self) -> None:
        """显示槽位值。"""
        try:
            tracker = await self.agent.get_tracker(self.sender_id)
            if tracker:
                slots = tracker.current_slot_values()
                click.echo()
                click.echo("当前槽位值:")
                for name, value in slots.items():
                    if value is not None:
                        click.echo(f"  {name}: {value}")
                    else:
                        click.echo(f"  {name}: (未设置)")
                click.echo()
            else:
                click.echo("会话不存在")
        except Exception as e:
            click.echo(f"获取槽位失败: {e}", err=True)
    
    async def _show_history(self) -> None:
        """显示对话历史。"""
        try:
            tracker = await self.agent.get_tracker(self.sender_id)
            if tracker:
                click.echo()
                click.echo("对话历史:")
                for i, msg in enumerate(tracker.messages):
                    role = "You" if msg.get("role") == "user" else "Bot"
                    text = msg.get("text", msg.get("content", ""))
                    click.echo(f"  [{i+1}] {role}: {text[:100]}...")
                click.echo()
            else:
                click.echo("会话不存在")
        except Exception as e:
            click.echo(f"获取历史失败: {e}", err=True)
    
    def _toggle_debug(self) -> None:
        """切换调试模式。"""
        current_level = logging.getLogger().level
        if current_level == logging.DEBUG:
            logging.getLogger().setLevel(logging.INFO)
            click.echo("调试模式: 关闭")
        else:
            logging.getLogger().setLevel(logging.DEBUG)
            click.echo("调试模式: 开启")
    
    async def _handle_message(self, text: str) -> None:
        """处理用户消息。"""
        try:
            # 发送消息到Agent
            responses = await self.agent.handle_message(
                message=text,
                sender_id=self.sender_id,
            )
            
            # 显示响应
            for response in responses:
                if isinstance(response, dict):
                    text = response.get("text", "")
                    if text:
                        click.echo(f"Bot: {text}")
                    
                    # 显示按钮
                    buttons = response.get("buttons", [])
                    if buttons:
                        click.echo("  选项:")
                        for i, btn in enumerate(buttons, 1):
                            title = btn.get("title", "")
                            click.echo(f"    [{i}] {title}")
                    
                    # 显示自定义数据
                    custom = response.get("custom")
                    if custom:
                        click.echo(f"  (自定义: {custom})")
                else:
                    click.echo(f"Bot: {response}")
            
            if not responses:
                click.echo("Bot: (无响应)")
                
        except Exception as e:
            click.echo(f"处理消息失败: {e}", err=True)
            logger.exception("Message handling error")


@click.command("shell", help="交互式对话测试")
@click.option(
    "--model", "-m",
    type=click.Path(exists=True),
    default=".",
    help="模型或项目目录路径",
)
@click.option(
    "--sender-id",
    type=str,
    default="shell_user",
    help="发送者ID",
)
@click.pass_context
def shell_command(
    ctx: click.Context,
    model: str,
    sender_id: str,
) -> None:
    """启动交互式对话Shell。
    
    提供命令行交互式对话测试环境。
    
    示例:
        atguigu shell
        atguigu shell --model ./my_project
        atguigu shell --sender-id test_user
    """
    verbose = ctx.obj.get("verbose", False)
    debug = ctx.obj.get("debug", False)
    
    model_path = Path(model)
    
    try:
        # 导入Agent
        from atguigu_ai.agent.agent import Agent
        
        # 加载Agent
        click.echo(f"加载模型: {model_path.absolute()}")
        agent = Agent.load(str(model_path))
        click.echo("模型加载完成")
        
        # 创建Shell
        shell = InteractiveShell(agent, sender_id)
        
        # 运行Shell
        asyncio.run(shell.run())
        
    except ImportError as e:
        click.echo(f"导入错误: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"启动Shell失败: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)


# 导出
__all__ = ["shell_command", "InteractiveShell"]
