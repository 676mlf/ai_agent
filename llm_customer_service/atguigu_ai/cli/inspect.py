# -*- coding: utf-8 -*-
"""
Inspect命令

启动调试页面服务。
"""

from __future__ import annotations

import logging
import webbrowser
from pathlib import Path
from typing import Optional

import click

from atguigu_ai.shared.constants import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
)

logger = logging.getLogger(__name__)


@click.command("inspect", help="启动调试页面")
@click.option(
    "--model", "-m",
    type=click.Path(exists=True),
    default=".",
    help="模型或项目目录路径",
)
@click.option(
    "--host", "-H",
    type=str,
    default=DEFAULT_SERVER_HOST,
    help="服务器监听地址",
)
@click.option(
    "--port", "-p",
    type=int,
    default=DEFAULT_SERVER_PORT,
    help="服务器监听端口",
)
@click.option(
    "--no-browser",
    is_flag=True,
    default=False,
    help="不自动打开浏览器",
)
@click.option(
    "--cors",
    type=str,
    multiple=True,
    default=["*"],
    help="CORS允许的源",
)
@click.pass_context
def inspect_command(
    ctx: click.Context,
    model: str,
    host: str,
    port: int,
    no_browser: bool,
    cors: tuple,
) -> None:
    """启动Inspect调试页面。
    
    提供可视化的对话调试界面，包括：
    - 实时对话窗口
    - Tracker状态查看
    - 命令和事件日志
    
    示例:
        atguigu inspect
        atguigu inspect --model ./my_bot --port 5005
        atguigu inspect --no-browser
    """
    verbose = ctx.obj.get("verbose", False)
    debug = ctx.obj.get("debug", False)
    
    model_path = Path(model)
    
    click.echo("=" * 50)
    click.echo("Atguigu AI - Inspect 调试页面")
    click.echo("=" * 50)
    
    click.echo(f"模型路径: {model_path.absolute()}")
    click.echo(f"服务地址: http://{host}:{port}")
    click.echo()
    
    try:
        # 导入必要模块
        from atguigu_ai.agent.agent import Agent
        from atguigu_ai.api.server import AtguiguServer
        
        # 加载Agent
        click.echo("加载Agent...")
        agent = Agent.load(str(model_path))
        click.echo("Agent加载完成")
        
        # 创建服务器
        server = AtguiguServer(
            agent=agent,
            cors_origins=list(cors),
            enable_inspect=True,
        )
        
        inspect_url = f"http://{host}:{port}/inspect"
        # 浏览器 URL 使用 localhost（0.0.0.0 在浏览器中无法访问）
        browser_host = "localhost" if host == "0.0.0.0" else host
        browser_url = f"http://{browser_host}:{port}/inspect"
        
        click.echo()
        click.echo(f"Inspect页面: {inspect_url}")
        click.echo(f"API文档: http://{host}:{port}/docs")
        click.echo("按 Ctrl+C 停止服务")
        click.echo()
        
        # 自动打开浏览器
        if not no_browser:
            import threading
            
            def open_browser():
                import time
                time.sleep(1.5)  # 等待服务器启动
                webbrowser.open(browser_url)
            
            threading.Thread(target=open_browser, daemon=True).start()
        
        # 运行服务器
        server.run(host=host, port=port)
        
    except KeyboardInterrupt:
        click.echo("\n服务已停止")
    except ImportError as e:
        click.echo(f"导入错误: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"启动失败: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)


# 导出
__all__ = ["inspect_command"]
