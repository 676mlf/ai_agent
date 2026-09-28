# -*- coding: utf-8 -*-
"""
运行服务命令

启动对话服务器。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional, List

import click

from atguigu_ai.shared.constants import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
)

logger = logging.getLogger(__name__)


@click.command("run", help="运行对话服务")
@click.option(
    "--model", "-m",
    type=click.Path(exists=True),
    default="models",
    help="模型目录或模型文件路径",
)
@click.option(
    "--endpoints",
    type=click.Path(exists=True),
    default=None,
    help="端点配置文件路径",
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
    "--cors",
    type=str,
    multiple=True,
    default=["*"],
    help="CORS允许的源 (可多次指定)",
)
@click.option(
    "--enable-api/--disable-api",
    default=True,
    help="启用/禁用REST API",
)
@click.option(
    "--enable-inspect/--disable-inspect",
    default=True,
    help="启用/禁用调试页面",
)
@click.option(
    "--channel",
    type=click.Choice(["rest", "socketio", "all"]),
    default="all",
    help="启用的通道类型",
)
@click.pass_context
def run_command(
    ctx: click.Context,
    model: str,
    endpoints: Optional[str],
    host: str,
    port: int,
    cors: tuple,
    enable_api: bool,
    enable_inspect: bool,
    channel: str,
) -> None:
    """运行对话服务。
    
    启动FastAPI服务器，提供对话API和WebSocket接口。
    
    示例:
        atguigu run
        atguigu run --model ./models/latest
        atguigu run --port 8080 --enable-inspect
    """
    verbose = ctx.obj.get("verbose", False)
    debug = ctx.obj.get("debug", False)
    
    click.echo("=" * 50)
    click.echo("Atguigu AI - 对话服务")
    click.echo("=" * 50)
    
    model_path = Path(model)
    
    click.echo(f"模型路径: {model_path.absolute()}")
    click.echo(f"服务地址: http://{host}:{port}")
    click.echo(f"调试页面: {'启用' if enable_inspect else '禁用'}")
    click.echo(f"CORS来源: {list(cors)}")
    click.echo()
    
    try:
        # 导入必要模块
        from atguigu_ai.agent.agent import Agent
        from atguigu_ai.api.server import AtguiguServer
        
        # 加载Agent
        click.echo("加载Agent...")
        
        # 检查是否存在打包的模型或项目目录
        if model_path.is_dir():
            # 项目目录模式
            domain_path = model_path / "domain.yml"
            config_path = model_path / "config.yml"
            
            if domain_path.exists() and config_path.exists():
                # 从项目目录加载
                agent = Agent.load(str(model_path))
            else:
                # 尝试从models目录加载最新模型
                model_files = list(model_path.glob("*.tar.gz"))
                if model_files:
                    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                    agent = Agent.load(str(latest_model))
                else:
                    # 回退到当前目录
                    click.echo("未找到模型文件，尝试从当前目录加载...")
                    agent = Agent.load(".")
        else:
            # 模型文件模式
            agent = Agent.load(str(model_path))
        
        click.echo("Agent加载完成")
        
        # 创建服务器
        click.echo("启动服务器...")
        server = AtguiguServer(
            agent=agent,
            cors_origins=list(cors),
            enable_inspect=enable_inspect,
        )
        
        click.echo()
        click.echo(f"服务已启动: http://{host}:{port}")
        click.echo(f"API文档: http://{host}:{port}/docs")
        if enable_inspect:
            click.echo(f"调试页面: http://{host}:{port}/inspect")
        click.echo("按 Ctrl+C 停止服务")
        click.echo()
        
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
        click.echo(f"运行失败: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)


# 导出
__all__ = ["run_command"]
