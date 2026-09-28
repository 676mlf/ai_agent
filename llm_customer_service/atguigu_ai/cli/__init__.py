# -*- coding: utf-8 -*-
"""
atguigu_ai CLI模块

提供命令行接口，使用Click库实现。
支持的主要命令：
- init: 初始化新项目
- train: 训练对话模型
- run: 运行对话服务
- shell: 交互式对话测试
- inspect: 调试页面
- data: 数据处理命令
"""

import click
import sys
import platform
from typing import Optional

from atguigu_ai import __version__
from atguigu_ai.shared.constants import DEFAULT_SERVER_PORT


# 自定义Click组，用于美化帮助信息
class AtguiguCLI(click.Group):
    """自定义CLI组，提供更好的帮助信息格式化。"""
    
    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """格式化帮助信息。"""
        formatter.write_paragraph()
        formatter.write_text(
            "Atguigu AI - 教学版对话系统 CLI\n"
            "LLM驱动的对话系统，用于学习和教学。"
        )
        formatter.write_paragraph()
        super().format_help(ctx, formatter)


def print_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """打印版本信息。"""
    if not value or ctx.resilient_parsing:
        return
    
    click.echo(f"Atguigu AI Version : {__version__}")
    click.echo(f"Python Version     : {platform.python_version()}")
    click.echo(f"Operating System   : {platform.platform()}")
    click.echo(f"Python Path        : {sys.executable}")
    ctx.exit()


@click.group(cls=AtguiguCLI)
@click.option(
    '--version', '-V',
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help='显示版本信息'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    default=False,
    help='启用详细日志输出'
)
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help='启用调试模式'
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool) -> None:
    """Atguigu AI 命令行工具
    
    LLM驱动的教学版对话系统。
    使用 'atguigu <command> --help' 查看各命令的详细帮助。
    """
    # 确保ctx.obj存在
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['debug'] = debug
    
    # 配置日志级别
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    elif verbose:
        import logging
        logging.basicConfig(level=logging.INFO)


def main(args: Optional[list] = None) -> None:
    """CLI主入口函数。
    
    Args:
        args: 命令行参数列表，如果为None则从sys.argv读取
    """
    import os
    from pathlib import Path
    import dotenv
    
    # 自动加载用户工程目录下的 .env 文件
    # 用户可在工程根目录放置 .env 文件配置 API KEY 等环境变量
    cwd = Path.cwd()
    env_file = cwd / ".env"
    if env_file.exists():
        dotenv.load_dotenv(env_file)
    else:
        # 如果当前目录没有，尝试默认搜索
        dotenv.load_dotenv()
    
    # 将当前目录添加到Python路径，以便导入自定义模块
    sys.path.insert(0, str(cwd))
    
    try:
        # 导入并注册子命令
        from atguigu_ai.cli.init import init_command
        from atguigu_ai.cli.train import train_command
        from atguigu_ai.cli.run import run_command
        from atguigu_ai.cli.shell import shell_command
        from atguigu_ai.cli.export import export_command
        from atguigu_ai.cli.inspect import inspect_command
        
        cli.add_command(init_command)
        cli.add_command(train_command)
        cli.add_command(run_command)
        cli.add_command(shell_command)
        cli.add_command(export_command)
        cli.add_command(inspect_command)
        
        # 执行CLI
        cli(args)
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        if '--debug' in (args or sys.argv):
            raise
        sys.exit(1)


# 导出
__all__ = [
    'cli',
    'main',
    'AtguiguCLI',
]
