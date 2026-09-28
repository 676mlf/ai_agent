# -*- coding: utf-8 -*-
"""
训练命令

提供模型训练和验证功能。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)


@click.command("train", help="训练对话模型")
@click.option(
    "--data", "-d",
    type=click.Path(exists=True),
    default="data",
    help="训练数据目录",
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    default="config.yml",
    help="配置文件路径",
)
@click.option(
    "--domain",
    type=click.Path(exists=True),
    default=None,
    help="Domain文件或目录路径（默认自动检测 domain/ 或 domain.yml）",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="models",
    help="模型输出目录",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="仅验证，不生成模型",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="强制重新训练",
)
@click.pass_context
def train_command(
    ctx: click.Context,
    data: str,
    config: str,
    domain: Optional[str],
    output: str,
    dry_run: bool,
    force: bool,
) -> None:
    """训练对话模型。
    
    验证配置和数据，生成可部署的模型包。
    
    示例:
        atguigu train
        atguigu train --data ./my_data --output ./my_models
        atguigu train --dry-run  # 仅验证
    """
    verbose = ctx.obj.get("verbose", False)
    debug = ctx.obj.get("debug", False)
    
    click.echo("=" * 50)
    click.echo("Atguigu AI - 模型训练")
    click.echo("=" * 50)
    
    # 自动检测 domain 路径
    if domain is None:
        # 优先检测 domain/ 目录，其次是 domain.yml 文件
        if Path("domain").is_dir():
            domain = "domain"
        elif Path("domain.yml").exists():
            domain = "domain.yml"
        else:
            click.echo("错误: 未找到 domain/ 目录或 domain.yml 文件", err=True)
            raise SystemExit(1)
    
    # 验证路径
    data_path = Path(data)
    config_path = Path(config)
    domain_path = Path(domain)
    output_path = Path(output)
    
    if not data_path.exists():
        click.echo(f"错误: 数据目录不存在: {data_path}", err=True)
        raise SystemExit(1)
    
    if not config_path.exists():
        click.echo(f"错误: 配置文件不存在: {config_path}", err=True)
        raise SystemExit(1)
    
    if not domain_path.exists():
        click.echo(f"错误: Domain文件不存在: {domain_path}", err=True)
        raise SystemExit(1)
    
    click.echo(f"数据目录: {data_path.absolute()}")
    click.echo(f"配置文件: {config_path.absolute()}")
    click.echo(f"Domain文件: {domain_path.absolute()}")
    click.echo(f"输出目录: {output_path.absolute()}")
    click.echo()
    
    try:
        # 导入训练模块
        from atguigu_ai.training.trainer import Trainer, TrainerConfig
        from atguigu_ai.shared.config import AtguiguConfig
        
        # 加载配置（用于验证）
        click.echo("加载配置...")
        atguigu_config = AtguiguConfig.load(str(config_path))
        click.echo("  配置加载成功")
        
        # 创建训练器配置
        trainer_config = TrainerConfig(
            domain_path=domain,
            config_path=config,
            data_path=data,
            output_path=output,
            force_training=force,
        )
        
        # 创建训练器
        trainer = Trainer(config=trainer_config)
        
        if dry_run:
            # 仅验证模式
            click.echo("验证Domain和Flows...")
            from atguigu_ai.core.domain import Domain
            from atguigu_ai.dialogue_understanding.flow import FlowLoader
            
            # 加载Domain
            domain_obj = Domain.load(str(domain_path))
            click.echo(f"  Domain加载成功: {len(domain_obj.responses)} 个响应模板")
            
            # 加载Flows
            loader = FlowLoader()
            flows = loader.load(data_path)
            click.echo(f"  Flows加载成功: {len(flows)} 个flow")
            
            click.echo()
            click.echo("验证完成 (dry-run模式，未生成模型)")
        else:
            # 执行训练
            click.echo("执行训练...")
            result = trainer.train()
            
            if result.success:
                click.echo()
                click.echo("=" * 50)
                click.echo("训练完成!")
                click.echo(f"模型路径: {result.model_path}")
                if result.flows:
                    click.echo(f"Flow数量: {len(result.flows)}")
                click.echo(f"训练耗时: {result.training_time:.2f}秒")
                click.echo("=" * 50)
            else:
                click.echo("训练失败:", err=True)
                for error in result.errors:
                    click.echo(f"  - {error}", err=True)
                raise SystemExit(1)
        
    except ImportError as e:
        click.echo(f"导入错误: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"训练失败: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)


# 导出
__all__ = ["train_command"]
