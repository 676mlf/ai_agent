# -*- coding: utf-8 -*-
"""
导出命令

导出模型、配置或数据。
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import click

logger = logging.getLogger(__name__)


@click.command("export", help="导出模型或配置")
@click.option(
    "--model", "-m",
    type=click.Path(exists=True),
    default=".",
    help="模型或项目目录路径",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    required=True,
    help="输出路径",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["tar.gz", "zip", "dir", "json"]),
    default="tar.gz",
    help="输出格式",
)
@click.option(
    "--include-data/--no-data",
    default=True,
    help="包含训练数据",
)
@click.option(
    "--include-config/--no-config",
    default=True,
    help="包含配置文件",
)
@click.pass_context
def export_command(
    ctx: click.Context,
    model: str,
    output: str,
    format: str,
    include_data: bool,
    include_config: bool,
) -> None:
    """导出模型或配置。
    
    将模型、配置和数据打包导出。
    
    示例:
        atguigu export --output ./export/model.tar.gz
        atguigu export --format zip --output ./export/model.zip
        atguigu export --format json --output ./export/domain.json
    """
    verbose = ctx.obj.get("verbose", False)
    debug = ctx.obj.get("debug", False)
    
    model_path = Path(model)
    output_path = Path(output)
    
    click.echo("=" * 50)
    click.echo("Atguigu AI - 导出")
    click.echo("=" * 50)
    
    click.echo(f"源路径: {model_path.absolute()}")
    click.echo(f"输出路径: {output_path.absolute()}")
    click.echo(f"格式: {format}")
    click.echo()
    
    try:
        if format == "json":
            # 导出domain为JSON
            _export_json(model_path, output_path)
        elif format == "dir":
            # 导出为目录
            _export_directory(
                model_path,
                output_path,
                include_data,
                include_config,
            )
        else:
            # 导出为压缩包
            _export_archive(
                model_path,
                output_path,
                format,
                include_data,
                include_config,
            )
        
        click.echo()
        click.echo("=" * 50)
        click.echo(f"导出完成: {output_path}")
        click.echo("=" * 50)
        
    except Exception as e:
        click.echo(f"导出失败: {e}", err=True)
        if debug:
            raise
        raise SystemExit(1)


def _export_json(source: Path, output: Path) -> None:
    """导出为JSON格式。"""
    from atguigu_ai.core.domain import Domain
    
    domain_file = source / "domain.yml"
    if not domain_file.exists():
        domain_file = source / "domain.yaml"
    
    if not domain_file.exists():
        raise FileNotFoundError(f"未找到domain文件: {source}")
    
    click.echo("加载domain...")
    domain = Domain.from_yaml(str(domain_file))
    
    click.echo("导出JSON...")
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "w", encoding="utf-8") as f:
        json.dump(domain.as_dict(), f, ensure_ascii=False, indent=2)


def _export_directory(
    source: Path,
    output: Path,
    include_data: bool,
    include_config: bool,
) -> None:
    """导出为目录。"""
    click.echo("创建导出目录...")
    output.mkdir(parents=True, exist_ok=True)
    
    # 复制domain
    for domain_file in ["domain.yml", "domain.yaml"]:
        src = source / domain_file
        if src.exists():
            click.echo(f"  复制 {domain_file}")
            shutil.copy2(src, output / domain_file)
            break
    
    # 复制配置
    if include_config:
        for config_file in ["config.yml", "config.yaml"]:
            src = source / config_file
            if src.exists():
                click.echo(f"  复制 {config_file}")
                shutil.copy2(src, output / config_file)
                break
        
        # 复制endpoints
        for endpoints_file in ["endpoints.yml", "endpoints.yaml"]:
            src = source / endpoints_file
            if src.exists():
                click.echo(f"  复制 {endpoints_file}")
                shutil.copy2(src, output / endpoints_file)
                break
    
    # 复制数据
    if include_data:
        data_dir = source / "data"
        if data_dir.exists():
            click.echo("  复制 data/")
            shutil.copytree(data_dir, output / "data", dirs_exist_ok=True)
    
    # 复制actions
    actions_dir = source / "actions"
    if actions_dir.exists():
        click.echo("  复制 actions/")
        shutil.copytree(actions_dir, output / "actions", dirs_exist_ok=True)
    
    # 写入元数据
    metadata = {
        "exported_at": datetime.now().isoformat(),
        "source": str(source.absolute()),
        "version": "0.1.0",
    }
    
    with open(output / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _export_archive(
    source: Path,
    output: Path,
    format: str,
    include_data: bool,
    include_config: bool,
) -> None:
    """导出为压缩包。"""
    import tempfile
    
    # 先导出到临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "export"
        _export_directory(source, temp_path, include_data, include_config)
        
        # 创建压缩包
        output.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "tar.gz":
            click.echo("创建tar.gz压缩包...")
            archive_name = str(output).replace(".tar.gz", "")
            shutil.make_archive(archive_name, "gztar", temp_path)
            # 重命名如果需要
            created = Path(f"{archive_name}.tar.gz")
            if created != output:
                shutil.move(str(created), str(output))
        
        elif format == "zip":
            click.echo("创建zip压缩包...")
            archive_name = str(output).replace(".zip", "")
            shutil.make_archive(archive_name, "zip", temp_path)
            # 重命名如果需要
            created = Path(f"{archive_name}.zip")
            if created != output:
                shutil.move(str(created), str(output))


# 导出
__all__ = ["export_command"]
