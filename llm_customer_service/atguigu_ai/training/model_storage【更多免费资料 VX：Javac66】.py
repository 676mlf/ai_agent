# -*- coding: utf-8 -*-
"""
模型存储

提供模型打包和解压功能，将训练产物打包为 .tar.gz 压缩包。

模型包结构:
    model-YYYYMMDD-HHMMSS.tar.gz
    ├── metadata.json    # 模型元数据
    ├── config.yml       # 配置文件
    ├── domain.yml       # Domain定义
    └── data/            # 训练数据
        └── flows.yml    # Flow定义
"""

from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# 模型包内的路径
MODEL_ARCHIVE_METADATA_FILE = "metadata.json"
MODEL_ARCHIVE_CONFIG_FILE = "config.yml"
MODEL_ARCHIVE_DOMAIN_FILE = "domain.yml"
MODEL_ARCHIVE_DATA_DIR = "data"


@dataclass
class ModelMetadata:
    """模型元数据。
    
    存储模型的基本信息，用于模型管理和版本追踪。
    """
    name: str = ""
    created_at: str = ""
    version: str = "1.0"
    flows_count: int = 0
    actions_count: int = 0
    responses_count: int = 0
    assistant_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def as_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "name": self.name,
            "created_at": self.created_at,
            "version": self.version,
            "flows_count": self.flows_count,
            "actions_count": self.actions_count,
            "responses_count": self.responses_count,
            "assistant_id": self.assistant_id,
            "extra": self.extra,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """从字典反序列化。"""
        return cls(
            name=data.get("name", ""),
            created_at=data.get("created_at", ""),
            version=data.get("version", "1.0"),
            flows_count=data.get("flows_count", 0),
            actions_count=data.get("actions_count", 0),
            responses_count=data.get("responses_count", 0),
            assistant_id=data.get("assistant_id"),
            extra=data.get("extra", {}),
        )


def create_model_package(
    project_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: Optional[str] = None,
    domain_path: str = "domain.yml",
    config_path: str = "config.yml",
    data_path: str = "data",
    metadata: Optional[ModelMetadata] = None,
) -> str:
    """创建模型压缩包。
    
    将项目文件打包为 .tar.gz 格式的模型包。
    
    Args:
        project_path: 项目目录路径
        output_dir: 输出目录
        model_name: 模型名称（不含扩展名），默认自动生成
        domain_path: Domain文件相对路径
        config_path: 配置文件相对路径
        data_path: 数据目录相对路径
        metadata: 模型元数据，如果为None则自动生成
        
    Returns:
        模型压缩包的完整路径
    """
    project_path = Path(project_path)
    output_dir = Path(output_dir)
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成模型名称
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if model_name is None:
        model_name = f"model-{timestamp}"
    
    # 确保文件名以 .tar.gz 结尾
    if not model_name.endswith(".tar.gz"):
        model_name = f"{model_name}.tar.gz"
    
    model_archive_path = output_dir / model_name
    
    logger.info(f"Creating model package: {model_archive_path}")
    
    # 使用临时目录组织文件
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 复制配置文件
        src_config = project_path / config_path
        if src_config.exists():
            shutil.copy(src_config, temp_path / MODEL_ARCHIVE_CONFIG_FILE)
            logger.debug(f"Added config: {config_path}")
        
        # 复制Domain文件
        src_domain = project_path / domain_path
        if src_domain.exists():
            if src_domain.is_dir():
                # 如果是目录，复制整个目录
                shutil.copytree(src_domain, temp_path / "domain")
            else:
                shutil.copy(src_domain, temp_path / MODEL_ARCHIVE_DOMAIN_FILE)
            logger.debug(f"Added domain: {domain_path}")
        
        # 复制数据目录
        src_data = project_path / data_path
        if src_data.exists():
            shutil.copytree(src_data, temp_path / MODEL_ARCHIVE_DATA_DIR)
            logger.debug(f"Added data: {data_path}")
        
        # 复制 actions 目录（如果存在）
        src_actions = project_path / "actions"
        if src_actions.exists():
            shutil.copytree(src_actions, temp_path / "actions")
            logger.debug("Added actions directory")
        
        # 复制 endpoints.yml（如果存在）
        src_endpoints = project_path / "endpoints.yml"
        if src_endpoints.exists():
            shutil.copy(src_endpoints, temp_path / "endpoints.yml")
            logger.debug("Added endpoints.yml")
        
        # 复制 credentials.yml（如果存在）
        src_credentials = project_path / "credentials.yml"
        if src_credentials.exists():
            shutil.copy(src_credentials, temp_path / "credentials.yml")
            logger.debug("Added credentials.yml")
        
        # 创建/保存元数据
        if metadata is None:
            metadata = ModelMetadata(
                name=model_name.replace(".tar.gz", ""),
                created_at=timestamp,
            )
        
        metadata_path = temp_path / MODEL_ARCHIVE_METADATA_FILE
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.as_dict(), f, ensure_ascii=False, indent=2)
        
        # 创建 tar.gz 压缩包
        with tarfile.open(model_archive_path, "w:gz") as tar:
            # 添加临时目录中的所有文件
            for item in temp_path.iterdir():
                tar.add(item, arcname=item.name)
    
    logger.info(f"Model package created: {model_archive_path}")
    return str(model_archive_path)


def extract_model_archive(
    model_archive_path: Union[str, Path],
    target_dir: Union[str, Path],
) -> ModelMetadata:
    """解压模型压缩包。
    
    将 .tar.gz 格式的模型包解压到目标目录。
    
    Args:
        model_archive_path: 模型压缩包路径
        target_dir: 目标解压目录
        
    Returns:
        模型元数据
        
    Raises:
        FileNotFoundError: 模型文件不存在
        ValueError: 模型格式无效
    """
    model_archive_path = Path(model_archive_path)
    target_dir = Path(target_dir)
    
    if not model_archive_path.exists():
        raise FileNotFoundError(f"Model archive not found: {model_archive_path}")
    
    if not model_archive_path.suffix == ".gz" or not model_archive_path.name.endswith(".tar.gz"):
        raise ValueError(f"Invalid model format, expected .tar.gz: {model_archive_path}")
    
    logger.info(f"Extracting model archive: {model_archive_path}")
    
    # 确保目标目录存在
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 解压
    with tarfile.open(model_archive_path, "r:gz") as tar:
        # 安全解压：过滤危险路径
        members = _get_safe_members(tar)
        tar.extractall(target_dir, members=members)
    
    # 加载元数据
    metadata_path = target_dir / MODEL_ARCHIVE_METADATA_FILE
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
        metadata = ModelMetadata.from_dict(metadata_dict)
    else:
        # 如果没有元数据文件，创建默认的
        metadata = ModelMetadata(
            name=model_archive_path.stem.replace(".tar", ""),
        )
    
    logger.info(f"Model extracted to: {target_dir}")
    return metadata


def _get_safe_members(tar: tarfile.TarFile) -> List[tarfile.TarInfo]:
    """获取安全的 tar 成员列表。
    
    过滤掉危险的路径（如绝对路径、路径穿越等）。
    """
    safe_members = []
    for member in tar.getmembers():
        # 跳过绝对路径
        if Path(member.name).is_absolute():
            logger.warning(f"Skipping absolute path: {member.name}")
            continue
        
        # 跳过路径穿越
        if ".." in member.name:
            logger.warning(f"Skipping path traversal: {member.name}")
            continue
        
        # 跳过设备文件和符号链接
        if member.isdev() or member.issym():
            logger.warning(f"Skipping special file: {member.name}")
            continue
        
        safe_members.append(member)
    
    return safe_members


def get_latest_model(models_dir: Union[str, Path]) -> Optional[str]:
    """获取最新的模型压缩包。
    
    从模型目录中找到最新的 .tar.gz 文件。
    优先按文件名中的时间戳排序（格式: model-YYYYMMDD-HHMMSS.tar.gz）。
    如果文件名不包含时间戳，则按文件修改时间排序。
    
    Args:
        models_dir: 模型目录路径
        
    Returns:
        最新模型的路径，如果没有找到返回 None
    """
    import re
    
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        return None
    
    # 查找所有 .tar.gz 文件
    model_files = list(models_dir.glob("*.tar.gz"))
    
    if not model_files:
        return None
    
    def extract_timestamp(filepath: Path) -> tuple:
        """从文件名提取时间戳，返回 (是否有时间戳, 时间戳值或修改时间)"""
        filename = filepath.name
        # 匹配格式: model-YYYYMMDD-HHMMSS.tar.gz 或类似格式
        # 支持: model-20250128-143000.tar.gz, ecs_demo-20250128-143000.tar.gz 等
        match = re.search(r'(\d{8})-(\d{6})\.tar\.gz$', filename)
        if match:
            date_str = match.group(1)  # YYYYMMDD
            time_str = match.group(2)  # HHMMSS
            try:
                timestamp = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
                return (True, timestamp)
            except ValueError:
                pass
        # 没有有效时间戳，使用文件修改时间
        return (False, datetime.fromtimestamp(filepath.stat().st_mtime))
    
    # 按时间戳排序：优先有时间戳的文件，然后按时间戳从新到旧
    # 返回最新的（最大的时间戳）
    latest = max(model_files, key=lambda p: extract_timestamp(p))
    logger.debug(f"选择最新模型: {latest}")
    return str(latest)


def get_model_path(
    model_path: Union[str, Path],
    models_dir: str = "models",
) -> Optional[str]:
    """获取模型路径。
    
    支持以下输入：
    - 直接指定 .tar.gz 文件路径
    - 指定包含 .tar.gz 文件的目录（返回最新的）
    - 指定项目目录（从 models/ 子目录查找）
    
    Args:
        model_path: 模型路径或目录
        models_dir: 模型子目录名称
        
    Returns:
        模型压缩包路径，如果找不到返回 None
    """
    model_path = Path(model_path)
    
    # 情况1: 直接是 .tar.gz 文件
    if model_path.is_file() and model_path.name.endswith(".tar.gz"):
        return str(model_path)
    
    # 情况2: 是目录
    if model_path.is_dir():
        # 先尝试在当前目录查找 .tar.gz
        latest = get_latest_model(model_path)
        if latest:
            return latest
        
        # 再尝试在 models/ 子目录查找
        models_subdir = model_path / models_dir
        if models_subdir.exists():
            return get_latest_model(models_subdir)
    
    return None


def load_metadata_from_archive(
    model_archive_path: Union[str, Path],
) -> ModelMetadata:
    """从压缩包加载元数据（不解压整个包）。
    
    Args:
        model_archive_path: 模型压缩包路径
        
    Returns:
        模型元数据
    """
    model_archive_path = Path(model_archive_path)
    
    with tarfile.open(model_archive_path, "r:gz") as tar:
        # 查找 metadata.json
        try:
            member = tar.getmember(MODEL_ARCHIVE_METADATA_FILE)
            f = tar.extractfile(member)
            if f:
                metadata_dict = json.load(f)
                return ModelMetadata.from_dict(metadata_dict)
        except KeyError:
            pass
    
    # 如果没有元数据，返回默认的
    return ModelMetadata(
        name=model_archive_path.stem.replace(".tar", ""),
    )


# 导出
__all__ = [
    "ModelMetadata",
    "create_model_package",
    "extract_model_archive",
    "get_latest_model",
    "get_model_path",
    "load_metadata_from_archive",
    "MODEL_ARCHIVE_METADATA_FILE",
    "MODEL_ARCHIVE_CONFIG_FILE",
    "MODEL_ARCHIVE_DOMAIN_FILE",
    "MODEL_ARCHIVE_DATA_DIR",
]
