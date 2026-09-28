# -*- coding: utf-8 -*-
"""
训练器

负责训练对话系统模型。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from atguigu_ai.core.domain import Domain
from atguigu_ai.dialogue_understanding.flow import FlowsList, FlowLoader
from atguigu_ai.shared.yaml_loader import read_yaml_file

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """训练器配置。
    
    Attributes:
        domain_path: Domain文件路径
        config_path: 配置文件路径
        data_path: 训练数据路径
        output_path: 输出模型路径
        force_training: 是否强制训练（忽略缓存）
    """
    domain_path: str = "domain.yml"
    config_path: str = "config.yml"
    data_path: str = "data"
    output_path: str = "models"
    force_training: bool = False


@dataclass
class TrainingResult:
    """训练结果。
    
    Attributes:
        model_path: 训练后的模型路径
        domain: 加载的Domain
        flows: 加载的Flows
        training_time: 训练耗时（秒）
        success: 是否成功
        errors: 错误信息列表
    """
    model_path: str = ""
    domain: Optional[Domain] = None
    flows: Optional[FlowsList] = None
    training_time: float = 0.0
    success: bool = True
    errors: List[str] = field(default_factory=list)


class Trainer:
    """训练器。
    
    负责训练对话系统模型。由于本架构主要依赖LLM，
    "训练"主要是验证和打包配置文件。
    
    训练流程：
    1. 加载并验证Domain
    2. 加载并验证Flows
    3. 加载并验证配置
    4. 打包模型文件
    """
    
    def __init__(self, config: Optional[TrainerConfig] = None):
        """初始化训练器。
        
        Args:
            config: 训练器配置
        """
        self.config = config or TrainerConfig()
    
    def train(
        self,
        project_path: Optional[Union[str, Path]] = None,
    ) -> TrainingResult:
        """执行训练。
        
        Args:
            project_path: 项目路径（如果为None则使用当前目录）
            
        Returns:
            训练结果
        """
        import time
        start_time = time.time()
        
        result = TrainingResult()
        
        # 确定项目路径
        if project_path:
            project_path = Path(project_path)
        else:
            project_path = Path.cwd()
        
        logger.info(f"Training model from {project_path}")
        
        try:
            # 1. 加载Domain
            domain_path = project_path / self.config.domain_path
            if domain_path.exists():
                result.domain = Domain.load(str(domain_path))
                logger.info(f"Loaded domain from {domain_path}")
            else:
                result.errors.append(f"Domain file not found: {domain_path}")
            
            # 2. 加载Flows
            data_path = project_path / self.config.data_path
            if data_path.exists():
                loader = FlowLoader()
                try:
                    result.flows = loader.load(data_path)
                    logger.info(f"Loaded {len(result.flows)} flows")
                except Exception as e:
                    result.errors.append(f"Failed to load flows: {e}")
            
            # 3. 加载配置
            config_path = project_path / self.config.config_path
            if config_path.exists():
                config_data = read_yaml_file(str(config_path))
                logger.info(f"Loaded config from {config_path}")
            else:
                result.errors.append(f"Config file not found: {config_path}")
            
            # 4. 验证
            validation_errors = self._validate(result.domain, result.flows)
            result.errors.extend(validation_errors)
            
            # 5. 打包模型
            if not result.errors or self.config.force_training:
                model_path = self._package_model(
                    project_path,
                    result.domain,
                    result.flows,
                )
                result.model_path = model_path
                result.success = True
                logger.info(f"Model saved to {model_path}")
            else:
                result.success = False
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            result.errors.append(str(e))
            result.success = False
        
        result.training_time = time.time() - start_time
        return result
    
    def _validate(
        self,
        domain: Optional[Domain],
        flows: Optional[FlowsList],
    ) -> List[str]:
        """验证Domain和Flows。
        
        Args:
            domain: Domain定义
            flows: Flow列表
            
        Returns:
            错误信息列表
        """
        errors = []
        
        if domain is None:
            errors.append("Domain is required")
            return errors
        
        # 验证响应模板存在
        if not domain.responses:
            errors.append("No responses defined in domain")
        
        # 验证Flows中引用的动作存在
        if flows:
            for flow in flows:
                for step in flow.steps:
                    if step.action:
                        # 检查非utter_动作是否在domain中定义
                        if step.action.startswith("action_"):
                            if not domain.has_action(step.action):
                                errors.append(
                                    f"Action '{step.action}' in flow '{flow.id}' "
                                    f"not found in domain"
                                )
        
        return errors
    
    def _package_model(
        self,
        project_path: Path,
        domain: Optional[Domain],
        flows: Optional[FlowsList],
    ) -> str:
        """打包模型文件为 .tar.gz 压缩包。
        
        Args:
            project_path: 项目路径
            domain: Domain定义
            flows: Flow列表
            
        Returns:
            模型压缩包路径
        """
        from atguigu_ai.training.model_storage import (
            create_model_package,
            ModelMetadata,
        )
        
        # 创建输出目录
        output_dir = project_path / self.config.output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成模型名称
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = f"model-{timestamp}"
        
        # 创建元数据
        metadata = ModelMetadata(
            name=model_name,
            created_at=timestamp,
            flows_count=len(flows) if flows else 0,
            actions_count=len(domain.actions) if domain else 0,
            responses_count=len(domain.responses) if domain else 0,
        )
        
        # 使用 model_storage 创建压缩包
        model_path = create_model_package(
            project_path=project_path,
            output_dir=output_dir,
            model_name=model_name,
            domain_path=self.config.domain_path,
            config_path=self.config.config_path,
            data_path=self.config.data_path,
            metadata=metadata,
        )
        
        return model_path


# 便捷函数

def train(
    project_path: Optional[Union[str, Path]] = None,
    domain: Optional[str] = None,
    config: Optional[str] = None,
    output: Optional[str] = None,
) -> TrainingResult:
    """便捷训练函数。
    
    Args:
        project_path: 项目路径
        domain: Domain路径
        config: 配置文件路径
        output: 输出路径
        
    Returns:
        训练结果
    """
    trainer_config = TrainerConfig()
    
    if domain:
        trainer_config.domain_path = domain
    if config:
        trainer_config.config_path = config
    if output:
        trainer_config.output_path = output
    
    trainer = Trainer(config=trainer_config)
    return trainer.train(project_path)


# 导出
__all__ = [
    "Trainer",
    "TrainerConfig",
    "TrainingResult",
    "train",
]
