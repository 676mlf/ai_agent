# -*- coding: utf-8 -*-
"""
Flow加载器

从YAML文件加载Flow定义。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from atguigu_ai.dialogue_understanding.flow.flow import Flow, FlowsList
from atguigu_ai.shared.yaml_loader import read_yaml_file, read_yaml_files
from atguigu_ai.shared.exceptions import ConfigurationException

logger = logging.getLogger(__name__)


class FlowLoader:
    """Flow加载器。
    
    从YAML文件加载Flow定义。
    
    支持的文件格式：
    - 单个flows.yml文件
    - flows目录下的多个YAML文件
    """
    
    def __init__(self):
        """初始化加载器。"""
        pass
    
    def load(self, path: Union[str, Path]) -> FlowsList:
        """加载Flow。
        
        Args:
            path: 文件或目录路径
            
        Returns:
            FlowsList实例
        """
        path = Path(path)
        
        if path.is_file():
            return self._load_from_file(path)
        elif path.is_dir():
            return self._load_from_directory(path)
        else:
            raise ConfigurationException(f"Flow path not found: {path}")
    
    def _load_from_file(self, file_path: Path) -> FlowsList:
        """从单个文件加载。
        
        Args:
            file_path: 文件路径
            
        Returns:
            FlowsList实例
        """
        logger.debug(f"Loading flows from file: {file_path}")
        
        data = read_yaml_file(str(file_path))
        if not data:
            logger.warning(f"Empty or invalid flow file: {file_path}")
            return FlowsList()
        
        return self._parse_flows_data(data)
    
    def _load_from_directory(self, dir_path: Path) -> FlowsList:
        """从目录加载。
        
        Args:
            dir_path: 目录路径
            
        Returns:
            FlowsList实例
        """
        logger.debug(f"Loading flows from directory: {dir_path}")
        
        # 查找所有YAML文件
        yaml_files = list(dir_path.glob("*.yml")) + list(dir_path.glob("*.yaml"))
        
        if not yaml_files:
            logger.warning(f"No YAML files found in: {dir_path}")
            return FlowsList()
        
        # 加载所有文件
        all_flows = FlowsList()
        
        for yaml_file in yaml_files:
            try:
                flows_list = self._load_from_file(yaml_file)
                for flow in flows_list:
                    all_flows.add_flow(flow)
            except Exception as e:
                logger.error(f"Failed to load flow file {yaml_file}: {e}")
        
        return all_flows
    
    def _parse_flows_data(self, data: Dict[str, Any]) -> FlowsList:
        """解析Flow数据。
        
        Args:
            data: YAML数据
            
        Returns:
            FlowsList实例
        """
        flows = []
        
        # 支持两种格式：
        # 1. flows: { flow_id: {...}, ... }
        # 2. 直接是 { flow_id: {...}, ... }
        flows_data = data.get("flows", data)
        
        if not isinstance(flows_data, dict):
            logger.warning(f"Invalid flows data format: {type(flows_data)}")
            return FlowsList()
        
        for flow_id, flow_config in flows_data.items():
            # 跳过非Flow字段
            if flow_id in ("version", "metadata", "imports"):
                continue
            
            if not isinstance(flow_config, dict):
                logger.warning(f"Invalid flow config for {flow_id}: {type(flow_config)}")
                continue
            
            try:
                flow = Flow.from_dict(flow_id, flow_config)
                flows.append(flow)
                logger.debug(f"Loaded flow: {flow_id} with {len(flow.steps)} steps")
            except Exception as e:
                logger.error(f"Failed to parse flow {flow_id}: {e}")
        
        return FlowsList(flows=flows)
    
    def load_from_string(self, yaml_string: str) -> FlowsList:
        """从YAML字符串加载。
        
        Args:
            yaml_string: YAML字符串
            
        Returns:
            FlowsList实例
        """
        from atguigu_ai.shared.yaml_loader import read_yaml_string
        
        data = read_yaml_string(yaml_string)
        if not data:
            return FlowsList()
        
        return self._parse_flows_data(data)


# 便捷函数

def load_flows(path: Union[str, Path]) -> FlowsList:
    """便捷函数：加载Flow。
    
    Args:
        path: 文件或目录路径
        
    Returns:
        FlowsList实例
    """
    loader = FlowLoader()
    return loader.load(path)


def load_flows_from_string(yaml_string: str) -> FlowsList:
    """便捷函数：从字符串加载Flow。
    
    Args:
        yaml_string: YAML字符串
        
    Returns:
        FlowsList实例
    """
    loader = FlowLoader()
    return loader.load_from_string(yaml_string)


# 导出
__all__ = [
    "FlowLoader",
    "load_flows",
    "load_flows_from_string",
]
