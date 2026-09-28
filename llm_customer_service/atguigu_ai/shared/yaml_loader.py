# -*- coding: utf-8 -*-
"""
yaml_loader - YAML文件安全加载器

提供YAML文件的安全加载功能，防止YAML注入攻击。
支持多文档YAML文件和Unicode编码。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Union

import yaml

from atguigu_ai.shared.constants import DEFAULT_ENCODING


class SafeLineLoader(yaml.SafeLoader):
    """安全的YAML加载器，保留行号信息
    
    继承自yaml.SafeLoader，添加行号追踪功能，便于错误定位。
    """
    pass


def _construct_mapping(loader: SafeLineLoader, node: yaml.Node) -> Dict[str, Any]:
    """构造映射时保留行号信息"""
    loader.flatten_mapping(node)
    pairs = loader.construct_pairs(node)
    return dict(pairs)


# 注册自定义构造函数
SafeLineLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping
)


def read_yaml_file(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """读取YAML文件
    
    安全地读取YAML文件内容，返回解析后的字典。
    
    参数：
        path: YAML文件路径
        
    返回：
        解析后的字典，如果文件为空则返回None
        
    异常：
        FileNotFoundError: 文件不存在
        yaml.YAMLError: YAML格式错误
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"YAML文件不存在: {path}")
    
    with open(path, "r", encoding=DEFAULT_ENCODING) as f:
        content = yaml.safe_load(f)
    
    return content


def read_yaml_string(yaml_string: Text) -> Optional[Dict[str, Any]]:
    """解析YAML字符串
    
    参数：
        yaml_string: YAML格式的字符串
        
    返回：
        解析后的字典，如果字符串为空则返回None
        
    异常：
        yaml.YAMLError: YAML格式错误
    """
    return yaml.safe_load(yaml_string)


def read_yaml_files(paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
    """读取多个YAML文件
    
    参数：
        paths: YAML文件路径列表
        
    返回：
        解析后的字典列表(过滤掉空文件)
    """
    results = []
    for path in paths:
        content = read_yaml_file(path)
        if content is not None:
            results.append(content)
    return results


def read_yaml_multi_document(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """读取多文档YAML文件
    
    支持使用---分隔的多文档YAML文件。
    
    参数：
        path: YAML文件路径
        
    返回：
        解析后的字典列表
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"YAML文件不存在: {path}")
    
    with open(path, "r", encoding=DEFAULT_ENCODING) as f:
        documents = list(yaml.safe_load_all(f))
    
    return [doc for doc in documents if doc is not None]


def write_yaml_file(
    data: Dict[str, Any],
    path: Union[str, Path],
    allow_unicode: bool = True,
    default_flow_style: bool = False,
) -> None:
    """写入YAML文件
    
    将字典数据写入YAML文件。
    
    参数：
        data: 要写入的数据
        path: 目标文件路径
        allow_unicode: 是否允许Unicode字符(默认True)
        default_flow_style: 是否使用流式风格(默认False，使用块风格)
    """
    path = Path(path)
    
    # 确保父目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding=DEFAULT_ENCODING) as f:
        yaml.dump(
            data,
            f,
            allow_unicode=allow_unicode,
            default_flow_style=default_flow_style,
            sort_keys=False,
        )


def dump_yaml_string(
    data: Dict[str, Any],
    allow_unicode: bool = True,
    default_flow_style: bool = False,
) -> str:
    """将数据转换为YAML字符串
    
    参数：
        data: 要转换的数据
        allow_unicode: 是否允许Unicode字符
        default_flow_style: 是否使用流式风格
        
    返回：
        YAML格式的字符串
    """
    return yaml.dump(
        data,
        allow_unicode=allow_unicode,
        default_flow_style=default_flow_style,
        sort_keys=False,
    )


def merge_yaml_files(paths: List[Union[str, Path]]) -> Dict[str, Any]:
    """合并多个YAML文件
    
    后面的文件会覆盖前面文件中的同名键。
    
    参数：
        paths: YAML文件路径列表
        
    返回：
        合并后的字典
    """
    result: Dict[str, Any] = {}
    
    for path in paths:
        content = read_yaml_file(path)
        if content:
            result = _deep_merge(result, content)
    
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并两个字典
    
    递归合并嵌套的字典，override中的值会覆盖base中的值。
    
    参数：
        base: 基础字典
        override: 覆盖字典
        
    返回：
        合并后的字典
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result
