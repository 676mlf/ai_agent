# -*- coding: utf-8 -*-
"""
检索模块

提供向量检索基类，通过继承 InformationRetrieval 实现自定义检索器。

使用方式：
1. 创建自定义检索器类，继承 InformationRetrieval
2. 实现 connect() 和 search() 方法
3. 在 config.yml 的 policies.EnterpriseSearchPolicy.vector_store 中指定全类名
4. 在 endpoints.yml 的 vector_store 中配置连接参数

示例：
    class Neo4jRetriever(InformationRetrieval):
        def connect(self, config):
            self.driver = GraphDatabase.driver(
                config["uri"],
                auth=(config["user"], config["password"])
            )
        
        async def search(self, query, top_k=5):
            # 执行检索
            return [SearchResult(text=..., score=...)]
"""

import importlib
import logging
from typing import Any, Dict, Optional

from atguigu_ai.retrieval.base_retriever import (
    InformationRetrieval,
    Retriever,  # 向后兼容别名
    SearchResult,
    Document,
)
from atguigu_ai.retrieval.embedder import (
    Embedder,
    EmbedderConfig,
    OpenAIEmbedder,
    LocalEmbedder,
    create_embedder,
)
from atguigu_ai.retrieval.flow_retriever import FlowRetriever, FlowDocument

logger = logging.getLogger(__name__)


def create_retriever(
    class_path: str,
    connect_config: Optional[Dict[str, Any]] = None,
) -> Optional[InformationRetrieval]:
    """根据类路径创建 Retriever 实例。
    
    从 config.yml 的 policies.EnterpriseSearchPolicy.vector_store 读取类路径，
    从 endpoints.yml 的 vector_store 读取连接配置。
    
    Args:
        class_path: 检索器类的完整路径，如 "my_module.Neo4jRetriever"
        connect_config: 传递给 retriever.connect() 的配置字典
        
    Returns:
        InformationRetrieval 实例，如果创建失败则返回 None
    """
    if not class_path:
        logger.debug("未指定检索器类路径")
        return None
    
    # 动态加载类
    try:
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        retriever_class = getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        logger.error(f"加载检索器类 '{class_path}' 失败: {e}")
        raise ImportError(f"无法加载检索器类 '{class_path}': {e}")
    
    # 验证是否继承自 InformationRetrieval
    if not issubclass(retriever_class, InformationRetrieval):
        raise TypeError(
            f"检索器类 '{class_path}' 必须继承自 InformationRetrieval"
        )
    
    # 创建实例
    try:
        retriever = retriever_class()
        
        # 调用 connect 初始化连接
        if connect_config:
            retriever.connect(connect_config)
            logger.info(f"创建检索器: {class_name}")
        
        return retriever
        
    except Exception as e:
        logger.error(f"创建检索器实例失败: {e}")
        raise


__all__ = [
    # 基类
    "InformationRetrieval",
    "Retriever",  # 向后兼容
    "SearchResult",
    "Document",
    # Embedder
    "Embedder",
    "EmbedderConfig",
    "OpenAIEmbedder",
    "LocalEmbedder",
    "create_embedder",
    # Flow 检索（特殊用途）
    "FlowRetriever",
    "FlowDocument",
    # 工厂函数
    "create_retriever",
]
