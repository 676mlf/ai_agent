# -*- coding: utf-8 -*-
"""
检索器基类

定义向量检索的最小化抽象接口：connect + search。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from atguigu_ai.retrieval.embedder import Embedder


@dataclass
class Document:
    """文档类 - 向后兼容。
    
    Attributes:
        id: 文档ID
        content: 文档内容
        metadata: 元数据
    """
    id: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """搜索结果。
    
    支持两种构造方式：
    1. 新方式：SearchResult(text="...", score=0.9)
    2. 旧方式：SearchResult(document=Document(...), score=0.9)
    
    Attributes:
        text: 检索到的文本内容
        metadata: 元数据（来源、页码等）
        score: 相似度分数
        document: 向后兼容的Document对象
    """
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    document: Optional[Document] = None
    
    def __post_init__(self):
        """初始化后处理：兼容旧的document参数。"""
        if self.document is not None and not self.text:
            self.text = self.document.content
            if not self.metadata:
                self.metadata = self.document.metadata.copy()
    
    @property
    def source(self) -> str:
        """获取来源。"""
        return self.metadata.get("source", "unknown")
    
    # 向后兼容：提供content属性
    @property
    def content(self) -> str:
        """获取内容（兼容旧接口）。"""
        return self.text


class InformationRetrieval(ABC):
    """检索器抽象基类。
    
    只定义两个核心方法：
    - connect(): 连接/初始化向量索引
    - search(): 语义搜索
    
    自定义检索器只需继承此类并实现这两个方法即可。
    
    示例:
        class MyRetriever(InformationRetrieval):
            def connect(self, config):
                self.client = MyVectorDB(**config)
            
            async def search(self, query, top_k=5):
                results = await self.client.search(query, limit=top_k)
                return [SearchResult(text=r.text, score=r.score) for r in results]
    """
    
    def __init__(self, embeddings: Optional["Embedder"] = None) -> None:
        """初始化检索器。
        
        Args:
            embeddings: 嵌入向量化器（可选）
        """
        self.embeddings = embeddings
    
    @abstractmethod
    def connect(self, config: Optional[Dict[str, Any]] = None) -> None:
        """连接/初始化向量索引。
        
        Args:
            config: 连接配置（如索引路径、数据库地址等）
        """
        raise NotImplementedError()
    
    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        tracker_state: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """语义搜索。
        
        Args:
            query: 搜索查询文本
            top_k: 返回结果数量
            tracker_state: 对话状态（用于获取用户信息和历史对话）
            
        Returns:
            搜索结果列表
        """
        raise NotImplementedError()


# 向后兼容：保留Retriever别名
Retriever = InformationRetrieval


# 导出
__all__ = [
    "InformationRetrieval",
    "Retriever",  # 向后兼容
    "Document",   # 向后兼容
    "SearchResult",
]
