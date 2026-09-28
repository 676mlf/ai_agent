# -*- coding: utf-8 -*-
"""
Flow检索器

提供Flow语义检索功能。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from atguigu_ai.retrieval.base_retriever import Retriever, Document, SearchResult
from atguigu_ai.retrieval.embedder import Embedder, create_embedder

if TYPE_CHECKING:
    from atguigu_ai.dialogue_understanding.flow import Flow, FlowsList

logger = logging.getLogger(__name__)


@dataclass
class FlowDocument:
    """Flow文档。
    
    Attributes:
        flow_id: Flow ID
        flow_name: Flow名称
        description: Flow描述
        triggers: 触发条件
        embedding: 向量表示
    """
    flow_id: str
    flow_name: str
    description: str
    triggers: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


class FlowRetriever(Retriever):
    """Flow检索器。
    
    根据用户输入检索最相关的Flow。
    """
    
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        top_k: int = 3,
        threshold: float = 0.5,
    ):
        """初始化Flow检索器。
        
        Args:
            embedder: 向量化器
            top_k: 返回结果数量
            threshold: 相似度阈值
        """
        super().__init__()
        self.embedder = embedder or create_embedder()
        self.top_k = top_k
        self.threshold = threshold
        
        self._flow_documents: List[FlowDocument] = []
        self._index_built = False
    
    async def index_flows(self, flows: List["Flow"]) -> None:
        """索引Flow列表。
        
        Args:
            flows: Flow列表
        """
        self._flow_documents.clear()
        
        for flow in flows:
            # 构建Flow文档
            doc = FlowDocument(
                flow_id=flow.id,
                flow_name=flow.name,
                description=flow.description or "",
                triggers=getattr(flow, "triggers", []),
            )
            
            # 构建检索文本
            text_parts = [flow.name]
            if flow.description:
                text_parts.append(flow.description)
            if doc.triggers:
                text_parts.extend(doc.triggers)
            
            text = " ".join(text_parts)
            
            # 生成向量
            doc.embedding = await self.embedder.embed(text)
            self._flow_documents.append(doc)
        
        self._index_built = True
        logger.info(f"Indexed {len(self._flow_documents)} flows")
    
    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """搜索最相关的Flow。
        
        Args:
            query: 查询文本
            top_k: 返回数量
            **kwargs: 额外参数
            
        Returns:
            搜索结果列表
        """
        if not self._index_built:
            logger.warning("Flow index not built")
            return []
        
        top_k = top_k or self.top_k
        
        # 生成查询向量
        query_embedding = await self.embedder.embed(query)
        
        # 计算相似度
        results = []
        for doc in self._flow_documents:
            if doc.embedding is None:
                continue
            
            # 余弦相似度
            similarity = float(np.dot(query_embedding, doc.embedding))
            
            if similarity >= self.threshold:
                results.append(SearchResult(
                    document=Document(
                        id=doc.flow_id,
                        content=doc.description,
                        metadata={
                            "flow_id": doc.flow_id,
                            "flow_name": doc.flow_name,
                            "triggers": doc.triggers,
                        },
                    ),
                    score=similarity,
                ))
        
        # 排序并返回top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    async def add_document(self, document: Document) -> None:
        """添加文档（不支持）。"""
        raise NotImplementedError("Use index_flows to add flows")
    
    async def delete_document(self, doc_id: str) -> bool:
        """删除文档。"""
        for i, doc in enumerate(self._flow_documents):
            if doc.flow_id == doc_id:
                self._flow_documents.pop(i)
                return True
        return False
    
    def get_flow_by_id(self, flow_id: str) -> Optional[FlowDocument]:
        """根据ID获取Flow文档。
        
        Args:
            flow_id: Flow ID
            
        Returns:
            Flow文档
        """
        for doc in self._flow_documents:
            if doc.flow_id == flow_id:
                return doc
        return None


# 导出
__all__ = [
    "FlowDocument",
    "FlowRetriever",
]
