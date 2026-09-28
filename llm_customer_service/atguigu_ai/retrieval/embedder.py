# -*- coding: utf-8 -*-
"""
向量化接口

提供文本嵌入向量化功能。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbedderConfig:
    """向量化器配置。
    
    Attributes:
        provider: 提供商 (openai, local, custom)
        model: 模型名称
        api_key: API密钥
        dimension: 向量维度
        batch_size: 批处理大小
        normalize: 是否归一化
    """
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    dimension: int = 1536
    batch_size: int = 100
    normalize: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class Embedder(ABC):
    """向量化器抽象基类。
    
    将文本转换为向量表示。
    """
    
    def __init__(self, config: Optional[EmbedderConfig] = None):
        """初始化向量化器。
        
        Args:
            config: 向量化器配置
        """
        self.config = config or EmbedderConfig()
    
    @property
    def dimension(self) -> int:
        """向量维度。"""
        return self.config.dimension
    
    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """将单个文本转换为向量。
        
        Args:
            text: 输入文本
            
        Returns:
            向量表示
        """
        raise NotImplementedError()
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """批量将文本转换为向量。
        
        Args:
            texts: 输入文本列表
            
        Returns:
            向量列表
        """
        raise NotImplementedError()
    
    def embed_sync(self, text: str) -> np.ndarray:
        """同步版本的embed。"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.embed(text))
    
    def embed_batch_sync(self, texts: List[str]) -> List[np.ndarray]:
        """同步版本的embed_batch。"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.embed_batch(texts))


class OpenAIEmbedder(Embedder):
    """OpenAI向量化器。
    
    使用OpenAI Embeddings API进行向量化。
    """
    
    def __init__(self, config: Optional[EmbedderConfig] = None):
        """初始化OpenAI向量化器。"""
        if config is None:
            config = EmbedderConfig(provider="openai")
        super().__init__(config)
        self._client = None
    
    def _get_client(self):
        """获取或创建OpenAI客户端。"""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("openai package required for OpenAIEmbedder")
        return self._client
    
    async def embed(self, text: str) -> np.ndarray:
        """将文本转换为向量。"""
        client = self._get_client()
        
        response = await client.embeddings.create(
            model=self.config.model,
            input=text,
        )
        
        embedding = np.array(response.data[0].embedding)
        
        if self.config.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """批量将文本转换为向量。"""
        if not texts:
            return []
        
        client = self._get_client()
        results = []
        
        # 分批处理
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            response = await client.embeddings.create(
                model=self.config.model,
                input=batch,
            )
            
            for item in response.data:
                embedding = np.array(item.embedding)
                if self.config.normalize:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                results.append(embedding)
        
        return results


class LocalEmbedder(Embedder):
    """本地向量化器。
    
    使用本地模型进行向量化（适用于离线场景）。
    """
    
    def __init__(self, config: Optional[EmbedderConfig] = None):
        """初始化本地向量化器。"""
        if config is None:
            config = EmbedderConfig(
                provider="local",
                dimension=384,  # 常见的小模型维度
            )
        super().__init__(config)
        self._model = None
    
    def _get_model(self):
        """获取或创建模型。"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.model or "all-MiniLM-L6-v2"
                self._model = SentenceTransformer(model_name)
            except ImportError:
                logger.warning("sentence-transformers not installed, using random embeddings")
                self._model = "random"
        return self._model
    
    async def embed(self, text: str) -> np.ndarray:
        """将文本转换为向量。"""
        model = self._get_model()
        
        if model == "random":
            # 降级为随机向量
            return np.random.randn(self.config.dimension).astype(np.float32)
        
        embedding = model.encode(text, convert_to_numpy=True)
        
        if self.config.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """批量将文本转换为向量。"""
        if not texts:
            return []
        
        model = self._get_model()
        
        if model == "random":
            return [np.random.randn(self.config.dimension).astype(np.float32) for _ in texts]
        
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        results = []
        for embedding in embeddings:
            if self.config.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            results.append(embedding)
        
        return results


def create_embedder(
    provider: str = "openai",
    **kwargs: Any,
) -> Embedder:
    """创建向量化器工厂函数。
    
    Args:
        provider: 提供商名称
        **kwargs: 配置参数
        
    Returns:
        向量化器实例
    """
    config = EmbedderConfig(provider=provider, **kwargs)
    
    if provider == "openai":
        return OpenAIEmbedder(config)
    elif provider == "local":
        return LocalEmbedder(config)
    else:
        raise ValueError(f"Unknown embedder provider: {provider}")


# 导出
__all__ = [
    "EmbedderConfig",
    "Embedder",
    "OpenAIEmbedder",
    "LocalEmbedder",
    "create_embedder",
]
