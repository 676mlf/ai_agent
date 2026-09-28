# -*- coding: utf-8 -*-
"""
句子改写器

提供句子改写功能，用于数据增强。
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from atguigu_ai.shared.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ParaphraserConfig:
    """改写器配置。
    
    Attributes:
        use_llm: 是否使用LLM改写
        llm_model: LLM模型
        temperature: 生成温度
        num_variations: 每次生成的变体数
        preserve_intent: 是否保持意图
    """
    use_llm: bool = True
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    num_variations: int = 3
    preserve_intent: bool = True


class Paraphraser:
    """句子改写器。
    
    将句子改写为语义相同但表达不同的形式。
    """
    
    # 简单的同义词替换规则
    SYNONYM_MAP = {
        "想要": ["想", "需要", "希望", "要"],
        "帮助": ["帮忙", "协助", "支持"],
        "查询": ["查看", "查找", "搜索", "检索"],
        "订购": ["订", "预订", "预约", "下单"],
        "取消": ["撤销", "退订", "不要了"],
        "修改": ["更改", "变更", "调整"],
        "确认": ["确定", "核实", "验证"],
    }
    
    # 句式变换模板
    SENTENCE_PATTERNS = [
        ("我想{action}", "请帮我{action}"),
        ("我想{action}", "能不能{action}"),
        ("我想{action}", "我需要{action}"),
        ("帮我{action}", "请{action}"),
        ("帮我{action}", "我要{action}"),
    ]
    
    def __init__(
        self,
        config: Optional[ParaphraserConfig] = None,
        llm_client: Optional["LLMClient"] = None,
    ):
        """初始化改写器。
        
        Args:
            config: 配置
            llm_client: LLM客户端
        """
        self.config = config or ParaphraserConfig()
        self.llm_client = llm_client
    
    async def paraphrase(
        self,
        text: str,
        num_variations: Optional[int] = None,
    ) -> str:
        """改写句子。
        
        Args:
            text: 原始句子
            num_variations: 变体数量
            
        Returns:
            改写后的句子
        """
        if self.config.use_llm and self.llm_client:
            try:
                return await self._llm_paraphrase(text)
            except Exception as e:
                logger.warning(f"LLM paraphrase failed: {e}")
        
        # 降级到规则改写
        return self._rule_based_paraphrase(text)
    
    async def paraphrase_batch(
        self,
        texts: List[str],
        num_variations: Optional[int] = None,
    ) -> List[str]:
        """批量改写句子。
        
        Args:
            texts: 原始句子列表
            num_variations: 每个句子的变体数量
            
        Returns:
            改写后的句子列表
        """
        results = []
        for text in texts:
            paraphrased = await self.paraphrase(text, num_variations)
            results.append(paraphrased)
        return results
    
    async def generate_variations(
        self,
        text: str,
        num_variations: Optional[int] = None,
    ) -> List[str]:
        """生成多个变体。
        
        Args:
            text: 原始句子
            num_variations: 变体数量
            
        Returns:
            变体列表
        """
        num = num_variations or self.config.num_variations
        variations = set()
        variations.add(text)  # 包含原始句子
        
        for _ in range(num * 2):  # 多尝试几次以获得足够的变体
            if len(variations) >= num + 1:
                break
            
            paraphrased = await self.paraphrase(text)
            if paraphrased and paraphrased != text:
                variations.add(paraphrased)
        
        return list(variations)
    
    async def _llm_paraphrase(self, text: str) -> str:
        """使用LLM改写。"""
        prompt = f"""请将以下句子改写为语义相同但表达不同的形式。
保持原意，但使用不同的词汇和句式。
只输出改写后的句子，不要解释。

原句：{text}

改写："""
        
        messages = [
            {"role": "system", "content": "你是一个句子改写专家。"},
            {"role": "user", "content": prompt},
        ]
        
        response = await self.llm_client.complete(messages)
        return response.strip()
    
    def _rule_based_paraphrase(self, text: str) -> str:
        """基于规则的改写。"""
        result = text
        
        # 1. 同义词替换
        for original, synonyms in self.SYNONYM_MAP.items():
            if original in result:
                replacement = random.choice(synonyms)
                result = result.replace(original, replacement, 1)
                break
        
        # 2. 句式变换
        for pattern, replacement in self.SENTENCE_PATTERNS:
            # 简单模式匹配
            if text.startswith(pattern.split("{")[0]):
                action_start = len(pattern.split("{")[0])
                action = text[action_start:]
                result = replacement.format(action=action)
                break
        
        return result
    
    def add_synonym(self, word: str, synonyms: List[str]) -> None:
        """添加同义词。
        
        Args:
            word: 原词
            synonyms: 同义词列表
        """
        if word in self.SYNONYM_MAP:
            self.SYNONYM_MAP[word].extend(synonyms)
        else:
            self.SYNONYM_MAP[word] = synonyms


# 导出
__all__ = [
    "ParaphraserConfig",
    "Paraphraser",
]
