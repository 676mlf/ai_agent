# -*- coding: utf-8 -*-
"""
微调数据生成器

生成用于LLM微调的训练数据。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from atguigu_ai.core.domain import Domain
    from atguigu_ai.dialogue_understanding.flow import Flow

logger = logging.getLogger(__name__)


@dataclass
class FinetuneExample:
    """微调样本。
    
    Attributes:
        messages: 对话消息列表
        metadata: 元数据
    """
    messages: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """转换为OpenAI微调格式。"""
        return {"messages": self.messages}
    
    def to_jsonl(self) -> str:
        """转换为JSONL格式。"""
        return json.dumps(self.to_openai_format(), ensure_ascii=False)


@dataclass
class FinetuneConfig:
    """微调配置。
    
    Attributes:
        system_prompt: 系统提示
        num_examples_per_flow: 每个Flow生成的样本数
        include_negative_examples: 是否包含负样本
        augmentation_factor: 数据增强倍数
        output_format: 输出格式 (openai, alpaca, custom)
    """
    system_prompt: str = "你是一个智能对话助手。"
    num_examples_per_flow: int = 5
    include_negative_examples: bool = True
    augmentation_factor: int = 3
    output_format: str = "openai"


class FinetuneDataGenerator:
    """微调数据生成器。
    
    从Flow定义和对话数据生成微调训练样本。
    """
    
    def __init__(
        self,
        config: Optional[FinetuneConfig] = None,
        paraphraser: Optional["Paraphraser"] = None,
    ):
        """初始化生成器。
        
        Args:
            config: 配置
            paraphraser: 句子改写器
        """
        self.config = config or FinetuneConfig()
        self.paraphraser = paraphraser
    
    async def generate_from_flows(
        self,
        flows: List["Flow"],
        domain: Optional["Domain"] = None,
    ) -> List[FinetuneExample]:
        """从Flow生成训练数据。
        
        Args:
            flows: Flow列表
            domain: Domain定义
            
        Returns:
            微调样本列表
        """
        examples = []
        
        for flow in flows:
            flow_examples = await self._generate_flow_examples(flow, domain)
            examples.extend(flow_examples)
        
        logger.info(f"Generated {len(examples)} examples from {len(flows)} flows")
        return examples
    
    async def _generate_flow_examples(
        self,
        flow: "Flow",
        domain: Optional["Domain"],
    ) -> List[FinetuneExample]:
        """为单个Flow生成训练样本。"""
        examples = []
        
        # 生成正样本
        for i in range(self.config.num_examples_per_flow):
            example = await self._create_positive_example(flow, domain, i)
            if example:
                examples.append(example)
        
        # 数据增强
        if self.paraphraser and self.config.augmentation_factor > 1:
            augmented = []
            for example in examples:
                aug_examples = await self._augment_example(example)
                augmented.extend(aug_examples)
            examples.extend(augmented)
        
        # 生成负样本
        if self.config.include_negative_examples:
            neg_examples = await self._create_negative_examples(flow, domain)
            examples.extend(neg_examples)
        
        return examples
    
    async def _create_positive_example(
        self,
        flow: "Flow",
        domain: Optional["Domain"],
        index: int,
    ) -> Optional[FinetuneExample]:
        """创建正样本。"""
        messages = [
            {"role": "system", "content": self.config.system_prompt}
        ]
        
        # 用户消息 - 模拟触发flow的意图
        user_message = self._generate_user_message(flow, index)
        messages.append({"role": "user", "content": user_message})
        
        # 助手响应 - 应该启动对应的flow
        assistant_response = f"StartFlow({flow.id})"
        messages.append({"role": "assistant", "content": assistant_response})
        
        return FinetuneExample(
            messages=messages,
            metadata={
                "flow_id": flow.id,
                "flow_name": flow.name,
                "example_type": "positive",
                "index": index,
            },
        )
    
    def _generate_user_message(self, flow: "Flow", index: int) -> str:
        """生成用户消息。"""
        # 从flow描述或名称生成
        if flow.description:
            return f"我想{flow.description}"
        return f"帮我{flow.name}"
    
    async def _create_negative_examples(
        self,
        flow: "Flow",
        domain: Optional["Domain"],
    ) -> List[FinetuneExample]:
        """创建负样本。"""
        examples = []
        
        # 不相关查询的负样本
        negative_queries = [
            "今天天气怎么样？",
            "给我讲个笑话",
            "你好",
        ]
        
        for query in negative_queries:
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": "ChitChat()"},
            ]
            
            examples.append(FinetuneExample(
                messages=messages,
                metadata={
                    "flow_id": flow.id,
                    "example_type": "negative",
                },
            ))
        
        return examples
    
    async def _augment_example(
        self,
        example: FinetuneExample,
    ) -> List[FinetuneExample]:
        """数据增强。"""
        augmented = []
        
        if not self.paraphraser:
            return augmented
        
        # 找到用户消息并改写
        for msg in example.messages:
            if msg["role"] == "user":
                original_text = msg["content"]
                
                # 生成改写版本
                for _ in range(self.config.augmentation_factor - 1):
                    paraphrased = await self.paraphraser.paraphrase(original_text)
                    if paraphrased and paraphrased != original_text:
                        new_messages = []
                        for m in example.messages:
                            if m["role"] == "user" and m["content"] == original_text:
                                new_messages.append({"role": "user", "content": paraphrased})
                            else:
                                new_messages.append(m.copy())
                        
                        augmented.append(FinetuneExample(
                            messages=new_messages,
                            metadata={
                                **example.metadata,
                                "augmented": True,
                                "original": original_text,
                            },
                        ))
                break
        
        return augmented
    
    def save_to_file(
        self,
        examples: List[FinetuneExample],
        output_path: str,
    ) -> None:
        """保存到文件。
        
        Args:
            examples: 样本列表
            output_path: 输出路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(example.to_jsonl() + "\n")
        
        logger.info(f"Saved {len(examples)} examples to {output_path}")


# 导出
__all__ = [
    "FinetuneExample",
    "FinetuneConfig",
    "FinetuneDataGenerator",
]
