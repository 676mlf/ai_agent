# -*- coding: utf-8 -*-
"""
响应重述器

使用LLM对模板响应进行重述，使回复更加自然和个性化。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from atguigu_ai.nlg.nlg_generator import NLGGenerator, NLGConfig, NLGResponse
from atguigu_ai.shared.llm import create_llm_client
from atguigu_ai.shared.llm.base_client import LLMClient

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker
    from atguigu_ai.core.domain import Domain

logger = logging.getLogger(__name__)


@dataclass
class RephraserConfig:
    """重述器配置。
    
    Attributes:
        enabled: 是否启用重述
        llm_type: LLM类型 (openai/qwen/azure/anthropic)
        llm_model: LLM模型
        temperature: 生成温度
        rephrase_threshold: 触发重述的最小文本长度
        preserve_slots: 重述时是否保留槽位占位符
        style: 重述风格 (friendly/professional/casual)
        language: 目标语言
    """
    enabled: bool = True
    llm_type: str = "openai"
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    rephrase_threshold: int = 10
    preserve_slots: bool = True
    style: str = "friendly"
    language: str = "zh"


# 重述提示词模板
REPHRASE_PROMPT_TEMPLATE = """请将以下机器人回复重述得更加自然和{style}。

原始回复：
{original_text}

重述要求：
1. 保持原意不变
2. 使语言更加自然流畅
3. 采用{style}的语气
4. 使用{language}
5. 保持简洁，不要过度扩展
{slot_instruction}

请直接输出重述后的文本，不要包含任何解释：
"""

STYLE_DESCRIPTIONS = {
    "friendly": "友好、亲切",
    "professional": "专业、正式",
    "casual": "轻松、随意",
    "empathetic": "富有同理心、温暖",
}


class ResponseRephraser(NLGGenerator):
    """响应重述器。
    
    对底层NLG生成器的响应进行LLM重述，使回复更加自然。
    
    支持两种工作模式：
    1. 装饰器模式：包装另一个NLG生成器
    2. 独立模式：直接重述传入的文本
    
    重述触发条件：
    - 配置中enabled=True
    - 响应metadata中rephrase=True，或
    - 响应文本长度超过rephrase_threshold
    
    使用示例：
    ```python
    # 装饰器模式
    template_nlg = TemplateNLG(domain)
    rephraser = ResponseRephraser(base_generator=template_nlg)
    response = await rephraser.generate("utter_greet", tracker, domain)
    
    # 独立模式
    rephraser = ResponseRephraser()
    rephrased = await rephraser.rephrase("您好，请问有什么可以帮您？")
    ```
    """
    
    def __init__(
        self,
        config: Optional[RephraserConfig] = None,
        base_generator: Optional[NLGGenerator] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        """初始化重述器。
        
        Args:
            config: 重述器配置
            base_generator: 底层NLG生成器
            llm_client: LLM客户端
        """
        super().__init__()
        self.rephrase_config = config or RephraserConfig()
        self.base_generator = base_generator
        self._llm_client = llm_client
    
    @property
    def llm_client(self) -> LLMClient:
        """获取LLM客户端（延迟初始化）。"""
        if self._llm_client is None:
            self._llm_client = create_llm_client(
                type=self.rephrase_config.llm_type,
                model=self.rephrase_config.llm_model,
                temperature=self.rephrase_config.temperature,
            )
        return self._llm_client
    
    async def generate(
        self,
        utter_action: str,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> NLGResponse:
        """生成回复。
        
        首先使用底层生成器生成响应，然后根据条件决定是否重述。
        
        Args:
            utter_action: 响应动作名称
            tracker: 对话状态追踪器
            domain: Domain定义
            **kwargs: 额外参数
            
        Returns:
            NLG响应
        """
        # 如果没有底层生成器，直接返回空响应
        if self.base_generator is None:
            logger.warning("No base generator configured, returning empty response")
            return NLGResponse(text="")
        
        # 使用底层生成器生成原始响应
        original_response = await self.base_generator.generate(
            utter_action, tracker, domain, **kwargs
        )
        
        # 检查是否应该重述
        if self._should_rephrase(original_response):
            try:
                rephrased_text = await self.rephrase(
                    original_response.text,
                    context=self._build_context(tracker),
                )
                
                # 创建新响应，保留原始响应的其他属性
                return NLGResponse(
                    text=rephrased_text,
                    buttons=original_response.buttons,
                    image=original_response.image,
                    custom=original_response.custom,
                    metadata={
                        **original_response.metadata,
                        "rephrased": True,
                        "original_text": original_response.text,
                    },
                )
            except Exception as e:
                logger.warning(f"Rephrase failed, using original: {e}")
                return original_response
        
        return original_response
    
    def _should_rephrase(self, response: NLGResponse) -> bool:
        """判断是否应该重述。
        
        Args:
            response: 原始响应
            
        Returns:
            是否应该重述
        """
        if not self.rephrase_config.enabled:
            return False
        
        # 检查metadata中的rephrase标记
        if response.metadata.get("rephrase", False):
            return True
        
        # 检查文本长度阈值
        if len(response.text) >= self.rephrase_config.rephrase_threshold:
            return True
        
        return False
    
    async def rephrase(
        self,
        text: str,
        style: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """重述文本。
        
        Args:
            text: 原始文本
            style: 重述风格（覆盖配置）
            context: 上下文信息
            
        Returns:
            重述后的文本
        """
        if not text or not text.strip():
            return text
        
        style = style or self.rephrase_config.style
        style_desc = STYLE_DESCRIPTIONS.get(style, style)
        
        # 构建槽位保留指令
        slot_instruction = ""
        if self.rephrase_config.preserve_slots:
            slot_instruction = "6. 保留所有{xxx}格式的占位符不变"
        
        # 构建提示词
        prompt = REPHRASE_PROMPT_TEMPLATE.format(
            original_text=text,
            style=style_desc,
            language=self.rephrase_config.language,
            slot_instruction=slot_instruction,
        )
        
        # 调用LLM
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm_client.chat(messages)
            
            rephrased = response.content.strip()
            
            # 验证重述结果
            if self._validate_rephrase(text, rephrased):
                logger.debug(f"Rephrased: '{text[:50]}...' -> '{rephrased[:50]}...'")
                return rephrased
            else:
                logger.warning("Rephrase validation failed, using original")
                return text
                
        except Exception as e:
            logger.error(f"LLM rephrase error: {e}")
            return text
    
    def _validate_rephrase(self, original: str, rephrased: str) -> bool:
        """验证重述结果。
        
        Args:
            original: 原始文本
            rephrased: 重述文本
            
        Returns:
            是否有效
        """
        # 基本检查
        if not rephrased or not rephrased.strip():
            return False
        
        # 长度检查：重述不应过度扩展或缩减
        original_len = len(original)
        rephrased_len = len(rephrased)
        
        if rephrased_len < original_len * 0.3:
            return False
        if rephrased_len > original_len * 3:
            return False
        
        # 如果需要保留槽位，检查槽位占位符
        if self.rephrase_config.preserve_slots:
            import re
            original_slots = set(re.findall(r'\{(\w+)\}', original))
            rephrased_slots = set(re.findall(r'\{(\w+)\}', rephrased))
            
            if original_slots and original_slots != rephrased_slots:
                return False
        
        return True
    
    def _build_context(
        self,
        tracker: "DialogueStateTracker",
    ) -> Dict[str, Any]:
        """构建重述上下文。
        
        Args:
            tracker: 对话状态追踪器
            
        Returns:
            上下文字典
        """
        context = {
            "sender_id": tracker.sender_id,
        }
        
        # 获取最近的用户消息
        if tracker.latest_message:
            context["last_user_message"] = tracker.latest_message.get("text", "")
        
        # 获取当前流程
        if tracker.active_flow:
            context["active_flow"] = tracker.active_flow
        
        return context


# 导出
__all__ = [
    "ResponseRephraser",
    "RephraserConfig",
]
