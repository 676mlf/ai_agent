# -*- coding: utf-8 -*-
"""
模板NLG

基于Domain中定义的响应模板生成回复。
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, Optional, TYPE_CHECKING

from atguigu_ai.nlg.nlg_generator import NLGGenerator, NLGConfig, NLGResponse

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker
    from atguigu_ai.core.domain import Domain


class TemplateNLG(NLGGenerator):
    """模板NLG生成器。
    
    从Domain的responses中获取模板并生成回复。
    支持变量替换（使用{slot_name}语法）。
    """
    
    async def generate(
        self,
        utter_action: str,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> NLGResponse:
        """生成回复。
        
        Args:
            utter_action: 响应动作名称
            tracker: 对话状态追踪器
            domain: Domain定义
            **kwargs: 额外参数
            
        Returns:
            NLG响应
        """
        if domain is None:
            return NLGResponse(text="")
        
        # 获取响应模板
        templates = domain.get_response(utter_action)
        if not templates:
            return NLGResponse(text="")
        
        # 随机选择一个模板
        template = random.choice(templates)
        
        # 获取文本
        text = template.text or ""
        
        # 替换槽位变量
        text = self._fill_slots(text, tracker)
        
        # 构建响应
        response = NLGResponse(
            text=text,
            buttons=template.buttons or [],
            image=template.image,
            custom=template.custom,
            metadata=template.metadata or {},
        )
        
        return response
    
    def _fill_slots(
        self,
        text: str,
        tracker: "DialogueStateTracker",
    ) -> str:
        """替换文本中的槽位变量。
        
        Args:
            text: 原始文本
            tracker: 对话状态追踪器
            
        Returns:
            替换后的文本
        """
        if not text:
            return text
        
        # 获取所有槽位值
        slots = tracker.get_all_slots()
        
        # 使用正则表达式查找所有 {slot_name} 模式
        pattern = r'\{(\w+)\}'
        
        def replace_slot(match: re.Match) -> str:
            slot_name = match.group(1)
            value = slots.get(slot_name)
            if value is not None:
                return str(value)
            return match.group(0)  # 保持原样
        
        return re.sub(pattern, replace_slot, text)


# 导出
__all__ = ["TemplateNLG"]
