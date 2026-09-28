# -*- coding: utf-8 -*-
"""
Prompt构建器

使用Jinja2模板引擎构建发送给LLM的提示词。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker, UserMessage
    from atguigu_ai.core.domain import Domain
    from atguigu_ai.dialogue_understanding.flow import Flow, FlowsList

logger = logging.getLogger(__name__)


# 模板目录
TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass
class PromptBuilder:
    """Prompt构建器。
    
    使用Jinja2模板引擎根据对话上下文构建发送给LLM的提示词。
    
    Attributes:
        template_name: 主模板文件名
        max_history_turns: 包含的最大历史轮数
        include_slots: 是否包含槽位信息
        include_flows: 是否包含Flow信息
    """
    
    template_name: str = "command_prompt.jinja2"
    max_history_turns: int = 5
    include_slots: bool = True
    include_flows: bool = True
    
    def __post_init__(self):
        """初始化Jinja2环境。"""
        self._env = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._template = self._env.get_template(self.template_name)
    
    def build_prompt(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        flows: Optional["FlowsList"] = None,
    ) -> str:
        """构建完整的提示词。
        
        使用Jinja2模板渲染提示词。
        
        Args:
            tracker: 对话状态追踪器
            domain: Domain定义
            flows: 可用的Flow列表
            
        Returns:
            完整的提示词
        """
        context = self._build_template_context(tracker, domain, flows)
        return self._template.render(**context)
    
    def build_messages(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        flows: Optional["FlowsList"] = None,
    ) -> List[Dict[str, str]]:
        """构建消息列表格式的提示词。
        
        返回适合发送给LLM API的消息列表。
        
        Args:
            tracker: 对话状态追踪器
            domain: Domain定义
            flows: 可用的Flow列表
            
        Returns:
            消息列表 [{"role": "user", "content": "..."}]
        """
        prompt = self.build_prompt(tracker, domain, flows)
        
        # 使用单个用户消息包含完整提示词
        # 这样设计是因为提示词已经包含了系统指令和上下文
        return [{"role": "user", "content": prompt}]
    
    def _build_template_context(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        flows: Optional["FlowsList"] = None,
    ) -> Dict[str, Any]:
        """构建模板渲染上下文。
        
        Args:
            tracker: 对话状态追踪器
            domain: Domain定义
            flows: 可用的Flow列表
            
        Returns:
            模板上下文字典
        """
        context: Dict[str, Any] = {}
        
        # 用户消息
        if tracker.latest_message:
            context["user_message"] = tracker.latest_message.text
        else:
            context["user_message"] = ""
        
        # Flows信息
        if self.include_flows and flows:
            context["flows"] = self._prepare_flows_data(flows, domain)
        else:
            context["flows"] = []
        
        # 槽位信息
        if self.include_slots:
            context["slot_info"] = self._prepare_slot_info(domain)
        else:
            context["slot_info"] = {}
        
        # 当前Flow状态
        context["current_flow"] = tracker.active_flow
        context["current_slot"] = self._get_current_slot(tracker)
        context["current_slot_description"] = self._get_current_slot_description(
            tracker, domain
        )
        context["flow_slots"] = self._get_flow_slots(tracker, domain, flows)
        
        # 对话历史
        context["conversation_history"] = self._get_conversation_history(tracker)
        
        return context
    
    def _prepare_flows_data(
        self,
        flows: "FlowsList",
        domain: Optional["Domain"] = None,
    ) -> List[Dict[str, Any]]:
        """准备Flow数据用于模板渲染。
        
        Args:
            flows: Flow列表
            domain: Domain定义
            
        Returns:
            Flow数据列表
        """
        flows_data = []
        for flow in flows:
            flow_data = {
                "id": flow.id,
                "description": flow.description or flow.name or flow.id,
                "slots_to_collect": flow.get_slots_to_collect(),
            }
            flows_data.append(flow_data)
        return flows_data
    
    def _prepare_slot_info(
        self,
        domain: Optional["Domain"] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """准备槽位信息用于模板渲染。
        
        Args:
            domain: Domain定义
            
        Returns:
            槽位信息字典
        """
        slot_info = {}
        if domain and domain.slots:
            for name, slot in domain.slots.items():
                info = {
                    "type": getattr(slot, "slot_type", "text"),
                    "description": getattr(slot, "description", ""),
                }
                # 获取允许的值（如果是分类槽位）
                allowed_values = getattr(slot, "allowed_values", None)
                if allowed_values:
                    info["allowed_values"] = allowed_values
                slot_info[name] = info
        return slot_info
    
    def _get_current_slot(
        self,
        tracker: "DialogueStateTracker",
    ) -> Optional[str]:
        """获取当前正在收集的槽位。
        
        Args:
            tracker: 对话状态追踪器
            
        Returns:
            槽位名，如果没有则返回None
        """
        # 从对话栈获取当前需要收集的槽位
        top_frame = tracker.dialogue_stack.top()
        if top_frame:
            from atguigu_ai.dialogue_understanding.stack.stack_frame import FlowStackFrame
            if isinstance(top_frame, FlowStackFrame):
                return getattr(top_frame, "slot_to_collect", None)
        return None
    
    def _get_current_slot_description(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
    ) -> Optional[str]:
        """获取当前槽位的描述。
        
        Args:
            tracker: 对话状态追踪器
            domain: Domain定义
            
        Returns:
            槽位描述
        """
        current_slot = self._get_current_slot(tracker)
        if current_slot and domain and domain.slots:
            slot = domain.slots.get(current_slot)
            if slot:
                return getattr(slot, "description", None)
        return None
    
    def _get_flow_slots(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        flows: Optional["FlowsList"] = None,
    ) -> List[Dict[str, Any]]:
        """获取当前Flow的槽位状态。
        
        Args:
            tracker: 对话状态追踪器
            domain: Domain定义
            flows: Flow列表
            
        Returns:
            槽位状态列表
        """
        flow_slots = []
        active_flow = tracker.active_flow
        
        if not active_flow or not flows:
            return flow_slots
        
        # 获取当前Flow（兼容FlowsList和普通列表）
        flow = None
        if hasattr(flows, 'get_flow'):
            flow = flows.get_flow(active_flow)
        elif isinstance(flows, list):
            for f in flows:
                if f.id == active_flow:
                    flow = f
                    break
        
        if not flow:
            return flow_slots
        
        # 获取Flow需要收集的槽位
        slots_to_collect = flow.get_slots_to_collect()
        
        for slot_name in slots_to_collect:
            slot_data = {
                "name": slot_name,
                "value": tracker.get_slot(slot_name),
            }
            
            # 从domain获取槽位元信息
            if domain and domain.slots:
                slot_def = domain.slots.get(slot_name)
                if slot_def:
                    slot_data["type"] = getattr(slot_def, "slot_type", "text")
                    slot_data["description"] = getattr(slot_def, "description", "")
            
            flow_slots.append(slot_data)
        
        return flow_slots
    
    def _get_conversation_history(
        self,
        tracker: "DialogueStateTracker",
    ) -> List[Dict[str, str]]:
        """获取对话历史。
        
        Args:
            tracker: 对话状态追踪器
            
        Returns:
            对话历史列表
        """
        history = tracker.get_messages_for_llm(max_turns=self.max_history_turns)
        
        formatted_history = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                formatted_history.append({"role": "用户", "content": content})
            elif role == "assistant":
                formatted_history.append({"role": "助手", "content": content})
        
        return formatted_history


# 导出
__all__ = [
    "PromptBuilder",
    "TEMPLATES_DIR",
]
