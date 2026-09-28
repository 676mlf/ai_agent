# -*- coding: utf-8 -*-
"""
domain - 领域定义

定义对话系统的领域(Domain)，包括槽位、动作、响应模板等。
Domain是对话系统配置的核心，描述了系统能够处理的对话范围。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Text, Union

from atguigu_ai.shared.constants import (
    ACTION_DEFAULT_FALLBACK,
    ACTION_LISTEN,
    ACTION_RESTART,
    ACTION_SESSION_START,
    DEFAULT_DOMAIN_PATH,
)
from atguigu_ai.shared.exceptions import DomainValidationError
from atguigu_ai.shared.yaml_loader import read_yaml_file, merge_yaml_files
from atguigu_ai.core.slots import Slot, create_slot


@dataclass
class ResponseTemplate:
    """响应模板
    
    定义Bot的响应内容，支持多种响应变体和条件。
    
    属性：
        text: 响应文本
        buttons: 按钮列表
        image: 图片URL
        custom: 自定义数据
        condition: 条件表达式
        channel: 指定通道
    """
    text: Optional[str] = None
    buttons: List[Dict[str, Any]] = field(default_factory=list)
    image: Optional[str] = None
    custom: Optional[Dict[str, Any]] = None
    condition: Optional[str] = None
    channel: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Union[Dict[str, Any], str]) -> "ResponseTemplate":
        """从字典或字符串创建响应模板"""
        if isinstance(data, str):
            return cls(text=data)
        
        return cls(
            text=data.get("text"),
            buttons=data.get("buttons", []),
            image=data.get("image"),
            custom=data.get("custom"),
            condition=data.get("condition"),
            channel=data.get("channel"),
            metadata=data.get("metadata", {}),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result: Dict[str, Any] = {}
        
        if self.text:
            result["text"] = self.text
        if self.buttons:
            result["buttons"] = self.buttons
        if self.image:
            result["image"] = self.image
        if self.custom:
            result["custom"] = self.custom
        if self.condition:
            result["condition"] = self.condition
        if self.channel:
            result["channel"] = self.channel
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result


@dataclass
class Domain:
    """领域定义类
    
    管理对话系统的领域配置，包括：
    - 槽位定义
    - 动作列表
    - 响应模板
    - Flow列表
    - 表单定义
    
    属性：
        slots: 槽位字典
        actions: 动作名称集合
        responses: 响应模板字典
        flows: Flow名称列表
        forms: 表单定义字典
        session_config: 会话配置
    """
    slots: Dict[str, Slot] = field(default_factory=dict)
    actions: Set[str] = field(default_factory=set)
    responses: Dict[str, List[ResponseTemplate]] = field(default_factory=dict)
    flows: List[str] = field(default_factory=list)
    forms: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    session_config: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    
    def __post_init__(self) -> None:
        """初始化后处理，添加默认动作"""
        self._add_default_actions()
    
    def _add_default_actions(self) -> None:
        """添加系统默认动作"""
        default_actions = {
            ACTION_LISTEN,
            ACTION_RESTART,
            ACTION_SESSION_START,
            ACTION_DEFAULT_FALLBACK,
        }
        self.actions.update(default_actions)
    
    @classmethod
    def load(
        cls,
        domain_path: Union[str, Path] = DEFAULT_DOMAIN_PATH,
    ) -> "Domain":
        """从文件加载Domain
        
        支持单个文件或目录(合并多个domain文件)。
        
        参数：
            domain_path: Domain文件或目录路径
            
        返回：
            Domain实例
            
        异常：
            DomainValidationError: Domain配置无效
        """
        path = Path(domain_path)
        
        if path.is_dir():
            # 目录模式：合并所有YAML文件
            yaml_files = list(path.glob("*.yml")) + list(path.glob("*.yaml"))
            if not yaml_files:
                raise DomainValidationError(f"目录中没有找到YAML文件: {path}")
            domain_dict = merge_yaml_files(yaml_files)
        elif path.exists():
            # 单文件模式
            domain_dict = read_yaml_file(path)
        else:
            raise DomainValidationError(f"Domain文件不存在: {path}")
        
        if not domain_dict:
            domain_dict = {}
        
        return cls.from_dict(domain_dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Domain":
        """从字典创建Domain
        
        参数：
            data: Domain配置字典
            
        返回：
            Domain实例
        """
        # 解析槽位
        slots = {}
        for slot_name, slot_config in data.get("slots", {}).items():
            if isinstance(slot_config, dict):
                # 从 mappings 中提取 mapping_type
                mappings = slot_config.get("mappings", [])
                mapping_type = "from_llm"  # 默认值
                if mappings and isinstance(mappings, list) and len(mappings) > 0:
                    first_mapping = mappings[0]
                    if isinstance(first_mapping, dict):
                        mapping_type = first_mapping.get("type", "from_llm")
                
                # 构建 slot 参数
                slot_kwargs = {
                    "name": slot_name,
                    "slot_type": slot_config.get("type", "any"),
                    "mapping_type": mapping_type,
                    "description": slot_config.get("description"),
                    "initial_value": slot_config.get("initial_value"),
                    "influence_conversation": slot_config.get("influence_conversation", True),
                    "mappings": mappings,
                }
                # 分类槽位的可选值
                if "values" in slot_config:
                    slot_kwargs["values"] = slot_config.get("values")
                
                slots[slot_name] = create_slot(**slot_kwargs)
            else:
                # 简化格式: slot_name: type
                slots[slot_name] = create_slot(name=slot_name, slot_type=str(slot_config))
        
        # 解析动作
        actions = set(data.get("actions", []))
        
        # 解析响应模板
        responses = {}
        for response_name, response_list in data.get("responses", {}).items():
            if isinstance(response_list, list):
                responses[response_name] = [
                    ResponseTemplate.from_dict(r) for r in response_list
                ]
            else:
                responses[response_name] = [ResponseTemplate.from_dict(response_list)]
        
        # 解析其他配置
        flows = data.get("flows", [])
        forms = data.get("forms", {})
        session_config = data.get("session_config", {})
        version = data.get("version", "1.0")
        
        return cls(
            slots=slots,
            actions=actions,
            responses=responses,
            flows=flows,
            forms=forms,
            session_config=session_config,
            version=version,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "version": self.version,
            "slots": {name: slot.to_dict() for name, slot in self.slots.items()},
            "actions": list(self.actions),
            "responses": {
                name: [r.to_dict() for r in templates]
                for name, templates in self.responses.items()
            },
            "flows": self.flows,
            "forms": self.forms,
            "session_config": self.session_config,
        }
    
    def get_slot(self, slot_name: str) -> Optional[Slot]:
        """获取槽位
        
        参数：
            slot_name: 槽位名称
            
        返回：
            Slot实例，不存在返回None
        """
        return self.slots.get(slot_name)
    
    def get_response(self, response_name: str) -> List[ResponseTemplate]:
        """获取响应模板
        
        参数：
            response_name: 响应名称(如utter_greet)
            
        返回：
            响应模板列表，不存在返回空列表
        """
        return self.responses.get(response_name, [])
    
    def has_action(self, action_name: str) -> bool:
        """检查动作是否存在
        
        参数：
            action_name: 动作名称
            
        返回：
            存在返回True
        """
        return action_name in self.actions
    
    def has_flow(self, flow_name: str) -> bool:
        """检查Flow是否存在
        
        参数：
            flow_name: Flow名称
            
        返回：
            存在返回True
        """
        return flow_name in self.flows
    
    def add_slot(self, slot: Slot) -> None:
        """添加槽位"""
        self.slots[slot.name] = slot
    
    def add_action(self, action_name: str) -> None:
        """添加动作"""
        self.actions.add(action_name)
    
    def add_response(
        self,
        response_name: str,
        templates: List[ResponseTemplate],
    ) -> None:
        """添加响应模板"""
        self.responses[response_name] = templates
    
    def merge(self, other: "Domain") -> "Domain":
        """合并另一个Domain
        
        合并策略：
        - 槽位: 后者覆盖前者
        - 动作: 取并集
        - 响应: 后者覆盖前者
        - Flow: 合并列表(去重)
        
        参数：
            other: 要合并的Domain
            
        返回：
            新的合并后的Domain
        """
        merged_slots = {**self.slots, **other.slots}
        merged_actions = self.actions | other.actions
        merged_responses = {**self.responses, **other.responses}
        merged_flows = list(set(self.flows + other.flows))
        merged_forms = {**self.forms, **other.forms}
        
        return Domain(
            slots=merged_slots,
            actions=merged_actions,
            responses=merged_responses,
            flows=merged_flows,
            forms=merged_forms,
            session_config={**self.session_config, **other.session_config},
        )
    
    def __repr__(self) -> str:
        return (
            f"Domain(slots={len(self.slots)}, actions={len(self.actions)}, "
            f"responses={len(self.responses)}, flows={len(self.flows)})"
        )
