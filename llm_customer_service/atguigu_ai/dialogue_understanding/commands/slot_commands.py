# -*- coding: utf-8 -*-
"""
槽位相关命令

包含槽位设置等命令。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from atguigu_ai.dialogue_understanding.commands.base import Command, register_command

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker


def clean_extracted_value(value: str) -> Any:
    """清理从DSL中提取的值。
    
    处理引号、空格，以及特殊值（null, true, false, 数字）。
    
    Args:
        value: 原始字符串值
        
    Returns:
        清理后的值
    """
    if value is None:
        return None
    
    # 去除首尾空格
    value = value.strip()
    
    # 去除引号
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    
    # 处理特殊值
    value_lower = value.lower()
    if value_lower == "null" or value_lower == "none":
        return None
    if value_lower == "true":
        return True
    if value_lower == "false":
        return False
    
    # 尝试转换为数字
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    return value


@register_command
@dataclass
class SetSlotCommand(Command):
    """设置槽位命令。
    
    用于设置对话状态中的槽位值。LLM识别到用户提供的信息后，
    会生成此命令来记录信息。
    
    Attributes:
        name: 槽位名称
        value: 槽位值
        extractor: 提取器类型（llm, nlu, form等）
    """
    
    name: str
    value: Any
    extractor: str = "llm"
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "set_slot"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SetSlotCommand":
        """从字典创建命令。
        
        Args:
            data: 包含name和value字段的字典
            
        Returns:
            SetSlotCommand实例
            
        Raises:
            ValueError: 如果缺少必要字段
        """
        try:
            return SetSlotCommand(
                name=data["name"],
                value=data["value"],
                extractor=data.get("extractor", "llm"),
            )
        except KeyError as e:
            raise ValueError(f"Missing required field for SetSlotCommand: {e}") from e
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行设置槽位命令。
        
        在tracker上设置指定槽位的值。
        
        Args:
            tracker: 对话状态追踪器
            flows: 可用的Flow列表（未使用）
            
        Returns:
            产生的事件列表
        """
        # 获取槽位对象以检查类型
        slot_obj = tracker.slots.get(self.name)
        value_to_set = self.value
        
        # 如果槽位是 text 类型，将值转换为字符串
        if slot_obj and hasattr(slot_obj, 'type_name'):
            if slot_obj.type_name == "text" and value_to_set is not None:
                value_to_set = str(value_to_set)
        
        # 设置槽位值
        try:
            tracker.set_slot(self.name, value_to_set)
            return [{
                "event": "slot_set",
                "name": self.name,
                "value": value_to_set,
                "extractor": self.extractor,
                "timestamp": None,
            }]
        except Exception as e:
            # 槽位不存在或设置失败
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"SetSlotCommand failed to set slot '{self.name}': {e}")
            return []
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        # 根据值类型决定是否添加引号
        if isinstance(self.value, str):
            return f'set slot {self.name} "{self.value}"'
        elif self.value is None:
            return f"set slot {self.name} null"
        else:
            return f"set slot {self.name} {self.value}"
    
    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。"""
        # 支持多种格式:
        # - set slot <name> <value>
        # - set slot <name> "<value>"
        # - SetSlot(<name>, <value>)
        return r"""(?:set\s+slot\s+['"`]?([a-zA-Z_][a-zA-Z0-9_-]*)['"`]?\s+['"`]?(.+?)['"`]?$|SetSlot\(['"]?([a-zA-Z_][a-zA-Z0-9_-]*)['"]?,\s*['"]?(.*)['"]?\))"""
    
    @classmethod
    def _from_regex_match(cls, match: re.Match) -> Optional["SetSlotCommand"]:
        """从正则匹配创建命令。"""
        # 第一组是 set slot 格式，第三组是 SetSlot() 格式
        name = match.group(1) or match.group(3)
        value = match.group(2) or match.group(4)
        
        if name:
            return SetSlotCommand(
                name=name.strip(),
                value=clean_extracted_value(value) if value else None,
            )
        return None
    
    def __hash__(self) -> int:
        return hash((self.name, str(self.value)))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SetSlotCommand):
            return False
        return other.name == self.name and str(other.value).lower() == str(self.value).lower()


@register_command
@dataclass
class ResetSlotCommand(Command):
    """重置槽位命令。
    
    用于将槽位重置为初始值。
    
    Attributes:
        name: 要重置的槽位名称
    """
    
    name: str
    
    @classmethod
    def command_name(cls) -> str:
        """返回命令名称。"""
        return "reset_slot"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResetSlotCommand":
        """从字典创建命令。"""
        try:
            return ResetSlotCommand(name=data["name"])
        except KeyError as e:
            raise ValueError(f"Missing required field 'name' for ResetSlotCommand") from e
    
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行重置槽位命令。"""
        slot = tracker.slots.get(self.name)
        if slot:
            slot.reset()
            return [{
                "event": "slot_reset",
                "name": self.name,
                "timestamp": None,
            }]
        return []
    
    def to_dsl(self) -> str:
        """转换为DSL字符串。"""
        return f"reset slot {self.name}"
    
    @classmethod
    def regex_pattern(cls) -> str:
        """返回正则表达式模式。"""
        return r"""reset\s+slot\s+['"`]?([a-zA-Z_][a-zA-Z0-9_-]*)['"`]?"""
    
    @classmethod
    def _from_regex_match(cls, match: re.Match) -> Optional["ResetSlotCommand"]:
        """从正则匹配创建命令。"""
        name = match.group(1)
        if name:
            return ResetSlotCommand(name=name.strip())
        return None


# 导出
__all__ = [
    "SetSlotCommand",
    "ResetSlotCommand",
    "clean_extracted_value",
]
