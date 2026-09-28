# -*- coding: utf-8 -*-
"""
slots - 槽位系统

定义对话系统中的槽位类型，用于存储对话过程中收集的信息。
槽位是对话状态的核心组成部分，支持多种数据类型。

槽位映射类型：
- from_llm: 由LLM从用户输入中提取并填充
- controlled: 由Action填充，不由LLM自动提取
"""

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Text, Type, Union

from atguigu_ai.shared.constants import (
    SLOT_TYPE_ANY,
    SLOT_TYPE_BOOL,
    SLOT_TYPE_CATEGORICAL,
    SLOT_TYPE_FLOAT,
    SLOT_TYPE_LIST,
    SLOT_TYPE_TEXT,
)
from atguigu_ai.shared.exceptions import InvalidSlotValueError


class SlotMappingType(str, Enum):
    """槽位映射类型。
    
    定义槽位如何被填充：
    - FROM_LLM: LLM从用户输入中提取值填充
    - CONTROLLED: Action代码控制填充，LLM不自动提取
    """
    FROM_LLM = "from_llm"
    CONTROLLED = "controlled"


class Slot(ABC):
    """槽位基类
    
    所有槽位类型的抽象基类，定义槽位的基本属性和行为。
    
    属性：
        name: 槽位名称
        value: 当前值
        initial_value: 初始值
        influence_conversation: 是否影响对话流程
        mappings: 槽位映射规则
        mapping_type: 映射类型 (from_llm/controlled)
        description: 槽位描述（供LLM理解用途）
    """
    
    type_name: str = "any"
    
    def __init__(
        self,
        name: Text,
        initial_value: Any = None,
        influence_conversation: bool = True,
        mappings: Optional[List[Dict[str, Any]]] = None,
        mapping_type: Union[SlotMappingType, str] = SlotMappingType.FROM_LLM,
        description: Optional[str] = None,
    ) -> None:
        """初始化槽位
        
        参数：
            name: 槽位名称
            initial_value: 初始值
            influence_conversation: 是否影响对话流程
            mappings: 槽位映射规则
            mapping_type: 映射类型 (from_llm/controlled)
            description: 槽位描述
        """
        self.name = name
        self.initial_value = initial_value
        self._value = initial_value
        self.influence_conversation = influence_conversation
        self.mappings = mappings or []
        
        # 槽位映射类型支持
        if isinstance(mapping_type, str):
            mapping_type = SlotMappingType(mapping_type)
        self.mapping_type = mapping_type
        self.description = description
    
    @property
    def value(self) -> Any:
        """获取槽位当前值"""
        return self._value
    
    @value.setter
    def value(self, new_value: Any) -> None:
        """设置槽位值
        
        会进行类型验证，无效值将抛出异常。
        """
        if new_value is not None and not self._validate_value(new_value):
            raise InvalidSlotValueError(
                f"槽位 '{self.name}' 的值 '{new_value}' 类型无效，"
                f"期望类型: {self.type_name}"
            )
        self._value = new_value
    
    def _validate_value(self, value: Any) -> bool:
        """验证槽位值是否有效
        
        子类应重写此方法实现具体的验证逻辑。
        
        参数：
            value: 待验证的值
            
        返回：
            有效返回True，否则返回False
        """
        return True
    
    def reset(self) -> None:
        """重置槽位为初始值"""
        self._value = self.initial_value
    
    def is_set(self) -> bool:
        """检查槽位是否已设置(非空)"""
        return self._value is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = {
            "name": self.name,
            "type": self.type_name,
            "value": self._value,
            "initial_value": self.initial_value,
            "influence_conversation": self.influence_conversation,
            "mapping_type": self.mapping_type.value,
        }
        if self.description:
            data["description"] = self.description
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Slot":
        """从字典创建槽位实例
        
        工厂方法，根据type字段创建对应类型的槽位。
        """
        slot_type = data.get("type", SLOT_TYPE_ANY)
        slot_class = SLOT_TYPE_MAP.get(slot_type, AnySlot)
        
        slot = slot_class(
            name=data["name"],
            initial_value=data.get("initial_value"),
            influence_conversation=data.get("influence_conversation", True),
            mappings=data.get("mappings"),
            mapping_type=data.get("mapping_type", SlotMappingType.FROM_LLM),
            description=data.get("description"),
        )
        
        # 设置当前值
        if "value" in data:
            slot._value = data["value"]
        
        return slot
    
    def is_from_llm(self) -> bool:
        """检查槽位是否由LLM填充"""
        return self.mapping_type == SlotMappingType.FROM_LLM
    
    def is_controlled(self) -> bool:
        """检查槽位是否由Action控制填充"""
        return self.mapping_type == SlotMappingType.CONTROLLED
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, value={self._value})"


class TextSlot(Slot):
    """文本槽位
    
    存储字符串类型的值。
    """
    
    type_name = SLOT_TYPE_TEXT
    
    def _validate_value(self, value: Any) -> bool:
        """验证是否为字符串"""
        return isinstance(value, str)


class BoolSlot(Slot):
    """布尔槽位
    
    存储布尔类型的值。
    """
    
    type_name = SLOT_TYPE_BOOL
    
    def _validate_value(self, value: Any) -> bool:
        """验证是否为布尔值"""
        return isinstance(value, bool)


class FloatSlot(Slot):
    """浮点槽位
    
    存储数值类型的值，支持设置取值范围。
    
    属性：
        min_value: 最小值
        max_value: 最大值
    """
    
    type_name = SLOT_TYPE_FLOAT
    
    def __init__(
        self,
        name: Text,
        initial_value: Any = None,
        influence_conversation: bool = True,
        mappings: Optional[List[Dict[str, Any]]] = None,
        mapping_type: Union[SlotMappingType, str] = SlotMappingType.FROM_LLM,
        description: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> None:
        super().__init__(name, initial_value, influence_conversation, mappings, mapping_type, description)
        self.min_value = min_value
        self.max_value = max_value
    
    def _validate_value(self, value: Any) -> bool:
        """验证是否为有效数值"""
        if not isinstance(value, (int, float)):
            return False
        
        if self.min_value is not None and value < self.min_value:
            return False
        
        if self.max_value is not None and value > self.max_value:
            return False
        
        return True


class ListSlot(Slot):
    """列表槽位
    
    存储列表类型的值。
    """
    
    type_name = SLOT_TYPE_LIST
    
    def __init__(
        self,
        name: Text,
        initial_value: Any = None,
        influence_conversation: bool = True,
        mappings: Optional[List[Dict[str, Any]]] = None,
        mapping_type: Union[SlotMappingType, str] = SlotMappingType.FROM_LLM,
        description: Optional[str] = None,
    ) -> None:
        # 确保初始值为列表
        if initial_value is None:
            initial_value = []
        super().__init__(name, initial_value, influence_conversation, mappings, mapping_type, description)
    
    def _validate_value(self, value: Any) -> bool:
        """验证是否为列表"""
        return isinstance(value, list)
    
    def append(self, item: Any) -> None:
        """向列表添加元素"""
        if self._value is None:
            self._value = []
        self._value.append(item)


class CategoricalSlot(Slot):
    """分类槽位
    
    只允许设置预定义的值列表中的值。
    
    属性：
        values: 允许的值列表
    """
    
    type_name = SLOT_TYPE_CATEGORICAL
    
    def __init__(
        self,
        name: Text,
        initial_value: Any = None,
        influence_conversation: bool = True,
        mappings: Optional[List[Dict[str, Any]]] = None,
        mapping_type: Union[SlotMappingType, str] = SlotMappingType.FROM_LLM,
        description: Optional[str] = None,
        values: Optional[List[Any]] = None,
    ) -> None:
        super().__init__(name, initial_value, influence_conversation, mappings, mapping_type, description)
        self.values = values or []
    
    def _validate_value(self, value: Any) -> bool:
        """验证值是否在允许列表中"""
        if not self.values:
            return True
        return value in self.values


class AnySlot(Slot):
    """任意类型槽位
    
    接受任何类型的值，不进行类型验证。
    """
    
    type_name = SLOT_TYPE_ANY
    
    def _validate_value(self, value: Any) -> bool:
        """接受任何值"""
        return True


# 槽位类型映射
SLOT_TYPE_MAP: Dict[str, Type[Slot]] = {
    SLOT_TYPE_TEXT: TextSlot,
    SLOT_TYPE_BOOL: BoolSlot,
    SLOT_TYPE_FLOAT: FloatSlot,
    SLOT_TYPE_LIST: ListSlot,
    SLOT_TYPE_CATEGORICAL: CategoricalSlot,
    SLOT_TYPE_ANY: AnySlot,
}


def create_slot(
    name: Text,
    slot_type: Text = SLOT_TYPE_ANY,
    mapping_type: Union[SlotMappingType, str] = SlotMappingType.FROM_LLM,
    description: Optional[str] = None,
    **kwargs: Any,
) -> Slot:
    """创建槽位实例
    
    工厂函数，根据类型名创建对应的槽位对象。
    
    参数：
        name: 槽位名称
        slot_type: 槽位数据类型 (text/bool/float/list/categorical/any)
        mapping_type: 映射类型 (from_llm/controlled)
        description: 槽位描述（供LLM理解用途）
        **kwargs: 额外的槽位配置
        
    返回：
        Slot实例
        
    示例：
        >>> # 创建由LLM填充的文本槽位
        >>> slot = create_slot("food_type", "text", mapping_type="from_llm")
        >>> slot.value = "pizza"
        
        >>> # 创建由Action控制的槽位
        >>> slot = create_slot("order_id", "text", mapping_type="controlled")
    """
    slot_class = SLOT_TYPE_MAP.get(slot_type, AnySlot)
    return slot_class(
        name=name, 
        mapping_type=mapping_type, 
        description=description,
        **kwargs
    )


# 导出
__all__ = [
    "Slot",
    "TextSlot",
    "BoolSlot",
    "FloatSlot",
    "ListSlot",
    "CategoricalSlot",
    "AnySlot",
    "SlotMappingType",
    "SLOT_TYPE_MAP",
    "create_slot",
]
