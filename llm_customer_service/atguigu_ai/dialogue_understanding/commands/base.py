# -*- coding: utf-8 -*-
"""
命令基类

定义所有命令的抽象基类，提供统一的接口规范。

## Command 与 Action 的职责边界

在本架构中，Command 和 Action 有明确的职责分工：

### Command（命令）
- **来源**：由 LLM 根据用户输入生成
- **职责**：解析用户意图，更新对话状态
- **执行方式**：通过 CommandProcessor 执行
- **输出**：直接修改 Tracker 状态（如压入栈帧、设置槽位）

### Action（动作）
- **来源**：由 Policy 根据栈帧状态选择
- **职责**：执行具体操作，压入栈帧
- **执行方式**：通过 Agent 执行
- **输出**：返回 ActionResult（事件、响应）

### 处理流程
1. 用户输入 → LLM 生成 Command
2. CommandProcessor 执行 Command → 更新 Tracker（可能压入栈帧）
3. Policy 检测栈帧 → 选择 Action
4. Agent 执行 Action → 生成响应

### 栈帧化 Action
部分 Action 采用"栈帧化"设计：
- Action 只压入栈帧（如 SearchStackFrame）
- Policy 检测栈帧并执行实际操作（如检索）
- 这种设计实现了 Action 与响应生成的解耦

### 示例
```
用户: "帮我查一下订单"
↓
LLM 生成: KnowledgeAnswerCommand
↓
CommandProcessor: 确定 next_action = action_trigger_search
↓
ActionTriggerSearch: 压入 SearchStackFrame
↓
EnterpriseSearchPolicy: 检测到 SearchStackFrame，执行检索，生成响应
```
"""

from __future__ import annotations

import re
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker


# 命令注册表
_COMMAND_REGISTRY: Dict[str, Type["Command"]] = {}


def register_command(cls: Type["Command"]) -> Type["Command"]:
    """命令注册装饰器。
    
    用于将命令类注册到全局注册表中，便于动态创建命令。
    
    Args:
        cls: 要注册的命令类
        
    Returns:
        注册后的命令类
    """
    _COMMAND_REGISTRY[cls.command_name()] = cls
    return cls


def get_command_class(name: str) -> Optional[Type["Command"]]:
    """根据命令名获取命令类。
    
    Args:
        name: 命令名称
        
    Returns:
        命令类，如果不存在则返回None
    """
    return _COMMAND_REGISTRY.get(name)


def get_all_command_classes() -> Dict[str, Type["Command"]]:
    """获取所有已注册的命令类。
    
    Returns:
        命令名到命令类的映射
    """
    return _COMMAND_REGISTRY.copy()


@dataclass
class Command(ABC):
    """命令基类。
    
    命令是本架构中的核心概念，表示对话系统可以执行的原子操作。
    所有具体的命令类型都应继承此基类。
    
    命令的生命周期：
    1. LLM生成器根据用户输入生成命令文本
    2. 命令解析器将文本解析为命令对象
    3. 命令处理器执行命令并更新对话状态
    """
    
    @classmethod
    @abstractmethod
    def command_name(cls) -> str:
        """返回命令名称。
        
        用于识别命令类型，应该是唯一的。
        例如: "start_flow", "set_slot", "cancel_flow"
        
        Returns:
            命令名称字符串
        """
        raise NotImplementedError()
    
    @classmethod
    def command_type(cls) -> str:
        """返回命令类型（别名）。
        
        Returns:
            命令类型字符串
        """
        return cls.command_name()
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Command":
        """从字典创建命令实例。
        
        Args:
            data: 包含命令参数的字典
            
        Returns:
            命令实例
            
        Raises:
            ValueError: 如果参数缺失或无效
        """
        raise NotImplementedError()
    
    def as_dict(self) -> Dict[str, Any]:
        """将命令转换为字典。
        
        Returns:
            包含命令参数的字典
        """
        data = dataclasses.asdict(self)
        data["command"] = self.command_name()
        return data
    
    @abstractmethod
    def run(
        self,
        tracker: "DialogueStateTracker",
        flows: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行命令。
        
        在对话状态追踪器上执行此命令，并返回产生的事件。
        
        Args:
            tracker: 对话状态追踪器
            flows: 可用的Flow列表
            
        Returns:
            执行命令产生的事件列表
        """
        raise NotImplementedError()
    
    @classmethod
    def from_dsl(cls, text: str) -> Optional["Command"]:
        """从DSL文本解析命令。
        
        Args:
            text: DSL格式的命令文本
            
        Returns:
            命令实例，如果解析失败则返回None
        """
        pattern = cls.regex_pattern()
        if not pattern:
            return None
        
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            return cls._from_regex_match(match)
        return None
    
    @classmethod
    def _from_regex_match(cls, match: re.Match) -> Optional["Command"]:
        """从正则匹配结果创建命令。
        
        子类应该覆盖此方法以实现具体的解析逻辑。
        
        Args:
            match: 正则匹配结果
            
        Returns:
            命令实例
        """
        raise NotImplementedError()
    
    def to_dsl(self) -> str:
        """将命令转换为DSL文本。
        
        Returns:
            DSL格式的命令字符串
        """
        return f"{self.command_name()}"
    
    @classmethod
    def regex_pattern(cls) -> Optional[str]:
        """返回用于解析此命令的正则表达式模式。
        
        Returns:
            正则表达式字符串，如果不支持DSL解析则返回None
        """
        return None
    
    def __hash__(self) -> int:
        """计算命令的哈希值。"""
        return hash(self.command_name())
    
    def __eq__(self, other: object) -> bool:
        """判断两个命令是否相等。"""
        if not isinstance(other, Command):
            return False
        return self.as_dict() == other.as_dict()
    
    def __repr__(self) -> str:
        """返回命令的字符串表示。"""
        return f"{self.__class__.__name__}({self.as_dict()})"


@staticmethod
def command_from_dict(data: Dict[str, Any]) -> Command:
    """从字典创建命令对象。
    
    根据字典中的command字段确定命令类型，然后创建对应的命令对象。
    
    Args:
        data: 包含命令信息的字典
        
    Returns:
        命令对象
        
    Raises:
        ValueError: 如果命令类型未知
    """
    command_name = data.get("command")
    if not command_name:
        raise ValueError("Missing 'command' field in data")
    
    command_cls = get_command_class(command_name)
    if command_cls is None:
        raise ValueError(f"Unknown command type: {command_name}")
    
    return command_cls.from_dict(data)


def parse_command_from_text(text: str) -> Optional[Command]:
    """从文本解析命令。
    
    尝试使用所有已注册的命令类的正则模式解析文本。
    
    Args:
        text: 待解析的文本
        
    Returns:
        命令对象，如果解析失败则返回None
    """
    text = text.strip()
    for command_cls in _COMMAND_REGISTRY.values():
        try:
            command = command_cls.from_dsl(text)
            if command is not None:
                return command
        except (NotImplementedError, ValueError):
            continue
    return None


# 导出
__all__ = [
    "Command",
    "register_command",
    "get_command_class",
    "get_all_command_classes",
    "command_from_dict",
    "parse_command_from_text",
]
