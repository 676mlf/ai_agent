# -*- coding: utf-8 -*-
"""
命令解析器

负责解析LLM输出的文本，将其转换为命令对象。
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

from atguigu_ai.dialogue_understanding.commands.base import (
    Command,
    get_all_command_classes,
    parse_command_from_text,
)
from atguigu_ai.dialogue_understanding.commands.error_commands import ParseErrorCommand

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """解析结果。
    
    Attributes:
        commands: 成功解析的命令列表
        errors: 解析错误列表
        raw_lines: 原始输入行
    """
    commands: List[Command] = field(default_factory=list)
    errors: List[Tuple[str, str]] = field(default_factory=list)  # (raw_text, error_message)
    raw_lines: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """是否成功解析了至少一个命令。"""
        return len(self.commands) > 0
    
    @property
    def has_errors(self) -> bool:
        """是否有解析错误。"""
        return len(self.errors) > 0


class CommandParser:
    """命令解析器。
    
    负责将LLM输出的文本解析为命令对象列表。
    
    支持的命令格式：
    1. DSL格式: start flow booking, set slot name "John"
    2. 函数格式: StartFlow(booking), SetSlot(name, "John")
    """
    
    def __init__(self):
        """初始化解析器。"""
        self._command_classes = get_all_command_classes()
    
    def parse(self, text: str) -> ParseResult:
        """解析文本中的命令。
        
        Args:
            text: LLM输出的文本
            
        Returns:
            解析结果
        """
        result = ParseResult()
        
        # 清理文本
        text = self._clean_text(text)
        
        # 按行分割
        lines = self._split_lines(text)
        result.raw_lines = lines
        
        # 逐行解析
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 跳过注释行
            if line.startswith('#') or line.startswith('//'):
                continue
            
            try:
                command = self._parse_line(line)
                if command:
                    result.commands.append(command)
                else:
                    # 无法识别的行
                    result.errors.append((line, "Unrecognized command format"))
            except Exception as e:
                result.errors.append((line, str(e)))
                logger.debug(f"Failed to parse line: {line}, error: {e}")
        
        return result
    
    def parse_single(self, text: str) -> Optional[Command]:
        """解析单个命令。
        
        Args:
            text: 命令文本
            
        Returns:
            命令对象，如果解析失败则返回None
        """
        result = self.parse(text)
        return result.commands[0] if result.commands else None
    
    def _clean_text(self, text: str) -> str:
        """清理文本。
        
        移除markdown代码块标记等。
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        # 移除markdown代码块
        text = re.sub(r'```[\w]*\n?', '', text)
        
        # 移除内联代码的反引号（如 `cannot_handle` -> cannot_handle）
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # 移除开头的编号（如 "1. ", "- "）
        lines = []
        for line in text.split('\n'):
            # 移除列表标记
            line = re.sub(r'^\s*[-*•]\s*', '', line)
            # 移除编号
            line = re.sub(r'^\s*\d+\.\s*', '', line)
            lines.append(line)
        
        return '\n'.join(lines)
    
    def _split_lines(self, text: str) -> List[str]:
        """分割文本行。
        
        Args:
            text: 文本
            
        Returns:
            行列表
        """
        # 按换行符分割
        lines = text.split('\n')
        
        # 也可能用分号分隔
        result = []
        for line in lines:
            if ';' in line:
                result.extend(line.split(';'))
            else:
                result.append(line)
        
        return [l.strip() for l in result if l.strip()]
    
    def _parse_line(self, line: str) -> Optional[Command]:
        """解析单行命令。
        
        Args:
            line: 命令行文本
            
        Returns:
            命令对象，如果无法解析则返回None
        """
        line = line.strip()
        
        # 尝试使用注册的命令类解析
        command = parse_command_from_text(line)
        if command:
            return command
        
        # 尝试逐个命令类的模式匹配
        for command_name, command_cls in self._command_classes.items():
            try:
                pattern = command_cls.regex_pattern()
                if pattern:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        return command_cls._from_regex_match(match)
            except (NotImplementedError, AttributeError):
                continue
        
        # 无法解析
        return None
    
    def validate_commands(self, commands: List[Command]) -> List[Command]:
        """验证命令列表。
        
        过滤掉无效的命令。
        
        Args:
            commands: 命令列表
            
        Returns:
            有效的命令列表
        """
        valid_commands = []
        for cmd in commands:
            if self._is_valid_command(cmd):
                valid_commands.append(cmd)
        return valid_commands
    
    def _is_valid_command(self, command: Command) -> bool:
        """检查命令是否有效。
        
        Args:
            command: 命令对象
            
        Returns:
            是否有效
        """
        # 基本验证：命令不是None
        if command is None:
            return False
        
        # 检查必要字段
        try:
            # StartFlowCommand 需要 flow 字段
            if hasattr(command, 'flow') and not command.flow:
                return False
            # SetSlotCommand 需要 name 字段
            if hasattr(command, 'name') and not command.name:
                return False
        except Exception:
            return False
        
        return True


# 创建默认解析器实例
default_parser = CommandParser()


def parse_commands(text: str) -> List[Command]:
    """便捷函数：解析命令。
    
    Args:
        text: LLM输出的文本
        
    Returns:
        命令列表
    """
    result = default_parser.parse(text)
    return result.commands


def parse_single_command(text: str) -> Optional[Command]:
    """便捷函数：解析单个命令。
    
    Args:
        text: 命令文本
        
    Returns:
        命令对象
    """
    return default_parser.parse_single(text)


# 导出
__all__ = [
    "CommandParser",
    "ParseResult",
    "parse_commands",
    "parse_single_command",
    "default_parser",
]
