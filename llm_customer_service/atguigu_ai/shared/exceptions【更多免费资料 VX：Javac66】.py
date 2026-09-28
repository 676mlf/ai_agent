# -*- coding: utf-8 -*-
"""
exceptions - 异常类定义

定义系统中使用的所有自定义异常类，形成统一的异常层次结构。
所有异常都继承自AtguiguException基类。
"""

from typing import Optional, Text


class AtguiguException(Exception):
    """atguigu_ai系统的基础异常类
    
    所有自定义异常都应继承此类，便于统一捕获和处理。
    
    属性：
        message: 异常描述信息
    """
    
    def __init__(self, message: Optional[Text] = None) -> None:
        """初始化异常
        
        参数：
            message: 异常描述信息
        """
        self.message = message or self.__class__.__name__
        super().__init__(self.message)
    
    def __str__(self) -> str:
        return self.message


# =============================================================================
# 配置相关异常
# =============================================================================

class ConfigurationException(AtguiguException):
    """配置异常
    
    当配置文件格式错误、缺少必要配置项或配置值无效时抛出。
    """
    pass


class InvalidConfigException(ConfigurationException):
    """无效配置异常
    
    当配置值不符合预期格式或范围时抛出。
    """
    pass


class MissingConfigException(ConfigurationException):
    """缺少配置异常
    
    当必需的配置项未提供时抛出。
    """
    pass


# =============================================================================
# 模型相关异常
# =============================================================================

class ModelNotFound(AtguiguException):
    """模型未找到异常
    
    当指定路径没有找到模型文件时抛出。
    """
    pass


class ModelLoadException(AtguiguException):
    """模型加载异常
    
    当模型文件损坏或格式不正确导致加载失败时抛出。
    """
    pass


class ModelTrainingException(AtguiguException):
    """模型训练异常
    
    当训练过程中出现错误时抛出。
    """
    pass


# =============================================================================
# LLM相关异常
# =============================================================================

class LLMException(AtguiguException):
    """LLM异常基类
    
    所有LLM相关异常的基类。
    """
    pass


class LLMConnectionError(LLMException):
    """LLM连接错误
    
    当无法连接到LLM API服务时抛出。
    """
    pass


class LLMTimeoutError(LLMException):
    """LLM超时错误
    
    当LLM API请求超时时抛出。
    """
    pass


class LLMResponseError(LLMException):
    """LLM响应错误
    
    当LLM返回无效或无法解析的响应时抛出。
    """
    pass


class LLMAuthenticationError(LLMException):
    """LLM认证错误
    
    当API密钥无效或认证失败时抛出。
    """
    pass


class LLMRateLimitError(LLMException):
    """LLM速率限制错误
    
    当超过API调用速率限制时抛出。
    """
    pass


# =============================================================================
# Tracker Store相关异常
# =============================================================================

class TrackerStoreException(AtguiguException):
    """Tracker存储异常基类
    
    所有Tracker存储相关异常的基类。
    """
    pass


class TrackerStoreConnectionError(TrackerStoreException):
    """Tracker存储连接错误
    
    当无法连接到存储后端时抛出。
    """
    pass


class TrackerSerializationError(TrackerStoreException):
    """Tracker序列化错误
    
    当Tracker对象无法正确序列化或反序列化时抛出。
    """
    pass


# =============================================================================
# 对话理解相关异常
# =============================================================================

class DialogueUnderstandingException(AtguiguException):
    """对话理解异常基类
    
    所有对话理解模块相关异常的基类。
    """
    pass


class CommandParseError(DialogueUnderstandingException):
    """命令解析错误
    
    当LLM输出无法解析为有效命令时抛出。
    """
    pass


class FlowNotFoundError(DialogueUnderstandingException):
    """Flow未找到错误
    
    当尝试启动不存在的Flow时抛出。
    """
    pass


class FlowExecutionError(DialogueUnderstandingException):
    """Flow执行错误
    
    当Flow执行过程中出现错误时抛出。
    """
    pass


class InvalidSlotValueError(DialogueUnderstandingException):
    """无效槽位值错误
    
    当槽位值不符合定义的类型或约束时抛出。
    """
    pass


# =============================================================================
# Action相关异常
# =============================================================================

class ActionException(AtguiguException):
    """Action异常基类
    
    所有Action相关异常的基类。
    """
    pass


class ActionNotFoundError(ActionException):
    """Action未找到错误
    
    当尝试执行不存在的Action时抛出。
    """
    pass


class ActionExecutionError(ActionException):
    """Action执行错误
    
    当Action执行过程中出现错误时抛出。
    """
    pass


class ActionServerConnectionError(ActionException):
    """Action服务器连接错误
    
    当无法连接到远程Action服务器时抛出。
    """
    pass


# =============================================================================
# 通道相关异常
# =============================================================================

class ChannelException(AtguiguException):
    """通道异常基类
    
    所有通道相关异常的基类。
    """
    pass


class ChannelNotFoundError(ChannelException):
    """通道未找到错误
    
    当尝试使用不存在的通道时抛出。
    """
    pass


class ChannelConnectionError(ChannelException):
    """通道连接错误
    
    当通道连接失败时抛出。
    """
    pass


# =============================================================================
# 图引擎相关异常
# =============================================================================

class GraphException(AtguiguException):
    """图引擎异常基类
    
    所有图引擎相关异常的基类。
    """
    pass


class GraphValidationError(GraphException):
    """图验证错误
    
    当图结构无效(如存在循环依赖)时抛出。
    """
    pass


class GraphExecutionError(GraphException):
    """图执行错误
    
    当图执行过程中出现错误时抛出。
    """
    pass


# =============================================================================
# 检索相关异常
# =============================================================================

class RetrievalException(AtguiguException):
    """检索异常基类
    
    所有检索相关异常的基类。
    """
    pass


class VectorStoreError(RetrievalException):
    """向量存储错误
    
    当向量存储操作失败时抛出。
    """
    pass


class EmbeddingError(RetrievalException):
    """向量化错误
    
    当文本向量化失败时抛出。
    """
    pass


# =============================================================================
# 数据验证相关异常
# =============================================================================

class ValidationException(AtguiguException):
    """验证异常基类
    
    所有数据验证相关异常的基类。
    """
    pass


class DomainValidationError(ValidationException):
    """Domain验证错误
    
    当Domain定义不符合规范时抛出。
    """
    pass


class FlowValidationError(ValidationException):
    """Flow验证错误
    
    当Flow定义不符合规范时抛出。
    """
    pass


class DataValidationError(ValidationException):
    """数据验证错误
    
    当训练数据不符合规范时抛出。
    """
    pass
