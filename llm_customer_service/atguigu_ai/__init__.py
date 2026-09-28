# -*- coding: utf-8 -*-
"""
atguigu_ai - 教学版对话系统

基于LLM驱动的对话架构的精简教学版对话系统。
本项目适合教学和理解对话系统核心原理。

核心模块：
- agent: 对话代理，基于LangGraph实现图式消息处理流程
- dialogue_understanding: 对话理解模块(DU)，包含命令生成、处理、Flow执行
- core: 核心对话管理，包含Tracker、Domain、Slot、Store
- policies: 对话策略，包含FlowPolicy、EnterpriseSearchPolicy
- nlg: 自然语言生成（预留）
- retrieval: 检索增强(FAISS向量检索)（预留）
- training: 训练模块
- api: Web服务(FastAPI)
- channels: 对话通道（REST、SocketIO、Console）
- shared: 共享工具和配置
"""

__version__ = "0.1.0"
__author__ = "atguigu"

from atguigu_ai.shared.constants import (
    DEFAULT_SERVER_PORT,
    DEFAULT_MODELS_PATH,
)

__all__ = [
    "__version__",
    "DEFAULT_SERVER_PORT",
    "DEFAULT_MODELS_PATH",
]
