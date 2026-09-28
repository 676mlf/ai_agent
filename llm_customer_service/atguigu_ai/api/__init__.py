# -*- coding: utf-8 -*-
"""
atguigu_ai API模块

提供基于FastAPI的Web服务接口。
"""

from atguigu_ai.api.server import AtguiguServer, create_app

__all__ = [
    "AtguiguServer",
    "create_app",
]
