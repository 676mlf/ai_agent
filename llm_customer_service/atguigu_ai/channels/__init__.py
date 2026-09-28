# -*- coding: utf-8 -*-
"""
通道模块

提供与用户交互的通道（Channel）实现。
"""

from atguigu_ai.channels.base_channel import (
    InputChannel,
    OutputChannel,
    UserMessage,
    CollectingOutputChannel,
)
from atguigu_ai.channels.rest_channel import RestChannel
from atguigu_ai.channels.socketio_channel import SocketIOChannel
from atguigu_ai.channels.console_channel import ConsoleChannel
from atguigu_ai.channels.inspect_proxy import InspectProxy

__all__ = [
    "InputChannel",
    "OutputChannel",
    "UserMessage",
    "CollectingOutputChannel",
    "RestChannel",
    "SocketIOChannel",
    "ConsoleChannel",
    "InspectProxy",
]
