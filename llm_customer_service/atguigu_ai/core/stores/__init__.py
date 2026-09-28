# -*- coding: utf-8 -*-
"""
stores - Tracker存储系统

提供对话状态的持久化存储，支持多种存储后端：
- JSON: 本地文件存储，适合开发和测试
- MySQL: 关系数据库存储，适合生产环境
- Memory: 内存存储，适合临时测试
"""

from atguigu_ai.core.stores.tracker_store import TrackerStore
from atguigu_ai.core.stores.json_store import JsonTrackerStore
from atguigu_ai.core.stores.mysql_store import MySQLTrackerStore

__all__ = [
    "TrackerStore",
    "JsonTrackerStore", 
    "MySQLTrackerStore",
    "create_tracker_store",
]


def create_tracker_store(
    store_type: str = "json",
    **kwargs,
) -> TrackerStore:
    """创建Tracker存储实例
    
    工厂函数，根据类型创建对应的存储后端。
    
    参数：
        store_type: 存储类型 (json/mysql/memory)
        **kwargs: 存储配置参数
        
    返回：
        TrackerStore实例
        
    示例：
        >>> store = create_tracker_store("json", path="./trackers")
        >>> await store.save(tracker)
    """
    store_type = store_type.lower()
    
    if store_type == "json":
        return JsonTrackerStore(**kwargs)
    elif store_type == "mysql":
        return MySQLTrackerStore(**kwargs)
    elif store_type == "memory":
        # 内存存储使用空的JSON存储实现
        return JsonTrackerStore(path=None, in_memory=True)
    else:
        raise ValueError(
            f"不支持的存储类型: {store_type}。"
            f"支持的类型: json, mysql, memory"
        )
