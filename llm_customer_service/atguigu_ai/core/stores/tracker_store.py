# -*- coding: utf-8 -*-
"""
tracker_store - Tracker存储基类

定义Tracker存储的抽象接口，所有存储后端都需要实现这些方法。
"""

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, Text

from atguigu_ai.core.tracker import DialogueStateTracker
from atguigu_ai.core.domain import Domain


class TrackerStore(ABC):
    """Tracker存储抽象基类
    
    定义Tracker持久化存储的统一接口。
    
    所有存储后端必须实现以下方法：
    - save: 保存Tracker
    - retrieve: 获取Tracker
    - delete: 删除Tracker
    - keys: 获取所有sender_id
    """
    
    def __init__(self, domain: Optional[Domain] = None) -> None:
        """初始化存储
        
        参数：
            domain: Domain实例，用于创建Tracker时初始化槽位
        """
        self.domain = domain
    
    @abstractmethod
    async def save(self, tracker: DialogueStateTracker) -> None:
        """保存Tracker状态
        
        将Tracker序列化后存储到后端。
        
        参数：
            tracker: 要保存的Tracker
            
        异常：
            TrackerStoreException: 保存失败
        """
        pass
    
    @abstractmethod
    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """获取Tracker状态
        
        从存储后端加载指定sender_id的Tracker。
        
        参数：
            sender_id: 会话ID
            
        返回：
            DialogueStateTracker实例，不存在返回None
            
        异常：
            TrackerStoreException: 获取失败
        """
        pass
    
    async def retrieve_full_tracker(
        self,
        sender_id: Text,
    ) -> Optional[DialogueStateTracker]:
        """获取完整的Tracker状态
        
        获取包含所有历史会话的Tracker。
        默认实现与retrieve相同，子类可重写以支持会话分割。
        
        参数：
            sender_id: 会话ID
            
        返回：
            DialogueStateTracker实例
        """
        return await self.retrieve(sender_id)
    
    @abstractmethod
    async def delete(self, sender_id: Text) -> None:
        """删除Tracker状态
        
        从存储后端删除指定sender_id的Tracker。
        
        参数：
            sender_id: 会话ID
            
        异常：
            TrackerStoreException: 删除失败
        """
        pass
    
    @abstractmethod
    async def keys(self) -> Iterable[Text]:
        """获取所有sender_id
        
        返回存储中所有Tracker的sender_id。
        
        返回：
            sender_id的可迭代对象
        """
        pass
    
    async def exists(self, sender_id: Text) -> bool:
        """检查Tracker是否存在
        
        参数：
            sender_id: 会话ID
            
        返回：
            存在返回True
        """
        tracker = await self.retrieve(sender_id)
        return tracker is not None
    
    def create_tracker(
        self,
        sender_id: Text,
    ) -> DialogueStateTracker:
        """创建新的Tracker
        
        使用Domain定义的槽位创建新的Tracker。
        
        参数：
            sender_id: 会话ID
            
        返回：
            新的DialogueStateTracker实例
        """
        slots = {}
        if self.domain:
            # 从Domain复制槽位定义
            from copy import deepcopy
            slots = {name: deepcopy(slot) for name, slot in self.domain.slots.items()}
        
        return DialogueStateTracker(sender_id=sender_id, slots=slots)
    
    async def get_or_create_tracker(
        self,
        sender_id: Text,
    ) -> DialogueStateTracker:
        """获取或创建Tracker
        
        如果Tracker存在则返回，否则创建新的。
        
        参数：
            sender_id: 会话ID
            
        返回：
            DialogueStateTracker实例
        """
        tracker = await self.retrieve(sender_id)
        
        if tracker is None:
            tracker = self.create_tracker(sender_id)
            await self.save(tracker)
        
        return tracker
    
    def set_domain(self, domain: Domain) -> None:
        """设置Domain
        
        参数：
            domain: Domain实例
        """
        self.domain = domain
