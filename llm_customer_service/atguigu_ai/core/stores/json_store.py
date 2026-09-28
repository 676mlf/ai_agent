# -*- coding: utf-8 -*-
"""
json_store - JSON文件存储

将Tracker状态以JSON文件格式存储到本地文件系统。
适合开发、测试和小规模部署场景。
"""

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Text

from atguigu_ai.shared.constants import DEFAULT_ENCODING
from atguigu_ai.shared.exceptions import TrackerSerializationError, TrackerStoreException
from atguigu_ai.core.tracker import DialogueStateTracker
from atguigu_ai.core.domain import Domain
from atguigu_ai.core.stores.tracker_store import TrackerStore


class JsonTrackerStore(TrackerStore):
    """JSON文件Tracker存储
    
    将每个Tracker保存为独立的JSON文件。
    
    文件结构：
        {path}/
        ├── {sender_id_1}.json
        ├── {sender_id_2}.json
        └── ...
    
    属性：
        path: 存储目录路径
        in_memory: 是否使用内存存储(不持久化)
    """
    
    def __init__(
        self,
        domain: Optional[Domain] = None,
        path: Optional[str] = "trackers",
        in_memory: bool = False,
    ) -> None:
        """初始化JSON存储
        
        参数：
            domain: Domain实例
            path: 存储目录路径
            in_memory: 是否使用内存存储
        """
        super().__init__(domain)
        
        self.in_memory = in_memory
        self._memory_store: Dict[str, Dict] = {}
        
        if not in_memory and path:
            self.path = Path(path)
            # 确保目录存在
            self.path.mkdir(parents=True, exist_ok=True)
        else:
            self.path = None
    
    def _get_file_path(self, sender_id: Text) -> Path:
        """获取Tracker文件路径
        
        参数：
            sender_id: 会话ID
            
        返回：
            文件路径
        """
        # 清理sender_id中的特殊字符
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in sender_id)
        return self.path / f"{safe_id}.json"
    
    async def save(self, tracker: DialogueStateTracker) -> None:
        """保存Tracker到JSON文件
        
        参数：
            tracker: 要保存的Tracker
        """
        try:
            tracker_data = tracker.to_dict()
            
            if self.in_memory:
                self._memory_store[tracker.sender_id] = tracker_data
            else:
                file_path = self._get_file_path(tracker.sender_id)
                with open(file_path, "w", encoding=DEFAULT_ENCODING) as f:
                    json.dump(tracker_data, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            raise TrackerSerializationError(f"保存Tracker失败: {e}")
    
    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """从JSON文件加载Tracker
        
        参数：
            sender_id: 会话ID
            
        返回：
            DialogueStateTracker实例，不存在返回None
        """
        try:
            if self.in_memory:
                tracker_data = self._memory_store.get(sender_id)
            else:
                file_path = self._get_file_path(sender_id)
                
                if not file_path.exists():
                    return None
                
                with open(file_path, "r", encoding=DEFAULT_ENCODING) as f:
                    tracker_data = json.load(f)
            
            if tracker_data is None:
                return None
            
            # 获取Domain槽位用于反序列化
            domain_slots = self.domain.slots if self.domain else None
            
            return DialogueStateTracker.from_dict(tracker_data, domain_slots)
        
        except json.JSONDecodeError as e:
            raise TrackerSerializationError(f"解析Tracker JSON失败: {e}")
        except Exception as e:
            raise TrackerStoreException(f"加载Tracker失败: {e}")
    
    async def delete(self, sender_id: Text) -> None:
        """删除Tracker文件
        
        参数：
            sender_id: 会话ID
        """
        try:
            if self.in_memory:
                self._memory_store.pop(sender_id, None)
            else:
                file_path = self._get_file_path(sender_id)
                
                if file_path.exists():
                    file_path.unlink()
        
        except Exception as e:
            raise TrackerStoreException(f"删除Tracker失败: {e}")
    
    async def keys(self) -> Iterable[Text]:
        """获取所有sender_id
        
        返回：
            sender_id列表
        """
        if self.in_memory:
            return list(self._memory_store.keys())
        
        if self.path is None:
            return []
        
        sender_ids = []
        for file_path in self.path.glob("*.json"):
            # 从文件名恢复sender_id
            sender_id = file_path.stem
            sender_ids.append(sender_id)
        
        return sender_ids
    
    async def clear_all(self) -> None:
        """清除所有Tracker
        
        警告：此操作不可逆！
        """
        if self.in_memory:
            self._memory_store.clear()
        else:
            for sender_id in await self.keys():
                await self.delete(sender_id)
