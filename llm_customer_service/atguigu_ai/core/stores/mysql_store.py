# -*- coding: utf-8 -*-
"""
mysql_store - MySQL数据库存储

将Tracker状态存储到MySQL数据库。
适合生产环境和需要高可用、多实例部署的场景。
"""

import json
from typing import Iterable, Optional, Text

from atguigu_ai.shared.exceptions import (
    TrackerSerializationError,
    TrackerStoreConnectionError,
    TrackerStoreException,
)
from atguigu_ai.core.tracker import DialogueStateTracker
from atguigu_ai.core.domain import Domain
from atguigu_ai.core.stores.tracker_store import TrackerStore


class MySQLTrackerStore(TrackerStore):
    """MySQL Tracker存储
    
    使用MySQL数据库存储Tracker状态。
    
    表结构：
        CREATE TABLE trackers (
            sender_id VARCHAR(255) PRIMARY KEY,
            state JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        );
    
    属性：
        host: 数据库主机
        port: 数据库端口
        db: 数据库名称
        username: 用户名
        password: 密码
        table_name: 表名
    """
    
    def __init__(
        self,
        domain: Optional[Domain] = None,
        host: str = "localhost",
        port: int = 3306,
        db: str = "atguigu",
        username: str = "root",
        password: str = "",
        url: Optional[str] = None,
        table_name: str = "trackers",
        **kwargs,
    ) -> None:
        """初始化MySQL存储
        
        参数：
            domain: Domain实例
            host: 数据库主机
            port: 数据库端口
            db: 数据库名称
            username: 用户名
            password: 密码
            url: 完整的数据库连接URL(优先于其他参数)
            table_name: 表名
        """
        super().__init__(domain)
        
        self.host = host
        self.port = port
        self.db = db
        self.username = username
        self.password = password
        self.url = url
        self.table_name = table_name
        
        self._engine = None
        self._session_maker = None
        self._table = None
        self._initialized = False
    
    def _get_connection_url(self) -> str:
        """获取数据库连接URL"""
        if self.url:
            return self.url
        
        return (
            f"mysql+pymysql://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}?charset=utf8mb4"
        )
    
    async def _ensure_initialized(self) -> None:
        """确保数据库连接已初始化"""
        if self._initialized:
            return
        
        try:
            from sqlalchemy import (
                Column,
                DateTime,
                MetaData,
                String,
                Table,
                Text as SQLText,
                create_engine,
                func,
            )
            from sqlalchemy.orm import sessionmaker
            
            # 创建引擎
            connection_url = self._get_connection_url()
            self._engine = create_engine(
                connection_url,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            
            # 创建表结构
            metadata = MetaData()
            self._table = Table(
                self.table_name,
                metadata,
                Column("sender_id", String(255), primary_key=True),
                Column("state", SQLText, nullable=False),
                Column("created_at", DateTime, default=func.now()),
                Column("updated_at", DateTime, default=func.now(), onupdate=func.now()),
            )
            
            # 创建表(如果不存在)
            metadata.create_all(self._engine)
            
            # 创建会话工厂
            self._session_maker = sessionmaker(bind=self._engine)
            
            self._initialized = True
        
        except ImportError:
            raise ImportError(
                "使用MySQL存储需要安装sqlalchemy和pymysql包。"
                "请运行: pip install sqlalchemy pymysql -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
        except Exception as e:
            raise TrackerStoreConnectionError(f"连接MySQL失败: {e}")
    
    async def save(self, tracker: DialogueStateTracker) -> None:
        """保存Tracker到MySQL
        
        使用UPSERT语义，存在则更新，不存在则插入。
        
        参数：
            tracker: 要保存的Tracker
        """
        await self._ensure_initialized()
        
        try:
            from sqlalchemy import insert
            from sqlalchemy.dialects.mysql import insert as mysql_insert
            
            tracker_data = tracker.to_dict()
            state_json = json.dumps(tracker_data, ensure_ascii=False)
            
            with self._session_maker() as session:
                # MySQL特有的ON DUPLICATE KEY UPDATE
                stmt = mysql_insert(self._table).values(
                    sender_id=tracker.sender_id,
                    state=state_json,
                )
                
                stmt = stmt.on_duplicate_key_update(
                    state=stmt.inserted.state,
                )
                
                session.execute(stmt)
                session.commit()
        
        except Exception as e:
            raise TrackerSerializationError(f"保存Tracker到MySQL失败: {e}")
    
    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """从MySQL加载Tracker
        
        参数：
            sender_id: 会话ID
            
        返回：
            DialogueStateTracker实例，不存在返回None
        """
        await self._ensure_initialized()
        
        try:
            from sqlalchemy import select
            
            with self._session_maker() as session:
                stmt = select(self._table.c.state).where(
                    self._table.c.sender_id == sender_id
                )
                result = session.execute(stmt).fetchone()
                
                if result is None:
                    return None
                
                state_json = result[0]
                tracker_data = json.loads(state_json)
                
                domain_slots = self.domain.slots if self.domain else None
                return DialogueStateTracker.from_dict(tracker_data, domain_slots)
        
        except json.JSONDecodeError as e:
            raise TrackerSerializationError(f"解析Tracker JSON失败: {e}")
        except Exception as e:
            raise TrackerStoreException(f"从MySQL加载Tracker失败: {e}")
    
    async def delete(self, sender_id: Text) -> None:
        """从MySQL删除Tracker
        
        参数：
            sender_id: 会话ID
        """
        await self._ensure_initialized()
        
        try:
            from sqlalchemy import delete
            
            with self._session_maker() as session:
                stmt = delete(self._table).where(
                    self._table.c.sender_id == sender_id
                )
                session.execute(stmt)
                session.commit()
        
        except Exception as e:
            raise TrackerStoreException(f"从MySQL删除Tracker失败: {e}")
    
    async def keys(self) -> Iterable[Text]:
        """获取所有sender_id
        
        返回：
            sender_id列表
        """
        await self._ensure_initialized()
        
        try:
            from sqlalchemy import select
            
            with self._session_maker() as session:
                stmt = select(self._table.c.sender_id)
                results = session.execute(stmt).fetchall()
                
                return [row[0] for row in results]
        
        except Exception as e:
            raise TrackerStoreException(f"获取sender_id列表失败: {e}")
    
    async def close(self) -> None:
        """关闭数据库连接"""
        if self._engine:
            self._engine.dispose()
            self._initialized = False
