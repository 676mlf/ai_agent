# -*- coding: utf-8 -*-
"""
物流相关Action

实现物流查询功能。
适配atguigu_ai框架Action接口，复刻参考实现。
"""

import logging
from typing import Any, Optional

from atguigu_ai.agent.actions import Action, ActionResult

logger = logging.getLogger(__name__)


class ActionGetLogisticsCompanys(Action):
    """查询支持的快递公司"""
    
    @property
    def name(self) -> str:
        return "action_get_logistics_companys"
    
    async def run(
        self,
        tracker: Any,
        domain: Optional[Any] = None,
        **kwargs: Any,
    ) -> ActionResult:
        from actions.db import SessionLocal
        from actions.db_table_class import LogisticsCompany
        
        result = ActionResult()
        
        try:
            with SessionLocal() as session:
                logistics_companys = session.query(LogisticsCompany).all()
            
            # 按照company_name字段，拼接快递公司名称
            logistics_list = "".join(
                [f"- {i.company_name}\n" for i in logistics_companys]
            )
            
            # 如果没有查询到快递公司名称，返回"- 无"
            if logistics_list == "":
                logistics_list = "- 无"
            
            result.add_response(f"支持的快递有:\n{logistics_list}")
        except Exception as e:
            logger.error(f"查询快递公司失败: {e}")
            result.add_response("查询快递公司时出错，请稍后重试。")
        
        return result


class ActionGetLogisticsInfo(Action):
    """查询指定订单的物流信息"""
    
    @property
    def name(self) -> str:
        return "action_get_logistics_info"
    
    async def run(
        self,
        tracker: Any,
        domain: Optional[Any] = None,
        **kwargs: Any,
    ) -> ActionResult:
        from actions.db import SessionLocal
        from actions.db_table_class import OrderInfo
        from sqlalchemy.orm import joinedload
        
        result = ActionResult()
        order_id = tracker.get_slot("order_id")
        
        if not order_id or order_id == "false":
            result.add_response("请先选择要查询物流的订单。")
            return result
        
        try:
            with SessionLocal() as session:
                order_info = (
                    session.query(OrderInfo)
                    .options(joinedload(OrderInfo.logistics))
                    .options(joinedload(OrderInfo.order_detail))
                    .filter_by(order_id=order_id)
                    .first()
                )
                
                if not order_info:
                    result.add_response("未找到该订单，请检查订单号是否正确。")
                    return result
                
                if not order_info.logistics:
                    result.add_response("该订单暂无物流信息，可能还未发货。")
                    return result
                
                # 获取订单物流信息
                logistics = order_info.logistics[0]
                
                # 拼接订单id
                message = [f"- **订单ID**：{order_id}"]
                
                # 拼接sku名称和对应的数量
                for order_detail in order_info.order_detail:
                    message.append(f"  - {order_detail.sku_name} × {order_detail.sku_count}")
                
                # 拼接物流id
                message.append(f"- **物流ID**：{logistics.logistics_id}")
                
                # 拼接"物流信息"标题
                message.append("- **物流信息**：")
                
                # 拼接物流详情，对logistics表的logistics_tracking字段按照换行符进行分割
                if logistics.logistics_tracking:
                    message.append("  - " + "\n  - ".join(logistics.logistics_tracking.split("\n")))
                
                result.add_response("\n".join(message))
                
                # 将物流id存入槽中
                tracker.set_slot("logistics_id", logistics.logistics_id)
                
        except Exception as e:
            logger.error(f"查询物流失败: {e}")
            result.add_response("查询物流时出错，请稍后重试。")
        
        return result
