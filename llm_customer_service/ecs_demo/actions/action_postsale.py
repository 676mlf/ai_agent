# -*- coding: utf-8 -*-
"""
售后相关Action

实现退换货申请功能。
适配atguigu_ai框架Action接口。
"""

import logging
from typing import Any, Optional
from datetime import datetime
from uuid import uuid4

from atguigu_ai.agent.actions import Action, ActionResult

logger = logging.getLogger(__name__)


class ActionAskOrderIdAfterDelivered(Action):
    """查询已签收的订单，用于售后申请"""
    
    @property
    def name(self) -> str:
        return "action_ask_order_id_after_delivered"
    
    async def run(
        self,
        tracker: Any,
        domain: Optional[Any] = None,
        **kwargs: Any,
    ) -> ActionResult:
        from actions.db import SessionLocal
        from actions.db_table_class import OrderInfo, OrderStatus
        from sqlalchemy.orm import joinedload
        
        result = ActionResult()
        user_id = tracker.get_slot("user_id") or "1001"
        
        try:
            with SessionLocal() as session:
                # 查询已签收、售后中、已完成的订单
                order_infos = (
                    session.query(OrderInfo)
                    .join(OrderInfo.order_status_)
                    .options(joinedload(OrderInfo.order_detail))
                    .filter(
                        OrderInfo.user_id == user_id,
                        OrderInfo.order_status != "已取消",
                        OrderStatus.status_code >= 330  # 已签收及之后
                    )
                    .order_by(OrderInfo.create_time.desc())
                    .all()
                )
                
                if not order_infos:
                    result.add_response("您没有可申请售后的订单。")
                    tracker.set_slot("order_id", "NO_ORDER")
                    return result
                
                msg = "请选择要申请售后的订单："
                buttons = []
                
                for i, order in enumerate(order_infos, 1):
                    sku_info = []
                    for detail in order.order_detail:
                        sku_info.append(f"{detail.sku_name}×{detail.sku_count}")
                    sku_str = ", ".join(sku_info[:2])
                    
                    # 添加按钮
                    buttons.append({
                        "title": f"{order.order_id[:12]}... ({sku_str[:15]})",
                        "payload": f"/SetSlots(order_id={order.order_id})"
                    })
                
                result.add_response(msg, buttons=buttons)
        except Exception as e:
            logger.error(f"查询订单失败: {e}")
            result.add_response("查询订单时出错，请稍后重试。")
        
        return result


class ActionCheckPostsaleEligible(Action):
    """检查订单是否支持售后"""
    
    @property
    def name(self) -> str:
        return "action_check_postsale_eligible"
    
    async def run(
        self,
        tracker: Any,
        domain: Optional[Any] = None,
        **kwargs: Any,
    ) -> ActionResult:
        from actions.db import SessionLocal
        from actions.db_table_class import OrderInfo, OrderStatus
        from sqlalchemy.orm import joinedload
        
        result = ActionResult()
        order_id = tracker.get_slot("order_id")
        
        if not order_id or order_id == "false":
            result.add_response("请先选择要申请售后的订单。")
            return result
        
        try:
            with SessionLocal() as session:
                order_info = (
                    session.query(OrderInfo)
                    .join(OrderInfo.order_status_)
                    .options(joinedload(OrderInfo.order_detail))
                    .filter_by(order_id=order_id)
                    .first()
                )
                
                if not order_info:
                    result.add_response("未找到该订单。")
                    tracker.set_slot("order_id", "NO_ORDER")
                    return result
                
                # 检查订单状态是否支持售后 (已签收及之后)
                if order_info.order_status_.status_code < 330:
                    result.add_response("抱歉，该订单当前状态不支持售后申请。需要订单已签收后才能申请售后。")
                    return result
                
                # 检查是否在售后期限内（7天内）
                check_time = order_info.delivered_time or order_info.create_time
                if check_time:
                    days_since = (datetime.now() - check_time).days
                    if days_since > 7:
                        result.add_response("抱歉，该订单已超过7天售后期限。")
                        return result
                
                msg = f"订单 {order_id} 符合售后条件。\n"
                msg += "请选择售后类型：\n"
                msg += "1. 退款 - 仅退款不退货\n"
                msg += "2. 退货 - 退货并退款\n"
                msg += "3. 换货 - 退回商品换新\n"
                
                result.add_response(msg)
        except Exception as e:
            logger.error(f"检查售后资格失败: {e}")
            result.add_response("检查售后资格时出错，请稍后重试。")
        
        return result


class ActionAskPostsaleReason(Action):
    """询问售后原因"""
    
    @property
    def name(self) -> str:
        return "action_ask_postsale_reason"
    
    async def run(
        self,
        tracker: Any,
        domain: Optional[Any] = None,
        **kwargs: Any,
    ) -> ActionResult:
        from actions.db import SessionLocal
        from actions.db_table_class import PostsaleReason, OrderInfo, OrderDetail, SkuInfo
        from sqlalchemy import or_
        from sqlalchemy.orm import joinedload
        
        result = ActionResult()
        order_id = tracker.get_slot("order_id")
        
        try:
            with SessionLocal() as session:
                # 获取订单中商品的类别
                order_info = (
                    session.query(OrderInfo)
                    .options(joinedload(OrderInfo.order_detail))
                    .filter_by(order_id=order_id)
                    .first()
                )
                
                if not order_info or not order_info.order_detail:
                    result.add_response("请输入售后原因：")
                    return result
                
                # 获取商品类别
                sku_ids = [detail.sku_id for detail in order_info.order_detail]
                sku_infos = session.query(SkuInfo).filter(SkuInfo.sku_id.in_(sku_ids)).all()
                categories = list(set(sku.sku_category for sku in sku_infos))
                
                # 查询适用的售后原因
                reasons = (
                    session.query(PostsaleReason)
                    .filter(
                        or_(
                            PostsaleReason.product_category.in_(categories),
                            PostsaleReason.product_category.is_(None)
                        )
                    )
                    .all()
                )
                
                if reasons:
                    msg = "请选择售后原因：\n"
                    for i, reason in enumerate(reasons, 1):
                        msg += f"{i}. {reason.postsale_reason}\n"
                    msg += f"{len(reasons)+1}. 其他原因\n"
                    result.add_response(msg)
                else:
                    result.add_response("请输入售后原因：")
                    
        except Exception as e:
            logger.error(f"获取售后原因失败: {e}")
            result.add_response("请输入售后原因：")
        
        return result


class ActionApplyPostsale(Action):
    """提交售后申请"""
    
    @property
    def name(self) -> str:
        return "action_apply_postsale"
    
    async def run(
        self,
        tracker: Any,
        domain: Optional[Any] = None,
        **kwargs: Any,
    ) -> ActionResult:
        from actions.db import SessionLocal
        from actions.db_table_class import Postsale, OrderInfo, PostsaleStatus
        from sqlalchemy.orm import joinedload
        
        result = ActionResult()
        order_id = tracker.get_slot("order_id")
        postsale_type = tracker.get_slot("postsale_type")  # 退款/退货/换货
        postsale_reason = tracker.get_slot("postsale_reason")
        user_id = tracker.get_slot("user_id") or "1001"
        
        if not all([order_id, postsale_type, postsale_reason]):
            result.add_response("售后信息不完整，请重新申请。")
            return result
        
        try:
            with SessionLocal() as session:
                # 获取订单信息
                order_info = (
                    session.query(OrderInfo)
                    .options(joinedload(OrderInfo.order_detail))
                    .filter_by(order_id=order_id)
                    .first()
                )
                
                if not order_info:
                    result.add_response("未找到该订单。")
                    return result
                
                # 确定初始售后状态
                status_map = {
                    "退款": "退款待审核",
                    "退货": "退货待审核", 
                    "换货": "换货待审核",
                }
                initial_status = status_map.get(postsale_type, "退款待审核")
                
                # 为每个订单明细创建售后记录
                created_postsales = []
                for detail in order_info.order_detail:
                    postsale = Postsale(
                        postsale_id="pts" + uuid4().hex[:16],
                        create_time=datetime.now(),
                        order_detail_id=detail.order_detail_id,
                        postsale_reason=postsale_reason,
                        postsale_status=initial_status,
                        receive_id=order_info.receive_id,
                        refund_amount=detail.final_amount if postsale_type != "换货" else None,
                        postsale_type=postsale_type,
                    )
                    session.add(postsale)
                    created_postsales.append(postsale)
                
                # 更新订单状态为售后中
                order_info.order_status = "售后中"
                
                session.commit()
                
                logger.info(f"售后申请已创建: 订单{order_id}, 类型{postsale_type}, 数量{len(created_postsales)}")
            
            type_text = {"退款": "退款", "退货": "退货退款", "换货": "换货"}.get(postsale_type, "售后")
            result.add_response(
                f"您的{type_text}申请已提交！\n\n"
                f"- 订单号: {order_id}\n"
                f"- 申请类型: {postsale_type}\n"
                f"- 申请原因: {postsale_reason}\n\n"
                f"我们会在1-3个工作日内处理，请耐心等待。"
            )
        except Exception as e:
            logger.error(f"提交售后申请失败: {e}")
            result.add_response("提交申请时出错，请稍后重试。")
        
        return result
