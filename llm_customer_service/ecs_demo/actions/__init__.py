# -*- coding: utf-8 -*-
"""
电商客服Demo Actions模块

导出所有自定义Action供atguigu_ai框架使用。
"""

from .action_order import (
    ActionAskOrderId,
    ActionGetOrderDetail,
    ActionAskReceiveId,
    ActionAskReceiveProvince,
    ActionAskReceiveCity,
    ActionAskReceiveDistrict,
    ActionAskSetReceiveInfo,
    ActionCancelOrder,
)
from .action_logistics import (
    ActionGetLogisticsCompanys,
    ActionGetLogisticsInfo,
)
from .action_postsale import (
    ActionAskOrderIdAfterDelivered,
    ActionCheckPostsaleEligible,
    ActionAskPostsaleReason,
    ActionApplyPostsale,
)

# 导出所有Action类
__all__ = [
    # 订单相关
    "ActionAskOrderId",
    "ActionGetOrderDetail",
    "ActionAskReceiveId",
    "ActionAskReceiveProvince",
    "ActionAskReceiveCity",
    "ActionAskReceiveDistrict",
    "ActionAskSetReceiveInfo",
    "ActionCancelOrder",
    # 物流相关
    "ActionGetLogisticsCompanys",
    "ActionGetLogisticsInfo",
    # 售后相关
    "ActionAskOrderIdAfterDelivered",
    "ActionCheckPostsaleEligible",
    "ActionAskPostsaleReason",
    "ActionApplyPostsale",
]
