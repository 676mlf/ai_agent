# -*- coding: utf-8 -*-
"""
Action系统

定义对话系统的动作（Action）基类和内置动作。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker
    from atguigu_ai.core.domain import Domain


@dataclass
class ActionResult:
    """动作执行结果。
    
    Attributes:
        responses: 要发送的响应列表
        events: 产生的事件列表
        success: 是否执行成功
        metadata: 额外元数据
    """
    responses: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_response(self, text: str, **kwargs: Any) -> None:
        """添加文本响应。"""
        response = {"text": text}
        response.update(kwargs)
        self.responses.append(response)
    
    def add_event(self, event_type: str, **kwargs: Any) -> None:
        """添加事件。"""
        event = {"event": event_type}
        event.update(kwargs)
        self.events.append(event)


class Action(ABC):
    """动作基类。
    
    动作是对话系统执行具体操作的单元。
    每个动作都有一个名称，并实现run方法来执行具体逻辑。
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """动作名称。"""
        raise NotImplementedError()
    
    @abstractmethod
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行动作。
        
        Args:
            tracker: 对话状态追踪器
            domain: Domain定义
            **kwargs: 额外参数
            
        Returns:
            执行结果
        """
        raise NotImplementedError()
    
    def run_sync(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """同步版本的run方法。"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.run(tracker, domain, **kwargs))


# =============================================================================
# 内置动作
# =============================================================================

class ActionListen(Action):
    """监听动作。
    
    等待用户输入，不执行任何操作。
    """
    
    @property
    def name(self) -> str:
        return "action_listen"
    
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """等待用户输入。"""
        return ActionResult()


class ActionRestart(Action):
    """重启动作。
    
    重置对话状态。
    """
    
    @property
    def name(self) -> str:
        return "action_restart"
    
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """重启对话。"""
        tracker.restart()
        
        result = ActionResult()
        result.add_event("conversation_restarted")
        return result


class ActionSessionStart(Action):
    """会话开始动作。
    
    初始化新会话，重置对话状态。
    """
    
    @property
    def name(self) -> str:
        return "action_session_start"
    
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """开始新会话。"""
        result = ActionResult()
        
        # 重置tracker状态
        tracker.restart()
        
        result.add_event("session_started")
        return result


class ActionDefaultFallback(Action):
    """默认降级动作。
    
    压入CannotHandleStackFrame，标记当前无法处理。
    实际降级响应由EnterpriseSearchPolicy生成。
    """
    
    @property
    def name(self) -> str:
        return "action_default_fallback"
    
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """压入CannotHandleStackFrame触发降级。"""
        result = ActionResult()
        
        # 获取降级原因
        reason = kwargs.get("reason", "无法理解用户意图")
        
        # 压入 CannotHandleStackFrame
        from atguigu_ai.dialogue_understanding.stack.stack_frame import CannotHandleStackFrame
        
        tracker.dialogue_stack.push(CannotHandleStackFrame(reason=reason))
        result.add_event("cannot_handle_triggered", reason=reason)
        
        return result


class ActionChitChatResponse(Action):
    """闲聊响应动作。
    
    压入ChitChatStackFrame，标记当前处于闲聊模式。
    实际闲聊回复由EnterpriseSearchPolicy生成。
    """
    
    @property
    def name(self) -> str:
        return "action_chitchat_response"
    
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """压入ChitChatStackFrame触发闲聊。"""
        result = ActionResult()
        
        # 压入 ChitChatStackFrame
        from atguigu_ai.dialogue_understanding.stack.stack_frame import ChitChatStackFrame
        
        tracker.dialogue_stack.push(ChitChatStackFrame())
        result.add_event("chitchat_triggered")
        
        return result


class ActionCancelFlow(Action):
    """取消流程动作。
    
    结束当前活动的Flow。通过tracker.end_flow()操作dialogue_stack。
    """
    
    @property
    def name(self) -> str:
        return "action_cancel_flow"
    
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """取消当前流程。"""
        result = ActionResult()
        
        # 检查是否有活动的Flow并结束
        if tracker.active_flow:
            flow_id = tracker.active_flow
            tracker.end_flow()
            result.add_event("flow_cancelled", flow_id=flow_id)
            result.add_response("好的，已取消当前操作。")
        else:
            result.add_response("当前没有进行中的流程。")
        
        return result


class ActionChangeFlow(Action):
    """切换流程动作。
    
    结束当前Flow，启动新Flow。通过tracker方法操作dialogue_stack。
    """
    
    @property
    def name(self) -> str:
        return "action_change_flow"
    
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """切换流程：结束旧流程，启动新流程。"""
        result = ActionResult()
        
        target_flow = kwargs.get("target_flow", "")
        
        if not target_flow:
            result.add_response("无法切换流程：未指定目标流程。")
            return result
        
        # 结束当前Flow（如果有）
        if tracker.active_flow:
            old_flow_id = tracker.active_flow
            tracker.end_flow()
            result.add_event("flow_cancelled", flow_id=old_flow_id)
        
        # 启动新Flow
        tracker.start_flow(target_flow)
        result.add_event("flow_started", flow_id=target_flow)
        
        return result


class ActionCleanStack(Action):
    """清理对话栈动作。
    
    清空dialogue_stack中的所有帧。
    """
    
    @property
    def name(self) -> str:
        return "action_clean_stack"
    
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """清理对话栈。"""
        result = ActionResult()
        
        cleared_count = len(tracker.dialogue_stack.frames)
        
        # 清空对话栈（cancel_flow内部调用dialogue_stack.clear()）
        tracker.cancel_flow()
        
        result.add_event("stack_cleaned", frames_cleared=cleared_count)
        return result


class ActionExtractSlots(Action):
    """提取槽位动作。
    
    从用户消息中提取槽位值，简化版本使用LLM提取。
    """
    
    @property
    def name(self) -> str:
        return "action_extract_slots"
    
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """从消息中提取槽位。"""
        result = ActionResult()
        
        # 获取需要提取的槽位
        slots_to_extract = kwargs.get("slots_to_extract", [])
        user_message = ""
        if tracker.latest_message:
            user_message = tracker.latest_message.text
        
        if not user_message or not slots_to_extract:
            return result
        
        # 尝试使用LLM提取槽位
        try:
            from atguigu_ai.shared.llm import create_llm_client
            
            llm_config = kwargs.get("llm_config", {})
            if llm_config.get("api_key"):
                client = create_llm_client(
                    type=llm_config.get("type", "openai"),
                    model=llm_config.get("model", "gpt-4o-mini"),
                    api_key=llm_config["api_key"],
                    api_base=llm_config.get("api_base"),
                    temperature=0.0,
                )
                
                # 构建提取提示词
                slot_descriptions = []
                for slot_name in slots_to_extract:
                    slot_descriptions.append(f"- {slot_name}")
                
                prompt = f"""从以下用户消息中提取指定的信息。

用户消息: {user_message}

需要提取的字段:
{chr(10).join(slot_descriptions)}

请以JSON格式返回提取结果，如果无法提取某个字段，值设为null。
只返回JSON，不要其他内容。"""
                
                messages = [{"role": "user", "content": prompt}]
                response = await client.complete(messages)
                
                if response and response.content:
                    import json
                    try:
                        # 尝试解析JSON
                        content = response.content.strip()
                        if content.startswith("```"):
                            content = content.split("```")[1]
                            if content.startswith("json"):
                                content = content[4:]
                        
                        extracted = json.loads(content)
                        
                        # 设置提取到的槽位
                        for slot_name, slot_value in extracted.items():
                            if slot_value is not None:
                                tracker.set_slot(slot_name, slot_value)
                                result.add_event(
                                    "slot_set",
                                    name=slot_name,
                                    value=slot_value,
                                )
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            # LLM调用失败，跳过提取
            pass
        
        return result


class ActionSendText(Action):
    """发送文本动作。
    
    直接发送指定的文本响应。
    """
    
    @property
    def name(self) -> str:
        return "action_send_text"
    
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """发送文本。"""
        result = ActionResult()
        
        text = kwargs.get("text", "")
        if text:
            result.add_response(text)
        
        return result


class ActionHandleHelp(Action):
    """处理帮助请求动作。
    
    处理用户的帮助请求，提供通用帮助信息。
    """
    
    @property
    def name(self) -> str:
        return "action_handle_help"
    
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """处理帮助请求。"""
        result = ActionResult()
        
        # 获取用户最近的消息，尝试提供针对性帮助
        help_text = "我可以帮助您完成以下任务：\n"
        help_text += "1. 回答问题\n"
        help_text += "2. 提供信息查询\n"
        help_text += "3. 进行日常对话\n"
        help_text += "\n请告诉我您具体需要什么帮助？"
        
        result.add_response(help_text)
        return result


class ActionClarify(Action):
    """澄清动作。
    
    当系统无法理解用户意图或需要更多信息时，
    向用户请求澄清。
    """
    
    @property
    def name(self) -> str:
        return "action_clarify"
    
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """请求用户澄清。"""
        result = ActionResult()
        
        # 获取澄清问题（如果有）
        question = kwargs.get("question", "")
        options = kwargs.get("options", [])
        
        if question:
            clarify_text = question
        else:
            clarify_text = "抱歉，我不太确定您的意思。能否请您再说明一下？"
        
        # 如果有选项，添加选项提示
        if options:
            clarify_text += "\n\n您可以选择："
            for i, opt in enumerate(options, 1):
                clarify_text += f"\n{i}. {opt}"
        
        result.add_response(clarify_text)
        result.add_event("clarification_requested", question=question, options=options)
        
        return result


class ActionHumanHandoff(Action):
    """人工转接动作。
    
    当系统无法处理用户请求或用户明确要求时，
    将对话转接给人工客服。
    
    职责：压入HumanHandoffStackFrame，由EnterpriseSearchPolicy生成响应。
    """
    
    @property
    def name(self) -> str:
        return "action_human_handoff"
    
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """压入HumanHandoffStackFrame触发人工转接。"""
        result = ActionResult()
        
        # 获取转接原因（如果有）
        reason = kwargs.get("reason", "")
        
        # 压入HumanHandoffStackFrame
        from atguigu_ai.dialogue_understanding.stack.stack_frame import HumanHandoffStackFrame
        
        tracker.dialogue_stack.push(HumanHandoffStackFrame(reason=reason))
        result.add_event("human_handoff_triggered", reason=reason)
        
        return result


class ActionTriggerSearch(Action):
    """触发搜索动作。
    
    将SearchStackFrame压入对话栈，标记当前处于搜索模式。
    不存储query，实际检索时从latest_message获取。
    实际检索由EnterpriseSearchPolicy执行。
    """
    
    @property
    def name(self) -> str:
        return "action_trigger_search"
    
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """压入SearchStackFrame触发搜索。"""
        result = ActionResult()
        
        # 验证是否有消息（不存储query，只验证）
        if not tracker.latest_message or not tracker.latest_message.text:
            result.add_response("请提供您的问题。")
            return result
        
        # 压入空的SearchStackFrame
        from atguigu_ai.dialogue_understanding.stack.stack_frame import SearchStackFrame
        
        tracker.dialogue_stack.push(SearchStackFrame())
        result.add_event("search_triggered")
        
        return result


class ActionFlowCompleted(Action):
    """Flow完成动作。
    
    当Flow执行完成后，压入CompletedStackFrame。
    由EnterpriseSearchPolicy检测栈帧并生成询问用户是否还有其他需求的响应。
    
    职责：压入CompletedStackFrame，由Policy生成响应。
    """
    
    @property
    def name(self) -> str:
        return "action_flow_completed"
    
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """压入CompletedStackFrame标记Flow完成。"""
        result = ActionResult()
        
        # 获取刚完成的Flow信息
        completed_flow = kwargs.get("completed_flow", "")
        
        # 压入CompletedStackFrame
        from atguigu_ai.dialogue_understanding.stack.stack_frame import CompletedStackFrame
        
        tracker.dialogue_stack.push(CompletedStackFrame(previous_flow_name=completed_flow))
        result.add_event("flow_completed_handled", flow_id=completed_flow)
        
        return result


class ActionUtter(Action):
    """模板响应动作。
    
    从domain中获取响应模板并发送。
    """
    
    def __init__(self, utter_name: str):
        """初始化。
        
        Args:
            utter_name: 响应模板名称
        """
        self._utter_name = utter_name
    
    @property
    def name(self) -> str:
        return self._utter_name
    
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行模板响应。"""
        result = ActionResult()
        
        if domain:
            responses = domain.get_response(self._utter_name)
            if responses:
                import random
                response = random.choice(responses)
                
                # 替换槽位变量
                text = response.text
                if text:
                    all_slots = tracker.get_all_slots()
                    for slot_name, slot_value in all_slots.items():
                        text = text.replace(f"{{{slot_name}}}", str(slot_value or ""))
                
                # 构建响应，包含 buttons
                result.add_response(
                    text or "",
                    buttons=response.buttons or [],
                    image=response.image,
                    custom=response.custom,
                )
        
        return result


# =============================================================================
# Action注册和工厂
# =============================================================================

# 内置动作注册表
_BUILTIN_ACTIONS: Dict[str, type] = {
    "action_listen": ActionListen,
    "action_restart": ActionRestart,
    "action_session_start": ActionSessionStart,
    "action_default_fallback": ActionDefaultFallback,
    "action_chitchat_response": ActionChitChatResponse,
    "action_cancel_flow": ActionCancelFlow,
    "action_change_flow": ActionChangeFlow,
    "action_clean_stack": ActionCleanStack,
    "action_extract_slots": ActionExtractSlots,
    "action_send_text": ActionSendText,
    "action_handle_help": ActionHandleHelp,
    "action_clarify": ActionClarify,
    "action_human_handoff": ActionHumanHandoff,
    "action_flow_completed": ActionFlowCompleted,
    "action_trigger_search": ActionTriggerSearch,
}

# 自定义动作注册表
_CUSTOM_ACTIONS: Dict[str, Action] = {}


def register_action(action: Action) -> None:
    """注册自定义动作。
    
    Args:
        action: 动作实例
    """
    _CUSTOM_ACTIONS[action.name] = action


def get_action(name: str) -> Optional[Action]:
    """获取动作实例。
    
    Args:
        name: 动作名称
        
    Returns:
        动作实例，如果不存在则返回None
    """
    # 先检查自定义动作
    if name in _CUSTOM_ACTIONS:
        return _CUSTOM_ACTIONS[name]
    
    # 检查内置动作
    if name in _BUILTIN_ACTIONS:
        return _BUILTIN_ACTIONS[name]()
    
    # 检查是否是utter_动作
    if name.startswith("utter_"):
        return ActionUtter(name)
    
    return None


def get_all_action_names() -> List[str]:
    """获取所有已注册的动作名称。"""
    names = list(_BUILTIN_ACTIONS.keys())
    names.extend(_CUSTOM_ACTIONS.keys())
    return names


# 导出
__all__ = [
    "Action",
    "ActionResult",
    "ActionListen",
    "ActionRestart",
    "ActionSessionStart",
    "ActionDefaultFallback",
    "ActionChitChatResponse",
    "ActionCancelFlow",
    "ActionChangeFlow",
    "ActionCleanStack",
    "ActionExtractSlots",
    "ActionSendText",
    "ActionHandleHelp",
    "ActionClarify",
    "ActionHumanHandoff",
    "ActionFlowCompleted",
    "ActionTriggerSearch",
    "ActionUtter",
    "register_action",
    "get_action",
    "get_all_action_names",
]
