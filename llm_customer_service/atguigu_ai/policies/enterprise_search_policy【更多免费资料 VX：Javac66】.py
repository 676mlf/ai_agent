# -*- coding: utf-8 -*-
"""
企业搜索策略

基于知识库检索的策略，实现RAG功能和降级机制。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from atguigu_ai.policies.base_policy import Policy, PolicyConfig, PolicyPrediction
from atguigu_ai.shared.constants import DegradationReason, ACTION_DEFAULT_FALLBACK
from atguigu_ai.shared.llm import create_llm_client
from atguigu_ai.shared.llm.base_client import LLMClient
from atguigu_ai.retrieval.base_retriever import SearchResult

if TYPE_CHECKING:
    from atguigu_ai.core.tracker import DialogueStateTracker
    from atguigu_ai.core.domain import Domain
    from atguigu_ai.dialogue_understanding.flow import FlowsList
    from atguigu_ai.dialogue_understanding.stack.stack_frame import StackFrame

logger = logging.getLogger(__name__)


@dataclass
class _InternalRetrievalConfig:
    """内部检索配置（简化版）。"""
    enabled: bool = True
    top_k: int = 3
    similarity_threshold: float = 0.5


@dataclass
class EnterpriseSearchPolicyConfig(PolicyConfig):
    """企业搜索策略配置。
    
    Attributes:
        priority: 策略优先级
        retrieval: 检索配置
        llm_type: LLM类型 (openai/qwen/azure/anthropic)
        llm_model: LLM模型名
        enable_citation: 是否启用引用
        enable_relevancy_check: 是否启用相关性检查
        chitchat_enabled: 是否启用闲聊降级
    """
    priority: int = 50  # 中等优先级
    retrieval: _InternalRetrievalConfig = field(default_factory=_InternalRetrievalConfig)
    llm_type: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    enable_citation: bool = False
    enable_relevancy_check: bool = True
    chitchat_enabled: bool = True


class EnterpriseSearchPolicy(Policy):
    """企业搜索策略。
    
    基于知识库检索实现RAG功能，并包含内置的降级机制。
    
    降级链：
    1. Flow匹配 → 执行Flow
    2. 知识库检索 → 生成RAG回答
    3. 闲聊识别 → 生成闲聊回复
    4. 无法处理 → 返回默认回复
    
    工作流程：
    1. 检索相关文档
    2. 检查相关性
    3. 使用LLM生成回答
    4. 如果无相关答案，降级到闲聊或默认回复
    """
    
    DEFAULT_PRIORITY = 50
    
    # RAG提示词模板
    RAG_PROMPT_TEMPLATE = """你是一个专业的客服助手，正在根据知识库文档回答用户问题。

### 参考文档
{context}

### 用户问题
{question}

### 回答要求
严格基于上述文档内容回答：
1. 直接回答问题，不要添加问候语或寒暄
2. 禁止使用 emoji 表情符号
3. 使用专业、简洁的语气
4. 只陈述文档中明确提到的信息
5. 如果文档包含具体的产品名称、品牌、规格等，必须准确引用
6. 最多2-3句话，避免冗余
7. 如果文档信息不足以回答问题，仅回复"[NO_RAG_ANSWER]"

回答：
"""
    
    # 闲聊提示词模板
    CHITCHAT_PROMPT_TEMPLATE = """你是一个友好的AI助手。用户发送了一条闲聊消息，请用自然、友好的方式回复。

用户: {message}

请回复（保持简短友好）：
"""
    
    def __init__(
        self,
        config: Optional[EnterpriseSearchPolicyConfig] = None,
        llm_client: Optional[LLMClient] = None,
        retriever: Optional[Any] = None,
        **kwargs: Any,
    ):
        """初始化企业搜索策略。
        
        Args:
            config: 策略配置
            llm_client: LLM客户端
            retriever: 检索器
            **kwargs: 额外参数
        """
        super().__init__(config or EnterpriseSearchPolicyConfig(), **kwargs)
        self.config: EnterpriseSearchPolicyConfig = self.config
        
        self._llm_client = llm_client
        self._retriever = retriever
    
    @property
    def llm_client(self) -> LLMClient:
        """获取LLM客户端（延迟初始化）。"""
        if self._llm_client is None:
            self._llm_client = create_llm_client(
                type=self.config.llm_type,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
            )
        return self._llm_client
    
    def does_support_stack_frame(self, frame: Optional[Any] = None) -> bool:
        """检查策略是否支持处理指定栈帧。
        
        支持：SearchStackFrame、ChitChatStackFrame、CannotHandleStackFrame、
              CompletedStackFrame、HumanHandoffStackFrame
        
        Args:
            frame: 要检查的栈帧
            
        Returns:
            是否支持处理该栈帧
        """
        from atguigu_ai.dialogue_understanding.stack.stack_frame import (
            SearchStackFrame,
            ChitChatStackFrame,
            CannotHandleStackFrame,
            CompletedStackFrame,
            HumanHandoffStackFrame,
        )
        return isinstance(frame, (
            SearchStackFrame, 
            ChitChatStackFrame, 
            CannotHandleStackFrame,
            CompletedStackFrame,
            HumanHandoffStackFrame,
        ))
    
    async def predict(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        flows: Optional["FlowsList"] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """预测下一步动作。
        
        检测栈帧类型并分发处理：
        - SearchStackFrame → 执行检索
        - ChitChatStackFrame → 生成闲聊回复
        - CannotHandleStackFrame → 返回降级响应
        - CompletedStackFrame → 询问是否还有其他需求
        - HumanHandoffStackFrame → 执行人工转接
        
        Args:
            tracker: 对话状态追踪器
            domain: Domain定义
            flows: Flow列表
            **kwargs: 额外参数
            
        Returns:
            预测结果
        """
        from atguigu_ai.dialogue_understanding.stack.stack_frame import (
            SearchStackFrame,
            ChitChatStackFrame,
            CannotHandleStackFrame,
            CompletedStackFrame,
            HumanHandoffStackFrame,
        )
        
        # 获取栈顶帧
        top_frame = tracker.dialogue_stack.top()
        
        # 检查是否已经有 bot 响应（如果刚执行了动作，则放弃）
        # 但是对于需要立即处理的栈帧（如 CompletedStackFrame），不应放弃
        from atguigu_ai.dialogue_understanding.stack.stack_frame import (
            CompletedStackFrame as CompletedFrame,
            HumanHandoffStackFrame as HandoffFrame,
        )
        needs_immediate_handling = isinstance(top_frame, (CompletedFrame, HandoffFrame))
        
        if (tracker.latest_action_name 
            and tracker.latest_action_name != "action_listen"
            and not needs_immediate_handling):
            logger.debug(f"Action {tracker.latest_action_name} just executed, abstaining")
            return PolicyPrediction.abstain(self.name)
        
        # 获取用户消息（统一从latest_message获取）
        user_message = ""
        if tracker.latest_message:
            user_message = tracker.latest_message.text
        
        # 根据栈帧类型分发处理
        if isinstance(top_frame, CompletedStackFrame):
            return await self._handle_completed_frame(tracker, top_frame, domain)
        
        if isinstance(top_frame, HumanHandoffStackFrame):
            return await self._handle_human_handoff_frame(tracker, top_frame, domain)
        
        if isinstance(top_frame, ChitChatStackFrame):
            return await self._handle_chitchat_frame(tracker, user_message)
        
        if isinstance(top_frame, CannotHandleStackFrame):
            return await self._handle_cannot_handle_frame(tracker, top_frame, domain)
        
        if isinstance(top_frame, SearchStackFrame):
            return await self._handle_search_frame(tracker, user_message)
        
        # 没有特定栈帧，放弃处理
        return PolicyPrediction.abstain(self.name)
    
    async def _handle_search_frame(
        self,
        tracker: "DialogueStateTracker",
        user_message: str,
    ) -> PolicyPrediction:
        """处理SearchStackFrame - 执行检索。"""
        if not user_message:
            return PolicyPrediction.abstain(self.name)
        
        logger.info(f"[EnterpriseSearchPolicy] SearchStackFrame processing: {user_message}")
        
        try:
            # 尝试知识库检索
            if self.config.retrieval.enabled and self._retriever:
                search_results = await self._search(user_message, tracker)
                
                if search_results:
                    logger.info(f"[EnterpriseSearchPolicy] 检索到 {len(search_results)} 条结果，开始生成RAG回答")
                    answer = await self._generate_rag_answer(user_message, search_results)
                    logger.info(f"[EnterpriseSearchPolicy] RAG回答: {answer[:200] if answer else 'None'}...")
                    
                    if answer and "[NO_RAG_ANSWER]" not in answer:
                        # 检索成功，弹出栈帧
                        tracker.dialogue_stack.pop()
                        # 记录 Pattern 执行历史
                        tracker.record_pattern("search")
                        logger.debug("SearchStackFrame popped after successful retrieval")
                        
                        return PolicyPrediction(
                            action="action_send_text",
                            confidence=0.9,
                            policy_name=self.name,
                            metadata={
                                "text": answer,
                                "degradation_reason": DegradationReason.DEFAULT,
                                "search_results": [r.content for r in search_results],
                            },
                        )
            
            # 降级到闲聊
            if self.config.chitchat_enabled:
                chitchat_answer = await self._generate_chitchat_answer(user_message)
                if chitchat_answer:
                    tracker.dialogue_stack.pop()
                    # 记录 Pattern 执行历史
                    tracker.record_pattern("search")
                    logger.debug("SearchStackFrame popped after chitchat fallback")
                    
                    return PolicyPrediction(
                        action="action_send_text",
                        confidence=0.7,
                        policy_name=self.name,
                        metadata={
                            "text": chitchat_answer,
                            "degradation_reason": DegradationReason.CHITCHAT,
                        },
                    )
            
            # 无法处理
            tracker.dialogue_stack.pop()
            # 记录 Pattern 执行历史
            tracker.record_pattern("search")
            return PolicyPrediction(
                action=ACTION_DEFAULT_FALLBACK,
                confidence=0.5,
                policy_name=self.name,
                metadata={"degradation_reason": DegradationReason.CANNOT_HANDLE},
            )
            
        except Exception as e:
            logger.error(f"Search frame error: {e}")
            try:
                tracker.dialogue_stack.pop()
                # 记录 Pattern 执行历史
                tracker.record_pattern("search")
            except Exception:
                pass
            return PolicyPrediction(
                action=ACTION_DEFAULT_FALLBACK,
                confidence=0.3,
                policy_name=self.name,
                metadata={"degradation_reason": DegradationReason.INTERNAL_ERROR, "error": str(e)},
            )
    
    async def _handle_chitchat_frame(
        self,
        tracker: "DialogueStateTracker",
        user_message: str,
    ) -> PolicyPrediction:
        """处理ChitChatStackFrame - 生成闲聊回复。"""
        logger.debug(f"ChitChatStackFrame processing: {user_message}")
        
        # 弹出栈帧
        tracker.dialogue_stack.pop()
        # 记录 Pattern 执行历史
        tracker.record_pattern("chitchat")
        
        if not user_message:
            return PolicyPrediction(
                action="action_send_text",
                confidence=0.8,
                policy_name=self.name,
                metadata={"text": "你好！有什么可以帮您的吗？"},
            )
        
        try:
            chitchat_answer = await self._generate_chitchat_answer(user_message)
            if chitchat_answer:
                return PolicyPrediction(
                    action="action_send_text",
                    confidence=0.9,
                    policy_name=self.name,
                    metadata={
                        "text": chitchat_answer,
                        "degradation_reason": DegradationReason.CHITCHAT,
                    },
                )
        except Exception as e:
            logger.error(f"Chitchat generation error: {e}")
        
        # 默认回复
        return PolicyPrediction(
            action="action_send_text",
            confidence=0.7,
            policy_name=self.name,
            metadata={"text": "你好！很高兴和你聊天。"},
        )
    
    async def _handle_cannot_handle_frame(
        self,
        tracker: "DialogueStateTracker",
        frame: Any,
        domain: Optional["Domain"],
    ) -> PolicyPrediction:
        """处理CannotHandleStackFrame - 返回降级响应。"""
        logger.debug(f"CannotHandleStackFrame processing, reason: {getattr(frame, 'reason', '')}")
        
        # 弹出栈帧
        tracker.dialogue_stack.pop()
        # 记录 Pattern 执行历史
        tracker.record_pattern("cannot_handle")
        
        # 尝试从domain获取默认回复
        fallback_text = "抱歉，我没有理解您的意思。请换一种方式表达。"
        if domain:
            responses = domain.get_response("utter_default")
            if responses:
                import random
                fallback_text = random.choice(responses).text
        
        return PolicyPrediction(
            action="action_send_text",
            confidence=0.5,
            policy_name=self.name,
            metadata={
                "text": fallback_text,
                "degradation_reason": DegradationReason.CANNOT_HANDLE,
                "reason": getattr(frame, 'reason', ''),
            },
        )
    
    async def _handle_completed_frame(
        self,
        tracker: "DialogueStateTracker",
        frame: Any,
        domain: Optional["Domain"],
    ) -> PolicyPrediction:
        """处理CompletedStackFrame - 询问是否还有其他需求。
        
        当Flow完成后，系统会询问用户是否还有其他需要帮助的。
        """
        previous_flow = getattr(frame, 'previous_flow_name', '')
        logger.debug(f"CompletedStackFrame processing, previous_flow: {previous_flow}")
        
        # 弹出栈帧
        tracker.dialogue_stack.pop()
        # 记录 Pattern 执行历史
        tracker.record_pattern("completed")
        
        # 尝试从domain获取完成响应
        completed_text = "还有什么我可以帮您的吗？"
        if domain:
            responses = domain.get_response("utter_can_do_something_else")
            if responses:
                import random
                completed_text = random.choice(responses).text
        
        return PolicyPrediction(
            action="action_send_text",
            confidence=0.9,
            policy_name=self.name,
            metadata={
                "text": completed_text,
                "previous_flow": previous_flow,
            },
        )
    
    async def _handle_human_handoff_frame(
        self,
        tracker: "DialogueStateTracker",
        frame: Any,
        domain: Optional["Domain"],
    ) -> PolicyPrediction:
        """处理HumanHandoffStackFrame - 执行人工转接。
        
        当需要转接人工客服时，生成转接响应。
        """
        reason = getattr(frame, 'reason', '')
        logger.debug(f"HumanHandoffStackFrame processing, reason: {reason}")
        
        # 弹出栈帧
        tracker.dialogue_stack.pop()
        # 记录 Pattern 执行历史
        tracker.record_pattern("human_handoff")
        
        # 尝试从domain获取转接响应
        handoff_text = "好的，正在为您转接人工客服，请稍候..."
        if domain:
            responses = domain.get_response("utter_human_handoff")
            if responses:
                import random
                handoff_text = random.choice(responses).text
        
        return PolicyPrediction(
            action="action_send_text",
            confidence=0.95,
            policy_name=self.name,
            metadata={
                "text": handoff_text,
                "human_handoff": True,
                "reason": reason,
            },
        )
    
    async def _search(
        self,
        query: str,
        tracker: "DialogueStateTracker" = None,
    ) -> List[SearchResult]:
        """执行知识库搜索。
        
        Args:
            query: 搜索查询
            tracker: 对话状态追踪器（用于获取用户信息和历史对话）
            
        Returns:
            搜索结果列表
        """
        if not self._retriever:
            logger.debug("未配置检索器，跳过知识库搜索")
            return []
        
        try:
            logger.info(f"[EnterpriseSearchPolicy] 调用检索器: {type(self._retriever).__name__}")
            logger.info(f"[EnterpriseSearchPolicy] 查询: '{query}', top_k={self.config.retrieval.top_k}")
            
            # 构建 tracker_state 用于检索器获取用户信息和历史对话
            tracker_state = tracker.to_dict() if tracker else None
            
            # 调用检索器
            results = await self._retriever.search(
                query,
                top_k=self.config.retrieval.top_k,
                tracker_state=tracker_state,
            )
            
            logger.info(f"[EnterpriseSearchPolicy] 检索器返回 {len(results)} 条结果")
            
            # 过滤低相似度结果（score 为 None 时视为通过过滤）
            threshold = self.config.retrieval.similarity_threshold
            filtered = [
                r for r in results
                if r.score is None or r.score >= threshold
            ]
            
            logger.info(
                f"[EnterpriseSearchPolicy] 过滤后剩余 {len(filtered)} 条结果 "
                f"(阈值: {self.config.retrieval.similarity_threshold})"
            )
            
            return filtered
            
        except Exception as e:
            logger.error(f"[EnterpriseSearchPolicy] 搜索错误: {e}")
            return []
    
    async def _generate_rag_answer(
        self,
        question: str,
        search_results: List[SearchResult],
    ) -> Optional[str]:
        """生成RAG回答。
        
        Args:
            question: 用户问题
            search_results: 搜索结果
            
        Returns:
            生成的回答
        """
        if not search_results:
            return None
        
        # 构建上下文
        context_parts = []
        for i, result in enumerate(search_results, 1):
            source = result.source
            content = result.content
            # 格式：编号. 来源\n内容
            context_parts.append(f"{i}. {source}\n{content}")
        
        context = "\n\n".join(context_parts)
        logger.info(f"[EnterpriseSearchPolicy] RAG上下文:\n{context}")
        
        # 构建提示词
        prompt = self.RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )
        
        # 调用LLM
        try:
            response = await self.llm_client.complete([
                {"role": "user", "content": prompt}
            ])
            logger.info(f"[EnterpriseSearchPolicy] RAG回答: {response.content[:200] if response.content else 'None'}...")
            return response.content
        except Exception as e:
            logger.error(f"RAG generation error: {e}")
            return None
    
    async def _generate_chitchat_answer(self, message: str) -> Optional[str]:
        """生成闲聊回答。
        
        Args:
            message: 用户消息
            
        Returns:
            生成的回答
        """
        prompt = self.CHITCHAT_PROMPT_TEMPLATE.format(message=message)
        
        try:
            response = await self.llm_client.complete([
                {"role": "user", "content": prompt}
            ])
            return response.content
        except Exception as e:
            logger.error(f"Chitchat generation error: {e}")
            return None
    
    def set_retriever(self, retriever: Any) -> None:
        """设置检索器。
        
        Args:
            retriever: 检索器实例
        """
        self._retriever = retriever


# 导出
__all__ = [
    "EnterpriseSearchPolicy",
    "EnterpriseSearchPolicyConfig",
    "SearchResult",  # 从base_retriever导入
]
