# -*- coding: utf-8 -*-
"""
Agent主类

提供对话系统的核心Agent实现。
基于 LangGraph 图式编排核心组件的执行流程。
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from atguigu_ai.agent.message_processor import (
    MessageProcessor,
    ProcessorConfig,
    MessageResponse,
)
from atguigu_ai.agent.actions import register_action, Action
from atguigu_ai.agent.graph import (
    get_message_processing_graph,
    create_initial_state,
)
from atguigu_ai.core.tracker import DialogueStateTracker
from atguigu_ai.core.domain import Domain
from atguigu_ai.core.stores import create_tracker_store, TrackerStore
from atguigu_ai.dialogue_understanding.flow import FlowsList, FlowLoader
from atguigu_ai.dialogue_understanding.generator import LLMCommandGenerator
from atguigu_ai.dialogue_understanding.processor import CommandProcessor
from atguigu_ai.policies import PolicyEnsemble, FlowPolicy, EnterpriseSearchPolicy
from atguigu_ai.shared.yaml_loader import read_yaml_file
from atguigu_ai.shared.config import AtguiguConfig, LLMConfig

logger = logging.getLogger(__name__)


def _load_custom_actions(actions_path: Path) -> List[str]:
    """从用户工程的 actions 目录自动加载自定义 Action。
    
    扫描指定目录下的所有 Python 文件，发现继承自 Action 基类的类，
    自动实例化并注册。
    
    Args:
        actions_path: actions 目录路径
        
    Returns:
        成功注册的 Action 名称列表
    """
    if not actions_path.exists() or not actions_path.is_dir():
        return []
    
    registered_actions = []
    
    # 将 actions 目录的父目录添加到 sys.path，以便正确导入
    parent_path = str(actions_path.parent)
    if parent_path not in sys.path:
        sys.path.insert(0, parent_path)
    
    try:
        # 扫描 actions 目录下的所有 .py 文件
        for py_file in actions_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue  # 跳过 __init__.py 等
            
            module_name = f"actions.{py_file.stem}"
            
            try:
                # 动态导入模块
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                # 扫描模块中的类，找到继承自 Action 的类
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # 检查是否是 Action 的子类（但不是 Action 本身）
                    if (issubclass(obj, Action) and 
                        obj is not Action and
                        obj.__module__ == module_name):
                        try:
                            # 实例化并注册
                            action_instance = obj()
                            register_action(action_instance)
                            logger.info(f"Registered custom action: {action_instance.name}")
                            registered_actions.append(action_instance.name)
                        except Exception as e:
                            logger.warning(f"Failed to register action {name}: {e}")
                            
            except Exception as e:
                logger.warning(f"Failed to load actions from {py_file}: {e}")
                
    finally:
        # 清理 sys.path（可选，保留以便后续使用）
        pass
    
    return registered_actions


@dataclass
class AgentConfig:
    """Agent配置。
    
    Attributes:
        domain_path: Domain文件路径
        flows_path: Flows文件/目录路径
        config_path: 配置文件路径
        endpoints_path: 端点配置路径
        tracker_store_type: Tracker存储类型
        tracker_store_config: Tracker存储配置
        llm_config: LLM配置
    """
    domain_path: str = "domain.yml"
    flows_path: str = "data/flows"
    config_path: str = "config.yml"
    endpoints_path: str = "endpoints.yml"
    tracker_store_type: str = "memory"
    tracker_store_config: Dict[str, Any] = field(default_factory=dict)
    llm_config: Optional[LLMConfig] = None


class Agent:
    """对话系统Agent。
    
    Agent是对话系统的核心类，负责：
    - 加载和管理配置
    - 处理用户消息
    - 管理对话状态
    - 协调各个组件
    
    使用示例：
    ```python
    agent = Agent.load("./my_bot")
    response = await agent.handle_message("你好", sender_id="user1")
    print(response.messages)
    ```
    """
    
    def __init__(
        self,
        domain: Optional[Domain] = None,
        flows: Optional[FlowsList] = None,
        tracker_store: Optional[TrackerStore] = None,
        policy_ensemble: Optional[PolicyEnsemble] = None,
        command_generator: Optional[LLMCommandGenerator] = None,
        nlg_generator: Optional[Any] = None,
        config: Optional[AgentConfig] = None,
    ):
        """初始化Agent。
        
        Args:
            domain: Domain定义
            flows: Flow列表
            tracker_store: Tracker存储
            policy_ensemble: 策略集成器
            command_generator: 命令生成器
            nlg_generator: NLG生成器（可选，用于响应重述）
            config: Agent配置
        """
        self.domain = domain or Domain()
        self.flows = flows or FlowsList()
        self.config = config or AgentConfig()
        
        # 初始化Tracker存储
        if tracker_store:
            self.tracker_store = tracker_store
        else:
            self.tracker_store = create_tracker_store(
                self.config.tracker_store_type,
                **self.config.tracker_store_config,
            )
        self.tracker_store.set_domain(self.domain)
        
        # 初始化策略
        if policy_ensemble:
            self.policy_ensemble = policy_ensemble
        else:
            self.policy_ensemble = PolicyEnsemble(policies=[
                FlowPolicy(flows=self.flows),
                EnterpriseSearchPolicy(),
            ])
        
        # 初始化命令生成器
        self.command_generator = command_generator
        
        # 初始化NLG生成器
        self.nlg_generator = nlg_generator
        
        # 初始化命令处理器
        self.command_processor = CommandProcessor(
            domain=self.domain,
            flows=self.flows.flows if self.flows else [],
        )
        
        # 获取 LangGraph 消息处理图（惰性初始化的单例）
        self.graph = get_message_processing_graph()
        
        # 保留消息处理器作为备用（向后兼容）
        self.message_processor = MessageProcessor(
            domain=self.domain,
            flows=self.flows,
            policy_ensemble=self.policy_ensemble,
            command_generator=self.command_generator,
        )
    
    async def handle_message(
        self,
        message: str,
        sender_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MessageResponse:
        """处理用户消息。
        
        使用 LangGraph 图式编排执行消息处理流程。
        
        Args:
            message: 用户消息文本
            sender_id: 发送者ID
            metadata: 消息元数据
            
        Returns:
            处理响应
        """
        # 获取或创建Tracker
        tracker = await self.tracker_store.get_or_create_tracker(sender_id)
        
        # 构建初始状态
        initial_state = create_initial_state(
            tracker=tracker,
            input_message=message,
            domain=self.domain,
            flows=self.flows,
            metadata=metadata,
            max_actions=10,
            command_generator=self.command_generator,
            command_processor=self.command_processor,
            policy_ensemble=self.policy_ensemble,
        )
        
        # 执行图
        logger.info(f"[Agent] 使用 LangGraph 处理消息: {message[:50]}...")
        final_state = await self.graph.ainvoke(initial_state)
        
        # 从最终状态提取结果
        updated_tracker = final_state.get("tracker", tracker)
        final_responses = final_state.get("final_responses", [])
        node_history = final_state.get("node_history", [])
        error = final_state.get("error")
        
        # 保存Tracker
        await self.tracker_store.save(updated_tracker)
        
        # 构建响应
        response = MessageResponse(
            messages=final_responses,
            metadata={
                "node_history": node_history,
                "error": error,
            },
        )
        
        logger.info(
            f"[Agent] 处理完成, 节点路径: {' -> '.join(node_history)}, "
            f"响应数: {len(final_responses)}"
        )
        
        return response
    
    def handle_message_sync(
        self,
        message: str,
        sender_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MessageResponse:
        """同步版本的消息处理。"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.handle_message(message, sender_id, metadata)
        )
    
    async def get_tracker(self, sender_id: str) -> Optional[DialogueStateTracker]:
        """获取指定用户的Tracker。
        
        Args:
            sender_id: 发送者ID
            
        Returns:
            Tracker实例，如果不存在则返回None
        """
        return await self.tracker_store.retrieve(sender_id)
    
    async def reset_tracker(self, sender_id: str) -> None:
        """重置指定用户的对话状态。
        
        Args:
            sender_id: 发送者ID
        """
        tracker = await self.tracker_store.retrieve(sender_id)
        if tracker:
            tracker.restart()
            await self.tracker_store.save(tracker)
    
    def register_action(self, action: Action) -> None:
        """注册自定义动作。
        
        Args:
            action: 动作实例
        """
        register_action(action)
    
    @classmethod
    def load(
        cls,
        project_path: Union[str, Path],
        config: Optional[AgentConfig] = None,
    ) -> "Agent":
        """从项目目录或模型压缩包加载Agent。
        
        支持以下输入：
        - .tar.gz 模型压缩包路径
        - 包含 .tar.gz 文件的目录（自动选择最新）
        - 项目目录（直接加载配置文件）
        
        Args:
            project_path: 项目目录路径或模型压缩包路径
            config: Agent配置（覆盖默认值）
            
        Returns:
            Agent实例
        """
        import tempfile
        from atguigu_ai.training.model_storage import (
            extract_model_archive,
            get_model_path,
            get_latest_model,
        )
        
        project_path = Path(project_path)
        
        if config is None:
            config = AgentConfig()
        
        # 确定实际的工作目录
        # 情况1: 输入是 .tar.gz 文件
        if project_path.is_file() and project_path.name.endswith(".tar.gz"):
            logger.info(f"Loading agent from model archive: {project_path}")
            # 解压到临时目录
            temp_dir = tempfile.mkdtemp(prefix="atguigu_model_")
            extract_model_archive(project_path, temp_dir)
            working_path = Path(temp_dir)
            logger.info(f"Extracted model to: {working_path}")
        
        # 情况2: 输入是目录
        elif project_path.is_dir():
            # 检查是否有 models/ 子目录包含 .tar.gz 文件
            models_dir = project_path / "models"
            latest_model = None
            if models_dir.exists():
                latest_model = get_latest_model(models_dir)
            
            if latest_model:
                # 找到了模型压缩包，解压并使用
                logger.info(f"Found model archive: {latest_model}")
                temp_dir = tempfile.mkdtemp(prefix="atguigu_model_")
                extract_model_archive(latest_model, temp_dir)
                working_path = Path(temp_dir)
                logger.info(f"Extracted model to: {working_path}")
            else:
                # 没有找到压缩包，直接使用项目目录（向后兼容）
                working_path = project_path
                logger.info(f"Loading agent from project directory: {project_path}")
        else:
            raise FileNotFoundError(f"Path not found: {project_path}")
        
        logger.info(f"Working path: {working_path}")
        
        # 将项目目录和工作目录添加到 sys.path，以便加载用户自定义模块（如 addons/）
        # 注意：当使用模型压缩包时，working_path 是临时目录，但用户自定义代码在原始 project_path 中
        project_path_str = str(project_path.absolute())
        working_path_str = str(working_path.absolute())
        
        # 优先添加原始项目目录（用户自定义代码所在位置）
        if project_path_str not in sys.path:
            sys.path.insert(0, project_path_str)
            logger.info(f"Added project path to sys.path: {project_path_str}")
        
        # 如果工作目录与项目目录不同，也添加工作目录
        if working_path_str != project_path_str and working_path_str not in sys.path:
            sys.path.insert(0, working_path_str)
            logger.info(f"Added working path to sys.path: {working_path_str}")
        
        # 加载Domain
        # 支持两种格式: domain.yml 文件或 domain/ 目录
        domain_path = working_path / config.domain_path
        domain = None
        if domain_path.exists():
            domain = Domain.load(str(domain_path))
            logger.info(f"Loaded domain from {domain_path}")
        else:
            # 如果配置的路径不存在，尝试查找 domain 目录（兼容模型压缩包）
            domain_dir = working_path / "domain"
            if domain_dir.exists() and domain_dir.is_dir():
                domain = Domain.load(str(domain_dir))
                logger.info(f"Loaded domain from {domain_dir}")
        
        # 加载Flows
        flows_path = working_path / config.flows_path
        flows = FlowsList()
        if flows_path.exists():
            loader = FlowLoader()
            flows = loader.load(flows_path)
            logger.info(f"Loaded {len(flows)} flows from {flows_path}")
        
        # 加载用户自定义 Actions
        # 自动发现 actions/ 目录中的 Action 类并注册
        actions_path = working_path / "actions"
        custom_action_names = _load_custom_actions(actions_path)
        if custom_action_names:
            logger.info(f"Loaded {len(custom_action_names)} custom actions from {actions_path}")
            # 将自动发现的 actions 同步到 domain 中
            if domain:
                for action_name in custom_action_names:
                    domain.add_action(action_name)
                logger.debug(f"Synced custom actions to domain: {custom_action_names}")
        
        # 加载endpoints配置（包含模型定义）
        endpoints_path = working_path / config.endpoints_path
        from atguigu_ai.shared.config import EndpointsConfig
        endpoints_config = EndpointsConfig.load(endpoints_path) if endpoints_path.exists() else EndpointsConfig()
        
        # 加载config配置
        config_path = working_path / config.config_path
        llm_config = None
        retrieval_config = None
        nlg_config = None
        enterprise_llm_config = None
        enterprise_embeddings_config = None
        retriever_class_path = None
        if config_path.exists():
            config_data = read_yaml_file(str(config_path))
            if config_data:
                # 从 pipeline 配置中获取 LLMCommandGenerator 的 llm 引用
                pipeline = config_data.get("pipeline", [])
                for component in pipeline:
                    if component.get("name") == "LLMCommandGenerator":
                        llm_ref = component.get("llm", "default")
                        llm_config = endpoints_config.get_model_config(llm_ref)
                        if llm_config:
                            logger.info(f"从 pipeline 配置加载 LLM '{llm_ref}'")
                        else:
                            logger.warning(f"endpoints.yml 中未找到模型 '{llm_ref}'")
                        break
                
                # 从 policies 配置中获取 EnterpriseSearchPolicy 的参数
                policies = config_data.get("policies", [])
                retriever_class_path = None
                for policy in policies:
                    if policy.get("name") == "EnterpriseSearchPolicy":
                        # 获取检索器类路径
                        retriever_class_path = policy.get("vector_store")
                        if retriever_class_path:
                            logger.info(f"从 policies 配置读取检索器类: {retriever_class_path}")
                        
                        # 获取策略的 llm 引用
                        policy_llm_ref = policy.get("llm", "default")
                        enterprise_llm_config = endpoints_config.get_model_config(policy_llm_ref)
                        if enterprise_llm_config:
                            logger.info(f"从 policies 配置加载 EnterpriseSearchPolicy LLM '{policy_llm_ref}'")
                        
                        # 获取策略的 embeddings 引用
                        policy_embeddings_ref = policy.get("embeddings", "default")
                        enterprise_embeddings_config = endpoints_config.get_embeddings_config(policy_embeddings_ref)
                        if enterprise_embeddings_config:
                            logger.info(f"从 policies 配置加载 EnterpriseSearchPolicy embeddings '{policy_embeddings_ref}'")
                        break
                
                # 加载检索配置
                if "retrieval" in config_data:
                    from atguigu_ai.shared.config import RetrievalConfig
                    retrieval_config = RetrievalConfig.from_dict(config_data.get("retrieval", {}))
        
        # 从 endpoints.yml 获取 NLG 配置
        nlg_config = endpoints_config.nlg
        
        # 创建命令生成器
        command_generator = None
        if llm_config:
            from atguigu_ai.dialogue_understanding.generator import (
                LLMCommandGenerator,
                LLMGeneratorConfig,
            )
            generator_config = LLMGeneratorConfig(
                type=llm_config.type,
                model=llm_config.model,
                api_key=llm_config.api_key,
                api_base=llm_config.api_base,
                temperature=llm_config.temperature,
                enable_thinking=llm_config.enable_thinking,
            )
            command_generator = LLMCommandGenerator(config=generator_config)
        
        # 从 endpoints.yml 获取 Tracker 存储配置
        tracker_store_config = endpoints_config.tracker_store
        tracker_store = create_tracker_store(
            tracker_store_config.type,
            path=tracker_store_config.path,
        )
        logger.info(f"创建 TrackerStore: type={tracker_store_config.type}, path={tracker_store_config.path}")
        
        # 创建策略
        from atguigu_ai.policies import EnterpriseSearchPolicyConfig
        flow_policy = FlowPolicy(flows=flows)
        
        # 创建 Retriever（类路径从 config.yml policies 读取，连接配置从 endpoints.yml 读取）
        retriever = None
        if retriever_class_path:
            try:
                from atguigu_ai.retrieval import create_retriever
                connect_config = endpoints_config.vector_store.to_connect_config()
                retriever = create_retriever(retriever_class_path, connect_config)
                if retriever:
                    logger.info(f"创建检索器: {retriever_class_path}")
            except Exception as e:
                logger.warning(f"创建检索器失败: {e}")
        
        # 创建NLG生成器（如果配置了重述）
        nlg_generator = None
        if nlg_config and nlg_config.rephrase_enabled:
            try:
                from atguigu_ai.nlg import ResponseRephraser, RephraserConfig, TemplateNLG
                
                # 获取重述用的LLM配置
                rephrase_llm_config = None
                if nlg_config.rephrase_model:
                    rephrase_llm_config = endpoints_config.get_model_config(nlg_config.rephrase_model)
                if not rephrase_llm_config and llm_config:
                    rephrase_llm_config = llm_config  # 回退到主LLM配置
                
                if rephrase_llm_config:
                    rephrase_config = RephraserConfig(
                        enabled=True,
                        llm_type=rephrase_llm_config.type,
                        llm_model=rephrase_llm_config.model,
                        style=nlg_config.rephrase_style,
                        rephrase_threshold=nlg_config.rephrase_threshold,
                        preserve_slots=nlg_config.preserve_slots,
                        language=nlg_config.language,
                    )
                    
                    # 创建LLM客户端
                    from atguigu_ai.shared.llm import create_llm_client
                    rephrase_llm = create_llm_client(
                        type=rephrase_llm_config.type,
                        model=rephrase_llm_config.model,
                        api_key=rephrase_llm_config.api_key,
                        api_base=rephrase_llm_config.api_base,
                        temperature=0.7,  # 重述使用较高温度
                    )
                    
                    # 创建模板NLG作为底层
                    template_nlg = TemplateNLG(domain=domain)
                    
                    nlg_generator = ResponseRephraser(
                        config=rephrase_config,
                        base_generator=template_nlg,
                        llm_client=rephrase_llm,
                    )
                    logger.info(f"Loaded NLG rephraser with style: {nlg_config.rephrase_style}")
            except Exception as e:
                logger.warning(f"Failed to create NLG generator: {e}")
        
        # 使用 policies 配置中的 LLM 配置创建 EnterpriseSearchPolicy
        # 优先使用 policies 中指定的 llm，否则回退到 pipeline 中的 llm
        policy_llm_config = enterprise_llm_config or llm_config
        if policy_llm_config:
            enterprise_config = EnterpriseSearchPolicyConfig(
                llm_type=policy_llm_config.type,
                llm_model=policy_llm_config.model,
            )
            from atguigu_ai.shared.llm import create_llm_client
            llm_client = create_llm_client(
                type=policy_llm_config.type,
                model=policy_llm_config.model,
                api_key=policy_llm_config.api_key,
                api_base=policy_llm_config.api_base,
                temperature=policy_llm_config.temperature,
                enable_thinking=policy_llm_config.enable_thinking,
            )
            enterprise_policy = EnterpriseSearchPolicy(
                config=enterprise_config,
                llm_client=llm_client,
                retriever=retriever,
            )
            logger.info(f"创建 EnterpriseSearchPolicy: llm={policy_llm_config.model}")
        else:
            enterprise_policy = EnterpriseSearchPolicy(retriever=retriever)
        
        policy_ensemble = PolicyEnsemble(policies=[
            flow_policy,
            enterprise_policy,
        ])
        
        return cls(
            domain=domain,
            flows=flows,
            tracker_store=tracker_store,
            policy_ensemble=policy_ensemble,
            command_generator=command_generator,
            nlg_generator=nlg_generator,
            config=config,
        )
    
    @classmethod
    async def create(
        cls,
        domain: Optional[Domain] = None,
        flows: Optional[FlowsList] = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> "Agent":
        """创建Agent实例的便捷方法。
        
        Args:
            domain: Domain定义
            flows: Flow列表
            llm_provider: LLM提供商
            llm_model: LLM模型
            **kwargs: 额外配置
            
        Returns:
            Agent实例
        """
        from atguigu_ai.dialogue_understanding.generator import (
            LLMCommandGenerator,
            LLMGeneratorConfig,
        )
        
        # 创建命令生成器
        generator_config = LLMGeneratorConfig(
            provider=llm_provider,
            model=llm_model,
        )
        command_generator = LLMCommandGenerator(config=generator_config)
        
        return cls(
            domain=domain,
            flows=flows,
            command_generator=command_generator,
        )


# 导出
__all__ = [
    "Agent",
    "AgentConfig",
]
