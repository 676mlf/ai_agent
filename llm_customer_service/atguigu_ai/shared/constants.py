# -*- coding: utf-8 -*-
"""
constants - 全局常量定义

定义系统中使用的所有常量，包括：
- 默认路径和端口
- 降级原因常量
- 命令和模式名称
- 槽位类型
"""

from typing import Final

# =============================================================================
# 服务器配置
# =============================================================================

DEFAULT_SERVER_PORT: Final[int] = 5005
"""默认服务器端口"""

DEFAULT_SERVER_HOST: Final[str] = "0.0.0.0"
"""默认服务器主机地址"""

DEFAULT_SERVER_FORMAT: Final[str] = "{0}://localhost:{1}"
"""服务器URL格式模板"""

# =============================================================================
# 默认路径
# =============================================================================

DEFAULT_MODELS_PATH: Final[str] = "models"
"""默认模型存储路径"""

DEFAULT_CONFIG_PATH: Final[str] = "config.yml"
"""默认配置文件路径"""

DEFAULT_DOMAIN_PATH: Final[str] = "domain.yml"
"""默认Domain文件路径"""

DEFAULT_ENDPOINTS_PATH: Final[str] = "endpoints.yml"
"""默认端点配置文件路径"""

DEFAULT_DATA_PATH: Final[str] = "data"
"""默认训练数据路径"""

DEFAULT_FLOWS_PATH: Final[str] = "flows"
"""默认Flow定义路径"""

DEFAULT_ACTIONS_PATH: Final[str] = "actions"
"""默认Action模块路径"""

# =============================================================================
# 降级原因常量
# =============================================================================

class DegradationReason:
    """降级原因常量类。
    
    封装企业搜索策略中的降级原因常量。
    降级链: Flow -> RAG -> Chitchat -> CannotHandle
    """
    DEFAULT: str = "default"
    CHITCHAT: str = "chitchat"
    NOT_SUPPORTED: str = "not_supported"
    INVALID_INTENT: str = "invalid_intent"
    NO_RELEVANT_ANSWER: str = "no_relevant_answer"
    INTERNAL_ERROR: str = "internal_error"
    CANNOT_HANDLE: str = "cannot_handle"

# =============================================================================
# 命令类型常量
# =============================================================================

COMMAND_START_FLOW: Final[str] = "start_flow"
"""启动Flow命令"""

COMMAND_CANCEL_FLOW: Final[str] = "cancel_flow"
"""取消Flow命令"""

COMMAND_SET_SLOT: Final[str] = "set_slot"
"""设置槽位命令"""

COMMAND_CHITCHAT: Final[str] = "chitchat"
"""闲聊命令"""

COMMAND_KNOWLEDGE_ANSWER: Final[str] = "knowledge_answer"
"""知识库回答命令"""

COMMAND_CLARIFY: Final[str] = "clarify"
"""澄清命令"""

COMMAND_CANNOT_HANDLE: Final[str] = "cannot_handle"
"""无法处理命令"""

COMMAND_SESSION_START: Final[str] = "session_start"
"""会话开始命令"""

# =============================================================================
# 槽位类型常量
# =============================================================================

SLOT_TYPE_TEXT: Final[str] = "text"
"""文本槽位类型"""

SLOT_TYPE_BOOL: Final[str] = "bool"
"""布尔槽位类型"""

SLOT_TYPE_FLOAT: Final[str] = "float"
"""浮点槽位类型"""

SLOT_TYPE_LIST: Final[str] = "list"
"""列表槽位类型"""

SLOT_TYPE_ANY: Final[str] = "any"
"""任意类型槽位"""

SLOT_TYPE_CATEGORICAL: Final[str] = "categorical"
"""分类槽位类型"""

# =============================================================================
# Action名称常量
# =============================================================================

ACTION_LISTEN: Final[str] = "action_listen"
"""监听用户输入动作"""

ACTION_RESTART: Final[str] = "action_restart"
"""重启会话动作"""

ACTION_SESSION_START: Final[str] = "action_session_start"
"""会话开始动作"""

ACTION_DEFAULT_FALLBACK: Final[str] = "action_default_fallback"
"""默认降级动作"""

ACTION_DEACTIVATE_LOOP: Final[str] = "action_deactivate_loop"
"""停用循环动作"""

ACTION_BACK: Final[str] = "action_back"
"""返回上一步动作"""

# =============================================================================
# LLM相关常量
# =============================================================================

LLM_TYPE_OPENAI: Final[str] = "openai"
"""OpenAI类型（支持OpenAI API和兼容服务如vLLM）"""

LLM_TYPE_QWEN: Final[str] = "qwen"
"""通义千问类型（阿里云DashScope）"""

LLM_TYPE_AZURE: Final[str] = "azure"
"""Azure OpenAI类型"""

LLM_TYPE_ANTHROPIC: Final[str] = "anthropic"
"""Anthropic Claude类型"""

DEFAULT_LLM_TEMPERATURE: Final[float] = 0.0
"""默认LLM温度参数"""

DEFAULT_LLM_MAX_TOKENS: Final[int] = 1024
"""默认LLM最大token数"""

DEFAULT_LLM_TIMEOUT: Final[int] = 30
"""默认LLM请求超时(秒)"""

# =============================================================================
# Tracker Store类型常量
# =============================================================================

TRACKER_STORE_JSON: Final[str] = "json"
"""JSON本地存储类型"""

TRACKER_STORE_MYSQL: Final[str] = "mysql"
"""MySQL存储类型"""

TRACKER_STORE_MEMORY: Final[str] = "memory"
"""内存存储类型"""

# =============================================================================
# 通道名称常量
# =============================================================================

CHANNEL_REST: Final[str] = "rest"
"""REST API通道"""

CHANNEL_SOCKETIO: Final[str] = "socketio"
"""SocketIO通道"""

CHANNEL_CONSOLE: Final[str] = "console"
"""控制台通道"""

# =============================================================================
# 其他常量
# =============================================================================

DEFAULT_SENDER_ID: Final[str] = "default"
"""默认发送者ID"""

DEFAULT_ENCODING: Final[str] = "utf-8"
"""默认文件编码"""

SESSION_START_METADATA_SLOT: Final[str] = "session_started_metadata"
"""会话开始元数据槽位"""
