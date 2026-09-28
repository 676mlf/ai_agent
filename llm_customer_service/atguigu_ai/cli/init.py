# -*- coding: utf-8 -*-
"""
atguigu_ai init命令

用于初始化新的对话系统项目，创建项目结构和模板文件。
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional
from enum import Enum

import click

from atguigu_ai.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_MODELS_PATH,
    DEFAULT_ACTIONS_PATH,
    DEFAULT_FLOWS_PATH,
)


class ProjectTemplate(str, Enum):
    """项目模板类型。"""
    DEFAULT = "default"
    BASIC = "basic"
    TUTORIAL = "tutorial"
    
    def __str__(self) -> str:
        return self.value


# 默认的config.yml内容
DEFAULT_CONFIG_CONTENT = '''# Atguigu AI 配置文件
# LLM驱动的对话系统配置

assistant_id: "{assistant_id}"

# Pipeline配置 - 对话理解组件
pipeline:
  # LLM命令生成器：将用户输入转换为系统命令
  - name: "LLMCommandGenerator"
    llm: "default"  # 引用endpoints.yml中的模型

# 策略配置
policies:
  # Flow策略：处理对话流程
  - name: "FlowPolicy"
  # 企业搜索策略：处理知识检索和降级
  - name: "EnterpriseSearchPolicy"
    # vector_store: "my_module.MyRetriever"  # 自定义检索器全类名（可选）
    llm: "default"  # 引用endpoints.yml中的模型
    embeddings: "default"  # 引用endpoints.yml中的嵌入模型
'''

# 默认的domain.yml内容
DEFAULT_DOMAIN_CONTENT = '''# Domain配置文件
# 定义对话系统的槽位、动作、响应等

version: "1.0"

# 槽位定义
# 每个槽位必须包含: type（数据类型）和 mappings（映射方式）
slots:
  user_name:
    type: text
    mappings:
      - type: from_llm  # 由LLM从用户输入中提取
    description: "用户姓名"
  
  current_topic:
    type: categorical
    values:
      - greeting
      - help
      - goodbye
    mappings:
      - type: from_llm
    description: "当前对话主题"
  
  order_id:
    type: text
    mappings:
      - type: controlled  # 由Action代码控制填充
    description: "订单编号，由系统生成"

# 响应模板
responses:
  utter_greet:
    - text: "你好！我是Atguigu AI助手，有什么可以帮助你的吗？"
    - text: "您好！很高兴为您服务。"
  
  utter_goodbye:
    - text: "再见！祝您生活愉快！"
    - text: "感谢使用，再见！"
  
  utter_default:
    - text: "抱歉，我没有理解您的意思，能否换一种方式表达？"
  
  utter_ask_help:
    - text: "请问您需要什么帮助？"

# 动作列表
actions:
  - action_greet
  - action_goodbye
  - action_handle_help

# Forms (表单)
forms: ~
'''

# 默认的endpoints.yml内容
DEFAULT_ENDPOINTS_CONTENT = '''# Endpoints配置文件
# 定义模型服务和外部端点

# 模型组配置 - 统一管理所有LLM和嵌入模型
models:
  # 默认LLM模型
  default:
    type: "qwen"  # openai, qwen, azure, anthropic
    model: "qwen-plus"
    temperature: 0.0
    max_tokens: 1024
    # api_key: ${DASHSCOPE_API_KEY}  # 从环境变量读取
    # api_base: ""  # 自定义API地址（用于vLLM等自部署服务）
    # enable_thinking: false  # 启用深度思考模式
  
  # 可定义多个模型用于不同用途
  # rephrase:
  #   type: "qwen"
  #   model: "qwen-turbo"
  #   temperature: 0.7

# 嵌入模型配置
embeddings:
  default:
    type: "openai"
    model: "text-embedding-3-small"
    # api_key: ${OPENAI_API_KEY}

# NLG配置 - 响应生成和重述
nlg:
  # 响应重述功能
  rephrase:
    enabled: false
    model: "default"  # 引用上面models中的模型名称
    style: "friendly"  # friendly, professional, casual, empathetic
    language: "zh"

# 向量存储配置 - 传递给 retriever.connect() 的连接参数
# 检索器类名在 config.yml 的 policies.EnterpriseSearchPolicy.vector_store 中指定
vector_store:
  # 示例：Neo4j 连接配置
  # uri: "bolt://localhost:7687"
  # user: "neo4j"
  # password: "${NEO4J_PASSWORD}"
  # 示例：Elasticsearch 连接配置
  # hosts: ["http://localhost:9200"]
  # index: "knowledge_base"

# Tracker Store配置 - 对话状态存储
tracker_store:
  type: json  # json, mysql, memory
  path: "./tracker_store"
'''

# 默认的Flow文件内容
DEFAULT_FLOW_CONTENT = '''# Flows定义文件
# 定义对话流程

flows:
  greet_flow:
    description: "问候流程"
    steps:
      - id: "start"
        action: action_greet
        next: "end"
  
  help_flow:
    description: "帮助流程"
    steps:
      - id: "ask_topic"
        action: utter_ask_help
        next: "handle"
      - id: "handle"
        action: action_handle_help
        next: "end"
'''

# 默认的actions.py内容
DEFAULT_ACTIONS_CONTENT = '''# -*- coding: utf-8 -*-
"""
自定义Action实现

在这里定义你的自定义动作逻辑。
每个自定义动作需要继承Action基类并实现run方法。

注意：
- Action 类会在 Agent.load() 时自动发现和注册
- run 方法必须返回 ActionResult 对象
- 使用 result.add_response(text) 添加响应文本
"""

from typing import Any, Text

from atguigu_ai.agent.actions import Action, ActionResult


class ActionGreet(Action):
    """问候动作。"""
    
    @property
    def name(self) -> Text:
        return "action_greet"
    
    async def run(
        self,
        tracker: Any,
        domain: Any,
        **kwargs: Any,
    ) -> ActionResult:
        """执行问候动作。
        
        根据用户是否已知，返回个性化问候。
        """
        result = ActionResult()
        user_name = tracker.get_slot("user_name") if hasattr(tracker, 'get_slot') else None
        if user_name:
            result.add_response(f"你好，{user_name}！有什么可以帮助你的吗？")
        else:
            result.add_response("你好！我是Atguigu AI助手，有什么可以帮助你的吗？")
        return result


class ActionGoodbye(Action):
    """告别动作。"""
    
    @property
    def name(self) -> Text:
        return "action_goodbye"
    
    async def run(
        self,
        tracker: Any,
        domain: Any,
        **kwargs: Any,
    ) -> ActionResult:
        """执行告别动作。"""
        result = ActionResult()
        result.add_response("再见！祝您生活愉快！")
        return result


class ActionHandleHelp(Action):
    """处理帮助请求的动作。"""
    
    @property
    def name(self) -> Text:
        return "action_handle_help"
    
    async def run(
        self,
        tracker: Any,
        domain: Any,
        **kwargs: Any,
    ) -> ActionResult:
        """处理帮助请求。
        
        根据用户最新消息提供相应帮助。
        """
        result = ActionResult()
        latest_message = tracker.latest_message if hasattr(tracker, 'latest_message') else None
        if latest_message:
            text = latest_message.text if hasattr(latest_message, 'text') else ""
            if text:
                result.add_response(f"好的，我来帮您处理关于'{text}'的问题。")
                return result
        result.add_response("请告诉我您需要什么帮助。")
        return result
'''

# README内容
DEFAULT_README_CONTENT = '''# Atguigu AI 项目

LLM驱动的对话系统项目。

## 项目结构

```
.
├── config.yml          # 系统配置（pipeline、policies）
├── domain.yml          # Domain定义（slots、responses、actions）
├── endpoints.yml       # 端点配置（models、nlg、tracker_store）
├── data/
│   └── flows.yml       # Flow定义
├── actions/
│   └── actions.py      # 自定义动作
└── models/             # 训练后的模型
```

## 快速开始

1. 安装依赖:
   ```bash
   pip install atguigu-ai
   ```

2. 配置环境变量:
   ```bash
   # 阿里云DashScope
   export DASHSCOPE_API_KEY=your-api-key
   
   # 或 OpenAI
   export OPENAI_API_KEY=your-api-key
   ```

3. 训练模型:
   ```bash
   atguigu train
   ```

4. 运行服务:
   ```bash
   atguigu run
   ```

5. 交互测试:
   ```bash
   atguigu shell
   ```

## 配置说明

### 模型配置 (endpoints.yml)

在 `endpoints.yml` 中配置LLM模型:

```yaml
models:
  default:
    type: "qwen"  # openai, qwen, azure, anthropic
    model: "qwen-plus"
    # api_key: ${DASHSCOPE_API_KEY}
    # api_base: "http://localhost:8000/v1"  # vLLM等自部署服务
    # enable_thinking: false  # 深度思考模式
```

### 槽位配置 (domain.yml)

```yaml
slots:
  user_name:
    type: text
    mappings:
      - type: from_llm  # 由LLM提取
    description: "用户姓名"
  
  order_id:
    type: text
    mappings:
      - type: controlled  # 由Action控制
    description: "订单编号"
```

### Pipeline配置 (config.yml)

```yaml
pipeline:
  - name: "LLMCommandGenerator"
    llm: "default"  # 引用endpoints.yml中的模型

policies:
  - name: "FlowPolicy"
  - name: "EnterpriseSearchPolicy"
    vector_store: "faiss"
    llm: "default"
```

## 更多信息

请参考 Atguigu AI 文档获取更多信息。
'''


def generate_assistant_id() -> str:
    """生成随机的助手ID。"""
    import random
    import string
    
    # 生成类似 "happy-robot-123" 的ID
    adjectives = ['happy', 'smart', 'quick', 'calm', 'bright', 'wise', 'kind']
    nouns = ['robot', 'assistant', 'helper', 'agent', 'bot', 'buddy']
    
    adj = random.choice(adjectives)
    noun = random.choice(nouns)
    num = ''.join(random.choices(string.digits, k=3))
    
    return f"{adj}-{noun}-{num}"


def create_project_structure(
    path: Path,
    template: ProjectTemplate = ProjectTemplate.DEFAULT
) -> None:
    """创建项目目录结构和模板文件。
    
    Args:
        path: 项目路径
        template: 项目模板类型
    """
    # 创建目录
    directories = [
        path / "data",
        path / "actions",
        path / "models",
        path / "tests",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    # 生成assistant_id
    assistant_id = generate_assistant_id()
    
    # 创建配置文件
    files = {
        path / "config.yml": DEFAULT_CONFIG_CONTENT.format(assistant_id=assistant_id),
        path / "domain.yml": DEFAULT_DOMAIN_CONTENT,
        path / "endpoints.yml": DEFAULT_ENDPOINTS_CONTENT,
        path / "data" / "flows.yml": DEFAULT_FLOW_CONTENT,
        path / "actions" / "actions.py": DEFAULT_ACTIONS_CONTENT,
        path / "actions" / "__init__.py": "# Custom actions module\n",
        path / "README.md": DEFAULT_README_CONTENT,
    }
    
    for file_path, content in files.items():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
    
    # 创建.gitignore
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.env

# Models
models/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Tracker store
tracker_store/

# Logs
*.log
logs/
'''
    (path / ".gitignore").write_text(gitignore_content, encoding='utf-8')


@click.command(name='init')
@click.option(
    '--path', '-p',
    type=click.Path(),
    default='.',
    help='项目创建路径，默认为当前目录'
)
@click.option(
    '--template', '-t',
    type=click.Choice([t.value for t in ProjectTemplate]),
    default=ProjectTemplate.DEFAULT.value,
    help='项目模板类型'
)
@click.option(
    '--no-prompt',
    is_flag=True,
    default=False,
    help='跳过交互式提示，使用默认值'
)
@click.pass_context
def init_command(
    ctx: click.Context,
    path: str,
    template: str,
    no_prompt: bool
) -> None:
    """初始化新的Atguigu AI项目。
    
    创建项目目录结构和基本配置文件，包括：
    - config.yml: 系统配置
    - domain.yml: Domain定义
    - endpoints.yml: 端点配置
    - data/flows.yml: Flow定义
    - actions/actions.py: 自定义动作
    
    示例:
        atguigu init                    # 在当前目录初始化
        atguigu init --path ./my_bot    # 在指定目录初始化
        atguigu init --template basic   # 使用basic模板
    """
    click.echo("欢迎使用 Atguigu AI!\n")
    click.echo("正在创建新项目...\n")
    
    project_path = Path(path).resolve()
    
    # 检查路径是否存在
    if not project_path.exists():
        if no_prompt:
            click.echo(f"创建目录: {project_path}")
            project_path.mkdir(parents=True, exist_ok=True)
        else:
            if click.confirm(f"目录 '{project_path}' 不存在，是否创建？"):
                project_path.mkdir(parents=True, exist_ok=True)
            else:
                click.echo("已取消。")
                sys.exit(0)
    
    # 检查目录是否为空
    if project_path.exists() and any(project_path.iterdir()):
        if not no_prompt:
            if not click.confirm(f"目录 '{project_path}' 不为空，是否继续？"):
                click.echo("已取消。")
                sys.exit(0)
    
    # 创建项目结构
    try:
        template_enum = ProjectTemplate(template)
        create_project_structure(project_path, template_enum)
        
        click.echo(f"[OK] 项目已创建于: {project_path}")
        click.echo("")
        click.echo("项目结构:")
        click.echo("  +-- config.yml        # 系统配置（pipeline、policies）")
        click.echo("  +-- domain.yml        # Domain定义（slots、responses）")
        click.echo("  +-- endpoints.yml     # 端点配置（models、nlg、tracker）")
        click.echo("  +-- data/")
        click.echo("  |   +-- flows.yml     # Flow定义")
        click.echo("  +-- actions/")
        click.echo("  |   +-- actions.py    # 自定义动作")
        click.echo("  +-- models/           # 模型目录")
        click.echo("")
        click.echo("下一步:")
        click.echo("  1. 编辑 config.yml 配置LLM")
        click.echo("  2. 设置环境变量 OPENAI_API_KEY")
        click.echo("  3. 运行 'atguigu train' 训练模型")
        click.echo("  4. 运行 'atguigu shell' 测试对话")
        click.echo("")
        
    except Exception as e:
        click.echo(f"错误: 创建项目失败 - {e}", err=True)
        if ctx.obj.get('debug'):
            raise
        sys.exit(1)


# 导出
__all__ = ['init_command', 'ProjectTemplate', 'create_project_structure']
