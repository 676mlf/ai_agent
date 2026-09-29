# LLM 智能对话系统（atguigu_ai）

基于 **LLM 驱动**的教学版对话系统，采用 LangGraph 图式编排消息处理流程，内置一个完整的**电商客服 Demo**（订单 / 物流 / 售后），适合学习和理解现代对话系统（任务型对话 + RAG 问答）的核心原理。

> 版本：0.1.0 ｜ Python ≥ 3.10 ｜ License: MIT

## 项目亮点

- **图式对话流程**：基于 LangGraph 将「理解 → 策略 → 动作 → 响应 → 守卫」编排为可视化状态图
- **LLM 命令生成**：通过 LLM 将用户自然语言解析为结构化命令（槽位填充），支持 Flow 与自由对话混合
- **双策略引擎**：`FlowPolicy`（任务型流程对话）+ `EnterpriseSearchPolicy`（RAG 企业知识库检索），带降级链 Flow → RAG → Chitchat → CannotHandle
- **对话理解模块（DU）**：命令生成、命令处理、Flow 执行器、对话栈（Dialogue Stack）
- **检索增强**：本地 Embedding（bge-base-zh-v1.5）+ FAISS / Neo4j GraphRAG 知识图谱检索
- **多通道接入**：REST API、WebSocket(SocketIO)、Console，并提供可视化调试页面
- **完整工程化**：Click CLI、FastAPI 服务、YAML 配置、MySQL/Neo4j 存储、模型训练与导出

## 目录结构

```
.
├── README.md
├── llm_customer_service/
│   ├── setup.py                     # 包安装配置（安装后提供 atguigu 命令）
│   ├── requirements-atguigu.txt     # 依赖列表
│   ├── atguigu_ai/                  # 核心框架包
│   │   ├── agent/                   # 对话代理（LangGraph 图式消息处理）
│   │   │   └── graph/               #   图构建、状态、节点（understand/policy/action/response/guard）
│   │   ├── dialogue_understanding/  # 对话理解 DU（命令生成/处理、Flow 执行、对话栈）
│   │   ├── core/                    # Tracker、Domain、Slot、Store
│   │   ├── policies/                # FlowPolicy / EnterpriseSearchPolicy / 策略集成
│   │   ├── nlg/                     # 自然语言生成
│   │   ├── retrieval/               # 向量检索（Embedder、FAISS/GraphRAG）
│   │   ├── training/                # 训练与微调数据生成
│   │   ├── api/                     # FastAPI 服务（/api/messages、/inspect 调试页）
│   │   ├── channels/                # REST / SocketIO / Console 通道
│   │   ├── cli/                     # atguigu 命令行（init/train/run/shell/export/inspect）
│   │   └── shared/                  # 配置、常量、LLM 客户端、代理处理
│   └── ecs_demo/                    # 电商客服 Demo（订单/物流/售后）
│       ├── config.yml               # Pipeline 与策略配置
│       ├── endpoints.yml            # LLM / Neo4j / MySQL 端点配置
│       ├── domain/                  # Domain 定义（槽位、表单、响应）
│       ├── data/flows/              # Flow 流程定义（YAML）
│       ├── actions/                 # 自定义 Action（查订单、物流、售后等）
│       ├── addons/                  # GraphRAG 检索与索引构建
│       ├── models/                  # 嵌入模型与训练产物
│       └── gen_data.py              # 测试数据生成
```

## 环境要求

- Python **3.10+**
- （可选，企业知识库检索需要）Neo4j 5.x、MySQL 8.x
- （可选）OpenAI 兼容的 LLM API Key，如[硅基流动](https://cloud.siliconflow.cn/)、阿里云百炼等

## 安装

```bash
# 1. 克隆仓库
git clone https://github.com/676mlf/ai_agent.git
cd ai_agent/llm_customer_service

# 2. （推荐）创建虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. 以开发模式安装（安装依赖并注册 atguigu 命令）
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 依赖较多（含 torch、sentence-transformers），如不使用本地嵌入模型，可参考 `requirements-atguigu.txt` 注释精简安装。

## 配置

在 `llm_customer_service/ecs_demo/` 目录下创建 `.env` 文件（该文件已被 `.gitignore` 忽略，不会提交密钥）：

```env
# LLM API Key（硅基流动控制台: https://cloud.siliconflow.cn/account/ak）
SILICONFLOW_API_KEY=sk-xxxx
LLM_API_BASE=https://api.siliconflow.cn/v1
LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507

# GraphRAG Cypher 生成模型（可与 LLM_MODEL 相同）
CYPHER_LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507

# 本地/国内模型服务直连，避免系统代理导致 ProxyError
NO_PROXY=*
no_proxy=*

# 可选：企业知识库检索（GraphRAG / 业务数据库）
NEO4J_PASSWORD=your-neo4j-password
MYSQL_PASSWORD=your-mysql-password
EMBEDDING_MODEL=models/bge-base-zh-v1.5
```

- `ecs_demo/endpoints.yml` 中的 LLM、Neo4j、MySQL 配置均支持 `${ENV_VAR:默认值}` 形式的环境变量插值，可按需修改。
- 仅体验对话（不启用企业检索）时，只需配置 `SILICONFLOW_API_KEY` 即可。

## 快速开始

进入 Demo 目录后，即可使用 `atguigu` 命令：

```bash
cd llm_customer_service/ecs_demo

# 方式一：启动可视化调试页面（推荐，自动打开浏览器）
atguigu inspect

# 方式二：启动完整对话服务（REST + WebSocket + 调试页）
atguigu run

# 方式三：命令行交互式对话
atguigu shell
```

服务默认监听 `0.0.0.0:5005`，启动后可访问：

| 入口 | 地址 | 说明 |
| --- | --- | --- |
| 调试页面 | http://localhost:5005/inspect | 实时对话窗口 + Tracker 状态查看 |
| API 文档 | http://localhost:5005/docs | Swagger UI |
| 健康检查 | http://localhost:5005/health | 服务状态 |

试试问它：*“查一下订单 20260501001”*、*“我的快递到哪了”*、*“我要退货”*。

## CLI 命令

安装后全局提供 `atguigu` 命令（也可用 `python -m atguigu_ai <command>` 等价调用）：

| 命令 | 说明 |
| --- | --- |
| `atguigu init` | 初始化一个新对话项目（生成 config.yml、domain、flows 等脚手架） |
| `atguigu train` | 训练/打包对话模型（`--dry-run` 仅校验） |
| `atguigu run` | 启动对话服务（`--port`、`--enable-inspect`、`--channel` 等） |
| `atguigu shell` | 命令行交互式对话测试 |
| `atguigu inspect` | 启动可视化调试页面（`--no-browser` 禁止自动打开浏览器） |
| `atguigu export` | 导出模型 |
| `atguigu --version` / `-V` | 显示版本信息 |
| `atguigu --help` | 查看帮助 |

## REST API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/messages` | 发送消息，获取对话回复 |
| GET | `/api/sessions/{session_id}` | 获取会话状态 |
| POST | `/api/sessions/{session_id}/reset` | 重置会话 |
| GET | `/api/domain` | 获取 Domain 配置 |
| GET | `/api/flows` | 获取所有 Flow 定义 |
| GET | `/api/tracker/{session_id}/full` | 获取完整 Tracker 状态 |

## 对话处理流程

```
用户输入
   │
   ▼
┌──────────────┐   LLM 解析为结构化命令（含槽位）
│ understand   │──────────────────────────────┐
└──────────────┘                              ▼
┌──────────────┐   FlowPolicy /      ┌────────────────┐
│   policy     │───EnterpriseSearch──▶│  生成命令事件   │
└──────────────┘      Policy          └────────────────┘
                                              │
┌──────────────┐   执行 Action（查库/调接口）   │
│   action     │◀─────────────────────────────┘
└──────────────┘
      │
      ▼
┌──────────────┐   NLG 生成自然语言回复
│   response   │
└──────────────┘
      │
      ▼
┌──────────────┐   guard：安全检查与降级处理
│    guard     │
└──────────────┘
      │
      ▼
   返回用户
```

## 常见问题

**Q: 对话没有回复 / 报 ProxyError？**
国内模型服务需直连，检查 `.env` 中是否配置了 `NO_PROXY=*`、`no_proxy=*`（CLI 启动时会自动绕过系统代理）。

**Q: 提示 `Model does not exist`？**
`LLM_MODEL` / `CYPHER_LLM_MODEL` 必须是 API 平方账号模型列表中真实存在的模型 ID。

**Q: 企业知识库检索报错？**
`EnterpriseSearchPolicy` 依赖 Neo4j（GraphRAG）与 MySQL，需确保服务已启动且 `.env` 中密码正确；不需要该功能时可在 `ecs_demo/config.yml` 的 `policies` 中移除。

**Q: 修改代码后需要重装吗？**
不需要，`pip install -e .` 为可编辑模式，源码修改即时生效。

## License

MIT
