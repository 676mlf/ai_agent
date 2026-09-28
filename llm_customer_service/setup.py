# -*- coding: utf-8 -*-
"""
atguigu_ai 安装配置文件 (setup.py)

================================================================================
什么是 setup.py？
================================================================================
setup.py 是 Python 包的"安装说明书"，它告诉 pip：
1. 这个包叫什么名字
2. 需要安装哪些依赖
3. 包含哪些代码文件
4. 提供哪些命令行工具

================================================================================
为什么 setup.py 要放在这个位置？
================================================================================
目录结构：
    llm_customer_service_B/          <-- 项目根目录
    ├── setup.py                     <-- setup.py 放在这里！
    ├── requirements-atguigu.txt     <-- 依赖列表也放在这里
    ├── atguigu_ai/                  <-- 要安装的 Python 包
    │   ├── __init__.py
    │   ├── agent/
    │   ├── core/
    │   └── ...
    ├── reference/                    <-- 参考代码（不会被安装）
    └── ecs_demo/                    <-- 示例项目（不会被安装）

关键原则：setup.py 必须放在"要安装的包"的父目录！
- atguigu_ai/ 是我们要安装的包（里面有 __init__.py）
- 所以 setup.py 放在 atguigu_ai/ 的父目录，即项目根目录

================================================================================
如何使用？
================================================================================
# 第一步：进入项目根目录（setup.py 所在的目录）
cd llm_customer_service_B

# 第二步：安装（选择以下任一方式）

# 方式1：开发模式安装（推荐！修改代码后立即生效，无需重新安装）
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方式2：普通安装（代码会被复制到 site-packages，修改源码不会生效）
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方式3：安装时包含开发工具（pytest、black 等）
pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple

================================================================================
安装后会发生什么？
================================================================================
1. 依赖包会被自动安装（langchain、fastapi、openai 等）
2. atguigu_ai 包会被注册到 Python 环境，可以在任何地方 import
3. atguigu 命令会被注册到系统，可以在命令行直接使用：
   - atguigu init    # 初始化项目
   - atguigu train   # 训练模型
   - atguigu run     # 启动服务
   - atguigu inspect # 交互式测试

================================================================================
常见问题
================================================================================
Q: 为什么用 pip install -e . 而不是直接 python xxx.py？
A: -e 是"editable"（可编辑）模式，安装后修改源码立即生效，适合开发阶段。
   直接运行 python 脚本需要手动处理导入路径，容易出错。

Q: 那个点 "." 是什么意思？
A: 点表示"当前目录"，告诉 pip 在当前目录找 setup.py 进行安装。

Q: 安装后怎么卸载？
A: pip uninstall atguigu_ai

Q: 怎么查看是否安装成功？
A: pip show atguigu_ai  或者  python -c "import atguigu_ai; print(atguigu_ai.__version__)"
"""

from pathlib import Path
from setuptools import setup, find_packages


# ============================================================================
# 辅助函数：读取依赖列表
# ============================================================================
def read_requirements(filename: str) -> list:
    """从 requirements 文件读取依赖列表。
    
    为什么要从文件读取？
    - 依赖列表可能很长，写在 setup.py 里不好维护
    - 单独的 requirements.txt 可以被其他工具使用（如 pip install -r）
    
    Args:
        filename: requirements 文件名（相对于 setup.py 的路径）
        
    Returns:
        依赖包列表，如 ["langchain>=0.3.0", "fastapi>=0.100.0", ...]
    """
    # Path(__file__) 是当前文件（setup.py）的路径
    # .parent 获取父目录，即项目根目录
    requirements_path = Path(__file__).parent / filename
    
    # 如果文件不存在，返回空列表（不会报错）
    if not requirements_path.exists():
        return []
    
    requirements = []
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            # 跳过注释行（以 # 开头）
            if line.startswith("#"):
                continue
            # 处理条件依赖，如 "colorama>=0.4.0; sys_platform == 'win32'"
            # 这里简化处理，只取分号前的部分
            if ";" in line:
                line = line.split(";")[0].strip()
            requirements.append(line)
    
    return requirements


# ============================================================================
# 辅助函数：读取版本号
# ============================================================================
def read_version() -> str:
    """从 atguigu_ai/__init__.py 读取版本号。
    
    为什么版本号定义在 __init__.py 而不是这里？
    - 这样代码里可以通过 atguigu_ai.__version__ 获取版本
    - 版本号只需要维护一处，不会出现不一致
    """
    init_path = Path(__file__).parent / "atguigu_ai" / "__init__.py"
    if not init_path.exists():
        return "0.1.0"
    
    with open(init_path, "r", encoding="utf-8") as f:
        for line in f:
            # 找到 __version__ = "x.x.x" 这一行
            if line.startswith("__version__"):
                # 提取等号右边的值，去掉引号
                return line.split("=")[1].strip().strip('"').strip("'")
    
    return "0.1.0"


def read_long_description() -> str:
    """读取 README.md 作为包的详细描述。"""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# ============================================================================
# 依赖定义
# ============================================================================

# 核心依赖：运行 atguigu_ai 必须的包
# 从 requirements-atguigu.txt 文件读取
install_requires = read_requirements("requirements-atguigu.txt")

# 开发依赖：开发和测试时需要的包（用户正常使用不需要）
dev_requires = [
    "pytest>=7.0.0",           # 测试框架
    "pytest-asyncio>=0.21.0",  # 异步测试支持
    "pytest-cov>=4.0.0",       # 测试覆盖率
    "black>=23.0.0",           # 代码格式化
    "isort>=5.12.0",           # import 排序
    "mypy>=1.0.0",             # 类型检查
    "flake8>=6.0.0",           # 代码风格检查
]


# ============================================================================
# setup() - 核心配置
# ============================================================================
# setup() 函数是整个文件的核心，它接收一系列参数来描述这个包
setup(
    # ------------------------------------------------------------------------
    # 基本信息
    # ------------------------------------------------------------------------
    
    # 包名：安装后 import 用的名字，也是 pip install/uninstall 用的名字
    name="atguigu_ai",
    
    # 版本号：遵循语义化版本 (主版本.次版本.修订版本)
    version=read_version(),
    
    # 作者信息
    author="atguigu",
    author_email="",
    
    # 简短描述（一句话）
    description="教学版对话系统 - 基于 LLM 驱动的智能对话架构精简实现",
    
    # 详细描述（通常是 README 内容）
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    
    # 项目主页
    url="",
    
    # 开源协议
    license="MIT",
    
    # ------------------------------------------------------------------------
    # 包发现配置
    # ------------------------------------------------------------------------
    # find_packages() 会自动扫描目录，找到所有包含 __init__.py 的目录
    # 
    # include: 只包含这些包（atguigu_ai 和它的所有子包）
    # exclude: 排除这些目录（测试、文档、参考代码等）
    #
    # 为什么要排除 reference 和 ecs_demo？
    # - reference/ 是参考源码，不是我们要发布的代码
    # - ecs_demo/ 是示例项目，用户需要单独创建自己的项目
    packages=find_packages(
        include=["atguigu_ai", "atguigu_ai.*"],
        exclude=["tests", "tests.*", "docs", "docs.*", 
                 "reference", "reference.*", "ecs_demo", "ecs_demo.*"],
    ),
    
    # ------------------------------------------------------------------------
    # 非 Python 文件
    # ------------------------------------------------------------------------
    # 默认只打包 .py 文件，如果包里有模板、配置文件等，需要显式声明
    include_package_data=True,
    package_data={
        "atguigu_ai": [
            "templates/**/*",  # 模板文件
            "data/**/*",       # 数据文件
        ],
    },
    
    # ------------------------------------------------------------------------
    # Python 版本要求
    # ------------------------------------------------------------------------
    python_requires=">=3.10",
    
    # ------------------------------------------------------------------------
    # 依赖配置
    # ------------------------------------------------------------------------
    # install_requires: 必须安装的依赖，pip install 时会自动安装
    install_requires=install_requires,
    
    # extras_require: 可选依赖，通过 pip install "包名[extra]" 安装
    # 例如：pip install -e ".[dev]" 会安装核心依赖 + 开发依赖
    extras_require={
        "dev": dev_requires,                      # 开发依赖
        "all": install_requires + dev_requires,  # 全部依赖
    },
    
    # ------------------------------------------------------------------------
    # 命令行入口点（CLI）
    # ------------------------------------------------------------------------
    # 安装后会创建 atguigu 命令，执行时调用 atguigu_ai.cli 模块的 main 函数
    # 
    # 格式："命令名=模块路径:函数名"
    # 安装后可以直接在终端运行：atguigu init / atguigu train / atguigu run
    entry_points={
        "console_scripts": [
            "atguigu=atguigu_ai.cli:main",
        ],
    },
    
    # ------------------------------------------------------------------------
    # PyPI 分类信息（发布到 PyPI 时用于分类检索）
    # ------------------------------------------------------------------------
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    
    # 搜索关键词
    keywords="conversational-ai chatbot dialogue-system llm langchain",
)
