# -*- coding: utf-8 -*-
"""
atguigu_ai 程序入口

支持通过以下方式运行:
    python -m atguigu_ai <command>
    atguigu <command>  (需要安装后)
"""

import sys
import os
from pathlib import Path

import dotenv


def main() -> None:
    """主入口函数。"""
    # 自动加载用户工程目录下的 .env 文件
    # 用户可在工程根目录放置 .env 文件配置 API KEY 等环境变量
    # 显式指定当前工作目录的 .env 文件路径，确保在 Windows 上正确加载
    cwd = Path.cwd()
    env_file = cwd / ".env"
    if env_file.exists():
        dotenv.load_dotenv(env_file)
    else:
        # 如果当前目录没有，尝试默认搜索
        dotenv.load_dotenv()
    
    # 将当前目录添加到Python路径，以便导入自定义模块
    sys.path.insert(0, str(cwd))
    
    # 导入并运行CLI
    from atguigu_ai.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
