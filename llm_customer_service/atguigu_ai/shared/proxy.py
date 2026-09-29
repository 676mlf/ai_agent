# -*- coding: utf-8 -*-
"""
代理处理工具模块

背景：
    在 Windows/macOS 上开启系统代理（Clash、v2rayN 等）后，requests 会通过
    ``urllib.request.getproxies()`` 从系统注册表读取代理配置。注册表里的
    https 代理会被解析成 ``https://127.0.0.1:PORT`` 这种"用 TLS 连代理"的形式，
    而本地代理端口实际只监听明文 HTTP，于是 TLS 握手失败，抛出：

        requests.exceptions.ProxyError: ... Caused by ProxyError(
            'Unable to connect to proxy', SSLEOFError())

    表现为 LLM 调用（如阿里云 DashScope）直接失败、对话没有任何回复。

处理策略：
    国内模型服务（DashScope 等）本来就不需要经过代理，将其域名加入 ``NO_PROXY``
    强制直连即可；其他域名（如 HuggingFace）仍沿用用户原有的代理设置。

可通过环境变量调整：
    - ``ATGUIGU_NO_PROXY``            追加需要直连的域名（逗号分隔）
    - ``ATGUIGU_DISABLE_NO_PROXY=1``  关闭本模块的自动直连
"""

import logging
import os
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# 国内可直连的模型/服务对象域名（后缀匹配，可覆盖其所有子域名）
DOMESTIC_LLM_HOSTS = (
    "dashscope.aliyuncs.com",          # 阿里云百炼 / DashScope 兼容接口
    "dashscope-intl.aliyuncs.com",     # DashScope 国际站
    "aliyuncs.com",                    # 其他阿里云服务域名
    "bigmodel.cn",                     # 智谱
    "moonshot.cn",                     # Kimi
    "volces.com",                      # 火山方舟 Doubao
    "siliconflow.cn",                  # 硅基流动 SiliconFlow
    "api.siliconflow.cn",              # 硅基流动 API 网关(httpx/openai SDK 需精确匹配)
)

# 本地服务域名/地址：开启系统代理后 requests 默认仍会走代理，
# 会导致本地 Action Server、vLLM、Neo4j HTTP 端点等访问失败，一并直连。
LOCAL_HOSTS = (
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
)

_ENV_EXTRA = "ATGUIGU_NO_PROXY"
_ENV_DISABLED = "ATGUIGU_DISABLE_NO_PROXY"


def _split_entries(value: Optional[str]) -> list:
    """把 NO_PROXY 风格的字符串拆成去空、去重的条目列表。"""
    entries: list = []
    for item in (value or "").split(","):
        item = item.strip()
        if item and item not in entries:
            entries.append(item)
    return entries


def bypass_proxy_for_domestic_hosts(extra_hosts: Optional[Iterable[str]] = None) -> str:
    """将国内模型服务域名与本地地址加入 ``NO_PROXY``，使其绕过系统代理直连。

    幂等：重复调用不会产生重复条目；用户已设置的 ``NO_PROXY`` 条目会保留。
    若用户已设置 ``NO_PROXY=*``（全部直连），则不做任何修改。

    参数：
        extra_hosts: 额外需要直连的域名（后缀匹配）

    返回：
        生效后的 NO_PROXY 字符串（便于日志输出）
    """
    if os.environ.get(_ENV_DISABLED, "").strip().lower() in ("1", "true", "yes"):
        return os.environ.get("NO_PROXY", "")

    existing = _split_entries(os.environ.get("NO_PROXY")) or _split_entries(
        os.environ.get("no_proxy")
    )
    if "*" in existing:
        # 用户已选择全部直连，保持原样
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"
        return "*"

    hosts = list(DOMESTIC_LLM_HOSTS) + list(LOCAL_HOSTS)
    if extra_hosts:
        hosts.extend(extra_hosts)
    hosts.extend(_split_entries(os.environ.get(_ENV_EXTRA)))

    merged = list(existing)
    for host in hosts:
        host = (host or "").strip()
        if not host:
            continue
        if host.startswith("."):
            host = host[1:]
        # 已有条目能覆盖该域名（相等或作为后缀）时跳过
        if any(host == item or host.endswith("." + item) for item in merged):
            continue
        merged.append(host)

    value = ",".join(merged)
    # requests / httpx 等库读取的变量名大小写不统一，两种都写入
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value
    logger.debug(f"已设置 NO_PROXY 直连域名: {value}")
    return value


__all__ = [
    "DOMESTIC_LLM_HOSTS",
    "LOCAL_HOSTS",
    "bypass_proxy_for_domestic_hosts",
]
