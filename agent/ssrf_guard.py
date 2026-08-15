"""HTTP hook SSRF 防护（R16 #4）。

对齐 CCB ssrfGuard.ts + execHttpHook.ts 的核心子集，适配 requests：

- 地址段校验：DNS 解析目标主机名，所有结果 IP 落在禁达段即拒
  （0/8、10/8、100.64/10 CGNAT 云元数据、169.254/16 链路本地云元数据、
  172.16/12、192.168/16；IPv6 ::、fc00::/7 ULA、fe80::/10、
  ::ffff:<v4> 映射地址按内嵌 v4 判定）
- 环回放行（127/8 与 ::1）：本地 dev policy server 是 http hook 的
  主流用法（CC 同款取舍）
- IP 直连同样校验（http://169.254.169.254/ 直接拒）
- 环境代理激活时跳过校验（代理侧做 DNS，本侧校验会误伤企业内网代理；
  对齐 CC 的 envProxyActive 语义）
- URL 含 CR/LF/NUL 拒（请求行/头部注入）
- 禁重定向（allow_redirects=False 由调用方设置；重定向可绕过预检
  弹到内网地址）

与 CC 的差异（如实记录）：
- requests 无法注入自定义 DNS lookup，校验发生在请求前的 getaddrinfo——
  校验与连接之间存在 DNS rebinding 窗口（CC 用 axios lookup 选项把
  校验 IP 钉到 socket 连接消除了该窗口）。预检已挡住配置型 hook 指向
  元数据/内网的绝大多数场景。
- OmniMate http hook 无 headers/env 插值配置（headers 硬编码
  Content-Type），CC 的 env 插值白名单 + header CRLF 清洗无对应面；
  URL 层的 CRLF 清洗保留。

全部校验函数是纯函数（不发网络请求的部分），便于测试；
validate_url_for_ssrf 会做 DNS 解析。
"""
import ipaddress
import logging
import re
import socket
from typing import List, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# IPv4 禁达段（对齐 ssrfGuard isBlockedV4；127/8 环回放行）
_BLOCKED_V4_NETS = [
    ipaddress.ip_network("0.0.0.0/8"),        # "this" network
    ipaddress.ip_network("10.0.0.0/8"),       # 私有
    ipaddress.ip_network("100.64.0.0/10"),    # CGNAT（阿里云元数据 100.100.100.200）
    ipaddress.ip_network("169.254.0.0/16"),   # 链路本地（云元数据 169.254.169.254）
    ipaddress.ip_network("172.16.0.0/12"),    # 私有
    ipaddress.ip_network("192.168.0.0/16"),   # 私有
]
_LOOPBACK_V4 = ipaddress.ip_network("127.0.0.0/8")
# IPv6 禁达段（:: 与 ::1 特判；fc00::/7 ULA、fe80::/10 链路本地）
_BLOCKED_V6_NETS = [
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

# URL 里的控制字符（请求行/头部注入）
_URL_CTRL_RE = re.compile(r"[\r\n\x00]")


def is_blocked_address(address: str) -> bool:
    """地址是否落在 http hook 禁达段。

    环回（127.0.0.0/8、::1）放行——本地 dev policy server 是主流用法。
    非法 IP 字符串返回 False（交给真实 DNS 路径处理）。
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return _is_blocked_ip(ip)


def _is_blocked_ip(ip) -> bool:
    if ip.version == 4:
        if ip in _LOOPBACK_V4:
            return False
        return any(ip in net for net in _BLOCKED_V4_NETS)
    # IPv6
    if ip == ipaddress.ip_address("::1"):
        return False  # 环回放行
    if ip == ipaddress.ip_address("::"):
        return True   # 未指定地址
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        # ::ffff:<v4> 映射地址：按内嵌 v4 判定（防 hex 形态绕过）
        return _is_blocked_ip(mapped)
    return any(ip in net for net in _BLOCKED_V6_NETS)


def _env_proxy_active(url: str) -> bool:
    """该 URL 是否会走环境代理（HTTP_PROXY/HTTPS_PROXY，尊重 NO_PROXY）。

    走代理时目标 DNS 由代理解析，本侧校验会误伤企业内网代理场景 → 跳过。
    """
    try:
        from requests.utils import get_environ_proxies
        proxies = get_environ_proxies(url)
        return bool(proxies)
    except Exception:
        return False


def validate_url_for_ssrf(url: str) -> Optional[str]:
    """校验 http hook 的 URL 是否允许外呼。允许返回 None，拒绝返回原因。

    步骤：
    1. URL 含 CR/LF/NUL → 拒（请求行/头部注入）
    2. scheme 非 http/https → 拒
    3. 主机名是 IP 字面量 → 直接按地址段校验
    4. 环境代理激活 → 放行（代理侧负责 DNS 与域白名单）
    5. DNS 解析（getaddrinfo），任一结果 IP 落禁达段 → 拒
       （拒绝消息只含主机名与命中的地址段，不泄露解析细节）
    """
    if not url:
        return "URL 为空"
    if _URL_CTRL_RE.search(url):
        return "URL 含控制字符（CR/LF/NUL）"

    try:
        parts = urlsplit(url.strip())
    except ValueError as e:
        return f"URL 解析失败: {e}"
    if parts.scheme not in ("http", "https"):
        return f"非 http/https scheme: {parts.scheme or '(空)'}"
    hostname = parts.hostname
    if not hostname:
        return "URL 缺主机名"

    # IP 字面量（含 [::1] 括号形态——urlsplit 的 hostname 已剥括号）
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        ip = None
    if ip is not None:
        if _is_blocked_ip(ip):
            return f"目标 {hostname} 在禁达段（私网/链路本地/云元数据）"
        return None  # 环回或公网 IP 字面量放行

    # 环境代理 → 跳过（对齐 CC envProxyActive 语义）
    if _env_proxy_active(url):
        return None

    try:
        infos = socket.getaddrinfo(
            hostname, parts.port or (443 if parts.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as e:
        # 解析失败不拦（让 requests 报真实的连接错误，fail-open 对齐 hook 语义）
        logger.debug("SSRF 预检 DNS 解析失败（放行交给请求层）: %s: %s", hostname, e)
        return None

    for info in infos:
        addr = info[4][0]
        if is_blocked_address(addr):
            return f"主机 {hostname} 解析到禁达段地址（私网/链路本地/云元数据）"
    return None


def url_matches_pattern(url: str, pattern: str) -> bool:
    """URL 是否匹配 allowlist 模式（* 为通配符，对齐 CC urlMatchesPattern）。

    模式整串匹配（^...$），* 匹配任意字符。
    """
    escaped = re.escape(pattern)
    return re.fullmatch(escaped.replace(r"\*", ".*"), url) is not None


def check_url_against_allowlist(url: str, allowed: Optional[List[str]]) -> Optional[str]:
    """URL allowlist 检查（R16 #4）。

    对齐 CC allowedHttpHookUrls 语义：
    - None → 不限制（默认）
    - [] → 全拒
    - 非空 → 必须匹配其中至少一个模式
    """
    if allowed is None:
        return None
    for pat in allowed:
        try:
            if url_matches_pattern(url, str(pat)):
                return None
        except re.error:
            continue
    return f"URL 不在 allowlist（allowed: {allowed}）"
