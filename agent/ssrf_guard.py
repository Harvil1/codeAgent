"""HTTP hook（用 HTTP 请求触发的钩子）的 SSRF（服务器端请求伪造）防护，R16 安全专项第 4 项。

背景：http hook 会让程序主动向外发 HTTP 请求。如果配置里写了个内网地址，
攻击者就可能借它去摸云服务器的"元数据接口"（能拿到云账号密钥）或公司内网。
这个模块就是在请求发出去之前做一道安检，参照 CCB 的 ssrfGuard.ts +
execHttpHook.ts 核心子集实现，并适配 Python 的 requests 库：

- 地址段检查：把目标域名做 DNS 解析（域名变 IP），只要解析出来的 IP 落在
  "禁达名单"里就拒绝。禁达名单包括：0/8、10/8（家庭/公司内网）、
  100.64/10（运营商级 NAT 段，阿里云元数据服务 100.100.100.200 在这）、
  169.254/16（链路本地段，云元数据服务 169.254.169.254 在这）、
  172.16/12、192.168/16（内网）；IPv6 的 ::（未指定地址）、
  fc00::/7（内网唯一地址）、fe80::/10（链路本地）；
  ::ffff:<v4> 这种"IPv6 皮 IPv4 心"的映射地址按里面的 IPv4 判定
- 环回地址（127/8 和 ::1，也就是"本机"）放行——本地开发时 hook 常指向
  本机的 policy server，这是主流用法（CC 也是这么取舍的）
- 直接写 IP 的 URL 同样检查（http://169.254.169.254/ 直接拒）
- 配了环境代理（HTTP_PROXY 等）时跳过检查——代理会替我们做 DNS，
  本地再查会把走公司内网代理的正常场景误杀（对齐 CC 的 envProxyActive 语义）
- URL 里带换行/回车/NUL 控制字符就拒（防止伪造请求行或 HTTP 头）
- 禁止重定向（调用方设 allow_redirects=False）——重定向可能把一个
  安全的 URL"弹"到内网地址，绕过预检

和 CC 的已知差异（如实记录）：
- requests 没法自定义 DNS 解析函数，检查发生在请求前的 getaddrinfo——
  检查完到真正建连接之间理论上存在 DNS rebinding（域名解析结果被
  掉包）的窗口。CC 用 axios 的 lookup 选项把检查过的 IP 直接钉到
  socket 连接上，消除了这个窗口。预检已挡住配置型 hook 指向
  元数据/内网的绝大多数场景。
- OmniMate 的 http hook 没有 headers/env 插值配置（headers 写死
  Content-Type），所以 CC 的 env 插值白名单 + 头部 CRLF 清洗在这里
  没有对应攻击面；URL 层的 CRLF 清洗保留。

除 validate_url_for_ssrf 要做 DNS 解析外，其余校验函数都是纯函数
（不发网络请求），方便单测。
"""
import ipaddress
import logging
import re
import socket
from typing import List, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# IPv4 禁达段（对齐 ssrfGuard 的 isBlockedV4；127/8 环回放行不在这表里）
_BLOCKED_V4_NETS = [
    ipaddress.ip_network("0.0.0.0/8"),        # "本网络"保留段
    ipaddress.ip_network("10.0.0.0/8"),       # 内网
    ipaddress.ip_network("100.64.0.0/10"),    # 运营商级 NAT 段（阿里云元数据 100.100.100.200 在这）
    ipaddress.ip_network("169.254.0.0/16"),   # 链路本地段（云元数据 169.254.169.254 在这）
    ipaddress.ip_network("172.16.0.0/12"),    # 内网
    ipaddress.ip_network("192.168.0.0/16"),   # 内网
]
_LOOPBACK_V4 = ipaddress.ip_network("127.0.0.0/8")
# IPv6 禁达段（:: 和 ::1 走下面代码特判；fc00::/7 内网唯一地址、fe80::/10 链路本地）
_BLOCKED_V6_NETS = [
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

# URL 里不允许出现的控制字符（防伪造请求行/HTTP 头注入）
_URL_CTRL_RE = re.compile(r"[\r\n\x00]")


def is_blocked_address(address: str) -> bool:
    """判断一个 IP 地址是否落在 http hook 的禁达段里。

    背景：http hook 外呼前要做安检，这个函数负责"IP 落点"这一项。
    环回地址（127.0.0.0/8、::1，即本机）放行——本地开发的 policy server
    是主流用法。传进来的字符串不是合法 IP 时返回 False，交给后面真正的
    DNS 解析路径去处理。

    参数：
        address：IP 地址字符串（如 "192.168.1.1"）

    返回：True 表示落在禁达段（要拒）；False 表示放行或无法识别。
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return _is_blocked_ip(ip)


def _is_blocked_ip(ip) -> bool:
    """（内部）判断已解析好的 IP 对象是否命中禁达段，v4/v6 分开查表。"""
    if ip.version == 4:
        if ip in _LOOPBACK_V4:
            return False
        return any(ip in net for net in _BLOCKED_V4_NETS)
    # IPv6 分支
    if ip == ipaddress.ip_address("::1"):
        return False  # 本机环回，放行
    if ip == ipaddress.ip_address("::"):
        return True   # 未指定地址，拒绝
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        # ::ffff:<v4> 是"IPv6 皮 IPv4 心"的映射地址：按里面的 v4 判定，
        # 防止用十六进制写法绕过检查
        return _is_blocked_ip(mapped)
    return any(ip in net for net in _BLOCKED_V6_NETS)


def _env_proxy_active(url: str) -> bool:
    """判断这个 URL 是否会走环境变量配置的代理（HTTP_PROXY/HTTPS_PROXY，尊重 NO_PROXY 的豁免）。

    背景：如果流量走代理，目标域名是代理服务器去解析的，我们在本地做
    DNS 检查反而会把"公司内网代理"这种正常场景误杀，所以这种情况下
    整个 SSRF 检查跳过。

    参数：
        url：要请求的完整 URL

    返回：True 表示会走代理（应跳过校验）。
    """
    try:
        from requests.utils import get_environ_proxies
        proxies = get_environ_proxies(url)
        return bool(proxies)
    except Exception:
        return False


def validate_url_for_ssrf(url: str) -> Optional[str]:
    """校验 http hook 要访问的 URL 能不能放行外呼。允许返回 None，拒绝返回原因文字。

    背景：这是 http hook 请求发出前的总安检入口，逐层把关。

    步骤（按顺序）：
    1. URL 含回车/换行/NUL 控制字符 → 拒（防伪造请求行/HTTP 头）
    2. 协议不是 http/https → 拒
    3. 主机名直接就是个 IP 字面量 → 按禁达段查表
    4. 配了环境代理 → 放行（DNS 和域白名单由代理侧负责）
    5. 做真 DNS 解析（getaddrinfo），只要有一个结果 IP 落在禁达段 → 拒
       （拒绝消息只带主机名和命中的段名，不泄露内部解析细节）

    参数：
        url：http hook 要请求的完整 URL 字符串

    返回：None 表示放行；字符串表示拒绝原因（可直接给用户看）。
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

    # 主机名本身就是 IP 字面量的情况（含 [::1] 括号写法——urlsplit
    # 取 hostname 时已经把括号剥掉了）
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        ip = None
    if ip is not None:
        if _is_blocked_ip(ip):
            return f"目标 {hostname} 在禁达段（私网/链路本地/云元数据）"
        return None  # 本机环回或公网 IP 字面量，放行

    # 配了环境代理就跳过后续检查（对齐 CC envProxyActive 语义）
    if _env_proxy_active(url):
        return None

    try:
        infos = socket.getaddrinfo(
            hostname, parts.port or (443 if parts.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as e:
        # DNS 解析失败不在这拦——放行给 requests 报真实的连接错误
        # （fail-open，对齐 hook 的容错语义）
        logger.debug("SSRF 预检 DNS 解析失败（放行交给请求层）: %s: %s", hostname, e)
        return None

    for info in infos:
        addr = info[4][0]
        if is_blocked_address(addr):
            return f"主机 {hostname} 解析到禁达段地址（私网/链路本地/云元数据）"
    return None


def url_matches_pattern(url: str, pattern: str) -> bool:
    """判断 URL 是否匹配一个 allowlist（允许清单）模式，* 是通配符（对齐 CC 的 urlMatchesPattern）。

    背景：allowlist 里每条是一个模式串，模式要求整串匹配（相当于自动加
    ^...$），* 可以匹配任意字符。

    参数：
        url：要检查的完整 URL
        pattern：模式串，如 "https://example.com/*"

    返回：True 表示匹配。
    """
    escaped = re.escape(pattern)
    return re.fullmatch(escaped.replace(r"\*", ".*"), url) is not None


def check_url_against_allowlist(url: str, allowed: Optional[List[str]]) -> Optional[str]:
    """URL allowlist（允许清单）检查，R16 安全专项第 4 项。

    背景：除了禁达段，还可以配置"只许访问这些 URL"。语义对齐 CC 的
    allowedHttpHookUrls：
    - 传 None → 不限制（默认行为）
    - 传空列表 → 全部拒绝
    - 传非空列表 → 必须至少匹配其中一个模式，否则拒

    参数：
        url：要检查的完整 URL
        allowed：allowlist 模式列表，或 None 表示不限制

    返回：None 表示放行；字符串表示拒绝原因。
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
