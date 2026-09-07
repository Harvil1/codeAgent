"""MCP（Model Context Protocol，一个让 AI 接外部工具的通用插座标准）客户端。

这个文件负责"插上插座"：不管对面是 Jira、Notion 还是数据库，只要是
MCP 标准服务，就用同一套协议对接（tools/list 列工具 + tools/call 调
工具），不用为每个外部服务重写工具代码。上层是 tools/mcp_tool.py
（把 MCP 工具注册进工具表），被 cli.py 聚合使用。

支持四种 transport（传输方式）：
- stdio：在本机启动一个子进程，通过它的标准输入/输出对话
- http：发 JSON POST 请求、收 JSON 响应（用 httpx 库），支持 OAuth 令牌自动刷新
- sse：Server-Sent Events（服务器单向流式推送），httpx 的 SSE 实现，独立 transport
- websocket：长连接双向 JSON-RPC（用 websockets 库）

配置文件 ~/.codeAgent/.mcp.json 长这样：
    {
      "mcpServers": {
        "filesystem": {
          "transport": "stdio",
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
        },
        "github-http": {
          "transport": "http",
          "url": "https://api.github-mcp.com/v1",
          "headers": {"X-Custom": "v"}
        },
        "notion-oauth": {
          "transport": "http",
          "url": "https://mcp.notion.com/v1",
          "oauth": {
            "token_url": "...", "client_id": "...",
            "client_secret": "...", "refresh_token": "..."
          }
        },
        "remote-sse": {
          "transport": "sse",
          "url": "https://mcp.example.com/sse"
        },
        "remote-ws": {
          "transport": "websocket",
          "url": "wss://mcp.example.com/ws"
        }
      }
    }

工具暴露给 LLM 时叫 mcp__<server>__<tool>（双下划线隔开服务器名和工具名）。

Feature flags（功能开关，settings.json 里配）：
    mcp_http_transport: 打开 http/sse transport（默认关）
    mcp_websocket_transport: 打开 websocket transport（默认关）
    不打开时，对应 transport 的配置会被自动跳过（工具通过 check_fn 机制隐藏）。
"""

import json
import logging
import os
import queue
import subprocess
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transport（传输方式）抽象基类
# ---------------------------------------------------------------------------

class MCPTransport(ABC):
    """所有传输方式的共同抽象。

    stdio/HTTP/SSE/WebSocket 四种连法底层完全不同，但对上层
    （MCPClient）要长得一样，所以定一个统一接口。每种 transport 必须实现：
      - connect(): 建立连接 + 完成 MCP initialize 握手（先互相自报家门）
      - send_request(method, params) -> Optional[dict]：发一个请求，等一个响应
      - send_notification(method, params) -> None：发一个不等回复的通知
      - close(): 断开
      - is_connected 属性：现在连着吗

    可选实现：
      - set_notification_handler(handler)：注册"服务器 → 客户端"通知的
        处理器。默认什么都不做（老子类不用改），子类按需重载
        （StdioTransport 重载了，在 reader 线程里真分发通知）
    """

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def send_request(self, method: str, params: dict) -> Optional[dict]: ...

    @abstractmethod
    def send_notification(self, method: str, params: dict) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    def set_notification_handler(
        self,
        handler,  # Callable[[str, dict], None]
    ) -> None:
        """注册"服务器主动推送给客户端"的通知处理器。

        handler(notification_method, params) 会在收到 notifications/* 时被调用。
        默认实现只是存起来不干活（不强制子类重载，老代码如 HTTPTransport
        不用改就能继续用）。

        StdioTransport 重载了本方法：reader 线程读 stdout 时会真正分发通知。

        参数：
            handler：回调函数，签名 (notification_method: str, params: dict) -> None

        返回：无。
        """
        # 默认不强制子类实现，向后兼容
        self._notification_handler = handler

    @property
    def notification_handler(self):
        """查当前注册的通知处理器；返回 None 表示没注册过。"""
        return getattr(self, "_notification_handler", None)

    # MCP Resources 协议（服务器除工具外还能提供"资源"——
    # 可读的文件/数据）。这里给默认实现，子类不用重载：
    # send_request 在所有具体 transport 里都是通用的，基类直接复用；
    # 服务器不支持 resources（返回 JSON-RPC error）或出任何异常 →
    # 返回 None 不抛（fail-open，坏了不影响主流程）。
    def list_resources(self) -> Optional[list]:
        """问服务器要资源清单（MCP resources/list）。失败或对方不支持就返回 None。

        返回：[{uri, name, mimeType?, description?}, ...]，失败为 None。
        """
        try:
            resp = self.send_request("resources/list", {})
            return (resp or {}).get("resources")
        except Exception as e:
            # fail-open 但要大声：坏了照样返回 None，但必须在日志里留痕
            logger.warning("MCP list_resources 失败（fail-open 返回 None）: %s", e)
            return None

    def read_resource(self, uri: str) -> Optional[dict]:
        """读一个具体资源的内容（MCP resources/read）。失败返回 None。

        参数：
            uri：资源的 URI 地址

        返回：成功是 {contents: [{uri, text?|blob?, mimeType?}, ...]}，失败 None。
        """
        try:
            return self.send_request("resources/read", {"uri": uri})
        except Exception as e:
            # fail-open 但要大声：坏了照样返回 None，但必须在日志里留痕
            logger.warning("MCP read_resource 失败（fail-open 返回 None）: %s", e)
            return None

    # 各子类共用：MCP 标准握手流程（子类 connect() 末尾调用）
    def _do_initialize_handshake(self) -> None:
        """（内部）做标准 MCP initialize 握手——先自报家门（协议版本/客户端
        名），服务器应答后再发一个 initialized 通知确认握手完成。

        参数：无。返回：无；服务器没回应就抛 RuntimeError。
        """
        resp = self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "CodeAgent", "version": "0.1.0"},
        })
        if not resp:
            raise RuntimeError("MCP initialize 无响应")
        self.send_notification("notifications/initialized", {})


# ---------------------------------------------------------------------------
# stdio transport（本机子进程方式，从原 MCPClient 里拆出来的）
# ---------------------------------------------------------------------------

# 机密形状关键词：命中即不透传给 stdio server 子进程（大小写不敏感）。
# 第三方 server 不该默认拿到 harness 进程里的钥匙；真需要 key 的
# server 在 .mcp.json 的 env 里显式写（显式配置在擦洗之后叠加）。
_SECRET_ENV_PATTERNS = (
    "API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "PRIVATE_KEY",
)


def _scrub_env_for_child(env: dict) -> dict:
    """给 stdio MCP 子进程准备环境：剥掉机密形状变量和 CODEAGENT_* 自家变量。

    参数：
        env：当前进程的环境变量（通常是 os.environ）

    返回：擦洗后的副本（原 dict 不动）。
    """
    scrubbed = {}
    for key, value in env.items():
        upper = key.upper()
        if upper.startswith("CODEAGENT_"):
            continue
        if any(p in upper for p in _SECRET_ENV_PATTERNS):
            continue
        scrubbed[key] = value
    return scrubbed


class StdioTransport(MCPTransport):
    """stdio 传输：在本机启动一个子进程当 MCP 服务器，跟它的标准输入/输出
    管道里互发 JSON-RPC 消息（好比两个人各拿一根管子喊话）。

    起一个后台守护线程专门读子进程的 stdout——读到的
    "响应"放进队列（send_request 从队列里取），读到的"通知"转交给
    set_notification_handler 注册的处理器。对调用方完全透明（看起来还是
    同步阻塞等响应的老用法）。
    """

    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        """初始化 stdio transport（只存参数，真正启动在 connect()）。

        参数：
            command：要启动的命令，如 "npx"
            args：命令参数列表，如 ["-y", "@modelcontextprotocol/server-filesystem"]
            env：额外环境变量（叠加在当前进程环境上），可不填

        返回：无（构造函数）。
        """
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._connected = False
        # reader 线程相关
        self._response_queue: "queue.Queue" = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._notification_handler = None  # 默认 None（向后兼容）
        # send_request 等响应的超时秒数；子类/测试可覆盖
        self._response_timeout: float = 60.0

    def connect(self) -> None:
        """启动子进程并完成 MCP 握手。

        环境变量 = 当前进程的（先擦掉机密形状/自家变量）+ 配置里显式
        声明的（server 真需要 key 就在配置 env 里点名给）。
        握手失败就关掉子进程再抛错，不留半死进程。

        参数：无。返回：无；失败抛 RuntimeError。
        """
        # 环境擦洗：默认不给第三方 server 机密和自家变量；
        # 配置里显式声明的 env 在擦洗后叠加（用户点名要给的才给）
        full_env = _scrub_env_for_child(os.environ)
        full_env.update(self.env)
        self.process = subprocess.Popen(
            self._resolve_command_argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
            text=True,
            encoding="utf-8",
            # server 输出
            # 里可能混坏字节（GBK 日志/二进制），不能让 reader 线程炸——
            # replace 成 U+FFFD 替换符后当"非 JSON 行"跳过（fail-open），
            # 否则一条 UnicodeDecodeError 就让整条连接永久失效
            errors="replace",
            bufsize=1,  # 行缓冲
        )
        try:
            # 握手过程中 send_request 会顺手把 reader 线程懒启动起来
            self._do_initialize_handshake()
        except Exception:
            # 握手失败 → 收拾掉子进程再抛
            self._connected = False
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                pass
            self.process = None
            raise
        self._connected = True

    def _ensure_reader_started(self) -> None:
        """（内部）懒启动 reader 线程（第一次 send_request 时才起，之后复用）。

        为什么不在 connect() 末尾起：握手本身就要读响应，而握手在
        connect() 里、置成功标记之前——所以在 send_request 这个唯一
        的读入口里懒启动，一个入口管住所有读需求，最省事。
        """
        if self._reader_thread is not None and self._reader_thread.is_alive():
            return
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="mcp-stdio-reader",
        )
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        """后台守护线程的主体：不停地读子进程 stdout，一直读到流关闭。

        读到的每一行按类型分流：
        - 响应（带 id）→ 丢进 _response_queue 等人取
        - 通知（有 method 没 id）→ 交给 notification_handler
        - 不是 JSON 的行 → 跳过（多半是 server 自己打的调试日志）
        - handler 抛异常 → 记日志，不影响继续读
        - readline 返回空（EOF，管道关闭）→ 退出循环

        ⚠️ 注意：
        循环条件绝不能带 self._connected——握手期间（本线程被懒启动时）
        _connected 还是 False（connect() 要等握手成功才置 True），带上它
        reader 会立刻退出 → 响应永远读不到 → 握手 60s 超时。线程的退出
        只靠 EOF / 进程结束 / close()。
        """
        while self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
            except Exception as e:
                logger.warning("mcp reader readline 异常: %s", e)
                break
            if not line:
                break  # 读到 EOF（管道关了），收工
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("MCP 非 JSON 行: %s", line.strip())
                continue
            # 按 JSON-RPC 惯例分流：带 id 的是响应，只带 method 的是通知
            if "id" in data:
                # 响应（带 id）→ 进队列
                self._response_queue.put(data)
            elif "method" in data:
                # 通知 → 交 handler（fail-open：handler 抛异常不影响后续读）
                handler = self._notification_handler
                if handler is not None:
                    try:
                        handler(data["method"], data.get("params", {}))
                    except Exception as e:
                        logger.warning("notification handler 异常: %s", e)
            # 其他类型（既没 id 又没 method）直接忽略

    def _resolve_command_argv(self) -> List[str]:
        """（内部）算出真正用来启动子进程的命令行参数（处理跨平台差异）。

        Windows 上的经典坑：配置里写 `npx`，但 Windows 创建
        进程时找不到这个裸名字——实际可执行文件叫 npx.cmd，而且
        CreateProcess 不认 .cmd/.bat 脚本。处理办法：
          - 用 shutil.which 把命令解析成全路径（按 PATHEXT 自动补
            .exe/.cmd/.bat 后缀）
          - 解析出来是 .cmd/.bat 时，外面套一层 cmd.exe 来跑
            （CreateProcess 不能直接执行脚本）
        非 Windows 直接返回 [command, *args]。

        返回：参数字符串列表，可直接交给 subprocess.Popen。
        """
        if os.name != "nt":
            return [self.command, *self.args]

        import shutil
        resolved = shutil.which(self.command) or self.command
        if resolved.lower().endswith((".cmd", ".bat")):
            # cmd /c 的参数里路径带空格时要加引号，否则会被拆开
            quoted = f'"{resolved}"' if " " in resolved else resolved
            return ["cmd", "/c", quoted, *self.args]
        return [resolved, *self.args]

    def send_request(self, method: str, params: dict) -> Optional[dict]:
        """发一个 JSON-RPC 请求并阻塞等响应（老接口，调用方无需感知 reader 线程）。

        参数：
            method：MCP 方法名，如 "tools/list"
            params：请求参数 dict

        返回：响应里的 result 字段。超时或 server 报错抛 RuntimeError。
        """
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("MCP stdio server 未运行")

        # 懒启动 reader 线程（第一次调用时起，之后复用）
        self._ensure_reader_started()

        with self._lock:
            self._request_id += 1
            req_id = self._request_id
            msg = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params,
            }
            try:
                self.process.stdin.write(json.dumps(msg) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise RuntimeError(f"MCP stdio 写入失败: {e}")

        # 从队列里等响应（reader 线程已经把响应投进来了）
        # 注意：必须在 _lock 锁外等——队列本身线程安全，但如果持着锁
        # 阻塞等，reader 那边也拿不到写 stdin 要的同一把锁，就死锁了
        while True:
            try:
                data = self._response_queue.get(
                    timeout=self._response_timeout,
                )
            except queue.Empty:
                raise RuntimeError(
                    f"MCP stdio request 超时（{self._response_timeout}s）"
                )
            # 按 id 配对：队列里这条是不是我们这次请求的响应
            if data.get("id") == req_id:
                if "error" in data:
                    err = data["error"]
                    raise RuntimeError(
                        f"MCP 错误 {err.get('code')}: {err.get('message')}"
                    )
                return data.get("result")
            # id 对不上（多半是迟到的旧请求响应）→ 丢弃继续等
            logger.debug("MCP 丢弃过期 response: id=%s", data.get("id"))

    def send_notification(self, method: str, params: dict) -> None:
        """发一个不等回复的通知（fire-and-forget，写失败也只静默吞掉）。

        参数：
            method：通知方法名
            params：通知参数 dict

        返回：无。
        """
        if self.process is None:
            return
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self.process.stdin.write(json.dumps(msg) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def close(self) -> None:
        """关闭连接：先断 stdin，再客气地让子进程退出（3 秒不退就 kill）。

        参数：无。返回：无。
        """
        self._connected = False
        # reader 线程是 daemon 线程，随子进程退出自然结束
        # （stdout 一关，readline 返回空，循环就退了）。
        if self.process is None:
            return
        try:
            self.process.stdin.close()
        except Exception:
            pass
        try:
            self.process.terminate()
            self.process.wait(timeout=3)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        self.process = None

    @property
    def is_connected(self) -> bool:
        """还连着吗：握手成功过 且 子进程还活着。"""
        return self._connected and self.process is not None


# ---------------------------------------------------------------------------
# HTTP transport（基于 httpx 库）
# ---------------------------------------------------------------------------

class HTTPTransport(MCPTransport):
    """HTTP 传输（基于 httpx 库）。

    支持三种玩法：
    - 普通：发 JSON POST，收一个 JSON 响应
    - streamable-http：发 JSON POST，收 SSE 流式响应
    - OAuth：访问令牌自动刷新（收到 401 会换新令牌重试一次）

    连接前先做 HTTP 预检（快速探测）：URL 配错时能秒级报错，
    而不是傻等 60 秒超时。
    """

    PREFLIGHT_TIMEOUT_S = 3

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        oauth_config: Optional[dict] = None,
    ):
        """初始化 HTTP transport（只存配置，真正连接在 connect()）。

        参数：
            url：MCP 服务器地址，如 "https://api.github-mcp.com/v1"
            headers：额外自定义请求头，可不填
            oauth_config：OAuth 配置 dict（含 token_url/client_id/
                client_secret/refresh_token），可不填

        返回：无（构造函数）。
        """
        self.url = url
        self._headers = headers or {}
        self._oauth = oauth_config
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0
        self._client = None  # httpx.Client
        self._connected = False
        self._request_id = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        """建立连接：建 HTTP 客户端 → （配了的话）拿 OAuth 令牌 → 预检 → MCP 握手。

        参数：无。返回：无；任何一步失败都清理现场后抛 RuntimeError。
        """
        # 1. 建 httpx 客户端（缺依赖就报清楚的错）
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"缺少 HTTP 依赖（httpx）: {e}")
        import httpx
        self._client = httpx.Client(timeout=60.0)

        # 2. 配了 OAuth 就先拿访问令牌
        if self._oauth:
            self._refresh_access_token()

        # 3. HTTP 预检（3 秒内探一下端点像不像 MCP 服务器，URL 配错秒报）
        ok, reason = self._preflight()
        if not ok:
            self._client.close()
            self._client = None
            raise RuntimeError(f"MCP HTTP 预检失败 ({self.url}): {reason}")
        logger.info("MCP HTTP 预检通过: %s", reason)

        # 4. MCP initialize 握手（失败则关客户端再抛）
        try:
            self._do_initialize_handshake()
        except Exception:
            self._client.close()
            self._client = None
            raise
        self._connected = True

    def _preflight(self) -> Tuple[bool, str]:
        """（内部）预检：快速探测这个 URL 是不是一个像样的 MCP 端点。

        思路：先发个轻量的 HEAD，看响应类型是不是 JSON/SSE；不行再发
        GET 兜底——405（不允许 GET）也说明端点活着；401 多半是 OAuth
        配错了；返回 HTML 网页则说明这不是 MCP 端点。

        参数：无。

        返回：(是否通过, 人能看懂的原因说明)。
        """
        if self._client is None:
            return False, "client 未建立"
        probe_headers = {
            **self._headers,
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2024-11-05",
        }
        if self._access_token:
            probe_headers["Authorization"] = f"Bearer {self._access_token}"

        # 先试 HEAD（最轻量的探法）
        try:
            r = self._client.head(
                self.url, headers=probe_headers,
                timeout=self.PREFLIGHT_TIMEOUT_S, follow_redirects=True,
            )
            ct = r.headers.get("Content-Type", "")
            if "json" in ct.lower() or "event-stream" in ct.lower():
                return True, f"HEAD ok (status={r.status_code}, ct={ct})"
        except Exception:
            pass

        # HEAD 没结论再试 GET
        try:
            r = self._client.get(
                self.url, headers=probe_headers,
                timeout=self.PREFLIGHT_TIMEOUT_S,
            )
            ct = r.headers.get("Content-Type", "")
            if "json" in ct.lower() or "event-stream" in ct.lower():
                return True, f"GET ok (ct={ct})"
            if r.status_code == 405:
                return True, "405 (端点存活)"
            if r.status_code == 401:
                return False, "401 Unauthorized（OAuth 配置错？）"
            if "text/html" in ct.lower():
                return False, "不是 MCP 端点（返回 HTML 网页）"
            return False, f"未知响应 (status={r.status_code})"
        except Exception as e:
            return False, f"预检异常: {type(e).__name__}: {e}"

    def _refresh_access_token(self) -> None:
        """（内部）用 refresh_token 换一张新的 access_token（OAuth 刷新流程）——access_token 是短期门票，refresh_token 是长期身份证。

        做法：向 token_url 发
        POST（grant_type=refresh_token），拿回新 access_token 和有效期
        expires_in；记下过期时间，提前 60 秒主动刷新，不等它真过期。

        参数：无。返回：无；没配 OAuth 直接返回，刷新失败抛 RuntimeError。
        """
        if not self._oauth:
            return
        import time
        import httpx
        cfg = self._oauth
        try:
            r = httpx.post(
                cfg["token_url"],
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": cfg["refresh_token"],
                    "client_id": cfg["client_id"],
                    "client_secret": cfg.get("client_secret", ""),
                },
                timeout=10.0,
            )
            if r.status_code != 200:
                raise RuntimeError(
                    f"OAuth refresh 失败: {r.status_code} {r.text[:200]}"
                )
            data = r.json()
            self._access_token = data["access_token"]
            expires_in = data.get("expires_in", 3600)
            self._token_expires_at = time.time() + expires_in
            logger.info("MCP OAuth token 刷新成功，%ss 后过期", expires_in)
        except Exception as e:
            logger.error("OAuth refresh 失败: %s", e)
            raise

    def _ensure_token(self) -> None:
        """（内部）每次请求前检查门票：离过期不到 60 秒就提前刷新。"""
        if not self._oauth:
            return
        import time
        if self._access_token and time.time() < self._token_expires_at - 60:
            return
        self._refresh_access_token()

    def send_request(self, method: str, params: dict) -> Optional[dict]:
        """发一个 JSON-RPC 请求（POST）并等响应；响应是 SSE 流就按流解析。

        参数：
            method：MCP 方法名
            params：请求参数 dict

        返回：响应里的 result 字段。HTTP 非 200 或 server 报错抛 RuntimeError。
        """
        if self._client is None:
            raise RuntimeError("MCP HTTP client 未建立")
        self._ensure_token()

        headers = dict(self._headers)
        headers["Content-Type"] = "application/json"
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        with self._lock:
            self._request_id += 1
            req_id = self._request_id
            payload = {
                "jsonrpc": "2.0", "id": req_id,
                "method": method, "params": params,
            }

            r = self._client.post(
                self.url, json=payload, headers=headers, timeout=60.0,
            )
            # 收到 401（门票过期被拒）→ 换张新 token 重试一次
            if r.status_code == 401 and self._oauth:
                logger.info("MCP HTTP 401，刷新 token 后重试一次")
                self._access_token = None
                self._refresh_access_token()
                headers["Authorization"] = f"Bearer {self._access_token}"
                r = self._client.post(
                    self.url, json=payload, headers=headers, timeout=60.0,
                )

            if r.status_code != 200:
                raise RuntimeError(
                    f"MCP HTTP {r.status_code}: {r.text[:200]}"
                )

            ct = r.headers.get("Content-Type", "")
            if "event-stream" in ct.lower():
                return self._parse_sse_response(r.text)
            return r.json().get("result")

    def _parse_sse_response(self, text: str) -> Optional[dict]:
        """（内部）从 SSE 流文本里抠出最后一个 JSON-RPC result。

        SSE（Server-Sent Events，服务器单向流式推送）每行是一条
        "data: {...}" 事件，可能推多条，取最后那条有效结果；遇到 error
        事件直接抛。

        参数：
            text：整个 SSE 响应的原文

        返回：最后的 result dict；没有则 None。坏行跳过。
        """
        result = None
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                try:
                    data = json.loads(line[5:].strip())
                    if "result" in data:
                        result = data["result"]
                    elif "error" in data:
                        err = data["error"]
                        raise RuntimeError(
                            f"MCP 错误 {err.get('code')}: {err.get('message')}"
                        )
                except json.JSONDecodeError:
                    continue
        return result

    def send_notification(self, method: str, params: dict) -> None:
        """发一个不等回复的通知（POST 出去就不管了，失败静默吞掉）。"""
        if self._client is None:
            return
        headers = dict(self._headers)
        headers["Content-Type"] = "application/json"
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        try:
            self._client.post(
                self.url,
                json={"jsonrpc": "2.0", "method": method, "params": params},
                headers=headers, timeout=10.0,
            )
        except Exception:
            pass

    def close(self) -> None:
        """关闭连接：关掉 HTTP 客户端。"""
        self._connected = False
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    @property
    def is_connected(self) -> bool:
        """还连着吗（握手成功过）。"""
        return self._connected


# ---------------------------------------------------------------------------
# SSE transport（专用的 SSE 流式 transport）
# ---------------------------------------------------------------------------

class SSETransport(MCPTransport):
    """SSE（Server-Sent Events，服务器单向流式推送）专用 transport。

    和 HTTPTransport 里"POST 完收一个 SSE 响应"的 streamable-http 玩法不同：
    这个类用 GET 建一条长连接、持续读 SSE 事件流，请求则从另一条
    POST 通道发出去——两条道各走各的。

    适用场景：服务器需要保持长连接主动推送（比如远程 MCP server 的
    SSE 端点）。

    OAuth 刷新流程直接抄 HTTPTransport 的（令牌管理逻辑一样）。
    """

    PREFLIGHT_TIMEOUT_S = 3

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        oauth_config: Optional[dict] = None,
        post_url: Optional[str] = None,
    ):
        """初始化 SSE transport（只存配置，连接在 connect()）。

        参数：
            url：SSE 事件流地址（GET 长连接用）
            headers：额外自定义请求头，可不填
            oauth_config：OAuth 配置 dict，可不填
            post_url：发请求（POST）用的地址；不填默认和 url 相同

        返回：无（构造函数）。
        """
        self.url = url
        self._post_url = post_url or url  # POST 目标（默认同 URL）
        self._headers = headers or {}
        self._oauth = oauth_config
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0
        self._client = None  # httpx.Client
        self._connected = False
        self._request_id = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        """建立连接：建 HTTP 客户端 → （配了的话）拿 OAuth 令牌 → 预检 → MCP 握手。

        参数：无。返回：无；失败清理现场后抛 RuntimeError。
        """
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"缺少 HTTP 依赖（httpx）: {e}")
        import httpx
        self._client = httpx.Client(timeout=60.0)

        if self._oauth:
            self._refresh_access_token()

        # 预检（做法和 HTTPTransport 同款）
        ok, reason = self._preflight()
        if not ok:
            self._client.close()
            self._client = None
            raise RuntimeError(f"MCP SSE 预检失败 ({self.url}): {reason}")
        logger.info("MCP SSE 预检通过: %s", reason)

        try:
            self._do_initialize_handshake()
        except Exception:
            self._client.close()
            self._client = None
            raise
        self._connected = True

    def _preflight(self) -> Tuple[bool, str]:
        """（内部）预检：GET 一下端点，返回的类型得是 SSE 流或 JSON 才算过。

        返回：(是否通过, 原因说明)。
        """
        if self._client is None:
            return False, "client 未建立"
        probe_headers = {
            **self._headers,
            "Accept": "text/event-stream, application/json",
            "MCP-Protocol-Version": "2024-11-05",
        }
        if self._access_token:
            probe_headers["Authorization"] = f"Bearer {self._access_token}"
        try:
            r = self._client.get(
                self.url, headers=probe_headers,
                timeout=self.PREFLIGHT_TIMEOUT_S,
            )
            ct = r.headers.get("Content-Type", "")
            if "event-stream" in ct.lower() or "json" in ct.lower():
                return True, f"GET ok (ct={ct})"
            if r.status_code == 405:
                return True, "405 (端点存活)"
            if r.status_code == 401:
                return False, "401 Unauthorized"
            return False, f"未知响应 (status={r.status_code}, ct={ct})"
        except Exception as e:
            return False, f"预检异常: {type(e).__name__}: {e}"

    def _refresh_access_token(self) -> None:
        """（内部）用 refresh_token 换新 access_token（和 HTTPTransport 同一套流程）。"""
        if not self._oauth:
            return
        import time
        import httpx
        cfg = self._oauth
        try:
            r = httpx.post(
                cfg["token_url"],
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": cfg["refresh_token"],
                    "client_id": cfg["client_id"],
                    "client_secret": cfg.get("client_secret", ""),
                },
                timeout=10.0,
            )
            if r.status_code != 200:
                raise RuntimeError(
                    f"OAuth refresh 失败: {r.status_code} {r.text[:200]}"
                )
            data = r.json()
            self._access_token = data["access_token"]
            expires_in = data.get("expires_in", 3600)
            self._token_expires_at = time.time() + expires_in
            logger.info("MCP SSE OAuth token 刷新成功，%ss 后过期", expires_in)
        except Exception as e:
            logger.error("OAuth refresh 失败: %s", e)
            raise

    def _ensure_token(self) -> None:
        if not self._oauth:
            return
        import time
        if self._access_token and time.time() < self._token_expires_at - 60:
            return
        self._refresh_access_token()

    def send_request(self, method: str, params: dict) -> Optional[dict]:
        """发一个 JSON-RPC 请求（POST 到 post_url）并等响应；SSE 流或 JSON 都能解析。

        参数：
            method：MCP 方法名
            params：请求参数 dict

        返回：响应里的 result 字段；非 200 或 server 报错抛 RuntimeError。
        """
        if self._client is None:
            raise RuntimeError("MCP SSE client 未建立")
        self._ensure_token()

        headers = dict(self._headers)
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream, application/json"
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        with self._lock:
            self._request_id += 1
            req_id = self._request_id
            payload = {
                "jsonrpc": "2.0", "id": req_id,
                "method": method, "params": params,
            }

            # SSE transport：请求 POST 到 post_url，回来的可能是 SSE 流也可能是普通 JSON
            r = self._client.post(
                self._post_url, json=payload, headers=headers, timeout=60.0,
            )
            if r.status_code == 401 and self._oauth:
                logger.info("MCP SSE 401，刷新 token 后重试一次")
                self._access_token = None
                self._refresh_access_token()
                headers["Authorization"] = f"Bearer {self._access_token}"
                r = self._client.post(
                    self._post_url, json=payload, headers=headers, timeout=60.0,
                )

            if r.status_code != 200:
                raise RuntimeError(
                    f"MCP SSE {r.status_code}: {r.text[:200]}"
                )

            ct = r.headers.get("Content-Type", "")
            if "event-stream" in ct.lower():
                return self._parse_sse_response(r.text)
            return r.json().get("result")

    def _parse_sse_response(self, text: str) -> Optional[dict]:
        """（内部）从 SSE 流文本里抠出最后一个 JSON-RPC result（同 HTTPTransport）。"""
        result = None
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                try:
                    data = json.loads(line[5:].strip())
                    if "result" in data:
                        result = data["result"]
                    elif "error" in data:
                        err = data["error"]
                        raise RuntimeError(
                            f"MCP 错误 {err.get('code')}: {err.get('message')}"
                        )
                except json.JSONDecodeError:
                    continue
        return result

    def send_notification(self, method: str, params: dict) -> None:
        if self._client is None:
            return
        headers = dict(self._headers)
        headers["Content-Type"] = "application/json"
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        try:
            self._client.post(
                self._post_url,
                json={"jsonrpc": "2.0", "method": method, "params": params},
                headers=headers, timeout=10.0,
            )
        except Exception:
            pass

    def close(self) -> None:
        self._connected = False
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    @property
    def is_connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# WebSocket transport（用 websockets 库）
# ---------------------------------------------------------------------------

class WebSocketTransport(MCPTransport):
    """WebSocket transport。

    用 websockets 库建立一条长连接，JSON-RPC 消息双向收发（像打电话，
    两边都能随时开口）。websockets 库原生是 async 异步的，这里把协程
    统一交给 loop_host 常驻事件循环跑（run_async 桥接成同步接口）——
    连接对象从建立到关闭一直绑在同一个循环上，不再来回搬家。

    适用场景：需要低延迟双向通信的 MCP server（比如实时协作工具）。
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        oauth_config: Optional[dict] = None,
    ):
        """初始化 WebSocket transport（只存配置，连接在 connect()）。

        参数：
            url：WebSocket 地址（ws:// 或 wss:// 开头）
            headers：额外自定义请求头，可不填
            oauth_config：OAuth 配置 dict，可不填

        返回：无（构造函数）。
        """
        # websockets 库用 ws:// 或 wss:// 协议
        self.url = url
        self._headers = headers or {}
        self._oauth = oauth_config
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0
        self._ws = None  # websockets 连接对象
        # 驱动交给 loop_host 常驻循环（不再自建事件循环字段）
        self._connected = False
        self._request_id = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        """建立连接：拿 OAuth 令牌 → 连 WebSocket → MCP 握手。

        异步部分全部交给 loop_host 常驻循环驱动（不再自建事件循环，
        也不动调用线程的循环视图）。失败清理现场后抛。

        参数：无。返回：无；失败清理现场后抛。
        """
        try:
            import websockets  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"缺少 WebSocket 依赖（websockets）: {e}")

        # 惰性 import（防模块级循环依赖）；宿主循环也是首次用到才拉起
        from agent.loop_host import loop_host

        # 配了 OAuth 先拿门票
        if self._oauth:
            self._refresh_access_token()

        try:
            loop_host.run_async(self._ws_connect())
            self._do_initialize_handshake()
        except Exception:
            loop_host.run_async(self._ws_close())
            raise
        self._connected = True

    async def _ws_connect(self) -> None:
        """（内部）真正去连 WebSocket 服务器，把自定义头和令牌都带上。"""
        import websockets
        # 拼请求头（websockets 库的参数名叫 additional_headers）
        extra_headers = []
        for k, v in self._headers.items():
            extra_headers.append((k, v))
        if self._access_token:
            extra_headers.append(("Authorization", f"Bearer {self._access_token}"))

        self._ws = await websockets.connect(
            self.url, additional_headers=extra_headers,
        )

    def _refresh_access_token(self) -> None:
        """（内部）用 refresh_token 换新 access_token（和 HTTPTransport 同一套流程）。"""
        if not self._oauth:
            return
        import time
        import httpx
        cfg = self._oauth
        try:
            r = httpx.post(
                cfg["token_url"],
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": cfg["refresh_token"],
                    "client_id": cfg["client_id"],
                    "client_secret": cfg.get("client_secret", ""),
                },
                timeout=10.0,
            )
            if r.status_code != 200:
                raise RuntimeError(
                    f"OAuth refresh 失败: {r.status_code} {r.text[:200]}"
                )
            data = r.json()
            self._access_token = data["access_token"]
            expires_in = data.get("expires_in", 3600)
            self._token_expires_at = time.time() + expires_in
            logger.info("MCP WebSocket OAuth token 刷新成功，%ss 后过期", expires_in)
        except Exception as e:
            logger.error("OAuth refresh 失败: %s", e)
            raise

    def _ensure_token(self) -> None:
        """（内部）每次请求前检查门票：离过期不到 60 秒就提前刷新。"""
        if not self._oauth:
            return
        import time
        if self._access_token and time.time() < self._token_expires_at - 60:
            return
        self._refresh_access_token()

    def send_request(self, method: str, params: dict) -> Optional[dict]:
        """发一个 JSON-RPC 请求（走 WebSocket）并等响应。

        参数：
            method：MCP 方法名
            params：请求参数 dict

        返回：响应里的 result 字段；server 报错抛 RuntimeError。
        """
        if self._ws is None:
            raise RuntimeError("MCP WebSocket 未建立")
        self._ensure_token()

        # 惰性 import（防模块级循环依赖）
        from agent.loop_host import loop_host

        with self._lock:
            self._request_id += 1
            req_id = self._request_id
            msg = {
                "jsonrpc": "2.0", "id": req_id,
                "method": method, "params": params,
            }
            # 防死锁：send_request 的调用方是 to_thread 工具线程/启动期
            # 主线程，均不在宿主循环线程——loop_host.run_async 安全
            #（run_async 若在宿主循环线程内调用会自己等自己，死锁）
            return loop_host.run_async(self._ws_send_and_recv(msg, req_id))

    async def _ws_send_and_recv(self, msg: dict, expected_id: int) -> Optional[dict]:
        """（内部）把请求发出去，然后收消息直到等到 id 对得上的那条响应。

        参数：
            msg：完整 JSON-RPC 消息 dict
            expected_id：本次请求的 id，用来配对响应

        返回：响应里的 result 字段。
        """
        await self._ws.send(json.dumps(msg))
        while True:
            raw = await self._ws.recv()
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if data.get("id") == expected_id:
                if "error" in data:
                    err = data["error"]
                    raise RuntimeError(
                        f"MCP 错误 {err.get('code')}: {err.get('message')}"
                    )
                return data.get("result")

    def send_notification(self, method: str, params: dict) -> None:
        """发一个不等回复的通知（失败静默吞掉）。"""
        if self._ws is None:
            return
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            from agent.loop_host import loop_host
            loop_host.run_async(self._ws.send(json.dumps(msg)))
        except Exception:
            pass

    async def _ws_close(self) -> None:
        """（内部）关掉 WebSocket 连接（幂等，已关也不报错）。"""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    def close(self) -> None:
        """关闭连接：关 WebSocket（驱动在 loop_host 常驻循环上）。"""
        self._connected = False
        if self._ws is not None:
            try:
                from agent.loop_host import loop_host
                loop_host.run_async(self._ws_close())
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        """还连着吗（握手成功过）。"""
        return self._connected


# ---------------------------------------------------------------------------
# MCP 客户端（transport 的统一外壳）
# ---------------------------------------------------------------------------

class MCPClient:
    """单个 MCP server 的客户端连接（支持 4 种 transport）——上层（tools/mcp_tool.py）不想关心底下是子进程还是 HTTP，这个类挑好具体 transport 再包一层统一接口。

    用哪种 transport 的判定顺序：
    1. 配置里明写了 transport 字段（"stdio" / "http" / "sse" / "websocket"）
    2. 有 url → 看网址开头（ws/wss → websocket，其余当 http）
    3. 有 command → stdio

    Feature flag 门控：
    - websocket 要开 mcp_websocket_transport
    - 显式写 sse 要开 mcp_http_transport
    - http 不设门槛（向后兼容，保住老配置的行为）
    没开就构造抛 ValueError——故意抛给上层 connect_all 看，让它跳过并记日志。
    """

    def __init__(
        self,
        name: str,
        *,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        oauth: Optional[dict] = None,
        transport: Optional[str] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        config: Optional[dict] = None,
    ):
        """初始化：判定 transport 类型（含 feature flag 门控）并造好 transport 对象。

        参数：
            name：server 名字（会进工具全名 mcp__<name>__<tool>）
            command：stdio 方式要启动的命令，可不填
            args：stdio 命令的参数列表，可不填
            env：stdio 子进程的额外环境变量，可不填
            url：远程 server 的地址（http/sse/websocket 用），可不填
            headers：额外请求头，可不填
            oauth：OAuth 配置 dict，可不填
            transport：显式指定 "stdio"/"http"/"sse"/"websocket"，可不填（自动推断）
            include：只保留这些工具名（白名单），可不填
            exclude：排除这些工具名（黑名单），可不填
            config：应用配置 dict（查 feature flag 用），可不填

        返回：无（构造函数）。flag 没开或类型推不出来抛 ValueError。
        """
        self.name = name
        self.include = include
        self.exclude = exclude
        self._connected = False

        # 先查功能开关
        from agent.feature_flags import is_feature_enabled
        cfg = config or {}
        http_enabled = is_feature_enabled(cfg, "mcp_http_transport")
        ws_enabled = is_feature_enabled(cfg, "mcp_websocket_transport")

        # 推断该用哪种 transport
        transport_type = self._resolve_transport_type(
            transport, url, command,
        )

        # 按类型造 transport + flag 门控
        if transport_type == "websocket":
            if not ws_enabled:
                raise ValueError(
                    f"MCP server {name}: websocket transport 需要 "
                    f"开启 mcp_websocket_transport feature flag"
                )
            self._transport: MCPTransport = WebSocketTransport(
                url=url, headers=headers, oauth_config=oauth,
            )
        elif transport_type == "sse":
            # 显式写 sse 是专用 transport，要 flag 开着才让用
            if not http_enabled:
                raise ValueError(
                    f"MCP server {name}: sse transport 需要 "
                    f"开启 mcp_http_transport feature flag"
                )
            self._transport = SSETransport(
                url=url, headers=headers, oauth_config=oauth,
            )
        elif transport_type == "http":
            # http 不设 flag 门槛（保护老配置，行为不突变）
            self._transport = HTTPTransport(
                url=url, headers=headers, oauth_config=oauth,
            )
        elif transport_type == "stdio":
            self._transport = StdioTransport(command, args, env)
        else:
            raise ValueError(
                f"MCP server {name}: 无法确定 transport 类型"
            )

    @staticmethod
    def _resolve_transport_type(
        transport: Optional[str],
        url: Optional[str],
        command: Optional[str],
    ) -> str:
        """（内部）推断该用哪种 transport。

        优先级：配置里明写 > 看网址开头 > 有 command 就 stdio。

        参数：
            transport：配置里显式写的类型（可 None）
            url：远程地址（可 None）
            command：本机命令（可 None）

        返回："stdio"/"http"/"sse"/"websocket" 之一；推不出来抛 ValueError。
        """
        if transport:
            t = transport.lower().strip()
            if t in ("stdio", "http", "sse", "websocket"):
                return t
            raise ValueError(f"未知 transport 类型: {transport}")

        # 没明写 transport → 按网址开头猜
        if url:
            lower = url.lower()
            if lower.startswith(("ws://", "wss://")):
                return "websocket"
            return "http"  # 默认按 HTTP 处理（含 streamable-http）

        # 连 URL 都没有 → 必须给了 command，走 stdio
        if command:
            return "stdio"

        raise ValueError("必须配 transport / url / command 之一")

    def connect(self) -> None:
        """连接服务器（底下 transport 的 connect，含 MCP 握手）。"""
        self._transport.connect()
        self._connected = True
        logger.info("MCP server %s 已连接", self.name)

    def list_tools(self) -> List[dict]:
        """问服务器要工具清单（tools/list）。

        返回：工具描述 dict 的列表；拿不到就是空列表。
        """
        resp = self._transport.send_request("tools/list", {})
        return resp.get("tools", []) if resp else []

    def call_tool(self, name: str, arguments: dict) -> dict:
        """调用服务器上的一个工具（tools/call）。

        参数：
            name：工具名（不带 mcp__ 前缀的原始名）
            arguments：工具参数 dict

        返回：工具执行结果 dict；空结果给空 dict。
        """
        return self._transport.send_request("tools/call", {
            "name": name,
            "arguments": arguments or {},
        }) or {}

    # resources 透传（fail-open：返回 None = 对方不支持或失败）
    def list_resources(self) -> Optional[list]:
        """列出 server 的资源清单（转手调 transport.list_resources）。"""
        return self._transport.list_resources()

    def read_resource(self, uri: str) -> Optional[dict]:
        """读单个资源内容（转手调 transport.read_resource）。

        参数：
            uri：资源的 URI 地址
        """
        return self._transport.read_resource(uri)

    def close(self) -> None:
        """断开连接。"""
        self._connected = False
        self._transport.close()

    @property
    def connected(self) -> bool:
        """还连着吗：自己记录的标记 和 transport 的实时状态 都得为真。"""
        return self._connected and self._transport.is_connected


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_mcp_config(config_path=None) -> Dict[str, dict]:
    """加载用户级 MCP 配置文件 .mcp.json。

    参数：
        config_path：配置文件路径；不填默认 ~/.codeAgent/.mcp.json

    返回：{server名: 该server的配置dict}。stdio 型含 command/args/env；
    HTTP 型含 url/headers/oauth。文件不存在或读坏了返回空 dict（不抛）。
    """
    if config_path is None:
        try:
            from constants import get_codeagent_home
            config_path = get_codeagent_home() / ".mcp.json"
        except Exception:
            return {}

    config_path = __import__("pathlib").Path(config_path)
    if not config_path.exists():
        return {}

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return data.get("mcpServers", {}) or {}
    except Exception as e:
        logger.warning("加载 MCP 配置失败 %s: %s", config_path, e)
        return {}


def load_project_mcp_config() -> Tuple[Optional[Path], Dict[str, dict]]:
    """读项目级 MCP 配置：当前工作目录下的 .mcp.json。

    参数：无。

    返回：(配置文件路径或 None, {server名: 配置dict})。
    为什么单独搞项目级：项目里的 .mcp.json 不受用户直接控制——
    clone 一个陌生仓库就可能被带进来，所以调用方（initialize_mcp）
    必须先过"首连审批"（第一次连之前问用户）才能连。
    """
    try:
        from agent.workspace_context import get_workspace_cwd
        p = Path(get_workspace_cwd()) / ".mcp.json"
    except Exception:
        return None, {}
    if not p.exists():
        return None, {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        servers = data.get("mcpServers", {}) or {}
        return p, {str(k): v for k, v in servers.items() if isinstance(v, dict)}
    except Exception as e:
        logger.warning("加载项目 MCP 配置失败 %s: %s", p, e)
        return p, {}


# ---------------------------------------------------------------------------
# 多 server 管理
# ---------------------------------------------------------------------------

class MCPManager:
    """管家：同时管着多个 MCP server 的连接（增删查、列工具、调用分发）。"""

    def __init__(self):
        """初始化：空的连接表 + 一把锁（保护连接表的并发读写）。"""
        self._clients: Dict[str, MCPClient] = {}
        self._lock = threading.Lock()

    def connect_all(
        self,
        config: Optional[Dict[str, dict]] = None,
        *,
        app_config: Optional[dict] = None,
    ) -> None:
        """按配置把所有 server 都连一遍（单个失败只记日志跳过，不拖垮整体）。

        参数：
            config：{server名: 配置dict}；不填则自动去读 ~/.codeAgent/.mcp.json。
                配置字段按 transport 分：
                - stdio: transport="stdio", command, args, env
                - http:  transport="http", url, headers, oauth
                - sse:   transport="sse", url, headers, oauth
                - websocket: transport="websocket", url, headers, oauth
                - 通用:  include, exclude（工具白/黑名单过滤）
            app_config：应用配置字典（查 feature flag 用），可不填

        返回：无。
        """
        if config is None:
            config = load_mcp_config()

        for name, cfg in config.items():
            try:
                self.connect_one(name, cfg, app_config=app_config)
            except Exception as e:
                # 保持既有语义：单个 server 失败（包括 transport flag 没开）只跳过
                logger.warning("MCP server %s 连接失败: %s", name, e)

    def connect_one(
        self,
        name: str,
        cfg: dict,
        *,
        app_config: Optional[dict] = None,
    ) -> "MCPClient":
        """连接单个 server（给 agent 定义里内联的 mcpServers 用）。

        参数：
            name：server 名字
            cfg：单个 server 的配置 dict（字段同 connect_all 的说明）
            app_config：应用配置字典（查 feature flag 用），可不填

        返回：连好的 MCPClient。失败直接抛异常（要不要吞由调用方定，
        和 connect_all 的"只跳过"语义不同）。同名连接已存在且活着就直接
        复用（幂等，连两次不报错）。
        """
        with self._lock:
            existing = self._clients.get(name)
        if existing is not None and existing.connected:
            return existing
        client = MCPClient(
            name=name,
            command=cfg.get("command"),
            args=cfg.get("args"),
            env=cfg.get("env"),
            url=cfg.get("url"),
            headers=cfg.get("headers"),
            oauth=cfg.get("oauth"),
            transport=cfg.get("transport"),
            include=cfg.get("include"),
            exclude=cfg.get("exclude"),
            config=app_config,
        )
        client.connect()
        with self._lock:
            self._clients[name] = client
        return client

    def disconnect_one(self, name: str) -> bool:
        """断开并移除单个 server 连接（临时连接用完就断，
        不留残留——比如 agent 定义里的内联 server）。

        参数：
            name：server 名字

        返回：True 表示找到并断开了；False 表示本来就没这个连接。
        """
        with self._lock:
            client = self._clients.pop(name, None)
        if client is None:
            return False
        try:
            client.close()
        except Exception as e:
            logger.debug("MCP server %s 关闭异常（忽略）: %s", name, e)
        return True

    def get_all_tools(self) -> List[dict]:
        """汇总所有在线 server 的工具清单（工具名带上 mcp__server__ 前缀）。

        参数：无。

        返回：工具描述 dict 列表，每项含 server/original_name/full_name/
        description/inputSchema。单个 server 列工具失败只记日志跳过。
        """
        all_tools = []
        with self._lock:
            clients = list(self._clients.items())

        for server_name, client in clients:
            if not client.connected:
                continue
            try:
                tools = client.list_tools()
                include = getattr(client, "include", None)
                exclude = getattr(client, "exclude", None)
                if not isinstance(include, (list, type(None))):
                    include = None
                if not isinstance(exclude, (list, type(None))):
                    exclude = None

                for tool in tools:
                    tool_name = tool.get("name", "")
                    if include is not None:
                        if tool_name not in include:
                            continue
                    elif exclude:
                        if tool_name in exclude:
                            continue
                    all_tools.append({
                        "server": server_name,
                        "original_name": tool_name,
                        "full_name": f"mcp__{server_name}__{tool_name}",
                        "description": tool.get("description", ""),
                        "inputSchema": tool.get("inputSchema", {
                            "type": "object", "properties": {},
                        }),
                    })
            except Exception as e:
                logger.warning("MCP server %s 列工具失败: %s", server_name, e)
        return all_tools

    def call(self, full_name: str, arguments: dict) -> dict:
        """调用一个 MCP 工具。

        参数：
            full_name：工具全名，格式 mcp__server__tool（双下划线分隔）
            arguments：工具参数 dict

        返回：工具结果 dict；名字不合法/没连上/调用失败都返回
        {"error": ...} 形式的 dict（不抛异常）。
        """
        parts = full_name.split("__", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            return {"error": f"非法 MCP 工具名: {full_name}"}
        _, server_name, tool_name = parts

        with self._lock:
            client = self._clients.get(server_name)
        if client is None:
            return {"error": f"MCP server 未连接: {server_name}"}
        if not client.connected:
            return {"error": f"MCP server 已断开: {server_name}"}

        try:
            return client.call_tool(tool_name, arguments)
        except Exception as e:
            return {"error": f"MCP 调用失败: {e}"}

    # resources 协议的对外入口（给 mcp_resource 工具调）。
    # 错误风格与 call() 对齐：统一返回 dict，成功失败都能直接 JSON 序列化。
    def list_resources(self, server_name: str) -> dict:
        """列某个 server 的资源清单。

        参数：
            server_name：server 名字

        返回：成功是 {"server": ..., "resources": [...]};
        失败是 {"error": ..., "error_type": ...}（没连上/不支持/出错）。
        """
        with self._lock:
            client = self._clients.get(server_name)
        if client is None:
            return {
                "error": f"MCP server 未连接: {server_name}",
                "error_type": "mcp_server_not_connected",
            }
        if not client.connected:
            return {
                "error": f"MCP server 已断开: {server_name}",
                "error_type": "mcp_server_disconnected",
            }
        try:
            resources = client.list_resources()
        except Exception as e:
            return {"error": f"MCP list_resources 失败: {e}"}
        if resources is None:
            return {
                "error": f"MCP server {server_name} 不支持 resources 协议",
                "error_type": "mcp_resources_unsupported",
            }
        return {"server": server_name, "resources": resources}

    def read_resource(self, server_name: str, uri: str) -> dict:
        """读某个 server 上的单个资源内容。

        参数：
            server_name：server 名字
            uri：资源的 URI 地址（必填）

        返回：成功是资源内容 dict；失败是 {"error": ..., "error_type": ...}
        （没连上/缺参数/不存在/不支持）。
        """
        with self._lock:
            client = self._clients.get(server_name)
        if client is None:
            return {
                "error": f"MCP server 未连接: {server_name}",
                "error_type": "mcp_server_not_connected",
            }
        if not client.connected:
            return {
                "error": f"MCP server 已断开: {server_name}",
                "error_type": "mcp_server_disconnected",
            }
        if not uri:
            return {
                "error": "缺少 uri 参数",
                "error_type": "invalid_args",
            }
        try:
            result = client.read_resource(uri)
        except Exception as e:
            return {"error": f"MCP read_resource 失败: {e}"}
        if result is None:
            return {
                "error": (
                    f"读取 resource 失败（uri 可能不存在，或 server "
                    f"{server_name} 不支持 resources 协议）"
                ),
                "error_type": "mcp_resources_unsupported",
            }
        return result

    def close_all(self) -> None:
        """把所有连接一口气全关掉（退出时用；单个关闭失败也继续关别的）。"""
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            try:
                client.close()
            except Exception:
                pass

    @property
    def servers(self) -> List[str]:
        """现在管着哪些 server（名字列表）。"""
        with self._lock:
            return list(self._clients.keys())


# 全局单例（整个进程共用一个管家）
_mcp_manager = MCPManager()


def get_mcp_manager() -> MCPManager:
    return _mcp_manager


def is_mcp_tool(name: str) -> bool:
    """判断工具名是否是 MCP 工具（mcp__ 前缀）。"""
    return name.startswith("mcp__")


def collect_routing_hints(config_path=None) -> str:
    """从 .mcp.json 收集 keywords,生成给 system prompt 用的 routing hints 块。

    用户在 .mcp.json 配 server 时可加 keywords 字段:
        "postgres": {
            "command": "...",
            "keywords": ["订单", "数据库", "SQL", "查询"],
            "description": "数据库查询"
        }

    本函数扫描配置,有 keywords 的 server 集合生成一个提示块,
    让 LLM 看到用户提到"查订单"时立刻知道用 postgres server。

    返回空串表示无 routing hints(没配 keywords 的场景)。
    """
    config = load_mcp_config(config_path)
    if not config:
        return ""
    lines = []
    for name, cfg in config.items():
        keywords = cfg.get("keywords") or []
        if not keywords:
            continue
        description = cfg.get("description", "")
        kw_str = " / ".join(str(k) for k in keywords)
        line = f"- 提到 [{kw_str}] → 优先用 `{name}`"
        if description:
            line += f"（{description}）"
        lines.append(line)
    if not lines:
        return ""
    return (
        "## MCP 工具路由提示\n"
        "当用户消息涉及以下关键词时,优先用对应的 MCP server 工具:\n"
        + "\n".join(lines)
    )


def filter_tool_name(
    tool_name: str,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> bool:
    """判断单个工具是否应该保留。

    include 优先：include 非空时，只有在 include 里才保留。
    exclude：在 exclude 里的不保留。
    两者都空时保留所有。
    """
    if include is not None:
        return tool_name in include
    if exclude:
        return tool_name not in exclude
    return True
