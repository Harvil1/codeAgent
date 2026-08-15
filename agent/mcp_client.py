"""MCP（Model Context Protocol）客户端：stdio + HTTP/SSE/WebSocket transport。

MCP 是外部服务统一接入协议。不需要为每个外部服务（Jira、Notion、数据库）
重写工具代码，只需实现 MCP 标准接口（tools/list + tools/call）。

Phase 5 升级：从单一 stdio → 四种 transport：
- stdio：启动本地子进程（原有）
- http：JSON POST + JSON 响应（httpx），支持 OAuth 刷新
- sse：Server-Sent Events 流式响应（httpx SSE，专用 transport）
- websocket：长连接双向 JSON-RPC（websockets 库）

配置文件 ~/.OmniMate/.mcp.json：
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

工具暴露：mcp__<server>__<tool> 前缀。

Feature flags：
    mcp_http_transport: 开启 http/sse transport（默认 OFF）
    mcp_websocket_transport: 开启 websocket transport（默认 OFF）
    不开时对应 transport 配置自动跳过（check_fn 隐藏）。
"""

import asyncio
import json
import logging
import os
import queue
import subprocess
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transport 抽象基类
# ---------------------------------------------------------------------------

class MCPTransport(ABC):
    """MCP 传输层抽象（08）。

    所有 transport 必须实现：
      - connect(): 建立连接 + MCP initialize 握手
      - send_request(method, params) -> Optional[dict]
      - send_notification(method, params) -> None
      - close()
      - is_connected 属性

    可选（Task 9 加）：
      - set_notification_handler(handler)：注册 server → client notification 处理器
        默认 no-op（向后兼容），子类按需 override（StdioTransport 真起 reader 线程）
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
        """注册 server → client notification 处理器。

        handler(notification_method, params) 在收到 notifications/* 时调用。
        默认实现：no-op（不强制子类 override，向后兼容 HTTPTransport 等）。

        StdioTransport override：reader 线程读 stdout 时真 dispatch notification。
        """
        # 默认 no-op（不强制子类实现，向后兼容）
        self._notification_handler = handler

    @property
    def notification_handler(self):
        """返回当前注册的 notification handler（None 表示未注册）。"""
        return getattr(self, "_notification_handler", None)

    # CCAR12 Task 5：MCP Resources 协议（默认实现，子类无需 override）
    # send_request 在所有具体 transport 里已通用，基类直接复用；
    # 服务器不支持 resources（JSON-RPC error）或任何异常 → None（fail-open）。
    def list_resources(self) -> Optional[list]:
        """MCP resources/list。服务器不支持/失败返回 None（fail-open）。

        返回 [{uri, name, mimeType?, description?}, ...]。
        """
        try:
            resp = self.send_request("resources/list", {})
            return (resp or {}).get("resources")
        except Exception:
            return None

    def read_resource(self, uri: str) -> Optional[dict]:
        """MCP resources/read。失败返回 None（fail-open）。

        成功返回 {contents: [{uri, text?|blob?, mimeType?}, ...]}。
        """
        try:
            return self.send_request("resources/read", {"uri": uri})
        except Exception:
            return None

    # 共享：MCP 握手流程（子类 connect() 末尾调用）
    def _do_initialize_handshake(self) -> None:
        """标准 MCP initialize 握手。"""
        resp = self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "OmniMate", "version": "0.1.0"},
        })
        if not resp:
            raise RuntimeError("MCP initialize 无响应")
        self.send_notification("notifications/initialized", {})


# ---------------------------------------------------------------------------
# stdio transport（从原 MCPClient 提取）
# ---------------------------------------------------------------------------

class StdioTransport(MCPTransport):
    """stdio 传输：启动本地子进程通过 stdin/stdout 交换 JSON-RPC。

    Task 9 升级：后台 daemon thread 读 stdout，
    response 走 queue（send_request 从 queue 拿），
    notification dispatch 到 set_notification_handler 注册的 handler。
    对现有调用方透明（仍 sync 阻塞等响应）。
    """

    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._connected = False
        # Task 9：reader 线程相关
        self._response_queue: "queue.Queue" = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._notification_handler = None  # 默认 None（向后兼容）
        # send_request 等响应的超时（秒）；可被子类/测试覆盖
        self._response_timeout: float = 60.0

    def connect(self) -> None:
        full_env = {**os.environ, **self.env}
        self.process = subprocess.Popen(
            self._resolve_command_argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
            text=True,
            encoding="utf-8",
            bufsize=1,  # 行缓冲
        )
        try:
            # 握手期间 send_request 会懒启动 reader 线程
            self._do_initialize_handshake()
        except Exception:
            # 握手失败 → 关闭进程
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
        """懒启动 reader 线程（首次 send_request 时起）。

        为什么不在 connect() 末尾起：握手期间就需要读响应，
        所以 send_request 里懒启动更简单（一个入口覆盖所有读需求）。
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
        """daemon thread：持续读 stdout。

        - response（带 id）→ _response_queue
        - notification（有 method 无 id）→ notification_handler
        - 非 JSON 行 → 跳过（server debug 输出）
        - handler 抛异常 → log 不影响后续读
        - readline 返回空（EOF）→ 退出循环
        """
        while self._connected and self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
            except Exception as e:
                logger.warning("mcp reader readline 异常: %s", e)
                break
            if not line:
                break  # EOF
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("MCP 非 JSON 行: %s", line.strip())
                continue
            # 区分 response（带 id）vs notification（有 method 无 id）
            if "id" in data:
                # response（带 id）→ queue
                self._response_queue.put(data)
            elif "method" in data:
                # notification → handler（fail-open：handler 异常不影响后续）
                handler = self._notification_handler
                if handler is not None:
                    try:
                        handler(data["method"], data.get("params", {}))
                    except Exception as e:
                        logger.warning("notification handler 异常: %s", e)
            # 其他类型（无 id 无 method）忽略

    def _resolve_command_argv(self) -> List[str]:
        """解析 MCP 子进程的 argv（跨平台）。

        Windows 上常见坑：配置写 `npx`，但 CreateProcess 找不到裸命令
        （实际可执行文件是 npx.cmd，且 CreateProcess 不解析 .cmd/.bat）。
        修复：
          - 用 shutil.which 解析命令（按 PATHEXT 补 .exe/.cmd/.bat）
          - 解析到 .cmd/.bat 时用 cmd.exe 包装（CreateProcess 不能直接执行脚本）
        非 Windows 直接返回 [command, *args]。
        """
        if os.name != "nt":
            return [self.command, *self.args]

        import shutil
        resolved = shutil.which(self.command) or self.command
        if resolved.lower().endswith((".cmd", ".bat")):
            # cmd /c 对含空格路径要加引号
            quoted = f'"{resolved}"' if " " in resolved else resolved
            return ["cmd", "/c", quoted, *self.args]
        return [resolved, *self.args]

    def send_request(self, method: str, params: dict) -> Optional[dict]:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("MCP stdio server 未运行")

        # Task 9：懒启动 reader 线程（首次调用时起，之后复用）
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

        # 从 queue 拿响应（reader 线程已把 response 投递过来）
        # 注意：必须在 _lock 外等——queue 是 thread-safe 的，但持锁阻塞
        # 会让 reader 无法拿到写 stdin 需要的 _lock（死锁）。
        while True:
            try:
                data = self._response_queue.get(
                    timeout=self._response_timeout,
                )
            except queue.Empty:
                raise RuntimeError(
                    f"MCP stdio request 超时（{self._response_timeout}s）"
                )
            # 匹配 id
            if data.get("id") == req_id:
                if "error" in data:
                    err = data["error"]
                    raise RuntimeError(
                        f"MCP 错误 {err.get('code')}: {err.get('message')}"
                    )
                return data.get("result")
            # 不是我们要的 response（可能是迟到的旧 response）→ 丢
            logger.debug("MCP 丢弃过期 response: id=%s", data.get("id"))

    def send_notification(self, method: str, params: dict) -> None:
        if self.process is None:
            return
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self.process.stdin.write(json.dumps(msg) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def close(self) -> None:
        self._connected = False
        # reader 线程是 daemon，会随 _connected=False + process 退出自然结束
        # （readline 会因为 stdout 关闭返回空）。
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
        return self._connected and self.process is not None


# ---------------------------------------------------------------------------
# HTTP transport（Phase 5：requests → httpx）
# ---------------------------------------------------------------------------

class HTTPTransport(MCPTransport):
    """HTTP 传输（Phase 5 升级：基于 httpx）。

    支持：
    - JSON POST + JSON 响应（普通）
    - JSON POST + SSE 响应（streamable-http）
    - OAuth：access_token 自动刷新（401 重试一次）

    HTTP 预检避免 URL 配错时卡 60s。
    """

    PREFLIGHT_TIMEOUT_S = 3

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        oauth_config: Optional[dict] = None,
    ):
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
        # 1. 创建 httpx.Client
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"缺少 HTTP 依赖（httpx）: {e}")
        import httpx
        self._client = httpx.Client(timeout=60.0)

        # 2. OAuth 初始化（如配置）
        if self._oauth:
            self._refresh_access_token()

        # 3. HTTP 预检（避免 URL 配错卡 60s）
        ok, reason = self._preflight()
        if not ok:
            self._client.close()
            self._client = None
            raise RuntimeError(f"MCP HTTP 预检失败 ({self.url}): {reason}")
        logger.info("MCP HTTP 预检通过: %s", reason)

        # 4. MCP initialize 握手
        try:
            self._do_initialize_handshake()
        except Exception:
            self._client.close()
            self._client = None
            raise
        self._connected = True

    def _preflight(self) -> Tuple[bool, str]:
        """探测端点是否是合法 MCP server。返回 (ok, reason)。"""
        if self._client is None:
            return False, "client 未建立"
        probe_headers = {
            **self._headers,
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2024-11-05",
        }
        if self._access_token:
            probe_headers["Authorization"] = f"Bearer {self._access_token}"

        # 先 HEAD（轻量）
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

        # 再 GET
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
        """用 refresh_token 换新 access_token（OAuth refresh 流程）。

        参考 claude-code-main HTTPTransport：POST grant_type=refresh_token 到
        token_url，拿 access_token + expires_in。过期前 60s 主动刷新。
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
        """过期前 60s 主动刷新。"""
        if not self._oauth:
            return
        import time
        if self._access_token and time.time() < self._token_expires_at - 60:
            return
        self._refresh_access_token()

    def send_request(self, method: str, params: dict) -> Optional[dict]:
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
            # 401 → 刷新 token 重试一次
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
        """从 SSE 流里提取最后一个 JSON-RPC result。"""
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
                self.url,
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
# SSE transport（Phase 5 新增：专用 SSE 流式 transport）
# ---------------------------------------------------------------------------

class SSETransport(MCPTransport):
    """SSE（Server-Sent Events）专用 transport（Phase 5）。

    区别于 HTTPTransport 的 streamable-http（POST 后收 SSE 响应）：
    SSETransport 建立长连接 GET 请求持续读 SSE 事件流，
    请求通过独立 POST 发送。

    适用场景：server 需要保持长连接推送（如远程 MCP server 的 SSE 端点）。

    OAuth 流程复用 HTTPTransport 的实现（共用 token 管理）。
    """

    PREFLIGHT_TIMEOUT_S = 3

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        oauth_config: Optional[dict] = None,
        post_url: Optional[str] = None,
    ):
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
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"缺少 HTTP 依赖（httpx）: {e}")
        import httpx
        self._client = httpx.Client(timeout=60.0)

        if self._oauth:
            self._refresh_access_token()

        # 预检（复用 HTTP 风格）
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
        """预检：端点需返回 event-stream 或 json content-type。"""
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
        """OAuth refresh（与 HTTPTransport 一致）。"""
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

            # SSE transport：POST 请求发到 post_url，响应可能是 SSE 流或 JSON
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
        """从 SSE 流里提取最后一个 JSON-RPC result。"""
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
# WebSocket transport（Phase 5 新增：websockets 库）
# ---------------------------------------------------------------------------

class WebSocketTransport(MCPTransport):
    """WebSocket transport（Phase 5）。

    用 websockets 库建立长连接，JSON-RPC 消息双向交换。
    websockets 库原生 async，这里用 asyncio.run 桥接到同步接口
    （对齐 Plan 2A 桥接模式）。

    适用场景：需要低延迟双向通信的 MCP server（如实时协作工具）。
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        oauth_config: Optional[dict] = None,
    ):
        # websockets 库用 ws:// 或 wss:// 协议
        self.url = url
        self._headers = headers or {}
        self._oauth = oauth_config
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0
        self._ws = None  # websockets 连接对象
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connected = False
        self._request_id = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        try:
            import websockets  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"缺少 WebSocket 依赖（websockets）: {e}")

        # OAuth 初始化
        if self._oauth:
            self._refresh_access_token()

        # 新建 event loop（独立于主线程的 asyncio 循环）
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._ws_connect())
            self._do_initialize_handshake()
        except Exception:
            self._loop.run_until_complete(self._ws_close())
            self._loop.close()
            self._loop = None
            raise
        self._connected = True

    async def _ws_connect(self) -> None:
        """建立 WebSocket 连接。"""
        import websockets
        # 构造请求头（websockets 库用 additional_headers）
        extra_headers = []
        for k, v in self._headers.items():
            extra_headers.append((k, v))
        if self._access_token:
            extra_headers.append(("Authorization", f"Bearer {self._access_token}"))

        self._ws = await websockets.connect(
            self.url, additional_headers=extra_headers,
        )

    def _refresh_access_token(self) -> None:
        """OAuth refresh（与 HTTPTransport 一致）。"""
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
        if not self._oauth:
            return
        import time
        if self._access_token and time.time() < self._token_expires_at - 60:
            return
        self._refresh_access_token()

    def send_request(self, method: str, params: dict) -> Optional[dict]:
        if self._ws is None or self._loop is None:
            raise RuntimeError("MCP WebSocket 未建立")
        self._ensure_token()

        with self._lock:
            self._request_id += 1
            req_id = self._request_id
            msg = {
                "jsonrpc": "2.0", "id": req_id,
                "method": method, "params": params,
            }
            return self._loop.run_until_complete(self._ws_send_and_recv(msg, req_id))

    async def _ws_send_and_recv(self, msg: dict, expected_id: int) -> Optional[dict]:
        """发送请求并等待对应 id 的响应。"""
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
        if self._ws is None or self._loop is None:
            return
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self._loop.run_until_complete(self._ws.send(json.dumps(msg)))
        except Exception:
            pass

    async def _ws_close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    def close(self) -> None:
        self._connected = False
        if self._loop is not None:
            try:
                self._loop.run_until_complete(self._ws_close())
                self._loop.close()
            except Exception:
                pass
            self._loop = None

    @property
    def is_connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# MCP 客户端（transport wrapper）
# ---------------------------------------------------------------------------

class MCPClient:
    """单个 MCP server 的客户端连接（Phase 5：支持 4 种 transport）。

    transport 选择规则（按优先级）：
    1. 显式 transport 字段（"stdio" / "http" / "sse" / "websocket"）
    2. 有 url → 按 URL scheme 推断（ws/wss → websocket，否则 http）
    3. 有 command → stdio

    Feature flag 门控（Phase 5 集成）：
    - websocket 需 mcp_websocket_transport 开启
    - sse（显式）需 mcp_http_transport 开启
    - http 向后兼容（不强制 flag，保留旧行为）
    未开启时构造抛 ValueError（让上层 connect_all 跳过+log）。
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
        self.name = name
        self.include = include
        self.exclude = exclude
        self._connected = False

        # Feature flag 检查（Phase 5 集成）
        from agent.feature_flags import is_feature_enabled
        cfg = config or {}
        http_enabled = is_feature_enabled(cfg, "mcp_http_transport")
        ws_enabled = is_feature_enabled(cfg, "mcp_websocket_transport")

        # 推断 transport 类型
        transport_type = self._resolve_transport_type(
            transport, url, command,
        )

        # 按类型构造 + flag 门控
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
            # 显式 SSE 需要 flag（专用 transport）
            if not http_enabled:
                raise ValueError(
                    f"MCP server {name}: sse transport 需要 "
                    f"开启 mcp_http_transport feature flag"
                )
            self._transport = SSETransport(
                url=url, headers=headers, oauth_config=oauth,
            )
        elif transport_type == "http":
            # http 向后兼容：不强制 flag（旧行为保护）
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
        """推断 transport 类型。

        优先级：显式 transport > URL scheme > command 存在性。
        """
        if transport:
            t = transport.lower().strip()
            if t in ("stdio", "http", "sse", "websocket"):
                return t
            raise ValueError(f"未知 transport 类型: {transport}")

        # 无显式 transport → 按 URL scheme 推断
        if url:
            lower = url.lower()
            if lower.startswith(("ws://", "wss://")):
                return "websocket"
            return "http"  # 默认 HTTP（含 streamable-http）

        # 无 URL → command 必须存在
        if command:
            return "stdio"

        raise ValueError("必须配 transport / url / command 之一")

    def connect(self) -> None:
        self._transport.connect()
        self._connected = True
        logger.info("MCP server %s 已连接", self.name)

    def list_tools(self) -> List[dict]:
        resp = self._transport.send_request("tools/list", {})
        return resp.get("tools", []) if resp else []

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self._transport.send_request("tools/call", {
            "name": name,
            "arguments": arguments or {},
        }) or {}

    # CCAR12 Task 5：resources 透传（fail-open，None = 不支持/失败）
    def list_resources(self) -> Optional[list]:
        """列出 server 的 resources（transport.list_resources）。"""
        return self._transport.list_resources()

    def read_resource(self, uri: str) -> Optional[dict]:
        """读单个 resource 内容（transport.read_resource）。"""
        return self._transport.read_resource(uri)

    def close(self) -> None:
        self._connected = False
        self._transport.close()

    @property
    def connected(self) -> bool:
        return self._connected and self._transport.is_connected


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_mcp_config(config_path=None) -> Dict[str, dict]:
    """加载 .mcp.json 配置。

    返回 {server_name: cfg}，cfg 含 command/args/env（stdio）或
    url/headers/oauth（HTTP）。
    """
    if config_path is None:
        try:
            from constants import get_omnimate_home
            config_path = get_omnimate_home() / ".mcp.json"
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


# ---------------------------------------------------------------------------
# 多 server 管理
# ---------------------------------------------------------------------------

class MCPManager:
    """管理多个 MCP server 连接。"""

    def __init__(self):
        self._clients: Dict[str, MCPClient] = {}
        self._lock = threading.Lock()

    def connect_all(
        self,
        config: Optional[Dict[str, dict]] = None,
        *,
        app_config: Optional[dict] = None,
    ) -> None:
        """连接所有配置的 server。

        配置字段（按 transport）：
        - stdio: transport="stdio", command, args, env
        - http:  transport="http", url, headers, oauth
        - sse:   transport="sse", url, headers, oauth
        - websocket: transport="websocket", url, headers, oauth
        - 通用:  include, exclude（工具过滤）

        app_config：应用配置字典（用于 feature flag 检查）。
        """
        if config is None:
            config = load_mcp_config()

        for name, cfg in config.items():
            try:
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
            except Exception as e:
                logger.warning("MCP server %s 连接失败: %s", name, e)

    def get_all_tools(self) -> List[dict]:
        """获取所有 server 的工具列表（含 server 名前缀）。"""
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
        """调用 MCP 工具（full_name 格式：mcp__server__tool）。"""
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

    # CCAR12 Task 5：resources 协议入口（供 mcp_resource 工具调用）
    # 错误风格与 call() 对齐：统一返回 dict（成功/失败都是 JSON 可序列化）。
    def list_resources(self, server_name: str) -> dict:
        """列某个 server 的 resources。失败返回错误 dict。"""
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
        """读某个 server 的单个 resource。失败返回错误 dict。"""
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
        """关闭所有连接。"""
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
        with self._lock:
            return list(self._clients.keys())


# 全局单例
_mcp_manager = MCPManager()


def get_mcp_manager() -> MCPManager:
    return _mcp_manager


def is_mcp_tool(name: str) -> bool:
    """判断工具名是否是 MCP 工具（mcp__ 前缀）。"""
    return name.startswith("mcp__")


def collect_routing_hints(config_path=None) -> str:
    """从 .mcp.json 收集 keywords,生成给 system prompt 用的 routing hints 块。

    借鉴 DeerFlow:用户在 .mcp.json 配 server 时可加 keywords 字段:
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
