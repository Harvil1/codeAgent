"""MCP（Model Context Protocol）客户端：stdio + HTTP/SSE transport。

MCP 是外部服务统一接入协议。不需要为每个外部服务（Jira、Notion、数据库）
重写工具代码，只需实现 MCP 标准接口（tools/list + tools/call）。

08 升级：从单一 stdio → 两种 transport：
- stdio：启动本地子进程（原有）
- HTTP/SSE：远程 server，支持 OAuth 刷新

配置文件 ~/.agent/.mcp.json：
    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
        },
        "github-http": {
          "url": "https://api.github-mcp.com/v1",
          "headers": {"X-Custom": "v"}
        },
        "notion-oauth": {
          "url": "https://mcp.notion.com/v1",
          "oauth": {
            "token_url": "...", "client_id": "...",
            "client_secret": "...", "refresh_token": "..."
          }
        }
      }
    }

工具暴露：mcp__<server>__<tool> 前缀。
"""

import json
import logging
import os
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

    # 共享：MCP 握手流程（子类 connect() 末尾调用）
    def _do_initialize_handshake(self) -> None:
        """标准 MCP initialize 握手。"""
        resp = self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "HarvilAgent", "version": "0.1.0"},
        })
        if not resp:
            raise RuntimeError("MCP initialize 无响应")
        self.send_notification("notifications/initialized", {})


# ---------------------------------------------------------------------------
# stdio transport（从原 MCPClient 提取）
# ---------------------------------------------------------------------------

class StdioTransport(MCPTransport):
    """stdio 传输：启动本地子进程通过 stdin/stdout 交换 JSON-RPC。"""

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

    def connect(self) -> None:
        full_env = {**os.environ, **self.env}
        self.process = subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
            text=True,
            encoding="utf-8",
            bufsize=1,  # 行缓冲
        )
        try:
            self._do_initialize_handshake()
        except Exception:
            # 握手失败 → 关闭进程
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                pass
            self.process = None
            raise
        self._connected = True

    def send_request(self, method: str, params: dict) -> Optional[dict]:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("MCP stdio server 未运行")

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

            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError("MCP stdio server 断开")
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("MCP 非 JSON 行: %s", line.strip())
                    continue
                if data.get("id") == req_id:
                    if "error" in data:
                        err = data["error"]
                        raise RuntimeError(
                            f"MCP 错误 {err.get('code')}: {err.get('message')}"
                        )
                    return data.get("result")

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
# HTTP/SSE transport（08 新增）
# ---------------------------------------------------------------------------

class HTTPTransport(MCPTransport):
    """HTTP/SSE 传输（08）。

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
        self._session = None  # requests.Session
        self._connected = False
        self._request_id = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        # 1. 创建 session
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"缺少 HTTP 依赖（requests）: {e}")
        import requests
        self._session = requests.Session()

        # 2. OAuth 初始化（如配置）
        if self._oauth:
            self._refresh_access_token()

        # 3. HTTP 预检（避免 URL 配错卡 60s）
        ok, reason = self._preflight()
        if not ok:
            self._session.close()
            self._session = None
            raise RuntimeError(f"MCP HTTP 预检失败 ({self.url}): {reason}")
        logger.info("MCP HTTP 预检通过: %s", reason)

        # 4. MCP initialize 握手
        try:
            self._do_initialize_handshake()
        except Exception as e:
            self._session.close()
            self._session = None
            raise
        self._connected = True

    def _preflight(self) -> Tuple[bool, str]:
        """探测端点是否是合法 MCP server。返回 (ok, reason)。"""
        if self._session is None:
            return False, "session 未建立"
        probe_headers = {
            **self._headers,
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2024-11-05",
        }
        if self._access_token:
            probe_headers["Authorization"] = f"Bearer {self._access_token}"

        # 先 HEAD（轻量）
        try:
            r = self._session.head(
                self.url, headers=probe_headers,
                timeout=self.PREFLIGHT_TIMEOUT_S, allow_redirects=True,
            )
            ct = r.headers.get("Content-Type", "")
            if "json" in ct.lower() or "event-stream" in ct.lower():
                return True, f"HEAD ok (status={r.status_code}, ct={ct})"
        except Exception:
            pass

        # 再 GET
        try:
            r = self._session.get(
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
        """用 refresh_token 换新 access_token。"""
        if not self._oauth:
            return
        import requests
        import time
        cfg = self._oauth
        try:
            r = requests.post(
                cfg["token_url"],
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": cfg["refresh_token"],
                    "client_id": cfg["client_id"],
                    "client_secret": cfg.get("client_secret", ""),
                },
                timeout=10,
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
        if self._session is None:
            raise RuntimeError("MCP HTTP session 未建立")
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

            r = self._session.post(
                self.url, json=payload, headers=headers, timeout=60,
            )
            # 401 → 刷新 token 重试一次
            if r.status_code == 401 and self._oauth:
                logger.info("MCP HTTP 401，刷新 token 后重试一次")
                self._access_token = None
                self._refresh_access_token()
                headers["Authorization"] = f"Bearer {self._access_token}"
                r = self._session.post(
                    self.url, json=payload, headers=headers, timeout=60,
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
        if self._session is None:
            return
        headers = dict(self._headers)
        headers["Content-Type"] = "application/json"
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        try:
            self._session.post(
                self.url,
                json={"jsonrpc": "2.0", "method": method, "params": params},
                headers=headers, timeout=10,
            )
        except Exception:
            pass

    def close(self) -> None:
        self._connected = False
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    @property
    def is_connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# MCP 客户端（transport wrapper）
# ---------------------------------------------------------------------------

class MCPClient:
    """单个 MCP server 的客户端连接（08：自动选择 transport）。

    - command/args/env 配置 → StdioTransport
    - url/headers/oauth 配置 → HTTPTransport
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
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        self.name = name
        self.include = include
        self.exclude = exclude
        self._connected = False

        # 自动选择 transport
        if url:
            self._transport: MCPTransport = HTTPTransport(
                url=url, headers=headers, oauth_config=oauth,
            )
        elif command:
            self._transport = StdioTransport(command, args, env)
        else:
            raise ValueError(
                f"MCP server {name} 必须配 command (stdio) 或 url (http)"
            )

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
            from constants import get_agent_home
            config_path = get_agent_home() / ".mcp.json"
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

    def connect_all(self, config: Optional[Dict[str, dict]] = None) -> None:
        """连接所有配置的 server。

        配置字段（按 transport）：
        - stdio: command, args, env
        - HTTP:  url, headers, oauth
        - 通用:  include, exclude（工具过滤）
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
                    include=cfg.get("include"),
                    exclude=cfg.get("exclude"),
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
