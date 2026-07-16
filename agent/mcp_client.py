"""MCP（Model Context Protocol）客户端：stdio transport。

MCP 是外部服务统一接入协议。不需要为每个外部服务（Jira、Notion、数据库）
重写工具代码，只需实现 MCP 标准接口（tools/list + tools/call）。

本模块实现 stdio transport：启动子进程，通过 stdin/stdout 交换 JSON-RPC。

配置文件 ~/.agent/.mcp.json：
    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
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
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP 客户端（单个 server 连接）
# ---------------------------------------------------------------------------

class MCPClient:
    """单个 MCP server 的客户端连接。"""

    def __init__(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        # 工具过滤：include 优先于 exclude
        # include 只保留列出的；exclude 跳过列出的
        self.include = include
        self.exclude = exclude
        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._connected = False

    def connect(self) -> None:
        """启动子进程并发送 initialize 握手。"""
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

        # initialize 握手
        resp = self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "HarvilAgent", "version": "0.1.0"},
        })
        if not resp:
            raise RuntimeError(f"MCP server {self.name} initialize 无响应")

        # 发送 initialized 通知
        self._notify("notifications/initialized", {})
        self._connected = True
        logger.info("MCP server %s 已连接", self.name)

    def list_tools(self) -> List[dict]:
        """列出 server 提供的工具。"""
        resp = self._request("tools/list", {})
        return resp.get("tools", []) if resp else []

    def call_tool(self, name: str, arguments: dict) -> dict:
        """调用工具，返回结果。"""
        return self._request("tools/call", {
            "name": name,
            "arguments": arguments or {},
        }) or {}

    def _request(self, method: str, params: dict) -> Optional[dict]:
        """发送 JSON-RPC 请求并等待响应。"""
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError(f"MCP server {self.name} 未运行")

        with self._lock:
            self._request_id += 1
            msg = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            }
            try:
                self.process.stdin.write(json.dumps(msg) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise RuntimeError(f"MCP server {self.name} 写入失败: {e}")

            # 读响应（跳过 notification，只处理有 id 的响应）
            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError(f"MCP server {self.name} 断开")
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("MCP 非 JSON 行: %s", line.strip())
                    continue
                # 只处理对当前请求的响应
                if data.get("id") == self._request_id:
                    if "error" in data:
                        err = data["error"]
                        raise RuntimeError(
                            f"MCP 错误 {err.get('code')}: {err.get('message')}"
                        )
                    return data.get("result")

    def _notify(self, method: str, params: dict) -> None:
        """发送通知（不等待响应）。"""
        if self.process is None:
            return
        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        try:
            self.process.stdin.write(json.dumps(msg) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def close(self) -> None:
        """关闭连接。"""
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
    def connected(self) -> bool:
        return self._connected and self.process is not None


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_mcp_config(config_path=None) -> Dict[str, dict]:
    """加载 .mcp.json 配置。

    返回 {server_name: {command, args, env}}。
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

        配置里可选 include/exclude 字段（include 优先）：
            "github": {
                "command": "...",
                "include": ["create_issue"],
                "exclude": []
            }
        """
        if config is None:
            config = load_mcp_config()

        for name, cfg in config.items():
            try:
                client = MCPClient(
                    name=name,
                    command=cfg.get("command", ""),
                    args=cfg.get("args", []),
                    env=cfg.get("env", {}),
                    include=cfg.get("include"),
                    exclude=cfg.get("exclude"),
                )
                client.connect()
                with self._lock:
                    self._clients[name] = client
            except Exception as e:
                logger.warning("MCP server %s 连接失败: %s", name, e)

    def get_all_tools(self) -> List[dict]:
        """获取所有 server 的工具列表（含 server 名前缀）。

        应用每个 client 的 include/exclude 过滤（include 优先于 exclude）。
        """
        all_tools = []
        with self._lock:
            clients = list(self._clients.items())

        for server_name, client in clients:
            if not client.connected:
                continue
            try:
                tools = client.list_tools()
                # 读取 include/exclude（兼容 MagicMock 等动态属性）
                include = getattr(client, "include", None)
                exclude = getattr(client, "exclude", None)
                # 只接受真实 list/None，避免 MagicMock 属性干扰
                if not isinstance(include, (list, type(None))):
                    include = None
                if not isinstance(exclude, (list, type(None))):
                    exclude = None

                for tool in tools:
                    tool_name = tool.get("name", "")
                    # include/exlude 过滤（include 优先）
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
