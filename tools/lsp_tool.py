"""LSP 工具：跳转定义 / 查引用（R26 #17）。

比 grep 准的地方：能分清重名（只返回真正的符号定义/引用点）。
实现：subprocess 起 pylsp --stdio（shutil.which 探测，没装自动隐藏——
check_fn 门控），JSON-RPC over stdin/stdout，单实例复用。

窄腰裁决：核心工具 + check_fn 门控（pylsp 不进项目依赖，用户可选装
`pipx install python-lsp-server`）。isConcurrencySafe=False——子进程
stdin/stdout 有状态交互，串行保守。

相对任务简报的四处实现修正（语义不变，原因见各项）：
1. rootUri 用顶部 `import pathlib` 常规化（简报内联 walrus + __import__
   是演示性写法，任务指示要求清理）。
2. server 启动 + initialize 握手收敛进 _ensure_ready()（持锁调用，返回
   bool）：pylsp 不在 PATH 时返回 False 而非 raise——handler 据此跳过
   didOpen，使「fake _rpc_request 注入」的测试无需真起 pylsp；真实路径
   _rpc_request 里 False 会 raise，由 handler except 兜成 lsp_error。
   顺带修正消息顺序为 initialize → didOpen → request（简报原实现 didOpen
   先于 initialize 握手，违反 LSP 生命周期）。
3. 子进程管道用二进制模式（简报 text=True + encoding）：Windows 上
   TextIOWrapper(newline=None) 读侧把 \\r\\n 翻译成 \\n（header 定界
   "\\r\\n\\r\\n" 永远匹配不上）、写侧把 \\n 翻译成 \\r\\n（帧头 "\\r\\n"
   变 "\\r\\r\\n"），Content-Length 帧协议直接损坏。二进制 + 手工
   encode/decode 与 Content-Length 的字节语义严格一致。
4. _read 按响应 id 匹配、跳过通知帧：didOpen 必然触发 server 的
   publishDiagnostics 通知（无 id），"读一帧就当响应"会错位拿到
   通知的 result=None → 生产环境必然查不到结果。

已知边界（接受）：server 挂死不吐字节时 stdout.read(1) 阻塞，10s
deadline 只能兜"慢速滴流"和 EOF（server 崩溃）场景；完全不吐字节的
挂死靠 handler 层的异常重建兜底。
"""
import json
import logging
import os
import pathlib
import shutil
import subprocess
import threading
import time
from typing import Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

_SERVER = {"proc": None, "lock": threading.Lock(), "id": 0, "init": False}
_RPC_TIMEOUT = 10.0


def _lsp_available() -> bool:
    return shutil.which("pylsp") is not None


def _reset_server() -> None:
    """关掉 server（check_fn 变 False / 测试用）。"""
    with _SERVER["lock"]:
        proc = _SERVER.get("proc")
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.close()
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                logger.debug("关闭 pylsp 子进程失败（忽略，进程可能已退出）", exc_info=True)
        _SERVER["proc"] = None
        _SERVER["init"] = False


def _ensure_server() -> bool:
    """懒启动 pylsp --stdio（持锁调用）。返回 False = pylsp 不在 PATH。"""
    if _SERVER["proc"] is not None and _SERVER["proc"].poll() is None:
        return True
    exe = shutil.which("pylsp")
    if exe is None:
        return False
    # 二进制管道（不 text=True）：Windows 文本流 \r\n 翻译会损坏帧协议，见模块头 3
    proc = subprocess.Popen(
        [exe, "--stdio"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    _SERVER["proc"] = proc
    _SERVER["init"] = False
    return True


def _ensure_ready() -> bool:
    """确保 server 存在 + initialize 握手完成（持锁调用）。

    返回 False = pylsp 不可用（不 raise，调用方降级处理）。
    """
    if not _ensure_server():
        return False
    if not _SERVER["init"]:
        proc = _SERVER["proc"]
        _send(proc, 0, "initialize", {
            "processId": os.getpid(),
            "rootUri": pathlib.Path.cwd().as_uri(),
            "capabilities": {},
        })
        _read(proc, 0)
        _send_notification(proc, "initialized", {})
        _SERVER["init"] = True
    return True


def _rpc_request(method: str, params: dict) -> Optional[object]:
    """同步 JSON-RPC 请求（Content-Length 帧）。测试可 monkeypatch 本函数。

    首次调用经 _ensure_ready 自动完成 server 启动 + initialize 握手
    （workspace rootUri = cwd）。
    """
    with _SERVER["lock"]:
        if not _ensure_ready():
            raise RuntimeError("pylsp 未安装（pipx install python-lsp-server）")
        proc = _SERVER["proc"]
        _SERVER["id"] += 1
        rid = _SERVER["id"]
        _send(proc, rid, method, params)
        return _read(proc, rid)


def _send(proc, rid: int, method: str, params: dict) -> None:
    body = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
    _write_frame(proc, body)


def _send_notification(proc, method: str, params: dict) -> None:
    body = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
    _write_frame(proc, body)


def _write_frame(proc, body: str) -> None:
    """按 Content-Length 帧写一条消息（二进制，字节计数）。"""
    payload = body.encode("utf-8")
    header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
    proc.stdin.write(header + payload)
    proc.stdin.flush()


def _read_frame(proc, deadline: float) -> Optional[bytes]:
    """读一帧原始字节（按 Content-Length，二进制）。超时/EOF → None。"""
    header = b""
    # 逐字节读 header（pylsp 输出无行缓冲保证时的保守做法）
    while time.time() < deadline:
        ch = proc.stdout.read(1)
        if not ch:
            return None  # EOF：server 崩溃/退出
        header += ch
        if header.endswith(b"\r\n\r\n"):
            break
    else:
        return None  # deadline 到了 header 仍不完整
    try:
        length = int(
            next(line.split(b":", 1)[1] for line in header.strip().split(b"\r\n")
                 if line.lower().startswith(b"content-length"))
        )
    except (StopIteration, ValueError):
        return None
    body = proc.stdout.read(length)
    if len(body) < length:
        return None
    return body


def _read(proc, rid: int) -> Optional[object]:
    """读到 id 匹配的响应帧并取 result（超时/EOF/格式错 → None）。

    跳过通知帧（window/logMessage、publishDiagnostics 无 id）和乱序旧响应，
    只认 id == rid 的响应；错误响应 raise（由 handler 兜成 lsp_error）。
    """
    deadline = time.time() + _RPC_TIMEOUT
    while time.time() < deadline:
        frame = _read_frame(proc, deadline)
        if frame is None:
            return None
        try:
            msg = json.loads(frame)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(msg, dict):
            continue
        if msg.get("id") != rid:
            continue  # 通知帧 / 乱序响应 → 丢弃继续等
        if "error" in msg:
            raise RuntimeError(f"LSP 错误响应: {msg['error']}")
        return msg.get("result")
    return None


def _handle_lsp(args: dict, **kwargs) -> str:
    """lsp 工具 handler：definitions / references 两动作。"""
    if not _lsp_available():
        return json.dumps({
            "error": "pylsp 未安装（pipx install python-lsp-server 后重启会话）",
            "error_type": "lsp_unavailable",
        }, ensure_ascii=False)
    path = str(args.get("path", ""))
    line = int(args.get("line", 0))
    character = int(args.get("character", 0))
    action = str(args.get("action", ""))
    if not path or not pathlib.Path(path).exists():
        return json.dumps({"error": f"文件不存在: {path}", "error_type": "invalid_path"}, ensure_ascii=False)
    method = {"definitions": "textDocument/definition",
              "references": "textDocument/references"}.get(action)
    if method is None:
        return json.dumps({"error": f"未知 action: {action}", "error_type": "invalid_action"}, ensure_ascii=False)
    doc_uri = pathlib.Path(path).resolve().as_uri()
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    params = {
        "textDocument": {"uri": doc_uri},
        "position": {"line": line, "character": character},
    }
    if action == "references":
        params["context"] = {"includeDeclaration": True}
    try:
        with _SERVER["lock"]:
            # didOpen 让 server 拿到未落盘内容（也覆盖已落盘的）；
            # _ensure_ready False（pylsp 不可用）时跳过——真实路径
            # _rpc_request 会 raise 由 except 兜底，fake 注入路径直接返回结果
            if _ensure_ready():
                _send_notification(_SERVER["proc"], "textDocument/didOpen", {
                    "textDocument": {"uri": doc_uri, "languageId": "python",
                                     "version": 1, "text": text},
                })
        result = _rpc_request(method, params)
        items = result if isinstance(result, list) else ([result] if result else [])
        return json.dumps({
            "action": action,
            "results": [
                {"uri": it.get("uri"), "line": it.get("range", {}).get("start", {}).get("line", 0),
                 "character": it.get("range", {}).get("start", {}).get("character", 0)}
                for it in items if isinstance(it, dict)
            ][:50],
        }, ensure_ascii=False)
    except Exception as e:
        _reset_server()  # server 状态可疑 → 重建（对齐 CC discard 语义）
        return json.dumps({"error": f"LSP 调用失败: {e}", "error_type": "lsp_error"}, ensure_ascii=False)


registry.register(
    name="lsp",
    toolset="core",
    schema={
        "name": "lsp",
        "description": (
            "代码符号导航（比 grep 准：能分清重名）。"
            "action=definitions 跳转定义；references 查所有引用点。"
            "line/character 从 0 计。需要 pylsp（未安装时本工具自动隐藏）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["definitions", "references"],
                           "description": "definitions=跳定义 / references=查引用"},
                "path": {"type": "string", "description": "当前文件路径"},
                "line": {"type": "integer", "description": "符号所在行（0 起）"},
                "character": {"type": "integer", "description": "符号所在列（0 起）"},
            },
            "required": ["action", "path", "line", "character"],
        },
    },
    handler=_handle_lsp,
    check_fn=_lsp_available,
    emoji="🧭",
    isConcurrencySafe=False,  # 子进程 stdin/stdout 有状态，串行保守
)
