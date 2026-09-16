"""「代码导航」工具：跳转到定义 / 查所有引用。

查"这个函数在哪定义、谁在用它"时，文本搜索（grep）会被重名坑
（两个不同类都有 run 方法）。LSP（Language Server Protocol，代码编辑器
背后那套"懂语法"的服务）能分清重名，只返回真正的定义点/引用点。

实现方式：用 subprocess 拉起一个 pylsp 进程（python-lsp-server，
通过 stdio 通信），跟它说 JSON-RPC 协议问结果；进程全局只起一个反复用。
用户没装 pylsp 时工具自动从模型可见列表里消失（check_fn 门控：注册了
但按运行时条件决定显不显示）。

设计取舍：做成核心工具 + 门控显隐，而不是把 pylsp 塞进项目依赖——
想用的人自己 `pipx install python-lsp-server`。
标记为不可并发：与子进程的 stdin/stdout 对话是有状态的，两个请求
同时读写会把对话搅乱，保守排队执行。

实现要点：
1. 消息顺序必须是 initialize → didOpen → request（LSP 生命周期规定）。
2. 子进程管道用二进制模式（不能开 text=True）：Windows 文本模式会把换行
   自动转换（\\r\\n ↔ \\n），Content-Length 帧协议（按字节数定界的通信
   格式）直接坏掉；二进制 + 手工 encode/decode 才和"按字节计数"语义
   严格一致。
3. _read 按响应编号（id）对号入座、跳过通知帧：didOpen 一发，server
   必然回一条 publishDiagnostics 通知（没有 id），"读一条就当响应"会错
   拿到通知的空结果。

已知边界（接受）：server 挂死且一个字节都不吐时，逐字节读会卡住，
10 秒超时只能兜住"慢速滴流"和"进程崩了立刻 EOF"两种场景；
彻底挂死的靠 handler 层"出错就重建进程"兜底。
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
    """看机器上装没装 pylsp（既给工具门控用，也是 handler 的前置检查）。"""
    return shutil.which("pylsp") is not None


def _reset_server() -> None:
    """关掉 pylsp 子进程、清空就绪标记（下次查询会自动重新拉起）。

    进程状态不对（比如出过错）或测试需要干净环境时调用。
    """
    with _SERVER["lock"]:
        proc = _SERVER.get("proc")
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.close()
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                logger.warning("关闭 pylsp 子进程失败（忽略，进程可能已退出）", exc_info=True)
        _SERVER["proc"] = None
        _SERVER["init"] = False


def _ensure_server() -> bool:
    """需要时才启动 pylsp 子进程（必须在已持锁时调用）。

    返回：False 表示机器上没装 pylsp（PATH 里找不到）；True 表示进程可用
    （原本活着或刚启动的）。
    """
    if _SERVER["proc"] is not None and _SERVER["proc"].poll() is None:
        return True
    exe = shutil.which("pylsp")
    if exe is None:
        return False
    # 故意用二进制管道（不开 text=True）：Windows 文本模式会自动转换换行符，
    # 直接弄坏帧协议（详见模块头"实现要点"）
    proc = subprocess.Popen(
        [exe, "--stdio"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    _SERVER["proc"] = proc
    _SERVER["init"] = False
    return True


def _ensure_ready() -> bool:
    """确保 pylsp 进程已启动且完成过"握手"（initialize，必须在已持锁时调用）。

    LSP 协议规定先握手才能正经问事；握手只在第一次做，之后靠 init 标记跳过。

    返回：False = pylsp 不可用（不抛异常，调用方自己降级处理）。
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
    """给 pylsp 发一条正式请求并等对应答案（同步、按字节帧收发）。

    对外的主力入口——首次调用会自动完成进程启动和握手（工作区根目录用
    当前目录）。测试时可以用替身换掉本函数，不必真起 pylsp。

    参数：
    - method：LSP 方法名（如 "textDocument/definition"）
    - params：随请求带的参数字典（文件、位置等）

    返回：答案里的 result 字段；超时/连接断了返回 None；
    pylsp 没装时抛 RuntimeError（由上层兜成错误 JSON）。
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
    """发一条带编号的请求（有编号，等会儿才能对号收答案）。"""
    body = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
    _write_frame(proc, body)


def _send_notification(proc, method: str, params: dict) -> None:
    """发一条不用回信的通知（无编号，发了就完）。"""
    body = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
    _write_frame(proc, body)


def _write_frame(proc, body: str) -> None:
    """按 Content-Length 帧格式写一条消息。

    帧格式就是"报头写明正文有多少字节 + 正文"，必须按 utf-8 编码后的
    字节数算，不能按字符数。
    """
    payload = body.encode("utf-8")
    header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
    proc.stdin.write(header + payload)
    proc.stdin.flush()


def _read_frame(proc, deadline: float) -> Optional[bytes]:
    """收一帧完整的原始字节（先啃报头拿长度，再读定长的正文）。

    参数：
    - proc：pylsp 子进程
    - deadline：放弃读取的最后期限（时间戳）

    返回：正文的原始字节；超时或对方断线（EOF）返回 None。
    """
    header = b""
    # 一个字节一个字节啃报头：因为没法保证 pylsp 会按行输出，
    # 一次读多了会把正文混进报头里
    while time.time() < deadline:
        ch = proc.stdout.read(1)
        if not ch:
            return None  # 读到 EOF：对面进程没了（崩溃/退出）
        header += ch
        if header.endswith(b"\r\n\r\n"):
            break
    else:
        return None  # 到点了报头还没凑齐，按超时处理
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
    """一直收帧，直到等到编号对得上的那条答案，取出 result 字段。

    server 会主动塞各种通知（logMessage、publishDiagnostics，都没有编号），
    还可能有迟到的旧答案，必须对号入座——只认 id == rid 的那条。

    参数：
    - proc：pylsp 子进程
    - rid：自己发请求时用的编号，用来认领答案

    返回：答案的 result 字段；超时/断线/格式坏了返回 None；
    答案本身报错则抛 RuntimeError（由 handler 兜成 lsp_error 返回）。
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
            continue  # 不是我要的那条（通知或迟到的旧答案）→ 扔掉接着等
        if "error" in msg:
            raise RuntimeError(f"LSP 错误响应: {msg['error']}")
        return msg.get("result")
    return None


def _handle_lsp(args: dict, **kwargs) -> str:
    """查某个符号的定义位置或所有引用点，返回 JSON。

    模型给出文件+行列位置，本函数先确认 pylsp 在、文件在，再把文件内容
    同步给 server（didOpen）然后发查询。

    参数：
    - args：工具参数字典。action 二选一（definitions=跳定义 /
      references=查引用）；path 是文件路径；line/character 是符号
      所在行列（都从 0 数起）。
    - kwargs：运行时注入的命名上下文（本工具不依赖，占位满足统一签名）。

    返回：JSON 字符串，results 里每项含 uri/line/character（最多 50 项）；
    pylsp 没装/文件不存在/查询出错时返回对应 error。
    """
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
            # didOpen 把文件当前内容喂给 server（以内存里的为准，可能比
            # 磁盘上的新）；_ensure_ready 返回 False（没装 pylsp）就跳过
            # 这步——真实路径下后面的 _rpc_request 会抛异常被 except 兜住；
            # 测试用假请求替身时则能直接走到返回结果
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
        _reset_server()  # 出过错就不信任这个进程的状态，直接扔掉重建
        return json.dumps({"error": f"LSP 调用失败: {e}", "error_type": "lsp_error"}, ensure_ascii=False)


# import 本模块时顺手把工具登记进中央注册表（项目惯例：工具文件顶层自注册）
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
    isConcurrencySafe=False,  # 和 pylsp 子进程的对话有先后状态，两个请求并发会搅乱对话，保守排队
)
