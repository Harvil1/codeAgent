"""Agent Teams 工具：5 个 handler 注册到 registry。

工具：
- team_send: 发消息
- team_inbox: 读自己的收件箱（消费式）
- team_members: 列出团队成员
- team_spawn: 启动子 agent
- team_shutdown: 关闭子 agent

handler 通过 kwargs 接收 team_bus / team_coordinator / team_name（由 agent 透传）。
"""
import json
import logging

from tools.registry import registry

logger = logging.getLogger(__name__)


TEAM_SEND_SCHEMA = {
    "name": "team_send",
    "description": "向另一个 agent 发消息",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "接收者 agent 名"},
            "content": {"type": "string"},
            "msg_type": {
                "type": "string",
                "enum": ["message", "request", "response", "shutdown"],
                "default": "message",
            },
            "request_id": {"type": "string"},
        },
        "required": ["to", "content"],
    },
}

TEAM_INBOX_SCHEMA = {
    "name": "team_inbox",
    "description": "读自己收件箱的所有消息（消费式：读后清空）",
    "parameters": {"type": "object", "properties": {}},
}

TEAM_MEMBERS_SCHEMA = {
    "name": "team_members",
    "description": "列出所有团队成员及其状态",
    "parameters": {"type": "object", "properties": {}},
}

TEAM_SPAWN_SCHEMA = {
    "name": "team_spawn",
    "description": "启动一个子 agent 进程处理任务（一次性，完成后自动退出）",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "新 agent 的名字（必须唯一）"},
            "role": {"type": "string", "default": "worker"},
            "task": {"type": "string", "description": "交给新 agent 跑的 prompt"},
        },
        "required": ["name", "task"],
    },
}

TEAM_SHUTDOWN_SCHEMA = {
    "name": "team_shutdown",
    "description": "向另一个 agent 发 shutdown 消息并等其退出",
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
}


def _handle_team_send(args: dict, **kwargs) -> str:
    bus = kwargs.get("team_bus")
    team_name = kwargs.get("team_name", "main")
    if bus is None:
        return _err("team bus 未初始化", "team_unavailable")
    to = args.get("to")
    content = args.get("content")
    if not to or not content:
        return _err("to 和 content 必需", "invalid_args")
    try:
        mid = bus.send(
            from_=team_name, to=to,
            type_=args.get("msg_type", "message"),
            content=content,
            request_id=args.get("request_id"),
        )
        return json.dumps({"success": True, "id": mid, "to": to},
                          ensure_ascii=False)
    except ValueError as e:
        return _err(str(e), "invalid_args")
    except Exception as e:
        return _err(f"内部错误: {e}", "team_error")


def _handle_team_inbox(args: dict, **kwargs) -> str:
    bus = kwargs.get("team_bus")
    team_name = kwargs.get("team_name", "main")
    if bus is None:
        return _err("team bus 未初始化", "team_unavailable")
    try:
        msgs = bus.read_inbox(team_name)
        return json.dumps({
            "success": True, "count": len(msgs),
            "messages": [
                {
                    "id": m.id, "from": m.from_, "to": m.to,
                    "type": m.type, "content": m.content, "ts": m.ts,
                    "request_id": m.request_id,
                }
                for m in msgs
            ],
        }, ensure_ascii=False)
    except Exception as e:
        return _err(f"读取 inbox 失败: {e}", "team_error")


def _handle_team_members(args: dict, **kwargs) -> str:
    coord = kwargs.get("team_coordinator")
    if coord is None:
        return _err("team coordinator 未初始化", "team_unavailable")
    members = coord.list_members()
    return json.dumps({
        "success": True, "count": len(members),
        "members": [
            {
                "name": m.name, "role": m.role, "pid": m.pid,
                "status": m.status, "created_at": m.created_at,
            }
            for m in members
        ],
    }, ensure_ascii=False)


def _handle_team_spawn(args: dict, **kwargs) -> str:
    coord = kwargs.get("team_coordinator")
    if coord is None:
        return _err("team coordinator 未初始化", "team_unavailable")
    name = args.get("name")
    task = args.get("task")
    if not name or not task:
        return _err("name 和 task 必需", "invalid_args")
    role = args.get("role", "worker")
    try:
        member = coord.spawn(name=name, role=role, task=task)
        return json.dumps({
            "success": True, "name": name, "pid": member.pid,
            "status": member.status,
        }, ensure_ascii=False)
    except (ValueError, RuntimeError) as e:
        return _err(str(e), "team_spawn_error")
    except Exception as e:
        return _err(f"spawn 失败: {e}", "team_error")


def _handle_team_shutdown(args: dict, **kwargs) -> str:
    coord = kwargs.get("team_coordinator")
    bus = kwargs.get("team_bus")
    if coord is None:
        return _err("team coordinator 未初始化", "team_unavailable")
    name = args.get("name")
    if not name:
        return _err("name 必需", "invalid_args")
    ok = coord.shutdown(name)
    if not ok:
        return _err(f"未找到或仍在运行: {name}", "team_not_found")
    return json.dumps({
        "success": True, "name": name, "status": "shutdown",
    }, ensure_ascii=False)


def _err(msg: str, error_type: str) -> str:
    return json.dumps({"success": False, "error": msg,
                       "error_type": error_type}, ensure_ascii=False)


# ---- 注册 ----
registry.register(
    name="team_send", toolset="team",
    schema=TEAM_SEND_SCHEMA, handler=_handle_team_send, emoji="📤",
)
registry.register(
    name="team_inbox", toolset="team",
    schema=TEAM_INBOX_SCHEMA, handler=_handle_team_inbox, emoji="📥",
)
registry.register(
    name="team_members", toolset="team",
    schema=TEAM_MEMBERS_SCHEMA, handler=_handle_team_members, emoji="👥",
)
registry.register(
    name="team_spawn", toolset="team",
    schema=TEAM_SPAWN_SCHEMA, handler=_handle_team_spawn, emoji="🚀",
)
registry.register(
    name="team_shutdown", toolset="team",
    schema=TEAM_SHUTDOWN_SCHEMA, handler=_handle_team_shutdown, emoji="🛑",
)
