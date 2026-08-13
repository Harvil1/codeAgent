"""mailbox 工具：让 teammate / agent 异步通信。

3 个工具：
- mailbox_send: 异步投递邮件给另一个 agent（不等响应）
- mailbox_check: 检查自己 mailbox 的邮件（默认只看未读）
- mailbox_clear: 清空自己 mailbox

ctx 通过 kwargs["agent_ref"] 拿到 AIAgent 实例，再从 _mailbox/_agent_name
字段读取（Task 12 在 RuntimeContext 注入 mailbox 并 set_mailbox 到 agent）。

并发分类（Task F1 规则）：
- mailbox_send: isConcurrencySafe=False（写投递）
- mailbox_check: isConcurrencySafe=True（只读）
- mailbox_clear: isConcurrencySafe=False（删除）
"""
import json

from tools.registry import registry


MAILBOX_SEND_SCHEMA = {
    "name": "mailbox_send",
    "description": (
        "异步投递邮件给另一个 agent（不等响应）。"
        "用于跨 agent 的非阻塞通信（fire-and-forget）。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "收件人 agent 名"},
            "content": {"type": "string", "description": "邮件内容"},
            "kind": {
                "type": "string",
                "enum": ["message", "task", "status_update", "alert"],
                "default": "message",
            },
        },
        "required": ["to", "content"],
    },
}

MAILBOX_CHECK_SCHEMA = {
    "name": "mailbox_check",
    "description": "检查自己 mailbox 的邮件。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "unread_only": {"type": "boolean", "default": True},
        },
    },
}

MAILBOX_CLEAR_SCHEMA = {
    "name": "mailbox_clear",
    "description": "清空自己 mailbox。",
    "inputSchema": {"type": "object", "properties": {}},
}


def _resolve_mailbox_ctx(kwargs: dict):
    """从工具调用的 kwargs 里提取 mailbox + agent_name。

    Task 12 接线：registry.dispatch 把 agent_ref（AIAgent 实例）作为 kwargs 透传。
    AIAgent 把 mailbox 挂在 `_mailbox` 字段，agent_name 挂在 `_agent_name`。
    本函数集中处理 fallback（无 agent_ref / 字段缺失）。
    """
    agent_ref = kwargs.get("agent_ref")
    if agent_ref is None:
        return None, "main"
    # AIAgent 内部字段是 _mailbox / _agent_name（私有意），外部通过 agent_ref 读取
    mailbox = getattr(agent_ref, "_mailbox", None) or getattr(agent_ref, "mailbox", None)
    agent_name = getattr(agent_ref, "_agent_name", None) or getattr(
        agent_ref, "agent_name", "main"
    )
    return mailbox, agent_name


def _handle_mailbox_send(args: dict, **kwargs) -> str:
    mailbox, agent_name = _resolve_mailbox_ctx(kwargs)
    if mailbox is None:
        return json.dumps(
            {"error": "mailbox not configured", "error_type": "not_configured"}
        )
    msg_id = mailbox.send(
        to=args["to"],
        from_=agent_name,
        content=args["content"],
        kind=args.get("kind", "message"),
    )
    return json.dumps({"msg_id": msg_id, "to": args["to"]})


def _handle_mailbox_check(args: dict, **kwargs) -> str:
    mailbox, agent_name = _resolve_mailbox_ctx(kwargs)
    if mailbox is None:
        return json.dumps(
            {"error": "mailbox not configured", "error_type": "not_configured"}
        )
    unread_only = args.get("unread_only", True)
    if unread_only:
        msgs = mailbox.check_unread(agent_name)
    else:
        msgs = mailbox.check_all(agent_name)
    return json.dumps({"messages": msgs, "count": len(msgs)})


def _handle_mailbox_clear(args: dict, **kwargs) -> str:
    mailbox, agent_name = _resolve_mailbox_ctx(kwargs)
    if mailbox is None:
        return json.dumps(
            {"error": "mailbox not configured", "error_type": "not_configured"}
        )
    count = mailbox.clear(agent_name)
    return json.dumps({"cleared": count})


# 模块顶部注册（import 即生效）
registry.register(
    name="mailbox_send",
    schema=MAILBOX_SEND_SCHEMA,
    handler=_handle_mailbox_send,
    toolset="team",
    isConcurrencySafe=False,  # 写投递，有副作用
)
registry.register(
    name="mailbox_check",
    schema=MAILBOX_CHECK_SCHEMA,
    handler=_handle_mailbox_check,
    toolset="team",
    isConcurrencySafe=True,  # 只读
)
registry.register(
    name="mailbox_clear",
    schema=MAILBOX_CLEAR_SCHEMA,
    handler=_handle_mailbox_clear,
    toolset="team",
    isConcurrencySafe=False,  # 删除，有副作用
)
