"""信箱（mailbox）工具：让多个 AI 之间像发邮件一样异步通信。

打个比方：每个 agent 有一个信箱。寄信人把信扔进去就走了（不等回信，
即 fire-and-forget「发了就不管」），收信人有空时自己开箱看。区别于
团队工具（team_tool）的实时消息，这里强调异步——发和收完全解耦。

3 个工具：
- mailbox_send：给另一个 agent 寄信（不等回应）
- mailbox_check：开自己的信箱看信（默认只看没读过的）
- mailbox_clear：把自己信箱里的信全扔掉

怎么拿到信箱：从 kwargs["agent_ref"] 取 AIAgent 实例，再读它的
_mailbox / _agent_name 字段（RuntimeContext 里创建信箱，
并用 set_mailbox 挂到 agent 身上）。

并发分类（能不能同时跑）：
- mailbox_send：isConcurrencySafe=False（要写信箱，不能并发）
- mailbox_check：isConcurrencySafe=True（只读，可并发）
- mailbox_clear：isConcurrencySafe=False（要删信，不能并发）
"""
import json

from tools.registry import registry


MAILBOX_SEND_SCHEMA = {
    "name": "mailbox_send",
    "description": (
        "异步投递邮件给另一个 agent（不等响应）。"
        "用于跨 agent 的非阻塞通信（fire-and-forget）。"
    ),
    "parameters": {
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
    "parameters": {
        "type": "object",
        "properties": {
            "unread_only": {"type": "boolean", "default": True},
        },
    },
}

MAILBOX_CLEAR_SCHEMA = {
    "name": "mailbox_clear",
    "description": "清空自己 mailbox。",
    "parameters": {"type": "object", "properties": {}},
}


def _resolve_mailbox_ctx(kwargs: dict):
    """从工具调用的上下文里提取「信箱 + 自己的名字」这两样东西。

    背景：中央注册表分发工具时会把 agent_ref（AIAgent 实例）放在
    kwargs 里透传。信箱挂在 agent 的 _mailbox 字段、
    名字挂在 _agent_name 字段。所有「拿不到怎么办」的兜底逻辑集中在
    这一个函数里，三个 handler 不用各写一遍。

    参数：
    - kwargs：框架透传的上下文字典。

    返回：二元组 (mailbox, agent_name)。拿不到 agent_ref 时返回
    (None, "main")；字段缺失时回退到公开字段名（mailbox / agent_name）。
    """
    agent_ref = kwargs.get("agent_ref")
    if agent_ref is None:
        return None, "main"
    # AIAgent 的字段名带下划线（私有约定），外面统一经 agent_ref 读取
    mailbox = getattr(agent_ref, "_mailbox", None) or getattr(agent_ref, "mailbox", None)
    agent_name = getattr(agent_ref, "_agent_name", None) or getattr(
        agent_ref, "agent_name", "main"
    )
    return mailbox, agent_name


def _handle_mailbox_send(args: dict, **kwargs) -> str:
    """给另一个 agent 寄一封信（扔进对方信箱就走，不等回应）。

    参数：
    - args：工具参数，to（收件人 agent 名）和 content（信的内容）必填，
      kind 可选（信的类型：message/task/status_update/alert，默认 message）。
    - kwargs：框架透传的上下文，用来取信箱和自己的名字（作发件人）。

    返回：JSON 字符串，成功含 msg_id 和收件人；信箱没配置时返回
    not_configured 错误。
    """
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
    """开自己的信箱看信（默认只看没读过的）。

    参数：
    - args：工具参数，unread_only 可选（True 只看未读，False 看全部，
      默认 True）。
    - kwargs：框架透传的上下文，用来取信箱和自己的名字。

    返回：JSON 字符串，含消息列表和 count 总数；信箱没配置时返回
    not_configured 错误。
    """
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
    """把自己信箱里的信全部扔掉。

    参数：
    - args：工具参数（本工具不需要参数）。
    - kwargs：框架透传的上下文，用来取信箱和自己的名字。

    返回：JSON 字符串，cleared 是扔掉的信的数量；信箱没配置时返回
    not_configured 错误。
    """
    mailbox, agent_name = _resolve_mailbox_ctx(kwargs)
    if mailbox is None:
        return json.dumps(
            {"error": "mailbox not configured", "error_type": "not_configured"}
        )
    count = mailbox.clear(agent_name)
    return json.dumps({"cleared": count})


# 模块顶部注册：import 本文件即自动登记进中央注册表
# 历史踩坑：LLM 能不能看到工具由 toolsets._CORE_TOOLS 清单决定——
# 曾漏列 mailbox，导致只有 CLI 的 /mailbox 命令能用，LLM 调不到。
# 另外 mailbox_send 列入了 ASYNC_AGENT_DISALLOWED_TOOLS（后台子代理
# 不许发信，和 team_send 同一个道理：防止分身乱传消息）。
registry.register(
    name="mailbox_send",
    schema=MAILBOX_SEND_SCHEMA,
    handler=_handle_mailbox_send,
    toolset="team",
    isConcurrencySafe=False,  # 要写信箱，有副作用，不能并发
)
registry.register(
    name="mailbox_check",
    schema=MAILBOX_CHECK_SCHEMA,
    handler=_handle_mailbox_check,
    toolset="team",
    isConcurrencySafe=True,  # 只读，可并发
)
registry.register(
    name="mailbox_clear",
    schema=MAILBOX_CLEAR_SCHEMA,
    handler=_handle_mailbox_clear,
    toolset="team",
    isConcurrencySafe=False,  # 要删信，有副作用，不能并发
)
