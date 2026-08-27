"""团队协作（Agent Teams）工具：给「多个 AI 组队干活」提供的 6 个操作入口。

打个比方：主对话像项目经理，可以拉几个帮手（子代理，主对话派出去
帮忙干活的分身，各自是独立进程）组成一个小组，互相发消息、分活、收活。
本文件把这些操作做成工具，AI 在对话里就能直接调用：

- team_send：给另一个成员发消息
- team_inbox：读自己的收件箱（消费式——像取快递，取完就清空）
- team_members：列出团队都有谁、各自什么状态
- team_spawn：招一个新帮手（启动子代理进程）
- team_shutdown：让某个帮手下班（发关闭消息并等它退出）
- idle：帮手声明「我没活了」，进入待命状态等新任务

handler 被调用时，框架通过 kwargs 把三样东西透传进来：
team_bus（消息总线，成员间发信的邮局）、team_coordinator（协调员，
管成员名单和进程生死）、team_name（自己叫什么名字）。
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
    "description": (
        "启动一个子 agent 进程处理任务（一次性，完成后自动退出）。\n"
        "可选 task_id：若提供，worker 进程被绑定到该 task，"
        "其内部的 task_update / task_complete 只能操作该任务（防止 prompt 注入跨任务操作）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "新 agent 的名字（必须唯一）"},
            "role": {"type": "string", "default": "worker"},
            "task": {"type": "string", "description": "交给新 agent 跑的 prompt"},
            "task_id": {
                "type": "string",
                "description": (
                    "可选。绑定的 TaskStore 任务 ID。"
                    "若提供，Coordinator 会先 claim 该任务（owner=name, status=in_progress），"
                    "并把 OMNIMATE_KANBAN_TASK 注入子进程 env。"
                ),
            },
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

IDLE_SCHEMA = {
    "name": "idle",
    "description": (
        "声明当前没有更多工作要做，进入 IDLE 状态等新任务。"
        "仅在 autonomous worker 模式下有意义；主 agent 调用是 no-op。"
        "注意：idle 只对 team_name 不是 main 的 worker agent 生效。"
    ),
    "parameters": {"type": "object", "properties": {}},
}


def _handle_team_send(args: dict, **kwargs) -> str:
    """给另一个团队成员发消息（投递到对方收件箱，不等回复）。

    参数：
    - args：工具参数，to（收件人名字）和 content（消息内容）必填，
      msg_type 可选（message/request/response/shutdown 四种，默认
      message），request_id 可选（配对请求/响应用）。
    - kwargs：框架透传的上下文，取 team_bus（消息总线）和 team_name
      （自己的名字，作发件人）。

    返回：JSON 字符串，成功含消息 id；总线没初始化/参数缺失/内部出错
    时返回相应错误。
    """
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
    """读自己收件箱里的所有消息（消费式：像取快递，读完信箱就清空）。

    参数：
    - args：工具参数（本工具不需要参数）。
    - kwargs：框架透传的上下文，取 team_bus（消息总线）和 team_name
      （自己的名字，决定读哪个信箱）。

    返回：JSON 字符串，含消息列表（id/发件人/类型/内容/时间等）和
    count 总数；总线没初始化或读取出错时返回相应错误。
    """
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
    """列出团队所有成员及其状态（点名：都有谁、什么角色、活着没）。

    参数：
    - args：工具参数（本工具不需要参数）。
    - kwargs：框架透传的上下文，取 team_coordinator（管成员名单的协调员）。

    返回：JSON 字符串，含成员列表（名字/角色/进程号/状态/创建时间）和
    count 总数；协调员没初始化时返回错误。
    """
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
    """招一个新帮手：启动一个子代理进程去干指定的活（一次性，干完自动退出）。

    子代理是独立进程，不继承主对话的记忆，只靠 task 参数里写的指令干活。
    可选绑定一个任务 ID，绑了之后帮手只能操作那个任务——防 prompt 注入
    跨任务操作的安全设计。

    参数：
    - args：工具参数，name（新成员名字，不能重名）和 task（交给它干的
      活，一段指令文本）必填；role 可选（角色，默认 worker）；task_id
      可选（绑定的任务 ID）。
    - kwargs：框架透传的上下文，取 team_coordinator（真正去拉进程）、
      agent_ref（查当前嵌套深度）、config（读最大深度限制）。

    返回：JSON 字符串，成功含新成员名字/进程号/状态；超过最大嵌套
    深度、名字或活没给、任务 ID 不存在等情况返回相应错误。
    """
    coord = kwargs.get("team_coordinator")
    agent = kwargs.get("agent_ref")
    config = kwargs.get("config") or {}
    if coord is None:
        return _err("team coordinator 未初始化", "team_unavailable")
    name = args.get("name")
    task = args.get("task")
    if not name or not task:
        return _err("name 和 task 必需", "invalid_args")

    # 嵌套深度检查：帮手也能再招帮手，但不能无限套娃——
    # 当前层数达到上限就直接拒绝
    current_depth = getattr(agent, "spawn_depth", 0) if agent else 0
    max_depth = config.get("team", {}).get("max_depth", 2)
    if current_depth >= max_depth:
        return json.dumps({
            "success": False,
            "error": f"max_depth {max_depth} reached (current: {current_depth})",
            "error_type": "team_max_depth",
        }, ensure_ascii=False)

    role = args.get("role", "worker")
    # 空字符串和 None 统一当成「不绑定任务」处理，免得空串走绑定逻辑
    task_id = args.get("task_id") or None

    try:
        member = coord.spawn(
            name=name, role=role, task=task,
            depth=current_depth + 1,
            task_id=task_id,
        )
        return json.dumps({
            "success": True, "name": name, "pid": member.pid,
            "status": member.status,
            "task_id": task_id,
        }, ensure_ascii=False)
    except ValueError as e:
        # 比如 task_id 在任务库里不存在
        msg = str(e)
        error_type = "invalid_task_id" if "不存在" in msg else "team_spawn_error"
        return _err(msg, error_type)
    except RuntimeError as e:
        return _err(str(e), "team_spawn_error")
    except Exception as e:
        return _err(f"spawn 失败: {e}", "team_error")


def _handle_team_shutdown(args: dict, **kwargs) -> str:
    """让某个成员下班：给它发关闭消息并等它退出。

    参数：
    - args：工具参数，name 必填（要关闭的成员名字）。
    - kwargs：框架透传的上下文，取 team_coordinator（执行关闭）和
      team_bus（发关闭消息）。

    返回：JSON 字符串，成功含 status=shutdown；名字没给、找不到人或
      它已经不在运行时返回相应错误。
    """
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


def _handle_idle(args: dict, **kwargs) -> str:
    """帮手声明「我没活了」：进入待命（IDLE）状态等新任务。

    自动干活的帮手（autonomous worker）忙完一轮后调这个工具说一声
    「闲了」，等协调员派新活。

    参数：
    - args：工具参数（本工具不需要参数）。
    - kwargs：框架透传的上下文，取 agent_ref（要标记闲置的 agent）。

    返回：JSON 字符串（success=True + 说明信息）。

    两类安全兜底（都不真的闲置）：主对话（team_name 是 "main" 或
    None）调用是空操作，防止主对话误把自己中断；上下文里没有
    agent_ref 时同样空操作。
    """
    agent = kwargs.get("agent_ref")
    if agent is None:
        return json.dumps({
            "success": True, "message": "idle (no-op, no agent ref)",
        }, ensure_ascii=False)
    # 主对话调 idle 不生效（空操作），防止误中断自己
    team_name = getattr(agent, "team_name", None)
    if team_name is None or team_name == "main":
        return json.dumps({
            "success": True, "message": "idle (no-op for main agent)",
        }, ensure_ascii=False)
    agent._idle_requested = True
    return json.dumps({
        "success": True, "message": "idle requested",
    }, ensure_ascii=False)


def _err(msg: str, error_type: str) -> str:
    """拼一个统一格式的错误 JSON（success=False + 错误信息 + 错误类型）。

    参数：
    - msg：给人看的错误描述。
    - error_type：给程序分类用的错误类型码。

    返回：JSON 字符串。
    """
    return json.dumps({"success": False, "error": msg,
                       "error_type": error_type}, ensure_ascii=False)


# ---- 注册：import 本文件即自动登记进中央注册表 ----
registry.register(
    name="team_send", toolset="team",
    schema=TEAM_SEND_SCHEMA, handler=_handle_team_send, emoji="📤",
    isConcurrencySafe=False,  # 会改别人的收件箱（发消息进信箱），不能并发
)
registry.register(
    name="team_inbox", toolset="team",
    schema=TEAM_INBOX_SCHEMA, handler=_handle_team_inbox, emoji="📥",
    isConcurrencySafe=False,  # 读完就清空（消费式），并发会互相吞消息，不能并发
)
registry.register(
    name="team_members", toolset="team",
    schema=TEAM_MEMBERS_SCHEMA, handler=_handle_team_members, emoji="👥",
    isConcurrencySafe=True,  # 只读点名，无副作用，可并发
)
registry.register(
    name="team_spawn", toolset="team",
    schema=TEAM_SPAWN_SCHEMA, handler=_handle_team_spawn, emoji="🚀",
    isConcurrencySafe=False,  # 要拉起子代理进程，不能并发
)
registry.register(
    name="team_shutdown", toolset="team",
    schema=TEAM_SHUTDOWN_SCHEMA, handler=_handle_team_shutdown, emoji="🛑",
    isConcurrencySafe=False,  # 要关子代理进程，不能并发
)
registry.register(
    name="idle", toolset="team",
    schema=IDLE_SCHEMA, handler=_handle_idle, emoji="💤",
    isConcurrencySafe=False,  # 要改帮手状态机到 IDLE，不能并发
)
