"""工具 handler（真正干活的处理函数）共用的小帮手函数。

这里只放被多个工具模块共同用到的小函数，不放具体业务逻辑——
就像家里共用的工具箱，谁需要谁来拿，不往里塞私人物品。
"""
from typing import Optional


def get_mode_override_from_kwargs(kwargs: dict) -> Optional[str]:
    """从工具调用的附加参数里，取出子代理（主对话派出去帮忙干活的分身）的权限模式。

    子代理可携带与主对话不同的权限模式，权限检查需按实际调用方取模式。

    参数：
        kwargs: 工具调用时带上的命名上下文（里面有 agent_ref，即发起调用的 agent 引用）。

    返回：权限模式字符串，取值是 "bypassPermissions"（跳过审批）/ "default"（默认，该问就问）/
    "acceptEdits"（自动接受编辑类操作）/ "autoDeny"（一律拒绝，async 子代理用）；
    如果没传 agent_ref 或模式值不认识，返回 None（表示不覆盖，走默认）。
    只影响本次调用，不改全局状态，多线程同时调也安全。
    """
    agent_ref = kwargs.get("agent_ref")
    if agent_ref is None:
        return None
    mode = getattr(agent_ref, "permission_mode", None)
    # 四种模式都必须在此透传——漏了任一种，对应调用方会悄悄退回 default（功能失效）
    if mode in ("default", "bypassPermissions", "acceptEdits", "autoDeny"):
        return mode
    return None
