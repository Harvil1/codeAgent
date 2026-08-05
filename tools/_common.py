"""工具 handler 共用的小工具。

只放真正跨多个 tool 模块复用的 helper，不放业务逻辑。
"""
from typing import Optional


def get_mode_override_from_kwargs(kwargs: dict) -> Optional[str]:
    """从工具调用的 kwargs 里提取子代理 permission_mode override。

    必修 1：工具读 kwargs["agent_ref"].permission_mode，作为本次 check 的 mode override。
    线程安全：mode override 只影响本次调用，不修改全局 checker 状态。
    返回 "bypassPermissions" / "default" / None（无 agent_ref 时）。
    """
    agent_ref = kwargs.get("agent_ref")
    if agent_ref is None:
        return None
    mode = getattr(agent_ref, "permission_mode", None)
    if mode in ("default", "bypassPermissions"):
        return mode
    return None
