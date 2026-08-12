"""工具 handler 共用的小工具。

只放真正跨多个 tool 模块复用的 helper，不放业务逻辑。
"""
from typing import Optional


def get_mode_override_from_kwargs(kwargs: dict) -> Optional[str]:
    """从工具调用的 kwargs 里提取子代理 permission_mode override。

    必修 1：工具读 kwargs["agent_ref"].permission_mode，作为本次 check 的 mode override。
    线程安全：mode override 只影响本次调用，不修改全局 checker 状态。
    返回 "bypassPermissions" / "default" / "acceptEdits" / "autoDeny" / None（无 agent_ref 时）。
    """
    agent_ref = kwargs.get("agent_ref")
    if agent_ref is None:
        return None
    mode = getattr(agent_ref, "permission_mode", None)
    # S2 fix: acceptEdits 必须透传，否则模式静默降级到 default（功能失效）
    # Task J fix: autoDeny 必须透传，否则 async 子代理破坏性命令静默降级到
    # default 走 approval_callback（auto_deny 短路失效，生产路径不触发）
    if mode in ("default", "bypassPermissions", "acceptEdits", "autoDeny"):
        return mode
    return None
