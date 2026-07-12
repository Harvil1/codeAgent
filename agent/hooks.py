"""Hooks 系统：扩展 agent 主循环行为的注册表机制。

4 种 event：USER_PROMPT_SUBMIT / PRE_TOOL_USE / POST_TOOL_USE / STOP
2 种注册：programmatic（Python 函数）/ declarative（子进程脚本，T3 实现）
失败 fail-open 默认（log + 视为 None）；PreToolUse 可选 fail_closed。
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class HookEvent(Enum):
    USER_PROMPT_SUBMIT = "user_prompt_submit"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    STOP = "stop"


# 程序式 hook 的 4 种签名
UserPromptSubmitFn = Callable[[str], Optional[str]]
PreToolUseFn = Callable[[str, dict], Optional[dict]]
PostToolUseFn = Callable[[str, dict, str], Optional[str]]
StopFn = Callable[[], Optional[str]]


@dataclass
class HookScriptConfig:
    """声明式 hook 的子进程配置。"""
    command: list  # list[str]，如 ["python", "./hooks/audit.py"]
    timeout: float = 10.0
    env: Optional[dict] = None


@dataclass
class Hook:
    """统一包装：程序式或声明式。"""
    name: str
    event: HookEvent
    kind: str  # "programmatic" | "declarative"
    fn: Optional[Callable] = None
    script: Optional[HookScriptConfig] = None
    fail_closed: bool = False


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class HookRegistry:
    """管理所有 hook 注册和执行。实例由 RuntimeContext 持有，注入 AIAgent。"""

    def __init__(self):
        self._hooks: dict = {e: [] for e in HookEvent}
        self._stop_fire_count: int = 0

    # ---- 注册 ----
    def register_user_prompt_submit(self, fn, *, name=None):
        self._hooks[HookEvent.USER_PROMPT_SUBMIT].append(
            Hook(name=name or "anonymous", event=HookEvent.USER_PROMPT_SUBMIT,
                 kind="programmatic", fn=fn)
        )

    def register_pre_tool_use(self, fn, *, name=None, fail_closed=False):
        self._hooks[HookEvent.PRE_TOOL_USE].append(
            Hook(name=name or "anonymous", event=HookEvent.PRE_TOOL_USE,
                 kind="programmatic", fn=fn, fail_closed=fail_closed)
        )

    def register_post_tool_use(self, fn, *, name=None):
        self._hooks[HookEvent.POST_TOOL_USE].append(
            Hook(name=name or "anonymous", event=HookEvent.POST_TOOL_USE,
                 kind="programmatic", fn=fn)
        )

    def register_stop(self, fn, *, name=None):
        self._hooks[HookEvent.STOP].append(
            Hook(name=name or "anonymous", event=HookEvent.STOP,
                 kind="programmatic", fn=fn)
        )

    def register_declarative(self, hook: Hook):
        """T3 实现：把已构造好的 Hook 加到 registry。"""
        self._hooks[hook.event].append(hook)

    def clear(self, event=None):
        """清空（测试用）。"""
        if event is None:
            for e in self._hooks:
                self._hooks[e] = []
        else:
            self._hooks[event] = []
        self._stop_fire_count = 0

    # ---- 执行：USER_PROMPT_SUBMIT ----
    def run_user_prompt_submit(self, prompt: str, *, session_id: str) -> str:
        """链式：每个 hook 看到前一个的输出。失败 fail-open。"""
        for hook in self._hooks[HookEvent.USER_PROMPT_SUBMIT]:
            try:
                if hook.kind == "programmatic":
                    new_prompt = hook.fn(prompt)
                else:
                    # T3 接入 declarative
                    new_prompt = self._invoke_declarative_user_prompt(hook, prompt, session_id)
                if new_prompt is not None:
                    prompt = new_prompt
            except Exception as e:
                logger.warning("hook %s 异常（视为 None）: %s", hook.name, e)
        return prompt

    def _invoke_declarative_user_prompt(self, hook, prompt, session_id):
        """跑子进程，按 IPC 协议解析。返回新 prompt 或 None。"""
        from agent.hook_exec import run_script_hook  # 懒加载避免循环
        payload = {
            "event": "user_prompt_submit",
            "session_id": session_id,
            "timestamp": _now_iso(),
            "hook_name": hook.name,
            "prompt": prompt,
        }
        result = run_script_hook(hook, payload)
        if result is None:
            return None
        # IPC: {"prompt": "..."} → 替换；其他/空 → None
        return result.get("prompt")

    # ---- 执行：PRE_TOOL_USE ----
    def run_pre_tool_use(self, tool_name: str, args: dict, *,
                         session_id: str):
        """短路：首个 deny 胜出。返回 (deny_reason, modified_args)。

        - deny: Optional[str]，非 None 时拒绝
        - modified_args: Optional[dict]，非 None 时累计替换 args
        """
        deny_reason = None
        modified_args = None
        current_args = args
        for hook in self._hooks[HookEvent.PRE_TOOL_USE]:
            try:
                if hook.kind == "programmatic":
                    result = hook.fn(tool_name, current_args)
                else:
                    # T3 接入 declarative
                    result = self._invoke_declarative_pre_tool(hook, tool_name, current_args, session_id)
                if result is None:
                    continue
                if "deny" in result:
                    deny_reason = result["deny"]
                    return deny_reason, modified_args  # 短路
                if "modify_args" in result:
                    current_args = result["modify_args"]
                    modified_args = current_args
            except Exception as e:
                if hook.fail_closed:
                    logger.warning("hook %s fail_closed（视为拒绝）: %s", hook.name, e)
                    return str(e), None
                logger.warning("hook %s 异常（视为 None）: %s", hook.name, e)
        return deny_reason, modified_args

    def _invoke_declarative_pre_tool(self, hook, tool_name, args, session_id):
        """跑子进程，按 IPC 协议解析。返回 {deny: ...}/{modify_args: ...}/None。"""
        from agent.hook_exec import run_script_hook
        payload = {
            "event": "pre_tool_use",
            "session_id": session_id,
            "timestamp": _now_iso(),
            "hook_name": hook.name,
            "tool_name": tool_name,
            "args": args,
        }
        result = run_script_hook(hook, payload)
        if result is None:
            return None
        action = result.get("action", "allow")
        if action == "deny":
            return {"deny": result.get("reason", "unspecified")}
        if action == "modify":
            return {"modify_args": result.get("args", args)}
        return None

    # ---- 执行：POST_TOOL_USE ----
    def run_post_tool_use(self, tool_name: str, args: dict, result: str,
                          *, session_id: str) -> str:
        """链式：每个 hook 看到前一个的输出。"""
        for hook in self._hooks[HookEvent.POST_TOOL_USE]:
            try:
                if hook.kind == "programmatic":
                    new_result = hook.fn(tool_name, args, result)
                else:
                    new_result = self._invoke_declarative_post_tool(
                        hook, tool_name, args, result, session_id)
                if new_result is not None:
                    result = new_result
            except Exception as e:
                logger.warning("hook %s 异常（视为 None）: %s", hook.name, e)
        return result

    def _invoke_declarative_post_tool(self, hook, tool_name, args, result, session_id):
        """跑子进程，按 IPC 协议解析。返回新 result 或 None。"""
        from agent.hook_exec import run_script_hook
        payload = {
            "event": "post_tool_use",
            "session_id": session_id,
            "timestamp": _now_iso(),
            "hook_name": hook.name,
            "tool_name": tool_name,
            "args": args,
            "result": result,
        }
        proc_result = run_script_hook(hook, payload)
        if proc_result is None:
            return None
        return proc_result.get("result")

    # ---- 执行：STOP ----
    def run_stop(self, *, session_id: str, max_fires: int = 3) -> Optional[str]:
        """首个非 None 胜出。超过 max_fires 强制返回 None（防失控）。"""
        if self._stop_fire_count >= max_fires:
            logger.info("STOP hook 触发上限（%d/%d），本次跳过",
                        self._stop_fire_count, max_fires)
            return None
        for hook in self._hooks[HookEvent.STOP]:
            try:
                if hook.kind == "programmatic":
                    msg = hook.fn()
                else:
                    msg = self._invoke_declarative_stop(hook, session_id)
                if msg is not None:
                    self._stop_fire_count += 1
                    return msg
            except Exception as e:
                logger.warning("hook %s 异常（视为 None）: %s", hook.name, e)
        return None

    def _invoke_declarative_stop(self, hook, session_id):
        """跑子进程，按 IPC 协议解析。返回 continue 消息或 None。"""
        from agent.hook_exec import run_script_hook
        payload = {
            "event": "stop",
            "session_id": session_id,
            "timestamp": _now_iso(),
            "hook_name": hook.name,
        }
        result = run_script_hook(hook, payload)
        if result is None:
            return None
        return result.get("continue")
