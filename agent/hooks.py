"""Hooks 系统：扩展 agent 主循环行为的注册表机制。

11 种 event（P2-13 扩展后）：
  核心 6 种：USER_PROMPT_SUBMIT / PRE_TOOL_USE / POST_TOOL_USE / STOP
           + PRE_LLM_CALL / POST_LLM_CALL（batch2-T2）
  新增 5 种（P2-13）：SESSION_START / SESSION_END
           + PRE_COMPACT / POST_COMPACT + CONFIG_CHANGE
2 种注册：programmatic（Python 函数）/ declarative（子进程脚本）
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
    # batch2-T2: LLM 调用前后的 hook
    PRE_LLM_CALL = "pre_llm_call"
    POST_LLM_CALL = "post_llm_call"
    # P2-13 NEW: 会话/压缩/配置 5 个新事件
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PRE_COMPACT = "pre_compact"
    POST_COMPACT = "post_compact"
    CONFIG_CHANGE = "config_change"


# 程序式 hook 的签名
UserPromptSubmitFn = Callable[[str], Optional[str]]
PreToolUseFn = Callable[[str, dict], Optional[dict]]
PostToolUseFn = Callable[[str, dict, str], Optional[str]]
StopFn = Callable[[], Optional[str]]
# batch2-T2: LLM hooks
PreLLMCallFn = Callable[[list, Optional[list]], Optional[tuple]]
PostLLMCallFn = Callable[[object], Optional[object]]
# P2-13: 新事件用统一 payload 风格（dict 进，Optional[dict] 出）
# - SESSION_START/END/POST_COMPACT/CONFIG_CHANGE: 纯通知型，返回值忽略
# - PRE_COMPACT: 可返回 {"abort": True} 阻止该层压缩
PayloadFn = Callable[[dict], Optional[dict]]


@dataclass
class HookScriptConfig:
    """声明式 hook 配置（支持 5 种 handler 类型）。

    - command:  本地子进程（向后兼容，老 hook_loader 总会传它）
    - http:     POST JSON 到 url，解析响应
    - mcp_tool: 调 MCP 工具 mcp_server.mcp_tool
    - prompt:   单轮 aux_llm 评估
    - agent:    多轮子代理（delegate）评估
    """
    handler_type: str = "command"   # command | http | mcp_tool | prompt | agent
    command: Optional[list] = None  # list[str]，command 类型用（老配置仍是必需）
    url: Optional[str] = None       # http 类型用
    mcp_server: Optional[str] = None  # mcp_tool 类型用
    mcp_tool: Optional[str] = None    # mcp_tool 类型用
    prompt: Optional[str] = None      # prompt / agent 类型用（模板字符串，.format(**payload)）
    agent_name: Optional[str] = None  # agent 类型用（自定义子代理名，可选）
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

    # ---- P2-13 NEW: 会话/压缩/配置事件的注册 ----
    def register_session_start(self, fn, *, name=None):
        """fn(payload: dict) -> None。payload: {session_id, started_at, agent_home}。"""
        self._hooks[HookEvent.SESSION_START].append(
            Hook(name=name or "anonymous", event=HookEvent.SESSION_START,
                 kind="programmatic", fn=fn)
        )

    def register_session_end(self, fn, *, name=None):
        """fn(payload: dict) -> None。payload: {session_id, reason, ended_at}。"""
        self._hooks[HookEvent.SESSION_END].append(
            Hook(name=name or "anonymous", event=HookEvent.SESSION_END,
                 kind="programmatic", fn=fn)
        )

    def register_pre_compact(self, fn, *, name=None):
        """fn(payload: dict) -> Optional[{"abort": True}]。payload: {layer, messages_count, est_tokens}。"""
        self._hooks[HookEvent.PRE_COMPACT].append(
            Hook(name=name or "anonymous", event=HookEvent.PRE_COMPACT,
                 kind="programmatic", fn=fn)
        )

    def register_post_compact(self, fn, *, name=None):
        """fn(payload: dict) -> None。payload: {messages_before, messages_after, layer}。"""
        self._hooks[HookEvent.POST_COMPACT].append(
            Hook(name=name or "anonymous", event=HookEvent.POST_COMPACT,
                 kind="programmatic", fn=fn)
        )

    def register_config_change(self, fn, *, name=None):
        """fn(payload: dict) -> None。payload: {changed_keys, old, new}。"""
        self._hooks[HookEvent.CONFIG_CHANGE].append(
            Hook(name=name or "anonymous", event=HookEvent.CONFIG_CHANGE,
                 kind="programmatic", fn=fn)
        )

    # ---- batch2-T2: LLM hooks 注册 ----
    def register_pre_llm_call(self, fn, *, name=None):
        """注册 PRE_LLM_CALL hook。

        fn 签名: (messages: list, tools: Optional[list]) -> Optional[tuple[list, Optional[list]]]
        返回 (messages, tools) 元组以修改；返回 None 表示不修改。
        """
        self._hooks[HookEvent.PRE_LLM_CALL].append(
            Hook(name=name or "anonymous", event=HookEvent.PRE_LLM_CALL,
                 kind="programmatic", fn=fn)
        )

    def register_post_llm_call(self, fn, *, name=None):
        """注册 POST_LLM_CALL hook。

        fn 签名: (response) -> Optional[response]
        返回新 response 以修改；返回 None 表示不修改。
        """
        self._hooks[HookEvent.POST_LLM_CALL].append(
            Hook(name=name or "anonymous", event=HookEvent.POST_LLM_CALL,
                 kind="programmatic", fn=fn)
        )

    def register_declarative(self, hook: Hook):
        """注册一个声明式 hook（已构造好的 Hook 对象）。"""
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
                    # declarative hook
                    new_prompt = self._invoke_declarative_user_prompt(hook, prompt, session_id)
                if new_prompt is not None:
                    prompt = new_prompt
            except Exception as e:
                logger.warning("hook %s 异常（视为 None）: %s", hook.name, e)
        return prompt

    def _invoke_declarative_user_prompt(self, hook, prompt, session_id):
        """跑子进程，按 IPC 协议解析。返回新 prompt 或 None。"""
        from agent.hook_exec import dispatch_hook  # 懒加载避免循环
        payload = {
            "event": "user_prompt_submit",
            "session_id": session_id,
            "timestamp": _now_iso(),
            "hook_name": hook.name,
            "prompt": prompt,
        }
        result = dispatch_hook(hook, payload)
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
                    # declarative hook
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
        from agent.hook_exec import dispatch_hook
        payload = {
            "event": "pre_tool_use",
            "session_id": session_id,
            "timestamp": _now_iso(),
            "hook_name": hook.name,
            "tool_name": tool_name,
            "args": args,
        }
        result = dispatch_hook(hook, payload)
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
        from agent.hook_exec import dispatch_hook
        payload = {
            "event": "post_tool_use",
            "session_id": session_id,
            "timestamp": _now_iso(),
            "hook_name": hook.name,
            "tool_name": tool_name,
            "args": args,
            "result": result,
        }
        proc_result = dispatch_hook(hook, payload)
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
        from agent.hook_exec import dispatch_hook
        payload = {
            "event": "stop",
            "session_id": session_id,
            "timestamp": _now_iso(),
            "hook_name": hook.name,
        }
        result = dispatch_hook(hook, payload)
        if result is None:
            return None
        return result.get("continue")

    def _invoke_declarative_script(self, hook, session_id: str, event: str,
                                   **extra) -> Optional[dict]:
        """跑声明式 hook 子进程（统一入口）。返回 dispatch_hook 的 dict 或 None。

        声明式 hook 的 payload 统一含 event/session_id/timestamp/hook_name，
        事件特定字段通过 extra 传入（不传超大内容，只传元信息）。
        dispatch_hook 按 hook.script.handler_type 分发到对应执行器
        （command/http/mcp_tool/prompt/agent）。
        """
        from agent.hook_exec import dispatch_hook
        payload = {
            "event": event,
            "session_id": session_id,
            "timestamp": _now_iso(),
            "hook_name": hook.name,
        }
        payload.update(extra)
        return dispatch_hook(hook, payload)

    # ---- batch2-T2: 执行 PRE_LLM_CALL / POST_LLM_CALL ----
    def run_pre_llm_call(self, messages: list, tools: Optional[list],
                         *, session_id: str = "") -> tuple:
        """链式：每个 hook 可修改 messages 和 tools。

        fn 签名: (messages, tools) -> Optional[(messages, tools)]
        返回 None 时不修改（保持上一轮输出）。
        声明式 hook：跑子进程（通知型），不修改 messages/tools。
        失败 fail-open（视为 None）。
        """
        for hook in self._hooks[HookEvent.PRE_LLM_CALL]:
            try:
                if hook.kind == "programmatic":
                    result = hook.fn(messages, tools)
                else:
                    # 声明式：通知型，只传消息数量等元信息（避免超大 payload）
                    self._invoke_declarative_script(
                        hook, session_id, "pre_llm_call",
                        message_count=len(messages),
                    )
                    result = None
                if result is not None:
                    # 解包元组
                    if isinstance(result, tuple) and len(result) == 2:
                        messages, tools = result
                    else:
                        logger.warning(
                            "PRE_LLM_CALL hook %s 返回非 (messages, tools) 元组，忽略",
                            hook.name,
                        )
            except Exception as e:
                logger.warning("PRE_LLM_CALL hook %s 异常（视为 None）: %s",
                               hook.name, e)
        return messages, tools

    def run_post_llm_call(self, response, *, session_id: str = ""):
        """链式：每个 hook 可修改 response。

        fn 签名: (response) -> Optional[response]
        返回 None 时不修改。
        声明式 hook：跑子进程（通知型），不修改 response。
        失败 fail-open。
        """
        for hook in self._hooks[HookEvent.POST_LLM_CALL]:
            try:
                if hook.kind == "programmatic":
                    result = hook.fn(response)
                else:
                    # 声明式：通知型
                    self._invoke_declarative_script(hook, session_id, "post_llm_call")
                    result = None
                if result is not None:
                    response = result
            except Exception as e:
                logger.warning("POST_LLM_CALL hook %s 异常（视为 None）: %s",
                               hook.name, e)
        return response

    # ---- P2-13 NEW: 会话/压缩/配置事件的执行 ----
    def run_session_start(self, payload: dict) -> None:
        """通知型：所有 SESSION_START hook 都被调，返回值忽略。失败 fail-open。"""
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.SESSION_START]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "session_start",
                        session_title=payload.get("title", ""),
                    )
            except Exception as e:
                logger.warning("SESSION_START hook %s 异常（忽略）: %s",
                               hook.name, e)

    def run_session_end(self, payload: dict) -> None:
        """通知型：所有 SESSION_END hook 都被调。失败 fail-open。"""
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.SESSION_END]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "session_end",
                        message_count=payload.get("message_count", 0),
                    )
            except Exception as e:
                logger.warning("SESSION_END hook %s 异常（忽略）: %s",
                               hook.name, e)

    def run_pre_compact(self, payload: dict) -> dict:
        """可短路：任一 hook 返回 {"abort": True} 则停止后续 + 通知调用方。

        返回 {"abort": bool}。abort=True 时调用方应跳过该层压缩。
        """
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.PRE_COMPACT]:
            try:
                if hook.kind == "programmatic":
                    result = hook.fn(payload)
                else:
                    result = self._invoke_declarative_script(
                        hook, session_id, "pre_compact",
                        layer=payload.get("layer", ""),
                    )
                if result and result.get("abort"):
                    logger.info(
                        "PRE_COMPACT hook %s 请求 abort（layer=%s）",
                        hook.name, payload.get("layer"),
                    )
                    return {"abort": True, "blocked_by": hook.name}
            except Exception as e:
                logger.warning("PRE_COMPACT hook %s 异常（视为 None）: %s",
                               hook.name, e)
        return {"abort": False}

    def run_post_compact(self, payload: dict) -> None:
        """通知型：压缩完成后通知所有 hook（metrics 收集、日志等）。"""
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.POST_COMPACT]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "post_compact",
                        layer=payload.get("layer", ""),
                    )
            except Exception as e:
                logger.warning("POST_COMPACT hook %s 异常（忽略）: %s",
                               hook.name, e)

    def run_config_change(self, payload: dict) -> None:
        """通知型：配置变更后通知所有 hook（审计、缓存失效等）。"""
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.CONFIG_CHANGE]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "config_change",
                        changed_keys=payload.get("changed_keys", []),
                    )
            except Exception as e:
                logger.warning("CONFIG_CHANGE hook %s 异常（忽略）: %s",
                               hook.name, e)
