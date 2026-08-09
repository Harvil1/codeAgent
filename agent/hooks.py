"""Hooks 系统：扩展 agent 主循环行为的注册表机制。

21 种 event（核心 6 + P2-13 扩展 5 + round3 新增 7 + P3.3-P3.4 新增 3）：
  核心 6 种：USER_PROMPT_SUBMIT / PRE_TOOL_USE / POST_TOOL_USE / STOP
           + PRE_LLM_CALL / POST_LLM_CALL（batch2-T2）
  新增 5 种（P2-13）：SESSION_START / SESSION_END
           + PRE_COMPACT / POST_COMPACT + CONFIG_CHANGE
  round3 新增 7 种：POST_TOOL_USE_FAILURE / SUBAGENT_START / SUBAGENT_STOP
           + TASK_CREATED / TASK_COMPLETED + PERMISSION_REQUEST + PERMISSION_DENIED
  P3.3-P3.4 新增 3 种：STOP_FAILURE + WORKTREE_CREATE + WORKTREE_REMOVE
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
    # round3 NEW: 关键生命周期/审计事件
    POST_TOOL_USE_FAILURE = "post_tool_use_failure"
    SUBAGENT_START = "subagent_start"
    SUBAGENT_STOP = "subagent_stop"
    TASK_CREATED = "task_created"
    TASK_COMPLETED = "task_completed"
    PERMISSION_REQUEST = "permission_request"
    PERMISSION_DENIED = "permission_denied"
    # P3.3 NEW: STOP 失败变体（主循环异常退出时触发）
    STOP_FAILURE = "stop_failure"
    # P3.4 NEW: worktree 生命周期事件（隔离工作区创建/清理通知）
    WORKTREE_CREATE = "worktree_create"
    WORKTREE_REMOVE = "worktree_remove"


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

    P3.5 新增字段：
    - if_condition: 声明式条件过滤（permission rule 语法），
      仅适用 PRE_TOOL_USE / POST_TOOL_USE / POST_TOOL_USE_FAILURE / PERMISSION_REQUEST。
      不匹配时跳过该 hook（省资源）。格式："ToolName(arg_pattern)"，如 "terminal(git *)"。
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
    # P3.5 NEW: 条件过滤（permission rule 语法，None/空 = 无条件匹配）
    if_condition: Optional[str] = None


@dataclass
class Hook:
    """统一包装：程序式或声明式。"""
    name: str
    event: HookEvent
    kind: str  # "programmatic" | "declarative"
    fn: Optional[Callable] = None
    script: Optional[HookScriptConfig] = None
    fail_closed: bool = False
    # P3.6 NEW: once=True 的 hook 跑一次后被消费（从 registry 移除/跳过）
    once: bool = False
    # P3.8 NEW: 声明式 command hook 是否套 sandbox（仅 Unix 生效，Windows fail-open）
    use_sandbox: bool = False


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class HookRegistry:
    """管理所有 hook 注册和执行。实例由 RuntimeContext 持有，注入 AIAgent。"""

    def __init__(self):
        self._hooks: dict = {e: [] for e in HookEvent}
        self._stop_fire_count: int = 0
        # P3.6 NEW: once=True 的 hook 已被消费的 id 集合（按 Hook 对象 id）
        self._consumed: set = set()

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
        self._consumed.clear()  # P3.6: 同步清消费记录

    # ---- P3.5/P3.6 helpers ----

    def _is_consumed(self, hook: "Hook") -> bool:
        """P3.6: once=True 的 hook 是否已被消费。"""
        return hook.once and id(hook) in self._consumed

    def _mark_consumed_if_once(self, hook: "Hook") -> None:
        """P3.6: 声明式 hook 跑完后如果是 once，标记为已消费。"""
        if hook.once:
            self._consumed.add(id(hook))

    def _matches_if_condition(self, hook: "Hook", tool_name: str, args: dict) -> bool:
        """P3.5: 声明式 hook 的 if 条件过滤。

        - hook.script 无 if_condition → True（无条件匹配）
        - 有 if_condition → 调 agent.hook_filter.match_if_condition
        - 程序式 hook 不走这里（调用方应只在 declarative 时调）
        """
        if hook.script is None:
            return True
        cond = getattr(hook.script, "if_condition", None)
        if not cond:
            return True
        from agent.hook_filter import match_if_condition
        return match_if_condition(tool_name, args, cond)

    # ---- 执行：USER_PROMPT_SUBMIT ----
    def run_user_prompt_submit(self, prompt: str, *, session_id: str) -> str:
        """链式：每个 hook 看到前一个的输出。失败 fail-open。

        P3.6: 声明式 hook 支持 once（跑一次后消费）。
        """
        for hook in self._hooks[HookEvent.USER_PROMPT_SUBMIT]:
            if hook.kind == "declarative":
                if self._is_consumed(hook):
                    continue
            try:
                if hook.kind == "programmatic":
                    new_prompt = hook.fn(prompt)
                else:
                    # declarative hook
                    new_prompt = self._invoke_declarative_user_prompt(hook, prompt, session_id)
                    self._mark_consumed_if_once(hook)
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

        P3.5: 声明式 hook 支持 if 条件过滤（permission rule 语法）。
        P3.6: 声明式 hook 支持 once（跑一次后消费）。
        """
        deny_reason = None
        modified_args = None
        current_args = args
        for hook in self._hooks[HookEvent.PRE_TOOL_USE]:
            # P3.5/P3.6: 声明式 hook 的 if 条件 + once 消费检查
            if hook.kind == "declarative":
                if self._is_consumed(hook):
                    continue
                if not self._matches_if_condition(hook, tool_name, current_args):
                    continue
            try:
                if hook.kind == "programmatic":
                    result = hook.fn(tool_name, current_args)
                else:
                    # declarative hook
                    result = self._invoke_declarative_pre_tool(hook, tool_name, current_args, session_id)
                    self._mark_consumed_if_once(hook)
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
        """跑子进程，按 IPC 协议解析。返回 {deny: ...}/{modify_args: ...}/None。

        P3.7: 处理 exit code 2 blocking 协议。
            dispatch_hook 返回 {"action": "block", "reason": stderr} → 转 {"deny": reason}
        """
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
        # P3.7: block（exit 2）转 deny
        if action in ("deny", "block"):
            return {"deny": result.get("reason", "unspecified")}
        if action == "modify":
            return {"modify_args": result.get("args", args)}
        return None

    # ---- 执行：POST_TOOL_USE ----
    def run_post_tool_use(self, tool_name: str, args: dict, result: str,
                          *, session_id: str) -> str:
        """链式：每个 hook 看到前一个的输出。

        P3.5/P3.6: 声明式 hook 支持 if 条件过滤 + once 消费。
        """
        for hook in self._hooks[HookEvent.POST_TOOL_USE]:
            if hook.kind == "declarative":
                if self._is_consumed(hook):
                    continue
                if not self._matches_if_condition(hook, tool_name, args):
                    continue
            try:
                if hook.kind == "programmatic":
                    new_result = hook.fn(tool_name, args, result)
                else:
                    new_result = self._invoke_declarative_post_tool(
                        hook, tool_name, args, result, session_id)
                    self._mark_consumed_if_once(hook)
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
        """首个非 None 胜出。超过 max_fires 强制返回 None（防失控）。

        P3.6: 声明式 hook 支持 once（跑一次后消费）。
        """
        if self._stop_fire_count >= max_fires:
            logger.info("STOP hook 触发上限（%d/%d），本次跳过",
                        self._stop_fire_count, max_fires)
            return None
        for hook in self._hooks[HookEvent.STOP]:
            if hook.kind == "declarative":
                if self._is_consumed(hook):
                    continue
            try:
                if hook.kind == "programmatic":
                    msg = hook.fn()
                else:
                    msg = self._invoke_declarative_stop(hook, session_id)
                    self._mark_consumed_if_once(hook)
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

    # ---- round3 NEW: 7 个关键事件 ----

    def register_post_tool_use_failure(self, fn, *, name=None):
        self._hooks[HookEvent.POST_TOOL_USE_FAILURE].append(
            Hook(name=name or "anonymous", event=HookEvent.POST_TOOL_USE_FAILURE,
                 kind="programmatic", fn=fn))

    def run_post_tool_use_failure(self, payload: dict) -> None:
        """通知型：工具调用失败（result 含 error）时触发。fail-open。"""
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.POST_TOOL_USE_FAILURE]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "post_tool_use_failure",
                        tool=payload.get("tool"), error=payload.get("error"),
                        error_type=payload.get("error_type"),
                    )
            except Exception as e:
                logger.warning("POST_TOOL_USE_FAILURE hook %s 异常: %s", hook.name, e)

    def register_subagent_start(self, fn, *, name=None):
        self._hooks[HookEvent.SUBAGENT_START].append(
            Hook(name=name or "anonymous", event=HookEvent.SUBAGENT_START,
                 kind="programmatic", fn=fn))

    def run_subagent_start(self, payload: dict) -> None:
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.SUBAGENT_START]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "subagent_start",
                        subagent=payload.get("subagent"), goal=payload.get("goal"),
                        spawn_depth=payload.get("spawn_depth"),
                    )
            except Exception as e:
                logger.warning("SUBAGENT_START hook %s 异常: %s", hook.name, e)

    def register_subagent_stop(self, fn, *, name=None):
        self._hooks[HookEvent.SUBAGENT_STOP].append(
            Hook(name=name or "anonymous", event=HookEvent.SUBAGENT_STOP,
                 kind="programmatic", fn=fn))

    def run_subagent_stop(self, payload: dict) -> None:
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.SUBAGENT_STOP]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "subagent_stop",
                        subagent=payload.get("subagent"), goal=payload.get("goal"),
                        success=payload.get("success"),
                    )
            except Exception as e:
                logger.warning("SUBAGENT_STOP hook %s 异常: %s", hook.name, e)

    def register_task_created(self, fn, *, name=None):
        self._hooks[HookEvent.TASK_CREATED].append(
            Hook(name=name or "anonymous", event=HookEvent.TASK_CREATED,
                 kind="programmatic", fn=fn))

    def run_task_created(self, payload: dict) -> None:
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.TASK_CREATED]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "task_created",
                        task_id=payload.get("task_id"), subject=payload.get("subject"),
                        owner=payload.get("owner"),
                    )
            except Exception as e:
                logger.warning("TASK_CREATED hook %s 异常: %s", hook.name, e)

    def register_task_completed(self, fn, *, name=None):
        self._hooks[HookEvent.TASK_COMPLETED].append(
            Hook(name=name or "anonymous", event=HookEvent.TASK_COMPLETED,
                 kind="programmatic", fn=fn))

    def run_task_completed(self, payload: dict) -> None:
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.TASK_COMPLETED]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "task_completed",
                        task_id=payload.get("task_id"),
                        unblocked=payload.get("unblocked"),
                    )
            except Exception as e:
                logger.warning("TASK_COMPLETED hook %s 异常: %s", hook.name, e)

    def register_permission_request(self, fn, *, name=None):
        self._hooks[HookEvent.PERMISSION_REQUEST].append(
            Hook(name=name or "anonymous", event=HookEvent.PERMISSION_REQUEST,
                 kind="programmatic", fn=fn))

    def run_permission_request(self, payload: dict) -> None:
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.PERMISSION_REQUEST]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "permission_request",
                        command=payload.get("command"), reason=payload.get("reason"),
                    )
            except Exception as e:
                logger.warning("PERMISSION_REQUEST hook %s 异常: %s", hook.name, e)

    def register_permission_denied(self, fn, *, name=None):
        self._hooks[HookEvent.PERMISSION_DENIED].append(
            Hook(name=name or "anonymous", event=HookEvent.PERMISSION_DENIED,
                 kind="programmatic", fn=fn))

    def run_permission_denied(self, payload: dict) -> None:
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.PERMISSION_DENIED]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "permission_denied",
                        command=payload.get("command"), reason=payload.get("reason"),
                        deny_type=payload.get("deny_type"),
                    )
            except Exception as e:
                logger.warning("PERMISSION_DENIED hook %s 异常: %s", hook.name, e)

    # ---- P3.3 NEW: STOP_FAILURE 事件 ----

    def register_stop_failure(self, fn, *, name=None):
        """注册 STOP_FAILURE hook（通知型，返回值忽略）。

        触发时机：agent 主循环 LLM 调用失败/异常退出时（与正常 STOP 区分）。
        payload: {session_id, error, error_type, timestamp}。
        """
        self._hooks[HookEvent.STOP_FAILURE].append(
            Hook(name=name or "anonymous", event=HookEvent.STOP_FAILURE,
                 kind="programmatic", fn=fn))

    def run_stop_failure(self, payload: dict) -> None:
        """通知型：所有 STOP_FAILURE hook 都被调。fail-open。"""
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.STOP_FAILURE]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "stop_failure",
                        error=payload.get("error"),
                        error_type=payload.get("error_type"),
                    )
            except Exception as e:
                logger.warning("STOP_FAILURE hook %s 异常: %s", hook.name, e)

    # ---- P3.4 NEW: WORKTREE_CREATE / WORKTREE_REMOVE 事件 ----

    def register_worktree_create(self, fn, *, name=None):
        """注册 WORKTREE_CREATE hook（通知型）。

        触发时机：tools/worktree.py:create_isolated_workspace 创建 worktree 成功后。
        payload: {session_id, path, branch, workspace_type}。
        """
        self._hooks[HookEvent.WORKTREE_CREATE].append(
            Hook(name=name or "anonymous", event=HookEvent.WORKTREE_CREATE,
                 kind="programmatic", fn=fn))

    def run_worktree_create(self, payload: dict) -> None:
        """通知型：worktree 创建后通知所有 hook（审计、清理注册等）。fail-open。"""
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.WORKTREE_CREATE]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "worktree_create",
                        path=payload.get("path"),
                        branch=payload.get("branch"),
                    )
            except Exception as e:
                logger.warning("WORKTREE_CREATE hook %s 异常: %s", hook.name, e)

    def register_worktree_remove(self, fn, *, name=None):
        """注册 WORKTREE_REMOVE hook（通知型）。

        触发时机：tools/worktree.py 的 cleanup 函数执行后。
        payload: {session_id, path, branch}。
        """
        self._hooks[HookEvent.WORKTREE_REMOVE].append(
            Hook(name=name or "anonymous", event=HookEvent.WORKTREE_REMOVE,
                 kind="programmatic", fn=fn))

    def run_worktree_remove(self, payload: dict) -> None:
        """通知型：worktree 清理后通知所有 hook。fail-open。"""
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.WORKTREE_REMOVE]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "worktree_remove",
                        path=payload.get("path"),
                    )
            except Exception as e:
                logger.warning("WORKTREE_REMOVE hook %s 异常: %s", hook.name, e)
