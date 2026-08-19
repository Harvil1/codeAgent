"""Hooks 系统：扩展 agent 主循环行为的注册表机制。

27 种 event（核心 6 + P2-13 扩展 5 + round3 新增 7 + P3.3-P3.4 新增 3 + Task N 新增 6）：
  核心 6 种：USER_PROMPT_SUBMIT / PRE_TOOL_USE / POST_TOOL_USE / STOP
           + PRE_LLM_CALL / POST_LLM_CALL（batch2-T2）
  新增 5 种（P2-13）：SESSION_START / SESSION_END
           + PRE_COMPACT / POST_COMPACT + CONFIG_CHANGE
  round3 新增 7 种：POST_TOOL_USE_FAILURE / SUBAGENT_START / SUBAGENT_STOP
           + TASK_CREATED / TASK_COMPLETED + PERMISSION_REQUEST + PERMISSION_DENIED
  P3.3-P3.4 新增 3 种：STOP_FAILURE + WORKTREE_CREATE + WORKTREE_REMOVE
  Task N 新增 6 种：FILE_CHANGED / CWD_CHANGED / INSTRUCTIONS_LOADED
           + SETUP / TEAMMATE_IDLE / ELICITATION_STARTED
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
    # === Task N 新增 6 种（借鉴 Claude Code）===
    FILE_CHANGED = "file_changed"                   # 文件写后触发（IDE 集成基础）
    CWD_CHANGED = "cwd_changed"                     # worktree 切换（workspace_context 配合）
    INSTRUCTIONS_LOADED = "instructions_loaded"     # CLAUDE.md/OMNIMATE.md 加载完
    SETUP = "setup"                                 # 启动时一次（cli.py initialize）
    TEAMMATE_IDLE = "teammate_idle"                 # team 成员进 idle
    ELICITATION_STARTED = "elicitation_started"     # ask_user 弹窗前


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
    # C4（CCB 借鉴）：async hook（仅 command 类型）
    # - async_run: 后台线程跑，dispatch 立即返回 None 不阻塞主流程
    #   （⚠ gating 语义失效：async 的 PRE_TOOL_USE deny 来不及拦——
    #   async 只该用于通知/审计型 hook）
    # - async_rewake: 后台跑完 exit 2（block）时推 rewake 通知，
    #   agent 下一轮 drain 为 ephemeral <rewake_notification> 让模型跟进
    # - status_message: 展示文案（rewake 通知附带；日志/UI 用）
    async_run: bool = False
    async_rewake: bool = False
    status_message: Optional[str] = None


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
        """R30g-H5：并行执行 + deny>ask>allow 聚合。返回 (deny_reason, modified_args)。

        - 所有匹配 hook 先跑完再聚合（声明式子进程 hook 用线程池并行——
          一个慢 hook 不拖累其他 hook 与主循环；programmatic 是进程内快函数串行）
        - 聚合优先级：deny > ask > allow/None；modify_args 按注册顺序叠加
          （注意：并行下每个 hook 的判决基于**原始参数**计算——多 hook 同时
          改参的链式语义与旧串行版略有差异，罕见场景）
        - ask 档（对齐 CCB ask 语义）：本项目 pre_tool_use 无通用审批 UI，
          ask 兑现为 fail-closed 拒绝，错误信息注明是 ask（用户可调整 hook）
        - fail_closed hook 异常聚合为 deny（不再吞掉——R30c-A6 语义保留）

        P3.5: 声明式 hook 支持 if 条件过滤（permission rule 语法）。
        P3.6: 声明式 hook 支持 once（跑一次后消费）。
        """
        matched = []
        for hook in self._hooks[HookEvent.PRE_TOOL_USE]:
            if hook.kind == "declarative":
                if self._is_consumed(hook):
                    continue
                if not self._matches_if_condition(hook, tool_name, args):
                    continue
            matched.append(hook)
        if not matched:
            return None, None

        def _run_one(hook, hook_args):
            """单 hook 执行 → (hook, result 或 Exception)。"""
            try:
                if hook.kind == "programmatic":
                    return hook, hook.fn(tool_name, hook_args)
                result = self._invoke_declarative_pre_tool(hook, tool_name, hook_args, session_id)
                self._mark_consumed_if_once(hook)
                return hook, result
            except Exception as e:  # noqa: BLE001
                return hook, e

        # programmatic：进程内快函数——**串行链式**（后一个看到前一个的
        # modify_args，保持旧语义）；declarative：子进程慢——并行执行
        # （判决基于原始参数；modify_args 为罕见场景，按注册顺序替换叠加）
        prog = [h for h in matched if h.kind == "programmatic"]
        decl = [h for h in matched if h.kind != "programmatic"]
        outcomes = []
        current_args = args
        for h in prog:
            outcome = _run_one(h, current_args)
            outcomes.append(outcome)
            # 链式语义（同旧串行版）：后一个 hook 看到前一个的 modify_args
            if (not isinstance(outcome[1], Exception)
                    and isinstance(outcome[1], dict)
                    and "modify_args" in outcome[1]):
                current_args = outcome[1]["modify_args"]
        if len(decl) == 1:
            outcomes.append(_run_one(decl[0], args))
        elif decl:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(
                max_workers=min(4, len(decl)),
                thread_name_prefix="pre-tool-hook",
            ) as ex:
                outcomes.extend(ex.map(lambda h: _run_one(h, args), decl))

        denies = []   # [(hook_name, reason)]
        asks = []     # [(hook_name, reason)]
        modified_args = None
        for hook, result in outcomes:
            if isinstance(result, Exception):
                if hook.fail_closed:
                    denies.append((hook.name, f"hook 异常（fail_closed）: {result}"))
                else:
                    logger.warning("hook %s 异常（视为 None）: %s", hook.name, result)
                continue
            if result is None:
                continue
            if "deny" in result:
                denies.append((hook.name, result["deny"]))
            if "ask" in result:
                asks.append((hook.name, result.get("ask") or "unspecified"))
            if "modify_args" in result:
                mod = result["modify_args"]
                if not isinstance(mod, dict):
                    logger.warning("hook %s modify_args 非 dict，忽略", hook.name)
                    continue
                # R30 审计 L14：按注册顺序叠加合并（outcomes 本身按注册序）——
                # 旧实现后到者整体替换，先到 hook 的修改静默丢失。合并语义：
                # 首个 hook 的完整返回为基底，后续按键覆盖（同键后到胜、异键并集）
                if modified_args is None:
                    modified_args = dict(mod)
                else:
                    modified_args.update(mod)

        if denies:
            name, reason = denies[0]
            if len(denies) > 1:
                logger.info("pre_tool_use 聚合：%d 个 hook deny，取 %s", len(denies), name)
            return reason, None
        if asks:
            name, reason = asks[0]
            return (
                f"hook {name} 要求人工确认（ask）: {reason}"
                "（本环境未接入 hook 审批 UI，默认拒绝；如需放行请调整该 hook 的规则）"
            ), None
        return None, modified_args

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
        # R30b-A6：propagate_error=True——fail_closed hook 执行失败时异常
        # 向上抛，让 run_pre_tool_use 的 except 分支转为 deny（此前
        # dispatch_hook 内部吞掉一切异常，fail_closed 分支永不可达）
        result = dispatch_hook(hook, payload, propagate_error=True)
        if result is None:
            return None
        action = result.get("action", "allow")
        # P3.7: block（exit 2）转 deny
        if action in ("deny", "block"):
            return {"deny": result.get("reason", "unspecified")}
        # R30g-H5: ask 档（升审批语义——run_pre_tool_use 聚合层兑现）
        if action == "ask":
            return {"ask": result.get("reason", "unspecified")}
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
                                   timeout_cap: float = None, **extra) -> Optional[dict]:
        """跑声明式 hook 子进程（统一入口）。返回 dispatch_hook 的 dict 或 None。

        声明式 hook 的 payload 统一含 event/session_id/timestamp/hook_name，
        事件特定字段通过 extra 传入（不传超大内容，只传元信息）。
        dispatch_hook 按 hook.script.handler_type 分发到对应执行器
        （command/http/mcp_tool/prompt/agent）。
        R30g-M8：timeout_cap 非空时钳制 hook 超时（取 min，SessionEnd 用）。
        """
        from agent.hook_exec import dispatch_hook
        payload = {
            "event": event,
            "session_id": session_id,
            "timestamp": _now_iso(),
            "hook_name": hook.name,
        }
        payload.update(extra)
        return dispatch_hook(hook, payload, timeout_cap=timeout_cap)

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

    # R30g-M8：SessionEnd hook 独立短超时（对齐 CCB 默认 1500ms）——
    # teardown 不能被慢 hook 卡死（退出等 10s 体验极差）
    SESSION_END_TIMEOUT_CAP = 1.5

    def run_session_end(self, payload: dict) -> None:
        """通知型：所有 SESSION_END hook 都被调。失败 fail-open。

        R30g-M8：声明式 hook 超时钳制到 SESSION_END_TIMEOUT_CAP
        （hook 自配更短则尊重更短值）。
        """
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.SESSION_END]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "session_end",
                        message_count=payload.get("message_count", 0),
                        timeout_cap=self.SESSION_END_TIMEOUT_CAP,
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

    # ---- Task N NEW: 6 个新事件 ----

    def register_file_changed(self, fn, *, name=None):
        """注册 FILE_CHANGED hook（通知型）。

        触发时机：write_file / str_replace 等文件写入成功后（IDE 集成基础）。
        payload: {session_id, path, op}。op ∈ {"write", "append", "edit"}。
        """
        self._hooks[HookEvent.FILE_CHANGED].append(
            Hook(name=name or "anonymous", event=HookEvent.FILE_CHANGED,
                 kind="programmatic", fn=fn))

    def run_file_changed(self, payload: dict) -> None:
        """通知型：文件写入成功后通知所有 hook（IDE 同步、审计、热重载等）。fail-open。"""
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.FILE_CHANGED]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "file_changed",
                        path=payload.get("path"),
                        op=payload.get("op"),
                    )
            except Exception as e:
                logger.warning("FILE_CHANGED hook %s 异常: %s", hook.name, e)

    def register_cwd_changed(self, fn, *, name=None):
        """注册 CWD_CHANGED hook（通知型）。

        触发时机：workspace_context 切换时（子代理 worktree 进入/退出）。
        payload: {session_id, old, new}。
        """
        self._hooks[HookEvent.CWD_CHANGED].append(
            Hook(name=name or "anonymous", event=HookEvent.CWD_CHANGED,
                 kind="programmatic", fn=fn))

    def run_cwd_changed(self, payload: dict) -> None:
        """通知型：workspace cwd 切换后通知所有 hook。fail-open。"""
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.CWD_CHANGED]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "cwd_changed",
                        old=payload.get("old"),
                        new=payload.get("new"),
                    )
            except Exception as e:
                logger.warning("CWD_CHANGED hook %s 异常: %s", hook.name, e)

    def register_instructions_loaded(self, fn, *, name=None):
        """注册 INSTRUCTIONS_LOADED hook（通知型）。

        触发时机：prompt_builder 加载完项目 CLAUDE.md/OMNIMATE.md 后。
        payload: {session_id, source, bytes}。
        """
        self._hooks[HookEvent.INSTRUCTIONS_LOADED].append(
            Hook(name=name or "anonymous", event=HookEvent.INSTRUCTIONS_LOADED,
                 kind="programmatic", fn=fn))

    def run_instructions_loaded(self, payload: dict) -> None:
        """通知型：项目记忆加载完通知所有 hook。fail-open。"""
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.INSTRUCTIONS_LOADED]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "instructions_loaded",
                        source=payload.get("source"),
                    )
            except Exception as e:
                logger.warning("INSTRUCTIONS_LOADED hook %s 异常: %s", hook.name, e)

    def register_setup(self, fn, *, name=None):
        """注册 SETUP hook（通知型）。

        触发时机：cli.py RuntimeContext.initialize 末尾（启动时一次）。
        payload: {session_id, agent_home, started_at}。
        """
        self._hooks[HookEvent.SETUP].append(
            Hook(name=name or "anonymous", event=HookEvent.SETUP,
                 kind="programmatic", fn=fn))

    def run_setup(self, payload: dict) -> None:
        """通知型：agent 启动时触发一次。fail-open。"""
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.SETUP]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "setup",
                        agent_home=payload.get("agent_home"),
                    )
            except Exception as e:
                logger.warning("SETUP hook %s 异常: %s", hook.name, e)

    def register_teammate_idle(self, fn, *, name=None):
        """注册 TEAMMATE_IDLE hook（通知型）。

        触发时机：team 成员进 idle 状态（team 协作时）。
        payload: {session_id, member}。
        """
        self._hooks[HookEvent.TEAMMATE_IDLE].append(
            Hook(name=name or "anonymous", event=HookEvent.TEAMMATE_IDLE,
                 kind="programmatic", fn=fn))

    def run_teammate_idle(self, payload: dict) -> None:
        """通知型：team 成员进 idle 时通知所有 hook。fail-open。"""
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.TEAMMATE_IDLE]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "teammate_idle",
                        member=payload.get("member"),
                    )
            except Exception as e:
                logger.warning("TEAMMATE_IDLE hook %s 异常: %s", hook.name, e)

    def register_elicitation_started(self, fn, *, name=None):
        """注册 ELICITATION_STARTED hook（通知型）。

        触发时机：ask_user 工具弹窗前（UI 集成用）。
        payload: {session_id, prompt}。
        """
        self._hooks[HookEvent.ELICITATION_STARTED].append(
            Hook(name=name or "anonymous", event=HookEvent.ELICITATION_STARTED,
                 kind="programmatic", fn=fn))

    def run_elicitation_started(self, payload: dict) -> None:
        """通知型：ask_user 弹窗前通知所有 hook。fail-open。"""
        session_id = payload.get("session_id", "") or ""
        for hook in self._hooks[HookEvent.ELICITATION_STARTED]:
            try:
                if hook.kind == "programmatic":
                    hook.fn(payload)
                else:
                    self._invoke_declarative_script(
                        hook, session_id, "elicitation_started",
                        prompt=payload.get("prompt"),
                    )
            except Exception as e:
                logger.warning("ELICITATION_STARTED hook %s 异常: %s", hook.name, e)
