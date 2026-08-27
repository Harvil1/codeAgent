"""Hooks 系统——给 agent 主循环装"外挂插件"的注册中心。

hook（钩子）是什么：在对话流程的固定节点上自动触发的外挂小程序，
像电梯里的楼层按钮，到层就响。比如"每次用户提交问题前，先跑一遍敏感词过滤"。

本文件在项目里的位置：agent 核心的横向扩展层。cli.py 组装 RuntimeContext 时
创建 HookRegistry 实例注入 AIAgent；真正的执行细节（子进程/HTTP/沙箱）在
agent/hook_exec.py，配置文件解析在 agent/hook_loader.py。本文件只管
"有哪些 hook、什么时候触发、结果怎么合并"。

一共 27 种事件：
  核心 6 种：用户提交问题（USER_PROMPT_SUBMIT）、工具调用前
    （PRE_TOOL_USE）、工具调用后（POST_TOOL_USE）、回答结束（STOP）、
    LLM 调用前后（PRE_LLM_CALL / POST_LLM_CALL）
  会话/压缩/配置 5 种：会话开始/结束（SESSION_START / SESSION_END）、
    上下文压缩前/后（PRE_COMPACT / POST_COMPACT）、配置变更（CONFIG_CHANGE）
  关键生命周期/留痕 7 种：工具调用失败（POST_TOOL_USE_FAILURE）、子代理
    开始/结束（SUBAGENT_START / SUBAGENT_STOP）、任务创建/完成
    （TASK_CREATED / TASK_COMPLETED）、权限请求/拒绝
    （PERMISSION_REQUEST / PERMISSION_DENIED）
  回答异常结束（STOP_FAILURE）、worktree 隔离工作区
    创建/清理（WORKTREE_CREATE / WORKTREE_REMOVE）3 种
  集成/通知类 6 种：文件被改（FILE_CHANGED）、工作目录切换
    （CWD_CHANGED）、项目说明文件加载完（INSTRUCTIONS_LOADED）、
    启动一次性事件（SETUP）、队友空闲（TEAMMATE_IDLE）、
    弹窗问用户前（ELICITATION_STARTED）

hook 有两种注册方式：programmatic（直接给一个 Python 函数）和
declarative（写配置，由子进程脚本/HTTP/MCP 工具/LLM 等执行）。

两条核心语义（tests/test_hooks.py 有用例对着考）：
1. hook 执行不拖累主循环——声明式子进程 hook 用线程池并行跑，
   一个慢 hook 不会卡住其他 hook 和主循环。
2. 多个 hook 都要改工具参数时，按注册顺序叠加（前一个的修改是后一个的
   底，后一个按 key 覆盖），而不是后来者整体替换。

失败处理默认 fail-open（出异常就记条日志、当作这个 hook 不存在）；
只有 PreToolUse 可以配成 fail_closed（出错当作拒绝执行）。
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
    # LLM 调用前后各触发一次
    PRE_LLM_CALL = "pre_llm_call"
    POST_LLM_CALL = "post_llm_call"
    # 会话/压缩/配置相关 5 个事件
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PRE_COMPACT = "pre_compact"
    POST_COMPACT = "post_compact"
    CONFIG_CHANGE = "config_change"
    # 关键生命周期/留痕审计事件
    POST_TOOL_USE_FAILURE = "post_tool_use_failure"
    SUBAGENT_START = "subagent_start"
    SUBAGENT_STOP = "subagent_stop"
    TASK_CREATED = "task_created"
    TASK_COMPLETED = "task_completed"
    PERMISSION_REQUEST = "permission_request"
    PERMISSION_DENIED = "permission_denied"
    # STOP 的失败变体（主循环异常退出时才触发，区别于正常结束）
    STOP_FAILURE = "stop_failure"
    # worktree 隔离工作区创建/清理的通知事件
    WORKTREE_CREATE = "worktree_create"
    WORKTREE_REMOVE = "worktree_remove"
    # === 集成/通知类 6 种 ===
    FILE_CHANGED = "file_changed"                   # 文件写入成功后触发（做 IDE 集成的基础）
    CWD_CHANGED = "cwd_changed"                     # worktree 切目录时（配合 workspace_context）
    INSTRUCTIONS_LOADED = "instructions_loaded"     # CLAUDE.md/OMNIMATE.md 加载完后
    SETUP = "setup"                                 # 启动时触发一次（cli.py initialize）
    TEAMMATE_IDLE = "teammate_idle"                 # 团队协作的成员进入空闲
    ELICITATION_STARTED = "elicitation_started"     # ask_user 弹窗问用户之前


# 程序式（直接给 Python 函数）hook 的函数签名
UserPromptSubmitFn = Callable[[str], Optional[str]]
PreToolUseFn = Callable[[str, dict], Optional[dict]]
PostToolUseFn = Callable[[str, dict, str], Optional[str]]
StopFn = Callable[[], Optional[str]]
# LLM 调用前后两个事件的签名
PreLLMCallFn = Callable[[list, Optional[list]], Optional[tuple]]
PostLLMCallFn = Callable[[object], Optional[object]]
# 新事件统一用 dict 进、Optional[dict] 出的风格：
# - SESSION_START/END、POST_COMPACT、CONFIG_CHANGE 是纯通知型，返回值直接忽略
# - PRE_COMPACT 特殊：返回 {"abort": True} 可以阻止这一层压缩
PayloadFn = Callable[[dict], Optional[dict]]


@dataclass
class HookScriptConfig:
    """声明式 hook 的配置（一个 hook 可以由 5 种不同的"执行器"来跑）。

    声明式 hook 就是写在配置里的 hook，不用写 Python 代码。
    具体怎么执行由 handler_type 决定，共 5 种：
    - command:  起一个本地子进程跑（默认值——hook_loader
      加载配置时总会填这个字段）
    - http:     往 url 发一个 POST JSON 请求，解析响应
    - mcp_tool: 调一个 MCP 外部工具（mcp_server 服务器上的 mcp_tool）
    - prompt:   让辅助小模型（aux_llm）单轮评估一次
    - agent:    让子代理（多轮）评估

    条件过滤字段：
    - if_condition: 声明式的条件过滤，写法沿用 permission rule 语法。
      只对 PRE_TOOL_USE / POST_TOOL_USE / POST_TOOL_USE_FAILURE /
      PERMISSION_REQUEST 这四种事件有意义；条件不匹配就跳过这个 hook
      （省得每次都白跑一遍）。格式如 "terminal(git *)"——
      意思是"只有 terminal 工具且参数以 git 开头才触发"。
    """
    handler_type: str = "command"   # 五种执行器之一：command | http | mcp_tool | prompt | agent
    command: Optional[list] = None  # 命令及参数列表，command 类型用（老配置里必填）
    url: Optional[str] = None       # 目标网址，http 类型用
    mcp_server: Optional[str] = None  # MCP 服务器名，mcp_tool 类型用
    mcp_tool: Optional[str] = None    # MCP 工具名，mcp_tool 类型用
    prompt: Optional[str] = None      # 提示词模板，prompt / agent 类型用（用 .format(**payload) 填充）
    agent_name: Optional[str] = None  # 自定义子代理的名字，agent 类型用（可不填）
    timeout: float = 10.0
    env: Optional[dict] = None
    # 条件过滤（permission rule 语法；None/空字符串 = 不过滤、每次都匹配）
    if_condition: Optional[str] = None
    # 异步 hook（只有 command 类型支持）
    # - async_run: 放到后台线程跑，dispatch 立刻返回 None，不阻塞主流程。
    #   ⚠ 代价是"拦截"语义失效：异步的 PRE_TOOL_USE 就算想 deny 也来不及拦——
    #   所以 async 只该用在通知/审计这类"事后知道就行"的 hook 上
    # - async_rewake: 后台跑完且退出码为 2（block）时，推一条 rewake 通知，
    #   agent 下一轮把它作为临时 <rewake_notification> 消息喂给模型跟进处理
    # - status_message: 给人看的说明文字（随 rewake 通知带出；日志/UI 显示用）
    async_run: bool = False
    async_rewake: bool = False
    status_message: Optional[str] = None


@dataclass
class Hook:
    """一个 hook 的统一外壳：程序式（带 fn）或声明式（带 script）二选一。"""
    name: str
    event: HookEvent
    kind: str  # "programmatic"（Python 函数） | "declarative"（配置声明）
    fn: Optional[Callable] = None
    script: Optional[HookScriptConfig] = None
    fail_closed: bool = False
    # once=True 表示"一次性"hook，跑过一次后就消费掉（后续触发直接跳过）
    once: bool = False
    # 声明式 command hook 是否套 OS 沙箱（只在 Unix 生效；Windows 上降级跳过沙箱）
    use_sandbox: bool = False


def _now_iso() -> str:
    """当前时间的 ISO 字符串（精确到秒），给 hook payload 的时间戳字段用。"""
    return datetime.now().isoformat(timespec="seconds")


class HookRegistry:
    """所有 hook 的登记处兼触发器：登记谁、什么时候挨个跑、结果怎么合并。

    实例由 cli.py 的 RuntimeContext 持有并注入 AIAgent——agent 主循环
    在各个节点上调这里的 run_xxx 方法来触发对应事件。
    """

    def __init__(self):
        self._hooks: dict = {e: [] for e in HookEvent}
        self._stop_fire_count: int = 0
        # once=True 且已经跑过（被"消费"）的 hook 对象 id 集合
        self._consumed: set = set()

    # ---- 注册（往登记簿上添条目） ----
    def register_user_prompt_submit(self, fn, *, name=None):
        """登记一个"用户提交问题前"触发的 hook。

        参数：
        - fn：hook 函数，签名 (prompt) -> 新 prompt 或 None（None = 不改）
        - name：hook 名字（日志里显示用，不填就记 "anonymous"）
        """
        self._hooks[HookEvent.USER_PROMPT_SUBMIT].append(
            Hook(name=name or "anonymous", event=HookEvent.USER_PROMPT_SUBMIT,
                 kind="programmatic", fn=fn)
        )

    def register_pre_tool_use(self, fn, *, name=None, fail_closed=False):
        """登记一个"工具调用前"触发的 hook（可以拦下工具调用）。

        参数：
        - fn：hook 函数，签名 (tool_name, args) -> dict 或 None；
          返回 {"deny": 理由} 拒绝执行、{"modify_args": 新参数} 改参数、None 不管
        - name：hook 名字
        - fail_closed：True 时 fn 抛异常按"拒绝"处理（默认按"不管"处理）
        """
        self._hooks[HookEvent.PRE_TOOL_USE].append(
            Hook(name=name or "anonymous", event=HookEvent.PRE_TOOL_USE,
                 kind="programmatic", fn=fn, fail_closed=fail_closed)
        )

    def register_post_tool_use(self, fn, *, name=None):
        """登记一个"工具调用后"触发的 hook（可以改写工具结果）。

        参数：
        - fn：hook 函数，签名 (tool_name, args, result) -> 新 result 或 None
        - name：hook 名字
        """
        self._hooks[HookEvent.POST_TOOL_USE].append(
            Hook(name=name or "anonymous", event=HookEvent.POST_TOOL_USE,
                 kind="programmatic", fn=fn)
        )

    def register_stop(self, fn, *, name=None):
        """登记一个"回答结束"触发的 hook（可以让模型继续说话）。

        参数：
        - fn：hook 函数，签名 () -> 消息字符串或 None；
          返回消息会作为"继续"的理由喂回模型
        - name：hook 名字
        """
        self._hooks[HookEvent.STOP].append(
            Hook(name=name or "anonymous", event=HookEvent.STOP,
                 kind="programmatic", fn=fn)
        )

    # ---- 会话/压缩/配置事件的注册 ----
    def register_session_start(self, fn, *, name=None):
        """登记会话开始事件 hook。fn(payload) -> None；
        payload 含 session_id、started_at、agent_home。参数：fn 为 hook 函数，name 为名字。"""
        self._hooks[HookEvent.SESSION_START].append(
            Hook(name=name or "anonymous", event=HookEvent.SESSION_START,
                 kind="programmatic", fn=fn)
        )

    def register_session_end(self, fn, *, name=None):
        """登记会话结束事件 hook。fn(payload) -> None；
        payload 含 session_id、reason（结束原因）、ended_at。参数：fn 为 hook 函数，name 为名字。"""
        self._hooks[HookEvent.SESSION_END].append(
            Hook(name=name or "anonymous", event=HookEvent.SESSION_END,
                 kind="programmatic", fn=fn)
        )

    def register_pre_compact(self, fn, *, name=None):
        """登记"压缩前"事件 hook——这是唯一能叫停压缩的：
        fn(payload) -> {"abort": True} 可阻止这一层压缩；
        payload 含 layer、messages_count、est_tokens。参数：fn 为 hook 函数，name 为名字。"""
        self._hooks[HookEvent.PRE_COMPACT].append(
            Hook(name=name or "anonymous", event=HookEvent.PRE_COMPACT,
                 kind="programmatic", fn=fn)
        )

    def register_post_compact(self, fn, *, name=None):
        """登记"压缩后"通知 hook。fn(payload) -> None；
        payload 含 messages_before、messages_after、layer。参数：fn 为 hook 函数，name 为名字。"""
        self._hooks[HookEvent.POST_COMPACT].append(
            Hook(name=name or "anonymous", event=HookEvent.POST_COMPACT,
                 kind="programmatic", fn=fn)
        )

    def register_config_change(self, fn, *, name=None):
        """登记配置变更通知 hook。fn(payload) -> None；
        payload 含 changed_keys、old、new。参数：fn 为 hook 函数，name 为名字。"""
        self._hooks[HookEvent.CONFIG_CHANGE].append(
            Hook(name=name or "anonymous", event=HookEvent.CONFIG_CHANGE,
                 kind="programmatic", fn=fn)
        )

    # ---- LLM 调用前后的 hook 注册 ----
    def register_pre_llm_call(self, fn, *, name=None):
        """登记"LLM 调用前"触发的 hook（可以在请求发出去之前改消息和工具表）。

        参数：
        - fn：hook 函数，签名 (messages, tools) -> (messages, tools) 元组或 None；
          返回元组就替换成新值，返回 None 表示不动
        - name：hook 名字
        """
        self._hooks[HookEvent.PRE_LLM_CALL].append(
            Hook(name=name or "anonymous", event=HookEvent.PRE_LLM_CALL,
                 kind="programmatic", fn=fn)
        )

    def register_post_llm_call(self, fn, *, name=None):
        """登记"LLM 调用后"触发的 hook（可以改写模型响应）。

        参数：
        - fn：hook 函数，签名 (response) -> 新 response 或 None（None 不动）
        - name：hook 名字
        """
        self._hooks[HookEvent.POST_LLM_CALL].append(
            Hook(name=name or "anonymous", event=HookEvent.POST_LLM_CALL,
                 kind="programmatic", fn=fn)
        )

    def register_declarative(self, hook: Hook):
        """登记一个声明式 hook（Hook 对象已在外面的 hook_loader 构造好）。

        参数：
        - hook：装配完毕的 Hook 对象（kind="declarative"，带 script 配置）
        """
        self._hooks[hook.event].append(hook)

    def clear(self, event=None):
        """清空登记簿（测试用，让每个测试从干净状态开始）。

        参数：
        - event：只清这一种事件；None 表示全部清空
        """
        if event is None:
            for e in self._hooks:
                self._hooks[e] = []
        else:
            self._hooks[event] = []
        self._stop_fire_count = 0
        self._consumed.clear()  # 一次性 hook 的消费记录也要一并清掉

    # ---- 一次性/条件过滤的小工具 ----

    def _is_consumed(self, hook: "Hook") -> bool:
        """判断一个 once=True 的 hook 是不是已经跑过（被消费）了。"""
        return hook.once and id(hook) in self._consumed

    def _mark_consumed_if_once(self, hook: "Hook") -> None:
        """声明式 hook 跑完后，如果是 once 就记下"已消费"，下次跳过。"""
        if hook.once:
            self._consumed.add(id(hook))

    def _matches_if_condition(self, hook: "Hook", tool_name: str, args: dict) -> bool:
        """声明式 hook 的 if 条件过滤——条件不匹配就不触发这个 hook。

        参数：
        - hook：要检查的 hook
        - tool_name：当前要执行的工具名
        - args：当前工具的参数

        规则：hook.script 没配 if_condition 就一律匹配；配了就交给
        agent.hook_filter.match_if_condition 去比对。程序式 hook 不走
        这里（调用方保证只在声明式时才调这个方法）。
        """
        if hook.script is None:
            return True
        cond = getattr(hook.script, "if_condition", None)
        if not cond:
            return True
        from agent.hook_filter import match_if_condition
        return match_if_condition(tool_name, args, cond)

    # ---- 执行：用户提交问题事件 ----
    def run_user_prompt_submit(self, prompt: str, *, session_id: str) -> str:
        """把用户输入的 prompt 依次过一遍所有 hook（像流水线，每个 hook
        都能看到并改写前一个的输出），返回最终版本的 prompt——送进模型
        前可做改写、过滤、注入上下文等。
        单个 hook 出异常就记条日志跳过（fail-open，当它不存在）。

        参数：
        - prompt：用户原始输入
        - session_id：当前会话 id（拼进 hook 的 payload）

        声明式 hook 支持 once（跑一次后消费）。
        """
        for hook in self._hooks[HookEvent.USER_PROMPT_SUBMIT]:
            if hook.kind == "declarative":
                if self._is_consumed(hook):
                    continue
            try:
                if hook.kind == "programmatic":
                    new_prompt = hook.fn(prompt)
                else:
                    # 声明式 hook 走子进程/外部执行器
                    new_prompt = self._invoke_declarative_user_prompt(hook, prompt, session_id)
                    self._mark_consumed_if_once(hook)
                if new_prompt is not None:
                    prompt = new_prompt
            except Exception as e:
                logger.warning("hook %s 异常（视为 None）: %s", hook.name, e)
        return prompt

    def _invoke_declarative_user_prompt(self, hook, prompt, session_id):
        """跑一个声明式 hook 并按约定的返回格式解析。

        参数：
        - hook：要跑的声明式 hook
        - prompt：当前 prompt
        - session_id：会话 id

        返回：新 prompt（hook 返回 {"prompt": "..."} 时）或 None（不改）。
        """
        from agent.hook_exec import dispatch_hook  # 懒加载，避免和 hook_exec 循环 import
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
        # 约定的返回格式：{"prompt": "..."} 表示替换；其他/空一律当 None
        return result.get("prompt")

    # ---- 执行：工具调用前事件 ----
    def run_pre_tool_use(self, tool_name: str, args: dict, *,
                         session_id: str):
        """工具执行前把所有匹配的 hook 跑一遍，汇总出"要不要拦、要不要改参数"。

        这是唯一能"否决"工具调用的 hook 事件，所以要考虑多个 hook
        意见不一致时听谁的。返回 (deny_reason, modified_args)：
        deny_reason 非 None 就拒绝执行工具；modified_args 非 None 就替换参数。

        怎么跑（并行执行 + 聚合）：
        - 所有匹配的 hook 先全跑完再合并结论。声明式（子进程，慢）的用
          线程池并行——一个慢 hook 不拖累其他 hook 和主循环；程序式
          （进程内的快函数）保持串行。
        - 合并优先级：deny（拒）> ask（要人批）> allow/None（放行）。
          多个 hook 都要改参数时按注册顺序叠加（先到的做基底，
          后到的按键覆盖，异键保留并集）。
          注意：并行跑时每个 hook 是基于**原始参数**做的判断——
          "多个 hook 同时改参互相看得见"的旧串行语义在并行下略有出入
          （罕见场景才碰得到）。
        - ask 档：本项目 pre_tool_use 没有通用的
          审批界面，ask 就按"宁可拒绝"（fail-closed）处理，报错里注明
          是 ask（用户可以去调整 hook 规则）。
        - 配了 fail_closed 的 hook 出异常按 deny 汇总（异常不能吞——
          吞掉的话 fail_closed 分支根本走不到）。

        参数：
        - tool_name：要执行的工具名
        - args：工具的原始参数
        - session_id：会话 id

        声明式 hook 支持 if 条件过滤（permission rule 语法）和 once（跑一次后消费）。
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
            """跑单个 hook。返回 (hook, 判决结果) 或 (hook, 异常对象)。"""
            try:
                if hook.kind == "programmatic":
                    return hook, hook.fn(tool_name, hook_args)
                result = self._invoke_declarative_pre_tool(hook, tool_name, hook_args, session_id)
                self._mark_consumed_if_once(hook)
                return hook, result
            except Exception as e:  # noqa: BLE001
                return hook, e

        # 程序式是进程内的快函数——**串行流水线**（后一个能看到前一个
        # 改过的参数，保持旧语义）；声明式要起子进程、慢——并行跑
        # （判决都基于原始参数；同时改参数属罕见场景，按注册顺序叠加合并）
        prog = [h for h in matched if h.kind == "programmatic"]
        decl = [h for h in matched if h.kind != "programmatic"]
        outcomes = []
        current_args = args
        for h in prog:
            outcome = _run_one(h, current_args)
            outcomes.append(outcome)
            # 流水线语义（同旧串行版）：后一个 hook 看到前一个改过的参数
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
                # 多个 hook 改参数必须按注册顺序叠加合并
                # （outcomes 本身就是按注册序收集的）——后到者整体替换会让
                # 先到的 hook 改的东西静默丢掉。合并规则：
                # 第一个 hook 的完整返回做基底，后面的按键覆盖（同键后到胜、异键并集）
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
        """跑一个声明式 pre_tool_use hook 并把返回翻译成统一判决。

        参数：
        - hook：要跑的声明式 hook
        - tool_name / args：工具名和参数
        - session_id：会话 id

        返回：{"deny": 理由} / {"modify_args": 新参数} / None（放行）。

        支持子进程 exit code 2 的拦截协议——
        dispatch_hook 会把它转成 {"action": "block", "reason": stderr}，
        这里再转成 {"deny": reason}。
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
        # 必须传 propagate_error=True——fail_closed
        # 的 hook 执行失败时异常要向上抛，让 run_pre_tool_use 的 except 分支
        # 转成 deny（不传的话 dispatch_hook 内部把一切异常都吞了，fail_closed
        # 分支永远走不到，等于白配）
        result = dispatch_hook(hook, payload, propagate_error=True)
        if result is None:
            return None
        action = result.get("action", "allow")
        # block（exit code 2）等同于 deny
        if action in ("deny", "block"):
            return {"deny": result.get("reason", "unspecified")}
        # ask 档（升审批语义——由 run_pre_tool_use 聚合层兑现）
        if action == "ask":
            return {"ask": result.get("reason", "unspecified")}
        if action == "modify":
            return {"modify_args": result.get("args", args)}
        return None

    # ---- 执行：工具调用后事件 ----
    def run_post_tool_use(self, tool_name: str, args: dict, result: str,
                          *, session_id: str) -> str:
        """工具结果依次过一遍所有 hook（流水线：每个 hook 看到前一个的
        输出），返回最终版本的结果。单个 hook 出异常就跳过（fail-open）。

        参数：
        - tool_name：刚执行的工具名
        - args：工具参数
        - result：工具的原始结果
        - session_id：会话 id

        声明式 hook 支持 if 条件过滤 + once 消费。
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
        """跑一个声明式 post_tool_use hook。

        参数：hook 为要跑的 hook；tool_name/args 为工具名和参数；
        result 为工具结果；session_id 为会话 id。
        返回：新 result（hook 返回 {"result": "..."} 时）或 None（不改）。
        """
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

    # ---- 执行：回答结束事件 ----
    def run_stop(self, *, session_id: str, max_fires: int = 3) -> Optional[str]:
        """模型答完一轮时挨个问 hook"要不要让它继续"。

        规则：任何一个 hook 返回了消息就立刻以它为准（第一个
        非 None 胜出），这条消息会喂回模型让它继续干活。为防止 hook
        之间互相触发形成死循环，触发超过 max_fires 次后强制返回 None
        （不再继续）。

        参数：
        - session_id：会话 id
        - max_fires：本会话最多让 STOP hook"续命"几次（默认 3）

        声明式 hook 支持 once（跑一次后消费）。
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
        """跑一个声明式 stop hook。

        参数：hook 为要跑的 hook；session_id 为会话 id。
        返回："继续"的消息（hook 返回 {"continue": "..."} 时）或 None。
        """
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
        """跑声明式 hook 的统一入口：拼好 payload 交给 dispatch_hook 执行——通知型事件（会话开始/结束等）的声明式 hook 都从这里走，免得每个事件重复拼 payload。

        payload 固定带 event/session_id/
        timestamp/hook_name 四个字段，事件自己的字段用 extra 传
        （注意别塞超大内容，只传摘要信息——子进程间传大块数据太亏）。
        dispatch_hook 会按 hook.script.handler_type 挑对应的执行器
        （command/http/mcp_tool/prompt/agent 五种）。

        参数：
        - hook：要跑的声明式 hook
        - session_id：会话 id
        - event：事件名字符串
        - timeout_cap：超时上限（非空时和 hook 自配的超时取较小者；
          会话结束时用——退出不能被慢 hook 卡住）
        - **extra：事件特定字段，原样并进 payload

        返回：dispatch_hook 的 dict 结果，或 None。
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

    # ---- 执行 LLM 调用前/后事件 ----
    def run_pre_llm_call(self, messages: list, tools: Optional[list],
                         *, session_id: str = "") -> tuple:
        """请求发给 LLM 之前，把消息列表和工具表依次过一遍 hook（流水线）。

        每个 hook 返回 (messages, tools) 元组就替换，返回 None 就保持
        上一轮的输出不动。声明式 hook 只当通知跑子进程，不改内容。
        单个 hook 出异常就跳过（fail-open）。

        参数：
        - messages：即将发给 LLM 的消息历史
        - tools：即将随请求发给 LLM 的工具定义列表
        - session_id：会话 id

        返回：处理后的 (messages, tools) 元组。
        """
        for hook in self._hooks[HookEvent.PRE_LLM_CALL]:
            try:
                if hook.kind == "programmatic":
                    result = hook.fn(messages, tools)
                else:
                    # 声明式：通知型，只传消息条数这类摘要信息（别把大 payload 塞给子进程）
                    self._invoke_declarative_script(
                        hook, session_id, "pre_llm_call",
                        message_count=len(messages),
                    )
                    result = None
                if result is not None:
                    # 拆开 (messages, tools) 元组
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
        """LLM 返回响应后，把 response 依次过一遍 hook（流水线）。

        每个 hook 返回新 response 就替换，返回 None 不动。声明式 hook
        只当通知跑子进程，不改 response。单个 hook 出异常就跳过（fail-open）。

        参数：
        - response：LLM 的原始响应对象
        - session_id：会话 id

        返回：处理后的 response。
        """
        for hook in self._hooks[HookEvent.POST_LLM_CALL]:
            try:
                if hook.kind == "programmatic":
                    result = hook.fn(response)
                else:
                    # 声明式：纯通知
                    self._invoke_declarative_script(hook, session_id, "post_llm_call")
                    result = None
                if result is not None:
                    response = result
            except Exception as e:
                logger.warning("POST_LLM_CALL hook %s 异常（视为 None）: %s",
                               hook.name, e)
        return response

    # ---- 会话/压缩/配置事件的执行 ----
    def run_session_start(self, payload: dict) -> None:
        """通知型：会话开始时把所有 SESSION_START hook 都叫一遍，返回值
        一律忽略。单个 hook 出异常就跳过（fail-open）。

        参数：
        - payload：事件数据（session_id、started_at、agent_home 等）
        """
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

    # 会话结束 hook 的独立短超时（默认 1500ms）——
    # 收尾阶段不能被慢 hook 卡死（退出时干等 10 秒体验太差）
    SESSION_END_TIMEOUT_CAP = 1.5

    def run_session_end(self, payload: dict) -> None:
        """通知型：会话结束时把所有 SESSION_END hook 都叫一遍。
        单个 hook 出异常就跳过（fail-open）。

        参数：
        - payload：事件数据（session_id、reason、ended_at 等）

        声明式 hook 超时被钳到 SESSION_END_TIMEOUT_CAP
        （hook 自己配得更短就尊重更短的值）。
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
        """压缩前问一遍 hook"这层压缩要不要做"。

        任何一个 hook 返回 {"abort": True} 就立刻停下（后面的 hook 不再
        跑），并把"叫停了"告诉调用方。

        参数：
        - payload：事件数据（layer、messages_count、est_tokens 等）

        返回：{"abort": bool}（abort=True 时调用方应跳过这一层压缩；
        带 blocked_by 说明是哪个 hook 叫停的）。
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
        """通知型：压缩做完后告知所有 hook（供指标收集、记日志等用）。

        参数：
        - payload：事件数据（messages_before、messages_after、layer 等）
        """
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
        """通知型：配置变更后告知所有 hook（供审计、缓存失效等用）。

        参数：
        - payload：事件数据（changed_keys、old、new）
        """
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

    # ---- 7 个关键事件 ----

    def register_post_tool_use_failure(self, fn, *, name=None):
        """登记"工具调用失败后"通知 hook（result 里带 error 时触发）。
        参数：fn 为 hook 函数（收 payload dict，无返回值）；name 为名字。"""
        self._hooks[HookEvent.POST_TOOL_USE_FAILURE].append(
            Hook(name=name or "anonymous", event=HookEvent.POST_TOOL_USE_FAILURE,
                 kind="programmatic", fn=fn))

    def run_post_tool_use_failure(self, payload: dict) -> None:
        """通知型：工具调用失败（result 含 error）时告知所有 hook。
        单个 hook 出异常就跳过（fail-open）。

        参数：
        - payload：事件数据（session_id、tool、error、error_type 等）
        """
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
        """登记"子代理启动时"通知 hook。
        参数：fn 为 hook 函数（收 payload dict，无返回值）；name 为名字。"""
        self._hooks[HookEvent.SUBAGENT_START].append(
            Hook(name=name or "anonymous", event=HookEvent.SUBAGENT_START,
                 kind="programmatic", fn=fn))

    def run_subagent_start(self, payload: dict) -> None:
        """通知型：子代理启动时告知所有 hook。fail-open。

        参数：
        - payload：事件数据（session_id、subagent、goal、spawn_depth 等）
        """
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
        """登记"子代理结束时"通知 hook。
        参数：fn 为 hook 函数（收 payload dict，无返回值）；name 为名字。"""
        self._hooks[HookEvent.SUBAGENT_STOP].append(
            Hook(name=name or "anonymous", event=HookEvent.SUBAGENT_STOP,
                 kind="programmatic", fn=fn))

    def run_subagent_stop(self, payload: dict) -> None:
        """通知型：子代理结束时告知所有 hook。fail-open。

        参数：
        - payload：事件数据（session_id、subagent、goal、success 等）
        """
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
        """登记"任务创建时"通知 hook。
        参数：fn 为 hook 函数（收 payload dict，无返回值）；name 为名字。"""
        self._hooks[HookEvent.TASK_CREATED].append(
            Hook(name=name or "anonymous", event=HookEvent.TASK_CREATED,
                 kind="programmatic", fn=fn))

    def run_task_created(self, payload: dict) -> None:
        """通知型：任务创建时告知所有 hook。fail-open。

        参数：
        - payload：事件数据（session_id、task_id、subject、owner 等）
        """
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
        """登记"任务完成时"通知 hook。
        参数：fn 为 hook 函数（收 payload dict，无返回值）；name 为名字。"""
        self._hooks[HookEvent.TASK_COMPLETED].append(
            Hook(name=name or "anonymous", event=HookEvent.TASK_COMPLETED,
                 kind="programmatic", fn=fn))

    def run_task_completed(self, payload: dict) -> None:
        """通知型：任务完成时告知所有 hook。fail-open。

        参数：
        - payload：事件数据（session_id、task_id、unblocked 等）
        """
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
        """登记"权限请求时"通知 hook（工具要用户审批前触发）。
        参数：fn 为 hook 函数（收 payload dict，无返回值）；name 为名字。"""
        self._hooks[HookEvent.PERMISSION_REQUEST].append(
            Hook(name=name or "anonymous", event=HookEvent.PERMISSION_REQUEST,
                 kind="programmatic", fn=fn))

    def run_permission_request(self, payload: dict) -> None:
        """通知型：权限请求时告知所有 hook。fail-open。

        参数：
        - payload：事件数据（session_id、command、reason 等）
        """
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
        """登记"权限被拒时"通知 hook。
        参数：fn 为 hook 函数（收 payload dict，无返回值）；name 为名字。"""
        self._hooks[HookEvent.PERMISSION_DENIED].append(
            Hook(name=name or "anonymous", event=HookEvent.PERMISSION_DENIED,
                 kind="programmatic", fn=fn))

    def run_permission_denied(self, payload: dict) -> None:
        """通知型：权限被拒后告知所有 hook。fail-open。

        参数：
        - payload：事件数据（session_id、command、reason、deny_type 等）
        """
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

    # ---- STOP_FAILURE（回答异常结束）事件 ----

    def register_stop_failure(self, fn, *, name=None):
        """登记 STOP_FAILURE hook（通知型，返回值忽略）。

        什么时候触发：agent 主循环 LLM 调用失败/异常退出时——注意和正常
        回答结束（STOP）区分开，一个是"摔了"，一个是"正常干完"。

        参数：
        - fn：hook 函数，收 payload dict（session_id、error、error_type、timestamp）
        - name：hook 名字
        """
        self._hooks[HookEvent.STOP_FAILURE].append(
            Hook(name=name or "anonymous", event=HookEvent.STOP_FAILURE,
                 kind="programmatic", fn=fn))

    def run_stop_failure(self, payload: dict) -> None:
        """通知型：主循环异常退出时告知所有 hook。fail-open。

        参数：
        - payload：事件数据（session_id、error、error_type 等）
        """
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

    # ---- worktree（隔离工作区）创建/清理事件 ----

    def register_worktree_create(self, fn, *, name=None):
        """登记 WORKTREE_CREATE hook（通知型）。

        什么时候触发：tools/worktree.py 的 create_isolated_workspace
        成功创建出隔离工作区之后。

        参数：
        - fn：hook 函数，收 payload dict（session_id、path、branch、workspace_type）
        - name：hook 名字
        """
        self._hooks[HookEvent.WORKTREE_CREATE].append(
            Hook(name=name or "anonymous", event=HookEvent.WORKTREE_CREATE,
                 kind="programmatic", fn=fn))

    def run_worktree_create(self, payload: dict) -> None:
        """通知型：worktree 创建后告知所有 hook（供审计、登记待清理等）。
        fail-open。

        参数：
        - payload：事件数据（session_id、path、branch 等）
        """
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
        """登记 WORKTREE_REMOVE hook（通知型）。

        什么时候触发：tools/worktree.py 的 cleanup 清理函数执行完之后。

        参数：
        - fn：hook 函数，收 payload dict（session_id、path、branch）
        - name：hook 名字
        """
        self._hooks[HookEvent.WORKTREE_REMOVE].append(
            Hook(name=name or "anonymous", event=HookEvent.WORKTREE_REMOVE,
                 kind="programmatic", fn=fn))

    def run_worktree_remove(self, payload: dict) -> None:
        """通知型：worktree 清理后告知所有 hook。fail-open。

        参数：
        - payload：事件数据（session_id、path 等）
        """
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

    # ---- 6 个通知/集成事件 ----

    def register_file_changed(self, fn, *, name=None):
        """登记 FILE_CHANGED hook（通知型）。

        什么时候触发：write_file / str_replace 等文件写入成功之后
        （做 IDE 集成的基础）。

        参数：
        - fn：hook 函数，收 payload dict（session_id、path、op），
          op 取 "write" / "append" / "edit" 三种
        - name：hook 名字
        """
        self._hooks[HookEvent.FILE_CHANGED].append(
            Hook(name=name or "anonymous", event=HookEvent.FILE_CHANGED,
                 kind="programmatic", fn=fn))

    def run_file_changed(self, payload: dict) -> None:
        """通知型：文件写入成功后告知所有 hook（供 IDE 同步、审计、
        热重载等）。fail-open。

        参数：
        - payload：事件数据（session_id、path、op 等）
        """
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
        """登记 CWD_CHANGED hook（通知型）。

        什么时候触发：workspace_context 切换工作目录时
        （子代理进/出 worktree 的场景）。

        参数：
        - fn：hook 函数，收 payload dict（session_id、old、new——旧新目录）
        - name：hook 名字
        """
        self._hooks[HookEvent.CWD_CHANGED].append(
            Hook(name=name or "anonymous", event=HookEvent.CWD_CHANGED,
                 kind="programmatic", fn=fn))

    def run_cwd_changed(self, payload: dict) -> None:
        """通知型：工作目录切换后告知所有 hook。fail-open。

        参数：
        - payload：事件数据（session_id、old、new）
        """
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
        """登记 INSTRUCTIONS_LOADED hook（通知型）。

        什么时候触发：prompt_builder 加载完项目的 CLAUDE.md/OMNIMATE.md
        （项目使用说明）之后。

        参数：
        - fn：hook 函数，收 payload dict（session_id、source、bytes）
        - name：hook 名字
        """
        self._hooks[HookEvent.INSTRUCTIONS_LOADED].append(
            Hook(name=name or "anonymous", event=HookEvent.INSTRUCTIONS_LOADED,
                 kind="programmatic", fn=fn))

    def run_instructions_loaded(self, payload: dict) -> None:
        """通知型：项目使用说明加载完后告知所有 hook。fail-open。

        参数：
        - payload：事件数据（session_id、source 等）
        """
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
        """登记 SETUP hook（通知型）。

        什么时候触发：cli.py 的 RuntimeContext.initialize 收尾时——
        每次启动只触发一次。

        参数：
        - fn：hook 函数，收 payload dict（session_id、agent_home、started_at）
        - name：hook 名字
        """
        self._hooks[HookEvent.SETUP].append(
            Hook(name=name or "anonymous", event=HookEvent.SETUP,
                 kind="programmatic", fn=fn))

    def run_setup(self, payload: dict) -> None:
        """通知型：agent 启动时触发一次。fail-open。

        参数：
        - payload：事件数据（session_id、agent_home 等）
        """
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
        """登记 TEAMMATE_IDLE hook（通知型）。

        什么时候触发：团队协作中某个成员进入空闲状态时。

        参数：
        - fn：hook 函数，收 payload dict（session_id、member）
        - name：hook 名字
        """
        self._hooks[HookEvent.TEAMMATE_IDLE].append(
            Hook(name=name or "anonymous", event=HookEvent.TEAMMATE_IDLE,
                 kind="programmatic", fn=fn))

    def run_teammate_idle(self, payload: dict) -> None:
        """通知型：团队成员空闲时告知所有 hook。fail-open。

        参数：
        - payload：事件数据（session_id、member）
        """
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
        """登记 ELICITATION_STARTED hook（通知型）。

        什么时候触发：ask_user 工具要弹窗问用户之前（给 UI 集成用的）。

        参数：
        - fn：hook 函数，收 payload dict（session_id、prompt——问的问题）
        - name：hook 名字
        """
        self._hooks[HookEvent.ELICITATION_STARTED].append(
            Hook(name=name or "anonymous", event=HookEvent.ELICITATION_STARTED,
                 kind="programmatic", fn=fn))

    def run_elicitation_started(self, payload: dict) -> None:
        """通知型：ask_user 弹窗前告知所有 hook。fail-open。

        参数：
        - payload：事件数据（session_id、prompt）
        """
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
