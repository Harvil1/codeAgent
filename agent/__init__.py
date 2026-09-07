"""AIAgent 主类：整个系统的「大脑 + 主循环」，全项目最核心的文件。

大白话：一个 AIAgent 实例就是一场对话（一个会话）。你发一句话进来，
它反复「问大模型 → 大模型说要用工具 → 执行工具 → 把结果喂回去再问」，
直到大模型给出不用工具的最终回答为止。对话记录通过 session_id 存进
SessionStore（会话库，JSONL 文件），重开会话能恢复。

主循环（run_conversation）伪代码：
    while 迭代预算还没用完 and 用户没按中断:
        把 system prompt + 历史消息拼成 messages
        调 LLM（大模型）
        如果大模型要求调工具（tool_calls）：
            执行每个工具，结果追加到历史
            continue（再来一轮）
        否则：
            返回最终回答

几个必须知道的设计：
- 整个循环是 async 的（异步），只读类工具可以并发跑
- system prompt（系统提示词）只在会话开头构建一次然后缓存——中途改它
  会让 LLM 服务商的前缀缓存（prompt cache）失效，token 费用翻倍
- 中断是「商量着来」的：Ctrl+C 只是设个标志，循环每轮自己检查后退出，
  不硬杀（硬杀可能把消息历史弄坏）
- 迭代预算用完后额外给一次「遗言机会」（grace call），让模型把刚拿到的
  工具结果消化完、说句收尾的话再结束
"""

import asyncio
import contextvars
import json
import logging
import os
import threading
import time
from typing import Optional

from agent.budget import IterationBudget
from agent.context_pipeline import CompressionSessionState, strip_internal_fields
from agent.context_compressor import reset_compact_circuit_breaker
# 临时注入纯函数区拆到独立模块（行为零变化搬迁）；带下划线别名 re-export
# 保住旧调用点与习惯引用
from agent.ephemeral_inject import (
    LoopExitReason,
    build_goal_continue_message as _build_goal_continue_message,
    build_channel_injection as _build_channel_injection,
    build_mail_injection as _build_mail_injection,
    drop_leading_system as _drop_leading_system,
)
from agent.prompt_builder import build_system_prompt
# 批间摘要 + 条件技能激活五件拆到独立模块（行为零变化搬迁）；属性全留
# AIAgent，自由函数第一参收 agent 实例（原 self）
from agent.tool_batch_summary import (
    activate_conditional_skills,
    flush_skill_activations,
    generate_tool_batch_summary,
    queue_skill_activation,
    start_tool_batch_summary,
)
from tools.registry import registry

logger = logging.getLogger(__name__)


def _spawn_detached(coro, name: str):
    """把后台协程交给常驻事件循环宿主（loop_host）跑——发出去就不管。

    旧实现是「独立 daemon 线程 + 独立事件循环」：因为当时每回合
    asyncio.run 会把关联任务全部取消，后台任务只能另立门户（线程数
    随任务涨）。现在回合跑在常驻宿主循环上，后台任务直接 submit 上去
    （注册进回合栅栏的豁免名单，不会被回合结束误杀）；submit 会以
    调用方线程的 contextvars 上下文创建 Task（create_task(context=)），
    工作目录等上下文对后台任务可见。

    参数：
        coro: 要在后台跑的协程（async 函数调用后产生的对象）
        name: 任务名（日志里好认）

    返回：concurrent.futures.Future。调用方留着引用可以防它被回收；
    loop_host 侧也有豁免名单持有任务，不需要 join。
    """
    from agent.loop_host import loop_host
    return loop_host.submit(coro, name=name)


class AIAgent:
    """核心 Agent 类。一个实例 = 一场对话，主循环在下面的 run_conversation。"""

    # 哨兵标记：代表「刚做了紧急压缩（reactive_compact），本轮没产出，主循环请重试」
    _REACTIVE_RETRY = object()

    def __init__(
        self,
        *,
        base_url: str = None,
        api_key: str = None,
        auth_token: str = None,            # DeepSeek 的 Anthropic 端点要 Bearer 认证
        effort_level: str = None,          # 思考强度档位: max / high / medium / low
        model: str = "deepseek-chat",
        model_format: str = "openai",      # openai / anthropic
        fallback_model: str = None,
        max_iterations: int = 200,
        enabled_toolsets: list = None,
        session_id: str = None,
        system_prompt_override: str = None,
        memory_store=None,
        memory_manager=None,
        session_store=None,
        codeagent_home=None,
        on_tool_call=None,
        on_response=None,
        config: dict = None,
        hooks_registry=None,   # 钩子注册表
        bg_manager=None,       # 后台任务管理器
        cron_scheduler=None,   # 定时任务调度器
        team_bus=None,           # 团队消息总线
        team_coordinator=None,   # 团队协调器
        team_name=None,          # 团队里的名字
        spawn_depth: int = 0,    # 嵌套深度：主代理 0、子代理 1+
        aux_llm_router=None,     # 辅助小模型路由器
        plan_approval_callback=None,  # 计划审批回调（PlanMode 引入）
        stream_callback=None,    # 流式输出回调（04 引入）
        ask_user_bridge=None,    # ask_user 工具与 CLI 的桥（渲染问题+读选择）
        checkpoint_manager=None, # 文件快照/回滚管理器
        permission_mode: str = "default",  # 权限模式：default | bypassPermissions
        initial_messages: list = None,  # fork 子代理的初始消息
        omit_project_memory: bool = False,  # 子代理跳过项目 CODEAGENT.md
        trace_sink=None,  # 本地轨迹记录 sink
        goal_state=None,  # 目标驱动状态机
        channel_inbox=None,  # MCP 推送通知收件箱
        mailbox=None,  # 队友异步邮箱
        agent_name: str = "main",  # 自己的 agent 名，邮箱收件人
        stream_idle_timeout: float = None,  # 流空闲看门狗秒数（None=用默认 90，<=0 禁用）
    ):
        """初始化一个会话。参数很多，绝大多数是「外围零件」，由 CLI 层配好塞进来。

        参数：
            base_url: LLM API 的根地址（OpenAI 兼容格式）
            api_key: API 密钥
            auth_token: Bearer 认证令牌（DeepSeek 的 Anthropic 端点要用）
            effort_level: 思考强度档位（max / high / medium / low）
            model: 模型名（如 "deepseek-chat"）
            model_format: 用哪家的消息格式（"openai" 或 "anthropic"）
            fallback_model: 备用模型名（主模型重试耗尽后顶上）；None = 没有
            max_iterations: LLM 调用次数上限（防止模型无限循环跑飞）
            enabled_toolsets: 启用哪些工具集（如 ["core"]）
            session_id: 会话 ID（持久化、恢复都靠它）
            system_prompt_override: 直接给定 system prompt，跳过默认构建
                （子代理场景常用）；None = 走默认构建
            memory_store: 记忆仓库（MEMORY.md + USER.md 那套）
            memory_manager: 外部记忆管理器（跨 provider 同步用）
            session_store: 会话库（历史落盘）
            codeagent_home: 数据根目录（默认 ~/.codeAgent）；None 时自动解析
            on_tool_call: 工具开始执行时的回调（CLI 拿它打印进度）
            on_response: 拿到最终回答时的回调
            config: 完整配置 dict（压缩阈值、hooks 开关等都从这读）
            hooks_registry: 钩子注册表（用户挂在各事件上的自定义脚本）
            bg_manager: 后台任务管理器（长时间命令挂后台跑）
            cron_scheduler: 定时任务调度器
            team_bus: 团队消息总线（多 agent 同步通信）
            team_coordinator: 团队协调器
            team_name: 本 agent 在团队里的名字
            spawn_depth: 嵌套层数（主代理=0，子代理=1+；用于限制递归派生）
            aux_llm_router: 辅助小模型路由器（摘要/检索等杂活用它，省钱）
            plan_approval_callback: 计划审批回调，签名 (plan) -> (approved,
                feedback[, clear_context])；None = 自动批准（测试场景）
            stream_callback: 流式输出回调，LLM 每吐一段就调一次；None = 非流式
            ask_user_bridge: ask_user 工具与 CLI 的桥（渲染问题、读用户选择）
            checkpoint_manager: 文件快照/回滚管理器
            permission_mode: 权限模式（default=每次问 / bypassPermissions=跳过审批）
            initial_messages: 初始对话历史（fork 子代理继承父代理前缀用）；
                None = 从空历史开始
            omit_project_memory: True 时跳过项目 CODEAGENT.md 注入（子代理可配）
            trace_sink: 本地轨迹记录 sink（/trace 命令查的那个）
            goal_state: 目标驱动状态机（自动多轮推进）；None = 未启用
            channel_inbox: MCP 推送通知收件箱
            mailbox: 团队异步邮箱（agent_name 是收件人）
            agent_name: 自己的 agent 名（默认 "main"）
            stream_idle_timeout: 流式空闲看门狗秒数（流卡多久算超时）；
                None = 用 client 默认 90 秒；<=0 = 禁用
        """
        # 创建 LLM 客户端（按 model_format 选 OpenAI 兼容格式或 Anthropic 原生格式）
        from agent.llm_client import create_llm_client
        model_config = {
            "format": model_format,
            "base_url": base_url,
            "api_key": api_key,
            "auth_token": auth_token,
            "effort_level": effort_level,
            "model": model,
        }
        # 流式空闲看门狗（流卡住多久算超时，settings.json 的 llm.stream_idle_timeout_seconds）
        if stream_idle_timeout is not None:
            model_config["stream_idle_timeout"] = stream_idle_timeout
        self.llm_client = create_llm_client(model_config)
        self.base_url = base_url
        self.api_key = api_key
        self.auth_token = auth_token
        self.model = model
        self.model_format = model_format
        self.effort_level = effort_level
        self.fallback_model = fallback_model

        # 备用 LLM client（主 client 重试耗尽时切换）
        self.fallback_llm_client = None
        if fallback_model:
            fb_config = dict(model_config)
            fb_config["model"] = fallback_model
            try:
                self.fallback_llm_client = create_llm_client(fb_config)
            except Exception as e:
                logger.warning("创建 fallback LLM client 失败: %s", e)

        self.max_iterations = max_iterations
        self.enabled_toolsets = enabled_toolsets or ["core"]
        self.session_id = session_id
        # 给本会话建一个专属 env 文件，路径放进环境变量给 hook 用
        self._session_env_path = None
        self._setup_session_env_file()
        self.memory_store = memory_store
        self.memory_manager = memory_manager
        self.session_store = session_store
        # codeagent_home=None 时要解析成默认 ~/.codeAgent，
        # 否则下游拿 None 拼路径会直接 TypeError
        if codeagent_home is not None:
            self.codeAgent_home = codeagent_home
        else:
            from constants import get_codeagent_home
            self.codeAgent_home = get_codeagent_home()
        self.on_tool_call = on_tool_call
        self.on_response = on_response

        # 迭代预算（限制一条用户消息最多循环多少轮，每个会话独立）
        self.iteration_budget = IterationBudget(max_iterations)

        # 中断标志（Ctrl+C 时置 True，主循环每轮自己检查）
        self._interrupt_requested = False

        # 预算耗尽后的「遗言轮」标志：再给模型一次机会消化工具结果
        self._budget_grace_call = False
        # 遗言轮只能触发一次——不然「跑工具→遗言→
        # 又跑工具→又遗言」能无限循环
        self._grace_triggered = False

        # 系统提示词：会话开头构建一次，之后用缓存（中途改会打穿前缀缓存）
        self._system_prompt_built = system_prompt_override is not None
        # 分两层缓存（stable=整个会话不变 / context=本会话不变）
        self._stable_prompt: Optional[str] = system_prompt_override
        self._context_prompt: Optional[str] = ""
        # 自定义子代理可跳过项目 CODEAGENT.md 注入
        self.omit_project_memory = bool(omit_project_memory)

        # 对话历史（注意：不含 system prompt，system 每次单独拼在最前）
        # initial_messages 支持（fork 出的子代理继承父代理的前缀历史）
        self.conversation_history: list = list(initial_messages) if initial_messages else []

        # 上下文压缩开关与计数
        self.compression_enabled = True
        self._compression_attempts = 0
        # 完整配置 dict（压缩阈值、hooks 开关等都从这里读）
        self.config: dict = config or {}

        # === hooks（钩子）系统 ===
        self.hooks_registry = hooks_registry
        self._stop_fire_count = 0
        self._stop_hook_forced = False  # STOP hook 拦下收尾时置 True，让主循环再跑一轮

        # === 后台任务管理器 ===
        self.bg_manager = bg_manager

        # === cron（定时任务）调度器 ===
        self.cron_scheduler = cron_scheduler

        # === 团队消息总线 + 协调器 ===
        self.team_bus = team_bus
        self.team_coordinator = team_coordinator
        self.team_name = team_name

        # === idle（主动停下）标志 + 派生深度 ===
        self._idle_requested = False
        self.spawn_depth = spawn_depth

        # 上下文压缩的会话级状态（每实例一份，跨轮追踪冷却时间和触发次数）
        self._compress_session_state = CompressionSessionState()
        # 这是「本 agent 专属」的子代理完成队列——
        # 不能做成全局单例，否则同进程里多个 AIAgent（团队工人等）会把别人
        # 子代理的结果捞走。这里延迟导入是防循环依赖（delegate_tool 依赖 agent 包）。
        try:
            from tools.delegate_tool import DelegationCompletionQueue
            self._delegation_queue = DelegationCompletionQueue()
        except Exception:
            self._delegation_queue = None
        # 模型运行期间用户敲的斜杠命令先攒这里（不喂给模型），
        # 本轮对话结束后由 CLI 主循环取走执行
        self._queued_cli_commands: list = []
        # 已经注入过的记忆 id 集合（跨轮去重，防止同一条记忆反复挤占上下文）
        self._surfaced_memory_ids: set = set()
        # 按模型分别记账的用量追踪器（CLI 注入；None = 不追踪）
        self._usage_tracker = None
        # 这里刻意不清空模块级的「工具结果落盘决策表」——
        # 那张表是同进程所有 agent 共享的，__init__ 里清空会把别的正在跑的
        # agent 的决策一起抹掉，下一轮工具结果会被原样还原、重新全量落盘，
        # 破坏逐字节重放（等于打穿 prompt cache）。不用担心的理由：
        # tool_call_id 是服务商随机生成的（call_xxx）跨会话不撞车；
        # 内存也有 _OFFLOAD_DECISIONS_LIMIT 的 LRU 上限兜底。
        # 新会话要重置「摘要熔断器」（防止上个会话的失败计数污染本会话）
        reset_compact_circuit_breaker()
        # 新会话重置 LLM 观察器的熔断器和调用计数（同样防跨会话污染）
        try:
            from agent.skill_learning.llm_observer import reset_llm_observer_state
            reset_llm_observer_state()
        except Exception as e:
            logger.debug("reset_llm_observer_state 失败（fail-open）: %s", e)
        # 新会话重置缓存监控状态（防上个会话的基线数据污染本会话）
        try:
            from agent.cache_monitor import reset_cache_monitor
            reset_cache_monitor()
        except Exception as e:
            logger.debug("reset_cache_monitor 失败（fail-open）: %s", e)
        # 从 config 读「diff 文件列表」的 LRU 上限，传给缓存监控
        try:
            from agent.cache_monitor import set_diff_limit
            _diff_limit = (
                (config or {}).get("context", {}).get(
                    "max_cache_break_diff_files", 100,
                )
            )
            set_diff_limit(_diff_limit)
        except Exception as e:
            logger.debug("set_diff_limit 失败（fail-open）: %s", e)

        # === 辅助 LLM 路由器（杂活走便宜小模型）===
        self.aux_llm_router = aux_llm_router

        # === 轨迹记录 sink 接到钩子上（fail-open）===
        # 挂 6 个钩子点（LLM 调用前后 + 工具成功/失败 + 子代理开始/结束）。
        # 钩子内部自带异常保护，写盘失败只记日志。
        # 防「静默死代码」：必须在构造函数里真的调用
        # _register_trace_hooks——否则单元测试能过但生产路径一条轨迹都不出。
        self._trace_sink = trace_sink
        if trace_sink is not None and hooks_registry is not None:
            try:
                from agent.trace import _register_trace_hooks
                _register_trace_hooks(hooks_registry, trace_sink)
            except Exception as e:
                logger.warning("trace hook 注册失败（不影响主流程）: %s", e)

        # === PlanMode：计划模式状态 + 审批回调 ===
        # plan_mode=True 时下一轮切到 ["plan"] 工具集（全只读，只能调研）
        # plan_approval_callback(plan: str) -> (approved: bool, feedback: str)
        # 传 None 表示自动批准（测试/当库用的场景）
        self.plan_mode: bool = False
        self.plan_approval_callback = plan_approval_callback
        # 记住最近一次批准的计划全文——上下文压缩后
        # 由恢复模块重新注入，防止模型「失忆」不知道自己在执行什么计划
        self._last_approved_plan: str = ""
        # ask_user 桥接（CLI 用它渲染问题、读用户选择；GUI 走 HTTP）
        # None = 没桥接（快速失败，不无限等）
        self.ask_user_bridge = ask_user_bridge
        # Checkpoint：文件快照/回滚（编辑工具通过 _checkpoint_track 记录改过的文件）
        self.checkpoint_manager = checkpoint_manager
        # === 权限模式（default=正常问 / bypassPermissions=跳过审批），
        # 由 CLI 启动参数或 /permission 命令切换 ===
        self.permission_mode = permission_mode
        # 压缩后重注入用的：最近读过的文件 + 加载过的技能
        self._recent_read_files: list = []
        self._recent_skills: list = []
        # 上下文管理提示：接近上限时建议用户主动 /compact 或 /new，只提示一次
        self._context_tip_shown = False
        # 上次提醒更新 PROGRESS.md 时的 LLM 轮次号（-10^6 = 从没提醒过；
        # 首轮即满足节流条件，长任务信号挡住短会话）
        self._last_progress_reminder_turn = -10**6

        # === LLM 用量统计（给 prompt cache 记账）===
        self._llm_usage_stats = {
            "total_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_creation_tokens": 0,
        }

        # === 子代理清单（Ctrl+C 时中断要级联传给它们）===
        self._children: list = []

        # === 自动心跳桥 ===
        # 派出去干活的工人 agent 每调一次工具就自动更新任务的「最后心跳时间」
        # （主代理没配任务环境变量，这里等于空操作）
        try:
            from agent.team.auto_heartbeat import register as _register_auto_heartbeat
            _register_auto_heartbeat(self.hooks_registry)
        except Exception as e:
            logger.warning("注册 auto_heartbeat hook 失败（不影响主流程）: %s", e)

        # === 流式输出回调 ===
        # None 时走非流式（兼容老测试）；非 None 时 LLM 每吐一小段就调一次。
        # 回调签名：callback(event: dict) -> None
        # event["type"]: "content"（文本增量）| "tool_call_start" | "done"
        self._stream_callback = stream_callback

        # === 最近一次 LLM 错误的分类（prompt_too_long/stream_idle/other）===
        # 「扣留-恢复」（先藏住错误、试过恢复才认输）失败时记下分类，
        # 主循环据此细分终止原因。
        self._last_llm_error_kind = "other"
        # === 最近一次主调用的工具 schema 清单（fork 摘要前缀复用）===
        # 压缩发生在准备工具集之前的轮边界，fork 出的请求沿用上一轮的工具
        # 清单保持前缀一致（计划模式切换那一轮会失配一次，可接受）。
        self._last_tool_schemas = None
        # === 权威 token 锚点（消息条数, 服务商报的真实输入 token）===
        # 由 _record_llm_usage 更新；压缩阈值用它做「真实值+粗估值」混合计数。
        self._last_usage_anchor = None
        # === 批间工具摘要（后台生成 + 下一轮临时注入）===
        self._pending_tool_batch_summary = None  # 待注入的一句话摘要
        self._tool_summary_task = None           # 任务引用（防被垃圾回收）
        # === 条件技能动态激活的会话级去重集合 ===
        self._activated_conditional_skills = set()
        # === 对话级轻量记忆提取（增量游标 + 互斥 + 节流）===
        self._auto_extract_cursor = 0        # 已经提取到历史列表的哪个下标
        self._memory_touched_this_turn = False  # 本轮模型是否自己调过记忆写入工具
        self._auto_extract_turn_count = 0    # 回合计数（每 N 回合才提取一次）
        self._auto_extract_task = None       # 任务引用（防被垃圾回收）
        # === 记忆检索并行预取（先发起，组装消息时再等结果）===
        self._memory_prefetch_task = None
        # === 用户输入队列（模型跑时排队，工具批结束后回流给模型）===
        self._input_queue = None  # queue.Queue（cli.py 的输入线程往里灌）
        # === 流式预执行结果（流式路径产出，工具分发阶段消费）===
        self._streaming_preset_results = {}
        # === max_tokens（单次回复长度上限）升级机制 ===
        # 输出被截断（finish_reason=length）时先调大上限重试一整次，
        # 而不是急着让模型续写打断思路。整个会话复用；升级最多做一次。
        from agent.llm_retry import MaxTokensEscalator
        self._max_tokens_escalator = MaxTokensEscalator()

        # === 韧性状态：紧急压缩已触发标记 ===
        # 紧急压缩（reactive_compact）支持多次触发（冷却 60 秒 +
        # 每会话上限 5 次），次数控制收进了它自己内部，这里不再单独记。
        # 字段保留是向后兼容（有的测试会读它）。
        self._reacted: bool = False

        # === 失败重试检测 ===
        # 连续 N 次工具失败就注入一条提醒，防止模型一根筋死循环重试
        self._tool_failure_streak: int = 0
        self._last_tool_error: str = ""
        self._failure_threshold: int = 3
        # 本条用户消息内是否有过成功的工具调用（goal「踢一脚」的
        # 判据——「最近有进展、不是在空转」；与失败计数对称，成功就置 True）
        self._last_turn_had_tool_success: bool = False

        # === 任务级反思引擎 ===
        # 每次任务正常结束时后台触发：用辅助小模型从对话轨迹里提炼经验写进
        # 记忆仓库。辅助模型不可用就跳过；config["reflection"]["enabled"]=False 可关。
        self._reflection_enabled = (
            (config or {}).get("reflection", {}).get("enabled", True)
        )
        # 节流：防止连续对话起一堆反思线程烧 token
        # - 任意时刻最多 1 个反思在跑（_active_reflections）
        # - 距上次启动不足 N 轮时跳过（_last_reflection_turn + cooldown_turns）
        self._reflection_lock = __import__("threading").Lock()
        self._active_reflections = 0
        self._last_reflection_turn = -1
        self._reflection_cooldown_turns = int(
            (config or {}).get("reflection", {}).get("cooldown_turns", 3)
        )

        # === Goal/Channel/Mailbox 三件套 ===
        # goal_state：目标驱动的状态机（让模型自动一轮轮推进）；None=未启用
        # channel_inbox：MCP 服务器推送通知的收件箱
        # mailbox：队友的异步邮箱（_agent_name 是收件人）
        # 三个都既支持构造传入也支持 setter 后注入（最灵活）
        self._goal_state = goal_state
        self._channel_inbox = channel_inbox
        self._mailbox = mailbox
        self._agent_name = agent_name
        # 待注入的临时消息队列：goal continue 跨轮注入用。
        # 设计：主循环要继续 goal 时把临时消息塞这里（不进正式历史），
        # 下一轮组装消息时取出消费并清空——模型看得到，但不污染持久化
        # 的对话历史（保住落盘记录 + 前缀缓存）
        self._pending_ephemeral_messages: list = []
        # 等着激活的条件技能的文件路径（先收集、后批量处理）
        # 工具回调只负责收集；组装消息开头统一处理——一轮多个工具
        # 只扫一次技能目录，省掉重复的磁盘 IO
        self._pending_skill_paths: list = []
        # goal「踢一脚」每条用户消息最多一次
        # （防连环踢死循环——模型坚持说做完了时，第二次就放行正常收尾）
        self._nudged_this_turn: bool = False
        # 降级快照只注入一次的标志。
        # 没有辅助小模型时，主循环降级为注入记忆索引快照；为对齐旧「会话级
        # 冻结」语义，注入一次后本会话不再重复注入（免得每轮塞同一份索引）
        self._snapshot_injected: bool = False

    def cleanup(self):
        """释放这个 agent 占用的资源（由 CLI 的 RuntimeContext.shutdown 调用）。

        进程退出前要把临时文件、HTTP 连接池等收拾干净。
        幂等（调用多少次都安全）；每项清理各自 try/except，一项失败不拖累其他。

        参数：无。返回：无。
        """
        # 删掉会话专属 env 文件 + 清掉对应环境变量
        try:
            if getattr(self, "_session_env_path", None):
                self._session_env_path.unlink(missing_ok=True)
                self._session_env_path = None
            if "CODEAGENT_ENV_FILE" in os.environ:
                del os.environ["CODEAGENT_ENV_FILE"]
        except Exception as e:
            logger.warning("清理 session env 文件失败: %s", e)

        # 关闭 LLM 客户端（释放 HTTP 连接池），防进程退出前泄漏
        for client_attr in ("llm_client", "fallback_llm_client"):
            client = getattr(self, client_attr, None)
            if client is not None:
                try:
                    close_fn = getattr(client, "close", None)
                    if callable(close_fn):
                        close_fn()
                except Exception as e:
                    logger.warning("关闭 %s 失败（忽略）: %s", client_attr, e)

    def _setup_session_env_file(self):
        """会话启动时创建 .session/{session_id}.env 文件并设 CODEAGENT_ENV_FILE 环境变量。

        SessionStart 钩子跑的时候能从环境变量读到这个路径，往文件里
        写 `export K=V` 行；之后 terminal 工具执行命令会把这些变量合并进去。
        失败只打日志（fail-open）。

        参数：无（用 self.session_id）。返回：无。
        """
        try:
            from constants import session_env_file
            env_path = session_env_file(self.session_id)
            env_path.parent.mkdir(parents=True, exist_ok=True)
            if not env_path.exists():
                env_path.touch()
            os.environ["CODEAGENT_ENV_FILE"] = str(env_path)
            self._session_env_path = env_path
        except Exception as e:
            logger.warning("创建 session env file 失败: %s", e)
    def interrupt(self):
        """请求中断（CLI 的 Ctrl+C 处理器调用这个）。

        用的是「商量式」中断——不直接杀线程（硬杀可能把消息历史弄坏），
        只设个标志让主循环自己看到后退出。中断会像多米诺一样传给所有
        活跃的子代理（主对话派出去帮忙干活的分身）。

        参数：无。返回：无。
        """
        self._interrupt_requested = True
        # 级联传播给子代理
        for child in self._children:
            try:
                child.interrupt()
            except Exception as e:
                logger.warning("子 agent 中断失败: %s", e)

    def cleanup_runtime(self) -> None:
        """子代理退出时清理它遗留的运行状态。

        子代理跑完/被砍时，它自己派生的孙代理和后台任务可能还活着，
        得收尾。幂等 + fail-open（逐项清，单项失败不拖累其他）：
        - 中断级联到 children（异步孙代理线程会在自己的循环顶上协作式退出，
          不再往已死的父代理的结果队列里塞东西）
        - 有后台任务管理器就 shutdown（防御性——当前子代理默认不配 bg）
        由 delegate_tool 的 _run_child 在 finally 里调用；主代理走
        RuntimeContext.shutdown 自己的清理链。

        参数：无。返回：无。
        """
        try:
            self.interrupt()  # 含 _children 级联
        except Exception as e:
            logger.debug("cleanup_runtime 中断级联失败: %s", e)
        if getattr(self, "bg_manager", None) is not None:
            try:
                self.bg_manager.shutdown()
            except Exception as e:
                logger.debug("cleanup_runtime bg shutdown 失败: %s", e)

    def _record_llm_usage(self, response, sent_message_count: int = None) -> None:
        """记一笔 LLM 调用的 token 用量账（/usage 命令、缓存命中分析都要有账可查）。

        sent_message_count 不为 None 时，顺手记一个「权威锚点」
        ``_last_usage_anchor = (消息条数, 输入 token 数)``——输入 token 的
        口径按 usage 字段名判语义（DeepSeek 命名 → prompt_tokens 已含缓存
        直接用；Anthropic 命名 → 三项相加；详见下方注释）。压缩阈值判定
        拿它做「权威值 + 新消息粗估」的混合计数，比全程粗估准。

        参数：
            response: LLM 返回的响应对象（从它的 usage 字段取数）
            sent_message_count: 本次实际发送的消息条数；None 时不记锚点

        返回：无（直接累加到 self._llm_usage_stats）。全程 fail-open。
        """
        self._llm_usage_stats["total_calls"] += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        try:
            prompt_t = getattr(usage, "prompt_tokens", 0) or 0
            self._llm_usage_stats["total_prompt_tokens"] += prompt_t
            self._llm_usage_stats["total_completion_tokens"] += (
                getattr(usage, "completion_tokens", 0) or 0
            )
            # 前缀缓存相关字段（DeepSeek / OpenAI / Anthropic 各叫各的名，都试一遍）
            cache_read = (
                getattr(usage, "prompt_cache_hit_tokens", 0)
                or getattr(usage, "cache_read_input_tokens", 0)
                or 0
            )
            cache_creation = (
                getattr(usage, "prompt_cache_miss_tokens", 0)
                or getattr(usage, "cache_creation_input_tokens", 0)
                or 0
            )
            self._llm_usage_stats["total_cache_read_tokens"] += cache_read
            self._llm_usage_stats["total_cache_creation_tokens"] += cache_creation
            # 记权威锚点（压缩阈值混合计数用）。
            # 口径按 usage 字段名判语义——不同服务商的 prompt_tokens 含义不同：
            # - DeepSeek 命名（prompt_cache_hit_tokens）：prompt_tokens 本身
            #   已含缓存命中+未命中，直接用它（旧版三项相加 = 双倍，5 万真实
            #   token 会谎报成 10 万，过早触发有损压缩）
            # - Anthropic 命名（cache_read_input_tokens）：prompt_tokens 不含
            #   缓存部分，三项相加才是真实输入
            # - OpenAI 官方（两者皆无）：prompt_tokens 即全量
            if sent_message_count:
                if hasattr(usage, "prompt_cache_hit_tokens"):
                    anchor_tokens = prompt_t
                else:
                    anchor_tokens = prompt_t + cache_read + cache_creation
                self._last_usage_anchor = (sent_message_count, anchor_tokens)
            # 按模型分四项累计（有 tracker 才记；失败不炸）
            if getattr(self, "_usage_tracker", None) is not None:
                try:
                    self._usage_tracker.record(
                        model=str(
                            getattr(response, "model", None)
                            or self.model or "unknown"
                        ),
                        prompt=prompt_t,
                        completion=getattr(usage, "completion_tokens", 0) or 0,
                        cache_read=cache_read,
                        cache_creation=cache_creation,
                    )
                except Exception as te:
                    logger.debug("usage_tracker 记录失败（fail-open）: %s", te)
        except Exception as e:
            logger.debug("记录 LLM usage 失败（fail-open）: %s", e)

    # ------------------------------------------------------------------
    # Goal/Channel/Mailbox 的 setter + goal 持久化路径
    # ------------------------------------------------------------------
    # 设计：构造参数和 setter 都支持——构造参数用于子代理场景，
    # setter 用于 CLI 启动后按需注入（比如 /goal 命令触发后才建 GoalState）

    def set_goal_state(self, goal_state) -> None:
        """注入目标状态机（传 None = 清除）。

        参数：goal_state: GoalState 实例或 None。返回：无。
        """
        self._goal_state = goal_state

    def set_channel_inbox(self, inbox) -> None:
        """注入 MCP 推送收件箱（传 None = 清除）。

        参数：inbox: ChannelInbox 实例或 None。返回：无。
        """
        self._channel_inbox = inbox

    def set_input_queue(self, q) -> None:
        """注入用户输入队列（CLI 的输入线程往里灌用户敲的字）。

        模型正在回复时用户敲的字先排队（不打断当前回复）；一批工具
        跑完后取出，以临时消息 <queued_user_input> 的形式回流——模型下一轮
        看得到并回应。

        参数：q: queue.Queue 队列对象。返回：无。
        """
        self._input_queue = q

    def _drain_queued_input(self) -> None:
        """把输入队列里攒的用户输入捞出来，包成临时消息回流给模型。

        非阻塞取，出错也只打日志。只有主代理（派生深度为 0）
        才做——子代理没有用户交互面。多条输入合并成一条临时消息（同一轮
        消化）；队列空就什么都不做。

        参数：无。返回：无。
        """
        # getattr 防御：测试用 __new__ 造的轻量 agent 没这些属性，直接读会炸
        input_queue = getattr(self, "_input_queue", None)
        if input_queue is None or getattr(self, "spawn_depth", 0) != 0:
            return
        try:
            lines = []
            commands = []
            while True:
                try:
                    item = input_queue.get_nowait()
                except Exception:
                    break
                # 跳过不是字符串的项（CLI 输入线程用「对象哨兵」表示
                # EOF/中断，用户敲的字再怪也是字符串，不会和它撞车）
                if not isinstance(item, str):
                    continue
                # 排队输入里的斜杠命令不喂模型——分流到
                # _queued_cli_commands，让 CLI 主循环在本轮对话结束后执行
                #（不分流的话 /compact /quit 会被当普通文本吞掉）
                if item.startswith("/") and self._is_cli_command_like(item):
                    commands.append(item)
                    continue
                lines.append(item)
            if commands:
                self._queued_cli_commands.extend(commands)
                logger.info("排队 slash 命令转交 CLI 执行（%d 条）", len(commands))
            if not lines:
                return
            joined = "\n".join(l for l in lines if l and l.strip())
            if not joined.strip():
                return
            self._pending_ephemeral_messages.append({
                "role": "user",
                "content": (
                    "<queued_user_input>（模型执行期间用户发来的消息，"
                    "请在完成当前工作后回应）\n" + joined + "\n</queued_user_input>"
                ),
                "_ephemeral": True,
            })
            logger.info("排队输入回流（%d 条）", len(lines))
        except Exception as e:
            logger.debug("输入队列 drain fail-open: %s", e)

    @staticmethod
    def _is_cli_command_like(line: str) -> bool:
        """判断一行字长得像不像斜杠命令（规则和 cli.py 的启发式一致）。

        排队输入要分流——`/compact`、`/skills_list` 这种「斜杠+单词」
        形态算命令交给 CLI；`/etc/passwd 是什么`（含路径分隔符等非命令
        字符）不算，还是当普通消息喂给模型。

        参数：
            line: 用户敲的一行字

        返回：True = 像 /word 形态的命令；False = 不像。
        """
        name = line.split()[0][1:] if line.split() else ""
        return bool(
            name
            and name[0].isalpha()
            and all(c.isalnum() or c in "_-" for c in name)
        )

    def _recent_active_tools(self, lookback: int = 8, limit: int = 3) -> list:
        """列出最近用过哪几个工具（给记忆检索当降噪信号）。

        记忆检索时知道「正在干哪种活」能滤掉不相关的记忆。
        从对话历史尾部倒着扫 assistant 消息里的工具调用，去重后收集，
        最多 limit 个；倒扫保住「最近」，最后插回头部恢复时间正序。

        参数：
            lookback: 往回看多少条消息（默认 8）
            limit: 最多收集几个工具名（默认 3）

        返回：工具名列表（时间正序）。
        """
        names: list = []
        for m in reversed((self.conversation_history or [])[-lookback:]):
            if not isinstance(m, dict) or m.get("role") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                fn = ((tc.get("function") or {}) or {}).get("name") or ""
                if fn and fn not in names:
                    names.insert(0, fn)
            if len(names) >= limit:
                break
        return names[:limit]

    def set_mailbox(self, mailbox, agent_name: str = None) -> None:
        """注入团队邮箱。agent_name 传空就保留现有名字。

        参数：
            mailbox: Mailbox 实例（传 None = 清除）
            agent_name: 自己的 agent 名（收件人）；None/空 = 不改

        返回：无。
        """
        self._mailbox = mailbox
        if agent_name:
            self._agent_name = agent_name

    def set_usage_tracker(self, tracker) -> None:
        """注入按模型记账的用量追踪器（_record_llm_usage 里消费）。

        参数：tracker: 用量追踪器实例。返回：无。
        """
        self._usage_tracker = tracker

    def _goal_state_path(self):
        """goal 状态的落盘路径：~/.codeAgent/.goal/current.json。

        参数：无。返回：pathlib.Path 路径对象。
        """
        from pathlib import Path
        return Path(self.codeAgent_home) / ".goal" / "current.json"

    def _check_all_goal_tasks_done(self) -> bool:
        """goal 关联的任务是否全部完成了。

        真的查任务库：
        - 没有 goal_state → False
        - 否则调 agent.goal 的 check_all_tasks_done 查任务状态

        参数：无。返回：True = 全部 completed。查询出错按 False 算（保守）。
        """
        if self._goal_state is None:
            return False
        from agent.goal import check_all_tasks_done
        try:
            return check_all_tasks_done(self._goal_state)
        except Exception as e:
            logger.warning("_check_all_goal_tasks_done 查询失败（fail-open False）: %s", e)
            return False

    def _extract_turn_tokens(self, response) -> int:
        """从响应的 usage 字段里取本轮花了多少 token（输入 + 输出）。

        goal 的 token 预算要在每轮结束后累加，数据就从这取。
        响应没有 usage 字段就返回 0（不炸）。

        参数：
            response: LLM 响应对象（可以是 None）

        返回：本轮 prompt + completion 的 token 总数（整数）。
        """
        if response is None:
            return 0
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0
        try:
            prompt = getattr(usage, "prompt_tokens", 0) or 0
            completion = getattr(usage, "completion_tokens", 0) or 0
            return int(prompt) + int(completion)
        except Exception:
            return 0

    @staticmethod
    def _extract_cache_read(usage) -> int:
        """从 usage 里取「前缀缓存命中读的 token 数」。

        各家服务商字段名不一样——DeepSeek 叫 prompt_cache_hit_tokens，
        Anthropic 叫 cache_read_input_tokens。流式路径合成的 usage 对象两个
        字段都塞了值（见 _call_llm_streaming 末尾），所以这里用 or 短路，
        哪个非零用哪个。dict 和对象两种形态都兼容，出错返回 0。

        参数：
            usage: usage 对象或 dict（可能是 None/空）

        返回：缓存读 token 数（整数，出错为 0）。
        """
        if not usage:
            return 0
        try:
            if isinstance(usage, dict):
                return (
                    usage.get("prompt_cache_hit_tokens", 0)
                    or usage.get("cache_read_input_tokens", 0)
                    or 0
                )
            return (
                getattr(usage, "prompt_cache_hit_tokens", 0)
                or getattr(usage, "cache_read_input_tokens", 0)
                or 0
            )
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # 流式调 LLM（边生成边吐字）
    # ------------------------------------------------------------------

    async def _call_llm_streaming(self, *, messages, tools):
        """流式调用 LLM：每收到一小段就调 stream_callback 报告一次。

        流式让用户边生成边看到字，不用干等。流式失败时自动退回
        非流式重试（带备用客户端）。返回值和非流式路径完全同构
        （用 SimpleNamespace 拼出 OpenAI 响应的形状），下游的用量记账 /
        hook / 工具调用处理代码一行都不用改。

        注意：这个方法是「async 函数返回 response 对象」，不是 async 生成器。
        流式过程通过 stream_callback 回调报告，最终结果用 return 返回。

        stream_callback 会收到的事件：
            {"type": "content", "delta": str, "accumulated": str}  # 文本增量
            {"type": "tool_call_start", "name": str, "id": str}    # 工具调用开始
            {"type": "done", "finish_reason": str}                  # 流结束

        参数：
            messages: 发给 LLM 的消息列表
            tools: 工具 schema 列表（没有工具传 None）

        返回：拼装好的 OpenAI 兼容响应对象（SimpleNamespace）。
        """
        from types import SimpleNamespace
        full_content = ""
        tool_call_buffers: dict[int, dict] = {}  # idx → {id, name, arguments}
        final_usage = None
        finish_reason = "stop"
        reasoning_content = None   # DeepSeek 的思考内容（下次带工具调用回传时要带上）
        thinking_signature = None

        # === 流式并发执行（只读的安全工具趁模型还在吐字先跑起来）===
        # 开关在 config 的 agent.streaming_tool_execution（默认关）。预执行
        # 结果存 _streaming_preset_results，工具分发阶段按调用 id 取用、跳过重复执行。
        _executor = None
        if (self.config or {}).get("agent", {}).get(
            "streaming_tool_execution", False,
        ):
            try:
                from agent.streaming_executor import StreamingToolExecutor
                _executor = StreamingToolExecutor(self)
            except Exception as e:
                logger.debug("流式执行器构造失败（退正常路径）: %s", e)
        _last_seen_idx = None

        try:
            # 从 config 读 max_tokens（用户可在 settings.json 的 llm 块配
            # "max_tokens": 8192）。不配就不传，让 API 用默认值——换模型不用改代码
            _extra = {}
            _cfg_mt = (
                (self.config or {}).get("model", {}).get("max_tokens")
                or (self.config or {}).get("llm", {}).get("max_tokens")
            )
            if _cfg_mt:
                _extra["max_tokens"] = _cfg_mt
            async for delta in self.llm_client.chat_completions_stream(
                messages, tools=tools, **_extra,
            ):
                # 内容流式
                delta_text = delta.get("content") or ""
                if delta_text:
                    full_content += delta_text
                    if self._stream_callback is not None:
                        try:
                            self._stream_callback({
                                "type": "content",
                                "delta": delta_text,
                                "accumulated": full_content,
                            })
                        except Exception as cb_err:
                            logger.warning(
                                "stream_callback(content) 异常（忽略）: %s", cb_err
                            )

                # 工具调用增量累积
                for tc in delta.get("tool_calls") or []:
                    idx = getattr(tc, "index", 0)
                    # 出现新的 index 说明上一个工具的参数已经拼完整了，
                    # 安全工具立刻预执行（和模型继续吐字的时间重叠，省等待）
                    if (
                        _executor is not None
                        and _last_seen_idx is not None
                        and idx != _last_seen_idx
                        and _last_seen_idx in tool_call_buffers
                    ):
                        _executor.complete(
                            _last_seen_idx, tool_call_buffers[_last_seen_idx],
                        )
                    _last_seen_idx = idx
                    buf = tool_call_buffers.setdefault(
                        idx, {"id": "", "name": "", "arguments": ""}
                    )
                    tc_id = getattr(tc, "id", None)
                    if tc_id:
                        buf["id"] = tc_id
                    func = getattr(tc, "function", None)
                    if func is not None:
                        fname = getattr(func, "name", None)
                        if fname:
                            buf["name"] = fname
                        fargs = getattr(func, "arguments", None)
                        if fargs:
                            buf["arguments"] += fargs
                    # 第一次拿到 name 时通知 callback
                    if buf["name"] and not buf.get("_notified"):
                        buf["_notified"] = True
                        if self._stream_callback is not None:
                            try:
                                self._stream_callback({
                                    "type": "tool_call_start",
                                    "name": buf["name"],
                                    "id": buf["id"],
                                })
                            except Exception as cb_err:
                                logger.warning(
                                    "stream_callback(tool_call_start) 异常: %s",
                                    cb_err,
                                )

                # 最后一个数据块里带 finish_reason / usage / 思考内容
                if delta.get("finish_reason"):
                    finish_reason = delta["finish_reason"]
                if delta.get("usage"):
                    final_usage = delta["usage"]
                # DeepSeek 思考内容提取（后续带工具调用的请求要回传）
                if delta.get("reasoning_content"):
                    reasoning_content = delta["reasoning_content"]
                    # 思考流也通知回调（CLI 画暗色思考框用）。加法式：
                    # 没回调/回调不认识该类型时零行为变化。
                    if self._stream_callback is not None:
                        try:
                            self._stream_callback({
                                "type": "reasoning",
                                "delta": delta["reasoning_content"],
                            })
                        except Exception as cb_err:
                            logger.warning(
                                "stream_callback(reasoning) 异常（忽略）: %s",
                                cb_err,
                            )
                if delta.get("thinking_signature"):
                    thinking_signature = delta["thinking_signature"]
        except Exception as stream_err:
            # 流式失败：退回非流式重试（带备用客户端）
            logger.warning(
                "流式调用失败，fallback 到非流式重试: %s", stream_err
            )
            # 流出错 → 等预执行任务跑完但扔掉结果（防留僵尸任务）
            if _executor is not None:
                await _executor.drain()
            # 显式扔掉已累积的半截状态（防御性「墓碑」清理，出错也放行）
            try:
                self._discard_partial_stream_state()
            except Exception:
                pass
            from agent.llm_retry import call_with_retry
            response = await call_with_retry(
                self.llm_client,
                messages,
                tools=tools,
                fallback_llm_client=self.fallback_llm_client,
                config=self.config,
                heartbeat_cb=self._llm_retry_heartbeat,  # 长退避心跳
            )
            # 流式回调已经错过，但至少把完整内容回放给 callback
            choice_msg = response.choices[0].message
            if choice_msg.content and self._stream_callback is not None:
                try:
                    self._stream_callback({
                        "type": "content",
                        "delta": choice_msg.content,
                        "accumulated": choice_msg.content,
                    })
                except Exception:
                    pass
            return response

        # 流正常结束 → 补完最后一个工具的完整化 + 收集预执行结果
        if _executor is not None:
            try:
                if _last_seen_idx is not None and _last_seen_idx in tool_call_buffers:
                    _executor.complete(
                        _last_seen_idx, tool_call_buffers[_last_seen_idx],
                    )
                self._streaming_preset_results = await _executor.collect()
            except Exception as e:
                logger.debug("流式预执行 collect 失败（弃用）: %s", e)
                self._streaming_preset_results = {}

        # 合成 tool_calls 列表（按 idx 排序，过滤掉没 name 的）
        tool_calls_out = []
        for idx in sorted(tool_call_buffers.keys()):
            buf = tool_call_buffers[idx]
            if not buf["name"]:
                continue
            tool_calls_out.append(SimpleNamespace(
                id=buf["id"],
                type="function",
                function=SimpleNamespace(
                    name=buf["name"],
                    arguments=buf["arguments"] or "{}",
                ),
            ))

        # === max_tokens 截断的「调大上限重试」 ===
        # finish_reason=length 说明输出被单次回复长度上限掐断了。
        # DeepSeek-reasoner 还有一种隐蔽截断：纯思考（正文空、思考有值）——
        # 思考把长度额度用光了，正文没地方写，但 finish_reason 可能还是 "stop"。
        # 策略：先调大 max_tokens 整个重试一次（走非流式，避免把半截正文重复发
        # 一遍）；调大后还是空才认输，交给主循环处理。
        is_pure_thinking = (
            not full_content and not tool_calls_out and bool(reasoning_content)
        )
        new_max = None
        if finish_reason == "length" or is_pure_thinking:
            trigger_reason = "纯 thinking（content 空）" if is_pure_thinking else "finish_reason=length"
            new_max = self._try_escalate_max_tokens(trigger_reason)
        # 拿到新上限才重试；None = 不该升级/已升过级，沿用截断响应
        if new_max is not None:
            try:
                from agent.llm_retry import call_with_retry
                retried = await call_with_retry(
                    self.llm_client,
                    messages,
                    tools=tools,
                    fallback_llm_client=self.fallback_llm_client,
                    max_tokens=new_max,
                    config=self.config,
                    heartbeat_cb=self._llm_retry_heartbeat,  # 长退避心跳
                )
                retried_choice = retried.choices[0]
                retried_msg = retried_choice.message
                # 用重试结果整体覆盖（重试拿到的是完整响应）。
                # 不能只在「重试有 tool_calls」时才覆盖，
                # 重试结果没有工具调用时会把截断那次的半截 tool_calls 残留进
                # 最终响应——必须无条件清空。
                finish_reason = (
                    getattr(retried_choice, "finish_reason", None) or "stop"
                )
                full_content = retried_msg.content or ""
                tool_calls_out = list(
                    getattr(retried_msg, "tool_calls", None) or []
                )
                # 把重试结果回放给回调（和 fallback 路径同款做法）
                if retried_msg.content and self._stream_callback is not None:
                    try:
                        self._stream_callback({
                            "type": "content",
                            "delta": retried_msg.content,
                            "accumulated": retried_msg.content,
                        })
                    except Exception:
                        pass
                # 更新 usage：截断那次 + 升级重试这次都真实花过钱，两边加总
                # （直接覆盖会漏记截断那次的花费）
                final_usage = self._merge_usage_tokens(final_usage, retried)
            except Exception as esc_err:
                logger.warning(
                    "max_tokens 升级重试失败（沿用截断响应）: %s", esc_err
                )

        # 通知回调：流结束了
        if self._stream_callback is not None:
            try:
                self._stream_callback({
                    "type": "done",
                    "finish_reason": finish_reason,
                })
            except Exception:
                pass

        # 拼一个 OpenAI 兼容的响应对象（让记账 / hook 等下游代码不用改）
        message = SimpleNamespace(
            content=full_content if full_content else None,
            tool_calls=tool_calls_out if tool_calls_out else None,
            reasoning_content=reasoning_content,
            thinking_signature=thinking_signature,
        )
        usage_ns = None
        if final_usage is not None:
            usage_ns = SimpleNamespace(
                prompt_tokens=final_usage.get("prompt_tokens", 0),
                completion_tokens=final_usage.get("completion_tokens", 0),
                prompt_cache_hit_tokens=final_usage.get("cache_read", 0),
                cache_read_input_tokens=final_usage.get("cache_read", 0),
                prompt_cache_miss_tokens=final_usage.get("cache_creation", 0),
                cache_creation_input_tokens=final_usage.get("cache_creation", 0),
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=message,
                finish_reason=finish_reason,
            )],
            usage=usage_ns,
        )

    @property
    def llm_usage_stats(self) -> dict:
        """用量统计的只读副本，给 /usage 命令展示用。

        参数：无。返回：统计 dict 的浅拷贝（改它不影响内部账本）。
        """
        return dict(self._llm_usage_stats)

    def _get_system_prompt(self) -> str:
        """拿系统提示词。第一次调用时现场构建，之后一直返回缓存的那份。

        为什么缓存？LLM 服务商有「前缀缓存」——请求开头不变，
        后面就能按折扣价算。中途改系统提示词等于每次都换开头，费用翻倍。

        分两层缓存 + 一层不缓存：
        - stable（稳定层）：跨会话都不变（身份、指导），几乎 100% 命中前缀缓存
        - context（会话层）：本会话内不变（记忆/技能索引/CODEAGENT.md）
        - volatile（易变层）：每轮都可能变（提醒类），不进缓存

        参数：无。返回：stable + context 拼接成的完整系统提示词字符串。
        """
        if not self._system_prompt_built:
            from agent.prompt_builder import build_system_prompt_layers
            # C6：输出风格解析（出错只打日志；结果拼进 context 层）
            style_text = ""
            try:
                from agent.output_styles import (
                    resolve_output_style, render_style_section,
                )
                try:
                    from agent.workspace_context import get_workspace_cwd
                    _cwd = get_workspace_cwd()
                except Exception:
                    _cwd = os.getcwd()
                _style = resolve_output_style(
                    self.config, _cwd, self.codeAgent_home,
                )
                if _style is not None:
                    style_text = render_style_section(_style)
            except Exception as e:
                logger.debug("输出风格解析失败（fail-open）: %s", e)
            layers = build_system_prompt_layers(
                memory_store=self.memory_store,
                memory_manager=self.memory_manager,
                enabled_toolsets=self.enabled_toolsets,
                omit_project_memory=self.omit_project_memory,
                output_style_text=style_text,
            )
            # stable + context 两层缓存；volatile 层每次现取
            self._stable_prompt = layers.stable
            self._context_prompt = layers.context
            self._system_prompt_built = True
        return "\n\n".join(p for p in (self._stable_prompt, self._context_prompt) if p)

    def _get_volatile_prompt(self) -> str:
        """每轮现算的易变部分（05）。

        目前是空的（任务状态模型自己用工具查，没有系统级提醒）。
        留这个入口方便以后扩展；它不进 stable/context 缓存，直接拼在
        系统提示词末尾。

        参数：无。返回：易变段文本（当前恒为空串）。
        """
        return ""

    def invalidate_system_prompt(self):
        """把缓存的系统提示词作废，下次调用时重新构建。

        警告：这会让前缀缓存失效、成本变高，只在上下文压缩这种非做不可的
        场景用。05 优化：压缩后只重建 context 层（stable 不动，稳定段的
        前缀缓存照样命中）。

        参数：无。返回：无。
        """
        self._context_prompt = None
        self._system_prompt_built = False
        # _stable_prompt 故意保留（整个会话内 stable 理论上永不变）

    async def run_conversation(
        self, user_message: str, cancel_event=None,
    ) -> str:
        """处理一条用户消息，返回助手的最终回答。整个系统的心脏就是这个方法。

        大白话流程：消息进来 → 先做准备工作（跑钩子、收外部消息、查记忆）→
        进主循环反复「组装消息 → 压缩 → 调 LLM → 执行模型要的工具」→
        模型不再要工具了就收尾返回。cancel_event（取消旗子）：父代理一举旗，
        循环在下一轮开头立刻退出，
        并把已完成的部分结果带回去。

        参数：
            user_message: 用户发来的消息文本
            cancel_event: 可选的 threading.Event（线程安全的开关旗）；
                None = 不检查取消（向后兼容）

        返回：字符串——助手的最终回答（或中断/预算耗尽的兜底说明）。

        主循环结构（自顶向下读）：
            循环前：钩子 → 收外部消息 → 开场记忆检索 → 用户消息入历史
            循环内：组装 messages → 压缩 → 工具集/警告/钩子 → 调 LLM → 分发工具调用
            循环后：兜底响应（预算耗尽/中断）

        具体细节都在各个 _xxx 辅助方法里。
        """
        # === 每次调用都重置「主动停下」标志 ===
        # 同一个 agent 实例在自主生命周期的多个工作周期里会复用，
        # 上一次的停下请求不能漏到下一次。
        self._idle_requested = False
        # 清掉上一回合残留的中断标志——上次
        # Ctrl+C 异常退出时旗子已插但循环没消费，不清的话用户下一条消息
        # 第一轮就被吞（直接回「[已被用户中断]」）。
        self._interrupt_requested = False
        # 每条用户消息重置迭代预算：预算只管「这一条消息」的循环轮数。
        # 不重置的话，长会话里多条消息累计消耗，中途突然耗尽就静默断线
        # （工具调完预算没了直接 break，没有任何提示）。
        self.iteration_budget.reset()
        # 每条用户消息独立的「遗言轮」机会（_grace_triggered 防同一条消息内反复触发）
        self._grace_triggered = False
        self._budget_grace_call = False
        # goal 踢一脚标志 + 工具成功标志每条消息重置
        #（踢一脚一条消息最多一次；「最近有进展」从本条消息重新算）
        self._nudged_this_turn = False
        self._last_turn_had_tool_success = False
        # 上一条消息异常退出（预算耗尽/中断）时已入队的
        # 技能激活路径，在这里处理掉而不是扔掉——激活通知进临时队列，
        # 本条消息组装时透出；有去重集合保证不会重复激活
        flush_skill_activations(self)
        # 记下本轮历史的起点下标（轮末的行为学习只看本轮轨迹）
        self._sl_turn_start = len(self.conversation_history)

        # ---------- 循环前准备 ----------
        # 钩子链挪出事件循环线程跑（慢钩子不会冻住流式输出）
        user_message = await asyncio.to_thread(
            self._run_prompt_submit_hook, user_message,
        )
        # 这里故意不收消息——每轮 while 里都收，
        # 不然多轮工具调用中途新到的消息挤不进模型上下文

        # 开场不做记忆注入——记忆统一走按需检索的临时注入
        # （见下面 _pending_ephemeral_messages 块），用户消息原样进历史
        # （保护前缀缓存 + 不污染持久化）
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        # === 按需检索记忆并注入（只有主代理做，子代理不做）===
        # 记忆走临时消息注入（保护前缀缓存），不塞进系统提示词。
        # 每条用户消息一次：放在用户输入刚进主循环的地方（不在工具循环里反复做）。
        # 检索 query 用「用户消息 + 上下文签名」增强——长任务后期用户常说
        # 「继续/好」，光靠这几个字查记忆没有信号。细节见 _kick_memory_prefetch。
        self._kick_memory_prefetch(user_message)

        system_prompt = self._get_system_prompt()

        # 延迟导入（用到才 import），避免模块间互相 import 死锁
        from model_tools import get_tool_definitions, handle_function_call

        # ---------- 主循环 ----------
        api_call_count = 0
        turn_exit_reason = "normal"
        goal_decision = None  # 本轮目标状态机的决策（正常返回路径记轨迹用）

        while (
            api_call_count < self.max_iterations
            and self.iteration_budget.remaining > 0
        ) or self._budget_grace_call:
            # 中断检查（商量式：看标志，不是硬杀）
            if self._interrupt_requested:
                turn_exit_reason = "interrupted_by_user"
                self._interrupt_requested = False  # 消费掉标志，别影响下条消息
                break

            # === 取消旗检查（父代理触发）===
            # 每轮开头查一次（不是每条消息查）——开销小，也足够及时。
            # 父代理一举旗，本子代理优雅
            # 退出，把已完成的 assistant 消息打包成部分结果带回去
            if cancel_event is not None and cancel_event.is_set():
                logger.info(
                    "Task K: 子代理被 cancel_event 中断，返回 partial result"
                )
                return self._extract_partial_result()

            # === 防休眠——goal 循环或后台任务跑着时别让电脑睡 ===
            # 每轮开头评估忙不忙，只在忙闲切换时
            # 真正调 acquire/release（失败也放行，防休眠失手绝不影响主对话）
            try:
                self._update_prevent_sleep()
            except Exception as e:
                logger.debug("prevent_sleep fail-open: %s", e)

            # 消耗预算（遗言轮不消耗预算）
            consumed_this_iter = False
            if not self._budget_grace_call:
                if not self.iteration_budget.consume():
                    turn_exit_reason = LoopExitReason.BUDGET_EXHAUSTED
                    break
                consumed_this_iter = True
            else:
                # 遗言轮进入时立刻清标志，保证只跑一次。
                # 不清的话会「跑工具→遗言→又跑工具」无限循环
                self._budget_grace_call = False

            # 组装 messages + 注入 bg/cron/team/plan_mode 等临时消息
            # 每轮都重新收一遍外部消息——不能只在循环前
            # 收一次，多轮工具调用中途新到的后台/定时/团队消息挤不进模型上下文
            injected = self._drain_injected_messages()
            messages = self._assemble_turn_messages(system_prompt, injected)

            # 上下文压缩（快到 token 上限时触发，可能会重建系统提示词）
            messages, system_prompt, compacted_this_turn = (
                await self._run_context_compression(messages, system_prompt)
            )

            # 剥掉内部字段（_timestamp 之类的记账标记）——必须放在压缩之后、
            # 发给 LLM 之前。剥早了压缩层读不到 _timestamp，「按时间清理旧
            # 工具结果」功能会失效；放这里压缩层能读到时间
            # 标记（按时清理真正生效），发给 LLM 的消息又不带这些内部字段
            # （保护前缀缓存）。重复剥也无害（每次都建新 dict）。
            # 先存一份 strip 前快照：下面「发送前预检」若触发 force 压缩，
            # 输入必须用这份（_ephemeral 还在）——用剥过的消息跑压缩，
            # 历史同步过滤器的 not m.get("_ephemeral") 全部失守，
            # 本轮临时注入会被焊进正式历史随边界落库。
            pre_strip = messages
            messages = strip_internal_fields(messages)

            # 等记忆检索预取的结果（并行窗口覆盖收消息/组装/压缩/剥字段
            # 全程；结果直接追加到本轮 messages——天然是临时消息，不进正式历史）
            messages = await self._consume_memory_prefetch(messages)

            # 刷新工具集（计划模式切换）+ 失败重试警告 + PRE_LLM_CALL 钩子
            tool_schemas = await self._prepare_toolset_and_injections(messages)
            # 记下最新工具清单，轮边界的 fork 摘要靠它保持前缀一致
            self._last_tool_schemas = tool_schemas

            # strip 之后才追加的尾部（记忆预取 + 重试警告/钩子追加的提醒，
            # _ephemeral 天然完好）——快照必须取在上面两步**之后**：它们只会
            # 往尾部原地追加，取早了这些追加就落在快照外，force 预检触发时
            # 被 force_in 替换掉、当轮静默丢失。同时要在配对修复之前取：
            # 修复可能增删头部消息改变列表长度，之后按下标切就不准了。
            # strip 只换 dict 不变长度、上面几步只追加尾部，
            # len(pre_strip) 对齐依然成立。
            post_strip_appendix = messages[len(pre_strip):]

            # 防孤儿兜底：发送前修复工具调用配对。「孤儿」= 有工具结果却找不到
            # 对应的工具调用（压缩边界/流式断连/工具异常都可能造出）——Anthropic
            # API 见到就直接 400 报错。_fix_tool_call_pairs 补一个假结果，或
            # 删掉无主的结果。
            try:
                from agent.context_compressor import _fix_tool_call_pairs
                messages = _fix_tool_call_pairs(messages)
            except Exception as pair_err:
                logger.warning("发送前配对修复失败（忽略）: %s", pair_err)

            # === 发送前窗口预检（最后防线） ===
            # 循环顶部压缩之后又追加了记忆注入等增量；估算失准（尤其恢复后
            # 第一轮没有锚点）时，这里拦住「直接顶到 PTL → reactive 只留
            # 5 条」的灾难通道：超窗口 90% 且本回合还没压过，就再跑一次
            # 优雅压缩管线（force 绕过冷却但守熔断）。还超就照发——
            # PTL 救火机制保持最终兜底。预检自身异常绝不挡发送。
            # _force_out：force 压缩的「未剥输出」暂存——若这回合真跑了
            # force，下面调 LLM 时 reactive 的 flag 完好输入要用它（消息集
            # 就是最新真身）；None = 这回合没跑 force。
            _force_out = None
            try:
                from agent.context_pipeline import needs_pre_send_compaction
                if (not compacted_this_turn
                        and needs_pre_send_compaction(
                            messages, self._last_usage_anchor, self.model)):
                    logger.warning(
                        "发送前预检：估算超窗口 90%，先跑一次 force 压缩再发",
                    )
                    # force 压缩的输入不能用 strip 后的 messages——_ephemeral
                    # 已被剥掉，压缩内部「同步回正式历史」的过滤器会把本轮
                    # 临时注入（task_notification 等）当正式消息焊进历史。
                    # 拼法：头段用 strip 前快照（flag 完好）+ 尾段用 strip 后
                    # 追加的临时消息（记忆预取/重试警告/钩子提醒，flag 也完好）。
                    # 配对修复若动过头部，压缩管线收尾自己会再修一遍配对，无碍。
                    force_in = pre_strip + post_strip_appendix
                    messages, system_prompt, compacted_this_turn = (
                        await self._run_context_compression(
                            force_in, system_prompt, force=True,
                        )
                    )
                    _force_out = messages
                    messages = strip_internal_fields(messages)
            except Exception as guard_err:
                logger.warning("发送前预检失败（fail-open 照发）: %s", guard_err)

            # 调 LLM（含 max_tokens 升级 + 输入超长时的紧急压缩）
            # 本方法是异步的
            # reactive 紧急压缩要 flag 完好版输入（理由见函数 docstring）；
            # force 预检这回合跑过的话，用它的未剥输出（消息集就是最新真身）
            _flagged = (
                _force_out if _force_out is not None
                else pre_strip + post_strip_appendix
            )
            response = await self._call_llm_with_escalation(
                messages, tool_schemas, system_prompt,
                flagged_messages=_flagged,
            )
            if response is self._REACTIVE_RETRY:
                # 紧急压缩后的重试是「恢复」不是「新一轮」——
                # 本轮没有成功的 LLM 产出，预算要退回（遗言轮本来就没消费，
                # 不退，防止多退）。不退的话连续超长白扣预算会提前「预算耗尽」；
                # 失控风险由紧急压缩自身的冷却 + 次数上限兜底，退预算不会造
                # 无限循环。
                if consumed_this_iter:
                    self.iteration_budget.refund()
                system_prompt = self._get_system_prompt()
                continue  # 紧急压缩已改写历史，重试本轮
            if response is None:
                # 按之前记下的错误分类，细分终止原因
                if self._last_llm_error_kind == "prompt_too_long":
                    turn_exit_reason = LoopExitReason.PROMPT_TOO_LONG
                elif self._last_llm_error_kind == "stream_idle":
                    turn_exit_reason = LoopExitReason.STREAM_IDLE
                else:
                    turn_exit_reason = LoopExitReason.MODEL_ERROR  # 错误信息已写回历史
                break

            # === 调大上限后仍被截断 → 续写恢复（最多 3 次，恢复过程不进历史）===
            response = await self._recover_output_truncation(
                response, messages, tool_schemas,
            )

            api_call_count += 1
            # 记 LLM 用量账（前缀缓存记账）
            # 附带消息条数 → 记权威锚点（压缩阈值混合计数用）
            self._record_llm_usage(response, sent_message_count=len(messages))

            # === POST_LLM_CALL 钩子（LLM 返回后、处理工具调用前）===
            response = self._run_post_llm_call_hook(response)

            # 同步累加压缩会话状态的轮次——L4 层的冷却
            # 计时就靠这个值，漏加会导致冷却判断错乱
            self._compress_session_state.increment_turn()

            assistant_msg = response.choices[0].message

            # 分支：模型要调工具 → 分发执行；不要 → 这就是最终回答
            if assistant_msg.tool_calls:
                # 工具分发是异步的
                should_continue = await self._dispatch_tool_calls(
                    assistant_msg, handle_function_call,
                )
                if not should_continue:
                    turn_exit_reason = LoopExitReason.IDLE_REQUESTED
                    break  # 工具要求停下
                # 工具跑完发现预算没了，就开一轮遗言——
                # 让下轮 LLM 至少能看到工具结果再收尾。
                # 模型刚调完工具还没消化
                # 结果就因预算耗尽退出，体验很差。
                # 配套：_grace_triggered 保证整条消息只触发一次（防「跑工具→
                # 遗言→又跑工具」无限循环）
                if self.iteration_budget.remaining <= 0 and not self._grace_triggered:
                    self._budget_grace_call = True
                    self._grace_triggered = True
                    logger.info(
                        "迭代预算耗尽，触发 grace call 让 LLM 看到本轮工具结果再结束"
                    )
                # 回到循环顶，让下轮 LLM 看到工具结果
                continue

            # 没有 tool_calls = 最终回答，走收尾
            final_content = await self._finalize_response(assistant_msg, user_message)
            if self._stop_hook_forced:
                # STOP 钩子拦下了收尾（注入了强制消息），跳回 while 让模型再跑一轮
                continue

            # === goal continue（目标驱动的自动多轮）===
            # 设计：
            # - 只在 goal 状态机 active 且本轮有 LLM 响应时评估
            # - 决策 continue → 把 <continue_goal> 临时消息塞进待注入队列
            #   （**不进正式历史**，保护持久化 + 前缀缓存）
            # - 下一轮组装消息时消费这个队列、追加到 messages（模型看得到）
            # - 决策 pause/complete → 正常返回收尾
            # - 缓存保护：不动系统提示词，只加 user 消息
            if self._goal_state is not None and self._goal_state.status == "active":
                # === 预算还没花完且最近有成功工具调用 → 踢一脚让它继续 ===
                # 场景：模型修了 14/20 个文件就宣布全做完了（判定全完成 → 会走
                # 完成收尾退出），可预算明明还剩很多 → 踢它一脚继续干或验证，
                # 别过早收摊。
                # 必须在状态机评估**之前**查：一旦走了完成路径，状态就变成
                # "completed"，踢一脚的前置条件（active）永远过不去了。
                # 「全部任务完成」是「本轮评估即将收尾退出」的信号——普通继续
                # 轮不触发（continue_goal 机制已覆盖，避免重复注入）。
                # _nudged_this_turn 每条用户消息最多踢一次：模型坚持说做完了，
                # 第二次就放行让它正常收尾（防连环踢死循环）。
                if (not self._nudged_this_turn
                        and self._goal_state.should_nudge(
                            recent_tool_success=self._last_turn_had_tool_success,
                        )
                        and self._check_all_goal_tasks_done()):
                    self._nudged_this_turn = True
                    self._pending_ephemeral_messages.append({
                        "role": "user",
                        "content": (
                            "[goal nudge] 预算尚有余量"
                            f"（已用 {self._goal_state.token_budget}/"
                            f"{self._goal_state.token_budget_limit}），"
                            "且最近一轮仍有进展。请继续推进目标；"
                            "若确已完成，请验证关键结果后明确说明完成依据。"
                        ),
                        "_ephemeral": True,
                    })
                    logger.info(
                        "goal nudge（预算未用完不收尾）: budget=%d/%d, iteration=%d",
                        self._goal_state.token_budget,
                        self._goal_state.token_budget_limit,
                        self._goal_state.iteration_count,
                    )
                    # 被踢的这轮不进状态机评估（迭代数/token 不累加，状态保持
                    # active 原样，交下一轮重新评估）
                    continue
                # 取本轮 token 花费（从响应的 usage 字段）
                turn_tokens = self._extract_turn_tokens(response)
                decision = self._goal_state.evaluate_after_turn(
                    tokens_used=turn_tokens,
                    all_tasks_done=self._check_all_goal_tasks_done(),
                )
                # goal 状态落盘
                self._goal_state.save(self._goal_state_path())

                if decision == "continue":
                    # 注入临时 user 消息驱动下一轮（不进正式历史）
                    cont_msg = _build_goal_continue_message(self._goal_state)
                    if cont_msg is not None:
                        self._pending_ephemeral_messages.append(cont_msg)
                    logger.info(
                        "goal continue: iteration=%d, tokens=%d",
                        self._goal_state.iteration_count,
                        self._goal_state.token_budget,
                    )
                    continue
                # pause / complete / fail → 正常返回最终回答（不走循环退出兜底）
                goal_decision = decision  # 记轨迹用
                logger.info(
                    "goal %s: reason=%s",
                    decision, self._goal_state.pause_reason,
                )

            # === 轮末行为学习观察 + 累积够了就演化成技能 ===
            # fail-open：学习链路任何异常只打 debug 日志，绝不影响主对话返回
            # 观察二件已拆到 skill_learning/turn_observer，主循环体内惰性导入
            from agent.skill_learning.turn_observer import maybe_skill_learning
            await maybe_skill_learning(self, user_message)

            # 对话级轻量记忆提取（发出去就不管，失败也放行）
            self._maybe_auto_extract()

            # 正常返回路径记轨迹（有 goal 决策时用对应的退出原因）
            self._emit_loop_exit_trace(
                {
                    "pause": LoopExitReason.GOAL_PAUSE,
                    "complete": LoopExitReason.GOAL_COMPLETE,
                    "fail": LoopExitReason.GOAL_FAIL,
                }.get(goal_decision, LoopExitReason.COMPLETED),
                api_calls=api_call_count,
            )
            return final_content

        # ---------- 循环结束（预算耗尽或中断）----------
        # 这些「失败退场」的轨迹同样有价值（失败恢复的信号常在这里）
        # while 条件自然退出时，细分是「到轮数上限」还是「预算耗尽」
        if turn_exit_reason == LoopExitReason.NORMAL:
            if api_call_count >= self.max_iterations:
                turn_exit_reason = LoopExitReason.MAX_TURNS
            else:
                turn_exit_reason = LoopExitReason.BUDGET_EXHAUSTED
        from agent.skill_learning.turn_observer import maybe_skill_learning
        await maybe_skill_learning(self, user_message)
        # 循环退出路径同样推进提取游标（失败也放行）
        self._maybe_auto_extract()
        return self._handle_loop_exit(turn_exit_reason, user_message)

    # ------------------------------------------------------------------
    # run_conversation 的辅助方法（按主循环调用顺序排列）
    # 纯重构：从原 run_conversation 抽出来的，行为完全等价。
    # ------------------------------------------------------------------

    def _run_prompt_submit_hook(self, user_message: str) -> str:
        """跑 USER_PROMPT_SUBMIT 钩子（钩子可能改写用户消息）。

        参数：
            user_message: 原始用户消息

        返回：钩子处理后的用户消息（没注册钩子或出错时原样返回）。
        """
        if (self.hooks_registry
                and self.config.get("hooks", {}).get("enabled", True)):
            try:
                user_message = self.hooks_registry.run_user_prompt_submit(
                    user_message, session_id=self.session_id or "",
                )
            except Exception as e:
                logger.warning("USER_PROMPT_SUBMIT 编排异常: %s", e)
        return user_message

    def _drain_injected_messages(self) -> dict:
        """把各路外部异步消息一次收齐（后台任务/定时任务/团队信箱/异步子代理）。

        这些消息到了不会自己长腿走进模型上下文，得每轮主动来取；
        取完即清空（本轮注入后下轮不重复）。

        参数：无。

        返回：dict，含五个 key（任何一个都可能是空）：
            bg_notifications: 后台任务通知列表
            cron_messages: 到点的定时任务消息列表
            team_messages_text: 团队信箱消息（拼好的文本）
            delegation_results: 异步子代理的完成结果列表
            rewake_notifications: 异步钩子的唤醒通知列表
        """
        bg_notifications = []
        if self.bg_manager:
            try:
                bg_notifications = self.bg_manager.drain_notifications()
                # 后台任务完成/失败时弹桌面通知（用户切走
                # 也能感知）。标题带 task_id 前 8 位——不同任务标题
                # 不同，30 秒同标题节流就不会互相吞掉（批量完成时每个都能
                # 通知到）。失败也放行，不影响主循环。
                try:
                    import asyncio as _aio_notify
                    from agent.notifier import notify as _bg_notify
                    for n in bg_notifications:
                        status = n.get("status")
                        if status in ("completed", "failed"):
                            _tid = str(n.get("task_id") or "?")
                            # toast 要 spawn PowerShell 子进程（可达数百 ms），
                            # 在事件循环线程里同步等会把流式输出/并发工具全
                            # 冻住——丢线程池 fire-and-forget（不等待、不挡路）。
                            # 没有运行中的事件循环（同步测试上下文）就降级直调。
                            try:
                                _loop = _aio_notify.get_running_loop()
                                _loop.create_task(_aio_notify.to_thread(
                                    _bg_notify,
                                    f"后台任务:{_tid[:8]}",  # id 不足 8 位时切片即全量
                                    f"{_tid} {status}",
                                ))
                            except RuntimeError:
                                _bg_notify(f"后台任务:{_tid[:8]}", f"{_tid} {status}")
                except Exception as notify_err:
                    logger.debug("bg notify fail-open: %s", notify_err)
            except Exception as e:
                logger.warning("drain_notifications 异常: %s", e)

        cron_messages = []
        if self.cron_scheduler:
            try:
                cron_messages = self.cron_scheduler.drain_due()
            except Exception as e:
                logger.warning("cron drain_due 异常: %s", e)

        team_messages_text = ""
        if self.team_bus and self.team_name:
            try:
                msgs = self.team_bus.read_inbox(self.team_name)
                if msgs:
                    team_messages_text = "\n".join(
                        f"[from {m.from_} ({m.type})] {m.content}"
                        for m in msgs
                    )
            except Exception as e:
                logger.warning("team inbox drain 异常: %s", e)

        # 异步子代理的完成通知
        # 取队列出错不炸主循环；读自家实例的队列（塞的一侧已定向
        # 到本 agent）；万一实例队列没建成，退回全局队列
        delegation_results = []
        try:
            queue = getattr(self, "_delegation_queue", None)
            if queue is None:
                from tools.delegate_tool import get_delegation_queue
                queue = get_delegation_queue()
            if queue.has_pending():
                delegation_results = queue.drain()
        except Exception as e:
            logger.warning("delegation_queue drain 异常: %s", e)

        # 异步钩子后台跑完发现阻断问题时的唤醒通知
        rewake_notifications = []
        try:
            from agent.hook_exec import drain_rewake_notifications
            rewake_notifications = drain_rewake_notifications()
        except Exception as e:
            logger.debug("rewake drain 失败（fail-open）: %s", e)

        return {
            "bg_notifications": bg_notifications,
            "cron_messages": cron_messages,
            "team_messages_text": team_messages_text,
            "delegation_results": delegation_results,
            "rewake_notifications": rewake_notifications,
        }

    def has_pending_wake_payload(self) -> bool:
        """预检：有没有"后台完成"类消息待取（给 CLI 的唤醒哨兵用）。

        （idle wake）CLI 收到唤醒哨兵后先问这里——通知已经被
        正在跑的回合消费掉了就别再空跑一轮 LLM（防哨兵风暴/空唤醒）。

        参数：无。

        返回：True=有 bg 任务通知或异步子代理结果待取；全空/出错返回 False。
        """
        try:
            if self.bg_manager is not None and self.bg_manager.has_notifications():
                return True
        except Exception as e:
            logger.debug("has_notifications 预检失败（fail-open 视为无）: %s", e)
        try:
            queue = getattr(self, "_delegation_queue", None)
            if queue is not None and queue.has_pending():
                return True
        except Exception as e:
            logger.debug("delegation_queue 预检失败（fail-open 视为无）: %s", e)
        return False

    def _build_bg_running_note(self):
        """把"仍在跑的后台任务/异步子代理"整理成一段状态说明（没有返回 None）。

        （idle wake 配套）让模型每轮都看得见"还有哪些活儿在后台跑、
        跑完会自动通知/唤醒"，从而不瞎轮询、收工时会主动向用户交代。
        只读状态不消费（区别于通知类的一次性注入），每轮现算。

        参数：无。

        返回：说明文本；一个在跑的都没有时返回 None。
        """
        lines = []
        try:
            if self.bg_manager is not None:
                for t in self.bg_manager.list_tasks():
                    if getattr(t, "status", "") != "running":
                        continue
                    cmd = " ".join(str(c) for c in (t.command or []))
                    if len(cmd) > 60:
                        cmd = cmd[:60] + "…"
                    lines.append(f"- {t.task_id} (running): {cmd}")
        except Exception as e:
            logger.debug("bg running 状态收集失败（fail-open）: %s", e)
        try:
            from tools.delegate_tool import _async_tasks
            for del_id, info in _async_tasks.items():
                if not isinstance(info, dict):
                    continue
                thread = info.get("thread")
                if thread is None or not thread.is_alive():
                    continue
                goal = str(info.get("goal", ""))
                if len(goal) > 60:
                    goal = goal[:60] + "…"
                lines.append(f"- {del_id} (async 子代理): {goal}")
        except Exception as e:
            logger.debug("async 子代理状态收集失败（fail-open）: %s", e)
        if not lines:
            return None
        shown = lines[:10]
        more = f"\n（另有 {len(lines) - 10} 个未列出）" if len(lines) > 10 else ""
        return (
            f"以下 {len(lines)} 个后台任务仍在运行，完成后会以 "
            f"<task_notification>/<delegation_completion> 通知你"
            f"（主对话空闲时会自动唤醒继续处理，无需轮询）：\n"
            + "\n".join(shown) + more
        )

    async def _consume_memory_prefetch(self, messages: list) -> list:
        """等记忆检索预取的结果，把它作为临时消息追加到本轮 messages。

        预取任务在 run_conversation 开场就发起了（把记忆
        检索并行化）；这里等到结果为止。等待点特意放在压缩/剥字段之后，
        并行窗口最大——检索的辅助模型调用延迟大半被主循环的准备工作
        吸收掉了。结果只追加到本轮局部 messages（不进正式历史，和待注入
        队列同语义）；出错放行；一次性消费（紧急压缩重试回到循环时任务
        已清空，不会重复注入）。

        参数：
            messages: 本轮已组装好的消息列表

        返回：追加了记忆消息（若有）的消息列表。
        """
        if self._memory_prefetch_task is None:
            return messages
        task, self._memory_prefetch_task = self._memory_prefetch_task, None
        try:
            msg = await task
            if msg is not None:
                messages.append(msg)
                logger.debug("记忆 prefetch 已消费（并行检索完成）")
        except Exception as e:
            logger.debug("记忆 prefetch 消费 fail-open: %s", e)
        return messages

    def _kick_memory_prefetch(self, user_message: str) -> None:
        """按当前用户消息预取相关记忆（检索式注入的发起端）。

        检索 query 用「用户消息 + 上下文签名」增强（build_augmented_query：
        最近文件/进行中任务/最近回复三信号机械拼接）——长任务后期用户
        常说「继续/好/下一步」，光靠这几个字查记忆没有信号。
        检索是并行预取（先发起任务，组装消息时再等结果）——等待窗口覆盖
        压缩（含 LLM 调用）+ 工具集准备，检索延迟大半被并行吸收。
        降级链：没有辅助小模型 → 退回注入记忆索引快照（有实例级标志，
        只注一次，防每轮重复塞同一份索引）。全部 fail-open。
        """
        # 主代理 only（spawn_depth==0）；快照降级本会话只注入一次
        if (self.spawn_depth == 0 and self.memory_store is not None
                and self._pending_ephemeral_messages is not None):
            from agent.memory_injection import (
                build_relevant_memories_message, build_augmented_query,
                reset_injection_cache, _fallback_snapshot_message,
            )
            # 每轮开头清缓存（防上一轮的缓存串到这一轮）
            reset_injection_cache()
            try:
                if self.aux_llm_router is not None:
                    # 检索路径：并行预取（先不阻塞；组装消息时再等结果）
                    # 带上「最近在用的工具」降噪 + 已注入过的记忆跨轮去重
                    self._memory_prefetch_task = asyncio.create_task(
                        build_relevant_memories_message(
                            query=build_augmented_query(user_message, self),
                            memory_store=self.memory_store,
                            aux_llm_router=self.aux_llm_router,
                            active_tools=self._recent_active_tools(),
                            surfaced=self._surfaced_memory_ids,
                        )
                    )
                elif not self._snapshot_injected:
                    # 降级路径：没有辅助模型 → 注入一次快照（本会话仅此一次）
                    msg = _fallback_snapshot_message(self.memory_store)
                    if msg is not None:
                        self._snapshot_injected = True
                        self._pending_ephemeral_messages.append(msg)
            except Exception as e:
                logger.debug("记忆注入 fail-open: %s", e)

    def _assemble_turn_messages(self, system_prompt: str, injected: dict) -> list:
        """组装本轮要发给 LLM 的消息：系统提示词 + 历史 + 各种临时消息。

        模型每轮看到的完整上下文在这里拼装。injected 里的后台/定时/
        团队消息注入后会被原地清空（防止下轮重复注入）；计划模式提醒每轮
        现算（不消费）。channel/mailbox 走临时注入（失败放行），不进正式
        历史（保护前缀缓存 + 持久化）。

        参数：
            system_prompt: 系统提示词文本
            injected: _drain_injected_messages 返回的外部消息 dict

        返回：拼装好的消息列表（末尾附加各种临时注入）。
        """
        # 开头先把上一批工具触碰收集的技能激活路径处理掉——必须赶
        # 在本方法末尾消费临时消息队列之前，激活通知才能搭上同一轮组装的
        # 车被模型看到，不晚一拍。
        flush_skill_activations(self)

        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history,
        ]

        # === channel/mailbox 临时注入（失败放行）===
        # 用模块级纯函数：出错只打 warning，不影响主流程。
        # 注入位置在历史之后、后台/定时/团队消息之前——channel/mailbox 优先级
        # 更高（外部协作消息比内部任务通知更紧急）
        channel_msg = _build_channel_injection(self._channel_inbox)
        if channel_msg is not None:
            messages.append(channel_msg)

        mail_msg = _build_mail_injection(self._mailbox, self._agent_name)
        if mail_msg is not None:
            messages.append(mail_msg)

        # 后台任务通知（取一次就没了的消费型消息）
        if injected.get("bg_notifications"):
            bg = injected["bg_notifications"]
            notif_text = "\n".join(
                f"[task {n['task_id']} {n['status']}] "
                f"exit={n.get('exit_code')} "
                f"stdout_tail={(n.get('stdout') or '')[-200:]}"
                for n in bg
            )
            messages.append({
                "role": "user",
                "content": f"<task_notification>\n{notif_text}\n</task_notification>",
                # 消费型消息一律标 ephemeral——压缩过滤器
                # 只保留非 ephemeral 的；不标的话它会被焊进内存里的历史、但
                # 又从没落过会话库 → 恢复会话时两边对不上
                "_ephemeral": True,
            })
            injected["bg_notifications"] = []

        # 定时任务到点消息（消费型）
        if injected.get("cron_messages"):
            cron = injected["cron_messages"]
            sched_text = "\n".join(
                f"[Scheduled: {m['job_id']}] {m['message']}"
                for m in cron
            )
            messages.append({
                "role": "user",
                "content": f"<scheduled_message>\n{sched_text}\n</scheduled_message>",
                "_ephemeral": True,  # 消费型注入（同上）
            })
            injected["cron_messages"] = []

        # 团队信箱消息（消费型）
        if injected.get("team_messages_text"):
            team_text = injected["team_messages_text"]
            messages.append({
                "role": "user",
                "content": f"<team_messages>\n{team_text}\n</team_messages>",
                "_ephemeral": True,  # 消费型注入（同上）
            })
            injected["team_messages_text"] = ""

        # 异步子代理完成通知（消费型，每条转成一条临时 user 消息）
        # 失败放行 + 结果截断（防上下文被撑爆）
        delegation_results = injected.get("delegation_results") or []
        for r in delegation_results:
            try:
                success = r.get("success", False)
                delegation_id = r.get("delegation_id", "?")
                goal = r.get("goal", "")
                if success:
                    result_text = r.get("result", "")
                    if len(result_text) > 2000:
                        # 全文落盘留预览 + 找回路径——深度调研结果不再闷头
                        # 砍掉（对齐 L2 落盘哲学；maybe_offload 自带 IO 失败
                        # 降级为截断）
                        try:
                            from agent.output_offload import maybe_offload
                            result_text = maybe_offload(
                                result_text,
                                tool_call_id=f"delegation_{delegation_id}",
                                agent_home=self.codeAgent_home,
                                threshold=2000,
                                preview_chars=2000,
                            )
                        except Exception as e:
                            logger.warning(
                                "async 结果落盘失败（降级截断）: %s", e,
                            )
                            result_text = result_text[:2000] + (
                                f"...[truncated {len(result_text)} chars]"
                            )
                    text = (
                        f"[后台子代理完成] task_id={delegation_id}\n"
                        f"任务: {goal}\n"
                        f"结果: {result_text}"
                    )
                else:
                    error_text = r.get("error", "")
                    text = (
                        f"[后台子代理失败] task_id={delegation_id}\n"
                        f"任务: {goal}\n"
                        f"错误: {error_text}"
                    )
                messages.append({
                    "role": "user",
                    "content": f"<delegation_completion>\n{text}\n</delegation_completion>",
                    "_ephemeral": True,  # 消费型注入（同上）
                })
            except Exception as e:
                logger.warning("delegation_results 注入异常: %s", e)
        if delegation_results:
            injected["delegation_results"] = []

        # 异步钩子的阻断通知（消费型临时消息）
        rewakes = injected.get("rewake_notifications") or []
        for note in rewakes:
            try:
                status = note.get("status_message") or ""
                messages.append({
                    "role": "user",
                    "content": (
                        "<rewake_notification>（异步 hook 跑完发现阻断性问题，"
                        "请评估是否跟进处理）\n"
                        f"[hook {note.get('hook', '?')}] "
                        f"原因: {note.get('reason', '')}"
                        + (f"（{status}）" if status else "")
                        + "\n</rewake_notification>"
                    ),
                    "_ephemeral": True,  # 消费型注入（同上）
                })
            except Exception as e:
                logger.warning("rewake 注入异常: %s", e)
        if rewakes:
            injected["rewake_notifications"] = []

        # idle wake 配套：仍在运行的后台任务/异步子代理清单（每轮现算、
        # 只读状态不消费）。让模型保持"后台还有活儿在跑"的预期——收工时
        # 向用户交代、完成后等通知/自动唤醒，而不是瞎轮询。
        bg_running_note = self._build_bg_running_note()
        if bg_running_note:
            messages.append({
                "role": "user",
                "content": (
                    "<background_tasks_running>\n"
                    f"{bg_running_note}\n"
                    "</background_tasks_running>"
                ),
                "_ephemeral": True,  # 状态类临时注入（每轮重算，不进历史）
            })

        # 计划模式提醒（每轮现算重新注入）
        if self.plan_mode:
            messages.append({
                "role": "user",
                "content": (
                    "<plan_mode_reminder>\n"
                    "你处于【计划模式】，只能调研，不能修改任何东西。\n"
                    "完成调研后必须调 exit_plan_mode(plan=...) 提交计划等待用户审批。\n"
                    "计划要包含：要改什么文件、为什么、步骤、风险点。\n"
                    "</plan_mode_reminder>"
                ),
                "_ephemeral": True,  # 每轮重算的临时注入（同上）
            })

        # 上下文管理提示（快到上限时建议用户主动 /compact 或 /new）
        self._maybe_inject_context_tip(messages)
        # 长任务定期提醒更新进度外存（双条件节流）
        self._maybe_inject_progress_reminder(messages)

        # === 消费待注入临时队列（goal continue 的注入口）===
        # 主循环要继续 goal 时把临时消息塞进这个队列（不进正式历史），
        # 本轮组装时取出消费并清空——模型看得到，但不污染持久化
        if self._pending_ephemeral_messages:
            messages.extend(self._pending_ephemeral_messages)
            self._pending_ephemeral_messages = []

        # === 批间摘要注入（上一批工具调用的一句话总结，临时消息）===
        # 由后台辅助模型任务生成（工具分发末尾发出去就不管），这里消费并
        # 清空；还没生成完就本轮跳过（不等它，把延迟藏在后台）
        if self._pending_tool_batch_summary:
            messages.append({
                "role": "user",
                "content": (
                    "<tool_batch_summary>（上一批工具调用的一句话总结）\n"
                    f"{self._pending_tool_batch_summary}\n</tool_batch_summary>"
                ),
                "_ephemeral": True,
            })
            self._pending_tool_batch_summary = None

        # 注意：这里故意不剥 _timestamp——按时间清理旧工具结果需要读它；
        # 剥的操作挪到了主循环压缩之后、发 LLM 之前
        return messages

    def _maybe_inject_context_tip(self, messages: list) -> None:
        """上下文快到上限时给模型塞一条「管理建议」提示。

        context rot（上下文太长模型变笨）——
        自动压缩总发生在模型「最不聪明」的时刻，不如主动 /compact
        并说明保留重点；换新任务用 /new；读大文件委托子代理只带摘要回来。
        每会话只提示一次，免得每轮刷屏。

        参数：
            messages: 本轮消息列表（原地追加提示）

        返回：无。
        """
        if self._context_tip_shown:
            return
        try:
            from agent.context_compressor import estimate_message_tokens
            est = estimate_message_tokens(messages)
            token_threshold = self.config.get("context", {}).get(
                "llm_compact_token_threshold", 100000,
            )
            if self.model and "[1m]" in str(self.model):
                token_threshold = max(token_threshold, 700000)
            if est >= token_threshold * 0.7:
                self._context_tip_shown = True
                pct = int(est / token_threshold * 100) if token_threshold else 0
                messages.append({
                    "role": "user",
                    "content": (
                        "<context_management_tip>\n上下文接近上限（约 "
                        f"{pct}%）。为避免自动压缩发生在效果最差时：\n"
                        "1. 继续当前任务 → 建议主动 /compact 并说明保留哪些重点\n"
                        "2. 换新任务 → 建议 /new 新开对话（避免 context rot）\n"
                        "3. 大文件读取 → 用 subagent 委托子代理，只带摘要回主上下文\n"
                        "</context_management_tip>"
                    ),
                    "_ephemeral": True,  # 一次性管理提示（同上）
                })
        except Exception as e:
            logger.debug("上下文管理提示注入失败（忽略）: %s", e)

    def _is_long_task(self) -> bool:
        """长任务信号：历史条数 > 100 或本场触发过 L4 压缩。

        F（进度提醒）和 E（批间摘要自启）共用——单一事实源，
        阈值别在两处各写一份（会漂移）。
        """
        return (
            len(self.conversation_history or []) > 100
            or self._compress_session_state.llm_compact_count > 0
        )

    def _maybe_inject_progress_reminder(self, messages: list) -> None:
        """长任务里定期提醒更新 PROGRESS.md（双条件节流，ephemeral）。

        进度外存此前只在压缩醒来那一刻被提醒写——两次压缩之间中断
        的话外存是旧的。这里双条件兜底：
        1. 长任务信号：历史条数 > 100 或本场触发过 L4 压缩
        2. 节流：距上次提醒 >= progress_reminder_turns 个 LLM 轮（0 关闭）

        提醒走 ephemeral 临时注入，不进历史不碰缓存。
        """
        try:
            cfg = (self.config or {}).get("context", {})
            interval = int(cfg.get("progress_reminder_turns", 40))
            if interval <= 0:
                return
            if not self._is_long_task():
                return
            turn = self._compress_session_state.current_turn
            if turn - self._last_progress_reminder_turn < interval:
                return
            self._last_progress_reminder_turn = turn

            from agent.scratchpad import scratchpad_dir
            sp = scratchpad_dir(
                getattr(self, "session_id", "") or "default",
                getattr(self, "codeagent_home", None),
            )
            messages.append({
                "role": "user",
                "content": (
                    "<progress_reminder>\n"
                    "这是段长任务。如果自上次更新以来有新的关键结论"
                    "（重要决策/发现/架构判断/已完成步骤），请追加写入\n"
                    f"{sp / 'PROGRESS.md'}\n"
                    "（一行一条、最新在前）——上下文压缩时它会原样回读，"
                    "跨会话恢复时也会回读。\n"
                    "有跨会话价值的用户/项目事实，记得用 memory 工具保存。\n"
                    "</progress_reminder>"
                ),
                "_ephemeral": True,
            })
        except Exception as e:
            logger.debug("progress reminder 注入失败（fail-open）: %s", e)

    async def _run_context_compression(
        self, messages: list, system_prompt: str, *, force: bool = False,
    ) -> tuple:
        """快到 token 上限时压缩上下文（整体已异步化）。

        上下文太大不但贵还会超限报错，得分层瘦身。
        返回 (messages, system_prompt, compressed: bool)：
        - 无损变化（大结果落盘 / 按时间清旧工具结果 / 折叠）：只同步历史，
          返回 compressed=False（不做重建提示词、注入简报这些「仪式」，
          尽量保住缓存）
        - L4 的 LLM 摘要压缩（真有损总结）：重建系统提示词并注入
          <post_compress_brief>「醒来简报」，返回 compressed=True

        参数：
            messages: 本轮消息列表
            system_prompt: 当前系统提示词

        返回：三元组 (压缩后的 messages, 可能重建的 system_prompt, 是否真压缩)。
        """
        if not self.compression_enabled:
            return messages, system_prompt, False

        from agent.context_pipeline import compress_if_needed
        ctx_cfg = self.config.get("context", {})
        messages, changed, compacted = await compress_if_needed(
            messages,
            llm_client=self.llm_client,
            model=self.model,
            config=ctx_cfg,
            session_state=self._compress_session_state,
            agent_home=self.codeAgent_home,
            session_id=self.session_id,
            hooks_registry=self.hooks_registry,
            tools=self._last_tool_schemas,  # fork 摘要前缀复用
            authoritative_tokens=self._last_usage_anchor,  # 混合计数
            session_store=self.session_store,  # L4 前落 [COMPACT_START] 事务标记
            force=force,  # 发送前预检的强制压缩：绕冷却但守熔断
        )
        if not changed:
            return messages, system_prompt, False

        # 任何一层动了 messages → 同步回正式历史（不同步的话，按时间清理
        # 下一轮会把原始内容原样塞回来；折叠/落盘同理）。
        # 顺手滤掉临时消息（它们不该进正式历史，保护持久化）。
        # 不能无脑切 messages[1:]——system 缺失时会
        # 悄悄丢掉第一条真实消息；只在 [0] 确实是 system 时才剥。
        self.conversation_history = [
            m for m in _drop_leading_system(messages) if not m.get("_ephemeral")
        ]

        if not compacted:
            # 无损变化（落盘/按时间清理/折叠）
            # 到此为止——作废缓存 / 打压缩边界标记 / 注入「历史已被总结」
            # 简报这些仪式，只有真做了 LLM 摘要压缩才有意义（一次大工具结果
            # 落盘就跑全套仪式会误导模型 + 白白打穿缓存）。
            return messages, system_prompt, False

        # 压缩前调记忆管理器提取事实（趁旧消息还在）
        if self.memory_manager:
            try:
                self.memory_manager.on_pre_compress(None, messages)
            except Exception as e:
                logger.warning("on_pre_compress 编排异常: %s", e)

        self.invalidate_system_prompt()
        system_prompt = self._get_system_prompt()
        self._compression_attempts += 1

        # 压缩把早期对话（含当时注入的记忆内容）摘要掉了——已注入记忆的
        # 跨轮去重集合不再挡着，重新允许注入（否则长任务后期对这些记忆
        # 彻底失明，而压缩后恰恰最需要它们补上下文）
        try:
            self._surfaced_memory_ids.clear()
        except Exception:
            pass

        # 压缩后重新对齐：注入一条「刚醒来」简报
        brief_parts = [
            "你刚经历了上下文压缩，历史已被总结。"
            "身份和 system prompt 不变。"
        ]
        mode_text = (
            "计划模式（只能调研，不能修改）"
            if self.plan_mode
            else "正常执行模式"
        )
        brief_parts.append(f"当前模式：{mode_text}")

        # 压缩后把最近加载的技能正文 + 读过的
        # 文件重新注入，让 agent 压缩后不「失忆」（免得反复手动读文件、重新
        # 加载技能）。具体活儿委托给 post_compact_recovery 模块（失败放行，
        # 写文件走 safe_path 白名单校验）。
        try:
            from agent.post_compact_recovery import build_post_compact_brief
            reinject = build_post_compact_brief(self)
        except Exception as e:
            logger.debug("post_compact_recovery fail-open: %s", e)
            reinject = ""
        if reinject:
            brief_parts.append(
                "以下是你最近加载的技能和读过的文件（压缩后重注入，帮助恢复上下文）：\n"
                f"{reinject}"
            )

        # 长任务进度外存：引导把关键中间结论写进会话涂鸦区的 PROGRESS.md——
        # 摘要每段 200 字有损耗，这个文件下次压缩时被原样回读，零损耗
        try:
            from agent.scratchpad import ensure_scratchpad
            sp_dir = ensure_scratchpad(self.session_id, self.codeAgent_home)
            if sp_dir is not None:
                brief_parts.append(
                    "长任务建议：把关键中间结论（重要决策/发现/架构判断/已完成步骤）"
                    f"追加写入 {sp_dir / 'PROGRESS.md'}（一行一条，最新在前）。"
                    "下次压缩时该文件会被原样回读，防止长任务细节在反复压缩中丢失。"
                )
        except Exception as e:
            logger.debug("进度外存提示失败（fail-open）: %s", e)

        brief_parts.append("请继续之前的工作。")
        messages.append({
            "role": "user",
            "content": (
                "<post_compress_brief>\n"
                + "\n".join(brief_parts)
                + "\n</post_compress_brief>"
            ),
        })

        return messages, system_prompt, True

    async def _prepare_toolset_and_injections(self, messages: list) -> list:
        """发请求前的最后准备：刷新工具集（计划模式切换）+ 注入重试警告 + 跑 PRE_LLM_CALL 钩子。

        会原地改 messages（往里追加提醒）。这里不做记忆注入——
        记忆统一走按需检索的临时注入（每条用户消息一次，不重复、不污染历史）。
        方法保持 async 是因为 PRE_LLM_CALL 钩子等异步依赖还要用。

        参数：
            messages: 本轮消息列表（原地追加提醒）

        返回：工具 schema 列表（可能被钩子改过）。
        """
        from model_tools import get_tool_definitions

        # 计划模式：强制切到 plan 工具集（全只读）
        effective_toolsets = ["plan"] if self.plan_mode else self.enabled_toolsets
        # 从 config 透传禁用工具清单（子代理自定义 .md 里声明的
        # disallowedTools，从这传给工具定义层过滤）
        _disabled = (self.config or {}).get("disabled_tools")
        tool_schemas = get_tool_definitions(
            effective_toolsets, disabled_tools=_disabled, agent=self)

        # 失败重试检测：连续 N 次工具失败 → 塞一条「别再用同样方式重试」的提醒
        # （_ephemeral：临时提醒只给模型看一眼，不落盘——漏标的话
        #   当轮触发压缩时会被历史同步收进正式对话记录）
        if self._tool_failure_streak >= self._failure_threshold:
            messages.append({
                "role": "user",
                "content": (
                    f"<retry_warning>\n"
                    f"你已连续 {self._tool_failure_streak} 次工具调用失败。\n"
                    f"最近错误: {self._last_tool_error}\n\n"
                    f"**不要用完全相同的方式重试**。建议:\n"
                    f"1. 分析错误根因(看 stderr / error 字段)\n"
                    f"2. 换一种方法(改命令 / 改路径 / 改参数)\n"
                    f"3. 如果是环境问题(路径冲突 / 权限 / 版本不兼容),"
                    f"**停下来告诉用户**具体问题和解决建议\n"
                    f"</retry_warning>"
                ),
                "_ephemeral": True,
            })
            self._tool_failure_streak = 0  # 重置（提醒一次就够）

        # PRE_LLM_CALL hook
        if (self.hooks_registry
                and self.config.get("hooks", {}).get("enabled", True)):
            try:
                # 挪到线程里跑，别冻住事件循环
                messages, tool_schemas = await asyncio.to_thread(
                    self.hooks_registry.run_pre_llm_call,
                    messages, tool_schemas,
                    session_id=self.session_id or "",
                )
            except Exception as e:
                logger.warning("PRE_LLM_CALL hook 编排异常: %s", e)

        return tool_schemas

    async def _call_llm_with_escalation(
        self, messages: list, tool_schemas: list, system_prompt: str,
        flagged_messages: Optional[list] = None,
    ):
        """调 LLM，并处理两类「意外抢救」：max_tokens 升级重试 + 输入超长的紧急压缩。

        返回值三种：
            response 对象         - 正常拿到响应
            self._REACTIVE_RETRY  - 刚做了紧急压缩，主循环应重试本轮
            None                  - LLM 出错（错误信息已写回历史），主循环应退出

        参数：
            messages: 要发的消息列表
            tool_schemas: 工具 schema 列表（可为 None）
            system_prompt: 当前系统提示词
            flagged_messages: _ephemeral 未剥的消息版本（reactive 紧急压缩的
                重建历史要用它——strip 后的消息 flag 已剥，滤临时注入会失守；
                None 时退回 messages）

        另：调用前后会做缓存监控快照（fail-open，绝不影响主流程）。
        """
        # === 调用前：缓存监控给 prompt 状态拍 12 维快照（失败放行）===
        # 12 个维度 + 每个工具单独 hash
        cache_state = None
        try:
            from agent.cache_monitor import record_prompt_state
            # 从 config 提取 LLM 调用参数（和 _call_llm_streaming 取 max_tokens 的逻辑保持一致）
            _cfg = self.config or {}
            _mt = (
                _cfg.get("model", {}).get("max_tokens")
                or _cfg.get("llm", {}).get("max_tokens")
            ) or 0
            _temp = _cfg.get("model", {}).get("temperature")
            # 第一条 user 消息的开头片段（用来捕捉用户消息有没有被改）
            _ucp = ""
            if messages:
                for m in messages:
                    if m.get("role") == "user":
                        _ucp = str(m.get("content", ""))[:500]
                        break
            cache_state = record_prompt_state(
                system_prompt=system_prompt or "",
                tools=tool_schemas or [],
                model=self.model or "",
                max_tokens=_mt,
                temperature=_temp,
                stream_mode=self._stream_callback is not None,
                tool_choice=_cfg.get("model", {}).get("tool_choice"),
                betas=_cfg.get("model", {}).get("betas"),
                user_content_prefix=_ucp,
                messages_count=len(messages),
            )
        except Exception as e:
            logger.debug("cache_monitor pre-call fail-open: %s", e)

        try:
            if self._stream_callback is not None:
                response = await self._call_llm_streaming(
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                )
            else:
                from agent.llm_retry import call_with_retry, detect_length_finish
                response = await call_with_retry(
                    self.llm_client,
                    messages,
                    tools=tool_schemas if tool_schemas else None,
                    fallback_llm_client=self.fallback_llm_client,
                    config=self.config,
                    heartbeat_cb=self._llm_retry_heartbeat,  # 长退避心跳
                )
                # 非流式路径也支持 max_tokens 升级
                new_max = None
                if detect_length_finish(response):
                    new_max = self._try_escalate_max_tokens("非流式")
                # 拿到新上限才重试；None = 不该升级/已升过级，沿用截断响应
                if new_max is not None:
                    try:
                        response = await call_with_retry(
                            self.llm_client,
                            messages,
                            tools=tool_schemas if tool_schemas else None,
                            fallback_llm_client=self.fallback_llm_client,
                            max_tokens=new_max,
                            config=self.config,
                            heartbeat_cb=self._llm_retry_heartbeat,  # 长退避心跳
                        )
                    except Exception as esc_err:
                        logger.warning(
                            "max_tokens 升级重试失败（沿用截断响应）: %s", esc_err,
                        )

            # === 调用后：检查缓存有没有被打穿（失败放行）===
            if cache_state is not None:
                try:
                    from agent.cache_monitor import check_cache_break
                    cache_read = self._extract_cache_read(
                        getattr(response, "usage", None)
                    )
                    check_cache_break(
                        current_state=cache_state,
                        cache_read_tokens=cache_read,
                        query_source="main",
                    )
                except Exception as e:
                    logger.debug("cache_monitor post-call fail-open: %s", e)

            return response

        except Exception as e:
            # 紧急压缩（reactive_compact）：API 报「输入超长」时马上压缩再重试。
            # 支持多次触发（冷却 60 秒 + 每会话上限 5 次），次数控制
            # 收在 reactive_compact 自己内部
            err_str = str(e).lower()
            self._last_llm_error_kind = "other"

            # === 扣留-恢复：流式卡死超时 → 先按住不报，转非流式重试一次 ===
            # （可恢复的错误先藏住，恢复路径确认没救才报）
            from agent.llm_client import LLMStreamIdleTimeout
            if isinstance(e, LLMStreamIdleTimeout):
                logger.warning("流空闲超时（看门狗），扣留转非流式重试一次")
                # 显式扔掉已累积的半截状态（防御性「墓碑」清理，出错也放行）
                try:
                    self._discard_partial_stream_state()
                except Exception:
                    pass
                from agent.llm_retry import call_with_retry as _cwr
                try:
                    response = await _cwr(
                        self.llm_client,
                        messages,
                        tools=tool_schemas if tool_schemas else None,
                        fallback_llm_client=self.fallback_llm_client,
                        config=self.config,
                        heartbeat_cb=self._llm_retry_heartbeat,  # 长退避心跳
                    )
                    logger.info("流空闲超时恢复成功（非流式路径）")
                    return response
                except Exception as recover_err:
                    # 恢复也失败了：把非流式路径的错误抛出来（它还带着 400 溢出
                    # 后自动下调 max_tokens 的机会）
                    e = recover_err
                    err_str = str(e).lower()
                    self._last_llm_error_kind = "stream_idle"
                    logger.warning(
                        "流空闲超时的非流式恢复也失败（透出）: %s", e,
                    )

            is_prompt_too_long = (
                "prompt_too_long" in err_str
                or "context_length" in err_str
                or "maximum context" in err_str
            )
            # 输入超长（PTL）一律先试紧急压缩抢救（不再受功能开关
            # 门控——超长是可恢复错误，恢复优先于报错；
            # 防无限循环的冷却/次数上限在紧急压缩内部生效。旧的功能开关语义
            # 已废弃：超长抢救是韧性底线，不是可选功能）
            if is_prompt_too_long:
                self._last_llm_error_kind = "prompt_too_long"
                from agent.context_pipeline import reactive_compact
                ctx_cfg = self.config.get("context", {})
                # 输入用 flag 完好版：reactive 重建历史时有 _ephemeral 可滤，
                # 本轮临时注入不会被焊进 conversation_history（与 force 预检同款修法）
                _reactive_src = (
                    flagged_messages if flagged_messages is not None else messages
                )
                messages, changed = reactive_compact(
                    _reactive_src,
                    session_state=self._compress_session_state,
                    keep_recent=ctx_cfg.get("reactive_keep_recent", 5),
                    cooldown_seconds=ctx_cfg.get(
                        "reactive_compact_cooldown_seconds", 60),
                    max_per_session=ctx_cfg.get(
                        "reactive_compact_max_per_session", 5),
                )
                if changed:
                    self._reacted = True  # 向后兼容标记（有测试读它）
                    # 重建历史时滤掉临时消息（保护持久化）；
                    # 剥 system 用防御版函数，防 system 缺失时丢消息
                    self.conversation_history = [
                        m for m in _drop_leading_system(messages)
                        if not m.get("_ephemeral")
                    ]
                    # 边界占位落库（与 L4 同款）：恢复时能按
                    # [COMPACT_BOUNDARY] 裁掉紧急压缩前的旧历史
                    try:
                        from agent.context_pipeline import (
                            take_last_compact_placeholder,
                        )
                        _ph = take_last_compact_placeholder()
                        if _ph:
                            self._persist_session_message("user", _ph)
                    except Exception as _e:
                        logger.warning("reactive 边界落库失败（fail-open）: %s", _e)
                    self.invalidate_system_prompt()
                    logger.warning("reactive_compact 后重试本轮")
                    return self._REACTIVE_RETRY
                # changed=False：还在冷却中或到次数上限了，走正常报错路径

            logger.error("LLM API 调用失败（重试后）: %s", e)

            # === goal 遇网络异常自动暂停 ===
            # 触发关键词：529 / overloaded / timeout / connection / network
            # 只在 goal 进行中时暂停（没有 goal 就不搞副作用）；
            # 暂停这个动作本身失败也只打日志
            if self._goal_state is not None and self._goal_state.status == "active":
                network_keywords = (
                    "529", "overloaded", "timeout",
                    "connection", "network", "timed out",
                    "connectionerror", "connectionreseterror",
                )
                if any(kw in err_str for kw in network_keywords):
                    try:
                        self._goal_state.pause(reason="network")
                        self._goal_state.save(self._goal_state_path())
                        logger.warning(
                            "goal 自动 pause（网络异常）: %s", self._goal_state.pause_reason,
                        )
                        # 暂停的桌面通知已集中到 GoalState.pause() 里发，
                        # 这里不再另发（防重复弹两遍）
                    except Exception as pause_err:
                        logger.warning("goal pause 失败（fail-open）: %s", pause_err)

            # 把错误包装成一条 assistant 消息塞回历史——模型下一轮看到，
            # 有机会自己想办法补救
            self.conversation_history.append({
                "role": "assistant",
                "content": f"[API 错误: {e}]",
                "_timestamp": time.time(),
            })
            return None

    def _try_escalate_max_tokens(self, trigger_desc):
        """max_tokens 截断后的「升级判定」：看能不能调大上限，能就调并打日志。

        大白话：先过两道门——没装升级器（self._max_tokens_escalator 为
        None）、或本轮已经升过级（防无限套娃）——任一道挡住就返回 None，
        调用方沿用截断响应。两道门都过了才真正调 escalate() 拿新上限
        （这步有副作用：标记「已升级」），并打一条 info 日志。

        参数：
            trigger_desc: 触发原因文案，原样进日志（如「非流式」、
                「finish_reason=length」、「纯 thinking（content 空）」）

        返回：新的 max_tokens 上限；不该/不能升级时返回 None。
        """
        if (self._max_tokens_escalator is None
                or self._max_tokens_escalator.has_escalated):
            return None
        new_max = self._max_tokens_escalator.escalate()
        logger.info(
            "max_tokens 截断（%s），升级到 %d 重试", trigger_desc, new_max
        )
        return new_max

    @staticmethod
    def _merge_usage_tokens(final_usage, retried):
        """把升级重试响应的 usage 逐字段加总进旧账本。

        大白话：截断那次和升级重试这次是两笔真实花费，直接拿新 usage
        覆盖会漏记前一笔，所以四个字段（prompt/completion/两类缓存）
        逐项相加——Anthropic 字段优先、DeepSeek 字段兜底，加总顺序与
        字段名和原内联实现逐字节一致。重试响应没带 usage 就原样返回
        旧账本（新调用一分钱没记）。

        参数：
            final_usage: 已有用量 dict（不是 dict 时按空账本处理）
            retried: 升级重试拿到的响应对象

        返回：加总后的新用量 dict（不原地改旧 dict）。
        """
        if getattr(retried, "usage", None) is None:
            return final_usage
        u = retried.usage
        retry_usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0),
            "completion_tokens": getattr(u, "completion_tokens", 0),
            "cache_read": (
                getattr(u, "cache_read_input_tokens", 0)
                or getattr(u, "prompt_cache_hit_tokens", 0)
            ),
            "cache_creation": (
                getattr(u, "cache_creation_input_tokens", 0)
                or getattr(u, "prompt_cache_miss_tokens", 0)
            ),
        }
        prev_usage = final_usage if isinstance(final_usage, dict) else {}
        return {
            k: (prev_usage.get(k, 0) or 0) + (retry_usage.get(k, 0) or 0)
            for k in retry_usage
        }

    async def _recover_output_truncation(self, response, messages, tool_schemas):
        """调大上限后仍被截断 → 让模型「从断点接着写」的续写恢复。

        做法：把截断的半截回答 +
        一条「从中断处直接继续、不道歉不复述」的指令，追加到**只在本次请求
        里用的局部消息**上再调 LLM，把续写拼上去；最多重试
        llm.output_recovery_limit 次（默认 3）。

        设计要点：
        - 局部 messages 不进正式历史——恢复成功后以「拼接好的完整回答」
          一条消息返回（主循环正常入史），不污染会话记录
        - 只处理纯文本截断；工具调用被截断（罕见）或没有可续内容就原样返回
        - 还没做过上限升级的截断不接手（那是上游升级路径的事）
        - 失败放行：恢复调用挂了就返回已拼接的部分（保留截断标记，
          主循环当最终响应处理）

        参数：
            response: 被截断的响应对象
            messages: 本轮消息列表
            tool_schemas: 工具 schema 列表

        返回：恢复后的响应对象（或原样返回）。
        """
        from agent.llm_retry import (
            DEFAULT_OUTPUT_RECOVERY_LIMIT,
            call_with_retry,
            detect_length_finish,
        )
        if not detect_length_finish(response):
            return response
        if (self._max_tokens_escalator is not None
                and not self._max_tokens_escalator.has_escalated):
            return response
        msg = response.choices[0].message
        if getattr(msg, "tool_calls", None):
            return response
        accumulated = msg.content or ""
        if not accumulated.strip():
            return response

        try:
            limit = int(
                (self.config or {}).get("llm", {}).get(
                    "output_recovery_limit", DEFAULT_OUTPUT_RECOVERY_LIMIT,
                )
            )
        except (TypeError, ValueError):
            limit = DEFAULT_OUTPUT_RECOVERY_LIMIT
        limit = max(0, limit)

        recovery_meta = (
            "你的上一条回复因输出 token 上限被截断。"
            "从中断处直接继续——不要道歉、不要复述已写内容，"
            "从被切断的那个位置接着写。把剩余工作拆成小块完成。"
        )
        recovery_max_tokens = (
            self._max_tokens_escalator.get_next_max_tokens()
            if self._max_tokens_escalator is not None else None
        )
        local_messages = list(messages) + [
            {"role": "assistant", "content": accumulated},
            {"role": "user", "content": recovery_meta},
        ]
        for attempt in range(1, limit + 1):
            try:
                resp = await call_with_retry(
                    self.llm_client,
                    local_messages,
                    tools=tool_schemas if tool_schemas else None,
                    fallback_llm_client=self.fallback_llm_client,
                    max_tokens=recovery_max_tokens,
                    config=self.config,
                    heartbeat_cb=self._llm_retry_heartbeat,  # 长退避心跳
                )
            except Exception as e:
                logger.warning("续写恢复调用失败（返回已拼接内容）: %s", e)
                break
            piece = (resp.choices[0].message.content or "")
            if piece:
                accumulated += piece
            if not detect_length_finish(resp):
                logger.info("续写恢复成功（第 %d 次），拼接 %d 字符", attempt, len(accumulated))
                return self._merge_continuation_response(
                    response, resp, accumulated, finished=True,
                )
            local_messages = list(local_messages) + [
                {"role": "assistant", "content": piece},
                {"role": "user", "content": recovery_meta},
            ]
        if limit > 0:
            logger.warning("续写恢复 %d 次后仍截断，返回已拼接内容", limit)
        return self._merge_continuation_response(
            response, None, accumulated, finished=False,
        )

    @staticmethod
    def _merge_continuation_response(
        truncated_response, last_response, content, *, finished: bool,
    ):
        """续写恢复的最后一步：把截断响应和续写响应合并成一个。

        拼上内容、取最后一轮的用量，形状对齐截断前的响应结构。
        最后一轮续写如果带工具调用，必须保留并把
        结束原因标成 "tool_calls"（主循环按它分发工具；硬编码成
        「无工具调用 + stop」会把模型明确要调工具的意图吞掉还伪装成正常
        完成）。还没写完（仍截断）就维持 "length"。

        参数：
            truncated_response: 最初被截断的响应
            last_response: 最后一轮续写的响应（可能为 None）
            content: 拼接好的完整文本
            finished: 关键字参数，True = 恢复完成不再截断

        返回：合并后的响应对象。
        """
        from types import SimpleNamespace
        src_msg = truncated_response.choices[0].message
        last_msg = None
        if last_response is not None:
            try:
                last_msg = last_response.choices[0].message
            except (IndexError, AttributeError):
                last_msg = None
        tool_calls = getattr(last_msg, "tool_calls", None) if last_msg else None
        merged = SimpleNamespace(
            content=content if content else None,
            tool_calls=tool_calls,
            reasoning_content=getattr(src_msg, "reasoning_content", None),
            thinking_signature=getattr(src_msg, "thinking_signature", None),
        )
        usage = (
            getattr(last_response, "usage", None)
            if last_response is not None
            else getattr(truncated_response, "usage", None)
        )
        if not finished:
            finish_reason = "length"
        else:
            finish_reason = "tool_calls" if tool_calls else "stop"
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=merged,
                finish_reason=finish_reason,
            )],
            usage=usage,
        )

    def _run_post_llm_call_hook(self, response):
        """跑 POST_LLM_CALL 钩子（钩子可能改写响应对象）。

        参数：response: LLM 响应对象。
        返回：钩子处理后的响应（没钩子或出错时原样返回）。
        """
        if (self.hooks_registry
                and self.config.get("hooks", {}).get("enabled", True)):
            try:
                response = self.hooks_registry.run_post_llm_call(
                    response, session_id=self.session_id or "",
                )
            except Exception as e:
                logger.warning("POST_LLM_CALL hook 编排异常: %s", e)
        return response

    def _checkpoint_track(self, path: str) -> None:
        """编辑工具成功改完文件后调用：记下这个文件，供快照/回滚用。

        参数：path: 被修改的文件路径。返回：无（出错只打 debug 日志）。
        """
        if self.checkpoint_manager:
            try:
                self.checkpoint_manager.track_file(path)
            except Exception as e:
                logger.debug("checkpoint track 失败: %s", e)

    def _maybe_auto_extract(self) -> None:
        """对话级轻量记忆提取的启动器（主循环末尾调）。

        让 agent 顺手从对话里捞事实存记忆。门控条件：config 里
        memory.auto_extract.enabled 开着（默认关）+ 每 N 回合才一次 +
        只有主代理做（子代理有独立记忆目录，不走这条链）。
        互斥：本轮模型自己调过记忆写入工具 → 跳过并推进游标
        （已经写过了，别抢着重复写）。
        发出去就不管：用 _spawn_detached 跑提取（辅助模型单轮；submit 到
        常驻宿主循环、进回合栅栏豁免名单——不会被回合收尾误杀）；
        任务引用保活；出错放行。

        参数：无。返回：无。
        """
        try:
            cfg = (self.config or {}).get("memory", {}).get("auto_extract", {})
            if not cfg.get("enabled", False):
                return
            # 游标始终前移（本轮内容要么被提取、要么被互斥/节流跳过，都不回头再看）
            start_idx = self._auto_extract_cursor
            self._auto_extract_cursor = len(self.conversation_history)
            self._auto_extract_turn_count += 1
            every = int(cfg.get("every_n_turns", 3) or 3)
            should_run = (
                self._auto_extract_turn_count % every == 0
                and self.spawn_depth == 0
                and not self._memory_touched_this_turn
                and self.aux_llm_router is not None
                and self.conversation_history
                and start_idx < self._auto_extract_cursor
            )
            self._memory_touched_this_turn = False  # 互斥标记每回合重置
            if not should_run:
                return
            from agent.auto_extract import run_auto_extract
            self._auto_extract_task = _spawn_detached(
                run_auto_extract(self, start_idx), "auto-extract",
            )
        except Exception as e:
            logger.debug("auto_extract 启动失败（fail-open）: %s", e)

    def _record_recent(self, kind: str, key: str) -> None:
        """记下最近读过的文件 / 加载过的技能（去重、保序，只留最近 10 个）。

        上下文压缩后把这些重新注入，让 agent 不「失忆」。

        参数：
            kind: "read"（读过的文件）或 "skill"（加载的技能）
            key: 文件路径或技能名

        返回：无。
        """
        bucket = self._recent_read_files if kind == "read" else self._recent_skills
        if key in bucket:
            bucket.remove(key)
        bucket.append(key)
        del bucket[:-10]

    def _load_skill_body(self, name: str) -> str:
        """跨目录找技能并读正文（剥掉文件头的 frontmatter 元数据）。

        参数：name: 技能名。返回：技能正文文本；找不到或出错返回空串。
        """
        try:
            from tools.skill_tools import _find_skill_md, _get_skills_dirs
            from agent.skill_commands import parse_frontmatter
            md = _find_skill_md(
                name,
                _get_skills_dirs({"codeagent_home": str(self.codeAgent_home)}),
            )
            if md is None:
                return ""
            content = md.read_text(encoding="utf-8")
            _, body = parse_frontmatter(content)
            return body.strip()
        except Exception as e:
            logger.debug("加载技能正文失败 %s: %s", name, e)
            return ""

    def _persist_session_message(self, role, content, *, tool_calls=None,
                                 tool_call_id=None, name=None) -> None:
        """把一条消息写进会话库（工具轮次也完整入库）。

        内存里的对话历史本来就含工具轮次；这里把 assistant（带工具
        调用）和工具结果同步写进会话库，重开会话恢复时才能完整重放。
        用户输入和最终回答由 cli.py 负责持久化，这里只补工具轮次，避免
        重复写。失败不阻塞主流程。

        参数：
            role: 消息角色（"assistant" / "tool" 等）
            content: 消息文本
            tool_calls: 工具调用列表（assistant 消息用，可省）
            tool_call_id: 对应的工具调用 id（tool 消息用，可省）
            name: 工具名（tool 消息用，可省）

        返回：无。
        """
        if not self.session_store or not self.session_id:
            return
        try:
            self.session_store.append_message(
                self.session_id, role, content or "",
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                name=name,
            )
        except Exception as e:
            logger.warning("持久化 %s 消息失败: %s", role, e)

    async def _dispatch_tool_calls(self, assistant_msg, handle_function_call) -> bool:
        """执行模型要求的这批工具调用，并处理计划审批 + 失败统计 + 停下检查。

        模型的 assistant 消息（含工具调用和思考字段）会先入历史。
        返回 True 表示主循环继续；False 表示工具要求停下了，得退出。

        并发策略：把工具分成两组——
        - safe（安全，标记了可并发：read_file/list_dir 等只读的）用
          asyncio.gather 并发跑
        - unsafe（不安全：write_file/terminal/记忆写入等有副作用的）挨个排队跑
        - 结果按模型原始顺序回填历史（_merge_results_in_order），保证工具
          调用 id 和结果严格一一配对，不破坏「调用和结果必须交替」的消息规矩
        - safe 组里单个失败不连累其他（异常转成 JSON 错误结果）

        参数：
            assistant_msg: 模型的 assistant 消息（含 tool_calls）
            handle_function_call: 实际执行工具的函数

        返回：True = 继续主循环；False = 已请求停下。
        """
        # assistant 消息入历史（DeepSeek 要求回传工具调用时带上思考字段）
        assistant_entry = {
            "role": "assistant",
            "content": assistant_msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_msg.tool_calls
            ],
        }
        rc = getattr(assistant_msg, "reasoning_content", None)
        sig = getattr(assistant_msg, "thinking_signature", None)
        if rc:
            assistant_entry["reasoning_content"] = rc
        if sig:
            assistant_entry["thinking_signature"] = sig
        assistant_entry["_timestamp"] = time.time()
        self.conversation_history.append(assistant_entry)
        # assistant 的工具调用消息也落盘，恢复会话时可重放
        self._persist_session_message(
            "assistant", assistant_entry.get("content") or "",
            tool_calls=assistant_entry.get("tool_calls"),
        )

        # ---- 按安全/不安全分两组 ----
        tool_calls = list(assistant_msg.tool_calls)

        # === 消费流式预执行的结果（已经跑过的调用不再重复执行）===
        # 预执行走的也是完整的工具执行链（含钩子/权限检查）；这里只补上
        # 计划审批 / 失败统计（和 safe 组的后处理同款）。
        preset_processed = []
        try:
            preset = getattr(self, "_streaming_preset_results", None) or {}
        except AttributeError:
            preset = {}
        self._streaming_preset_results = {}
        if preset:
            remaining = []
            for tc in tool_calls:
                if tc.id in preset:
                    content = self._maybe_handle_plan_approval(tc, preset[tc.id])
                    self._update_failure_streak(content)
                    preset_processed.append((tc, content))
                else:
                    remaining.append(tc)
            tool_calls = remaining
            if preset_processed:
                logger.info(
                    "流式预执行命中 %d 个工具（跳过重复执行）", len(preset_processed),
                )

        safe_calls = []
        unsafe_calls = []
        for tc in tool_calls:
            entry = registry.get(tc.function.name)
            is_safe = bool(entry.isConcurrencySafe) if entry else False
            # terminal 工具整体按不安全处理（保守
            # 串行），但单条调用看命令内容——只读命令（git status/ls/cat 等）
            # 放进并发组。工具级的并发安全标记不动，只在这里分组时放宽。
            if not is_safe and tc.function.name == "terminal":
                try:
                    _t_args = json.loads(tc.function.arguments or "{}")
                    from agent.permission import is_readonly_command
                    if is_readonly_command(str(_t_args.get("command", ""))):
                        is_safe = True
                except Exception:
                    pass
            if is_safe:
                safe_calls.append(tc)
            else:
                unsafe_calls.append(tc)

        # ---- safe 组：先按顺序跑执行前的记录/通知回调，再并发执行工具。
        # 异常不抛出而是收集起来（单个失败转成 JSON 错误，不连累其他）。
        safe_results_raw = await self._run_safe_group_concurrently(
            safe_calls, handle_function_call,
        )

        # ---- unsafe 组：挨个串行执行，保留计划审批 / 失败统计等完整逻辑 ----
        unsafe_results = []
        for idx, tc in enumerate(unsafe_calls):
            # 批内商量式中断——每跑一个工具前查一次中断标志。
            # 中断后剩下的工具不跑了，补上「error_type=interrupted」的假结果
            # 保住调用与结果的配对完整（全跑完才退出
            # 会在库里留下没有结果的孤儿调用）。
            # getattr 防御：测试用 __new__ 造的轻量 agent 没这些属性
            if getattr(self, "_interrupt_requested", False):
                logger.info(
                    "批内中断：unsafe 工具 %s 及后续 %d 个不再执行",
                    tc.function.name, len(unsafe_calls) - idx,
                )
                unsafe_results.extend(
                    json.dumps({
                        "error": "interrupted by user",
                        "error_type": "interrupted",
                    }, ensure_ascii=False)
                    for _ in unsafe_calls[idx:]
                )
                break
            tool_content = await self._run_unsafe_tool_call(tc, handle_function_call)
            unsafe_results.append(tool_content)

        # ---- 按模型原始调用顺序把结果回填历史 ----
        # safe 组结果也要过一遍计划审批检查和失败统计（很少触发，但 read_file
        # 返回错误也该计入失败连击；退出计划模式是不安全工具，不会出现在
        # safe 组）。
        safe_processed = []
        for tc, raw in zip(safe_calls, safe_results_raw):
            tool_name = tc.function.name
            if isinstance(raw, Exception):
                content = json.dumps({
                    "error": f"concurrent dispatch failed: {raw}",
                    "error_type": "concurrent_dispatch_error",
                }, ensure_ascii=False)
            else:
                content = raw
            # 计划审批检查（safe 组理论上不会触发，但保持对称，防以后工具
            # 分类变化——比如某个只读工具也可能返回「需要计划审批」）
            content = self._maybe_handle_plan_approval(tc, content)
            # 失败统计
            self._update_failure_streak(content)
            safe_processed.append((tc, content))

        unsafe_processed = [(tc, c) for tc, c in zip(unsafe_calls, unsafe_results)]

        # 按模型原始调用顺序合并三路结果并回填历史
        # （预执行的也一起进合并——合并函数按完整调用序列对齐）
        self._merge_results_in_order(
            list(assistant_msg.tool_calls),
            preset_processed + safe_processed,
            unsafe_processed,
        )

        # 工具批结束 → 捞排队的用户输入（临时消息回流，下轮消化）
        self._drain_queued_input()

        # 批间摘要（发出去就不管；已请求停下时不启动）
        if not self._idle_requested:
            start_tool_batch_summary(self, tool_calls, safe_processed, unsafe_processed)

        # 检查「主动停下」标志
        if self._idle_requested:
            logger.info("idle 已请求，退出 run_conversation")
            return False
        return True

    def _run_tool_pre_callbacks(self, tc) -> None:
        """工具执行前的记录/通知类小事（safe 组顺序版和流式预执行共用）。

        真正执行工具前要记账——记最近读的文件、条件技能激活入队、
        记忆互斥标记、通知回调。各回调自带异常保护（fail-open）。

        参数：tc: 工具调用对象。返回：无。
        """
        tool_name = tc.function.name
        try:
            tool_args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            tool_args = {}
        if tool_name == "read_file" and tool_args.get("path"):
            self._record_recent("read", str(tool_args["path"]))
        elif tool_name == "load_skill" and tool_args.get("name"):
            self._record_recent("skill", str(tool_args["name"]))
        # 碰到文件 → 条件技能动态激活（paths 匹配）
        # 先收集后批量（回调里只入队，统一处理见组装消息处）
        if tool_name in ("read_file", "write_file", "str_replace") and tool_args.get("path"):
            queue_skill_activation(self, str(tool_args["path"]))
        # 主代理本轮写过记忆 → 记互斥标记（自动提取要避让）
        if tool_name == "memory" and tool_args.get("action") in ("save", "update"):
            self._memory_touched_this_turn = True
        if self.on_tool_call:
            try:
                self.on_tool_call(tool_name, tool_args)
            except Exception:
                pass

    async def _run_safe_group_concurrently(self, safe_calls, handle_function_call):
        """safe 组的并发执行。

        执行前的记录/通知回调按原顺序先同步跑一遍（都是廉价的记账活，
        不值得进并发），然后 asyncio.gather 并发执行工具。
        异常收集不抛出，保证单个失败不连累其他。

        参数：
            safe_calls: safe 组的工具调用列表
            handle_function_call: 实际执行工具的函数

        返回：结果列表（元素可能是异常对象，由调用方转成 JSON 错误）。
        """
        # 按顺序跑执行前回调（保持「先记录再执行」的语义）
        for tc in safe_calls:
            self._run_tool_pre_callbacks(tc)

        # 并发执行工具
        async def _one(tc):
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
            return await handle_function_call(
                tc.function.name, tool_args,
                session_id=self.session_id,
                memory_store=self.memory_store,
                session_store=self.session_store,
                codeagent_home=self.codeAgent_home,
                tool_call_id=tc.id,
                config=self.config,
                hooks_registry=self.hooks_registry,
                bg_manager=self.bg_manager,
                team_bus=self.team_bus,
                team_coordinator=self.team_coordinator,
                team_name=self.team_name,
                agent_ref=self,
            )

        if not safe_calls:
            return []
        return await asyncio.gather(*[_one(tc) for tc in safe_calls],
                                    return_exceptions=True)

    async def _run_unsafe_tool_call(self, tc, handle_function_call):
        """unsafe 组单个工具的执行（一个跑完再跑下一个）。

        保留完整逻辑：执行前回调 → 真执行 → 计划审批处理 → 失败统计。
        返回最终要回填历史的工具结果文本（可能已被计划审批覆盖）。

        参数：
            tc: 工具调用对象
            handle_function_call: 实际执行工具的函数

        返回：工具结果（JSON 字符串）。
        """
        tool_name = tc.function.name
        try:
            tool_args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            tool_args = {}

        # 压缩后重注入用的记账：最近读过的文件 / 加载的技能
        if tool_name == "read_file" and tool_args.get("path"):
            self._record_recent("read", str(tool_args["path"]))
        elif tool_name == "load_skill" and tool_args.get("name"):
            self._record_recent("skill", str(tool_args["name"]))
        # 碰到文件 → 条件技能动态激活（paths 匹配）
        # 先收集后批量（回调里只入队，统一处理见组装消息处）
        if tool_name in ("read_file", "write_file", "str_replace") and tool_args.get("path"):
            queue_skill_activation(self, str(tool_args["path"]))
        # 主代理本轮写过记忆 → 记互斥标记（自动提取要避让）
        if tool_name == "memory" and tool_args.get("action") in ("save", "update"):
            self._memory_touched_this_turn = True

        if self.on_tool_call:
            try:
                self.on_tool_call(tool_name, tool_args)
            except Exception:
                pass

        result = await handle_function_call(
            tool_name, tool_args,
            session_id=self.session_id,
            memory_store=self.memory_store,
            session_store=self.session_store,
            codeagent_home=self.codeAgent_home,
            tool_call_id=tc.id,
            config=self.config,
            hooks_registry=self.hooks_registry,
            bg_manager=self.bg_manager,
            team_bus=self.team_bus,
            team_coordinator=self.team_coordinator,
            team_name=self.team_name,
            agent_ref=self,
        )

        # 计划审批处理（exit_plan_mode 之类；只有 unsafe 组走这条路径）
        tool_content = self._maybe_handle_plan_approval(tc, result)
        # 失败统计
        self._update_failure_streak(tool_content)
        return tool_content

    def _apply_post_plan_clear(self, plan_text: str) -> None:
        """计划批准后「清空上下文再执行」。

        大白话：调研阶段的一大堆对话执行时用不上，白白占 token——批准后：
        - 把对话历史截成一条 <post_plan_brief> user 消息（含计划全文 +
          「开始执行」指令），调研对话从模型上下文里移走
        - 系统提示词的 context 层重建（stable 段不动 → 稳定段的前缀缓存
          照样命中）
        - 会话库/转录文件的落盘记录不动（完整调研记录还能查、能恢复——
          只清模型上下文，不做不可逆删除）

        出错放行：任何异常只打日志，历史保持现状（宁可不清也不能清错）。

        参数：plan_text: 被批准的计划全文。返回：无。
        """
        try:
            brief = (
                "<post_plan_brief>\n"
                "用户刚批准了以下计划。调研阶段的对话已清空（执行阶段专注执行，"
                "省 token），从现在开始按计划执行。\n\n"
                f"{plan_text}\n\n"
                "</post_plan_brief>"
            )
            self.conversation_history = [{"role": "user", "content": brief}]
            self.invalidate_system_prompt()
            logger.info("T9: post-plan 清空上下文执行，history 截断为计划 brief")
        except Exception as e:
            logger.warning("T9: post-plan 清空失败（fail-open 保留现状）: %s", e)

    def _maybe_handle_plan_approval(self, tc, result):
        """如果工具结果是「需要计划审批」，跑审批回调并给出最终回复内容。

        不是审批请求就原样返回。语义：回调为 None 时默认批准（测试场景）；
        审批后返回 plan_approved / 被拒返回 plan_rejected 的 JSON 内容。

        参数：
            tc: 工具调用对象
            result: 工具返回的结果（JSON 字符串）

        返回：处理后的最终结果文本（非审批请求时就是原 result）。
        """
        try:
            result_data = json.loads(result) if isinstance(result, str) else {}
        except (json.JSONDecodeError, ValueError):
            result_data = {}

        if result_data.get("error_type") != "plan_approval_required":
            return result

        plan_text = result_data.get("plan", "")
        clear_context = False
        try:
            if self.plan_approval_callback is not None:
                cb_result = self.plan_approval_callback(plan_text)
                # 回调返回三元组 (批准?, 反馈, 是否清上下文)；
                # 二元组也兼容（清上下文默认 False）
                if isinstance(cb_result, tuple) and len(cb_result) >= 3:
                    approved, feedback, clear_context = (
                        cb_result[0], cb_result[1], cb_result[2],
                    )
                else:
                    approved, feedback = cb_result
            else:
                approved, feedback = True, ""
        except Exception as cb_exc:
            logger.warning("plan_approval_callback 异常: %s", cb_exc)
            approved = False
            feedback = f"审批回调异常: {cb_exc}"

        if approved:
            self.plan_mode = False
            # 记下批准的计划全文（压缩后恢复注入用）
            self._last_approved_plan = plan_text
            # 清空上下文执行——历史截断成计划指令，调研消息全部让位
            if clear_context and plan_text.strip():
                self._apply_post_plan_clear(plan_text)
                return json.dumps({
                    "plan_approved": True,
                    "context_cleared": True,
                    "message": (
                        "用户已批准计划并清空上下文执行。历史已截断为计划指令"
                        "（完整调研记录在会话库/transcripts 可查），现在开始执行："
                        "用 task_create 列出步骤，每步完成调 task_complete。"
                    ),
                }, ensure_ascii=False)
            return json.dumps({
                "plan_approved": True,
                "message": "用户已批准计划。现在可以开始执行：用 task_create 列出步骤，每步完成调 task_complete，依赖关系用 blocked_by。",
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "plan_rejected": True,
                "feedback": feedback or "用户未提供拒绝原因",
                "message": "用户拒绝了计划。请根据 feedback 修订后重新调 exit_plan_mode。",
            }, ensure_ascii=False)

    def _update_failure_streak(self, tool_content):
        """按单个工具结果更新失败连击计数和最近错误。

        语义：结果里有 error 字段、或退出码非 0 就算失败，连击 +1；
        成功则清零。

        参数：tool_content: 工具结果（JSON 字符串）。返回：无。
        """
        try:
            rd = json.loads(tool_content) if isinstance(tool_content, str) else {}
            is_error = bool(rd.get("error")) or rd.get("exit_code", 0) != 0
        except (json.JSONDecodeError, TypeError):
            is_error = False
        if is_error:
            self._tool_failure_streak += 1
            err_snippet = (
                rd.get("error", "") or str(rd.get("stderr", ""))[:200]
                if isinstance(rd, dict) else ""
            )
            self._last_tool_error = str(err_snippet)[:200]
        else:
            self._tool_failure_streak = 0
            # 任何一个工具没报错 → 本条用户消息内有进展
            #（goal 踢一脚的「最近有成功」判据）
            self._last_turn_had_tool_success = True

    def _merge_results_in_order(self, all_calls, safe_processed, unsafe_processed):
        """按模型原始调用顺序合并 safe/unsafe 两路结果，回填历史并落盘。

        safe 组和 unsafe 组是分开跑的，但模型那边的调用顺序不能乱；
        这里按原顺序对号入座，保证调用 id 和结果严格配对（不破坏
        「调用和结果必须交替」的消息规矩）。

        参数：
            all_calls: 模型原始的完整调用列表（顺序基准）
            safe_processed: safe 组结果，List[(tool_call, content)]
            unsafe_processed: unsafe 组结果，List[(tool_call, content)]

        返回：无。
        """
        safe_map = {tc.id: content for tc, content in safe_processed}
        unsafe_map = {tc.id: content for tc, content in unsafe_processed}
        for tc in all_calls:
            if tc.id in safe_map:
                content = safe_map[tc.id]
            elif tc.id in unsafe_map:
                content = unsafe_map[tc.id]
            else:
                # 理论上不该走到这：缺结果的兜底（补个错误结果，防空结果
                # 破坏 API 要求的调用-结果配对）
                content = json.dumps({
                    "error": "missing result for tool_call_id={}".format(tc.id),
                    "error_type": "missing_tool_result",
                }, ensure_ascii=False)
            self.conversation_history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": content,
                "_timestamp": time.time(),
            })
            self._persist_session_message(
                "tool", content, tool_call_id=tc.id, name=tc.function.name,
            )

    async def _finalize_response(self, assistant_msg, user_message: str) -> str:
        """处理「模型不再调工具」的最终回答：入历史 + 跑 STOP 钩子 + 触发反思。

        本方法是 async——STOP 钩子链挪到线程里跑，
        免得慢钩子冻住事件循环。如果 STOP 钩子注入了强制消息，会置
        self._stop_hook_forced = True，主循环看到这个标志要继续跑而不是返回。

        空响应处理（防「突然断开」的坑）：
        - 正文空 + 思考内容有值 → 拿思考内容当回复（思考模型的纯思考输出）
        - 全空（正文和思考都没字）→ 给一句友好的兜底说明（绝不静默返回空串）

        参数：
            assistant_msg: 模型的 assistant 消息
            user_message: 本轮用户消息（反思/记忆同步用）

        返回：最终的回答文本。
        """
        final_content = assistant_msg.content or ""

        # 空响应处理：先用思考内容顶上，全空再给兜底
        if not final_content:
            reasoning = getattr(assistant_msg, "reasoning_content", None)
            if reasoning:
                # 思考模型的纯思考响应：思考内容对用户有价值，直接当回复
                final_content = (
                    "[模型只产出了思考过程，未给最终回复。以下是思考内容：]\n\n"
                    f"<thinking>\n{reasoning}\n</thinking>"
                )
                logger.info(
                    "LLM 返回纯 thinking 响应（content 空 + reasoning_content 有值），"
                    "用 reasoning 作为回复"
                )
            else:
                # 完全空响应（正文和思考都没字）：友好兜底。
                # 绝不静默返回空串——用户会看到「突然断开」
                # 以为 agent 崩了
                final_content = (
                    "[LLM 返回了空响应（content 和 reasoning_content 都为空）。"
                    "可能是网络抖动、流式断连或 provider bug。请重试。]"
                )
                logger.warning(
                    "LLM 返回完全空响应（content + reasoning_content 都空），"
                    "可能是网络问题或模型 bug；返回友好兜底消息（不静默返回空串）"
                )

        self.conversation_history.append({
            "role": "assistant",
            "content": final_content,
            "_timestamp": time.time(),
        })

        if self.on_response:
            try:
                self.on_response(final_content)
            except Exception:
                pass

        # 异步同步到外部记忆服务（不阻塞返回）
        self._sync_memory(user_message, final_content)

        # STOP 钩子（可能注入强制消息让循环继续跑）
        self._stop_hook_forced = False
        if (self.hooks_registry
                and self.config.get("hooks", {}).get("enabled", True)
                and self._stop_fire_count < self.config.get(
                    "hooks", {}).get("stop_hook_max_fires", 3)):
            try:
                # 挪到线程里跑，别冻住事件循环
                force_msg = await asyncio.to_thread(
                    self.hooks_registry.run_stop,
                    session_id=self.session_id or "",
                    max_fires=self.config.get("hooks", {}).get(
                        "stop_hook_max_fires", 3),
                )
            except Exception as e:
                logger.warning("STOP hook 编排异常: %s", e)
                force_msg = None

            if force_msg:
                self._stop_fire_count += 1
                self.conversation_history.append({
                    "role": "user",
                    "content": f"[stop_hook]: {force_msg}",
                })
                self._stop_hook_forced = True
                return final_content  # 主循环看到标志会跳回去继续跑

        # 任务级反思（后台跑，不阻塞返回）
        if self._reflection_enabled:
            try:
                from agent.reflection import trigger_reflection_async
                trigger_reflection_async(self)
            except Exception as e:
                logger.warning("触发反思失败（不影响主流程）: %s", e)

        return final_content

    def _extract_partial_result(self) -> str:
        """被中断时把「已经做完的部分」捞出来。

        做法：找最后一条有内容的 assistant 消息，加上 [PARTIAL] 前缀——
        父代理看到前缀就知道「这是被掐断的半成品，不是完整结果」；
        没有任何 assistant 消息就返回空串。历史状态异常也返回空串不抛错，
        保证父代理的取消路径不因子代理内部状态坏掉而崩。

        使用场景：
            1. 取消旗被举起 → 主循环退出时调这个
            2. 父代理的取消路径要拿部分结果时调这个

        参数：无。返回：带 [PARTIAL] 前缀的部分结果文本（或空串）。
        """
        try:
            history = getattr(self, "conversation_history", None) or []
            for msg in reversed(history):
                if (isinstance(msg, dict)
                        and msg.get("role") == "assistant"
                        and msg.get("content")):
                    content = msg["content"]
                    if isinstance(content, str) and content.strip():
                        return f"[PARTIAL] {content}"
            return ""
        except Exception as e:
            logger.warning("_extract_partial_result 异常（fail-open）: %s", e)
            return ""

    def _update_prevent_sleep(self) -> None:
        """看一眼忙不忙，忙就按住电脑不让它休眠。

        每轮 while 开头调。goal 进行中或有后台任务在跑就算「忙」；
        忙且 config 开了防休眠就 acquire，闲了就 release。用实例标志保证
        只在忙闲切换时真正调——每轮都调的话引用计数会无限涨。
        任何异常由调用处兜底（防休眠失败不影响主对话）。

        参数：无。返回：无。
        """
        try:
            from agent import prevent_sleep
        except Exception:
            return  # 失败放行：模块都加载不了就别硬撑（理论上不会发生）
        try:
            _busy = bool(
                (self._goal_state is not None
                 and self._goal_state.status == "active")
                or (self.bg_manager is not None and any(
                    t.status == "running" for t in self.bg_manager.list_tasks()
                ))
            )
            enabled = (self.config or {}).get("security", {}).get(
                "prevent_sleep", True
            )
            if _busy and enabled:
                if not getattr(self, "_prevent_sleep_held", False):
                    prevent_sleep.acquire("busy")
                    self._prevent_sleep_held = True
            elif getattr(self, "_prevent_sleep_held", False):
                # 非 Windows 上 release 返回 False 是正常空操作，不算错
                prevent_sleep.release("busy")
                self._prevent_sleep_held = False
        except Exception as e:
            logger.debug("prevent_sleep 状态切换 fail-open: %s", e)

    def _emit_loop_exit_trace(self, reason: str, **extra) -> None:
        """把循环退出原因记进轨迹 sink（失败放行；没配 sink 就什么都不做）。

        参数：reason: 退出原因字符串；**extra: 附加字段（如 api_calls）。
        返回：无。
        """
        if self._trace_sink is None:
            return
        try:
            self._trace_sink.emit("loop_exit", reason=reason, **extra)
        except Exception as e:
            logger.debug("loop_exit trace fail-open: %s", e)

    def _llm_retry_heartbeat(self, elapsed: float, total: float) -> None:
        """LLM 长退避期间的心跳（超过 30 秒的退避才触发，每 30 秒一跳）。

        目的：等几分钟重试的时候，用户别以为 agent 死了。
        只记 info 日志（进日志文件和轨迹），不弹通知（30 秒一弹会刷屏）。

        参数：
            elapsed: 已等待秒数
            total: 预计总共要等多久

        返回：无。
        """
        logger.info("LLM 重试退避中：已等待 %.0fs / 预计共 %.0fs", elapsed, total)

    def _discard_partial_stream_state(self) -> None:
        """流式失败后显式扔掉半截累积状态（防御性的「墓碑」清理）。

        说明：半截增量只进 UI 回调、不进历史（那些变量是流式函数的
        局部变量，出作用域自己就没了），所以这个方法目前是保险带——万一
        以后有人把增量提前塞进历史或暂存区，这里负责清痕迹 + 打日志提醒，
        顺手清空流式预执行结果。

        参数：无。返回：无。
        """
        self._streaming_preset_results = {}
        logger.debug("流式失败：半截增量已丢弃（不入 history）")

    def _handle_loop_exit(self, turn_exit_reason: str, user_message: str) -> str:
        """循环退场（预算耗尽或中断）时的兜底回复。

        按 LoopExitReason 枚举生成对应消息（旧的三个值行为不变，
        新增的细分原因各有各的说法）。

        参数：
            turn_exit_reason: 退出原因（LoopExitReason 的值）
            user_message: 本轮用户消息（记忆同步用）

        返回：兜底回复文本（同时已入历史、已触发对应钩子）。
        """
        self._emit_loop_exit_trace(turn_exit_reason)
        if turn_exit_reason == LoopExitReason.INTERRUPTED:
            fallback = "[已被用户中断]"
        elif turn_exit_reason == LoopExitReason.MODEL_ERROR:
            fallback = (
                "[LLM 调用失败，本轮已中断] 重试或检查模型连接。"
                "详见日志（LLM API 调用失败）。"
            )
            # 触发 STOP_FAILURE 钩子（和正常收尾的 STOP 区分开，给审计/告警用）
            self._trigger_stop_failure_hook(
                error="LLM 调用失败", error_type="LLMError",
            )
        elif turn_exit_reason == LoopExitReason.PROMPT_TOO_LONG:
            fallback = (
                "[上下文超限（自动压缩未成功），本轮已中断] "
                "可用 /compact 手动压缩后重试。"
            )
            self._trigger_stop_failure_hook(
                error="prompt_too_long", error_type="ContextLengthError",
            )
        elif turn_exit_reason == LoopExitReason.STREAM_IDLE:
            fallback = (
                "[流式响应超时（自动恢复未成功），本轮已中断] 请重试。"
            )
            self._trigger_stop_failure_hook(
                error="stream_idle_timeout", error_type="StreamIdleTimeout",
            )
        elif turn_exit_reason == LoopExitReason.IDLE_REQUESTED:
            fallback = "[已按请求停止本轮]"
        elif turn_exit_reason == LoopExitReason.BUDGET_EXHAUSTED:
            fallback = "[迭代预算耗尽，强制停止]"
        else:
            # normal / max_turns（旧行为）
            fallback = "[已达最大迭代次数，强制停止]"

        self.conversation_history.append({
            "role": "assistant",
            "content": fallback,
            "_timestamp": time.time(),
        })
        # 同步到外部记忆服务（即使被打断也保留部分上下文）
        self._sync_memory(user_message, fallback)
        return fallback

    def _trigger_stop_failure_hook(self, *, error: str, error_type: str) -> None:
        """触发 STOP_FAILURE 钩子（失败放行，异常不影响主流程）。

        参数：
            error: 错误描述
            error_type: 错误类型标记

        返回：无。
        """
        if not self.hooks_registry:
            return
        if not self.config.get("hooks", {}).get("enabled", True):
            return
        try:
            self.hooks_registry.run_stop_failure({
                "session_id": self.session_id or "",
                "error": error,
                "error_type": error_type,
            })
        except Exception as e:
            logger.warning("STOP_FAILURE hook 编排异常: %s", e)

    async def chat(self, message: str, cancel_event=None) -> str:
        """简单接口：发一条消息，拿回答。就是个薄壳，转给 run_conversation。

        参数：
            message: 用户消息文本
            cancel_event: 可选的取消旗（子代理场景传，让父代理能叫停）

        返回：助手回答文本。
        """
        if cancel_event is not None:
            return await self.run_conversation(message, cancel_event=cancel_event)
        return await self.run_conversation(message)

    def _sync_memory(self, user_message: str, assistant_message: str) -> None:
        """把这一轮对话同步到外部记忆服务（如果配了）。

        内置的文件记忆（Memory.md 那套）由记忆工具主动写；这里只管
        外部服务的逐轮同步。出错只打 debug 日志。

        参数：
            user_message: 用户消息
            assistant_message: 助手回答

        返回：无。
        """
        if self.memory_manager is None:
            return
        try:
            self.memory_manager.sync_all(user_message, assistant_message)
        except Exception as e:
            logger.debug("memory_manager.sync_all 失败: %s", e)
