"""配置管理模块——整个项目所有配置默认值的唯一源头。

这个文件是干嘛的：
  存一份"出厂默认配置"（DEFAULT_CONFIG 这个大字典），再把它和用户的
  settings.json 合并，得到程序实际用的配置。

配置怎么算出来的（默认路径下）：
  最终值 = DEFAULT_CONFIG（出厂默认）← 深合并 ← settings.json（~/.OmniMate/settings.json）
  - settings.json 由 agent/settings.py 负责读写；第一次启动时会自动把
    config.yaml / .env / .mcp.json 迁进 settings.json
  - 如果调用时显式传了 config_file 指向某个 yaml 文件，就走 yaml
    加载逻辑（现在基本只有测试在用）
  - settings.json 读取失败时，也会退回 yaml 逻辑兜底

什么是"深合并"（deep-merge），为什么不用整体替换：
  比如默认配置里有 context 整段几十个键，用户只想改其中一个阈值。
  整体替换会把其他几十个键全弄丢；深合并是逐层递归地只覆盖用户
  写了的字段。好处：
  - 新版本程序加了新字段，用户的旧配置也能自动补上默认值
  - 只有重命名键、改动结构这种"伤筋动骨"的变更才需要把
    _config_version 版本号 +1 触发迁移

注意：save_config/save_config_value 写的是 yaml。运行时真正想
持久化配置，唯一正道是 agent/settings.py 的 save_settings——
写 yaml 是"断轨"的，写完下次启动也读不回来。
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from constants import config_path as _config_path

logger = logging.getLogger(__name__)


# 配置结构版本号：只有重命名键/改结构这种大改动才 +1（加了新字段不用动它）
_config_version = 1


DEFAULT_CONFIG: Dict[str, Any] = {
    "_config_version": _config_version,

    # 输出风格名。None = 关闭。
    # 谁在读它：AIAgent._get_system_prompt → resolve_output_style
    # （注意这是真实有代码在读的键，不是摆设）；/output-style 命令负责写入。
    "output_style": None,

    # 模型配置（默认 DeepSeek，可换成 OpenAI/OpenRouter/Anthropic 等）
    "model": {
        "provider": "deepseek",
        "name": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0.7,
        "max_tokens": None,
        "fallback_model": None,
    },

    # 辅助 LLM 配置（可选）：给检索/上下文压缩/记忆提取这些"打杂"任务
    # 用一个便宜模型跑，省 token。没配置（None）就直接用主模型。
    "aux_model": None,

    # 输出语言（"zh" 中文 / "en" 英文），影响 system prompt 里
    # "我是谁"和"输出约定"这两段的措辞
    "language": "zh",

    # Agent 主循环行为
    "agent": {
        "max_iterations": 200,
        "compression_enabled": True,
        "system_prompt": None,
        # 流式并发执行：模型还在边想边吐字时，就把"安全"的工具先偷偷跑了
        # （等模型说完直接用结果，省一轮等待）。默认关——还在灰度验证，
        # 确认稳定后才考虑默认打开
        "streaming_tool_execution": False,
    },

    # 上下文压缩管线：对话太长时一层层瘦身，防止撑爆模型的上下文窗口
    "context": {
        # ── L3 落盘层：把超大块内容挪到磁盘，上下文里只留预览 ──
        # 单个工具输出超过 5 万字符就落盘（阈值 per-tool）
        "output_offload_threshold": 50000,
        "output_offload_preview": 2000,
        "message_offload_threshold": 200000,        # 单条消息累计超过 20 万字符就落盘
        "offload_decision_freeze": True,            # 落盘决定跨轮次不变（反复变会让缓存失效）
        # ── L1 裁剪层 ──
        # 是否压缩主要看 token 量，消息条数阈值放宽，
        # 避免动不动就把中间的对话裁掉
        "snip_message_threshold": 200,
        "snip_release_threshold": 30,
        "snip_keep_first": 3,
        "snip_keep_last": 47,
        # ── L2 微压缩：按单条消息大小折叠老的工具结果，最近 3 条不动 ──
        "micro_keep_recent_results": 3,
        # 按时间清理：距最后一条 AI 回复超过 60 分钟的旧工具结果，
        # 替换成"已清理"标记（最近 5 条保留）
        "time_based_mc_enabled": True,
        "time_based_mc_gap_minutes": 60,
        "time_based_mc_keep_recent": 5,
        # ── L4 大压缩：让 LLM 把整段对话总结成摘要 ──
        # 10 万 token 约等于 30 万字符。DeepSeek/OpenAI 的上下文上限
        # 在 64K-128K 之间。消息条数阈值只是兜底——真正的主判据是
        # token 逼近窗口（100 万 token 的模型可放宽到 2000 条）
        "llm_compact_token_threshold": 100000,
        "llm_compact_message_threshold": 500,
        "llm_compact_keep_recent": 30,
        "llm_compact_cooldown_turns": 5,
        # 不设"每会话最多压缩 N 次"的总量帽——长会话用完次数后会永久
        # 失去 L4，只能频繁紧急截断，得不偿失；只留冷却轮数 + 连续失败熔断
        # 单轮增长预估（防"压完马上又涨回去"的震荡）：判断条件是
        # "当前量 + 预计下一轮增量 >= 阈值"就提前压；增量取最近
        # growth_window 轮里最猛的一轮，历史数据不够就用默认值
        "llm_compact_growth_window": 3,
        "llm_compact_growth_default": 8000,
        # 9 段摘要 Files/Errors 段字数分档：被摘要消息数过 (60, 150) 阈值
        # 就把上限从 200 放宽到 400/600（长任务路径+报错多，200 字装不下）
        "summary_scale_thresholds": [60, 150],
        "summary_files_errors_limits": [200, 400, 600],
        # 压缩后恢复信息的统一 token 预算：计划/异步状态 > 最近文件 >
        # 技能正文，预算不够就按这个优先级从低往高砍
        "post_compact_recovery_budget": 40000,
        # ── Reactive 被动压缩：上下文已经超了才救火 ──
        # 支持一救再救，但有冷却窗口防失控
        "reactive_keep_recent": 10,
        "reactive_compact_cooldown_seconds": 60,   # 60 秒内最多救一次
        "reactive_compact_max_per_session": 5,     # 单会话最多救 5 次（防死循环）
        "reactive_once_per_session": False,        # 已废弃的"每会话只救一次"开关（别开）
        # 工具批间摘要：让辅助模型一句话总结这批工具结果，下一轮再悄悄
        # 注入，把"出结果"和"看结果"的时间差藏起来。默认关
        "tool_batch_summary_enabled": False,
        # 对话轨迹落盘
        "transcript_enabled": True,
        "transcript_trigger": "pre_llm_compact",
        "transcript_retention": 20,
        # 缓存失效原因 diff 文件最多留 100 个，超了删最旧的（LRU）
        "max_cache_break_diff_files": 100,
        # 压缩后主动恢复：把最近用过的文件和技能重新塞回上下文，
        # 免得压缩把关键信息剪没了
        "post_compact_recovery_enabled": True,          # 总开关（False = 压缩后不重注入）
        "post_compact_recovery_max_files": 5,            # 最多恢复几个最近文件
        "post_compact_recovery_max_skills": 5,           # 最多恢复几个最近技能
    },

    # Hook 系统：在固定事件点（如工具调用前后）自动执行用户配置的脚本
    "hooks": {
        "enabled": True,                            # 总开关；False 时所有 hook 都不执行
        "settings_path": None,                      # None → 默认 ~/.OmniMate/.hooks/settings.json
        "script_timeout_default": 10.0,            # 声明式 hook 默认超时（秒）
        "stop_hook_max_fires": 3,                  # Stop hook 每会话最多触发 3 次（防失控）
        "fail_closed_default": False,              # 声明式 hook 默认不启用 fail_closed
    },

    # 后台任务：让命令在后台跑，不堵主对话
    "bg_task": {
        "enabled": True,                        # False 时 bg_* 系列工具对模型隐藏
        "max_concurrent": 5,
        "default_timeout": 600,                 # bg_start 默认超时（秒）
        "notification_stdout_cap": 500,         # 桌面通知里最多带 500 字符输出
        "result_stdout_cap": 5000,              # bg_result 返回的输出最多 5000 字符
        "default_detach": False,                # bg_start 默认不脱离（detach）
        "stall_timeout": 45.0,                  # 停滞看门狗（秒）。0=禁用；开启后连续这么
                                                # 多秒输出没动静就弹通知，让模型自己判断咋办
        "idle_wake": True,                      # 主对话空闲时后台任务/异步子代理完成
                                                # → 自动唤醒主循环跑一轮处理结果（哨兵走
                                                # 输入队列；False 恢复"等用户下次发消息"）
    },

    # Cron 定时调度：到点自动执行任务
    "cron": {
        "enabled": True,                        # False 时整个定时功能关闭
        "jobs_path": None,                      # None → 默认 ~/.OmniMate/.cron/jobs.json
        "poll_interval_seconds": 30.0,          # 后台线程每隔多少秒看一次表
        "max_age_days": 7,                      # 周期任务超过 7 天没跑就自动停用（防僵尸任务）
    },

    # Team 多 agent 协作：多个 agent 分工干活（像拉了个工作群）
    "team": {
        "enabled": True,                        # False 时 team_* 系列工具对模型隐藏
        "team_dir": None,                       # None → 默认 ~/.OmniMate/.team/
        "default_role": "worker",               # 新成员默认角色
        "spawn_timeout": 600,                   # 启动一个子 agent 的超时（秒）
        "max_members": 10,                      # 单队最多 10 个成员
        # autonomous（自主）模式：成员空闲时自己轮询领活干
        "autonomous_idle_timeout": 60.0,        # 空闲多久算 IDLE（秒）
        "autonomous_poll_interval": 5.0,        # 空闲时隔几秒看一次有没有新活
        "max_depth": 2,                         # 成员最多再往下派几层（防无限套娃）
    },

    # 终端命令执行
    "terminal": {
        "cwd": None,
        "default_timeout": 120,
        "shell": None,
    },

    # 记忆：agent 对用户和环境的长期认知
    "memory": {
        "enabled": True,
        "provider": None,                # 外部记忆服务（如 honcho），None = 用内置方案
        "memory_char_limit": 2200,
        "user_char_limit": 1375,
        "multifile_enabled": True,       # 多文件模式开关
        "memory_dir": None,              # 记忆目录（None → ~/.OmniMate/.memory/）
        "retrieval_enabled": True,       # 按需检索开关（每轮挑相关的记忆注入）
        "retrieval_max_results": 5,      # 每次最多检索几条
        "retrieval_model": None,         # 检索用哪个模型（None → 用主模型）
        # 记忆管理员（curator）：定期在后台整理记忆库的后台工人
        "curator": {
            "enabled": True,             # False 时整个记忆整理关闭
            "interval_hours": 168,       # 每 7 天跑一轮
            "min_idle_hours": 1,         # agent 至少闲 1 小时才跑（不打扰正事）
            "llm_review_enabled": True,  # 第 2 阶段是否让 LLM 复核
            "max_batch_size": 30,        # 每桶每批发给 LLM 的最大条数
            "archive_after_multiplier": 2,  # 归档阈值 = N × 有效天数
        },
        # 对话级轻量记忆提取：辅助模型每 N 回合顺手从对话里捞点事实存下来。
        # 默认关——每回合都多一次辅助模型调用的开销；且和主 agent
        # 主动写记忆互斥（一轮只允许一个写）
        "auto_extract": {
            "enabled": False,
            "every_n_turns": 3,
        },
    },

    # 技能：沉淀成 Markdown 文件的"操作手册"，agent 能自己创建和改进
    "skills": {
        "dir": None,                     # 自定义技能目录
        "external_dirs": [],
        # 技能 frontmatter 的 files 附件：单个文件注入时最多带 8000 字符
        "file_attachment_max_chars": 8000,
    },

    # 技能管理员（curator，跟记忆管理员是两套）：定期整理技能库
    "curator": {
        "enabled": True,
        "interval_hours": 168,           # 每 7 天跑一轮
        "min_idle_hours": 1,
        "stale_after_days": 30,
        "archive_after_days": 90,
        "prune_builtins": False,
        "backup": {
            "enabled": True,
            "max_snapshots": 5,
        },
    },

    # 子代理（主对话派出去帮忙干活的分身）相关配置
    "delegation": {
        "max_concurrent_children": 5,
        # workflow（确定性工作流引擎）的总 token 预算
        # （按子代理产出的 token 估算口径算）
        "workflow_budget_total": 500000,
        "max_spawn_depth": 2,
        "child_timeout_seconds": 600,
        "orchestrator_enabled": True,
        "subagent_auto_approve": False,
        "max_iterations": 50,
        # async（后台跑、不等结果）子代理的工具白名单开关。
        # 默认开——安全优先；关掉的话后台子代理能用全部工具（危险）
        "async_tool_whitelist_enabled": True,
        # 用户追加的禁用列表（在内置黑名单基础上再删工具）
        "async_disallowed_tools": [],
        # worktree（独立 git 工作区）智能清理：
        # True = 一律清理（老行为，不管有没有改动）；
        # False = 有改动就保留现场，没改动才清理
        "worktree_always_cleanup": False,
        # fork 子代理：完整继承父对话的上下文前缀，让 API 前缀缓存
        # 也能命中（省 token）。True 时 subagent(fork=True) 生效
        "fork_subagent_enabled": True,
        # 交接复审：辅助模型检查子代理的产出，危险结果加警告前缀。
        # 默认关——每交接一次多花一次辅助调用
        "handoff_review_enabled": False,
        # fork 时继承父对话最近 N 条 AI 回复（继承越多缓存命中越好，
        # 但也越容易把父上下文的信息带偏子代理，权衡值）
        "fork_max_parent_turns": 3,
        # fork="full"（全量继承）模式最多带多少条 AI 回复（防失控）
        "fork_full_history_max_turns": 50,
        # 子代理对话轨迹落盘到 ~/.OmniMate/.agent-sessions/（便于 resume）
        "subagent_persistence_enabled": True,
        # 已完成子代理的记录保留 N 天，超期清理省磁盘
        "subagent_persistence_retention_days": 7,
        # async 子代理默认拒绝审批：后台子代理遇到需要用户点头的
        # 破坏性命令时直接拒绝（它没法弹窗问你，宁可拒也不放行）。
        # 安全优先，默认开；关掉不推荐
        "async_auto_deny_permission": True,
        # async 子代理用的权限模式名。默认 autoDeny；可改成
        # "default"/"bypassPermissions"/"acceptEdits" 让后台子代理
        # 走别的权限逻辑（仅当上面那个开关为 True 时生效）
        "async_permission_mode": "autoDeny",
        # 同步子代理超时后，先举"取消牌"再等这么多秒让它自己收尾退出
        # （它在每次调 LLM 前都会看一眼牌子，看到就撤，不白烧 token）
        "sync_cancel_timeout_seconds": 2.0,
        # 是否把 subagent_kill（击杀后台子代理）工具暴露给模型
        "async_kill_enabled": True,
    },

    # 安全
    "security": {
        "command_approval": "ask",       # 命令审批策略：ask（问用户）| always | never
        "dangerous_commands": [],
        "redact_secrets": True,
        "permission_mode": "default",    # 权限模式："default" | "bypassPermissions"（全放行）| "acceptEdits"（编辑自动批）
        # terminal 超时上限（秒）——模型传再大的 timeout 也会被压到这个值
        "max_terminal_timeout": 600,
        # OS 沙箱：给命令执行再套一层操作系统级隔离
        "sandbox_mode": "off",           # "off" | "on"（启动时灌进 PermissionChecker）
        "sandbox_writable_roots": [],    # 沙箱里额外允许写的目录（默认已含 cwd + ~/.OmniMate）
        # /add-dir 命令持久化的写白名单：运行时用户加目录就追加到这里
        # 并写回；下次启动 RuntimeContext 读取生效
        "extra_allowed_roots": [],
        # 只读命令快速通道：git status/ls/cat 这类看了不改东西的命令
        # 自动放行 + 可以同轮并发跑（默认开）
        "readonly_fastpath_enabled": True,
        # Windows 防休眠：goal 循环/后台任务跑着的时候保持系统清醒
        # （非 Windows 上自动什么都不做）
        "prevent_sleep": True,
        # http hook 的 URL 白名单：None = 不限制 / [] = 全拒 /
        # 非空 = 必须匹配其中一条（支持 * 通配）；
        # 内网地址段（SSRF）校验无论如何都开着
        "http_hook_allowed_urls": None,
        "http_hook_allowed_env_vars": [],  # http hook 里 ${VAR} 环境变量插值只认这个白名单
    },

    # 显示
    "display": {
        "quiet": False,
        "show_tool_progress": True,
        "spinner": True,
    },

    # 会话
    "sessions": {
        "auto_save": True,
        "auto_title": True,
        "db_path": None,                 # 默认 ~/.OmniMate/sessions.db
    },

    # 启用的工具集：core = 全部内置工具；mcp = 外部工具服务器
    # （动态的，服务器连上了工具才可见）
    "enabled_toolsets": ["core", "mcp"],

    # Plan Mode（先出计划、用户批准、再动手的模式）总开关
    "plan_mode": {
        "enabled": True,  # False 时输 /plan 会报"功能未启用"
    },

    # ────────────────────────────────────────────────────────────
    # 功能开关（feature flags）——新功能先藏在这里灰度
    # ────────────────────────────────────────────────────────────
    # 所有 flag 默认关（用户决策：装完默认全关，但每项都要有测试覆盖）。
    # 用户想开哪个，就在 ~/.OmniMate/settings.json 的 features 节里覆盖。
    # 启动时通过 _deep_merge 读一次，运行中不热加载（保护 prompt 缓存）。
    "features": {
        # Bash 命令 LLM 分类器：拿不准的命令让辅助模型判断危不危险
        "bash_llm_classifier": {
            "enabled": False,
            "model": "aux",          # 用辅助 LLM 跑分类
            "whitelist": [           # 这些命令直接放行，不劳烦模型
                "ls", "ll", "cat", "pwd", "echo",
                "grep", "find", "which", "where",
                "git status", "git diff", "git log", "git show",
                "python --version", "uv --version",
            ],
        },

        # 5 层压缩补强：L4 折叠（上下文用到 80% 触发）
        "context_collapse": {
            "enabled": False,
            "threshold_ratio": 0.8,  # 上下文用到 80% 触发
        },

        # 5 层压缩补强：L5 响应式回压（超限后救火）
        "reactive_compact": {
            "enabled": False,
        },

        # Bash 无人值守持久重试：没人盯着时命令失败了也一直重试
        "bash_unattended_retry": {
            "enabled": False,
            "max_hours": 24,         # 最多持续重试 24 小时
        },

        # MCP 走 HTTP/SSE 传输（而不只是本地进程）
        "mcp_http_transport": {
            "enabled": False,
            "default_timeout_sec": 30,
        },

        # MCP 走 WebSocket 传输
        "mcp_websocket_transport": {
            "enabled": False,
        },

        # Plan Mode V2：计划阶段多个 agent 并行干活
        "plan_mode_v2_parallel": {
            "enabled": False,
            "max_parallel_agents": 3,  # 最多同时 3 个（防失控）
        },

        # Hook handler 类型扩展：支持 http / mcp 工具 / agent 三种新形态
        "hook_http_handler": {
            "enabled": False,
        },
        "hook_mcp_tool_handler": {
            "enabled": False,
        },
        "hook_agent_handler": {
            "enabled": False,
        },
    },

    # ── goal / trace / handoff 三件套 ──
    # goal：给 agent 定个目标，它自己多轮推进直到完成（/goal 命令触发）
    "goal": {
        "enabled": True,
        "default_token_budget": 200_000,  # 默认预算上限（防无限烧 token）
        "reflection_interval": 5,          # 每 5 轮停下来复盘一次
    },

    # trace：本地轨迹记录（jsonl 文件落盘，/trace 命令查看）
    "trace": {
        "enabled": True,
        "retention_days": 7,  # 超过 7 天的轨迹文件自动清理
    },

    # statusline：每次 AI 回复完在末尾打一行紧凑状态
    #（用的什么模型/本会话烧了多少 token/goal 进度/项目名）
    "statusline": {
        "enabled": True,
    },

    # handoff（会话移交：把当前对话打包，换个会话/机器接着干）
    "handoff": {
        "auto_save_on_exit": True,   # 退出时自动打包一份
        "auto_save_max_keep": 20,    # 自动打包最多留 20 份（防占满磁盘）
    },

    # 桌面通知（Windows 右下角弹窗，用系统自带 PowerShell 实现，零依赖）。
    # 管着所有 notify() 调用：后台任务完成 / 权限审批 / goal 暂停三类时机
    "notifications": {
        "enabled": True,  # False 时所有 notify() 直接返回 False（不弹 PowerShell）
    },

    # skill_learning（行为学习）：agent 从自己的操作轨迹里总结出
    # "instinct"（直觉经验）并演化。默认关——学习链路要显式开启才跑。
    # 总开关管两件事：轮末观察 + 经验簇达标后演化
    "skill_learning": {
        "enabled": False,
        "observer": "heuristic",    # 观察方式："heuristic"（正则规则）|"llm"（辅助模型，失败回退正则）
        "evolve_threshold": 0.75,   # 经验簇平均置信度达到 0.75 才演化
        "evolve_min_cluster": 3,    # 经验簇至少攒够 3 条才演化
    },
}


# ---------------------------------------------------------------------------
# 环境变量元数据：.env 文件里允许放哪些密钥（用于引导用户配置）
# ---------------------------------------------------------------------------

OPTIONAL_ENV_VARS: Dict[str, dict] = {
    "DEEPSEEK_API_KEY": {
        "description": "DeepSeek API 密钥",
        "prompt": "DeepSeek API Key",
        "url": "https://platform.deepseek.com/api_keys",
        "password": True,
        "category": "provider",
    },
    "OPENAI_API_KEY": {
        "description": "OpenAI API 密钥",
        "prompt": "OpenAI API Key",
        "url": "https://platform.openai.com/api-keys",
        "password": True,
        "category": "provider",
    },
    "OPENROUTER_API_KEY": {
        "description": "OpenRouter API 密钥（多模型聚合）",
        "prompt": "OpenRouter API Key",
        "url": "https://openrouter.ai/keys",
        "password": True,
        "category": "provider",
    },
    "ANTHROPIC_API_KEY": {
        "description": "Anthropic API 密钥",
        "prompt": "Anthropic API Key",
        "url": "https://console.anthropic.com/",
        "password": True,
        "category": "provider",
    },
    "OMNIMATE_HOME": {
        "description": "agent home 目录（覆盖默认 ~/.OmniMate）",
        "prompt": "Agent Home",
        "password": False,
        "category": "system",
    },
}


# ---------------------------------------------------------------------------
# 路径辅助
# ---------------------------------------------------------------------------

# 说明：路径函数（get_omnimate_home / config_path / env_file）统一在
# constants.py。这里留个 config_path 别名纯粹是本模块自己用着
# 方便——外部代码请直接 import constants。

def config_path() -> Path:
    """拿到配置文件 config.yaml 的完整路径（实现在 constants.py，这里是本模块内部用的别名）。

    返回：
        Path 对象，指向 ~/.OmniMate/config.yaml（或 OMNIMATE_HOME 覆盖后的位置）。
    """
    return _config_path()



# ---------------------------------------------------------------------------
# 加载 / 保存
# ---------------------------------------------------------------------------

def load_config(
    config_file: Optional[Path] = None,
    *,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """加载配置，返回程序实际使用的那份配置字典。

    把出厂默认值、用户配置、命令行临时覆盖三层捏成一份。
    不传 config_file 时优先读 settings.json；传了就按 yaml 逻辑走
    （主要是测试在用）；settings.json 读挂了也会退回 yaml 兜底。

    参数：
        config_file：可选，显式指定一个 yaml 配置文件路径。
            不传（默认）就走 settings.json 逻辑；传了就强制走 yaml 逻辑。
        cli_overrides：可选，命令行临时覆盖项（字典）。
            优先级最高，最后合并进来。

    返回：
        合并好的配置字典（dict）。兼容形态的 config["model"]["name"]
        等字段会被构造好放进去。
    """
    # 默认路径：优先读 settings.json（新配置体系）
    if config_file is None:
        try:
            from agent.settings import (
                load_settings, get_current_model_config,
            )
            settings = load_settings()
            model_cfg = get_current_model_config(settings)

            config = dict(settings)
            # 拼一个兼容的 model 段：settings.json 的模型结构不同，
            # 读 config["model"]["xxx"] 的地方靠这里翻译成兼容形状
            config["model"] = {
                "provider": model_cfg.get("name", "opus"),
                "name": model_cfg.get("model", ""),
                "base_url": model_cfg.get("base_url"),
                "api_key": model_cfg.get("api_key", ""),
                "auth_token": model_cfg.get("auth_token", ""),
                "api_key_env": "",
                "format": model_cfg.get("format", "anthropic"),
                "effort_level": model_cfg.get("effort_level", ""),
                "api_timeout_ms": model_cfg.get("api_timeout_ms"),
            }

            # 新模式（llm 段）下额外注入 haiku 级轻量模型配置，
            # delegate_tool（子代理委派工具）要读它
            llm_cfg = settings.get("llm", {})
            if llm_cfg:
                haiku_name = settings.get("default_haiku_model", "haiku")
                haiku_model_name = llm_cfg.get(f"{haiku_name}_model")
                if haiku_model_name:
                    config["haiku_model"] = {
                        "format": "anthropic",
                        "base_url": llm_cfg.get("base_url"),
                        "auth_token": llm_cfg.get("auth_token", ""),
                        "model": haiku_model_name,
                    }

            if cli_overrides:
                config = _deep_merge(config, cli_overrides)
            return config
        except Exception as e:
            logger.warning("读 settings.json 失败，fallback 到 yaml: %s", e)

    # yaml 逻辑：传了 config_file 参数，或上面 settings.json 读失败了
    config = copy.deepcopy(DEFAULT_CONFIG)

    # 2. 加载用户 config.yaml（存在的话）
    if config_file is None:
        config_file = config_path()

    config_file = Path(config_file)
    if config_file.exists():
        try:
            user_config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            if user_config and isinstance(user_config, dict):
                # 深合并：只覆盖用户写了的字段，其他默认值保留
                config = _deep_merge(config, user_config)
        except Exception as e:
            logger.warning("加载配置失败 %s: %s", config_file, e)

    # 3. 应用命令行覆盖（优先级最高）
    if cli_overrides:
        config = _deep_merge(config, cli_overrides)

    # 4. 版本不匹配时做结构迁移
    config = _maybe_migrate(config)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并两个字典（深合并），返回一份新字典。

    规则（逐层下钻，避免用户没写的字段丢默认值）：
    - 两边都是字典 → 递归继续合
    - 其他情况 → 用 override（用户配置）的值直接覆盖
    - override 里的 None 不覆盖（让默认值有机会生效）

    参数：
        base：基础字典（通常是出厂默认配置）。
        override：覆盖字典（通常是用户配置），它的值优先。

    返回：
        合并结果，一个全新的深拷贝字典（不动入参）。
    """
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if value is None:
            continue  # None 不覆盖（见上）
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _maybe_migrate(config: dict) -> dict:
    """配置结构迁移：版本号对不上时把旧结构改成新结构。

    只有"重命名键、改 schema"这种结构性变更才需要迁移；
    单纯新增字段不需要（深合并会自动补默认值）。

    参数：
        config：刚加载完的用户配置字典。

    返回：
        迁移后的配置字典（版本一致时原样返回）。
    """
    user_version = config.get("_config_version", 1)
    if user_version == _config_version:
        return config

    # 未来的迁移逻辑写这里，例如：
    # if user_version < 2:
    #     ...

    return config


def save_config(config: dict, config_file: Optional[Path] = None) -> None:
    """把配置字典写回 config.yaml 文件（遗留路径，运行时别用）。

    运行时改配置应走 agent/settings.py 的 save_settings（写 settings.json），
    写 yaml 下次启动读不回来。

    参数：
        config：要保存的完整配置字典。
        config_file：可选，目标 yaml 路径；不传用默认路径。
    """
    if config_file is None:
        config_file = config_path()

    config_file = Path(config_file)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # 值为 None 的键不写进文件——留空让默认值在下次加载时生效
    clean = _remove_none(config)

    yaml_text = yaml.dump(
        clean,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    config_file.write_text(yaml_text, encoding="utf-8")


def _remove_none(obj):
    """递归删掉字典/列表里所有值为 None 的条目。

    参数：
        obj：任意结构（dict / list / 其他）。

    返回：
        清理后的新结构（dict/list 会被重建，其他原样返回）。
    """
    if isinstance(obj, dict):
        return {k: _remove_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_remove_none(i) for i in obj if i is not None]
    return obj


def ensure_default_config() -> None:
    """配置文件不存在时，把出厂默认配置写一份到磁盘（首启生成可编辑模板）。"""
    cf = config_path()
    if not cf.exists():
        cf.parent.mkdir(parents=True, exist_ok=True)
        save_config(DEFAULT_CONFIG, cf)
        logger.info("已写入默认配置: %s", cf)


# ---------------------------------------------------------------------------
# dotpath 访问：用 "model.name" 这种点分字符串读写嵌套配置
# ---------------------------------------------------------------------------

def save_config_value(dotpath: str, value: Any, config_file: Optional[Path] = None) -> None:
    """用点分路径保存单个配置值（写 yaml，遗留路径）。

    参数：
        dotpath：点分键路径，比如 "model.name" 会写 config["model"]["name"]。
        value：要写入的值。
        config_file：可选，目标 yaml 路径；不传用默认路径。
    """
    config = load_config(config_file)
    keys = dotpath.split(".")
    target = config
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value
    save_config(config, config_file)


def get_config_value(config: dict, dotpath: str, default: Any = None) -> Any:
    """用点分路径读配置值，读不到就返回默认值。

    参数：
        config：配置字典。
        dotpath：点分键路径，如 "model.name"。
        default：可选，路径不通或值是 None 时返回它。

    返回：
        查到的值；查不到（中间层不是字典、键不存在、值为 None）返回 default。

    示例：get_config_value(config, "model.name") → "deepseek-chat"
    """
    keys = dotpath.split(".")
    value = config
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
        if value is None:
            return default
    return value
