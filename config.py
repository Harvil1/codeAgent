"""配置管理（完整版）。

三层配置：
  优先级（从低到高）：
    1. DEFAULT_CONFIG（代码内置默认值）
    2. config.yaml（用户配置文件）
    3. 命令行参数（临时覆盖）

最终值 = DEFAULT_CONFIG ← deep-merge ← config.yaml ← CLI 参数

deep-merge 策略：
  - DEFAULT_CONFIG 是基础
  - 用户的 config.yaml 覆盖（而不是替换）对应字段
  - 新增字段不需要 bump _config_version
  - 重命名/结构性变更才需要 bump
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from constants import config_path as _config_path

logger = logging.getLogger(__name__)


# 当前配置版本（结构性变更才 bump）
_config_version = 1


DEFAULT_CONFIG: Dict[str, Any] = {
    "_config_version": _config_version,

    # 模型配置（默认 DeepSeek，可切换到 OpenAI/OpenRouter/Anthropic 等）
    "model": {
        "provider": "deepseek",
        "name": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0.7,
        "max_tokens": None,
        "fallback_model": None,
    },

    # batch2-T3: 辅助 LLM 配置（可选，用于检索/压缩/记忆提取等辅助任务）
    # 未配置（None）时 fallback 到主 model 配置
    "aux_model": None,

    # B1 NEW: vision/image 工具配置（image_analyze / image_ocr）
    "vision": {
        "enabled": True,
        "provider": "",          # 空字符串 = 用主 provider
        "model": "",             # 空字符串 = 用主 model（假设支持 vision）
        "max_bytes": 20 * 1024 * 1024,  # 20 MB
    },

    # B6 NEW: 输出语言（"zh" 中文 / "en" 英文）
    # 影响 system prompt 的身份声明和输出约定段
    "language": "zh",

    # Agent 行为
    "agent": {
        "max_iterations": 200,
        "compression_enabled": True,
        "system_prompt": None,
    },

    # 上下文压缩管线（Phase 1）
    "context": {
        # L3 offload
        # 改造点 ①：精细化落盘阈值（per-tool 50K + per-message 200K + 决策冻结）
        "output_offload_threshold": 50000,
        "output_offload_preview": 2000,
        "message_offload_threshold": 200000,        # per-message 聚合阈值
        "offload_decision_freeze": True,            # 跨轮次决策冻结（保护 prompt cache）
        # L1 snip
        # 对齐 Claude Code：压缩由 L4 token 主导，消息数阈值放宽避免频繁裁中间
        "snip_message_threshold": 200,
        "snip_release_threshold": 30,
        "snip_keep_first": 3,
        "snip_keep_last": 47,
        # L2 micro（对齐 Claude Code microCompact：按单条大小折叠，保护最近 3 条）
        "micro_keep_recent_results": 3,
        # 改造点 ④：time-based MC（60min 清旧工具结果）
        # 距最后一条 assistant 消息 > gap_minutes 时，把旧 tool result 替换为清除标记，
        # 保留最近 keep_recent 个（对齐 claude-code-main microCompact:evaluateTimeBasedTrigger）
        "time_based_mc_enabled": True,
        "time_based_mc_gap_minutes": 60,
        "time_based_mc_keep_recent": 5,
        # L4 llm
        # 100K tokens ≈ 300K 字符。DeepSeek/OpenAI context 上限 64K-128K。
        # 消息数阈值只是兜底（token 接近窗口才 LLM 压缩，1M 模型放宽到 2000）
        "llm_compact_token_threshold": 100000,
        "llm_compact_message_threshold": 500,
        "llm_compact_keep_recent": 30,
        "llm_compact_cooldown_turns": 5,
        "max_compress_attempts": 3,
        # Reactive（Task D：多次触发 + 冷却窗口）
        "reactive_keep_recent": 10,
        "reactive_compact_cooldown_seconds": 60,   # 冷却窗口（60s 内最多 1 次）
        "reactive_compact_max_per_session": 5,     # 单会话最多触发 5 次（防失控）
        "reactive_once_per_session": False,        # 向后兼容 flag（已废弃，默认 False）
        # Transcript
        "transcript_enabled": True,
        "transcript_trigger": "pre_llm_compact",
        "transcript_retention": 20,
        # CCAR4 Task A：cache break diff 文件 LRU 上限（默认 100，超过删最旧）
        "max_cache_break_diff_files": 100,
        # CCAR4 Task B：post-compact 主动恢复（最近文件 + invoked skills 重注入）
        "post_compact_recovery_enabled": True,          # 开关（False 时 compact 后不重注入）
        "post_compact_recovery_max_files": 5,            # 最近文件数上限
        "post_compact_recovery_max_skills": 5,           # invoked skills 数上限
    },

    # Hooks 系统（Phase 2a）
    "hooks": {
        "enabled": True,                            # 全局开关；False 时跳过所有 hook 调用
        "settings_path": None,                      # None → 默认 ~/.OmniMate/.hooks/settings.json
        "script_timeout_default": 10.0,            # 声明式 hook 默认超时（秒）
        "stop_hook_max_fires": 3,                  # Stop hook 每会话最多触发次数（防失控）
        "fail_closed_default": False,              # 声明式 hook 默认 fail_closed
    },

    # 后台任务（Phase 2b）
    "bg_task": {
        "enabled": True,                        # False 时 bg_* 工具隐藏
        "max_concurrent": 5,
        "default_timeout": 600,                 # bg_start 默认超时（秒）
        "notification_stdout_cap": 500,         # 通知里 stdout 字符上限
        "result_stdout_cap": 5000,              # bg_result 返回的 stdout 字符上限
        "default_detach": False,                # bg_start 默认 detach
        "stall_timeout": 45.0,                  # P1-3: 停滞看门狗（秒）。0=禁用，>0 启用：连续 N 秒 stdout 无新增 → push 通知让 LLM 决定
    },

    # Cron 调度（Phase 2c）
    "cron": {
        "enabled": True,                        # False 时整个 cron 关闭
        "jobs_path": None,                      # None → 默认 ~/.OmniMate/.cron/jobs.json
        "poll_interval_seconds": 30.0,          # 后台线程 tick 间隔
        "max_age_days": 7,                      # === CronRecurringExpiry NEW === 超过自动 disable（防僵尸）
    },

    # Team 多 agent 协作（Phase 4a）
    "team": {
        "enabled": True,                        # False 时 team_* 工具隐藏
        "team_dir": None,                       # None → 默认 ~/.OmniMate/.team/
        "default_role": "worker",               # 新成员默认角色
        "spawn_timeout": 600,                   # 子 agent 启动超时（秒）
        "max_members": 10,                      # 单队最大成员数
        # === P4b-T4: autonomous 模式配置 ===
        "autonomous_idle_timeout": 60.0,        # IDLE 状态超时秒数
        "autonomous_poll_interval": 5.0,        # IDLE 轮询间隔
        "max_depth": 2,                         # spawn 递归深度上限
    },

    # 终端
    "terminal": {
        "cwd": None,
        "default_timeout": 120,
        "shell": None,
    },

    # 记忆
    "memory": {
        "enabled": True,
        "provider": None,                # 外部 provider（如 honcho）
        "memory_char_limit": 2200,
        "user_char_limit": 1375,
        "multifile_enabled": True,       # 多文件模式开关
        "memory_dir": None,              # 记忆目录（None → ~/.OmniMate/.memory/）
        "retrieval_enabled": True,       # 语义检索开关
        "retrieval_max_results": 5,      # 检索最大结果数
        "retrieval_model": None,         # 检索模型（None → 用主模型）
        # === Memory Curator（NEW，Task 10）===
        "curator": {
            "enabled": True,             # False 时整个 memory curator 关闭
            "interval_hours": 168,       # 7 天
            "min_idle_hours": 1,         # agent 至少空闲 1 小时才跑
            "llm_review_enabled": True,  # 第 2 阶段 LLM 总开关
            "max_batch_size": 30,        # 单桶每批发给 LLM 的最大条数
            "archive_after_multiplier": 2,  # archived 阈值 = N × valid_days
        },
    },

    # 技能
    "skills": {
        "dir": None,                     # 自定义技能目录
        "external_dirs": [],
    },

    # Curator
    "curator": {
        "enabled": True,
        "interval_hours": 168,           # 7 天
        "min_idle_hours": 1,
        "stale_after_days": 30,
        "archive_after_days": 90,
        "prune_builtins": False,
        "backup": {
            "enabled": True,
            "max_snapshots": 5,
        },
    },

    # 委托
    "delegation": {
        "max_concurrent_children": 5,
        "max_spawn_depth": 2,
        "child_timeout_seconds": 600,
        "orchestrator_enabled": True,
        "subagent_auto_approve": False,
        "max_iterations": 50,
        # Task F: async 子代理工具白名单（对齐 claude-code-main ASYNC_AGENT_ALLOWED_TOOLS）
        # 默认开（安全默认 > 事后补救）；False 时 async 子代理能用全部工具
        "async_tool_whitelist_enabled": True,
        # 用户扩展禁用列表（在 ASYNC_AGENT_DISALLOWED_TOOLS 基础上追加）
        "async_disallowed_tools": [],
        # Task G: worktree 智能清理
        # True=总是清理（旧行为，无论有无改动）；False=智能清理（有改动保留）
        "worktree_always_cleanup": False,
        # Task H: fork 子代理路径（cache-identical 省 token）
        # True 时 subagent(fork=True) 继承父 system prompt + 父对话前缀
        "fork_subagent_enabled": True,
        # 继承父最近 N 个 assistant turn（cache 命中范围 vs context 污染权衡）
        "fork_max_parent_turns": 3,
        # Task I: 子代理 sidechain transcript 持久化
        # True 时子代理对话历史落盘到 ~/.OmniMate/.agent-sessions/
        "subagent_persistence_enabled": True,
        # 已完成子代理记录保留 N 天（超期清理，节省磁盘）
        "subagent_persistence_retention_days": 7,
        # Task J: async 子代理默认拒审批（对齐 claude-code-main shouldAvoidPermissionPrompts）
        # True 时 async 子代理（background=True）permission_mode='autoDeny'，
        # 所有需 user approval 的破坏性命令直接 permission_denied（fail-closed）。
        # 安全默认 > 事后补救；False 时不注入（不推荐）
        "async_auto_deny_permission": True,
        # async 子代理用的 permission_mode 名称（默认 autoDeny）
        # 用户可改成其他 mode（"default"/"bypassPermissions"/"acceptEdits"）让 async 子代理
        # 走特定权限逻辑（仅当 async_auto_deny_permission=True 时生效）
        "async_permission_mode": "autoDeny",
    },

    # 安全
    "security": {
        "command_approval": "ask",       # ask | always | never
        "dangerous_commands": [],
        "redact_secrets": True,
        "permission_mode": "default",    # "default" | "bypassPermissions" | "acceptEdits"
        # OS 沙箱（对齐 Claude Code /sandbox）
        "sandbox_mode": "off",           # "off" | "on"（启动时灌进 PermissionChecker）
        "sandbox_writable_roots": [],    # 额外允许写的目录（默认 cwd + ~/.OmniMate 已含）
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

    # 启用的工具集：core 全部内置工具 + mcp 外部 server（动态，server 连上才可见）
    "enabled_toolsets": ["core", "mcp"],

    # Plan Mode 总开关（Task 7）
    "plan_mode": {
        "enabled": True,  # Plan Mode 总开关；False 时 /plan 命令报错"功能未启用"
    },

    # ────────────────────────────────────────────────────────────
    # Feature flags（spec 2026-08-09-claude-code-borrow-design §5.3）
    # ────────────────────────────────────────────────────────────
    # 所有 flag 默认 OFF（用户决策：装完默认关，但每项要有测试覆盖）。
    # 用户通过 ~/.OmniMate/settings.json 的 features 节覆盖默认值。
    # 启动时通过 _deep_merge 加载一次，运行时不热加载（保护 prompt cache）。
    "features": {
        # ── 批次 3 (5): Bash LLM 分类器 ──
        "bash_llm_classifier": {
            "enabled": False,
            "model": "aux",          # 用 aux_llm 跑分类
            "whitelist": [           # 白名单快速通道（spec §4 决策 4）
                "ls", "ll", "cat", "pwd", "echo",
                "grep", "find", "which", "where",
                "git status", "git diff", "git log", "git show",
                "python --version", "uv --version",
            ],
        },

        # ── 批次 3 (1): 5 层压缩（补 L4 折叠） ──
        "context_collapse": {
            "enabled": False,
            "threshold_ratio": 0.8,  # 上下文用到 80% 触发
        },

        # ── 批次 3 (1): 5 层压缩（补 L5 响应式回压） ──
        "reactive_compact": {
            "enabled": False,
        },

        # ── 批次 3 (2): Bash 持久重试（unattended） ──
        "bash_unattended_retry": {
            "enabled": False,
            "max_hours": 24,         # 最多持续重试 24 小时
        },

        # ── 批次 3 (6): MCP HTTP/SSE transport ──
        "mcp_http_transport": {
            "enabled": False,
            "default_timeout_sec": 30,
        },

        # ── 批次 3 (6): MCP WebSocket transport ──
        "mcp_websocket_transport": {
            "enabled": False,
        },

        # ── 批次 3 (7): Plan Mode V2 多 Agent 并行 ──
        "plan_mode_v2_parallel": {
            "enabled": False,
            "max_parallel_agents": 3,  # 决策 6：上限 3
        },

        # ── 批次 3 (4): Hook handler 类型扩展 ──
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
}


# ---------------------------------------------------------------------------
# 环境变量元数据（.env 里允许的密钥）
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

# 注：路径函数（get_omnimate_home / config_path / env_file）已统一收敛到 constants.py，
# 此处保留 config_path alias 仅为本模块内部调用便利（外部请直接 import constants）。

def config_path() -> Path:
    """配置文件路径（委托给 constants.config_path）。"""
    return _config_path()



# ---------------------------------------------------------------------------
# 加载 / 保存
# ---------------------------------------------------------------------------

def load_config(
    config_file: Optional[Path] = None,
    *,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """加载配置。

    默认（config_file=None）读 settings.json（新 JSON 配置，类 业界），
    自动从旧 config.yaml + .env 迁移。构造兼容的 model 段供旧代码使用。

    传 config_file 参数时走旧 yaml 逻辑（测试用）。

    合并顺序：DEFAULT ← settings.json ← CLI 覆盖
    """
    # 默认路径：读 settings.json
    if config_file is None:
        try:
            from agent.settings import (
                load_settings, get_current_model_config,
            )
            settings = load_settings()
            model_cfg = get_current_model_config(settings)

            config = dict(settings)
            # 构造兼容 model 段（旧代码用 config["model"]["name"] 等）
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

            # 新模式(llm 段):注入 haiku_model 配置,让 delegate_tool 能读到
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

    # 旧 yaml 逻辑（传 config_file 参数时，或 settings.json 读失败）
    config = copy.deepcopy(DEFAULT_CONFIG)

    # 2. 加载用户 config.yaml
    if config_file is None:
        config_file = config_path()

    config_file = Path(config_file)
    if config_file.exists():
        try:
            user_config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            if user_config and isinstance(user_config, dict):
                # 深合并（不是替换）
                config = _deep_merge(config, user_config)
        except Exception as e:
            logger.warning("加载配置失败 %s: %s", config_file, e)

    # 3. 应用 CLI 覆盖
    if cli_overrides:
        config = _deep_merge(config, cli_overrides)

    # 4. 迁移检查
    config = _maybe_migrate(config)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并字典。

    - 字典：递归合并
    - 其他类型：直接覆盖
    - override 中的 None 不覆盖（让默认值生效）
    """
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if value is None:
            continue  # None 不覆盖
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
    """配置迁移（如果版本不匹配）。

    只在结构性变更（重命名 key、改变 schema）时需要。
    新增字段不需要迁移（deep-merge 自动处理）。
    """
    user_version = config.get("_config_version", 1)
    if user_version == _config_version:
        return config

    # 示例迁移（未来版本）：
    # if user_version < 2:
    #     ...

    return config


def save_config(config: dict, config_file: Optional[Path] = None) -> None:
    """保存配置到 config.yaml。"""
    if config_file is None:
        config_file = config_path()

    config_file = Path(config_file)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # 移除 None 值（让默认值生效）
    clean = _remove_none(config)

    yaml_text = yaml.dump(
        clean,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    config_file.write_text(yaml_text, encoding="utf-8")


def _remove_none(obj):
    """递归移除 None 值。"""
    if isinstance(obj, dict):
        return {k: _remove_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_remove_none(i) for i in obj if i is not None]
    return obj


def ensure_default_config() -> None:
    """如果配置文件不存在，写入默认配置。"""
    cf = config_path()
    if not cf.exists():
        cf.parent.mkdir(parents=True, exist_ok=True)
        save_config(DEFAULT_CONFIG, cf)
        logger.info("已写入默认配置: %s", cf)


# ---------------------------------------------------------------------------
# dotpath 访问（model.name / agent.max_iterations 等）
# ---------------------------------------------------------------------------

def save_config_value(dotpath: str, value: Any, config_file: Optional[Path] = None) -> None:
    """保存单个配置值。

    dotpath 格式："model.name" → config["model"]["name"]
    """
    config = load_config(config_file)
    keys = dotpath.split(".")
    target = config
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value
    save_config(config, config_file)


def get_config_value(config: dict, dotpath: str, default: Any = None) -> Any:
    """按 dotpath 获取配置值。

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
