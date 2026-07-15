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

    # Agent 行为
    "agent": {
        "max_iterations": 90,
        "compression_enabled": True,
        "system_prompt": None,
    },

    # 上下文压缩管线（Phase 1）
    "context": {
        # L3 offload
        "output_offload_threshold": 30000,
        "output_offload_preview": 2000,
        # L1 snip
        "snip_message_threshold": 50,
        "snip_release_threshold": 30,
        "snip_keep_first": 3,
        "snip_keep_last": 47,
        # L2 micro
        "micro_keep_recent_results": 3,
        # L4 llm
        "llm_compact_token_threshold": 100000,
        "llm_compact_message_threshold": 100,
        "llm_compact_keep_recent": 10,
        "llm_compact_cooldown_turns": 5,
        "max_compress_attempts": 3,
        # Reactive
        "reactive_keep_recent": 5,
        "reactive_once_per_session": True,
        # Transcript
        "transcript_enabled": True,
        "transcript_trigger": "pre_llm_compact",
        "transcript_retention": 20,
        # 功能开关（双轨期；Commit 6 改 True）
        "use_new_pipeline": True,
    },

    # Hooks 系统（Phase 2a）
    "hooks": {
        "enabled": True,                            # 全局开关；False 时跳过所有 hook 调用
        "settings_path": None,                      # None → 默认 ~/.agent/.hooks/settings.json
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
        "max_concurrent_children": 3,
        "max_spawn_depth": 2,
        "child_timeout_seconds": 600,
        "orchestrator_enabled": True,
        "subagent_auto_approve": False,
        "max_iterations": 50,
    },

    # 安全
    "security": {
        "command_approval": "ask",       # ask | always | never
        "dangerous_commands": [],
        "redact_secrets": True,
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
        "db_path": None,                 # 默认 ~/.agent/sessions.db
    },

    # 启用的工具集
    "enabled_toolsets": ["core"],
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
    "AGENT_HOME": {
        "description": "agent home 目录（覆盖默认 ~/.agent）",
        "prompt": "Agent Home",
        "password": False,
        "category": "system",
    },
}


# ---------------------------------------------------------------------------
# 路径辅助
# ---------------------------------------------------------------------------

def get_agent_home() -> Path:
    """获取 agent home 目录（支持 AGENT_HOME 环境变量）。"""
    home_env = __import__("os").environ.get("AGENT_HOME")
    if home_env:
        return Path(home_env).expanduser()
    return Path.home() / ".agent"


def config_path() -> Path:
    """配置文件路径。"""
    return get_agent_home() / "config.yaml"


def env_path() -> Path:
    """.env 文件路径（只放密钥）。"""
    return get_agent_home() / ".env"


# ---------------------------------------------------------------------------
# 加载 / 保存
# ---------------------------------------------------------------------------

def load_config(
    config_file: Optional[Path] = None,
    *,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """加载配置。

    默认（config_file=None）读 settings.json（新 JSON 配置，类 Claude Code），
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
                "provider": model_cfg.get("name", "deepseek"),
                "name": model_cfg.get("model", "deepseek-chat"),
                "base_url": model_cfg.get("base_url"),
                "api_key": model_cfg.get("api_key", ""),
                "api_key_env": "",                  # 已废弃（key 直接存 JSON）
                "format": model_cfg.get("format", "openai"),
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
