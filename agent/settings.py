"""settings.json 配置管理（类 Claude Code）。

所有配置统一在 ~/.agent/settings.json：
  - models：多模型配置（含 format/api_key/base_url/model）
  - mcpServers：MCP 外部服务器
  - agent/memory/curator/security/sessions/display：行为配置

启动时自动从 config.yaml + .env 迁移（如果 settings.json 不存在）。
旧文件改名 .bak 保留。

API key 直接存 JSON（用户明确要求 JSON 化）。⚠️ 注意权限保护。
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS: Dict[str, Any] = {
    "models": {
        "deepseek": {
            "format": "openai",                       # openai / anthropic
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "",                            # 直接存（用户要求）
            "model": "deepseek-chat",
        },
    },
    "default_model": "deepseek",

    "mcpServers": {},                                 # MCP 配置（原 .mcp.json）

    "agent": {
        "max_iterations": 90,
        "compression_enabled": True,
    },
    "memory": {
        "enabled": True,
        "memory_char_limit": 2200,
        "user_char_limit": 1375,
    },
    "curator": {
        "enabled": True,
        "interval_hours": 168,
        "stale_after_days": 30,
        "archive_after_days": 90,
    },
    "security": {
        "command_approval": "ask",                    # ask / never
    },
    "sessions": {
        "auto_save": True,
        "auto_title": True,
    },
    "display": {
        "show_tool_progress": True,
    },
    "enabled_toolsets": ["core"],
}


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

def settings_path() -> Path:
    """settings.json 路径。"""
    from constants import get_agent_home
    return get_agent_home() / "settings.json"


def approved_commands_path() -> Path:
    """审批白名单 JSON 路径。"""
    from constants import get_agent_home
    return get_agent_home() / "approved_commands.json"


def mcp_config_path() -> Path:
    """旧 .mcp.json 路径（迁移用）。"""
    from constants import get_agent_home
    return get_agent_home() / ".mcp.json"


# ---------------------------------------------------------------------------
# 加载 / 保存
# ---------------------------------------------------------------------------

def load_settings() -> Dict[str, Any]:
    """加载 settings.json。

    不存在时触发迁移：从 config.yaml + .env + .mcp.json 合并生成。
    都没有则写默认配置。
    """
    path = settings_path()
    if not path.exists():
        migrated = migrate_from_legacy()
        if not migrated:
            save_settings(DEFAULT_SETTINGS)
            logger.info("已写入默认 settings.json: %s", path)
            return _deep_merge(dict(DEFAULT_SETTINGS), {})

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return dict(DEFAULT_SETTINGS)
        return _deep_merge(dict(DEFAULT_SETTINGS), data)
    except Exception as e:
        logger.warning("读取 settings.json 失败，用默认: %s", e)
        return dict(DEFAULT_SETTINGS)


def save_settings(settings: Dict[str, Any]) -> None:
    """保存 settings.json。"""
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并（override 覆盖 base）。"""
    if not isinstance(override, dict):
        return override
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# ---------------------------------------------------------------------------
# 从旧配置迁移（config.yaml + .env + .mcp.json → settings.json）
# ---------------------------------------------------------------------------

def migrate_from_legacy() -> bool:
    """从 config.yaml + .env + .mcp.json 迁移到 settings.json。

    旧文件改名 .bak 保留。返回是否执行了迁移。
    """
    from constants import get_agent_home
    home = get_agent_home()

    config_yaml = home / "config.yaml"
    env_file = home / ".env"
    mcp_file = home / ".mcp.json"

    if not any(f.exists() for f in (config_yaml, env_file, mcp_file)):
        return False

    settings = json.loads(json.dumps(DEFAULT_SETTINGS))  # 深拷贝
    migration_log = []

    # 1. 读 config.yaml
    if config_yaml.exists():
        try:
            import yaml
            yaml_config = yaml.safe_load(config_yaml.read_text(encoding="utf-8")) or {}

            old_model = yaml_config.get("model", {}) or {}
            provider = old_model.get("provider", "deepseek")
            name = old_model.get("name", "deepseek-chat")
            base_url = old_model.get("base_url") or "https://api.deepseek.com/v1"
            api_key_env = old_model.get("api_key_env", "DEEPSEEK_API_KEY")

            # 用 provider 名作为模型 key（标准化）
            model_key = provider.lower().replace("-", "_")
            settings["models"] = {
                model_key: {
                    "format": "openai",
                    "base_url": base_url,
                    "api_key": "",  # 从 .env 填
                    "model": name,
                    "_api_key_env": api_key_env,  # 临时记录，下面填
                },
            }
            settings["default_model"] = model_key

            # 迁移其他段
            for key in ("agent", "memory", "curator", "sessions", "display"):
                if key in yaml_config:
                    settings[key] = yaml_config[key]
            if "enabled_toolsets" in yaml_config:
                settings["enabled_toolsets"] = yaml_config["enabled_toolsets"]

            migration_log.append(f"config.yaml → models.{model_key}")
        except Exception as e:
            logger.warning("迁移 config.yaml 失败: %s", e)

    # 2. 读 .env 填 API key
    env_values = {}
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                env_values[k.strip()] = v.strip().strip('"').strip("'")
        except Exception as e:
            logger.warning("读取 .env 失败: %s", e)

    for model_key, model_cfg in settings.get("models", {}).items():
        # 优先用 _api_key_env 指定的变量名
        env_name = model_cfg.pop("_api_key_env", None)
        candidates = []
        if env_name:
            candidates.append(env_name)
        candidates.extend([
            f"{model_key.upper()}_API_KEY",
            f"{model_key.replace('_', '').upper()}_API_KEY",
        ])
        for cand in candidates:
            if env_values.get(cand):
                model_cfg["api_key"] = env_values[cand]
                break
        # 同时也填其他 provider 的 key（如 OPENAI_API_KEY）
    # 兜底：常见 API key 环境变量也填进对应模型
    for env_name, value in env_values.items():
        if not value or "_API_KEY" not in env_name:
            continue
        provider_hint = env_name.replace("_API_KEY", "").lower().replace("-", "_")
        # 如果 settings.models 里有同名的，已填过；否则加一个新模型条目
        if provider_hint not in settings.get("models", {}):
            # 跳过（避免臆造配置）
            pass

    # 3. 读 .mcp.json
    if mcp_file.exists():
        try:
            mcp_data = json.loads(mcp_file.read_text(encoding="utf-8"))
            settings["mcpServers"] = mcp_data.get("mcpServers", {})
            migration_log.append(".mcp.json → mcpServers")
        except Exception as e:
            logger.warning("迁移 .mcp.json 失败: %s", e)

    # 4. 保存 settings.json
    save_settings(settings)
    logger.info("已迁移到 settings.json: %s", "; ".join(migration_log) or "（空）")

    # 5. 旧文件改名 .bak（失败不阻塞）
    for f in (config_yaml, env_file, mcp_file):
        if f.exists():
            bak = f.with_suffix(f.suffix + ".bak")
            try:
                if bak.exists():
                    bak.unlink()
                f.rename(bak)
            except Exception as e:
                logger.debug("改名 %s 失败: %s", f, e)

    return True


# ---------------------------------------------------------------------------
# 当前模型配置
# ---------------------------------------------------------------------------

def get_current_model_config(settings: Optional[Dict] = None) -> Dict[str, Any]:
    """获取当前激活模型的完整配置（含 name/format/api_key/base_url/model）。"""
    if settings is None:
        settings = load_settings()

    name = settings.get("default_model", "")
    models = settings.get("models", {})

    if name not in models:
        if models:
            name = next(iter(models))
        else:
            return {"name": "none", "format": "openai", "api_key": "", "model": ""}

    cfg = json.loads(json.dumps(models[name]))  # 深拷贝
    cfg["name"] = name
    cfg.setdefault("format", "openai")
    cfg.setdefault("api_key", "")
    cfg.setdefault("base_url", None)
    cfg.setdefault("model", name)
    return cfg


def list_models(settings: Optional[Dict] = None) -> Dict[str, Dict]:
    """列出所有配置的模型。"""
    if settings is None:
        settings = load_settings()
    return settings.get("models", {})


def set_default_model(name: str) -> bool:
    """切换默认模型并持久化。"""
    settings = load_settings()
    if name not in settings.get("models", {}):
        return False
    settings["default_model"] = name
    save_settings(settings)
    return True


def add_model(name: str, config: Dict[str, Any]) -> bool:
    """添加新模型配置并持久化。"""
    settings = load_settings()
    settings.setdefault("models", {})[name] = config
    save_settings(settings)
    return True


def ensure_default_settings() -> None:
    """如果 settings.json 不存在，创建默认配置（含迁移）。"""
    if not settings_path().exists():
        load_settings()  # 触发迁移或写默认
