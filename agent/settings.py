"""settings.json 配置管理——运行时可写配置的唯一正道。

这个文件管 ~/.OmniMate/settings.json 的读写，里面装着：
  - llm：模型配置（用哪个服务商、哪个模型、API key）
  - mcpServers：MCP 外部工具服务器配置
  - agent/memory/curator/security/sessions/display：各种行为配置

第一次启动时会自动把旧的三样（config.yaml + .env + .mcp.json）
搬进 settings.json，旧文件改名 .bak 留着不删。

注意：API key 是明文直接存 JSON 里的（用户明确要求 JSON 化），
所以这个文件的权限保护很重要。
"""

import copy
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
    # LLM 配置（扁平模式：像环境变量一样平铺，不带品牌前缀）。
    # 思路：所有模型共用同一个 base_url + auth_token，只按"角色"
    # 区分模型名——换模型只改一个字段就行
    "llm": {
        "base_url": "https://api.deepseek.com/anthropic",
        "auth_token": "",                             # API token 填这里
        "api_timeout_ms": 3000000,                    # 50 分钟超时（给复杂思考任务留足时间）
        "effort_level": "max",                        # 思考强度：max/high/medium/low
        "auto_compact_window": 1000000,               # 1M 上下文的自动压缩窗口
        # 按角色分层选模型（强/标准/轻三档）
        "opus_model": "deepseek-v4-pro[1m]",          # 强模型（主对话/复杂推理）
        "sonnet_model": "deepseek-v4-pro[1m]",        # 标准模型
        "haiku_model": "deepseek-v4-flash",           # 轻量模型（子代理/打杂任务）
    },
    "default_model": "opus",                          # 主对话默认用 opus 档
    "default_haiku_model": "haiku",                   # 子代理/辅助任务默认用 haiku 档

    "mcpServers": {},                                 # MCP 外部工具服务器配置

    # 工具可见性规则：deny 里的整类隐藏，allow 里的可以豁免某条 deny
    "permissions": {
        "allow": [],
        "deny": [],
    },

    "agent": {
        "max_iterations": 200,
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
        "command_approval": "ask",                    # 命令审批：ask（问用户）/ never（不问）
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
    """拿到 settings.json 的完整路径。

    返回：
        Path 对象，指向 <agent home>/settings.json。
    """
    from constants import get_omnimate_home
    return get_omnimate_home() / "settings.json"


def approved_commands_path() -> Path:
    """拿到"已批准命令"白名单 JSON 的路径。

    返回：
        Path 对象，指向 <agent home>/approved_commands.json。
    """
    from constants import get_omnimate_home
    return get_omnimate_home() / "approved_commands.json"


def approved_paths_path() -> Path:
    """拿到"已批准写入路径"白名单 JSON 的路径（用户点头过的写目录）。

    返回：
        Path 对象，指向 <agent home>/approved_paths.json。
    """
    from constants import get_omnimate_home
    return get_omnimate_home() / "approved_paths.json"


def mcp_config_path() -> Path:
    """拿到旧 .mcp.json 的路径（只在迁移旧配置时用）。

    返回：
        Path 对象，指向 <agent home>/.mcp.json。
    """
    from constants import get_omnimate_home
    return get_omnimate_home() / ".mcp.json"


# ---------------------------------------------------------------------------
# 加载 / 保存
# ---------------------------------------------------------------------------

def load_settings() -> Dict[str, Any]:
    """加载 settings.json，返回"默认值 + 用户配置"合并后的结果。

    文件不存在时先尝试从 config.yaml + .env + .mcp.json 迁移生成；
    三样都没有就写入一份纯默认配置。

    返回值必须是深拷贝——浅拷贝会让调用方"读→改→写回"的改动污染进
    模块级 DEFAULT_SETTINGS（如 extra_allowed_roots 泄进后续所有
    "默认配置"加载）。

    返回：
        配置字典（深拷贝，随便改不伤默认值）。
    """
    path = settings_path()
    if not path.exists():
        migrated = migrate_from_legacy()
        if not migrated:
            save_settings(DEFAULT_SETTINGS)
            logger.info("已写入默认 settings.json: %s", path)
            return copy.deepcopy(DEFAULT_SETTINGS)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return copy.deepcopy(DEFAULT_SETTINGS)
        # 必须深拷贝基底再合并：_deep_merge 会原地修改 base 里的嵌套
        # 字典，浅拷贝（dict(...)）等于把用户数据泄进 DEFAULT_SETTINGS
        # 这个全局默认值里
        return _deep_merge(copy.deepcopy(DEFAULT_SETTINGS), data)
    except Exception as e:
        logger.warning("读取 settings.json 失败，用默认: %s", e)
        return copy.deepcopy(DEFAULT_SETTINGS)


def save_settings(settings: Dict[str, Any]) -> None:
    """把配置字典写入 settings.json（原子写，写一半断电不会留半个文件）。

    参数：
        settings：完整的配置字典（整体覆盖写入）。
    """
    from agent.atomic_io import atomic_write_text
    path = settings_path()
    atomic_write_text(path, json.dumps(settings, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# 写路径"总是允许"白名单的持久化（/add-dir 命令走的就是这一通道）
# ---------------------------------------------------------------------------

def persist_extra_allowed_root(root: str) -> bool:
    """把一个目录加入写白名单（存到 settings.json 的
    security.extra_allowed_roots），让 agent 以后写这个目录不用再问。

    做法：读出现有配置（已含默认值）→ 追加（查重）→ 原子写回。
    重复添加不会产生重复条目（幂等）。

    参数：
        root：要放行的目录路径字符串。

    返回：
        bool——True 表示这次新写入了一条；False 表示它本来就在白名单里。
    """
    from pathlib import Path as _Path

    data = load_settings()
    sec = data.get("security")
    if not isinstance(sec, dict):
        sec = {}
        data["security"] = sec
    roots = [r for r in (sec.get("extra_allowed_roots") or []) if isinstance(r, str)]
    target = str(_Path(root).expanduser().resolve())
    if target in roots or root in roots:
        return False
    roots.append(target)
    sec["extra_allowed_roots"] = roots
    save_settings(data)
    return True


def remove_extra_allowed_root(root: str) -> bool:
    """把一个目录从写白名单里移除（persist_extra_allowed_root 的反操作）。

    参数：
        root：要移除的目录路径字符串。字符串完全相同、或解析成
            绝对路径后相同，都算命中。

    返回：
        bool——True 表示找到并移除了；False 表示白名单里没有它。
    """
    from pathlib import Path as _Path

    data = load_settings()
    sec = data.get("security")
    if not isinstance(sec, dict):
        return False
    roots = [r for r in (sec.get("extra_allowed_roots") or []) if isinstance(r, str)]
    if not roots:
        return False
    try:
        target = str(_Path(root).expanduser().resolve())
    except (OSError, ValueError):
        target = root
    new_roots = [r for r in roots if not (r == root or _same_path(r, target))]
    if len(new_roots) == len(roots):
        return False
    sec["extra_allowed_roots"] = new_roots
    save_settings(data)
    return True


def _same_path(a: str, b: str) -> bool:
    """判断两条路径字符串是不是指同一个目录（同一目录可有相对/绝对/~ 简写多种写法，直接比字符串会误判）。

    参数：
        a、b：两条路径字符串。

    返回：
        bool——把 ~ 展开、转成绝对路径后相等即为 True；
        转换过程出错（如非法字符）则退回直接比字符串。
    """
    from pathlib import Path as _Path
    try:
        return _Path(a).expanduser().resolve() == _Path(b).expanduser().resolve()
    except (OSError, ValueError):
        return a == b


# ---------------------------------------------------------------------------
# 项目级 .mcp.json 首连审批：第一次连某个项目自带的外部工具服务器
# 时要问用户一次，批准结果持久化在这里（settings.json 的 mcp 段）
# ---------------------------------------------------------------------------

def is_project_mcp_approved(key: str) -> bool:
    """查某个项目级 MCP 服务器是否已被用户批准过。

    参数：
        key：审批键（项目路径 + 服务器名拼出来的字符串，
            见 mcp_approval_key）。

    返回：
        bool——在已批准列表里返回 True；没批准过或读取失败返回 False
        （读不到就当没批准，宁可多问一次，fail-safe）。
    """
    try:
        data = load_settings()
        mcp_sec = data.get("mcp")
        if not isinstance(mcp_sec, dict):
            return False
        return key in [str(k) for k in (mcp_sec.get("approved_project_servers") or [])]
    except Exception:
        return False


def persist_project_mcp_approval(key: str) -> None:
    """把某次项目级 MCP 服务器的批准记下来（幂等，重复记不重复加）。

    参数：
        key：审批键，同 is_project_mcp_approved。
    """
    data = load_settings()
    mcp_sec = data.get("mcp")
    if not isinstance(mcp_sec, dict):
        mcp_sec = {}
        data["mcp"] = mcp_sec
    keys = [str(k) for k in (mcp_sec.get("approved_project_servers") or [])]
    if key not in keys:
        keys.append(key)
    mcp_sec["approved_project_servers"] = keys
    save_settings(data)


def server_cfg_fingerprint(cfg: dict) -> str:
    """给一份 MCP 服务器配置算"指纹"（规范化 JSON 后取 sha256 前 8 位）。

    审批键里带上指纹后，配置一变指纹就变 → 旧批准的键对不上 → 重新问用户
    （失效方向是"多问一次"，绝不漏拦）。

    参数：
        cfg：MCP 服务器的配置字典。

    返回：
        8 个十六进制字符组成的指纹字符串。
    """
    import hashlib
    payload = json.dumps(cfg or {}, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def mcp_approval_key(proj_key: str, server_name: str, cfg: Optional[dict] = None) -> str:
    """拼出项目级 MCP 审批用的键字符串。

    参数：
        proj_key：项目标识（项目路径小写等规范形式）。
        server_name：服务器名。
        cfg：可选，服务器的配置字典。带上它键里就多一段配置指纹
            （配置一变键就变，逼着重新审批）；不传则生成无指纹的
            老式键。

    返回：
        键字符串：cfg=None 时是 `<proj>::<name>`；
        有 cfg 时是 `<proj>::<name>::<指纹8位>`。
    """
    if cfg is None:
        return f"{proj_key}::{server_name}"
    return f"{proj_key}::{server_name}::{server_cfg_fingerprint(cfg)}"


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并两个字典（override 的值优先），原地修改 base 并返回——用户配置通常只写几个字段，逐层下钻补齐默认值（整体替换会丢字段）。

    参数：
        base：基础字典（会被原地修改，调用方要传拷贝）。
        override：覆盖字典，它的值优先。

    返回：
        合并后的字典（就是被改过的 base）。
    """
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
    """把旧的 config.yaml + .env + .mcp.json 无损搬进 settings.json，搬完把旧文件改名 .bak 留底（永不删除）。

    返回：
        bool——True 表示三个旧文件至少有一个存在、执行了迁移；
        False 表示什么旧文件都没有（无需迁移）。
    """
    from constants import get_omnimate_home
    home = get_omnimate_home()

    config_yaml = home / "config.yaml"
    env_file = home / ".env"
    mcp_file = home / ".mcp.json"

    if not any(f.exists() for f in (config_yaml, env_file, mcp_file)):
        return False

    settings = json.loads(json.dumps(DEFAULT_SETTINGS))  # 深拷贝默认值当底子
    migration_log = []

    # 1. 搬 config.yaml
    if config_yaml.exists():
        try:
            import yaml
            yaml_config = yaml.safe_load(config_yaml.read_text(encoding="utf-8")) or {}

            old_model = yaml_config.get("model", {}) or {}
            provider = old_model.get("provider", "deepseek")
            name = old_model.get("name", "deepseek-chat")
            base_url = old_model.get("base_url") or "https://api.deepseek.com/v1"
            api_key_env = old_model.get("api_key_env", "DEEPSEEK_API_KEY")

            # 用 provider 名当模型条目的键（统一命名格式）
            model_key = provider.lower().replace("-", "_")
            settings["models"] = {
                model_key: {
                    "format": "openai",
                    "base_url": base_url,
                    "api_key": "",  # 先留空，下面从 .env 填
                    "model": name,
                    "_api_key_env": api_key_env,  # 临时记一下变量名，稍后回填
                },
            }
            settings["default_model"] = model_key

            # 其他配置段原样搬
            for key in ("agent", "memory", "curator", "sessions", "display"):
                if key in yaml_config:
                    settings[key] = yaml_config[key]
            if "enabled_toolsets" in yaml_config:
                settings["enabled_toolsets"] = yaml_config["enabled_toolsets"]

            migration_log.append(f"config.yaml → models.{model_key}")
        except Exception as e:
            logger.warning("迁移 config.yaml 失败: %s", e)

    # 2. 读 .env，把 API key 填进对应模型
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
        # 优先用旧配置里指定的环境变量名
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
        # 顺带把其他服务商的 key 也填上（如 OPENAI_API_KEY）
    # 兜底：.env 里常见的 API key 环境变量也归置一下
    for env_name, value in env_values.items():
        if not value or "_API_KEY" not in env_name:
            continue
        provider_hint = env_name.replace("_API_KEY", "").lower().replace("-", "_")
        # models 里已有同名的就说明填过了；没有的不凭空捏造条目
        # （避免臆造配置，宁缺勿错）
        if provider_hint not in settings.get("models", {}):
            pass

    # 3. 搬 .mcp.json
    if mcp_file.exists():
        try:
            mcp_data = json.loads(mcp_file.read_text(encoding="utf-8"))
            settings["mcpServers"] = mcp_data.get("mcpServers", {})
            migration_log.append(".mcp.json → mcpServers")
        except Exception as e:
            logger.warning("迁移 .mcp.json 失败: %s", e)

    # 4. 写出 settings.json
    save_settings(settings)
    logger.info("已迁移到 settings.json: %s", "; ".join(migration_log) or "（空）")

    # 5. 旧文件改名 .bak 留底（改不成也不影响迁移结果）
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
    """拿到当前正在用的那个模型的完整配置。

    支持两种配置模式（并存）：
    1. 新模式（推荐）：llm 段扁平配置——所有模型共用一份
       base_url/auth_token，按 opus/sonnet/haiku 三档选模型名。
    2. 老模式（向后兼容）：models 嵌套——每个模型独立配一套
       base_url/auth_token。

    参数：
        settings：可选，配置字典；不传就现场 load_settings() 读一份。

    返回：
        当前模型的配置字典（含 name/format/base_url/auth_token/model 等）。
        完全没配模型时返回一个 name="none" 的空壳。
    """
    if settings is None:
        settings = load_settings()

    name = settings.get("default_model", "opus")

    # 新模式：llm 段扁平配置
    llm_cfg = settings.get("llm", {})
    if llm_cfg:
        model_name = llm_cfg.get(f"{name}_model")
        if model_name:
            return {
                "name": name,
                "format": "anthropic",
                "base_url": llm_cfg.get("base_url"),
                "auth_token": llm_cfg.get("auth_token", ""),
                "model": model_name,
                "effort_level": llm_cfg.get("effort_level", ""),
                "api_timeout_ms": llm_cfg.get("api_timeout_ms"),
            }

    # 老模式：models 嵌套（向后兼容）
    models = settings.get("models", {})
    if name not in models:
        if models:
            name = next(iter(models))
        else:
            return {"name": "none", "format": "anthropic", "auth_token": "", "model": ""}

    cfg = json.loads(json.dumps(models[name]))
    cfg["name"] = name
    cfg.setdefault("format", "anthropic")
    cfg.setdefault("auth_token", "")
    cfg.setdefault("api_key", "")
    cfg.setdefault("base_url", None)
    cfg.setdefault("model", name)
    return cfg


def list_models(settings: Optional[Dict] = None) -> Dict[str, Dict]:
    """列出所有已配置的模型。

    参数：
        settings：可选，配置字典；不传就现场 load_settings() 读一份。

    返回：
        字典，键是模型名，值是该模型的配置字典。
        新模式下从 opus/sonnet/haiku 三档构造；老模式下
        直接返回 models 段。
    """
    if settings is None:
        settings = load_settings()

    # 新模式：从 llm 段的三档模型名构造
    llm_cfg = settings.get("llm", {})
    if llm_cfg:
        models = {}
        for tier in ("opus", "sonnet", "haiku"):
            model_name = llm_cfg.get(f"{tier}_model")
            if model_name:
                models[tier] = {
                    "format": "anthropic",
                    "base_url": llm_cfg.get("base_url"),
                    "auth_token": llm_cfg.get("auth_token", ""),
                    "model": model_name,
                    "effort_level": llm_cfg.get("effort_level", "") if tier != "haiku" else "",
                }
        return models

    # 老模式：models 嵌套直接给
    return settings.get("models", {})


def set_default_model(name: str) -> bool:
    """切换默认模型并写进 settings.json。

    参数：
        name：模型名。新模式下是档位名（opus/sonnet/haiku）；
            老模式下是 models 段里的键名。

    返回：
        bool——True 表示切换成功并已持久化；False 表示没找到
        这个名字的模型（配置没动）。
    """
    settings = load_settings()

    # 新模式：llm 段里得有对应档位的模型名
    llm_cfg = settings.get("llm", {})
    if llm_cfg:
        if llm_cfg.get(f"{name}_model"):
            settings["default_model"] = name
            save_settings(settings)
            return True
        return False

    # 老模式：models 段里得有这个键
    if name in settings.get("models", {}):
        settings["default_model"] = name
        save_settings(settings)
        return True
    return False


def add_model(name: str, config: Dict[str, Any]) -> bool:
    """新增一个模型配置并写进 settings.json。

    参数：
        name：模型条目的键名。
        config：该模型的配置字典（base_url/auth_token/model 等）。

    返回：
        bool——目前恒为 True（同名会直接覆盖旧配置）。
    """
    settings = load_settings()
    settings.setdefault("models", {})[name] = config
    save_settings(settings)
    return True


def ensure_default_settings() -> None:
    """settings.json 不存在时创建它（顺带完成旧配置迁移）——程序启动时调用一次，保证配置文件总是存在。"""
    if not settings_path().exists():
        load_settings()  # 走 load 的"不存在"分支：迁移或写默认
