"""config 工具（CCAR12 Task 7）：LLM 安全改配置（白名单 9 键精确匹配）。

为什么需要白名单：配置里混着敏感键（llm.auth_token、security.* 等），
LLM 直接改配置一旦被 prompt 注入就是灾难。白名单 frozenset 精确匹配
（非前缀——"notifications" 不命中 "notifications.enabled"），
白名单外一律 permission_denied。

config_set 三步走：
  1. 读-改-写 settings.json（load_settings/save_settings，CCAR11 轨道）
  2. runtime 生效：同步改 dispatch_kwargs["agent_ref"].config 同键
     （嵌套路径 set——本会话立即生效，不用重启）
  3. CONFIG_CHANGE hook（审计/缓存失效，fail-open）

value 类型：按现有值类型强转（现有 bool → bool(value)，int → int(value)），
防 LLM 传 JSON 字符串数字落盘后类型漂移。
"""
import json
import logging

from agent.settings import load_settings, save_settings
from tools.registry import registry

logger = logging.getLogger(__name__)


# 白名单（模块级 frozenset，精确匹配）。只放"改坏了也只影响体验"的开关类键。
# review 修正：换掉 2 个 dead key——trace.retention_days（trace.py 无 retention
# 逻辑，全仓无读取点）和 context.reactive_compact_enabled（真实开关是
# features.reactive_compact.enabled）——写黑洞还假报生效是 silent-dead-code。
# 换入 context.reactive_compact_cooldown_seconds / max_per_session
# （agent/__init__.py reactive_compact 分支有真实读取点）。
_CONFIG_WHITELIST = frozenset({
    "notifications.enabled",
    "statusline.enabled",
    "memory.curator.enabled",
    "memory.curator.interval_hours",
    "trace.enabled",
    "context.reactive_compact_cooldown_seconds",
    "context.reactive_compact_max_per_session",
    # CCAR15 Task 4：skill_learning 开关 + 观察后端（读取点在
    # agent/__init__.py:_maybe_skill_learning，start/stop 也可走 CLI
    # /skill-learning，这里给 LLM 一条持久化通道）
    "skill_learning.enabled",
    "skill_learning.observer",
})

# 这些键在 cli initialize 一次性装配（如 TraceSink），会话中改 config 不会重接线
# ——runtime_applied 必须如实报 "next_session"，不能假报 True。
_NEXT_SESSION_KEYS = frozenset({
    "trace.enabled",
})

# 枚举键：值只允许列出的集合（大小写不敏感归一）。
# CCAR15 T4 review 快修：observer 设 "LLM"/拼错会静默走启发式还假报
# runtime_applied=True——"假报生效"教训的值级变体，在校验层拦掉。
_ENUM_KEYS = {
    "skill_learning.observer": frozenset({"heuristic", "llm"}),
}


CONFIG_GET_SCHEMA = {
    "name": "config_get",
    "description": (
        "读配置项的当前值（只限白名单内的 9 个安全键："
        "notifications.enabled / statusline.enabled / memory.curator.enabled / "
        "memory.curator.interval_hours / trace.enabled / "
        "context.reactive_compact_cooldown_seconds"
        "（仅在 features.reactive_compact.enabled 开启时生效） / "
        "context.reactive_compact_max_per_session"
        "（仅在 features.reactive_compact.enabled 开启时生效） / "
        "skill_learning.enabled / skill_learning.observer"
        "（heuristic|llm，llm 失败自动回退启发式））。"
        "优先返回运行时实际生效值。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "配置键（点分路径，必须在白名单内）",
            },
        },
        "required": ["key"],
    },
}


CONFIG_SET_SCHEMA = {
    "name": "config_set",
    "description": (
        "修改配置项（只限白名单内的 9 个安全键，其余拒绝）。"
        "写入 settings.json 持久化 + 当前会话立即生效 + 触发 CONFIG_CHANGE hook。"
        "value 类型按现有值强转（bool/int）。"
        "个别键（trace.enabled）在启动时一次性装配，runtime_applied='next_session'"
        "表示下次会话才生效。"
        "context.reactive_compact_cooldown_seconds"
        "（仅在 features.reactive_compact.enabled 开启时生效）/ "
        "context.reactive_compact_max_per_session"
        "（仅在 features.reactive_compact.enabled 开启时生效）。"
        "skill_learning.enabled（默认 False）/"
        "skill_learning.observer（heuristic|llm）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "配置键（点分路径，必须在白名单内）",
            },
            "value": {
                "description": (
                    "新值（JSON 值，bool/int）。按现有值类型强转："
                    "现有 bool → 布尔，现有 int → 整数。"
                ),
            },
        },
        "required": ["key", "value"],
    },
}


def _split_key(key: str) -> list:
    """点分路径拆段。"""
    return [seg for seg in key.split(".") if seg]


def _get_nested(d: dict, parts: list):
    """按路径读嵌套值。中间层缺失/非 dict → None。"""
    cur = d
    for seg in parts:
        if not isinstance(cur, dict) or seg not in cur:
            return None
        cur = cur[seg]
    return cur


def _set_nested(d: dict, parts: list, value) -> None:
    """按路径写嵌套值（中间层 setdefault 逐层创建）。"""
    cur = d
    for seg in parts[:-1]:
        cur = cur.setdefault(seg, {})
    cur[parts[-1]] = value


def _coerce_value(value, current):
    """按现有值类型强转。bool 先判（bool 是 int 的子类，顺序不能反）。

    不可转（int 键传 "abc"）抛 ValueError/TypeError，由调用方转 invalid_args。
    """
    if isinstance(current, bool):
        return bool(value)
    if isinstance(current, int):
        return int(value)
    return value


def _denied_json(key: str) -> str:
    """白名单外拒绝（带白名单提示，LLM 可自行纠正到合法键）。"""
    return json.dumps(
        {
            "error": (
                f"配置键不在白名单内: {key}。"
                f"允许的键: {sorted(_CONFIG_WHITELIST)}"
            ),
            "error_type": "permission_denied",
            "key": key,
        },
        ensure_ascii=False,
    )


def _handle_config_get(args: dict, **dispatch_kwargs) -> str:
    """读配置项（只读）。

    优先读 agent_ref.config（运行时实际生效值），无 agent_ref 时读
    settings.json。返回 JSON 字符串（统一契约）。
    """
    key = (args.get("key") or "").strip()
    if not key:
        return json.dumps(
            {"error": "key 必填", "error_type": "invalid_args"},
            ensure_ascii=False,
        )
    if key not in _CONFIG_WHITELIST:
        return _denied_json(key)

    try:
        parts = _split_key(key)
        agent_ref = dispatch_kwargs.get("agent_ref")
        runtime_cfg = getattr(agent_ref, "config", None)
        # runtime 优先：反映本会话已生效的值（磁盘可能落后）
        if isinstance(runtime_cfg, dict):
            value = _get_nested(runtime_cfg, parts)
            if value is not None:
                return json.dumps(
                    {"key": key, "value": value}, ensure_ascii=False
                )
        value = _get_nested(load_settings(), parts)
        return json.dumps({"key": key, "value": value}, ensure_ascii=False)
    except Exception as e:
        logger.warning("config_get 异常: %s", e, exc_info=True)
        return json.dumps(
            {"error": str(e), "error_type": type(e).__name__},
            ensure_ascii=False,
        )


def _handle_config_set(args: dict, **dispatch_kwargs) -> str:
    """改配置项（写盘 + runtime 生效 + hook）。

    流程：白名单校验 → load_settings 读现有值（类型推断）→ 强转 →
    save_settings 落盘 → 同步 agent_ref.config → CONFIG_CHANGE hook（fail-open）。
    """
    key = (args.get("key") or "").strip()
    if not key:
        return json.dumps(
            {"error": "key 必填", "error_type": "invalid_args"},
            ensure_ascii=False,
        )
    if key not in _CONFIG_WHITELIST:
        return _denied_json(key)
    if "value" not in args:
        return json.dumps(
            {"error": "value 必填", "error_type": "invalid_args"},
            ensure_ascii=False,
        )

    agent_ref = dispatch_kwargs.get("agent_ref")
    runtime_cfg = getattr(agent_ref, "config", None)

    try:
        parts = _split_key(key)
        settings = load_settings()

        # 类型推断：settings 现有值优先，磁盘没有再参考 runtime（都无 → 原样存）
        current = _get_nested(settings, parts)
        if current is None and isinstance(runtime_cfg, dict):
            current = _get_nested(runtime_cfg, parts)

        try:
            coerced = _coerce_value(args["value"], current)
        except (ValueError, TypeError) as e:
            return json.dumps(
                {
                    "error": f"value 类型转换失败（现有类型 {type(current).__name__}）: {e}",
                    "error_type": "invalid_args",
                    "key": key,
                },
                ensure_ascii=False,
            )

        # 枚举键校验：大小写归一 + 非法值拒绝（不落盘，错误消息列合法值）
        allowed = _ENUM_KEYS.get(key)
        if allowed is not None:
            normalized = str(coerced).strip().lower() if isinstance(
                coerced, str) else None
            if normalized not in allowed:
                return json.dumps(
                    {
                        "error": (
                            f"{key} 只接受 {'/'.join(sorted(allowed))}，"
                            f"收到: {coerced!r}"
                        ),
                        "error_type": "invalid_args",
                        "key": key,
                    },
                    ensure_ascii=False,
                )
            coerced = normalized

        old_value = current

        # 1. 读-改-写 settings.json（CCAR11 轨道）
        _set_nested(settings, parts, coerced)
        save_settings(settings)

        # 2. runtime 生效：同步 agent_ref.config 同键（嵌套路径 set）。
        #    例外：_NEXT_SESSION_KEYS 在 cli initialize 一次性装配（如 TraceSink
        #    接线），会话中改 config 不会重接线——值照常同步（下次会话生效），
        #    但 runtime_applied 如实报 "next_session"，不假报本会话已生效。
        runtime_applied = False
        if isinstance(runtime_cfg, dict):
            _set_nested(runtime_cfg, parts, coerced)
            if key in _NEXT_SESSION_KEYS:
                runtime_applied = "next_session"
            else:
                runtime_applied = True

        # 3. CONFIG_CHANGE hook（fail-open：hook 异常不影响配置写入）
        hooks_registry = dispatch_kwargs.get("hooks_registry")
        if hooks_registry is not None:
            try:
                hooks_registry.run_config_change({
                    "session_id": dispatch_kwargs.get("session_id") or "",
                    "changed_keys": [key],
                    "key": key,
                    "old_value": old_value,
                    "new_value": coerced,
                })
            except Exception as e:
                logger.warning("CONFIG_CHANGE hook 异常（忽略）: %s", e)

        return json.dumps(
            {
                "key": key,
                "value": coerced,
                "old_value": old_value,
                "runtime_applied": runtime_applied,
                "persisted": True,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.warning("config_set 异常: %s", e, exc_info=True)
        return json.dumps(
            {"error": str(e), "error_type": type(e).__name__},
            ensure_ascii=False,
        )


# 模块级注册（import 时自动执行）
registry.register(
    name="config_get",
    toolset="core",
    schema=CONFIG_GET_SCHEMA,
    handler=_handle_config_get,
    emoji="⚙️",
    isConcurrencySafe=True,  # 只读：读配置值，无副作用
)
registry.register(
    name="config_set",
    toolset="core",
    schema=CONFIG_SET_SCHEMA,
    handler=_handle_config_set,
    emoji="⚙️",
    isConcurrencySafe=False,  # 写 settings.json + 改 runtime config + hook
)
