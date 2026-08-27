"""config 工具：让 LLM 能安全地改运行时配置。

只开放白名单里的 9 个键，而且必须精确匹配。在项目里的位置：tools 层的
core 工具，读写都走 agent/settings.py 的 load_settings/save_settings，
并同步 agent_ref 上挂的运行时配置。

为什么非要白名单：配置里混着敏感键（比如 llm.auth_token、security.* 这类），
如果 LLM 什么都能改，一旦提示词被恶意注入（prompt 注入攻击），后果不堪设想。
所以用 frozenset（不可变集合）做精确匹配——不是前缀匹配，
比如 "notifications" 不会误中 "notifications.enabled"；不在名单里的键一律拒绝
（permission_denied）。

config_set（改配置）分三步：
  1. 读-改-写 settings.json（走 load_settings/save_settings，这是
     唯一合法持久化通道——config.yaml 不在读取路径上，写了也读不回来）
  2. 运行时立即生效：把 agent_ref.config 里的同一个键改掉
     （按嵌套路径写入——本会话马上生效，不用重启）
  3. 触发 CONFIG_CHANGE hook（钩子，供审计/缓存失效用；hook 出错也不影响写配置）

新值的类型：按现有值的类型强转（现有是 bool 就转成 bool，是 int 就转成 int）——
LLM 有时会把数字传成 JSON 字符串（"5" 而不是 5），不转的话落盘后类型就漂了，
后面按 int 读会出错。
"""
import json
import logging

from agent.settings import load_settings, save_settings
from tools.registry import registry

logger = logging.getLogger(__name__)


# 白名单（模块级不可变集合，精确匹配）。只收「改坏了顶多影响体验」的开关类键。
# 只收有真实读取点的键——写进没人读的"死键"还谎报"已生效"，是最危险的
# 静默失败（silent-dead-code）。新收录键之前必须先 grep 确认有消费方。
_CONFIG_WHITELIST = frozenset({
    "notifications.enabled",
    "statusline.enabled",
    "memory.curator.enabled",
    "memory.curator.interval_hours",
    "trace.enabled",
    "context.reactive_compact_cooldown_seconds",
    "context.reactive_compact_max_per_session",
    # skill_learning 开关 + 观察后端（读取点在
    # agent/__init__.py:_maybe_skill_learning，start/stop 也可走 CLI
    # /skill-learning，这里给 LLM 一条持久化通道）
    "skill_learning.enabled",
    "skill_learning.observer",
})

# 这些键只在 CLI 启动装配阶段接线一次（比如 TraceSink 日志收集器），会话中途改
# config 也不会重新接线——所以 runtime_applied 必须如实报 "next_session"（下次会话
# 才生效），不能谎报 True。「假报生效」是本项目反复强调要防的坑。
_NEXT_SESSION_KEYS = frozenset({
    "trace.enabled",
})

# 枚举键：值只能是列出来的那几个（统一转小写再比较）。
# observer 若被设成 "LLM" 或拼错单词，
# 会静默回落到启发式，还谎报 runtime_applied=True——这就是"假报生效"在取值层面的
# 变体，所以在校验这一层直接拦死。
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
    """把 "a.b.c" 这种点分隔的配置键拆成 ["a", "b", "c"]。

    参数：
        key: 点分路径形式的配置键
    返回：拆好的段列表（空段自动丢弃）。
    """
    return [seg for seg in key.split(".") if seg]


def _get_nested(d: dict, parts: list):
    """按 ["a","b","c"] 这样的路径，从嵌套 dict 里一层层往下读值。

    参数：
        d: 要读的字典（可以是多层嵌套的）
        parts: 路径段列表（_split_key 的产物）
    返回：读到叶子上的值；中途某一层不存在或不是字典 → 返回 None。
    """
    cur = d
    for seg in parts:
        if not isinstance(cur, dict) or seg not in cur:
            return None
        cur = cur[seg]
    return cur


def _set_nested(d: dict, parts: list, value) -> None:
    """按路径往嵌套 dict 里写值；中间层不存在就顺手建出来。

    参数：
        d: 要写入的字典
        parts: 路径段列表
        value: 要写的值
    返回：无（直接原地修改 d）。
    """
    cur = d
    for seg in parts[:-1]:
        cur = cur.setdefault(seg, {})
    cur[parts[-1]] = value


def _coerce_value(value, current):
    """按现有值的类型把新值"掰"成同类型。

    LLM 可能把数字传成字符串 "5"，不掰的话落盘后类型就乱了。
    必须先判 bool 再判 int——因为 bool 是 int 的子类，顺序反了布尔值会被
    误当成整数处理。

    参数：
        value: LLM 传来的新值
        current: 现有值（用它的类型做参照）
    返回：强转后的新值；转不动（比如 int 键传 "abc"）就抛
    ValueError/TypeError，由调用方包装成 invalid_args 错误返回。
    """
    if isinstance(current, bool):
        return bool(value)
    if isinstance(current, int):
        return int(value)
    return value


def _denied_json(key: str) -> str:
    """拒绝白名单外的键，并在错误信息里列出所有合法键。

    把合法键直接告诉 LLM，它下次就能自己改对。

    参数：
        key: 被拒绝的配置键
    返回：JSON 字符串，error_type 为 permission_denied。
    """
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
    """config_get 的处理函数：查一个配置键当前的值（纯只读）。

    参数：
        args: LLM 传来的参数，只用到 key（要查的配置键）
        dispatch_kwargs: 命名上下文，用到 agent_ref（拿它身上的运行时配置）
    返回：JSON 字符串（统一契约）。优先读 agent_ref.config——那是本会话
    实际生效的值；拿不到 agent_ref 时才退回去读 settings.json 文件。
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
        # 运行时值优先：反映本会话已生效的状态（磁盘文件可能还没跟上）
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
    """config_set 的处理函数：改一个配置键（写盘 + 本会话生效 + 通知 hook）。

    流程：白名单校验 → 读 settings.json 里的现有值（用来推断类型）→
    把新值强转成同类型 → 写回 settings.json 落盘 → 同步改 agent_ref.config →
    触发 CONFIG_CHANGE hook（hook 出错不影响配置写入）。

    参数：
        args: LLM 传来的参数——key（配置键）和 value（新值）
        dispatch_kwargs: 命名上下文——agent_ref（运行时配置挂它身上）、
                         hooks_registry（触发 hook 用）、session_id（hook 载荷）
    返回：JSON 字符串，带 runtime_applied（本会话生效 / "next_session"）和
    persisted（是否已写盘）。
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

        # 推断类型用哪个值当参照：磁盘 settings 里现有值优先，磁盘没有再看
        # 运行时配置（两边都没有 → 不强转，按 LLM 传的原样存）
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

        # 枚举键校验：统一转小写再比；非法值直接拒绝（不落盘），错误消息里列出合法选项
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

        # 第 1 步：读-改-写 settings.json（唯一合法持久化通道）
        _set_nested(settings, parts, coerced)
        save_settings(settings)

        # 第 2 步：让它在本会话立即生效——同步改 agent_ref.config 里的同一个键
        #    （按嵌套路径写入）。例外：_NEXT_SESSION_KEYS 里的键（如 trace.enabled）
        #    只在 CLI 启动装配阶段接线一次（比如 TraceSink），会话中途改了也不会
        #    重新接线——值照样同步（下次会话读到），但 runtime_applied 要如实报
        #    "next_session"，不能谎报本会话已生效。
        runtime_applied = False
        if isinstance(runtime_cfg, dict):
            _set_nested(runtime_cfg, parts, coerced)
            if key in _NEXT_SESSION_KEYS:
                runtime_applied = "next_session"
            else:
                runtime_applied = True

        # 第 3 步：触发 CONFIG_CHANGE hook（供审计/缓存失效用；hook 崩了也不回滚配置）
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


# 模块级注册：本文件一被 import 就自动登记这两个工具
registry.register(
    name="config_get",
    toolset="core",
    schema=CONFIG_GET_SCHEMA,
    handler=_handle_config_get,
    emoji="⚙️",
    isConcurrencySafe=True,  # 只读：查配置值，没有副作用，并发跑也没事
)
registry.register(
    name="config_set",
    toolset="core",
    schema=CONFIG_SET_SCHEMA,
    handler=_handle_config_set,
    emoji="⚙️",
    isConcurrencySafe=False,  # 有写副作用：写 settings.json + 改运行时配置 + 触发 hook，须串行
)
