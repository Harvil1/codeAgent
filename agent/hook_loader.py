"""把 settings.json 里用户声明式配置的 hooks（钩子——在特定事件发生时自动执行的小程序）装进 registry（工具/钩子注册表）。

这是 hook 系统的"装载入口"：上层是 settings.json 的用户配置，下层是 agent/hooks.py 的 Hook 数据结构和 hook_exec 的执行器。

settings.json 里的写生长这样：
{
  "hooks": {
    "user_prompt_submit": [{name, command, timeout?, env?, fail_closed?}, ...],
    "pre_tool_use": [...],
    "post_tool_use": [...],
    "stop": [...]
  }
}

校验规则（哪些错要报、哪些错放过）：
- 顶层缺 hooks 字段 → raise ValueError（用户写了 hooks 相关配置却写错了，必须让他知道）
- event 名不合法 → raise ValueError（同上，拼错事件名要报）
- 单个 hook 缺 name / command → 跳过 + log warning（一条坏了不连累其他）
- 文件不存在 → 静默返回 0（没配置 hook 是正常状态，不强制）

SnapshotCache（配置快照缓存）：启动加载完就把原始配置锁进内存，
运行期不再重读磁盘。设计取舍（安全语义）：
"hook 配置在会话启动时锁定，运行期改文件不会立即生效"，防止会话中途被人
篡改配置文件偷偷加钩子。
"""
import copy
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from agent.hooks import Hook, HookEvent, HookScriptConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SnapshotCache：启动时装一份配置进内存锁死，运行期只读（防会话中途被篡改）
# ---------------------------------------------------------------------------

_snapshot_cache: Optional[Dict] = None  # 内存里的配置快照，形如 {事件名: [hook配置dict, ...]}
_snapshot_settings_path: Optional[Path] = None


def load_declarative_hooks(registry, settings_path: Path) -> int:
    """从 settings.json 读取用户声明的 hooks 并逐个注册进 registry。

    背景：这是会话启动时 hook 系统的装载动作，只跑一次；跑完把原始配置
    存进 SnapshotCache 锁定（运行期改磁盘文件不生效，防篡改）。

    参数：
        registry：hook 注册表对象，解析成功的 hook 会调它的 register_declarative 登记
        settings_path：settings.json 的路径

    返回：
        int——成功加载的 hook 数量。settings.json 不存在时安静返回 0（没配 hook 很正常）。
    """
    global _snapshot_cache, _snapshot_settings_path

    if not settings_path.exists():
        return 0

    text = settings_path.read_text(encoding="utf-8")
    config = json.loads(text)  # 为什么直接抛：JSON 都写坏了说明配置有硬错误，必须让用户看到

    if "hooks" not in config:
        raise ValueError(
            f"settings.json 缺少顶层 'hooks' 字段: {settings_path}"
        )

    raw = config["hooks"]
    if not isinstance(raw, dict):
        raise ValueError(f"settings.json 'hooks' 必须是 dict，实际是 {type(raw).__name__}")

    count = 0
    for event_str, hook_list in raw.items():
        try:
            event = HookEvent(event_str)
        except ValueError:
            raise ValueError(
                f"settings.json 包含未知 event 名: '{event_str}'，"
                f"合法值: {[e.value for e in HookEvent]}"
            )
        if not isinstance(hook_list, list):
            raise ValueError(
                f"settings.json event '{event_str}' 必须是 list，"
                f"实际是 {type(hook_list).__name__}"
            )
        for h_cfg in hook_list:
            hook = _parse_hook(h_cfg, event)
            if hook is not None:
                registry.register_declarative(hook)
                count += 1

    # 深拷贝一份存内存——之后磁盘文件怎么改都不影响本会话（防篡改语义）
    _snapshot_cache = copy.deepcopy(raw)
    _snapshot_settings_path = settings_path
    return count


def get_snapshot() -> Dict:
    """拿到启动时锁定的那份 hook 配置快照（运行期内容不变）。

    返回：dict 快照；还没加载过时返回空 dict。
    """
    return _snapshot_cache or {}


def get_disk_version() -> Dict:
    """现场重读一遍磁盘上的 hook 配置，返回当前磁盘版本。

    背景：/hooks diff 命令用它和内存快照对比，让用户看出
    "我改了配置文件但还没重启会话，改动尚未生效"。

    返回：dict；文件不存在或解析失败时返回 {}（对比命令不能因此崩）。
    """
    if not _snapshot_settings_path or not _snapshot_settings_path.exists():
        return {}
    try:
        config = json.loads(_snapshot_settings_path.read_text(encoding="utf-8"))
        return config.get("hooks", {}) if isinstance(config, dict) else {}
    except Exception as e:
        logger.debug("读取磁盘 hook 配置失败: %s", e)
        return {}


def reset_snapshot() -> None:
    """清空内存里的配置快照，回到"从未加载"状态。

    背景：主要给测试用——每个测试用例都要从干净的快照状态开始，否则上一个个例子的配置会串场。
    """
    global _snapshot_cache, _snapshot_settings_path
    _snapshot_cache = None
    _snapshot_settings_path = None


def _parse_hook(h_cfg: dict, event: HookEvent):
    """把 settings.json 里的一条 hook 配置翻译成 Hook 对象。

    支持 5 种类型：command（跑命令）/ http（调 URL）/ mcp_tool（调 MCP 工具）/
    prompt（调 LLM）/ agent（起子代理）。哪类缺了关键字段就跳过这条
    （返回 None 并打 warning，不连累其他 hook）。

    参数：
        h_cfg：单个 hook 的配置 dict（如 {"name": ..., "type": "command", "command": [...]}）
        event：这条 hook 挂在哪个事件上（如 PRE_TOOL_USE）

    返回：
        组装好的 Hook 对象；配置不完整/类型不认识时返回 None（调用方跳过）。
    """
    name = h_cfg.get("name")
    if not name:
        logger.warning("settings.json hook 缺 name 字段，跳过: %s", h_cfg)
        return None

    ht = h_cfg.get("type", "command")
    timeout = h_cfg.get("timeout", 10.0)
    # if 条件过滤——只在 4 个工具相关事件上生效，挂到别的事件上会被忽略并提醒
    if_cond = h_cfg.get("if")
    if if_cond and event.value not in (
        "pre_tool_use", "post_tool_use", "post_tool_use_failure", "permission_request"
    ):
        logger.warning(
            "hook '%s' if 条件 '%s' 不适用事件 %s（仅 tool 相关事件支持），已忽略",
            name, if_cond, event.value,
        )
        if_cond = None

    if ht == "command":
        command = h_cfg.get("command")
        if not command or not isinstance(command, list):
            logger.warning("hook '%s' 缺 command（或非 list），跳过", name)
            return None
        script = HookScriptConfig(handler_type="command", command=command, timeout=timeout,
                                  env=h_cfg.get("env"), if_condition=if_cond,
                                  # 异步 hook——放后台跑，退出码 2 时把 agent 重新唤醒
                                  async_run=bool(h_cfg.get("async", False)),
                                  async_rewake=bool(h_cfg.get("async_rewake", False)),
                                  status_message=h_cfg.get("status_message"))
    elif ht == "http":
        url = h_cfg.get("url")
        if not url:
            logger.warning("hook '%s' (http) 缺 url，跳过", name)
            return None
        script = HookScriptConfig(handler_type="http", url=url, timeout=timeout,
                                  if_condition=if_cond)
    elif ht == "mcp_tool":
        server = h_cfg.get("server")
        tool = h_cfg.get("tool")
        if not server or not tool:
            logger.warning("hook '%s' (mcp_tool) 缺 server/tool，跳过", name)
            return None
        script = HookScriptConfig(handler_type="mcp_tool", mcp_server=server,
                                  mcp_tool=tool, timeout=timeout, if_condition=if_cond)
    elif ht == "prompt":
        prompt = h_cfg.get("prompt")
        if not prompt:
            logger.warning("hook '%s' (prompt) 缺 prompt，跳过", name)
            return None
        script = HookScriptConfig(handler_type="prompt", prompt=prompt, timeout=timeout,
                                  if_condition=if_cond)
    elif ht == "agent":
        prompt = h_cfg.get("prompt") or ""
        agent_name = h_cfg.get("agent")
        if not prompt and not agent_name:
            logger.warning("hook '%s' (agent) 缺 prompt/agent，跳过", name)
            return None
        script = HookScriptConfig(handler_type="agent", prompt=prompt,
                                  agent_name=agent_name, timeout=timeout,
                                  if_condition=if_cond)
    else:
        logger.warning("hook '%s' 未知 type '%s'，跳过", name, ht)
        return None

    return Hook(
        name=name,
        event=event,
        kind="declarative",
        script=script,
        fail_closed=h_cfg.get("fail_closed", False),
        # if 条件过滤（借用 permission rule 的写法，详见 agent/hook_filter.py）
        once=h_cfg.get("once", False),  # 一次性 hook——触发一次后自动失效
        use_sandbox=h_cfg.get("use_sandbox", False),  # 给 hook 进程套 OS 沙箱再跑
    )
