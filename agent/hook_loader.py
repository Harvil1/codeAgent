"""加载 settings.json 中的声明式 hooks 到 registry。

文件格式：
{
  "hooks": {
    "user_prompt_submit": [{name, command, timeout?, env?, fail_closed?}, ...],
    "pre_tool_use": [...],
    "post_tool_use": [...],
    "stop": [...]
  }
}

校验规则：
- 顶层缺 hooks 字段 → raise ValueError
- event 名不合法 → raise ValueError
- 单个 hook 缺 name / command → 跳过 + log warning（不阻塞其他）
- 文件不存在 → 静默返回 0（不强制用户配置）

阶段 4 新增：SnapshotCache —— 启动加载后锁定 raw 配置，
运行期不重读磁盘。对齐 Claude Code "hook 配置在会话启动时锁定，运行期修改不立即生效" 的安全语义。
"""
import copy
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from agent.hooks import Hook, HookEvent, HookScriptConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SnapshotCache：启动时锁定，运行期只读（防篡改）
# ---------------------------------------------------------------------------

_snapshot_cache: Optional[Dict] = None  # {event_str: [hook_dict, ...]}
_snapshot_settings_path: Optional[Path] = None


def load_declarative_hooks(registry, settings_path: Path) -> int:
    """从 settings.json 加载声明式 hooks。

    返回加载成功的 hook 数量。
    文件不存在 = 静默返回 0。
    末尾把 raw 配置存到 SnapshotCache（防篡改）。
    """
    global _snapshot_cache, _snapshot_settings_path

    if not settings_path.exists():
        return 0

    text = settings_path.read_text(encoding="utf-8")
    config = json.loads(text)  # JSON 解析失败直接抛

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

    # 阶段 4：deepcopy 锁定，运行期不重读
    _snapshot_cache = copy.deepcopy(raw)
    _snapshot_settings_path = settings_path
    return count


def get_snapshot() -> Dict:
    """启动时锁定的 hook 配置快照（运行期不变）。"""
    return _snapshot_cache or {}


def get_disk_version() -> Dict:
    """重新读盘返回当前磁盘版本（用于 /hooks diff 检测）。

    文件不存在/解析失败 → 返回 {}。
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
    """测试用：重置 snapshot 状态。"""
    global _snapshot_cache, _snapshot_settings_path
    _snapshot_cache = None
    _snapshot_settings_path = None


def _parse_hook(h_cfg: dict, event: HookEvent):
    """解析单个 hook 配置。缺关键字段时返回 None + log warning。"""
    name = h_cfg.get("name")
    if not name:
        logger.warning("settings.json hook 缺 name 字段，跳过: %s", h_cfg)
        return None

    ht = h_cfg.get("type", "command")
    timeout = h_cfg.get("timeout", 10.0)
    # P3.5 NEW: if 条件过滤（仅 4 个工具相关事件生效，其他事件忽略）
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
                                  env=h_cfg.get("env"), if_condition=if_cond)
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
        # P3.5 NEW: if 条件过滤（permission rule 语法）
        once=h_cfg.get("once", False),  # P3.6 NEW: 一次性 hook
        use_sandbox=h_cfg.get("use_sandbox", False),  # P3.8 NEW: 套 sandbox
    )
