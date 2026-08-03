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
"""
import json
import logging
from pathlib import Path

from agent.hooks import Hook, HookEvent, HookScriptConfig

logger = logging.getLogger(__name__)


def load_declarative_hooks(registry, settings_path: Path) -> int:
    """从 settings.json 加载声明式 hooks。

    返回加载成功的 hook 数量。
    文件不存在 = 静默返回 0。
    """
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
    return count


def _parse_hook(h_cfg: dict, event: HookEvent):
    """解析单个 hook 配置。缺关键字段时返回 None + log warning。"""
    name = h_cfg.get("name")
    if not name:
        logger.warning("settings.json hook 缺 name 字段，跳过: %s", h_cfg)
        return None

    ht = h_cfg.get("type", "command")
    timeout = h_cfg.get("timeout", 10.0)

    if ht == "command":
        command = h_cfg.get("command")
        if not command or not isinstance(command, list):
            logger.warning("hook '%s' 缺 command（或非 list），跳过", name)
            return None
        script = HookScriptConfig(handler_type="command", command=command, timeout=timeout,
                                  env=h_cfg.get("env"))
    elif ht == "http":
        url = h_cfg.get("url")
        if not url:
            logger.warning("hook '%s' (http) 缺 url，跳过", name)
            return None
        script = HookScriptConfig(handler_type="http", url=url, timeout=timeout)
    elif ht == "mcp_tool":
        server = h_cfg.get("server")
        tool = h_cfg.get("tool")
        if not server or not tool:
            logger.warning("hook '%s' (mcp_tool) 缺 server/tool，跳过", name)
            return None
        script = HookScriptConfig(handler_type="mcp_tool", mcp_server=server,
                                  mcp_tool=tool, timeout=timeout)
    elif ht == "prompt":
        prompt = h_cfg.get("prompt")
        if not prompt:
            logger.warning("hook '%s' (prompt) 缺 prompt，跳过", name)
            return None
        script = HookScriptConfig(handler_type="prompt", prompt=prompt, timeout=timeout)
    elif ht == "agent":
        prompt = h_cfg.get("prompt") or ""
        agent_name = h_cfg.get("agent")
        if not prompt and not agent_name:
            logger.warning("hook '%s' (agent) 缺 prompt/agent，跳过", name)
            return None
        script = HookScriptConfig(handler_type="agent", prompt=prompt,
                                  agent_name=agent_name, timeout=timeout)
    else:
        logger.warning("hook '%s' 未知 type '%s'，跳过", name, ht)
        return None

    return Hook(
        name=name,
        event=event,
        kind="declarative",
        script=script,
        fail_closed=h_cfg.get("fail_closed", False),
    )
