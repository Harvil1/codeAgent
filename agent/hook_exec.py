"""声明式 hook 的执行入口 + JSON IPC 协议。

5 种 handler_type（F1 扩展）：
- command:  本地子进程。stdin 收 payload JSON，stdout 期望合法 JSON。
- http:     POST JSON 到 url，解析响应 JSON。
- mcp_tool: 调 MCP 工具 mcp_server.mcp_tool。
- prompt:   单轮 LLM 评估（走 aux_llm_router，若未注入则 fail-open）。
- agent:    多轮子代理判断（复用 delegate_tool._run_child）。

子进程协议：
- stdin 收到 payload 的 JSON
- stdout 期望是合法 JSON（空 stdout = 空 dict）
- 超时 kill
- exit code != 0 / 启动失败 / JSON 解析失败 → None（fail-open）

全部 handler 异常 → None（fail-open）。
"""
import json
import logging
import os
import subprocess
from typing import Optional

import requests  # 模块级，便于测试 monkeypatch（he.requests.post）

logger = logging.getLogger(__name__)


# ============================================================================
# aux_llm router 注入（F1）
#
# agent/aux_llm.py 的 AuxLLMRouter 需要在 cli.py / RuntimeContext 构造时拿
# main_client + endpoints，没有全局单例 getter。我们提供一个模块级 provider
# 注入点：cli.py 启动时调 set_aux_router_provider(lambda: aux_llm_router)，
# run_prompt_hook 通过 provider 拿 router。未注入时返回 None → fail-open。
# ============================================================================
_AUX_ROUTER_PROVIDER = None


def set_aux_router_provider(provider) -> None:
    """注入一个返回 AuxLLMRouter 实例（或 None）的 callable。

    cli.py 在构造完 aux_llm_router 后调一次：
        from agent.hook_exec import set_aux_router_provider
        set_aux_router_provider(lambda: aux_llm_router)
    """
    global _AUX_ROUTER_PROVIDER
    _AUX_ROUTER_PROVIDER = provider


def _get_aux_router():
    """取注入的 AuxLLMRouter（可能为 None）。"""
    if _AUX_ROUTER_PROVIDER is None:
        return None
    try:
        return _AUX_ROUTER_PROVIDER()
    except Exception as e:
        logger.warning("aux_router provider 抛异常（视为不可用）: %s", e)
        return None


# ============================================================================
# config 注入（P3.2）：dispatch_hook 需要 config 读 feature flag 门控
# http / mcp_tool / agent 三种 handler 类型。模块级 provider（与 aux_router 同模式），
# cli.py 启动时注入 lambda: self.config，dispatch_hook 通过它拿 config dict。
# 未注入 → 视为全部 handler 允许（向后兼容，避免破坏现有测试）。
# ============================================================================
_CONFIG_PROVIDER = None


def set_config_provider(provider) -> None:
    """注入一个返回 config dict（或 None）的 callable。

    cli.py 在 RuntimeContext 构造完后调一次：
        from agent.hook_exec import set_config_provider
        set_config_provider(lambda: self.config)
    """
    global _CONFIG_PROVIDER
    _CONFIG_PROVIDER = provider


def _get_config():
    """取注入的 config dict（可能为 None）。"""
    if _CONFIG_PROVIDER is None:
        return None
    try:
        return _CONFIG_PROVIDER()
    except Exception as e:
        logger.warning("config provider 抛异常（视为不可用）: %s", e)
        return None


# P3.2: handler_type → feature flag 名的映射
_HANDLER_FLAG_MAP = {
    "http": "hook_http_handler",
    "mcp_tool": "hook_mcp_tool_handler",
    "agent": "hook_agent_handler",
}


def _is_handler_allowed(handler_type: str) -> bool:
    """检查该 handler_type 是否被 feature flag 允许。

    - command / prompt：无 flag 门控，永远允许（基线 handler）
    - http / mcp_tool / agent：需要对应 flag enabled
    - config 未注入（None）：向后兼容，全部允许
    """
    flag_name = _HANDLER_FLAG_MAP.get(handler_type)
    if flag_name is None:
        return True  # command / prompt 无门控
    config = _get_config()
    if config is None:
        return True  # 未注入 config（测试场景），全放开
    from agent.feature_flags import is_feature_enabled
    return is_feature_enabled(config, flag_name)


# ============================================================================
# 总入口：dispatch_hook
# ============================================================================


def dispatch_hook(hook, payload: dict) -> Optional[dict]:
    """按 hook.script.handler_type 分发到对应执行器。

    所有执行器异常都被吞掉返回 None（fail-open），与原 run_script_hook 一致。

    P3.2: http / mcp_tool / agent 三种 handler 受 feature flag 门控。
    flag 未开启时直接返回 None（视为 skip），不调对应执行器。
    command / prompt 永远允许（基线）。
    """
    if hook.script is None:
        return None
    ht = getattr(hook.script, "handler_type", "command") or "command"
    # P3.2: flag 门控
    if not _is_handler_allowed(ht):
        logger.info(
            "hook %s handler_type '%s' 被门控关闭（feature flag 未开启），跳过",
            hook.name, ht,
        )
        return None
    try:
        if ht == "command":
            return run_script_hook(hook, payload)
        if ht == "http":
            return run_http_hook(hook, payload)
        if ht == "mcp_tool":
            return run_mcp_tool_hook(hook, payload)
        if ht == "prompt":
            return run_prompt_hook(hook, payload)
        if ht == "agent":
            return run_agent_hook(hook, payload)
        logger.warning("未知 handler_type %s（hook %s）", ht, hook.name)
        return None
    except Exception as e:
        logger.warning("hook %s (%s) 执行失败（fail-open）: %s", hook.name, ht, e)
        return None


# ============================================================================
# command 类型（原 run_script_hook，保持不动）
# ============================================================================


def run_script_hook(hook, payload: dict) -> Optional[dict]:
    """在子进程中执行声明式 hook。

    参数：
        hook: Hook 实例（kind="declarative"，script 非 None）
        payload: 要传给子进程的 dict（含 event/session_id/timestamp/事件字段）

    返回：
        解析后的 dict（可能为空 dict 表示 allow），或 None（任何故障）
    """
    if hook.script is None:
        logger.warning("hook %s 缺 script 配置", hook.name)
        return None

    if not hook.script.command:
        logger.warning("hook %s command 类型但 command 为空", hook.name)
        return None

    payload_json = json.dumps(payload, ensure_ascii=False)
    env = {**os.environ, **(hook.script.env or {})}

    try:
        proc = subprocess.run(
            hook.script.command,
            input=payload_json,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=hook.script.timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.warning("hook %s 超时 (%.1fs)", hook.name, hook.script.timeout)
        return None
    except OSError as e:
        logger.warning("hook %s 启动失败: %s", hook.name, e)
        return None

    if proc.returncode != 0:
        logger.warning("hook %s exit %d: %s",
                       hook.name, proc.returncode, (proc.stderr or "")[:200])
        return None

    stdout = (proc.stdout or "").strip()
    if not stdout:
        return {}  # 空 stdout = allow

    try:
        parsed = json.loads(stdout)
        if not isinstance(parsed, dict):
            logger.warning("hook %s stdout 非合法 JSON dict: %r", hook.name, parsed)
            return None
        return parsed
    except json.JSONDecodeError as e:
        logger.warning("hook %s stdout 非合法 JSON: %s", hook.name, e)
        return None


# ============================================================================
# http 类型
# ============================================================================


def run_http_hook(hook, payload: dict) -> Optional[dict]:
    """POST JSON payload 到 hook.script.url，解析响应 JSON dict。

    非 200 / 响应非 dict / JSON 解析失败 → None（fail-open）。
    """
    if not hook.script.url:
        logger.warning("http hook %s 缺 url", hook.name)
        return None
    resp = requests.post(
        hook.script.url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=hook.script.timeout,
    )
    if resp.status_code != 200:
        logger.warning("http hook %s 返回 %d", hook.name, resp.status_code)
        return None
    try:
        parsed = resp.json()
        return parsed if isinstance(parsed, dict) else None
    except Exception as e:
        logger.warning("http hook %s 响应非合法 JSON: %s", hook.name, e)
        return None


# ============================================================================
# mcp_tool 类型
# ============================================================================


def run_mcp_tool_hook(hook, payload: dict) -> Optional[dict]:
    """调 MCP 工具 mcp__{server}__{tool}。

    payload 整体作为 arguments 传入。失败 → None（fail-open）。
    """
    from agent.mcp_client import get_mcp_manager
    if not hook.script.mcp_server or not hook.script.mcp_tool:
        logger.warning("mcp_tool hook %s 缺 mcp_server/mcp_tool", hook.name)
        return None
    mgr = get_mcp_manager()
    full = f"mcp__{hook.script.mcp_server}__{hook.script.mcp_tool}"
    try:
        result = mgr.call(full, payload)
        return result if isinstance(result, dict) else {"result": result}
    except Exception as e:
        logger.warning("mcp_tool hook %s 调用失败: %s", hook.name, e)
        return None


# ============================================================================
# prompt 类型（单轮 aux_llm）
# ============================================================================


def run_prompt_hook(hook, payload: dict) -> Optional[dict]:
    """单轮 LLM 评估（走注入的 aux_llm_router）。

    hook.script.prompt 是模板字符串，会 .format(**payload)。
    router 未注入 → None（fail-open，不打断主流程）。
    """
    router = _get_aux_router()
    if router is None:
        logger.warning(
            "prompt hook %s 跳过：aux_llm router 未注入"
            "（cli.py 未调 set_aux_router_provider）",
            hook.name,
        )
        return None

    # 安全 format：payload 里缺字段时保持原样
    prompt_text = (hook.script.prompt or "").format(**payload) if hook.script.prompt else ""
    full_prompt = (
        f"{prompt_text}\n\n"
        f"payload: {json.dumps(payload, ensure_ascii=False)}\n"
        '返回 JSON：{"permissionDecision": "allow"|"deny", "reason": "..."}'
    )

    try:
        resp = router.chat_completions(
            messages=[{"role": "user", "content": full_prompt}],
        )
    except Exception as e:
        logger.warning("prompt hook %s LLM 调用失败: %s", hook.name, e)
        return None

    if not resp:
        return None

    # 响应可能是字符串或 OpenAI 风格 dict
    text = resp if isinstance(resp, str) else ""
    if isinstance(resp, dict):
        # OpenAI 风格：{choices: [{message: {content: "..."}}]}
        try:
            text = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = json.dumps(resp, ensure_ascii=False)
    elif hasattr(resp, "content"):
        text = getattr(resp, "content", "")

    try:
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    return None


# ============================================================================
# agent 类型（多轮子代理，复用 delegate）
# ============================================================================


def run_agent_hook(hook, payload: dict) -> Optional[dict]:
    """多轮子代理判断（复用 delegate_tool._run_child）。

    hook.script.prompt 是 goal 模板（.format(**payload)）。
    hook.script.agent_name 透传给 _run_child（自定义子代理名，可选）。
    返回 dict（解析子代理输出里的 JSON），解析失败返回 {"decision": "review", ...}。

    约束：hook_exec 是模块级函数，无父 agent 上下文，因此 _run_child 不传
    config/agent_ref。后果：
    - config：_run_child 内部从 load_config() 自取（fallback 路径），可正常运行。
    - agent_ref：无中断传播、无 pendingToolUseSummary、无 checkpoint 追踪。
    agent hook 设计用于一次性事件评估（不长任务运行），影响可控。
    如需补全，可参考 _AUX_ROUTER_PROVIDER 模式增设 _AGENT_REF_PROVIDER 注入点。
    """
    # 懒加载避免循环
    from tools.delegate_tool import _run_child

    base_prompt = hook.script.prompt or "判断以下事件是否允许，返回 allow/deny"
    # 安全 format
    try:
        goal = base_prompt.format(**payload)
    except (KeyError, IndexError):
        goal = base_prompt

    full_goal = goal + f"\n\npayload: {json.dumps(payload, ensure_ascii=False)}"

    # _run_child(goal, context, role, **kwargs)：agent_name 走 kwargs 透传
    kwargs = {}
    if hook.script.agent_name:
        kwargs["agent_name"] = hook.script.agent_name

    try:
        result = _run_child(
            goal=full_goal,
            context="",
            role="leaf",
            **kwargs,
        )
    except Exception as e:
        logger.warning("agent hook %s _run_child 失败: %s", hook.name, e)
        return None

    if not isinstance(result, str):
        return None

    try:
        import re
        m = re.search(r"\{.*\}", result, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
    except Exception:
        pass
    return {"decision": "review", "raw": result[:500]}
