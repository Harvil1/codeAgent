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
import re
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


def _wrap_with_sandbox(hook) -> list:
    """P3.8: 把 hook 命令用 sandbox_runner 包装。

    平台不支持/二进制未装时 fail-open 返回原始 command（log warning）。
    返回 argv list。
    """
    from agent import sandbox_runner
    try:
        # hook.script.command 是 list[str]（如 ["/bin/sh", "-c", "..."]），
        # sandbox_runner.wrap_command 接受 str（shell 命令），
        # 我们把 list 拼成 shell 命令字符串。
        raw_cmd = hook.script.command
        if isinstance(raw_cmd, list):
            # 简单拼接：用 shlex.quote 保护每个参数
            import shlex
            shell_cmd = " ".join(shlex.quote(str(x)) for x in raw_cmd)
        else:
            shell_cmd = str(raw_cmd)

        # 并发子代理 workspace：优先读 ContextVar（线程隔离），fallback 到 os.getcwd()
        from agent.workspace_context import get_workspace_cwd
        cwd = get_workspace_cwd()
        writable_roots = []  # hook 命令默认只可写 cwd
        return sandbox_runner.wrap_command(
            shell_cmd, cwd=cwd, writable_roots=writable_roots,
        )
    except Exception as e:
        # fail-open：沙箱不可用 → 降级到无沙箱（与 terminal_tool 同语义）
        logger.warning(
            "hook %s use_sandbox=True 但沙箱不可用，降级到无沙箱: %s",
            hook.name, e,
        )
        return hook.script.command


# ============================================================================
# 总入口：dispatch_hook
# ============================================================================


class HookExecutionError(RuntimeError):
    """声明式 hook 执行失败（启动失败/超时/非零退出/输出解析失败）。

    R30b-A6：fail_closed=True 的 hook 在失败路径抛出。只有 pre_tool_use
    路径（dispatch_hook(propagate_error=True)）会把异常放行给 registry
    转为 deny；其他事件维持 fail-open（吞掉返回 None）。
    """


def _fail_closed_or_none(hook, reason: str) -> None:
    """失败分叉：fail_closed hook 抛 HookExecutionError，否则返回 None。"""
    if getattr(hook, "fail_closed", False):
        raise HookExecutionError(reason)
    return None


def dispatch_hook(
    hook, payload: dict, *, propagate_error: bool = False,
    timeout_cap: float = None,
) -> Optional[dict]:
    """按 hook.script.handler_type 分发到对应执行器。

    所有执行器异常都被吞掉返回 None（fail-open），与原 run_script_hook 一致。

    R30b-A6：propagate_error=True（仅 pre_tool_use 路径用）且 hook.fail_closed
    时，异常向上抛而不是吞掉——否则 registry 的 fail_closed 分支永远不可达
    （声明式 hook 配了 fail_closed: true 出错仍是 fail-open 放行）。

    R30g-M8：timeout_cap 非空时钳制执行器超时（取 min；SessionEnd 用 1.5s
    防 teardown 被慢 hook 卡死）。仅对有超时语义的 command/http 生效。

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
        # R30g-M8：仅在 cap 存在时传参（保持旧两参调用/测试 mock 兼容）
        _tkw = {"timeout_cap": timeout_cap} if timeout_cap is not None else {}
        if ht == "command":
            return run_script_hook(hook, payload, **_tkw)
        if ht == "http":
            return run_http_hook(hook, payload, **_tkw)
        if ht == "mcp_tool":
            return run_mcp_tool_hook(hook, payload)
        if ht == "prompt":
            return run_prompt_hook(hook, payload)
        if ht == "agent":
            return run_agent_hook(hook, payload)
        logger.warning("未知 handler_type %s（hook %s）", ht, hook.name)
        return None
    except Exception as e:
        if propagate_error and getattr(hook, "fail_closed", False):
            raise  # pre_tool_use 的 fail_closed 语义：异常 → registry 转 deny
        logger.warning("hook %s (%s) 执行失败（fail-open）: %s", hook.name, ht, e)
        return None


# ============================================================================
# command 类型（原 run_script_hook，保持不动）
# ============================================================================


def run_script_hook(hook, payload: dict, timeout_cap: float = None) -> Optional[dict]:
    """在子进程中执行声明式 hook。

    参数：
        hook: Hook 实例（kind="declarative"，script 非 None）
        payload: 要传给子进程的 dict（含 event/session_id/timestamp/事件字段）
        timeout_cap: R30g-M8 可选超时钳制（取 min；SessionEnd 用 1.5s）

    返回：
        解析后的 dict（可能为空 dict 表示 allow），或 None（任何故障）

    P3.8: hook.use_sandbox=True 时，命令被 sandbox_runner 包装（仅 Unix 可用）。
          Windows/平台不支持时 fail-open 降级到无沙箱（log warning）。
    CCAR14 Task 2: Windows + use_sandbox=True 走 Job Object 模式——命令不包装，
          正常 Popen 启动后 attach_job（对齐 terminal_tool 的 CCAR12 分支）；
          attach 失败 fail-open 继续（log warning）。
    R30b-A6：失败分支（启动失败/超时/非零退出/输出解析失败）在 hook.fail_closed
          时抛 HookExecutionError——经 pre_tool_use 路径转为 deny；exit 2 是
          主动 block 决策不算失败，维持原协议。
    """
    if hook.script is None:
        logger.warning("hook %s 缺 script 配置", hook.name)
        return _fail_closed_or_none(hook, f"hook {hook.name} 缺 script 配置")

    if not hook.script.command:
        logger.warning("hook %s command 类型但 command 为空", hook.name)
        return _fail_closed_or_none(hook, f"hook {hook.name} command 为空")

    payload_json = json.dumps(payload, ensure_ascii=False)
    # R30d-B6a：hook 子进程 env 不再继承宿主全量（含 API key 等敏感变量），
    # 改用 terminal 同款 build_safe_env（洗掉密钥类）+ hook 自身声明的 env 覆盖
    from agent.sandbox_env import build_safe_env
    env = {**build_safe_env(), **(hook.script.env or {})}
    # R30g-M8：超时钳制（cap 更小则用 cap）
    effective_timeout = hook.script.timeout
    if timeout_cap is not None:
        effective_timeout = min(effective_timeout, float(timeout_cap))

    # P3.8: 可选 sandbox 包装
    argv = hook.script.command
    use_sandbox = getattr(hook, "use_sandbox", False)
    # CCAR14 Task 2: Windows Job Object 模式（命令不包装，启动后挂 job）
    win_job_mode = False
    if use_sandbox:
        from agent import sandbox_runner
        try:
            win_job_mode = (
                sandbox_runner.uses_job_object()
                and sandbox_runner.is_available()
            )
        except Exception as e:
            # 查询失败按非 Job 模式处理（下面走 Unix wrapper 降级）
            logger.warning("hook %s 沙箱模式查询失败（降级到 wrapper 路径）: %s",
                           hook.name, e)
            win_job_mode = False
        if not win_job_mode:
            argv = _wrap_with_sandbox(hook)

    if win_job_mode:
        # ============================================================
        # CCAR14 Task 2: Windows Job Object 分支
        # 公共路径提取到 sandbox_runner.run_with_job_object（照 terminal_tool
        # 模式：命令不包装正常启动 → attach_job → try communicate /
        # finally job.close()，句柄保活到进程结束：早关会在子进程还在跑时
        # 触发全树 kill——那是误杀；超时收尸后重抛，错误语义留在本函数）
        # ============================================================
        from agent.sandbox_runner import run_with_job_object
        try:
            result = run_with_job_object(
                argv,
                shell=False,
                timeout=effective_timeout,
                env=env,
                input=payload_json,
            )
        except OSError as e:
            logger.warning("hook %s 启动失败: %s", hook.name, e)
            return _fail_closed_or_none(hook, f"hook {hook.name} 启动失败: {e}")
        except subprocess.TimeoutExpired:
            logger.warning("hook %s 超时 (%.1fs)", hook.name, effective_timeout)
            return _fail_closed_or_none(hook, f"hook {hook.name} 超时")
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode
    else:
        # 原路径：subprocess.run（无沙箱 / Unix wrapper 包装后）
        try:
            proc = subprocess.run(
                argv,
                input=payload_json,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=effective_timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.warning("hook %s 超时 (%.1fs)", hook.name, effective_timeout)
            return _fail_closed_or_none(hook, f"hook {hook.name} 超时")
        except OSError as e:
            logger.warning("hook %s 启动失败: %s", hook.name, e)
            return _fail_closed_or_none(hook, f"hook {hook.name} 启动失败: {e}")
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode

    if returncode != 0:
        # P3.7: exit code 2 = blocking 协议（对齐 claude-code-main）
        # stderr 作为阻塞原因，返回特殊 dict 让调用方识别为 block。
        # 与 fail-open（None）区分：None = 静默失败，block = 主动拒绝。
        if returncode == 2:
            stderr_txt = (stderr or "").strip()
            logger.info("hook %s exit 2 (block): %s", hook.name, stderr_txt[:200])
            return {"action": "block", "reason": stderr_txt or "hook blocked (exit 2)"}
        logger.warning("hook %s exit %d: %s",
                       hook.name, returncode, (stderr or "")[:200])
        return _fail_closed_or_none(
            hook, f"hook {hook.name} exit {returncode}: {(stderr or '')[:200]}"
        )

    stdout = (stdout or "").strip()
    if not stdout:
        return {}  # 空 stdout = allow

    try:
        parsed = json.loads(stdout)
        if not isinstance(parsed, dict):
            logger.warning("hook %s stdout 非合法 JSON dict: %r", hook.name, parsed)
            return _fail_closed_or_none(hook, f"hook {hook.name} stdout 非 JSON dict")
        return parsed
    except json.JSONDecodeError as e:
        logger.warning("hook %s stdout 非合法 JSON: %s", hook.name, e)
        return _fail_closed_or_none(hook, f"hook {hook.name} stdout 非合法 JSON: {e}")


# ============================================================================
# http 类型
# ============================================================================


# R25 #5 → #8：http hook ${VAR} 插值（仅白名单变量）
_ENV_INTERP_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def interpolate_env_vars(value: str, allowed: list) -> str:
    """对字符串做 ${VAR} 环境变量插值，只允许白名单里的变量名。

    对齐 CCB httpHookAllowedEnvVars：非白名单引用保留原样并告警
    （防 hook 配置把 API key 等敏感 env 悄悄发出去；也防用户以为
    会插值实际没插的静默错配）。
    """
    import os as _os
    warned = set()

    def _sub(m):
        name = m.group(1)
        if name in allowed and name in _os.environ:
            return _os.environ[name]
        warned.add(name)
        return m.group(0)

    out = _ENV_INTERP_RE.sub(_sub, value)
    for n in sorted(warned):
        logger.warning(
            "http hook 引用了未白名单环境变量 ${%s}（保留原样）；"
            "如需插值请把它加进 security.http_hook_allowed_env_vars", n,
        )
    return out


def run_http_hook(hook, payload: dict, timeout_cap: float = None) -> Optional[dict]:
    """POST JSON payload 到 hook.script.url，解析响应 JSON dict。

    R16 #4 SSRF 防护（agent/ssrf_guard.py）：
    - URL allowlist（config security.http_hook_allowed_urls：None 不限/[] 全拒/
      非空必须匹配，* 通配）
    - DNS 预检：解析 IP 落私网/链路本地/云元数据段即拒；环回放行；
      环境代理激活时跳过预检
    - 禁重定向（重定向可绕过预检弹内网）
    - URL 含 CR/LF/NUL 拒
    命中防护 → 不外呼，返回 None（与 hook fail-open 语义一致：hook 没跑）。

    非 200 / 响应非 dict / JSON 解析失败 → None（fail-open）。
    """
    if not hook.script.url:
        logger.warning("http hook %s 缺 url", hook.name)
        return None
    url = hook.script.url.strip()

    config = _get_config()
    # R25 #8：${VAR} 插值（仅白名单；默认 [] = 完全不插值，原样发送）
    allowed_env = []
    if config is not None:
        allowed_env = [
            str(v) for v in
            ((config.get("security") or {}).get("http_hook_allowed_env_vars") or [])
        ]
    url = interpolate_env_vars(url, allowed_env)

    from agent.ssrf_guard import check_url_against_allowlist, validate_url_for_ssrf

    # URL allowlist
    allowed = None
    if config is not None:
        allowed = (config.get("security") or {}).get("http_hook_allowed_urls")
    block = check_url_against_allowlist(url, allowed)
    if block:
        logger.warning("http hook %s 被 URL allowlist 拦截: %s", hook.name, block)
        return None

    # SSRF 地址段预检
    err = validate_url_for_ssrf(url)
    if err:
        logger.warning("http hook %s SSRF 防护拦截: %s", hook.name, err)
        return None

    resp = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=(
            min(hook.script.timeout, float(timeout_cap))
            if timeout_cap is not None else hook.script.timeout
        ),  # R30g-M8：SessionEnd 等场景钳制
        allow_redirects=False,  # R16 #4: 重定向可绕过预检
    )
    # R30d-H1：响应体大小上限（防恶意/异常 server 用超大 body 打爆内存；
    # requests 已下载完成，这里至少阻止超大 JSON 的解析放大）。
    # getattr 容错：非 requests 传输/测试替身无 content 属性时按小 body 处理。
    _body = getattr(resp, "content", None) or b""
    if len(_body) > _MAX_HTTP_HOOK_BODY_BYTES:
        logger.warning(
            "http hook %s 响应体 %d bytes 超上限 %d，丢弃",
            hook.name, len(_body), _MAX_HTTP_HOOK_BODY_BYTES,
        )
        return None
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


# R30d-H1：http hook 响应体上限（1MB——hook 协议只需小 JSON 决策）
_MAX_HTTP_HOOK_BODY_BYTES = 1_000_000


class _SafeFormatDict(dict):
    """format_map 的安全 dict：缺 key 时保留 {key} 原样（不抛 KeyError）。"""

    def __missing__(self, key):
        return "{" + key + "}"


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

    # R30d-H2：真·安全 format——format_map + _SafeFormatDict，payload 缺字段
    # 保留 {field} 原样（此前裸 .format(**payload) 缺 key 直接 KeyError 抛穿）
    prompt_text = (
        (hook.script.prompt or "").format_map(_SafeFormatDict(payload))
        if hook.script.prompt else ""
    )
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
