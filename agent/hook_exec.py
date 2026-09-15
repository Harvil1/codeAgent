"""声明式 hook 的执行器：按配置把 hook 真正跑起来，并约定 JSON 通信格式。

本文件在项目里的位置：agent/hooks.py 的 HookRegistry 只管"什么时候触发"，
具体"怎么执行"全在这里。被 agent/hook_loader.py（配置解析）和
cli.py（启动时注入依赖）使用。执行方式共 5 种：
- command:  起本地子进程跑命令。把 payload JSON 喂给它的标准输入（stdin），
            期望它往标准输出（stdout）吐一段合法 JSON 作为判决。
- http:     往 url POST 一份 JSON，把响应 JSON 当判决。
- mcp_tool: 调一个 MCP 外部工具（mcp_server 服务器上的 mcp_tool）。
- prompt:   让辅助小模型（aux_llm）单轮评估一次；router 没注入就跳过。
- agent:    让子代理多轮判断（复用 delegate_tool 的 _run_child）。

子进程的通信约定（协议）：
- 子进程从 stdin 收到 payload 的 JSON
- 子进程的 stdout 应该是合法 JSON（什么都没输出 = 空 dict，即"放行"）
- 超时就 kill 掉
- 退出码非 0 / 启动失败 / JSON 解析不出来 → 返回 None（当作没这个 hook）

兜底原则：所有执行器出异常都返回 None（fail-open，出问题的 hook
不能拖垮主流程），除非 hook 显式配了 fail_closed。
"""
import json
import logging
import os
import queue
import re
import subprocess
from typing import Optional

import requests  # 放在模块级，方便测试 monkeypatch（he.requests.post）

logger = logging.getLogger(__name__)


# ============================================================================
# 辅助小模型 router 的注入
#
# agent/aux_llm.py 的 AuxLLMRouter 没有全局单例的 getter，这里留一个
# 模块级的"提供者"注入点：cli.py 启动时调
# set_aux_router_provider(lambda: aux_llm_router)，run_prompt_hook 再通过
# provider 拿 router。没注入就返回 None → fail-open 跳过。
# ============================================================================
_AUX_ROUTER_PROVIDER = None


def set_aux_router_provider(provider) -> None:
    """注入一个"能返回 AuxLLMRouter 实例（或 None）的函数"。

    cli.py 在构造完 aux_llm_router 后调一次：
        from agent.hook_exec import set_aux_router_provider
        set_aux_router_provider(lambda: aux_llm_router)

    参数：
    - provider：无参函数，调用后返回 AuxLLMRouter 实例或 None
    """
    global _AUX_ROUTER_PROVIDER
    _AUX_ROUTER_PROVIDER = provider


def _get_aux_router():
    """取出注入的 AuxLLMRouter（可能为 None——没注入或注入方出异常时）。"""
    if _AUX_ROUTER_PROVIDER is None:
        return None
    try:
        return _AUX_ROUTER_PROVIDER()
    except Exception as e:
        logger.warning("aux_router provider 抛异常（视为不可用）: %s", e)
        return None


# ============================================================================
# config 的注入：dispatch_hook 需要读配置里的 feature flag
# （功能开关），决定 http / mcp_tool / agent 三种执行器允不允许用。
# 和上面 aux_router 同一套模式：cli.py 启动时注入 lambda: self.config，
# dispatch_hook 通过它拿 config dict。没注入就当作"全允许"
# （向后兼容，免得现有测试直接挂掉）。
# ============================================================================
_CONFIG_PROVIDER = None


def set_config_provider(provider) -> None:
    """注入一个"能返回 config dict（或 None）的函数"。

    cli.py 在 RuntimeContext 构造完后调一次：
        from agent.hook_exec import set_config_provider
        set_config_provider(lambda: self.config)

    参数：
    - provider：无参函数，调用后返回 config dict 或 None
    """
    global _CONFIG_PROVIDER
    _CONFIG_PROVIDER = provider


def _get_config():
    """取出注入的 config dict（可能为 None——没注入或注入方出异常时）。"""
    if _CONFIG_PROVIDER is None:
        return None
    try:
        return _CONFIG_PROVIDER()
    except Exception as e:
        logger.warning("config provider 抛异常（视为不可用）: %s", e)
        return None


# 执行器类型 → 对应 feature flag 名字的对照表
_HANDLER_FLAG_MAP = {
    "http": "hook_http_handler",
    "mcp_tool": "hook_mcp_tool_handler",
    "agent": "hook_agent_handler",
}


def _is_handler_allowed(handler_type: str) -> bool:
    """检查这种执行器类型有没有被 feature flag（功能开关）放行。

    参数：
    - handler_type：执行器类型字符串

    规则：
    - command / prompt：没有开关管着，永远允许（基线能力）
    - http / mcp_tool / agent：需要对应的 flag 处于开启状态
    - config 没注入（None）：向后兼容，全部允许
    """
    flag_name = _HANDLER_FLAG_MAP.get(handler_type)
    if flag_name is None:
        return True  # command / prompt 不受门控
    config = _get_config()
    if config is None:
        return True  # 没注入 config（多见于测试场景），全放开
    from agent.feature_flags import is_feature_enabled
    return is_feature_enabled(config, flag_name)


def _wrap_with_sandbox(hook) -> list:
    """把 hook 要跑的命令用 sandbox_runner（OS 沙箱）包一层，限制它能碰的文件范围。

    平台不支持或沙箱程序没装时降级——返回原始 command 并记条警告
    （fail-open，不能因为沙箱缺失让 hook 跑不了）。

    参数：
    - hook：要包装的声明式 hook

    返回：包装后的 argv 列表（可直接交给 Popen）。
    """
    from agent import sandbox_runner
    try:
        # hook.script.command 是 list[str]（如 ["/bin/sh", "-c", "..."]），
        # 而 sandbox_runner.wrap_command 只收 str（shell 命令字符串），
        # 所以先把 list 拼回 shell 命令
        raw_cmd = hook.script.command
        if isinstance(raw_cmd, list):
            # 用 shlex.quote 给每个参数加引号保护，防参数里带空格/特殊字符被拆错
            import shlex
            shell_cmd = " ".join(shlex.quote(str(x)) for x in raw_cmd)
        else:
            shell_cmd = str(raw_cmd)

        # 并发子代理各自有工作目录：优先读 ContextVar（线程隔离不串号），
        # 读不到再退回 os.getcwd()
        from agent.workspace_context import get_workspace_cwd
        cwd = get_workspace_cwd()
        writable_roots = []  # hook 命令默认只允许写 cwd（不额外开白名单）
        return sandbox_runner.wrap_command(
            shell_cmd, cwd=cwd, writable_roots=writable_roots,
        )
    except Exception as e:
        # fail-open：沙箱不可用就退回无沙箱执行（和 terminal_tool 同一套语义）
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

    fail_closed=True 的 hook 要在失败路径把这个异常
    抛出去。只有 pre_tool_use 路径（dispatch_hook 传 propagate_error=True）
    会放行异常，让 registry 把它转成 deny（拒绝执行工具）；其他事件仍按
    fail-open 处理（吞掉异常返回 None）。
    """


def _fail_closed_or_none(hook, reason: str) -> None:
    """失败时的分岔路：hook 配了 fail_closed 就抛 HookExecutionError，
    否则安静返回 None（当作 hook 没跑）。

    参数：hook 为出错的 hook；reason 为失败原因（进异常信息）。"""
    if getattr(hook, "fail_closed", False):
        raise HookExecutionError(reason)
    return None


# ============================================================================
# 异步 hook 的 rewake（重新叫醒）通知队列
# ============================================================================
# 异步 hook 在后台跑完后如果退出码是 2（block）且配了 async_rewake=True，
# 就往这个队列推一条通知；agent 主循环每轮用 _drain_injected_messages
# 把队列里的内容作为临时 <rewake_notification> 消息喂给模型。
_REWAKE_QUEUE: "queue.Queue" = queue.Queue()


def _push_rewake(hook_name: str, reason: str, status_message: str = "") -> None:
    """往队列推一条 rewake 通知（线程安全——后台线程和主循环都会碰它）。

    参数：
    - hook_name：哪个 hook 发的
    - reason：为什么叫醒（block 理由）
    - status_message：给人看的说明文字（可空）
    """
    import time as _time
    _REWAKE_QUEUE.put({
        "hook": hook_name,
        "reason": reason,
        "status_message": status_message or "",
        "pushed_at": _time.time(),
    })


def drain_rewake_notifications() -> list:
    """把队列里攒的 rewake 通知一次性取干取净（agent 每轮开头调）。

    返回：通知 dict 的列表（队列空了就是空列表）。
    """
    notes = []
    while True:
        try:
            notes.append(_REWAKE_QUEUE.get_nowait())
        except Exception:
            return notes


def _run_async_hook(hook, payload: dict, timeout_cap: float = None) -> None:
    """异步 command hook——丢到后台线程跑，dispatch 立刻返回不等它。

    后台跑完退出码是 2（block）且配了 async_rewake 时，推一条 rewake 通知
    （模型下一轮能看到并跟进）。其他结果直接扔掉——异步的语义本来就是
    "不等待判决"，所以要拦东西的（gating 类）hook 别用异步。

    参数：
    - hook：要跑的异步 hook
    - payload：事件数据
    - timeout_cap：超时上限（可空）
    """
    def _bg():
        try:
            result = run_script_hook(
                hook, payload,
                **({"timeout_cap": timeout_cap} if timeout_cap is not None else {}),
            )
            if (isinstance(result, dict) and result.get("action") == "block"
                    and hook.script.async_rewake):
                _push_rewake(
                    hook.name,
                    result.get("reason") or "async hook blocked (exit 2)",
                    status_message=hook.script.status_message or "",
                )
        except Exception as e:
            logger.warning("async hook %s 后台执行失败: %s", hook.name, e)

    import threading as _threading
    _threading.Thread(
        target=_bg, daemon=True, name=f"hook-async-{hook.name}",
    ).start()


def dispatch_hook(
    hook, payload: dict, *, propagate_error: bool = False,
    timeout_cap: float = None,
) -> Optional[dict]:
    """执行一个声明式 hook 的总入口：看配置选哪种执行器去跑。

    参数：
    - hook：要执行的 Hook 对象（script 配置里写了用哪种执行器）
    - payload：事件数据（会原样传给执行器）
    - propagate_error：True 时 fail_closed hook 的异常向上抛而不是吞掉
      （只有 pre_tool_use 路径用）
    - timeout_cap：超时上限（非空时和 hook 自配超时取较小者）

    返回：执行器产出的 dict（判决），或 None（没跑/失败/被跳过）。

    默认所有执行器的异常都吞掉返回 None（fail-open，和老的 run_script_hook
    行为一致）。

    propagate_error=True 且 hook 配了 fail_closed 时，
    异常必须向上抛而不是吞掉——否则 registry 的 fail_closed 分支永远走不到
    （等于声明式 hook 配了 fail_closed: true，出错时却照样 fail-open 放行）。

    timeout_cap 非空时钳住执行器超时（取较小值；会话结束时
    传 1.5 秒，防止收尾被慢 hook 卡死）。只对 command/http 这两种有超时
    概念的执行器生效。

    http / mcp_tool / agent 三种执行器受 feature flag 门控，
    flag 没开就直接返回 None（当作跳过），不碰对应执行器；
    command / prompt 永远允许（基线能力）。
    """
    if hook.script is None:
        return None
    ht = getattr(hook.script, "handler_type", "command") or "command"
    # 异步 command hook——后台跑不阻塞，立刻返回 None
    if (ht == "command"
            and getattr(hook.script, "async_run", False)):
        _run_async_hook(hook, payload, timeout_cap)
        return None
    # feature flag 门控
    if not _is_handler_allowed(ht):
        logger.info(
            "hook %s handler_type '%s' 被门控关闭（feature flag 未开启），跳过",
            hook.name, ht,
        )
        return None
    try:
        # 只在 cap 存在时才传参（保持旧的两参调用/测试 mock 兼容）
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
            raise  # pre_tool_use 的 fail_closed 语义：异常抛上去让 registry 转成 deny
        logger.warning("hook %s (%s) 执行失败（fail-open）: %s", hook.name, ht, e)
        return None


# ============================================================================
# command 类型：起子进程跑命令
# ============================================================================


def run_script_hook(hook, payload: dict, timeout_cap: float = None) -> Optional[dict]:
    """在子进程里执行声明式 hook 命令，按约定解析它的输出。

    参数：
        hook: Hook 实例（kind="declarative"，script 非 None）
        payload: 要传给子进程的 dict（含 event/session_id/timestamp/事件字段）
        timeout_cap: 可选的超时上限（取较小值；
                     会话结束时传 1.5s 防收尾被卡）

    返回：
        解析后的 dict（空 dict 表示"放行"），或 None（出了任何故障）

    沙箱相关：
    hook 配了 use_sandbox=True 时，命令会被 sandbox_runner 包装
          （只有 Unix 可用）；Windows/平台不支持时降级成无沙箱跑（记警告）。
    Windows + use_sandbox=True 走 Job Object（进程管控）模式——
          命令不做包装，正常 Popen 启动后把进程挂进 job（与 terminal_tool
          同一套流程）；挂 job 失败也降级继续跑（记警告）。
    各失败分支（启动失败/超时/非零退出/输出解析失败）
          在 hook.fail_closed 时要抛 HookExecutionError——经 pre_tool_use
          路径转成 deny；注意 exit 2 是 hook 主动"拦截"的决策、不算失败，
          维持原协议不走异常。
    """
    if hook.script is None:
        logger.warning("hook %s 缺 script 配置", hook.name)
        return _fail_closed_or_none(hook, f"hook {hook.name} 缺 script 配置")

    if not hook.script.command:
        logger.warning("hook %s command 类型但 command 为空", hook.name)
        return _fail_closed_or_none(hook, f"hook {hook.name} command 为空")

    payload_json = json.dumps(payload, ensure_ascii=False)
    # hook 子进程的环境变量不原样继承宿主的全量
    # （里面含 API key 等敏感信息），用 terminal 同款 build_safe_env
    # （把密钥类洗掉）+ hook 配置里自己声明的 env 覆盖
    from agent.sandbox_env import build_safe_env
    env = {**build_safe_env(), **(hook.script.env or {})}
    # 超时钳制（cap 更小就听 cap 的）
    effective_timeout = hook.script.timeout
    if timeout_cap is not None:
        effective_timeout = min(effective_timeout, float(timeout_cap))

    # 可选的沙箱包装
    argv = hook.script.command
    use_sandbox = getattr(hook, "use_sandbox", False)
    # Windows Job Object 模式（命令不包装，启动后再挂进 job）
    win_job_mode = False
    if use_sandbox:
        from agent import sandbox_runner
        try:
            win_job_mode = (
                sandbox_runner.uses_job_object()
                and sandbox_runner.is_available()
            )
        except Exception as e:
            # 查询失败就当非 Job 模式处理（下面走 Unix wrapper 的降级路径）
            logger.warning("hook %s 沙箱模式查询失败（降级到 wrapper 路径）: %s",
                           hook.name, e)
            win_job_mode = False
        if not win_job_mode:
            argv = _wrap_with_sandbox(hook)

    if win_job_mode:
        # ============================================================
        # Windows Job Object 分支
        # 公共流程提取到了 sandbox_runner.run_with_job_object（照抄
        # terminal_tool 的模式：命令不包装、正常启动 → 挂进 job →
        # try 里 communicate / finally 里关 job。job 句柄必须保活到进程
        # 结束——提前关会触发整棵进程树被杀，那是误杀；超时先收尸再
        # 把异常重抛出来，错误怎么报留给本函数决定）
        # ============================================================
        from agent.sandbox_runner import run_with_job_object
        try:
            result = run_with_job_object(
                argv,
                shell=False,
                timeout=effective_timeout,
                env=env,
                input=payload_json,
                # GBK 控制台脚本输出中文时严格解码会抛 UnicodeDecodeError
                # 穿透拦截协议（exit 2 的 block 判决整条丢失）——replace 兜底
                errors="replace",
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
        # 非 Job 模式路径：直接 subprocess.run（没开沙箱，或 Unix 下被 wrapper 包装过）
        try:
            proc = subprocess.run(
                argv,
                input=payload_json,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",  # GBK 输出兜底（同上，防解码炸穿 exit 2 协议）
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
        # exit code 2 = "拦截"协议——
        # stderr 里写的是拦截原因，返回特殊 dict 让调用方识别成 block。
        # 要和 fail-open（None）区分开：None = 出故障静默跳过，
        # block = hook 主动说"不行"。
        # 注意顺序：exit 2 的判断在非零退出分支里、fail_closed 之前——
        # 主动拦截不算失败，不能被转成异常。
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
        return {}  # 什么都没输出 = 默认放行（空 dict）

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
# http 类型：发 HTTP 请求
# ============================================================================


# http hook 的 ${VAR} 环境变量插值（只允许白名单里的变量）
_ENV_INTERP_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def interpolate_env_vars(value: str, allowed: list) -> str:
    """把字符串里的 ${VAR} 换成环境变量的值——但只换白名单点过名的变量（不加限制可能把 API key 等敏感变量悄悄发到外部网址）。

    不在白名单的引用一律原样保留 ${VAR} 字样并记警告——既防泄漏，也防
    "用户以为会插值、实际没插"的静默错配。

    参数：
    - value：待处理的字符串（通常是 url）
    - allowed：允许插值的变量名列表

    返回：处理后的字符串。
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
    """把 payload POST 到 hook 配置的 url，把响应 JSON 当判决返回。

    参数：
    - hook：要执行的 http 类型 hook
    - payload：事件数据（POST 的请求体）
    - timeout_cap：超时上限（非空时取较小值）

    返回：响应里的 JSON dict，或 None（没跑/失败）。

    SSRF 防护（防"让服务器替攻击者访问内网"，实现在
    agent/ssrf_guard.py），发请求之前过四道检查：
    - URL 白名单（配置 security.http_hook_allowed_urls：None = 不限 /
      空列表 = 全拒 / 非空必须匹配上，支持 * 通配）
    - DNS 预检：把域名解析成 IP，落在私网段/链路本地段/云元数据段就拒；
      环回地址（127.x、::1）放行；环境里配了代理时跳过预检
    - 禁止重定向（跟着重定向走可以绕过预检摸进内网）
    - URL 里带换行符（CR/LF）或 NUL 就拒
    命中任何一条 → 根本不发请求，返回 None（和 hook 的 fail-open 语义
    一致：等于这个 hook 没跑）。

    此外：响应不是 200 / 响应不是 JSON dict / JSON 解析失败 → None。
    """
    if not hook.script.url:
        logger.warning("http hook %s 缺 url", hook.name)
        return None
    url = hook.script.url.strip()

    config = _get_config()
    # ${VAR} 插值（只认白名单；默认空列表 = 完全不插值，原样发送）
    allowed_env = []
    if config is not None:
        allowed_env = [
            str(v) for v in
            ((config.get("security") or {}).get("http_hook_allowed_env_vars") or [])
        ]
    url = interpolate_env_vars(url, allowed_env)

    from agent.ssrf_guard import check_url_against_allowlist, validate_url_for_ssrf

    # 第一道：URL 白名单
    allowed = None
    if config is not None:
        allowed = (config.get("security") or {}).get("http_hook_allowed_urls")
    block = check_url_against_allowlist(url, allowed)
    if block:
        logger.warning("http hook %s 被 URL allowlist 拦截: %s", hook.name, block)
        # fail_closed 对 pre_tool_use 要兑现成"拦截"——静默 None 等于放行
        return _fail_closed_or_none(
            hook, f"http hook {hook.name} 被 URL allowlist 拦截: {block}")

    # 第二道：SSRF 地址段预检
    err = validate_url_for_ssrf(url)
    if err:
        logger.warning("http hook %s SSRF 防护拦截: %s", hook.name, err)
        return _fail_closed_or_none(
            hook, f"http hook {hook.name} SSRF 防护拦截: {err}")

    resp = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=(
            min(hook.script.timeout, float(timeout_cap))
            if timeout_cap is not None else hook.script.timeout
        ),  # 会话结束等场景的超时钳制
        allow_redirects=False,  # 跟着重定向走能绕过预检，所以直接禁掉
    )
    # 响应体大小上限（防恶意/异常服务器用超大 body 打爆内存；
    # requests 拿到手时已经下载完了，这里至少挡住超大 JSON 的解析放大效应）。
    # getattr 容错：非 requests 传输/测试替身没有 content 属性时按小 body 处理。
    _body = getattr(resp, "content", None) or b""
    if len(_body) > _MAX_HTTP_HOOK_BODY_BYTES:
        logger.warning(
            "http hook %s 响应体 %d bytes 超上限 %d，丢弃",
            hook.name, len(_body), _MAX_HTTP_HOOK_BODY_BYTES,
        )
        return _fail_closed_or_none(
            hook, f"http hook {hook.name} 响应体超上限")
    if resp.status_code != 200:
        logger.warning("http hook %s 返回 %d", hook.name, resp.status_code)
        return _fail_closed_or_none(
            hook, f"http hook {hook.name} 返回 {resp.status_code}")
    try:
        parsed = resp.json()
        return parsed if isinstance(parsed, dict) else None
    except Exception as e:
        logger.warning("http hook %s 响应非合法 JSON: %s", hook.name, e)
        return _fail_closed_or_none(
            hook, f"http hook {hook.name} 响应非合法 JSON: {e}")


# ============================================================================
# mcp_tool 类型：调 MCP 外部工具
# ============================================================================


def run_mcp_tool_hook(hook, payload: dict) -> Optional[dict]:
    """调用 MCP 工具 mcp__{server}__{tool}，把工具返回当判决。

    参数：
    - hook：要执行的 mcp_tool 类型 hook
    - payload：事件数据（整体作为工具的 arguments 传进去）

    返回：工具返回的 dict（不是 dict 就包成 {"result": ...}），或 None
    （没配 server/tool 名 / 调用失败，fail-open）。
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
        return _fail_closed_or_none(
            hook, f"mcp_tool hook {hook.name} 调用失败: {e}")


# ============================================================================
# prompt 类型：让辅助小模型单轮评估
# ============================================================================


# http hook 响应体上限（1MB——hook 协议只需要很小的 JSON 决策）
_MAX_HTTP_HOOK_BODY_BYTES = 1_000_000


class _SafeFormatDict(dict):
    """给 format_map 用的"宽容" dict：模板里引用了 payload 没有的字段时，
    保留 {key} 原样（不抛 KeyError），渲染继续。"""

    def __missing__(self, key):
        return "{" + key + "}"


def run_prompt_hook(hook, payload: dict) -> Optional[dict]:
    """让辅助小模型（aux_llm_router）单轮评估这个事件，返回 JSON 判决——有些判决逻辑写不成死规则，让小模型看一眼更灵活。

    hook 配置里的 prompt 是模板字符串，用 payload 的字段填空。

    参数：
    - hook：要执行的 prompt 类型 hook
    - payload：事件数据（填模板 + 拼进最终提示词）

    返回：小模型回答里解析出的 JSON dict，或 None（router 没注入 /
    调用失败 / 解析不出 JSON——一律 fail-open，不打断主流程）。
    """
    router = _get_aux_router()
    if router is None:
        logger.warning(
            "prompt hook %s 跳过：aux_llm router 未注入"
            "（cli.py 未调 set_aux_router_provider）",
            hook.name,
        )
        return None

    # 必须用 format_map + _SafeFormatDict 这种"安全填充"——
    # payload 缺字段时保留 {field} 原样继续渲染（裸 .format(**payload)
    # 缺 key 会直接 KeyError 抛穿，整个 hook 挂掉）
    prompt_text = (
        (hook.script.prompt or "").format_map(_SafeFormatDict(payload))
        if hook.script.prompt else ""
    )
    full_prompt = (
        f"{prompt_text}\n\n"
        f"payload: {json.dumps(payload, ensure_ascii=False)}\n"
        '返回 JSON：{"permissionDecision": "allow"|"deny", "reason": "..."}'
    )

    # chat_completions 是 async 的（aux_llm 公开契约：跨线程同步等结果一律
    # 走 loop_host.run_async）——本函数是同步函数，在钩子的 worker 线程里跑
    # （pre_tool_use 经 model_tools 的 asyncio.to_thread 进来），直接调
    # coroutine 只会拿到协程对象（判空/类型检查全空转，hook 永远不生效）。
    # 防死锁保险：万一在宿主循环线程里被同步调用（自己等自己），fail-open
    # 放行并大声告警——宁可跳过评估也不能把事件循环冻死。
    try:
        import asyncio as _aio
        _aio.get_running_loop()
        logger.warning(
            "prompt hook %s 跳过：在事件循环线程内无法同步等待 LLM（防死锁）",
            hook.name,
        )
        return None
    except RuntimeError:
        pass  # 当前线程没有在跑的事件循环——可以安全阻塞等

    try:
        from agent.loop_host import loop_host
        resp = loop_host.run_async(
            router.chat_completions(
                messages=[{"role": "user", "content": full_prompt}],
            ),
            # hook 评估可能从后台线程发起、跨回合边界才返回——豁免回合栅栏
            #（同 reflection 的取法），防被回合收尾当遗留误杀
            exempt_from_fence=True,
        )
    except Exception as e:
        logger.warning("prompt hook %s LLM 调用失败: %s", hook.name, e)
        return _fail_closed_or_none(
            hook, f"prompt hook {hook.name} LLM 评估失败: {e}")

    if not resp:
        return None

    # 响应格式不保证统一：可能是纯字符串 / OpenAI 风格的 dict / 响应对象
    text = ""
    if isinstance(resp, str):
        text = resp
    elif isinstance(resp, dict):
        # OpenAI 风格结构：{choices: [{message: {content: "..."}}]}，取正文
        try:
            text = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = json.dumps(resp, ensure_ascii=False)
    else:
        # 响应对象（正常返回形态）：choices[0].message.content
        try:
            text = resp.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            text = ""

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
# agent 类型：让子代理多轮评估（复用 delegate 机制）
# ============================================================================


def run_agent_hook(hook, payload: dict) -> Optional[dict]:
    """派一个子代理去多轮评估这个事件，解析它回答里的 JSON 当判决——单轮小模型不够用的复杂判断，交给能多轮思考（还能查资料）的子代理（复用 delegate_tool 的 _run_child）。

    参数：
    - hook：要执行的 agent 类型 hook
    - payload：事件数据（填进 prompt 模板 + 拼进最终目标）

    hook 配置里的 prompt 是目标模板（.format(**payload) 填空）；
    agent_name 透传给 _run_child（自定义子代理名，可不填）。
    返回：子代理输出里解析出的 dict；解析不出 JSON 时返回
    {"decision": "review", "raw": 前 500 字}（宁可让人复审，别当放行）。

    已知约束（设计取舍）：hook_exec 是模块级函数、没有父 agent 上下文，
    所以调 _run_child 时不传 config/agent_ref。后果：
    - config：_run_child 内部会自己 load_config()（兜底路径），能正常跑。
    - agent_ref：没有中断传播、没有 pendingToolUseSummary、没有 checkpoint
      追踪。agent hook 的定位就是"一次性事件评估"（不是长任务），影响可控。
      以后要补全的话，可以照 _AUX_ROUTER_PROVIDER 的模式加个
      _AGENT_REF_PROVIDER 注入点。
    """
    # 懒加载，避免和 delegate_tool 循环 import
    from tools.delegate_tool import _run_child

    base_prompt = hook.script.prompt or "判断以下事件是否允许，返回 allow/deny"
    # 安全填充：payload 缺字段就保留模板原样，不让整个 hook 挂掉
    try:
        goal = base_prompt.format(**payload)
    except (KeyError, IndexError):
        goal = base_prompt

    full_goal = goal + f"\n\npayload: {json.dumps(payload, ensure_ascii=False)}"

    # _run_child 的签名是 (goal, context, role, **kwargs)：agent_name 走 kwargs 透传
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
        return _fail_closed_or_none(
            hook, f"agent hook {hook.name} 子代理评估失败: {e}")

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
