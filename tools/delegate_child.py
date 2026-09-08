"""子代理执行心脏：_run_child 从 delegate_tool 拆出（纯搬迁）。

_run_child 是同步/异步/批量三条委托路最终汇聚的「把子代理真正跑起来」的函数：
造 AIAgent 实例、配工具集/模型/权限、fork 前缀继承、隔离 worktree、内联 MCP
临时连接、轨迹落盘、asyncio.run 驱动对话、结果后处理，最后 finally 里八步
清理链收尾。整个委托链里最厚的一块，单独一个文件好找好维护。

搬迁铁律（委托链拆分约定）：
  - 行为零变化——函数体与 delegate_tool 原文逐字节一致；
  - asyncio.run(child.chat(...))（R5 定案）一字不动——子代理在独立线程里
    跑（线程里没有事件循环），用 asyncio.run 驱动 AIAgent.chat 这个协程；
  - finally 八步清理链顺序零变化：cleanup_runtime → 内联 MCP 断开 →
    failed 标记 → _children 划名 → workspace token reset → worktree
    智能清理 → subagent_stop 钩子 → client 关闭；
  - _workspace_cwd 的 contextvars token 手动管理原样（不重新缩进 try 块）；
  - 函数内 26 处延迟 import（agent.AIAgent / subagent_persistence /
    agent_defs / fork_messages / mcp_client / progress 等）原样保留，
    不提升到模块顶层；
  - 唯一改写（兄弟跨调允许项）：对 delegate_result（后处理五件）与
    delegate_setup（两件配置函数）的调用改为模块级 import 后同名直调——
    调用点文本一字未动，名字从 delegate_tool 的 re-export 全局名换成
    本模块的 import（两者本就指向同一函数对象，无环）；
  - 模块顶层不调 registry.register()——注册仍由 delegate_tool 完成；
    主文件 `from tools.delegate_child import _run_child` re-export 保住
    三条链调用点、外部延迟 import（hook_exec / workflow_engine /
    plan_mode_tool）与 verify 的 patch("tools.delegate_tool._run_child")
    全部照旧命中主文件全局名。
"""

import asyncio
import concurrent.futures
import logging
import os
import time
from pathlib import Path

from tools.delegate_result import (
    _review_handoff,
    _offload_child_result,
    _attach_full_result_pointer,
    _summarize_child_result,
    _build_child_system_prompt,
)
from tools.delegate_setup import (
    inline_mcp_spawn_allowed,
    _validate_toolset_names,
)

logger = logging.getLogger(__name__)


def _run_child(
    goal: str,
    context: str,
    role: str,
    **kwargs,
) -> str:
    """创建并跑起一个子代理——同步/异步/批量三条路最终都汇聚到这个函数。

    子代理是一个全新的 AIAgent 实例，跟主对话各过各的，自带：
    - 会话 ID
    - 迭代预算（最多循环多少轮，默认 50）
    - 工具集（leaf 角色受限）
    - 工作目录（可隔离成独立 worktree）

    铁律：子代理不继承父代理的对话历史（只通过 goal/context 拿必要信息）。

    其它职责：
    - 把子代理登记到父代理的 _children 名单，支持中断往下传（batch1-T4）；
    - 把子代理的执行轨迹（transcript）落盘，供事后查证/恢复（出错不挡主流程）。

    参数：
      - goal：任务描述
      - context：补充背景
      - role：leaf / orchestrator
      - **kwargs：运行时上下文（config、agent_ref、cancel_event、fork、
        subagent_type、isolated_workspace、permission_mode 等）

    返回：子代理产出的结果文本（可能已被摘要压缩）。
    """
    # 延迟导入，避免和 agent 包互相 import 死锁
    from agent import AIAgent

    # === 子代理轨迹落盘初始化（出错只 debug 记录，不影响主流程）===
    _persistence_enabled = (kwargs.get("config") or {}).get(
        "delegation", {},
    ).get("subagent_persistence_enabled", True)
    _child_agent_id = None
    if _persistence_enabled:
        try:
            from agent.subagent_persistence import (
                generate_agent_id, write_metadata as _sp_write_meta,
            )
            # 调用方预先生成的 id 优先用（_delegate_sync 强制扔下子代理时要靠它标状态）
            _child_agent_id = kwargs.pop("subagent_agent_id", None) \
                or generate_agent_id(
                    parent_session_id=kwargs.get("session_id", ""),
                )
            _sp_write_meta(_child_agent_id, {
                "agent_type": kwargs.get("subagent_type", "general-purpose"),
                "parent_session_id": kwargs.get("session_id", ""),
                "description": f"{goal[:100]}",
                "status": "running",
                "created_at": time.time(),
            })
            # === 用户指令先进轨迹文件（作为轨迹开头）===
            # 真被中断的子代理之后 resume 时，原始指令就是对话的起点。
            from agent.subagent_persistence import append_message as _sp_append
            _sp_append(_child_agent_id, {
                "role": "user",
                "content": f"{goal}\n上下文: {context}" if context else goal,
                "_ts": time.time(),
            })
            logger.debug("Task I: 子代理 transcript 持久化启用: %s", _child_agent_id)
        except Exception as e:
            logger.debug("Task I: transcript 持久化初始化失败（fail-open）: %s", e)
            _child_agent_id = None

    # LLM 连接配置：优先用 kwargs 里带的，缺了再从 config 补
    base_url = kwargs.get("base_url")
    api_key = kwargs.get("api_key")
    auth_token = kwargs.get("auth_token")
    model = kwargs.get("model")
    model_format = kwargs.get("model_format")

    if not (api_key or auth_token) or not model:
        # 缺关键配置 → 从 config 补齐
        try:
            from config import load_config
            config = load_config()

            # 子代理优先用轻量便宜的小模型（default_haiku_model），省 token——
            # 相当于业界的「主对话用旗舰、跑腿的用小模型」模式
            haiku_name = config.get("default_haiku_model", "")
            # 新配置形态：config["haiku_model"]（llm 段注入的）
            haiku_cfg = config.get("haiku_model")
            if haiku_cfg:
                sub_cfg = haiku_cfg
            elif haiku_name and haiku_name in config.get("models", {}):
                # 旧式配置（仍兼容）：config["models"][小模型名]
                sub_cfg = config["models"][haiku_name]
            else:
                # 都没有 → 退回主模型配置段
                sub_cfg = config.get("model", {})

            if not base_url:
                base_url = sub_cfg.get("base_url")
            if not model:
                model = sub_cfg.get("model") or sub_cfg.get("name")
            if not api_key:
                api_key = sub_cfg.get("api_key") or ""
            if not auth_token:
                auth_token = sub_cfg.get("auth_token") or ""
            # 向后兼容：旧式 config.yaml 用 api_key_env 写环境变量名，去环境里取真值
            if not api_key and not auth_token:
                api_key_env = sub_cfg.get("api_key_env") or ""
                if api_key_env:
                    api_key = os.environ.get(api_key_env) or ""
            if not model_format:
                model_format = sub_cfg.get("format", "anthropic")
        except Exception as e:
            raise RuntimeError(f"子代理无法获取 LLM 配置: {e}")

    if not api_key and not auth_token:
        raise RuntimeError(
            "子代理无法获取 API key（settings.json 的 models 配置为空，"
            "且未设置环境变量）"
        )

    # 构造子代理的 system prompt（开场设定词）
    system_prompt = _build_child_system_prompt(goal, context, role)

    # 计算子代理的派生深度（第几层套娃）。线程安全的做法：靠参数传递，
    # 不写 os.environ（环境变量是进程级全局，并发线程会互相覆盖）
    parent_agent = kwargs.get("agent_ref")
    if parent_agent is not None and hasattr(parent_agent, "spawn_depth"):
        parent_depth = parent_agent.spawn_depth
    else:
        parent_depth = int(kwargs.get("spawn_depth", 0))
    child_spawn_depth = parent_depth + 1

    # 工具集选择
    # 提前解析类型 + 自定义定义：为了让 isolation=worktree 能赶在创建 worktree
    # 的分支之前生效（isolated 要在自定义定义写入 kwargs 之后再读，
    # 读早了 worktree 永远创建不出来）
    stype = kwargs.get("subagent_type", "general-purpose")
    custom_def = None
    if stype not in ("general-purpose", "custom"):
        # 传的是自定义子代理名：从 .md 定义文件加载
        from agent.agent_defs import get_agent_def
        custom_def = get_agent_def(stype)
        if custom_def is None:
            raise RuntimeError(
                f"未找到子代理定义: {stype}"
                f"（检查 ~/.codeAgent/agents/ 和 ./.codeAgent/agents/）"
            )

    # 可选：隔离工作区（自定义 .md 定义 isolation=worktree 也会开启）
    # 千万别用 os.chdir 切目录——那是进程级全局操作，线程池里
    # 并发跑的多个子代理会互相踩对方的工作目录。所以用
    # workspace_cwd_context（contextvars.ContextVar，线程各一份互不干扰），
    # 子代理内的工具调 get_workspace_cwd() 拿到的就是自己的 worktree。
    isolated = kwargs.get("isolated_workspace", False) or (
        custom_def is not None and custom_def.isolation == "worktree")
    workspace_cleanup = None
    workspace_path = None
    if isolated:
        try:
            from tools.worktree import create_isolated_workspace
            workspace_path, workspace_cleanup = create_isolated_workspace(
                name=f"delegate-{goal[:20].replace(' ', '-')}",
            )
            logger.info("子代理在隔离工作区运行: %s", workspace_path)
        except Exception as e:
            logger.warning("创建隔离工作区失败，用当前目录: %s", e)

    # 工作目录上下文手动管理 token（不为这个把 200 行 try/finally 重新缩进一层）。
    # 用 contextvars 替代 os.chdir：并发子代理（线程池里）每线程各有一份，
    # 不会互相踩工作目录。token 在 finally 末尾 reset（出 try 块 = 出子代理的上下文）。
    from agent.workspace_context import _workspace_cwd
    _workspace_cwd_token = None
    if workspace_path is not None:
        _workspace_cwd_token = _workspace_cwd.set(str(workspace_path))

    # 拿父代理引用（用于中断传播）
    parent_agent = kwargs.get("agent_ref")

    child = None
    # 成功标志（SUBAGENT_START/STOP 钩子要用；默认 False，正常走到 try 末尾才设 True）
    _fork_success = False
    # 内联 MCP 临时连接名单必须在这就初始化（而不是 try 体内）：try 里任何
    # 早退 raise（深度守卫/套餐名校验等）都会走 finally 清理，finally 读到
    # 未赋值变量会 UnboundLocalError，把真正的报错盖掉
    _inline_mcp_connected: list = []
    # 父代理的钩子注册表（用于 SUBAGENT_START/STOP 审计事件）
    _parent_hooks = getattr(parent_agent, "hooks_registry", None) if parent_agent else None
    # === 工作目录变化钩子：切进 worktree 时通知一声（出错全吞，不挡主流程）===
    if workspace_path is not None and _parent_hooks is not None:
        try:
            _parent_hooks.run_cwd_changed({
                "session_id": kwargs.get("session_id", ""),
                "old": str(Path.cwd()),  # 进程当前目录（近似值）
                "new": str(workspace_path),
            })
        except Exception:
            pass  # fail-open：钩子失败不影响子代理
    try:
        # 工具集选择：
        # - 自定义名：custom_def 已提前加载，按定义配置工具集/模型/权限/轮数上限
        # - custom：用显式传的 enabled_toolsets
        # - general-purpose：按角色给默认
        # permission_mode（权限模式）优先级：
        #   ① 自定义 .md 里显式写的 permission_mode（最高）
        #   ② kwargs 里的（_delegate_async 注入的 autoDeny，或调用方显式传的）
        #   ③ "default"（兜底）
        _injected_perm_mode = kwargs.get("permission_mode")
        if custom_def:
            # 按自定义定义配置
            child_toolsets = custom_def.tools or (
                ["core"] if role == "orchestrator" else ["minimal"])
            # 自定义 .md 的 disallowedTools 直接覆盖父 config（取替不是合并）
            disabled = custom_def.disallowed_tools or None
            child_model = custom_def.model or model
            # 自定义 .md 显式指定 > kwargs 注入（后台模式的 autoDeny）
            child_perm_mode = custom_def.permission_mode or _injected_perm_mode or "default"
            child_max_iter = custom_def.max_turns or kwargs.get("child_max_iterations", 50)
        elif stype == "custom":
            child_toolsets = kwargs.get("enabled_toolsets") or (
                ["core"] if role == "orchestrator" else ["minimal"])
            disabled = None
            child_model = model
            child_perm_mode = _injected_perm_mode or "default"
            child_max_iter = kwargs.get("child_max_iterations", 50)
        elif role == "leaf":
            child_toolsets = kwargs.get("enabled_toolsets") or ["minimal"]
            disabled = None
            child_model = model
            child_perm_mode = _injected_perm_mode or "default"
            child_max_iter = kwargs.get("child_max_iterations", 50)
        else:  # orchestrator
            child_toolsets = kwargs.get("enabled_toolsets") or ["core"]
            disabled = None
            child_model = model
            child_perm_mode = _injected_perm_mode or "default"
            child_max_iter = kwargs.get("child_max_iterations", 50)

        # fail-loud：四条赋值路径汇合后、真正用之前验一遍套餐名——
        # 未知名静默给空工具比报错危险得多（silent dead agent）。
        # raise 会被 _delegate_sync/_delegate_async 的调用方 try/except
        # 捕获，作为委托错误结果返回给 LLM/用户，不会崩主循环。
        _ts_err = _validate_toolset_names(child_toolsets)
        if _ts_err:
            raise ValueError(_ts_err)

        # === 套娃深度守卫：按"可见工具含 subagent"判定，不按角色名 ===
        # 为什么看工具不看 role：内置 coordinator / 自定义 .md 派生时 role 默认
        # "leaf"，但套餐里带 subagent，照样能继续往下派——只认
        # role=="orchestrator" 会让 coordinator→coordinator→… 链绕过深度封顶
        # （审查发现的盲区）。所以守卫从 _handle_delegate_task 入口挪到这里，
        # 在工具集成型之后统一判定：解析出的可见工具里含能派生的工具
        # （subagent 一族），且父深度已达上限 → 拒绝（报错语义与旧守卫一致）。
        # 宁可多拦：async 路径的 gate 2 之后会剥掉 subagent，但判定发生在剥除
        # 之前——真到深度上限说明上面层级确实派下来了，多拦不破坏功能；
        # "mcp" 套餐解析成空表，天然不受影响。
        _max_depth = int(kwargs.get("max_spawn_depth", 2))
        from toolsets import resolve_toolset
        _spawn_family = {"subagent", "subagent_kill", "subagent_resume"}
        if parent_depth >= _max_depth and any(
            _spawn_family.intersection(resolve_toolset(ts)) for ts in child_toolsets
        ):
            raise ValueError(f"已达最大嵌套深度 {_max_depth}")

        # 自定义子代理的 system_prompt 覆盖（用定义里的重写一份）
        if custom_def and custom_def.system_prompt:
            system_prompt = _build_child_system_prompt(
                goal, context, role, override=custom_def.system_prompt)

        # === 关键提醒（critical_reminder）拼到 system_prompt 末尾 ===
        # 拼在这里对缓存友好（system_prompt 只构建一次就走缓存，不必每轮重复注入）
        if custom_def and custom_def.critical_reminder:
            system_prompt += (
                f"\n\n## CRITICAL REMINDER\n{custom_def.critical_reminder}"
            )
            logger.debug(
                "Task N: critical_reminder 已拼到子代理 system_prompt (%d 字符)",
                len(custom_def.critical_reminder),
            )

        # === memory 字段：给子代理开独立记忆目录 ===
        # 默认继承父代理的记忆库（没传则子代理自己新建默认的）
        child_memory_store = kwargs.get("memory_store")
        if custom_def and custom_def.memory:
            from constants import get_codeagent_home
            agent_memory_home = get_codeagent_home() / ".agent-memory" / custom_def.name
            try:
                agent_memory_home.mkdir(parents=True, exist_ok=True)
                from agent.memory_store import MemoryStore
                child_memory_store = MemoryStore(codeagent_home=agent_memory_home)
                logger.info(
                    "子代理 %s 使用独立记忆目录: %s",
                    custom_def.name, agent_memory_home,
                )
            except Exception as e:
                logger.warning("创建子代理独立记忆目录失败（用父 store）: %s", e)

        # === skills 字段：把技能正文预装进 system_prompt ===
        # 注意时序：必须在 system_prompt 定稿之后、AIAgent 构造之前
        if custom_def and custom_def.skills:
            try:
                from agent.skill_commands import parse_frontmatter
                from constants import all_skills_dirs
                for skill_name in custom_def.skills:
                    found = False
                    for d in all_skills_dirs():
                        p = Path(d) / skill_name / "SKILL.md"
                        if p.exists():
                            _, body = parse_frontmatter(p.read_text(encoding="utf-8"))
                            system_prompt += (
                                f"\n\n## 预装技能：{skill_name}\n{body.strip()}\n"
                            )
                            found = True
                            break
                    if not found:
                        logger.warning("子代理预装技能未找到: %s", skill_name)
            except Exception as e:
                logger.warning("子代理预装技能失败（继续）: %s", e)

        # disabled_tools 传递方式：AIAgent.__init__ 没有这个参数，只能走 config 转交
        # （get_tool_definitions 运行时会从 self.config 读 disabled_tools）
        #
        # 两个来源的禁用清单要合并（取并集、保序、去重）：
        #   ① 自定义 .md 的 disallowed_tools（custom_def 路径，上面赋给了 `disabled`）
        #   ② _delegate_async 注入到 kwargs["config"]["disabled_tools"] 的后台黑名单兜底
        #      —— 非 custom_def 路径会把它弄丢，靠这里补上
        _injected_disabled = (
            (kwargs.get("config") or {}).get("disabled_tools")
            if isinstance(kwargs.get("config"), dict)
            else None
        )
        _all_disabled = []
        for _src in (disabled, _injected_disabled):
            if _src:
                for _tool in _src:
                    if _tool not in _all_disabled:
                        _all_disabled.append(_tool)

        child_config = None
        if _all_disabled:
            # 继承父 config（如果有的话），再补上 disabled_tools
            parent_cfg = kwargs.get("config")
            child_config = dict(parent_cfg) if isinstance(parent_cfg, dict) else {}
            child_config["disabled_tools"] = _all_disabled

        # === mcp_servers 字段：子代理只暴露定义里列出的 MCP 服务器 ===
        # 时序：必须在 child_config 构造之后、AIAgent 构造之前
        if custom_def and custom_def.mcp_servers:
            if child_config is None:
                parent_cfg = kwargs.get("config")
                child_config = dict(parent_cfg) if isinstance(parent_cfg, dict) else {}
            child_config["mcp_server_filter"] = custom_def.mcp_servers

        # === 内联 mcpServers：派生时临时连接，结束时断开（不留全局残留）===
        # （不进全局 .mcp.json 注册；连接进共享的 MCPManager，工具以
        #   mcp__<服务器名>__<工具名> 前缀动态注册；finally 里断开。
        #   名单变量在 try 之前已初始化，这里只往里追加）
        if custom_def and getattr(custom_def, "inline_mcp_servers", None):
            try:
                from agent.mcp_client import get_mcp_manager
                from tools.mcp_tool import register_mcp_tools
                _mgr = get_mcp_manager()
                _app_cfg = kwargs.get("config")
                for _sname, _scfg in custom_def.inline_mcp_servers.items():
                    # 项目来源的内联 MCP 必须已获首连审批，否则跳过不连（宁可不放行）
                    if not inline_mcp_spawn_allowed(custom_def, _sname, _scfg):
                        logger.warning(
                            "内联 MCP server %s（agent %s，项目来源）未获首连审批，"
                            "跳过连接（启动时会请求审批，或手动在 settings.json "
                            "mcp.approved_project_servers 加 key）",
                            _sname, custom_def.name,
                        )
                        continue
                    _mgr.connect_one(_sname, _scfg, app_config=_app_cfg)
                    _inline_mcp_connected.append(_sname)
                if _inline_mcp_connected:
                    register_mcp_tools(_mgr, servers=_inline_mcp_connected)
                    # 让子代理能看到这些服务器的工具（跟 mcp_servers 过滤合并）
                    _filter = child_config.get("mcp_server_filter") if child_config else None
                    if child_config is None:
                        _pcfg = kwargs.get("config")
                        child_config = dict(_pcfg) if isinstance(_pcfg, dict) else {}
                    child_config["mcp_server_filter"] = (
                        (list(_filter) if _filter else []) + _inline_mcp_connected
                    )
            except Exception as e:
                logger.warning("内联 mcpServers 连接失败（fail-open 继续无 MCP）: %s", e)

        # === fork 子代理路径（前缀跟父代理一字不差，蹭 prompt cache 省 token）===
        # fork=True 时：子代理继承父代理的 system prompt 原字节 + 父对话前缀
        # （最近 N 轮 assistant 发言），构造出「缓存等价」的前缀，prompt cache
        # 一命中 token 省 50% 以上。构造失败就退回普通非 fork 路径（只打 warning）
        fork_mode = kwargs.get("fork", False)
        child_initial_messages = None
        if fork_mode:
            # 读 config 开关（默认 True）
            _cfg = kwargs.get("config") or {}
            _delegation_cfg = _cfg.get("delegation") if isinstance(_cfg, dict) else {}
            _fork_enabled = (_delegation_cfg or {}).get("fork_subagent_enabled", True)
            _max_turns = int((_delegation_cfg or {}).get("fork_max_parent_turns", 3))

            if _fork_enabled and parent_agent is not None:
                try:
                    from agent.fork_messages import (
                        build_forked_messages,
                        build_forked_system_prompt,
                    )
                    parent_messages = parent_agent.conversation_history or []
                    parent_sysprompt = parent_agent._get_system_prompt() or ""
                    # 把 system_prompt 换成 fork 版（父字节 + fork 标记）
                    system_prompt = build_forked_system_prompt(
                        parent_sysprompt, child_role=role,
                    )
                    # fork 覆盖了 system_prompt，得再补一次关键提醒
                    # （critical_reminder 常是安全提醒，fork 路径不能丢）
                    if custom_def and custom_def.critical_reminder:
                        system_prompt += (
                            f"\n\n## CRITICAL REMINDER\n{custom_def.critical_reminder}"
                        )
                    # 构造初始消息（父前缀 + 给子代理的指令）
                    # fork="full" 走全量模式（完整 user/assistant 对话流，
                    # 截到 delegation.fork_full_history_max_turns 上限）
                    _fork_full = fork_mode == "full"
                    _full_max = int(
                        (_delegation_cfg or {}).get("fork_full_history_max_turns", 50)
                    )
                    child_initial_messages = build_forked_messages(
                        parent_messages=parent_messages,
                        parent_system_prompt=parent_sysprompt,
                        child_directive=f"{goal}\n上下文: {context}" if context else goal,
                        max_parent_turns=_max_turns,
                        full_history=_fork_full,
                        full_history_max_turns=_full_max,
                    )
                    logger.info(
                        "Task H: fork 子代理启用，继承 %d 条 messages",
                        len(child_initial_messages),
                    )
                except Exception as e:
                    # 出错兜底：fork 构造失败，退回普通非 fork 路径
                    logger.warning(
                        "Task H: fork 构造失败，fallback 到非 fork 路径: %s", e,
                    )
                    child_initial_messages = None
                    # system_prompt 本该保留前面 _build_child_system_prompt 的结果，
                    # 但如果上面已被 fork 版覆盖到一半又出错，得整个重建一遍
                    system_prompt = _build_child_system_prompt(goal, context, role)
                    if custom_def and custom_def.system_prompt:
                        system_prompt = _build_child_system_prompt(
                            goal, context, role, override=custom_def.system_prompt)
                    # 兜底路径同样要补关键提醒（别在 fallback 里丢了安全提醒）
                    if custom_def and custom_def.critical_reminder:
                        system_prompt += (
                            f"\n\n## CRITICAL REMINDER\n{custom_def.critical_reminder}"
                        )

        # === 每轮把轨迹落盘（靠 POST_LLM_CALL 程序式钩子实现）===
        # 为什么自己建一套：子代理的钩子注册表不跟主代理共享（_run_child
        # 不传，子代理拿到的是 None）→ 新建一个空的独立 HookRegistry 注册
        # 程序式钩子，完全不碰主代理的注册表。
        # 落什么：轨迹 = 用户指令 + 每轮 assistant 正文；tool_calls 和工具结果
        # 不落盘（POST_LLM_CALL 只拿得到 LLM 响应；存了带 tool_calls 但没有
        # 配对结果的消息会造出「孤儿消息」→ API 直接报 400）。这样 resume 时
        # 初始消息就是纯 user/assistant 文本流，配对天然完整。
        # 已知耦合：走的是 AIAgent._run_post_llm_call_hook，受总开关
        # config["hooks"]["enabled"] 门控（默认开）；用户显式关掉 hooks 的话
        # 轮级记录会停（只剩 user 指令那一条）。
        _child_hooks = None
        if _child_agent_id:
            try:
                from agent.hooks import HookRegistry

                def _extract_turn_text(response):
                    """从 LLM 响应里抽出 assistant 的正文文本（None 安全，两种格式都兼容）。

                    参数：
                      - response：LLM 响应对象
                    返回：正文文本；抽不出来就返回 None。
                    """
                    try:
                        msg = response.choices[0].message
                    except Exception:
                        return None
                    content = getattr(msg, "content", None)
                    if isinstance(content, list):
                        # Anthropic 风格的 content 是块列表 → 只把 text 块拼起来
                        parts = [
                            b.get("text", "") for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        content = "\n".join(p for p in parts if p)
                    return content or None

                def _on_llm_turn(response):
                    """每轮 LLM 响应完，把 assistant 正文追加进轨迹文件（出错全吞不挡路）。

                    参数：
                      - response：LLM 响应对象
                    返回：None（不改动响应本身）。
                    """
                    try:
                        text = _extract_turn_text(response)
                        if text:
                            from agent.subagent_persistence import append_message
                            append_message(_child_agent_id, {
                                "role": "assistant",
                                "content": text,
                                "_ts": time.time(),
                            })
                    except Exception:
                        pass  # fail-open：落盘失败不影响子代理
                    return None  # 不修改 response

                _child_hooks = HookRegistry()
                _child_hooks.register_post_llm_call(_on_llm_turn)
            except Exception as e:
                logger.debug(
                    "Task 3: 轮级 transcript hook 注册失败（fail-open）: %s", e)
                _child_hooks = None

        child = AIAgent(
            base_url=base_url,
            api_key=api_key or None,
            auth_token=auth_token or None,
            model=child_model,
            model_format=model_format or "anthropic",
            max_iterations=child_max_iter,
            enabled_toolsets=child_toolsets,
            system_prompt_override=system_prompt,
            spawn_depth=child_spawn_depth,
            permission_mode=child_perm_mode,
            effort_level=(custom_def.effort if custom_def else None) or getattr(parent_agent, "effort_level", None),
            config=child_config,
            memory_store=child_memory_store,
            initial_messages=child_initial_messages,
            # 轮级轨迹持久化：独立空 registry + POST_LLM_CALL
            # 程序式钩子每轮追加；最终响应的 append 已删掉，避免同一内容写两遍
            hooks_registry=_child_hooks,
            omit_project_memory=bool(custom_def.omit_claude_md) if custom_def else False,
        )

        # 登记到父代理的 _children 名单（中断时能顺着名单传下去）
        if parent_agent is not None:
            try:
                parent_agent._children.append(child)
            except Exception:
                pass

        # === UI 直播：子代理的工具活动上报给 live 面板（纯展示，fail-open）===
        # 大白话：子代理在下面干活，屏幕上的树（├─ 描述 · N tool uses）
        # 靠这里每次工具调用敲一笔。只注册钩子不改执行逻辑，任何异常
        # 都吞——展示线断了也不许断任务线。
        _ui_child_key = kwargs.get("ui_child_key")
        if _ui_child_key:
            try:
                import cli_live

                def _ui_on_pre(tool_name, args, **_kw):
                    try:
                        from cli_events import summarize_args
                        cli_live.note_child_tool(
                            _ui_child_key,
                            f"{tool_name}({summarize_args(tool_name, args or {})})",
                        )
                    except Exception:
                        pass
                    return None   # 不拦不改变量——纯旁观

                child.hooks_registry.register_pre_tool_use(
                    _ui_on_pre, name="cli_live_child")
            except Exception as e:
                logger.debug("live 面板钩子注册失败（fail-open）: %s", e)

        # === 长任务进行中的进度播报（P1-10）===
        # 用辅助小模型周期性生成「正在做什么」的摘要，推给父代理的
        # stream_callback，让前端不至于干等。辅助模型不可用时降级成心跳。
        from agent.progress import ProgressReporter
        progress_stream = getattr(parent_agent, "_stream_callback", None) if parent_agent else None
        progress_aux = getattr(parent_agent, "aux_llm_router", None) if parent_agent else None
        progress_interval = (kwargs.get("config") or {}).get(
            "delegation", {},
        ).get("progress_interval", 30.0)

        with ProgressReporter(
            goal=goal,
            stream_callback=progress_stream,
            aux_llm_router=progress_aux,
            interval=progress_interval,
        ):
            # 子代理开始事件（在 child.chat 之前触发；出错全吞）
            if _parent_hooks is not None:
                try:
                    _parent_hooks.run_subagent_start({
                        "session_id": kwargs.get("session_id", ""),
                        "subagent": stype,
                        "goal": goal,
                        "spawn_depth": child_spawn_depth,
                    })
                except Exception:
                    pass  # fail-open

            # 正式跑子代理
            # AIAgent.chat 是 async。_run_child 在
            # 独立线程里跑（同步/异步两条路都起的 threading.Thread），线程里
            # 没有事件循环 → 用 asyncio.run 驱动它。
            # cancel_event 也传给子代理的对话主循环：它每轮开头检查信号，
            # 一旦被按下就退出并返回已完成的部分结果
            import asyncio
            _cancel_event = kwargs.get("cancel_event")
            # === initial_prompt 前置到第一条 user 消息 ===
            # 类似斜杠命令的预处理
            _child_first_msg = f"请执行任务: {goal}"
            if custom_def and custom_def.initial_prompt:
                _child_first_msg = (
                    f"{custom_def.initial_prompt}\n\n{_child_first_msg}"
                )
                logger.debug(
                    "Task N: initial_prompt 前置到子代理首 user (%d 字符)",
                    len(custom_def.initial_prompt),
                )
            if _cancel_event is not None:
                result = asyncio.run(
                    child.chat(_child_first_msg, cancel_event=_cancel_event)
                )
            else:
                result = asyncio.run(child.chat(_child_first_msg))

        # 幻觉检测（赶在摘要压缩之前做，这样警告能保留进摘要）
        # 注意：Path.cwd() 是进程级的（就是 os.getcwd），
        # 并发子代理会互相踩。优先用 kwargs 里的 cwd，没有就读线程局部的
        # get_workspace_cwd()。
        try:
            from agent.team.hallucination_check import verify_claims, append_warning
            from agent.workspace_context import get_workspace_cwd
            verification = verify_claims(
                result,
                task_store=kwargs.get("task_store"),
                fs_cwd=kwargs.get("cwd") or Path(get_workspace_cwd()),
            )
            result = append_warning(result, verification)
        except Exception as e:
            logger.warning("幻觉检测失败（fail-open）: %s", e)

        # === 放权模式下的交接复审 ===
        # bypass/auto 权限模式下，子代理产出会直接进父代理上下文——危险产出
        # （破坏命令证据/数据外发/凭证修改痕迹）由辅助 LLM 复审一遍，命中就在
        # 结果前面附警告（注意不拦截——怎么处理由父代理和用户自己决断）。
        # 由 feature flag delegation.handoff_review_enabled 门控（默认关）；出错全吞。
        try:
            _hr_cfg = (kwargs.get("config") or {}).get("delegation") or {}
            if _hr_cfg.get("handoff_review_enabled", False):
                result = _review_handoff(result, parent_agent)
        except Exception as e:
            logger.warning("交接复审失败（fail-open）: %s", e)

        # summary_only：结果太长就用 LLM 压成摘要，省父代理的上下文空间
        # summary_len：摘要长度（默认 300；长任务/深度调研可调大，见 schema 说明）
        # 原文先落盘再摘要：摘要是有损压缩，落盘后父代理想看细节时
        # 有 full_at 指针可读回（旧版摘要后原文即蒸发）
        summary_only = kwargs.get("summary_only", True)
        if summary_only and len(result) > 500:
            offloaded_json = _offload_child_result(result, kwargs)
            try:
                summary_len = int(kwargs.get("summary_len", 300))
            except (TypeError, ValueError):
                summary_len = 300
            summary_len = max(100, min(2000, summary_len))
            result = _summarize_child_result(
                result, child.llm_client, child.model, max_chars=summary_len,
            )
            if offloaded_json:
                result = _attach_full_result_pointer(result, offloaded_json)

        # 走到这里说明成功了，把成功标志立起来（finally 里靠它触发结束事件）
        _fork_success = True

        # 把轨迹状态标成 completed
        if _child_agent_id:
            try:
                from agent.subagent_persistence import mark_completed
                mark_completed(_child_agent_id, "completed")
            except Exception:
                pass  # fail-open

        return result
    finally:
        # === 清杀子代理留下的运行状态 ===
        # 级联中断孙代理（免得后台线程往已死的父代理推结果）+ 停后台任务；
        # 可重复执行且出错不挡路，放在清理链最前（后面步骤不再依赖子代理活着）
        try:
            child.cleanup_runtime()
        except Exception:
            pass  # fail-open：这步失败不挡后面的清理

        # === 断开临时连的内联 MCP 服务器（别留全局残留连接）===
        if _inline_mcp_connected:
            try:
                from agent.mcp_client import get_mcp_manager
                _mgr = get_mcp_manager()
                for _sname in _inline_mcp_connected:
                    _mgr.disconnect_one(_sname)
            except Exception:
                pass  # fail-open：断不开也不挡清理链

        # === 出错时把轨迹标成 failed（_fork_success=False 说明没走到 return）===
        if _child_agent_id and not _fork_success:
            try:
                from agent.subagent_persistence import mark_completed
                mark_completed(_child_agent_id, "failed")
            except Exception:
                pass  # fail-open

        # 从父代理的 _children 名单里划掉自己
        if parent_agent is not None and child is not None:
            try:
                if child in parent_agent._children:
                    parent_agent._children.remove(child)
            except Exception:
                pass
        # 恢复工作目录上下文（替代 os.chdir 的回切）
        # ContextVar 的 token reset 只影响当前线程，踩不到别的并发子代理
        if _workspace_cwd_token is not None:
            try:
                _workspace_cwd.reset(_workspace_cwd_token)
            except Exception:
                pass
        if workspace_cleanup:
            # 智能清理 worktree：子代理有改动就保留现场，没改动才删
            # config.delegation.worktree_always_cleanup=True → 恢复旧行为（无脑总清理）
            _cfg = kwargs.get("config") or {}
            _delegation_cfg = _cfg.get("delegation") if isinstance(_cfg, dict) else {}
            _always_cleanup = (_delegation_cfg or {}).get("worktree_always_cleanup", False)
            try:
                if _always_cleanup:
                    workspace_cleanup(force=True)
                else:
                    cleaned = workspace_cleanup()
                    if cleaned is False and workspace_path is not None:
                        logger.warning(
                            "worktree 保留（子代理有改动）: %s", workspace_path,
                        )
            except Exception as e:
                logger.warning("worktree 智能清理异常（fail-open）: %s", e)

        # 子代理结束事件（成功失败都触发；出错全吞）
        if _parent_hooks is not None:
            try:
                _parent_hooks.run_subagent_stop({
                    "session_id": kwargs.get("session_id", ""),
                    "subagent": stype,
                    "goal": goal,
                    "success": _fork_success,
                })
            except Exception:
                pass  # fail-open

        # === 子代理 client 用后即关（放 finally 尾部）===
        # 这 client 是专为 child 新建的（一代理一池，AIAgent 构造时
        # create_llm_client 现造，不共享父代理的）——child.chat 虽仍跑
        # 在自己的 asyncio.run 里，但 httpx 连接池不随循环关闭自动回收，
        # 显式关才稳。
        # 放尾部是因为走到这时 _summarize_child_result 等真正用
        # child.llm_client 的步骤都已完成（它们全在 try 体的 return
        # 之前），这时关不碰任何人。fail-open：关不上只警告。
        if child is not None:
            try:
                from agent.llm_client import aclose_llm_client
                from agent.loop_host import loop_host
                loop_host.run_async(aclose_llm_client(child.llm_client))
            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                # 回合栅栏恰好落下时关闭协程被顺带取消——CancelledError 是
                # BaseException，except Exception 接不住会打穿本 finally，
                # 把子代理真正的结果/异常盖成取消错误。池留给进程退出
                # 收尾，安静放行
                pass
            except Exception as e:
                logger.warning("子代理 client 关闭失败（fail-open）: %s", e)
