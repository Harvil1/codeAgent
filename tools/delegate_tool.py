"""subagent 工具：派生子代理执行独立任务（对齐 Claude Code Agent）。

两种角色：
  - leaf（默认）：执行者，不能再委托
    有 terminal/read_file 等工具
    不能调用 subagent/clarify/memory/send_message（通过工具集隔离）

  - orchestrator：协调者，可以继续委托
    可以调用 subagent 派生自己的子代理
    受 max_spawn_depth 限制（默认 2 层）

三种模式：
  - 同步（默认）：等待子代理完成
  - 异步（background=True）：立即返回，结果通过队列送达
  - 批量（tasks=[...]）：并行执行多个子代理
"""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from tools.registry import registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 异步委托结果队列
# ---------------------------------------------------------------------------

class DelegationCompletionQueue:
    """异步委托完成结果的队列。

    后台子代理完成后，结果进入队列。
    父代理在每次循环开始时检查队列。
    """

    def __init__(self):
        self._queue: List[dict] = []
        self._lock = threading.Lock()

    def push(self, result: dict) -> None:
        """子代理完成时调用。"""
        with self._lock:
            self._queue.append(result)

    def drain(self) -> List[dict]:
        """父代理消费所有结果。"""
        with self._lock:
            results = list(self._queue)
            self._queue.clear()
            return results

    def has_pending(self) -> bool:
        with self._lock:
            return len(self._queue) > 0


# 全局单例（每个 agent 实例应该有自己的，这里简化）
_delegation_queue = DelegationCompletionQueue()


def get_delegation_queue() -> DelegationCompletionQueue:
    """获取全局委托完成队列。"""
    return _delegation_queue


# ---------------------------------------------------------------------------
# subagent 工具
# ---------------------------------------------------------------------------

DELEGATE_TASK_SCHEMA = {
    "name": "subagent",
    "description": (
        "派生子代理（subagent）执行独立任务（对齐 Claude Code Agent 工具）。"
        "子代理有独立的上下文和工具，只把结果带回主代理。\n\n"
        "主入口用 prompt 描述任务；subagent_type 选子代理类型。\n\n"
        "两种模式：\n"
        "- 同步（默认）：等待子代理完成后继续\n"
        "- 异步（background=True）：立即继续，结果稍后送达\n\n"
        "**并行规则（重要）**：需要同时派多个子代理（如并行探索多个模块）时，"
        "必须用 tasks=[...] 一次批量调用（内部真并行）。"
        "禁止发多个独立的 subagent 调用——独立调用是串行执行的，会逐个等待，浪费大量时间。\n\n"
        "角色：\n"
        "- leaf（默认）：执行者，不能再委托\n"
        "- orchestrator：可继续派生（受 max_spawn_depth 限制）"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "子代理任务（Claude Code Agent 主入口，必填之一）",
            },
            "subagent_type": {
                "type": "string",
                "default": "general-purpose",
                "description": (
                    "子代理类型：general-purpose=通用（minimal 工具集）；"
                    "custom=显式 enabled_toolsets；"
                    "或自定义子代理名（扫描 ~/.OmniMate/agents/*.md 和 ./.claude/agents/*.md 的 name 字段）。"
                    "自定义名时按定义的 model/tools/permissionMode/isolation/maxTurns 配置子代理。"
                ),
            },
            "goal": {
                "type": "string",
                "description": "子代理的目标（兼容旧字段，等同 prompt）",
            },
            "context": {
                "type": "string",
                "description": "给子代理的额外上下文",
            },
            "tasks": {
                "type": "array",
                "items": {"type": "object"},
                "description": "批量任务列表（并行执行）。要并行多个任务就用这个字段，不要多次调用 subagent",
            },
            "background": {
                "type": "boolean",
                "description": "是否后台运行（默认 false）",
                "default": False,
            },
            "role": {
                "type": "string",
                "enum": ["leaf", "orchestrator"],
                "default": "leaf",
            },
            "enabled_toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "子代理启用的工具集（默认继承父代理）",
            },
            "summary_only": {
                "type": "boolean",
                "description": "是否只返回摘要（默认 True）。超长结果用 LLM 压缩成 300 字摘要，节省父代理 context。",
                "default": True,
            },
            "isolated_workspace": {
                "type": "boolean",
                "description": "是否在隔离工作区执行（默认 False）。True 时创建独立 worktree/临时目录，避免文件冲突。",
                "default": False,
            },
        },
    },
}


def _handle_delegate_task(args: dict, **kwargs) -> str:
    """处理委托请求（subagent）。"""
    # prompt 是 Claude Code Agent 主入口，兼容 goal
    goal = args.get("goal") or args.get("prompt", "")
    tasks = args.get("tasks")
    background = args.get("background", False)
    role = args.get("role", "leaf")
    subagent_type = args.get("subagent_type", "general-purpose")

    # 批量模式
    if tasks:
        return _delegate_batch(tasks, background=background, **kwargs)

    if not goal:
        return json.dumps({"error": "goal 不能为空"}, ensure_ascii=False)

    # 检查角色权限（orchestrator 受深度限制）
    # 深度优先从父 agent 字段读（线程安全的私份），fallback 到 kwargs 或 0
    parent_agent = kwargs.get("agent_ref")
    if parent_agent is not None and hasattr(parent_agent, "spawn_depth"):
        current_depth = parent_agent.spawn_depth
    else:
        current_depth = int(kwargs.get("spawn_depth", 0))
    max_depth = kwargs.get("max_spawn_depth", 2)

    if role == "orchestrator" and current_depth >= max_depth:
        return json.dumps({
            "error": f"已达最大嵌套深度 {max_depth}",
            "current_depth": current_depth,
        }, ensure_ascii=False)

    # 透传 subagent_type 给子代理创建（工具集选择）
    kwargs["subagent_type"] = subagent_type

    if background:
        return _delegate_async(goal, args.get("context", ""), role, **kwargs)
    else:
        return _delegate_sync(goal, args.get("context", ""), role, **kwargs)


def _delegate_sync(
    goal: str,
    context: str,
    role: str,
    **kwargs,
) -> str:
    """同步委托：等待子代理完成。带超时（防止无限挂起）。

    用后台线程跑子代理，主线程 join(timeout)。超时后返回错误，
    子代理留在 daemon 线程继续（结果丢弃，与 async 模式一致）。
    """
    child_timeout = float(kwargs.get("child_timeout", 600))
    box: dict = {}

    def _run():
        try:
            box["result"] = _run_child(goal, context, role, **kwargs)
        except Exception as e:  # noqa: BLE001
            box["error"] = e

    thread = threading.Thread(target=_run, daemon=True, name="delegate-sync")
    thread.start()
    thread.join(timeout=child_timeout)

    if thread.is_alive():
        return json.dumps({
            "success": False,
            "error": f"子代理执行超时（{child_timeout}s），已放弃等待",
            "mode": "sync",
        }, ensure_ascii=False)
    if "error" in box:
        logger.exception("子代理执行失败")
        return json.dumps({
            "success": False,
            "error": str(box["error"]),
            "mode": "sync",
        }, ensure_ascii=False)
    return json.dumps({
        "success": True,
        "result": box["result"],
        "mode": "sync",
    }, ensure_ascii=False)


def _delegate_async(
    goal: str,
    context: str,
    role: str,
    **kwargs,
) -> str:
    """异步委托：立即返回，后台执行。"""
    delegation_id = f"del_{datetime.now(timezone.utc).strftime('%H%M%S%f')}"

    def _background():
        try:
            result = _run_child(goal, context, role, **kwargs)
            _delegation_queue.push({
                "delegation_id": delegation_id,
                "goal": goal,
                "success": True,
                "result": result,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            _delegation_queue.push({
                "delegation_id": delegation_id,
                "goal": goal,
                "success": False,
                "error": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })

    # 后台线程执行
    thread = threading.Thread(target=_background, daemon=True)
    thread.start()

    return json.dumps({
        "success": True,
        "mode": "async",
        "delegation_id": delegation_id,
        "message": f"子代理已启动（ID: {delegation_id}），完成后会通知你",
    }, ensure_ascii=False)


def _delegate_batch(tasks: list, *, background: bool, **kwargs) -> str:
    """批量并行委托。"""
    # 从 config 读并发上限（config.delegation.max_concurrent_children），
    # 不依赖 kwargs（之前永远 fallback 3，配置不生效）
    _cfg = (kwargs.get("config") or {}) if isinstance(kwargs.get("config"), dict) else {}
    max_concurrent = int((_cfg.get("delegation") or {}).get("max_concurrent_children", 5))
    child_timeout = float(kwargs.get("child_timeout", 600))

    results = []
    # 不用 `with ThreadPoolExecutor`：它退出时 shutdown(wait=True)，
    # Ctrl+C 的 KeyboardInterrupt 会卡在等子线程跑完 → 界面"没反应"。
    # 手动管理，KeyboardInterrupt 时传播中断 + 不阻塞等待。
    executor = ThreadPoolExecutor(max_workers=max_concurrent)
    futures = {}
    try:
        for i, task in enumerate(tasks):
            goal = task.get("goal", "")
            context = task.get("context", "")
            role = task.get("role", "leaf")

            future = executor.submit(_run_child, goal, context, role, **kwargs)
            futures[future] = i

        for future in futures:
            idx = futures[future]
            try:
                result = future.result(timeout=child_timeout)
                results.append({
                    "task_index": idx,
                    "success": True,
                    "result": result,
                })
            except Exception as e:
                results.append({
                    "task_index": idx,
                    "success": False,
                    "error": str(e),
                })
    except KeyboardInterrupt:
        # Ctrl+C：先传播中断给子代理（让它们在下轮迭代退出），
        # 取消未启动任务，shutdown(wait=False) 不阻塞，再向上抛。
        parent = kwargs.get("agent_ref")
        if parent is not None and hasattr(parent, "interrupt"):
            try:
                parent.interrupt()
            except Exception:
                pass
        for f in futures:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    executor.shutdown(wait=True)

    return json.dumps({
        "success": True,
        "mode": "batch",
        "results": results,
    }, ensure_ascii=False)


def _run_child(
    goal: str,
    context: str,
    role: str,
    **kwargs,
) -> str:
    """创建并运行一个子代理。

    子代理是一个独立的 AIAgent 实例，有自己的：
    - 会话 ID
    - 迭代预算（默认 50 次）
    - 工具集（leaf 模式受限）
    - 工作目录

    子代理不继承父代理的对话历史。

    batch1-T4: 子 agent 注册到父 agent._children，支持中断传播。
    """
    # 延迟导入避免循环
    from agent import AIAgent

    # 从 kwargs 或 config 获取 LLM 配置
    base_url = kwargs.get("base_url")
    api_key = kwargs.get("api_key")
    auth_token = kwargs.get("auth_token")
    model = kwargs.get("model")
    model_format = kwargs.get("model_format")

    if not (api_key or auth_token) or not model:
        # 从 config 读取
        try:
            from config import load_config
            config = load_config()

            # 子代理优先用轻量模型(default_haiku_model),省 token
            # 类 业界 模式:主对话用 opus,子代理用 haiku
            haiku_name = config.get("default_haiku_model", "")
            # 新模式:config["haiku_model"](llm 段注入的)
            haiku_cfg = config.get("haiku_model")
            if haiku_cfg:
                sub_cfg = haiku_cfg
            elif haiku_name and haiku_name in config.get("models", {}):
                # 老模式:config["models"][haiku_name]
                sub_cfg = config["models"][haiku_name]
            else:
                # fallback 到主模型兼容段
                sub_cfg = config.get("model", {})

            if not base_url:
                base_url = sub_cfg.get("base_url")
            if not model:
                model = sub_cfg.get("model") or sub_cfg.get("name")
            if not api_key:
                api_key = sub_cfg.get("api_key") or ""
            if not auth_token:
                auth_token = sub_cfg.get("auth_token") or ""
            # 向后兼容:api_key_env 指向环境变量(老 config.yaml 格式)
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

    # 构造子代理的 system prompt
    system_prompt = _build_child_system_prompt(goal, context, role)

    # 计算子代理的 spawn 深度（线程安全：参数传递，不写 os.environ）
    parent_agent = kwargs.get("agent_ref")
    if parent_agent is not None and hasattr(parent_agent, "spawn_depth"):
        parent_depth = parent_agent.spawn_depth
    else:
        parent_depth = int(kwargs.get("spawn_depth", 0))
    child_spawn_depth = parent_depth + 1

    # 工具集选择（对齐 Claude Code Agent subagent_type）
    # 提前解析 stype + custom_def，让 isolation=worktree 能在 worktree 创建分支前生效
    # （曾有时序 bug：isolated 在 custom_def 写入 kwargs 前读取，worktree 永不创建）
    stype = kwargs.get("subagent_type", "general-purpose")
    custom_def = None
    if stype not in ("general-purpose", "custom"):
        # 自定义子代理名：从 .md 定义加载
        from agent.agent_defs import get_agent_def
        custom_def = get_agent_def(stype)
        if custom_def is None:
            raise RuntimeError(
                f"未找到子代理定义: {stype}"
                f"（检查 ~/.OmniMate/agents/ 和 ./.claude/agents/）"
            )

    # 可选：隔离工作区（自定义 .md 定义 isolation=worktree 也开启）
    isolated = kwargs.get("isolated_workspace", False) or (
        custom_def is not None and custom_def.isolation == "worktree")
    original_cwd = os.getcwd()
    workspace_cleanup = None
    if isolated:
        try:
            from tools.worktree import create_isolated_workspace
            workspace_path, workspace_cleanup = create_isolated_workspace(
                name=f"delegate-{goal[:20].replace(' ', '-')}",
            )
            os.chdir(workspace_path)
            logger.info("子代理在隔离工作区运行: %s", workspace_path)
        except Exception as e:
            logger.warning("创建隔离工作区失败，用当前目录: %s", e)

    # batch1-T4: 获取父 agent 引用（用于中断传播）
    parent_agent = kwargs.get("agent_ref")

    child = None
    try:
        # 工具集选择（对齐 Claude Code Agent subagent_type）
        # - 自定义名：custom_def 已在 worktree 分支前加载，按定义配置 toolsets/model/perm/maxTurns
        # - custom：用显式 enabled_toolsets
        # - general-purpose：按角色默认
        if custom_def:
            # 按定义配置
            child_toolsets = custom_def.tools or (
                ["core"] if role == "orchestrator" else ["minimal"])
            # 自定义 .md 的 disallowedTools 覆盖父 config（非 union）
            disabled = custom_def.disallowed_tools or None
            child_model = custom_def.model or model
            child_perm_mode = custom_def.permission_mode or "default"
            child_max_iter = custom_def.max_turns or kwargs.get("child_max_iterations", 50)
        elif stype == "custom":
            child_toolsets = kwargs.get("enabled_toolsets") or (
                ["core"] if role == "orchestrator" else ["minimal"])
            disabled = None
            child_model = model
            child_perm_mode = "default"
            child_max_iter = kwargs.get("child_max_iterations", 50)
        elif role == "leaf":
            child_toolsets = kwargs.get("enabled_toolsets") or ["minimal"]
            disabled = None
            child_model = model
            child_perm_mode = "default"
            child_max_iter = kwargs.get("child_max_iterations", 50)
        else:  # orchestrator
            child_toolsets = kwargs.get("enabled_toolsets") or ["core"]
            disabled = None
            child_model = model
            child_perm_mode = "default"
            child_max_iter = kwargs.get("child_max_iterations", 50)

        # 自定义子代理 system_prompt 覆盖（重建 system_prompt）
        if custom_def and custom_def.system_prompt:
            system_prompt = _build_child_system_prompt(
                goal, context, role, override=custom_def.system_prompt)

        # disabled_tools 透传：AIAgent.__init__ 无此参数，走 config 透传
        # （get_tool_definitions 运行时从 self.config 读 disabled_tools，见 D2）
        child_config = None
        if disabled:
            # 继承父 config（如有）再加 disabled_tools
            parent_cfg = kwargs.get("config")
            child_config = dict(parent_cfg) if isinstance(parent_cfg, dict) else {}
            child_config["disabled_tools"] = disabled

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
            config=child_config,
        )

        # batch1-T4: 注册到父 agent._children（中断传播）
        if parent_agent is not None:
            try:
                parent_agent._children.append(child)
            except Exception:
                pass

        # === P1-10: pendingToolUseSummary ===
        # 长任务执行期间用 aux_llm 周期生成进度摘要，推到父 agent 的 stream_callback
        # 让前端知道"还在做什么"。aux_llm 不可用时降级为心跳。
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
            # 运行子代理
            result = child.chat(f"请执行任务: {goal}")

        # 06 NEW: 幻觉检测（在 summary_only 压缩前做，保留警告进摘要）
        try:
            from agent.team.hallucination_check import verify_claims, append_warning
            verification = verify_claims(
                result,
                task_store=kwargs.get("task_store"),
                fs_cwd=kwargs.get("cwd") or Path.cwd(),
            )
            result = append_warning(result, verification)
        except Exception as e:
            logger.warning("幻觉检测失败（fail-open）: %s", e)

        # summary_only：超长结果用 LLM 生成摘要，节省父代理 context
        summary_only = kwargs.get("summary_only", True)
        if summary_only and len(result) > 500:
            result = _summarize_child_result(result, child.llm_client, child.model)

        return result
    finally:
        # batch1-T4: 从父 agent._children 移除
        if parent_agent is not None and child is not None:
            try:
                if child in parent_agent._children:
                    parent_agent._children.remove(child)
            except Exception:
                pass
        # 恢复工作目录
        if workspace_cleanup:
            try:
                os.chdir(original_cwd)
            except Exception:
                pass
            workspace_cleanup()


def _summarize_child_result(result: str, client, model: str) -> str:
    """用 LLM 把子代理结果总结成 300 字以内的摘要。

    摘要失败时返回原文（不阻塞委托流程）。
    """
    prompt = (
        "把以下子代理执行结果总结成 300 字以内的摘要，保留：\n"
        "1. 核心结论\n"
        "2. 关键发现和数据\n"
        "3. 重要的文件路径、命令、错误信息\n"
        "4. 待办事项\n\n"
        f"子代理结果：\n{result[:8000]}"
    )
    try:
        from agent.llm_retry import call_with_retry
        response = call_with_retry(
            client,  # child.llm_client（LLMClient 实例）
            [{"role": "user", "content": prompt}],
        )
        summary = response.choices[0].message.content
        return f"[摘要] {summary}\n\n[完整结果 {len(result)} 字符已省略]"
    except Exception as e:
        logger.debug("子代理结果摘要失败，返回原文: %s", e)
        return result


def _build_child_system_prompt(goal: str, context: str, role: str, override: str = None) -> str:
    """构建子代理的 system prompt。

    override 非空时，base 用 override（自定义子代理 .md 的 system_prompt），
    仅追加上下文/约束/角色提示。None 时走默认构建（与历史行为一致）。
    """
    if override:
        parts = [override]
    else:
        parts = [
            "你是一个子代理，由父代理派生执行独立任务。",
            f"\n你的角色: {role}",
            f"\n你的任务目标: {goal}",
        ]

    if context:
        parts.append(f"\n来自父代理的上下文:\n{context}")

    # 自定义 override 也补一份通用约束（不含默认 goal/role 文本）
    if override:
        parts.append(
            "\n## 约束\n"
            "- 独立执行，不假设父代理历史\n"
            "- 结果要具体（文件路径、命令、数据）\n"
        )
    else:
        parts.append(
            "\n要求:\n"
            "- 专注完成任务，不要偏离目标\n"
            "- 完成后给出清晰的总结\n"
            "- 遇到不可解决的阻碍时，返回错误说明\n"
            "- 不要做任务范围外的事"
        )

    if role == "leaf":
        parts.append("\n你是 leaf 角色，不能再派生子代理。")

    return "\n".join(parts)


def _delegate_schema_overrides(schema: dict, runtime_ctx: dict) -> dict:
    """根据运行时状态改 subagent schema description（03）。

    让 LLM 看到当前剩余并发槽位，避免"试 spawn 被拒"浪费一轮。
    """
    agent = runtime_ctx.get("agent") if runtime_ctx else None
    if agent is None:
        return schema

    active = len(getattr(agent, "_children", []) or [])
    cfg = getattr(agent, "config", None) or {}
    max_children = (
        (cfg.get("delegation") or {}).get("max_concurrent_children", 5)
        if isinstance(cfg, dict) else 5
    )
    remaining = max(0, max_children - active)

    new_schema = dict(schema)
    desc = new_schema.get("description", "")
    status_line = (
        f"\n\n[运行时状态] 当前活跃子代理: {active}/{max_children}，"
        f"剩余可委派: {remaining}"
    )
    if remaining == 0:
        status_line += "\n⚠️ 已达并发上限，再委派会被拒绝。"
    new_schema["description"] = desc + status_line
    return new_schema


# 注册到 core 工具集（让 resolve("core") 能找到）
registry.register(
    name="subagent",
    toolset="core",
    schema=DELEGATE_TASK_SCHEMA,
    handler=_handle_delegate_task,
    schema_overrides_fn=_delegate_schema_overrides,
    emoji="🤝",
)
# 兼容 alias：历史会话/记忆里的 delegate_task 调用仍可 dispatch 命中
# toolset="_compat"：dispatch 按名字命中（不依赖 toolset），但 resolve_toolset("core")
# 不会包含它——LLM 只看到 subagent，delegate_task 仅给老调用兜底
registry.register(
    name="delegate_task",
    toolset="_compat",
    schema=DELEGATE_TASK_SCHEMA,
    handler=_handle_delegate_task,
    schema_overrides_fn=_delegate_schema_overrides,
    emoji="🤝",
    override=True,
)
