"""delegate_task 工具：派生子代理执行独立任务。

两种角色：
  - leaf（默认）：执行者，不能再委托
    有 terminal/read_file 等工具
    不能调用 delegate_task/clarify/memory/send_message（通过工具集隔离）

  - orchestrator：协调者，可以继续委托
    可以调用 delegate_task 派生自己的子代理
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
# delegate_task 工具
# ---------------------------------------------------------------------------

DELEGATE_TASK_SCHEMA = {
    "name": "delegate_task",
    "description": (
        "派生子代理执行独立任务。子代理有独立的上下文和工具。\n\n"
        "两种模式：\n"
        "- 同步（默认）：等待子代理完成后继续\n"
        "- 异步（background=True）：立即继续，结果稍后送达\n\n"
        "批量模式：传 tasks=[...] 并行执行多个子代理。\n\n"
        "角色：\n"
        "- leaf（默认）：执行者，不能再委托\n"
        "- orchestrator：可继续派生（受 max_spawn_depth 限制）"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "子代理的目标（单任务模式）",
            },
            "context": {
                "type": "string",
                "description": "给子代理的额外上下文",
            },
            "tasks": {
                "type": "array",
                "items": {"type": "object"},
                "description": "批量任务列表（并行执行）",
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
    """处理委托请求。"""
    goal = args.get("goal", "")
    tasks = args.get("tasks")
    background = args.get("background", False)
    role = args.get("role", "leaf")

    # 批量模式
    if tasks:
        return _delegate_batch(tasks, background=background, **kwargs)

    if not goal:
        return json.dumps({"error": "goal 不能为空"}, ensure_ascii=False)

    # 检查角色权限（orchestrator 受深度限制）
    current_depth = int(os.environ.get("_SPAWN_DEPTH", "0"))
    max_depth = kwargs.get("max_spawn_depth", 2)

    if role == "orchestrator" and current_depth >= max_depth:
        return json.dumps({
            "error": f"已达最大嵌套深度 {max_depth}",
            "current_depth": current_depth,
        }, ensure_ascii=False)

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
    """同步委托：等待子代理完成。"""
    try:
        result = _run_child(goal, context, role, **kwargs)
        return json.dumps({
            "success": True,
            "result": result,
            "mode": "sync",
        }, ensure_ascii=False)
    except Exception as e:
        logger.exception("子代理执行失败")
        return json.dumps({
            "success": False,
            "error": str(e),
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
    max_concurrent = kwargs.get("max_concurrent_children", 3)

    results = []
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {}
        for i, task in enumerate(tasks):
            goal = task.get("goal", "")
            context = task.get("context", "")
            role = task.get("role", "leaf")

            future = executor.submit(_run_child, goal, context, role, **kwargs)
            futures[future] = i

        for future in futures:
            idx = futures[future]
            try:
                result = future.result(timeout=kwargs.get("child_timeout", 600))
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
    model = kwargs.get("model")

    if not api_key or not model:
        # 从 config 读取
        try:
            from config import load_config
            config = load_config()
            if not base_url:
                base_url = config["model"].get("base_url")
            if not model:
                model = config["model"]["name"]
            if not api_key:
                api_key_env = config["model"]["api_key_env"]
                api_key = os.environ.get(api_key_env)
        except Exception as e:
            raise RuntimeError(f"子代理无法获取 LLM 配置: {e}")

    if not api_key:
        raise RuntimeError("子代理无法获取 API key（未设置环境变量）")

    # 构造子代理的 system prompt
    system_prompt = _build_child_system_prompt(goal, context, role)

    # 设置环境（深度 +1，通过环境变量传递）
    current_depth = int(os.environ.get("_SPAWN_DEPTH", "0"))
    os.environ["_SPAWN_DEPTH"] = str(current_depth + 1)

    # 可选：隔离工作区
    isolated = kwargs.get("isolated_workspace", False)
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
        # leaf 角色：限制工具集
        if role == "leaf":
            child_toolsets = kwargs.get("enabled_toolsets") or ["minimal"]
        else:  # orchestrator
            child_toolsets = kwargs.get("enabled_toolsets") or ["core"]

        child = AIAgent(
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_iterations=kwargs.get("child_max_iterations", 50),
            enabled_toolsets=child_toolsets,
            system_prompt_override=system_prompt,
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
        # 恢复环境
        os.environ["_SPAWN_DEPTH"] = str(current_depth)


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


def _build_child_system_prompt(goal: str, context: str, role: str) -> str:
    """构建子代理的 system prompt。"""
    parts = [
        "你是一个子代理，由父代理派生执行独立任务。",
        f"\n你的角色: {role}",
        f"\n你的任务目标: {goal}",
    ]

    if context:
        parts.append(f"\n来自父代理的上下文:\n{context}")

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
    """根据运行时状态改 delegate_task schema description（03）。

    让 LLM 看到当前剩余并发槽位，避免"试 spawn 被拒"浪费一轮。
    """
    agent = runtime_ctx.get("agent") if runtime_ctx else None
    if agent is None:
        return schema

    active = len(getattr(agent, "_children", []) or [])
    cfg = getattr(agent, "config", None) or {}
    max_children = (
        cfg.get("delegate", {}).get("max_concurrent_children", 5)
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
    name="delegate_task",
    toolset="core",
    schema=DELEGATE_TASK_SCHEMA,
    handler=_handle_delegate_task,
    schema_overrides_fn=_delegate_schema_overrides,
    emoji="🤝",
)
