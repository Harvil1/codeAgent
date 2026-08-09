"""Plan Mode 工具：exit_plan_mode + plan_mode_v2_dispatch（多 Agent 并行）。

LLM 在计划模式下调研完成后，调此工具请求用户审批计划。
handler 不真正"完成"任务，而是返回特殊 error_type="plan_approval_required"，
由 agent 主循环捕获并调用 plan_approval_callback 走审批流程。

P6 新增（spec §7.6，flag plan_mode_v2_parallel）：
- plan_mode_v2_dispatch 工具：父 Agent 拆 N 个子任务 → N 个 Plan 子代理并行调研
  → 合并 N 份子计划 → 父 Agent 拿合并结果继续（可再调 exit_plan_mode 提交审批）
- _run_parallel_planners：并发调度（ThreadPoolExecutor + _run_child(subagent_type=Plan)）
- _merge_plans：LLM 合并 N 份子计划（PLAN_MERGE_PROMPT_TEMPLATE）

discover_builtin_tools() 通过 AST 扫描自动发现本模块（顶层有 registry.register）。
"""
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

from tools.registry import registry

logger = logging.getLogger(__name__)


def handle_exit_plan_mode(args: dict, **kwargs) -> str:
    """exit_plan_mode handler。

    返回特殊 error_type 让主循环走审批分支：
    - 空 plan → invalid_args（LLM 重试）
    - 非空 plan → plan_approval_required（主循环捕获，调回调）
    """
    plan = (args or {}).get("plan", "")
    if not isinstance(plan, str):
        plan = str(plan) if plan else ""
    plan = plan.strip()
    if not plan:
        return json.dumps({
            "error": "plan 字段不能为空。请提供完整的实施计划摘要：要改什么文件、为什么、步骤、风险点。",
            "error_type": "invalid_args",
        }, ensure_ascii=False)

    return json.dumps({
        "error": "等待用户审批计划",
        "error_type": "plan_approval_required",
        "plan": plan,
    }, ensure_ascii=False)


# ============================================================================
# P6.1: 多 Agent 并行调度（flag plan_mode_v2_parallel）
# ============================================================================

# 硬上限：即使配置写 100 也只放这么多（防爆资源）
_ABS_MAX_PARALLEL = 8


def _run_parallel_planners(subtasks: List[str], max_n: int, **kwargs) -> List[str]:
    """并发调度 N 个 Plan 子代理，返回每份子计划文本。

    spec §7.6：进入 plan mode V2 时（flag 开），父 Agent 拆任务，启动 N 个 Plan 子代理
    （N ≤ max_parallel_agents，默认 3），等所有完成（ThreadPoolExecutor 并行，
    依赖 Plan 2A 的 async 基础已在 _run_child 内部用 asyncio.run 驱动）。

    Args:
        subtasks: 子任务描述列表（每个是一段调研目标 + 范围）
        max_n: 并发上限（config.features.plan_mode_v2_parallel.max_parallel_agents）
        **kwargs: 透传给 _run_child 的参数（base_url/api_key/model/agent_ref 等）

    Returns:
        成功完成的子计划文本列表（顺序与 subtasks 一致）。
        失败的子代理被跳过（兜底：用剩余结果合并），对应位置不出现。
    """
    # 延迟导入避免循环
    from tools.delegate_tool import _run_child

    if not subtasks:
        return []

    # 实际并发数：min(用户要的 N, 配置上限, 硬上限)
    n = len(subtasks)
    effective_max = max(1, min(max_n, _ABS_MAX_PARALLEL))
    workers = min(n, effective_max)

    # 给每个 Plan 子代理的 kwargs 加 subagent_type=Plan（用 OmniMate 内置 plan.md）
    base_kwargs = dict(kwargs)
    base_kwargs["subagent_type"] = "Plan"

    results: List[str] = []
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    try:
        for i, goal in enumerate(subtasks):
            # context 标明这是并行计划任务的第几份，便于子代理知道自己在多 Agent 协作中
            ctx = (
                f"[Plan Mode V2 并行任务 {i + 1}/{n}] "
                f"请在最终回复直接输出这一份子计划正文（不调任何提交计划类工具）。"
            )
            future = executor.submit(_run_child, goal, ctx, "leaf", **base_kwargs)
            futures[future] = i

        for future in futures:
            idx = futures[future]
            try:
                # 超时沿用 kwargs.child_timeout（默认 600s），与 _delegate_batch 一致
                child_timeout = float(kwargs.get("child_timeout", 600))
                result = future.result(timeout=child_timeout)
                results.append(result)
                logger.info(
                    "Plan V2 子代理 %d/%d 完成（%d 字符）",
                    idx + 1, n, len(result),
                )
            except Exception as e:
                # 兜底：单个子代理失败不影响其他，用剩余结果合并（spec §7.6 韧性）
                logger.warning(
                    "Plan V2 子代理 %d/%d 失败（跳过，用剩余合并）: %s",
                    idx + 1, n, e,
                )
    except KeyboardInterrupt:
        for f in futures:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=True)

    return results


# ============================================================================
# P6.2: Plan 合并逻辑（PLAN_MERGE_PROMPT_TEMPLATE + _merge_plans）
# ============================================================================

PLAN_MERGE_PROMPT_TEMPLATE = """你是 OmniMate 的计划合并器。

下面是 {n} 份并行的子计划。请合并成一份最终的、连贯的实施计划。

## 合并规则

1. **去重**：不同子计划里重复的步骤只保留一份（取最详细的版本）。
2. **互补**：A 没覆盖但 B 覆盖的点补进来。
3. **排顺序**：按依赖关系重排步骤（先底层后上层，先准备后执行）。
4. **冲突裁决**：如果两份计划对同一改动有矛盾（比如一个说加文件 X，一个说不加），
   你来裁决——选更合理的那边，并在计划末尾的「裁决说明」段简述为什么。
5. **保留可执行性**：每个步骤必须含具体文件路径 + 改动点 + 验证方式。

## 输出格式

```
## 最终实施计划

### 目标
<一句话>

### 步骤
1. [文件路径] <改动点> —— 验证：<怎么验>
2. ...

### 风险点
- ...

### 裁决说明（如有冲突）
- <冲突点> → 选 <方案> 因为 <理由>
```

## 子计划列表

{sub_plans_block}
"""


def _merge_plans(sub_plans: List[str], llm_client, model: str) -> str:
    """用 LLM 合并 N 份子计划成一份最终计划。

    Args:
        sub_plans: 子计划文本列表（已过滤掉失败的，至少 1 份）
        llm_client: LLM 客户端（用 call_with_retry 走重试/退避）
        model: 模型名

    Returns:
        合并后的最终计划文本。合并失败时回退成拼接（fail-open，不阻塞）。
    """
    if not sub_plans:
        return "（无子计划可合并——所有并行子代理都失败了）"
    if len(sub_plans) == 1:
        return sub_plans[0]

    # 拼子计划块（每份加分隔标题）
    blocks = []
    for i, p in enumerate(sub_plans, 1):
        # 截断超长子计划避免 prompt 爆炸（单份最多 6000 字符）
        truncated = p[:6000]
        if len(p) > 6000:
            truncated += f"\n...(已截断，原文 {len(p)} 字符)"
        blocks.append(f"### 子计划 {i}\n{truncated}")
    sub_plans_block = "\n\n".join(blocks)

    prompt = PLAN_MERGE_PROMPT_TEMPLATE.format(
        n=len(sub_plans),
        sub_plans_block=sub_plans_block,
    )

    try:
        from agent.llm_retry import call_with_retry
        response = asyncio.run(call_with_retry(
            llm_client,
            [{"role": "user", "content": prompt}],
        ))
        return response.choices[0].message.content
    except Exception as e:
        # fail-open：LLM 合并失败就退化为拼接（带分隔标题）
        logger.warning("Plan 合并 LLM 调用失败（退化为拼接）: %s", e)
        header = (
            f"（LLM 合并失败，退化为 {len(sub_plans)} 份子计划拼接）\n\n"
        )
        return header + sub_plans_block


def handle_plan_mode_v2_dispatch(args: dict, **kwargs) -> str:
    """plan_mode_v2_dispatch handler。

    父 Agent 拆出 N 个子任务 → 并行调 N 个 Plan 子代理 → 合并 N 份子计划 →
    返回合并后的最终计划给父 Agent。父 Agent 拿到后可以再调 exit_plan_mode 提交审批。

    flag 门控：config.features.plan_mode_v2_parallel.enabled=False 时返回
    feature_disabled，LLM 会看到提示走原单 Agent 路径（不调本工具）。
    """
    config = kwargs.get("config") or {}
    if not isinstance(config, dict):
        config = {}

    from agent.feature_flags import is_feature_enabled, get_feature_config
    if not is_feature_enabled(config, "plan_mode_v2_parallel"):
        return json.dumps({
            "error": (
                "plan_mode_v2_parallel 功能未开启。"
                "请在 settings.json 的 features.plan_mode_v2_parallel.enabled=true "
                "后使用；未开启时走原单 Agent 计划路径（/plan + exit_plan_mode）。"
            ),
            "error_type": "feature_disabled",
        }, ensure_ascii=False)

    subtasks = args.get("subtasks") or []
    if not subtasks or not isinstance(subtasks, list):
        return json.dumps({
            "error": "subtasks 必须是非空字符串数组（每个是一段子调研目标）。",
            "error_type": "invalid_args",
        }, ensure_ascii=False)

    # N 由 LLM 自决（决策 6）：subtasks 长度就是 LLM 想要的并发数，
    # 但受 max_parallel_agents 上限钳制（_run_parallel_planners 内部）
    flag_cfg = get_feature_config(config, "plan_mode_v2_parallel")
    max_n = int(flag_cfg.get("max_parallel_agents", 3))

    sub_plans = _run_parallel_planners(subtasks, max_n, **kwargs)

    if not sub_plans:
        return json.dumps({
            "error": "所有 Plan 子代理都失败了（详见日志）。请简化子任务或稍后重试。",
            "error_type": "all_subagents_failed",
        }, ensure_ascii=False)

    # 合并：用父 agent 的 LLM client（避免子代理的轻量模型做合并这种要全局视角的活）
    parent_agent = kwargs.get("agent_ref")
    llm_client = getattr(parent_agent, "llm_client", None) if parent_agent else None
    model = getattr(parent_agent, "model", "") if parent_agent else ""

    if llm_client is not None and model:
        merged = _merge_plans(sub_plans, llm_client, model)
    else:
        # 父 agent 引用拿不到（测试/非标准调用路径）→ 拼接兜底
        merged = _merge_plans(sub_plans, None, "")
        merged = f"（无父 agent LLM，退化为拼接）\n\n{merged}"

    succeeded = len(sub_plans)
    attempted = len(subtasks)
    return json.dumps({
        "success": True,
        "subplans_succeeded": succeeded,
        "subtasks_attempted": attempted,
        "max_parallel_agents": max_n,
        "merged_plan": merged,
        "message": (
            f"已并行调度 {attempted} 个 Plan 子代理，{succeeded} 个成功，"
            f"合并成 1 份最终计划。请审阅 merged_plan，"
            f"如需提交审批再调 exit_plan_mode。"
        ),
    }, ensure_ascii=False)


# 模块顶层 register —— 被 discover_builtin_tools AST 扫描自动发现
registry.register(
    name="exit_plan_mode",
    toolset="plan",
    schema={
        "name": "exit_plan_mode",
        "description": (
            "Plan 写好后调此工具请求用户审批。批准后才能进入执行模式。"
            "调用此工具前必须已完成调研。"
            "plan 参数要包含完整的实施步骤：要改什么文件、为什么、步骤、风险点。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "完整的实施计划摘要。",
                }
            },
            "required": ["plan"],
        },
    },
    handler=handle_exit_plan_mode,
    emoji="📋",
    isConcurrencySafe=False,  # 状态变更：触发审批流程（切换 plan/execute 模式），必须串行
)


# P6: 多 Agent 并行调度工具（flag plan_mode_v2_parallel 门控）
# 注册到 plan 工具集：进入 plan mode 后 LLM 可见。flag OFF 时 handler 返回
# feature_disabled，LLM 会走原单 Agent 路径（/plan + exit_plan_mode）。
registry.register(
    name="plan_mode_v2_dispatch",
    toolset="plan",
    schema={
        "name": "plan_mode_v2_dispatch",
        "description": (
            "Plan Mode V2：多 Agent 并行调度。把一个复杂任务拆成 N 个子调研目标，"
            "并行调 N 个 Plan 子代理（N ≤ max_parallel_agents），合并 N 份子计划成 1 份。"
            "适用于：任务规模大、可切成独立调研块（如「前端改 3 个页面 + 后端加 2 个接口」）。"
            "需要 settings.json 的 features.plan_mode_v2_parallel.enabled=true。"
            "返回 merged_plan 后，请审阅并按需调 exit_plan_mode 提交审批。"
            "N 由你自决——subtasks 数组长度就是并发数（受 max_parallel_agents 钳制）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "子调研目标数组。每个元素是一段独立的调研范围 + 目标描述，"
                        "会被分派给一个 Plan 子代理。"
                        "例：[\"调研前端登录页改动点\", \"调研后端 auth 接口\", \"调研测试覆盖\"]"
                    ),
                }
            },
            "required": ["subtasks"],
        },
    },
    handler=handle_plan_mode_v2_dispatch,
    emoji="🔀",
    isConcurrencySafe=False,  # 起多个子代理 + 合并，串行更稳（避免并发风暴）
)
