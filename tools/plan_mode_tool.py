"""计划模式（Plan Mode）的工具端：exit_plan_mode（提交计划求审批）+ plan_mode_v2_dispatch（多 Agent 并行调研）。

计划模式下 LLM 只做调研出方案、不许直接动手改东西；调研完调 exit_plan_mode
把计划交给用户审批，批了才切到执行模式。注意 handler 本身并不真正「完成」任务，
而是返回一个特殊错误码 error_type="plan_approval_required"，由 agent 主循环
捕获后调用审批回调走人工审批流程——相当于工具和主循环之间约定的暗号。

并行版（功能开关 plan_mode_v2_parallel）：
- plan_mode_v2_dispatch 工具：父 Agent 把大任务拆成 N 份 → 同时派 N 个 Plan 子代理
  （主对话派出去帮忙干活的分身）各自调研 → 把 N 份子计划合并成一份 → 交回父 Agent
  （之后仍可再调 exit_plan_mode 提交审批）
- _run_parallel_planners：并发调度（线程池 + 复用 delegate_tool 的 _run_child）
- _merge_plans：用 LLM 把 N 份子计划合并成一份（提示词模板见 PLAN_MERGE_PROMPT_TEMPLATE）

在项目里的位置：属于工具层（tools/），模块顶层有 registry.register，
会被 discover_builtin_tools() 的 AST 扫描自动发现并注册。
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

from tools.registry import registry

logger = logging.getLogger(__name__)


def handle_exit_plan_mode(args: dict, **kwargs) -> str:
    """exit_plan_mode 的实际处理函数：校验计划文本，然后发「等审批」暗号。

    计划不能是空的——空计划说明 LLM 还没想清楚。

    参数：
    - args：LLM 传的工具参数，只看必填的 plan（完整实施计划文本）
    - kwargs：运行时上下文（本函数用不到）

    返回：JSON 字符串。计划为空 → error_type=invalid_args（提示 LLM 补全后重试）；
    计划非空 → error_type=plan_approval_required（主循环看到这个码就去走用户审批）。
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
# 多 Agent 并行调度（功能开关 plan_mode_v2_parallel）
# ============================================================================

# 并发数硬上限：配置文件就算写 100 也只放到这么多，防止把机器资源打爆
_ABS_MAX_PARALLEL = 8


def _run_parallel_planners(subtasks: List[str], max_n: int, **kwargs) -> List[str]:
    """同时派出 N 个 Plan 子代理分头调研，收回每份子计划的文本。

    计划模式 V2 开启时，父 Agent 把大任务拆开，同时启动 N 个 Plan 子代理
    （N 不超过配置的 max_parallel_agents，默认 3），等全部跑完再汇总。
    用线程池并行；子代理内部的异步逻辑由 _run_child 自己用 asyncio.run 驱动。

    参数：
    - subtasks：子任务描述列表，每条是一段调研目标 + 范围
    - max_n：并发上限（来自 config.features.plan_mode_v2_parallel.max_parallel_agents）
    - **kwargs：原样转交给 _run_child 的参数（base_url/api_key/model/agent_ref 等）

    返回：成功完成的子计划文本列表。哪个子代理失败就跳过哪个
    （兜底思路：拿剩下的照样合并，不让一颗老鼠屎坏一锅粥），失败的那份不会出现。
    """
    # 函数内才 import，避免两个模块在加载阶段互相 import 卡死
    from tools.delegate_tool import _run_child

    if not subtasks:
        return []

    # 实际并发数取三者最小：任务数、配置上限、硬上限 8
    n = len(subtasks)
    effective_max = max(1, min(max_n, _ABS_MAX_PARALLEL))
    workers = min(n, effective_max)

    # 给每个子代理的参数里塞上 subagent_type=Plan，让它用内置的 Plan 角色定义跑
    base_kwargs = dict(kwargs)
    base_kwargs["subagent_type"] = "Plan"

    results: List[str] = []
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    try:
        for i, goal in enumerate(subtasks):
            # 给子代理的指令里标明「这是 N 份并行任务里的第几份」，让它知道自己在团队协作中
            ctx = (
                f"[Plan Mode V2 并行任务 {i + 1}/{n}] "
                f"请在最终回复直接输出这一份子计划正文（不调任何提交计划类工具）。"
            )
            future = executor.submit(_run_child, goal, ctx, "leaf", **base_kwargs)
            futures[future] = i

        for future in futures:
            idx = futures[future]
            try:
                # 超时用 kwargs.child_timeout（默认 600 秒），口径与批量委派保持一致
                child_timeout = float(kwargs.get("child_timeout", 600))
                result = future.result(timeout=child_timeout)
                results.append(result)
                logger.info(
                    "Plan V2 子代理 %d/%d 完成（%d 字符）",
                    idx + 1, n, len(result),
                )
            except Exception as e:
                # 兜底（spec §7.6 韧性要求）：单个子代理挂了不连累其他，拿剩下的合并
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
# 子计划合并逻辑（PLAN_MERGE_PROMPT_TEMPLATE 提示词 + _merge_plans）
# ============================================================================

PLAN_MERGE_PROMPT_TEMPLATE = """你是 CodeAgent 的计划合并器。

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
    """用 LLM 把 N 份子计划揉成一份连贯的最终计划。

    多个子代理各写各的，直接堆一起会有重复和冲突，需要一个「编辑部统稿」环节。

    参数：
    - sub_plans：子计划文本列表（失败的那时已剔除，至少 1 份）
    - llm_client：LLM 客户端（底层走 call_with_retry，自带重试和退避）
    - model：模型名

    返回：合并后的最终计划文本。LLM 调用失败也不报错，
    退化为「带标题的简单拼接」（fail-open：宁可产出降质结果也不阻塞流程）。
    """
    if not sub_plans:
        return "（无子计划可合并——所有并行子代理都失败了）"
    if len(sub_plans) == 1:
        return sub_plans[0]

    # 先把子计划拼成一块（每份前面加个编号标题）
    blocks = []
    for i, p in enumerate(sub_plans, 1):
        # 单份超 6000 字符就截断，防止拼起来的提示词大到撑爆请求
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
        # call_with_retry 是 async，而本函数经 sync 工具 handler 跑在
        # to_thread 工作线程里（不在宿主循环线程）——交给进程级常驻循环
        # 宿主同步等结果（等价旧的 asyncio.run，但这里用的是父 agent 的
        # 主 client：迁移后绑宿主循环，跨循环使用清零）
        from agent.loop_host import loop_host
        response = loop_host.run_async(call_with_retry(
            llm_client,
            [{"role": "user", "content": prompt}],
        ), exempt_from_fence=True)  # 后台线程长活，豁免回合栅栏（见 run_async docstring）
        return response.choices[0].message.content
    except Exception as e:
        # fail-open：LLM 合并失败就退化成拼接（至少保住内容，不阻塞流程）
        logger.warning("Plan 合并 LLM 调用失败（退化为拼接）: %s", e)
        header = (
            f"（LLM 合并失败，退化为 {len(sub_plans)} 份子计划拼接）\n\n"
        )
        return header + sub_plans_block


def handle_plan_mode_v2_dispatch(args: dict, **kwargs) -> str:
    """plan_mode_v2_dispatch 的实际处理函数：并行派活调研再合并，把最终计划交回父 Agent。

    流程：父 Agent 拆出 N 个子任务 → 同时派 N 个 Plan 子代理调研 →
    合并 N 份子计划 → 把合并结果返回。父 Agent 审阅后可再调 exit_plan_mode 提交审批。

    参数：
    - args：LLM 传的工具参数，只看必填的 subtasks（子调研目标的字符串数组）
    - kwargs：运行时上下文（config、agent_ref、base_url 等会转交子代理）

    返回：JSON 字符串，含 merged_plan（合并后的计划）和成功/总数统计。
    功能开关没开时返回 feature_disabled（LLM 看到提示会改走常规的单 Agent 路径）；
    子任务参数不合法返回 invalid_args；所有子代理都挂了返回 all_subagents_failed。
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

    # 并发数 N 由 LLM 自己定：subtasks 数组多长就派几个，
    # 但会被 max_parallel_agents 上限压住（在 _run_parallel_planners 里钳制）
    flag_cfg = get_feature_config(config, "plan_mode_v2_parallel")
    max_n = int(flag_cfg.get("max_parallel_agents", 3))

    sub_plans = _run_parallel_planners(subtasks, max_n, **kwargs)

    if not sub_plans:
        return json.dumps({
            "error": "所有 Plan 子代理都失败了（详见日志）。请简化子任务或稍后重试。",
            "error_type": "all_subagents_failed",
        }, ensure_ascii=False)

    # 合并这活用父 agent 的 LLM 客户端：合并需要全局视角，不能交给子代理的轻量模型
    parent_agent = kwargs.get("agent_ref")
    llm_client = getattr(parent_agent, "llm_client", None) if parent_agent else None
    model = getattr(parent_agent, "model", "") if parent_agent else ""

    if llm_client is not None and model:
        merged = _merge_plans(sub_plans, llm_client, model)
    else:
        # 拿不到父 agent 引用（测试或非标准调用路径）→ 只能拼接兜底
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


# 模块顶层注册——discover_builtin_tools 的 AST 扫描靠发现这些注册调用来自动收录本模块
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
    isConcurrencySafe=False,  # 会触发审批流程、切换计划/执行模式这种状态变更，必须串行
)


# 多 Agent 并行调度工具（功能开关 plan_mode_v2_parallel 门控）
# 注册到 plan 工具集：进入计划模式后 LLM 才看得见它。开关没开时 handler 返回
# feature_disabled，LLM 会看到提示改走常规的单 Agent 路径（/plan + exit_plan_mode）。
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
    isConcurrencySafe=False,  # 要起一串子代理再做合并，串行更稳（防止并发风暴）
)
