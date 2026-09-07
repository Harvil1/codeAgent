# -*- coding: utf-8 -*-
"""轮末技能学习观察二件（从 agent/__init__.py 平移而来）。

大白话：这两个函数原本是 AIAgent 的两个方法，拆出来只为给
agent/__init__.py 减负——行为零变化，搬的是同一份代码：

- maybe_skill_learning：轮末做行为观察（instinct，直觉观察记录），
  攒够一簇就演化成技能；门槛（主代理/开关/记忆仓库）和演化参数
  全从 config 读，任何异常只打 debug 日志（学习是旁路）。
- collect_turn_tool_trace：从本轮历史片段里抽出观察器要的
  工具调用/结果对（解析 JSON、下标对齐、坏数据跳过）。

本模块三条铁律（跟拆分一期约定一致）：
1. 属性全留 AIAgent——函数不自己存状态，一律读写 ``agent._xxx``
   （如轮起点 ``agent._sl_turn_start``，写入点仍在 AIAgent 主循环），
   第一参固定收 agent 实例（原 ``self``）。
2. 禁止模块级 import agent root（防循环导入）——skill_learning
   包内依赖沿用函数内延迟导入。
3. 函数体与原方法逐字节平移，唯一改写是 ``self`` → ``agent``；
   同模块兄弟函数直接按自由函数名调。
"""

import json
import logging

logger = logging.getLogger(__name__)


async def maybe_skill_learning(agent, user_message: str) -> None:
    """轮末做行为观察（instinct，直觉观察记录），攒够一簇就演化成技能。

    门槛：只有主代理（派生深度 0）+ config 里 skill_learning.enabled
    显式开着（默认关）+ 有记忆仓库。
    观察后端按 config 选——"llm" 走辅助模型提取（内部有熔断/
    限流，失败自动退回规则式），其他走规则式观察；演化的门槛（置信度/
    最小簇大小）都从 config 读，不再写死。
    任何异常只打 debug 日志——学习是旁路，绝不影响主对话。

    参数：user_message: 本轮用户消息。返回：无。
    """
    sl_cfg = agent.config.get("skill_learning", {}) or {}
    if (agent.spawn_depth != 0
            or not sl_cfg.get("enabled")
            or agent.memory_store is None):
        return
    try:
        from pathlib import Path

        from agent.skill_learning import (
            maybe_evolve, observe_turn, observe_turn_llm,
        )
        from agent.skill_learning.store import InstinctStore

        store = InstinctStore(Path(agent.codeAgent_home) / ".skill-learning")
        calls, results = collect_turn_tool_trace(
            agent, getattr(agent, "_sl_turn_start", 0))
        if sl_cfg.get("observer") == "llm":
            # LLM 后端：熔断（3 次失败）/冷却（30 秒）/会话上限（20 次）
            # 都在函数内部处理，失败自动退回规则式观察
            await observe_turn_llm(
                user_text=user_message or "",
                tool_calls=calls,
                tool_results=results,
                aux_llm_router=getattr(agent, "aux_llm_router", None),
                store=store,
                scope="global",
            )
        else:
            # 规则式后端：scope="global"——纠错/恢复/序列类信号全局共享；
            # 项目约定类信号由观察器内部按信号类型归入项目隔离区
            observe_turn(
                user_text=user_message or "",
                tool_calls=calls,
                tool_results=results,
                store=store,
                scope="global",
            )
        skills_dir = Path(agent.codeAgent_home) / "skills"
        # 只演化全局区：项目约定类的直觉记录只存着，
        # 不参与自动演化——生成到全局技能目录会跨项目泄漏（破坏记忆
        # 分层项目隔离），而且约定簇的触发词恒为「项目约定」会撞名。项目约定
        # 的演化留作后续（需要项目级技能目录或 paths 门控）。
        maybe_evolve(
            store, "global", skills_dir,
            min_avg_confidence=sl_cfg.get("evolve_threshold", 0.75),
            min_members=sl_cfg.get("evolve_min_cluster", 3),
        )
    except Exception as e:
        logger.debug("skill_learning fail-open: %s", e)


def collect_turn_tool_trace(agent, start_idx: int):
    """从本轮历史片段里抽出观察器要的工具调用/结果对。

    历史里存的参数和结果都是 JSON 字符串，观察器要的是解析后的
    形状（[{"name","arguments"}] / [{"name","error","content"}]）。
    结果是按原调用顺序回填的，两个列表下标天然对齐；解析失败的条目
    跳过，不让观察器吃坏数据。

    参数：start_idx: 本轮历史的起点下标。

    返回：(calls, results) 两个列表。
    """
    calls, results = [], []
    for msg in agent.conversation_history[start_idx:]:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = (tc or {}).get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                calls.append({"name": fn.get("name"), "arguments": args})
        elif msg.get("role") == "tool":
            content = msg.get("content")
            err = None
            try:
                rd = json.loads(content) if isinstance(content, str) else {}
                if isinstance(rd, dict) and rd.get("error"):
                    err = str(rd["error"])
            except (json.JSONDecodeError, TypeError):
                pass
            results.append({
                "name": msg.get("name"), "content": content, "error": err,
            })
    return calls, results
