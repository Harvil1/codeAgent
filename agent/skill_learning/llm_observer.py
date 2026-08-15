# -*- coding: utf-8 -*-
"""LLMObserver：LLM 观察后端（CCAR15 Task 4，对标 CCB sessionObserver 的 LLM 模式）。

启发式观察器（observer.py）只认四类固定正则信号；LLM 后端把整轮轨迹
（用户消息 + 工具调用 + 工具结果）交给 aux_llm 提取——prompt 对齐 CCB：

    "从观察提取最多 3 条原子 instinct，纯 JSON 数组
     [{trigger, action, confidence}]，没有返 []，不猜"

韧性设计（对齐 context_compressor 熔断器的模块级状态写法）：
* **熔断**：连续失败 3 次开闸；开闸后不再调 LLM，直接回退启发式
* **冷却**：开闸 30s 后自动合闸重试（合闸时清零失败计数，半开语义）
* **会话上限**：每会话最多调 LLM 20 次（防观察器吃掉配额），超限回退
* **回退**：任何失败（无 router / LLM 异常 / 解析失败）→ 回退启发式
  observe_turn；返回值统一为"实际入库的 Instinct 列表"（回退路径用
  recorder 包装 store 透传 upsert 并记录，不落两套语义）

注意：**空数组 [] 是合法成功**（这轮确实没什么可记的），重置熔断计数，
不算失败——只有调不动 / 解析不出才是失败。
"""
import json
import math
import logging
import time
from datetime import datetime, timezone
from typing import List

from agent.skill_learning.observer import _clip, observe_turn
from agent.skill_learning.store import Instinct

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 熔断 / 限流：模块级状态（对齐 context_compressor 的写法）
# ---------------------------------------------------------------------------
_consecutive_failures = 0
_circuit_open = False
_circuit_opened_at = 0.0
_session_call_count = 0

MAX_CONSECUTIVE_FAILURES = 3      # 连续失败 N 次开闸
COOLDOWN_SECONDS = 30.0           # 开闸后冷却 N 秒可重试
MAX_CALLS_PER_SESSION = 20        # 每会话 LLM 调用上限
MAX_INSTINCTS_PER_TURN = 3        # 单轮最多提取条数（prompt 已声明）

# LLM 条目缺 confidence / 非法时的默认值（单次 LLM 观察，取偏保守的 0.4）
_DEFAULT_CONFIDENCE = 0.4

# 可 patch 的时钟（测试用；monotonic 不受系统时间回拨影响）
_now = time.monotonic

_OBSERVER_SYSTEM_PROMPT = "你是行为观察助手。只输出 JSON 数组，不要其他文字。"


def reset_llm_observer_state() -> None:
    """会话开始时重置熔断/限流状态（避免跨会话污染）。

    在 AIAgent.__init__ 调用（对齐 reset_compact_circuit_breaker 的用法）。
    """
    global _consecutive_failures, _circuit_open, _circuit_opened_at, _session_call_count
    _consecutive_failures = 0
    _circuit_open = False
    _circuit_opened_at = 0.0
    _session_call_count = 0


class _RecordingStore:
    """启发式回退路径的透传 recorder：upsert 原样落盘 + 记录写入的条目。"""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.written: List[Instinct] = []

    def upsert(self, inst: Instinct) -> Instinct:
        self.written.append(inst)
        return self._inner.upsert(inst)


def _strip_code_fence(text: str) -> str:
    """剥掉 LLM 可能加的 ```json ... ``` 包装（对齐 goal.py 的写法）。"""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _build_observer_prompt(user_text: str, tool_calls, tool_results) -> str:
    """把本轮轨迹浓缩成观察 prompt（防长轨迹刷 token：各段截断）。"""
    parts = []
    if user_text:
        parts.append(f"用户消息：{_clip(user_text, 2000)}")
    if tool_calls:
        lines = []
        for i, c in enumerate(tool_calls or []):
            c = c or {}
            name = c.get("name") or "?"
            try:
                args = json.dumps(c.get("arguments", {}), ensure_ascii=False)
            except (TypeError, ValueError):
                args = str(c.get("arguments"))
            lines.append(f"{i + 1}. {name}({_clip(args, 200)})")
        parts.append("本轮工具调用：\n" + "\n".join(lines))
    if tool_results:
        lines = []
        for i, r in enumerate(tool_results or []):
            r = r or {}
            name = r.get("name") or "?"
            if r.get("error"):
                lines.append(f"{i + 1}. {name} → 失败：{_clip(str(r['error']), 150)}")
            else:
                lines.append(f"{i + 1}. {name} → 成功：{_clip(str(r.get('content') or ''), 150)}")
        parts.append("工具结果：\n" + "\n".join(lines))
    body = "\n\n".join(parts) or "（本轮无有效观察内容）"
    return (
        f"{body}\n\n"
        "从以上观察提取最多 3 条原子 instinct（trigger→action 的条件反射式"
        "习惯，trigger 是触发情境、action 是建议行为）。\n"
        '只输出纯 JSON 数组，格式 [{"trigger": "...", "action": "...", '
        '"confidence": 0.0~1.0}]，不要任何其他文字。\n'
        "没有值得记的就返回 []，不要猜测。"
    )


def _extract_content(response) -> str:
    """从 LLM 响应取文本（兼容对象式 choices / dict 式返回）。"""
    raw = ""
    choices = getattr(response, "choices", None) or []
    if choices:
        raw = choices[0].message.content or ""
    if not raw and isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            raw = (choices[0].get("message", {}) or {}).get("content", "")
    return raw


def _parse_instincts(raw: str) -> List[dict]:
    """解析 LLM 输出为合法的 [{trigger, action, confidence}] 列表。

    非 JSON / 非数组 / 一条合法条目都没有 → 抛 ValueError（调用方计入
    熔断并回退启发式）；空数组 [] 是合法结果返回 []（不算失败）。
    单条缺 trigger/action 跳过；confidence 非法回退默认值。
    """
    text = _strip_code_fence(raw)
    data = json.loads(text)  # JSONDecodeError 向上抛（也计入熔断）
    if not isinstance(data, list):
        raise ValueError(f"LLM 输出不是 JSON 数组: {type(data).__name__}")
    items = []
    for it in data[:MAX_INSTINCTS_PER_TURN]:
        if not isinstance(it, dict):
            continue
        trigger = str(it.get("trigger") or "").strip()
        action = str(it.get("action") or "").strip()
        if not trigger or not action:
            continue  # 不猜：缺关键字段的条目直接丢
        try:
            conf = float(it.get("confidence"))
            # NaN/Infinity 不能进 clamp：min(1.0, nan) 会返回 1.0（nan < 1.0
            # 为 False），方向反了——落保守默认值（T4 review 快修）
            if math.isnan(conf) or math.isinf(conf):
                raise ValueError
        except (TypeError, ValueError):
            conf = _DEFAULT_CONFIDENCE
        items.append({
            "trigger": trigger, "action": action,
            "confidence": max(0.0, min(1.0, conf)),
        })
    if data and not items:
        raise ValueError("LLM 输出数组里没有一条合法 instinct")
    return items


def _heuristic_fallback(*, user_text, tool_calls, tool_results, store,
                        scope) -> List[Instinct]:
    """回退启发式 observe_turn，返回实际写入 store 的条目列表。"""
    recorder = _RecordingStore(store)
    try:
        observe_turn(
            user_text=user_text or "",
            tool_calls=tool_calls,
            tool_results=tool_results,
            store=recorder,
            scope=scope,
        )
    except Exception as e:
        # observe_turn 自身 fail-open，理论上不会到这；再兜一层
        logger.debug("llm_observer 启发式回退异常（fail-open）: %s", e)
    return recorder.written


async def observe_turn_llm(*, user_text: str, tool_calls, tool_results,
                           aux_llm_router, store, scope: str = "global"
                           ) -> List[Instinct]:
    """LLM 观察后端：aux_llm 提取 → 解析 → upsert 入库。

    Args:
        user_text: 本轮用户消息原文（可为空）。
        tool_calls: [{"name", "arguments"}]，与 tool_results 下标对齐。
        tool_results: [{"name", "error"(可选), "content"}]。
        aux_llm_router: AuxLLMRouter（或任何有 async chat_completions 的
            对象）；None 直接回退启发式。
        store: InstinctStore（LLM 条目与回退条目都写这里）。
        scope: 入库 scope（LLM 提取的通用习惯默认 global）。

    Returns:
        实际写入 store 的 Instinct 列表（LLM 成功条目 / 回退启发式条目）。

    fail-open：任何失败回退启发式 observe_turn，永不抛。
    """
    global _consecutive_failures, _circuit_open, _circuit_opened_at, _session_call_count

    # 1. 熔断检查（冷却期内直接回退；冷却期满合闸半开重试）
    if _circuit_open:
        if _now() - _circuit_opened_at >= COOLDOWN_SECONDS:
            _circuit_open = False
            _consecutive_failures = 0  # 半开：重试从零计数
            logger.info("llm_observer 熔断冷却期满，合闸重试")
        else:
            return _heuristic_fallback(
                user_text=user_text, tool_calls=tool_calls,
                tool_results=tool_results, store=store, scope=scope,
            )

    # 2. 会话上限 / 无 router：不调 LLM 直接回退
    if _session_call_count >= MAX_CALLS_PER_SESSION:
        return _heuristic_fallback(
            user_text=user_text, tool_calls=tool_calls,
            tool_results=tool_results, store=store, scope=scope,
        )
    if aux_llm_router is None:
        return _heuristic_fallback(
            user_text=user_text, tool_calls=tool_calls,
            tool_results=tool_results, store=store, scope=scope,
        )

    _session_call_count += 1
    try:
        response = await aux_llm_router.chat_completions(
            [
                {"role": "system", "content": _OBSERVER_SYSTEM_PROMPT},
                {"role": "user", "content": _build_observer_prompt(
                    user_text, tool_calls, tool_results)},
            ],
            max_tokens=600,
            temperature=0.2,
        )
        items = _parse_instincts(_extract_content(response))
        # 成功（含空数组）→ 重置熔断计数
        _consecutive_failures = 0

        now_iso = datetime.now(timezone.utc).isoformat()
        evidence = [f"llm 观察：{_clip(user_text, 80)}"] if user_text else ["llm 观察"]
        result: List[Instinct] = []
        for it in items:
            inst = Instinct(
                trigger=it["trigger"],
                action=it["action"],
                confidence=it["confidence"],
                evidence=evidence,
                scope=scope,
                updated_at=now_iso,
            )
            store.upsert(inst)
            result.append(inst)
        return result
    except Exception as e:
        # 失败（LLM 异常 / 解析失败）→ 累计熔断 + 回退启发式
        _consecutive_failures += 1
        if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            _circuit_open = True
            _circuit_opened_at = _now()
            logger.warning(
                "llm_observer 熔断开启（连续 %d 次失败），冷却 %.0fs",
                _consecutive_failures, COOLDOWN_SECONDS,
            )
        logger.debug("llm_observer 失败回退启发式: %s", e)
        return _heuristic_fallback(
            user_text=user_text, tool_calls=tool_calls,
            tool_results=tool_results, store=store, scope=scope,
        )
