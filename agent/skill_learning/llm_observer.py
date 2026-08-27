# -*- coding: utf-8 -*-
"""LLM 观察后端：用辅助 LLM（aux_llm）做行为观察。

启发式观察器（observer.py）只会认四类写死的正则模式；这个后端更聪明——
把整轮轨迹（用户消息 + 工具调用 + 工具结果）整理好交给一个便宜的辅助
LLM 去提炼习惯，要求它只回 JSON：

    "从观察提取最多 3 条原子习惯，纯 JSON 数组
     [{trigger, action, confidence}]，没有返 []，不猜"

韧性设计（写法对齐 context_compressor 的熔断器，都是模块级状态）：
* **熔断**：连续失败 3 次就「拉闸」；拉闸期间不再调 LLM，直接回退启发式
* **冷却**：拉闸 30 秒后自动「合闸」重试（合闸时清零失败计数——半开语义，
  给它一次重新证明自己的机会）
* **会话上限**：每个会话最多调 LLM 20 次（防观察器这个小角色吃掉太多
  配额），超限回退启发式
* **回退**：任何失败（没有 router / LLM 报错 / 解析不出 JSON）→ 回退
  启发式 observe_turn；返回值统一是「实际入库的 Instinct 列表」（回退
  路径用一个 recorder 包装 store，透传写入并顺手记账，不搞两套语义）

注意：**空数组 [] 算成功不算失败**（这轮确实没什么可记的），会重置熔断
计数——只有调不动 / 解析不出才算失败。
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
# 熔断 / 限流的模块级状态（写法对齐 context_compressor——同样是全局可重置）
# ---------------------------------------------------------------------------
_consecutive_failures = 0
_circuit_open = False
_circuit_opened_at = 0.0
_session_call_count = 0

MAX_CONSECUTIVE_FAILURES = 3      # 连续失败这么多次就拉闸
COOLDOWN_SECONDS = 30.0           # 拉闸后等这么久才允许重试
MAX_CALLS_PER_SESSION = 20        # 每个会话最多调 LLM 这么多次
MAX_INSTINCTS_PER_TURN = 3        # 单轮最多提炼几条习惯（prompt 里也是这么要求的）

# LLM 给的条目缺 confidence 或值非法时用的默认值——单次观察没经过
# 重复验证，取偏保守的 0.4
_DEFAULT_CONFIDENCE = 0.4

# 时钟函数做成变量：测试时可以换成假的。用 monotonic 而不是系统时间，
# 是为了不受用户改系统时间/时间回拨的影响
_now = time.monotonic

_OBSERVER_SYSTEM_PROMPT = "你是行为观察助手。只输出 JSON 数组，不要其他文字。"


def reset_llm_observer_state() -> None:
    """新会话开始时把熔断/限流状态归零（模块级状态会跨会话残留，不重置
    会把上个会话的「拉闸中」带进来）。

    调用时机：AIAgent.__init__ 里调一次（对齐 reset_compact_circuit_breaker
    的用法）。

    返回：无。
    """
    global _consecutive_failures, _circuit_open, _circuit_opened_at, _session_call_count
    _consecutive_failures = 0
    _circuit_open = False
    _circuit_opened_at = 0.0
    _session_call_count = 0


class _RecordingStore:
    """回退启发式时用的「记账」包装：写入照常透传给真存储，同时把写了
    什么记在 written 列表里——这样回退路径也能统一返回「实际入库的条目」。

    参数（__init__）：
        inner：真正的 InstinctStore。
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.written: List[Instinct] = []

    def upsert(self, inst: Instinct) -> Instinct:
        self.written.append(inst)
        return self._inner.upsert(inst)


def _strip_code_fence(text: str) -> str:
    """剥掉 LLM 回答外面可能裹的 ```json ... ``` 代码围栏。

    背景：你让它「只输出 JSON」，它经常还是习惯性套一层 Markdown 代码块，
    直接 json.loads 会炸，所以先剥掉。

    参数：
        text：LLM 的原始回答。

    返回：剥掉围栏后的文本（本来就没有围栏就原样返回）。
    """
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
    """把这一轮发生的事浓缩成给 LLM 看的观察材料。

    为什么每段都要截断：工具结果动辄几千字，整段塞进去 token 会爆——
    观察只需要「大概发生了什么」，不需要全文。

    参数：
        user_text：本轮用户消息原文。
        tool_calls：本轮工具调用列表（参数会 JSON 化后截 200 字）。
        tool_results：本轮工具结果列表（每条截 150 字）。

    返回：拼好的完整 prompt 文本（末尾带输出格式要求）。
    """
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
    """从 LLM 的响应对象里把文本抠出来。

    背景：aux router 有时返回 SDK 的对象（属性访问）、有时返回字典
    （键访问），两种都得认。

    参数：
        response：LLM 响应（对象或 dict）。

    返回：回答文本；取不到就空串。
    """
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
    """把 LLM 的输出解析成合法的习惯条目列表。

    参数：
        raw：LLM 的原始回答文本。

    返回：[{trigger, action, confidence}]，最多 3 条（多给的截掉）。

    失败判定：不是 JSON / 不是数组 / 数组里一条合法的都没有 → 抛
    ValueError（调用方会计入熔断并回退启发式）。空数组 [] 是合法结果，
    正常返回（这轮确实没什么可记的，不算失败）。
    单条缺 trigger 或 action 的直接丢（不猜）；confidence 非法回退默认值。
    """
    text = _strip_code_fence(raw)
    data = json.loads(text)  # JSON 坏了就让异常向上抛——要计入熔断，不能吞
    if not isinstance(data, list):
        raise ValueError(f"LLM 输出不是 JSON 数组: {type(data).__name__}")
    items = []
    for it in data[:MAX_INSTINCTS_PER_TURN]:
        if not isinstance(it, dict):
            continue
        trigger = str(it.get("trigger") or "").strip()
        action = str(it.get("action") or "").strip()
        if not trigger or not action:
            continue  # 关键字段缺的条目直接丢——宁缺毋滥，不猜
        try:
            conf = float(it.get("confidence"))
            # 历史踩坑：NaN 不能直接走 min/max 收敛——
            # min(1.0, nan) 会返回 1.0（nan 参与比较是 False，方向反了），
            # 把最不可信的值洗成满分。所以 NaN/Infinity 先拦下，落保守默认值。
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
    """LLM 路子走不通时，退回用启发式观察器干同样的活。

    参数（全是 keyword-only）：
        user_text / tool_calls / tool_results：本轮轨迹，原样传给启发式。
        store：真正的 InstinctStore。
        scope：作用域。

    返回：启发式实际写入 store 的条目列表（用 _RecordingStore 记的账）。
    """
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
        # observe_turn 自己就 fail-open，按理走不到这里——再兜一层保险
        logger.debug("llm_observer 启发式回退异常（fail-open）: %s", e)
    return recorder.written


async def observe_turn_llm(*, user_text: str, tool_calls, tool_results,
                           aux_llm_router, store, scope: str = "global"
                           ) -> List[Instinct]:
    """LLM 观察主流程：整理轨迹 → 调辅助 LLM 提炼 → 解析 → 入库。

    参数（全是 keyword-only）：
        user_text：本轮用户消息原文（可为空）。
        tool_calls：本轮工具调用列表 [{"name", "arguments"}]，
            与 tool_results 按下标一一对应。
        tool_results：本轮工具结果列表 [{"name", "error"(可选), "content"}]。
        aux_llm_router：辅助 LLM 路由器（或任何有 async chat_completions
            方法的对象）；传 None 直接走启发式回退。
        store：InstinctStore（LLM 提炼的和回退启发式写的都进这里）。
        scope：入库作用域（LLM 提炼的通用习惯默认存全局）。

    返回：实际写入 store 的 Instinct 列表——LLM 成功就用 LLM 的条目，
    走了回退就是启发式的条目，两套路径一个口径。

    fail-open：任何失败都回退启发式 observe_turn，永不抛异常。
    """
    global _consecutive_failures, _circuit_open, _circuit_opened_at, _session_call_count

    # 第 1 关：熔断检查。拉闸中且还在冷却期 → 直接回退启发式；
    # 冷却期满 → 合闸并把失败计数清零（半开，给一次重试机会）
    if _circuit_open:
        if _now() - _circuit_opened_at >= COOLDOWN_SECONDS:
            _circuit_open = False
            _consecutive_failures = 0  # 半开语义：重新计数，不背着旧账
            logger.info("llm_observer 熔断冷却期满，合闸重试")
        else:
            return _heuristic_fallback(
                user_text=user_text, tool_calls=tool_calls,
                tool_results=tool_results, store=store, scope=scope,
            )

    # 第 2 关：会话调用次数超上限、或压根没有 router——不调 LLM，直接回退
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
        # 成功（含空数组——这轮确实没得记）→ 失败计数清零
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
        # 失败（LLM 报错或解析不出）→ 失败计数 +1，够数就拉闸，然后回退启发式
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
