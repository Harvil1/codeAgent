# -*- coding: utf-8 -*-
"""HeuristicObserver：四类启发信号观察器（CCAR15 Task 2，对标 CCB sessionObserver）。

每轮对话结束后扫一遍（user_text + tool_calls + tool_results），从轨迹里
"顺手"提炼 instinct 行为记忆。四类信号：

1. **用户纠错**（confidence=0.5，global）——"不要用 X，用 Y"式指令，
   trigger="使用 {X}" / action="改用 {Y}"。这是用户最直接的习惯声明。
2. **失败恢复**（0.4，global）——同一工具 error 后紧接着成功，说明 agent
   自己找到了绕过办法，值得记成 "{tool} 失败时 → 重试/换参（diff）"。
3. **重复 3-元组序列**（0.3，global）——单轮内 read→search→edit 这类固定
   三连出现 ≥2 次，说明存在稳定的操作节奏（目的词取序列终点工具）。
4. **项目约定**（0.45，project scope）——"必须/一律/always"声明的规矩，
   属于当前项目（复用 CCAR9 的 get_project_memory_key），跨项目不通用。

设计约定：
* **fail-open**：整体 try/except 永不抛——观察器挂了不能影响主对话流程；
  单个信号检测失败也不连坐其他信号。
* evidence：截 80 字/条的原文片段（用户原话 / 错误信息 / 序列描述）。
* 每类信号单轮最多记 3 条（防一条长消息刷屏灌库）。
* scope 参数语义：信号 1/2/3 恒为 "global"；信号 4 的项目 scope 内部算
  （scope 传入的是当前项目 key 时直接用，否则 get_project_memory_key）。
"""
import re
from datetime import datetime, timezone
from typing import List

from agent.project_scope import get_project_memory_key
from agent.skill_learning.store import Instinct

# ---------------------------------------------------------------------------
# 常量：confidence 值 + 每信号上限
# ---------------------------------------------------------------------------
_CONF_CORRECTION = 0.5
_CONF_RECOVERY = 0.4
_CONF_SEQUENCE = 0.3
_CONF_CONVENTION = 0.45

_MAX_PER_SIGNAL = 3
_EVIDENCE_CLIP = 80

# 信号 1：用户纠错（中英文）。
# "不要/别/don't/stop" + "用/使用/use/using" + X ... （≤20 字间隔）... "改用/换成/用/use" + Y
# 相比 brief 的原始正则把 `\s+` 放宽为 `\s*`（中文"不要用grep"中间没有空格），
# 动词 alternation 按"长在前"排序（"using" 先于 "use"，否则会吃剩 "ing" 当工具名）。
_CORRECTION_RE = re.compile(
    r"(?:不要|别|don'?t|stop)\s*(?:使用|using|用|use)\s*([^\s，,。.;；的]+)"
    r".{0,20}?(?:改用|换用|换成|用|use)\s*([^\s，,。.;；]+)",
    re.IGNORECASE,
)

# 信号 4：项目约定。brief 正则的 `.{5,80}` 细化为不出句（句号后是另一句
# 话的约定，拼进同一条 action 会串味）。
_CONVENTION_RE = re.compile(r"(必须|一律|always|convention)[^。！？!?\n]{5,80}", re.IGNORECASE)


def _clip(text: str, limit: int = _EVIDENCE_CLIP) -> str:
    """截断到 limit 字符（evidence 条目 / action 的统一口径）。"""
    text = str(text or "").strip()
    return text[:limit]


def _emit(store, *, trigger: str, action: str, confidence: float,
          evidence: str, scope: str) -> None:
    """构造 Instinct 并写入 store（置信度累积/去重由 InstinctStore 负责）。"""
    store.upsert(Instinct(
        trigger=trigger,
        action=action,
        confidence=confidence,
        evidence=[_clip(evidence)],
        scope=scope,
        updated_at=datetime.now(timezone.utc).isoformat(),
    ))


def _project_scope(scope: str) -> str:
    """信号 4 的项目 scope：传入项目 key 直接用，否则内部算当前项目 key。

    传入 "project:xxx"（完整 scope）也接受；"global"/空 回退内部计算。
    get_project_memory_key 本身 fail-open（非 git 退 cwd），这里再兜一层。
    """
    if scope and scope.startswith("project:"):
        return scope
    key = scope if scope and scope != "global" else ""
    if not key:
        try:
            key = get_project_memory_key()
        except Exception:
            key = "default"
    return f"project:{key}"


# ---------------------------------------------------------------------------
# 四类信号检测器（每个返回本信号新增的 instinct 数）
# ---------------------------------------------------------------------------

def _detect_user_correction(user_text: str, store) -> int:
    """信号 1：用户纠错。"不要用 X，用 Y" → 使用 X 时改用 Y。"""
    count = 0
    seen = set()
    for m in _CORRECTION_RE.finditer(user_text or ""):
        old, new = m.group(1), m.group(2)
        if (old, new) in seen:
            continue
        seen.add((old, new))
        _emit(store, trigger=f"使用 {old}", action=f"改用 {new}",
              confidence=_CONF_CORRECTION, evidence=m.group(0), scope="global")
        count += 1
        if count >= _MAX_PER_SIGNAL:
            break
    return count


def _param_diff(old_args, new_args, tool_name: str) -> str:
    """失败→成功两次调用的参数 diff，浓缩成一句话。"""
    if not isinstance(old_args, dict) or not isinstance(new_args, dict):
        return f"重试 {tool_name}"
    parts: List[str] = []
    for k in dict.fromkeys(list(old_args.keys()) + list(new_args.keys())):
        ov, nv = old_args.get(k), new_args.get(k)
        if ov != nv:
            parts.append(f"{k}: {ov}→{nv}")
    if not parts:
        return f"重试 {tool_name}（参数未变）"
    return f"重试/换参（{'；'.join(parts)}）"


def _detect_failure_recovery(tool_calls, tool_results, store) -> int:
    """信号 2：某工具 error 后同工具紧接着成功（agent 自己找到了恢复路径）。

    tool_calls 与 tool_results 按下标对齐；同名工具多次恢复只记第一次。
    """
    n = min(len(tool_calls or []), len(tool_results or []))
    recovered = set()
    count = 0
    for i in range(n):
        call_i = tool_calls[i] or {}
        res_i = tool_results[i] or {}
        if not res_i.get("error"):
            continue
        name = res_i.get("name") or call_i.get("name")
        if not name or name in recovered:
            continue
        for j in range(i + 1, n):
            call_j = tool_calls[j] or {}
            res_j = tool_results[j] or {}
            name_j = res_j.get("name") or call_j.get("name")
            if name_j == name and not res_j.get("error"):
                action = _param_diff(call_i.get("arguments"),
                                     call_j.get("arguments"), name)
                _emit(store, trigger=f"{name} 失败时", action=action,
                      confidence=_CONF_RECOVERY,
                      evidence=res_i.get("error"), scope="global")
                recovered.add(name)
                count += 1
                break
        if count >= _MAX_PER_SIGNAL:
            break
    return count


def _detect_repeated_sequence(tool_calls, store) -> int:
    """信号 3：单轮内相同 3-元组序列（a→b→c）出现 ≥2 次。

    目的词取序列终点工具（read→search→edit 服务的是"编辑"），trigger
    写成 "需要 {c}"。序列去重后按出现次数降序取前 3 条。
    """
    names = [(c or {}).get("name") or "?" for c in (tool_calls or [])]
    counts = {}
    for i in range(len(names) - 2):
        tri = (names[i], names[i + 1], names[i + 2])
        counts[tri] = counts.get(tri, 0) + 1
    repeated = sorted(
        ((t, c) for t, c in counts.items() if c >= 2),
        key=lambda kv: kv[1], reverse=True,
    )[:_MAX_PER_SIGNAL]
    for tri, c in repeated:
        seq = f"{tri[0]}→{tri[1]}→{tri[2]}"
        _emit(store, trigger=f"需要 {tri[2]}", action=f"序列 {seq}",
              confidence=_CONF_SEQUENCE,
              evidence=f"重复序列 {seq} ×{c}", scope="global")
    return len(repeated)


def _detect_project_convention(user_text: str, store, scope: str) -> int:
    """信号 4：项目约定。"必须/一律/always ..." 原句截 80 字记进项目区。"""
    count = 0
    seen = set()
    for m in _CONVENTION_RE.finditer(user_text or ""):
        action = _clip(m.group(0))
        if action in seen:
            continue
        seen.add(action)
        _emit(store, trigger="项目约定", action=action,
              confidence=_CONF_CONVENTION, evidence=m.group(0),
              scope=_project_scope(scope))
        count += 1
        if count >= _MAX_PER_SIGNAL:
            break
    return count


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

def observe_turn(*, user_text: str, tool_calls, tool_results, store,
                 scope: str = "global") -> int:
    """扫一轮对话轨迹，把四类启发信号写入 InstinctStore。

    Args:
        user_text: 本轮用户消息原文（可为空）。
        tool_calls: [{"name", "arguments"}]，与 tool_results 下标对齐。
        tool_results: [{"name", "error"(可选), "content"}]。
        store: InstinctStore（或任何带 upsert(inst) 的对象）。
        scope: 当前项目 key（信号 4 用；信号 1/2/3 恒为 "global"）。

    Returns:
        本轮新增（upsert）的 instinct 条数。

    fail-open：单个信号检测抛异常只跳过该信号，整体永不抛。
    """
    total = 0
    detectors = (
        lambda: _detect_user_correction(user_text, store),
        lambda: _detect_failure_recovery(tool_calls, tool_results, store),
        lambda: _detect_repeated_sequence(tool_calls, store),
        lambda: _detect_project_convention(user_text, store, scope),
    )
    for detect in detectors:
        try:
            total += detect()
        except Exception:
            continue  # 观察器是旁路，绝不能把异常带进主对话流程
    return total
