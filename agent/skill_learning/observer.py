# -*- coding: utf-8 -*-
"""启发式观察器：用四类固定规则从对话轨迹里「顺手」提炼行为记忆。

每轮对话结束后扫一遍这轮发生了什么（用户原话 + 工具调用 + 工具结果），
按四类固定模式找值得记住的 instinct（「遇到什么情况 → 该怎么做」的习惯）：

1. **用户纠错**（初始置信度 0.5，存全局）——「不要用 X，改用 Y」这类话，
   记成「使用 X 时 → 改用 Y」。这是用户最直接的习惯声明。
2. **失败恢复**（0.4，全局）——同一个工具先报错、紧接着又成功了，
   说明 agent 自己摸出了绕过办法，值得记成「X 失败时 → 重试/换参」。
3. **重复三连招**（0.3，全局）——单轮内「读→搜→改」这种固定的三步
   连招出现 2 次以上，说明存在稳定的操作节奏，记下这个套路。
4. **项目约定**（0.45，存项目隔离区）——「必须/一律/always」声明的
   规矩，只属于当前项目（复用 get_project_memory_key 算项目
   归属），别的项目不通用。

设计约定：
* **fail-open（坏了也不能碍事）**：整体 try/except 永不抛——观察器挂了
  不能影响主对话；单个信号检测失败也不连坐其他信号。
* evidence（证据）：每条截 80 字的原文片段（用户原话/错误信息/序列描述）。
* 每类信号单轮最多记 3 条（防一条长消息刷屏灌库）。
* scope（作用域）参数：信号 1/2/3 永远存全局；信号 4 的项目归属在内部
  算（传入的 scope 是当前项目 key 时直接用，否则现场算）。
"""
import re
from datetime import datetime, timezone
from typing import List

from agent.project_scope import get_project_memory_key
from agent.skill_learning.store import Instinct

# ---------------------------------------------------------------------------
# 常量：四类信号的初始置信度 + 单轮条数/证据长度上限
# ---------------------------------------------------------------------------
_CONF_CORRECTION = 0.5
_CONF_RECOVERY = 0.4
_CONF_SEQUENCE = 0.3
_CONF_CONVENTION = 0.45

_MAX_PER_SIGNAL = 3
_EVIDENCE_CLIP = 80

# 信号 1 的正则（用户纠错，中英文都认）：
# 「不要/别/don't/stop」+「用/使用/use/using」+ X ……（中间 ≤20 字）……「改用/换成/用/use」+ Y
# 两个历史调参点（踩坑换来的，别改回去）：
# 1. 空白从 `\s+` 放宽为 `\s*`——中文「不要用grep」中间没有空格，要求空格会漏；
# 2. 英文动词候选按「长的在前」排——"using" 不排在 "use" 前面的话，
#    匹配到 "use" 会把剩下的 "ing" 当成工具名。
_CORRECTION_RE = re.compile(
    r"(?:不要|别|don'?t|stop)\s*(?:使用|using|用|use)\s*([^\s，,。.;；的]+)"
    r".{0,20}?(?:改用|换用|换成|用|use)\s*([^\s，,。.;；]+)",
    re.IGNORECASE,
)

# 信号 4 的正则（项目约定）。范围限定为「不出句」：遇到句号/问号/换行就停——
# 否则会把下一句话里不相干的约定也拼进同一条记忆，内容串味。
_CONVENTION_RE = re.compile(r"(必须|一律|always|convention)[^。！？!?\n]{5,80}", re.IGNORECASE)


def _clip(text: str, limit: int = _EVIDENCE_CLIP) -> str:
    """把文本截到 limit 个字（证据条目和行为描述统一用这个长度口径）。

    参数：
        text：原始文本（None 也收，当空串处理）。
        limit：最长保留字符数，默认 80。

    返回：去掉首尾空白后截断的文本。
    """
    text = str(text or "").strip()
    return text[:limit]


def _emit(store, *, trigger: str, action: str, confidence: float,
          evidence: str, scope: str) -> None:
    """把一条习惯打包成 Instinct 写进存储。

    参数（全是 keyword-only）：
        store：InstinctStore（存储对象）。
        trigger：触发情境（如「使用 grep」）。
        action：建议行为（如「改用 ripgrep」）。
        confidence：本次观察的初始置信度。
        evidence：证据原文（内部会截 80 字）。
        scope：作用域（"global" 或 "project:<项目key>"）。

    说明：重复观察时的置信度累积和去重不用这里操心，InstinctStore.upsert
    会做——这里只负责打包写入。
    """
    store.upsert(Instinct(
        trigger=trigger,
        action=action,
        confidence=confidence,
        evidence=[_clip(evidence)],
        scope=scope,
        updated_at=datetime.now(timezone.utc).isoformat(),
    ))


def _project_scope(scope: str) -> str:
    """算信号 4（项目约定）该存到哪个项目作用域。

    参数：
        scope：调用方传进来的 scope 字符串。

    返回："project:<项目key>"。规则：传进来的已经是 "project:xxx" 完整
    形态就直接用；传 "global"/空/别的 key 则现场算当前项目 key。
    get_project_memory_key 本身就 fail-open（不是 git 仓库就退回用 cwd），
    这里万一它还炸了就再兜一层用 "default"。
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
# 四类信号检测器（每个返回本信号新写入的习惯条数）
# ---------------------------------------------------------------------------

def _detect_user_correction(user_text: str, store) -> int:
    """信号 1：用户纠错。从用户原话里找「不要用 X，改用 Y」。

    参数：
        user_text：本轮用户消息原文。
        store：InstinctStore，找到就写进去。

    返回：新写入的条数（同一对 X→Y 去重，单轮最多 3 条）。
    """
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
    """对比失败那次和成功那次的参数差异，浓缩成一句话。

    参数：
        old_args：失败调用的参数字典。
        new_args：成功调用的参数字典。
        tool_name：工具名（两边参数没差时用于「重试 xxx」的说法）。

    返回：如「重试/换参（path: a.py→b.py）」；参数完全一样时返回
    「重试 xxx（参数未变）」。
    """
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
    """信号 2：失败恢复。某工具报错后同一个工具紧接着成功了。

    背景：先错后对说明 agent 自己摸出了绕过办法（改了参数或重试），
    这种经验值得记下来下次直接用。

    参数：
        tool_calls：本轮工具调用列表，[{"name", "arguments"}]。
        tool_results：本轮工具结果列表，与 tool_calls 按下标一一对应。
        store：InstinctStore。

    返回：新写入的条数。同一个工具多次恢复只记第一次；单轮最多 3 条。
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
    """信号 3：重复三连招。单轮内同一组三步序列（a→b→c）出现 2 次以上。

    例子：一轮里「read→search→edit」这个组合出现两次，说明 agent 干这类
    活有固定套路。触发词取序列的终点工具（read→search→edit 最终是为了
    「编辑」），记成「需要 edit 时 → 按 read→search→edit 的顺序来」。

    参数：
        tool_calls：本轮工具调用列表（只看工具名）。
        store：InstinctStore。

    返回：新写入的条数。序列去重后按出现次数从多到少取前 3 条。
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
    """信号 4：项目约定。「必须/一律/always ...」的规矩记进当前项目区。

    参数：
        user_text：本轮用户消息原文。
        store：InstinctStore。
        scope：作用域（用于算项目归属，见 _project_scope）。

    返回：新写入的条数。原话截 80 字；单轮最多 3 条。
    """
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
# 公开入口（主循环每轮结束后调这个）
# ---------------------------------------------------------------------------

def observe_turn(*, user_text: str, tool_calls, tool_results, store,
                 scope: str = "global") -> int:
    """扫一轮对话轨迹，把四类信号找到的习惯写进存储。

    参数（全是 keyword-only）：
        user_text：本轮用户消息原文（可为空）。
        tool_calls：本轮工具调用列表 [{"name", "arguments"}]，
            与 tool_results 按下标一一对应。
        tool_results：本轮工具结果列表 [{"name", "error"(可选), "content"}]。
        store：InstinctStore（或任何带 upsert(inst) 方法的对象）。
        scope：当前项目 key——只有信号 4 用它算项目归属，
            信号 1/2/3 永远存全局。

    返回：本轮实际写入（upsert）的习惯条数。

    fail-open：某个信号检测抛异常只跳过它，整体永不抛——观察器是旁路，
    绝不能把异常带进主对话。
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
            continue  # 某个信号炸了只跳过它——观察器是旁路，不能把异常带进主对话
    return total
