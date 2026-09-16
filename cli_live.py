"""live 区状态仓——claude code 底部「正在发生什么」告示牌。

大白话：终端最底下（spinner 那一带）有一块小告示牌，分三层叠着：

1. spinner 行：``✶ Cooking… (3m 11s · ↓ 17.6k tokens)``（随机英文动词）
2. 子代理树：批量委托时每个子代理一行（├─/└─），带工具计数和当前活动
3. 任务清单：□ 待办 / ■ 进行中 / √ 完成（task 工具的实时黑板）

数据流：**任何线程**都能往这儿塞事件（事件行钩子/委托批量/子代理工具
钩子都跑在工作线程），渲染闭包（cli_layout 的 live 面板）被 spinner
线程每 0.1s 拍一拍读一次快照。所以全部接口都加锁、全部 fail-open
——纯视觉，任何异常都不许挡业务。

为什么单独一个文件：cli_layout（渲染）和 cli_events/delegate（生产端）
互相不该 import 对方，这里当个中立的小黑板。
"""

import logging
import random
import re
import threading
import time

logger = logging.getLogger(__name__)


def _oneline(s, n: int) -> str:
    """压成单行再截断：换行/连续空白全折叠成一个空格。

    为什么必须：live 面板一行就是一行（高度按行数算）——任务描述里
    带个真实换行符（LLM 的 goal 经常多行），一行就会裂成多行，面板
    高度对不上、整块布局撕碎。
    """
    try:
        s = re.sub(r"\s+", " ", str(s or "")).strip()
    except Exception:
        s = str(s or "")
    return s if len(s) <= n else s[:n] + "…"

# ---------------------------------------------------------------------------
# 动词表（claude code 同款：spinner 一会儿 Cooking 一会儿 Brewing，
# 回合收尾行用过去式 Cooked for 45s）
# ---------------------------------------------------------------------------

# (进行中, 过去式) 成对出现——收尾行要能从进行式翻回过去式
_VERBS = [
    ("Cooking", "Cooked"),
    ("Brewing", "Brewed"),
    ("Baking", "Baked"),
    ("Simmering", "Simmered"),
    ("Sautéing", "Sautéed"),
    ("Wibbling", "Wibbled"),
    ("Vibing", "Vibed"),
    ("Pondering", "Pondered"),
    ("Noodling", "Noodled"),
    ("Canoodling", "Canoodled"),
    ("Channelling", "Channelled"),
    ("Conjuring", "Conjured"),
    ("Marinating", "Marinated"),
    ("Percolating", "Percolated"),
    ("Herding cats", "Herded cats"),
]

# spinner 前面的装饰字符（claude code 不用转圈动画，用 ✶ ✻ ✢ 之类的
# 小星星慢速轮换——这里按帧号分桶轮换，0.8s 换一个，不闪眼）
_SPIN_MARKS = ["✶", "✻", "✢", "·", "*", "✽"]

# 动词多久换一次（秒）——一直 Cooking 太单调，换太勤又看不清
_VERB_INTERVAL = 25.0

# live 面板（不含 spinner 行）最多画几行——告示牌不是历史区，超了截尾
_PANEL_MAX_LINES = 14

_lock = threading.RLock()

# 子代理黑板：key → {"desc","status","tools","activity"}
# status ∈ {"running","done","failed","cancelled"}
_agents: dict = {}
_agents_order: list = []

# 任务清单快照：[{subject, status}]（task 工具事件来了就整表重拉）
_tasks: list = []
# 本回合任务清单动没动过（动过才在回合收尾落一段静态快照）
_tasks_touched = False

# 面板开关（ctrl+t 切换隐藏/显示）
_panel_hidden = False

# 回合状态：动词 + token 基线（spinner 的 ↓ 段要算「本回合新增」）
_verb = _VERBS[0]
_verb_at = 0.0
_token_base = None


def _pick_verb() -> None:
    """换一个动词（记住换的时刻，_VERB_INTERVAL 内不重复换）。"""
    global _verb, _verb_at
    _verb = random.choice(_VERBS)
    _verb_at = time.monotonic()


def pick_finish_verb() -> str:
    """随机拿一个过去式动词（回合收尾行 `✻ Cooked for 45s` 用）。"""
    return random.choice(_VERBS)[1]


# ---------------------------------------------------------------------------
# 格式化小工具（纯函数）
# ---------------------------------------------------------------------------

def fmt_elapsed(seconds) -> str:
    """秒 → claude code 风格计时文案：45s / 3m 11s。"""
    s = max(0, int(seconds or 0))
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60}s"


def fmt_tokens(n) -> str:
    """token 数 → 17.6k 风格（claude code 的 ↓ 段同款）。"""
    try:
        n = int(n or 0)
    except Exception:
        return "0"
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k"


def _agent_usage_total(agent) -> int:
    """从 agent 的用量统计里抠累计 token 数（拿不到返回 0，绝不炸）。"""
    try:
        stats = getattr(agent, "_llm_usage_stats", None)
        if isinstance(stats, dict):
            return (
                int(stats.get("total_prompt_tokens", 0) or 0)
                + int(stats.get("total_completion_tokens", 0) or 0)
            )
        return int(getattr(agent, "session_total_tokens", 0) or 0)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# spinner 行
# ---------------------------------------------------------------------------

def turn_started(agent) -> None:
    """回合开始：换动词 + 记 token 基线 + 清任务触板标记。"""
    global _token_base
    try:
        with _lock:
            _pick_verb()
            _token_base = _agent_usage_total(agent)
            _mark_tasks_untouched()
    except Exception:
        pass


def turn_ended() -> None:
    """回合结束：把没收场的子代理落成静态行，再清黑板。

    旧版直接 clear——面板上的子代理树无声蒸发，中断后屏上什么都不剩
    （用户视角：智能体调用、任务"都没有了"）。正常收场的条目不落
    （POST 的 finished 静态块已经打过了，落了就重复）；只有还挂着
    running 的条目（中断/被吞的收尾）落一行「已中断」留痕，对齐
    claude code 中断后仍能看到每个子代理终态的行为。
    """
    try:
        with _lock:
            snap = [(k, dict(_agents[k]))
                    for k in _agents_order if k in _agents]
        lines = []
        for _, e in snap:
            if e.get("status") != "running":
                continue
            tools = int(e.get("tools", 0) or 0)
            lines.append((
                "dim",
                f"  ⎿  {e.get('desc', '')} · {tools} tool uses · 已中断",
            ))
        if lines:
            try:
                from cli_events import print_style_lines
                print_style_lines(lines)
            except Exception:
                pass
        with _lock:
            _agents.clear()
            _agents_order.clear()
    except Exception:
        pass


def dump_panel_snapshot() -> None:
    """强退前的面板遗照：未收场子代理行 + 任务清单静态块，尽力落屏。

    强退路径（双击 Ctrl+C → os._exit）不走 _execute_turn 收尾——
    任务静态块永远没机会打。这里在进程消失前把面板内容落成滚动区
    静态行，用户回头还能看到当时有哪些子代理、任务进展到哪。
    """
    turn_ended()
    try:
        from cli_events import format_tasks_static_block, print_style_lines
        lines = format_tasks_static_block()
        if lines:
            print_style_lines(lines)
    except Exception:
        pass


def _mark_tasks_untouched():
    global _tasks_touched
    _tasks_touched = False


def spinner_text(frame: int, rt, turn_started_at, now: float) -> str:
    """spinner 行文案（纯函数）：空闲返回空串。

    长相（claude code 同款）：

        ✶ Cooking… (3m 11s · ↓ 17.6k tokens)

    frame 是 cli_layout 的 spinner 线程递增的帧号——装饰字符按帧号
    分桶慢速轮换（0.8s 一个），不像盲文那样疯狂转圈。
    动词到期自动换（_VERB_INTERVAL），一个回合里会轮着来。
    """
    try:
        if not getattr(rt, "turn_active", False):
            return ""
        elapsed = (now - turn_started_at) if turn_started_at else 0.0
        # 装饰字符：每 2 帧（0.2s）换一格——转得利落才有「在动」的感觉，
        # 0.8s 一格慢得像卡死
        mark = _SPIN_MARKS[(frame // 2) % len(_SPIN_MARKS)]
        # 动词到期换新（写全局有锁就锁，锁失败也无所谓——纯视觉）
        global _verb, _verb_at
        try:
            with _lock:
                if now - _verb_at > _VERB_INTERVAL:
                    _pick_verb()
                v = _verb[0]
        except Exception:
            v = _verb[0]
        # token 增量：当前累计 - 回合基线（基线没记就显示累计）
        try:
            agent = getattr(rt, "agent", None)
            total = _agent_usage_total(agent)
            with _lock:
                base = _token_base
            delta = total - base if base is not None else total
            tok = f" · ↓ {fmt_tokens(delta)} tokens"
        except Exception:
            tok = ""
        return f"{mark} {v}… ({fmt_elapsed(elapsed)}{tok})"
    except Exception:
        return ""   # 纯视觉，任何异常都当「不显示」


# ---------------------------------------------------------------------------
# 子代理黑板
# ---------------------------------------------------------------------------

def agents_begin(pairs: list) -> None:
    """批量进场：重置黑板，塞入 [(key, desc), ...]（顺序即显示顺序）。"""
    try:
        with _lock:
            _agents.clear()
            _agents_order.clear()
            for key, desc in pairs:
                _agents[str(key)] = {
                    "desc": _oneline(desc, 48), "status": "running",
                    "tools": 0, "activity": "",
                }
                _agents_order.append(str(key))
    except Exception:
        pass


def agent_begin(key, desc) -> None:
    """单个进场（同步子代理）：追加一条，不重置黑板。"""
    import time as _t
    try:
        with _lock:
            key = str(key)
            if key not in _agents:
                _agents_order.append(key)
            _agents[key] = {
                "desc": _oneline(desc, 48), "status": "running",
                "tools": 0, "activity": "",
                "started_at": _t.monotonic(),  # 面板显示已耗时用
            }
    except Exception:
        pass


def agent_update(key, *, status=None, activity=None) -> None:
    """更新某个子代理的状态/当前活动（fail-open）。"""
    try:
        with _lock:
            entry = _agents.get(str(key))
            if entry is None:
                return
            if status is not None:
                entry["status"] = str(status)
            if activity is not None:
                entry["activity"] = str(activity)
    except Exception:
        pass


def note_child_tool(key, activity: str) -> None:
    """子代理每调一次工具上报一次：计数 +1、活动行更新。

    activity 形如 ``read_file(D:/x.py)``（调用方拼好）。

    只更新 live 面板黑板（每个子代理固定一行的「当前活动」，spinner
    0.1s 一拍原地刷新）——**不往对话流打字**。旧版每次工具调用都往
    滚动历史刷一行 ⎿，3 个子代理跑几分钟就是几十行刷屏（对齐
    claude code：运行过程看面板一行，细节等收尾块）。
    """
    try:
        with _lock:
            entry = _agents.get(str(key))
            if entry is None:
                return
            entry["tools"] = int(entry.get("tools", 0)) + 1
            entry["activity"] = _oneline(activity, 120)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 运行中工具黑板：PRE 入栈、POST 出栈，live 面板顶部画动画行
# （● 闪烁 = 字符 ●◐◑○ 随时间轮换，0.1s 一换——spinner 线程本来就
#  在这个节拍上重绘，白蹭）
# ---------------------------------------------------------------------------

_running_tools: list = []          # 受 _lock 保护：["Name(摘要)", ...]
_SPIN_FRAMES = "●◐◑○"


def running_tool_start(label: str) -> None:
    """一个工具开始跑：入栈（label 由 cli_events 拼好「Name(摘要)」）。"""
    try:
        with _lock:
            _running_tools.append(str(label or "?"))
    except Exception:
        pass


def running_tool_end(label: str) -> None:
    """工具跑完：出栈（先精确匹配，兜底按前缀删第一个同名）。"""
    try:
        with _lock:
            name = str(label or "?")
            if name in _running_tools:
                _running_tools.remove(name)
                return
            base = name.split("(")[0]
            for i, x in enumerate(_running_tools):
                if x == base or x.split("(")[0] == base:
                    del _running_tools[i]
                    return
    except Exception:
        pass


def running_tool_lines(width=80) -> list:
    """运行中工具的动画行（panel_lines 顶部用）。

    帧号从墙上时钟推（monotonic×10 取模）——不存状态，重绘到哪算哪；
    spinner 线程 0.1s 一拍刷帧，● 就「闪」起来了。
    """
    try:
        with _lock:
            items = list(_running_tools)
        if not items:
            return []
        import time as _t
        frame = _SPIN_FRAMES[int(_t.monotonic() * 10) % len(_SPIN_FRAMES)]
        w = max(20, width or 80)
        return [("class:live-dim", f"  {frame} {x}"[:w]) for x in items[:4]]
    except Exception:
        return []


def agent_finish(key, status="done") -> None:
    """某个子代理收场（done/failed/cancelled）。"""
    agent_update(key, status=status, activity="")


def agents_snapshot() -> list:
    """黑板快照：[(key, entry拷贝), ...] 按进场顺序（渲染/静态块用）。"""
    try:
        with _lock:
            return [(k, dict(_agents[k])) for k in _agents_order if k in _agents]
    except Exception:
        return []


def agent_tools_count(key) -> int:
    """某个子代理的工具调用次数（静态收尾块用，查不到返回 0）。"""
    try:
        with _lock:
            return int(_agents.get(str(key), {}).get("tools", 0))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# 任务清单黑板
# ---------------------------------------------------------------------------

# 任务面板的会话过滤域：None=不过滤（测试/无 CLI）；否则只显示归属该
# 会话的任务。任务库是全局单仓库——不过滤的话别的会话/项目的残留任务
# 会串进本会话的面板和恢复注入（用户视角"冒出两份任务清单"）。
_task_scope: "str | None" = None


def set_task_scope(session_id) -> None:
    """设置任务面板的会话过滤域（建会话/换会话/恢复会话时调）。"""
    global _task_scope
    try:
        with _lock:
            _task_scope = str(session_id) if session_id else None
    except Exception:
        pass


def refresh_tasks() -> None:
    """从 task_store 重拉任务快照（task 工具事件来了就调一次）。

    只留 pending/in_progress/completed/blocked 四类（deleted 剔除），
    顺序按 store 现有顺序；设了会话过滤域（set_task_scope）就只拉
    归属该会话的任务。拉不到就保持原样（fail-open）。
    """
    global _tasks, _tasks_touched
    try:
        from agent.task_store import get_task_store
        store = get_task_store(None)
        with _lock:
            scope = _task_scope
        rows = [
            {"subject": t.get("subject", ""), "status": t.get("status", "")}
            for t in store.list_all()
            if t.get("status") not in ("deleted",)
            and (scope is None or t.get("session_id") == scope)
        ]
        with _lock:
            _tasks = rows
            _tasks_touched = True
    except Exception:
        pass


def tasks_snapshot() -> list:
    """任务清单快照拷贝（渲染/静态块用）。"""
    try:
        with _lock:
            return [dict(t) for t in _tasks]
    except Exception:
        return []


def take_tasks_touched() -> bool:
    """读本回合任务清单动没动过（读后不清——收尾块自己决定）。"""
    try:
        with _lock:
            return _tasks_touched
    except Exception:
        return False


def tasks_summary() -> str:
    """`5 tasks (1 done, 4 open)` 汇总行（空表返回空串）。"""
    try:
        with _lock:
            rows = list(_tasks)
        if not rows:
            return ""
        done = sum(1 for t in rows if t.get("status") == "completed")
        inprog = sum(1 for t in rows if t.get("status") == "in_progress")
        open_ = len(rows) - done
        if inprog:
            return (f"{len(rows)} tasks "
                    f"({done} done, {inprog} in progress, {open_ - inprog} open)")
        return f"{len(rows)} tasks ({done} done, {open_} open)"
    except Exception:
        return ""


_TASK_MARKS = {"completed": "√", "in_progress": "■", "blocked": "⊘"}


def tasks_lines(width=80) -> list:
    """任务清单行：□/■/√ 符号 + 标题（渲染和静态快照共用一套长相）。

    返回 [(style, text), ...]；style ∈ {"", "dim"}。
    """
    try:
        with _lock:
            rows = [dict(t) for t in _tasks]
        out = []
        limit = max(20, (width or 80) - 8)
        for t in rows:
            mark = _TASK_MARKS.get(t.get("status", ""), "□")
            subject = _oneline(t.get("subject", ""), limit)
            style = "dim" if t.get("status") == "completed" else ""
            out.append((style, f"  {mark} {subject}"))
        return out
    except Exception:
        return []


# ---------------------------------------------------------------------------
# live 面板渲染（spinner 行之下、分隔线之上的整块内容）
# ---------------------------------------------------------------------------

def toggle_panel() -> bool:
    """ctrl+t：切换面板隐藏/显示，返回切换后的隐藏状态。"""
    global _panel_hidden
    try:
        with _lock:
            _panel_hidden = not _panel_hidden
            return _panel_hidden
    except Exception:
        return _panel_hidden


def panel_hidden() -> bool:
    try:
        with _lock:
            return _panel_hidden
    except Exception:
        return False


def panel_lines(width=80) -> list:
    """live 面板内容（不含 spinner 行）：子代理树 + 任务清单。

    返回 [(style, text), ...]，style 是 cli_layout 样式表里的类名
    （"" 默认 / "class:live-dim" 暗色）。空列表 = 没东西可画。
    面板被 ctrl+t 藏起来时返回空。
    """
    try:
        with _lock:
            hidden = _panel_hidden
        if hidden:
            return []
        w = max(20, width or 80)   # 行宽上限：超了截断，防终端软换行撕面板
        out = []
        # ---- 运行中工具（● 闪烁动画行，工具跑完 POST 出栈自动消失）----
        out.extend(running_tool_lines(width))
        # ---- 子代理树（running 的行带已耗时+活动行实时更新）----
        snap = agents_snapshot()
        if snap:
            import time as _t_now
            for i, (_, e) in enumerate(snap):
                last = i == len(snap) - 1
                branch = "└─" if last else "├─"
                running = e["status"] == "running"
                # 已耗时（running 才显示，spinner 线程 0.1s 重绘=秒表在走）
                _elapsed = ""
                if running and e.get("started_at"):
                    _sec = int(_t_now.monotonic() - e["started_at"])
                    if _sec >= 60:
                        _elapsed = f" ({_sec // 60}m{_sec % 60:02d}s)"
                    elif _sec >= 3:
                        _elapsed = f" ({_sec}s)"
                status_bit = {
                    "done": " · Done",
                    "failed": " · ✗",
                    "cancelled": " · cancelled",
                }.get(e["status"], f" · {e['tools']} tool uses")
                out.append(("class:live-dim",
                            f"   {branch} {e['desc']}{status_bit}{_elapsed}"[:w]))
                activity = e["activity"]
                if running and activity:
                    pipe = "   " if last else "│  "
                    out.append(("class:live-dim",
                                f"   {pipe} ⎿  {activity}"[:w]))
        # ---- 任务清单 ----
        rows = tasks_lines(width)
        if rows:
            out.append(("class:live-dim", "  ⎿"))
            out.extend(rows)
        # 截尾：告示牌不许比屏幕高
        if len(out) > _PANEL_MAX_LINES:
            out = out[:_PANEL_MAX_LINES]
            out.append(("class:live-dim", "  ⎿ …"))
        return out
    except Exception:
        return []


def panel_height(width=80) -> int:
    """live 面板行数（cli_layout 的高度回调读它；0 = 整块隐藏）。"""
    try:
        return len(panel_lines(width))
    except Exception:
        return 0
