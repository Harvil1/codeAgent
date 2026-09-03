"""事件行渲染器——把工具/子代理/任务/提问的动作打成一行行紧凑事件。

设计一句话：**完成才打行，等待靠工具栏**。
- PRE（工具开始）：只记时间戳 + 维护 rt.event_pending（状态栏 ◐ 段/spinner
  行读它）；子代理出发是时刻事件，PRE 就打行
- POST（工具结束）：配对算耗时，打完成行（✓/✗ + 参数摘要 + 结果摘要）；
  失败也走 POST（结果含 error → ✗ 行）——不另挂 FAILURE 钩子，
  否则一次失败会打两行 ✗（model_tools 对含 error 的结果两个钩子都发）
- 全部 try/except 吞掉——纯视觉，绝不挡工具执行（fail-open）
"""

import json
import logging
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# 参数摘要表：工具名 → 从 args 里挑哪个字段当"这行在干嘛"的摘要
_ARG_FIELDS = {
    "read_file": "file_path",
    "write_file": "file_path",
    "str_replace": "file_path",
    "terminal": "command",
    "search_files": "pattern",
    "glob": "pattern",
    "web_fetch": "url",
}

# 工具行样式表：工具名 → (emoji, 动词)。格式：┊ emoji 动词 摘要 耗时
# （跟 hermes 的 get_cute_tool_message 学的——一行扫出「谁在干什么」）
_TOOL_STYLES = {
    "terminal": ("💻", "$"),
    "read_file": ("📖", "read"),
    "write_file": ("✍️", "write"),
    "str_replace": ("🔧", "patch"),
    "search_files": ("🔎", "grep"),
    "glob": ("🔎", "find"),
    "web_fetch": ("📄", "fetch"),
    "memory": ("🧠", "memory"),
    "subagent": ("🔀", "delegate"),
    "delegate_task": ("🔀", "delegate"),
    "task_create": ("📋", "plan"),
    "task_complete": ("📋", "plan"),
}


def _tool_prefix() -> str:
    """工具行前缀（皮肤可换，默认 ┊）。"""
    try:
        import cli_skin
        return cli_skin.get_active_skin().tool_prefix or "┊"
    except Exception:
        return "┊"


def _cut(s: str, n: int) -> str:
    """字符串截断：超长加省略号（事件行一行的信息量守恒）。"""
    s = str(s).strip()
    return s if len(s) <= n else s[:n] + "…"


def summarize_args(tool_name: str, args: dict) -> str:
    """按工具名挑最有信息量的参数字段当摘要；取不到返回空串。"""
    args = args or {}
    field = _ARG_FIELDS.get(tool_name)
    if field:
        v = args.get(field)
        limit = 40 if tool_name in ("terminal", "web_fetch") else 30
        return _cut(v, limit) if v else ""
    # 没登记的工具：拿第一个字符串型参数值顶上
    for v in args.values():
        if isinstance(v, str) and v:
            return _cut(v, 30)
    return ""


def format_duration(dt) -> str:
    """耗时格式化：太快就 <0.1s，否则 0.1s 精度。dt 为 None 返回空串。"""
    if dt is None:
        return ""
    return "<0.1s" if dt < 0.05 else f"{dt:.1f}s"


def result_preview(result_str: str):
    """从工具结果 JSON 里抠一行摘要。

    返回：(ok, 摘要)——result 含 error 键 → (False, 错误前 40 字)；
    成功按 preview → output → content → text 顺序找第一个非空键取前 30 字；
    都没有/解析失败 → (True, "")。
    """
    try:
        data = json.loads(result_str or "")
    except Exception:
        return True, ""
    if isinstance(data, dict):
        if data.get("error"):
            return False, _cut(data["error"], 40)
        for key in ("preview", "output", "content", "text"):
            v = data.get(key)
            if v:
                return True, _cut(v, 30)
    return True, ""


def format_tool_line(tool_name, args, dt, result_str) -> str:
    """通用工具完成行：┊ emoji 动词 摘要 耗时 ✓/✗（结果摘要）。

    跟 hermes 的工具行学的一行扫读格式；失败时 ✗ + 错误前缀。
    """
    ok, preview = result_preview(result_str)
    mark = "✓" if ok else "✗"
    emoji, verb = _TOOL_STYLES.get(tool_name, ("⚡", tool_name[:9]))
    prefix = _tool_prefix()
    s = summarize_args(tool_name, args)
    dur = format_duration(dt)
    head = f"{prefix} {emoji} {verb:<9}"
    if s:
        head += f" {s}"
    tail = f" {dur} {mark}".strip()
    line = f"{head}  {tail}" if dur else head
    if not ok and preview:
        return f"{line} {preview}"
    return line


def format_subagent_depart(args) -> str:
    """子代理出发行：┊ 🔀 delegate 子代理 X 出发：任务摘要。"""
    name = (args or {}).get("subagent_type") or "general-purpose"
    task = _cut((args or {}).get("prompt") or "", 40)
    head = f"{_tool_prefix()} 🔀 delegate 子代理 {name} 出发"
    return f"{head}：{task}" if task else head


def format_subagent_done(args, dt, result_str) -> str:
    """子代理完成行：┊ 🔀 delegate 子代理 X 完成（耗时）：结果摘要。"""
    name = (args or {}).get("subagent_type") or "general-purpose"
    dur = format_duration(dt)
    head = f"{_tool_prefix()} 🔀 delegate 子代理 {name} 完成"
    if dur:
        head += f"（{dur}）"
    ok, preview = result_preview(result_str)
    if not ok:
        return f"{head}：✗ {preview}" if preview else f"{head}：✗"
    if preview:
        return f"{head}：{_cut(preview, 50)}"
    return head


def is_async_subagent_result(result_str: str) -> bool:
    """判断 subagent 工具的返回是不是「后台异步已受理」而不是真结果。"""
    try:
        data = json.loads(result_str or "{}")
    except Exception:
        return False
    return isinstance(data, dict) and (
        data.get("mode") == "async" or ("delegation_id" in data and data.get("success"))
    )


def format_task_event(tool_name, result_str) -> str:
    """任务事件行：创建/完成两种（update 不打，太吵）。

    subject/计数尽量从 result JSON 和 task_store 挖；挖不到就退化成
    无细节形态（fail-open——展示行不能因为数据不全就报错）。
    """
    try:
        data = json.loads(result_str or "{}")
    except Exception:
        data = {}
    if tool_name == "task_create":
        subject = (data.get("task") or {}).get("subject", "")
        return f"📋 新任务：{subject}" if subject else "📋 新任务"
    # task_complete：subject 优先从 result 挖，挖不到查 store；
    # done/total 计数总是现查 store（便宜且总是准）
    subject = (data.get("task") or {}).get("subject", "") or data.get("subject", "")
    done = total = None
    try:
        task_id = data.get("task_id") or (data.get("task") or {}).get("id")
        from agent.task_store import get_task_store
        store = get_task_store(None)
        if task_id and not subject:
            t = store.get(task_id)
            if t:
                subject = t.get("subject", "")
        tasks = [t for t in store.list_all()
                 if t.get("status") != "deleted"]
        total = len(tasks)
        done = sum(1 for t in tasks if t.get("status") == "completed")
    except Exception:
        pass
    if subject and done is not None:
        return f"📋 完成 {done}/{total}：{subject}"
    if subject:
        return f"📋 完成：{subject}"
    return "📋 任务完成"


class EventPairer:
    """PRE/POST 配对器：同名同参并发场景靠双端队列天然配对。"""

    def __init__(self):
        self._starts = defaultdict(deque)   # (名, 参数json) → 时间戳队列
        self._pending = defaultdict(int)    # 名字 → 未完成数（工具栏 ◐ 段读）

    @staticmethod
    def _key(name, args):
        return (name, json.dumps(args or {}, sort_keys=True,
                                 ensure_ascii=False, default=str))

    def record(self, name, args):
        self._starts[self._key(name, args)].append(time.monotonic())
        self._pending[name] += 1

    def pop(self, name, args):
        key = self._key(name, args)
        q = self._starts.get(key)
        if not q:
            return None
        dt = time.monotonic() - q.popleft()
        if not q:
            del self._starts[key]   # 键用完就删：参数json可能含整个文件内容，不留在内存
        self._pending[name] = max(0, self._pending.get(name, 1) - 1)
        return dt

    def clear(self):
        """全部清空（回合开始时调用，防 PRE 无 POST 配对的幻影 ◐ 残留）。"""
        self._starts.clear()
        self._pending.clear()

    def pending_names(self) -> list:
        """当前没跑完的工具名列表（按记录顺序自然去重）。"""
        return [n for n, c in self._pending.items() if c > 0]


def reset_pending(rt) -> None:
    """回合开始时清 ◐ 黑板（cli 主循环每回合调用一次）。

    为什么需要：钩子拒绝/参数改写等路径会让 PRE 记的起点永远配不上
    POST——不清板的话工具栏会一直挂着幻影「◐ 工具名」。
    """
    try:
        rt.event_pending = []
        pairer = getattr(rt, "_event_pairer", None)
        if pairer is not None:
            pairer.clear()
        # 写前快照一并清（钩子拒绝等路径会让 PRE 存的快照配不上 POST）
        snaps = getattr(rt, "_write_snapshots", None)
        if isinstance(snaps, dict):
            snaps.clear()
    except Exception:
        pass


_SUBAGENT_TOOLS = {"subagent", "delegate_task"}
_TASK_TOOLS = {"task_create", "task_complete"}
# 写类工具：完成行后面追加 inline diff（红删绿增，跟 hermes 学的）
_WRITE_TOOLS = {"write_file", "str_replace"}
# 快照/对比的文件大小上限（超过就不画 diff——几十万行的 diff 没人看）
_DIFF_SNAPSHOT_MAX_BYTES = 200_000
_DIFF_MAX_LINES = 30


def build_edit_diff(old_text, new_text, max_lines=_DIFF_MAX_LINES):
    """两段文本 → inline diff 行列表（difflib.unified_diff 的紧凑版）。

    返回：(kind, line) 元组列表，kind ∈ {"-", "+", "@", " "}；
    超过 max_lines 截断并补一行「还有更多」提示。纯函数可单测。
    """
    import difflib
    diff = difflib.unified_diff(
        (old_text or "").splitlines(), (new_text or "").splitlines(),
        lineterm="", n=1,   # n=1：上下文只要 1 行，视觉紧凑
    )
    out = []
    for ln in diff:
        if ln.startswith("---") or ln.startswith("+++"):
            continue   # ---/+++ 文件头行是噪声，不要
        kind = ln[:1] if ln[:1] in ("-", "+", "@") else " "
        out.append((kind, ln))
        if len(out) >= max_lines:
            hidden = sum(1 for _ in diff)   # 吃掉剩余迭代器算总数
            if hidden:
                out.append(("…", f"… 还有 {hidden} 行差异未展示"))
            break
    return out


def _snapshot_write_target(args):
    """写类工具的目标路径（write_file/str_replace 的参数名统一取一遍）。"""
    args = args or {}
    return str(args.get("path") or args.get("file_path") or "").strip()


def _print_edit_diff(lines) -> None:
    """把 diff 行画到屏幕：红删绿增、@@ 暗青，缩进 4 格对齐事件行。

    用 rich 的 Text 对象（不解析 markup）——文件内容里带 [ ] 方括号
    不会被误当成富文本标签。
    """
    from rich.text import Text
    styles = {"-": "red", "+": "green", "@": "cyan dim", " ": "", "…": "dim"}
    for kind, text in lines:
        style = styles.get(kind, "")
        console.print(Text(f"    {text}", style=style) if style
                      else Text(f"    {text}"))


def install_event_lines(rt) -> None:
    """把两类钩子装到 agent 上（run_interactive 装配区调用一次）。

    取代老的 pre-only ⏺ 行钩子：PRE 只记时间戳/子代理出发行，
    POST 打完成行。rt.event_pending 是给工具栏 ◐ 段读的黑板。
    不挂 FAILURE 钩子——POST 对含 error 的结果已打 ✗ 行，
    再挂会一次失败打两行（model_tools 两个钩子都发）。
    """
    from cli_ui import console

    pairer = EventPairer()
    rt._event_pairer = pairer   # 挂到 rt 上：回合开始时 reset_pending 清板用
    rt.event_pending = []
    # 写类工具的「写前快照」：(名, 参数json) → 旧文件内容。
    # PRE 存、POST 取——纯 UI 层读文件，不给核心工具加一个字的负担。
    # 挂到 rt 上：回合开始时 reset_pending 一并清（防 PRE 无 POST 的残留）
    _write_snapshots = {}
    rt._write_snapshots = _write_snapshots

    def _update_pending():
        try:
            rt.event_pending = pairer.pending_names()
        except Exception:
            pass

    def _snapshot_old(tool_name, args):
        """PRE 时把要写的文件现状拍下来（读不到/太大就放弃 diff）。"""
        try:
            path = _snapshot_write_target(args)
            if not path:
                return
            from pathlib import Path
            p = Path(path)
            if not p.is_file() or p.stat().st_size > _DIFF_SNAPSHOT_MAX_BYTES:
                return
            _write_snapshots[EventPairer._key(tool_name, args)] = \
                p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

    def _emit_edit_diff(tool_name, args, result):
        """POST 时对比新旧内容画 diff（失败/没快照就静默跳过）。"""
        try:
            ok, _ = result_preview(result)
            if not ok:
                return   # 写都失败了，没有 diff 可画
            key = EventPairer._key(tool_name, args)
            old = _write_snapshots.pop(key, None)
            if old is None:
                return
            from pathlib import Path
            path = _snapshot_write_target(args)
            p = Path(path) if path else None
            if p is None or not p.is_file():
                return
            new = p.read_text(encoding="utf-8", errors="replace")
            lines = build_edit_diff(old, new)
            if lines:
                _print_edit_diff(lines)
        except Exception:
            pass

    def _on_pre(tool_name, args, **_kw):
        """记起点 + 子代理出发行 + 写类工具旧内容快照。异常全吞。"""
        try:
            args = args or {}
            pairer.record(tool_name, args)
            _update_pending()
            if tool_name in _SUBAGENT_TOOLS:
                console.print(f"[dim]{format_subagent_depart(args)}[/dim]")
            elif tool_name in _WRITE_TOOLS:
                _snapshot_old(tool_name, args)
            # 普通工具的等待期反馈归 spinner 行/状态栏 ◐ 段（cli_layout），
            # 这里不再补打 ◇ 出发行
        except Exception:
            pass

    def _on_post(tool_name, args, result, **_kw):
        """打完成行 + 写类工具 inline diff。POST 是流水线——最后原样
        return result。"""
        try:
            args = args or {}
            dt = pairer.pop(tool_name, args)
            _update_pending()
            if tool_name in _SUBAGENT_TOOLS:
                # 后台异步派发立即返回受理回执（子代理还没跑完）——
                # 只当「已出发」处理，不当「已完成」
                if is_async_subagent_result(result):
                    return result
                console.print(f"[dim]{format_subagent_done(args, dt, result)}[/dim]")
            elif tool_name in _TASK_TOOLS:
                console.print(f"[dim]{format_task_event(tool_name, result)}[/dim]")
            else:
                console.print(f"[dim]{format_tool_line(tool_name, args, dt, result)}[/dim]")
                if tool_name in _WRITE_TOOLS:
                    _emit_edit_diff(tool_name, args, result)
        except Exception:
            pass
        return result

    try:
        rt.agent.hooks_registry.register_pre_tool_use(
            _on_pre, name="cli_tool_event_line")
        rt.agent.hooks_registry.register_post_tool_use(
            _on_post, name="cli_tool_event_line")
    except Exception as e:
        logger.error("事件行钩子登记失败（进度展示将缺失）: %s", e)
