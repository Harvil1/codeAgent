"""事件行渲染器——claude code 风格：● 头行 + ⎿ 结果块。

设计一句话：**完成才打行，等待靠工具栏**。
- PRE（工具开始）：只记时间戳 + 维护 rt.event_pending（状态栏 ◐ 段/spinner
  行读它）；写类工具顺手拍一份旧文件内容快照（POST 画 diff 用）；
  子代理出发是时刻事件，PRE 就打 ● 头行
- POST（工具结束）：配对算耗时 → 打 ● 头行 + ⎿ 结果块（stdout 预览 /
  带行号 diff / 匹配条数……按工具类型挑最有信息量的几行）；
  失败也走 POST（结果含 error → ✗）——不另挂 FAILURE 钩子，
  否则一次失败会打两行 ✗（model_tools 对含 error 的结果两个钩子都发）
- 全部 try/except 吞掉——纯视觉，绝不挡工具执行（fail-open）

长相（claude code 同款）：

    ● Bash(ls D:\\project)
      ⎿  src/  docs/  package.json
         … +12 lines

    ● Update(cli_ui.py)
      ⎿  Added 9 lines, removed 2 lines
         51 +### 新段落 ###
         52 -旧段落
"""

import json
import logging
import re
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# 参数摘要表：工具名 → 从 args 里挑哪个字段当"这行在干嘛"的摘要
# （字段名必须跟工具 schema 一致：read/write/str_replace 都是 path）
_ARG_FIELDS = {
    "read_file": "path",
    "write_file": "path",
    "str_replace": "path",
    "terminal": "command",
    "search_files": "pattern",
    "glob": "pattern",
    "web_fetch": "url",
}

# claude code 风格显示名：工具名 → 界面上展示的动词
# （write_file 特殊：文件原本就存在时展示成 Update——POST 画 diff 时判断）
_TOOL_DISPLAY = {
    "terminal": "Bash",
    "read_file": "Read",
    "write_file": "Write",
    "str_replace": "Update",
    "search_files": "Grep",
    "glob": "Glob",
    "web_fetch": "WebFetch",
    "memory": "Memory",
    "subagent": "Agent",
    "delegate_task": "Agent",
}

# 结果块里正文最多展示几行（超出的折叠成「… +N lines」）
_BLOCK_BODY_LINES = 3
# diff 最多展示几行
_DIFF_MAX_LINES = 15


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
        limit = 60 if tool_name in ("terminal", "web_fetch") else 48
        return _cut(v, limit) if v else ""
    # 没登记的工具：拿第一个字符串型参数值顶上
    for v in args.values():
        if isinstance(v, str) and v:
            return _cut(v, 48)
    return ""


def format_duration(dt) -> str:
    """耗时格式化：太快就 <0.1s，否则 0.1s 精度。dt 为 None 返回空串。"""
    if dt is None:
        return ""
    return "<0.1s" if dt < 0.05 else f"{dt:.1f}s"


def result_preview(result_str: str):
    """从工具结果 JSON 里抠一行摘要。

    返回：(ok, 摘要)——result 含 error 键 → (False, 错误前 60 字)；
    成功按 preview → output → content → text 顺序找第一个非空键取前 60 字；
    都没有/解析失败 → (True, "")。
    """
    try:
        data = json.loads(result_str or "")
    except Exception:
        return True, ""
    if isinstance(data, dict):
        if data.get("error"):
            return False, _cut(data["error"], 60)
        for key in ("preview", "output", "content", "text"):
            v = data.get(key)
            if v:
                return True, _cut(v, 60)
    return True, ""


def format_tool_line(tool_name, args, dt=None, result_str=None, *,
                     is_update=False) -> str:
    """工具头行（claude code 同款）：● Name(参数摘要)。

    失败时（result 含 error）行尾追加 ✗；耗时只有子代理完成行用，
    普通工具行不带耗时（claude code 同款——干净）。
    """
    display = _TOOL_DISPLAY.get(tool_name) or tool_name
    if tool_name == "write_file" and is_update:
        display = "Update"   # 覆盖已有文件 = Update；新文件 = Write
    s = summarize_args(tool_name, args)
    head = f"● {display}({s})" if s else f"● {display}"
    if result_str is not None:
        ok, _ = result_preview(result_str)
        if not ok:
            head += "  ✗"
    return head


# ---------------------------------------------------------------------------
# 带行号 diff（claude code 同款：上下文/+ 用新行号，- 用旧行号）
# ---------------------------------------------------------------------------

def _diff_counts(old_text: str, new_text: str):
    """数一数这次改动加了几行、删了几行（n=0 最省事的数法）。"""
    import difflib
    add = rem = 0
    for ln in difflib.unified_diff(
        (old_text or "").splitlines(), (new_text or "").splitlines(),
        lineterm="", n=0,
    ):
        if ln.startswith("+++") or ln.startswith("---"):
            continue
        if ln.startswith("+"):
            add += 1
        elif ln.startswith("-"):
            rem += 1
    return add, rem


def build_numbered_diff(old_text, new_text, max_lines=_DIFF_MAX_LINES):
    """新旧文本 → [(kind, 行号, 文本)] 带行号 diff（unified n=1 的改版）。

    kind ∈ {" ", "+", "-"}；行号规则：上下文/+ 行用新文件行号，
    - 行用旧文件行号（claude code 同款）。超过 max_lines 截断，末尾补
    ("…", None, 隐藏行数)。纯函数可单测。
    """
    import difflib
    a = (old_text or "").splitlines()
    b = (new_text or "").splitlines()

    entries = []
    old_no = new_no = None
    for ln in difflib.unified_diff(a, b, lineterm="", n=1):
        if ln.startswith("---") or ln.startswith("+++"):
            continue   # 文件头行是噪声
        if ln.startswith("@"):
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", ln)
            if m:
                old_no, new_no = int(m.group(1)), int(m.group(2))
            continue
        if ln.startswith("-"):
            if old_no is not None:
                entries.append(("-", old_no, ln[1:]))
                old_no += 1
        elif ln.startswith("+"):
            if new_no is not None:
                entries.append(("+", new_no, ln[1:]))
                new_no += 1
        else:
            if new_no is not None:
                entries.append((" ", new_no, ln[1:] if len(ln) > 1 else ""))
                old_no += 1
                new_no += 1
    if len(entries) > max_lines:
        hidden = len(entries) - max_lines
        return entries[:max_lines] + [("…", None, str(hidden))]
    return entries


def _diff_block_lines(old_text, new_text):
    """写类工具的结果块：⎿ Added/Removed 统计 + 带行号红删绿增。"""
    lines = []
    add, rem = _diff_counts(old_text, new_text)
    if not (old_text or "").strip():
        n = len((new_text or '').splitlines())
        lines.append(("dim", f"  ⎿  Wrote {n} line{'s' if n != 1 else ''}"))
    else:
        lines.append(("dim",
                      f"  ⎿  Added {add} line{'s' if add != 1 else ''}, "
                      f"removed {rem} line{'s' if rem != 1 else ''}"))
    for kind, no, text in build_numbered_diff(old_text, new_text):
        if kind == "+":
            lines.append(("green", f"     {no:>4} +{text}"))
        elif kind == "-":
            lines.append(("red", f"     {no:>4} -{text}"))
        elif kind == "…":
            lines.append(("dim", f"     … +{text} more lines"))
        else:
            lines.append(("dim", f"     {no:>4}  {text}"))
    return lines


# ---------------------------------------------------------------------------
# 结果块：按工具类型从返回 JSON 里挑最有信息量的几行
# ---------------------------------------------------------------------------

def format_result_block(tool_name, result_str, *, old_text=None,
                        new_text=None) -> list:
    """工具结果 → ⎿ 结果块行列表 [(style, text), ...]（纯函数可单测）。

    style ∈ {"", "red", "green", "dim"}；text 已含缩进（首行 "  ⎿  "，
    后续行 5 格对齐）。失败（error 键）统一 ✗ 红行。
    写类工具传 old_text/new_text 才有 diff 块。
    """
    try:
        data = json.loads(result_str or "")
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}

    # 失败优先：红 ✗ + 错误摘要
    if data.get("error"):
        return [("red", f"  ⎿  ✗ {_cut(data['error'], 80)}")]

    # 写类工具：统计 + 带行号 diff
    if tool_name in ("write_file", "str_replace") \
            and new_text is not None:
        return _diff_block_lines(old_text, new_text)

    if tool_name == "terminal":
        stdout = str(data.get("stdout") or "")
        stderr = str(data.get("stderr") or "")
        body = stdout.strip() or stderr.strip()
        if not body:
            return []
        all_lines = body.splitlines()
        shown, rest = all_lines[:_BLOCK_BODY_LINES], all_lines[_BLOCK_BODY_LINES:]
        out = [("dim", f"  ⎿  {shown[0]}")]
        out += [("dim", f"     {ln}") for ln in shown[1:]]
        if rest:
            out.append(("dim", f"     … +{len(rest)} lines"))
        return out

    if tool_name == "read_file":
        n = data.get("total_lines")
        if n:
            return [("dim", f"  ⎿  读取 {n} 行")]
        return []

    if tool_name == "search_files":
        matches = data.get("matches") or []
        cnt = data.get("match_count", len(matches))
        out = [("dim", f"  ⎿  找到 {cnt} 处匹配")]
        for m in matches[:_BLOCK_BODY_LINES]:
            out.append(("dim",
                        f"     {_cut(m.get('file', '?'), 40)}:{m.get('line', '?')}"
                        f"  {_cut(m.get('content', ''), 50)}"))
        more = cnt - min(len(matches), _BLOCK_BODY_LINES)
        if more > 0:
            out.append(("dim", f"     … 还有 {more} 处"))
        return out

    if tool_name == "glob":
        matches = data.get("matches") or []
        cnt = data.get("count", len(matches))
        out = [("dim", f"  ⎿  找到 {cnt} 个文件")]
        for m in matches[:_BLOCK_BODY_LINES]:
            out.append(("dim", f"     {_cut(str(m), 60)}"))
        if cnt > _BLOCK_BODY_LINES:
            out.append(("dim", f"     … 还有 {cnt - _BLOCK_BODY_LINES} 个"))
        return out

    # 没特判的工具：给一行结果摘要（没有就算了，不打空块）
    ok, preview = result_preview(result_str)
    if preview:
        return [("dim", f"  ⎿  {preview}")]
    return []


# ---------------------------------------------------------------------------
# 子代理 / 任务事件
# ---------------------------------------------------------------------------

def format_subagent_depart(args) -> str:
    """子代理头行：● Agent(任务摘要) 角色名。"""
    name = (args or {}).get("subagent_type") or "general-purpose"
    task = _cut((args or {}).get("prompt") or "", 60)
    return f"● Agent({task}) {name}" if task else f"● Agent {name}"


def format_subagent_done(args, dt, result_str) -> list:
    """子代理完成块：⎿ Done（耗时）/ 失败给 ✗ + 摘要。"""
    dur = format_duration(dt)
    ok, preview = result_preview(result_str)
    if not ok:
        line = f"  ⎿  ✗ {_cut(preview, 60)}" if preview else "  ⎿  ✗"
        return [("red", line)]
    return [("dim", f"  ⎿  Done（{dur}）" if dur else "  ⎿  Done")]


def is_async_subagent_result(result_str: str) -> bool:
    """判断 subagent 工具的返回是不是「后台异步已受理」而不是真结果。"""
    try:
        data = json.loads(result_str or "{}")
    except Exception:
        return False
    return isinstance(data, dict) and (
        data.get("mode") == "async" or ("delegation_id" in data and data.get("success"))
    )


def format_task_event(tool_name, result_str) -> list:
    """任务事件块：● TaskCreate(subject) / ● TaskComplete(subject) ⎿ 进度。

    subject/计数尽量从 result JSON 和 task_store 挖；挖不到就退化成
    无细节形态（fail-open——展示行不能因为数据不全就报错）。
    """
    try:
        data = json.loads(result_str or "{}")
    except Exception:
        data = {}
    if tool_name == "task_create":
        subject = (data.get("task") or {}).get("subject", "")
        head = f"● TaskCreate({_cut(subject, 50)})" if subject else "● TaskCreate"
        return [("", head)]
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
    head = f"● TaskComplete({_cut(subject, 50)})" if subject else "● TaskComplete"
    if done is not None:
        return [("", head), ("dim", f"  ⎿  完成 {done}/{total}")]
    return [("", head)]


# ---------------------------------------------------------------------------
# PRE/POST 配对器
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 钩子装配
# ---------------------------------------------------------------------------

_SUBAGENT_TOOLS = {"subagent", "delegate_task"}
_TASK_TOOLS = {"task_create", "task_complete"}
# 写类工具：POST 画 Write/Update + 带行号 diff（纯 UI 层读文件，
# 不给核心工具加一个字的负担）
_WRITE_TOOLS = {"write_file", "str_replace"}
# 快照/对比的文件大小上限（超过就不画 diff——几十万行的 diff 没人看）
_DIFF_SNAPSHOT_MAX_BYTES = 200_000


def _snapshot_write_target(args):
    """写类工具的目标路径（write_file/str_replace 的参数名统一取一遍）。"""
    args = args or {}
    return str(args.get("path") or args.get("file_path") or "").strip()


def install_event_lines(rt) -> None:
    """把两类钩子装到 agent 上（run_interactive 装配区调用一次）。

    PRE 只记时间戳/拍快照/子代理 ● 头行；POST 打 ● 头行 + ⎿ 结果块。
    rt.event_pending 是给工具栏 ◐ 段读的黑板。
    不挂 FAILURE 钩子——POST 对含 error 的结果已打 ✗，
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

    def _print_block(lines) -> None:
        """把 (style, text) 行列表画上屏（rich Text，不解析 markup——
        文件内容里带 [ ] 方括号不会被误当成富文本标签）。"""
        from rich.text import Text
        for style, text in lines:
            try:
                console.print(Text(text, style=style) if style else Text(text))
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

    def _read_new_text(args):
        """POST 时把写完的文件读回来（读不到返回 None，跳过 diff）。"""
        try:
            from pathlib import Path
            path = _snapshot_write_target(args)
            p = Path(path) if path else None
            if p is None or not p.is_file() \
                    or p.stat().st_size > _DIFF_SNAPSHOT_MAX_BYTES:
                return None
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

    def _on_pre(tool_name, args, **_kw):
        """记起点 + 子代理 ● 头行 + 写类工具旧内容快照。异常全吞。"""
        try:
            args = args or {}
            pairer.record(tool_name, args)
            _update_pending()
            if tool_name in _SUBAGENT_TOOLS:
                _print_block([("", format_subagent_depart(args))])
            elif tool_name in _WRITE_TOOLS:
                _snapshot_old(tool_name, args)
            # 普通工具的等待期反馈归 spinner 行/状态栏 ◐ 段（cli_layout）
        except Exception:
            pass

    def _on_post(tool_name, args, result, **_kw):
        """打 ● 头行 + ⎿ 结果块。POST 是流水线——最后原样 return result。"""
        try:
            args = args or {}
            dt = pairer.pop(tool_name, args)
            _update_pending()
            if tool_name in _SUBAGENT_TOOLS:
                # 后台异步派发立即返回受理回执（子代理还没跑完）——
                # 只当「已出发」处理，不当「已完成」
                if is_async_subagent_result(result):
                    return result
                _print_block(format_subagent_done(args, dt, result))
            elif tool_name in _TASK_TOOLS:
                _print_block(format_task_event(tool_name, result))
            else:
                head = format_tool_line(tool_name, args, dt, result)
                if tool_name in _WRITE_TOOLS:
                    ok, _ = result_preview(result)
                    if ok:
                        key = EventPairer._key(tool_name, args)
                        old = _write_snapshots.pop(key, None)
                        new = _read_new_text(args)
                        if new is not None:
                            _print_block(
                                [("", format_tool_line(
                                    tool_name, args, dt, result,
                                    is_update=bool((old or "").strip())))]
                                + format_result_block(
                                    tool_name, result,
                                    old_text=old, new_text=new))
                            return result
                _print_block(
                    [("", head)] + format_result_block(tool_name, result))
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
