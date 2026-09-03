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
    """通用工具完成行：⏺ 名字 摘要 ✓/✗ 耗时（结果摘要）。"""
    ok, preview = result_preview(result_str)
    mark = "✓" if ok else "✗"
    parts = [f"⏺ {tool_name}"]
    s = summarize_args(tool_name, args)
    if s:
        parts.append(s)
    head = " ".join(parts)
    dur = format_duration(dt)
    tail = f"{mark} {dur}".strip()
    if preview:
        return f"{head} {tail}（{preview}）"
    return f"{head} {tail}"


def format_subagent_depart(args) -> str:
    """子代理出发行：⦿ 子代理 X 出发：任务摘要。"""
    name = (args or {}).get("subagent_type") or "general-purpose"
    task = _cut((args or {}).get("prompt") or "", 40)
    return f"⦿ 子代理 {name} 出发：{task}" if task else f"⦿ 子代理 {name} 出发"


def format_subagent_done(args, dt, result_str) -> str:
    """子代理完成行：⦿ 子代理 X 完成（耗时）：结果摘要。"""
    name = (args or {}).get("subagent_type") or "general-purpose"
    dur = format_duration(dt)
    head = f"⦿ 子代理 {name} 完成"
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
    except Exception:
        pass


_SUBAGENT_TOOLS = {"subagent", "delegate_task"}
_TASK_TOOLS = {"task_create", "task_complete"}


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

    def _update_pending():
        try:
            rt.event_pending = pairer.pending_names()
        except Exception:
            pass

    def _on_pre(tool_name, args, **_kw):
        """记起点 + 子代理出发行。异常全吞（fail-open）。"""
        try:
            args = args or {}
            pairer.record(tool_name, args)
            _update_pending()
            if tool_name in _SUBAGENT_TOOLS:
                console.print(f"[dim]{format_subagent_depart(args)}[/dim]")
            # 普通工具的等待期反馈归 spinner 行/状态栏 ◐ 段（cli_layout），
            # 这里不再补打 ◇ 出发行
        except Exception:
            pass

    def _on_post(tool_name, args, result, **_kw):
        """打完成行。POST 钩子是流水线——最后必须原样 return result。"""
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
