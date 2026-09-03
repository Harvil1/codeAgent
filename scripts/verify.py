"""复刻检查清单的验证脚本：跑 24 项小检查，确认 agent 的核心功能还活着。

本项目按复刻指南（11-scaffold.md）实现，这个脚本验收指南里「这些功能
必须存在且能用」的清单——每项做一件小事（建个文件、发个工具调用），
看结果对不对。适合改完代码后快速回归，比跑全量测试快得多。

用法：
    uv run python scripts/verify.py

每项输出 PASS（通过）/ FAIL（失败）/ SKIPPED（跳过），最后给汇总。
SKIPPED 的项需要真实 API key 才能验证。
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# 以脚本方式运行时 Python 只把 scripts/ 加进搜索路径，手动把项目根也加进去
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _result(status, detail=""):
    """拼一个检查结果字典。

    参数：
        status  结果状态："PASS" / "FAIL" / "SKIPPED"
        detail  补充说明文字（会显示在终端）

    返回：{"status": ..., "detail": ...} 字典。
    """
    return {"status": status, "detail": detail}


def _ok(detail=""):
    """生成一个 PASS 结果。参数 detail 为附带的说明文字。"""
    return _result("PASS", detail)


def _fail(detail=""):
    """生成一个 FAIL 结果。参数 detail 为失败原因。"""
    return _result("FAIL", detail)


def _skip(detail=""):
    """生成一个 SKIPPED 结果。参数 detail 为跳过原因。"""
    return _result("SKIPPED", detail)


# ---------------------------------------------------------------------------
# 基础对话
# ---------------------------------------------------------------------------

def check_agent_initialization():
    """验证 AIAgent 对象能正常创建（不发起任何真实 API 请求）。

    返回：PASS/FAIL 结果。
    """
    from agent import AIAgent
    agent = AIAgent(api_key="fake", model="test", enabled_toolsets=[])
    return _ok(f"max_iter={agent.max_iterations}, budget={agent.iteration_budget.remaining}")


def check_tool_definitions():
    """验证工具定义能正常加载，且核心工具（terminal/read/write/memory）都在。

    返回：PASS/FAIL 结果。
    """
    from model_tools import get_tool_definitions
    tools = get_tool_definitions(["core"])
    names = [t["function"]["name"] for t in tools]
    expected = {"terminal", "read_file", "write_file", "memory"}
    missing = expected - set(names)
    if missing:
        return _fail(f"缺少工具: {missing}")
    return _ok(f"暴露 {len(names)} 个工具")


def check_terminal_tool():
    """验证 terminal 工具真的能执行一条命令（跑 echo 看输出）。

    返回：PASS/FAIL 结果。
    """
    from tools.registry import registry
    result = asyncio.run(registry.dispatch("terminal", {"command": "echo verify_ok"}))
    data = json.loads(result)
    if "verify_ok" in data.get("stdout", ""):
        return _ok("echo 输出正确")
    return _fail(f"输出异常: {data}")


def check_read_file_tool(tmp):
    """验证 read_file 工具能读回文件内容。

    参数：
        tmp  临时目录 Path，测试文件写在这底下

    返回：PASS/FAIL 结果。
    """
    from tools.registry import registry
    f = tmp / "sample.txt"
    f.write_text("line1\nline2\n", encoding="utf-8")
    result = asyncio.run(registry.dispatch("read_file", {"path": str(f)}))
    data = json.loads(result)
    if "line1" in data.get("content", ""):
        return _ok("读取到内容")
    return _fail(f"读取失败: {data}")


def check_interrupt():
    """验证中断机制：调用 interrupt() 后，中断标志确实被设置
    （用户按 Ctrl+C 优雅打断 agent 的底层开关）。

    返回：PASS/FAIL 结果。
    """
    from agent import AIAgent
    agent = AIAgent(api_key="fake", model="test", enabled_toolsets=[])
    agent.interrupt()
    if agent._interrupt_requested:
        return _ok("中断标志已设置")
    return _fail("中断标志未设置")


# ---------------------------------------------------------------------------
# 记忆系统
# ---------------------------------------------------------------------------

def check_memory_tool_write(tmp):
    """验证 memory 工具能保存一条记忆并落盘。

    参数：
        tmp  临时目录 Path，当作隔离的 agent home 用（记忆写到 tmp/.memory/）

    返回：PASS/FAIL 结果。
    """
    from tools.registry import registry
    from agent.memory_store import MemoryStore
    store = MemoryStore(codeagent_home=tmp)

    result = asyncio.run(registry.dispatch(
        "memory",
        {"action": "save", "name": "验证测试", "description": "验证测试",
         "type": "other", "body": "验证测试"},
        memory_store=store,
    ))
    data = json.loads(result)
    if data.get("success"):
        entries = store.list_all()
        if any("验证测试" in (e.body or "") for e in entries):
            return _ok("已写入 .memory/")
    return _fail(f"写入失败: {data}")


def check_memory_persist(tmp):
    """验证保存记忆后，磁盘上真的出现了记忆文件（.memory/ 目录或 MEMORY.md 索引）。

    参数：
        tmp  临时目录 Path，当作隔离的 agent home

    返回：PASS/FAIL 结果。
    """
    from agent.memory_store import MemoryStore
    store = MemoryStore(codeagent_home=tmp)
    store.add("memory", "持久化测试")
    # 存储是多文件模式：条目是 .memory/ 下的 .md 文件，MEMORY.md 只是索引
    memory_dir = tmp / ".memory"
    has_files = memory_dir.exists() and any(memory_dir.glob("*.md"))
    if has_files or (tmp / "MEMORY.md").exists():
        return _ok(".memory/ 已创建")
    return _fail("记忆文件未创建")


def check_memory_reload(tmp):
    """验证记忆的持久性：新建一个 MemoryStore 实例（模拟重启），旧记忆还在。

    参数：
        tmp  临时目录 Path，当作隔离的 agent home

    返回：PASS/FAIL 结果。
    """
    from agent.memory_store import MemoryStore
    s1 = MemoryStore(codeagent_home=tmp)
    s1.add("memory", "重启测试")

    s2 = MemoryStore(codeagent_home=tmp)
    entries = s2.list_all()
    if any("重启测试" in (e.body or "") for e in entries):
        return _ok("记忆已跨实例加载")
    return _fail("记忆丢失")


# ---------------------------------------------------------------------------
# 技能系统
# ---------------------------------------------------------------------------

def _setup_skill(tmp):
    """在临时目录里造一个名叫 hello 的示例技能，供技能类检查复用。

    参数：
        tmp  临时目录 Path，技能建在 tmp/skills/hello/

    返回：技能库目录的 Path。
    """
    skills = tmp / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    (skills / "hello").mkdir(parents=True, exist_ok=True)
    (skills / "hello" / "SKILL.md").write_text(
        '---\nname: hello\ndescription: "打招呼"\n---\n# Hello\n你好技能',
        encoding="utf-8",
    )
    return skills


def check_skill_trigger(tmp):
    """验证技能能被发现，且触发时正文内容会被注入对话。

    参数：
        tmp  临时目录 Path（会先在里面造一个 hello 技能）

    返回：PASS/FAIL 结果。
    """
    from agent.skill_commands import scan_skill_commands, execute_skill
    skills = _setup_skill(tmp)
    cmds = scan_skill_commands(skills)
    if "/hello" not in cmds:
        return _fail("技能未被发现")
    injected = execute_skill(cmds["/hello"]["skill_md_path"], "test")
    if "Hello" in injected:
        return _ok("技能内容已注入")
    return _fail("技能触发失败")


def check_skills_list(tmp):
    """验证 skills_list 工具能列出技能库里的技能。

    参数：
        tmp  临时目录 Path（会先造一个 hello 技能）

    返回：PASS/FAIL 结果。
    """
    from tools.registry import registry
    skills = _setup_skill(tmp)
    result = asyncio.run(registry.dispatch("skills_list", {}, codeagent_home=tmp))
    data = json.loads(result)
    names = [s["name"] for s in data["skills"]]
    if "hello" in names:
        return _ok("列出了 hello 技能")
    return _fail(f"未列出: {data}")


def check_skill_view(tmp):
    """验证 skill_view 工具能查看指定技能的内容。

    参数：
        tmp  临时目录 Path（会先造一个 hello 技能）

    返回：PASS/FAIL 结果。
    """
    from tools.registry import registry
    _setup_skill(tmp)
    result = asyncio.run(registry.dispatch("skill_view", {"name": "hello"}, codeagent_home=tmp))
    data = json.loads(result)
    if "Hello" in data.get("content", ""):
        return _ok("查看了 hello 技能")
    return _fail(f"查看失败: {data}")


def check_skill_manage_create(tmp):
    """验证 skill_manage 工具能创建新技能（SKILL.md 真的出现在磁盘上）。

    参数：
        tmp  临时目录 Path，技能建在 tmp/skills/ 下

    返回：PASS/FAIL 结果。
    """
    from tools.registry import registry
    asyncio.run(registry.dispatch(
        "skill_manage",
        {"action": "create", "name": "new-skill", "content": "---\nname: x\n---\nbody"},
        codeagent_home=tmp,
    ))
    if (tmp / "skills" / "new-skill" / "SKILL.md").exists():
        return _ok("创建了 new-skill")
    return _fail("创建失败")


def check_usage_stats(tmp):
    """验证技能使用统计：用一次技能后 .usage.json 里的计数会 +1。

    参数：
        tmp  临时目录 Path（会先造一个 hello 技能）

    返回：PASS/FAIL 结果。
    """
    from tools.skill_usage import bump_use, load_usage
    skills = _setup_skill(tmp)
    bump_use(skills, "hello")
    data = load_usage(skills)
    if data.get("hello", {}).get("use_count") == 1:
        return _ok("use_count=1")
    return _fail("统计未记录")


# ---------------------------------------------------------------------------
# 会话存储
# ---------------------------------------------------------------------------

def check_sessions_db(tmp):
    """验证会话存储初始化时会创建 .sessions/ 目录（JSONL 文件目录，
    append-only 方便恢复；传 sessions.db 路径会自动转成该目录）。

    参数：
        tmp  临时目录 Path

    返回：PASS/FAIL 结果。
    """
    from agent.session_store import SessionStore
    db = tmp / "sessions.db"
    store = SessionStore(db)
    # 兼容老参数：传 sessions.db 路径会自动转成 .sessions/ 目录
    sessions_dir = tmp / ".sessions"
    if sessions_dir.exists():
        return _ok(str(sessions_dir))
    return _fail("会话目录未创建")


def check_session_search(tmp):
    """验证 session_search 工具能搜到历史对话内容。

    参数：
        tmp  临时目录 Path，会话存在这底下

    返回：PASS/FAIL 结果。
    """
    from agent.session_store import SessionStore
    from tools.registry import registry
    store = SessionStore(tmp / "s.db")
    sid = store.create_session()
    store.append_message(sid, "user", "Python 测试内容")

    result = asyncio.run(registry.dispatch(
        "session_search",
        {"query": "Python"},
        session_store=store,
    ))
    data = json.loads(result)
    if data.get("total", 0) > 0:
        return _ok(f"找到 {data['total']} 条")
    return _fail("未搜到")


# ---------------------------------------------------------------------------
# Curator
# ---------------------------------------------------------------------------

def check_curator_status(tmp):
    """验证读取 curator 状态不报错（状态为空也是正常情况）。

    参数：
        tmp  临时目录 Path

    返回：PASS/FAIL 结果。
    """
    from agent.curator import load_state
    state = load_state(tmp / "skills")
    # 从未跑过 curator 时状态文件不存在，返回空 dict 也算通过
    return _ok(f"state keys: {list(state.keys()) or '(空)'}")


def check_curator_dry_run(tmp):
    """验证 curator 的 dry-run 模式：只预览要做的转换，不动文件。

    参数：
        tmp  临时目录 Path

    返回：PASS/FAIL 结果。
    """
    from agent.curator import run_curator_review
    skills = tmp / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    report = run_curator_review(skills, dry_run=True)
    if report["dry_run"] is True:
        return _ok(f"transitions={report['transitions']}")
    return _fail("dry_run 标志错误")


def check_curator_archive(tmp):
    """验证技能归档：archive 后技能目录从原地消失、出现在 .archive/ 下
    （「完全可逆」铁律——归档不删除，只是挪到 .archive/ 藏起来）。

    参数：
        tmp  临时目录 Path

    返回：PASS/FAIL 结果。
    """
    from tools.skill_usage import archive_skill, mark_agent_created, bump_use
    skills = tmp / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    (skills / "old").mkdir(parents=True, exist_ok=True)
    (skills / "old" / "SKILL.md").write_text("# old\n", encoding="utf-8")
    bump_use(skills, "old")
    mark_agent_created(skills, "old")

    ok, _ = archive_skill(skills, "old")
    if ok and (skills / ".archive" / "old").exists():
        return _ok("已归档到 .archive/")
    return _fail("归档失败")


def check_curator_restore(tmp):
    """验证归档可逆：restore 能把 .archive/ 里的技能放回原位。

    参数：
        tmp  临时目录 Path

    返回：PASS/FAIL 结果。
    """
    from tools.skill_usage import archive_skill, restore_skill, bump_use
    skills = tmp / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    (skills / "lost").mkdir(parents=True, exist_ok=True)
    (skills / "lost" / "SKILL.md").write_text("# lost\n", encoding="utf-8")
    bump_use(skills, "lost")

    archive_skill(skills, "lost")
    ok, _ = restore_skill(skills, "lost")
    if ok and (skills / "lost").exists():
        return _ok("已恢复")
    return _fail("恢复失败")


# ---------------------------------------------------------------------------
# 委托
# ---------------------------------------------------------------------------

def check_delegate_sync(tmp):
    """验证 subagent 同步委托链路（用假实现替掉子代理，不真跑 LLM）。

    参数：
        tmp  临时目录 Path（本项未用到，保持签名统一）

    返回：PASS/FAIL 结果。
    """
    from unittest.mock import patch
    from tools.registry import registry
    with patch("tools.delegate_tool._run_child", return_value="子代理完成"):
        result = asyncio.run(registry.dispatch(
            "subagent",
            {"goal": "测试任务"},
            base_url=None, api_key="fake", model="test",
        ))
    data = json.loads(result)
    if data.get("success") and data.get("result") == "子代理完成":
        return _ok("同步委托 OK")
    return _fail(f"委托失败: {data}")


def check_delegate_batch(tmp):
    """验证 subagent 批量模式：一次派 3 个任务并全部拿到结果（假实现，不真跑）。

    参数：
        tmp  临时目录 Path（本项未用到，保持签名统一）

    返回：PASS/FAIL 结果。
    """
    from unittest.mock import patch
    from tools.registry import registry
    with patch("tools.delegate_tool._run_child", return_value="ok"):
        result = asyncio.run(registry.dispatch(
            "subagent",
            {"tasks": [{"goal": "a"}, {"goal": "b"}, {"goal": "c"}]},
            base_url=None, api_key="fake", model="test",
        ))
    data = json.loads(result)
    if data["mode"] == "batch" and len(data["results"]) == 3:
        return _ok("3 个任务并行完成")
    return _fail(f"批量失败: {data}")


# ---------------------------------------------------------------------------
# 上下文压缩
# ---------------------------------------------------------------------------

def check_context_compress():
    """验证上下文压缩：消息条数超阈值时新管线会把历史压短（LLM 用假实现）。

    这里把触发阈值调得很低，确保压缩一定会发生。

    返回：PASS/FAIL 结果。
    """
    from agent.context_pipeline import compress_if_needed, CompressionSessionState
    from config import DEFAULT_CONFIG

    def fake_create(**kw):
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="总结内容")
            )]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    msgs = [{"role": "system", "content": "sys"}]
    for i in range(30):
        msgs.append({"role": "user", "content": f"消息 {i}"})
        msgs.append({"role": "assistant", "content": f"回复 {i}"})

    ctx_cfg = dict(DEFAULT_CONFIG.get("context", {}))
    # 把触发阈值调低到必触发，否则样例对话不够长压不动
    ctx_cfg["snip_message_threshold"] = 50
    state = CompressionSessionState()
    new_msgs, compressed, _compacted = asyncio.run(compress_if_needed(
        msgs,
        llm_client=client,
        model="deepseek-chat",
        config=ctx_cfg,
        session_state=state,
        agent_home=None,
        session_id="verify",
    ))
    if compressed and len(new_msgs) < len(msgs):
        return _ok(f"{len(msgs)} → {len(new_msgs)} 条")
    return _fail("未压缩")


# ---------------------------------------------------------------------------
# slash 命令注册表
# ---------------------------------------------------------------------------

def check_slash_registry():
    """验证 slash 命令注册表：命令一条没丢、每个 handler 都可调用。"""
    import cli_commands as cc
    # 触发命令模块 import（自登记发生在 import 时）
    import cli  # noqa: F401
    import cli_diag_cmds  # noqa: F401
    import cli_session_cmds  # noqa: F401
    import cli_skill_memory_cmds  # noqa: F401
    cmds = cc.all_commands()
    if len(cmds) < 37:
        return _fail(f"注册表只有 {len(cmds)} 条（基线 37），命令迁移丢了")
    for c in cmds:
        if not callable(c.handler):
            return _fail(f"命令 {c.name} 的 handler 不可调用")
    return _ok(f"{len(cmds)} 条命令全部登记，handler 可调用")


def check_cli_completer():
    """验证三级补全器：命令名/参数两级都能出候选，异常不外泄。"""
    from prompt_toolkit.document import Document
    from prompt_toolkit.completion import CompleteEvent
    import cli_commands as cc
    import cli  # noqa: F401  触发自登记
    import cli_diag_cmds, cli_session_cmds, cli_skill_memory_cmds  # noqa: F401
    from cli_layout import SlashCompleter

    # amap 只算一次，构造补全器和二级检查共用同一份数据
    amap = cc.arg_completer_map()
    comp = SlashCompleter(
        registry_tokens=cc.all_tokens(),
        arg_completers=amap,
        dynamic_tokens_fn=lambda: [],
    )
    got = [c.text for c in comp.get_completions(Document("/hel"), CompleteEvent())]
    if "/help" not in got:
        return _fail(f"一级补全没出 /help：{got[:5]}")
    if not amap:
        return _ok("一级补全正常；二级未实测（暂无命令登记参数补全器）")
    # 二级：逐个验证登记了参数补全器的命令（候选能出才算过——
    # 补全器返回了候选却一个都没渲染出来就是真 bug）
    for token, fn in amap.items():
        cands = [c.text for c in comp.get_completions(
            Document(f"{token} "), CompleteEvent())]
        if not cands and fn(f"{token} "):
            return _fail(f"{token} 的参数补全没出候选")
    return _ok("三级补全器两级候选正常")


def check_event_lines():
    """验证事件行渲染器：hermes 风格工具行 + 失败标记 + 启动询问已删。"""
    import cli_events as ce
    line = ce.format_tool_line("terminal", {"command": "pytest -q"}, 4.06,
                               '{"output": "21 passed"}')
    for frag in ("┊", "💻", "pytest -q", "✓", "4.1s"):
        if frag not in line:
            return _fail(f"工具行缺 {frag!r}：{line}")
    bad = ce.format_tool_line("terminal", {"command": "boom"}, 0.2,
                              '{"error": "exit 1"}')
    if "✗" not in bad or "exit 1" not in bad:
        return _fail(f"失败行缺 ✗/错误摘要：{bad}")
    # 摘要字段名对齐工具 schema（read_file 用 path 不是 file_path）
    rl = ce.format_tool_line("read_file", {"path": "src/app.py"}, 0.05,
                             '{"content": "x"}')
    if "src/app.py" not in rl:
        return _fail(f"read_file 行缺路径摘要（字段名对不上 schema？）：{rl}")
    # inline diff：红删绿增行 + 截断提示
    dlines = ce.build_edit_diff("a\nb\nc\n", "a\nX\nc\nd\n")
    kinds = [k for k, _ in dlines]
    if "-" not in kinds or "+" not in kinds:
        return _fail(f"diff 缺删/增行：{dlines}")
    dcap = ce.build_edit_diff("", "\n".join(f"line{i}" for i in range(100)))
    if not any(k == "…" for k, _ in dcap):
        return _fail("diff 超长没有截断提示")
    p = ce.EventPairer()
    p.record("x", {})
    if p.pop("x", {}) is None or p.pop("x", {}) is not None:
        return _fail("配对队列进出异常")
    import cli_session_cmds
    if hasattr(cli_session_cmds, "_maybe_prompt_resume"):
        return _fail("启动询问函数 _maybe_prompt_resume 还在")
    return _ok("事件行渲染器 + 启动回归正常")


def check_stream_box():
    """验证流式回答框：框头/框尾、思考框押后、CJK 表格重排。"""
    from cli_stream import StreamBoxRenderer

    got = []
    r = StreamBoxRenderer(print_fn=got.append, width_fn=lambda: 60)

    # 思考流 → 正文（押后）：思考框必须排在回答框前面
    r.on_event({"type": "reasoning", "delta": "先想一想\n"})
    r.on_event({"type": "content", "delta": "| 名字 | 数量 |\n|---|---|\n"})
    r.on_event({"type": "content",
                "delta": "| 苹果 | 1 |\n| 香蕉香蕉 | 22 |\n\n"})
    r.on_event({"type": "content", "delta": "回答结束"})
    r.on_event({"type": "done"})
    # 剥掉 ANSI 色码再断言（默认皮肤给正文上真彩色）
    import re
    text = re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(got))

    if "┌─思考" not in text:
        return _fail(f"思考框头缺失：{text[:120]!r}")
    if text.index("思考") > text.index("╭─"):
        return _fail("思考框必须排在回答框前面")
    if "╭─" not in text or "╰" not in text:
        return _fail(f"回答框头/尾缺失：{text[:120]!r}")
    if "回答结束" not in text:
        return _fail("done 后正文丢失")
    # CJK 表格重排：表头行和数据行的竖线位置必须一致（占宽对齐）
    lines = [ln for ln in text.split("\n") if ln.startswith("    |")]
    if len(lines) < 4:
        return _fail(f"表格行数不对：{lines}")
    pipe_pos = {ln.index("|", 4) for ln in (lines[0], lines[2])}
    if len(pipe_pos) != 1:
        return _fail(f"CJK 列没对齐：{lines}")
    # 表格半行兜底：done 冲掉一切残留
    r2 = StreamBoxRenderer(print_fn=lambda s: None, width_fn=lambda: 60)
    r2.on_event({"type": "content", "delta": "半行"})
    r2.on_event({"type": "done"})
    if r2._buf != "":
        return _fail("done 后半行缓冲没清")
    return _ok("框头/尾 + 思考押后 + CJK 表格对齐正常")


# ---------------------------------------------------------------------------
# CLI 骨架（块①：常驻 Application 操作台）
# ---------------------------------------------------------------------------

def _gbk_safe(s: str) -> str:
    """把状态栏文案里的 emoji（⚡📂☂ 等 GBK 编不了的）替换成问号——
    verify 直跑时 stdout 是 GBK 控制台，详情里带 emoji 会炸 print。"""
    return s.encode("gbk", "replace").decode("gbk")


def check_status_bar_tiers():
    """验证状态栏三档宽度：该出现的段出现、不该出现的段不出现。"""
    from types import SimpleNamespace as _NS
    from cli_layout import status_bar_segments

    rt = _NS(
        agent=_NS(model="test-model"),
        workspace_cwd="D:/x/codeAgent",
        bg_count=2,
        event_pending=["terminal"],
        turn_active=True,
    )
    # 窄屏 <52：模型 + 计时；目录/后台/工具/按键提示都不出现
    segs = status_bar_segments(rt, width=40, elapsed_s=12.0)
    joined = " │ ".join(segs)
    if "test-model" not in joined or "12s" not in joined:
        return _fail(f"窄屏缺模型/计时: {_gbk_safe(joined)}")
    if "codeAgent" in joined or "后台" in joined or "terminal" in joined:
        return _fail(f"窄屏不该有目录/后台/工具段: {_gbk_safe(joined)}")
    # 中屏 <76：目录/后台/工具出现，按键提示还没有
    segs = status_bar_segments(rt, width=60, elapsed_s=12.0)
    joined = " │ ".join(segs)
    for frag in ("test-model", "codeAgent", "后台", "terminal", "12s"):
        if frag not in joined:
            return _fail(f"中屏缺 {frag!r}: {_gbk_safe(joined)}")
    if "Enter" in joined:
        return _fail(f"中屏不该有按键提示: {_gbk_safe(joined)}")
    # 宽屏 >=76：按键提示出现
    segs = status_bar_segments(rt, width=100, elapsed_s=12.0)
    joined = " │ ".join(segs)
    if "Enter" not in joined:
        return _fail(f"宽屏缺按键提示: {_gbk_safe(joined)}")
    return _ok("三档宽度内容正确")


def check_enter_routing():
    """验证提交小函数：入队成功、输入框清空、空输入不入队。"""
    import queue as _q
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.history import InMemoryHistory
    from cli_layout import submit_input

    buf = Buffer(history=InMemoryHistory())
    buf.text = "hello /world"
    q = _q.Queue()
    submit_input(buf, q)
    if q.empty() or q.get_nowait() != "hello /world":
        return _fail("提交后队列里没有原文")
    if buf.text != "":
        return _fail(f"提交后输入框没清空: {buf.text!r}")
    # 空白输入不入队（老语义：空行直接 continue，不烧一轮）
    buf.text = "   "
    submit_input(buf, q)
    if not q.empty():
        return _fail("空白输入不该入队")
    return _ok("提交路由正确")


def check_cli_layout():
    """验证能构建出 Application（无头模式）+ 退出请求不炸。"""
    import os
    from types import SimpleNamespace as _NS
    from cli_layout import build_application, request_app_exit

    rt = _NS(
        agent=_NS(model="test-model"), workspace_cwd="D:/x",
        bg_count=0, event_pending=[], turn_active=False,
    )
    # 无头开关：verify 在管道里跑，拿不到真终端输出；不开这个开关
    # build_application 会（正确地）返回 None 走降级
    os.environ["CODEAGENT_LAYOUT_HEADLESS"] = "1"
    try:
        app = build_application(rt, input_queue=None, eof_sentinel=None)
    finally:
        os.environ.pop("CODEAGENT_LAYOUT_HEADLESS", None)
    if app is None:
        return _fail("build_application 返回 None")
    if app.full_screen:
        return _fail("必须是非全屏模式（full_screen=False）")
    request_app_exit(app)   # app 没在跑也不许炸（内部全吞）
    return _ok("Application 构建 + 安全退出请求正常")


def check_skin_engine():
    """验证皮肤引擎：四套内置、YAML overlay、热切换、样式覆盖。"""
    import tempfile
    from pathlib import Path
    import cli_skin

    skins = cli_skin.list_skins()
    names = {s["name"] for s in skins}
    for expect in ("default", "mono", "slate", "daylight"):
        if expect not in names:
            return _fail(f"内置皮肤缺 {expect}: {sorted(names)}")

    # YAML overlay：只覆盖一节，其余继承 default
    with tempfile.TemporaryDirectory() as td:
        skin_dir = cli_skin._skins_dir()
        fake = Path(td) / "verify_skin.yaml"
        fake.write_text(
            "name: verify_skin\ndescription: 临时验证\n"
            "colors:\n  prompt: \"#123456\"\nbranding:\n  prompt_symbol: \"»\"\n",
            encoding="utf-8",
        )
        # 借用户皮肤目录放一下（放不进去就跳过 overlay 段，保底不挂）
        overlay_tested = False
        try:
            skin_dir.mkdir(parents=True, exist_ok=True)
            target = skin_dir / "_verify_skin.yaml"
            target.write_text(fake.read_text(encoding="utf-8"), encoding="utf-8")
            try:
                skin = cli_skin.load_skin("verify_skin")
                overlay_tested = (
                    skin.get_color("prompt") == "#123456"
                    and skin.get_branding("prompt_symbol") == "»"
                    # 未覆盖的键继承 default
                    and skin.tool_prefix == "┊"
                )
            finally:
                target.unlink(missing_ok=True)
        except Exception:
            pass
        if not overlay_tested:
            return _fail("YAML overlay 覆盖/继承不对")

    # 热切换 + 样式覆盖
    old = cli_skin.get_active_skin_name()
    try:
        cli_skin.set_active_skin("mono")
        mono_style = cli_skin.get_pt_style_overrides()
        if "prompt" in mono_style:
            return _fail("mono 不该有 prompt 颜色覆盖")
        cli_skin.set_active_skin("default")
        if not cli_skin.get_pt_style_overrides().get("prompt", "").startswith("fg:#00aa88"):
            return _fail("default 的 prompt 颜色覆盖丢失")
        if cli_skin.hex_to_truecolor_ansi("#FFD700") != "\033[38;2;255;215;0m":
            return _fail("真彩 ANSI 换算错误")
        if cli_skin.hex_to_truecolor_ansi("") != "":
            return _fail("空颜色必须返回空 ANSI")
    finally:
        cli_skin.set_active_skin(old)
    return _ok("四套内置 + overlay + 热切换正常")


def check_pt_extras():
    """验证键盘协议别名：安装器生效、ANSI_SEQUENCES 表真的改了。"""
    import cli_pt_extras
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    from prompt_toolkit.keys import Keys

    n = cli_pt_extras.install_all()
    if n < 0:
        return _fail("安装器返回了负数")
    alt_enter = (Keys.Escape, Keys.ControlM)
    for seq in ("\x1b[13;2u", "\x1b[27;2;13~", "\x1b[13;5u", "\x1b[27;5;13~"):
        if ANSI_SEQUENCES.get(seq) != alt_enter:
            return _fail(f"{seq!r} 没映射到 Alt+Enter")
    for seq in ("\x1b[I", "\x1b[O"):
        if ANSI_SEQUENCES.get(seq) != Keys.Ignore:
            return _fail(f"{seq!r} 没映射到 Ignore")
    return _ok("Shift/Ctrl+Enter 别名 + 焦点噪声忽略生效")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    """主流程：建临时目录 → 按分类逐项跑检查并打印 PASS/FAIL → 汇总退出。

    返回：进程退出码，0 = 全部通过，1 = 有失败项。
    """
    print("=" * 60)
    print("  CodeAgent 复刻检查清单验证")
    print("=" * 60)

    tmp = Path(tempfile.mkdtemp(prefix="codeAgent_verify_"))
    checks = [
        ("基础对话", [
            ("agent 初始化", check_agent_initialization),
            ("工具定义加载", check_tool_definitions),
            ("terminal 工具", check_terminal_tool),
            ("read_file 工具", lambda: check_read_file_tool(tmp)),
            ("中断机制", check_interrupt),
        ]),
        ("记忆系统", [
            ("memory 工具写入", lambda: check_memory_tool_write(tmp)),
            ("MEMORY.md 创建", lambda: check_memory_persist(tmp)),
            ("记忆跨实例加载", lambda: check_memory_reload(tmp)),
        ]),
        ("技能系统", [
            ("/hello 触发", lambda: check_skill_trigger(tmp)),
            ("skills_list 工具", lambda: check_skills_list(tmp)),
            ("skill_view 工具", lambda: check_skill_view(tmp)),
            ("skill_manage 创建", lambda: check_skill_manage_create(tmp)),
            ("使用统计", lambda: check_usage_stats(tmp)),
        ]),
        ("会话存储", [
            ("sessions.db 创建", lambda: check_sessions_db(tmp)),  # 名字来自清单原文，实际查的是 .sessions/ 目录
            ("session_search", lambda: check_session_search(tmp)),
        ]),
        ("Curator", [
            ("status 状态", lambda: check_curator_status(tmp)),
            ("run --dry-run", lambda: check_curator_dry_run(tmp)),
            ("归档到 .archive/", lambda: check_curator_archive(tmp)),
            ("restore 恢复", lambda: check_curator_restore(tmp)),
        ]),
        ("委托", [
            ("subagent 同步", lambda: check_delegate_sync(tmp)),
            ("批量委托并行", lambda: check_delegate_batch(tmp)),
        ]),
        ("上下文压缩", [
            ("自动压缩", check_context_compress),
        ]),
        ("slash 注册表", [
            ("slash 注册表", check_slash_registry),
        ]),
        ("CLI 补全器", [
            ("CLI 补全器", check_cli_completer),
        ]),
        ("CLI 事件行", [
            ("CLI 事件行", check_event_lines),
        ]),
        ("CLI 骨架", [
            ("状态栏三档", check_status_bar_tiers),
            ("提交路由", check_enter_routing),
            ("布局构建", check_cli_layout),
        ]),
        ("CLI 皮肤/输入", [
            ("皮肤引擎", check_skin_engine),
            ("键盘别名", check_pt_extras),
        ]),
        ("CLI 流式框", [
            ("流式回答框", check_stream_box),
        ]),
    ]

    total = 0
    passed = 0
    failed = 0

    for category, items in checks:
        print(f"\n[ {category} ]")
        for name, fn in items:
            total += 1
            try:
                r = fn()
            except Exception as e:
                r = _fail(f"异常: {e}")
            if r["status"] == "PASS":
                passed += 1
                tag = "[ OK ]"
            else:
                failed += 1
                tag = "[FAIL]"
            detail = f" ({r['detail']})" if r["detail"] else ""
            print(f"  {tag}  {name}{detail}")

    print("\n" + "=" * 60)
    tag = "[ALL PASS]" if failed == 0 else "[HAS FAIL]"
    print(f"  {tag}  总计 {total}：{passed} 通过，{failed} 失败")
    print("=" * 60)

    # 收尾：删掉临时目录（删不掉也不报错，留着无妨）
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
