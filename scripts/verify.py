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


def check_rules_param_equiv(tmp):
    """验证 rules= 传参与无参调用判定一致（stat 风暴修复的前提）。"""
    import os
    os.environ["CODEAGENT_HOME"] = str(tmp / "rules_home")
    try:
        (tmp / "rules_home").mkdir(parents=True, exist_ok=True)
        (tmp / "rules_home" / "settings.json").write_text(
            '{"permissions": {"deny": ["terminal"]}}', encoding="utf-8")
        from agent import tool_permissions as tp
        tp.reset_rules_cache()
        rules = tp.load_tool_permission_rules()
        for name in ("terminal", "read_file"):
            with_param = tp.is_tool_denied(name, rules=rules)
            without_param = tp.is_tool_denied(name)
            if with_param != without_param:
                return _fail(f"{name}: 传参={with_param} 无参={without_param} 不一致")
        if not tp.is_tool_denied("terminal", rules=rules):
            return _fail("deny 规则没生效")
        return _ok("rules 传参与无参判定一致")
    finally:
        os.environ.pop("CODEAGENT_HOME", None)
        tp.reset_rules_cache()


def check_tool_names_cache():
    """验证工具名清单缓存：同键两调只扫一次 registry，generation 变化自动失效。

    大白话：get_tool_definitions 里"算出本轮有哪些工具名"这段（toolset 展开 +
    mcp 扫描 + deny 过滤）带缓存；registry.generation（每次登记/注销 +1）
    是失效信号。这里三连测：注册新工具能立刻看见、同参数连调两次结果一致
    且第二次不重扫、注销后立刻隐身。
    """
    from model_tools import get_tool_definitions
    from tools.registry import registry

    def _handler(args, **kw):
        return '{"ok": true}'

    # core 套餐是写死的静态名单，登记新工具不会进 resolve_toolset 的结果；
    # 真正"动态"的名字来源是 mcp__ 扫描分支——探针得起 mcp__ 开头的名字才测得到
    probe = "mcp__verify_cache__probe"
    registry.register(
        name=probe,
        toolset="mcp",
        schema={"type": "function", "function": {
            "name": probe,
            "description": "verify 专用探针",
            "parameters": {"type": "object", "properties": {}},
        }},
        handler=_handler,
        override=True,
    )
    try:
        # 数 registry.list_all 被扫了几次：同键第二次调用应命中缓存、不再扫
        orig_list_all = registry.list_all
        scans = [0]

        def _counting_list_all():
            scans[0] += 1
            return orig_list_all()

        registry.list_all = _counting_list_all
        try:
            names = [t["function"]["name"] for t in get_tool_definitions(["core", "mcp"])]
            names2 = [t["function"]["name"] for t in get_tool_definitions(["core", "mcp"])]
        finally:
            del registry.list_all  # 摘掉实例上的影子属性，恢复类里定义的原方法

        if probe not in names:
            return _fail("注册后清单没包含新工具（generation 失效没生效）")
        if names != names2:
            return _fail("两次结果不一致")
        if scans[0] > 1:
            return _fail(f"同键两调扫了 {scans[0]} 次 registry（缓存没生效）")

        # 注销让 generation 再 +1：缓存必须跟着失效，名单里不能再有探针
        registry.unregister(probe)
        names3 = [t["function"]["name"] for t in get_tool_definitions(["core", "mcp"])]
        if probe in names3:
            return _fail("注销后清单还残留旧工具（generation 失效没生效）")
        return _ok("名字解析缓存 + generation 失效正常")
    finally:
        registry.unregister(probe)


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


def check_compact_boundary_marker():
    """验证压缩边界占位统一带 [COMPACT_BOUNDARY] 大写前缀且可被一次性取走。

    旧 bug：生成方写小写 [compact_boundary]，恢复侧/持久化侧只认大写或
    旧前缀——边界从不落库，重启后全量载入撑爆上下文。
    """
    from agent.context_pipeline import (
        _build_compact_boundary, reactive_compact,
        take_last_compact_placeholder,
    )
    # 1) L4 边界标注块以大写前缀开头
    text = _build_compact_boundary("消息 1-50", "最近 30 条", transcript_path=None)
    if not text.startswith("[COMPACT_BOUNDARY]"):
        return _fail(f"边界标注缺大写前缀: {text[:40]!r}")
    # 2) reactive 占位同样带前缀，且成功后可被一次性取走
    fake_state = SimpleNamespace(reactive_last_at=0.0, reactive_count=0)
    msgs = [{"role": "system", "content": "s"}] + [
        {"role": "user", "content": f"m{i}"} for i in range(20)
    ]
    new_msgs, changed = reactive_compact(msgs, session_state=fake_state)
    if not changed:
        return _fail("reactive_compact 未触发")
    first = new_msgs[1]["content"]
    if not first.startswith("[COMPACT_BOUNDARY]"):
        return _fail(f"reactive 占位缺前缀: {first[:40]!r}")
    taken = take_last_compact_placeholder()
    if not taken or not taken.startswith("[COMPACT_BOUNDARY]"):
        return _fail(f"一次性取走失败: {str(taken)[:40]!r}")
    if take_last_compact_placeholder() is not None:
        return _fail("取走后应清空（一次性消费），第二次应返回 None")
    return _ok("边界占位前缀统一 + 暂存机制正常")


def check_compact_boundary_persist(tmp):
    """验证边界占位能落进会话库、恢复裁剪能按它工作（E2E 无 LLM 版）。

    用 reactive_compact 造出占位 → 走持久化函数落进假会话库 →
    按 cli 的恢复裁剪逻辑载入——验证「压缩→落库→恢复裁剪」全链。
    """
    from agent.context_pipeline import (
        reactive_compact, take_last_compact_placeholder,
        _persist_compact_marker,
    )
    from cli import _truncate_at_last_compact_boundary

    # 假会话库：只记录调用（不真写盘）
    recorded = []
    fake_store = SimpleNamespace(
        append_message=lambda sid, role, content, **kw: recorded.append(
            (sid, role, content)),
    )
    fake_state = SimpleNamespace(reactive_last_at=0.0, reactive_count=0)
    msgs = [{"role": "system", "content": "s"}] + [
        {"role": "user", "content": f"m{i}"} for i in range(20)
    ]
    new_msgs, changed = reactive_compact(msgs, session_state=fake_state)
    if not changed:
        return _fail("reactive_compact 未触发")
    placeholder_content = take_last_compact_placeholder()
    if not placeholder_content:
        return _fail("占位暂存被提前消费了")
    _persist_compact_marker(fake_store, "s1", placeholder_content)
    if len(recorded) != 1 or recorded[0][0] != "s1":
        return _fail(f"落库记录不对: {recorded}")

    # 恢复裁剪：库里 = 旧消息 + START + 边界占位 + 尾部新消息；
    # 尾部再塞一条 START，模拟「中断后又手动压缩过」的混合态
    db_msgs = (
        [{"role": "user", "content": "旧消息1"}, {"role": "assistant", "content": "旧答1"}]
        + [{"role": "user", "content": "[COMPACT_START] L4 开跑"}]
        + [{"role": "user", "content": recorded[0][2]}]
        + [{"role": "user", "content": "压缩后的新消息"},
           {"role": "user", "content": "[COMPACT_START] 又一次没跑完"}]
    )
    loaded = _truncate_at_last_compact_boundary(db_msgs)
    contents = [m["content"] for m in loaded]
    if "旧消息1" in contents:
        return _fail("边界之前的旧消息没被裁掉")
    if "压缩后的新消息" not in contents:
        return _fail("边界之后的新消息丢了")
    if any(c.startswith("[COMPACT_BOUNDARY]") for c in contents):
        return _fail("边界标记行应被剥掉（摘要正文保留）")
    if any(c.startswith("[COMPACT_START]") for c in contents):
        return _fail("孤立的 [COMPACT_START] 没被剔除（会当废话发给模型）")
    summary_body = recorded[0][2].replace("[COMPACT_BOUNDARY]\n", "", 1)
    if not any(summary_body in c for c in contents):
        return _fail("摘要正文丢了（边界消息可能被整条跳过——切片下标错位）")
    return _ok("压缩→落库→恢复裁剪全链正常（含 START 剔除）")


def check_summary_input_fidelity():
    """验证 L4 摘要输入保真：工具参数进场、结果头尾、offload 指针、ephemeral 过滤。

    摘要 prompt 要求「文件路径/错误消息逐字保留」，但旧版排版把工具调用
    参数整个丢掉、结果只留头 200 字符——摘要层是无米之炊。
    """
    from agent.context_compressor import _format_dialog_for_summary
    offload_json = json.dumps({
        "truncated": True, "orig_chars": 90000,
        "preview": "P" * 2000,
        "full_at": r"D:\home\.task_outputs\tool-results\call_abc.txt",
        "hint": "完整结果已落盘",
    }, ensure_ascii=False)
    msgs = [
        {"role": "user", "content": "帮我修 login.py 的 bug"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "function": {
                "name": "read_file",
                "arguments": json.dumps(
                    {"path": "D:/project/src/login.py", "offset": 100},
                    ensure_ascii=False),
            },
        }]},
        {"role": "tool", "content": "头部分析结论…\n" + "X" * 800 + "\nTraceback 错误在尾部"},
        {"role": "tool", "content": offload_json},
        {"role": "user", "content": "<background_tasks_running>3</background_tasks_running>",
         "_ephemeral": True},
    ]
    out = _format_dialog_for_summary(msgs)
    # 1) 工具参数进场（文件路径可见）
    if "D:/project/src/login.py" not in out:
        return _fail("工具调用参数没进摘要输入（文件路径丢了）")
    if "read_file" not in out:
        return _fail("工具名丢了")
    # 2) 结果头尾保留（错误在尾部）
    if "头部分析结论" not in out or "Traceback 错误在尾部" not in out:
        return _fail("工具结果头尾没都保留（尾部报错丢了）")
    # 3) offload 占位的 full_at 指针存活
    if "call_abc.txt" not in out:
        return _fail("offload 占位的 full_at 指针被截丢了")
    # 4) ephemeral 瞬时消息不进摘要
    if "background_tasks_running" in out:
        return _fail("ephemeral 瞬时消息混进了摘要输入")
    return _ok("摘要输入原料保真")


def check_token_estimation_and_threshold():
    """验证 CJK 感知估算、窗口对齐阈值、锚点口径三件套。

    旧版 chars÷3 对纯中文低估约一半（压得太晚→顶线 PTL→紧急截断丢信息）；
    L4 默认阈值 10 万对 64k 窗口的 DeepSeek 形同虚设。
    """
    from agent.context_compressor import estimate_message_tokens
    # 1) 纯中文 200 字：旧公式给 66，实际 120~300——新公式应 >=150（不再低估）
    cn = estimate_message_tokens([{"role": "user", "content": "汉" * 200}])
    if cn < 150:
        return _fail(f"中文仍低估: {cn}")
    # 2) 纯 ASCII 400 字：应约 100（÷4），不能暴涨
    en = estimate_message_tokens([{"role": "user", "content": "a" * 400}])
    if not (80 <= en <= 140):
        return _fail(f"ASCII 估算异常: {en}")
    # 3) 工具调用参数也计费
    with_args = estimate_message_tokens([{
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"arguments": "b" * 400}}],
    }])
    if with_args < 80:
        return _fail(f"tool_calls 参数没计进: {with_args}")
    # 4) 阈值对齐窗口：DeepSeek(64k) 封顶 0.9 窗口
    #    （_get_model_max_tokens 对 deepseek 返回 65536，0.9 倍 = 58982）
    from agent.context_pipeline import _effective_llm_compact_threshold
    ds = _effective_llm_compact_threshold({}, "deepseek-chat")
    if ds != 58982:
        return _fail(f"DeepSeek 阈值应为 58982（65536×0.9）: {ds}")
    big = _effective_llm_compact_threshold({}, "claude-x[1m]")
    if big != 700000:
        return _fail(f"1M 模型阈值应为 700000（保现行行为）: {big}")
    claude = _effective_llm_compact_threshold({}, "claude-3-5-sonnet")
    if claude != 100000:
        return _fail(f"200k 窗口模型维持配置默认 100000: {claude}")
    # 5) 锚点口径：DeepSeek 命名（prompt_tokens 已含缓存命中/未命中）不双计
    from agent import AIAgent
    agent = AIAgent(api_key="fake", model="t", enabled_toolsets=[])
    usage = SimpleNamespace(
        prompt_tokens=8000,
        prompt_cache_hit_tokens=5000,
        prompt_cache_miss_tokens=3000,
        completion_tokens=10,
    )
    resp = SimpleNamespace(usage=usage, model="deepseek-chat")
    agent._record_llm_usage(resp, sent_message_count=10)
    anchor = agent._last_usage_anchor
    if not anchor or anchor[1] != 8000:
        return _fail(f"DeepSeek 锚点应取 prompt_tokens=8000（旧版会双计成 16000）: {anchor}")
    return _ok("CJK 估算 + 窗口对齐 + 锚点口径正常")


def check_dispatch_output_cap(tmp):
    """验证 dispatch 层统一输出封顶：没自觉接 offload 的工具也会被兜底。

    造一个返回 10 万字符的临时工具直接走 handle_function_call 总出口，
    结果应自动落盘（含 preview/full_at），文件真实存在。
    """
    import asyncio as _aio
    from tools.registry import registry
    from model_tools import handle_function_call

    def _big_handler(args, **kw):
        return "X" * 100000

    # 临时工具用完就注销（finally 兜底）——不注销的话它会留在全局注册表
    # 里污染后续检查项（工具集多了仓名工具，schema 白花 token）。
    try:
        registry.register(
            name="verify_big_output",
            toolset="core",
            schema={
                "type": "function",
                "function": {
                    "name": "verify_big_output",
                    "description": "verify 专用：返回超长文本",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            handler=_big_handler,
            override=True,
        )
        result = _aio.run(handle_function_call(
            "verify_big_output", {},
            tool_call_id="verify_cap_call_1",
            codeagent_home=tmp,
            config={},
        ))
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return _fail(f"结果不是 JSON: {result[:120]!r}")
        if not data.get("truncated") or "full_at" not in data:
            return _fail(f"大输出没被统一封顶: {str(data)[:150]!r}")
        if not Path(data["full_at"]).exists():
            return _fail(f"落盘文件不存在: {data['full_at']}")
        if "X" * 50 not in data.get("preview", ""):
            return _fail("预览内容不对")
        return _ok(f"统一封顶生效（{data['orig_chars']} 字符落盘）")
    finally:
        registry.unregister("verify_big_output")


def check_delegate_offload(tmp):
    """验证子代理结果先落盘再摘要：父代理想看细节时有 full_at 可读回。"""
    from tools.delegate_tool import _offload_child_result, _attach_full_result_pointer
    result = "子代理跑了很久的调研结果" + "Y" * 3000
    kwargs = {
        "tool_call_id": "verify_delegate_call_1",
        "codeagent_home": tmp,
        "task_id": "t-9",
    }
    off = _offload_child_result(result, kwargs)
    if not off:
        return _fail("没有产出 offload 占位")
    data = json.loads(off)
    if not data.get("full_at") or data.get("orig_chars") != len(result):
        return _fail(f"占位字段不对: {str(data)[:150]!r}")
    if not Path(data["full_at"]).exists():
        return _fail(f"原文没落盘: {data['full_at']}")
    final = _attach_full_result_pointer("[摘要] 调研完成，结论X", off)
    if "full_at" not in final or data["full_at"] not in final:
        return _fail(f"摘要没带上找回指针: {final[-120:]!r}")
    return _ok("子代理结果落盘+指针正常")


def check_memory_retrieval_fallback():
    """验证记忆检索兜底：关键词匹配 + ID 纠错（aux LLM 单点时的保底）。"""
    from agent.memory_retriever import keyword_fallback_ids, correct_memory_id
    index_text = (
        "## 项目记忆\n"
        "- [project] codeagent 压缩策略 (proj#abc123): L4 分层压缩\n"
        "- [user] 用户偏好中文回复 (user#def456): 保姆语言\n"
        "- [reference] gitee 仓库地址 (ref#xyz789): https://gitee.com/x\n"
    )
    ids = keyword_fallback_ids("压缩策略 怎么配置", index_text, max_results=2)
    if "proj#abc123" not in ids:
        return _fail(f"关键词兜底没命中: {ids}")
    ids2 = keyword_fallback_ids(
        "压缩策略", index_text, exclude_ids={"proj#abc123"},
    )
    if "proj#abc123" in ids2:
        return _fail("exclude_ids 没生效")
    # ID 纠错：LLM 抄掉尾部两位也能救回
    fixed = correct_memory_id("proj#abc1", index_text)
    if fixed != "proj#abc123":
        return _fail(f"前缀纠错失败: {fixed}")
    fixed2 = correct_memory_id("PROJ#ABC123", index_text)
    if fixed2 != "proj#abc123":
        return _fail(f"大小写归一纠错失败: {fixed2}")
    if correct_memory_id("totally#nope", index_text) is not None:
        return _fail("救不回的 ID 应返回 None")
    # 真实索引行是 markdown 链接（.memory/{topic}.jsonl#{uid}，见
    # memory_store._entry_link），兜底必须抠出裸 {topic}#{uid}——
    # retrieve_relevant 的返回值和 MemoryStore.get() 认的都是裸 ID
    real_index = (
        "- [project] codeagent 压缩策略 (.memory/project.jsonl#abc123): L4 分层压缩 [age: 3d]\n"
        "- [reference] gitee 地址 (.memory/projects/x/reference.jsonl#xyz789): url [age: 5d]\n"
    )
    rids = keyword_fallback_ids("压缩策略", real_index)
    if rids[:1] != ["project#abc123"]:
        return _fail(f"真实链接格式没抠出裸 ID: {rids}")
    fixed3 = correct_memory_id("reference#xyz78", real_index)
    if fixed3 != "reference#xyz789":
        return _fail(f"真实链接格式前缀纠错失败: {fixed3}")
    return _ok("关键词兜底 + ID 纠错正常")


def check_memory_injection_wiring():
    """验证注入链兜底接线：LLM 空手→关键词顶上；抄错 ID→纠错救回；计数器工作。"""
    import asyncio as _aio
    from agent import memory_injection as mi

    class _FakeStore:
        def full_index_text_with_age(self):
            return (
                "- [project] codeagent 压缩策略 (proj#abc123): L4 分层压缩\n"
                "- [user] 用户偏好 (user#def456): 中文回复\n"
            )

        def get(self, mid):
            if mid == "proj#abc123":
                return SimpleNamespace(
                    type="project", name="压缩策略", body="L4 分层压缩",
                    updated_at=None,
                )
            return None

    class _DeadLLM:
        async def chat_completions(self, *a, **kw):
            raise RuntimeError("aux LLM 挂了")

    mi.reset_injection_cache()
    before = dict(mi._retrieval_stats)
    # 注意：query 得带分隔符（空格）——_query_tokens 按空白/标点切词、
    # 不做 CJK 分词，整串"压缩策略怎么配置"会成一个无法命中索引行的
    # 长 token（brief 原文无空格，此处按意图修正为可分词写法）
    msg = _aio.run(mi.build_relevant_memories_message(
        query="压缩策略 怎么配置",
        memory_store=_FakeStore(),
        aux_llm_router=_DeadLLM(),
        max_results=3,
        surfaced=set(),
    ))
    if msg is None:
        return _fail("LLM 挂了+兜底在场时不该返回 None")
    if "proj#abc123" not in msg["content"] and "压缩策略" not in msg["content"]:
        return _fail(f"兜底没把相关记忆注入: {str(msg)[:150]!r}")
    after = mi._retrieval_stats
    if after["requests"] <= before["requests"]:
        return _fail("遥测计数没涨")
    return _ok(f"兜底接线+遥测正常（requests={after['requests']}）")


def check_pre_send_guard():
    """验证发送前预检判定：混合估算超窗口 90% 才要求再压一次。"""
    from agent.context_pipeline import needs_pre_send_compaction
    big = [{"role": "user", "content": "汉" * 200000}]  # ≈20 万 token
    if not needs_pre_send_compaction(big, None, "deepseek-chat"):
        return _fail("64k 窗口 + 20 万 token 应触发预检")
    small = [{"role": "user", "content": "你好"}]
    if needs_pre_send_compaction(small, None, "deepseek-chat"):
        return _fail("小上下文不该触发预检")
    # 1M 窗口下 20 万 token 不触发
    if needs_pre_send_compaction(big, None, "claude-x[1m]"):
        return _fail("1M 窗口 + 20 万 token 不该触发")
    # 坏锚点不炸（fail-open 全量估算）
    if not needs_pre_send_compaction(big, ("x", "y"), "deepseek-chat"):
        return _fail("坏锚点应回退全量估算并触发")
    return _ok("发送前预检判定正常")


def check_loop_host():
    """验证事件循环宿主十件套：run_async 阻塞等结果且跑在宿主循环上、
    异常穿透、submit 后台任务执行、run_turn 栅栏取消回合遗留 task、
    submit 豁免不被栅栏误杀、submit 出生即豁免（pending 集）防栅栏竞态误杀、
    run_async(exempt_from_fence=True) 跨回合长活不被栅栏误杀、
    submit 的 contextvars 从调用方线程传播进后台任务（create_task(context=)）、
    run_async 超时抛 TimeoutError 且协程被取消、stop 后拒绝复活
    （run_async 抛 RuntimeError，submit 返回已设异常的 future）。"""
    import asyncio
    from agent.loop_host import loop_host, cancel_current_turn

    where = {}
    async def _coro():
        where["loop"] = asyncio.get_running_loop()
        return 42
    if loop_host.run_async(_coro()) != 42:
        return _fail("run_async 结果不对")
    if where.get("loop") is not loop_host.loop:
        return _fail("协程没跑在宿主循环上")

    async def _boom():
        raise ValueError("boom")
    try:
        loop_host.run_async(_boom())
        return _fail("异常没穿透")
    except ValueError:
        pass

    done = {}
    async def _bg():
        done["yes"] = True
    loop_host.submit(_bg(), name="verify").result(timeout=5)
    if not done.get("yes"):
        return _fail("submit 没执行")

    # 栅栏：回合内裸 create_task 的遗留被取消
    fate = {}
    async def _leaked():
        try:
            await asyncio.sleep(30)
            fate["leaked"] = "done"
        except asyncio.CancelledError:
            fate["leaked"] = "cancelled"
    async def _turn():
        asyncio.ensure_future(_leaked())
        await asyncio.sleep(0.05)
        return "ok"
    if loop_host.run_turn(_turn()) != "ok":
        return _fail("run_turn 结果不对")
    if fate.get("leaked") != "cancelled":
        return _fail(f"栅栏没取消遗留任务: {fate}")

    # 豁免：submit 的慢后台任务在另一回合栅栏后仍活着
    bg_alive = {}
    async def _slow_bg():
        try:
            await asyncio.sleep(0.3)
            bg_alive["ok"] = True
        except asyncio.CancelledError:
            bg_alive["cancelled"] = True
    bgfut = loop_host.submit(_slow_bg(), name="verify-slow")
    async def _turn2():
        await asyncio.sleep(0.05)
        return "ok2"
    loop_host.run_turn(_turn2())
    bgfut.result(timeout=5)
    if bg_alive.get("cancelled"):
        return _fail("submit 的后台任务被回合栅栏误杀")

    # C-1 竞态用例（确定性复现）：回合挂起等事件期间排两个回调——
    # ①唤醒回调（让回合的最后一步入队）②submit 的建任务回调（后台
    # task 创建、首步排在回合收尾之后）。栅栏落下时后台 task「已创建
    # 未启动」：注册进 _bg_tasks 要等首步跑，此刻还没轮到——没有
    # pending 豁免集时它会被栅栏静默取消（协程体一行不跑、WARNING
    # 都不打）。两个回调都从回合协程内发起，入队顺序在循环线程上
    # 百分百确定（唤醒先、submit 后，否则首步会先于回合收尾执行）。
    c1_fate = {}
    c1_holder = []
    async def _c1_bg():
        try:
            await asyncio.sleep(0.2)
            c1_fate["r"] = "done"
        except asyncio.CancelledError:
            c1_fate["r"] = "cancelled"
    async def _c1_turn():
        ev = asyncio.Event()
        loop_host.call_soon_threadsafe(ev.set)  # ① 唤醒回调先入队
        c1_holder.append(loop_host.submit(_c1_bg(), name="verify-c1"))  # ② 建任务回调紧随
        await ev.wait()  # 回合挂起；唤醒后的下一步就是收尾（栅栏落下）
        return "ok3"
    if loop_host.run_turn(_c1_turn()) != "ok3":
        return _fail("C-1 用例回合结果不对")
    try:
        c1_holder[0].result(timeout=5)
    except BaseException:
        pass  # 未修复时后台任务出生即被栅栏取消，future 以 CancelledError 收场
    if c1_fate.get("r") == "cancelled":
        return _fail("C-1 竞态复现：已创建未启动的后台任务被栅栏误杀")
    if c1_fate.get("r") != "done":
        return _fail(f"C-1 竞态复现：后台任务没跑完（出生即被栅栏取消）: {c1_fate}")

    # cancel_current_turn 空转不炸（没有回合在跑）
    cancel_current_turn()

    # 豁免回归：exempt_from_fence 的 run_async 长活不被回合栅栏杀——
    # 后台线程发起、活过一次回合栅栏（0.25s 长活 vs 0.08s 短回合），
    # 修好前它会被栅栏当回合遗留取消（CancelledError 还是 BaseException，
    # 会穿透调用方全部 except Exception fail-open 防线）。
    # 时序确定性：runner 线程先等 _turn3 开门再提交——开门发生在栅栏
    # before 快照之后（同一同步段内），长活铁定「生于回合期间」，既不在
    # before 也不在 _bg_tasks，只靠 pending 豁免集活命（不握手的话线程
    # 可能抢在快照前提交，测试退化为无条件通过、测不到栅栏）
    ex_fate = {}
    async def _ex_long():
        try:
            await asyncio.sleep(0.25)
            ex_fate["r"] = "done"
        except asyncio.CancelledError:
            ex_fate["r"] = "cancelled"
    import threading as _th2
    _box = {}
    _gate = _th2.Event()
    def _runner():
        _gate.wait(timeout=5)  # 等回合真正开跑（快照已拍）再提交
        _box["r"] = loop_host.run_async(_ex_long(), exempt_from_fence=True)
    _t2 = _th2.Thread(target=_runner, daemon=True)
    _t2.start()
    async def _turn3():
        _gate.set()  # 开门：runner 此刻提交的长活必然生于快照之后
        await asyncio.sleep(0.08)
        return "ok4"
    if loop_host.run_turn(_turn3()) != "ok4":
        return _fail("exempt 用例回合结果不对")
    _t2.join(timeout=5)
    if ex_fate.get("r") == "cancelled":
        return _fail("exempt 的 run_async 被回合栅栏误杀")
    if ex_fate.get("r") != "done" or _box.get("r") is not None:
        return _fail(f"exempt 长活未跑完: {ex_fate} {_box}")

    # 第八件（I-1 传播断言）：submit 的后台任务要读到调用方线程 set 的
    # ContextVar。修好前 Task 拷的是宿主循环线程的上下文（协程对象不
    # 绑定 context，ctx.run 白做），调用方 set 的变量全丢——worktree
    # 会话的后台记忆提取会拿到主进程目录、写错项目分区
    import contextvars as _cv
    _probe_var = _cv.ContextVar("verify_loop_host_probe", default="unset")
    _got = {}
    async def _ctx_bg():
        _got["v"] = _probe_var.get()
    def _setter_thread():
        _probe_var.set("from-caller")
        loop_host.submit(_ctx_bg(), name="verify-ctx").result(timeout=5)
    _ct = _th2.Thread(target=_setter_thread, daemon=True)
    _ct.start(); _ct.join(timeout=5)
    if _got.get("v") != "from-caller":
        return _fail(f"contextvars 没传播到后台任务: {_got}")

    # 第九件（timeout 取消语义）：run_async 超时抛 TimeoutError 且协程
    # 真被取消——不然调用方都超时走了，协程还赖在常驻循环上白跑到天荒地老
    t_fate = {}
    async def _t_long():
        try:
            await asyncio.sleep(5)
            t_fate["r"] = "done"
        except asyncio.CancelledError:
            t_fate["r"] = "cancelled"
    try:
        loop_host.run_async(_t_long(), timeout=0.1)
        return _fail("timeout 没抛 TimeoutError")
    except TimeoutError:
        pass
    import time
    _t0 = time.monotonic()
    while t_fate.get("r") is None and time.monotonic() - _t0 < 2:
        time.sleep(0.02)
    if t_fate.get("r") != "cancelled":
        return _fail(f"timeout 后协程没被取消: {t_fate}")

    # 第十件（stop 拒绝复活）：stop 后 straggler 线程再来调 run_async/submit
    # 必须被拒——旧版会悄悄拉起一个新循环（用独立实例验证，不碰全局单例）
    from agent.loop_host import AgentLoopHost
    h2 = AgentLoopHost()
    async def _smoke():
        return 1
    if h2.run_async(_smoke()) != 1:
        return _fail("独立实例基础语义坏了")
    h2.stop()
    try:
        h2.run_async(_smoke())
        return _fail("stop 后 run_async 没拒绝")
    except RuntimeError:
        pass
    bgf = h2.submit(_smoke(), name="verify-stopped")
    try:
        bgf.result(timeout=2)
        return _fail("stop 后 submit 的 future 不该成功")
    except RuntimeError:
        pass
    return _ok("loop_host 语义十件套正常（含 timeout 取消与 stop 拒绝）")


def check_split_symbol_surface():
    """验证拆分后符号面不缩：agent root 的旧符号照常可用，新模块独立可导。"""
    import agent
    from agent import AIAgent, _spawn_detached, LoopExitReason, _drop_leading_system
    from agent import (
        _build_goal_continue_message, _build_channel_injection, _build_mail_injection,
    )
    import agent.ephemeral_inject as ei
    if ei.LoopExitReason is not LoopExitReason:
        return _fail("LoopExitReason 双导不一致")
    if not callable(_spawn_detached):
        return _fail("_spawn_detached 丢了")
    # 五个新拆模块全部可独立 import（turn_observer 借助模块级 hoist 由
    # import agent 传递覆盖，这里显式断言防退回惰性）
    import agent.ephemeral_inject  # noqa: F401
    import agent.tool_batch_summary  # noqa: F401
    import agent.usage_accounting  # noqa: F401
    import agent.skill_learning.turn_observer  # noqa: F401
    from agent.reflection import trigger_reflection_async as _tra
    if not callable(_tra):
        return _fail("trigger_reflection_async 不可用")
    # 拆分二期块 A：llm_retry 的五个新自由函数（max_tokens 升级/续写恢复
    # 四件 + 长退避心跳），agent root 模块级引入防退回惰性
    from agent.llm_retry import (
        try_escalate_max_tokens, merge_usage_tokens,
        recover_output_truncation, merge_continuation_response,
        llm_retry_heartbeat,
    )
    if not all(callable(f) for f in (
        try_escalate_max_tokens, merge_usage_tokens,
        recover_output_truncation, merge_continuation_response,
        llm_retry_heartbeat,
    )):
        return _fail("块 A 五自由函数有不可调用者")
    # 拆分二期块 B：流式调用心脏（call_llm_streaming + 墓碑清理）拆到
    # llm_streaming，agent root 模块级引入防退回惰性
    import agent.llm_streaming  # noqa: F401
    from agent.llm_streaming import (
        call_llm_streaming, discard_partial_stream_state,
    )
    if not all(callable(f) for f in (
        call_llm_streaming, discard_partial_stream_state,
    )):
        return _fail("块 B 二自由函数有不可调用者")
    # WS transport 不再自建循环（set_event_loop 会污染调用线程的循环视图）
    import inspect
    import agent.mcp_client as _mc
    _src = inspect.getsource(_mc)
    if "set_event_loop" in _src or "new_event_loop" in _src:
        return _fail("mcp_client 仍有自建事件循环残留")
    if "loop_host.run_async" not in _src:
        return _fail("WS transport 没走 loop_host")
    return _ok("拆分符号面契约成立")


def check_anthropic_usage_fields():
    """验证 Anthropic 响应包装后的 usage 带 cache 字段（缓存记账/锚点口径依赖）。"""
    from types import SimpleNamespace as NS
    from agent.llm_client import AnthropicClient
    fake = NS(
        content=[NS(type="text", text="hi")],
        usage=NS(input_tokens=10, output_tokens=5,
                 cache_read_input_tokens=7, cache_creation_input_tokens=3),
    )
    client = AnthropicClient.__new__(AnthropicClient)  # 不跑 __init__（不碰网络）
    wrapped = client._wrap_response(fake)
    u = wrapped.usage
    if getattr(u, "cache_read_input_tokens", None) != 7:
        return _fail(f"cache_read_input_tokens 丢失: {u}")
    if getattr(u, "cache_creation_input_tokens", None) != 3:
        return _fail(f"cache_creation_input_tokens 丢失: {u}")
    return _ok("Anthropic usage 缓存字段齐全")


def check_session_append_perf(tmp):
    """验证 append 不再 O(n²)：连写 200 条 turn_index 连续正确、文件行数
    吻合、index 卡片去抖中途真的延迟、flush 后最终一致。"""
    import json as _j
    from agent.session_store import SessionStore
    store = SessionStore(tmp / "perf_sessions")
    sid = store.create_session(title="perf", model="t")
    # 去抖延迟断言：另开一个会话连写 3 条，磁盘卡片应停在首条 flush 的
    # 状态（首条触发 _index_last_flush=0.0 的立即写盘，之后 3 条在窗口内
    # 只进内存）——若实现退化为每条都写盘，这里会 FAIL
    sid_d = store.create_session(title="debounce", model="t")
    for i in range(3):
        store.append_message(sid_d, "user", f"d{i}")
    disk_d = _j.loads((tmp / "perf_sessions" / "index.json").read_text(encoding="utf-8"))
    entry_d = next(s for s in disk_d["sessions"] if s["id"] == sid_d)
    if entry_d.get("message_count") != 1:
        return _fail(f"去抖没延迟：3 条后磁盘计数应为 1（首条 flush），实际 {entry_d.get('message_count')}")
    # 交替 user/assistant 各 100 条：user 各开新轮
    for i in range(100):
        store.append_message(sid, "user", f"问{i}")
        store.append_message(sid, "assistant", f"答{i}")
    msgs = store._read_session_msgs(sid)
    if len(msgs) != 200:
        return _fail(f"行数不符: {len(msgs)}")
    turns = [m.get("turn_index") for m in msgs]
    # 第 1 条 user 是第 1 轮；第 i 对 user 是第 i 轮，同轮 assistant 沿用
    if turns[0] != 1 or turns[1] != 1 or turns[198] != 100 or turns[199] != 100:
        return _fail(f"turn_index 序列不对: {turns[:4]}...{turns[-4:]}")
    # 去抖：磁盘卡片允许落后，但 flush 后必须追平
    store.flush_index()
    disk = _j.loads((tmp / "perf_sessions" / "index.json").read_text(encoding="utf-8"))
    entry = next(s for s in disk["sessions"] if s["id"] == sid)
    if entry.get("message_count") != 200:
        return _fail(f"flush 后磁盘卡片计数不追平: {entry.get('message_count')}")
    # 二次实例（模拟重启）从磁盘引导 turn_index：接着写 user 应开 101 轮
    store2 = SessionStore(tmp / "perf_sessions")
    tid = store2.append_message(sid, "user", "重启后再问")
    m2 = store2._read_session_msgs(sid)[-1]
    if m2.get("turn_index") != 101:
        return _fail(f"重启引导 turn_index 不对: {m2.get('turn_index')}")
    return _ok("append 线性化 + 卡片去抖 + 重启引导全正常")


def check_session_field_passthrough(tmp):
    """验证 pinned/timestamp 落库往返：append 带参 → get_messages 带回。"""
    from agent.session_store import SessionStore
    store = SessionStore(tmp / "passthrough")
    sid = store.create_session(title="pt", model="t")
    store.append_message(sid, "user", "钉住这条", pinned=True)
    store.append_message(sid, "assistant", "普通回复")
    msgs = store.get_messages(sid)
    if msgs[0].get("pinned") is not True:
        return _fail(f"pinned 没往返: {msgs[0]}")
    if not msgs[0].get("timestamp"):
        return _fail(f"timestamp 没透传: {msgs[0]}")
    if "pinned" in msgs[1]:
        return _fail(f"未钉住的消息不该带 pinned: {msgs[1]}")
    return _ok("pinned/timestamp 往返正常")


def check_msgs_cache_lru(tmp):
    """验证消息缓存有淘汰：加载 20 个会话后容量不超过上限（防常驻内存无限涨）。"""
    from agent.session_store import SessionStore, _MSGS_CACHE_CAP
    store = SessionStore(tmp / "lru_sessions")
    for i in range(_MSGS_CACHE_CAP + 4):
        sid = store.create_session(title=f"s{i}", model="t")
        store.append_message(sid, "user", f"hello {i}")
        store.get_messages(sid)  # 触发加载进缓存
    if len(store._msgs_cache) > _MSGS_CACHE_CAP:
        return _fail(f"缓存无淘汰: {len(store._msgs_cache)} > {_MSGS_CACHE_CAP}")
    return _ok(f"LRU 生效（容量 {len(store._msgs_cache)} ≤ {_MSGS_CACHE_CAP}）")


# ---------------------------------------------------------------------------
# 上下文压缩
# ---------------------------------------------------------------------------

def check_context_compress():
    """验证上下文压缩：消息条数超阈值时新管线会把历史压短（LLM 用假实现）。

    这里把触发阈值调得很低，确保压缩一定会发生；再给一个记录型的假
    会话库，断言 L4 成功后 [COMPACT_BOUNDARY] 边界真的落了库——
    不落库的话压完重启恢复全量载入，压缩等于白压（落库接线回归）。

    返回：PASS/FAIL 结果。
    """
    from agent.context_pipeline import compress_if_needed, CompressionSessionState
    from config import DEFAULT_CONFIG

    # 假 LLM 客户端：_summarize_conversation 认的是异步 chat_completions
    # 接口（fork 前缀 / 独立调用两条路都走它），返回固定摘要文本
    async def fake_chat_completions(messages, **kw):
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="总结内容：这是测试摘要。")
            )]
        )

    client = SimpleNamespace(chat_completions=fake_chat_completions)

    msgs = [{"role": "system", "content": "sys"}]
    for i in range(30):
        # 内容塞长一点：被摘要段必须比摘要占位大，L4 收敛检查才过得去
        msgs.append({"role": "user", "content": f"消息 {i} " + "细节" * 20})
        msgs.append({"role": "assistant", "content": f"回复 {i} " + "答复" * 20})

    ctx_cfg = dict(DEFAULT_CONFIG.get("context", {}))
    # 把触发阈值调低到必触发，否则样例对话不够长压不动
    ctx_cfg["snip_message_threshold"] = 50
    # token 阈值压到 1：强制走 L4（假 LLM 摘要）成功路径，才验得到边界落库
    ctx_cfg["llm_compact_token_threshold"] = 1
    # agent_home=None 落不了 transcript 快照，干脆关掉省 WARNING 噪音
    ctx_cfg["transcript_enabled"] = False
    state = CompressionSessionState()
    # 假会话库：只记录 append_message 的调用，最后断言边界标记真落了库
    recorded = []

    def _fake_append(sid, role, text, **kw):
        recorded.append((sid, role, text))

    fake_store = SimpleNamespace(append_message=_fake_append)
    new_msgs, compressed, _compacted = asyncio.run(compress_if_needed(
        msgs,
        llm_client=client,
        model="deepseek-chat",
        config=ctx_cfg,
        session_state=state,
        agent_home=None,
        session_id="verify",
        session_store=fake_store,
    ))
    if not (compressed and len(new_msgs) < len(msgs)):
        return _fail("未压缩")
    boundaries = [t for (_s, _r, t) in recorded if t.startswith("[COMPACT_BOUNDARY]")]
    if not boundaries:
        return _fail(f"L4 压缩成功但边界没落库（会话库记录 {len(recorded)} 条）")
    return _ok(
        f"{len(msgs)} → {len(new_msgs)} 条，边界已落库（共 {len(recorded)} 条标记）"
    )


def check_compress_profile():
    """验证单遍 profile 统计正确（触发判定改吃它的前提）。"""
    from agent.context_pipeline import _profile_messages
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "问"},
        {"role": "assistant", "content": "答", "_timestamp": 111},
        {"role": "tool", "content": "T" * 300},
        {"role": "tool", "content": "T" * 500},
        {"role": "assistant", "content": "答2", "_timestamp": 222},
    ]
    p = _profile_messages(msgs)
    if p["msg_count"] != 5:
        return _fail(f"msg_count 不对（system 不计）: {p['msg_count']}")
    if p["total_chars"] != 1 + 1 + 300 + 500 + 2:
        return _fail(f"total_chars 不对: {p['total_chars']}")
    if p["max_tool_chars"] != 500:
        return _fail(f"max_tool_chars 不对: {p['max_tool_chars']}")
    if p["last_assistant_ts"] != 222:
        return _fail(f"last_assistant_ts 不对: {p['last_assistant_ts']}")
    empty = _profile_messages([{"role": "system", "content": "s"}])
    if empty["msg_count"] != 0 or empty["last_assistant_ts"] is not None:
        return _fail(f"空会话 profile 不对: {empty}")
    return _ok("profile 单遍统计正确")


def check_low_threshold_offload_equivalence(tmp):
    """低 offload 阈值下 L2/L2.5/L2.6 守卫仍放行现场重算（R4T6 守卫回归门）。

    背景：压缩管线的各层触发判定吃 profile 统计量（省扫描），但强制
    落盘对短消息（内容 ≤ 预览长度）是「全文 + JSON 包装」替换、净变长
    ~270 字符/条——低 offload 阈值配置下 profile 会低估现场总量。
    本门防的是「profile 早退门跳过本应触发的现场重算」（R4T6 修复的
    守卫回归）：L2.6 的守卫臂（or c0 or c_freeze or c2 or c_per_msg）
    必须在前层动过消息时放行现场重算，否则 L2 折叠后现场总量超预算
    无人管（等价破坏）。

    为什么预算是 80000（2026-09 实测数字）：40 条 1900 字符工具结果，
    profile 总量 76001+slack(16×82=1312)≈77.3k；L2 折 37 条（留最近
    3 条保护圈）后现场总量 ~84.5k。80000 正好卡在 profile 低估与现场
    真值之间——
      守卫在场：c2=True 放行现场重算 → L2.6 补折 3 条保护圈，占位 40
        处、现场总量 86630 ≤ 80000×1.1（折短消息净变长，压不回 80000
        整，×1.1 富余盖住 40 条落盘 JSON 包装的开销）；
      守卫被删：只剩 profile 臂，77.3k < 80k 不触发 → 只折 L2 的 37
        条、现场总量 85.8k 超预算无人管 → 占位 37 < 38 断言 FAIL。
    删守卫会 FAIL 已本地实测（临时摘掉守卫臂跑同场景：占位 40 → 37）。
    注意总量断言单独并不带电（守卫被删时 85.8k 也在 1.1 内）——本门
    的电来自占位计数那条臂，两个条件同时断言缺一不可。

    参数：
        tmp  临时目录 Path（落盘文件写在 tmp/.task_outputs/ 下）

    返回：PASS/FAIL 结果。
    """
    from agent.context_pipeline import (
        compress_if_needed, CompressionSessionState, reset_offload_decisions,
    )
    import time as _t

    # 冻结决策表是模块级全局（跨检查共享），先清空保证本检查从零开始
    # （别的检查若碰巧用过 c0..c39 这类 id，照抄旧预览会搅乱断言）
    reset_offload_decisions()

    # 40 条 1900 字符工具结果：低阈值（1000）下 L2 会把它们折成
    # 「全文+包装」（净变长），L2.6 的守卫臂必须放行现场重算而不是
    # 被 profile 统计跳过（llm_client=None 时 L4 不触发，验的是无损层）
    msgs = [{"role": "system", "content": "s"}]
    msgs.append({"role": "user", "content": "go", "_timestamp": _t.time()})
    for i in range(40):
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": f"c{i}", "function": {
                         "name": "t", "arguments": "{}"}}],
                     "_timestamp": _t.time()})
        msgs.append({"role": "tool", "content": "X" * 1900,
                     "tool_call_id": f"c{i}", "_timestamp": _t.time()})
    budget = 80000
    new_msgs, changed, _ = asyncio.run(compress_if_needed(
        msgs, llm_client=None, model="deepseek-chat",
        config={"output_offload_threshold": 1000,
                "tool_result_total_budget": budget},
        session_state=CompressionSessionState(),
        agent_home=tmp, session_id="verify-low-th",
    ))
    # 断言 (a)：必须变化。落盘发生了但 changed=False 是记账 bug（调用方
    # 靠 changed 把新消息同步回对话历史，False 等于白折）——旧版
    # 「not changed 且占位在场也算过」的宽松象限已删
    if not changed:
        return _fail("无损层折叠了却不报 changed（记账 bug，新历史不会被同步回对话）")
    tool_msgs = [m for m in new_msgs if m.get("role") == "tool"]
    n_placeholders = sum(
        str(m.get("content", "")).count('"full_at"') for m in tool_msgs)
    tool_total = sum(len(str(m.get("content", ""))) for m in tool_msgs)
    if n_placeholders == 0:
        return _fail("changed 但没有落盘占位，行为异常")
    # 断言 (b)：占位条数。实测 40（L2 折 37 + L2.6 补折 3 条保护圈）；
    # >= 38 是对守卫敏感的形状——守卫被删时 L2.6 不触发、只剩 37 处
    if n_placeholders < 38:
        return _fail(
            f"落盘占位仅 {n_placeholders} 处（应 40：L2 折 37 + L2.6 补折"
            " 3 条保护圈）——守卫回归：profile 早退门跳过了本应触发的现场重算")
    # 断言 (c)：现场总量压回预算×1.1 内（实测 86630）。与 (b) 同时成立
    # 才算过门——这条单独不带电（守卫被删时 85.8k 也过），撑住的是
    # 「现场重算真把总量管住了」的语义
    if tool_total > budget * 1.1:
        return _fail(
            f"现场总量 {tool_total} 超预算×1.1（{int(budget * 1.1)}）无人管")
    return _ok(
        f"占位 {n_placeholders} 处、现场总量 {tool_total} ≤ {int(budget * 1.1)}"
        "（守卫敏感形状过门）"
    )


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
    """验证事件行渲染器：claude code 风格 ● 头行 + ⎿ 结果块 + 带行号 diff。"""
    import json as _json
    import cli_events as ce
    head = ce.format_tool_line("terminal", {"command": "pytest -q"}, 4.06,
                               '{"stdout": "ok"}')
    if "● Bash(pytest -q)" not in head:
        return _fail(f"工具头行不对（应 ● Bash(命令)）：{head!r}")
    bad = ce.format_tool_line("terminal", {"command": "boom"}, 0.2,
                              '{"error": "exit 1"}')
    if "✗" not in bad:
        return _fail(f"失败头行缺 ✗：{bad!r}")
    # 摘要字段名对齐工具 schema（read_file 用 path 不是 file_path）
    rl = ce.format_tool_line("read_file", {"path": "src/app.py"}, 0.05, None)
    if "● Read(src/app.py)" not in rl:
        return _fail(f"read 头行缺路径摘要（字段名对不上 schema？）：{rl!r}")
    # write_file：覆盖已有文件 → Update；新文件 → Write
    if "Update(" not in ce.format_tool_line(
            "write_file", {"path": "a.py"}, 0.1, "{}", is_update=True):
        return _fail("write_file 覆盖已有文件应显示 Update")
    if "Write(" not in ce.format_tool_line(
            "write_file", {"path": "a.py"}, 0.1, "{}"):
        return _fail("write_file 新文件应显示 Write")
    # 结果块：stdout 预览 + 折叠提示（5 行只展示 3 行，剩 2 行折叠）
    blk = ce.format_result_block("terminal", _json.dumps(
        {"stdout": "l1\nl2\nl3\nl4\nl5\n"}))
    btext = "\n".join(t for _, t in blk)
    if "l1" not in btext or "+2 lines" not in btext or not \
            any(t.startswith("  ") for _, t in blk):
        return _fail(f"stdout 预览块不对：{blk}")
    err = ce.format_result_block("terminal", '{"error": "exit 1"}')
    if "✗" not in err[0][1] or "exit 1" not in err[0][1]:
        return _fail(f"错误块不对：{err}")
    rd = ce.format_result_block("read_file", '{"total_lines": 135}')
    if "读取 135 行" not in rd[0][1]:
        return _fail(f"read 块不对：{rd}")
    # 带行号 diff：红删绿增 + 截断提示 + 行号必须是整数
    d = ce.build_numbered_diff("a\nb\nc\n", "a\nX\nc\nd\n")
    kinds = [k for k, _, _ in d]
    if "-" not in kinds or "+" not in kinds:
        return _fail(f"diff 缺删/增行：{d}")
    if not all(isinstance(no, int) for k, no, _ in d if k in " +-"):
        return _fail(f"diff 行号不是整数：{d}")
    dcap = ce.build_numbered_diff(
        "", "\n".join(f"line{i}" for i in range(100)))
    if not any(k == "…" for k, _, _ in dcap):
        return _fail("diff 超长没有截断提示")
    if ce._diff_counts("a\nb\n", "a\n") != (0, 1):
        return _fail("_diff_counts 数错了")
    p = ce.EventPairer()
    p.record("x", {})
    if p.pop("x", {}) is None or p.pop("x", {}) is not None:
        return _fail("配对队列进出异常")
    import cli_session_cmds
    if hasattr(cli_session_cmds, "_maybe_prompt_resume"):
        return _fail("启动询问函数 _maybe_prompt_resume 还在")
    return _ok("claude code 风格事件行（头行+结果块+行号diff）正常")


def check_stream_box():
    """验证流式渲染：无框直排、思考押后先行、CJK 表格重排。"""
    from cli_stream import StreamBoxRenderer

    got = []
    r = StreamBoxRenderer(print_fn=got.append)

    # 思考流 → 正文（押后）：思考必须排在正文前面
    r.on_event({"type": "reasoning", "delta": "先想一想\n"})
    r.on_event({"type": "content", "delta": "| 名字 | 数量 |\n|---|---|\n"})
    r.on_event({"type": "content",
                "delta": "| 苹果 | 1 |\n| 香蕉香蕉 | 22 |\n\n"})
    r.on_event({"type": "content", "delta": "回答结束"})
    r.on_event({"type": "done"})
    # 剥掉 ANSI 色码再断言
    import re
    text = re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(got))

    if "先想一想" not in text:
        return _fail(f"思考流丢失：{text[:120]!r}")
    if text.index("先想一想") > text.index("| 名字"):
        return _fail("思考必须排在正文前面")
    # 无框直排：不允许再出现任何框线
    for frag in ("╭─", "╰", "┌─", "└"):
        if frag in text:
            return _fail(f"还有框线 {frag!r}（应为无框直排）")
    if "回答结束" not in text:
        return _fail("done 后正文丢失")
    # CJK 表格重排：表头行和数据行的竖线位置必须一致（占宽对齐）
    lines = [ln for ln in text.split("\n") if ln.startswith("|")]
    if len(lines) < 4:
        return _fail(f"表格行数不对：{lines}")
    pipe_pos = {ln.index("|", 1) for ln in (lines[0], lines[2])}
    if len(pipe_pos) != 1:
        return _fail(f"CJK 列没对齐：{lines}")
    # 表格半行兜底：done 冲掉一切残留
    r2 = StreamBoxRenderer(print_fn=lambda s: None)
    r2.on_event({"type": "content", "delta": "半行"})
    r2.on_event({"type": "done"})
    if r2._buf != "":
        return _fail("done 后半行缓冲没清")
    return _ok("无框直排 + 思考押后 + CJK 表格对齐正常")


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


def check_console_bridge():
    """验证共享 Console 两座桥：print 走 pt 通道、input 桥跨线程回传。"""
    import threading as _th
    import cli_ui

    # print 桥：rich 渲染产物（含 ANSI 色码）进 emit_ansi，不直写 stdout
    got = []
    orig = cli_ui.emit_ansi
    cli_ui.emit_ansi = got.append
    try:
        cli_ui.console.print("[red]你好[/red]")
    finally:
        cli_ui.emit_ansi = orig
    if len(got) != 1 or "\x1b[" not in got[0] or "你好" not in got[0]:
        return _fail(f"print 桥产物异常：{got!r}")

    # input 桥：工作线程的提问经桥执行并回传（假桥不真读 stdin）
    result = {}
    cli_ui.set_input_bridge(lambda f: "answered")
    try:
        t = _th.Thread(
            target=lambda: result.setdefault("v", cli_ui.console.input("问：")),
            daemon=True,
        )
        t.start()
        t.join(timeout=5)
    finally:
        cli_ui.set_input_bridge(None)
    if result.get("v") != "answered":
        return _fail(f"input 桥回传异常：{result}")
    return _ok("print 走 pt 通道 + input 桥正常")


def check_cc_double_press():
    """验证 Ctrl+C 双击检测：窗口内第二击命中、超时重新计、命中后清零。"""
    from cli_layout import is_double_press

    st = {}
    if is_double_press(st, 100.0):
        return _fail("第一击不该命中")
    if not is_double_press(st, 101.5):       # 1.5s 后 → 双击
        return _fail("窗口内第二击没命中")
    if is_double_press(st, 102.0):           # 命中后清零 → 这算新第一击
        return _fail("命中后没清零（连击误判）")
    if is_double_press(st, 110.0):           # 超窗口 → 新第一击
        return _fail("超时第二击误判为双击")
    return _ok("双击窗口/清零/超时判定正确")


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------

def check_mcp_method_names():
    """验证 MCP 方法名字面量没被反斜杠转义污染（resources\\read 回车转义事故的回归门）。

    大白话：源码字符串里写 "resources\\read"（反斜杠）时，Python 会把
    \\r 解读成回车转义——真正发出去的方法名是 resources<CR>ead，MCP
    服务器永远不认，而且这种错在界面上毫无声响。这里直接翻源码文本
    （inspect.getsource），两头卡：反斜杠形态不许出现 + 四个正常方法
    名（resources/list、resources/read、tools/list、tools/call）必须都在。

    返回：PASS/FAIL 结果。
    """
    import inspect
    import agent.mcp_client as mc
    src = inspect.getsource(mc)
    # 反斜杠转义会把 r 变回车："resources\read" 实际发的是 resources<CR>ead
    if "resources\\read" in src or "resources\\xread" in src:
        return _fail("方法名字面量含反斜杠转义（resources\\read bug 回归）")
    # 正常形态：resources/list、resources/read、tools/list、tools/call
    for good in ("resources/list", "resources/read", "tools/list", "tools/call"):
        if f'"{good}"' not in src:
            return _fail(f"缺正常方法名字面量: {good}")
    return _ok("MCP 方法名字面量干净")


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
            ("rules 传参等价", lambda: check_rules_param_equiv(tmp)),
            ("工具名清单缓存", check_tool_names_cache),
            ("terminal 工具", check_terminal_tool),
            ("read_file 工具", lambda: check_read_file_tool(tmp)),
            ("中断机制", check_interrupt),
            ("事件循环宿主", check_loop_host),
            ("拆分符号面契约", check_split_symbol_surface),
            ("Anthropic usage 字段", check_anthropic_usage_fields),
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
            ("压缩边界占位", check_compact_boundary_marker),
            ("边界落库与恢复裁剪", lambda: check_compact_boundary_persist(tmp)),
            ("摘要输入保真", check_summary_input_fidelity),
            ("token 估算与阈值", check_token_estimation_and_threshold),
            ("dispatch 统一封顶", lambda: check_dispatch_output_cap(tmp)),
            ("子代理结果落盘", lambda: check_delegate_offload(tmp)),
            ("记忆检索兜底", check_memory_retrieval_fallback),
            ("记忆注入兜底接线", check_memory_injection_wiring),
            ("发送前窗口预检", check_pre_send_guard),
            ("session append 线性化", lambda: check_session_append_perf(tmp)),
            ("session 字段透传", lambda: check_session_field_passthrough(tmp)),
            ("消息缓存 LRU", lambda: check_msgs_cache_lru(tmp)),
        ]),
        ("上下文压缩", [
            ("自动压缩", check_context_compress),
            ("压缩管线 profile", check_compress_profile),
            ("低阈值 offload 等价", lambda: check_low_threshold_offload_equivalence(tmp)),
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
            ("Console 桥", check_console_bridge),
            ("Ctrl+C 双击", check_cc_double_press),
        ]),
        ("CLI 流式框", [
            ("流式回答框", check_stream_box),
        ]),
        ("MCP", [
            ("方法名字面量", check_mcp_method_names),
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
