"""按 11-scaffold.md 的检查清单验证 agent 功能。

用法：
    uv run python scripts/verify.py

输出每项的 PASS/FAIL/SKIPPED，最终汇总。
SKIPPED 项需要真实 API key 才能验证。
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# 确保项目根目录在 path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _result(status, detail=""):
    return {"status": status, "detail": detail}


def _ok(detail=""):
    return _result("PASS", detail)


def _fail(detail=""):
    return _result("FAIL", detail)


def _skip(detail=""):
    return _result("SKIPPED", detail)


# ---------------------------------------------------------------------------
# 基础对话
# ---------------------------------------------------------------------------

def check_agent_initialization():
    """agent 能初始化（无 API 调用）。"""
    from agent import AIAgent
    agent = AIAgent(api_key="fake", model="test", enabled_toolsets=[])
    return _ok(f"max_iter={agent.max_iterations}, budget={agent.iteration_budget.remaining}")


def check_tool_definitions():
    """工具定义能加载。"""
    from model_tools import get_tool_definitions
    tools = get_tool_definitions(["core"])
    names = [t["function"]["name"] for t in tools]
    expected = {"terminal", "read_file", "write_file", "memory"}
    missing = expected - set(names)
    if missing:
        return _fail(f"缺少工具: {missing}")
    return _ok(f"暴露 {len(names)} 个工具")


def check_terminal_tool():
    """terminal 工具能执行命令。"""
    from tools.registry import registry
    result = registry.dispatch("terminal", {"command": "echo verify_ok"})
    data = json.loads(result)
    if "verify_ok" in data.get("stdout", ""):
        return _ok("echo 输出正确")
    return _fail(f"输出异常: {data}")


def check_read_file_tool(tmp):
    """read_file 工具能读文件。"""
    from tools.registry import registry
    f = tmp / "sample.txt"
    f.write_text("line1\nline2\n", encoding="utf-8")
    result = registry.dispatch("read_file", {"path": str(f)})
    data = json.loads(result)
    if "line1" in data.get("content", ""):
        return _ok("读取到内容")
    return _fail(f"读取失败: {data}")


def check_interrupt():
    """中断机制工作。"""
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
    """memory 工具能写入记忆。"""
    from tools.registry import registry
    from agent.memory_store import MemoryStore
    store = MemoryStore(tmp)

    result = registry.dispatch(
        "memory",
        {"action": "add", "target": "memory", "content": "验证测试"},
        memory_store=store,
    )
    data = json.loads(result)
    if data.get("success") and "验证测试" in store.memory_entries:
        return _ok("已写入 MEMORY.md")
    return _fail(f"写入失败: {data}")


def check_memory_persist(tmp):
    """MEMORY.md 文件被创建。"""
    from agent.memory_store import MemoryStore
    store = MemoryStore(tmp)
    store.add("memory", "持久化测试")
    if (tmp / "MEMORY.md").exists():
        return _ok("MEMORY.md 已创建")
    return _fail("MEMORY.md 未创建")


def check_memory_reload(tmp):
    """重启后记忆能加载。"""
    from agent.memory_store import MemoryStore
    s1 = MemoryStore(tmp)
    s1.add("memory", "重启测试")

    s2 = MemoryStore(tmp)
    if "重启测试" in s2.memory_entries:
        return _ok("记忆已跨实例加载")
    return _fail("记忆丢失")


# ---------------------------------------------------------------------------
# 技能系统
# ---------------------------------------------------------------------------

def _setup_skill(tmp):
    """创建示例技能。"""
    skills = tmp / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    (skills / "hello").mkdir(parents=True, exist_ok=True)
    (skills / "hello" / "SKILL.md").write_text(
        '---\nname: hello\ndescription: "打招呼"\n---\n# Hello\n你好技能',
        encoding="utf-8",
    )
    return skills


def check_skill_trigger(tmp):
    """/hello 技能能被触发。"""
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
    """skills_list 工具。"""
    from tools.registry import registry
    skills = _setup_skill(tmp)
    result = registry.dispatch("skills_list", {}, harvil_home=tmp)
    data = json.loads(result)
    names = [s["name"] for s in data["skills"]]
    if "hello" in names:
        return _ok("列出了 hello 技能")
    return _fail(f"未列出: {data}")


def check_skill_view(tmp):
    """skill_view 工具。"""
    from tools.registry import registry
    _setup_skill(tmp)
    result = registry.dispatch("skill_view", {"name": "hello"}, harvil_home=tmp)
    data = json.loads(result)
    if "Hello" in data.get("content", ""):
        return _ok("查看了 hello 技能")
    return _fail(f"查看失败: {data}")


def check_skill_manage_create(tmp):
    """skill_manage 能创建新技能。"""
    from tools.registry import registry
    registry.dispatch(
        "skill_manage",
        {"action": "create", "name": "new-skill", "content": "---\nname: x\n---\nbody"},
        harvil_home=tmp,
    )
    if (tmp / "skills" / "new-skill" / "SKILL.md").exists():
        return _ok("创建了 new-skill")
    return _fail("创建失败")


def check_usage_stats(tmp):
    """.usage.json 记录使用统计。"""
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
    """sessions.db 被创建。"""
    from agent.session_store import SessionStore
    db = tmp / "sessions.db"
    store = SessionStore(db)
    if db.exists():
        return _ok(str(db))
    return _fail("sessions.db 未创建")


def check_session_search(tmp):
    """session_search 能搜到历史对话。"""
    from agent.session_store import SessionStore
    from tools.registry import registry
    store = SessionStore(tmp / "s.db")
    sid = store.create_session()
    store.append_message(sid, "user", "Python 测试内容")

    result = registry.dispatch(
        "session_search",
        {"query": "Python"},
        session_store=store,
    )
    data = json.loads(result)
    if data.get("total", 0) > 0:
        return _ok(f"找到 {data['total']} 条")
    return _fail("未搜到")


# ---------------------------------------------------------------------------
# Curator
# ---------------------------------------------------------------------------

def check_curator_status(tmp):
    """curator status 显示状态。"""
    from agent.curator import load_state
    state = load_state(tmp / "skills")
    # 空状态也是有效的
    return _ok(f"state keys: {list(state.keys()) or '(空)'}")


def check_curator_dry_run(tmp):
    """curator run --dry-run 能预览。"""
    from agent.curator import run_curator_review
    skills = tmp / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    report = run_curator_review(skills, dry_run=True)
    if report["dry_run"] is True:
        return _ok(f"transitions={report['transitions']}")
    return _fail("dry_run 标志错误")


def check_curator_archive(tmp):
    """归档的技能移到 .archive/。"""
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
    """curator restore 能恢复。"""
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
    """delegate_task 同步模式（mock）。"""
    from unittest.mock import patch
    from tools.registry import registry
    with patch("tools.delegate_tool._run_child", return_value="子代理完成"):
        result = registry.dispatch(
            "delegate_task",
            {"goal": "测试任务"},
            base_url=None, api_key="fake", model="test",
        )
    data = json.loads(result)
    if data.get("success") and data.get("result") == "子代理完成":
        return _ok("同步委托 OK")
    return _fail(f"委托失败: {data}")


def check_delegate_batch(tmp):
    """批量委托并行执行（mock）。"""
    from unittest.mock import patch
    from tools.registry import registry
    with patch("tools.delegate_tool._run_child", return_value="ok"):
        result = registry.dispatch(
            "delegate_task",
            {"tasks": [{"goal": "a"}, {"goal": "b"}, {"goal": "c"}]},
            base_url=None, api_key="fake", model="test",
        )
    data = json.loads(result)
    if data["mode"] == "batch" and len(data["results"]) == 3:
        return _ok("3 个任务并行完成")
    return _fail(f"批量失败: {data}")


# ---------------------------------------------------------------------------
# 上下文压缩
# ---------------------------------------------------------------------------

def check_context_compress():
    """长对话触发压缩（新管线，mock LLM）。"""
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
    # 降低阈值确保 snip 触发
    ctx_cfg["snip_message_threshold"] = 50
    state = CompressionSessionState()
    new_msgs, compressed = compress_if_needed(
        msgs,
        llm_client=client,
        model="deepseek-chat",
        config=ctx_cfg,
        session_state=state,
        agent_home=None,
        session_id="verify",
    )
    if compressed and len(new_msgs) < len(msgs):
        return _ok(f"{len(msgs)} → {len(new_msgs)} 条")
    return _fail("未压缩")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  HarvilAgent 复刻检查清单验证")
    print("=" * 60)

    tmp = Path(tempfile.mkdtemp(prefix="hermes_verify_"))
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
            ("sessions.db 创建", lambda: check_sessions_db(tmp)),
            ("session_search", lambda: check_session_search(tmp)),
        ]),
        ("Curator", [
            ("status 状态", lambda: check_curator_status(tmp)),
            ("run --dry-run", lambda: check_curator_dry_run(tmp)),
            ("归档到 .archive/", lambda: check_curator_archive(tmp)),
            ("restore 恢复", lambda: check_curator_restore(tmp)),
        ]),
        ("委托", [
            ("delegate_task 同步", lambda: check_delegate_sync(tmp)),
            ("批量委托并行", lambda: check_delegate_batch(tmp)),
        ]),
        ("上下文压缩", [
            ("自动压缩", check_context_compress),
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

    # 清理临时目录
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
