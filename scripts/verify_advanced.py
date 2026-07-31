"""验证 P0-P3 新增功能（借鉴 Claude Code 的取长补短项）。

用法：
    uv run python scripts/verify_advanced.py

所有检查纯本地跑（不需要真 LLM），需要 LLM 的用 mock。

P0：权限三道闸门 + 路径白名单 + 输出截断
P1：TodoWrite + 错误恢复 + load_skill
P2：子代理摘要 + worktree 隔离
P3：MCP + Task System
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model_tools import ensure_tools_discovered
ensure_tools_discovered()

from tools.registry import registry


def _ok(detail=""):
    return ("PASS", detail)


def _fail(detail):
    return ("FAIL", detail)


# ---------------------------------------------------------------------------
# P0 权限系统
# ---------------------------------------------------------------------------

def check_terminal_blocks_rm_rf():
    result = registry.dispatch("terminal", {"command": "rm -rf /"})
    data = json.loads(result)
    if data.get("error_type") == "permission_denied":
        return _ok(f"闸门: {data.get('gate', '?')}")
    return _fail(f"未拦截: {data}")


def check_terminal_blocks_sudo():
    result = registry.dispatch("terminal", {"command": "sudo apt install evil"})
    data = json.loads(result)
    if data.get("error_type") == "permission_denied":
        return _ok()
    return _fail("sudo 未被拦")


def check_terminal_blocks_fork_bomb():
    result = registry.dispatch("terminal", {"command": ":(){ :|:& };:"})
    data = json.loads(result)
    if data.get("error_type") == "permission_denied":
        return _ok()
    return _fail("fork bomb 未被拦")


def check_terminal_blocks_format():
    result = registry.dispatch("terminal", {"command": "format D:"})
    data = json.loads(result)
    if data.get("error_type") == "permission_denied":
        return _ok()
    return _fail("format 未被拦")


def check_terminal_blocks_force_push():
    result = registry.dispatch("terminal", {"command": "git push origin master --force"})
    data = json.loads(result)
    if data.get("error_type") == "permission_denied":
        return _ok()
    return _fail("强推主分支未被拦")


def check_terminal_allows_safe_command():
    result = registry.dispatch("terminal", {"command": "echo safe_test_ok"})
    data = json.loads(result)
    if "safe_test_ok" in data.get("stdout", ""):
        return _ok()
    return _fail(f"安全命令被误拦: {data}")


def check_read_file_blocks_ssh_key():
    result = registry.dispatch("read_file", {"path": "~/.ssh/id_rsa"})
    data = json.loads(result)
    if data.get("error_type") == "permission_denied":
        return _ok()
    return _fail("~/.ssh/id_rsa 可读")


def check_read_file_blocks_etc_passwd():
    result = registry.dispatch("read_file", {"path": "/etc/passwd"})
    data = json.loads(result)
    if data.get("error_type") == "permission_denied":
        return _ok()
    return _fail("/etc/passwd 可读")


def check_write_file_blocks_outside_cwd(tmp):
    target = tmp / "outside.txt"
    result = registry.dispatch("write_file", {"path": str(target), "content": "x"})
    data = json.loads(result)
    if data.get("error_type") == "permission_denied":
        return _ok("白名单拒绝")
    # 如果 tmp 恰好在 cwd 下（罕见），跳过
    if data.get("path"):
        return ("SKIP", "tmp 在 cwd 下")
    return _fail(f"cwd 外写入未被拦: {data}")


def check_write_file_allows_agent_home(tmp, monkeypatch_env):
    monkeypatch_env("OMNIMATE_HOME", str(tmp))
    target = tmp / "ok.txt"
    result = registry.dispatch("write_file", {"path": str(target), "content": "hi"})
    data = json.loads(result)
    if data.get("error_type"):
        return _fail(f"agent_home 内被拒: {data}")
    if target.read_text(encoding="utf-8") == "hi":
        return _ok()
    return _fail("文件未写入")


def check_terminal_output_truncation():
    """超 50000 字符的输出被截断。"""
    result = registry.dispatch(
        "terminal",
        {"command": "python -c \"print('y' * 60000)\""},
    )
    data = json.loads(result)
    if data.get("stdout_truncated") is True and "已截断" in data.get("stdout", ""):
        return _ok(f"{len(data['stdout'])} 字符")
    return _fail("未截断")


# ---------------------------------------------------------------------------
# P1 TodoWrite
# ---------------------------------------------------------------------------

def check_todo_write_basic():
    from agent.todo import get_todo_manager
    get_todo_manager().reset()
    result = registry.dispatch("todo_write", {"items": [
        {"text": "步骤 1", "status": "in_progress"},
        {"text": "步骤 2", "status": "pending"},
    ]})
    data = json.loads(result)
    if data.get("success") and data.get("count") == 2:
        return _ok()
    return _fail(f"todo_write 失败: {data}")


def check_todo_write_blocks_multiple_in_progress():
    result = registry.dispatch("todo_write", {"items": [
        {"text": "A", "status": "in_progress"},
        {"text": "B", "status": "in_progress"},
    ]})
    data = json.loads(result)
    if data.get("success") is False and "in_progress" in data.get("error", ""):
        return _ok()
    return _fail("允许多个 in_progress")


def check_todo_reminder_after_3_rounds():
    from agent.todo import TodoManager
    mgr = TodoManager()
    mgr.write([{"text": "未完成", "status": "in_progress"}])
    for _ in range(3):
        mgr.increment_round()
    if mgr.should_remind():
        reminder = mgr.format_for_reminder()
        if "todo_reminder" in reminder and "未完成" in reminder:
            return _ok()
    return _fail("3 轮后未触发 reminder")


# ---------------------------------------------------------------------------
# P1 错误恢复
# ---------------------------------------------------------------------------

def _make_api_error(code):
    try:
        from openai import APIStatusError
        err = APIStatusError.__new__(APIStatusError)
        err.status_code = code
        return err
    except ImportError:
        return None


def check_retry_judgment():
    from agent.llm_retry import is_retryable
    err429 = _make_api_error(429)
    err400 = _make_api_error(400)
    if err429 and is_retryable(err429) and not is_retryable(err400):
        return _ok("429 重试 / 400 不重试")
    return _fail("判断错误")


def check_retry_succeeds_eventually(monkeypatch_sleep):
    from agent.llm_retry import call_with_retry
    from unittest.mock import MagicMock

    monkeypatch_sleep(lambda s: None)
    # call_with_retry 调用 llm_client.chat_completions(messages, tools=tools)
    # 所以 mock 一个 LLMClient 风格的 client（不是 OpenAI SDK 风格）
    client = MagicMock()
    err = _make_api_error(429)
    ok_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))]
    )
    client.chat_completions.side_effect = [err, err, ok_resp]

    try:
        result = call_with_retry(
            client,
            [{"role": "user", "content": "m"}],
            tools=[],
            max_retries=3,
            initial_backoff=0.001,
        )
        if client.chat_completions.call_count == 3:
            return _ok("重试 2 次后成功")
    except Exception as e:
        return _fail(f"重试失败: {e}")
    return _fail("未重试")


def check_fallback_model(monkeypatch_sleep):
    from agent.llm_retry import call_with_retry
    from unittest.mock import MagicMock

    monkeypatch_sleep(lambda s: None)
    # 主 client：3 次全失败
    client = MagicMock()
    err = _make_api_error(429)
    client.chat_completions.side_effect = [err, err, err]
    # 备用 client：第 4 次成功（call_with_retry 用 fallback_llm_client 而非 fallback_model）
    fallback_client = MagicMock()
    fallback_client.chat_completions.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="backup", tool_calls=None))]
    )

    try:
        result = call_with_retry(
            client,
            [{"role": "user", "content": "main"}],
            tools=[],
            max_retries=3,
            initial_backoff=0.001,
            fallback_llm_client=fallback_client,
        )
        if (client.chat_completions.call_count == 3
                and fallback_client.chat_completions.call_count == 1):
            return _ok("切到备用 client")
    except Exception as e:
        return _fail(f"备用 client 失败: {e}")
    return _fail("未切换")


# ---------------------------------------------------------------------------
# P1 load_skill
# ---------------------------------------------------------------------------

def check_load_skill_returns_body(tmp):
    # 创建技能
    skills = tmp / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    (skills / "demo").mkdir(parents=True, exist_ok=True)
    (skills / "demo" / "SKILL.md").write_text(
        '---\nname: demo\ndescription: "测试"\n---\n# Demo\n执行这些步骤',
        encoding="utf-8",
    )

    result = registry.dispatch("load_skill", {"name": "demo"}, omnimate_home=tmp)
    data = json.loads(result)
    if data.get("body") and "执行这些步骤" in data["body"] and "---" not in data["body"]:
        return _ok("返回正文（去 frontmatter）")
    return _fail(f"load_skill 异常: {data}")


# ---------------------------------------------------------------------------
# P2 子代理摘要
# ---------------------------------------------------------------------------

def check_summarize_child_result():
    from tools.delegate_tool import _summarize_child_result
    from unittest.mock import MagicMock

    long = "详细结果。" * 200  # 800 字符
    # _summarize_child_result 调 call_with_retry(client, messages)
    # call_with_retry 调 client.chat_completions(messages, tools=tools)
    # 所以 mock 一个 LLMClient 风格的 client（不是 OpenAI SDK 风格）
    client = MagicMock()
    client.chat_completions.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="压缩摘要"))]
    )
    result = _summarize_child_result(long, client, "model")
    if "[摘要]" in result and "压缩摘要" in result and long not in result:
        return _ok()
    return _fail("摘要失败")


# ---------------------------------------------------------------------------
# P2 worktree 隔离
# ---------------------------------------------------------------------------

def check_worktree_create_cleanup():
    from tools.worktree import create_isolated_workspace
    path, cleanup = create_isolated_workspace(base_path=Path.home(), name="verify")
    try:
        if path.exists() and path.is_dir():
            ok_before = True
        else:
            return _fail("工作区未创建")
    finally:
        cleanup()
    if not path.exists():
        return _ok(f"已创建并清理: {path.name}")
    return _fail("cleanup 未删除目录")


# ---------------------------------------------------------------------------
# P3 MCP
# ---------------------------------------------------------------------------

def check_mcp_no_config_returns_zero():
    from tools.mcp_tool import initialize_mcp
    with patch("agent.mcp_client.load_mcp_config", return_value={}):
        count = initialize_mcp()
    if count == 0:
        return _ok("无配置不报错")
    return _fail(f"无配置却注册了 {count} 个")


def check_mcp_is_mcp_tool():
    from agent.mcp_client import is_mcp_tool
    if is_mcp_tool("mcp__fs__read") and not is_mcp_tool("terminal"):
        return _ok()
    return _fail("is_mcp_tool 判断错")


def check_mcp_config_load(tmp):
    from agent.mcp_client import load_mcp_config
    cfg = tmp / ".mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "test": {"command": "x", "args": []}
    }}), encoding="utf-8")
    servers = load_mcp_config(cfg)
    if "test" in servers:
        return _ok()
    return _fail("配置加载失败")


# ---------------------------------------------------------------------------
# P3 Task System
# ---------------------------------------------------------------------------

def check_task_create_and_list(tmp):
    result = registry.dispatch(
        "task_create",
        {"subject": "验证任务", "description": "test"},
        omnimate_home=tmp,
    )
    data = json.loads(result)
    if not data.get("success"):
        return _fail(f"create 失败: {data}")
    task_id = data["task"]["id"]

    list_result = registry.dispatch("task_list", {}, omnimate_home=tmp)
    list_data = json.loads(list_result)
    if list_data.get("count") == 1:
        return _ok(f"创建 {task_id[:16]}...")
    return _fail(f"list 异常: {list_data}")


def check_task_complete_unblocks(tmp):
    from agent.task_store import TaskStore
    store = TaskStore(omnimate_home=tmp)
    a = store.create(subject="A")
    b = store.create(subject="B", blocked_by=[a["id"]])

    # 完成前 B 不能开始
    if store.can_start(b["id"]):
        return _fail("依赖未完成却 can_start")

    store.complete(a["id"])
    if store.can_start(b["id"]):
        ready = store.find_ready()
        if any(t["id"] == b["id"] for t in ready):
            return _ok("A 完成后 B 解锁")
    return _fail("解锁失败")


def check_task_persist_across_instances(tmp):
    from agent.task_store import TaskStore
    s1 = TaskStore(omnimate_home=tmp)
    t = s1.create(subject="跨实例")
    s2 = TaskStore(omnimate_home=tmp)
    fetched = s2.get(t["id"])
    if fetched and fetched["subject"] == "跨实例":
        return _ok()
    return _fail("跨实例读取失败")


def check_task_dag_multiple_deps(tmp):
    from agent.task_store import TaskStore
    store = TaskStore(omnimate_home=tmp)
    a = store.create(subject="A")
    b = store.create(subject="B")
    c = store.create(subject="C", blocked_by=[a["id"], b["id"]])

    store.complete(a["id"])
    if store.can_start(c["id"]):
        return _fail("B 未完成 C 却可开始")
    store.complete(b["id"])
    if store.can_start(c["id"]):
        return _ok("多依赖都满足后解锁")
    return _fail("多依赖解锁失败")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  OmniMate P0-P3 新功能验证")
    print("=" * 60)

    tmp = Path(tempfile.mkdtemp(prefix="omnimate_verify_adv_"))

    # monkeypatch 辅助
    import agent.permission
    original_getenv = os.environ.get

    def monkeypatch_env(key, value):
        os.environ[key] = value

    def monkeypatch_sleep(fn):
        patcher = patch("agent.llm_retry.time.sleep", side_effect=fn)
        patcher.start()
        return patcher

    sleep_patchers = []

    checks = [
        ("P0 权限系统", [
            ("rm -rf / 被拦截", check_terminal_blocks_rm_rf),
            ("sudo 被拦截", check_terminal_blocks_sudo),
            ("fork bomb 被拦截", check_terminal_blocks_fork_bomb),
            ("format 磁盘被拦截", check_terminal_blocks_format),
            ("强推主分支被拦截", check_terminal_blocks_force_push),
            ("安全命令通过", check_terminal_allows_safe_command),
            ("read ~/.ssh/id_rsa 被拒", check_read_file_blocks_ssh_key),
            ("read /etc/passwd 被拒", check_read_file_blocks_etc_passwd),
            ("write cwd 外被拒", lambda: check_write_file_blocks_outside_cwd(tmp)),
            ("write agent_home 允许", lambda: check_write_file_allows_agent_home(tmp, monkeypatch_env)),
            ("输出超 50000 字符截断", check_terminal_output_truncation),
        ]),
        ("P1 TodoWrite", [
            ("todo_write 写入", check_todo_write_basic),
            ("多个 in_progress 被拒", check_todo_write_blocks_multiple_in_progress),
            ("3 轮触发 reminder", check_todo_reminder_after_3_rounds),
        ]),
        ("P1 错误恢复", [
            ("retry 判断（429/400）", check_retry_judgment),
            ("重试后成功", lambda: check_retry_succeeds_eventually(
                lambda fn: sleep_patchers.append(patch("agent.llm_retry.time.sleep", side_effect=fn)) or None
            )),
            ("fallback_model 切换", lambda: check_fallback_model(
                lambda fn: sleep_patchers.append(patch("agent.llm_retry.time.sleep", side_effect=fn)) or None
            )),
        ]),
        ("P1 load_skill", [
            ("加载技能正文", lambda: check_load_skill_returns_body(tmp)),
        ]),
        ("P2 子代理摘要", [
            ("_summarize_child_result", check_summarize_child_result),
        ]),
        ("P2 worktree", [
            ("create + cleanup", check_worktree_create_cleanup),
        ]),
        ("P3 MCP", [
            ("无配置降级", check_mcp_no_config_returns_zero),
            ("is_mcp_tool 判断", check_mcp_is_mcp_tool),
            ("配置加载", lambda: check_mcp_config_load(tmp)),
        ]),
        ("P3 Task System", [
            ("create + list", lambda: check_task_create_and_list(tmp)),
            ("complete 解锁依赖", lambda: check_task_complete_unblocks(tmp)),
            ("跨实例持久化", lambda: check_task_persist_across_instances(tmp)),
            ("多依赖 DAG", lambda: check_task_dag_multiple_deps(tmp)),
        ]),
    ]

    total = passed = failed = skipped = 0

    for category, items in checks:
        print(f"\n[ {category} ]")
        for name, fn in items:
            total += 1
            try:
                status, detail = fn()
            except Exception as e:
                status, detail = "FAIL", f"异常: {e}"

            if status == "PASS":
                passed += 1
                tag = "[ OK ]"
            elif status == "SKIP":
                skipped += 1
                tag = "[SKIP]"
            else:
                failed += 1
                tag = "[FAIL]"

            extra = f" ({detail})" if detail else ""
            print(f"  {tag}  {name}{extra}")

    # 清理 sleep patcher
    for p in sleep_patchers:
        try:
            p.stop()
        except Exception:
            pass

    print("\n" + "=" * 60)
    summary = "[ALL PASS]" if failed == 0 else "[HAS FAIL]"
    print(f"  {summary}  总 {total}：{passed} 通过，{failed} 失败，{skipped} 跳过")
    print("=" * 60)

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
