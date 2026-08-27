# -*- coding: utf-8 -*-
"""行为与安全修复回归测试。

覆盖：
  _offload_decisions 不被 AIAgent.__init__ 全局清空（test_offload_refined.py）
  续写恢复保留 tool_calls
  memory save() 新建路径时间戳 UTC
  curator stale 状态机落盘 + update() 支持 state
  get_task_store 按 home 键控缓存（防单例污染）
  声明式 hook fail_closed 语义传导
  危险删除切分正则含 & 和 \r\n
  deny/ask 激进剥 env 前缀 + allow 保守
  AST 解析失败时 allow 不生效
  git diff --output 不判只读
"""
from types import SimpleNamespace


# ======================================================================
# 危险删除换行/后台符绕过
# ======================================================================

def test_dangerous_removal_newline_bypass_b1():
    from agent.permission import check_dangerous_removal
    # 换行切分：第二行 rm 才是删除动词（切分必须认 \r\n/&，否则整段 verb=echo 漏过）
    assert check_dangerous_removal("echo hi\nrm -rf /usr") is not None
    assert check_dangerous_removal("echo hi\rrm -rf /usr") is not None
    assert check_dangerous_removal("echo hi & rm -rf /usr") is not None
    # 非危险目标不受影响
    assert check_dangerous_removal("echo hi\nrm -rf ./build") is None


# ======================================================================
# git diff --output 不判只读
# ======================================================================

def test_readonly_git_diff_output_rejected_b4():
    from agent.permission import _is_readonly_command
    assert _is_readonly_command("git diff") is True
    assert _is_readonly_command("git diff --output=out.patch") is False
    assert _is_readonly_command("git show --output=s.patch HEAD") is False


# ======================================================================
# 内容级规则的双形态归一化 + AST 失败 fail-safe
# ======================================================================

def test_suspicious_env_prefix_deny_aggressive_b2():
    from agent.tool_permissions import check_command_rules
    rules = {"deny": ["Bash(rm -rf:*)"], "allow": [], "ask": []}
    # 引号/$ 形态的 env 赋值若不剥掉 → deny 被绕过
    assert check_command_rules('FOO="x" rm -rf build/x', rules) == "deny"
    assert check_command_rules("FOO=$x rm -rf build/x", rules) == "deny"


def test_suspicious_env_prefix_allow_conservative_b2():
    from agent.tool_permissions import check_command_rules
    rules = {"allow": ["Bash(git status:*)"], "deny": [], "ask": []}
    # 保守形态不剥可疑 env → allow 不命中（防 FOO=$(evil) git status 激进剥离后误放行）
    assert check_command_rules("FOO=$(evil) git status", rules) == "none"
    # 干净 env 前缀 allow 仍生效
    assert check_command_rules("FOO=1 git status", rules) == "allow"


def test_ast_failure_allow_not_applied_b3(monkeypatch):
    import agent.bash_ast as bash_ast_mod
    from agent.tool_permissions import check_command_rules
    monkeypatch.setattr(bash_ast_mod, "parse_info", lambda cmd: None)
    rules = {"allow": ["Bash(git status:*)"], "deny": [], "ask": []}
    # AST 解析失败：无法证明 allow 覆盖全部段 → 不放宽
    assert check_command_rules("git status", rules) == "none"
    # deny 整串仍生效（收紧方向不受影响）
    rules2 = {"deny": ["Bash(rm -rf:*)"], "allow": [], "ask": []}
    assert check_command_rules("rm -rf build/x", rules2) == "deny"


# ======================================================================
# memory save() 新建路径 UTC 时间戳
# ======================================================================

def test_save_created_at_utc_aware_a3(tmp_path):
    from datetime import datetime, timedelta, timezone
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    # _now_iso 是秒级截断（timespec="seconds"），边界对齐到秒 +1s 容差
    before = datetime.now(timezone.utc).replace(microsecond=0)
    eid = store.save(name="n", description="d", type="user", body="b")
    entry = store.get(eid)
    after = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=1)
    assert entry.created_at.tzinfo is not None, (
        "created_at 必须是 UTC aware（此前 naive 本地时间，东八区年龄恒偏 8h）"
    )
    assert before <= entry.created_at <= after


# ======================================================================
# curator stale 状态机真正落盘 + state-only 更新不刷 updated_at
# ======================================================================

def test_update_state_only_no_updated_at_bump_a4(tmp_path):
    import time
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    eid = store.save(name="n", description="d", type="user", body="b")
    e1 = store.get(eid)
    time.sleep(1.1)  # 秒级时间戳，跨过一格才可断言"未刷新"
    store.update(eid, state="stale")
    e2 = store.get(eid)
    assert e2.state == "stale"
    assert e2.updated_at == e1.updated_at, "state-only 更新不应刷新 updated_at（防年龄翻转）"
    store.update(eid, description="d2")
    e3 = store.get(eid)
    assert e3.state == "stale"
    assert e3.updated_at > e1.updated_at, "内容更新仍应刷新 updated_at"


def test_apply_transitions_stale_persists_a4(tmp_path):
    from datetime import datetime, timedelta, timezone
    from agent.memory_curator import apply_automatic_transitions
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    eid = store.save(name="n", description="d", type="user", body="b",
                     expected_valid_days=365)
    memory_dir = tmp_path / ".memory"
    created = datetime.now(timezone.utc)
    # 400 天后 → 超过 valid_days 但未到 2×：应标 stale 且落盘（此前 no-op）
    counts = apply_automatic_transitions(
        memory_dir, now=created + timedelta(days=400), store=store,
    )
    assert counts["marked_stale"] == 1
    assert store.get(eid).state == "stale"
    # 回到当下 → reactivated 且落盘
    counts2 = apply_automatic_transitions(
        memory_dir, now=created + timedelta(days=1), store=store,
    )
    assert counts2["reactivated"] == 1
    assert store.get(eid).state == "active"


# ======================================================================
# get_task_store 按 home 键控缓存（防单例污染）
# ======================================================================

def test_get_task_store_keyed_cache_a5(tmp_path):
    import agent.task_store as ts
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    store_a1 = ts.get_task_store(omnimate_home=str(home_a))
    store_a2 = ts.get_task_store(omnimate_home=str(home_a))
    store_b = ts.get_task_store(omnimate_home=str(home_b))
    # 同 home 复用实例（旧实现每次带参调用都重建并覆盖全局）
    assert store_a1 is store_a2
    # 不同 home 互不可见（旧实现 B 会顶掉 A 的全局单例）
    assert store_a1 is not store_b
    assert store_b._dir == home_b.resolve() / ".tasks"
    assert ts.get_task_store(omnimate_home=str(home_a)) is store_a1


# ======================================================================
# 声明式 hook fail_closed 语义传导
# ======================================================================

def _make_declarative_hook(name, fail_closed):
    from agent.hooks import Hook, HookEvent, HookScriptConfig
    return Hook(
        name=name,
        event=HookEvent.PRE_TOOL_USE,
        kind="declarative",
        script=HookScriptConfig(
            command=["definitely-missing-cmd-xyz-r30b"], timeout=2.0,
        ),
        fail_closed=fail_closed,
    )


def test_declarative_pre_tool_fail_closed_denies_a6():
    from agent.hooks import HookEvent, HookRegistry
    reg = HookRegistry()
    reg._hooks[HookEvent.PRE_TOOL_USE].append(
        _make_declarative_hook("guard", fail_closed=True)
    )
    deny, _modified = reg.run_pre_tool_use(
        "terminal", {"command": "ls"}, session_id="s",
    )
    # 启动失败 + fail_closed → deny（此前 dispatch_hook 吞异常，fail_closed 永不可达）
    assert deny is not None


def test_declarative_pre_tool_fail_open_default_a6():
    from agent.hooks import HookEvent, HookRegistry
    reg = HookRegistry()
    reg._hooks[HookEvent.PRE_TOOL_USE].append(
        _make_declarative_hook("guard2", fail_closed=False)
    )
    deny, _modified = reg.run_pre_tool_use(
        "terminal", {"command": "ls"}, session_id="s",
    )
    assert deny is None  # 默认 fail-open 语义不变


# ======================================================================
# 续写恢复保留 tool_calls
# ======================================================================

def _truncated_response(content="part"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=content, tool_calls=None,
                reasoning_content=None, thinking_signature=None,
            ),
            finish_reason="length",
        )],
        usage=None,
    )


def test_merge_continuation_preserves_tool_calls_a2():
    from agent import AIAgent
    tc = [{"id": "call_1", "type": "function",
           "function": {"name": "f", "arguments": "{}"}}]
    last = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="tail", tool_calls=tc),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(total_tokens=1),
    )
    merged = AIAgent._merge_continuation_response(
        _truncated_response(), last, "parttail", finished=True,
    )
    assert merged.choices[0].message.tool_calls == tc
    assert merged.choices[0].finish_reason == "tool_calls"


def test_merge_continuation_plain_stop_a2():
    from agent import AIAgent
    last = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="tail", tool_calls=None),
            finish_reason="stop",
        )],
        usage=None,
    )
    merged = AIAgent._merge_continuation_response(
        _truncated_response(), last, "parttail", finished=True,
    )
    assert merged.choices[0].message.tool_calls is None
    assert merged.choices[0].finish_reason == "stop"


def test_merge_continuation_unfinished_keeps_length_a2():
    from agent import AIAgent
    merged = AIAgent._merge_continuation_response(
        _truncated_response(), None, "parttail", finished=False,
    )
    assert merged.choices[0].finish_reason == "length"
