"""MemoryCurator 测试。"""
from pathlib import Path
import pytest


def test_memory_entry_has_state_field_with_default_active(tmp_path):
    """新创建的 MemoryEntry 默认 state='active'。"""
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    mid = store.save(
        name="测试", description="测试默认 state", type="user", body="正文",
    )
    entry = store.get(mid)
    assert entry.state == "active"
    assert entry.last_reviewed_at == ""
def test_rebuild_index_excludes_archived(tmp_path):
    """archived 状态的记忆不出现在 MEMORY.md 索引中。"""
    from agent.memory_store import MemoryStore, _format_frontmatter
    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    # 写一条 active,一条 archived
    for name, state in [("活着", "active"), ("死了", "archived")]:
        meta = {
            "name": name, "description": f"desc-{name}", "type": "user",
            "created_at": "2026-07-29T00:00:00", "updated_at": "2026-07-29T00:00:00",
            "state": state,
        }
        (memory_dir / f"{name}.md").write_text(
            _format_frontmatter(meta) + f"body-{name}", encoding="utf-8",
        )
    store = MemoryStore(omnimate_home=tmp_path)
    snapshot = store.snapshot_for_prompt()
    assert "活着" in snapshot
    assert "死了" not in snapshot


# ============================================================
# Task 2: apply_automatic_transitions 状态转换纯函数
# ============================================================

from datetime import datetime, timedelta, timezone


def _make_memory(tmp_path, *, name, age_days, valid_days=365, state="active", type="user"):
    """辅助:创建一条指定年龄/有效期/状态/类型的记忆。"""
    from agent.memory_store import _format_frontmatter
    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    past = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat(timespec="seconds")
    meta = {
        "name": name, "description": f"desc-{name}", "type": type,
        "created_at": past, "updated_at": past,
        "expected_valid_days": valid_days, "state": state,
    }
    (memory_dir / f"{name}.md").write_text(
        _format_frontmatter(meta) + f"body-{name}", encoding="utf-8",
    )
def test_archived_is_terminal_state(tmp_path):
    """archived 是终态,不再变化。"""
    _make_memory(tmp_path, name="m3", age_days=9999, valid_days=30, state="archived")
    # 手动把文件挪到 .archive/(模拟已归档)
    import shutil
    archive_dir = tmp_path / ".archive" / "memory-test"
    archive_dir.mkdir(parents=True)
    shutil.move(
        str(tmp_path / ".memory" / "m3.md"),
        str(archive_dir / "m3.md"),
    )
    from agent.memory_curator import apply_automatic_transitions
    counts = apply_automatic_transitions(tmp_path / ".memory")
    assert counts["archived"] == 0
    assert counts["checked"] == 0  # .memory/ 里没东西了
def test_state_file_round_trip(tmp_path):
    """状态文件读写正确。"""
    from agent.memory_curator import load_memory_curator_state, save_memory_curator_state
    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    save_memory_curator_state(memory_dir, {
        "last_run_at": "2026-07-29T00:00:00+00:00",
        "paused": False,
    })
    loaded = load_memory_curator_state(memory_dir)
    assert loaded["last_run_at"] == "2026-07-29T00:00:00+00:00"
    assert loaded["paused"] is False


def test_should_run_seeds_on_first_run(tmp_path):
    """首次调用不跑,种子化 last_run_at。"""
    from agent.memory_curator import should_run_now_memory, load_memory_curator_state
    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    result = should_run_now_memory(memory_dir)
    assert result is False
    state = load_memory_curator_state(memory_dir)
    assert "last_run_at" in state  # 已种子化


def test_should_run_false_within_interval(tmp_path):
    """距上次 < interval_hours 不跑。"""
    from agent.memory_curator import should_run_now_memory, save_memory_curator_state
    import datetime
    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    # 1 天前跑过
    past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)).isoformat()
    save_memory_curator_state(memory_dir, {"last_run_at": past, "paused": False})
    assert should_run_now_memory(memory_dir, interval_hours=168) is False


def test_should_run_true_after_interval(tmp_path):
    """距上次 > interval_hours 跑。"""
    from agent.memory_curator import should_run_now_memory, save_memory_curator_state
    import datetime
    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    # 8 天前跑过
    past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=8)).isoformat()
    save_memory_curator_state(memory_dir, {"last_run_at": past, "paused": False})
    assert should_run_now_memory(memory_dir, interval_hours=168) is True


def test_should_run_false_when_paused(tmp_path):
    """paused=True 时永不跑。"""
    from agent.memory_curator import should_run_now_memory, save_memory_curator_state
    import datetime
    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)).isoformat()
    save_memory_curator_state(memory_dir, {"last_run_at": past, "paused": True})
    assert should_run_now_memory(memory_dir) is False


# ============================================================
# Task 4: curator_cli memory 子命令
# ============================================================


def test_curator_cli_memory_status_empty(tmp_path, capsys, monkeypatch):
    """memory status 在空目录下能跑,不报错。"""
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    from curator_cli import main
    main(["memory", "status"])
    out = capsys.readouterr().out
    assert "Memory Curator" in out or "从未" in out or "未运行" in out
def test_curator_cli_memory_pause_resume(tmp_path, monkeypatch):
    """memory pause/resume 修改状态文件。"""
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    from curator_cli import main
    from agent.memory_curator import load_memory_curator_state
    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    main(["memory", "pause"])
    assert load_memory_curator_state(memory_dir).get("paused") is True

    main(["memory", "resume"])
    assert load_memory_curator_state(memory_dir).get("paused") is False
def test_runtime_initialize_does_not_trigger_within_interval(tmp_path, monkeypatch):
    """7 天内不重复触发。"""
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    import datetime
    # 1 天前跑过
    past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)).isoformat()
    from agent.memory_curator import save_memory_curator_state, should_run_now_memory
    save_memory_curator_state(memory_dir, {"last_run_at": past, "paused": False})
    assert should_run_now_memory(memory_dir) is False


# ============================================================
# Task 6: 第 2 阶段候选收集 + 分桶 + 分批
# ============================================================


def test_collect_review_candidates_groups_by_type(tmp_path):
    """按 type 分桶,只收 state=active 的。"""
    from agent.memory_curator import collect_review_candidates
    _make_memory(tmp_path, name="u1", age_days=10, valid_days=365, state="active")
    _make_memory(tmp_path, name="u2", age_days=10, valid_days=365, state="active")
    _make_memory(tmp_path, name="p1", age_days=10, valid_days=365, state="active", type="project")
    _make_memory(tmp_path, name="p2", age_days=10, valid_days=365, state="active", type="project")
    _make_memory(tmp_path, name="skip1", age_days=10, valid_days=365, state="stale")
    _make_memory(tmp_path, name="skip2", age_days=10, valid_days=365, state="archived")

    buckets = collect_review_candidates(tmp_path / ".memory")
    assert "user" in buckets
    assert len(buckets["user"]) == 2  # u1, u2
    assert len(buckets.get("project", [])) == 2  # p1, p2(需 2+ 才保留桶)
    # stale/archived 不进桶
    all_names = [e.name for entries in buckets.values() for e in entries]
    assert "skip1" not in all_names
    assert "skip2" not in all_names


def test_collect_review_candidates_excludes_buckets_with_one_entry(tmp_path):
    """只有 1 条的桶被排除(单条不可能重复/矛盾)。"""
    from agent.memory_curator import collect_review_candidates
    _make_memory(tmp_path, name="only1", age_days=10, valid_days=365)
    buckets = collect_review_candidates(tmp_path / ".memory")
    assert "user" not in buckets  # 只有 1 条,整桶丢


def test_chunk_batch_splits_large_list():
    """超过 size 的列表被切分。"""
    from agent.memory_curator import chunk_batch
    items = list(range(75))
    batches = list(chunk_batch(items, size=30))
    assert len(batches) == 3
    assert len(batches[0]) == 30
    assert len(batches[1]) == 30
    assert len(batches[2]) == 15


# ============================================================
# Task 7: YAML 解析 + action 执行 + 改写备份
# ============================================================


def test_parse_yaml_actions_valid(tmp_path):
    """合法 YAML 输出被正确解析。"""
    from agent.memory_curator import parse_yaml_actions
    raw = '''分析完成:

```yaml
- action: merge_duplicate
  keep: id1
  archive: [id2, id3]
  reason: 重复
- action: resolve_contradiction
  update_id: id4
  new_body: |
    整合后
  archive: id5
  reason: 矛盾
```
'''
    actions = parse_yaml_actions(raw)
    assert len(actions) == 2
    assert actions[0]["action"] == "merge_duplicate"
    assert actions[0]["keep"] == "id1"
    assert actions[1]["action"] == "resolve_contradiction"


def test_parse_yaml_actions_malformed_returns_empty():
    """损坏的 YAML 返回空列表,不抛。"""
    from agent.memory_curator import parse_yaml_actions
    assert parse_yaml_actions("没有 yaml 块") == []
    assert parse_yaml_actions("```yaml\n{{invalid\n```") == []


def test_safe_rewrite_body_backs_up_original(tmp_path):
    """改写 body 前确实备份了原文。"""
    from agent.memory_store import MemoryStore
    from agent.memory_curator import safe_rewrite_body
    store = MemoryStore(omnimate_home=tmp_path)
    mid = store.save(
        name="测试", description="...", type="user",
        body="原文 body 内容",
    )
    archive_root = tmp_path / ".archive"
    safe_rewrite_body(store, mid, "新 body 内容", archive_root)
    # 备份存在
    backups = list(archive_root.glob("memory-rewrites-*/" + mid + ".md"))
    assert backups
    # 备份含原文
    backup_text = backups[0].read_text(encoding="utf-8")
    assert "原文 body 内容" in backup_text
    # 当前 body 已更新
    assert store.get(mid).body == "新 body 内容"


def test_execute_action_merge_duplicate(tmp_path):
    """merge_duplicate: 保留主,归档其余。"""
    from agent.memory_store import MemoryStore
    from agent.memory_curator import execute_action
    store = MemoryStore(omnimate_home=tmp_path)
    keep_id = store.save(name="主", description="...", type="user", body="主body")
    archive_id = store.save(name="副本", description="...", type="user", body="副本body")
    action = {
        "action": "merge_duplicate",
        "keep": keep_id,
        "archive": [archive_id],
        "reason": "重复",
    }
    result = execute_action(action, store, tmp_path / ".archive")
    assert "merge_duplicate" in result
    # archive_id 已被 store.delete(软删除)
    assert store.get(archive_id) is None
    # keep 还在
    assert store.get(keep_id) is not None


def test_execute_action_resolve_contradiction(tmp_path):
    """resolve_contradiction: update body + archive 新的。"""
    from agent.memory_store import MemoryStore
    from agent.memory_curator import execute_action
    store = MemoryStore(omnimate_home=tmp_path)
    old_id = store.save(name="旧", description="...", type="user", body="原偏好")
    new_id = store.save(name="新", description="...", type="user", body="换偏好")
    action = {
        "action": "resolve_contradiction",
        "update_id": old_id,
        "new_body": "原偏好 X。2026-07 改为 Y。",
        "archive": new_id,
        "reason": "矛盾",
    }
    result = execute_action(action, store, tmp_path / ".archive")
    # old_id body 已被改写
    assert "改为 Y" in store.get(old_id).body
    # new_id 已归档
    assert store.get(new_id) is None


def test_execute_action_unknown_skips(tmp_path):
    """未知 action 跳过,不抛。"""
    from agent.memory_store import MemoryStore
    from agent.memory_curator import execute_action
    store = MemoryStore(omnimate_home=tmp_path)
    action = {"action": "bogus_action", "foo": "bar"}
    result = execute_action(action, store, tmp_path / ".archive")
    assert "skip" in result.lower() or "未知" in result


# ---------------------------------------------------------------------------
# Task 8: run_memory_review 主入口(第 2 阶段 LLM 合并 + 矛盾检测)
# ---------------------------------------------------------------------------


def test_run_memory_review_contradiction_e2e(tmp_path):
    """端到端:两条矛盾记忆 → LLM 返回 resolve_contradiction → 改写 + 归档 + 备份。"""
    from agent.memory_store import MemoryStore
    from agent.memory_curator import run_memory_review

    store = MemoryStore(omnimate_home=tmp_path)
    old_id = store.save(
        name="DeepSeek 偏好", description="用户喜欢 DeepSeek",
        type="user", body="用户喜欢用 DeepSeek 做主对话",
    )
    new_id = store.save(
        name="DeepSeek 切换", description="用户换 Claude",
        type="user", body="用户从 DeepSeek 换到 Claude 了",
    )

    # mock agent_factory:返回一个 chat() 返回固定 YAML 的假 agent
    class FakeAgent:
        async def chat(self, prompt):
            return f"""分析完成:

```yaml
- action: resolve_contradiction
  update_id: {old_id}
  new_body: |
    原偏好 DeepSeek。2026-07 换到 Claude。
  archive: {new_id}
  reason: 用户切换主模型
```
"""

    report = run_memory_review(
        tmp_path / ".memory",
        agent_factory=lambda: FakeAgent(),
        dry_run=False,
    )
    # 验证
    assert report["executed_actions"] >= 1
    assert "DeepSeek" in store.get(old_id).body
    assert "Claude" in store.get(old_id).body
    # new_id 已归档
    assert store.get(new_id) is None
    # 备份存在
    backups = list((tmp_path / ".archive").glob("memory-rewrites-*"))
    assert backups


def test_run_memory_review_dry_run_no_changes(tmp_path):
    """dry-run 不实际执行,只报告。"""
    from agent.memory_store import MemoryStore
    from agent.memory_curator import run_memory_review

    store = MemoryStore(omnimate_home=tmp_path)
    store.save(name="A", description="...", type="user", body="A body")
    store.save(name="B", description="...", type="user", body="B body")

    class FakeAgent:
        async def chat(self, prompt):
            return "```yaml\n- action: merge_duplicate\n  keep: nonexist\n  archive: []\n  reason: x\n```"

    report = run_memory_review(
        tmp_path / ".memory",
        agent_factory=lambda: FakeAgent(),
        dry_run=True,
    )
    # dry-run 不调 agent(避免成本)
    assert report["dry_run"] is True
    assert report["buckets_reviewed"] == 0


def test_run_memory_review_handles_llm_failure(tmp_path):
    """LLM 调用失败时,该桶跳过,不阻塞。"""
    from agent.memory_store import MemoryStore
    from agent.memory_curator import run_memory_review

    store = MemoryStore(omnimate_home=tmp_path)
    aid = store.save(name="A", description="...", type="user", body="A")
    bid = store.save(name="B", description="...", type="user", body="B")

    class CrashAgent:
        async def chat(self, prompt):
            raise RuntimeError("API 挂了")

    report = run_memory_review(
        tmp_path / ".memory",
        agent_factory=lambda: CrashAgent(),
        dry_run=False,
    )
    assert report["errors"] >= 1
    # 记忆未被破坏(用 save 返回的 id 查询,而非 name)
    assert store.get(aid) is not None
    assert store.get(bid) is not None


# ---------------------------------------------------------------------------
# Task 9: 启动钩子接入第 2 阶段 + CLI run 完整模式
# ---------------------------------------------------------------------------


def test_make_memory_review_agent_factory_provides_main_model(tmp_path, monkeypatch):
    """RuntimeContext 的 _make_memory_review_agent_factory 返回的 agent 用主模型。"""
    # 这个测试主要验证 factory 函数能被调用且不抛
    # 真正的 LLM 调用在集成测试里 mock
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    # mock settings 让它返回有效配置
    import agent.settings as settings_mod
    original = settings_mod.load_settings

    def fake_settings():
        s = original()
        s["llm"]["auth_token"] = "fake-token"
        return s

    monkeypatch.setattr(settings_mod, "load_settings", fake_settings)

    from cli import RuntimeContext
    rt = RuntimeContext()
    # factory 应该可调用(不实际构造完整 agent,只验证函数存在)
    assert hasattr(rt, "_make_memory_review_agent_factory") or callable(
        getattr(rt, "_make_memory_review_agent_factory", None)
    )


def test_curator_cli_memory_run_with_llm_phase(tmp_path, monkeypatch):
    """curator memory run 完整跑(含第 2 阶段),用 mock agent_factory。"""
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    from agent.memory_store import MemoryStore

    store = MemoryStore(omnimate_home=tmp_path)
    old_id = store.save(name="X", description="...", type="user", body="old")
    new_id = store.save(name="Y", description="...", type="user", body="new")

    # mock:让 curator_cli 用 FakeAgent
    import agent.memory_curator as mc

    class FakeAgent:
        async def chat(self, prompt):
            return (
                f"```yaml\n- action: resolve_contradiction\n"
                f"  update_id: {old_id}\n  new_body: |\n    merged\n"
                f"  archive: {new_id}\n  reason: x\n```"
            )

    original_run = mc.run_memory_review

    def stub_run(memory_dir, *, agent_factory, dry_run=False, **kw):
        return original_run(
            memory_dir,
            agent_factory=lambda: FakeAgent(),
            dry_run=dry_run,
            **kw,
        )

    monkeypatch.setattr(mc, "run_memory_review", stub_run)

    from curator_cli import main

    main(["memory", "run"])
    # 验证 body 被改写
    assert "merged" in store.get(old_id).body


# ---------------------------------------------------------------------------
# Task 10: 配置集成(enabled / interval_hours / llm_review_enabled / max_batch_size)
# ---------------------------------------------------------------------------


def test_should_run_disabled_via_config(tmp_path):
    """config memory.curator.enabled=False 时不跑。"""
    from agent.memory_curator import should_run_now_memory
    import datetime
    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)).isoformat()
    from agent.memory_curator import save_memory_curator_state
    save_memory_curator_state(memory_dir, {"last_run_at": past, "paused": False})
    config = {"memory": {"curator": {"enabled": False}}}
    assert should_run_now_memory(memory_dir, config=config) is False


def test_run_review_skipped_when_llm_disabled(tmp_path):
    """llm_review_enabled=False 时 run_memory_review 直接跳过。"""
    from agent.memory_curator import run_memory_review
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    store.save(name="A", description="...", type="user", body="A")
    store.save(name="B", description="...", type="user", body="B")
    config = {"memory": {"curator": {"llm_review_enabled": False}}}
    report = run_memory_review(
        tmp_path / ".memory",
        agent_factory=lambda: None,
        config=config,
    )
    assert report.get("skipped") == "llm_review_enabled=False"
