"""Curator 测试。"""

from datetime import datetime, timedelta, timezone

import pytest

from agent.curator import (
    should_run_now, apply_automatic_transitions, run_curator_review,
    load_state, save_state, _parse_iso, _parse_consolidation_output,
    _latest_activity,
)
from tools.skill_usage import (
    bump_use, bump_view, mark_agent_created, set_state, set_pinned,
    archive_skill, STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED,
)


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_home(tmp_path):
    """提供空 agent_home（无状态文件）。"""
    home = tmp_path / "home"
    home.mkdir()
    skills = home / "skills"
    skills.mkdir()
    return home


@pytest.fixture
def seeded_home(fresh_home):
    """已种子化 last_run_at 的 home（should_run_now 第二次起的行为）。"""
    skills = fresh_home / "skills"
    save_state(skills, {"last_run_at": datetime.now(timezone.utc).isoformat()})
    return fresh_home


def _make_skill(skills_dir, name, created_by="agent", state=None, days_ago=0):
    """创建一个有使用记录的技能。"""
    import json
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test\n---\n# {name}\n",
        encoding="utf-8",
    )

    # 通过 bump_use 创建记录
    bump_use(skills_dir, name)
    bump_view(skills_dir, name)

    # 加载并修改记录
    from tools.skill_usage import load_usage, save_usage
    data = load_usage(skills_dir)
    rec = data[name]
    rec["created_by"] = created_by
    if state:
        rec["state"] = state

    # 调整时间戳（模拟 N 天前的活动）
    if days_ago > 0:
        old = datetime.now(timezone.utc) - timedelta(days=days_ago)
        rec["last_used_at"] = old.isoformat()
        rec["last_viewed_at"] = old.isoformat()
        rec["created_at"] = old.isoformat()

    save_usage(skills_dir, data)
    return rec


# ---------------------------------------------------------------------------
# should_run_now
# ---------------------------------------------------------------------------

def test_should_run_first_time_postponed(fresh_home):
    """首次运行被推迟（种子化）。"""
    skills = fresh_home / "skills"
    assert should_run_now(skills) is False
    # 状态文件被写入
    state = load_state(skills)
    assert "last_run_at" in state
    assert "推迟" in state["last_run_summary"]


def test_should_run_within_interval(seeded_home):
    """周期内不触发。"""
    assert should_run_now(seeded_home / "skills") is False


def test_should_run_after_interval(seeded_home):
    """超过周期触发。"""
    skills = seeded_home / "skills"
    # 把 last_run_at 设为 8 天前
    old = datetime.now(timezone.utc) - timedelta(days=8)
    state = load_state(skills)
    state["last_run_at"] = old.isoformat()
    save_state(skills, state)

    assert should_run_now(skills) is True


def test_should_run_paused(seeded_home):
    """暂停时不触发。"""
    skills = seeded_home / "skills"
    state = load_state(skills)
    state["paused"] = True
    save_state(skills, state)

    # 把 last_run_at 设为很久以前
    state["last_run_at"] = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    save_state(skills, state)

    assert should_run_now(skills) is False


# ---------------------------------------------------------------------------
# apply_automatic_transitions
# ---------------------------------------------------------------------------

def test_transitions_skip_user_skills(seeded_home):
    """用户创建的技能不被转换。"""
    skills = seeded_home / "skills"
    _make_skill(skills, "user-skill", created_by="user", days_ago=120)

    counts = apply_automatic_transitions(skills)
    assert counts["checked"] == 0  # 用户技能不计入
    assert counts["archived"] == 0


def test_transitions_archive_old(seeded_home):
    """90+ 天无活动 → archived。"""
    skills = seeded_home / "skills"
    _make_skill(skills, "old-skill", created_by="agent", days_ago=120)

    counts = apply_automatic_transitions(skills)
    assert counts["checked"] == 1
    assert counts["archived"] == 1
    assert not (skills / "old-skill").exists()
    assert (skills / ".archive" / "old-skill").exists()


def test_transitions_mark_stale(seeded_home):
    """30+ 天无活动 + active → stale。"""
    skills = seeded_home / "skills"
    _make_skill(skills, "stale-skill", created_by="agent", days_ago=45)

    counts = apply_automatic_transitions(skills)
    assert counts["marked_stale"] == 1

    from tools.skill_usage import load_usage
    rec = load_usage(skills)["stale-skill"]
    assert rec["state"] == STATE_STALE


def test_transitions_reactivate(seeded_home):
    """最近活动 + stale → active。"""
    skills = seeded_home / "skills"
    # 创建一个 stale 状态、最近有活动的技能
    _make_skill(skills, "revived", created_by="agent", state=STATE_STALE, days_ago=5)

    counts = apply_automatic_transitions(skills)
    assert counts["reactivated"] == 1

    from tools.skill_usage import load_usage
    rec = load_usage(skills)["revived"]
    assert rec["state"] == STATE_ACTIVE


def test_transitions_pinned_immune(seeded_home):
    """pinned 技能不被转换。"""
    skills = seeded_home / "skills"
    _make_skill(skills, "pinned-old", created_by="agent", days_ago=120)
    set_pinned(skills, "pinned-old", True)

    counts = apply_automatic_transitions(skills)
    assert counts["checked"] == 0  # pinned 不计入 checked
    assert counts["archived"] == 0
    assert (skills / "pinned-old").exists()  # 未被归档


# ---------------------------------------------------------------------------
# run_curator_review
# ---------------------------------------------------------------------------

def test_run_curator_dry_run(seeded_home):
    """dry-run 不做任何转换。"""
    skills = seeded_home / "skills"
    _make_skill(skills, "old", created_by="agent", days_ago=120)

    report = run_curator_review(skills, dry_run=True)
    assert report["dry_run"] is True
    assert report["transitions"]["archived"] == 0
    # 状态文件仍更新（last_run_at）
    state = load_state(skills)
    assert "last_run_at" in state


def test_run_curator_no_agent_factory(seeded_home):
    """无 agent_factory 时只跑第 1 阶段。"""
    skills = seeded_home / "skills"
    _make_skill(skills, "old", created_by="agent", days_ago=120)

    report = run_curator_review(skills)
    assert report["transitions"]["archived"] == 1
    assert report["consolidations"]["consolidations"] == []


# ---------------------------------------------------------------------------
# _parse_consolidation_output
# ---------------------------------------------------------------------------

def test_parse_consolidation_yaml():
    output = """一些文字
```yaml
consolidations:
  - from: old-skill
    into: umbrella
    reason: 主题重叠
prunings:
  - name: useless
    reason: 已废弃
```
更多文字"""
    result = _parse_consolidation_output(output)
    assert len(result["consolidations"]) == 1
    assert result["consolidations"][0]["from"] == "old-skill"
    assert len(result["prunings"]) == 1


def test_parse_consolidation_no_yaml():
    output = "纯文本，无 yaml 块"
    result = _parse_consolidation_output(output)
    assert result["consolidations"] == []
    assert result["prunings"] == []


# ---------------------------------------------------------------------------
# 跨会话 transcript 整理（should_consolidate + consolidate_transcripts）
# ---------------------------------------------------------------------------

import json as _json
import types as _types


class _FakeLLM:
    """对齐现场 client 形态：chat_completions 是 async（LLMClient/AuxLLMRouter 均如此）。

    payload 传 str（合法响应内容）或 Exception（模拟调用失败）。
    """

    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    async def chat_completions(self, messages, **kwargs):
        self.calls.append(messages)
        if isinstance(self._payload, Exception):
            raise self._payload
        msg = _types.SimpleNamespace(content=self._payload)
        return _types.SimpleNamespace(choices=[_types.SimpleNamespace(message=msg)])


class _FakeSessionStore:
    """list_sessions 返回 N 个会话；get_messages 返回固定消息（empty=True 全空）。"""

    def __init__(self, n=5, empty=False):
        self.sessions = [
            {"id": f"session-{i}-abcd1234", "title": f"会话{i}"}
            for i in range(n)
        ]
        self.empty = empty

    def list_sessions(self, *, limit=50, offset=0):
        return self.sessions[offset:offset + limit]

    def get_messages(self, session_id, *, limit=None):
        if self.empty:
            return []
        return [
            {"role": "user", "content": "怎么配置 pytest fixture"},
            {"role": "assistant", "content": "在 conftest.py 里定义 fixture"},
        ]


class _FakeMemoryStore:
    def __init__(self):
        self.saved = []

    def save(self, **kwargs):
        self.saved.append(kwargs)
        return f"general#fake{len(self.saved)}"


class TestConsolidateTranscripts:
    def test_should_consolidate_gates(self):
        import time as _t
        from agent.curator import should_consolidate
        now = _t.time()
        state = {"last_consolidate_at": now - 25 * 3600, "sessions_seen": 10}
        assert should_consolidate(state, now=now, new_sessions_since=6) is True
        # 间隔不够
        assert should_consolidate(
            {"last_consolidate_at": now - 3600, "sessions_seen": 10},
            now=now, new_sessions_since=6) is False
        # 会话不够
        assert should_consolidate(
            {"last_consolidate_at": now - 25 * 3600, "sessions_seen": 10},
            now=now, new_sessions_since=2) is False

    def test_consolidate_saves_new_memories(self, tmp_path):
        """从最近 5 个会话提炼 → memory_store.save 被调（type=project/reference）。"""
        from agent.curator import consolidate_transcripts
        items = [
            {"type": "project", "name": "依赖统一用 uv",
             "description": "项目依赖管理约定", "summary": "加包用 uv add",
             "body": "uv add <包名>；禁止 pip install"},
            {"type": "reference", "name": "复刻指南位置",
             "description": "设计权衡文档位置", "summary": "replication-guide 目录",
             "body": "hermes-agent-main/replication-guide"},
            {"type": "user", "name": "非法类型被过滤",
             "description": "只允许 project/reference", "summary": "", "body": ""},
        ]
        llm = _FakeLLM(_json.dumps(items, ensure_ascii=False))
        mem = _FakeMemoryStore()
        saved = consolidate_transcripts(_FakeSessionStore(5), mem, llm=llm)
        assert saved == 2                    # == fake llm 给的合法条数
        assert len(mem.saved) == 2
        assert all(m["type"] in ("project", "reference") for m in mem.saved)
        assert mem.saved[0]["source_session_id"] == "curator:consolidate"
        assert len(llm.calls) == 1           # LLM 只调一次

    def test_consolidate_failopen_llm_error(self, tmp_path):
        """LLM 抛异常 → fail-open 返回 0（不抛出）。"""
        from agent.curator import consolidate_transcripts
        llm = _FakeLLM(RuntimeError("api down"))
        saved = consolidate_transcripts(
            _FakeSessionStore(5), _FakeMemoryStore(), llm=llm)
        assert saved == 0

    def test_consolidate_fewer_than_5_sessions(self, tmp_path):
        """非空会话不足 5 个 → 不调 LLM 直接返回 0。"""
        from agent.curator import consolidate_transcripts
        llm = _FakeLLM("[]")
        saved = consolidate_transcripts(
            _FakeSessionStore(3), _FakeMemoryStore(), llm=llm)
        assert saved == 0
        assert llm.calls == []               # 没到门槛不烧 token


def test_run_curator_consolidate_wiring(seeded_home):
    """run_curator_review 传齐组件 + 双门控通过 → consolidate 跑且状态两键落盘。"""
    skills = seeded_home / "skills"
    items = [{"type": "project", "name": "测试约定", "description": "pytest 跑",
              "summary": "uv run pytest tests/", "body": ""}]
    report = run_curator_review(
        skills,
        session_store=_FakeSessionStore(5),
        memory_store=_FakeMemoryStore(),
        llm=_FakeLLM(_json.dumps(items, ensure_ascii=False)),
    )
    assert report["consolidated_memories"] == 1
    state = load_state(skills)
    assert "last_consolidate_at" in state
    assert state["sessions_seen"] == 5


def test_run_curator_consolidate_gate_closed(seeded_home):
    """门控未过（刚整理过）→ 不跑 consolidate，sessions_seen 不被改写。"""
    import time as _t
    skills = seeded_home / "skills"
    state = load_state(skills)
    state["last_consolidate_at"] = _t.time()   # 1 小时内整理过
    state["sessions_seen"] = 5
    save_state(skills, state)

    llm = _FakeLLM("[]")
    report = run_curator_review(
        skills,
        session_store=_FakeSessionStore(8),   # 3 个新会话 < 5，双门都关
        memory_store=_FakeMemoryStore(),
        llm=llm,
    )
    assert report["consolidated_memories"] == 0
    assert llm.calls == []
    state = load_state(skills)
    assert state["sessions_seen"] == 5        # 未被改写
