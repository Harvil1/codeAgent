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
