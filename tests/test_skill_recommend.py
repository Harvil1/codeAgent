"""技能评分 + 推荐测试。"""

import json
from pathlib import Path

import pytest

from tools.skill_usage import (
    set_rating,
    get_recommendations,
    load_usage,
    bump_use,
    bump_view,
    set_pinned,
)


@pytest.fixture
def skills_setup(tmp_path):
    """构造带 3 个技能的 skills_dir。"""
    sd = tmp_path / "skills"
    sd.mkdir()

    for name, desc in [
        ("python-tips", "Python 编程技巧"),
        ("git-flow", "Git 工作流"),
        ("debug-tricks", "调试技巧"),
    ]:
        skill_dir = sd / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\n\n内容",
            encoding="utf-8",
        )

    return sd


# ---------------------------------------------------------------------------
# set_rating
# ---------------------------------------------------------------------------

def test_set_rating_happy_path(skills_setup):
    ok, msg = set_rating(skills_setup, "python-tips", 5)
    assert ok is True
    assert "5 星" in msg

    data = load_usage(skills_setup)
    assert data["python-tips"]["rating"] == 5
    assert "rated_at" in data["python-tips"]


def test_set_rating_updates_existing(skills_setup):
    set_rating(skills_setup, "python-tips", 3)
    set_rating(skills_setup, "python-tips", 5)
    data = load_usage(skills_setup)
    assert data["python-tips"]["rating"] == 5


def test_set_rating_out_of_range(skills_setup):
    ok, msg = set_rating(skills_setup, "python-tips", 0)
    assert ok is False
    assert "越界" in msg

    ok, msg = set_rating(skills_setup, "python-tips", 6)
    assert ok is False


def test_set_rating_non_integer(skills_setup):
    ok, msg = set_rating(skills_setup, "python-tips", "three")
    assert ok is False


def test_set_rating_skill_not_found(skills_setup):
    ok, msg = set_rating(skills_setup, "nonexistent", 3)
    assert ok is False
    assert "不存在" in msg


# ---------------------------------------------------------------------------
# get_recommendations
# ---------------------------------------------------------------------------

def test_recommend_returns_skills_sorted_by_score(skills_setup):
    # python-tips: use_count=5, rating=4 → 5 + 8 = 13
    for _ in range(5):
        bump_use(skills_setup, "python-tips")
    set_rating(skills_setup, "python-tips", 4)

    # git-flow: use_count=10, rating=2 → 10 + 4 = 14
    for _ in range(10):
        bump_use(skills_setup, "git-flow")
    set_rating(skills_setup, "git-flow", 2)

    # debug-tricks: use_count=1, no rating → 1
    bump_use(skills_setup, "debug-tricks")

    recs = get_recommendations(skills_setup)
    assert len(recs) == 3
    # git-flow (14) > python-tips (13) > debug-tricks (1)
    assert recs[0]["name"] == "git-flow"
    assert recs[1]["name"] == "python-tips"
    assert recs[2]["name"] == "debug-tricks"


def test_recommend_respects_limit(skills_setup):
    for name in ["python-tips", "git-flow", "debug-tricks"]:
        bump_use(skills_setup, name)

    recs = get_recommendations(skills_setup, limit=2)
    assert len(recs) == 2


def test_recommend_pinned_gets_boost(skills_setup):
    """pinned 技能即使 use_count 略低也排前（+10 分 boost）。"""
    # debug-tricks 用 5 次
    for _ in range(5):
        bump_use(skills_setup, "debug-tricks")

    # python-tips 用 1 次，但 pinned（1 + 10 boost = 11 > 5）
    bump_use(skills_setup, "python-tips")
    # 直接改 pinned（set_pinned 要求 created_by=agent，绕过）
    data = load_usage(skills_setup)
    data["python-tips"]["created_by"] = "agent"
    data["python-tips"]["pinned"] = True
    from tools.skill_usage import save_usage
    save_usage(skills_setup, data)

    recs = get_recommendations(skills_setup)
    # python-tips (1 + 10 boost = 11) > debug-tricks (5)
    assert recs[0]["name"] == "python-tips"
    assert recs[0]["pinned"] is True


def test_recommend_empty_skills_dir(tmp_path):
    """空 skills_dir 返回空列表。"""
    sd = tmp_path / "empty"
    sd.mkdir()
    recs = get_recommendations(sd)
    assert recs == []


def test_recommend_includes_description(skills_setup):
    recs = get_recommendations(skills_setup)
    # 至少有一个含描述
    descs = [r["description"] for r in recs]
    assert any("Python" in d or "Git" in d or "调试" in d for d in descs)


def test_recommend_score_formula(skills_setup):
    """验证综合分公式：use*1 + rating*2 + view*0.1。"""
    bump_use(skills_setup, "python-tips")  # use=1
    bump_use(skills_setup, "python-tips")  # use=2
    bump_view(skills_setup, "python-tips")  # view=1
    bump_view(skills_setup, "python-tips")  # view=2
    bump_view(skills_setup, "python-tips")  # view=3
    set_rating(skills_setup, "python-tips", 5)  # rating=5

    recs = get_recommendations(skills_setup)
    py = next(r for r in recs if r["name"] == "python-tips")
    # 期望：2*1 + 5*2 + 3*0.1 = 2 + 10 + 0.3 = 12.3
    assert py["score"] == pytest.approx(12.3, abs=0.01)
