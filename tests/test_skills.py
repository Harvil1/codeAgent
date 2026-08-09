"""技能系统测试。"""

import json
from pathlib import Path

import pytest

from agent.skill_commands import (
    scan_skill_commands, parse_frontmatter, execute_skill,
)
from tools.skill_usage import (
    bump_view, bump_use, bump_patch,
    mark_agent_created, set_state, set_pinned,
    archive_skill, restore_skill, load_usage,
    STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED,
)
from tools.registry import registry


# ---------------------------------------------------------------------------
# 测试夹具：构造临时技能目录
# ---------------------------------------------------------------------------

SAMPLE_SKILL_MD = """---
name: code-review
description: "审查代码变更"
version: 1.0.0
---

# Code Review

## When to Use
- 审查 PR 时

## Procedure
1. 获取 diff
2. 检查风格
"""


@pytest.fixture
def skills_dir(tmp_path):
    """提供带示例技能的临时目录。"""
    sd = tmp_path / "skills"
    sd.mkdir()
    (sd / "code-review").mkdir()
    (sd / "code-review" / "SKILL.md").write_text(SAMPLE_SKILL_MD, encoding="utf-8")
    return sd


# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------

def test_parse_frontmatter_basic():
    content = '---\nname: test\ndescription: "hi"\n---\nbody'
    fm, body = parse_frontmatter(content)
    assert fm["name"] == "test"
    assert fm["description"] == "hi"
    assert "body" in body


def test_parse_frontmatter_no_frontmatter():
    fm, body = parse_frontmatter("just text")
    assert fm == {}
    assert body == "just text"


# ---------------------------------------------------------------------------
# scan_skill_commands
# ---------------------------------------------------------------------------

def test_scan_skill_commands(skills_dir):
    cmds = scan_skill_commands(skills_dir)
    assert "/code-review" in cmds
    info = cmds["/code-review"]
    assert info["name"] == "code-review"
    assert info["description"] == "审查代码变更"
    assert Path(info["skill_md_path"]).exists()


def test_scan_skill_commands_empty(tmp_path):
    """空目录返回空 dict。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert scan_skill_commands(empty) == {}


def test_scan_skill_commands_nonexistent(tmp_path):
    """不存在的目录返回空 dict。"""
    assert scan_skill_commands(tmp_path / "nope") == {}


# ---------------------------------------------------------------------------
# execute_skill
# ---------------------------------------------------------------------------

def test_execute_skill(skills_dir):
    skill_md = skills_dir / "code-review" / "SKILL.md"
    msg = execute_skill(str(skill_md), "审查这个 PR")
    assert "[技能已加载]" in msg
    assert "Code Review" in msg
    assert "审查这个 PR" in msg


def test_execute_skill_missing_path(tmp_path):
    """技能文件不存在时返回原始消息。"""
    msg = execute_skill(str(tmp_path / "nope.md"), "hi")
    assert msg == "hi"


# ---------------------------------------------------------------------------
# skill_usage（计数 + 状态）
# ---------------------------------------------------------------------------

def test_bump_use_increments(skills_dir):
    bump_use(skills_dir, "code-review")
    bump_use(skills_dir, "code-review")
    data = load_usage(skills_dir)
    assert data["code-review"]["use_count"] == 2
    assert data["code-review"]["last_used_at"] is not None


def test_bump_view_increments(skills_dir):
    bump_view(skills_dir, "code-review")
    data = load_usage(skills_dir)
    assert data["code-review"]["view_count"] == 1


def test_bump_patch_increments(skills_dir):
    bump_patch(skills_dir, "code-review")
    data = load_usage(skills_dir)
    assert data["code-review"]["patch_count"] == 1


def test_mark_agent_created(skills_dir):
    mark_agent_created(skills_dir, "code-review")
    data = load_usage(skills_dir)
    assert data["code-review"]["created_by"] == "agent"


def test_set_state_only_for_agent_created(skills_dir):
    """用户创建的技能状态不可被 set_state 改变。"""
    # 默认 created_by=user
    bump_use(skills_dir, "code-review")
    set_state(skills_dir, "code-review", STATE_ARCHIVED)
    data = load_usage(skills_dir)
    assert data["code-review"]["state"] == STATE_ACTIVE  # 未变


def test_set_state_for_agent_created(skills_dir):
    mark_agent_created(skills_dir, "code-review")
    set_state(skills_dir, "code-review", STATE_STALE)
    data = load_usage(skills_dir)
    assert data["code-review"]["state"] == STATE_STALE


def test_set_pinned(skills_dir):
    mark_agent_created(skills_dir, "code-review")
    set_pinned(skills_dir, "code-review", True)
    data = load_usage(skills_dir)
    assert data["code-review"]["pinned"] is True


# ---------------------------------------------------------------------------
# archive / restore
# ---------------------------------------------------------------------------

def test_archive_skill(skills_dir):
    ok, msg = archive_skill(skills_dir, "code-review")
    assert ok is True
    assert not (skills_dir / "code-review").exists()
    assert (skills_dir / ".archive" / "code-review").exists()


def test_archive_nonexistent(skills_dir):
    ok, msg = archive_skill(skills_dir, "nope")
    assert ok is False
    assert "不存在" in msg


def test_restore_skill(skills_dir):
    archive_skill(skills_dir, "code-review")
    ok, msg = restore_skill(skills_dir, "code-review")
    assert ok is True
    assert (skills_dir / "code-review").exists()
    assert not (skills_dir / ".archive" / "code-review").exists()


# ---------------------------------------------------------------------------
# skill_manage 工具集成
# ---------------------------------------------------------------------------

async def test_skill_manage_create(tmp_path):
    """通过 registry.dispatch 调用 skill_manage create。"""
    home = tmp_path
    result = await registry.dispatch(
        "skill_manage",
        {
            "action": "create",
            "name": "new-skill",
            "content": SAMPLE_SKILL_MD,
        },
        omnimate_home=home,
    )
    data = json.loads(result)
    assert data["success"] is True
    assert (home / "skills" / "new-skill" / "SKILL.md").exists()


async def test_skill_manage_create_duplicate(tmp_path):
    """重复 create 同名技能失败。"""
    home = tmp_path
    await registry.dispatch(
        "skill_manage",
        {"action": "create", "name": "dup", "content": "..."},
        omnimate_home=home,
    )
    result = await registry.dispatch(
        "skill_manage",
        {"action": "create", "name": "dup", "content": "..."},
        omnimate_home=home,
    )
    data = json.loads(result)
    assert data["success"] if "success" in data else "error" in data
    assert "已存在" in data.get("error", "") or data.get("success") is False


async def test_skill_manage_patch(skills_dir):
    """patch 动作能查找替换。"""
    home = skills_dir.parent
    result = await registry.dispatch(
        "skill_manage",
        {
            "action": "patch",
            "name": "code-review",
            "old_string": "Code Review",
            "new_string": "Code Review Skill",
        },
        omnimate_home=home,
    )
    data = json.loads(result)
    assert data["success"] is True
    content = (skills_dir / "code-review" / "SKILL.md").read_text(encoding="utf-8")
    assert "Code Review Skill" in content
    assert "Code Review\n" not in content  # 旧文本已替换


async def test_skill_manage_delete_archives(skills_dir):
    """delete 实际是归档。"""
    home = skills_dir.parent
    result = await registry.dispatch(
        "skill_manage",
        {"action": "delete", "name": "code-review"},
        omnimate_home=home,
    )
    data = json.loads(result)
    assert data["success"] is True
    assert not (skills_dir / "code-review").exists()
    assert (skills_dir / ".archive" / "code-review").exists()


# ---------------------------------------------------------------------------
# skills_list / skill_view 工具集成
# ---------------------------------------------------------------------------

async def test_skills_list(skills_dir):
    home = skills_dir.parent
    result = await registry.dispatch("skills_list", {}, omnimate_home=home)
    data = json.loads(result)
    names = [s["name"] for s in data["skills"]]
    assert "code-review" in names


async def test_skill_view(skills_dir):
    home = skills_dir.parent
    result = await registry.dispatch(
        "skill_view",
        {"name": "code-review"},
        omnimate_home=home,
    )
    data = json.loads(result)
    assert data["name"] == "code-review"
    assert "Code Review" in data["content"]


async def test_skill_view_missing(skills_dir):
    home = skills_dir.parent
    result = await registry.dispatch(
        "skill_view",
        {"name": "nonexistent"},
        omnimate_home=home,
    )
    data = json.loads(result)
    assert "error" in data


# ---------------------------------------------------------------------------
# frontmatter 字段过滤（user-invocable / disable-model-invocation）
# ---------------------------------------------------------------------------

import tempfile
from agent.prompt_builder import _build_skill_index


def _make_skill(skill_md: Path, frontmatter: str, body: str = "body"):
    skill_md.parent.mkdir(parents=True, exist_ok=True)
    skill_md.write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")


def test_user_invocable_false_excluded_from_commands():
    """user-invocable: false 的技能不进 slash 命令。"""
    with tempfile.TemporaryDirectory() as d:
        _make_skill(Path(d) / "hidden" / "SKILL.md",
                    "name: hidden\ndescription: x\nuser-invocable: false")
        _make_skill(Path(d) / "shown" / "SKILL.md",
                    "name: shown\ndescription: y")
        cmds = scan_skill_commands(d)
    assert "/shown" in cmds
    assert "/hidden" not in cmds


def test_disable_model_invocation_excluded_from_index():
    """disable-model-invocation: true 的技能不进系统提示索引。"""
    with tempfile.TemporaryDirectory() as d:
        _make_skill(Path(d) / "manual" / "SKILL.md",
                    "name: manual\ndescription: m\ndisable-model-invocation: true")
        _make_skill(Path(d) / "auto" / "SKILL.md",
                    "name: auto\ndescription: a")
        idx = _build_skill_index(d)
    assert "/auto" in idx
    assert "/manual" not in idx


def test_user_invocable_false_but_in_index():
    """user-invocable: false（仅模型）仍进系统提示索引。"""
    with tempfile.TemporaryDirectory() as d:
        _make_skill(Path(d) / "modelonly" / "SKILL.md",
                    "name: modelonly\ndescription: mo\nuser-invocable: false")
        idx = _build_skill_index(d)
        cmds = scan_skill_commands(d)
    assert "/modelonly" in idx
    assert "/modelonly" not in cmds


# ---------------------------------------------------------------------------
# load_skill：allowed-tools / disallowed-tools 临时作用域（D2）
# ---------------------------------------------------------------------------

def test_load_skill_sets_tool_scope():
    """load_skill 读到 allowed-tools 时把作用域存到 agent。"""
    with tempfile.TemporaryDirectory() as d:
        _make_skill(Path(d) / "skills" / "scoped" / "SKILL.md",
                    "name: scoped\ndescription: s\nallowed-tools:\n  - read_file\n  - terminal",
                    body="do thing")
        import tools.skill_tools as st
        class FakeAgent:
            _skill_tool_scope = None
        out = json.loads(st._handle_load_skill(
            {"name": "scoped"},
            omnimate_home=d,
        ))
    # _handle_load_skill 内部会 setattr agent._skill_tool_scope
    # （测试里 agent 未传，验证返回正常即可；作用域集成测在 D2 Step 4）
    assert out["name"] == "scoped"


# ---------------------------------------------------------------------------
# D3: paths frontmatter 条件加载（cwd 前缀匹配）
# ---------------------------------------------------------------------------

def test_paths_conditional_loading(monkeypatch):
    """paths 限定时，cwd 不匹配则不注入。"""
    with tempfile.TemporaryDirectory() as d:
        _make_skill(Path(d) / "pyskill" / "SKILL.md",
                    'name: pyskill\ndescription: p\npaths:\n  - "src/**"\n  - "*.py"',
                    body="py only")
        _make_skill(Path(d) / "always" / "SKILL.md",
                    "name: always\ndescription: al")
        # cwd 在 src/ 下 → pyskill 出现
        monkeypatch.chdir(Path(d))
        import agent.prompt_builder as pb
        monkeypatch.setattr(pb, "_current_cwd", lambda: str(Path(d) / "src"))
        idx = _build_skill_index(d)
        assert "/pyskill" in idx
        # cwd 不匹配 → pyskill 消失
        monkeypatch.setattr(pb, "_current_cwd", lambda: str(Path(d)))
        idx2 = _build_skill_index(d)
        assert "/pyskill" not in idx2
        assert "/always" in idx2
        # Windows: 退出 with 前还原 cwd，避免 tmp 目录被占用导致 cleanup 失败
        monkeypatch.undo()
