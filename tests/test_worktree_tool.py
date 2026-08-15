"""worktree_tool 测试（CCAR12 Task 6）：会话级 worktree 进出。

覆盖四层：
1. workspace_context 的 set/clear_session_workspace_cwd（长效 set + 幂等 clear）
2. 工具 handler 行为（git repo fixture：enter 建目录+切 cwd / exit 恢复 /
   复用 / keep 语义 / 非 git 降级临时目录）
3. handler dispatch 契约（args, **kwargs，防 silent-dead-code）+ schema 键契约
4. registry 注册 + 并发分类 + core 可见性
"""
import inspect
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import tools.worktree_tool  # noqa: 触发注册
from agent.workspace_context import (
    _workspace_cwd,
    clear_session_workspace_cwd,
    get_workspace_cwd,
    set_session_workspace_cwd,
)
from toolsets import resolve_toolset
from tools.registry import registry
from tools.worktree_tool import (
    WORKTREE_ENTER_SCHEMA,
    WORKTREE_EXIT_SCHEMA,
    _handle_worktree_enter,
    _handle_worktree_exit,
    _reset_session_worktree,
)


@pytest.fixture(autouse=True)
def _clean_session_state():
    """每个测试前后清空会话 worktree 状态（module-level，测试间必须隔离）。"""
    _reset_session_worktree()
    yield
    _reset_session_worktree()


def _make_agent_ref(hooks_registry=None):
    """最小 agent_ref mock：只带 hooks_registry（可为 None）。"""
    agent = MagicMock()
    agent.hooks_registry = hooks_registry
    return agent


def _init_git_repo(tmp_path: Path) -> Path:
    """建一个最小 git repo（有 1 个 commit，能建 worktree）。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    for cmd in (["git", "init"], ["git", "config", "user.email", "t@t"],
                ["git", "config", "user.name", "t"],
                ["git", "commit", "--allow-empty", "-m", "init"]):
        subprocess.run(cmd, cwd=str(repo), capture_output=True, check=True)
    return repo


def _cd(tmp_path: Path):
    """测试内临时切 os.getcwd（get_workspace_cwd 未设置时 fallback 它）。"""
    import os
    old = os.getcwd()
    os.chdir(str(tmp_path))
    yield tmp_path
    os.chdir(old)


# ---------------------------------------------------------------------------
# 1. workspace_context：set/clear_session_workspace_cwd
# ---------------------------------------------------------------------------

class TestSessionWorkspaceCwd:

    def test_set_then_get_returns_path(self, tmp_path):
        clear_session_workspace_cwd()
        set_session_workspace_cwd(str(tmp_path))
        assert get_workspace_cwd() == str(tmp_path)
        clear_session_workspace_cwd()

    def test_clear_restores_previous_value(self, tmp_path):
        clear_session_workspace_cwd()
        before = get_workspace_cwd()
        set_session_workspace_cwd(str(tmp_path))
        assert get_workspace_cwd() == str(tmp_path)
        clear_session_workspace_cwd()
        assert get_workspace_cwd() == before

    def test_clear_idempotent_when_not_set(self):
        clear_session_workspace_cwd()  # 不抛即过
        clear_session_workspace_cwd()
        assert _workspace_cwd.get() is None or get_workspace_cwd()


# ---------------------------------------------------------------------------
# 2. 工具 handler 行为（git repo fixture）
# ---------------------------------------------------------------------------

class TestWorktreeEnter:

    async def test_enter_denied_for_subagent(self):
        """spawn_depth>0 的子代理不能切主对话的会话 worktree（CCAR13 A1，
        CCAR12 final review follow-up：ContextVar 进程级共享，子代理 enter
        会劫持主对话 cwd）。"""
        agent = MagicMock()
        agent.spawn_depth = 1
        result = await _handle_worktree_enter({}, agent_ref=agent)
        data = json.loads(result)
        assert data["error_type"] == "permission_denied"

    async def test_enter_allowed_for_main_agent_mock(self, tmp_path):
        """agent_ref 是 MagicMock 但 spawn_depth 为真实 int 0 → 放行（isinstance
        守卫不误伤无 spawn_depth 属性的 mock/对象）。"""
        agent = MagicMock()
        agent.spawn_depth = 0
        repo = _init_git_repo(tmp_path)
        for d in _cd(repo):
            result = await _handle_worktree_enter({"name": "main-ok"}, agent_ref=agent)
            data = json.loads(result)
            assert "error" not in data, data
            await _handle_worktree_exit({"keep": False})

    async def test_enter_creates_worktree_and_switches_cwd(self, tmp_path):
        repo = _init_git_repo(tmp_path)
        for d in _cd(repo):
            result = await _handle_worktree_enter({"name": "feature-x"}, agent_ref=None)
            data = json.loads(result)
            assert "error" not in data, data
            wt = repo / ".worktrees" / "feature-x"
            assert Path(data["path"]) == wt
            assert wt.exists()
            assert data["reused"] is False
            assert data["workspace_type"] == "git"
            assert data["branch"]
            # 会话 cwd 已切换到 worktree
            assert get_workspace_cwd() == str(wt)
            # 退出恢复 + 清理（测试收尾）
            await _handle_worktree_exit({"keep": False})

    async def test_enter_default_name_when_missing(self, tmp_path):
        repo = _init_git_repo(tmp_path)
        for d in _cd(repo):
            result = await _handle_worktree_enter({}, agent_ref=None)
            data = json.loads(result)
            assert "error" not in data, data
            wt = Path(data["path"])
            assert wt.parent == repo / ".worktrees"
            assert wt.name.startswith("wt-")
            await _handle_worktree_exit({"keep": False})

    async def test_enter_reuses_existing_worktree(self, tmp_path):
        repo = _init_git_repo(tmp_path)
        for d in _cd(repo):
            # 第一次进入 + 退出（keep=True 保留目录）
            first = json.loads(await _handle_worktree_enter({"name": "reuse-me"}, agent_ref=None))
            assert first["reused"] is False
            await _handle_worktree_exit({"keep": True})

            # 第二次进入同名 → 复用（reused=True，同一路径）
            second = json.loads(await _handle_worktree_enter({"name": "reuse-me"}, agent_ref=None))
            assert second["reused"] is True
            assert second["path"] == first["path"]
            assert get_workspace_cwd() == first["path"]
            await _handle_worktree_exit({"keep": False})

    async def test_enter_rejects_when_already_in_worktree(self, tmp_path):
        repo = _init_git_repo(tmp_path)
        for d in _cd(repo):
            await _handle_worktree_enter({"name": "a"}, agent_ref=None)
            result = await _handle_worktree_enter({"name": "b"}, agent_ref=None)
            data = json.loads(result)
            assert data["error_type"] == "already_in_worktree"
            await _handle_worktree_exit({"keep": False})

    async def test_enter_sanitizes_bad_name(self, tmp_path):
        repo = _init_git_repo(tmp_path)
        for d in _cd(repo):
            result = await _handle_worktree_enter({"name": "..\\evil ..name"}, agent_ref=None)
            data = json.loads(result)
            assert "error" not in data, data
            name = Path(data["path"]).name
            assert ".." not in name and "\\" not in name and " " not in name
            await _handle_worktree_exit({"keep": False})

    async def test_enter_fires_cwd_changed_hook(self, tmp_path):
        repo = _init_git_repo(tmp_path)
        for d in _cd(repo):
            hooks = MagicMock()
            await _handle_worktree_enter({"name": "hooked"}, agent_ref=_make_agent_ref(hooks))
            assert hooks.run_cwd_changed.call_count == 1
            payload = hooks.run_cwd_changed.call_args[0][0]
            assert payload["new"] == str(repo / ".worktrees" / "hooked")
            assert "old" in payload
            await _handle_worktree_exit({"keep": False})

    async def test_enter_agent_ref_none_skips_hook(self, tmp_path):
        """agent_ref=None（hooks 也拿不到）→ 跳过 hook 不抛。"""
        repo = _init_git_repo(tmp_path)
        for d in _cd(repo):
            result = await _handle_worktree_enter({"name": "no-hook"}, agent_ref=None)
            assert "error" not in json.loads(result)
            await _handle_worktree_exit({"keep": False})

    async def test_enter_not_git_degrades_to_temp_dir(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        for d in _cd(plain):
            result = await _handle_worktree_enter({"name": "nofail"}, agent_ref=None)
            data = json.loads(result)
            assert "error" not in data, data
            assert data["workspace_type"] == "temp"
            assert data["reused"] is False
            # 临时目录不在 repo 内（系统 temp）
            assert not Path(data["path"]).is_relative_to(plain)
            assert get_workspace_cwd() == data["path"]
            await _handle_worktree_exit({"keep": False})


class TestWorktreeExit:

    async def test_exit_restores_cwd(self, tmp_path):
        repo = _init_git_repo(tmp_path)
        for d in _cd(repo):
            before = get_workspace_cwd()
            await _handle_worktree_enter({"name": "inout"}, agent_ref=None)
            assert get_workspace_cwd() != before
            result = await _handle_worktree_exit({"keep": True})
            assert "error" not in json.loads(result)
            assert get_workspace_cwd() == before

    async def test_exit_keep_true_preserves_worktree(self, tmp_path):
        repo = _init_git_repo(tmp_path)
        for d in _cd(repo):
            enter = json.loads(await _handle_worktree_enter({"name": "keepme"}, agent_ref=None))
            result = json.loads(await _handle_worktree_exit({"keep": True}))
            assert result["kept"] is True and result["cleaned"] is False
            assert Path(enter["path"]).exists()

    async def test_exit_keep_default_true(self, tmp_path):
        """keep 缺省 = True（保守默认，不删用户目录）。"""
        repo = _init_git_repo(tmp_path)
        for d in _cd(repo):
            enter = json.loads(await _handle_worktree_enter({"name": "default"}, agent_ref=None))
            result = json.loads(await _handle_worktree_exit({}))
            assert result["kept"] is True
            assert Path(enter["path"]).exists()

    async def test_exit_keep_false_clean_when_no_changes(self, tmp_path):
        repo = _init_git_repo(tmp_path)
        for d in _cd(repo):
            enter = json.loads(await _handle_worktree_enter({"name": "clean"}, agent_ref=None))
            result = json.loads(await _handle_worktree_exit({"keep": False}))
            assert result["cleaned"] is True and result["kept"] is False
            assert not Path(enter["path"]).exists()
            # 分支一并删除
            branches = subprocess.run(
                ["git", "branch", "--list", enter["branch"]],
                cwd=str(repo), capture_output=True, text=True,
            ).stdout.strip()
            assert branches == ""

    async def test_exit_keep_false_keeps_when_has_changes(self, tmp_path):
        repo = _init_git_repo(tmp_path)
        for d in _cd(repo):
            enter = json.loads(await _handle_worktree_enter({"name": "dirty"}, agent_ref=None))
            # 在 worktree 里写个未跟踪文件 → git status --porcelain 非空
            (Path(enter["path"]) / "new.txt").write_text("x", encoding="utf-8")
            result = json.loads(await _handle_worktree_exit({"keep": False}))
            assert result["kept"] is True and result["cleaned"] is False
            assert result["reason"] == "has_changes"
            assert Path(enter["path"]).exists()

    async def test_exit_not_in_worktree(self):
        result = json.loads(await _handle_worktree_exit({}))
        assert result["error_type"] == "not_in_worktree"

    async def test_exit_temp_workspace_clean_when_no_changes(self, tmp_path):
        """非 git 降级路径：空 temp 目录 keep=False → 清理（listdir 检测）。"""
        plain = tmp_path / "plain2"
        plain.mkdir()
        for d in _cd(plain):
            enter = json.loads(await _handle_worktree_enter({"name": "tmp"}, agent_ref=None))
            assert enter["workspace_type"] == "temp"
            result = json.loads(await _handle_worktree_exit({"keep": False}))
            assert result["cleaned"] is True
            assert not Path(enter["path"]).exists()


# ---------------------------------------------------------------------------
# 3. dispatch 契约（CCAR8 教训：handler 必须是 (args, **kwargs)）
# ---------------------------------------------------------------------------

def test_handler_signature_matches_dispatch_contract():
    """dispatch 调 handler(args, **kwargs)，两个 handler 签名必须兼容。"""
    for handler in (_handle_worktree_enter, _handle_worktree_exit):
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())
        assert params[0].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert params[0].name == "args"
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)


# ---------------------------------------------------------------------------
# 4. schema 键契约（CCAR11：OpenAI "parameters" 不是 "inputSchema"）
# ---------------------------------------------------------------------------

def test_schemas_use_openai_parameters_key():
    for schema in (WORKTREE_ENTER_SCHEMA, WORKTREE_EXIT_SCHEMA):
        assert "parameters" in schema, f"{schema.get('name')} 缺 parameters 键"
        assert "inputSchema" not in schema


def test_schema_fields():
    enter_props = WORKTREE_ENTER_SCHEMA["parameters"]["properties"]
    assert set(enter_props) == {"name"}
    exit_props = WORKTREE_EXIT_SCHEMA["parameters"]["properties"]
    assert set(exit_props) == {"keep"}
    assert exit_props["keep"]["default"] is True


# ---------------------------------------------------------------------------
# 5. registry 注册 + 分类 + core 可见性
# ---------------------------------------------------------------------------

def test_worktree_tools_registered_in_core():
    for name in ("worktree_enter", "worktree_exit"):
        entry = registry.get(name)
        assert entry is not None, f"{name} 未注册"
        assert entry.toolset == "core"
        assert entry.isConcurrencySafe is False  # 会话级状态变更，串行


def test_worktree_tools_in_core_toolset_visible():
    """两工具必须进 _CORE_TOOLS 才对 LLM 可见（发现 ≠ 可见）。"""
    core = set(resolve_toolset("core"))
    assert "worktree_enter" in core
    assert "worktree_exit" in core


def test_handlers_are_async_not_threaded():
    """【Task 6 fix Critical】handler 必须是 async def（is_async=True）。

    sync handler 经 dispatch 的 asyncio.to_thread 跑在 context 拷贝里——
    会话 cwd set 不回透主循环 + exit 的 token reset 跨 context 必炸。
    本测试防"改回 sync"回归（配合下面的 dispatch 端到端测试）。
    """
    for name in ("worktree_enter", "worktree_exit"):
        entry = registry.get(name)
        assert entry.is_async is True, f"{name} 必须注册 is_async=True"
        assert inspect.iscoroutinefunction(entry.handler), (
            f"{name} handler 必须是 async def（to_thread context 拷贝会让"
            "会话级 cwd 切换静默失效）"
        )


# ---------------------------------------------------------------------------
# 6. dispatch 端到端（Task 6 fix Critical 回归——单元直调绕过 to_thread，
#    这就是漏检原因；必须经 registry.dispatch 验证 context 回透）
# ---------------------------------------------------------------------------

async def test_enter_exit_via_registry_dispatch(tmp_path):
    """经 registry.dispatch 端到端：enter 真切 cwd，exit 真恢复（防 to_thread context 拷贝回归）。

    复现路径：sync handler → asyncio.to_thread（context 拷贝到 worker 线程）
    → set_session_workspace_cwd 只改拷贝（enter 静默失效）+ exit 在拷贝
    context 里 reset 主 token → ValueError → tool_exception（状态机死锁）。
    断言全部在 dispatch 外做——主 context 视角。
    """
    repo = _init_git_repo(tmp_path)
    for d in _cd(repo):
        before = get_workspace_cwd()

        enter = json.loads(
            await registry.dispatch("worktree_enter", {"name": "e2e"}, agent_ref=None)
        )
        assert "error" not in enter, enter
        assert Path(enter["path"]) == repo / ".worktrees" / "e2e"
        # dispatch 外、主 context 视角：cwd 必须真的切过去
        # （to_thread context 拷贝回归恰好死在这——handler 自报成功但主循环没切）
        assert get_workspace_cwd() == enter["path"]
        assert get_workspace_cwd() != before

        exit_result = json.loads(
            await registry.dispatch("worktree_exit", {"keep": True}, agent_ref=None)
        )
        assert "error" not in exit_result, exit_result
        assert exit_result["kept"] is True
        # 主 context 视角：cwd 恢复 + 无 tool_exception（跨 context reset 回归点）
        assert get_workspace_cwd() == before


async def test_reenter_via_registry_dispatch_not_deadlocked(tmp_path):
    """exit 后再 enter 不被 already_in_worktree 卡死（Critical 第 2 症状）。

    to_thread 回归形态：_session_worktree 是模块级全局（线程间共享，
    enter 会回写置位），但 exit 的 clear_session_workspace_cwd 在拷贝
    context 里 reset 主 token → ValueError → dispatch 捕获成
    tool_exception 且 _session_worktree 已置 None 也不清理干净 →
    主对话被卡在 already_in_worktree。端到端验证完整生命周期可循环。
    """
    repo = _init_git_repo(tmp_path)
    for d in _cd(repo):
        first = json.loads(
            await registry.dispatch("worktree_enter", {"name": "cycle"}, agent_ref=None)
        )
        assert "error" not in first
        out1 = json.loads(
            await registry.dispatch("worktree_exit", {"keep": False}, agent_ref=None)
        )
        assert "error" not in out1, out1
        assert out1["cleaned"] is True

        # 再进一次：不被 already_in_worktree 卡死
        second = json.loads(
            await registry.dispatch("worktree_enter", {"name": "cycle"}, agent_ref=None)
        )
        assert "error" not in second, second
        out2 = json.loads(
            await registry.dispatch("worktree_exit", {"keep": False}, agent_ref=None)
        )
        assert "error" not in out2, out2
