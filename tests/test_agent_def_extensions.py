"""Task N 测试：AgentDefinition 扩展 4 字段 + delegate_tool 接入。

覆盖：
1. AgentDefinition 加 4 字段（omit_claude_md / initial_prompt / required_mcp_servers / critical_reminder）
2. _parse_one 解析 frontmatter camelCase 字段
3. 默认值向后兼容（旧 .md 不含新字段不崩）
4. inject_cli_agents 接受新字段
5. omit_claude_md 端到端：child AIAgent 的 prompt_builder 跳过项目 OMNIMATE.md
"""
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. 字段可读写（dataclass 层）
# ---------------------------------------------------------------------------

def test_agent_definition_has_omit_claude_md_field():
    """AgentDefinition.omit_claude_md 默认 False，可设 True。"""
    from agent.agent_defs import AgentDefinition
    ad = AgentDefinition(name="x")
    assert ad.omit_claude_md is False  # 默认
    ad2 = AgentDefinition(name="y", omit_claude_md=True)
    assert ad2.omit_claude_md is True


def test_agent_definition_has_initial_prompt_field():
    """AgentDefinition.initial_prompt 默认空串，可设置。"""
    from agent.agent_defs import AgentDefinition
    ad = AgentDefinition(name="x")
    assert ad.initial_prompt == ""
    ad2 = AgentDefinition(name="y", initial_prompt="先检查 git status")
    assert ad2.initial_prompt == "先检查 git status"


def test_agent_definition_has_required_mcp_servers_field():
    """AgentDefinition.required_mcp_servers 默认空 list。"""
    from agent.agent_defs import AgentDefinition
    ad = AgentDefinition(name="x")
    assert ad.required_mcp_servers == []
    ad2 = AgentDefinition(
        name="y",
        required_mcp_servers=["github", "filesystem"],
    )
    assert ad2.required_mcp_servers == ["github", "filesystem"]


def test_agent_definition_has_critical_reminder_field():
    """AgentDefinition.critical_reminder 默认空串。"""
    from agent.agent_defs import AgentDefinition
    ad = AgentDefinition(name="x")
    assert ad.critical_reminder == ""
    ad2 = AgentDefinition(
        name="y",
        critical_reminder="绝对不能删除任何文件",
    )
    assert ad2.critical_reminder == "绝对不能删除任何文件"


# ---------------------------------------------------------------------------
# 2. frontmatter camelCase → snake_case 解析
# ---------------------------------------------------------------------------

def test_parse_frontmatter_all_4_new_fields(tmp_path):
    """frontmatter 含 omitClaudeMd / initialPrompt / requiredMcpServers / criticalReminder。"""
    from agent.agent_defs import _parse_one
    md = tmp_path / "rich.md"
    md.write_text(
        "---\n"
        "name: rich\n"
        "description: 富配置\n"
        "omitClaudeMd: true\n"
        'initialPrompt: "先看 README\\n再开始"\n'
        "requiredMcpServers:\n"
        "  - github\n"
        "  - filesystem\n"
        'criticalReminder: "禁止 rm -rf"\n'
        "---\n正文\n",
        encoding="utf-8",
    )
    ad = _parse_one(md)
    assert ad is not None
    assert ad.omit_claude_md is True
    assert "先看 README" in ad.initial_prompt
    assert ad.required_mcp_servers == ["github", "filesystem"]
    assert ad.critical_reminder == "禁止 rm -rf"


def test_parse_frontmatter_omit_claude_md_false_default(tmp_path):
    """frontmatter 不含 omitClaudeMd 时默认 False（向后兼容）。"""
    from agent.agent_defs import _parse_one
    md = tmp_path / "old.md"
    md.write_text(
        "---\nname: old\ndescription: 老定义\n---\n正文\n",
        encoding="utf-8",
    )
    ad = _parse_one(md)
    assert ad is not None
    assert ad.omit_claude_md is False
    assert ad.initial_prompt == ""
    assert ad.required_mcp_servers == []
    assert ad.critical_reminder == ""


def test_parse_frontmatter_initial_prompt_multiline(tmp_path):
    """initialPrompt 多行字符串（YAML block scalar）正确解析。"""
    from agent.agent_defs import _parse_one
    md = tmp_path / "multi.md"
    md.write_text(
        "---\n"
        "name: multi\n"
        "description: 多行\n"
        "initialPrompt: |\n"
        "  第一行\n"
        "  第二行\n"
        "---\n正文\n",
        encoding="utf-8",
    )
    ad = _parse_one(md)
    assert ad is not None
    assert "第一行" in ad.initial_prompt
    assert "第二行" in ad.initial_prompt


# ---------------------------------------------------------------------------
# 3. inject_cli_agents 接受新字段
# ---------------------------------------------------------------------------

def test_inject_cli_agents_accepts_new_fields():
    """CLI 注入也支持 4 个新字段。"""
    from agent.agent_defs import inject_cli_agents, clear_cli_injected, get_cli_injected
    clear_cli_injected()
    inject_cli_agents({
        "cli-agent": {
            "description": "from CLI",
            "prompt": "正文",
            "omitClaudeMd": True,
            "initialPrompt": "hello",
            "requiredMcpServers": ["s1"],
            "criticalReminder": "be careful",
        },
    })
    injected = get_cli_injected()
    assert "cli-agent" in injected
    ad = injected["cli-agent"]
    assert ad.omit_claude_md is True
    assert ad.initial_prompt == "hello"
    assert ad.required_mcp_servers == ["s1"]
    assert ad.critical_reminder == "be careful"
    clear_cli_injected()


# ---------------------------------------------------------------------------
# 4. scan_agent_defs 集成：项目级 .md 含新字段
# ---------------------------------------------------------------------------

def test_scan_finds_agent_with_new_fields(monkeypatch, tmp_path):
    """scan_agent_defs 正确加载含新字段的 .md 文件。"""
    user_dir = tmp_path / "u"; user_dir.mkdir()
    (user_dir / "rookie.md").write_text(
        "---\n"
        "name: rookie\n"
        "description: 新代理\n"
        "omitClaudeMd: true\n"
        "criticalReminder: 别用 rm\n"
        "---\n正文\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("agent.agent_defs._user_agents_dir", lambda: user_dir)
    monkeypatch.setattr("agent.agent_defs._project_agents_dir", lambda: tmp_path / "noexist")
    from agent.agent_defs import scan_agent_defs
    defs = scan_agent_defs()
    assert "rookie" in defs
    assert defs["rookie"].omit_claude_md is True
    assert defs["rookie"].critical_reminder == "别用 rm"


# ---------------------------------------------------------------------------
# 5. 端到端：build_system_prompt 支持 omit_project_memory flag
# ---------------------------------------------------------------------------

def test_build_system_prompt_layers_supports_omit_project_memory(tmp_path, monkeypatch):
    """omit_project_memory=True 时跳过项目 OMNIMATE.md 注入。"""
    # 在 tmp_path 下造一个 OMNIMATE.md
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "OMNIMATE.md").write_text(
        "# 项目规则\n禁止删除文件\n", encoding="utf-8",
    )
    # 让 workspace_context.get_workspace_cwd 指向 project_dir（函数内 import）
    import agent.workspace_context as wc
    monkeypatch.setattr(wc, "get_workspace_cwd", lambda: str(project_dir))

    from agent.prompt_builder import build_system_prompt_layers

    # 不带 omit：注入项目记忆
    layers_normal = build_system_prompt_layers()
    assert "项目规则" in layers_normal.context

    # 带 omit=True：不注入
    layers_omit = build_system_prompt_layers(omit_project_memory=True)
    assert "项目规则" not in layers_omit.context


# ---------------------------------------------------------------------------
# 6. 端到端：_run_child 把新字段传给 AIAgent + critical_reminder 拼到 system_prompt
# ---------------------------------------------------------------------------

def test_run_child_passes_omit_claude_md_to_aiagent(monkeypatch):
    """custom_def.omit_claude_md=True 时，AIAgent 收到 omit_project_memory=True。"""
    from agent.agent_defs import AgentDefinition
    from unittest.mock import patch

    custom_def = AgentDefinition(
        name="readonly-agent",
        description="read-only",
        omit_claude_md=True,
        critical_reminder="禁止写入",
        initial_prompt="先看 README",
    )
    monkeypatch.setattr("agent.agent_defs.get_agent_def", lambda n: custom_def)

    captured = {}

    class FakeChild:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_client = type("FakeClient", (), {})()
            self.model = kwargs.get("model")

        async def chat(self, msg):
            # 验证 initial_prompt 拼到了首 user turn
            assert "先看 README" in msg
            return "ok"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    from tools.delegate_tool import _run_child
    with patch("config.load_config", return_value={
        "model": {"name": "test", "api_key": "fake_key", "base_url": "http://x"},
    }):
        with patch("agent.AIAgent", FakeChild):
            with patch("agent.progress.ProgressReporter") as fake_prog:
                fake_prog.return_value.__enter__ = lambda s: None
                fake_prog.return_value.__exit__ = lambda s, *a: None
                with patch(
                    "agent.team.hallucination_check.verify_claims",
                    return_value=None,
                ):
                    with patch(
                        "agent.team.hallucination_check.append_warning",
                        lambda r, v: r,
                    ):
                        _run_child(
                            "goal", "ctx", "leaf",
                            subagent_type="readonly-agent")

    # omit_claude_md → omit_project_memory=True
    assert captured.get("omit_project_memory") is True
    # critical_reminder 拼到 system_prompt_override
    assert "禁止写入" in captured["system_prompt_override"]
    assert "CRITICAL REMINDER" in captured["system_prompt_override"]


def test_run_child_no_omit_when_flag_false(monkeypatch):
    """custom_def.omit_claude_md=False（默认）时，omit_project_memory=False。"""
    from agent.agent_defs import AgentDefinition
    from unittest.mock import patch

    custom_def = AgentDefinition(
        name="normal-agent",
        description="normal",
    )
    monkeypatch.setattr("agent.agent_defs.get_agent_def", lambda n: custom_def)

    captured = {}

    class FakeChild:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_client = type("FakeClient", (), {})()
            self.model = kwargs.get("model")

        async def chat(self, msg):
            return "ok"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    from tools.delegate_tool import _run_child
    with patch("config.load_config", return_value={
        "model": {"name": "test", "api_key": "fake_key", "base_url": "http://x"},
    }):
        with patch("agent.AIAgent", FakeChild):
            with patch("agent.progress.ProgressReporter") as fake_prog:
                fake_prog.return_value.__enter__ = lambda s: None
                fake_prog.return_value.__exit__ = lambda s, *a: None
                with patch(
                    "agent.team.hallucination_check.verify_claims",
                    return_value=None,
                ):
                    with patch(
                        "agent.team.hallucination_check.append_warning",
                        lambda r, v: r,
                    ):
                        _run_child(
                            "goal", "ctx", "leaf",
                            subagent_type="normal-agent")

    assert captured.get("omit_project_memory") is False
    # critical_reminder 默认空串 → 不拼 CRITICAL REMINDER 段
    assert "CRITICAL REMINDER" not in (captured.get("system_prompt_override") or "")


# ---------------------------------------------------------------------------
# 7. 端到端：FILE_CHANGED hook 在 write_file 成功后触发
# ---------------------------------------------------------------------------

def test_file_changed_hook_fires_on_write_file(tmp_path):
    """write_file 成功后触发 FILE_CHANGED hook。"""
    import json
    from agent.hooks import HookRegistry
    from agent.permission import add_extra_allowed_root, clear_extra_allowed_roots
    from tools.file_operations import _handle_write_file

    reg = HookRegistry()
    captured = []
    reg.register_file_changed(lambda p: captured.append(p), name="watcher")

    # 构造一个带 hooks_registry 的 fake agent_ref
    class FakeAgent:
        hooks_registry = reg
        session_id = "test-session"
        _checkpoint_track = None

    # CCAR13 Task 4: check_path 闸门 3 恢复白名单语义——tmp_path 不在
    # cwd 白名单内，注册为 extra root 才能走到写盘（finally clear 防泄漏）。
    add_extra_allowed_root(str(tmp_path))
    try:
        target = tmp_path / "out.txt"
        result = _handle_write_file(
            {"path": str(target), "content": "hello"},
            agent_ref=FakeAgent(),
        )
    finally:
        clear_extra_allowed_roots()
    data = json.loads(result)
    assert "bytes" in data  # 成功
    assert len(captured) == 1
    assert captured[0]["op"] == "write"
    assert captured[0]["path"] == str(target)
    assert captured[0]["session_id"] == "test-session"


def test_file_changed_hook_fires_on_str_replace(tmp_path):
    """str_replace 成功后触发 FILE_CHANGED hook（op=edit）。"""
    import json
    from agent.hooks import HookRegistry
    from agent.permission import add_extra_allowed_root, clear_extra_allowed_roots
    from tools.file_operations import _handle_str_replace

    reg = HookRegistry()
    captured = []
    reg.register_file_changed(lambda p: captured.append(p), name="watcher")

    class FakeAgent:
        hooks_registry = reg
        session_id = "edit-session"
        _checkpoint_track = None

    # 同上：注册 extra root 放行 tmp_path（finally clear 防泄漏）
    add_extra_allowed_root(str(tmp_path))
    try:
        target = tmp_path / "edit.txt"
        target.write_text("hello world", encoding="utf-8")
        result = _handle_str_replace(
            {"path": str(target), "old_str": "hello", "new_str": "hi"},
            agent_ref=FakeAgent(),
        )
    finally:
        clear_extra_allowed_roots()
    data = json.loads(result)
    assert "replaced" in data
    assert len(captured) == 1
    assert captured[0]["op"] == "edit"


def test_file_changed_no_hook_no_crash(tmp_path):
    """agent_ref 没有 hooks_registry 时不崩（fail-open 静默跳过）。"""
    from agent.permission import add_extra_allowed_root, clear_extra_allowed_roots
    from tools.file_operations import _handle_write_file

    add_extra_allowed_root(str(tmp_path))
    try:
        target = tmp_path / "silent.txt"
        # 没 agent_ref 也 OK
        result = _handle_write_file({"path": str(target), "content": "ok"})
    finally:
        clear_extra_allowed_roots()
    # 不应抛
    assert "bytes" in result


# ---------------------------------------------------------------------------
# 8. Task N Important fix: critical_reminder 在 fork 路径不丢失
# ---------------------------------------------------------------------------

def test_run_child_fork_preserves_critical_reminder(monkeypatch):
    """fork=True + custom_def.critical_reminder 同时设置时，子代理 system_prompt 仍含 critical_reminder。

    场景：custom agent 配了 criticalReminder + fork=True，fork 覆盖 system_prompt
    不应丢掉安全提醒。
    """
    from agent.agent_defs import AgentDefinition
    from unittest.mock import patch, MagicMock

    custom_def = AgentDefinition(
        name="safety-agent",
        description="safety-critical",
        critical_reminder="绝对不能删除任何文件",
    )
    monkeypatch.setattr("agent.agent_defs.get_agent_def", lambda n: custom_def)

    captured = {}

    class FakeChild:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_client = type("FakeClient", (), {})()
            self.model = kwargs.get("model")
            self.conversation_history = []

        async def chat(self, msg):
            return "ok"

    # 构造假父 agent（fork 需要父 system prompt + 对话历史）
    fake_parent = MagicMock()
    fake_parent.conversation_history = [
        {"role": "user", "content": "父问题"},
        {"role": "assistant", "content": "父回答"},
    ]
    fake_parent._get_system_prompt = MagicMock(return_value="父 system prompt")
    fake_parent.spawn_depth = 0
    fake_parent.effort_level = None
    fake_parent._children = []
    fake_parent.hooks_registry = None
    fake_parent.aux_llm_router = None
    fake_parent._stream_callback = None

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    from tools.delegate_tool import _run_child
    with patch("config.load_config", return_value={
        "model": {"name": "test", "api_key": "fake_key", "base_url": "http://x"},
    }):
        with patch("agent.AIAgent", FakeChild):
            with patch("agent.progress.ProgressReporter") as fake_prog:
                fake_prog.return_value.__enter__ = lambda s: None
                fake_prog.return_value.__exit__ = lambda s, *a: None
                with patch(
                    "agent.team.hallucination_check.verify_claims",
                    return_value=None,
                ):
                    with patch(
                        "agent.team.hallucination_check.append_warning",
                        lambda r, v: r,
                    ):
                        _run_child(
                            "goal", "ctx", "leaf",
                            fork=True,
                            agent_ref=fake_parent,
                            subagent_type="safety-agent",
                        )

    # fork 生效：system_prompt 前缀含父 system prompt
    sys_prompt = captured.get("system_prompt_override", "")
    assert "父 system prompt" in sys_prompt
    assert "FORK MODE" in sys_prompt
    # 关键断言：critical_reminder 没丢
    assert "CRITICAL REMINDER" in sys_prompt
    assert "绝对不能删除任何文件" in sys_prompt


def test_run_child_fork_no_critical_reminder_when_empty(monkeypatch):
    """fork=True 但 custom_def.critical_reminder 空（默认）→ 不拼 CRITICAL REMINDER 段。"""
    from agent.agent_defs import AgentDefinition
    from unittest.mock import patch, MagicMock

    custom_def = AgentDefinition(
        name="plain-agent",
        description="plain",
    )
    monkeypatch.setattr("agent.agent_defs.get_agent_def", lambda n: custom_def)

    captured = {}

    class FakeChild:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_client = type("FakeClient", (), {})()
            self.model = kwargs.get("model")
            self.conversation_history = []

        async def chat(self, msg):
            return "ok"

    fake_parent = MagicMock()
    fake_parent.conversation_history = [
        {"role": "user", "content": "父问题"},
        {"role": "assistant", "content": "父回答"},
    ]
    fake_parent._get_system_prompt = MagicMock(return_value="父 system prompt")
    fake_parent.spawn_depth = 0
    fake_parent.effort_level = None
    fake_parent._children = []
    fake_parent.hooks_registry = None
    fake_parent.aux_llm_router = None
    fake_parent._stream_callback = None

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    from tools.delegate_tool import _run_child
    with patch("config.load_config", return_value={
        "model": {"name": "test", "api_key": "fake_key", "base_url": "http://x"},
    }):
        with patch("agent.AIAgent", FakeChild):
            with patch("agent.progress.ProgressReporter") as fake_prog:
                fake_prog.return_value.__enter__ = lambda s: None
                fake_prog.return_value.__exit__ = lambda s, *a: None
                with patch(
                    "agent.team.hallucination_check.verify_claims",
                    return_value=None,
                ):
                    with patch(
                        "agent.team.hallucination_check.append_warning",
                        lambda r, v: r,
                    ):
                        _run_child(
                            "goal", "ctx", "leaf",
                            fork=True,
                            agent_ref=fake_parent,
                            subagent_type="plain-agent",
                        )

    sys_prompt = captured.get("system_prompt_override", "")
    assert "FORK MODE" in sys_prompt
    # 默认不拼 CRITICAL REMINDER
    assert "CRITICAL REMINDER" not in sys_prompt
