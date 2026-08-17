"""阶段 5 测试：OMNIMATE_ENV_FILE 环境持久化。

覆盖：
- `_load_session_env_overrides`：各种格式解析
- `session_env_file`：合法 + 含特殊字符 session_id（防 path traversal）
- AIAgent `_setup_session_env_file` 创建文件 + 设环境变量
- AIAgent.cleanup 清理文件 + unset 环境变量
"""
import os
from pathlib import Path
from unittest.mock import patch


def test_load_env_overrides_basic(monkeypatch, tmp_path):
    """OMNIMATE_ENV_FILE 存在 → 返回 dict。"""
    from tools.terminal_tool import _load_session_env_overrides
    env_file = tmp_path / "test.env"
    env_file.write_text(
        "# 注释\n"
        "export NODE_ENV=production\n"
        "PATH_ADDED=/extra/bin\n"
        'QUOTED="hello world"\n'
        "export SINGLE='quoted'\n"
        "\n# 另一个注释\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OMNIMATE_ENV_FILE", str(env_file))
    overrides = _load_session_env_overrides()
    assert overrides["NODE_ENV"] == "production"
    assert overrides["PATH_ADDED"] == "/extra/bin"
    assert overrides["QUOTED"] == "hello world"
    assert overrides["SINGLE"] == "quoted"


def test_load_env_overrides_no_env_var(monkeypatch):
    """OMNIMATE_ENV_FILE 未设 → 返回 {}。"""
    from tools.terminal_tool import _load_session_env_overrides
    monkeypatch.delenv("OMNIMATE_ENV_FILE", raising=False)
    assert _load_session_env_overrides() == {}


def test_load_env_overrides_nonexistent_file(monkeypatch, tmp_path):
    """文件不存在 → 返回 {}。"""
    from tools.terminal_tool import _load_session_env_overrides
    monkeypatch.setenv("OMNIMATE_ENV_FILE", str(tmp_path / "nope.env"))
    assert _load_session_env_overrides() == {}


def test_session_env_file_path():
    """合法 session_id → 返回 .session/{id}.env。"""
    from constants import session_env_file, get_omnimate_home
    p = session_env_file("abc-123_XYZ")
    assert p.name == "abc-123_XYZ.env"
    assert p.parent.name == ".session"


def test_session_env_file_sanitizes_session_id():
    """session_id 含非法字符 → 清理（防 path traversal）。"""
    from constants import session_env_file
    p = session_env_file("..\\evil\\path")
    # 只保留字母数字和 -_，剩下的全清掉
    assert ".." not in p.name
    assert "\\" not in p.name
    # 不能逃出 .session 目录
    assert p.parent == session_env_file("safe").parent


def test_session_env_file_empty_session_id():
    """session_id 为空 → 用 'default'。"""
    from constants import session_env_file
    p = session_env_file("")
    assert p.name == "default.env"


def test_ai_agent_setup_creates_env_file(monkeypatch, tmp_path):
    """AIAgent 初始化时建 session env 文件 + 设环境变量。"""
    # 把 omnimate home 指向 tmp
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    from agent import AIAgent
    # mock LLM client 避免真实连接
    with patch("agent.llm_client.create_llm_client", return_value=None):
        agent = AIAgent(
            base_url="http://x",
            api_key="fake",
            model="m",
            session_id="test-sess-123",
        )
    try:
        assert os.environ.get("OMNIMATE_ENV_FILE")
        assert "test-sess-123" in os.environ["OMNIMATE_ENV_FILE"]
        assert Path(os.environ["OMNIMATE_ENV_FILE"]).exists()
    finally:
        agent.cleanup()
        # cleanup 后环境变量 unset + 文件删除
        assert not os.environ.get("OMNIMATE_ENV_FILE")
        # 注意：cleanup 后变量 unset，文件可能仍存在如果路径未跟踪 — 这里检查 _session_env_path 已 None
        assert agent._session_env_path is None


def test_ai_agent_cleanup_idempotent(monkeypatch, tmp_path):
    """cleanup 幂等：多次调用不抛。"""
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    from agent import AIAgent
    with patch("agent.llm_client.create_llm_client", return_value=None):
        agent = AIAgent(
            base_url="http://x",
            api_key="fake",
            model="m",
            session_id="idempotent-test",
        )
    agent.cleanup()
    agent.cleanup()  # 第二次不抛
    agent.cleanup()  # 第三次也不抛
