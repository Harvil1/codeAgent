"""预制 hook 集成测试。

通过 subprocess 实际跑每个 hook，验证 IPC 协议正确。
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).parent.parent / "hooks"


def run_hook(script_name: str, payload: dict, env: dict = None) -> dict:
    """跑一个 hook 脚本，返回解析后的 stdout JSON（空 stdout → {}）。"""
    proc = subprocess.run(
        [sys.executable, str(HOOKS_DIR / script_name)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**__import__("os").environ, **(env or {})},
    )
    assert proc.returncode == 0, f"{script_name} exit {proc.returncode}: {proc.stderr}"
    out = proc.stdout.strip()
    return json.loads(out) if out else {}


# ---------------------------------------------------------------------------
# secrets_redactor
# ---------------------------------------------------------------------------

def test_secrets_redactor_strips_api_key():
    payload = {
        "event": "post_tool_use",
        "tool_name": "terminal",
        "result": "export API_KEY=sk-abc123def456ghi789jkl012mno345pqr678",
    }
    result = run_hook("secrets_redactor.py", payload)
    assert "sk-abc123" not in result["result"]
    assert "[REDACTED:" in result["result"]


def test_secrets_redactor_strips_bearer():
    payload = {
        "event": "post_tool_use",
        "result": "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturepart",
    }
    result = run_hook("secrets_redactor.py", payload)
    assert "Bearer" not in result["result"] or "[REDACTED:" in result["result"]


def test_secrets_redactor_strips_pem():
    payload = {
        "event": "post_tool_use",
        "result": "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----",
    }
    result = run_hook("secrets_redactor.py", payload)
    assert "[REDACTED:pem_key]" in result["result"]


def test_secrets_redactor_passthrough_no_secrets():
    """无密钥的输出不修改（空 stdout）。"""
    payload = {
        "event": "post_tool_use",
        "result": "ls -la\ntotal 0\nfile1.txt",
    }
    result = run_hook("secrets_redactor.py", payload)
    # 空 stdout = 不修改
    assert result == {}


def test_secrets_redactor_fail_open_on_bad_json():
    """坏 JSON 不应崩（exit 0 + 空 stdout）。"""
    proc = subprocess.run(
        [sys.executable, str(HOOKS_DIR / "secrets_redactor.py")],
        input="not-valid-json",
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


# ---------------------------------------------------------------------------
# audit_log
# ---------------------------------------------------------------------------

def test_audit_log_writes_user_prompt(tmp_path, monkeypatch):
    log_path = tmp_path / "audit.log"
    payload = {
        "event": "user_prompt_submit",
        "session_id": "abcdefgh-1234",
        "prompt": "hello world",
    }
    run_hook("audit_log.py", payload, env={"AUDIT_LOG_PATH": str(log_path)})

    content = log_path.read_text(encoding="utf-8")
    assert "USER: hello world" in content
    assert "abcdefgh" in content  # session id 前 8


def test_audit_log_writes_tool_result(tmp_path):
    log_path = tmp_path / "audit.log"
    payload = {
        "event": "post_tool_use",
        "session_id": "xyz12345",
        "tool_name": "terminal",
        "result": "command output here",
    }
    run_hook("audit_log.py", payload, env={"AUDIT_LOG_PATH": str(log_path)})

    content = log_path.read_text(encoding="utf-8")
    assert "TOOL terminal" in content
    assert "command output here" in content


def test_audit_log_truncates_long_prompt(tmp_path):
    log_path = tmp_path / "audit.log"
    long_prompt = "x" * 1000
    payload = {
        "event": "user_prompt_submit",
        "session_id": "abc",
        "prompt": long_prompt,
    }
    run_hook("audit_log.py", payload, env={"AUDIT_LOG_PATH": str(log_path)})

    content = log_path.read_text(encoding="utf-8")
    # 截断后应含 ... 且总长 < 1000
    assert "..." in content
    assert len(content) < 600


# ---------------------------------------------------------------------------
# python_syntax_check
# ---------------------------------------------------------------------------

def test_syntax_check_allows_valid_python():
    payload = {
        "event": "pre_tool_use",
        "tool_name": "write_file",
        "args": {"path": "foo.py", "content": "def f():\n    return 42\n"},
    }
    result = run_hook("python_syntax_check.py", payload)
    assert result == {}  # 空 = allow


def test_syntax_check_denies_invalid_python():
    payload = {
        "event": "pre_tool_use",
        "tool_name": "write_file",
        "args": {"path": "foo.py", "content": "def f(\n"},  # 语法错
    }
    result = run_hook("python_syntax_check.py", payload)
    assert result.get("action") == "deny"
    assert "语法错误" in result["reason"]


def test_syntax_check_ignores_non_python():
    payload = {
        "event": "pre_tool_use",
        "tool_name": "write_file",
        "args": {"path": "foo.txt", "content": "anything"},
    }
    result = run_hook("python_syntax_check.py", payload)
    assert result == {}


def test_syntax_check_ignores_non_write_file():
    payload = {
        "event": "pre_tool_use",
        "tool_name": "terminal",
        "args": {"command": "echo hi"},
    }
    result = run_hook("python_syntax_check.py", payload)
    assert result == {}


# ---------------------------------------------------------------------------
# block_large_writes
# ---------------------------------------------------------------------------

def test_block_large_writes_denies_oversize():
    payload = {
        "event": "pre_tool_use",
        "tool_name": "write_file",
        "args": {"path": "big.txt", "content": "x" * (10 * 1024 * 1024)},  # 10MB
    }
    result = run_hook("block_large_writes.py", payload)
    assert result.get("action") == "deny"
    assert "过大" in result["reason"]


def test_block_large_writes_allows_normal_size():
    payload = {
        "event": "pre_tool_use",
        "tool_name": "write_file",
        "args": {"path": "small.txt", "content": "hello"},
    }
    result = run_hook("block_large_writes.py", payload)
    assert result == {}


def test_block_large_writes_respects_env_threshold():
    """MAX_WRITE_BYTES 环境变量可调阈值。"""
    payload = {
        "event": "pre_tool_use",
        "tool_name": "write_file",
        "args": {"path": "med.txt", "content": "x" * 1000},  # 1KB
    }
    # 设阈值为 100 字节 → 1KB 应被拒
    result = run_hook("block_large_writes.py", payload, env={"MAX_WRITE_BYTES": "100"})
    assert result.get("action") == "deny"


def test_block_large_writes_ignores_non_write_file():
    payload = {
        "event": "pre_tool_use",
        "tool_name": "terminal",
        "args": {"command": "dd if=/dev/zero of=big bs=1M count=100"},
    }
    result = run_hook("block_large_writes.py", payload)
    assert result == {}


# ---------------------------------------------------------------------------
# 完整 IPC 协议（一个 hook 的所有路径）
# ---------------------------------------------------------------------------

def test_settings_example_is_valid_json():
    """example-settings.json 应是合法 JSON。"""
    example_path = HOOKS_DIR / "example-settings.json"
    data = json.loads(example_path.read_text(encoding="utf-8"))
    assert "hooks" in data
    # 至少注册了一个 post_tool_use hook
    assert "post_tool_use" in data["hooks"]
    assert len(data["hooks"]["post_tool_use"]) >= 1
