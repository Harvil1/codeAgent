"""hook_exec 模块测试：子进程执行 + JSON IPC。"""
import json
import sys
import time
from pathlib import Path

from agent.hooks import Hook, HookScriptConfig, HookEvent
from agent.hook_exec import run_script_hook


def _make_hook(command, timeout=10.0, name="test"):
    return Hook(
        name=name, event=HookEvent.POST_TOOL_USE, kind="declarative",
        script=HookScriptConfig(command=command, timeout=timeout),
    )


def test_run_script_returns_parsed_dict(tmp_path):
    """正常路径：子进程 stdout 合法 JSON。"""
    payload = {"event": "post_tool_use", "result": "hello"}
    script = "import sys; print(__import__('json').dumps({'result': 'WORLD'}))"
    hook = _make_hook([sys.executable, "-c", script])
    result = run_script_hook(hook, payload)
    assert result == {"result": "WORLD"}


def test_run_script_empty_stdout_returns_empty_dict(tmp_path):
    """子进程不输出 = 视为 allow（空 dict）。"""
    hook = _make_hook([sys.executable, "-c", ""])
    result = run_script_hook(hook, {"event": "stop"})
    assert result == {}


def test_run_script_timeout_returns_none():
    """超时 kill，返回 None。"""
    script = "import time; time.sleep(10)"
    hook = _make_hook([sys.executable, "-c", script], timeout=0.5)
    result = run_script_hook(hook, {"event": "stop"})
    assert result is None


def test_run_script_nonzero_exit_returns_none():
    """exit code != 0 → None。"""
    hook = _make_hook([sys.executable, "-c", "import sys; sys.exit(1)"])
    result = run_script_hook(hook, {"event": "stop"})
    assert result is None


def test_run_script_invalid_json_returns_none():
    """stdout 非合法 JSON → None。"""
    hook = _make_hook([sys.executable, "-c", "print('not json')"])
    result = run_script_hook(hook, {"event": "stop"})
    assert result is None


def test_run_script_command_not_found_returns_none():
    """可执行文件不存在 → None（不抛）。"""
    hook = _make_hook(["./nonexistent-script-xyz"])
    result = run_script_hook(hook, {"event": "stop"})
    assert result is None


def test_run_script_receives_payload_on_stdin(tmp_path):
    """payload 通过 stdin 传入，子进程能读到。"""
    script = """
import sys, json
data = json.load(sys.stdin)
print(json.dumps({"echo_prompt": data.get("prompt", "") + "_seen"}))
"""
    hook = _make_hook([sys.executable, "-c", script])
    result = run_script_hook(hook, {"prompt": "hello", "event": "user_prompt_submit"})
    assert result == {"echo_prompt": "hello_seen"}


def test_run_script_env_vars_passed():
    """hook config 的 env 被加到子进程环境。"""
    script = (
        "import os, sys; "
        "sys.stdout.write(__import__('json').dumps({'v': os.environ.get('MY_VAR', '')}))"
    )
    hook = _make_hook([sys.executable, "-c", script])
    hook.script.env = {"MY_VAR": "xyz"}
    result = run_script_hook(hook, {"event": "stop"})
    assert result == {"v": "xyz"}


# ===== http hook ${VAR} 插值白名单 =====

class TestInterpolateEnvVars:
    def test_whitelisted_var_interpolated(self, monkeypatch):
        from agent.hook_exec import interpolate_env_vars
        # 注：简报原文 env 值带 "http://" 前缀与模板重复（插值后 scheme 拼两次
        # 无法满足断言），此处最小修正为 host:port——模板带 scheme、env 提供地址
        monkeypatch.setenv("OMNI_TEST_HOOK_URL", "127.0.0.1:9")
        out = interpolate_env_vars(
            "http://${OMNI_TEST_HOOK_URL}/hook", ["OMNI_TEST_HOOK_URL"],
        )
        assert out == "http://127.0.0.1:9/hook"

    def test_non_whitelisted_left_literal_with_warning(self, monkeypatch, caplog):
        from agent.hook_exec import interpolate_env_vars
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret")
        out = interpolate_env_vars(
            "http://h/${DEEPSEEK_API_KEY}", [],
        )
        assert out == "http://h/${DEEPSEEK_API_KEY}"  # 不插值

    def test_no_pattern_unchanged(self):
        from agent.hook_exec import interpolate_env_vars
        assert interpolate_env_vars("http://plain/path", []) == "http://plain/path"

    def test_unset_whitelisted_left_literal(self, monkeypatch):
        from agent.hook_exec import interpolate_env_vars
        monkeypatch.delenv("OMNI_NOT_SET_VAR", raising=False)
        out = interpolate_env_vars("${OMNI_NOT_SET_VAR}", ["OMNI_NOT_SET_VAR"])
        assert out == "${OMNI_NOT_SET_VAR}"
