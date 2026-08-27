# -*- coding: utf-8 -*-
"""细分行为修复回归测试。

覆盖：
  hook env 最小化（build_safe_env）
  http hook 响应体上限 / prompt hook 安全 format
  L4 总量上限移除（见 test_context_pipeline.py 用例）
  _drop_leading_system 防御
  trace 中英混合 token 估算
  find_by_topic_name 目标区查重
  排队 slash 命令分流（agent drain → _queued_cli_commands）
  history 文件锁（行为不变，锁文件出现）
  粘贴内容寻址（见 test_r21_small.py 用例）
  /poor off 回滚快照
"""
import queue as queue_mod
from types import SimpleNamespace


# ======================================================================
# http hook 响应体上限
# ======================================================================

def test_http_hook_body_cap_h1(monkeypatch):
    import agent.hook_exec as he

    hook = SimpleNamespace(
        name="h", fail_closed=False,
        script=SimpleNamespace(
            handler_type="http", url="http://x.local/hook", timeout=2.0,
            env=None,
        ),
    )
    big = SimpleNamespace(status_code=200, content=b"x" * (he._MAX_HTTP_HOOK_BODY_BYTES + 1), json=lambda: {})
    monkeypatch.setattr(he.requests, "post", lambda *a, **kw: big)
    assert he.run_http_hook(hook, {"event": "e"}) is None, "超大响应体应被丢弃"


# ======================================================================
# prompt hook 安全 format
# ======================================================================

def test_prompt_hook_safe_format_h2(monkeypatch):
    import agent.hook_exec as he

    captured = {}

    class _Router:
        def chat_completions(self, messages=None, **kw):
            captured["prompt"] = messages[0]["content"]
            return '{"permissionDecision": "allow"}'

    monkeypatch.setattr(he, "_get_aux_router", lambda: _Router())
    hook = SimpleNamespace(
        name="p", fail_closed=False,
        script=SimpleNamespace(handler_type="prompt", prompt="评估 {tool_name}（{missing_field} 缺失）", timeout=2.0),
    )
    out = he.run_prompt_hook(hook, {"event": "pre_tool_use", "tool_name": "terminal"})
    # 缺字段不抛 KeyError，占位符原样保留
    assert "{missing_field}" in captured["prompt"]
    assert "terminal" in captured["prompt"]
    assert out is not None


# ======================================================================
# hook 子进程 env 最小化
# ======================================================================

def test_script_hook_env_sanitized_b6a(monkeypatch):
    import agent.hook_exec as he

    monkeypatch.setenv("FAKE_SECRET_API_KEY", "sk-super-secret")
    captured = {}

    def _fake_run(argv, **kw):
        captured["env"] = kw.get("env") or {}
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(he.subprocess, "run", _fake_run)
    hook = SimpleNamespace(
        name="s", fail_closed=False, use_sandbox=False,
        script=SimpleNamespace(handler_type="command", command=["whatever"], timeout=2.0, env=None),
    )
    he.run_script_hook(hook, {"event": "e"})
    assert "FAKE_SECRET_API_KEY" not in captured["env"], (
        "hook 子进程不应继承宿主的 API key 等敏感环境变量"
    )
    assert "PATH" in captured["env"], "常规变量（PATH 等）应保留"


# ======================================================================
# _drop_leading_system
# ======================================================================

def test_drop_leading_system_d3():
    from agent import _drop_leading_system
    sys_msg = {"role": "system", "content": "S"}
    u1 = {"role": "user", "content": "hi"}
    u2 = {"role": "user", "content": "yo"}
    assert _drop_leading_system([sys_msg, u1, u2]) == [u1, u2]
    # system 缺失：不丢第一条真实消息（旧裸切 [1:] 会静默丢 u1）
    assert _drop_leading_system([u1, u2]) == [u1, u2]
    assert _drop_leading_system([]) == []


# ======================================================================
# trace 中英混合估算
# ======================================================================

def test_trace_cjk_estimate_d6():
    from agent.trace import _estimate_messages_tokens
    ascii_tokens = _estimate_messages_tokens([{"role": "user", "content": "a" * 4000}])
    assert ascii_tokens == 1000
    # 中文 1.5 字符/token（此前按 4 字符/token 低估 ~2.7 倍）
    cjk_tokens = _estimate_messages_tokens([{"role": "user", "content": "中" * 1500}])
    assert cjk_tokens == 1000


# ======================================================================
# find_by_topic_name 目标区查重
# ======================================================================

def test_find_by_topic_name_type_scoped_d9(tmp_path):
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    store.save(name="dup", description="d", type="user", body="b")
    # type=user（同目标区）→ 命中
    assert store.find_by_topic_name("general", "dup", type="user") is not None
    # type=other（同全局区）→ 命中（无项目区时两 type 都路由全局）
    assert store.find_by_topic_name("general", "dup", type="other") is not None
    # 不存在的 name → None
    assert store.find_by_topic_name("general", "nope", type="user") is None
    # 旧签名（无 type）跨区查仍可用
    assert store.find_by_topic_name("general", "dup") is not None


# ======================================================================
# 排队 slash 命令分流
# ======================================================================

def test_drain_splits_cli_commands_c8(tmp_path):
    from agent import AIAgent

    agent = AIAgent(api_key="fake", model="t", omnimate_home=tmp_path,
                    enabled_toolsets=[])
    q = queue_mod.Queue()
    agent.set_input_queue(q)
    q.put("/compact")
    q.put("运行中补充的消息")
    q.put("/etc/passwd 是什么")  # 路径形态：不是命令，仍喂模型
    agent._drain_queued_input()
    # slash 命令分流到 _queued_cli_commands，不进 ephemeral
    assert agent._queued_cli_commands == ["/compact"]
    assert len(agent._pending_ephemeral_messages) == 1
    content = agent._pending_ephemeral_messages[0]["content"]
    assert "运行中补充的消息" in content
    assert "/etc/passwd 是什么" in content
    assert "/compact" not in content


# ======================================================================
# history 锁文件 + 行为不回归
# ======================================================================

def test_history_lock_and_behavior_c9(tmp_path):
    from agent.input_history import GlobalHistory
    h = GlobalHistory(tmp_path)
    h.append("a")
    h.append("b")
    assert h.recent(2) == ["b", "a"]
    assert (tmp_path / ".history.lock").exists(), "append 应持有/创建锁文件"


# ======================================================================
# /poor on 快照 + off 回滚
# ======================================================================

def test_poor_on_off_roundtrip_c11(tmp_path):
    from cli import _handle_poor_command

    rt = SimpleNamespace(config={"llm": {"reflection_enabled": True}})
    _handle_poor_command("on", rt)
    assert rt._poor_mode_on is True
    assert rt._poor_config_snapshot == {"llm": {"reflection_enabled": True}}
    # on 之后 config 被改（某 flag 关掉）
    rt.config["llm"]["reflection_enabled"] = False
    _handle_poor_command("off", rt)
    assert rt._poor_mode_on is False
    assert rt.config["llm"]["reflection_enabled"] is True, "off 应回滚开启前快照"
    assert rt._poor_config_snapshot is None
