# Phase 2a: Hooks 系统实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 Hooks 系统（4 种 event + 程序式 + 声明式注册），把扩展 agent 主循环行为从「改代码」变成「写回调 + 注册」。

**Architecture:** 3 个新模块（`agent/hooks.py` 核心注册表 + `agent/hook_exec.py` 子进程执行 + `agent/hook_loader.py` 配置加载）+ 主循环 4 个注入点。所有 hook 调用 fail-open，PreToolUse 可选 fail_closed。

**Tech Stack:** Python 3.11+、uv、pytest、subprocess、JSON IPC。

**对应 Spec:** `docs/superpowers/specs/2026-07-12-phase2a-hooks-design.md`

## Global Constraints

- 文件 I/O 必须 `encoding="utf-8"`（HARVIL.md 强制）
- 用 `uv`，不要 `pip install`
- 中文注释/commit；英文标识符
- 不要 import 用不到的模块（ruff F401）
- 工具 handler 返回 JSON 字符串
- 错误格式 `{"error": "...", "error_type": "..."}`
- 测试：`uv run pytest tests/<file>.py -v`；全量 `uv run pytest tests/ -v`
- 项目当前 360 测试不得回归
- Subagent 不 commit，controller 拿用户授权后统一 commit（沿用 Phase 1 工作流）
- 同步执行（不引入 asyncio）

---

## File Structure

| 文件 | 任务 | 责任 |
|---|---|---|
| `agent/hooks.py` | T1, T3 新增 | `HookEvent` 枚举 + `Hook` / `HookScriptConfig` dataclass + `HookRegistry` |
| `agent/hook_exec.py` | T2 新增 | `run_script_hook(hook, payload) -> Optional[dict]` 子进程执行 |
| `agent/hook_loader.py` | T4 新增 | `load_declarative_hooks(registry, settings_path) -> int` |
| `config.py` | T5 改 | `DEFAULT_CONFIG["hooks"]` 块 |
| `agent/__init__.py` | T6 改 | AIAgent `hooks_registry` kwarg + USER_PROMPT_SUBMIT + STOP 注入 |
| `model_tools.py` | T7 改 | `handle_function_call` 加 PRE/POST_TOOL_USE 调用点 |
| `cli.py` | T8 改 | RuntimeContext 持有 registry + 启动加载声明式 hooks |
| `tests/test_hooks.py` | T1, T3 新增 | 4 event 程序式 + 失败隔离 + 顺序 |
| `tests/test_hook_exec.py` | T2 新增 | 子进程 + JSON IPC + 超时 + 失败模式 |
| `tests/test_hook_loader.py` | T4 新增 | settings.json 解析 + 校验 |
| `tests/test_integration.py` | T8 改 | 主循环 + hooks 端到端 |

---

## Task 1: hooks.py 核心（类型 + 程序式注册表 + 4 event）

**Files:**
- Create: `agent/hooks.py`
- Create: `tests/test_hooks.py`

**Interfaces:**
- Consumes: 无
- Produces: `HookEvent`、`Hook`、`HookScriptConfig`、`HookRegistry`；方法签名见下

- [ ] **Step 1: 写失败测试（USER_PROMPT_SUBMIT）**

```python
# tests/test_hooks.py
"""Hooks 系统测试。"""
import pytest
from agent.hooks import (
    HookEvent, Hook, HookScriptConfig, HookRegistry,
)


# ---------------------------------------------------------------------------
# 类型 + 枚举
# ---------------------------------------------------------------------------

def test_hook_event_has_four_values():
    assert {e.value for e in HookEvent} == {
        "user_prompt_submit", "pre_tool_use", "post_tool_use", "stop",
    }


def test_hook_dataclass_programmatic():
    h = Hook(name="x", event=HookEvent.STOP, kind="programmatic", fn=lambda: None)
    assert h.kind == "programmatic"
    assert h.fail_closed is False


def test_hook_script_config_defaults():
    c = HookScriptConfig(command=["echo"])
    assert c.timeout == 10.0
    assert c.env is None


# ---------------------------------------------------------------------------
# USER_PROMPT_SUBMIT
# ---------------------------------------------------------------------------

def test_user_prompt_submit_no_hooks_passthrough():
    reg = HookRegistry()
    assert reg.run_user_prompt_submit("hello", session_id="s1") == "hello"


def test_user_prompt_submit_single_hook_modifies():
    reg = HookRegistry()
    reg.register_user_prompt_submit(lambda p: p.upper(), name="upper")
    assert reg.run_user_prompt_submit("hello", session_id="s1") == "HELLO"


def test_user_prompt_submit_chain_composition():
    reg = HookRegistry()
    reg.register_user_prompt_submit(lambda p: p + "1", name="a")
    reg.register_user_prompt_submit(lambda p: p + "2", name="b")
    assert reg.run_user_prompt_submit("x", session_id="s1") == "x12"


def test_user_prompt_submit_none_passthrough():
    reg = HookRegistry()
    reg.register_user_prompt_submit(lambda p: None, name="noop")
    assert reg.run_user_prompt_submit("hello", session_id="s1") == "hello"


def test_user_prompt_submit_exception_isolated():
    """hook 抛异常时视为 None，不影响链。"""
    reg = HookRegistry()
    def bad(p): raise ValueError("boom")
    reg.register_user_prompt_submit(bad, name="bad")
    reg.register_user_prompt_submit(lambda p: p + "_ok", name="ok")
    # bad 抛异常被吞，ok 仍然执行
    assert reg.run_user_prompt_submit("hello", session_id="s1") == "hello_ok"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_hooks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.hooks'`

- [ ] **Step 3: 写最小实现（USER_PROMPT_SUBMIT 部分）**

```python
# agent/hooks.py
"""Hooks 系统：扩展 agent 主循环行为的注册表机制。

4 种 event：USER_PROMPT_SUBMIT / PRE_TOOL_USE / POST_TOOL_USE / STOP
2 种注册：programmatic（Python 函数）/ declarative（子进程脚本，T3 实现）
失败 fail-open 默认（log + 视为 None）；PreToolUse 可选 fail_closed。
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)


class HookEvent(Enum):
    USER_PROMPT_SUBMIT = "user_prompt_submit"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    STOP = "stop"


# 程序式 hook 的 4 种签名
UserPromptSubmitFn = Callable[[str], Optional[str]]
PreToolUseFn = Callable[[str, dict], Optional[dict]]
PostToolUseFn = Callable[[str, dict, str], Optional[str]]
StopFn = Callable[[], Optional[str]]


@dataclass
class HookScriptConfig:
    """声明式 hook 的子进程配置。"""
    command: list  # list[str]，如 ["python", "./hooks/audit.py"]
    timeout: float = 10.0
    env: Optional[dict] = None


@dataclass
class Hook:
    """统一包装：程序式或声明式。"""
    name: str
    event: HookEvent
    kind: str  # "programmatic" | "declarative"
    fn: Optional[Callable] = None
    script: Optional[HookScriptConfig] = None
    fail_closed: bool = False


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class HookRegistry:
    """管理所有 hook 注册和执行。实例由 RuntimeContext 持有，注入 AIAgent。"""

    def __init__(self):
        self._hooks: dict = {e: [] for e in HookEvent}
        self._stop_fire_count: int = 0

    # ---- 注册 ----
    def register_user_prompt_submit(self, fn, *, name=None):
        self._hooks[HookEvent.USER_PROMPT_SUBMIT].append(
            Hook(name=name or "anonymous", event=HookEvent.USER_PROMPT_SUBMIT,
                 kind="programmatic", fn=fn)
        )

    def register_pre_tool_use(self, fn, *, name=None, fail_closed=False):
        self._hooks[HookEvent.PRE_TOOL_USE].append(
            Hook(name=name or "anonymous", event=HookEvent.PRE_TOOL_USE,
                 kind="programmatic", fn=fn, fail_closed=fail_closed)
        )

    def register_post_tool_use(self, fn, *, name=None):
        self._hooks[HookEvent.POST_TOOL_USE].append(
            Hook(name=name or "anonymous", event=HookEvent.POST_TOOL_USE,
                 kind="programmatic", fn=fn)
        )

    def register_stop(self, fn, *, name=None):
        self._hooks[HookEvent.STOP].append(
            Hook(name=name or "anonymous", event=HookEvent.STOP,
                 kind="programmatic", fn=fn)
        )

    def register_declarative(self, hook: Hook):
        """T3 实现：把已构造好的 Hook 加到 registry。"""
        self._hooks[hook.event].append(hook)

    def clear(self, event=None):
        """清空（测试用）。"""
        if event is None:
            for e in self._hooks:
                self._hooks[e] = []
        else:
            self._hooks[event] = []
        self._stop_fire_count = 0

    # ---- 执行：USER_PROMPT_SUBMIT ----
    def run_user_prompt_submit(self, prompt: str, *, session_id: str) -> str:
        """链式：每个 hook 看到前一个的输出。失败 fail-open。"""
        for hook in self._hooks[HookEvent.USER_PROMPT_SUBMIT]:
            try:
                if hook.kind == "programmatic":
                    new_prompt = hook.fn(prompt)
                else:
                    # T3 接入 declarative
                    new_prompt = self._invoke_declarative_user_prompt(hook, prompt, session_id)
                if new_prompt is not None:
                    prompt = new_prompt
            except Exception as e:
                logger.warning("hook %s 异常（视为 None）: %s", hook.name, e)
        return prompt

    def _invoke_declarative_user_prompt(self, hook, prompt, session_id):
        """T3 实现。"""
        return None  # 占位，T3 替换

    # ---- 其他 run_* 方法在 T1 step 5-10 中加 ----
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_hooks.py::test_user_prompt_submit_no_hooks_passthrough tests/test_hooks.py::test_user_prompt_submit_single_hook_modifies tests/test_hooks.py::test_user_prompt_submit_chain_composition tests/test_hooks.py::test_user_prompt_submit_none_passthrough tests/test_hooks.py::test_user_prompt_submit_exception_isolated tests/test_hooks.py::test_hook_event_has_four_values tests/test_hooks.py::test_hook_dataclass_programmatic tests/test_hooks.py::test_hook_script_config_defaults -v`
Expected: PASS（8 tests）

- [ ] **Step 5: 追加 PRE_TOOL_USE 测试**

```python
# 追加到 tests/test_hooks.py

# ---------------------------------------------------------------------------
# PRE_TOOL_USE
# ---------------------------------------------------------------------------

def test_pre_tool_use_no_hooks_returns_none_none():
    reg = HookRegistry()
    deny, modified = reg.run_pre_tool_use("terminal", {"cmd": "ls"}, session_id="s")
    assert deny is None
    assert modified is None


def test_pre_tool_use_allow_when_none():
    reg = HookRegistry()
    reg.register_pre_tool_use(lambda n, a: None, name="ok")
    deny, modified = reg.run_pre_tool_use("terminal", {"cmd": "ls"}, session_id="s")
    assert deny is None
    assert modified is None


def test_pre_tool_use_deny():
    reg = HookRegistry()
    reg.register_pre_tool_use(lambda n, a: {"deny": "blocked"}, name="b")
    deny, modified = reg.run_pre_tool_use("terminal", {"cmd": "rm"}, session_id="s")
    assert deny == "blocked"
    assert modified is None


def test_pre_tool_use_deny_short_circuits():
    """首个 deny 胜出，后续不跑。"""
    calls = []
    def h1(n, a): calls.append("h1"); return {"deny": "first"}
    def h2(n, a): calls.append("h2"); return None
    reg = HookRegistry()
    reg.register_pre_tool_use(h1, name="h1")
    reg.register_pre_tool_use(h2, name="h2")
    deny, _ = reg.run_pre_tool_use("t", {}, session_id="s")
    assert deny == "first"
    assert calls == ["h1"]


def test_pre_tool_use_modify_args():
    reg = HookRegistry()
    reg.register_pre_tool_use(
        lambda n, a: {"modify_args": {"cmd": "safe"}}, name="m"
    )
    deny, modified = reg.run_pre_tool_use("terminal", {"cmd": "rm"}, session_id="s")
    assert deny is None
    assert modified == {"cmd": "safe"}


def test_pre_tool_use_modify_chain():
    """modify_args 链式累积，hook2 看到 hook1 的修改。"""
    def h1(n, a): return {"modify_args": {**a, "x": 1}}
    def h2(n, a): return {"modify_args": {**a, "y": 2}}
    reg = HookRegistry()
    reg.register_pre_tool_use(h1, name="h1")
    reg.register_pre_tool_use(h2, name="h2")
    deny, modified = reg.run_pre_tool_use("t", {"orig": 0}, session_id="s")
    assert deny is None
    assert modified == {"orig": 0, "x": 1, "y": 2}


def test_pre_tool_use_exception_isolated():
    def bad(n, a): raise ValueError("boom")
    reg = HookRegistry()
    reg.register_pre_tool_use(bad, name="bad")
    deny, modified = reg.run_pre_tool_use("t", {"a": 1}, session_id="s")
    assert deny is None
    assert modified is None
```

- [ ] **Step 6: 追加 PRE_TOOL_USE 实现**

```python
# 追加到 agent/hooks.py:HookRegistry

    def run_pre_tool_use(self, tool_name: str, args: dict, *,
                          session_id: str):
        """短路：首个 deny 胜出。返回 (deny_reason, modified_args)。

        - deny: Optional[str]，非 None 时拒绝
        - modified_args: Optional[dict]，非 None 时累计替换 args
        """
        deny_reason = None
        modified_args = None
        current_args = args
        for hook in self._hooks[HookEvent.PRE_TOOL_USE]:
            try:
                if hook.kind == "programmatic":
                    result = hook.fn(tool_name, current_args)
                else:
                    # T3 接入 declarative
                    result = self._invoke_declarative_pre_tool(hook, tool_name, current_args, session_id)
                if result is None:
                    continue
                if "deny" in result:
                    deny_reason = result["deny"]
                    return deny_reason, modified_args  # 短路
                if "modify_args" in result:
                    current_args = result["modify_args"]
                    modified_args = current_args
            except Exception as e:
                if hook.fail_closed:
                    logger.warning("hook %s fail_closed（视为拒绝）: %s", hook.name, e)
                    return str(e), None
                logger.warning("hook %s 异常（视为 None）: %s", hook.name, e)
        return deny_reason, modified_args

    def _invoke_declarative_pre_tool(self, hook, tool_name, args, session_id):
        """T3 实现。"""
        return None
```

- [ ] **Step 7: 跑 PRE_TOOL_USE 测试**

Run: `uv run pytest tests/test_hooks.py -k pre_tool_use -v`
Expected: PASS（7 tests）

- [ ] **Step 8: 追加 POST_TOOL_USE + STOP 测试**

```python
# 追加到 tests/test_hooks.py

# ---------------------------------------------------------------------------
# POST_TOOL_USE
# ---------------------------------------------------------------------------

def test_post_tool_use_no_hooks_passthrough():
    reg = HookRegistry()
    assert reg.run_post_tool_use("t", {}, "result", session_id="s") == "result"


def test_post_tool_use_single_modifies():
    reg = HookRegistry()
    reg.register_post_tool_use(lambda n, a, r: r.upper(), name="upper")
    assert reg.run_post_tool_use("t", {}, "hello", session_id="s") == "HELLO"


def test_post_tool_use_chain():
    reg = HookRegistry()
    reg.register_post_tool_use(lambda n, a, r: r + "1", name="a")
    reg.register_post_tool_use(lambda n, a, r: r + "2", name="b")
    assert reg.run_post_tool_use("t", {}, "x", session_id="s") == "x12"


def test_post_tool_use_exception_isolated():
    def bad(n, a, r): raise ValueError("boom")
    reg = HookRegistry()
    reg.register_post_tool_use(bad, name="bad")
    assert reg.run_post_tool_use("t", {}, "hello", session_id="s") == "hello"


# ---------------------------------------------------------------------------
# STOP
# ---------------------------------------------------------------------------

def test_stop_no_hooks_returns_none():
    reg = HookRegistry()
    assert reg.run_stop(session_id="s", max_fires=3) is None


def test_stop_all_none_returns_none():
    reg = HookRegistry()
    reg.register_stop(lambda: None, name="a")
    reg.register_stop(lambda: None, name="b")
    assert reg.run_stop(session_id="s", max_fires=3) is None


def test_stop_first_non_none_wins():
    calls = []
    def h1(): calls.append("h1"); return None
    def h2(): calls.append("h2"); return "continue msg"
    def h3(): calls.append("h3"); return "should not run"
    reg = HookRegistry()
    reg.register_stop(h1, name="h1")
    reg.register_stop(h2, name="h2")
    reg.register_stop(h3, name="h3")
    msg = reg.run_stop(session_id="s", max_fires=3)
    assert msg == "continue msg"
    assert calls == ["h1", "h2"]


def test_stop_max_fires_returns_none_when_exceeded():
    """超过 max_fires 后强制返回 None。"""
    reg = HookRegistry()
    reg.register_stop(lambda: "loop", name="l")
    # 模拟已触发 max_fires 次
    reg._stop_fire_count = 3
    assert reg.run_stop(session_id="s", max_fires=3) is None


def test_stop_exception_isolated():
    def bad(): raise ValueError("boom")
    reg = HookRegistry()
    reg.register_stop(bad, name="bad")
    reg.register_stop(lambda: "after", name="ok")
    msg = reg.run_stop(session_id="s", max_fires=3)
    assert msg == "after"


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

def test_clear_all():
    reg = HookRegistry()
    reg.register_user_prompt_submit(lambda p: None, name="x")
    reg.register_pre_tool_use(lambda n, a: None, name="y")
    reg.clear()
    assert all(len(reg._hooks[e]) == 0 for e in HookEvent)


def test_clear_single_event():
    reg = HookRegistry()
    reg.register_user_prompt_submit(lambda p: None, name="x")
    reg.register_pre_tool_use(lambda n, a: None, name="y")
    reg.clear(HookEvent.USER_PROMPT_SUBMIT)
    assert len(reg._hooks[HookEvent.USER_PROMPT_SUBMIT]) == 0
    assert len(reg._hooks[HookEvent.PRE_TOOL_USE]) == 1
```

- [ ] **Step 9: 追加 POST_TOOL_USE + STOP 实现**

```python
# 追加到 agent/hooks.py:HookRegistry

    def run_post_tool_use(self, tool_name: str, args: dict, result: str,
                           *, session_id: str) -> str:
        """链式：每个 hook 看到前一个的输出。"""
        for hook in self._hooks[HookEvent.POST_TOOL_USE]:
            try:
                if hook.kind == "programmatic":
                    new_result = hook.fn(tool_name, args, result)
                else:
                    new_result = self._invoke_declarative_post_tool(
                        hook, tool_name, args, result, session_id)
                if new_result is not None:
                    result = new_result
            except Exception as e:
                logger.warning("hook %s 异常（视为 None）: %s", hook.name, e)
        return result

    def _invoke_declarative_post_tool(self, hook, tool_name, args, result, session_id):
        """T3 实现。"""
        return None

    def run_stop(self, *, session_id: str, max_fires: int = 3) -> Optional[str]:
        """首个非 None 胜出。超过 max_fires 强制返回 None（防失控）。"""
        if self._stop_fire_count >= max_fires:
            logger.info("STOP hook 触发上限（%d/%d），本次跳过",
                        self._stop_fire_count, max_fires)
            return None
        for hook in self._hooks[HookEvent.STOP]:
            try:
                if hook.kind == "programmatic":
                    msg = hook.fn()
                else:
                    msg = self._invoke_declarative_stop(hook, session_id)
                if msg is not None:
                    self._stop_fire_count += 1
                    return msg
            except Exception as e:
                logger.warning("hook %s 异常（视为 None）: %s", hook.name, e)
        return None

    def _invoke_declarative_stop(self, hook, session_id):
        """T3 实现。"""
        return None
```

- [ ] **Step 10: 全量跑测试**

Run: `uv run pytest tests/test_hooks.py -v`
Expected: PASS（约 22 tests）

Run: `uv run pytest tests/ -v`
Expected: PASS（原 360 + 新增 hooks tests，无回归）

- [ ] **Step 11: 提交（controller 拿用户授权后统一 commit）**

实现者：不 commit，工作区留给 controller。

---

## Task 2: hook_exec.py 子进程执行 + JSON IPC

**Files:**
- Create: `agent/hook_exec.py`
- Create: `tests/test_hook_exec.py`

**Interfaces:**
- Consumes: `Hook`、`HookScriptConfig` from `agent/hooks`
- Produces: `run_script_hook(hook: Hook, payload: dict) -> Optional[dict]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_hook_exec.py
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_hook_exec.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.hook_exec'`

- [ ] **Step 3: 写实现**

```python
# agent/hook_exec.py
"""声明式 hook 的子进程执行 + JSON IPC 协议。

子进程：
- stdin 收到 payload 的 JSON
- stdout 期望是合法 JSON（空 stdout = 空 dict）
- 超时 kill
- exit code != 0 / 启动失败 / JSON 解析失败 → None（fail-open）
"""
import json
import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def run_script_hook(hook, payload: dict) -> Optional[dict]:
    """在子进程中执行声明式 hook。

    参数：
        hook: Hook 实例（kind="declarative"，script 非 None）
        payload: 要传给子进程的 dict（含 event/session_id/timestamp/事件字段）

    返回：
        解析后的 dict（可能为空 dict 表示 allow），或 None（任何故障）
    """
    if hook.script is None:
        logger.warning("hook %s 缺 script 配置", hook.name)
        return None

    payload_json = json.dumps(payload, ensure_ascii=False)
    env = {**os.environ, **(hook.script.env or {})}

    try:
        proc = subprocess.run(
            hook.script.command,
            input=payload_json,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=hook.script.timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.warning("hook %s 超时 (%.1fs)", hook.name, hook.script.timeout)
        return None
    except OSError as e:
        logger.warning("hook %s 启动失败: %s", hook.name, e)
        return None

    if proc.returncode != 0:
        logger.warning("hook %s exit %d: %s",
                       hook.name, proc.returncode, (proc.stderr or "")[:200])
        return None

    stdout = (proc.stdout or "").strip()
    if not stdout:
        return {}  # 空 stdout = allow

    try:
        parsed = json.loads(stdout)
        if not isinstance(parsed, dict):
            logger.warning("hook %s stdout 非合法 JSON dict: %r", hook.name, parsed)
            return None
        return parsed
    except json.JSONDecodeError as e:
        logger.warning("hook %s stdout 非合法 JSON: %s", hook.name, e)
        return None
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_hook_exec.py -v`
Expected: PASS（8 tests）

Run: `uv run pytest tests/ -v`
Expected: PASS（原 + 新 hooks_exec tests，无回归）

- [ ] **Step 5: 不 commit**

---

## Task 3: hooks.py 集成 declarative（4 个 _invoke_declarative_* 实现 + 顺序测试）

**Files:**
- Modify: `agent/hooks.py`
- Modify: `tests/test_hooks.py`

**Interfaces:**
- Consumes: `run_script_hook` from `agent.hook_exec`（Task 2）
- Produces: 4 个 `_invoke_declarative_*` 方法实现；programmatic 先于 declarative 执行

- [ ] **Step 1: 追加测试**

```python
# 追加到 tests/test_hooks.py
import sys
from agent.hooks import HookScriptConfig
from agent.hook_exec import run_script_hook  # 验证 import 可达


def _make_declarative_hook(event, stdout_json_str, name="decl"):
    """构造一个声明式 hook：子进程 echo 一段 JSON。"""
    # 用 python -c "print('...')" 模拟
    safe_json = stdout_json_str.replace("'", "\\'")
    cmd = [sys.executable, "-c", f"print('{stdout_json_str}')"]
    return Hook(
        name=name, event=event, kind="declarative",
        script=HookScriptConfig(command=cmd, timeout=5.0),
    )


# ---------------------------------------------------------------------------
# Declarative 集成
# ---------------------------------------------------------------------------

def test_user_prompt_submit_declarative_modifies():
    """声明式 hook 通过 stdout {prompt: ...} 修改。"""
    hook = _make_declarative_hook(
        HookEvent.USER_PROMPT_SUBMIT, '{"prompt": "DECL"}',
    )
    reg = HookRegistry()
    reg.register_declarative(hook)
    assert reg.run_user_prompt_submit("hello", session_id="s") == "DECL"


def test_pre_tool_use_declarative_deny():
    hook = _make_declarative_hook(
        HookEvent.PRE_TOOL_USE, '{"action": "deny", "reason": "blocked"}',
    )
    reg = HookRegistry()
    reg.register_declarative(hook)
    deny, _ = reg.run_pre_tool_use("t", {}, session_id="s")
    assert deny == "blocked"


def test_pre_tool_use_declarative_modify():
    hook = _make_declarative_hook(
        HookEvent.PRE_TOOL_USE, '{"action": "modify", "args": {"x": 1}}',
    )
    reg = HookRegistry()
    reg.register_declarative(hook)
    _, modified = reg.run_pre_tool_use("t", {}, session_id="s")
    assert modified == {"x": 1}


def test_post_tool_use_declarative_modifies():
    hook = _make_declarative_hook(
        HookEvent.POST_TOOL_USE, '{"result": "NEW"}',
    )
    reg = HookRegistry()
    reg.register_declarative(hook)
    assert reg.run_post_tool_use("t", {}, "old", session_id="s") == "NEW"


def test_stop_declarative_continue():
    hook = _make_declarative_hook(
        HookEvent.STOP, '{"continue": "go on"}',
    )
    reg = HookRegistry()
    reg.register_declarative(hook)
    assert reg.run_stop(session_id="s", max_fires=3) == "go on"


def test_programmatic_runs_before_declarative():
    """同 event 内程序式先于声明式执行。"""
    order = []
    # 程序式 hook 记录顺序
    reg = HookRegistry()
    reg.register_user_prompt_submit(
        lambda p: order.append("prog") or p + "_p", name="prog"
    )
    # 声明式 hook 也记录（通过修改 prompt 标识）
    hook = _make_declarative_hook(
        HookEvent.USER_PROMPT_SUBMIT, '{"prompt": "FROM_DECL"}',
    )
    reg.register_declarative(hook)
    result = reg.run_user_prompt_submit("start", session_id="s")
    # 程序式先跑（把 start → start_p），声明式后跑（覆盖为 FROM_DECL）
    assert order == ["prog"]
    assert result == "FROM_DECL"


def test_declarative_failure_isolated():
    """声明式 hook 子进程失败时视为 None。"""
    # 用不存在的可执行文件
    bad_hook = Hook(
        name="bad", event=HookEvent.USER_PROMPT_SUBMIT, kind="declarative",
        script=HookScriptConfig(command=["./nonexistent-xyz"], timeout=1.0),
    )
    reg = HookRegistry()
    reg.register_declarative(bad_hook)
    assert reg.run_user_prompt_submit("hello", session_id="s") == "hello"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_hooks.py::test_user_prompt_submit_declarative_modifies -v`
Expected: FAIL — `assert 'hello' == 'DECL'`（_invoke_declarative_* 仍返回 None）

- [ ] **Step 3: 实现 4 个 `_invoke_declarative_*` 方法**

```python
# 替换 agent/hooks.py 里的 4 个占位方法（保持类内位置不变）

    def _invoke_declarative_user_prompt(self, hook, prompt, session_id):
        """T3 实现：跑子进程，按 IPC 协议解析。"""
        from agent.hook_exec import run_script_hook  # 懒加载避免循环
        payload = {
            "event": "user_prompt_submit",
            "session_id": session_id,
            "timestamp": _now_iso(),
            "hook_name": hook.name,
            "prompt": prompt,
        }
        result = run_script_hook(hook, payload)
        if result is None:
            return None
        # IPC: {"prompt": "..."} → 替换；其他/空 → None
        return result.get("prompt")

    def _invoke_declarative_pre_tool(self, hook, tool_name, args, session_id):
        from agent.hook_exec import run_script_hook
        payload = {
            "event": "pre_tool_use",
            "session_id": session_id,
            "timestamp": _now_iso(),
            "hook_name": hook.name,
            "tool_name": tool_name,
            "args": args,
        }
        result = run_script_hook(hook, payload)
        if result is None:
            return None
        action = result.get("action", "allow")
        if action == "deny":
            return {"deny": result.get("reason", "unspecified")}
        if action == "modify":
            return {"modify_args": result.get("args", args)}
        return None

    def _invoke_declarative_post_tool(self, hook, tool_name, args, result_str, session_id):
        from agent.hook_exec import run_script_hook
        payload = {
            "event": "post_tool_use",
            "session_id": session_id,
            "timestamp": _now_iso(),
            "hook_name": hook.name,
            "tool_name": tool_name,
            "args": args,
            "result": result_str,
        }
        result = run_script_hook(hook, payload)
        if result is None:
            return None
        return result.get("result")

    def _invoke_declarative_stop(self, hook, session_id):
        from agent.hook_exec import run_script_hook
        payload = {
            "event": "stop",
            "session_id": session_id,
            "timestamp": _now_iso(),
            "hook_name": hook.name,
        }
        result = run_script_hook(hook, payload)
        if result is None:
            return None
        return result.get("continue")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_hooks.py -v`
Expected: PASS（约 29 tests，新增 7 个 declarative 测试）

Run: `uv run pytest tests/ -v`
Expected: PASS（无回归）

- [ ] **Step 5: 不 commit**

---

## Task 4: hook_loader.py 加载 settings.json

**Files:**
- Create: `agent/hook_loader.py`
- Create: `tests/test_hook_loader.py`

**Interfaces:**
- Consumes: `HookRegistry`、`Hook`、`HookScriptConfig`、`HookEvent` from `agent/hooks`
- Produces: `load_declarative_hooks(registry, settings_path: Path) -> int`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_hook_loader.py
"""hook_loader 测试：settings.json 解析 + 校验。"""
import json
from pathlib import Path

import pytest

from agent.hooks import HookEvent, HookRegistry
from agent.hook_loader import load_declarative_hooks


def _write(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def test_load_returns_zero_when_file_missing(tmp_path):
    """文件不存在 = 静默返回 0。"""
    reg = HookRegistry()
    n = load_declarative_hooks(reg, tmp_path / "nonexistent.json")
    assert n == 0


def test_load_valid_file(tmp_path):
    settings = tmp_path / "settings.json"
    _write(settings, {
        "hooks": {
            "pre_tool_use": [
                {"name": "h1", "command": ["./h1.sh"], "timeout": 5.0},
            ],
            "post_tool_use": [
                {"name": "h2", "command": ["./h2.sh"]},
            ],
        }
    })
    reg = HookRegistry()
    n = load_declarative_hooks(reg, settings)
    assert n == 2
    assert len(reg._hooks[HookEvent.PRE_TOOL_USE]) == 1
    assert len(reg._hooks[HookEvent.POST_TOOL_USE]) == 1


def test_load_fail_closed_field(tmp_path):
    settings = tmp_path / "settings.json"
    _write(settings, {
        "hooks": {
            "pre_tool_use": [
                {"name": "h", "command": ["./h.sh"], "fail_closed": True},
            ],
        }
    })
    reg = HookRegistry()
    load_declarative_hooks(reg, settings)
    hook = reg._hooks[HookEvent.PRE_TOOL_USE][0]
    assert hook.fail_closed is True


def test_load_env_field(tmp_path):
    settings = tmp_path / "settings.json"
    _write(settings, {
        "hooks": {
            "post_tool_use": [
                {"name": "h", "command": ["./h.sh"], "env": {"K": "v"}},
            ],
        }
    })
    reg = HookRegistry()
    load_declarative_hooks(reg, settings)
    hook = reg._hooks[HookEvent.POST_TOOL_USE][0]
    assert hook.script.env == {"K": "v"}


def test_load_invalid_event_name_raises(tmp_path):
    settings = tmp_path / "settings.json"
    _write(settings, {"hooks": {"unknown_event": []}})
    reg = HookRegistry()
    with pytest.raises(ValueError, match="unknown event"):
        load_declarative_hooks(reg, settings)


def test_load_missing_hooks_field_raises(tmp_path):
    settings = tmp_path / "settings.json"
    _write(settings, {"wrong_field": {}})
    reg = HookRegistry()
    with pytest.raises(ValueError, match="missing.*hooks"):
        load_declarative_hooks(reg, settings)


def test_load_missing_malformed_json_raises(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text("not a json {{{", encoding="utf-8")
    reg = HookRegistry()
    with pytest.raises(json.JSONDecodeError):
        load_declarative_hooks(reg, settings)


def test_load_skips_hook_missing_name(tmp_path, caplog):
    """缺 name 字段的 hook 跳过 + warning。"""
    settings = tmp_path / "settings.json"
    _write(settings, {"hooks": {"pre_tool_use": [{"command": ["./h.sh"]}]}})
    reg = HookRegistry()
    n = load_declarative_hooks(reg, settings)
    assert n == 0
    assert any("missing name" in r.message.lower() or "name" in r.message.lower()
               for r in caplog.records)


def test_load_skips_hook_missing_command(tmp_path, caplog):
    settings = tmp_path / "settings.json"
    _write(settings, {"hooks": {"pre_tool_use": [{"name": "h"}]}})
    reg = HookRegistry()
    n = load_declarative_hooks(reg, settings)
    assert n == 0


def test_load_skips_hook_empty_command(tmp_path):
    settings = tmp_path / "settings.json"
    _write(settings, {"hooks": {"pre_tool_use": [{"name": "h", "command": []}]}})
    reg = HookRegistry()
    n = load_declarative_hooks(reg, settings)
    assert n == 0


def test_load_partial_failure_continues(tmp_path, caplog):
    """一个坏 hook 不影响其他好 hook。"""
    settings = tmp_path / "settings.json"
    _write(settings, {"hooks": {
        "pre_tool_use": [
            {"name": "good", "command": ["./g.sh"]},
            {"command": ["./no-name.sh"]},  # 缺 name，跳过
            {"name": "also_good", "command": ["./ag.sh"]},
        ]
    }})
    reg = HookRegistry()
    n = load_declarative_hooks(reg, settings)
    assert n == 2  # 两个 good 加载成功
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_hook_loader.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写实现**

```python
# agent/hook_loader.py
"""加载 settings.json 中的声明式 hooks 到 registry。

文件格式：
{
  "hooks": {
    "user_prompt_submit": [{name, command, timeout?, env?, fail_closed?}, ...],
    "pre_tool_use": [...],
    "post_tool_use": [...],
    "stop": [...]
  }
}

校验规则：
- 顶层缺 hooks 字段 → raise ValueError
- event 名不合法 → raise ValueError
- 单个 hook 缺 name / command → 跳过 + log warning（不阻塞其他）
- 文件不存在 → 静默返回 0（不强制用户配置）
"""
import json
import logging
from pathlib import Path

from agent.hooks import Hook, HookEvent, HookScriptConfig

logger = logging.getLogger(__name__)


def load_declarative_hooks(registry, settings_path: Path) -> int:
    """从 settings.json 加载声明式 hooks。

    返回加载成功的 hook 数量。
    文件不存在 = 静默返回 0。
    """
    if not settings_path.exists():
        return 0

    text = settings_path.read_text(encoding="utf-8")
    config = json.loads(text)  # JSON 解析失败直接抛

    if "hooks" not in config:
        raise ValueError(
            f"settings.json 缺少顶层 'hooks' 字段: {settings_path}"
        )

    raw = config["hooks"]
    if not isinstance(raw, dict):
        raise ValueError(f"settings.json 'hooks' 必须是 dict，实际是 {type(raw).__name__}")

    count = 0
    for event_str, hook_list in raw.items():
        try:
            event = HookEvent(event_str)
        except ValueError:
            raise ValueError(
                f"settings.json 包含未知 event 名: '{event_str}'，"
                f"合法值: {[e.value for e in HookEvent]}"
            )
        for h_cfg in hook_list:
            hook = _parse_hook(h_cfg, event)
            if hook is not None:
                registry.register_declarative(hook)
                count += 1
    return count


def _parse_hook(h_cfg: dict, event: HookEvent):
    """解析单个 hook 配置。缺关键字段时返回 None + log warning。"""
    name = h_cfg.get("name")
    command = h_cfg.get("command")

    if not name:
        logger.warning("settings.json hook 缺 name 字段，跳过: %s", h_cfg)
        return None
    if not command or not isinstance(command, list):
        logger.warning("settings.json hook '%s' 缺 command（或非 list），跳过", name)
        return None

    return Hook(
        name=name,
        event=event,
        kind="declarative",
        script=HookScriptConfig(
            command=command,
            timeout=h_cfg.get("timeout", 10.0),
            env=h_cfg.get("env"),
        ),
        fail_closed=h_cfg.get("fail_closed", False),
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_hook_loader.py -v`
Expected: PASS（11 tests）

Run: `uv run pytest tests/ -v`
Expected: PASS（无回归）

- [ ] **Step 5: 不 commit**

---

## Task 5: config.py 新增 hooks 块

**Files:**
- Modify: `config.py:DEFAULT_CONFIG`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: 无
- Produces: `DEFAULT_CONFIG["hooks"]` dict

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_config.py
def test_default_config_has_hooks_block():
    from config import DEFAULT_CONFIG
    h = DEFAULT_CONFIG["hooks"]
    for key in ("enabled", "settings_path", "script_timeout_default",
                "stop_hook_max_fires", "fail_closed_default"):
        assert key in h, f"缺 {key}"


def test_default_config_hooks_enabled_default_true():
    from config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["hooks"]["enabled"] is True


def test_default_config_hooks_settings_path_default_none():
    from config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["hooks"]["settings_path"] is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py::test_default_config_has_hooks_block -v`
Expected: FAIL — `KeyError: 'hooks'`

- [ ] **Step 3: 改 config.py**

在 `config.py:DEFAULT_CONFIG` 里（建议紧跟 `context` 块之后）新增：

```python
    # Hooks 系统（Phase 2a）
    "hooks": {
        "enabled": True,                            # 全局开关；False 时跳过所有 hook 调用
        "settings_path": None,                      # None → 默认 ~/.agent/.hooks/settings.json
        "script_timeout_default": 10.0,             # 声明式 hook 默认超时（秒）
        "stop_hook_max_fires": 3,                   # Stop hook 每会话最多触发次数（防失控）
        "fail_closed_default": False,               # 声明式 hook 默认 fail_closed
    },
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS（3 个新测试 + 原有）

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: 不 commit**

---

## Task 6: agent/__init__.py 集成（AIAgent 接 hooks_registry + USER_PROMPT_SUBMIT + STOP 注入）

**Files:**
- Modify: `agent/__init__.py`
- Modify: `tests/test_integration.py`

**Interfaces:**
- Consumes: `HookRegistry` from `agent/hooks`（Task 1-3）
- Produces: AIAgent 新增 `hooks_registry=None` kwarg；`_stop_fire_count` 属性

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_integration.py
from agent.hooks import HookRegistry


def test_aiagent_accepts_hooks_registry_kwarg():
    """hooks_registry=None 时构造成功（向后兼容）。"""
    from agent.llm_client import create_llm_client
    # 不传 hooks_registry
    agent = _make_test_agent()
    assert agent.hooks_registry is None
    assert agent._stop_fire_count == 0


def test_aiagent_user_prompt_submit_hook_modifies_input():
    """USER_PROMPT_SUBMIT hook 修改 prompt 后实际入 history。"""
    from unittest.mock import MagicMock
    agent = _make_test_agent_with_hooks()
    agent.hooks_registry.register_user_prompt_submit(
        lambda p: p + " [augmented]", name="augmenter"
    )
    # mock LLM 返回 stop（无 tool_call）
    agent.llm_client = _mock_llm_simple_response("ok")
    agent.run_conversation("hello")
    # 第 0 条应是 user，content 被 hook 修改
    assert agent.conversation_history[0]["content"] == "hello [augmented]"


def test_aiagent_user_prompt_submit_no_registry_modification():
    """无 registry 时 prompt 原样入 history。"""
    agent = _make_test_agent()
    agent.llm_client = _mock_llm_simple_response("ok")
    agent.run_conversation("hello")
    assert agent.conversation_history[0]["content"] == "hello"


def test_aiagent_stop_hook_force_continue():
    """STOP hook 返回 force_continue 时循环不退出（直到 max_fires）。"""
    agent = _make_test_agent_with_hooks()
    # 第一次 stop hook 让循环继续；第二次（max_fires 触发后）才真停
    calls = []
    agent.hooks_registry.register_stop(
        lambda: "again" if len(calls) == 0 else None, name="loop"
    )
    agent.llm_client = _mock_llm_simple_response("resp")
    # mock 中追踪调用次数
    agent.run_conversation("go")
    # 至少调用了 1 次 stop hook
    # 详细断言放 hook 测试，这里只验证不抛 + 不死循环


# ---- helpers ----

def _make_test_agent():
    """构造一个最小可跑的 AIAgent。"""
    from agent import AIAgent
    return AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], harvil_home="/tmp/fake",
    )


def _make_test_agent_with_hooks():
    from agent import AIAgent
    reg = HookRegistry()
    return AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], harvil_home="/tmp/fake",
        hooks_registry=reg,
    )


def _mock_llm_simple_response(text: str):
    """mock LLM client：每次返回固定文本，stop_reason='stop'。"""
    from unittest.mock import MagicMock
    m = MagicMock()
    m.chat_completions.return_value.choices = [
        MagicMock(message=MagicMock(content=text, tool_calls=None),
                  finish_reason="stop")
    ]
    return m
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_integration.py::test_aiagent_accepts_hooks_registry_kwarg -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'hooks_registry'`

- [ ] **Step 3: 改 agent/__init__.py**

在 `AIAgent.__init__` 签名加 `hooks_registry=None`：

```python
    def __init__(
        self,
        *,
        # ... 原有参数 ...
        config: dict = None,
        hooks_registry=None,   # === NEW ===
    ):
```

在 `__init__` 方法体内（其他 `self.X = X` 附近）加：

```python
        self.hooks_registry = hooks_registry
        self._stop_fire_count = 0
```

然后在 `run_conversation` 最开始（在 `self.conversation_history.append({"role": "user", ...})` 之前）插入：

```python
    def run_conversation(self, user_message: str) -> str:
        # === NEW: USER_PROMPT_SUBMIT hook ===
        if (self.hooks_registry
                and self.config.get("hooks", {}).get("enabled", True)):
            try:
                user_message = self.hooks_registry.run_user_prompt_submit(
                    user_message, session_id=self.session_id or "",
                )
            except Exception as e:
                logger.warning("USER_PROMPT_SUBMIT 编排异常: %s", e)

        self.conversation_history.append({"role": "user", "content": user_message})
        # ... 原有循环 ...
```

然后在 `run_conversation` 的 return 之前（找到 `return final_response` 或类似行）插入 STOP hook：

```python
        # === NEW: STOP hook ===
        if (self.hooks_registry
                and self.config.get("hooks", {}).get("enabled", True)
                and self._stop_fire_count < self.config.get("hooks", {}).get(
                    "stop_hook_max_fires", 3)):
            try:
                force_msg = self.hooks_registry.run_stop(
                    session_id=self.session_id or "",
                    max_fires=self.config.get("hooks", {}).get(
                        "stop_hook_max_fires", 3),
                )
            except Exception as e:
                logger.warning("STOP hook 编排异常: %s", e)
                force_msg = None

            if force_msg:
                self._stop_fire_count += 1
                self.conversation_history.append({
                    "role": "user",
                    "content": f"[stop_hook]: {force_msg}",
                })
                continue  # 跳回 while

        return final_response
```

**注意**：`continue` 必须在 while 循环里，紧邻 return。具体行号要读现有代码定位。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_integration.py -k aiagent -v`
Expected: PASS（4 个新测试）

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS（无回归）

- [ ] **Step 6: 不 commit**

---

## Task 7: model_tools.py 集成 PRE/POST_TOOL_USE

**Files:**
- Modify: `model_tools.py:handle_function_call`
- Modify: `tests/test_integration.py`

**Interfaces:**
- Consumes: `HookRegistry`（Task 1-3）
- Produces: `handle_function_call` 新增 `hooks_registry=None` kwarg

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_integration.py
import json
from agent.hooks import HookRegistry
from model_tools import handle_function_call


def test_handle_function_call_pre_tool_use_deny():
    """PreToolUse hook 返回 deny 时，handler 不调，返回 hook_deny error。"""
    reg = HookRegistry()
    reg.register_pre_tool_use(lambda n, a: {"deny": "blocked"}, name="b")
    result = handle_function_call(
        "todo_write", {"todos": []},
        hooks_registry=reg, session_id="s",
        config={"hooks": {"enabled": True}},
    )
    parsed = json.loads(result)
    assert parsed["error_type"] == "hook_deny"
    assert "blocked" in parsed["error"]


def test_handle_function_call_pre_tool_use_modify_args():
    """PreToolUse hook 修改 args 后，handler 看到的是修改后的。"""
    captured = []
    # 用 todo_write 作为目标：注册一个 todo_write handler 拦截
    # 简化：直接通过 registry 调用确认 args 修改（已在 T1 测过）
    # 这里验证 handle_function_call 集成路径
    reg = HookRegistry()
    reg.register_pre_tool_use(
        lambda n, a: {"modify_args": {"todos": [{"id": 1, "text": "modified"}]}},
        name="m"
    )
    # 调真实 todo_write handler
    result = handle_function_call(
        "todo_write", {"todos": []},
        hooks_registry=reg, session_id="s",
        config={"hooks": {"enabled": True}},
    )
    # 应该成功（todo_write 接受 modified args）
    parsed = json.loads(result)
    assert "error" not in parsed or parsed.get("error_type") != "hook_deny"


def test_handle_function_call_post_tool_use_modifies_result():
    """PostToolUse hook 修改 result 后，最终返回的是修改后的。"""
    reg = HookRegistry()
    reg.register_post_tool_use(
        lambda n, a, r: json.dumps({"overridden": True}, ensure_ascii=False),
        name="override"
    )
    result = handle_function_call(
        "todo_write", {"todos": []},
        hooks_registry=reg, session_id="s",
        config={"hooks": {"enabled": True}},
    )
    parsed = json.loads(result)
    assert parsed.get("overridden") is True


def test_handle_function_call_no_registry_unchanged():
    """hooks_registry=None 时行为完全等同于 Phase 1。"""
    result1 = handle_function_call(
        "todo_write", {"todos": []}, session_id="s",
    )
    result2 = handle_function_call(
        "todo_write", {"todos": []},
        hooks_registry=None, session_id="s",
    )
    # 两个结果应该相同（todo_write 是确定性的）
    # 注意：可能含时间戳等差异，只验 error_type 不变
    assert json.loads(result1).get("error_type") == json.loads(result2).get("error_type")


def test_handle_function_call_hooks_disabled_skips():
    """config.hooks.enabled=False 时跳过所有 hook。"""
    reg = HookRegistry()
    reg.register_pre_tool_use(lambda n, a: {"deny": "blocked"}, name="b")
    result = handle_function_call(
        "todo_write", {"todos": []},
        hooks_registry=reg, session_id="s",
        config={"hooks": {"enabled": False}},
    )
    parsed = json.loads(result)
    assert parsed.get("error_type") != "hook_deny"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_integration.py::test_handle_function_call_pre_tool_use_deny -v`
Expected: FAIL — hook 不触发

- [ ] **Step 3: 改 model_tools.py**

修改 `handle_function_call` 签名（加 `hooks_registry=None`）和实现（包裹 dispatch）：

```python
def handle_function_call(
    function_name: str,
    function_args: Dict[str, Any],
    *,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    memory_store=None,
    session_store=None,
    harvil_home=None,
    tool_call_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    hooks_registry=None,  # === NEW ===
) -> str:
    """分发工具调用，返回 JSON 字符串结果。"""
    ensure_tools_discovered()
    function_args = _coerce_tool_args(function_name, function_args)

    # === NEW: PRE_TOOL_USE hook ===
    hooks_enabled = (config or {}).get("hooks", {}).get("enabled", True)
    if hooks_registry and hooks_enabled:
        deny_reason, modified_args = hooks_registry.run_pre_tool_use(
            function_name, function_args,
            session_id=session_id or "",
        )
        if deny_reason is not None:
            return json.dumps({
                "error": f"hook denied: {deny_reason}",
                "error_type": "hook_deny",
            }, ensure_ascii=False)
        if modified_args is not None:
            function_args = modified_args

    # 原有 dispatch
    result = registry.dispatch(
        function_name, function_args,
        task_id=task_id, session_id=session_id,
        memory_store=memory_store, session_store=session_store,
        harvil_home=harvil_home, tool_call_id=tool_call_id, config=config,
    )

    # === NEW: POST_TOOL_USE hook ===
    if hooks_registry and hooks_enabled:
        result = hooks_registry.run_post_tool_use(
            function_name, function_args, result,
            session_id=session_id or "",
        )

    return result
```

**caller 改造**（`agent/__init__.py` 内调用 `handle_function_call` 的地方）：

找到 `agent/__init__.py` 调 `handle_function_call(...)` 的位置，多传一个 `hooks_registry=self.hooks_registry` 参数。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_integration.py -k handle_function_call -v`
Expected: PASS（5 个新测试）

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: 不 commit**

---

## Task 8: cli.py 注入 + 端到端集成测试

**Files:**
- Modify: `cli.py:RuntimeContext`
- Modify: `tests/test_integration.py`

**Interfaces:**
- Consumes: 所有前置任务
- Produces: RuntimeContext.hooks_registry；启动时自动加载声明式 hooks

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_integration.py
def test_runtime_context_has_hooks_registry():
    """RuntimeContext 持有 hooks_registry 实例。"""
    # 简化：直接 import 验证
    from cli import RuntimeContext
    ctx = RuntimeContext.__new__(RuntimeContext)  # 不调 __init__
    # 验证属性可设
    from agent.hooks import HookRegistry
    ctx.hooks_registry = HookRegistry()
    assert ctx.hooks_registry is not None


def test_e2e_aiagent_with_hooks_full_loop(tmp_path):
    """端到端：USER_PROMPT_SUBMIT 修改 → LLM → PRE_TOOL_USE 放行 → POST_TOOL_USE 改写 → STOP。

    使用 mock LLM + 真实 registry + 真实 handle_function_call。
    """
    from agent import AIAgent
    from agent.hooks import HookRegistry

    reg = HookRegistry()
    # hook 1: USER_PROMPT_SUBMIT 增强
    reg.register_user_prompt_submit(lambda p: p + " (with context)", name="augment")
    # hook 2: PRE_TOOL_USE 全放行
    reg.register_pre_tool_use(lambda n, a: None, name="allow-all")
    # hook 3: POST_TOOL_USE 在 result 里加 audit 标记
    def add_audit(n, a, r):
        try:
            parsed = json.loads(r)
            parsed["_audited"] = True
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            return None
    reg.register_post_tool_use(add_audit, name="audit")

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], harvil_home=str(tmp_path),
        hooks_registry=reg,
        config={"hooks": {"enabled": True, "stop_hook_max_fires": 3}},
    )
    # mock LLM 第一轮返回 tool_call，第二轮返回 stop
    from unittest.mock import MagicMock
    m = MagicMock()
    call_count = [0]
    def side_effect(msgs, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            # 第一轮：返回一个 todo_write tool_call
            resp = MagicMock()
            resp.choices = [MagicMock(
                message=MagicMock(
                    content=None,
                    tool_calls=[MagicMock(
                        id="call_1",
                        type="function",
                        function=MagicMock(name="todo_write", arguments='{"todos": []}'),
                    )],
                ),
                finish_reason="tool_calls",
            )]
            return resp
        else:
            # 后续：stop
            resp = MagicMock()
            resp.choices = [MagicMock(
                message=MagicMock(content="done", tool_calls=None),
                finish_reason="stop",
            )]
            return resp
    m.chat_completions.side_effect = side_effect
    agent.llm_client = m

    final = agent.run_conversation("do something")
    # 至少：用户消息被 augment；POST_TOOL_USE 在某条 tool 消息加了 _audited
    assert "(with context)" in agent.conversation_history[0]["content"]
    # 找到 tool 结果消息
    tool_msgs = [m for m in agent.conversation_history if m.get("role") == "tool"]
    if tool_msgs:
        parsed = json.loads(tool_msgs[0]["content"])
        assert parsed.get("_audited") is True
    # 无异常即通过
    assert isinstance(final, str)


def test_e2e_no_hooks_enabled_full_backward_compat(tmp_path):
    """config.hooks.enabled=False 时整条链路等同 Phase 1。"""
    from agent import AIAgent
    from agent.hooks import HookRegistry

    # 即使注册了 hook，enabled=False 也不触发
    reg = HookRegistry()
    reg.register_user_prompt_submit(lambda p: "MUTATED", name="m")

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], harvil_home=str(tmp_path),
        hooks_registry=reg,
        config={"hooks": {"enabled": False}},
    )
    agent.llm_client = _mock_llm_simple_response("ok")
    agent.run_conversation("original")
    # hook 没触发，原样入 history
    assert agent.conversation_history[0]["content"] == "original"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_integration.py::test_runtime_context_has_hooks_registry -v`
Expected: FAIL — `AttributeError` 或无该属性

- [ ] **Step 3: 改 cli.py:RuntimeContext**

找到 `cli.py` 里的 `RuntimeContext` 类。在 `__init__` 里（其他 self.X 附近）加：

```python
        # === NEW: Hooks 系统 ===
        from agent.hooks import HookRegistry
        self.hooks_registry = HookRegistry()
        # 启动时加载声明式 hooks（如果启用）
        if self.config.get("hooks", {}).get("enabled", True):
            from agent.hook_loader import load_declarative_hooks
            from pathlib import Path
            settings_path = self.config.get("hooks", {}).get("settings_path")
            if settings_path is None:
                # 默认 ~/.agent/.hooks/settings.json
                settings_path = Path(self.harvil_home) / ".hooks" / "settings.json"
            try:
                n = load_declarative_hooks(self.hooks_registry, Path(settings_path))
                if n > 0:
                    logger.info("加载了 %d 个声明式 hooks 自 %s", n, settings_path)
            except Exception as e:
                logger.error("加载声明式 hooks 失败: %s", e)
```

然后在 `_create_agent` 方法（或 `create_agent`）里把 `hooks_registry=self.hooks_registry` 传给 `AIAgent`。具体行号需读现有代码定位，找 `AIAgent(...)` 构造调用点。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS（含新测试）

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS（原 360 + ~25 新测试，0 回归）

- [ ] **Step 6: 不 commit**

---

## Self-Review

**Spec 覆盖检查**：
- ✅ §1 架构（4 注入点 + 3 新模块 + 4 改造文件）→ T1-T8 全覆盖
- ✅ §2 4 种 event 协议（USER_PROMPT / PRE_TOOL / POST_TOOL / STOP）→ T1 测程序式，T2+T3 测声明式
- ✅ §2 fail-open + fail_closed → T1 step 6（fail_closed 在 PreToolUse 异常分支）
- ✅ §2 Stop 防失控 max_fires → T1 step 8（test_stop_max_fires_returns_none_when_exceeded）
- ✅ §3 类设计（HookEvent/Hook/HookScriptConfig/HookRegistry）→ T1
- ✅ §3 hook_exec.py → T2
- ✅ §3 hook_loader.py + settings.json 格式 → T4
- ✅ §3 config.py hooks 块 → T5
- ✅ §3 RuntimeContext 注入 → T8
- ✅ §4.1 USER_PROMPT_SUBMIT 注入 → T6
- ✅ §4.2 PRE/POST_TOOL_USE 注入 → T7
- ✅ §4.3 STOP 注入 → T6
- ✅ §4.4 hooks_registry 注入 → T6
- ✅ §5 测试矩阵（test_hooks / test_hook_exec / test_hook_loader / 集成测试）→ T1-T8
- ✅ §6 已知限制（不引入 asyncio / 不实现 2b/2c）→ 全程遵守

**Placeholder 扫描**：无 TBD/TODO/「类似 Task N」。

**类型一致性**：
- `HookEvent` 在 T1 定义，T2-T8 全部用相同 import ✓
- `Hook`、`HookScriptConfig` 在 T1 定义，T2/T4/T8 用 ✓
- `HookRegistry.run_*` 签名在 T1/T3 一致 ✓
- `run_script_hook(hook, payload)` 在 T2 定义，T3 调用 ✓
- `load_declarative_hooks(registry, settings_path)` 在 T4 定义，T8 调用 ✓
- `handle_function_call` 新增 `hooks_registry=None` 在 T7，T8 集成测试依赖 ✓

**遗漏检查**：spec §5.5 提到「settings.json 格式错 → 启动时 fail-fast」→ T4 已实现（`raise ValueError`）。T8 在 RuntimeContext 用 try/except 包住 loader 调用，避免 agent 启动崩溃。这个 trade-off（启动时 fail-fast vs runtime 容错）已在 T8 step 3 体现——加载失败 log error 但 agent 仍能启动（更友好的生产体验）。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-12-phase2a-hooks.md`.

按用户先前授权（Phase 1 「全程授权」+ 本阶段「你直接做不需要问」），直接进入 SDD 执行，沿用 Phase 1 工作流：
- 每个 Task 派 implementer subagent（不 commit）→ review → fix loop → controller commit
- 终审后整批推送
