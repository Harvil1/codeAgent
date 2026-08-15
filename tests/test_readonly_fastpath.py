"""T7（核心机制对齐第 7 项）：只读命令识别 → 自动免审批 + concurrency-safe。

- _is_readonly_command：30+ 只读前缀表；复合命令（&&/||/;/|/$()/反引号）
  必须每段只读才算只读；重定向（>/>>）出现即非只读
- 闸门顺序：fatal → 黑名单 → 只读快速通道（自动批）→ 破坏性审批 → LLM 分类器
- terminal 工具 isConcurrencySafe 保持 False，但 _dispatch_tool_calls 按
  命令动态判定加入并发组
- config 开关 security.readonly_fastpath_enabled（默认 True）
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.permission import PermissionChecker, _is_readonly_command


# ---------------------------------------------------------------------------
# 纯函数：只读判定
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", [
    "ls -la",
    "cat foo.txt",
    "git status",
    "git log --oneline -5",
    "git diff HEAD~1",
    "git show abc123",
    "rg TODO agent/",
    "dir",
    "tree /F",
    "du -sh .",
    "df -h",
    "wc -l foo.py",
    "head -20 x.log",
    "tail -f app.log",
    "pip list",
    "pip show requests",
    "uv pip list",
    "python --version",
    "echo hello",
    "whoami",
    "find . -name '*.py'",
])
def test_readonly_commands(cmd):
    assert _is_readonly_command(cmd) is True


@pytest.mark.parametrize("cmd", [
    "git push",                      # 写远端
    "git commit -m x",               # 写历史
    "rm foo.txt",                    # 破坏性
    "git branch feature-x",          # branch 创建（git branch 带名字是写）
    "find . -delete",                # find -delete 破坏性
    "find . -exec rm {} ;",          # find -exec
    "cat a.txt > b.txt",             # 重定向写
    "echo hi >> log.txt",            # 追加重定向
    "ls; rm -rf /tmp/x",             # 复合：一段读一段写
    "cat a && cat b | rm x",         # 复合含非只读段
    "echo $(rm -rf /)",              # 子命令替换
    "cat `whoami`",                  # 反引号
    "pip install requests",          # 安装（写环境）
    "",
])
def test_non_readonly_commands(cmd):
    assert _is_readonly_command(cmd) is False


def test_compound_all_readonly_ok():
    assert _is_readonly_command("git status && git log -3 | head -5") is True
    assert _is_readonly_command("ls; pwd") is True


# ---------------------------------------------------------------------------
# 闸门：只读快速通道自动批
# ---------------------------------------------------------------------------

def test_readonly_fastpath_auto_approve(tmp_path, monkeypatch):
    """只读命令在审批之前自动批（回调不被调）。"""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    calls = []

    def cb(item):
        calls.append(item)
        return False  # 即使回调拒绝也不该被问到

    c = PermissionChecker(approval_callback=cb)
    # git push 不在只读表 → 正常流程（这里给审批路径）
    # 构造一个会命中破坏性审批的写命令确认回调仍工作
    r = c.check("git status")
    assert r.allowed is True
    assert "readonly" in r.reason.lower() or "只读" in r.reason
    assert len(calls) == 0


def test_readonly_fastpath_disabled_via_config(tmp_path, monkeypatch):
    """security.readonly_fastpath_enabled=False → 关闭快速通道（走原闸门）。"""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    c = PermissionChecker()
    c.set_config_provider(lambda: {"security": {"readonly_fastpath_enabled": False}})
    r = c.check("git status")
    # 走原闸门 → 闸门 3 默认通过（reason 不是只读快速通道）
    assert r.allowed is True
    assert "只读" not in r.reason and "readonly" not in r.reason.lower()


def test_fatal_gate_before_fastpath():
    """fatal 底线先于快速通道（rm -rf / 任何情况都拒）。"""
    c = PermissionChecker()
    r = c.check("cat /etc/passwd && rm -rf /")
    assert r.allowed is False


def test_config_key_present():
    from config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["security"]["readonly_fastpath_enabled"] is True


# ---------------------------------------------------------------------------
# 并发分组：terminal 只读命令动态判定
# ---------------------------------------------------------------------------

def test_dispatch_grouping_readonly_terminal_to_safe():
    """_dispatch_tool_calls 对只读 terminal 调用加入 safe 并发组（源码级防漏改）。"""
    import inspect
    from agent import AIAgent
    src = inspect.getsource(AIAgent._dispatch_tool_calls)
    assert "is_readonly_command" in src
    assert "terminal" in src


def test_is_readonly_command_public_api():
    """并发分组用的公开入口存在。"""
    from agent.permission import is_readonly_command
    assert is_readonly_command("git status") is True
    assert is_readonly_command("rm x") is False
