"""glob 工具测试。

文件名模式匹配，不读内容。
结果按 mtime 降序（最近改的在前）。
"""
import inspect
import json
import time
from pathlib import Path

from tools.glob_tool import GLOB_SCHEMA, _handle_glob


def _run(pattern, path=".", **kw):
    """便捷包装：调 _handle_glob 并解析 JSON。"""
    return json.loads(_handle_glob(
        {"pattern": pattern, "path": path, **kw}, cwd=".",
    ))


def test_glob_matches_simple(tmp_path, monkeypatch):
    """简单模式匹配：只返回 .py，不返回 .md。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("x", encoding="utf-8")
    result = _run("*.py")
    assert result["count"] == 1
    assert "a.py" in result["matches"][0]


def test_glob_recursive(tmp_path, monkeypatch):
    """递归匹配 **/*.py 跨子目录。"""
    monkeypatch.chdir(tmp_path)
    sub = tmp_path / "src" / "deep"
    sub.mkdir(parents=True)
    (sub / "x.py").write_text("x", encoding="utf-8")
    (tmp_path / "top.py").write_text("x", encoding="utf-8")
    result = _run("**/*.py")
    assert result["count"] == 2


def test_glob_mtime_order(tmp_path, monkeypatch):
    """mtime 降序：新改的文件排在前面。"""
    monkeypatch.chdir(tmp_path)
    f_old = tmp_path / "old.txt"
    f_new = tmp_path / "new.txt"
    f_old.write_text("x", encoding="utf-8")
    # 确保 mtime 有明显差异（Windows FAT/NTFS 精度不一，sleep 50ms）
    time.sleep(0.05)
    f_new.write_text("x", encoding="utf-8")
    result = _run("*.txt")
    assert "new.txt" in result["matches"][0]


def test_glob_truncation(tmp_path, monkeypatch):
    """max_results 钳制 + truncated 标志。"""
    monkeypatch.chdir(tmp_path)
    for i in range(10):
        (tmp_path / f"f{i}.txt").write_text("x", encoding="utf-8")
    result = _run("*.txt", max_results=3)
    assert result["count"] == 3
    assert result["truncated"] is True


def test_glob_no_matches(tmp_path, monkeypatch):
    """无匹配返回空列表。"""
    monkeypatch.chdir(tmp_path)
    result = _run("*.zzz")
    assert result["count"] == 0
    assert result["matches"] == []


def test_glob_protected_path_rejected():
    """safe_path 读保护路径拒绝（如 ~/.ssh）。

    使用 mock 确保行为稳定——测试环境里 ~/.ssh 可能不存在或 HOME
    被改写导致 safe_path 不拦，mock 更可靠。
    """
    from unittest.mock import patch
    from agent.permission import PermissionResult

    with patch("tools.glob_tool.safe_path") as mock_sp:
        mock_sp.return_value = PermissionResult(
            False, "受保护路径: .ssh", "protected",
        )
        result = json.loads(_handle_glob(
            {"pattern": "*", "path": "/fake/.ssh"}, cwd=".",
        ))
    assert result.get("error_type") == "permission_denied"


def test_glob_dispatch_contract():
    """handler 签名 (args, **kwargs)——防 silent-dead-code。"""
    sig = inspect.signature(_handle_glob)
    params = list(sig.parameters.values())
    assert params[0].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)


def test_glob_registered():
    """模块 import 触发注册，且 isConcurrencySafe=True（只读）。"""
    from tools.registry import registry
    import tools.glob_tool  # noqa: F401 触发注册
    entry = registry.get("glob")
    assert entry is not None
    assert entry.isConcurrencySafe is True
