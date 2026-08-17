from unittest.mock import MagicMock, patch


def _make_rt_with_snapshots():
    rt = MagicMock()
    rt.checkpoint_mgr = MagicMock()
    rt.checkpoint_mgr.list_snapshots.return_value = [
        {"id": "snap1", "ts": "2026-08-04T10:00:00", "files": ["/a.py"], "msg_count": 5}
    ]
    rt.checkpoint_mgr.restore_files.return_value = ["/a.py"]
    rt.checkpoint_mgr.get_conversation.return_value = [{"role": "user", "content": "hi"}]
    rt.agent = MagicMock()
    rt.agent.conversation_history = []
    return rt


def test_rewind_mode_1_full_restore(monkeypatch):
    """模式 1 全恢复：restore_files + 恢复对话。"""
    import cli
    rt = _make_rt_with_snapshots()
    # 输入：序号 0 → 模式 1
    inputs = iter(["0", "1"])
    monkeypatch.setattr("cli.console.input", lambda *a, **kw: next(inputs))
    cli._handle_rewind_command(rt, "")
    rt.checkpoint_mgr.restore_files.assert_called_once_with("snap1")
    rt.checkpoint_mgr.get_conversation.assert_called_once_with("snap1")
    assert len(rt.agent.conversation_history) == 1


def test_rewind_mode_2_dialog_only(monkeypatch):
    """模式 2 只对话：restore_files 不应被调用。"""
    import cli
    rt = _make_rt_with_snapshots()
    inputs = iter(["0", "2"])
    monkeypatch.setattr("cli.console.input", lambda *a, **kw: next(inputs))
    cli._handle_rewind_command(rt, "")
    rt.checkpoint_mgr.restore_files.assert_not_called()
    rt.checkpoint_mgr.get_conversation.assert_called_once_with("snap1")


def test_rewind_mode_3_code_only(monkeypatch):
    """模式 3 只代码：get_conversation 不应被调用。"""
    import cli
    rt = _make_rt_with_snapshots()
    inputs = iter(["0", "3"])
    monkeypatch.setattr("cli.console.input", lambda *a, **kw: next(inputs))
    cli._handle_rewind_command(rt, "")
    rt.checkpoint_mgr.restore_files.assert_called_once_with("snap1")
    rt.checkpoint_mgr.get_conversation.assert_not_called()


def test_rewind_mode_4_compact(monkeypatch):
    """模式 4 从此压缩：调 _summarize_rewind。"""
    import cli
    rt = _make_rt_with_snapshots()
    inputs = iter(["0", "4"])
    monkeypatch.setattr("cli.console.input", lambda *a, **kw: next(inputs))
    with patch("cli._summarize_rewind") as mock_summ:
        cli._handle_rewind_command(rt, "")
        mock_summ.assert_called_once_with(rt, "snap1")
