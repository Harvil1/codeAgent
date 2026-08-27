"""auto_extract 对话级记忆提取测试（机械查证）。"""
from agent import auto_extract


class TestExtractPathVerification:
    def test_fake_path_item_dropped(self, tmp_path, monkeypatch):
        """产物引用不存在的路径 → 该条丢弃。"""
        from agent import auto_extract

        monkeypatch.chdir(tmp_path)
        # tmp_path 下只有 real.py
        (tmp_path / "real.py").write_text("x=1", encoding="utf-8")

        class FakeStore:
            def __init__(self):
                self.saved = []

            def save(self, **kw):
                self.saved.append(kw)

        items = [
            {"type": "project", "name": "a", "description": "真实路径",
             "body": "配置在 real.py", "summary": ""},
            {"type": "project", "name": "b", "description": "幻觉路径",
             "body": "实现见 fake_path.py", "summary": ""},
        ]
        kept = auto_extract._filter_verified_items(items, str(tmp_path))
        assert [i["name"] for i in kept] == ["a"]

    def test_no_path_mentioned_kept(self, tmp_path):
        from agent import auto_extract
        items = [{"type": "user", "name": "c", "description": "无路径", "body": "用户偏好中文", "summary": ""}]
        assert len(auto_extract._filter_verified_items(items, str(tmp_path))) == 1
