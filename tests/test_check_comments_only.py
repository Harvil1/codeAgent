"""tests/test_check_comments_only.py——「只改注释」检查脚本的测试。"""

from scripts.check_comments_only import (
    check_file,
    expand_paths,
    strip_comments_and_docstrings,
)


def test_加行内注释后代码不变():
    old = "def add(a, b):\n    return a + b\n"
    new = "def add(a, b):\n    # 两数相加\n    return a + b\n"
    assert strip_comments_and_docstrings(old) == strip_comments_and_docstrings(new)


def test_改docstring后代码不变():
    old = 'def add(a, b):\n    """旧说明"""\n    return a + b\n'
    new = 'def add(a, b):\n    """新说明（大白话版）"""\n    return a + b\n'
    assert strip_comments_and_docstrings(old) == strip_comments_and_docstrings(new)


def test_删注释后代码不变():
    old = "import os  # 操作系统\n\n\ndef f():\n    return 1\n"
    new = "import os\n\n\ndef f():\n    return 1\n"
    assert strip_comments_and_docstrings(old) == strip_comments_and_docstrings(new)


def test_改代码会被发现():
    old = "def add(a, b):\n    return a + b\n"
    new = "def add(a, b):\n    return a - b\n"
    assert strip_comments_and_docstrings(old) != strip_comments_and_docstrings(new)


def test_改字符串字面量会被发现():
    old = 'x = "hello"\n'
    new = 'x = "world"\n'
    assert strip_comments_and_docstrings(old) != strip_comments_and_docstrings(new)


def test_模块类方法三层docstring都剥掉():
    src = (
        '"""模块说明"""\n\n'
        "class A:\n"
        '    """类说明"""\n\n'
        "    def m(self):\n"
        '        """方法说明"""\n'
        "        return 1\n"
    )
    out = strip_comments_and_docstrings(src)
    assert "模块说明" not in out
    assert "类说明" not in out
    assert "方法说明" not in out
    assert "return 1" in out


def test_函数体只有docstring时换成pass占位():
    src = 'def f():\n    """只有说明"""\n'
    out = strip_comments_and_docstrings(src)
    assert "pass" in out


def test_普通字符串不是docstring不剥():
    src = 'x = """长得像说明的字符串"""\n'
    out = strip_comments_and_docstrings(src)
    assert "长得像说明的字符串" in out


def test_目录展开成py文件列表(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.py").write_text("y = 2\n", encoding="utf-8")
    (sub / "c.txt").write_text("不是 py", encoding="utf-8")
    result = expand_paths([str(tmp_path)])
    names = [p.replace("\\", "/").split("/")[-1] for p in result]
    assert names == ["a.py", "b.py"]


def test_check_file_只改注释通过(tmp_path, monkeypatch):
    target = tmp_path / "mod.py"
    old = '"""旧说明"""\n\n\ndef f(x):\n    # 注释\n    return x\n'
    new = '"""新说明（大白话）"""\n\n\ndef f(x):\n    # 把输入原样交回\n    return x\n'
    target.write_text(new, encoding="utf-8")
    monkeypatch.setattr(
        "scripts.check_comments_only.load_head_version", lambda _p: old
    )
    assert check_file(str(target)) == []


def test_check_file_改了代码报问题(tmp_path, monkeypatch):
    target = tmp_path / "mod.py"
    old = "def f(x):\n    return x\n"
    new = "def f(x):\n    return x + 1\n"
    target.write_text(new, encoding="utf-8")
    monkeypatch.setattr(
        "scripts.check_comments_only.load_head_version", lambda _p: old
    )
    problems = check_file(str(target))
    assert problems and "代码被改动" in problems[0]


def test_check_file_改坏语法报问题(tmp_path, monkeypatch):
    target = tmp_path / "mod.py"
    target.write_text("def f(:\n", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.check_comments_only.load_head_version", lambda _p: "def f():\n    pass\n"
    )
    problems = check_file(str(target))
    assert problems and "解析失败" in problems[0]
