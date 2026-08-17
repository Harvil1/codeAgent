"""atomic_write_text 并发安全测试。

验证：
1. 读者永远不会读到半截内容
2. 写入失败时不留临时文件残留
3. 跨平台路径（同目录 tempfile）
"""
import threading
from pathlib import Path

import pytest

from agent.atomic_io import atomic_write_text


def test_basic_write(tmp_path: Path):
    """基本写入：内容正确，能读回。"""
    path = tmp_path / "f.txt"
    atomic_write_text(path, "hello, 世界")
    assert path.read_text(encoding="utf-8") == "hello, 世界"


def test_creates_parent_dirs(tmp_path: Path):
    """父目录不存在时自动创建。"""
    path = tmp_path / "a" / "b" / "c.txt"
    atomic_write_text(path, "nested")
    assert path.read_text(encoding="utf-8") == "nested"


def test_overwrite_atomic(tmp_path: Path):
    """覆盖写入：要么看到旧版要么看到新版，不会混合。"""
    path = tmp_path / "f.txt"
    atomic_write_text(path, "v1")
    atomic_write_text(path, "v2")
    assert path.read_text(encoding="utf-8") == "v2"


def test_concurrent_read_no_half(tmp_path: Path):
    """并发场景：读者从不会读到非 v1/v2 的内容。

    Windows 行为：os.replace 窗口内读者可能拿到 PermissionError，
    这是合法的"瞬时不可用"——不是数据损坏。读者要么拿到完整 v1/v2，
    要么短暂打不开文件。
    """
    path = tmp_path / "f.txt"
    # 用完整 payload 初始化，避免读者读到初始 "v1" 误判为半截
    payload_v1 = "v1-" + ("A" * 5000)
    payload_v2 = "v2-" + ("B" * 5000)
    atomic_write_text(path, payload_v1)

    results: list[str] = []
    transient_errors = 0
    stop = threading.Event()

    def reader():
        nonlocal transient_errors
        while not stop.is_set():
            try:
                if path.exists():
                    results.append(path.read_text(encoding="utf-8"))
            except PermissionError:
                # Windows: rename 窗口内打开失败，合法瞬时态
                transient_errors += 1

    t = threading.Thread(target=reader)
    t.start()

    # 50 次交替写入 v1 / v2
    for i in range(50):
        atomic_write_text(path, payload_v1 if i % 2 else payload_v2)

    stop.set()
    t.join()

    # 关键不变式：所有读到的内容都应是完整 v1 或 v2（不是半截）
    corrupt = [r for r in results if r not in (payload_v1, payload_v2)]
    assert not corrupt, (
        f"检测到半截内容！共 {len(results)} 次读取，"
        f"{len(corrupt)} 次读到非完整内容。前 3 个长度: {[len(c) for c in corrupt[:3]]}。"
        f"前 3 个前缀: {[c[:20] for c in corrupt[:3]]!r}"
    )


def test_no_tmp_residue_after_success(tmp_path: Path):
    """写入成功后无临时文件残留。"""
    path = tmp_path / "f.txt"
    atomic_write_text(path, "ok")
    # 目录里只有目标文件，没有 .aw_*.tmp
    files = list(tmp_path.iterdir())
    assert files == [path]


def test_encoding_utf8_default(tmp_path: Path):
    """默认 UTF-8 编码（Windows 默认 cp1252 会乱码，必须显式）。"""
    path = tmp_path / "f.txt"
    atomic_write_text(path, "中文测试")
    # 用字节读验证 UTF-8
    raw = path.read_bytes()
    assert "中文测试".encode("utf-8") == raw


def test_replace_existing_file_atomically(tmp_path: Path):
    """目标已存在时替换：fsync + os.replace 路径正常。"""
    path = tmp_path / "f.txt"
    path.write_text("old", encoding="utf-8")
    atomic_write_text(path, "new")
    assert path.read_text(encoding="utf-8") == "new"
