"""跨平台的独占文件锁（Windows 用 msvcrt，Linux/macOS 用 fcntl）。

大白话：多个进程（REPL 主程序、团队 worker、手动跑的 curator CLI）都要
改同一批文件时，谁先拿到锁文件谁先写，其他人等着——不锁的话就是
"后写的覆盖先写的"，先写的那次更新无声丢失。

原来这份实现长在 agent/input_history.py 里（history.jsonl 专用），
现在 memory_store 也要跨进程互斥，提炼成公共件（行为零变化搬迁，
input_history 改为引用本模块）。

用法：
    with exclusive_file_lock(lock_path, timeout=5.0) as acquired:
        # acquired 为 True 表示真拿到锁了；False = 等到超时还没等到
        #（拿没拿到由调用方自己决定接下来怎么办：跳过、照写都行）
        ...

注意：yield 的是 bool 而不是抛异常——等锁超时是"常见可接受"的场景
（宁可继续裸写也不把主流程搞崩），调用方按需处理。
"""
import sys
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def exclusive_file_lock(lock_path: Path, timeout: float = 5.0):
    """跨平台独占文件锁（防多进程并发读写互相覆盖）。

    参数：
        lock_path：锁文件路径（父目录会自动创建）
        timeout：最多等多久（秒）

    用法：
        with _file_lock(lock_path) as locked:
            # locked 为 True 表示真拿到锁了，False 表示等到超时
            # 还没等到（拿没拿到由调用方自己决定接下来怎么办）
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lf:
        acquired = False
        deadline = time.time() + timeout
        if sys.platform == "win32":
            import msvcrt
            while time.time() < deadline:
                try:
                    msvcrt.locking(lf.fileno(), msvcrt.LK_LOCK, 1)
                    acquired = True
                    break
                except OSError:
                    time.sleep(0.01)
        else:
            import fcntl
            while time.time() < deadline:
                try:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    time.sleep(0.01)
        try:
            yield acquired
        finally:
            if acquired:
                try:
                    if sys.platform == "win32":
                        import msvcrt
                        msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
