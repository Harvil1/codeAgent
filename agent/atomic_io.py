"""原子文件写入工具。

保证读者永远看到完整旧版或完整新版，不会读到半截。

使用场景：Curator 后台线程重建 MEMORY.md 时，Agent 主线程可能同时读取索引
注入 system prompt。如果直接 write_text，可能读到只写了一半的内容，导致
LLM 拿到残缺索引。原子写相当于"后台写好新招牌再一键挂上去"。
"""
import os
import tempfile
import time
from pathlib import Path
from typing import Union


def atomic_write_text(
    path: Union[str, Path],
    content: str,
    *,
    encoding: str = "utf-8",
    max_replace_retries: int = 10,
) -> None:
    """原子写入文本：tempfile + os.replace。

    tempfile 必须在同目录（同文件系统），os.replace 才能原子生效。
    写入后 fsync 保证崩溃恢复时数据已落盘。

    Windows 限制：os.replace 在目标文件被其他线程/进程读取时会抛
    PermissionError（WinError 5）。这是 OS 行为，重试可恢复——读者
    很快释放句柄。POSIX 没有这个限制（rename 即使文件打开也成功）。
    失败时清理临时文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=".aw_"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        # Windows 下目标文件被读时 os.replace 抛 PermissionError，短暂重试
        last_err: Exception | None = None
        for attempt in range(max_replace_retries + 1):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError as e:
                last_err = e
                if attempt < max_replace_retries:
                    time.sleep(0.005 * (attempt + 1))  # 5ms, 10ms, 15ms, ...
                continue
        # 重试耗尽，抛最后的错误
        assert last_err is not None
        raise last_err
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
