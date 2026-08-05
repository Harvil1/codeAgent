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
from typing import Optional, Union


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

    适用场景：低频关键写入（MEMORY.md / settings.json / config.yaml）。
    高频小写入请用 atomic_write_text_lite（无 fsync、无重试，更快）。
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


def atomic_write_text_lite(
    path: Union[str, Path],
    content: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """原子写入（轻量版）：tempfile + Path.replace，无 fsync、无重试。

    与 atomic_write_text 的差异：
    - 不调 fsync（崩溃恢复可能丢最近一次写入，但性能更好）
    - 不重试 Windows PermissionError（调用方需自行处理或容忍偶发失败）

    适用场景：高频小写入（offload 大输出落盘、transcript 快照）。
    异常路径下（replace 失败、权限拒绝等）清理临时文件，避免 .tmp 垃圾堆积。
    """
    path = Path(path)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            encoding=encoding,
            delete=False,
            suffix=".tmp",
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)  # 原子 rename
    except BaseException:
        # replace 抛异常时清理临时文件，避免磁盘上留下 *.tmp 垃圾
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
