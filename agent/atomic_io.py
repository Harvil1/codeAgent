"""原子写文件工具：写文件时读者永远不会看到「写了一半」的内容。

原理：先把新内容写到同目录的临时文件，全写好后再用 os.replace
「一键替换」到正式位置——替换在操作系统层面是一瞬间完成的，
不存在半旧半新的状态。被 memory_store / usage_tracker / settings
等「不能读到半截」的关键写入方使用。
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
    """安全地原子写入文本：先写临时文件，再用 os.replace 一键换过去。

    要点：
    - 临时文件必须建在目标同目录（同一块文件系统），os.replace 才是原子操作
    - 写完调 fsync 把数据真正刷到磁盘——万一程序或机器崩溃，恢复后数据还在
    - Windows 下目标正被别的线程/进程读时，os.replace 会抛 PermissionError
      （WinError 5），重试等读者放开句柄即可；Linux/Mac 文件开着也能 rename

    参数：
        path                 —— 目标文件路径（父目录不存在会自动创建）
        content              —— 要写入的文本内容
        encoding             —— 文件编码，默认 utf-8
        max_replace_retries  —— Windows 下 replace 被占读时的最大重试次数

    返回：无。失败（重试耗尽等）会抛异常，并清理掉临时文件。

    适用：低频关键写入（MEMORY.md / settings.json / config.yaml）。
    高频小写入请用 atomic_write_text_lite（不做 fsync、不重试，更快）。
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
        # Windows 下目标被读时 replace 会抛 PermissionError，短暂重试等读者松手
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
        # 重试用完还没成功，把最后一次的错误抛出去
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
    """原子写入的轻量版：只做「临时文件 + rename」，不 fsync、不重试。

    给高频小写入用（完整版每次 fsync + 重试循环太慢），牺牲一点
    可靠性换速度：
    - 不调 fsync（断电/崩溃恢复时可能丢最近一次写入，但快）
    - 不处理 Windows 的「目标被读导致 replace 失败」（调用方自己兜底，
      或者接受偶尔失败）

    参数：
        path     —— 目标文件路径
        content  —— 要写入的文本内容
        encoding —— 文件编码，默认 utf-8

    返回：无。出异常时会清理临时文件。

    适用：高频小写入（大输出落盘、子代理轨迹快照）。
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
        tmp_path.replace(path)  # 原子 rename，一瞬间完成新旧切换
    except BaseException:
        # replace 抛异常时把临时文件删掉，别在磁盘上留一堆 .tmp 垃圾
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
