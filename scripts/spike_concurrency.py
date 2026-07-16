"""Phase 4 并发 spike：验证跨进程文件操作是否可行。

测试 3 件事：
1. 多进程并发 append 到同一 JSONL 文件，是否会丢数据/交错
2. 用 msvcrt / fcntl 文件锁能否序列化
3. 「读取任务 → 标记认领」的 race 是否能原子化

用法：先跑 single-process 基线，再跑 multi-process 对比。
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from multiprocessing import Process, Manager


# ---------------------------------------------------------------------------
# 测试 1：无锁并发 append
# ---------------------------------------------------------------------------

def _append_no_lock(path: str, n: int, pid: int):
    """每个进程 append n 行到同一文件。"""
    with open(path, "a", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"pid": pid, "i": i}) + "\n")


def test_unsafe_concurrent_append(tmp_path: Path) -> tuple[int, int]:
    """5 进程并发各 append 100 行。返回 (写入总数, 实际行数)。

    无锁在 POSIX 上 O_APPEND 是原子的（小 buffer），Windows 上不一定。
    """
    path = tmp_path / "spike.jsonl"
    if path.exists():
        path.unlink()

    procs = []
    for pid in range(5):
        p = Process(target=_append_no_lock, args=(str(path), 100, pid))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()

    expected = 5 * 100
    actual = sum(1 for _ in path.open(encoding="utf-8"))
    return expected, actual


# ---------------------------------------------------------------------------
# 测试 2：有锁并发 append（Windows 用 msvcrt，POSIX 用 fcntl）
# ---------------------------------------------------------------------------

def _append_with_lock(path: str, n: int, pid: int):
    """每个进程 append n 行，用文件锁序列化。"""
    for i in range(n):
        with open(path, "a", encoding="utf-8") as f:
            _acquire_lock(f)
            try:
                f.write(json.dumps({"pid": pid, "i": i}) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                _release_lock(f)


def _acquire_lock(fileobj):
    """跨平台独占锁。"""
    if sys.platform == "win32":
        import msvcrt
        while True:
            try:
                msvcrt.locking(fileobj.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError:
                time.sleep(0.01)
    else:
        import fcntl
        fcntl.flock(fileobj.fileno(), fcntl.LOCK_EX)


def _release_lock(fileobj):
    if sys.platform == "win32":
        import msvcrt
        try:
            msvcrt.locking(fileobj.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        import fcntl
        fcntl.flock(fileobj.fileno(), fcntl.LOCK_UN)


def test_safe_concurrent_append(tmp_path: Path) -> tuple[int, int]:
    path = tmp_path / "spike_locked.jsonl"
    if path.exists():
        path.unlink()
    procs = []
    for pid in range(5):
        p = Process(target=_append_with_lock, args=(str(path), 100, pid))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
    expected = 5 * 100
    actual = sum(1 for _ in path.open(encoding="utf-8"))
    return expected, actual


# ---------------------------------------------------------------------------
# 测试 3：任务认领 race（read-modify-write 是否原子）
# ---------------------------------------------------------------------------

def _claim_task_race(tasks_file: str, pid: int, claimed: list, lock_file: str):
    """每个进程扫 tasks.json 找 pending，第一个尝试认领。"""
    time.sleep(0.05 * pid)  # stagger
    # 模拟 race：读 → 改 → 写，无锁
    with open(tasks_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    for t in tasks:
        if t["status"] == "pending":
            t["status"] = "in_progress"
            t["owner"] = f"agent_{pid}"
            claimed.append(t["id"])
            break
    time.sleep(0.01)  # 故意拉开 race 窗口
    with open(tasks_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f)


def test_task_claim_race(tmp_path: Path) -> tuple[int, int]:
    """5 个进程并发尝试认领同一 pending task。返回 (期望认领数, 实际认领数)。"""
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps([
        {"id": "t1", "status": "pending", "owner": None},
    ]), encoding="utf-8")

    with Manager() as mgr:
        claimed = mgr.list()
        procs = []
        for pid in range(5):
            p = Process(target=_claim_task_race,
                        args=(str(tasks_file), pid, claimed, str(tmp_path / "lock")))
            procs.append(p)
            p.start()
        for p in procs:
            p.join()
        return 1, len(claimed)


def _claim_task_safe(tasks_file: str, pid: int, claimed: list):
    """安全版：lock 文件 + re-read + 状态检查 + write。"""
    lock_path = tasks_file + ".lock"
    with open(lock_path, "w", encoding="utf-8") as lf:
        _acquire_lock(lf)
        try:
            with open(tasks_file, "r", encoding="utf-8") as f:
                tasks = json.load(f)
            for t in tasks:
                if t["status"] == "pending":
                    t["status"] = "in_progress"
                    t["owner"] = f"agent_{pid}"
                    claimed.append(t["id"])
                    break
            with open(tasks_file, "w", encoding="utf-8") as f:
                json.dump(tasks, f)
        finally:
            _release_lock(lf)


def test_task_claim_safe(tmp_path: Path) -> tuple[int, int]:
    tasks_file = str(tmp_path / "tasks_safe.json")
    Path(tasks_file).write_text(json.dumps([
        {"id": "t1", "status": "pending", "owner": None},
    ]), encoding="utf-8")
    Path(tasks_file + ".lock").touch()

    with Manager() as mgr:
        claimed = mgr.list()
        procs = []
        for pid in range(5):
            p = Process(target=_claim_task_safe, args=(tasks_file, pid, claimed))
            procs.append(p)
            p.start()
        for p in procs:
            p.join()
        return 1, len(claimed)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version_info[:2]}")
    print()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        print("=" * 60)
        print("测试 1: 无锁并发 append（预期可能丢/交错）")
        exp, act = test_unsafe_concurrent_append(tmp)
        status = "✓" if exp == act else "✗"
        print(f"  {status} 期望 {exp} 行，实际 {act} 行")
        print()

        print("=" * 60)
        print("测试 2: 有锁并发 append（预期正确）")
        exp, act = test_safe_concurrent_append(tmp)
        status = "✓" if exp == act else "✗"
        print(f"  {status} 期望 {exp} 行，实际 {act} 行")
        print()

        print("=" * 60)
        print("测试 3: 任务认领 race（无锁，预期多 agent 同时认领）")
        exp, act = test_task_claim_race(tmp)
        status = "✓" if act == 1 else f"race confirmed ({act} claimers)"
        print(f"  {status} 期望 {exp} 个认领者，实际 {act} 个")
        print()

        print("=" * 60)
        print("测试 4: 任务认领 安全（有锁，预期 1 个认领者）")
        exp, act = test_task_claim_safe(tmp)
        status = "✓" if act == 1 else f"lock failed ({act} claimers)"
        print(f"  {status} 期望 {exp} 个认领者，实际 {act} 个")
        print()

    print("=" * 60)
    print("Spike 结论：")
    print("- 测试 2 通过 → 文件锁可用于 JSONL inbox")
    print("- 测试 4 通过 → 文件锁可用于 task claim 原子化")
    print("- 若测试 1/3 也通过 → 平台已自带原子性（POSIX O_APPEND 可能如此）")


if __name__ == "__main__":
    main()
