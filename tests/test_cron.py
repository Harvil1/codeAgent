"""CronScheduler 测试。"""
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from agent.cron import CronScheduler, CronJob
from agent.cron_parser import cron_match


def _write_jobs(path: Path, jobs: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"jobs": jobs}, ensure_ascii=False), encoding="utf-8")


def test_load_jobs_valid(tmp_path: Path):
    p = tmp_path / "jobs.json"
    _write_jobs(p, [
        {"id": "j1", "cron": "*/5 * * * *", "message": "hello"},
        {"id": "j2", "cron": "0 9 * * *", "message": "9am", "enabled": False},
    ])
    sched = CronScheduler(jobs_path=p, enabled=False)
    assert len(sched._jobs) == 2
    assert sched._jobs[0].id == "j1"
    assert sched._jobs[1].enabled is False


def test_load_jobs_missing_file_starts_empty(tmp_path: Path):
    """文件不存在不抛，jobs 为空。"""
    sched = CronScheduler(jobs_path=tmp_path / "nonexistent.json", enabled=False)
    assert sched._jobs == []


def test_load_jobs_malformed_raises(tmp_path: Path):
    p = tmp_path / "jobs.json"
    p.write_text("not a json {{{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        CronScheduler(jobs_path=p, enabled=False)


def test_load_jobs_skips_invalid_entries(tmp_path: Path, caplog):
    """缺关键字段的 job 跳过 + warning。"""
    p = tmp_path / "jobs.json"
    _write_jobs(p, [
        {"id": "good", "cron": "* * * * *", "message": "ok"},
        {"id": "no_cron", "message": "misses cron"},
        {"no_message": "x"},
    ])
    sched = CronScheduler(jobs_path=p, enabled=False)
    assert len(sched._jobs) == 1
    assert sched._jobs[0].id == "good"


def test_drain_due_returns_and_clears(tmp_path: Path):
    sched = CronScheduler(jobs_path=tmp_path / "x.json", enabled=False)
    sched._notifications.append({"job_id": "test", "message": "hi"})
    result = sched.drain_due()
    assert len(result) == 1
    assert result[0]["job_id"] == "test"
    assert sched.drain_due() == []


def test_start_stop_idempotent(tmp_path: Path):
    sched = CronScheduler(jobs_path=tmp_path / "x.json", enabled=False)
    sched.start()
    sched.start()  # 幂等
    assert sched._thread is not None
    sched.stop()
    sched.stop()  # 幂等


def test_tick_fires_matching_job(tmp_path: Path):
    """manual tick 一个匹配的 job，notification 被 push。"""
    sched = CronScheduler(jobs_path=tmp_path / "x.json", enabled=False)
    sched._jobs = [CronJob(id="j1", cron="* * * * *", message="hi")]
    sched._tick(datetime(2026, 7, 12, 15, 30))
    notifs = sched.drain_due()
    assert len(notifs) == 1
    assert notifs[0]["job_id"] == "j1"
    assert notifs[0]["message"] == "hi"


def test_tick_skips_disabled_job(tmp_path: Path):
    sched = CronScheduler(jobs_path=tmp_path / "x.json", enabled=False)
    sched._jobs = [CronJob(id="j1", cron="* * * * *", message="hi", enabled=False)]
    sched._tick(datetime(2026, 7, 12, 15, 30))
    assert sched.drain_due() == []


def test_tick_dedupes_within_same_minute(tmp_path: Path):
    """同一分钟内多次 tick 只触发一次。"""
    sched = CronScheduler(jobs_path=tmp_path / "x.json", enabled=False)
    sched._jobs = [CronJob(id="j1", cron="* * * * *", message="hi")]
    sched._tick(datetime(2026, 7, 12, 15, 30))
    sched._tick(datetime(2026, 7, 12, 15, 30))
    notifs = sched.drain_due()
    assert len(notifs) == 1


def test_tick_fires_again_in_next_minute(tmp_path: Path):
    sched = CronScheduler(jobs_path=tmp_path / "x.json", enabled=False)
    sched._jobs = [CronJob(id="j1", cron="* * * * *", message="hi")]
    sched._tick(datetime(2026, 7, 12, 15, 30))
    sched._tick(datetime(2026, 7, 12, 15, 31))
    notifs = sched.drain_due()
    assert len(notifs) == 2


def test_reload_replaces_jobs(tmp_path: Path):
    p = tmp_path / "jobs.json"
    _write_jobs(p, [{"id": "j1", "cron": "* * * * *", "message": "old"}])
    sched = CronScheduler(jobs_path=p, enabled=False)
    assert len(sched._jobs) == 1
    _write_jobs(p, [
        {"id": "j1", "cron": "* * * * *", "message": "new"},
        {"id": "j2", "cron": "0 0 * * *", "message": "daily"},
    ])
    sched.reload(p)
    assert len(sched._jobs) == 2
    assert sched._jobs[0].message == "new"


def test_daemon_thread_starts_and_stops(tmp_path: Path):
    """集成：start + sleep + stop 不抛。"""
    sched = CronScheduler(
        jobs_path=tmp_path / "x.json",
        enabled=False,
        poll_interval_seconds=0.2,
    )
    sched.start()
    time.sleep(0.5)
    sched.stop()
    # thread 应已结束
    assert not sched._thread.is_alive()
