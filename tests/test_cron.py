"""CronScheduler 测试。"""
import json
import time
from datetime import datetime
from pathlib import Path

import pytest

from agent.cron import CronScheduler, CronJob


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


# ============================================================================
# Task 1: CronJob 字段扩展 + _parse_job 兼容 + _persist_jobs_unlocked
# ============================================================================

def test_parse_job_fills_created_at_when_missing():
    """老格式 jobs.json 缺 created_at 时，_parse_job 补 now。"""
    from agent.cron import CronScheduler
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sched = CronScheduler(jobs_path=Path(tmp) / "jobs.json", enabled=False)
        job = sched._parse_job({
            "id": "j1", "cron": "* * * * *", "message": "hi",
        })
        assert job is not None
        assert job.created_at, "created_at 不能为空"
        # 验证是合法 ISO 时间戳
        from datetime import datetime
        datetime.fromisoformat(job.created_at)  # 抛异常则测试失败


def test_parse_job_fills_recurring_default_true():
    """缺 recurring 时默认 True（既有行为不变）。"""
    from agent.cron import CronScheduler
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sched = CronScheduler(jobs_path=Path(tmp) / "jobs.json", enabled=False)
        job = sched._parse_job({
            "id": "j1", "cron": "* * * * *", "message": "hi",
        })
        assert job is not None
        assert job.recurring is True


def test_parse_job_reads_recurring_false():
    """recurring=False 能正确解析。"""
    from agent.cron import CronScheduler
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sched = CronScheduler(jobs_path=Path(tmp) / "jobs.json", enabled=False)
        job = sched._parse_job({
            "id": "j1", "cron": "* * * * *", "message": "hi",
            "recurring": False,
        })
        assert job is not None
        assert job.recurring is False


def test_parse_job_reads_explicit_created_at():
    """有 created_at 时原样保留。"""
    from agent.cron import CronScheduler
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sched = CronScheduler(jobs_path=Path(tmp) / "jobs.json", enabled=False)
        job = sched._parse_job({
            "id": "j1", "cron": "* * * * *", "message": "hi",
            "created_at": "2026-01-01T00:00:00",
        })
        assert job is not None
        assert job.created_at == "2026-01-01T00:00:00"


def test_persist_jobs_unlocked_writes_json():
    """_persist_jobs_unlocked 把 _jobs 写到 jobs.json。"""
    from agent.cron import CronScheduler, CronJob
    from pathlib import Path
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        jobs_path = Path(tmp) / "jobs.json"
        sched = CronScheduler(jobs_path=jobs_path, enabled=False)
        sched._jobs = [
            CronJob(
                id="j1", cron="* * * * *", message="hi",
                created_at="2026-01-01T00:00:00", recurring=False,
            ),
        ]
        with sched._lock:
            sched._persist_jobs_unlocked()

        assert jobs_path.exists(), "jobs.json 未创建"
        data = json.loads(jobs_path.read_text(encoding="utf-8"))
        assert "jobs" in data
        assert len(data["jobs"]) == 1
        j = data["jobs"][0]
        assert j["id"] == "j1"
        assert j["created_at"] == "2026-01-01T00:00:00"
        assert j["recurring"] is False


def test_persist_jobs_unlocked_atomic_replace():
    """持久化用 tmp + os.replace（不会留半写文件）。"""
    from agent.cron import CronScheduler, CronJob
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        jobs_path = Path(tmp) / "jobs.json"
        sched = CronScheduler(jobs_path=jobs_path, enabled=False)
        sched._jobs = [CronJob(id="j1", cron="* * * * *", message="x")]
        with sched._lock:
            sched._persist_jobs_unlocked()
        # 不应有 .tmp 残留
        tmp_files = list(Path(tmp).glob("*.tmp"))
        assert not tmp_files, f"残留临时文件: {tmp_files}"


# ============================================================================
# Task 2: _tick 加 7 天过期 + 一次性 disable
# ============================================================================

def test_tick_expires_job_over_7_days():
    """created_at 超 7 天的 job 被 disable，不入 notifications。"""
    from agent.cron import CronScheduler, CronJob
    from pathlib import Path
    from datetime import datetime, timedelta
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sched = CronScheduler(jobs_path=Path(tmp) / "jobs.json", enabled=False)
        eight_days_ago = (datetime.now() - timedelta(days=8)).isoformat(timespec="seconds")
        sched._jobs = [
            CronJob(
                id="old", cron="* * * * *", message="expired",
                created_at=eight_days_ago,
            ),
        ]
        sched._tick(datetime.now())
        assert sched._jobs[0].enabled is False, "过期 job 应被 disable"
        assert not sched._notifications, "过期 job 不应入 notifications"


def test_tick_keeps_job_under_7_days():
    """created_at 3 天前的 job 仍可正常触发。"""
    from agent.cron import CronScheduler, CronJob
    from pathlib import Path
    from datetime import datetime, timedelta
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sched = CronScheduler(jobs_path=Path(tmp) / "jobs.json", enabled=False)
        three_days_ago = (datetime.now() - timedelta(days=3)).isoformat(timespec="seconds")
        sched._jobs = [
            CronJob(
                id="recent", cron="* * * * *", message="ok",
                created_at=three_days_ago,
            ),
        ]
        sched._tick(datetime.now())
        assert sched._jobs[0].enabled is True, "3 天的 job 不应被 disable"
        assert len(sched._notifications) == 1
        assert sched._notifications[0]["job_id"] == "recent"


def test_tick_disables_non_recurring_after_fire():
    """recurring=False 触发后 enabled=False。"""
    from agent.cron import CronScheduler, CronJob
    from pathlib import Path
    from datetime import datetime
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sched = CronScheduler(jobs_path=Path(tmp) / "jobs.json", enabled=False)
        sched._jobs = [
            CronJob(
                id="once", cron="* * * * *", message="one-shot",
                recurring=False,
            ),
        ]
        sched._tick(datetime.now())
        assert sched._jobs[0].enabled is False, "一次性 job 触发后应 disable"
        assert len(sched._notifications) == 1


def test_tick_keeps_recurring_after_fire():
    """recurring=True 触发后仍 enabled。"""
    from agent.cron import CronScheduler, CronJob
    from pathlib import Path
    from datetime import datetime
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sched = CronScheduler(jobs_path=Path(tmp) / "jobs.json", enabled=False)
        sched._jobs = [
            CronJob(
                id="repeat", cron="* * * * *", message="daily",
                recurring=True,
            ),
        ]
        sched._tick(datetime.now())
        assert sched._jobs[0].enabled is True, "recurring=True 不应 disable"
        assert len(sched._notifications) == 1


def test_tick_respects_max_age_days_param():
    """max_age_days 参数生效（=1 时 2 天前的 job 过期）。"""
    from agent.cron import CronScheduler, CronJob
    from pathlib import Path
    from datetime import datetime, timedelta
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sched = CronScheduler(jobs_path=Path(tmp) / "jobs.json", enabled=False)
        two_days_ago = (datetime.now() - timedelta(days=2)).isoformat(timespec="seconds")
        sched._jobs = [
            CronJob(
                id="j", cron="* * * * *", message="x",
                created_at=two_days_ago,
            ),
        ]
        sched._tick(datetime.now(), max_age_days=1)
        assert sched._jobs[0].enabled is False, "max_age_days=1 时 2 天前应过期"


def test_tick_handles_bad_created_at_gracefully():
    """created_at 格式坏时不阻塞 tick（job 照常触发）。"""
    from agent.cron import CronScheduler, CronJob
    from pathlib import Path
    from datetime import datetime
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sched = CronScheduler(jobs_path=Path(tmp) / "jobs.json", enabled=False)
        sched._jobs = [
            CronJob(
                id="bad", cron="* * * * *", message="x",
                created_at="not-a-date",
            ),
        ]
        sched._tick(datetime.now())
        # 不应抛异常，job 照常触发
        assert len(sched._notifications) == 1
        assert sched._jobs[0].enabled is True


def test_tick_persists_disable_state():
    """disable 后 jobs.json 反映新状态。"""
    from agent.cron import CronScheduler, CronJob
    from pathlib import Path
    from datetime import datetime, timedelta
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        jobs_path = Path(tmp) / "jobs.json"
        sched = CronScheduler(jobs_path=jobs_path, enabled=False)
        eight_days_ago = (datetime.now() - timedelta(days=8)).isoformat(timespec="seconds")
        sched._jobs = [
            CronJob(id="old", cron="* * * * *", message="x", created_at=eight_days_ago),
        ]
        sched._tick(datetime.now())

        # 重读 jobs.json 验证持久化
        data = json.loads(jobs_path.read_text(encoding="utf-8"))
        assert data["jobs"][0]["enabled"] is False, "jobs.json 应反映 disabled 状态"


def test_tick_expired_takes_precedence_over_cron_match():
    """过期检查先于 cron_match（边界：既过期又匹配时优先 disable）。"""
    from agent.cron import CronScheduler, CronJob
    from pathlib import Path
    from datetime import datetime, timedelta
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sched = CronScheduler(jobs_path=Path(tmp) / "jobs.json", enabled=False)
        eight_days_ago = (datetime.now() - timedelta(days=8)).isoformat(timespec="seconds")
        sched._jobs = [
            CronJob(
                id="old", cron="* * * * *", message="x",
                created_at=eight_days_ago,
            ),
        ]
        sched._tick(datetime.now())
        # 过期优先，不应入 notifications
        assert not sched._notifications
        assert sched._jobs[0].enabled is False


# ============================================================================
# Task 3: config + __init__ 参数
# ============================================================================

def test_default_config_has_cron_max_age_days():
    """DEFAULT_CONFIG 含 cron.max_age_days，默认 7。"""
    from config import DEFAULT_CONFIG
    assert "cron" in DEFAULT_CONFIG
    assert DEFAULT_CONFIG["cron"].get("max_age_days") == 7


def test_scheduler_init_accepts_max_age_days():
    """CronScheduler 构造可传 max_age_days，存到 _max_age_days。"""
    from agent.cron import CronScheduler
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sched = CronScheduler(
            jobs_path=Path(tmp) / "jobs.json",
            enabled=False,
            max_age_days=30,
        )
        assert sched._max_age_days == 30


def test_scheduler_init_defaults_max_age_days_to_7():
    """不传 max_age_days 时默认 7。"""
    from agent.cron import CronScheduler
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sched = CronScheduler(
            jobs_path=Path(tmp) / "jobs.json",
            enabled=False,
        )
        assert sched._max_age_days == 7
