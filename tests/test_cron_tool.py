"""cron_tool 测试：LLM 可自主创建/列出/删除定时任务。

覆盖四层：
1. 工具 handler 行为（mock scheduler：创建/列表/删除/非法表达式 + not_configured）
2. handler dispatch 契约（args, **kwargs，防 silent-dead-code）
3. schema 键契约（OpenAI "parameters"）
4. 真实 CronScheduler 的 add_job/remove_job/list_jobs 方法（工具包装的底层）
"""
import inspect
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.cron import CronScheduler, CronJob
from agent.cron_parser import cron_match
import tools.cron_tool  # noqa: 触发注册
from tools.cron_tool import (
    _handle_cron_create,
    _handle_cron_list,
    _handle_cron_delete,
    CRON_CREATE_SCHEMA,
    CRON_LIST_SCHEMA,
    CRON_DELETE_SCHEMA,
)
from tools.registry import registry


def _make_agent_ref(scheduler=None):
    """构造带 cron_scheduler 属性的假 agent_ref。"""
    ref = MagicMock()
    ref.cron_scheduler = scheduler
    return ref


# ---------------------------------------------------------------------------
# 1. handler 行为（mock scheduler）
# ---------------------------------------------------------------------------

class TestCronCreate:
    def test_create_ok(self):
        """合法表达式 → 调 scheduler.add_job + 返回 job_id。"""
        sched = MagicMock()
        sched.add_job.return_value = CronJob(
            id="job_abc", cron="*/5 * * * *", message="提醒喝水",
        )
        result = _handle_cron_create(
            {"cron": "*/5 * * * *", "message": "提醒喝水", "catch_up": True},
            agent_ref=_make_agent_ref(sched),
        )
        data = json.loads(result)
        assert data["job_id"] == "job_abc"
        sched.add_job.assert_called_once_with(
            "*/5 * * * *", "提醒喝水", catch_up=True, recurring=True,
        )

    def test_create_invalid_expr(self):
        """非法表达式 → invalid_cron_expr 错误，不调 add_job。"""
        sched = MagicMock()
        result = _handle_cron_create(
            {"cron": "99 * * * *", "message": "x"},  # minute 超范围
            agent_ref=_make_agent_ref(sched),
        )
        data = json.loads(result)
        assert data["error_type"] == "invalid_cron_expr"
        sched.add_job.assert_not_called()

    def test_create_missing_args(self):
        """缺 cron 或 message → invalid_args。"""
        result = _handle_cron_create(
            {"message": "只有消息"}, agent_ref=_make_agent_ref(MagicMock())
        )
        data = json.loads(result)
        assert data["error_type"] == "invalid_args"

    def test_create_not_configured(self):
        """cron_scheduler 为 None（配置关闭/启动失败）→ not_configured。"""
        result = _handle_cron_create(
            {"cron": "* * * * *", "message": "x"},
            agent_ref=_make_agent_ref(None),
        )
        data = json.loads(result)
        assert data["error_type"] == "not_configured"

    def test_create_no_agent_ref(self):
        """dispatch_kwargs 没有 agent_ref → not_configured（fail-open 不抛）。"""
        result = _handle_cron_create({"cron": "* * * * *", "message": "x"})
        data = json.loads(result)
        assert data["error_type"] == "not_configured"


class TestCronList:
    def test_list_ok(self):
        sched = MagicMock()
        sched.list_jobs.return_value = [
            {"id": "j1", "cron": "*/5 * * * *", "message": "m1", "enabled": True},
            {"id": "j2", "cron": "0 9 * * *", "message": "m2", "enabled": False},
        ]
        result = _handle_cron_list({}, agent_ref=_make_agent_ref(sched))
        data = json.loads(result)
        assert data["count"] == 2
        assert data["jobs"][0]["id"] == "j1"
        assert data["jobs"][1]["enabled"] is False

    def test_list_empty(self):
        sched = MagicMock()
        sched.list_jobs.return_value = []
        result = _handle_cron_list({}, agent_ref=_make_agent_ref(sched))
        data = json.loads(result)
        assert data["count"] == 0
        assert data["jobs"] == []

    def test_list_not_configured(self):
        result = _handle_cron_list({}, agent_ref=_make_agent_ref(None))
        data = json.loads(result)
        assert data["error_type"] == "not_configured"


class TestCronDelete:
    def test_delete_ok(self):
        sched = MagicMock()
        sched.remove_job.return_value = True
        result = _handle_cron_delete(
            {"job_id": "j1"}, agent_ref=_make_agent_ref(sched)
        )
        data = json.loads(result)
        assert data["deleted"] is True
        sched.remove_job.assert_called_once_with("j1")

    def test_delete_not_found(self):
        sched = MagicMock()
        sched.remove_job.return_value = False
        result = _handle_cron_delete(
            {"job_id": "nope"}, agent_ref=_make_agent_ref(sched)
        )
        data = json.loads(result)
        assert data["deleted"] is False
        assert "error" in data

    def test_delete_missing_job_id(self):
        result = _handle_cron_delete({}, agent_ref=_make_agent_ref(MagicMock()))
        data = json.loads(result)
        assert data["error_type"] == "invalid_args"

    def test_delete_not_configured(self):
        result = _handle_cron_delete(
            {"job_id": "j1"}, agent_ref=_make_agent_ref(None)
        )
        data = json.loads(result)
        assert data["error_type"] == "not_configured"


# ---------------------------------------------------------------------------
# 2. dispatch 契约（handler 必须是 (args, **kwargs)）
# ---------------------------------------------------------------------------

def test_handler_signature_matches_dispatch_contract():
    """dispatch 调 handler(args, **kwargs)，三个 handler 签名必须兼容。"""
    for handler in (_handle_cron_create, _handle_cron_list, _handle_cron_delete):
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())
        assert params[0].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert params[0].name == "args"
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)


# ---------------------------------------------------------------------------
# 3. schema 键契约（OpenAI "parameters" 不是 "inputSchema"）
# ---------------------------------------------------------------------------

def test_schemas_use_openai_parameters_key():
    for schema in (CRON_CREATE_SCHEMA, CRON_LIST_SCHEMA, CRON_DELETE_SCHEMA):
        assert "parameters" in schema, f"{schema.get('name')} 缺 parameters 键"
        assert "inputSchema" not in schema


def test_create_schema_required_fields():
    props = CRON_CREATE_SCHEMA["parameters"]["properties"]
    assert set(props) == {"cron", "message", "catch_up", "recurring", "template"}
    # template 模式下 cron/message 可省略（取模板值），不硬性 required
    assert not CRON_CREATE_SCHEMA["parameters"].get("required")
    assert props["catch_up"]["default"] is False


# ---------------------------------------------------------------------------
# 4. registry 注册 + 分类
# ---------------------------------------------------------------------------

def test_cron_tools_registered_in_core():
    for name in ("cron_create", "cron_list", "cron_delete"):
        entry = registry.get(name)
        assert entry is not None, f"{name} 未注册"
        assert entry.toolset == "core"
        assert entry.isConcurrencySafe is False  # 定时任务有副作用，串行


def test_cron_tools_in_core_toolset_visible():
    """三工具必须进 _CORE_TOOLS 才对 LLM 可见（发现 ≠ 可见）。"""
    from toolsets import resolve_toolset
    core = resolve_toolset("core")
    for name in ("cron_create", "cron_list", "cron_delete"):
        assert name in core, f"{name} 不在 core 工具集"


def test_cron_tools_disabled_for_async_subagent():
    """async 子代理不应注册/删除定时任务（对齐 ASYNC_AGENT_DISALLOWED_TOOLS 既有注释意图）。"""
    from toolsets import ASYNC_AGENT_DISALLOWED_TOOLS
    assert "cron_create" in ASYNC_AGENT_DISALLOWED_TOOLS
    assert "cron_delete" in ASYNC_AGENT_DISALLOWED_TOOLS


# ---------------------------------------------------------------------------
# 5. 真实 CronScheduler 的 add_job / remove_job / list_jobs
# ---------------------------------------------------------------------------

class TestCronSchedulerCrud:
    def test_add_job_generates_id_and_persists(self, tmp_path: Path):
        p = tmp_path / "jobs.json"
        sched = CronScheduler(jobs_path=p, enabled=False)
        job = sched.add_job("*/5 * * * *", "提醒喝水", catch_up=True)
        assert job.id  # 自动生成
        assert job.catch_up is True
        assert job.enabled is True
        assert job.recurring is True
        assert job.created_at  # ISO 时间戳自动填
        # 持久化到磁盘
        data = json.loads(p.read_text(encoding="utf-8"))
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["id"] == job.id
        assert data["jobs"][0]["catch_up"] is True

    def test_add_job_invalid_expr_raises(self, tmp_path: Path):
        sched = CronScheduler(jobs_path=tmp_path / "jobs.json", enabled=False)
        with pytest.raises(ValueError):
            sched.add_job("not a cron", "x")

    def test_add_job_duplicate_id_raises(self, tmp_path: Path):
        sched = CronScheduler(jobs_path=tmp_path / "jobs.json", enabled=False)
        sched.add_job("* * * * *", "a", job_id="dup")
        with pytest.raises(ValueError):
            sched.add_job("0 9 * * *", "b", job_id="dup")

    def test_list_jobs(self, tmp_path: Path):
        sched = CronScheduler(jobs_path=tmp_path / "jobs.json", enabled=False)
        j1 = sched.add_job("*/5 * * * *", "m1")
        j2 = sched.add_job("0 9 * * *", "m2")
        jobs = sched.list_jobs()
        assert {j["id"] for j in jobs} == {j1.id, j2.id}
        # 字段齐（工具契约要求 id/cron/message/enabled）
        for j in jobs:
            assert {"id", "cron", "message", "enabled"} <= set(j)

    def test_remove_job(self, tmp_path: Path):
        p = tmp_path / "jobs.json"
        sched = CronScheduler(jobs_path=p, enabled=False)
        j1 = sched.add_job("* * * * *", "a")
        assert sched.remove_job(j1.id) is True
        # 删除后落盘 + 列表为空
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["jobs"] == []
        assert sched.list_jobs() == []
        # 再删（不存在）→ False
        assert sched.remove_job("nope") is False

    def test_removed_job_stops_firing(self, tmp_path: Path):
        """删掉的 job 不再触发（回归保护）。"""
        sched = CronScheduler(jobs_path=tmp_path / "jobs.json", enabled=False)
        j1 = sched.add_job("* * * * *", "a")
        sched.remove_job(j1.id)
        sched._tick(datetime.now())
        assert sched.drain_due() == []


# ---------------------------------------------------------------------------
# 6. 模板/显式 recurring 透传（模板的 recurring: false 必须透传到
#    add_job，否则一次性任务实际永久循环）
# ---------------------------------------------------------------------------

class TestCronTemplateRecurring:
    def test_template_recurring_false_passed(self, tmp_path, monkeypatch):
        """模板 recurring: false → add_job 收到 recurring=False。"""
        from agent import templates as T
        home = tmp_path / "home"
        (home / "templates").mkdir(parents=True)
        (home / "templates" / "once.md").write_text(
            "---\ncron: \"30 14 28 2 *\"\nrecurring: false\n---\n一次性提醒",
            encoding="utf-8",
        )
        monkeypatch.setenv("OMNIMATE_HOME", str(home))
        monkeypatch.setattr(
            "agent.workspace_context.get_workspace_cwd", lambda: str(tmp_path)
        )
        T._invalidate_cache()

        sched = MagicMock()
        sched.add_job.return_value = CronJob(
            id="job_once", cron="30 14 28 2 *", message="一次性提醒",
        )
        result = _handle_cron_create(
            {"template": "once"}, agent_ref=_make_agent_ref(sched),
        )
        data = json.loads(result)
        assert data["job_id"] == "job_once"
        sched.add_job.assert_called_once_with(
            "30 14 28 2 *", "一次性提醒", catch_up=False, recurring=False,
        )

    def test_explicit_recurring_overrides_template(self, tmp_path, monkeypatch):
        """显式 recurring=true 覆盖模板 false。"""
        from agent import templates as T
        home = tmp_path / "home"
        (home / "templates").mkdir(parents=True)
        (home / "templates" / "once.md").write_text(
            "---\ncron: \"30 14 28 2 *\"\nrecurring: false\n---\n一次性提醒",
            encoding="utf-8",
        )
        monkeypatch.setenv("OMNIMATE_HOME", str(home))
        monkeypatch.setattr(
            "agent.workspace_context.get_workspace_cwd", lambda: str(tmp_path)
        )
        T._invalidate_cache()

        sched = MagicMock()
        sched.add_job.return_value = CronJob(
            id="job_force", cron="30 14 28 2 *", message="一次性提醒",
        )
        result = _handle_cron_create(
            {"template": "once", "recurring": True},
            agent_ref=_make_agent_ref(sched),
        )
        data = json.loads(result)
        assert data["job_id"] == "job_force"
        sched.add_job.assert_called_once_with(
            "30 14 28 2 *", "一次性提醒", catch_up=False, recurring=True,
        )
