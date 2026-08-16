"""R26 #18：cron 任务模板（~/.OmniMate/templates + <cwd>/.omnimate/templates）。"""
import json
from unittest.mock import MagicMock

from agent import templates as T
from agent.cron import CronJob
from tools.cron_tool import _handle_cron_create


class TestLoadTaskTemplates:
    def test_discover_and_parse(self, tmp_path, monkeypatch):
        import json
        from agent import templates as T
        home = tmp_path / "home"
        proj = tmp_path / "proj"
        for d in (home / "templates", proj / ".omnimate" / "templates"):
            d.mkdir(parents=True)
        (home / "templates" / "nightly.md").write_text(
            "---\ncron: \"0 3 * * *\"\ncatch_up: true\n---\n每晚跑全量测试并汇总失败",
            encoding="utf-8",
        )
        (proj / ".omnimate" / "templates" / "demo.md").write_text(
            "---\ncron: \"*/10 * * * *\"\nmessage: 项目级模板\n---\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("OMNIMATE_HOME", str(home))
        monkeypatch.setattr("agent.workspace_context.get_workspace_cwd", lambda: str(proj))
        T._invalidate_cache()
        result = T.load_task_templates()
        assert result["nightly"]["cron"] == "0 3 * * *"
        assert "全量测试" in result["nightly"]["message"]
        assert result["nightly"]["catch_up"] is True
        assert result["demo"]["message"] == "项目级模板"
        # 项目级覆盖用户级同名
        (proj / ".omnimate" / "templates" / "nightly.md").write_text(
            "---\ncron: \"1 2 3 4 5\"\n---\n覆盖", encoding="utf-8")
        T._invalidate_cache()
        assert T.load_task_templates()["nightly"]["cron"] == "1 2 3 4 5"

    def test_empty_when_no_dirs(self, tmp_path, monkeypatch):
        from agent import templates as T
        monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path / "none"))
        monkeypatch.setattr("agent.workspace_context.get_workspace_cwd", lambda: str(tmp_path))
        T._invalidate_cache()
        assert T.load_task_templates() == {}


def _make_agent_ref(scheduler=None):
    """构造带 cron_scheduler 属性的假 agent_ref（对齐 test_cron_tool.py 手法）。"""
    ref = MagicMock()
    ref.cron_scheduler = scheduler
    return ref


class TestCronCreateWithTemplate:
    def test_create_from_template(self, tmp_path, monkeypatch):
        """cron_create(template=nightly) → job 用模板值。"""
        home = tmp_path / "home"
        (home / "templates").mkdir(parents=True)
        (home / "templates" / "nightly.md").write_text(
            "---\ncron: \"0 3 * * *\"\ncatch_up: true\n---\n每晚跑全量测试并汇总失败",
            encoding="utf-8",
        )
        monkeypatch.setenv("OMNIMATE_HOME", str(home))
        monkeypatch.setattr(
            "agent.workspace_context.get_workspace_cwd", lambda: str(tmp_path)
        )
        T._invalidate_cache()

        sched = MagicMock()
        sched.add_job.return_value = CronJob(
            id="job_tpl", cron="0 3 * * *", message="每晚跑全量测试并汇总失败",
        )
        result = _handle_cron_create(
            {"template": "nightly"}, agent_ref=_make_agent_ref(sched),
        )
        data = json.loads(result)
        assert data["job_id"] == "job_tpl"
        assert data["cron"] == "0 3 * * *"
        sched.add_job.assert_called_once_with(
            "0 3 * * *", "每晚跑全量测试并汇总失败", catch_up=True, recurring=True,
        )

    def test_explicit_args_override_template(self, tmp_path, monkeypatch):
        """显式传的 cron/message/catch_up 参数优先模板值。"""
        home = tmp_path / "home"
        (home / "templates").mkdir(parents=True)
        (home / "templates" / "nightly.md").write_text(
            "---\ncron: \"0 3 * * *\"\ncatch_up: true\n---\n每晚跑全量测试并汇总失败",
            encoding="utf-8",
        )
        monkeypatch.setenv("OMNIMATE_HOME", str(home))
        monkeypatch.setattr(
            "agent.workspace_context.get_workspace_cwd", lambda: str(tmp_path)
        )
        T._invalidate_cache()

        sched = MagicMock()
        sched.add_job.return_value = CronJob(
            id="job_x", cron="* * * * *", message="显式消息",
        )
        result = _handle_cron_create(
            {
                "template": "nightly",
                "cron": "* * * * *",
                "message": "显式消息",
                "catch_up": False,
            },
            agent_ref=_make_agent_ref(sched),
        )
        data = json.loads(result)
        assert data["job_id"] == "job_x"
        sched.add_job.assert_called_once_with(
            "* * * * *", "显式消息", catch_up=False, recurring=True
        )

    def test_template_not_found(self, tmp_path, monkeypatch):
        """模板不存在 → invalid_template + available 可用列表。"""
        home = tmp_path / "home"
        (home / "templates").mkdir(parents=True)
        (home / "templates" / "nightly.md").write_text(
            "---\ncron: \"0 3 * * *\"\n---\n每晚跑全量测试", encoding="utf-8",
        )
        monkeypatch.setenv("OMNIMATE_HOME", str(home))
        monkeypatch.setattr(
            "agent.workspace_context.get_workspace_cwd", lambda: str(tmp_path)
        )
        T._invalidate_cache()

        sched = MagicMock()
        result = _handle_cron_create(
            {"template": "nope"}, agent_ref=_make_agent_ref(sched),
        )
        data = json.loads(result)
        assert data["error_type"] == "invalid_template"
        assert data["available"] == ["nightly"]
        sched.add_job.assert_not_called()

    def test_without_template_missing_args_still_invalid(self):
        """不带 template 时原校验保留：缺 cron/message → invalid_args。"""
        result = _handle_cron_create(
            {"message": "只有消息"}, agent_ref=_make_agent_ref(MagicMock()),
        )
        data = json.loads(result)
        assert data["error_type"] == "invalid_args"
