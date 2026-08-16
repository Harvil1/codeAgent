"""R21 S 类专项测试。

#46 auto 分类器增强（拒绝回落 + 危险前缀剥离）
#48 任务全清 + verification nudge
#23 记忆路径安全
#41 孤儿并行工具结果修复
#37 yolo 交接复审
#8 记忆检索并行 prefetch
#45 审批命令解释器
#39 粘贴引用协议
#42 全局输入历史
"""

import json
from types import SimpleNamespace

import pytest

from model_tools import ensure_tools_discovered
from tools.registry import registry

ensure_tools_discovered()

from agent.permission import (
    LLM_DENIAL_MAX_CONSECUTIVE,
    LLM_DENIAL_MAX_TOTAL,
    PermissionChecker,
    _is_dangerous_whitelist_entry,
)


# ---------------------------------------------------------------------------
# R21 #46：auto 分类器增强
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry", [
    "python", "python3", "node", "npx", "bash", "sh", "ssh",
    "eval", "exec", "env", "xargs", "sudo", "curl", "wget",
    "python:*", "node:*",            # x:* 旧前缀形态
    "python -c", "npx serve",        # 首词形态
    "npm run", "npm run build",
])
def test_dangerous_whitelist_entries(entry):
    assert _is_dangerous_whitelist_entry(entry), f"应判危险: {entry}"


@pytest.mark.parametrize("entry", [
    "ls", "git status", "cat", "pip list", "echo", "", "  ",
])
def test_safe_whitelist_entries(entry):
    if entry.strip():
        assert not _is_dangerous_whitelist_entry(entry), f"不应判危险: {entry}"
    else:
        assert _is_dangerous_whitelist_entry(entry)  # 空条目排除


def _mk_classifier_checker(whitelist, verdicts):
    """构造闸门 4 全开的 checker（providers 注入 + mock aux）。"""
    calls = {"n": 0}

    class _Aux:
        async def chat_completions(self, msgs, **kw):
            calls["n"] += 1
            v = verdicts.pop(0) if verdicts else {"safe": True}
            content = json.dumps(v)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=content),
            )])

    checker = PermissionChecker()
    checker.set_aux_llm_provider(lambda: _Aux())
    checker.set_config_provider(lambda: {
        "features": {"bash_llm_classifier": {"enabled": True, "whitelist": whitelist}},
    })
    return checker, calls


def test_dangerous_prefix_stripped_from_whitelist():
    """白名单含 python → python 命令不走快速通道（仍走 LLM 分类）。"""
    checker, calls = _mk_classifier_checker(
        ["ls", "python"], [{"safe": True}],
    )
    r = checker.check("python -c 'print(1)'")
    assert calls["n"] == 1  # 没被白名单快速通道跳过，走了 LLM
    assert r.allowed and r.gate == "llm_safe"
    # 安全前缀仍走快速通道（0 LLM）——用非只读命令（ls 会被 T7 只读通道先截走）
    checker2, calls2 = _mk_classifier_checker(["pip install"], [])
    r2 = checker2.check("pip install requests")
    assert r2.gate == "whitelist"
    assert calls2["n"] == 0


def test_denial_fallback_consecutive():
    """连续 3 次判 unsafe → 闸门 4 停用（后续 0 LLM 调用，默认通过）。"""
    checker, calls = _mk_classifier_checker(
        [], [{"safe": False, "reason": "x"}, {"safe": False, "reason": "x"},
             {"safe": False, "reason": "x"}, {"safe": False, "reason": "x"}],
    )
    for _ in range(3):
        r = checker.check("some command")
        assert not r.allowed and r.gate == "llm_unsafe"
    assert checker._llm_denial_consecutive == LLM_DENIAL_MAX_CONSECUTIVE
    # 第 4 次：闸门 4 已停用 → 0 新 LLM 调用 → 默认通过
    r4 = checker.check("some command")
    assert r4.allowed and r4.gate == "ok"
    assert calls["n"] == 3


def test_denial_total_fallback():
    """累计 20 次拒绝（穿插 safe 重置连续计数，连续阈值不触发）→ 停用。"""
    verdicts = []
    for _ in range(10):
        verdicts.extend([{"safe": False, "reason": "x"}, {"safe": False, "reason": "x"},
                         {"safe": True}])
    checker, calls = _mk_classifier_checker([], verdicts)
    for _ in range(30):
        checker.check("cmd")
    assert checker._llm_denial_total == 20
    assert checker._llm_denial_total >= LLM_DENIAL_MAX_TOTAL
    # 第 29 次 unsafe 后 total=20 → 闸门停用；第 30 次不再调 LLM
    # （verdicts 里的 safe 没被消费）→ consecutive 保持 2
    assert checker._llm_denial_consecutive == 2
    # 后续 0 新调用
    n_before = calls["n"]
    checker.check("cmd")
    assert calls["n"] == n_before


# ---------------------------------------------------------------------------
# R21 #48：任务全清 + verification nudge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_task_all_done_cleanup_and_nudge(tmp_path):
    """全 completed → 自动软删清空；≥3 活跃无验证字样 → reminder。"""
    from agent.task_store import get_task_store

    class _FakeAgent:
        hooks_registry = None
    home = str(tmp_path)
    store = get_task_store(home)
    t1 = store.create(subject="做 A", description="实现 A")["id"]
    t2 = store.create(subject="做 B", description="实现 B")["id"]
    t3 = store.create(subject="做 C", description="实现 C")["id"]

    # 完成 2 个（剩 1 个活跃 → 无 nudge 无清空）
    r1 = json.loads(await registry.dispatch("task_complete", {"id": t1},
                                            omnimate_home=home, agent_ref=_FakeAgent()))
    assert "all_tasks_cleared" not in r1
    assert "reminder" not in r1

    # 完成第 3 个 → 全 completed → 清空
    await registry.dispatch("task_complete", {"id": t2},
                            omnimate_home=home, agent_ref=_FakeAgent())
    r3 = json.loads(await registry.dispatch("task_complete", {"id": t3},
                                            omnimate_home=home, agent_ref=_FakeAgent()))
    assert r3.get("all_tasks_cleared") is True
    assert r3.get("cleared_count") == 3
    # 软删（可恢复）：全部 status=deleted，活跃列表为空
    statuses = {t["status"] for t in store.list_all()}
    assert statuses == {"deleted"}
    assert store.list_all(status="pending") == []


@pytest.mark.asyncio
async def test_task_verification_nudge(tmp_path):
    """≥3 活跃且无验证字样 → reminder；描述含验证则不提醒。"""
    from agent.task_store import get_task_store

    class _FakeAgent:
        hooks_registry = None
    home = str(tmp_path)
    store = get_task_store(home)
    done = store.create(subject="D", description="已完成项")["id"]
    for i in range(3):
        store.create(subject=f"任务{i}", description=f"实现功能{i}")
    # 完成 done（剩 3 条活跃无验证字样）→ reminder 出现
    r = json.loads(await registry.dispatch("task_complete", {"id": done},
                                           omnimate_home=home, agent_ref=_FakeAgent()))
    assert "reminder" in r and "验证" in r["reminder"]

    # 描述含"测试"的活跃任务集 → 不提醒
    home2 = str(tmp_path / "s2")
    store2 = get_task_store(home2)
    d2 = store2.create(subject="D", description="x")["id"]
    for i in range(3):
        store2.create(subject=f"任务{i}", description=f"实现并测试功能{i}")
    r2 = json.loads(await registry.dispatch("task_complete", {"id": d2},
                                            omnimate_home=home2, agent_ref=_FakeAgent()))
    assert "reminder" not in r2
