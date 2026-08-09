"""Task System 测试（持久化 + DAG 依赖）。"""

import json

import pytest

from agent.task_store import TaskStore, get_task_store, VALID_STATUSES
from tools.registry import registry
from model_tools import ensure_tools_discovered

ensure_tools_discovered()


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return TaskStore(omnimate_home=tmp_path)


# ---------------------------------------------------------------------------
# create / get
# ---------------------------------------------------------------------------

def test_create_task(store):
    task = store.create(subject="做 A", description="详细说明")
    assert task["id"].startswith("task_")
    assert task["subject"] == "做 A"
    assert task["status"] == "pending"
    assert task["blocked_by"] == []
    # created_at 与 updated_at 在 create 时同源，但时间戳精度可能差几微秒，
    # 用 startswith 容忍（同秒级前缀即可）
    assert task["created_at"].startswith(task["updated_at"][:19]) or \
           task["updated_at"].startswith(task["created_at"][:19]), \
        f"created_at/updated_at 应同秒：{task['created_at']} vs {task['updated_at']}"


def test_get_task(store):
    created = store.create(subject="X")
    fetched = store.get(created["id"])
    assert fetched is not None
    assert fetched["id"] == created["id"]


def test_get_nonexistent(store):
    assert store.get("task_nope") is None


# ---------------------------------------------------------------------------
# update / status
# ---------------------------------------------------------------------------

def test_update_fields(store):
    t = store.create(subject="原始")
    updated = store.update(t["id"], subject="改后", description="新描述")
    assert updated["subject"] == "改后"
    assert updated["description"] == "新描述"
    # updated_at 变了
    assert updated["updated_at"] >= t["updated_at"]


def test_update_id_immutable(store):
    t = store.create(subject="X")
    updated = store.update(t["id"], id="task_hack")
    # id 不能被改
    assert updated["id"] == t["id"]


def test_set_status(store):
    t = store.create(subject="X")
    updated = store.set_status(t["id"], "in_progress")
    assert updated["status"] == "in_progress"


def test_set_invalid_status(store):
    t = store.create(subject="X")
    assert store.set_status(t["id"], "done") is None  # 非法状态


def test_claim(store):
    t = store.create(subject="X")
    claimed = store.claim(t["id"], owner="agent-1")
    assert claimed["owner"] == "agent-1"
    assert claimed["status"] == "in_progress"


def test_complete(store):
    t = store.create(subject="X")
    completed = store.complete(t["id"])
    assert completed["status"] == "completed"


# ---------------------------------------------------------------------------
# 持久化
# ---------------------------------------------------------------------------

def test_persist_across_instances(tmp_path):
    s1 = TaskStore(omnimate_home=tmp_path)
    t = s1.create(subject="持久化任务")

    s2 = TaskStore(omnimate_home=tmp_path)
    fetched = s2.get(t["id"])
    assert fetched is not None
    assert fetched["subject"] == "持久化任务"


# ---------------------------------------------------------------------------
# list_all
# ---------------------------------------------------------------------------

def test_list_all(store):
    store.create(subject="A")
    store.create(subject="B")
    tasks = store.list_all()
    assert len(tasks) == 2


def test_list_by_status(store):
    t1 = store.create(subject="done")
    t2 = store.create(subject="todo")
    store.complete(t1["id"])

    completed = store.list_all(status="completed")
    assert len(completed) == 1
    assert completed[0]["subject"] == "done"

    pending = store.list_all(status="pending")
    assert len(pending) == 1
    assert pending[0]["subject"] == "todo"


# ---------------------------------------------------------------------------
# 依赖（DAG）
# ---------------------------------------------------------------------------

def test_can_start_no_deps(store):
    t = store.create(subject="无依赖")
    assert store.can_start(t["id"]) is True


def test_can_start_with_unmet_deps(store):
    a = store.create(subject="A")
    b = store.create(subject="B", blocked_by=[a["id"]])
    # A 没完成
    assert store.can_start(b["id"]) is False


def test_can_start_after_dep_complete(store):
    a = store.create(subject="A")
    b = store.create(subject="B", blocked_by=[a["id"]])
    # A 完成后 B 可开始
    store.complete(a["id"])
    assert store.can_start(b["id"]) is True


def test_can_start_with_multiple_deps(store):
    a = store.create(subject="A")
    b = store.create(subject="B")
    c = store.create(subject="C", blocked_by=[a["id"], b["id"]])
    # A、B 都得完成
    store.complete(a["id"])
    assert store.can_start(c["id"]) is False
    store.complete(b["id"])
    assert store.can_start(c["id"]) is True


def test_find_ready(store):
    a = store.create(subject="A")
    b = store.create(subject="B", blocked_by=[a["id"]])
    c = store.create(subject="C")  # 无依赖

    ready = store.find_ready()
    ready_ids = [t["id"] for t in ready]
    assert a["id"] in ready_ids
    assert c["id"] in ready_ids
    assert b["id"] not in ready_ids  # 被阻塞


def test_find_blocked(store):
    a = store.create(subject="A")
    b = store.create(subject="B", blocked_by=[a["id"]])

    blocked = store.find_blocked()
    assert len(blocked) == 1
    assert blocked[0]["id"] == b["id"]


# ---------------------------------------------------------------------------
# 工具集成
# ---------------------------------------------------------------------------

async def test_task_create_tool(tmp_path):
    result = await registry.dispatch(
        "task_create",
        {"subject": "工具创建", "description": "via tool"},
        omnimate_home=tmp_path,
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["task"]["subject"] == "工具创建"


async def test_task_update_tool(tmp_path):
    # 先创建
    create_result = await registry.dispatch(
        "task_create", {"subject": "X"}, omnimate_home=tmp_path,
    )
    task_id = json.loads(create_result)["task"]["id"]

    # 更新
    result = await registry.dispatch(
        "task_update",
        {"id": task_id, "status": "in_progress"},
        omnimate_home=tmp_path,
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["task"]["status"] == "in_progress"


async def test_task_complete_tool_shows_unblocked(tmp_path):
    # 创建依赖链 A → B
    a_result = await registry.dispatch(
        "task_create", {"subject": "A"}, omnimate_home=tmp_path,
    )
    a_id = json.loads(a_result)["task"]["id"]
    await registry.dispatch(
        "task_create",
        {"subject": "B", "blocked_by": [a_id]},
        omnimate_home=tmp_path,
    )

    # 完成 A，应显示 B 被解锁
    result = await registry.dispatch(
        "task_complete", {"id": a_id}, omnimate_home=tmp_path,
    )
    data = json.loads(result)
    assert data["success"] is True


async def test_task_list_tool(tmp_path):
    await registry.dispatch(
        "task_create", {"subject": "T1"}, omnimate_home=tmp_path,
    )
    result = await registry.dispatch("task_list", {}, omnimate_home=tmp_path)
    data = json.loads(result)
    assert data["count"] >= 1


async def test_task_complete_returns_unblocked_with_subject(tmp_path):
    """task_complete 返回的 unblocked 含 id/subject/status 三字段。"""
    # 创建依赖链 A → B
    a_result = await registry.dispatch(
        "task_create", {"subject": "任务A"}, omnimate_home=tmp_path,
    )
    a_id = json.loads(a_result)["task"]["id"]
    b_result = await registry.dispatch(
        "task_create",
        {"subject": "任务B", "blocked_by": [a_id]},
        omnimate_home=tmp_path,
    )
    b_id = json.loads(b_result)["task"]["id"]

    # 完成 A，B 被解锁
    result = await registry.dispatch(
        "task_complete", {"id": a_id}, omnimate_home=tmp_path,
    )
    data = json.loads(result)
    assert data["success"] is True
    unblocked = data["unblocked"]
    assert isinstance(unblocked, list)
    assert len(unblocked) >= 1
    # 找到 B
    b_entries = [u for u in unblocked if u["id"] == b_id]
    assert len(b_entries) == 1
    entry = b_entries[0]
    assert entry["subject"] == "任务B"
    assert entry["status"] == "pending"


def test_task_tools_in_core():
    """4 个 task 工具在 core 工具集里。"""
    tools = ensure_tools_discovered() or []
    from model_tools import get_tool_definitions
    defs = get_tool_definitions(["core"])
    names = [t["function"]["name"] for t in defs]
    for name in ("task_create", "task_update", "task_complete", "task_list"):
        assert name in names, f"缺少工具: {name}"
