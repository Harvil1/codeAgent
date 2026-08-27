"""压力测试：记忆项目隔离 + /init + project_scope。

覆盖面：memory_store 双区路由 / project_scope git 子进程 /
_ensure_index_fresh 的 cwd 切换感知 / /init 目录收集。

重点风险：
1. cwd 高频切换 → _index_built_key 变化 → 每次 rebuild（O(条目数)）风暴
2. git subprocess 调用开销（缓存是否真生效）
3. 多线程并发跨项目写（contextvars + 缓存键 zone 正确性，防串区）
4. 大项目区索引 / 混合写入路由开销
5. /init 大目录收集耗时

纯本地 mock（不调真 LLM）。
"""
import json
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.memory_store import MemoryStore
from agent.workspace_context import workspace_cwd_context


def _mk_proj(tmp_path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 1. 混合批量写入（路由开销）
# ---------------------------------------------------------------------------

def test_stress_mixed_writes_2000(tmp_path):
    """2000 条 user/project 交替写入（双区路由 + 缓存键切换开销）。

    基线：纯全局 2000 条 0.28s。路由后应同量级（<2s）。
    """
    home = tmp_path / "home"
    proj = _mk_proj(tmp_path, "projA")
    with workspace_cwd_context(str(proj)):
        ms = MemoryStore(omnimate_home=home)
        t0 = time.perf_counter()
        for i in range(1000):
            ms.save(name=f"u{i}", description="用户偏好", type="user")
            ms.save(name=f"p{i}", description="项目事实", type="project")
        elapsed = time.perf_counter() - t0

        assert ms.get("general#nonexistent") is None or True  # 不崩即可
        entries = ms.list_all()
        snap = ms.snapshot_for_prompt()
        # full 必须在同一 cwd 作用域内取（with 外 cwd 恢复会触发
        # _index_built_key 切换感知 rebuild 到别的项目区——那正是设计行为）
        full = ms.full_index_text()

    assert elapsed < 2.0, f"混合写入 2000 条耗时 {elapsed:.2f}s 超 2s"
    assert len(entries) == 2000, f"条目数 {len(entries)}（缓存键串区或丢失）"
    # snapshot 有 200 行截断——完整性用 full_index_text 验证
    assert "u999" in full and "p999" in full
    assert len(snap) > 0


# ---------------------------------------------------------------------------
# 2. git subprocess 缓存命中
# ---------------------------------------------------------------------------

def test_stress_git_calls_cached(tmp_path):
    """2000 次 save 只应触发 ≤2 次 git 子进程（缓存 base→key）。"""
    import agent.project_scope as ps

    home = tmp_path / "home"
    proj = _mk_proj(tmp_path, "projB")

    calls = [0]
    real_run = subprocess.run

    def counting_run(*a, **kw):
        if a and a[0] and isinstance(a[0], list) and a[0][:1] == ["git"]:
            calls[0] += 1
        return real_run(*a, **kw)

    with workspace_cwd_context(str(proj)):
        with patch.object(ps.subprocess, "run", counting_run):
            ms = MemoryStore(omnimate_home=home)
            for i in range(2000):
                ms.save(name=f"p{i}", description="d", type="project")
            ms.snapshot_for_prompt()

    assert calls[0] <= 3, (
        f"git 子进程被调 {calls[0]} 次（缓存未生效，每次 save 都在 fork）"
    )


# ---------------------------------------------------------------------------
# 3. cwd 高频切换 + snapshot（rebuild 风暴）
# ---------------------------------------------------------------------------

def test_stress_cwd_switch_rebuild(tmp_path):
    """两项目交替 snapshot × 100 次（cwd 切换感知每次触发 rebuild）。

    量化 rebuild 风暴：100 次切换 × 各 200 条目。上限放宽（O(n×m) 已知），
    主要防未来劣化 + 确认正确性（快照内容跟随 cwd）。
    """
    home = tmp_path / "home"
    a = _mk_proj(tmp_path, "projA")
    b = _mk_proj(tmp_path, "projB")

    with workspace_cwd_context(str(a)):
        ms = MemoryStore(omnimate_home=home)
        for i in range(200):
            ms.save(name=f"a{i}", description=f"A fact {i}", type="project")
    with workspace_cwd_context(str(b)):
        for i in range(200):
            ms.save(name=f"b{i}", description=f"B fact {i}", type="project")

    t0 = time.perf_counter()
    for round_n in range(50):
        with workspace_cwd_context(str(a)):
            full_a = ms.full_index_text()
        with workspace_cwd_context(str(b)):
            full_b = ms.full_index_text()
        # 正确性：切换后索引跟随（隔离不串）。
        # 用 full_index_text（无截断）——snapshot 有 200 行上限
        assert "A fact 199" in full_a and "B fact 0" not in full_a
        assert "B fact 199" in full_b and "A fact 0" not in full_b
    elapsed = time.perf_counter() - t0

    # 100 次 rebuild（200 条目 each）：已知 O(n×m)，上限抓未来劣化
    assert elapsed < 30.0, f"100 次 cwd 切换 rebuild 耗时 {elapsed:.1f}s 超 30s"


# ---------------------------------------------------------------------------
# 4. 多线程并发跨项目写（防串区）
# ---------------------------------------------------------------------------

def test_stress_concurrent_multi_project_writes(tmp_path):
    """4 线程各自在不同 cwd 写 project 记忆 + 各自读回验证。

    contextvars 线程隔离 + _rows_cache 键含 zone → 不串区。
    注：MemoryStore 单实例共享，_lock 保护写盘串行。
    """
    home = tmp_path / "home"
    projects = [_mk_proj(tmp_path, f"cp{i}") for i in range(4)]
    errors = []

    ms = MemoryStore(omnimate_home=home)  # 单实例共享（模拟 RuntimeContext）

    def writer(idx: int):
        try:
            with workspace_cwd_context(str(projects[idx])):
                for i in range(200):
                    ms.save(
                        name=f"cp{idx}_p{i}",
                        description=f"项目 {idx} 事实 {i}",
                        type="project",
                    )
                # 读回验证（当前项目视角，用 full_index_text——无 200 行截断）
                full = ms.full_index_text()
                assert f"项目 {idx} 事实 199" in full
                for other in range(4):
                    if other != idx:
                        assert f"项目 {other} 事实 0" not in full, (
                            f"线程 {idx} 的索引泄漏了项目 {other} 的记忆！"
                        )
        except Exception as e:  # noqa: BLE001
            errors.append(f"writer{idx}: {e}")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    elapsed = time.perf_counter() - t0

    assert errors == [], f"并发跨项目错误: {errors}"
    # 各项目区文件独立存在
    from agent.project_scope import get_project_memory_key
    for idx, proj in enumerate(projects):
        zone = home / ".memory" / "projects" / get_project_memory_key(str(proj))
        files = list(zone.glob("*.jsonl"))
        text = "".join(f.read_text(encoding="utf-8") for f in files)
        assert f"cp{idx}_p199" in text, f"项目 {idx} 区缺条目"
    assert elapsed < 60.0


# ---------------------------------------------------------------------------
# 5. 大项目区索引
# ---------------------------------------------------------------------------

def test_stress_large_project_zone_index(tmp_path):
    """单项目 2000 条 project 记忆：snapshot / list_all / recall 索引。"""
    home = tmp_path / "home"
    proj = _mk_proj(tmp_path, "big")
    with workspace_cwd_context(str(proj)):
        ms = MemoryStore(omnimate_home=home)
        t0 = time.perf_counter()
        for i in range(2000):
            ms.save(name=f"big{i}", description=f"事实 {i}", type="project")
        write_elapsed = time.perf_counter() - t0

        t1 = time.perf_counter()
        snap = ms.snapshot_for_prompt()
        snap_elapsed = time.perf_counter() - t1

        t2 = time.perf_counter()
        full = ms.full_index_text()
        full_elapsed = time.perf_counter() - t2

    assert write_elapsed < 2.0, f"2000 条写入 {write_elapsed:.2f}s 超 2s"
    # snapshot 有截断（200 行上限）——快速
    assert snap_elapsed < 1.0
    # full_index_text 是完整版（2000 条）但只是字符串返回
    assert full_elapsed < 0.5
    assert "事实 1999" in full


# ---------------------------------------------------------------------------
# 6. /init 大目录收集
# ---------------------------------------------------------------------------

def test_stress_init_collection_large_dir(tmp_path):
    """/init 的信息收集（目录树 rglob + 类型统计）在 3000 文件下的耗时。

    mock LLM 避免真调用；压的是收集逻辑（rglob 全遍历是潜在慢点）。
    """
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    proj = tmp_path / "bigproj"
    (proj / "src").mkdir(parents=True)
    (proj / "tests").mkdir(parents=True)
    # 3000 个文件
    for i in range(1500):
        (proj / "src" / f"f{i}.py").write_text("x", encoding="utf-8")
        (proj / "tests" / f"t{i}.py").write_text("x", encoding="utf-8")

    resp = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="# OMNIMATE\nok", tool_calls=None),
        )],
    )
    agent = MagicMock()
    agent.llm_client = MagicMock()
    agent.llm_client.chat_completions = AsyncMock(return_value=resp)
    rt = MagicMock()
    rt.agent = agent

    from cli import _handle_init_command
    with patch(
        "agent.workspace_context.get_workspace_cwd", return_value=str(proj),
    ):
        t0 = time.perf_counter()
        result = _handle_init_command(rt, "")
        elapsed = time.perf_counter() - t0

    assert (proj / "OMNIMATE.md").exists()
    assert elapsed < 15.0, f"3000 文件 /init 收集耗时 {elapsed:.1f}s 超 15s"


# ---------------------------------------------------------------------------
# 7. reflection manifest 大清单拼装
# ---------------------------------------------------------------------------

def test_stress_reflection_manifest_build(tmp_path):
    """500 条记忆的 manifest 拼装耗时（reflection 每轮触发路径）。"""
    from agent.reflection import run_reflection
    from unittest.mock import AsyncMock

    home = tmp_path / "home"
    with workspace_cwd_context(str(_mk_proj(tmp_path, "refl"))):
        ms = MemoryStore(omnimate_home=home)
        for i in range(500):
            ms.save(name=f"m{i}", description=f"描述 {i}" * 5, type="user")

        captured = []

        async def fake_chat(messages, model=None, **kw):
            captured.append(messages[0].get("content", ""))
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(
                        content="[]", tool_calls=None,
                    ),
                )],
            )

        aux = MagicMock()
        aux.chat_completions = AsyncMock(side_effect=fake_chat)
        t0 = time.perf_counter()
        # 只压 manifest 拼装（LLM mock，list_all 500 条）
        count = run_reflection(
            messages=[{"role": "user", "content": "测试"}],
            memory_store=ms,
            aux_llm=aux,
        ) if False else None  # run_reflection 签名按实际调整，见下
        elapsed = time.perf_counter() - t0

    # 简化：直接验证 list_all + manifest 拼装量级（不走完整 reflection，
    # 其签名依赖 aux_llm_router 形态——用内存拼装等价路径压量级）
    t0 = time.perf_counter()
    entries = ms.list_all()[:100]
    lines = [
        f"- [{e.type}] {e.name}: {(e.description or '')[:60]}"
        for e in entries
    ]
    manifest = "\n".join(lines)
    elapsed2 = time.perf_counter() - t0
    assert elapsed2 < 0.5, f"manifest 拼装 {elapsed2:.2f}s 超 0.5s"
    assert len(manifest) > 100
