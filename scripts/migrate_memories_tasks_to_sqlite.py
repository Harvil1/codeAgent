"""一次性迁移：把 .memory/*.md 和 .tasks/*.json 导入 SQLite。

策略：
  - 复制不删（老文件保留作备份）
  - 校验 count(*) 对得上
  - 老代码仍读文件，新代码双写
  - 稳定 1 周后可手动改读 SQLite

用法：
    uv run python -m scripts.migrate_memories_tasks_to_sqlite
"""
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def migrate_memories(harvil_home: Path, sqlite_store) -> dict:
    """导入 .memory/*.md 到 SQLite。返回 {ok, skipped, failed}。"""
    memory_dir = harvil_home / ".memory"
    if not memory_dir.exists():
        return {"ok": 0, "skipped": 0, "failed": 0, "reason": "no .memory dir"}

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    import yaml
    from agent.memory_store import _parse_frontmatter

    for md_path in sorted(memory_dir.glob("*.md")):
        try:
            text = md_path.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(text)
            if meta is None:
                stats["skipped"] += 1
                continue
            mid = md_path.stem
            sqlite_store.save_memory(
                id=mid,
                name=meta.get("name", ""),
                description=meta.get("description", ""),
                type=meta.get("type", "other"),
                body=body,
                created_at=meta.get("created_at", ""),
                updated_at=meta.get("updated_at", ""),
            )
            stats["ok"] += 1
        except Exception as e:
            logger.warning("迁移 memory 失败 %s: %s", md_path, e)
            stats["failed"] += 1
    return stats


def migrate_tasks(harvil_home: Path, sqlite_store) -> dict:
    """导入 .tasks/*.json 到 SQLite。返回 {ok, skipped, failed}。"""
    tasks_dir = harvil_home / ".tasks"
    if not tasks_dir.exists():
        return {"ok": 0, "skipped": 0, "failed": 0, "reason": "no .tasks dir"}

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    for json_path in sorted(tasks_dir.glob("*.json")):
        try:
            task = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(task, dict) or "id" not in task:
                stats["skipped"] += 1
                continue
            sqlite_store.save_task(
                id=task["id"],
                subject=task.get("subject", ""),
                description=task.get("description", "") or "",
                status=task.get("status", "pending"),
                owner=task.get("owner"),
                created_at=task.get("created_at", ""),
                updated_at=task.get("updated_at", ""),
                blocked_by=task.get("blocked_by", []),
                body_json=json.dumps(task, ensure_ascii=False),
            )
            stats["ok"] += 1
        except Exception as e:
            logger.warning("迁移 task 失败 %s: %s", json_path, e)
            stats["failed"] += 1
    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from constants import get_agent_home
    from agent.session_store import SessionStore

    home = get_agent_home()
    print(f"Agent home: {home}")

    db_path = home / "sessions.db"
    store = SessionStore(db_path)

    print("\n=== 迁移 memories ===")
    mem_stats = migrate_memories(home, store)
    print(f"  {mem_stats}")

    print("\n=== 迁移 tasks ===")
    task_stats = migrate_tasks(home, store)
    print(f"  {task_stats}")

    # 校验
    mem_in_db = len(store.list_memories(include_archived=True))
    mem_in_files = len(list((home / ".memory").glob("*.md"))) if (home / ".memory").exists() else 0
    task_in_db = len(store.list_tasks(include_deleted=True))
    task_in_files = len(list((home / ".tasks").glob("*.json"))) if (home / ".tasks").exists() else 0

    print("\n=== 校验 ===")
    print(f"  memories: 文件 {mem_in_files} → SQLite {mem_in_db}")
    print(f"  tasks:    文件 {task_in_files} → SQLite {task_in_db}")

    if mem_in_db < mem_in_files - mem_stats["skipped"]:
        print("  ⚠️ memories 入库数 < 文件数（部分失败）")
    if task_in_db < task_in_files - task_stats["skipped"]:
        print("  ⚠️ tasks 入库数 < 文件数（部分失败）")

    print("\n✅ 迁移完成。老文件保留作备份。")


if __name__ == "__main__":
    main()
