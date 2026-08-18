"""workflow journal：断点恢复的记账层（R28 W2，蓝图 §4）。

目录布局 ~/.OmniMate/.workflows/<run_id>/：
  script.py（首跑快照——resume 只信它，防缓存投毒）
  script.sha256 / journal.jsonl（append-only）/ meta.json
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def call_key(prompt: str, schema: Optional[dict]) -> str:
    """调用键：sha256(prompt + schema 规范化 JSON)。"""
    payload = json.dumps(
        [prompt, schema], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class WorkflowJournal:
    """单 run 的 journal（非线程安全——engine 串行 append）。"""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.journal_path = self.run_dir / "journal.jsonl"
        self._entries: dict = {}
        self._seq = 0
        self._load()

    # ---- 生命周期 ----
    @classmethod
    def create(cls, run_dir: Path, script_source: str) -> "WorkflowJournal":
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "script.py").write_text(script_source, encoding="utf-8")
        h = hashlib.sha256(script_source.encode("utf-8")).hexdigest()
        (run_dir / "script.sha256").write_text(h, encoding="utf-8")
        j = cls(run_dir)
        j.save_meta({"status": "running", "created_at": time.time()})
        return j

    @classmethod
    def load(cls, run_dir: Path) -> "WorkflowJournal":
        """加载已有 run；脚本 hash 失配 → 截断全部条目（重跑语义）。"""
        run_dir = Path(run_dir)
        j = cls(run_dir)
        try:
            snap = (run_dir / "script.py").read_text(encoding="utf-8")
            if not j.verify_script(snap):
                logger.warning("workflow 脚本 hash 失配，journal 截断重跑: %s", run_dir)
                j.truncate_all()
        except OSError:
            pass
        return j

    def verify_script(self, source: str) -> bool:
        try:
            want = (self.run_dir / "script.sha256").read_text(encoding="utf-8").strip()
        except OSError:
            return False
        got = hashlib.sha256(source.encode("utf-8")).hexdigest()
        return want == got

    # ---- 条目 ----
    def _load(self) -> None:
        if not self.journal_path.exists():
            return
        for line in self.journal_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                self._entries[e["key"]] = e
                self._seq = max(self._seq, int(e.get("seq", 0)))
            except (json.JSONDecodeError, KeyError):
                continue  # 崩溃残留的半行丢弃（append-only 耐崩语义）

    def lookup(self, key: str) -> Optional[dict]:
        e = self._entries.get(key)
        return e.get("result") if e else None

    def append(self, key: str, result: dict) -> int:
        self._seq += 1
        entry = {"key": key, "seq": self._seq, "result": result}
        self._entries[key] = entry
        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return self._seq

    def truncate_all(self) -> None:
        self._entries.clear()
        self._seq = 0
        try:
            self.journal_path.write_text("", encoding="utf-8")
        except OSError:
            pass

    def __len__(self) -> int:
        return len(self._entries)

    # ---- meta ----
    def save_meta(self, data: dict) -> None:
        """合并写 meta.json（R30c-C7：改 tmp+rename 原子写——此前裸 write_text，
        进程中断会留下半截 JSON，resume 读 meta 直接失败）。"""
        from agent.atomic_io import atomic_write_text
        p = self.run_dir / "meta.json"
        old = {}
        try:
            old = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
        old.update(data)
        atomic_write_text(
            p, json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def load_meta(self) -> dict:
        try:
            return json.loads(
                (self.run_dir / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            return {}
