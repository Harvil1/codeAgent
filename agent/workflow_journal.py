"""工作流执行日志（journal）——为断点恢复记的账本（R28 W2，蓝图 §4）。

在项目里的位置：由 workflow_engine 在调子代理前后写入，resume（断点恢复）
时读取；对上服务 workflow_engine，对下只碰文件系统。

每个 run（一次工作流执行）在 ~/.OmniMate/.workflows/<run_id>/ 下有一套文件：
  script.py      首跑时的脚本快照——恢复时只信这份快照，防止有人改脚本后
                 借旧账本"投毒"（旧缓存是按旧脚本跑出来的）
  script.sha256  脚本的指纹（哈希），用来核对脚本有没有被动过
  journal.jsonl  账本本体：一行一条记录，只追加不修改（append-only——
                 追加写在中断时最多丢最后一行，不会把整本账写坏）
  meta.json      运行状态等元信息
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def call_key(prompt: str, schema: Optional[dict]) -> str:
    """给"一次子代理调用"算唯一指纹：prompt 和 schema 拼一起取哈希。

    用途：同样的调用（同 prompt 同 schema）指纹相同，账本里就能查到上次
    的结果直接复用，不用重跑。

    参数：
        prompt：发给子代理的指令。
        schema：要求的 JSON Schema（可为 None）。
    返回：十六进制哈希字符串。
    """
    payload = json.dumps(
        [prompt, schema], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class WorkflowJournal:
    """一个 run 专属的账本（不是线程安全的——引擎是串行记账，够用）。

    一般不直接 new：新 run 用 create()，恢复旧 run 用 load()。
    """

    def __init__(self, run_dir: Path):
        """记下 run 目录，并把盘上的账本读进内存。

        参数：
            run_dir：这个 run 的专属目录。
        """
        self.run_dir = Path(run_dir)
        self.journal_path = self.run_dir / "journal.jsonl"
        self._entries: dict = {}
        self._seq = 0
        self._load()

    # ---- 生命周期：新建 / 恢复 ----
    @classmethod
    def create(cls, run_dir: Path, script_source: str) -> "WorkflowJournal":
        """开一个新 run：建目录、存脚本快照和指纹、写初始 meta。

        参数：
            run_dir：要创建的 run 目录。
            script_source：本次要跑的脚本文本（原样快照下来）。
        返回：新建好的 WorkflowJournal（状态标为 running）。
        """
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
        """恢复一个旧 run：读回账本；发现脚本被动过就整本作废重跑。

        背景（防缓存投毒）：账本里的结果是按旧脚本跑出来的，脚本变了旧账
        就不可信——指纹（hash）对不上时清空全部条目，相当于从头再跑。

        参数：
            run_dir：要恢复的 run 目录。
        返回：恢复好的 WorkflowJournal（脚本指纹失配时是空账本）。
        """
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
        """核对一段脚本的指纹跟快照时记下的是否一致。

        参数：
            source：待核对的脚本文本。
        返回：True=没被动过；False=指纹对不上（或指纹文件读不到）。
        """
        try:
            want = (self.run_dir / "script.sha256").read_text(encoding="utf-8").strip()
        except OSError:
            return False
        got = hashlib.sha256(source.encode("utf-8")).hexdigest()
        return want == got

    # ---- 账本条目：读 / 查 / 记 ----
    def _load(self) -> None:
        """把盘上的 jsonl 账本逐行读进内存（同 key 后写的覆盖先写的）。"""
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
                continue  # 程序崩时写了一半的残行直接丢——这正是只追加写法耐崩的原因

    def lookup(self, key: str) -> Optional[dict]:
        """按调用指纹查账：这个调用之前跑过吗、结果是什么。

        参数：
            key：call_key 算出的调用指纹。
        返回：当初记下的 result dict；没跑过返回 None。
        """
        e = self._entries.get(key)
        return e.get("result") if e else None

    def append(self, key: str, result: dict) -> int:
        """记一笔账（往 jsonl 文件追加一行，同时在内存里存一份）。

        参数：
            key：调用指纹。
            result：结果 dict（如 {"kind": "ok", "output": ...}）。
        返回：这笔账的流水号 seq。
        """
        self._seq += 1
        entry = {"key": key, "seq": self._seq, "result": result}
        self._entries[key] = entry
        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return self._seq

    def truncate_all(self) -> None:
        """整本账作废：内存清空、文件清空、流水号归零。"""
        self._entries.clear()
        self._seq = 0
        try:
            self.journal_path.write_text("", encoding="utf-8")
        except OSError:
            pass

    def __len__(self) -> int:
        """账本里有多少条记录（len(journal) 直接可用）。"""
        return len(self._entries)

    # ---- meta：运行状态等元信息 ----
    def save_meta(self, data: dict) -> None:
        """把若干字段合并进 meta.json（已有的字段保留，不整文件覆盖）。

        历史踩坑（R30c-C7 修复）：以前直接 write_text，进程中断会留下
        半截 JSON，resume 读 meta 直接失败。现在改成"先写临时文件再改名"
        的原子写，中断最多丢这次更新，不会写坏整个文件。

        参数：
            data：要写入的字段 dict。
        """
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
        """读 meta.json；文件不存在或坏了返回空 dict（不抛错）。

        返回：元信息 dict。
        """
        try:
            return json.loads(
                (self.run_dir / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            return {}
