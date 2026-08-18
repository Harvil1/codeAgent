"""全局输入历史 + 粘贴引用协议（R21 #42/#39）。

对齐 CC history.ts（跨会话 ↑↓ 召回）+ pasteStore.ts（大段粘贴外存按需展开），
适配 OmniMate 纯 rich console（无 readline 行编辑）：

- **历史**：~/.OmniMate/history.jsonl 追加（与最近一条相同不记），倒序读；
  召回走 `/history` 命令（列表 + `/history N` 打印原文）——↑↓ 键绑定留待
  TUI 化（rich console.input 无行编辑，不引入平台终端 hack）
- **粘贴引用**：>1024 字符的输入外存 ~/.OmniMate/.paste/text_<n>.txt，
  消息中替换为 `[Pasted text #N +M lines]` 占位符（session 存占位符省空间），
  发送给 agent 前展开原文（expand_paste_references）；无文件时保留占位符
  （fail-open，不阻塞对话）

全部 fail-open：任何 I/O 异常返回原文/空列表，绝不影响输入主流程。
"""
import hashlib
import json
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

# 大段粘贴阈值（对齐 CC pasteStore 的 1024 字符）
PASTE_THRESHOLD = 1024
# 历史召回上限（对齐 CC 上限 100 条）
HISTORY_LIMIT = 100

# R30d-C10：占位符 id 兼容 hex（内容寻址）与旧数字编号
_PASTE_REF_RE = re.compile(r"\[Pasted text #([A-Za-z0-9]+)(?: \+\d+ lines)?\]")


@contextmanager
def _file_lock(lock_path: Path, timeout: float = 5.0):
    """跨平台独占文件锁（Windows msvcrt / POSIX fcntl，与 team bus 同款）。

    yield 是否真正拿到锁；超时 yield False（调用方自行决定降级动作）。
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lf:
        acquired = False
        deadline = time.time() + timeout
        if sys.platform == "win32":
            import msvcrt
            while time.time() < deadline:
                try:
                    msvcrt.locking(lf.fileno(), msvcrt.LK_LOCK, 1)
                    acquired = True
                    break
                except OSError:
                    time.sleep(0.01)
        else:
            import fcntl
            while time.time() < deadline:
                try:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    time.sleep(0.01)
        try:
            yield acquired
        finally:
            if acquired:
                try:
                    if sys.platform == "win32":
                        import msvcrt
                        msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass


class GlobalHistory:
    """history.jsonl 跨会话输入历史。"""

    def __init__(self, home):
        self._home = Path(home)
        self._path = self._home / "history.jsonl"
        self._lock_path = self._home / ".history.lock"

    def append(self, text: str) -> None:
        """追加一条（与最近一条相同不记；超上限裁剪旧条）。fail-open。

        R30d-C9：append + 软裁剪都在文件锁内——此前无锁的"整文件重写"
        会丢掉并发进程 append 的数据。锁超时时 append 照做（追加低风险），
        裁剪跳过（重写必须持锁，宁可不裁也不丢数据）。
        """
        if not text or not text.strip():
            return
        try:
            recent = self.recent(1)
            if recent and recent[0] == text:
                return  # 与最近一条相同
            with _file_lock(self._lock_path) as locked:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                # 软裁剪：超 2 倍上限时重写保留最新 HISTORY_LIMIT 条（不每次裁）
                # 注意 splitlines 已剥换行符——重写时补回（否则拼成一行废掉整个文件）
                if locked:
                    lines = self._read_all()
                    if len(lines) > HISTORY_LIMIT * 2:
                        kept = [l + "\n" for l in lines[-HISTORY_LIMIT:]]
                        self._path.write_text(
                            "".join(kept), encoding="utf-8",
                        )
        except Exception:
            pass

    def _read_all(self) -> List[str]:
        try:
            if not self._path.exists():
                return []
            return self._path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []

    def recent(self, n: int = 20) -> List[str]:
        """最近 n 条（最新在前）。fail-open 返回空列表。"""
        lines = self._read_all()
        out = []
        for line in reversed(lines):
            try:
                obj = json.loads(line)
                t = obj.get("text")
                if t:
                    out.append(t)
            except json.JSONDecodeError:
                continue
            if len(out) >= n:
                break
        return out

    def get(self, n: int) -> Optional[str]:
        """第 n 条（1-based，最新=1）。"""
        items = self.recent(n)
        return items[n - 1] if 0 < n <= len(items) else None


# ---------------------------------------------------------------------------
# R21 #39：粘贴引用协议
# ---------------------------------------------------------------------------

def _paste_dir(home) -> Path:
    return Path(home) / ".paste"


def _paste_id_for(text: str) -> str:
    """内容寻址 id：sha256 前 8 位 hex（R30d-C10——同内容只存一份）。"""
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:8]


def store_paste_if_large(text: str, home) -> Tuple[str, Optional[str]]:
    """大段输入外存 + 返回 (消息文本, 存储路径或 None)。

    > PASTE_THRESHOLD 字符 → 存 .paste/text_<hash8>.txt（**内容寻址**，
    R30d-C10——同内容重复粘贴复用同一文件，不再每次新写 text_<n>.txt），
    消息替换为 `[Pasted text #<hash8> +M lines]`；否则原样返回
    (text, None)。fail-open：存储失败返回原文。旧数字编号的占位符
    （text_5.txt）在 expand 侧仍可展开（regex 兼容）。
    """
    if not text or len(text) <= PASTE_THRESHOLD:
        return text, None
    try:
        d = _paste_dir(home)
        d.mkdir(parents=True, exist_ok=True)
        pid = _paste_id_for(text)
        path = d / f"text_{pid}.txt"
        if not path.exists():
            from agent.atomic_io import atomic_write_text_lite
            atomic_write_text_lite(path, text, encoding="utf-8")
        lines = text.count("\n") + 1
        return f"[Pasted text #{pid} +{lines} lines]", str(path)
    except Exception:
        return text, None


def expand_paste_references(text: str, home) -> str:
    """把消息里的 `[Pasted text #N ...]` 占位符展开为原文。

    无对应文件时保留占位符（fail-open——外存被清理不阻塞对话）。
    """
    if not text or "[Pasted text #" not in text:
        return text

    def _expand(m: "re.Match") -> str:
        try:
            # R30d-C10：id 是 alnum 字符串（hex 内容寻址 / 旧数字编号皆可）
            pid = m.group(1)
            path = _paste_dir(home) / f"text_{pid}.txt"
            if path.exists():
                return path.read_text(encoding="utf-8")
        except Exception:
            pass
        return m.group(0)  # 保留占位符

    return _PASTE_REF_RE.sub(_expand, text)
