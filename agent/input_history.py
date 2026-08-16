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
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

# 大段粘贴阈值（对齐 CC pasteStore 的 1024 字符）
PASTE_THRESHOLD = 1024
# 历史召回上限（对齐 CC 上限 100 条）
HISTORY_LIMIT = 100

_PASTE_REF_RE = re.compile(r"\[Pasted text #(\d+)(?: \+\d+ lines)?\]")


class GlobalHistory:
    """history.jsonl 跨会话输入历史。"""

    def __init__(self, home):
        self._path = Path(home) / "history.jsonl"

    def append(self, text: str) -> None:
        """追加一条（与最近一条相同不记；超上限裁剪旧条）。fail-open。"""
        if not text or not text.strip():
            return
        try:
            recent = self.recent(1)
            if recent and recent[0] == text:
                return  # 与最近一条相同
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            # 软裁剪：超 2 倍上限时重写保留最新 HISTORY_LIMIT 条（不每次裁）
            # 注意 splitlines 已剥换行符——重写时补回（否则拼成一行废掉整个文件）
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


def _next_paste_id(home) -> int:
    """下一个粘贴编号（扫描现有 text_*.txt 取最大 +1）。"""
    d = _paste_dir(home)
    max_id = 0
    try:
        if d.exists():
            for f in d.glob("text_*.txt"):
                try:
                    max_id = max(max_id, int(f.stem.split("_")[1]))
                except (IndexError, ValueError):
                    continue
    except Exception:
        pass
    return max_id + 1


def store_paste_if_large(text: str, home) -> Tuple[str, Optional[str]]:
    """大段输入外存 + 返回 (消息文本, 存储路径或 None)。

    > PASTE_THRESHOLD 字符 → 存 .paste/text_<n>.txt，消息替换为
    `[Pasted text #N +M lines]`；否则原样返回 (text, None)。fail-open：
    存储失败返回原文。
    """
    if not text or len(text) <= PASTE_THRESHOLD:
        return text, None
    try:
        d = _paste_dir(home)
        d.mkdir(parents=True, exist_ok=True)
        pid = _next_paste_id(home)
        path = d / f"text_{pid}.txt"
        path.write_text(text, encoding="utf-8")
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
            pid = int(m.group(1))
            path = _paste_dir(home) / f"text_{pid}.txt"
            if path.exists():
                return path.read_text(encoding="utf-8")
        except Exception:
            pass
        return m.group(0)  # 保留占位符

    return _PASTE_REF_RE.sub(_expand, text)
