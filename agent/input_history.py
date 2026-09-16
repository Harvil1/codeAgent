"""两件事：跨会话的输入历史 + 大段粘贴的"引用协议"。

- **输入历史**：跨会话召回旧输入。追加写 ~/.codeAgent/history.jsonl（和最近
  一条重复就不记），倒着读；召回靠 `/history` 命令（列表 + `/history N`
  打印第 N 条原文）。纯 rich console 没有按键级行编辑，不做 ↑↓ 键绑定。
- **粘贴引用**：大段粘贴先存盘、消息里只留占位符（要用时再展开）。
  超过 1024 字符的大段输入存到 ~/.codeAgent/.paste/ 外面，
  消息里只留 `[Pasted text #N +M lines]` 占位符（会话存占位符省空间），
  发给 agent 前再展开成原文（expand_paste_references）；对应文件找不到就
  保留占位符（fail-open——外存被清理也不耽误对话）

整个文件都 fail-open：任何读写异常都返回原文或空列表，绝不影响输入主流程。
"""
import hashlib
import json
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

# 大段粘贴判定阈值（1024 字符），超过就走外存
PASTE_THRESHOLD = 1024
# 输入历史最多留多少条（100 条上限）
HISTORY_LIMIT = 100

# 占位符 id 兼容两种形态：hex（内容寻址 id）和纯数字编号
_PASTE_REF_RE = re.compile(r"\[Pasted text #([A-Za-z0-9]+)(?: \+\d+ lines)?\]")


@contextmanager
def _file_lock(lock_path: Path, timeout: float = 5.0):
    """跨平台独占文件锁——实现已提炼到 agent/file_lock.py（行为零变化），
    这里保留同名薄壳，本模块的调用点不用改。
    """
    from agent.file_lock import exclusive_file_lock
    with exclusive_file_lock(lock_path, timeout) as acquired:
        yield acquired


class GlobalHistory:
    """跨会话共享的输入历史（history.jsonl）：这个会话敲过的命令，别的会话也能翻出来。"""

    def __init__(self, home):
        """建一个历史记录器。

        参数：
            home：CodeAgent 主目录（history.jsonl 和锁文件都放这里）
        """
        self._home = Path(home)
        self._path = self._home / "history.jsonl"
        self._lock_path = self._home / ".history.lock"

    def append(self, text: str) -> None:
        """记一条输入（和最近一条一模一样就不记；条数超了裁旧的）。全程 fail-open。

        约束：追加和"软裁剪"（超上限时整文件重写）必须持文件锁——重写不加锁
        会覆盖别的进程刚追加的数据；等锁超时时追加照做、裁剪放弃（宁可不裁
        也不丢数据）。

        参数：
            text：用户这次输入的文本（空白不记）

        返回：无。出任何异常都静默吞掉。
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
                # 软裁剪：超过 2 倍上限才动手重写、只留最新 HISTORY_LIMIT 条（不用每条都裁，省事）
                # 注意：splitlines 剥掉了换行符，重写时必须补回去，否则记录会粘成一行
                if locked:
                    lines = self._read_all()
                    if len(lines) > HISTORY_LIMIT * 2:
                        kept = [l + "\n" for l in lines[-HISTORY_LIMIT:]]
                        self._path.write_text(
                            "".join(kept), encoding="utf-8",
                        )
        except Exception:
            pass
            logger.warning("异常被吞(fail-open)", exc_info=True)

    def _read_all(self) -> List[str]:
        """内部：把历史文件的每一行读成列表。读不了返回空列表（fail-open）。"""
        try:
            if not self._path.exists():
                return []
            return self._path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []

    def recent(self, n: int = 20) -> List[str]:
        """拿最近 n 条历史（最新的排最前）。

        参数：
            n：要几条

        返回：
            字符串列表；一行 JSON 坏了就跳过那行，全坏返回空列表（fail-open）。
        """
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
        """按编号取一条历史（给 /history N 用）。

        参数：
            n：序号，从 1 开始数，1 = 最新一条

        返回：
            对应的文本；编号越界返回 None。
        """
        items = self.recent(n)
        return items[n - 1] if 0 < n <= len(items) else None


# ---------------------------------------------------------------------------
# 粘贴引用协议——大段文本存外面，消息里只留占位符
# ---------------------------------------------------------------------------

def _paste_dir(home) -> Path:
    """大段粘贴的外存目录：{home}/.paste。"""
    return Path(home) / ".paste"


def _paste_id_for(text: str) -> str:
    """给一段粘贴内容算 id：取内容 sha256 的前 8 位 hex（内容寻址——同样内容算出同样 id，重复粘贴复用同一个文件）。"""
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:8]


def store_paste_if_large(text: str, home) -> Tuple[str, Optional[str]]:
    """输入太长就转存到外部文件，消息里换成占位符。

    超过 PASTE_THRESHOLD 字符时：原文存 .paste/text_<hash8>.txt（内容寻址——
    同样内容重复粘贴复用同一个文件），
    消息替换成 `[Pasted text #<hash8> +M lines]`；没超长就原样返回 (text, None)。
    数字编号形态的占位符（text_5.txt）在展开那边也认识（正则做了兼容）。

    参数：
        text：用户输入的原文
        home：CodeAgent 主目录

    返回：
        (替换后的消息文本, 外存文件路径或 None)。存储失败返回 (原文, None)——
        fail-open，绝不能因为存不了外存就把用户的输入弄丢。
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
    """把消息里的 `[Pasted text #N ...]` 占位符换回粘贴的原文（发给 agent 之前做）。

    参数：
        text：可能带占位符的消息文本
        home：CodeAgent 主目录

    返回：
        展开后的文本；占位符对应的外存文件找不到时保留占位符原样
        （fail-open——外存被清理也不阻塞对话）。
    """
    if not text or "[Pasted text #" not in text:
        return text

    def _expand(m: "re.Match") -> str:
        try:
            # id 是字母数字串，hex（内容寻址）和数字编号都能对上
            pid = m.group(1)
            path = _paste_dir(home) / f"text_{pid}.txt"
            if path.exists():
                return path.read_text(encoding="utf-8")
        except Exception:
            pass
            logger.warning("异常被吞(fail-open)", exc_info=True)
        return m.group(0)  # 文件不在就保留占位符不动

    return _PASTE_REF_RE.sub(_expand, text)
