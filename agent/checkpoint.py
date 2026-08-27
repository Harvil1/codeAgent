"""Checkpoint（检查点）/ Rewind（回退）："后悔药"系统——给文件和对话拍快照，出了问题可以倒回去。

每次处理用户 prompt 之前，把 agent 改过的文件
都复制一份存起来；用户敲 /rewind 就能把文件和对话一起回滚到某个时间点。

边界（设计如此）：只追踪 write_file / str_replace 这类编辑工具的直接修改；
bash 命令改的文件不管——这套快照本来就明确不当 Git 用。
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """会话级检查点管理器：记下被改过的文件，提供快照 / 列表 / 回滚三个动作。

    快照的时机语义：create_snapshot() 在用户 prompt 开始执行前被调用，
    此刻复制 tracked 文件的内容存起来，就是"这条 prompt 动手前的状态"，
    /rewind 回滚的就是这些时间点。

    磁盘上的组织方式：
        {root_dir}/{session_id}/{snapshot_id}/
            files/{路径的md5}          # 文件副本（用 md5 当文件名避免路径里不能出现的字符）
            snapshot.json              # 快照清单：{ts, files:[{path, store}], conversation:[...]}
    """

    def __init__(
        self,
        root_dir,
        session_id: str,
        max_snapshots: int = 100,
    ):
        """建一个管理器。

        参数：
            root_dir：快照根目录（各会话各占一个子目录）
            session_id：会话 id（空的用 "nosession" 兜底）
            max_snapshots：最多留几份快照（最少 1），超了删最旧的
        """
        self._root = Path(root_dir)
        self._session_id = str(session_id or "nosession")
        self._max = max(1, int(max_snapshots))
        self._session_dir = self._root / self._session_id
        # 本次会话被编辑工具改过的文件集合（set 天然去重）
        self._tracked = set()

    # ------------------------------------------------------------------
    # 追踪
    # ------------------------------------------------------------------

    def track_file(self, path) -> None:
        """登记一个"被编辑工具改过的文件"（编辑工具每次写文件都来报个到）。

        参数：
            path：文件路径（空值忽略）

        返回：无。集合去重，同一文件报多次只记一次。
        """
        if path:
            self._tracked.add(str(Path(path).expanduser()))

    def tracked_files(self) -> list:
        """看看当前都盯着哪些文件。

        返回：
            排好序的路径字符串列表。
        """
        return sorted(self._tracked)

    # ------------------------------------------------------------------
    # 快照
    # ------------------------------------------------------------------

    def create_snapshot(self, conversation: Optional[list] = None) -> str:
        """拍快照：把当前追踪的所有文件复制一份存进新快照目录。

        参数：
            conversation：拍快照时的对话历史副本（之后 /rewind 连对话一起回滚要用）

        返回：
            snapshot_id（用 UTC 时间戳命名，如 20260820T101530123456）。
            个别文件复制失败只打 warning，不影响快照整体成立。
        """
        sid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        snap_dir = self._session_dir / sid
        files_dir = snap_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        files_meta = []
        for p in sorted(self._tracked):
            src = Path(p)
            if not src.exists() or not src.is_file():
                continue
            store_name = hashlib.md5(p.encode("utf-8")).hexdigest()
            try:
                shutil.copy2(src, files_dir / store_name)
                files_meta.append({"path": str(src), "store": store_name})
            except Exception as e:
                logger.warning("快照复制 %s 失败: %s", src, e)

        data = {
            "ts": sid,
            "files": files_meta,
            "conversation": conversation if conversation is not None else [],
        }
        try:
            (snap_dir / "snapshot.json").write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8",
            )
        except Exception as e:
            logger.warning("写 snapshot.json 失败: %s", e)

        self._prune_old()
        return sid

    def list_snapshots(self) -> list:
        """列出本会话的全部快照（给 /rewind 挑选用）。

        返回：
            列表，最新在前，每项形如 {id, ts, files, msg_count}；
            目录不存在或某份快照读坏了就跳过那份。
        """
        if not self._session_dir.exists():
            return []
        snaps = []
        for snap_dir in sorted(self._session_dir.iterdir()):
            if not snap_dir.is_dir():
                continue
            meta_path = snap_dir / "snapshot.json"
            if not meta_path.exists():
                continue
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            snaps.append({
                "id": snap_dir.name,
                "ts": data.get("ts", snap_dir.name),
                "files": [f.get("path", "") for f in data.get("files", [])],
                "msg_count": len(data.get("conversation", [])),
            })
        snaps.sort(key=lambda s: s["id"], reverse=True)  # 按时间戳 id 倒排 = 最新在前
        return snaps

    def restore_files(self, snapshot_id: str) -> list:
        """回滚文件：把指定快照里存的文件副本复制回原来的路径。

        参数：
            snapshot_id：要回到哪个快照

        返回：
            实际恢复成功的文件路径列表；快照不存在返回空列表，
            单个文件恢复失败只打 warning 继续恢复下一个。
        """
        snap_dir = self._session_dir / snapshot_id
        files_dir = snap_dir / "files"
        meta_path = snap_dir / "snapshot.json"
        if not files_dir.exists() or not meta_path.exists():
            return []
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        restored = []
        for f in data.get("files", []):
            src = files_dir / f["store"]
            if not src.exists():
                continue
            target = Path(f["path"])
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                restored.append(str(target))
            except Exception as e:
                logger.warning("回滚 %s 失败: %s", target, e)
        return restored

    def get_conversation(self, snapshot_id: str) -> list:
        """读某个快照当时存的对话历史副本（对话回滚的数据来源）。

        参数：
            snapshot_id：快照 id

        返回：
            消息列表；快照不存在或读坏了返回空列表。
        """
        meta_path = self._session_dir / snapshot_id / "snapshot.json"
        if not meta_path.exists():
            return []
        try:
            return json.loads(
                meta_path.read_text(encoding="utf-8"),
            ).get("conversation", [])
        except Exception:
            return []

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _prune_old(self) -> None:
        """内部清理：只保留最近 max_snapshots 份快照，更旧的整目录删掉。"""
        if not self._session_dir.exists():
            return
        snaps = sorted(
            d for d in self._session_dir.iterdir() if d.is_dir()
        )
        for old in snaps[: max(0, len(snaps) - self._max)]:
            try:
                shutil.rmtree(old)
            except Exception as e:
                logger.warning("清理旧快照 %s 失败: %s", old, e)
