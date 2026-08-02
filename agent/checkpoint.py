"""Checkpoint / Rewind：会话级文件快照 + 回滚（对齐 Claude Code）。

每个用户 prompt 前对 agent 修改过的文件做快照，/rewind 可回滚文件 + 对话。
只追踪 write_file/str_replace 等编辑工具的直接修改（bash 命令不追踪，
Claude Code 明确这不是 Git 的替代品）。
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
    """会话级 checkpoint：记录被编辑文件，支持快照 / 列表 / 回滚。

    快照语义：create_snapshot() 在用户 prompt 执行前调用，复制 tracked 文件
    的当前内容 = "prompt 前的状态"，回滚可回到该点。

    存储结构：
        {root_dir}/{session_id}/{snapshot_id}/
            files/{md5(path)}          # 文件副本
            snapshot.json              # {ts, files:[{path, store}], conversation:[...]}
    """

    def __init__(
        self,
        root_dir,
        session_id: str,
        max_snapshots: int = 100,
    ):
        self._root = Path(root_dir)
        self._session_id = str(session_id or "nosession")
        self._max = max(1, int(max_snapshots))
        self._session_dir = self._root / self._session_id
        # 本次会话被编辑工具改过的文件（去重）
        self._tracked = set()

    # ------------------------------------------------------------------
    # 追踪
    # ------------------------------------------------------------------

    def track_file(self, path) -> None:
        """记录被编辑工具修改的文件（去重）。"""
        if path:
            self._tracked.add(str(Path(path).expanduser()))

    def tracked_files(self) -> list:
        """当前追踪的文件列表（排序）。"""
        return sorted(self._tracked)

    # ------------------------------------------------------------------
    # 快照
    # ------------------------------------------------------------------

    def create_snapshot(self, conversation: Optional[list] = None) -> str:
        """复制所有 tracked 文件的当前内容到一个新快照，返回 snapshot_id。

        conversation：快照时的对话历史副本（用于对话回滚）。
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
        """列出全部快照（最新在前）。返回 [{id, ts, files, msg_count}]。"""
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
        snaps.sort(key=lambda s: s["id"], reverse=True)  # 最新在前
        return snaps

    def restore_files(self, snapshot_id: str) -> list:
        """把快照的文件副本复制回原路径。返回恢复的文件列表。"""
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
        """读快照存的对话历史副本（用于对话回滚）。"""
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
        """保留最近 max_snapshots 个快照，删最旧的。"""
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
