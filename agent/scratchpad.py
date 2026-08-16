"""会话级 scratchpad 涂鸦区（R24 #36，对齐 CC getScratchpadDir）。

coordinator/orchestrator 场景的跨 worker 共享草稿区：
- 路径 ``<omnimate_home>/.scratchpad/<session_id>/``（会话隔离）
- **免权限读写**：目录加进 safe_path 的 extra_allowed_roots（运行时白名单，
  不写 settings.json——会话级生命周期，进程重启自动失效）
- 结构自由（CC 语义："structure files however fits the work"——
  durable cross-worker knowledge）
- ensure 时创建目录；cleanup 清理旧会话目录（retention 天数，默认 7）

与 CC 差异：CC 是 0o700 权限隔离 + tengu gate 门控；OmniMate 简化为
目录白名单 + retention 清理（单人本地产品无多用户隔离需求）。
fail-open：白名单注入失败只 log（scratchpad 仍可写，只是要过审批）。
"""
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 7


def scratchpad_dir(session_id: str, omnimate_home=None) -> Path:
    """会话 scratchpad 目录路径（不创建）。"""
    if omnimate_home is None:
        try:
            from constants import get_omnimate_home
            omnimate_home = get_omnimate_home()
        except Exception:
            omnimate_home = Path.home() / ".OmniMate"
    safe_sid = "".join(c for c in str(session_id or "default") if c.isalnum() or c in "-_")
    return Path(omnimate_home) / ".scratchpad" / (safe_sid or "default")


def ensure_scratchpad(session_id: str, omnimate_home=None) -> Optional[Path]:
    """创建会话 scratchpad 目录并注入免权限白名单。返回目录路径。

    白名单注入：add_extra_allowed_root（运行时，进程级——与 /add-dir 同
    通道但持久化由调用方决定；这里只加运行时，不写 settings.json）。
    fail-open：白名单失败仍返回目录（写文件会走审批）。
    """
    try:
        d = scratchpad_dir(session_id, omnimate_home)
        d.mkdir(parents=True, exist_ok=True)
        try:
            from agent.permission import add_extra_allowed_root
            add_extra_allowed_root(d)
        except Exception as e:
            logger.warning("scratchpad 白名单注入失败（写入将走审批）: %s", e)
        return d
    except Exception as e:
        logger.warning("scratchpad 创建失败（fail-open）: %s", e)
        return None


def scratchpad_context_block(session_id: str, omnimate_home=None) -> str:
    """注入 orchestrator/coordinator 用户上下文的 scratchpad 说明（对齐 CC 文案）。"""
    d = scratchpad_dir(session_id, omnimate_home)
    return (
        f"\n\nScratchpad 目录：{d}\n"
        "worker 子代理可在此免权限读写。用它存放跨 worker 的持久知识"
        "（中间结论/共享状态/待合并草稿）——文件结构自由组织。"
    )


def cleanup_old_scratchpads(omnimate_home=None, retention_days: int = DEFAULT_RETENTION_DAYS) -> int:
    """清理超过 retention_days 未修改的旧会话 scratchpad 目录。返回清理数。

    ⚠️ 完全可逆铁律的例外说明：scratchpad 是**临时涂鸦区**（非知识库），
    会话隔离 + 按 mtime 清理是 CC 同款语义；真正的知识沉淀走记忆/技能系统。
    fail-open。
    """
    try:
        if omnimate_home is None:
            try:
                from constants import get_omnimate_home
                omnimate_home = get_omnimate_home()
            except Exception:
                return 0
        root = Path(omnimate_home) / ".scratchpad"
        if not root.exists():
            return 0
        cutoff = time.time() - retention_days * 86400
        removed = 0
        for d in root.iterdir():
            try:
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    import shutil
                    shutil.rmtree(d)
                    removed += 1
            except Exception:
                continue
        if removed:
            logger.info("scratchpad 清理 %d 个过期会话目录", removed)
        return removed
    except Exception:
        return 0
